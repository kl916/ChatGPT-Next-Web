import hashlib
import io
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from google import genai
from google.oauth2 import service_account
from google.genai import types
from google.cloud import storage

# ==========================================
# 1. CONFIGURATION & AUTH
# ==========================================
st.set_page_config(page_title="Veo: Storyboard to Video", layout="wide")
st.title("🎞️ Veo: Storyboard Director Mode")
st.caption(
    "Upload a storyboard → frames are cut out automatically → each pair of neighbouring "
    "frames (1→2, 2→3, …) becomes one video, using them as the FIRST and LAST frame."
)

# --- CSS: keep any video within the viewport so you never have to scroll to preview it ---
st.markdown(
    """
    <style>
        video {
            max-height: 60vh !important;
            width: auto !important;
            max-width: 100% !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

KEY_FILE = r'C:\Users\cklai\Desktop\GeminiVertex\ai-job-mapping-be2e5d3bb768.json'
PROJECT_ID = 'ai-job-mapping'
LOCATION = 'us-central1'
GCS_OUTPUT_URI = 'gs://ai_job_mapping_bucket/'
MY_SEED = 73238
MODELS = ['veo-3.1-fast-generate-001', 'veo-3.1-generate-001']

# Every generated take is also saved here, so nothing is lost if the browser tab reloads.
OUTPUT_ROOT = Path("storyboard_output")

# Frame size sent to Veo for each aspect ratio
TARGET_SIZE = {"16:9": (1280, 720), "9:16": (720, 1280)}
FIT_MODES = ["Crop centre", "Crop left", "Crop right", "Pad (blurred fill)"]
MAX_REFS = 4
REF_TYPES = {"Asset (character / object / place)": "ASSET", "Style": "STYLE"}

# --- SESSION STATE MANAGEMENT ---
# 'frames'   = list of {"num": 1-based storyboard frame number, "raw": PIL panel image}
# 'segments' = dict keyed "a->b" (frame numbers) with the job / result for that pair:
#              {"status": idle|queued|running|done|error, "op_name", "error",
#               "candidates": [{"uri", "path"}], "selected": index of the kept take}
defaults = {
    "frames": [],
    "segments": {},
    "storyboard_hash": None,
    "run_requested": False,
    "zip_bytes": None,
    "output_dir": OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S"),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "global_prompt" not in st.session_state:
    st.session_state.global_prompt = (
        "Cinematic live-action film, consistent characters and lighting, smooth continuous "
        "camera motion from the first frame to the last frame."
    )

# Streamlit forgets a widget's value on any run where that widget isn't drawn (e.g. the
# run that shows the progress panel). Re-assigning the keys each run keeps prompts,
# frame choices and selected takes safe.
for k in list(st.session_state.keys()):
    if k.startswith(("prompt_", "use_", "fit_", "sel_", "global_prompt", "bulk_prompts",
                     "reftype_", "ref_with_frames")):
        st.session_state[k] = st.session_state[k]


@st.cache_resource
def get_clients():
    creds = service_account.Credentials.from_service_account_file(
        KEY_FILE, scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    genai_client = genai.Client(
        vertexai=True, project=PROJECT_ID, location=LOCATION, credentials=creds
    )
    storage_client = storage.Client(project=PROJECT_ID, credentials=creds)
    return genai_client, storage_client


# NOTE: we connect to Google only when a video is generated (not at page load), so the
# storyboard upload and settings always appear even if the key file / network has a problem.


# ==========================================
# 2. STORYBOARD → FRAMES
# ==========================================
def _runs(mask):
    """Return (start, end) index pairs of consecutive True values."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def detect_panels(img, tol=30, gutter_ratio=0.92, min_frac=0.08):
    """Find panels separated by plain background gutters, in reading order.

    The background colour is taken from the image border. A row (then, inside each row
    band, a column) counts as gutter when almost all its pixels match that colour.
    Bands smaller than min_frac of the image (e.g. the title strip) are ignored.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    H, W, _ = rgb.shape
    border = np.concatenate([
        rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3),
        rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    is_bg = np.abs(rgb - bg).max(axis=2) <= tol

    boxes = []
    for y0, y1 in _runs(is_bg.mean(axis=1) < gutter_ratio):
        if y1 - y0 < min_frac * H:
            continue
        band = is_bg[y0:y1]
        for x0, x1 in _runs(band.mean(axis=0) < gutter_ratio):
            if x1 - x0 < min_frac * W:
                continue
            boxes.append((x0, y0, x1, y1))
    return boxes


def grid_panels(img, rows, cols, top_pct):
    """Fallback: split into a uniform rows × cols grid, skipping a header strip."""
    W, H = img.size
    top = int(H * top_pct / 100)
    ch, cw = (H - top) / rows, W / cols
    return [
        (int(c * cw), int(top + r * ch), int((c + 1) * cw), int(top + (r + 1) * ch))
        for r in range(rows) for c in range(cols)
    ]


def fit_to_aspect(img, aspect, mode):
    """Crop or pad a panel to the Veo aspect ratio and resize to the target size."""
    tw, th = TARGET_SIZE[aspect]
    target_ratio = tw / th
    w, h = img.size
    if mode == "Pad (blurred fill)":
        bg = fit_to_aspect(img, aspect, "Crop centre").filter(ImageFilter.GaussianBlur(30))
        scale = min(tw / w, th / h)
        fg = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        bg.paste(fg, ((tw - fg.width) // 2, (th - fg.height) // 2))
        return bg
    if w / h > target_ratio:  # too wide → cut the sides
        nw = int(h * target_ratio)
        x0 = {"Crop left": 0, "Crop right": w - nw}.get(mode, (w - nw) // 2)
        img = img.crop((x0, 0, x0 + nw, h))
    else:  # too tall → cut top/bottom evenly
        nh = int(w / target_ratio)
        y0 = (h - nh) // 2
        img = img.crop((0, y0, w, y0 + nh))
    return img.resize((tw, th), Image.LANCZOS)


def processed_frame(frame):
    """Panel → caption bars trimmed → fitted to aspect ratio. Returns a PIL image."""
    img = frame["raw"]
    w, h = img.size
    top = int(h * trim_top / 100)
    bottom = h - int(h * trim_bottom / 100)
    left = int(w * trim_side / 100)
    right = w - int(w * trim_side / 100)
    img = img.crop((left, top, max(right, left + 1), max(bottom, top + 1)))
    mode = st.session_state.get(f"fit_{frame['num']}", default_fit)
    return fit_to_aspect(img, aspect_ratio, mode)


def load_frames(images):
    """Replace the frame list (and forget old results) when a new storyboard arrives."""
    st.session_state.frames = [{"num": i + 1, "raw": im} for i, im in enumerate(images)]
    st.session_state.segments = {}
    st.session_state.zip_bytes = None
    for i in range(len(images)):
        st.session_state[f"use_{i + 1}"] = True
        st.session_state[f"fit_{i + 1}"] = default_fit


def to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ==========================================
# 3. GENERATION LOGIC (queue of first/last-frame jobs)
# ==========================================
def seg_key(a, b):
    return f"{a}->{b}"


def full_prompt(a, b):
    parts = [st.session_state.global_prompt.strip(),
             st.session_state.get(f"prompt_{a}_{b}", "").strip()]
    return "\n\n".join(p for p in parts if p)


def submit_segment(key, first, last):
    """Start one Veo job: first frame = image, last frame = config.last_frame,
    plus any reference images the user uploaded."""
    a, b = first["num"], last["num"]
    use_frames = not ref_images or st.session_state.get("ref_with_frames", True)
    config_kwargs = dict(
        number_of_videos=num_videos,
        duration_seconds=duration,
        aspect_ratio=aspect_ratio,
        generate_audio=generate_audio,
        output_gcs_uri=GCS_OUTPUT_URI,
        person_generation="allow_adult",
    )
    if use_frames:
        config_kwargs["last_frame"] = types.Image(
            image_bytes=to_png_bytes(processed_frame(last)), mime_type="image/png"
        )
    if ref_images:
        config_kwargs["reference_images"] = [
            types.VideoGenerationReferenceImage(
                image=types.Image(image_bytes=r["bytes"], mime_type="image/png"),
                reference_type=r["type"],
            )
            for r in ref_images
        ]
    # A fixed seed makes multiple takes identical, so only lock it for single takes.
    if num_videos == 1:
        config_kwargs["seed"] = MY_SEED

    client, _ = get_clients()
    operation = client.models.generate_videos(
        model=model_name,
        prompt=full_prompt(a, b),
        image=types.Image(
            image_bytes=to_png_bytes(processed_frame(first)), mime_type="image/png"
        ) if use_frames else None,
        config=types.GenerateVideosConfig(**config_kwargs),
    )
    seg = st.session_state.segments[key]
    seg.update(status="running", op_name=operation.name, error=None)


def collect_segment(key, operation):
    """Download every take of a finished job to disk and record it."""
    seg = st.session_state.segments[key]
    if getattr(operation, "error", None):
        seg.update(status="error", error=str(operation.error))
        return
    generated = (operation.response.generated_videos or []) if operation.response else []
    if not generated:
        reasons = getattr(operation.response, "rai_media_filtered_reasons", None)
        seg.update(status="error", error=f"No video returned (filtered?): {reasons}")
        return

    out_dir = st.session_state.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = key.split("->")
    candidates = []
    for i, gv in enumerate(generated):
        res_uri = gv.video.uri
        bucket_name = res_uri.split('/')[2]
        blob_path = '/'.join(res_uri.split('/')[3:])
        blob = get_clients()[1].bucket(bucket_name).blob(blob_path)
        stamp = datetime.now().strftime("%H%M%S")
        path = out_dir / f"segment_{int(a):02d}-{int(b):02d}_take{i + 1}_{stamp}.mp4"
        path.write_bytes(blob.download_as_bytes())
        candidates.append({"uri": res_uri, "path": str(path)})
    seg.update(status="done", candidates=candidates, selected=0, error=None)
    st.session_state[f"sel_{key}"] = 0


def run_queue(frame_by_num):
    """Keep up to `max_parallel` jobs running until every queued segment is finished.

    Job names are stored in session state, so if the page reruns mid-way the
    'Resume' button picks the running jobs back up instead of starting again.
    """
    segs = st.session_state.segments
    try:
        with st.spinner("Connecting to Google Cloud..."):
            get_clients()
    except Exception as e:
        st.error(f"Could not connect to Google Cloud — check KEY_FILE / PROJECT_ID.\n\n{e}")
        return False
    with st.status("🎥 Generating storyboard segments...", expanded=True) as status:
        progress = st.progress(0.0)
        total = sum(1 for s in segs.values() if s["status"] in ("queued", "running"))
        finished = 0
        while True:
            running = [k for k, s in segs.items() if s["status"] == "running"]
            queued = [k for k, s in segs.items() if s["status"] == "queued"]

            while queued and len(running) < max_parallel:
                key = queued.pop(0)
                a, b = (int(x) for x in key.split("->"))
                try:
                    submit_segment(key, frame_by_num[a], frame_by_num[b])
                    running.append(key)
                    st.write(f"▶️ Started segment {a} → {b}")
                except Exception as e:
                    segs[key].update(status="error", error=str(e))
                    finished += 1
                    st.write(f"❌ Segment {a} → {b} failed to start: {e}")

            if not running and not queued:
                break

            st.write(f"Rendering {len(running)} job(s), {len(queued)} waiting... (polling every 15s)")
            time.sleep(15)

            for key in running:
                try:
                    op = get_clients()[0].operations.get(
                        types.GenerateVideosOperation(name=segs[key]["op_name"])
                    )
                    if not op.done:
                        continue
                    collect_segment(key, op)
                except Exception as e:
                    segs[key].update(status="error", error=str(e))
                finished += 1
                a, b = key.split("->")
                if segs[key]["status"] == "done":
                    st.write(f"✅ Segment {a} → {b} ready")
                    st.video(segs[key]["candidates"][0]["path"])  # instant preview
                else:
                    st.write(f"❌ Segment {a} → {b}: {segs[key]['error']}")
            progress.progress(min(finished / max(total, 1), 1.0))

        status.update(label="✅ All requested segments finished", state="complete")
    return True


# ==========================================
# 4. SIDEBAR — GLOBAL SETTINGS
# ==========================================
st.sidebar.header("Video Settings")
model_name = st.sidebar.selectbox("Model", MODELS)
duration = st.sidebar.select_slider("Duration per segment (s)", options=[4, 6, 8], value=8)
aspect_ratio = st.sidebar.selectbox("Aspect Ratio", list(TARGET_SIZE))
num_videos = st.sidebar.slider("Takes per segment", min_value=1, max_value=4, value=1)
generate_audio = st.sidebar.checkbox("Generate audio", value=True)
max_parallel = st.sidebar.slider(
    "Jobs running at once", 1, 4, 2,
    help="Higher is faster but may hit your Vertex AI quota (quota errors show per segment).",
)

st.sidebar.header("Frame Clean-up")
st.sidebar.caption("Trim the text labels / caption bars printed on each panel, so they don't appear in the video.")
trim_top = st.sidebar.slider("Trim top %", 0, 40, 12)
trim_bottom = st.sidebar.slider("Trim bottom %", 0, 40, 17)
trim_side = st.sidebar.slider("Trim sides %", 0, 20, 0)
default_fit = st.sidebar.selectbox("Default fit to aspect ratio", FIT_MODES)

st.sidebar.divider()
st.sidebar.caption(f"Takes are saved to: `{st.session_state.output_dir}`")
if st.sidebar.button("🗑️ Reset all segments"):
    st.session_state.segments = {}
    st.session_state.zip_bytes = None
    st.rerun()


# ==========================================
# 5. STEP 1 — LOAD STORYBOARD & CUT FRAMES
# ==========================================
st.header("1️⃣ Storyboard")
source = st.radio(
    "Frame source", ["Storyboard image (auto-split)", "Individual frame images"], horizontal=True
)

if source == "Storyboard image (auto-split)":
    uploaded = st.file_uploader("Upload storyboard", type=["jpg", "jpeg", "png", "webp"])
    c1, c2 = st.columns([2, 3])
    with c1:
        split_mode = st.radio("Split method", ["Auto-detect panels", "Uniform grid"], horizontal=True)
    with c2:
        if split_mode == "Auto-detect panels":
            tol = st.slider("Gutter colour tolerance", 5, 80, 30,
                            help="Raise if panels are merged; lower if panels are split in pieces.")
        else:
            g1, g2, g3 = st.columns(3)
            rows = g1.number_input("Rows", 1, 10, 4)
            cols = g2.number_input("Columns", 1, 10, 4)
            header_pct = g3.number_input("Skip header %", 0, 30, 5)

    if uploaded:
        data = uploaded.getvalue()
        board = Image.open(io.BytesIO(data)).convert("RGB")
        if split_mode == "Auto-detect panels":
            boxes = detect_panels(board, tol=tol)
        else:
            boxes = grid_panels(board, rows, cols, header_pct)
        sig = hashlib.md5(data + repr(boxes).encode()).hexdigest()
        if sig != st.session_state.storyboard_hash:
            st.session_state.storyboard_hash = sig
            load_frames([board.crop(b) for b in boxes])
        with st.expander("Show uploaded storyboard", expanded=False):
            st.image(board, use_container_width=True)
else:
    files = st.file_uploader(
        "Upload frames (they are ordered by file name, e.g. 01.png, 02.png…)",
        type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
    )
    if files:
        files = sorted(files, key=lambda f: f.name)
        sig = hashlib.md5(b"".join(f.name.encode() + f.getvalue()[:4096] for f in files)).hexdigest()
        if sig != st.session_state.storyboard_hash:
            st.session_state.storyboard_hash = sig
            load_frames([Image.open(f).convert("RGB") for f in files])

frames = st.session_state.frames
if not frames:
    st.info("Upload a storyboard to begin.")
    st.stop()

# --- Frame review grid: see exactly what Veo will receive, choose which frames to use ---
st.subheader(f"Detected {len(frames)} frames")
st.caption("This is exactly what Veo receives (after trimming and fitting). Adjust the sidebar trims if text is still visible.")
per_row = 5
for start in range(0, len(frames), per_row):
    cols = st.columns(per_row)
    for col, frame in zip(cols, frames[start:start + per_row]):
        with col:
            st.image(processed_frame(frame), caption=f"Frame {frame['num']}", use_container_width=True)
            st.checkbox("Use", key=f"use_{frame['num']}")
            st.selectbox("Fit", FIT_MODES, key=f"fit_{frame['num']}", label_visibility="collapsed")

active = [f for f in frames if st.session_state.get(f"use_{f['num']}", True)]
frame_by_num = {f["num"]: f for f in frames}
pairs = [(active[i], active[i + 1]) for i in range(len(active) - 1)]
if not pairs:
    st.warning("Select at least two frames.")
    st.stop()

for first, last in pairs:
    key = seg_key(first["num"], last["num"])
    st.session_state.segments.setdefault(
        key, {"status": "idle", "op_name": None, "error": None, "candidates": [], "selected": 0}
    )
active_keys = [seg_key(a["num"], b["num"]) for a, b in pairs]

# --- Optional reference images (same set is sent with every segment) ---
st.subheader("🖼️ Reference images (optional, up to 4)")
st.caption(
    "Photos of your characters, costumes, props or location, so Veo keeps them consistent "
    "in every video. Veo 3.1 usually needs 8 s duration for reference images, and some "
    "models accept at most 3 — if Veo rejects the request, the reason is shown on the segment."
)
ref_files = st.file_uploader(
    "Upload reference images", type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True, key="ref_uploader",
)
if ref_files and len(ref_files) > MAX_REFS:
    st.warning(f"Only the first {MAX_REFS} reference images are used.")
ref_images = []
if ref_files:
    rcols = st.columns(MAX_REFS)
    for i, f in enumerate(ref_files[:MAX_REFS]):
        with rcols[i]:
            img = Image.open(f).convert("RGB")
            st.image(img, caption=f.name, use_container_width=True)
            label = st.selectbox("Type", list(REF_TYPES), key=f"reftype_{i}")
            ref_images.append({"bytes": to_png_bytes(img), "type": REF_TYPES[label]})
    if "ref_with_frames" not in st.session_state:
        st.session_state.ref_with_frames = True
    st.checkbox(
        "Also use the storyboard frames as first / last frame",
        key="ref_with_frames",
        help="Untick if Veo says reference images can't be combined with first/last frames. "
             "Each video is then made from its prompt + reference images only.",
    )

# ==========================================
# 6. STEP 2 — PROMPTS
# ==========================================
st.header("2️⃣ Prompts")
st.caption("Type the prompt for each video below before generating. The global prompt is added in front of every one.")
st.text_area(
    "Global prompt (added to every segment — describe characters, location, style)",
    key="global_prompt", height=90,
)

with st.expander("📋 Paste all segment prompts at once (one line per segment)"):
    bulk = st.text_area("Line 1 = frames 1→2, line 2 = frames 2→3, …", key="bulk_prompts", height=160)
    if st.button("Apply to segments"):
        lines = [ln.strip() for ln in bulk.splitlines() if ln.strip()]
        for (first, last), line in zip(pairs, lines):
            st.session_state[f"prompt_{first['num']}_{last['num']}"] = line
        st.rerun()

for idx, (first, last) in enumerate(pairs):
    a, b = first["num"], last["num"]
    t1, t2, t3 = st.columns([1, 1, 4])
    t1.image(processed_frame(first), caption=f"Frame {a}", use_container_width=True)
    t2.image(processed_frame(last), caption=f"Frame {b}", use_container_width=True)
    t3.text_area(
        f"Video {idx + 1} prompt (frame {a} → frame {b})", key=f"prompt_{a}_{b}", height=110,
        placeholder="What happens between these two frames? Action, camera movement, sound…",
    )

# ==========================================
# 7. STEP 3 — GENERATE
# ==========================================
st.header("3️⃣ Generate")
n_done = sum(1 for k in active_keys if st.session_state.segments[k]["status"] == "done")
st.write(f"**{len(pairs)} segments** · {n_done} done · {duration}s each · {num_videos} take(s) per segment"
         + (f" · {len(ref_images)} reference image(s)" if ref_images else ""))
empty = [i + 1 for i, (f, l) in enumerate(pairs)
         if not st.session_state.get(f"prompt_{f['num']}_{l['num']}", "").strip()]
if empty:
    st.warning(f"Video(s) {', '.join(map(str, empty))} have no prompt yet — only the global prompt will be used.")

b1, b2, b3 = st.columns(3)
if b1.button("🚀 Generate ALL segments", use_container_width=True, type="primary"):
    for k in active_keys:
        st.session_state.segments[k].update(status="queued", error=None)
    st.session_state.run_requested = True
if b2.button("⏭️ Generate missing / failed only", use_container_width=True):
    for k in active_keys:
        if st.session_state.segments[k]["status"] in ("idle", "error"):
            st.session_state.segments[k].update(status="queued", error=None)
    st.session_state.run_requested = True
pending = [k for k in active_keys if st.session_state.segments[k]["status"] in ("queued", "running")]
if pending and not st.session_state.run_requested:
    if b3.button(f"▶️ Resume {len(pending)} unfinished job(s)", use_container_width=True):
        st.session_state.run_requested = True

if st.session_state.run_requested:
    st.session_state.run_requested = False
    if run_queue(frame_by_num):
        st.rerun()

# ==========================================
# 8. STEP 4 — REVIEW EACH SEGMENT
# ==========================================
st.header("4️⃣ Review segments")
STATUS_ICON = {"idle": "⚪", "queued": "🕒", "running": "⏳", "done": "✅", "error": "❌"}

for idx, (first, last) in enumerate(pairs):
    a, b = first["num"], last["num"]
    key = seg_key(a, b)
    seg = st.session_state.segments[key]
    title = f"{STATUS_ICON[seg['status']]} Segment {idx + 1}: frame {a} → frame {b}"
    with st.expander(title, expanded=seg["status"] in ("done", "error")):
        left, right = st.columns([1, 2])
        with left:
            f1, f2 = st.columns(2)
            f1.image(processed_frame(first), caption=f"First: frame {a}", use_container_width=True)
            f2.image(processed_frame(last), caption=f"Last: frame {b}", use_container_width=True)
            st.caption("Prompt: " + (st.session_state.get(f"prompt_{a}_{b}", "").strip()
                                     or "_(empty — edit in step 2)_"))
            label = "🔁 Regenerate this segment" if seg["status"] == "done" else "🎬 Generate this segment"
            if st.button(label, key=f"gen_{key}", use_container_width=True,
                         disabled=seg["status"] in ("queued", "running")):
                seg.update(status="queued", error=None)
                st.session_state.run_requested = True
                st.rerun()

        with right:
            if seg["status"] == "error":
                st.error(seg["error"])
            if seg["candidates"]:
                takes = seg["candidates"]
                if len(takes) > 1:
                    seg["selected"] = st.radio(
                        "Keep take", list(range(len(takes))),
                        format_func=lambda i: f"Take {i + 1}", horizontal=True, key=f"sel_{key}",
                    )
                chosen = takes[seg["selected"]]
                st.video(chosen["path"])
                st.download_button(
                    "💾 Download this take", Path(chosen["path"]).read_bytes(),
                    f"segment_{idx + 1:02d}_frame{a}-{b}.mp4", mime="video/mp4", key=f"dl_{key}",
                )
            elif seg["status"] in ("queued", "running"):
                st.info("Rendering… press ▶️ Resume above if the progress panel disappeared.")
            elif seg["status"] == "idle":
                st.caption("Not generated yet.")

# --- Download every kept take in storyboard order ---
done_keys = [k for k in active_keys if st.session_state.segments[k]["candidates"]]
if done_keys:
    st.divider()
    if st.button(f"📦 Prepare ZIP of {len(done_keys)} kept take(s)"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for n, k in enumerate(active_keys, start=1):
                seg = st.session_state.segments[k]
                if seg["candidates"]:
                    a, b = k.split("->")
                    zf.write(seg["candidates"][seg["selected"]]["path"],
                             f"{n:02d}_frame{a}-{b}.mp4")
        st.session_state.zip_bytes = buf.getvalue()
    if st.session_state.zip_bytes:
        st.download_button("💾 Download ZIP", st.session_state.zip_bytes,
                           "storyboard_segments.zip", mime="application/zip")
