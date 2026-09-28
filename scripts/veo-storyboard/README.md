# Veo Storyboard → Video

Streamlit app that splits a storyboard image into frames and generates one Veo 3.1 video
per neighbouring pair of frames (1→2, 2→3, …), using them as first and last frame.

```
pip install streamlit google-genai google-cloud-storage pillow numpy
streamlit run geminiwebStoryboardVeo.py
```

Edit `KEY_FILE`, `PROJECT_ID`, `GCS_OUTPUT_URI` at the top of the script. Takes are also
saved locally under `storyboard_output/<timestamp>/`.
