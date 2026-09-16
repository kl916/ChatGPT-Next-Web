import { NextRequest, NextResponse } from "next/server";

// KMB Open Data API (published via data.gov.hk), see:
// https://data.gov.hk/en-data/dataset/hk-td-tis_21-etakmb
const KMB_BASE = "https://data.etabus.gov.hk/v1/transport/kmb";

async function fetchJson(url: string, revalidateSeconds?: number) {
  const res = await fetch(
    url,
    revalidateSeconds
      ? { next: { revalidate: revalidateSeconds } }
      : { cache: "no-store" },
  );
  if (!res.ok) {
    throw new Error(`KMB API responded with ${res.status}`);
  }
  return res.json();
}

export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl;
  const action = searchParams.get("action") ?? "eta";

  try {
    if (action === "stop") {
      const stopId = searchParams.get("stopId");
      if (!stopId) {
        return NextResponse.json({ error: "missing stopId" }, { status: 400 });
      }
      // Stop names/locations rarely change, cache for an hour.
      const data = await fetchJson(`${KMB_BASE}/stop/${stopId}`, 3600);
      return NextResponse.json(data);
    }

    const route = searchParams.get("route") ?? "962X";
    const serviceType = searchParams.get("serviceType") ?? "1";
    const data = await fetchJson(
      `${KMB_BASE}/route-eta/${encodeURIComponent(
        route,
      )}/${encodeURIComponent(serviceType)}`,
    );
    return NextResponse.json(data);
  } catch (e) {
    console.error("[bus-eta]", e);
    return NextResponse.json(
      { error: "Failed to fetch KMB data" },
      { status: 502 },
    );
  }
}
