"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import styles from "./bus-eta.module.scss";

const ROUTE = "962X";
const SERVICE_TYPE = "1";
const REFRESH_INTERVAL_MS = 30_000;

type Direction = "O" | "I";

interface EtaEntry {
  dir: Direction;
  seq: number;
  stop: string;
  dest_en: string;
  eta_seq: number;
  eta: string | null;
  rmk_en: string;
}

interface StopGroup {
  dir: Direction;
  seq: number;
  stop: string;
  dest_en: string;
  etas: { eta: string | null; rmk_en: string }[];
}

async function fetchEta(): Promise<EtaEntry[]> {
  const res = await fetch(
    `/api/bus-eta?route=${ROUTE}&serviceType=${SERVICE_TYPE}`,
  );
  const json = await res.json();
  if (!res.ok) throw new Error(json?.error ?? "failed to load ETA");
  return json.data ?? [];
}

async function fetchStopName(stopId: string): Promise<string> {
  const res = await fetch(`/api/bus-eta?action=stop&stopId=${stopId}`);
  const json = await res.json();
  if (!res.ok) throw new Error(json?.error ?? "failed to load stop");
  return json.data?.name_en ?? stopId;
}

function minutesFromNow(iso: string | null): string {
  if (!iso) return "-";
  const diffMs = new Date(iso).getTime() - Date.now();
  const minutes = Math.round(diffMs / 60000);
  if (minutes <= 0) return "Due";
  return `${minutes} min`;
}

export function BusEta() {
  const [direction, setDirection] = useState<Direction>("O");
  const [groups, setGroups] = useState<StopGroup[]>([]);
  const [stopNames, setStopNames] = useState<Record<string, string>>({});
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const stopNamesRef = useRef(stopNames);
  stopNamesRef.current = stopNames;

  const load = useCallback(async () => {
    try {
      const entries = await fetchEta();
      const byStop = new Map<string, StopGroup>();
      for (const entry of entries) {
        const key = `${entry.dir}-${entry.seq}-${entry.stop}`;
        const existing = byStop.get(key);
        if (existing) {
          existing.etas.push({ eta: entry.eta, rmk_en: entry.rmk_en });
        } else {
          byStop.set(key, {
            dir: entry.dir,
            seq: entry.seq,
            stop: entry.stop,
            dest_en: entry.dest_en,
            etas: [{ eta: entry.eta, rmk_en: entry.rmk_en }],
          });
        }
      }
      const sorted = Array.from(byStop.values()).sort(
        (a, b) => a.dir.localeCompare(b.dir) || a.seq - b.seq,
      );
      setGroups(sorted);
      setLastUpdated(new Date());
      setError(null);

      const missingStopIds = sorted
        .map((g) => g.stop)
        .filter((id) => !(id in stopNamesRef.current));
      const uniqueMissing = Array.from(new Set(missingStopIds));
      if (uniqueMissing.length > 0) {
        const names = await Promise.all(
          uniqueMissing.map(async (id) => {
            try {
              return [id, await fetchStopName(id)] as const;
            } catch {
              return [id, id] as const;
            }
          }),
        );
        setStopNames((prev) => {
          const next = { ...prev };
          for (const [id, name] of names) next[id] = name;
          return next;
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load ETA");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const timer = setInterval(load, REFRESH_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [load]);

  const visibleGroups = useMemo(
    () => groups.filter((g) => g.dir === direction),
    [groups, direction],
  );

  return (
    <div className={styles["bus-eta"]}>
      <div className={styles["bus-eta-header"]}>
        <h1>Route {ROUTE} Arrival Times</h1>
        <p className={styles["bus-eta-source"]}>
          Live data from{" "}
          <a
            href="https://data.gov.hk/en-data/dataset/hk-td-tis_21-etakmb"
            target="_blank"
            rel="noreferrer"
          >
            data.gov.hk
          </a>{" "}
          (KMB ETA)
        </p>
      </div>

      <div className={styles["bus-eta-tabs"]}>
        <button
          className={direction === "O" ? styles["active"] : ""}
          onClick={() => setDirection("O")}
        >
          Outbound
        </button>
        <button
          className={direction === "I" ? styles["active"] : ""}
          onClick={() => setDirection("I")}
        >
          Inbound
        </button>
        <button
          className={styles["bus-eta-refresh"]}
          onClick={() => {
            setLoading(true);
            load();
          }}
        >
          Refresh
        </button>
      </div>

      {error && <div className={styles["bus-eta-error"]}>{error}</div>}
      {loading && groups.length === 0 && !error && <div>Loading…</div>}

      <ul className={styles["bus-eta-list"]}>
        {visibleGroups.map((group) => (
          <li key={`${group.dir}-${group.seq}`} className={styles["bus-eta-stop"]}>
            <div className={styles["bus-eta-stop-name"]}>
              <span className={styles["bus-eta-seq"]}>{group.seq}</span>
              {stopNames[group.stop] ?? group.stop}
            </div>
            <div className={styles["bus-eta-times"]}>
              {group.etas
                .slice()
                .sort((a, b) => (a.eta ?? "").localeCompare(b.eta ?? ""))
                .map((e, i) => (
                  <span key={i} className={styles["bus-eta-time"]}>
                    {minutesFromNow(e.eta)}
                    {e.rmk_en ? ` (${e.rmk_en})` : ""}
                  </span>
                ))}
              {group.etas.every((e) => !e.eta) && (
                <span className={styles["bus-eta-time"]}>No ETA data</span>
              )}
            </div>
          </li>
        ))}
        {!loading && visibleGroups.length === 0 && !error && (
          <li>No stops found for this direction.</li>
        )}
      </ul>

      {lastUpdated && (
        <div className={styles["bus-eta-updated"]}>
          Last updated {lastUpdated.toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}
