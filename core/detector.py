"""Finds abnormal minutes and groups them into incidents.

Method: for each service and minute, compare error count and p95 latency with a
robust baseline (rolling median + MAD of the previous minutes). Several guards
stop healthy logs from producing false incidents: a minimum error count, a
minimum error rate, and a high z threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (BUCKET, MAX_INCIDENTS, MERGE_GAP_MIN, MIN_ERROR_RATE,
                    MIN_ERRORS_IN_BUCKET, MIN_LATENCY_JUMP_MS, ROLLING_WINDOW, Z_THRESHOLD)

ERR_LEVELS = {"ERROR", "FATAL"}


@dataclass
class Candidate:
    incident_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    services: list[str]
    first_service: str                  # service whose anomaly began earliest
    peak_error_rate: float
    signals: list[str] = field(default_factory=list)
    # Why each service was flagged, e.g. "latency spike only, no error spike (p95 3000ms vs ~45ms
    # normal)". A service that is slow but not erroring is invisible in an error timeline, so
    # the agents need this to know where to look.
    service_signals: dict[str, str] = field(default_factory=dict)


def minute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(minute=df["ts"].dt.floor(BUCKET), is_err=df["level"].isin(ERR_LEVELS))
    g = d.groupby(["service", "minute"])
    m = g.agg(events=("is_err", "size"), errors=("is_err", "sum"),
              p95_latency=("latency_ms", lambda s: s.quantile(0.95) if s.notna().any() else np.nan))
    m["error_rate"] = m["errors"] / m["events"]
    # Fill gaps so "no events" minutes exist as zeros.
    full = []
    for svc, part in m.groupby(level=0):
        part = part.droplevel(0)
        idx = pd.date_range(part.index.min(), part.index.max(), freq=BUCKET)
        part = part.reindex(idx)
        part[["events", "errors"]] = part[["events", "errors"]].fillna(0)
        part["error_rate"] = part["error_rate"].fillna(0)
        part["service"] = svc
        full.append(part.rename_axis("minute").reset_index())
    return pd.concat(full, ignore_index=True)


def _robust_z(x: pd.Series, floor: float) -> pd.Series:
    hist = x.shift(1).rolling(ROLLING_WINDOW, min_periods=3)
    med = hist.median().fillna(x.median())
    mad = hist.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True).fillna(0)
    return (x - med) / (1.4826 * mad + floor), med


def detect(df: pd.DataFrame) -> tuple[list[Candidate], pd.DataFrame]:
    """Return (incidents, per-minute metrics with an `anomaly` flag)."""
    m = minute_metrics(df)
    flags = []
    for svc, part in m.groupby("service"):
        part = part.sort_values("minute").copy()
        ez, _ = _robust_z(part["errors"], floor=1.0)
        part["error_z"] = ez
        err_flag = ((part["error_z"] >= Z_THRESHOLD)
                    & (part["errors"] >= MIN_ERRORS_IN_BUCKET)
                    & (part["error_rate"] >= MIN_ERROR_RATE))
        if part["p95_latency"].notna().sum() >= 5:
            lz, lmed = _robust_z(part["p95_latency"].ffill(), floor=max(1.0, part["p95_latency"].median() * 0.05))
            # Must be big in relative AND absolute terms: a 3ms cache going to 12ms is not an incident.
            lat_flag = ((lz >= Z_THRESHOLD) & (part["p95_latency"] > 3 * lmed)
                        & (part["p95_latency"] - lmed >= MIN_LATENCY_JUMP_MS) & (part["events"] >= 3))
        else:
            lat_flag = pd.Series(False, index=part.index)
        part["err_anomaly"], part["lat_anomaly"] = err_flag, lat_flag.fillna(False)
        part["anomaly"] = part["err_anomaly"] | part["lat_anomaly"]
        flags.append(part)
    m = pd.concat(flags, ignore_index=True)

    hits = m[m["anomaly"]].sort_values("minute")
    if hits.empty:
        return [], m

    # Merge flagged minutes (any service) into incident windows.
    groups, cur = [], [hits.iloc[0]]
    for _, row in hits.iloc[1:].iterrows():
        if (row["minute"] - cur[-1]["minute"]) <= pd.Timedelta(minutes=MERGE_GAP_MIN):
            cur.append(row)
        else:
            groups.append(cur)
            cur = [row]
    groups.append(cur)

    incidents = []
    for i, grp in enumerate(groups[:MAX_INCIDENTS]):
        g = pd.DataFrame(grp)
        first = g.sort_values(["minute", "error_z"], ascending=[True, False]).iloc[0]
        signals = []
        if g["err_anomaly"].any():
            signals.append("error spike")
        if g["lat_anomaly"].any():
            signals.append("latency spike")
        start, end = g["minute"].min(), g["minute"].max() + pd.Timedelta(minutes=1)
        incidents.append(Candidate(
            incident_id=f"DET-{i + 1:03d}", start=start, end=end,
            services=sorted(g["service"].unique().tolist()),
            first_service=str(first["service"]),
            peak_error_rate=float(g["error_rate"].max()), signals=signals,
            service_signals=_describe(g, m, start, end)))
    return incidents, m


def _describe(g: pd.DataFrame, m: pd.DataFrame, start, end) -> dict[str, str]:
    """One line per flagged service saying which signal fired, with numbers."""
    out = {}
    for svc, part in g.groupby("service"):
        err, lat = bool(part["err_anomaly"].any()), bool(part["lat_anomaly"].any())
        hist = m[(m["service"] == svc) & (m["minute"] < start)
                 & (m["minute"] >= start - pd.Timedelta(minutes=ROLLING_WINDOW))]
        win = m[(m["service"] == svc) & (m["minute"] >= start) & (m["minute"] < end)]
        normal, peak = hist["p95_latency"].median(), win["p95_latency"].max()
        latency = (f"p95 latency up to {peak:.0f}ms vs ~{normal:.0f}ms normal"
                   if pd.notna(peak) and pd.notna(normal) else "")
        errors = f"error rate up to {win['error_rate'].max():.0%}"
        if err and lat:
            out[svc] = f"error spike and latency spike ({errors}; {latency})"
        elif err:
            out[svc] = f"error spike ({errors})"
        else:
            out[svc] = f"latency spike only, NO error spike ({latency})"
    return out