"""Synthetic log generator with deliberately injected incidents.

Produces realistic-looking service logs plus a hidden answer key, so the system
can be evaluated: we know exactly which incidents exist and what caused them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd

from config import CANONICAL_COLUMNS

SERVICES = {
    "api-gateway":     {"endpoints": ["/home", "/search", "/cart"], "lat": 40},
    "auth-service":    {"endpoints": ["/login", "/token/refresh"], "lat": 35},
    "payment-service": {"endpoints": ["/pay", "/refund"], "lat": 120},
    "db-service":      {"endpoints": ["query", "write"], "lat": 15},
    "cache-service":   {"endpoints": ["get", "set"], "lat": 3},
}

# What each incident looks like. `root` is the true cause; `cascade` are victims.
INCIDENT_TYPES = {
    "db_timeout": {
        "root": "db-service", "cascade": ["payment-service", "api-gateway"],
        "root_msg": "connection timeout after 3000ms (pool exhausted 50/50)",
        "cascade_msg": "upstream db-service timed out", "lat": 3000, "status": 500,
    },
    "memory_leak": {
        "root": "payment-service", "cascade": ["api-gateway"],
        "root_msg": "java.lang.OutOfMemoryError: Java heap space",
        "cascade_msg": "upstream payment-service returned 503", "lat": 2500, "status": 503,
    },
    "auth_failure": {
        "root": "auth-service", "cascade": ["api-gateway"],
        "root_msg": "token validation failed: signing certificate expired",
        "cascade_msg": "auth check failed for request", "lat": 60, "status": 401,
    },
    "disk_full": {
        "root": "db-service", "cascade": ["payment-service"],
        "root_msg": "write failed: No space left on device",
        "cascade_msg": "upstream db-service write rejected", "lat": 40, "status": 507,
    },
    "cache_down": {
        "root": "cache-service", "cascade": ["db-service", "api-gateway"],
        "root_msg": "connection refused 10.0.3.12:6379",
        "cascade_msg": "cache miss storm, falling back to db", "lat": 900, "status": 500,
    },
    # Adversarial case: built to defeat the "victims name the cause" hint.
    # auth-service is CPU-throttled and becomes slow but logs only WARNINGS, so it
    # never shows up in an error timeline. payment-service holds database locks
    # while it waits for auth, so ITS errors blame the database, and the gateway's
    # errors blame payment-service. Following the names leads to the wrong answer
    # (db-service or payment-service). db-service itself stays healthy. Only the
    # latency metrics of auth-service, plus noticing that the "tx held" time
    # matches auth's latency, reveal the real cause.
    "auth_slowdown": {
        "root": "auth-service", "cascade": ["payment-service", "api-gateway"],
        "root_level": "WARN", "status": 200, "lat": 2900,
        "root_msg": "token introspection slow (cpu throttled to 20%)",
        "cascade_msg": "request failed",
        "cascade_msgs": {
            "payment-service": "db transaction rolled back: lock wait timeout (tx held 3100ms)",
            "api-gateway": "upstream payment-service returned 504",
        },
    },
}


LEAD_MINUTES = {"memory_leak": 10}       # warning signs logged before the incident starts


def plan_incidents(types: list[str], start: int, duration: int, minutes: int = 120,
                   min_gap: int = 15) -> tuple[list[dict], str | None]:
    """Place incidents evenly from `start` to the end of the logs, or explain why they can't fit.

    Returns (incidents, problem). `problem` is None when the plan is valid; otherwise
    it says what to change, and incidents is empty: nothing is moved or shortened
    silently. min_gap is the quiet time kept between incidents (default 15 min = the
    detector's history window), so neighbouring incidents never merge into one and
    each is judged against clean "normal" traffic.
    """
    if not types:
        return [], None
    n, end_limit = len(types), minutes - 2
    gap = min_gap + max(LEAD_MINUTES.get(t, 0) for t in types)
    hhmm = lambda m: f"{10 + m // 60}:{m % 60:02d}"                           # noqa: E731
    if n == 1:
        if start + duration > end_limit:
            return [], (f"An incident starting at {hhmm(start)} can last at most {end_limit - start} "
                        f"minutes before the logs end at {hhmm(end_limit)}. Shorten it or start earlier.")
        return [{"type": types[0], "at_minute": start, "duration": duration}], None
    step = (end_limit - duration - start) // (n - 1)
    if step < duration + gap:
        latest = end_limit - n * duration - (n - 1) * gap
        longest = (end_limit - start - (n - 1) * gap) // n
        fixes = []
        if latest >= 5:
            fixes.append(f"start at {hhmm(latest)} or earlier")
        if longest >= 2:
            fixes.append(f"use a duration of {longest} min or less")
        fixes.append("choose fewer incidents")
        return [], (f"{n} incidents of {duration} min need {gap} quiet minutes between them and "
                    f"don't fit between {hhmm(start)} and {hhmm(end_limit)}. To fix: "
                    + ", or ".join(fixes) + ".")
    return [{"type": t, "at_minute": start + i * step, "duration": duration}
            for i, t in enumerate(types)], None


@dataclass
class Incident:
    incident_id: str
    type: str
    root_service: str
    start: str
    end: str


def generate_logs(
    minutes: int = 120,
    events_per_min: int = 60,
    incidents: list[dict] | None = None,
    start: datetime | None = None,
    seed: int = 42,
    difficulty: str = "easy",
) -> tuple[pd.DataFrame, list[Incident]]:
    """Return (logs_df, answer_key).

    incidents: list of {"type": str, "at_minute": int, "duration": int}. Invalid
    entries are clamped or skipped, never raised, so user-driven input can't crash it.

    difficulty="hard": victims fail in the SAME minute as the cause and log MORE
    errors than it, so "first and loudest" no longer identifies the cause. Only
    reading the messages ("upstream db-service timed out") does.
    """
    hard = difficulty == "hard"
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 10, 14, 10, 0, 0)
    minutes = int(np.clip(minutes, 10, 24 * 60))
    events_per_min = int(np.clip(events_per_min, 5, 500))

    rows = []
    names = list(SERVICES)
    for m in range(minutes):
        n = rng.poisson(events_per_min)
        secs = np.sort(rng.uniform(0, 60, n))
        for s in secs:
            svc = names[rng.integers(len(names))]
            spec = SERVICES[svc]
            lat = max(1, int(rng.lognormal(np.log(spec["lat"]), 0.35)))
            is_err = rng.random() < 0.01          # 1% background noise
            rows.append({
                "ts": start + timedelta(minutes=m, seconds=float(s)),
                "level": "ERROR" if is_err else ("WARN" if rng.random() < 0.02 else "INFO"),
                "service": svc,
                "endpoint": spec["endpoints"][rng.integers(len(spec["endpoints"]))],
                "status": 500 if is_err else 200,
                "latency_ms": lat,
                "msg": "transient error, retried ok" if is_err else "ok",
            })

    key: list[Incident] = []
    for i, inc in enumerate(incidents or []):
        itype = inc.get("type")
        if itype not in INCIDENT_TYPES:
            continue
        spec = INCIDENT_TYPES[itype]
        at = int(np.clip(inc.get("at_minute", minutes // 2), 1, minutes - 2))
        dur = int(np.clip(inc.get("duration", 8), 2, max(2, minutes - at)))
        t0 = start + timedelta(minutes=at)
        t1 = t0 + timedelta(minutes=dur)
        key.append(Incident(f"INC-{i + 1:03d}", itype, spec["root"],
                            t0.isoformat(), t1.isoformat()))

        # Memory leak: latency creeps up for 10 minutes *before* the crash.
        if itype == "memory_leak":
            for k in range(min(10, at)):
                ts = t0 - timedelta(minutes=10 - k)
                for _ in range(6):
                    rows.append({"ts": ts + timedelta(seconds=float(rng.uniform(0, 60))),
                                 "level": "WARN", "service": spec["root"], "endpoint": "/pay",
                                 "status": 200, "latency_ms": int(200 + k * 150),
                                 "msg": f"GC pause {80 + k * 40}ms, heap {70 + k * 3}%"})

        for minute in range(dur):
            base = t0 + timedelta(minutes=minute)
            # Root cause errors start first.
            for _ in range(int(rng.integers(8, 14) if hard else rng.integers(15, 30))):
                rows.append({"ts": base + timedelta(seconds=float(rng.uniform(0, 60))),
                             "level": spec.get("root_level", "ERROR"), "service": spec["root"],
                             "endpoint": SERVICES[spec["root"]]["endpoints"][0],
                             "status": spec["status"],
                             "latency_ms": int(rng.normal(spec["lat"], spec["lat"] * 0.1)) + 1,
                             "msg": spec["root_msg"]})
            # Victims follow ~1 minute later; this ordering is what lets the
            # agents tell cause from symptom.
            if minute >= (0 if hard else 1):
                for victim in spec["cascade"]:
                    for _ in range(int(rng.integers(18, 30) if hard else rng.integers(8, 18))):
                        rows.append({"ts": base + timedelta(seconds=float(rng.uniform(0, 60))),
                                     "level": "ERROR", "service": victim,
                                     "endpoint": SERVICES[victim]["endpoints"][0],
                                     "status": 502, "latency_ms": int(rng.normal(1500, 200)) + 1,
                                     "msg": spec.get("cascade_msgs", {}).get(victim, spec["cascade_msg"])})

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS).sort_values("ts").reset_index(drop=True)
    return df, key


def to_jsonl(df: pd.DataFrame) -> str:
    out = df.copy()
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return "\n".join(json.dumps(r) for r in out.to_dict("records"))


def to_text(df: pd.DataFrame) -> str:
    """Plain format: 2026-10-14 10:02:11 INFO auth-service /login 200 45ms ok"""
    return "\n".join(
        f"{r.ts:%Y-%m-%d %H:%M:%S} {r.level} {r.service} {r.endpoint} {r.status} {r.latency_ms}ms {r.msg}"
        for r in df.itertuples()
    )


def key_to_df(key: list[Incident]) -> pd.DataFrame:
    return pd.DataFrame([asdict(k) for k in key],
                        columns=["incident_id", "type", "root_service", "start", "end"])