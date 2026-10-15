"""Read-only tools the agents can call. They never modify data and always return
small, bounded text, so neither a large log nor a confused agent can blow up
memory or the LLM context window."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from config import MAX_TOOL_RESULT_LINES, MIN_ERRORS_IN_BUCKET

RUNBOOK_PATH = Path(__file__).resolve().parent.parent / "data" / "runbook.md"
ERR = {"ERROR", "FATAL"}


def load_runbook(path: Path = RUNBOOK_PATH) -> list[dict]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    entries = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        title, _, body = block.partition("\n")
        fields = dict(re.findall(r"^(\w+):\s*(.+)$", body, flags=re.M))
        entries.append({"title": title.strip(), **{k.lower(): v for k, v in fields.items()}})
    return entries


class Toolbox:
    def __init__(self, df: pd.DataFrame, runbook: list[dict] | None = None):
        self.df = df
        self.runbook = runbook if runbook is not None else load_runbook()
        self.services = sorted(df["service"].unique().tolist())

    # -- helpers -----------------------------------------------------------
    def _window(self, start=None, end=None) -> pd.DataFrame:
        d = self.df
        try:
            if start:
                d = d[d["ts"] >= pd.Timestamp(start)]
            if end:
                d = d[d["ts"] <= pd.Timestamp(end)]
        except (ValueError, TypeError):
            pass                                   # bad time from the LLM: ignore the filter
        return d

    @staticmethod
    def _fmt(row) -> str:
        lat = "" if pd.isna(row.latency_ms) else f" {int(row.latency_ms)}ms"
        st = "" if pd.isna(row.status) else f" {int(row.status)}"
        return f"{row.ts:%H:%M:%S} {row.level} {row.service}{st}{lat} {row.msg[:160]}"

    # -- tools -------------------------------------------------------------
    def search_logs(self, service: str | None = None, start: str | None = None,
                    end: str | None = None, level: str | None = None,
                    contains: str | None = None, limit: int = 15) -> str:
        d = self._window(start, end)
        if service:
            if service not in self.services:
                return f"Unknown service '{service}'. Known services: {', '.join(self.services)}"
            d = d[d["service"] == service]
        if level:
            d = d[d["level"] == str(level).upper()]
        if contains:
            d = d[d["msg"].str.contains(re.escape(str(contains)[:100]), case=False, na=False)]
        limit = max(1, min(int(limit or 15), MAX_TOOL_RESULT_LINES))
        if d.empty:
            return "No matching log lines."
        lines = [self._fmt(r) for r in d.head(limit).itertuples()]
        return f"{len(d)} matching lines; first {len(lines)}:\n" + "\n".join(lines)

    def burst_starts(self, start=None, end=None) -> dict[str, pd.Timestamp]:
        """First error of each service's burst (first minute with >= MIN_ERRORS_IN_BUCKET errors)."""
        d = self._window(start, end)
        e = d[d["level"].isin(ERR)]
        out = {}
        for svc, part in e.groupby("service"):
            per_min = part.groupby(part["ts"].dt.floor("1min")).size()
            bursty = per_min[per_min >= MIN_ERRORS_IN_BUCKET]
            if not bursty.empty:
                out[svc] = part[part["ts"] >= bursty.index[0]]["ts"].min()
        return out

    def service_timeline(self, start: str | None = None, end: str | None = None) -> str:
        """When did each service's error BURST start, and with what message?

        Normal traffic contains scattered background errors. Reporting the very
        first error would point at that noise (e.g. a stray error 15 minutes
        before the outage), so the start is the first minute with at least
        MIN_ERRORS_IN_BUCKET errors; earlier stray errors are counted separately.
        """
        d = self._window(start, end)
        e = d[d["level"].isin(ERR)]
        if e.empty:
            return "No ERROR/FATAL events in this window."
        starts = self.burst_starts(start, end)
        rows = []
        for svc, part in e.groupby("service"):
            total = len(d[d["service"] == svc])
            if svc not in starts:
                rows.append((pd.Timestamp.max, f"{svc}: no error burst, only {len(part)} scattered "
                                               "background error(s)"))
                continue
            began = starts[svc]
            in_burst = part[part["ts"] >= began]
            stray = len(part) - len(in_burst)
            top = in_burst["msg"].value_counts().index[0][:120]
            rows.append((began, f"{svc}: error burst began {began:%H:%M:%S}, {len(in_burst)} errors "
                                f"({len(in_burst) / max(total, 1):.0%} of its events), top message: "
                                f"\"{top}\"" + (f"; {stray} scattered error(s) earlier, ignore as noise"
                                                if stray else "")))
        rows.sort(key=lambda r: r[0])
        return "\n".join(text for _, text in rows)

    def get_metrics(self, service: str, start: str | None = None, end: str | None = None) -> str:
        if service not in self.services:
            return f"Unknown service '{service}'. Known services: {', '.join(self.services)}"
        d = self._window(start, end)
        d = d[d["service"] == service]
        if d.empty:
            return "No events for this service in the window."
        g = d.set_index("ts").resample("1min").agg(
            {"level": lambda s: s.isin(ERR).sum(), "latency_ms": lambda s: s.quantile(.95)})
        g = g.tail(MAX_TOOL_RESULT_LINES)
        return "minute | errors | p95 latency ms\n" + "\n".join(
            f"{t:%H:%M} | {int(r.level)} | {'-' if pd.isna(r.latency_ms) else int(r.latency_ms)}"
            for t, r in g.iterrows())

    def search_runbook(self, query: str) -> str:
        words = set(re.findall(r"[a-z]{3,}", str(query).lower()))
        if not words or not self.runbook:
            return "No runbook entries matched."
        scored = []
        for e in self.runbook:
            hay = " ".join(e.values()).lower()
            scored.append((sum(w in hay for w in words), e))
        scored.sort(key=lambda x: -x[0])
        hits = [e for s, e in scored[:2] if s > 0]
        if not hits:
            return "No runbook entries matched."
        return "\n\n".join(f"[{e['title']}] category={e.get('category', '?')}\n"
                           f"Symptoms: {e.get('symptoms', '')}\nConfirm: {e.get('confirm', '')}\n"
                           f"Fix: {e.get('fix', '')}" for e in hits)

    # registry used by the agent loop (name -> (callable, allowed args))
    def registry(self):
        return {
            "search_logs": (self.search_logs, {"service", "start", "end", "level", "contains", "limit"}),
            "service_timeline": (self.service_timeline, {"start", "end"}),
            "get_metrics": (self.get_metrics, {"service", "start", "end"}),
            "search_runbook": (self.search_runbook, {"query"}),
        }

    def call(self, name: str, args: dict | None) -> str:
        reg = self.registry()
        if name not in reg:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(reg)}"
        fn, allowed = reg[name]
        args = {k: v for k, v in (args or {}).items() if k in allowed}
        try:
            return fn(**args)
        except Exception as e:                     # a tool must never crash the loop
            return f"ERROR: tool failed ({type(e).__name__}: {str(e)[:100]})"