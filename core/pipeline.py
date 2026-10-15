"""One entry point used by the app, the tests and the CLI."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from agents.triage import TriageResult, triage
from core.detector import Candidate, detect
from core.llm import LLM
from core.tools import Toolbox


@dataclass
class RunResult:
    df: pd.DataFrame
    metrics: pd.DataFrame
    candidates: list[Candidate]
    results: list[TriageResult] = field(default_factory=list)

    @property
    def no_incident(self) -> bool:
        return not self.candidates

    def summary(self) -> str:
        if self.no_incident:
            err = self.df["level"].isin({"ERROR", "FATAL"}).mean()
            return (f"No incident detected. {len(self.df):,} events across "
                    f"{self.df['service'].nunique()} services from {self.df['ts'].min():%H:%M} to "
                    f"{self.df['ts'].max():%H:%M}; overall error rate {err:.1%}, with no abnormal spike.")
        return f"{len(self.candidates)} incident(s) detected."


def run_pipeline(df: pd.DataFrame, llm: LLM, on_step=None) -> RunResult:
    candidates, metrics = detect(df)
    run = RunResult(df, metrics, candidates)
    if run.no_incident:                       # never invent a root cause for healthy logs
        return run
    tb = Toolbox(df)
    for cand in candidates:
        if on_step:
            on_step({"agent": "System", "kind": "system",
                     "text": f"--- {cand.incident_id}: {cand.start:%H:%M}-{cand.end:%H:%M}, "
                             f"anomalous: {', '.join(cand.services)} ---"})
        run.results.append(triage(cand, tb, llm, metrics, on_step))
    return run


def evaluate(results: list[TriageResult], key_df: pd.DataFrame, tolerance_min: int = 5) -> dict:
    """Compare detections with the injected answer key."""
    tol = pd.Timedelta(minutes=tolerance_min)
    key = key_df.assign(start=pd.to_datetime(key_df["start"]), end=pd.to_datetime(key_df["end"]))
    rows, matched_det = [], set()
    for _, k in key.iterrows():
        hit = next((r for r in results if r.candidate.start <= k["end"] + tol
                    and r.candidate.end >= k["start"] - tol
                    and r.candidate.incident_id not in matched_det), None)
        if hit:
            matched_det.add(hit.candidate.incident_id)
        rows.append({"injected": k["incident_id"], "type": k["type"], "true_root": k["root_service"],
                     "detected_as": hit.candidate.incident_id if hit else "-",
                     "predicted_root": hit.conclusion.root_service if hit else "-",
                     "root_correct": bool(hit and hit.conclusion.root_service == k["root_service"])})
    table = pd.DataFrame(rows)
    n = len(key)
    return {
        "table": table,
        "injected": n,
        "detected": int((table["detected_as"] != "-").sum()) if n else 0,
        "root_correct": int(table["root_correct"].sum()) if n else 0,
        "false_alarms": len(results) - len(matched_det),
    }
