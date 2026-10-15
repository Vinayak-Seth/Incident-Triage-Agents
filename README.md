# Incident triage agents

Upload server logs (or describe an outage in plain English) and two collaborating
LLM agents find the root cause: an **investigator** that uses tools to dig through
the logs, and a **verifier** that fact-checks it and sends objections back. The
result is a verified incident report with evidence and a recommended fix.

```
logs ──► parser ──► anomaly detector ──► investigator ⇄ verifier ──► report
        (any format)   (robust z-score)     (Gemini + read-only tools)
```

## Run it (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-key"        # optional: without it the app runs in offline mode
streamlit run app.py
```

Run the tests (no API key needed, they use a scripted fake LLM):

```powershell
python -m pytest -q
```

## Input modes

| Mode | What the user gives | Answer key? |
|---|---|---|
| Simulate incidents | pick incident types, time, difficulty | yes, accuracy shown |
| Describe a scenario | "payment service runs out of memory at 11 am" | yes |
| Upload a log file | .log / .txt / .jsonl / .csv, up to 5 MB | no |
| Paste log lines | at least 20 lines | no |

Accepted log formats: our plain text, JSON lines, CSV with a header, and unknown
formats (syslog, HDFS-style, etc.) via the format mapper.

## Project layout

```
app.py               Streamlit UI
config.py            every limit and threshold in one place
core/generator.py    synthetic logs + injected incidents + answer key
core/parser.py       any input -> canonical table, or a clear error
core/detector.py     per-minute anomaly detection, groups minutes into incidents
core/tools.py        read-only tools the agents call
core/llm.py          Gemini wrapper: retries, timeouts, offline switch
core/pipeline.py     orchestration + evaluation against the answer key
agents/triage.py     investigator, verifier, debate loop, report
agents/assist.py     scenario parser, format mapper, Q&A
data/runbook.md      known incident patterns the agents search
tests/               59 tests: bad inputs, agent failures, the app itself
```

## Deploy (Streamlit Community Cloud)

Push to GitHub, create an app pointing at `app.py`, and add `GEMINI_API_KEY`
under App settings → Secrets. Rehearse on the deployed URL, not localhost.
