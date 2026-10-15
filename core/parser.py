"""Turns ANY user input into the canonical event table, or fails with a clear message.

Pipeline:  raw bytes -> safe text -> detect format -> parse -> normalise -> validate

Supported out of the box: JSON lines, CSV with a header, our own plain-text format.
Anything else goes to the *format mapper*: the LLM labels which whitespace token
holds the time, level and service. The LLM never writes code or regex; it only
returns token positions, which we apply ourselves and then verify on a sample.
If the LLM is unavailable or wrong, a heuristic guesser does the same job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import re
import warnings

import pandas as pd

from config import (CANONICAL_COLUMNS, LEVELS, MAX_LINE_CHARS, MAX_LINES,
                    MAX_PASTE_CHARS, MAX_UPLOAD_BYTES, MIN_EVENTS)


class InputError(Exception):
    """Raised with a message that is safe and useful to show the user."""


@dataclass
class ParseResult:
    df: pd.DataFrame
    fmt: str
    notes: list[str] = field(default_factory=list)
    dropped: int = 0


EXPECTED_HELP = (
    "Expected one event per line, e.g.\n"
    "  2026-10-14 10:02:11 INFO auth-service /login 200 45ms ok\n"
    'or JSON lines: {"ts": "2026-10-14T10:02:11", "level": "ERROR", "service": "db-service", "msg": "..."}'
)

# ----------------------------------------------------------------- 1. safe text
def read_input(raw: bytes | str, is_paste: bool = False) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    if raw is None or len(raw) == 0:
        raise InputError("The input is empty. " + EXPECTED_HELP)

    if isinstance(raw, str):
        if is_paste and len(raw) > MAX_PASTE_CHARS:
            raw = raw[:MAX_PASTE_CHARS]
            notes.append(f"Pasted text was longer than {MAX_PASTE_CHARS:,} characters and was cut.")
        text = raw
    else:
        if len(raw) > MAX_UPLOAD_BYTES:
            raise InputError(f"File is {len(raw) / 1e6:.1f} MB; the limit is "
                             f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB. Upload a smaller slice of the log.")
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):        # UTF-16 BOM: contains NULs legitimately
            raw = raw.decode("utf-16").encode("utf-8")
        sample = raw[:4096]
        if sample.count(b"\x00") > len(sample) * 0.01:
            raise InputError("This looks like a binary file (image, zip, exe), not a text log.")
        for enc in ("utf-8-sig", "utf-16"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("latin-1")   # never fails; odd characters survive as-is
            notes.append("File was not UTF-8; decoded as Latin-1.")

    lines = [ln[:MAX_LINE_CHARS] for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise InputError("The input only contains blank lines. " + EXPECTED_HELP)
    if len(lines) > MAX_LINES:
        notes.append(f"Only the last {MAX_LINES:,} of {len(lines):,} lines were analysed.")
        lines = lines[-MAX_LINES:]
    return lines, notes


# ------------------------------------------------------------- 2. timestamps
_TS_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S,%f",
               "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f", "%y%m%d %H%M%S",
               "%d/%b/%Y:%H:%M:%S", "%Y/%m/%d %H:%M:%S",
               "%a %b %d %H:%M:%S %Y",          # Apache error log: [Sun Dec 04 04:47:44 2005]
               "%y/%m/%d %H:%M:%S",             # Spark: 17/06/09 20:10:40
               "%Y-%m-%d-%H.%M.%S.%f"]          # BGL: 2005-06-03-15.42.50.675872
_EPOCH = re.compile(r"^\d{10}(?:\.\d+)?$")       # unix seconds, e.g. 1117838570


def parse_ts(s: pd.Series, loose: bool = True) -> pd.Series:
    """Try known formats, keep the one that parses most rows; never raises.

    loose=True also lets pandas guess any other format. That's right for a column
    already known to hold timestamps (JSON/CSV), but wrong when guessing which
    words of a raw line are the timestamp: a lenient guesser happily reads "Dec"
    or "081109" alone as a date, which silently throws away the time of day.
    """
    s = s.astype(str).str.strip().str.strip("[]")
    best, best_ok = pd.Series(pd.NaT, index=s.index), -1.0
    # f=f binds each format now; a bare lambda would see only the last one in the list.
    candidates = [lambda f=f: pd.to_datetime(s, format=f, errors="coerce") for f in _TS_FORMATS]
    candidates += [
        # syslog "Dec 10 06:55:46" has no year: pin one (strptime can't parse it alone)
        lambda: pd.to_datetime("2026 " + s, format="%Y %b %d %H:%M:%S", errors="coerce"),
        lambda: pd.to_datetime(pd.to_numeric(s.where(s.str.match(_EPOCH)), errors="coerce"),
                               unit="s", errors="coerce"),
    ]
    for make in candidates:
        out = make()
        ok = out.notna().mean()
        if ok > best_ok:
            best, best_ok = out, ok
        if ok > 0.95:
            break
    if best_ok < 0.6 and loose:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
        if out.notna().mean() > best_ok:
            best = out
    if getattr(best.dt, "tz", None) is not None:
        best = best.dt.tz_convert(None)
    return best


# ------------------------------------------------------------- 3. normalise
_ALIASES = {
    "ts": ["ts", "timestamp", "time", "@timestamp", "datetime", "date", "t"],
    "level": ["level", "severity", "lvl", "loglevel", "log_level", "priority"],
    "service": ["service", "app", "component", "source", "logger", "host", "service_name"],
    "endpoint": ["endpoint", "path", "url", "route", "uri", "method"],
    "status": ["status", "status_code", "code", "http_status"],
    "latency_ms": ["latency_ms", "latency", "duration_ms", "duration", "response_time", "elapsed_ms"],
    "msg": ["msg", "message", "text", "log", "event", "content"],
}
_LEVEL_MAP = {"WARNING": "WARN", "ERR": "ERROR", "CRITICAL": "FATAL", "CRIT": "FATAL",
              "EMERG": "FATAL", "ALERT": "FATAL", "SEVERE": "ERROR", "NOTICE": "INFO",
              "TRACE": "DEBUG", "E": "ERROR", "W": "WARN", "I": "INFO", "D": "DEBUG", "F": "FATAL"}
_ERROR_WORDS = re.compile(r"error|exception|fail|refused|timeout|timed out|denied|panic|"
                          r"no space|out ?of ?memory|unavailable|fatal", re.I)


def _norm_level(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    v = re.sub(r"[^A-Za-z]", "", str(v)).upper()
    v = _LEVEL_MAP.get(v, v)
    return v if v in LEVELS else None


def clean_service(v) -> str:
    """Turn a raw component token into a stable service name.

    sshd[24200]:  -> sshd            (process ids differ per line)
    [QuorumPeer[myid=1]/0:0:0:2181:FastLeaderElection@774] -> FastLeaderElection
    org.apache.hadoop.mapreduce.v2.app.MRAppMaster: -> MRAppMaster
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "unknown"
    s = str(v).strip().rstrip(":")
    s = re.sub(r"\[\d+\]", "", s)              # [pid]
    s = re.sub(r"@\d+\]?$", "", s)             # @lineNumber
    s = s.strip("[]")
    if ":" in s or "/" in s:                   # thread/address prefixes: keep the last part
        parts = re.split(r"[:/]", s)
        if re.search(r"[A-Za-z]", parts[-1]):
            s = parts[-1]
        elif re.fullmatch(r"[A-Za-z][\w$]*", parts[0]):   # "LeaseRenewer:user@host:9000"
            s = parts[0]
    if s.count(".") >= 2 and re.fullmatch(r"[\w.$]+", s):   # java package path
        s = s.rsplit(".", 1)[-1]
    return s or "unknown"


def normalise(df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    notes: list[str] = []
    lower = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for canon, names in _ALIASES.items():
        src = next((lower[n] for n in names if n in lower), None)
        out[canon] = df[src] if src is not None else None

    if out["ts"].isna().all():
        raise InputError("Could not find a timestamp in the input. " + EXPECTED_HELP)
    out["ts"] = parse_ts(out["ts"])
    out["msg"] = out["msg"].fillna("").astype(str)
    out["level"] = out["level"].map(_norm_level)
    missing_lvl = out["level"].isna()
    if missing_lvl.any():
        # No level column: infer from message wording.
        out.loc[missing_lvl, "level"] = out.loc[missing_lvl, "msg"].map(
            lambda m: "ERROR" if _ERROR_WORDS.search(m) else "INFO")
        if missing_lvl.mean() > 0.5:
            notes.append("No log level found; ERROR/INFO was inferred from message text.")
    out["service"] = out["service"].map(clean_service)
    out["endpoint"] = out["endpoint"].fillna("").astype(str)
    out["status"] = pd.to_numeric(out["status"], errors="coerce")
    lat = out["latency_ms"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
    out["latency_ms"] = pd.to_numeric(lat, errors="coerce")

    before = len(out)
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    dropped = before - len(out)
    return out[CANONICAL_COLUMNS], dropped, notes


# ------------------------------------------------------------- 4. formats
_PLAIN = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+(?P<level>[A-Z]+)\s+(?P<service>\S+)\s+"
    r"(?P<endpoint>\S+)\s+(?P<status>\d{3})\s+(?P<latency_ms>\d+)ms\s*(?P<msg>.*)$")


def _try_jsonl(lines):
    rows, bad = [], 0
    for ln in lines:
        try:
            obj = json.loads(ln)
            rows.append(obj) if isinstance(obj, dict) else None
            bad += not isinstance(obj, dict)
        except (json.JSONDecodeError, ValueError):
            bad += 1
    return (pd.json_normalize(rows) if rows else None), bad


def _try_csv(lines):
    # The first line must BE column names, one of which is a timestamp column.
    # (Checking "contains a comma and the letter t" mistook Zookeeper/Hadoop lines,
    # whose timestamps contain commas, for CSV.)
    fields = [f.strip().strip('"').lower() for f in lines[0].split(",")]
    if len(fields) < 2 or not any(f in _ALIASES["ts"] for f in fields):
        return None
    try:
        return pd.read_csv(io.StringIO("\n".join(lines)), on_bad_lines="skip", dtype=str)
    except Exception:
        return None


def _apply_token_map(lines: list[str], m: dict) -> pd.DataFrame:
    """Apply a token-position mapping (from the LLM or the heuristic)."""
    ts_idx, lvl, svc, msg_from = m.get("ts_tokens", []), m.get("level"), m.get("service"), m.get("msg_from")
    rows = []
    for ln in lines:
        tok = ln.split()
        def get(i):
            return tok[i] if isinstance(i, int) and 0 <= i < len(tok) else None
        rest = " ".join(tok[msg_from:]) if isinstance(msg_from, int) else ln
        lat = re.search(r"(\d+(?:\.\d+)?)\s?ms\b", ln)
        st = re.search(r"\s([1-5]\d\d)\s", f" {rest} ")
        rows.append({
            "ts": " ".join(t for t in (get(i) for i in ts_idx) if t),
            "level": get(lvl), "service": get(svc), "msg": rest,
            "latency_ms": lat.group(1) if lat else None,
            "status": st.group(1) if st else None,
        })
    return pd.DataFrame(rows)


def guess_token_map(lines: list[str]) -> dict:
    """Heuristic stand-in for the LLM format mapper: which whitespace token is what."""
    sample = [ln.split() for ln in lines[:200]]
    width = min(12, min((len(t) for t in sample), default=0))
    if width == 0:
        return {"ts_tokens": [], "level": None, "service": None, "msg_from": 0}
    col = lambda i: [t[i] for t in sample]                       # noqa: E731
    lvl_idx = next((i for i in range(width)
                    if sum(_norm_level(x) is not None for x in col(i)) / len(sample) > 0.6), None)

    # Timestamp: try every run of 1-5 neighbouring tokens before the level. Among runs
    # that parse on >=80% of lines, keep the one giving the most DISTINCT times (the
    # finest resolution), so "Dec" alone never beats "Dec 10 06:55:46".
    limit = lvl_idx if lvl_idx is not None else min(width, 8)
    best_ts, best_key = [], (0.0, 0, 0, 0)
    for i in range(limit):
        for n in range(1, 6):
            combo = list(range(i, i + n))
            if combo[-1] >= limit:
                break
            joined = pd.Series([" ".join(t[j] for j in combo) for t in sample])
            parsed = parse_ts(joined, loose=False)
            ok = parsed.notna().mean()
            readable = 0 if joined.str.match(_EPOCH).mean() > 0.5 else 1   # prefer dates over epoch
            # ties go to the longer run, so "Sun Dec 04 04:47:44 2005" keeps its year
            key = (1.0 if ok >= 0.8 else ok, parsed.nunique(), n, readable)
            if key > best_key:
                best_ts, best_key = combo, key
    used = set(best_ts) | ({lvl_idx} if lvl_idx is not None else set())

    def is_name(i):          # mostly words, not numbers or punctuation
        return i < width and i not in used and sum(bool(re.search(r"[A-Za-z]", x)) for x in col(i)) / len(sample) > 0.8

    svc_idx = None
    # 1. A low-variety word right before the level is a category/component (BGL: "KERNEL INFO").
    if lvl_idx and is_name(lvl_idx - 1) and 1 < len(set(col(lvl_idx - 1))) <= 20:
        svc_idx = lvl_idx - 1
        msg_from = lvl_idx + 1
    else:
        start = (lvl_idx + 1) if lvl_idx is not None else (max(best_ts) + 1 if best_ts else 0)
        while start < width and all(x.isdigit() for x in col(start)[:20]):   # skip pids etc.
            start += 1
        if is_name(start):
            svc_idx = start
            # 2. syslog "LabSZ sshd[24200]: ...": a word identical on every line and NOT ending in
            #    ":", followed by one that does, is the hostname; the process comes next.
            if len(set(col(start))) == 1 and not col(start)[0].endswith(":") and is_name(start + 1) \
                    and sum(x.endswith(":") for x in col(start + 1)) / len(sample) > 0.8:
                svc_idx = start + 1
            # 3. A bracketed thread name followed by "Class:" (Hadoop "[main] org...MRAppMaster:").
            elif col(start)[0].startswith("[") and start + 1 < width and \
                    sum(x.endswith(":") for x in col(start + 1)) / len(sample) > 0.8:
                svc_idx = start + 1
        msg_from = (svc_idx + 1) if svc_idx is not None else start
    return {"ts_tokens": best_ts, "level": lvl_idx, "service": svc_idx, "msg_from": msg_from}


def _score_map(lines, m) -> float:
    """Share of sample lines whose timestamp parses AND carries a time of day."""
    try:
        df = _apply_token_map(lines[:200], m)
        if not len(df):
            return 0.0
        ts = parse_ts(df["ts"], loose=False)
        has_time = ts.notna() & ((ts.dt.hour + ts.dt.minute + ts.dt.second) > 0)
        return float(has_time.mean()) if ts.nunique() > 1 else 0.0
    except Exception:
        return 0.0


def parse_logs(lines: list[str], format_mapper=None) -> ParseResult:
    """format_mapper: optional callable(sample_lines) -> token map dict (LLM)."""
    notes: list[str] = []

    if sum(ln.lstrip().startswith("{") for ln in lines[:50]) > 0.8 * min(50, len(lines)):
        df, bad = _try_jsonl(lines)
        if df is not None:
            fmt = "JSON lines"
            if bad:
                notes.append(f"{bad} lines were not valid JSON and were skipped.")
        else:
            raise InputError("Lines start with '{' but none are valid JSON. " + EXPECTED_HELP)
    elif (csv_df := _try_csv(lines)) is not None:
        df, fmt = csv_df, "CSV"
    elif sum(bool(_PLAIN.match(ln)) for ln in lines[:200]) > 0.8 * min(200, len(lines)):
        df = pd.DataFrame([m.groupdict() for ln in lines if (m := _PLAIN.match(ln))])
        fmt = "plain text (native)"
    else:
        mapping, fmt = None, None
        if format_mapper is not None:
            try:
                cand = format_mapper(lines[:20])
                if cand and _score_map(lines, cand) >= 0.6:
                    mapping, fmt = cand, "unknown format, mapped by LLM"
            except Exception:
                pass       # LLM failure is not fatal; heuristic takes over
        if mapping is None:
            cand = guess_token_map(lines)
            if _score_map(lines, cand) >= 0.6:
                mapping, fmt = cand, "unknown format, mapped by heuristic"
        if mapping is None:
            raise InputError("Couldn't recognise this log format (no readable timestamps). "
                             + EXPECTED_HELP)
        df = _apply_token_map(lines, mapping)
        notes.append(f"Format mapping used: {mapping}")

    out, dropped, more = normalise(df)
    notes += more
    if dropped:
        notes.append(f"{dropped} lines had no readable timestamp and were skipped.")
    if len(out) < MIN_EVENTS:
        raise InputError(f"Only {len(out)} usable events found; need at least {MIN_EVENTS} "
                         "to tell normal behaviour from an incident.")
    return ParseResult(out, fmt, notes, dropped)