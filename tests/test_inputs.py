"""Every bad or unusual input a teacher could give must either work or fail with a clear message."""
import json

import pandas as pd

import pytest

from config import MAX_LINES, MAX_UPLOAD_BYTES
from core.detector import detect
from core.generator import generate_logs, to_jsonl, to_text
from core.parser import InputError, parse_logs, read_input


def parse(raw, **kw):
    lines, notes = read_input(raw, **kw)
    return parse_logs(lines)


# ---------------------------------------------------------- rejected cleanly
@pytest.mark.parametrize("raw", [b"", "", b"\n\n   \n"])
def test_empty_input(raw):
    with pytest.raises(InputError, match="empty|blank"):
        read_input(raw)


def test_binary_file():
    with pytest.raises(InputError, match="binary"):
        read_input(b"\x89PNG\r\n\x1a\n" + b"\x00" * 500)


def test_oversized_file():
    with pytest.raises(InputError, match="limit"):
        read_input(b"x" * (MAX_UPLOAD_BYTES + 1))


def test_garbage_text():
    with pytest.raises(InputError, match="recognise|timestamp"):
        parse(("hello world this is not a log at all\n" * 50).encode())


def test_too_few_events():
    with pytest.raises(InputError, match="need at least"):
        parse(to_text(generate_logs(minutes=10)[0].head(5)).encode())


def test_broken_json():
    with pytest.raises(InputError):
        parse(b"{not json\n" * 40)


# ---------------------------------------------------------- accepted
def test_native_formats_roundtrip():
    df, _ = generate_logs(minutes=30)
    for raw in (to_text(df), to_jsonl(df)):
        assert len(parse(raw.encode()).df) == len(df)


def test_huge_line_count_is_truncated_not_crashed():
    line = "2026-10-14 10:00:00 INFO api /x 200 5ms ok\n"
    lines, notes = read_input(line * (MAX_LINES + 100))
    assert len(lines) == MAX_LINES and any("last" in n for n in notes)


def test_jsonl_with_some_bad_lines():
    df, _ = generate_logs(minutes=20)
    raw = to_jsonl(df) + "\n{broken\n[1,2]\n"
    r = parse(raw.encode())
    assert len(r.df) == len(df) and any("not valid JSON" in n for n in r.notes)


def test_jsonl_with_other_field_names():
    rows = [{"@timestamp": f"2026-10-14T10:{i // 60:02d}:{i % 60:02d}Z", "severity": "warning",
             "app": "billing", "message": "slow"} for i in range(100)]
    r = parse("\n".join(json.dumps(x) for x in rows).encode())
    assert r.df["level"].eq("WARN").all() and r.df["service"].eq("billing").all()


def test_csv_upload():
    csv = "timestamp,level,service,message\n" + "\n".join(
        f"2026-10-14 10:{i:02d}:00,INFO,shop,ok" for i in range(40))
    assert parse(csv.encode()).fmt == "CSV"


def test_non_english_and_latin1():
    hindi = "\n".join(f"2026-10-14 10:{i:02d}:00 ERROR भुगतान-सेवा /pay 500 30ms डेटाबेस त्रुटि" for i in range(30))
    r = parse(hindi.encode("utf-8"))
    assert r.df["service"].iloc[0] == "भुगतान-सेवा"
    latin = "\n".join(f"2026-10-14 10:{i:02d}:00 INFO café /x 200 3ms résumé" for i in range(30))
    r2 = parse(latin.encode("latin-1"))
    assert len(r2.df) == 30


def test_utf16_file():
    txt = "\n".join(f"2026-10-14 10:{i:02d}:00 INFO api /x 200 3ms ok" for i in range(30))
    assert len(parse(txt.encode("utf-16")).df) == 30


def test_unknown_format_hdfs_style():
    """Loghub HDFS-like lines: yymmdd HHMMSS pid LEVEL component: message"""
    raw = "\n".join(f"081109 2036{i % 60:02d} 148 INFO dfs.DataNode$PacketResponder: block {i} terminating"
                    for i in range(60))
    r = parse(raw.encode())
    assert "heuristic" in r.fmt and r.df["service"].iloc[0].startswith("dfs.DataNode")
    assert r.df["ts"].notna().all()


def test_unknown_format_syslog_without_level():
    raw = "\n".join(f"Oct 14 10:{i // 60:02d}:{i % 60:02d} web01 nginx[{100 + i}]: "
                    + ("upstream timed out" if i % 7 == 0 else "request served") for i in range(80))
    r = parse(raw.encode())
    assert r.df["ts"].notna().all()
    assert r.df["level"].eq("ERROR").sum() == sum(i % 7 == 0 for i in range(80))


def test_mixed_garbage_lines_are_skipped():
    df, _ = generate_logs(minutes=20)
    raw = to_text(df) + "\n" + "\n".join(["!!! garbage", "---"] * 5)
    r = parse(raw.encode())
    assert len(r.df) == len(df)


def test_pasted_text_is_capped():
    big = "2026-10-14 10:00:00 INFO api /x 200 5ms ok\n" * 10_000
    _, notes = read_input(big, is_paste=True)
    assert any("cut" in n for n in notes)


# ---------------------------------------------------------- detector honesty
@pytest.mark.parametrize("seed", range(10))
def test_healthy_logs_never_raise_incidents(seed):
    df, _ = generate_logs(seed=seed)
    incidents, _ = detect(df)
    assert incidents == []


def test_short_logs_dont_crash_detector():
    df, _ = generate_logs(minutes=10, events_per_min=5)
    detect(df)


# ---------------------------------------------------------- real-world formats
# Lines below copy the exact FORMAT of public Loghub datasets (values are made up),
# because these formats broke the parser before: wrong dates, no time of day, or refused.
def _many(make, n=60):
    return "\n".join(make(i) for i in range(n)).encode()


@pytest.mark.parametrize("name,make,svc,first_ts", [
    ("openssh", lambda i: f"Dec 10 06:{i // 60 + 10:02d}:{i % 60:02d} LabSZ sshd[{24200 + i}]: Failed password for root",
     "sshd", "2026-12-10 06:10:00"),
    ("linux", lambda i: f"Jun 14 15:{i // 60 + 16:02d}:{i % 60:02d} combo sshd(pam_unix)[{19939 + i}]: authentication failure",
     "sshd(pam_unix)", "2026-06-14 15:16:00"),
    ("apache", lambda i: f"[Sun Dec 04 04:{i // 60 + 47:02d}:{i % 60:02d} 2005] [error] mod_jk child workerEnv in error state 6",
     "mod_jk", "2005-12-04 04:47:00"),
    ("hdfs", lambda i: f"081109 20{i // 60 + 36:02d}{i % 60:02d} 148 INFO dfs.DataNode$PacketResponder: block {i} terminating",
     "dfs.DataNode$PacketResponder", "2008-11-09 20:36:00"),
    ("zookeeper", lambda i: f"2015-07-29 17:{i // 60 + 41:02d}:{i % 60:02d},747 - WARN  [QuorumPeer[myid=1]/0:0:0:0:2181:FastLeaderElection@774] - Notification time out: 3200",
     "FastLeaderElection", "2015-07-29 17:41:00.747"),
    ("hadoop", lambda i: f"2015-10-18 18:{i // 60 + 1:02d}:{i % 60:02d},978 INFO [main] org.apache.hadoop.mapreduce.v2.app.MRAppMaster: Created MRAppMaster",
     "MRAppMaster", "2015-10-18 18:01:00.978"),
    ("spark", lambda i: f"17/06/09 20:{i // 60 + 10:02d}:{i % 60:02d} INFO executor.CoarseGrainedExecutorBackend: Registered signal handlers",
     "executor.CoarseGrainedExecutorBackend", "2017-06-09 20:10:00"),
    ("bgl", lambda i: f"- {1117838570 + i} 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.{i:02d}.675872 R02-M1-N0-C:J12-U11 RAS {'KERNEL' if i % 3 else 'APP'} INFO instruction cache parity error corrected",
     "KERNEL", "2005-06-03 15:42:00.675872"),
])
def test_real_world_formats(name, make, svc, first_ts):
    r = parse(_many(make))
    assert r.df["ts"].min() == pd.Timestamp(first_ts), name
    assert (r.df["ts"].diff().dropna() >= pd.Timedelta(0)).all()
    assert r.df["ts"].nunique() == 60, f"{name}: time of day was lost"
    assert svc in set(r.df["service"]), f"{name}: got {set(r.df['service'])}"


def test_bgl_expert_labels_never_reach_the_agents():
    """BGL's first word is the answer ('-' = normal, else an alert code). It must not leak."""
    lines = _many(lambda i: f"{'KERNDTLB' if i % 5 == 0 else '-'} {1118536327 + i} 2005.06.11 R30-M0-N9-C:J16-U01 "
                            f"2005-06-11-17.32.{i:02d}.581048 R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt")
    df = parse(lines).df
    assert not df["msg"].str.contains("KERNDTLB").any()
    assert not df["service"].str.contains("KERNDTLB").any()


def test_timestamps_with_commas_are_not_mistaken_for_csv():
    raw = _many(lambda i: f"2015-07-29 17:41:{i:02d},747 - INFO  [main:Server@1] - started")
    assert parse(raw).fmt != "CSV"


# ---------------------------------------------------------- placing several incidents
from core.generator import plan_incidents  # noqa: E402


def test_incidents_are_spread_with_quiet_gaps():
    incs, problem = plan_incidents(["db_timeout", "cache_down", "auth_failure"], 20, 8)
    assert problem is None and [i["at_minute"] for i in incs] == [20, 65, 110]
    for a, b in zip(incs, incs[1:]):
        assert b["at_minute"] - (a["at_minute"] + a["duration"]) >= 15


@pytest.mark.parametrize("types,start,dur,hint", [
    (["db_timeout"], 110, 20, "at most 8 minutes"),                      # runs past the end of the logs
    (["db_timeout", "cache_down"], 100, 8, "start at 11:27 or earlier"),  # second one wouldn't fit
    (["db_timeout", "cache_down", "auth_failure"], 45, 25, "duration of 14 min or less"),
])
def test_plans_that_dont_fit_are_refused_with_a_fix(types, start, dur, hint):
    incs, problem = plan_incidents(types, start, dur)
    assert incs == [] and hint in problem


def test_memory_leak_warning_period_gets_extra_room():
    incs, _ = plan_incidents(["db_timeout", "memory_leak"], 20, 8)
    # memory leak logs warnings for 10 minutes BEFORE it starts: those must not touch db_timeout
    assert incs[1]["at_minute"] - 10 - (incs[0]["at_minute"] + incs[0]["duration"]) >= 15


def test_planned_incidents_are_all_detected_separately():
    from core.llm import LLM
    from core.pipeline import evaluate, run_pipeline
    from core.generator import key_to_df
    incs, _ = plan_incidents(["db_timeout", "cache_down", "auth_failure"], 20, 8)
    df, key = generate_logs(incidents=incs, seed=3, difficulty="hard")
    ev = evaluate(run_pipeline(df, LLM(enabled=False)).results, key_to_df(key))
    assert ev["detected"] == 3 and ev["false_alarms"] == 0