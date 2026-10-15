"""Streamlit front end. Run with:  streamlit run app.py"""
from __future__ import annotations

from pathlib import Path
import pickle
import traceback

import altair as alt
import pandas as pd
import streamlit as st

from agents.assist import answer_question, make_format_mapper, scenario_to_settings
from config import BENCH_PAUSE_S, MAX_UPLOAD_BYTES
from core.generator import INCIDENT_TYPES, generate_logs, key_to_df, plan_incidents, to_text
from core.llm import LLM, probe
from core.parser import InputError, parse_logs, read_input
from core.pipeline import evaluate, run_pipeline

CACHE = Path(__file__).parent / "data" / "last_good_run.pkl"
ICONS = {"Investigator": ":material/search:", "Verifier": ":material/fact_check:",
         "System": ":material/settings:"}

st.set_page_config(page_title="Incident Triage Agents", page_icon=":material/emergency:", layout="wide")
ss = st.session_state
ss.setdefault("run", None)
ss.setdefault("key", None)
ss.setdefault("source", "")
ss.setdefault("notes", [])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle("Use Gemini (LLM agents)", value=True)
    # A real test call, cached for 5 minutes so it doesn't spend quota on every click.
    check = st.cache_data(ttl=300, show_spinner="Checking Gemini...")(probe)
    working, why = check(use_llm)
    llm = LLM(enabled=use_llm, prefer=working)
    if working:
        st.success(f"Gemini online: `{working}`", icon=":material/cloud_done:")
    else:
        st.warning(f"Offline reasoning. {why}", icon=":material/cloud_off:")
    if use_llm and st.button("Re-check Gemini", icon=":material/refresh:"):
        check.clear()
        st.rerun()
    replay = st.toggle("Replay last good run (demo backup)", value=False,
                       help="If the network or API fails during a presentation, show the last successful analysis.")
    st.caption(f"Upload limit {MAX_UPLOAD_BYTES // 1_000_000} MB. Logs are processed in memory and never stored.")


def show_step(container, s):
    container.markdown(f"{ICONS.get(s['agent'], '')} **{s['agent']}** · `{s['kind']}`  \n{s['text']}")


def analyse(df: pd.DataFrame, key, source: str, notes: list[str]):
    with st.status("Agents working...", expanded=True) as status:
        run = run_pipeline(df, llm, on_step=lambda s: show_step(status, s))
        status.update(label=run.summary(), state="complete", expanded=False)
    ss.run, ss.key, ss.source, ss.notes = run, key, source, notes
    if run.results and all(r.conclusion.mode == "llm" for r in run.results):
        try:
            CACHE.write_bytes(pickle.dumps({"run": run, "key": key, "source": source, "notes": notes}))
        except Exception:
            pass


def safely(fn):
    """Run an action; show friendly errors instead of a stack trace."""
    try:
        fn()
    except InputError as e:
        st.error(str(e), icon=":material/error:")
    except Exception as e:
        st.error(f"Unexpected problem ({type(e).__name__}). The input was not changed; try again "
                 "or switch off the LLM toggle.", icon=":material/error:")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


st.title("Incident triage agents")
st.caption("Logs in → anomaly detection → investigator and verifier agents → verified incident report")

tab_run, tab_eval, tab_ask = st.tabs(["Analyse", "Evaluation", "Ask about the results"])

# ------------------------------------------------------------------ input
with tab_run:
    if replay:
        try:
            cached = pickle.loads(CACHE.read_bytes())
            ss.run, ss.key, ss.source, ss.notes = cached["run"], cached["key"], cached["source"], cached["notes"]
            st.info("Showing the last successful run (replay mode).", icon=":material/replay:")
        except Exception:          # missing, or saved by an older version of the code
            st.warning("No usable saved run yet. Do one successful analysis with the LLM online first.")
    else:
        mode = st.radio("Input", ["Simulate incidents", "Describe a scenario", "Upload a log file",
                                  "Paste log lines"], horizontal=True)

        if mode == "Simulate incidents":
            difficulty = st.segmented_control(
                "Difficulty", ["easy", "hard"], default="easy",
                help="Hard: victim services fail in the same minute and louder than the cause, "
                     "so only reading the messages reveals the root cause.") or "easy"
            c1, c2, c3 = st.columns([2, 2, 1])
            types = c1.multiselect("Incidents to inject (empty = healthy logs)", list(INCIDENT_TYPES),
                                   default=["db_timeout"],
                                   format_func=lambda t: t.replace("_", " "))
            at = c2.slider("Start time", 5, 110, 45, format="10:00 + %d min")
            dur = c3.number_input("Duration (min)", 2, 30, 8)
            incs, problem = plan_incidents(types, at, dur)
            if problem:
                st.error(problem, icon=":material/schedule:")
            elif incs:
                fmt = lambda m: f"{10 + m // 60}:{m % 60:02d}"                    # noqa: E731
                st.caption("Planned: " + " · ".join(
                    f"{i['type'].replace('_', ' ')} {fmt(i['at_minute'])}-{fmt(i['at_minute'] + i['duration'])}"
                    for i in incs))
                if at < 15:
                    st.caption("Starts with under 15 minutes of normal traffic before it, so the "
                               "detector has less history to compare against.")
            if st.button("Generate and analyse", type="primary", disabled=bool(problem)):
                def go():
                    df, key = generate_logs(incidents=incs, seed=int(pd.Timestamp.now().timestamp()) % 10_000,
                                            difficulty=difficulty)
                    with st.expander("Raw log sample"):
                        st.code(to_text(df.iloc[len(df) // 2 - 10: len(df) // 2 + 10]))
                    analyse(df, key_to_df(key), "simulated", [])
                safely(go)

        elif mode == "Describe a scenario":
            text = st.text_input("Scenario", placeholder="payment service runs out of memory at 11:15 am")
            if st.button("Simulate and analyse", type="primary"):
                def go():
                    settings, how = scenario_to_settings(text, llm)
                    st.caption(f"Interpreted by {how}: {settings}")
                    df, key = generate_logs(**{k: v for k, v in settings.items() if k in ("minutes", "incidents")})
                    analyse(df, key_to_df(key), "scenario", [settings.get("note", "")] if settings.get("note") else [])
                safely(go)

        else:
            if mode == "Upload a log file":
                f = st.file_uploader("Log file (.log, .txt, .jsonl, .csv)", type=["log", "txt", "jsonl", "json", "csv"])
                raw, is_paste = (f.getvalue() if f else None), False
            else:
                raw = st.text_area("Paste log lines (at least 20)", height=200)
                is_paste = True
            if st.button("Analyse", type="primary", disabled=not raw):
                def go():
                    lines, notes = read_input(raw, is_paste=is_paste)
                    # The format helper gets its OWN Gemini connection: if that side call fails
                    # (rate limit, overload), the agents below still get a fresh try.
                    parsed = parse_logs(lines, format_mapper=make_format_mapper(
                        LLM(enabled=use_llm, prefer=working)))
                    st.caption(f"Detected format: {parsed.fmt} · {len(parsed.df):,} events")
                    analyse(parsed.df, None, "user", notes + parsed.notes)
                safely(go)

    # ------------------------------------------------------------------ results
    run = ss.run
    if run is not None:
        for n in ss.notes:
            st.info(n, icon=":material/info:")
        st.subheader(run.summary())

        m = run.metrics
        per_min = m.groupby("minute", as_index=False)[["errors", "events"]].sum()
        chart = alt.Chart(per_min).mark_line(color="#888").encode(
            # 24-hour clock, plus the date when the logs span more than a day: the default
            # 12-hour axis showed 18:06 as "06:06", which reads as morning.
            x=alt.X("minute:T", title=None, axis=alt.Axis(
                format="%H:%M" if per_min["minute"].max() - per_min["minute"].min() < pd.Timedelta(days=1)
                else "%d %b %H:%M")),
            y=alt.Y("errors:Q", title="errors per minute"))
        if run.candidates:
            win = pd.DataFrame([{"start": c.start, "end": c.end, "id": c.incident_id} for c in run.candidates])
            chart = alt.Chart(win).mark_rect(opacity=0.2, color="#E24B4A").encode(
                x="start:T", x2="end:T", tooltip=["id"]) + chart
        st.altair_chart(chart.properties(height=220), use_container_width=True)

        if run.no_incident:
            st.success("No incident detected. The logs look healthy, so no root cause was invented.",
                       icon=":material/check_circle:")
        for r in run.results:
            with st.container(border=True):
                st.markdown(r.report_md)
                with st.expander(f"Agent trace ({len(r.trace)} steps)"):
                    for s in r.trace:
                        show_step(st, s)
        if run.results:
            st.download_button("Download reports (.md)", "\n\n---\n\n".join(r.report_md for r in run.results),
                               file_name="incident_reports.md")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    run = ss.run
    if run is None:
        st.info("Run an analysis first.")
    elif ss.key is None:
        st.info("These logs came from you, so there is no answer key: the system can't score itself. "
                "Judge the report by its evidence. Simulated runs show an accuracy score here.")
    else:
        ev = evaluate(run.results, ss.key)
        c1, c2, c3 = st.columns(3)
        c1.metric("Incidents detected", f"{ev['detected']}/{ev['injected']}")
        c2.metric("Root cause correct", f"{ev['root_correct']}/{ev['injected']}")
        c3.metric("False alarms", ev["false_alarms"])
        if ev["injected"]:
            st.dataframe(ev["table"], hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Benchmark")
    b1, b2 = st.columns([3, 1])
    n = b1.slider("Random simulated runs", 5, 40, 20)
    bench_diff = b2.selectbox("Difficulty", ["hard", "easy"])
    st.caption("The same incidents every time (fixed seed), solved by the agents AND by the offline "
               f"rules, so the two can be compared fairly. With Gemini on it pauses {BENCH_PAUSE_S}s "
               "between runs to stay under the free rate limit.")
    if st.button("Run benchmark"):
        def go():
            import random
            import time
            rng = random.Random(7)
            offline = LLM(enabled=False)
            rows, bar, status = [], st.progress(0.0), st.empty()
            for i in range(n):
                healthy = i % 5 == 4
                incs = [] if healthy else [{"type": rng.choice(list(INCIDENT_TYPES)),
                                            "at_minute": rng.randint(20, 100), "duration": rng.randint(4, 12)}]
                df, key = generate_logs(incidents=incs, seed=1000 + i, difficulty=bench_diff)
                kdf = key_to_df(key)
                # A fresh connection per run: one rate-limit failure can't switch Gemini
                # off for every later run (that silently mixed modes before).
                run_llm = LLM(enabled=use_llm, prefer=working)
                status.caption(f"Run {i + 1}/{n}...")
                results = run_pipeline(df, run_llm).results
                ev = evaluate(results, kdf)
                base = evaluate(run_pipeline(df, offline).results, kdf)
                if healthy:
                    mode = "-"
                elif results and all(r.conclusion.mode == "llm" for r in results):
                    mode = "Gemini"
                else:
                    mode = "offline (Gemini failed)"
                rows.append({"run": i + 1, "injected": incs[0]["type"] if incs else "none (healthy)",
                             "solved by": mode,
                             "detected": ev["detected"] == ev["injected"],
                             "root_correct": None if healthy else ev["root_correct"] == ev["injected"],
                             "offline_rules_correct": None if healthy else base["root_correct"] == base["injected"],
                             "false_alarms": ev["false_alarms"]})
                bar.progress((i + 1) / n)
                if mode == "Gemini" and i < n - 1:     # only pause when quota was used
                    time.sleep(BENCH_PAUSE_S)
            status.empty()
            res = pd.DataFrame(rows)
            inc = res[res["injected"] != "none (healthy)"]
            gem = inc[inc["solved by"] == "Gemini"]
            fell_back = int((inc["solved by"] != "Gemini").sum())
            if len(gem):
                st.write(f"**Gemini agents, {bench_diff}:** root cause correct "
                         f"**{int(gem['root_correct'].sum())}/{len(gem)}** "
                         f"({gem['root_correct'].mean():.0%}) · offline rules on the same incidents: "
                         f"{int(gem['offline_rules_correct'].sum())}/{len(gem)} "
                         f"({gem['offline_rules_correct'].mean():.0%})")
            st.write(f"Detection {inc['detected'].mean():.0%} · false alarms {res['false_alarms'].sum()} "
                     f"across {len(res)} runs ({len(res) - len(inc)} healthy)")
            if fell_back:
                st.warning(f"{fell_back} run(s) fell back to offline rules because Gemini failed "
                           "(usually the free rate limit). They are excluded from the Gemini score. "
                           "Wait a few minutes and run again with fewer runs.", icon=":material/warning:")
            st.dataframe(res, hide_index=True, use_container_width=True)
        safely(go)

# ------------------------------------------------------------------ Q&A
with tab_ask:
    q = st.text_input("Question", placeholder="Why did you blame the database and not the gateway?")
    if st.button("Ask", disabled=not q):
        safely(lambda: st.markdown(answer_question(q, ss.run.results if ss.run else [], llm)))