"""Drives the real Streamlit app headlessly (no browser) through every input mode."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def at(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)     # offline: tests must not spend API quota
    a = AppTest.from_file(APP, default_timeout=120)
    a.run()
    assert not a.exception
    return a


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_simulated_incident(at):
    _button(at, "Generate and analyse").click().run()
    assert not at.exception and not at.error
    assert any("incident(s) detected" in s.value for s in at.subheader)
    assert any("Root cause correct" in m.label for m in at.metric)


def test_healthy_simulation_reports_no_incident(at):
    at.multiselect[0].set_value([]).run()
    _button(at, "Generate and analyse").click().run()
    assert any("No incident detected" in s.value for s in at.subheader)


def test_scenario_mode(at):
    at.radio[0].set_value("Describe a scenario").run()
    at.text_input[0].set_value("payment service runs out of memory at 11 am").run()
    _button(at, "Simulate and analyse").click().run()
    assert not at.exception and any("detected" in s.value for s in at.subheader)


def test_unsupported_scenario_shows_friendly_error(at):
    at.radio[0].set_value("Describe a scenario").run()
    at.text_input[0].set_value("aliens").run()
    _button(at, "Simulate and analyse").click().run()
    assert not at.exception and any("known incident type" in e.value for e in at.error)


def test_pasted_garbage_shows_friendly_error(at):
    at.radio[0].set_value("Paste log lines").run()
    at.text_area[0].set_value("this is not a log\n" * 30).run()
    _button(at, "Analyse").click().run()
    assert not at.exception and any("recognise" in e.value for e in at.error)


def test_pasted_real_logs_have_no_answer_key(at):
    from core.generator import generate_logs, to_text
    df, _ = generate_logs(minutes=60, incidents=[{"type": "disk_full", "at_minute": 30}])
    at.radio[0].set_value("Paste log lines").run()
    at.text_area[0].set_value(to_text(df)[:190_000]).run()
    _button(at, "Analyse").click().run()
    assert not at.exception and not at.error
    assert any("no answer key" in i.value for i in at.info)


def test_question_answering(at):
    _button(at, "Generate and analyse").click().run()
    at.text_input[0].set_value("why db-service?").run()
    _button(at, "Ask").click().run()
    assert not at.exception


def test_benchmark_never_reports_offline_runs_as_gemini(at):
    at.tabs[1].slider[0].set_value(5).run()
    next(b for b in at.button if b.label == "Run benchmark").click().run()
    assert not at.exception and not at.error
    # No key in tests, so every incident run falls back; that must be flagged, not scored.
    assert any("fell back to offline rules" in w.value for w in at.warning)
    assert not any("Gemini agents" in m.value for m in at.markdown)
    table = at.dataframe[-1].value
    assert set(table["solved by"]) <= {"offline (Gemini failed)", "-"}
    assert table.loc[table["injected"] == "none (healthy)", "root_correct"].isna().all()


def test_impossible_plan_shows_fix_and_disables_button(at):
    at.slider[0].set_value(100).run()                       # start 11:40
    at.multiselect[0].set_value(["db_timeout", "cache_down"]).run()
    assert any("start at 11:27 or earlier" in e.value for e in at.error)
    assert _button(at, "Generate and analyse").disabled


def test_valid_plan_is_shown_before_running(at):
    at.multiselect[0].set_value(["db_timeout", "cache_down"]).run()
    assert any("Planned: db timeout 10:45-10:53 · cache down 11:50-11:58" in c.value for c in at.caption)