"""The real LLM wrapper with a fake Gemini client: model fallback and key loading."""
from types import SimpleNamespace

import pytest
from google.genai import errors, types

import core.llm as llm_mod
from core.llm import LLM, LLMUnavailable


class FakeModels:
    """behaviour: model name -> list of 'ok' | HTTP code, consumed per call."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.seen = []

    def generate_content(self, model, contents, config):
        self.seen.append(model)
        step = self.behaviour[model].pop(0) if self.behaviour.get(model) else 503
        if step == "ok":
            return SimpleNamespace(text='{"ok": true}')
        cls = errors.ServerError if step >= 500 else errors.ClientError
        raise cls(step, {"error": {"code": step, "message": "x", "status": "x"}})


def make(behaviour, models, monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    llm = LLM(enabled=False)
    llm.models, llm._types = list(models), types
    llm.client = SimpleNamespace(models=FakeModels(behaviour))
    return llm


def test_overloaded_model_falls_back_and_stays_on_the_working_one(monkeypatch):
    llm = make({"a": [503, 503], "b": ["ok", "ok"]}, ["a", "b"], monkeypatch)
    assert llm.json("s", "p") == {"ok": True} and llm.model_used == "b"
    llm.json("s", "p")
    assert llm.client.models.seen == ["a", "a", "b", "b"]      # 'a' not retried on the 2nd call


def test_retired_or_rate_limited_model_is_skipped_immediately(monkeypatch):
    llm = make({"a": [404], "b": [429], "c": ["ok"]}, ["a", "b", "c"], monkeypatch)
    llm.json("s", "p")
    assert llm.client.models.seen == ["a", "b", "c"]


def test_bad_key_goes_offline_without_trying_other_models(monkeypatch):
    llm = make({"a": [403], "b": ["ok"]}, ["a", "b"], monkeypatch)
    with pytest.raises(LLMUnavailable, match="403"):
        llm.json("s", "p")
    assert not llm.available and llm.client is None


def test_all_models_down_goes_offline_and_stops_calling(monkeypatch):
    llm = make({}, ["a", "b"], monkeypatch)
    with pytest.raises(LLMUnavailable, match="All Gemini models failed"):
        llm.json("s", "p")
    assert not llm.available                                    # later calls fail fast


def test_preferred_model_is_tried_first():
    from config import GEMINI_MODELS
    llm = LLM(enabled=False, prefer=GEMINI_MODELS[-1])
    assert llm.models[0] == GEMINI_MODELS[-1] and len(llm.models) == len(GEMINI_MODELS)


@pytest.mark.parametrize("content,expected", [
    ('GEMINI_API_KEY = “abc123”\n', "typing mistake"),          # curly quotes
    ("GEMINI_API_KEY = abc123\n", "typing mistake"),                   # no quotes
    ('OTHER = "x"\n', "no GEMINI_API_KEY"),
])
def test_secrets_problems_are_explained(tmp_path, monkeypatch, content, expected):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p = tmp_path / "secrets.toml"
    p.write_text(content, encoding="utf-8")
    monkeypatch.setattr(llm_mod, "SECRETS_PATH", p)
    llm = LLM()
    assert not llm.available and expected in llm.reason
    assert "abc123" not in llm.reason                          # never echo the key


def test_valid_secrets_file_is_read(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p = tmp_path / "secrets.toml"
    p.write_text('GEMINI_API_KEY = "abc123"\n', encoding="utf-8")
    monkeypatch.setattr(llm_mod, "SECRETS_PATH", p)
    assert llm_mod._get_key() == ("abc123", "")