import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import LLMUnavailable  # noqa: E402


class FakeLLM:
    """Scripted stand-in for Gemini. `script` is a list of replies (dicts) or
    exceptions, returned in order. Lets us test the agent loops without an API key."""

    def __init__(self, script):
        self.script = list(script)
        self.prompts = []
        self.reason = ""

    @property
    def available(self):
        return True

    def json(self, system, prompt, temperature=0.2):
        self.prompts.append(prompt)
        if not self.script:
            raise LLMUnavailable("script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def offline_by_default(monkeypatch, tmp_path):
    """Tests must never read the developer's real key or call the real Gemini API:
    that would be slow, spend free quota, and make results depend on the network."""
    import streamlit

    import core.llm as llm_mod
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm_mod, "SECRETS_PATH", tmp_path / "no-secrets.toml")
    monkeypatch.setattr(streamlit, "secrets", {})
    streamlit.cache_data.clear()               # forget any cached "Gemini online" check


@pytest.fixture
def fake_llm():
    return FakeLLM