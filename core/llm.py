"""Thin, defensive wrapper around Gemini.

Rules it enforces:
- Every call returns parsed JSON or raises LLMUnavailable. Callers never see raw
  SDK errors, so one bad response can't crash the app.
- Server errors are retried; an overloaded, retired or rate-limited model is
  replaced by the next one in GEMINI_MODELS.
- Bad API key / every model failing -> the app switches to offline reasoning
  instead of dying mid-demo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import tomllib

from config import GEMINI_MODELS, LLM_RETRIES, LLM_TIMEOUT_S

SECRETS_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"


class LLMUnavailable(Exception):
    pass


def _get_key() -> tuple[str | None, str]:
    """Return (key, problem). `problem` explains why no key was found."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key, ""
    if SECRETS_PATH.exists():
        try:
            data = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8-sig"))
        except tomllib.TOMLDecodeError as e:   # message has line/column only, never the key
            return None, (f".streamlit/secrets.toml has a typing mistake ({e}). Write it as "
                          'GEMINI_API_KEY = "your-key" with straight double quotes.')
        if data.get("GEMINI_API_KEY"):
            return str(data["GEMINI_API_KEY"]), ""
        return None, ".streamlit/secrets.toml exists but has no GEMINI_API_KEY line."
    try:                                    # Streamlit Cloud secrets
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key, ""
    except Exception:
        pass
    return None, "No GEMINI_API_KEY found; running in offline mode."


def extract_json(text: str):
    """Pull the first JSON object out of a model reply, tolerating ``` fences."""
    if not text:
        raise ValueError("empty reply")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("no JSON object in reply")
        return json.loads(m.group(0))


class LLM:
    def __init__(self, enabled: bool = True, prefer: str | None = None):
        self.client = None
        self.reason = ""
        self.calls = 0
        # A model known to work right now (from the sidebar check) is tried first.
        self.models = ([prefer] if prefer else []) + [m for m in GEMINI_MODELS if m != prefer]
        self.model_idx = 0
        self.model_used: str | None = None
        if not enabled:
            self.reason = "LLM switched off (offline mode selected)."
            return
        key, problem = _get_key()
        if not key:
            self.reason = problem
            return
        try:
            from google import genai
            from google.genai import types
            self._types = types
            self.client = genai.Client(
                api_key=key, http_options=types.HttpOptions(timeout=LLM_TIMEOUT_S * 1000))
        except Exception as e:                       # import or config problem
            self.reason = f"Could not start Gemini client ({type(e).__name__})."

    @property
    def available(self) -> bool:
        return self.client is not None

    def json(self, system: str, prompt: str, temperature: float = 0.2) -> dict:
        if not self.available:
            raise LLMUnavailable(self.reason)
        from google.genai import errors
        cfg = self._types.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
            response_mime_type="application/json")
        last = None
        # Sticky: once a model fails, later calls in this run start from the next one.
        while self.model_idx < len(self.models):
            model = self.models[self.model_idx]
            for attempt in range(LLM_RETRIES):
                try:
                    self.calls += 1
                    resp = self.client.models.generate_content(
                        model=model, contents=prompt, config=cfg)
                    data = extract_json(resp.text or "")
                    if not isinstance(data, dict):
                        raise ValueError("reply is not a JSON object")
                    self.model_used = model
                    return data
                except errors.ClientError as e:
                    code = getattr(e, "code", 0)
                    if code in (404, 429):           # retired model / its quota used: next model
                        last = e
                        break
                    # 400/401/403: bad key or request; another model won't help
                    self.client = None
                    self.reason = f"Gemini rejected the request (HTTP {code}); switched to offline mode."
                    raise LLMUnavailable(self.reason) from e
                except errors.ServerError as e:      # 5xx, e.g. 503 overloaded
                    last = e
                    time.sleep(attempt + 1)
                except (ValueError, json.JSONDecodeError) as e:   # malformed reply: ask again
                    last = e
                except Exception as e:               # network / timeout
                    last = e
                    time.sleep(attempt + 1)
            self.model_idx += 1
        # Every model failed: stop calling for the rest of this run so it doesn't keep waiting.
        code = getattr(last, "code", None)
        self.client = None
        self.reason = (f"All Gemini models failed (last error: {code or type(last).__name__}); "
                       "using offline reasoning.")
        raise LLMUnavailable(self.reason)


def probe(enabled: bool) -> tuple[str | None, str]:
    """One tiny real call. Returns (working model, '') or (None, reason)."""
    llm = LLM(enabled=enabled)
    if not llm.available:
        return None, llm.reason
    try:
        llm.json("Reply with a JSON object only.", 'Return exactly {"ok": true}')
        return llm.model_used, ""
    except LLMUnavailable as e:
        return None, str(e)