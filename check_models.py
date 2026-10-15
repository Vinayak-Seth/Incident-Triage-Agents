"""Which Gemini models answer right now?  Run:  python check_models.py

Put the ones that print OK at the top of GEMINI_MODELS in config.py."""
import tomllib

from google import genai

from config import GEMINI_MODELS
from core.llm import SECRETS_PATH

key = tomllib.load(open(SECRETS_PATH, "rb"))["GEMINI_API_KEY"]
client = genai.Client(api_key=key)

extras = ["gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]
for m in GEMINI_MODELS + [e for e in extras if e not in GEMINI_MODELS]:
    try:
        r = client.models.generate_content(model=m, contents="Reply with the word ok")
        print(f"OK     {m}  ->  {r.text.strip()[:20]}")
    except Exception as e:
        print(f"FAIL   {m}  ->  {str(e)[:70]}")