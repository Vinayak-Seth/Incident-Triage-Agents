"""Central limits and settings. Every safety cap lives here so it is easy to defend in a viva."""

# ---- Input limits (protect memory, time and API quota) ----
MAX_UPLOAD_BYTES = 5 * 1024 * 1024      # 5 MB upload cap
MAX_LINES = 50_000                        # lines kept after reading
MAX_PASTE_CHARS = 200_000                 # pasted text cap
MIN_EVENTS = 20                           # fewer than this -> not enough data to judge
MAX_LINE_CHARS = 2_000                    # very long lines are truncated

# ---- Detection ----
BUCKET = "1min"                           # aggregation window
ROLLING_WINDOW = 15                       # minutes of history used as "normal"
Z_THRESHOLD = 4.0                         # how far above normal counts as anomalous
MIN_ERROR_RATE = 0.10                     # ignore tiny blips below 10% errors
MIN_ERRORS_IN_BUCKET = 5                  # need at least this many errors in a minute
MIN_LATENCY_JUMP_MS = 200                 # latency must rise by at least this much
MERGE_GAP_MIN = 3                         # anomalies closer than this are one incident
MAX_INCIDENTS = 5                         # investigate at most 5 incidents per run

# ---- Agents ----
MAX_TOOL_CALLS = 6                        # investigator tool budget per round
MAX_DEBATE_ROUNDS = 3                     # investigator <-> verifier rounds
MAX_TOOL_RESULT_LINES = 25                # lines returned by search_logs to the LLM
MAX_LLM_CONFIDENCE = 0.95                 # model self-ratings are not calibrated; never show 100%

# ---- LLM ----
# Tried in order. If one is overloaded (503), retired (404) or out of free quota
# (429), the next is used. Run check_models.py to see which ones answer today.
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]
LLM_TIMEOUT_S = 30
LLM_RETRIES = 2                           # attempts per model before moving to the next
BENCH_PAUSE_S = 20                        # pause between benchmark runs (free-tier rate limit)

# ---- Canonical schema every input is converted to ----
CANONICAL_COLUMNS = ["ts", "level", "service", "endpoint", "status", "latency_ms", "msg"]
LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]