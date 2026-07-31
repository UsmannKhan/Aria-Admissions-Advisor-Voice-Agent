"""Generation client and model selection.

ANSWER_MODEL is the controlled variable: the baseline and every specialist
answer with it, so the comparison isolates architecture rather than model.
PLANNER_MODEL is deliberately lighter -- understanding and assembly rewrite
and route, they do not reason over evidence.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

ANSWER_MODEL = "gemini-3.6-flash"
PLANNER_MODEL = "gemini-3.5-flash-lite"

# default 'high' cost 16s per call for no gain
THINKING_LEVEL = "low"

_client: "genai.Client | None" = None
_lock = threading.Lock()


def get_client(api_key: str | None = None) -> "genai.Client":
    """One client per process; agents run on separate threads."""
    global _client
    if _client is not None and api_key is None:
        return _client

    key = (api_key
           or os.environ.get("GEMINI_API_KEY")
           or os.environ.get("GOOGLE_API_KEY"))
    if not key:
        print(f"ERROR: no GEMINI_API_KEY in the environment or "
              f"{PROJECT_ROOT / '.env'}", file=sys.stderr)
        sys.exit(1)

    if api_key is not None:        
        return genai.Client(api_key=key)

    with _lock:
        if _client is None:
            _client = genai.Client(api_key=key)
    return _client
