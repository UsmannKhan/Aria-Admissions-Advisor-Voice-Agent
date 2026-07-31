"""Measure what streaming actually buys us.

Runs the same generation twice -- once blocking, once streamed -- and reports
when the first complete sentence became available to hand to TTS. Also checks
that Gemini emits schema fields in declaration order, which the incremental
extraction depends on.

    python stream_test.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

MODEL = "gemini-3.5-flash-lite"
THINKING = "low"

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit("no GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


class Claim(BaseModel):
    text: str
    sources: list[int]


class Answer(BaseModel):
    answer: str
    claims: list[Claim]
    sources_used: list[int]


SOURCES = """[Source 1] institution=lums | section: BS Computer Science, Fee Structure for First Year - FY 2026-27
| Particulars | Fall 2026 | Spring 2027 | Total |
| Admission Fee | 176,600 | - | 176,600 |
| Tuition Fee | 824,400 | 732,800 | 1,557,200 |
| SBASSE Fee | 134,600 | 134,600 | 269,200 |
| Semester Registration Fee | 56,100 | 56,100 | 112,200 |
| General Security | 67,000 | - | 67,000 |
| Total | 1,258,700 | 923,500 | 2,182,200 |

[Source 2] institution=lums | section: BS Computer Science, Important Points
* The above breakdown of fee structure is for Year 1 of the 4 year BS Computer Science.
* The per credit hour rate for FY 2026-27 is PKR 45,800/-
* Fees are payable by semester according to the schedule given in the Annual Fee bill.
"""

SYSTEM = (
    "You are an admissions counsellor for Pakistani students. Use only the "
    "sources given. Attribute facts to the institution. Be concise. No citation "
    "markers in the answer text. Also list the individual factual claims with "
    "the source numbers supporting each."
)

PROMPT = (f"SOURCES:\n{SOURCES}\n-----\n"
          "QUESTION: What is the tuition fee for BS Computer Science at LUMS?\n\n"
          "Answer using only the sources above.")

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    thinking_config=types.ThinkingConfig(thinking_level=THINKING),
    response_mime_type="application/json",
    response_schema=Answer,
)

SENTENCE_END = re.compile(r'(?<=[.!?])\s')


def extract_answer_so_far(raw: str) -> tuple[str, bool]:
    """Pull the `answer` string out of partial JSON. Returns (text, complete).

    Scans for the closing quote while respecting backslash escapes, so a quote
    inside the answer does not end it early.
    """
    m = re.search(r'"answer"\s*:\s*"', raw)
    if not m:
        return "", False
    i = m.end()
    out = []
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            out.append(raw[i:i + 2])
            i += 2
            continue
        if c == '"':
            return json.loads('"' + "".join(out) + '"'), True
        out.append(c)
        i += 1
    # unterminated: decode what we have, dropping a trailing partial escape
    partial = "".join(out).rstrip("\\")
    try:
        return json.loads('"' + partial + '"'), False
    except json.JSONDecodeError:
        return "", False


def run_blocking() -> float:
    t0 = time.perf_counter()
    resp = client.models.generate_content(model=MODEL, contents=PROMPT, config=CONFIG)
    elapsed = time.perf_counter() - t0
    parsed = resp.parsed
    print(f"  full response      {elapsed:6.2f}s")
    print(f"  answer chars       {len(parsed.answer)}")
    print(f"  claims             {len(parsed.claims)}")
    return elapsed


def run_streaming() -> tuple[float, float]:
    t0 = time.perf_counter()
    raw = ""
    spoken = 0
    t_first_sentence = None
    t_answer_done = None
    field_order = []

    for chunk in client.models.generate_content_stream(
            model=MODEL, contents=PROMPT, config=CONFIG):
        raw += chunk.text or ""

        for f in ("answer", "claims", "sources_used"):
            if f not in field_order and f'"{f}"' in raw:
                field_order.append(f)

        text, complete = extract_answer_so_far(raw)

        # everything up to the last completed sentence is speakable now
        parts = SENTENCE_END.split(text)
        ready = len(parts) - 1 if not complete else len(parts)
        if ready > spoken:
            if t_first_sentence is None:
                t_first_sentence = time.perf_counter() - t0
                print(f"  first sentence     {t_first_sentence:6.2f}s  "
                      f"-> {parts[0][:60]!r}")
            spoken = ready
        if complete and t_answer_done is None:
            t_answer_done = time.perf_counter() - t0

    total = time.perf_counter() - t0
    parsed = Answer.model_validate_json(raw)
    print(f"  answer complete    {t_answer_done:6.2f}s")
    print(f"  full response      {total:6.2f}s")
    print(f"  claims             {len(parsed.claims)}")
    print(f"  field order        {field_order}")
    return t_first_sentence or total, total


if __name__ == "__main__":
    print(f"model: {MODEL}\n")
    print("BLOCKING")
    blocking = run_blocking()
    print("\nSTREAMING")
    first, total = run_streaming()

    print("\n" + "-" * 46)
    print(f"  time to first audio, blocking   {blocking:6.2f}s")
    print(f"  time to first audio, streaming  {first:6.2f}s")
    if first < blocking:
        print(f"  saved                           {blocking - first:6.2f}s "
              f"({100 * (1 - first / blocking):.0f}% earlier)")
    else:
        print("  streaming did not start speech any sooner")
