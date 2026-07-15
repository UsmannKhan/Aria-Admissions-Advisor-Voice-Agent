"""
End-to-end RAG: retrieve (via retrieve.py) -> prompt -> Gemini -> cited answer.

Builds a grounded prompt from the retrieved chunks (labelled by institution +
heading), and asks Gemini to answer only from those sources, cite by
institution, and abstain when the answer isn't present.

    python generate.py "compare the eligibility criteria for CS at LUMS and Habib"
    python generate.py "what are the LUMS fees" --k 6 --mode dense

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# load project-root .env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from google import genai
from google.genai import types
from pydantic import BaseModel

from retrieve import retrieve

GEMINI_MODEL = "gemini-3.5-flash"


class Claim(BaseModel):
    """One factual claim and the source number(s) backing it. Self-reported by
    the generator; the Phase 2 verifier re-checks it against the source."""
    text: str
    sources: list[int]


class AdvisorResponse(BaseModel):
    """answer is clean prose (TTS-safe, no markers); claims is the
    claim->source breakdown; sources_used is the flat union for citations."""
    answer: str
    claims: list[Claim]
    sources_used: list[int]


SYSTEM_INSTRUCTION = (
    "You are Aria, a warm and capable admissions counsellor for Pakistani "
    "undergraduate applicants. You are friendly and approachable, but you get "
    "to the point. Rules:\n"
    "- Use only facts found in the provided sources. Do not add outside "
    "knowledge.\n"
    "- Attribute every fact to its institution (e.g. 'At LUMS, ...').\n"
    "- If the sources do not contain the answer, say so kindly and explicitly "
    "rather than guessing.\n"
    "- Answer the question that was actually asked. Do NOT open with emotional "
    "preamble and do NOT assume the student is worried or stressed unless they "
    "say so. If the student DOES express anxiety or emotion, then respond to "
    "it warmly and briefly; otherwise, just help them directly in a friendly "
    "tone.\n"
    "- You may reassure students about the process when relevant, but do NOT "
    "predict admission outcomes or chances, and do NOT invent reassurance "
    "about results you cannot know. If asked 'will I get in', explain the "
    "requirements and that the decision depends on factors beyond the "
    "available information, then offer to help with what you can.\n"
    "- For comparisons, address each institution separately, then summarise "
    "the difference.\n"
    "- Be concise; this answer will be read aloud, so avoid padding.\n"
    "- Do NOT put citation markers or source numbers in the answer text; keep "
    "the prose clean.\n"
    "- Also break the answer into individual factual claims; for each claim "
    "give its text and the source number(s) that support it. Only verifiable "
    "factual statements need to be claims. Record the union of all source "
    "numbers used in sources_used."
)


def build_context(results: list[dict]) -> str:
    """Format retrieved chunks into a labelled source block for the prompt."""
    lines = []
    for i, r in enumerate(results, start=1):
        meta = r.get("meta", {})
        entity = meta.get("entity_id", "?")
        headings = meta.get("headings", "")
        doc = r.get("doc", "")
        lines.append(
            f"[Source {i}] institution={entity} | section: {headings}\n{doc}\n"
        )
    return "\n".join(lines)


def language_clause(language: str) -> str:
    """Per-call instruction for the response language."""
    if language == "ur":
        return (
            "\n- Write the answer in simple, everyday spoken Urdu, the way a "
            "friendly counsellor would talk to a student. Avoid formal, "
            "literary, or heavily Sanskritised Urdu. Write numbers and percentages in Urdu script. For common study terms "
            "that students normally say in English, write the English word in "
            "Urdu script rather than translating to formal Urdu (for example "
            "write maths as 'میتھس' not 'ریاضی'; write 'کمپیوٹر سائنس', "
            "'مارکس', 'پرسنٹ'). The extracted 'claims' should also be in "
            "this simple Urdu."
        )
    return "\n- Write the answer in English."


def build_prompt(query: str, context: str) -> str:
    return (
        f"SOURCES:\n{context}\n"
        f"-----\n"
        f"QUESTION: {query}\n\n"
        f"Answer using only the sources above, following the rules."
    )


def print_grounding(results: list[dict], claims: list) -> None:
    """Print each claim with the source(s) it cites (number, entity, heading)."""
    if not claims:
        return
    print("\n" + "-" * 80)
    print("GROUNDING (claim -> source)")
    print("-" * 80)
    for c in claims:
        srcs = []
        for n in c.sources:
            if 1 <= n <= len(results):
                meta = results[n - 1].get("meta", {})
                srcs.append(f"#{n} [{meta.get('entity_id','?')}: "
                            f"{meta.get('headings','')}]")
            else:
                srcs.append(f"#{n} (out of range)")
        print(f"\n  CLAIM: {c.text}")
        print(f"    sources: {'; '.join(srcs) if srcs else '(none cited)'}")


def print_sources(results: list[dict], used_numbers: list[int]) -> None:
    """Print URLs for the source numbers the model reported using, grouped by
    institution and deduplicated. Falls back to all retrieved sources if the
    model reported none."""
    # Source N is 1-based; out-of-range numbers are ignored.
    chosen = []
    for n in used_numbers:
        if 1 <= n <= len(results):
            chosen.append(results[n - 1])
    if not chosen:
        chosen = results  # fallback: show everything retrieved

    by_entity: dict[str, list[str]] = {}
    for r in chosen:
        meta = r.get("meta", {})
        entity = meta.get("entity_id", "?")
        url = meta.get("source_url", "")
        if not url:
            continue
        urls = by_entity.setdefault(entity, [])
        if url not in urls:
            urls.append(url)

    if not by_entity:
        return
    print("\n" + "-" * 80)
    print("SOURCES")
    print("-" * 80)
    for entity, urls in by_entity.items():
        for url in urls:
            print(f"  [{entity}] {url}")


def get_client(api_key: str | None = None) -> "genai.Client":
    """Build a Gemini client once, at startup, and reuse it."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=api_key)


def answer_query(query: str, k: int, mode: str, client: "genai.Client",
                 results: list[dict] | None = None, language: str = "en"):
    """Retrieve (unless results passed in), call Gemini, return (results,
    parsed). Printing is left to the caller."""
    import time
    if results is None:
        t0 = time.time()
        results = retrieve(query, k=k, entity=None, mode=mode)
        print(f"[time] retrieve: {time.time() - t0:.1f}s")
    if not results:
        return results, None

    prompt = build_prompt(query, build_context(results))
    t0 = time.time()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION + language_clause(language),
            # instruction-following over given sources, not a reasoning task;
            # the default 'high' just costs latency here.
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            response_mime_type="application/json",
            response_schema=AdvisorResponse,
        ),
    )
    print(f"[time] gemini call: {time.time() - t0:.1f}s")
    return results, resp.parsed


def generate(query: str, k: int, mode: str, speak_answer: bool = False) -> None:
    client = get_client()

    results, parsed = answer_query(query, k, mode, client)
    if not results:
        print("\nNo sources retrieved; nothing to answer from.")
        return

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(parsed.answer)

    print_grounding(results, parsed.claims)
    print_sources(results, parsed.sources_used)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG generation over retrieved chunks")
    p.add_argument("query", help="The user question")
    p.add_argument("--k", type=int, default=6,
                   help="top-k per entity passed to retrieval (default 6)")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="dense",
                   help="retrieval mode (default dense; heading-prepend made it sufficient)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.query, args.k, args.mode)


if __name__ == "__main__":
    main()