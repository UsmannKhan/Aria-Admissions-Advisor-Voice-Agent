"""
Single-agent RAG: retrieve -> prompt -> ANSWER_MODEL -> cited answer. This is
the baseline arm of the evaluation, so its behaviour is pinned.

    python advisor/baseline/pipeline.py "compare CS eligibility at LUMS and NUST"
    python advisor/baseline/pipeline.py "what are the LUMS fees" --k 6 --mode dense
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):          # running as a script
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel

from advisor.core.retrieval import retrieve
from advisor.core.persona import language_clause
from advisor.core.llm import ANSWER_MODEL, generate as llm_generate

MODEL = ANSWER_MODEL


class Claim(BaseModel):
    """One factual claim and the source number(s) backing it, as reported by
    the generator."""
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


def build_prompt(query: str, context: str) -> str:
    return (
        f"SOURCES:\n{context}\n"
        f"-----\n"
        f"QUESTION: {query}\n\n"
        f"Answer using only the sources above, following the rules."
    )


def print_grounding(results: list[dict], claims: list) -> None:
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


def answer_query(query: str, k: int, mode: str,
                 results: list[dict] | None = None, language: str = "en"):
    """Retrieve (unless results passed in), call the answer model, return
    (results, parsed). Printing is left to the caller."""
    import time
    if results is None:
        t0 = time.time()
        results = retrieve(query, k=k, entity=None, mode=mode)
        print(f"[time] retrieve: {time.time() - t0:.1f}s")
    if not results:
        return results, None

    prompt = build_prompt(query, build_context(results))
    t0 = time.time()
    parsed = llm_generate(
        MODEL, prompt,
        system=SYSTEM_INSTRUCTION + language_clause(language),
        schema=AdvisorResponse,
    ).parsed
    print(f"[time] llm call: {time.time() - t0:.1f}s")
    return results, parsed


def generate(query: str, k: int, mode: str, speak_answer: bool = False) -> None:
    results, parsed = answer_query(query, k, mode)
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