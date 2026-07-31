"""Scholarships agent: financial aid, funding eligibility, awards, applying."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):          # running as a script
    sys.path.insert(0, str(PROJECT_ROOT))

from google.genai import types
from pydantic import BaseModel

from advisor.core.retrieval import retrieve
from advisor.core.persona import CLAIMS_INSTRUCTION, PERSONA
from advisor.core.llm import ANSWER_MODEL, THINKING_LEVEL, get_client

FRAMING = """
You are handling the scholarships part of this question: financial aid, funding
eligibility, award amounts, and how to apply.

- Several schemes are all described as "need-based" and must not be blurred
  together. For each one, say which body administers it and where the student
  actually applies, since those differ: some are applied for through the
  university, others through the awarding body's own portal.
- Some schemes exclude self-finance admissions, private universities, or
  affiliated campuses. State such a rule as the rule. Do not decide whether a
  particular institution falls under it unless the sources say so.
- Eligibility often turns on household income or BISP registration, which the
  sources do not contain. Explain the criteria, then ask the student directly
  for the one or two details you would need to tell them where they stand.
"""

MODEL = ANSWER_MODEL
THINKING = THINKING_LEVEL
DOMAIN = "scholarships"
K = 5
MODE = "dense"   

client = get_client()


class Claim(BaseModel):
    text: str
    sources: list[int]


class Answer(BaseModel):
    answer: str
    claims: list[Claim]
    sources_used: list[int]


def build_context(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        meta = r.get("meta", {})
        lines.append(
            f"[Source {i}] institution={meta.get('entity_id','?')} | "
            f"section: {meta.get('headings','')}\n{r.get('doc','')}\n"
        )
    return "\n".join(lines)


def partial(query: str, answer: str, claims: list[dict],
            sources: list[dict], abstained: bool) -> dict:
    return {"partials": [{
        "domain": DOMAIN,
        "query": query,
        "answer": answer,
        "claims": claims,
        "sources": sources,
        "abstained": abstained,
    }]}


def run(state: dict) -> dict:
    subtask = state["subtask"]
    query = subtask["query"]
    entities = subtask.get("entities") or []
    k = state.get("k", K)

    results: list[dict] = []
    if entities:
        for eid in entities:
            results.extend(retrieve(query, k=k, entity=eid, mode=MODE))
    else:
        results = retrieve(query, k=k, entity=None, mode=MODE)

    if not results:
        print(f"[{DOMAIN}] no chunks retrieved")
        return partial(query, f"I could not find anything about {query}",
                       [], [], True)

    profile = state.get("profile") or {}
    known = ""
    if profile:
        facts = "\n".join(f"- {k}: {v}" for k, v in profile.items())
        known = (f"\nWHAT THE STUDENT HAS ALREADY TOLD YOU:\n{facts}\n"
                 "Apply these where the sources make them relevant, and do not "
                 "ask again for anything listed here.\n")

    prompt = (
        f"SOURCES:\n{build_context(results)}\n-----{known}\n"
        f"QUESTION: {query}\n\nAnswer using only the sources above."
    )

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=PERSONA + FRAMING + CLAIMS_INSTRUCTION,
                thinking_config=types.ThinkingConfig(thinking_level=THINKING),
                response_mime_type="application/json",
                response_schema=Answer,
            ),
        )
        parsed = resp.parsed
        if not parsed:
            raise ValueError("empty response")
        claims = [{"text": c.text, "sources": c.sources} for c in parsed.claims]
        print(f"[{DOMAIN}] {len(results)} chunks, {len(claims)} claims")
        return partial(query, parsed.answer, claims, results, False)

    except Exception as exc:
        print(f"[{DOMAIN}] generation failed ({exc})")
        return partial(query, "I ran into a problem looking up the funding details.",
                       [], results, True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--entities", nargs="*", default=[])
    ap.add_argument("-k", type=int, default=K)
    args = ap.parse_args()

    out = run({"subtask": {"domain": DOMAIN, "entities": args.entities,
                           "query": args.query}, "k": args.k})
    p = out["partials"][0]
    print("\n" + "=" * 70)
    print(p["answer"])
    print("=" * 70)
    for c in p["claims"]:
        print(f"  {c['sources']} {c['text']}")