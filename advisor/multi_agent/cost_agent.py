"""Cost agent: fees, scholarships, financial aid, funding eligibility, awards."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):          # running as a script
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel

from advisor.core.retrieval import RetrievalSpec, search
from advisor.multi_agent import tools as T
from advisor.core.persona import (CLAIMS_INSTRUCTION, ELICITATION, PERSONA,
                                  SURFACING, language_clause, today_clause)
from advisor.core.llm import ANSWER_MODEL, generate
from advisor.multi_agent import verify as V

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
DOMAIN = "cost"
K = 5
MODE = "dense"
# entity types added to the planner's, never a filter
EXPAND_TYPES = ['scholarship']
# ranking prior only: content_category matches the text 32-90% of the time
PREFER_CATEGORIES = ['fees', 'scholarship_info']
# off: no verification. lite: V1+V2 flags decide, no judge. full: V1-V3.
VERIFY_MODE = V.env_mode()
VERIFY = VERIFY_MODE != "off"
# staleness is always checked; the live refetch is opt-in (a network round
# trip per turn)
LIVE_FETCH = os.environ.get("ADVISOR_LIVE_FETCH", "0") != "0"   

class Claim(BaseModel):
    text: str
    sources: list[int]


class Reading(BaseModel):
    """Spoken form of a figure in the answer, for the Urdu TTS (can't read
    bare digits). Empty for English."""
    written: str
    spoken: str


class Answer(BaseModel):
    answer: str
    claims: list[Claim]
    sources_used: list[int]
    number_readings: list[Reading] = []
    # one clarifying question or ""; separate from `answer` so assembly can
    # cap and dedupe
    question: str
    pivot: str


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
            sources: list[dict], abstained: bool,
            struck: list[dict] | None = None, as_of: str = "",
            scope_note: str = "", question: str = "",
            pivot: str = "", readings: list[dict] | None = None,
            verified: list[dict] | None = None,
            calibration: dict | None = None) -> dict:
    return {"partials": [{
        "domain": DOMAIN,
        "query": query,
        "answer": answer,
        "claims": claims,
        "sources": sources,
        "abstained": abstained,
        "struck": struck or [],
        "as_of": as_of,
        "scope_note": scope_note,
        "question": question,
        "pivot": pivot,
        "readings": readings or [],
        # every claim with its verdict (kept ones too), for verifier calibration
        "verified": verified or [],
        "calibration": calibration or {},
    }]}


def run(state: dict) -> dict:
    subtask = state["subtask"]
    query = subtask["query"]
    entities = subtask.get("entities") or []
    k = state.get("k", K)

    shape = subtask.get("shape", "lookup")
    language = state.get("language", "en")

    results, shape_block = T.gather_evidence(
        shape, query, entities, k, MODE, EXPAND_TYPES, PREFER_CATEGORIES,
        enumerate_kind='scholarship', enumerate_over=['university', 'scholarship'],
        programme=subtask.get("programme") or None)

    if VERIFY:
        # before generation, so the model never sees expired evidence
        expired = V.refresh_evidence(results, live=LIVE_FETCH)
        if expired:
            print(f"[{DOMAIN}] {expired} stale chunk(s) in evidence"
                  + (" (refreshed where possible)" if LIVE_FETCH else ""))

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

    tool_block = T.cost_summary(profile, results)
    ask_block = T.asked_before(state.get("history"))

    prompt = (
        f"SOURCES:\n{build_context(results)}\n-----{known}{shape_block}"
        f"{tool_block}{ask_block}\n"
        f"THE STUDENT ASKED: {state.get('raw_query') or query}\n"
        f"SEARCHED AS: {query}\n\n"
        f"Answer what they actually asked, using only the sources above."
    )

    try:
        parsed = generate(
            MODEL, prompt,
            system=(PERSONA + FRAMING + SURFACING + ELICITATION
                    + CLAIMS_INSTRUCTION + today_clause()
                    + language_clause(language)),
            schema=Answer,
        ).parsed
        if not parsed:
            raise ValueError("empty response")
        claims = [{"text": c.text, "sources": c.sources} for c in parsed.claims]
        answer = parsed.answer

        verified = []
        if VERIFY:
            verified, kept, struck = V.review(claims, results, query, profile,
                                              judge=VERIFY_MODE == "full")
            stale = V.hedge_stale_claims(verified, results)
            kept, struck = V.partition(verified)
            print(f"[{DOMAIN}] {len(results)} chunks, {len(claims)} claims, "
                  f"{V.summarise(verified)} {V.calibration(verified)}"
                  + (f" stale={stale}" if stale else ""))
            for s in struck:
                print(f"[{DOMAIN}] STRUCK ({s['verdict']}) {s['text']}")
                for n in s["notes"]:
                    print(f"[{DOMAIN}]        {n}")
            if struck and not kept:
                return partial(query, "I could not confirm any of that against "
                               "the official sources, so I would rather not "
                               "guess.", [], results, True, struck,
                               verified=verified,
                               calibration=V.calibration(verified))
            if struck:
                rewritten = V.rewrite_answer(query, kept, struck, PERSONA,
                                             language)
                if rewritten:
                    answer = rewritten
            claims = kept or claims
        else:
            struck = []
            print(f"[{DOMAIN}] {len(results)} chunks, {len(claims)} claims")

        return partial(query, answer, claims, results, False, struck,
                       V.as_of_date(verified) if VERIFY else "",
                       T.coverage_note(query, 'scholarship', language)
                       if shape == "enumeration" else "",
                       parsed.question.strip(), parsed.pivot.strip(),
                       [{'written': r.written, 'spoken': r.spoken}
                        for r in parsed.number_readings],
                       verified=verified,
                       calibration=V.calibration(verified) if VERIFY else {})

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