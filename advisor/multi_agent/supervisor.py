"""Supervisor: understand() before the agents, assemble() after them.

Rewrite and split happen in one call because they are the same reasoning, and
because the rewrite is needed on every voice query anyway -- a Python fast
path would almost never fire on speech-to-text input.

Splitting is by DOMAIN, never by institution. "Compare CS at LUMS and NUST"
stays ONE subtask listing both, so the agent compares them against both sets
of chunks at once rather than the comparison happening at merge time, away
from the evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):          # running as a script
    sys.path.insert(0, str(PROJECT_ROOT))

from google.genai import types
from pydantic import BaseModel

from advisor.core.persona import PERSONA
from advisor.core.entities import detect_entities
from advisor.core.llm import PLANNER_MODEL, THINKING_LEVEL, get_client

MODEL = PLANNER_MODEL
THINKING = THINKING_LEVEL

DOMAINS = ("admissions", "scholarships", "exams")
MAX_SUBTASKS = 3

ENTITIES_PATH = PROJECT_ROOT / "data" / "ingestion" / "entities.json"

client = get_client()


class _Subtask(BaseModel):
    domain: Literal["admissions", "scholarships", "exams"]
    entities: list[str]
    query: str


class _Fact(BaseModel):
    key: str
    value: str


class _Understanding(BaseModel):
    normalized_query: str
    corrections: list[str]
    subtasks: list[_Subtask]
    profile_updates: list[_Fact]


UNDERSTAND_PROMPT = """\
You prepare student questions about Pakistani university admissions for a \
retrieval system. You never answer the question.

Every message is either a QUESTION, or INFORMATION the student is supplying --
usually answering something you asked last turn ("merit seat", "I got 85%",
"pre-engineering"). Decide which before doing anything else.

If it is INFORMATION:
- record the facts in `profile_updates`
- set `normalized_query` to the earlier question this bears on, rewritten to
  include what they just told you. Never invent a different topic: if they say
  "merit seat" after you asked about their seat type, the question is still the
  scholarship question they originally asked, not a new one about merit seats
- only if nothing earlier relates should you treat it as a fresh question
- then split that question as below

If it is a QUESTION, rewrite it in clear, direct English:
- the input may be code-switched Urdu-English, written in Urdu script, or a \
rough speech-to-text transcript with mangled names
- correct garbled institution, exam, and scholarship names to the canonical \
ones listed below, but only when the intended one is obvious (usually a \
phonetic near-match). If you cannot tell which is meant, leave it alone
- never add an institution, programme, or scholarship the student did not \
mention or clearly imply. A general question stays general
- list every change you made in `corrections`

SECOND, split the rewritten question into subtasks, one per domain it touches:
- admissions: eligibility, programmes, fees, deadlines, how to apply
- scholarships: financial aid, scholarships, funding
- exams: entrance tests (NET, SAT), test format, scores, test rules

Rules for splitting:
- at most one subtask per domain, so never more than three
- most questions are ONE subtask. Only split when the question genuinely asks \
about different domains
- DO NOT split by institution. A comparison of two universities is ONE \
subtask listing both entity ids
- `entities` may only contain ids from the list below. Leave it empty if the \
question names no specific institution, exam, or scholarship
- each subtask `query` must stand on its own and be searchable by itself

THIRD, note any fact the student states about themselves that would change an
answer later, as `profile_updates`. Use short stable keys, for example:
seat_type (merit / self-finance), fsc_marks, matric_marks, board, intended_programme,
tests_taken, income_bracket, bisp_registered, domicile.
Record only what the student actually said, in this message or an earlier one.
Never infer, never guess, and return an empty list if they stated nothing.
"""

HISTORY_TURNS = 3
ANSWER_SNIPPET = 200


_catalogue: str | None = None


def entity_catalogue() -> str:
    """Known entity ids, given to the model so it can map a garbled or
    Urdu-script name ('looms', 'لمز') onto the right id ('lums')."""
    global _catalogue
    if _catalogue is None:
        try:
            data = json.loads(ENTITIES_PATH.read_text(encoding="utf-8"))
            _catalogue = "\n".join(
                f"- {eid}: {e.get('display_name', eid)} ({e.get('type', '')})"
                for eid, e in data.get("entities", {}).items()
            )
        except Exception as exc:
            print(f"[supervisor] could not read {ENTITIES_PATH}: {exc}")
            _catalogue = "(entity list unavailable)"
    return _catalogue


def reconcile_entities(raw_query: str, subtasks: list[dict]) -> list[str]:
    """Cross-check the model's entity ids against the alias matcher.

    Two checks, both per-subtask (never on the whole query -- that would leak
    every entity into every subtask):

      union        anything the matcher finds in the rewritten subtask query
                   but the model left out gets added.
      substitution if the raw query named entities and this subtask names a
                   completely different set, the model probably "corrected" a
                   name to the wrong one. Trust the raw query.

    Mutates `subtasks` in place, returns notes for logging.
    """
    raw_detected = set(detect_entities(raw_query))
    notes: list[str] = []

    for st in subtasks:
        model_said = set(st.get("entities") or [])
        in_rewrite = set(detect_entities(st["query"]))
        merged = model_said | in_rewrite

        missed = in_rewrite - model_said
        if missed:
            notes.append(f"{st['domain']}: matcher added {sorted(missed)}")

        if raw_detected and merged and not (raw_detected & merged):
            notes.append(
                f"{st['domain']}: {sorted(merged)} shares nothing with the raw "
                f"query's {sorted(raw_detected)} -- possible bad correction, "
                f"trusting the raw query"
            )
            merged = raw_detected

        st["entities"] = sorted(merged)

    return notes


def understand(state: dict) -> dict:
    """Rewrite the query and split it into subtasks.

    Any failure falls back to a single admissions subtask carrying the raw
    query, so an outage degrades to roughly baseline behaviour rather than
    taking the system down.
    """
    raw = state["raw_query"]

    parts = [f"Known entity ids:\n{entity_catalogue()}\n"]
    history = state.get("history") or []
    if history:
        recent = "\n".join(
            f"Q: {h.get('q','')}\nA: {h.get('a','')}" for h in history[-HISTORY_TURNS:]
        )
        parts.append(
            f"Earlier in this session:\n{recent}\n\n"
            "If the new question leans on this (pronouns, or a missing "
            "institution), make it stand on its own.\n"
        )
    profile = state.get("profile") or {}
    if profile:
        known = "\n".join(f"- {k}: {v}" for k, v in profile.items())
        parts.append(f"Already known about this student:\n{known}\n")
    parts.append(f"Question:\n{raw}")

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents="\n".join(parts),
            config=types.GenerateContentConfig(
                system_instruction=UNDERSTAND_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_level=THINKING),
                response_mime_type="application/json",
                response_schema=_Understanding,
            ),
        )
        u = resp.parsed
        if not u or not u.subtasks:
            raise ValueError("no subtasks returned")

        subtasks = [
            {"domain": s.domain, "entities": list(s.entities), "query": s.query}
            for s in u.subtasks
            if s.domain in DOMAINS
        ][:MAX_SUBTASKS]
        if not subtasks:
            raise ValueError("no valid subtasks after filtering")

        notes = reconcile_entities(raw, subtasks)

        print(f"\n[supervisor] {raw!r}\n         --> {u.normalized_query!r}")
        for c in u.corrections:
            print(f"    fixed: {c}")
        for n in notes:
            print(f"    check: {n}")
        for s in subtasks:
            print(f"    -> {s['domain']:13} {s['entities']} :: {s['query']!r}")

        profile = dict(state.get("profile") or {})
        for f in u.profile_updates:
            if f.key and f.value:
                profile[f.key] = f.value
                print(f"    learned: {f.key} = {f.value}")

        return {
            "normalized_query": u.normalized_query,
            "corrections": list(u.corrections),
            "entity_notes": notes,
            "subtasks": subtasks,
            "profile": profile,
            "partials": None,   # clear the previous turn's agent results
        }

    except Exception as exc:
        print(f"[supervisor] understanding failed ({exc}); using the raw query")
        return {
            "normalized_query": raw,
            "corrections": [],
            "entity_notes": [],
            "subtasks": [{"domain": "admissions",
                          "entities": detect_entities(raw),
                          "query": raw}],
            "profile": dict(state.get("profile") or {}),
            "partials": None,
        }


# The merged text is what the student hears, so it carries the same persona
# and safety rules as a single agent answer. Keep in step with the baseline's
# SYSTEM_INSTRUCTION.
ASSEMBLE_PROMPT = PERSONA + (
    "\nWrite ONLY the answer itself. No greeting, no list of claims, no source "
    "markers -- this text is read aloud to the student as-is.\n"
    "\nYou are combining answers that specialist colleagues produced for "
    "different parts of one student question, each grounded in its own "
    "sources.\n"
    "- Keep every factual claim exactly as given. Add nothing, soften nothing.\n"
    "- Where a colleague could not find an answer, say so plainly.\n"
    "- Do not repeat the same fact twice.\n"
    "- One coherent reply, not a list of separate answers."
)


def assemble(state: dict) -> dict:
    """Merge the agents' partial answers.

    A single partial passes straight through with no LLM call. Most questions
    are single-domain, so most questions skip merging entirely.
    """
    partials = state.get("partials") or []

    if not partials:
        return {"answer": "I could not find anything relevant to that question.",
                "claims": [], "sources": []}


    # each agent numbers its own sources from 1, so the numbers collide once
    # partials are merged. Renumber into the combined list as we go, otherwise
    # a claim cites another agent's chunk.
    claims: list[dict] = []
    sources: list[dict] = []
    for p in partials:
        offset = len(sources)
        sources.extend(p.get("sources", []))
        for c in p.get("claims", []):
            claims.append({
                "text": c["text"],
                "sources": [n + offset for n in c.get("sources", [])],
                "domain": p["domain"],
            })

    def _with_history(answer: str) -> dict:
        turn = {"q": state["raw_query"], "a": answer[:ANSWER_SNIPPET]}
        past = list(state.get("history") or [])[-(HISTORY_TURNS - 1):]
        return {"answer": answer, "claims": claims, "sources": sources,
                "history": past + [turn]}

    if len(partials) == 1:
        return _with_history(partials[0]["answer"])

    blocks = "\n\n".join(
        f"[{p['domain']} - "
        f"{'could not answer' if p.get('abstained') else 'answered'}]\n{p['answer']}"
        for p in partials
    )
    asked = state.get("normalized_query") or state["raw_query"]

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=f"The student asked: {asked}\n\n{blocks}",
            config=types.GenerateContentConfig(
                system_instruction=ASSEMBLE_PROMPT,
                thinking_config=types.ThinkingConfig(thinking_level=THINKING),
            ),
        )
        answer = (resp.text or "").strip()
        if not answer:
            raise ValueError("empty merge")
    except Exception as exc:
        print(f"[supervisor] assembly failed ({exc}); concatenating instead")
        answer = "\n\n".join(p["answer"] for p in partials)

    return _with_history(answer)


if __name__ == "__main__":
    # exercise understanding on its own -- no agents, no retrieval, no graph
    import argparse

    ap = argparse.ArgumentParser(description="test the supervisor's understanding step")
    ap.add_argument("query")
    ap.add_argument("--history", nargs="*", default=None)
    args = ap.parse_args()

    out = understand({"raw_query": args.query, "history": args.history})
    print(f"\nsubtasks: {len(out['subtasks'])}")