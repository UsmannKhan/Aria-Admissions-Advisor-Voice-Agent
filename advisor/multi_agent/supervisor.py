"""Supervisor: understand() before the agents, assemble() after them.

Rewrite and split in one call; voice queries always need the rewrite, so a
Python fast path would almost never fire.

Subtasks split by domain, not institution: "Compare CS at LUMS and NUST" is
one subtask, so one agent compares with both sets of chunks in front of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):          # running as a script
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel

from advisor.core.persona import PERSONA, language_clause, today_clause
from advisor.core.entities import detect_entities
from advisor.core.llm import PLANNER_MODEL, generate
from advisor.multi_agent import verify as V

MODEL = PLANNER_MODEL
# policy gate and conflict check follow ADVISOR_VERIFY too
MODE = V.env_mode()

DOMAINS = ("admissions", "cost", "exams")
MAX_SUBTASKS = 3

ENTITIES_PATH = PROJECT_ROOT / "data" / "ingestion" / "entities.json"


class _Subtask(BaseModel):
    domain: Literal["admissions", "cost", "exams"]
    shape: Literal["lookup", "comparison", "enumeration"]
    entities: list[str]
    # canonical programme name from the catalogue, or "". The model normalises
    # "کمپیوٹر سائنس" or "CS" here; the matcher downstream only knows clean names
    programme: str = ""
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
- admissions: eligibility, programmes, deadlines, how to apply, documents
- cost: anything about money -- fee structures, what a degree costs, financial
  aid, scholarships, funding, affordability. Fees and aid are ONE domain
  because "what will this cost me" needs both together
- exams: entrance tests (NET, SAT, MDCAT), test format, scores, test rules, and
  WHICH INSTITUTIONS ACCEPT A GIVEN TEST. "Which universities accept the SAT"
  is an exams question, not an admissions one, even though it names
  universities and a programme -- the subject is the test

ALSO give each subtask a SHAPE, which decides how it is executed:
- lookup: one thing about one institution, or a general question. Most
  questions are this
- comparison: the student names two or more institutions and wants them set
  against each other. "Compare CS eligibility at LUMS and NUST"
- enumeration: the student asks WHICH institutions do something, without
  naming them. "Which universities accept the SAT", "what scholarships can I
  apply for". The answer is a list of institutions, not a fact about one

Rules for splitting:
- at most one subtask per domain, so never more than three
- most questions are ONE subtask. Only split when the question genuinely asks \
about different domains
- DO NOT split by institution. A comparison of two universities is ONE \
subtask listing both entity ids
- `entities` may only contain ids from the list below. Leave it empty if the \
question names no specific institution, exam, or scholarship
- if the student names a degree programme -- in any language, script or \
abbreviation ("CS", "کمپیوٹر سائنس", BBA) -- set `programme` to its canonical \
name from the programme list below. Leave it empty when no programme is \
named, and never add one the student did not say: "what are the fees" names \
no programme, and which programme it is may be exactly what needs asking
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
_prog_catalogue: str | None = None


def prog_catalogue() -> str:
    """Ingested programmes and their institutions. Lazy, so routing still
    works when the corpus is down."""
    global _prog_catalogue
    if _prog_catalogue is None:
        try:
            from advisor.core.retrieval import programme_catalogue
            _prog_catalogue = "\n".join(f"- {p}" for p in programme_catalogue())
        except Exception as exc:
            print(f"[supervisor] programme catalogue unavailable: {exc}")
            _prog_catalogue = "(programme list unavailable)"
    return _prog_catalogue


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
    """Cross-check the model's entity ids against the alias matcher, per
    subtask (whole-query matching leaks every entity into every subtask).

    Adds entities the matcher finds in the subtask query but the model missed.
    If the raw query named entities and the subtask names a different set, the
    model probably "corrected" a name wrongly; the raw query wins.
    Mutates `subtasks`; returns notes for logging.
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


def reconcile_programmes(raw_query: str, subtasks: list[dict]) -> list[str]:
    """Same checks for the programme field. Drops programmes the corpus can't
    resolve (scoping to them matches nothing), fills in ones the matcher finds,
    and normalises to the slug form via detect_programme.
    """
    from advisor.core.retrieval import detect_programme

    notes: list[str] = []
    for st in subtasks:
        given = (st.get("programme") or "").strip()
        resolved = detect_programme(given) if given else ""
        if given and not resolved:
            notes.append(f"{st['domain']}: programme {given!r} matches nothing "
                         f"ingested, dropped")
        st["programme"] = resolved
        if not resolved:
            fallback = detect_programme(st["query"]) or detect_programme(raw_query)
            if fallback:
                st["programme"] = fallback
                notes.append(f"{st['domain']}: matcher added programme "
                             f"{fallback!r}")
    return notes


def understand(state: dict) -> dict:
    """Rewrite the query and split it into subtasks. On any failure: one
    admissions subtask with the raw query (roughly the baseline).
    """
    raw = state["raw_query"]

    parts = [f"Known entity ids:\n{entity_catalogue()}\n",
             f"Known programmes:\n{prog_catalogue()}\n"]
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
        u = generate(MODEL, "\n".join(parts),
                     system=UNDERSTAND_PROMPT, schema=_Understanding).parsed
        if not u or not u.subtasks:
            raise ValueError("no subtasks returned")

        subtasks = [
            {"domain": s.domain, "shape": s.shape,
             "entities": list(s.entities),
             "programme": s.programme.strip(), "query": s.query}
            for s in u.subtasks
            if s.domain in DOMAINS
        ][:MAX_SUBTASKS]
        if not subtasks:
            raise ValueError("no valid subtasks after filtering")

        notes = reconcile_entities(raw, subtasks)
        notes += reconcile_programmes(raw, subtasks)

        print(f"\n[supervisor] {raw!r}\n         --> {u.normalized_query!r}")
        for c in u.corrections:
            print(f"    fixed: {c}")
        for n in notes:
            print(f"    check: {n}")
        for s in subtasks:
            prog = f" [{s['programme']}]" if s.get("programme") else ""
            print(f"    -> {s['domain']:11} {s['shape']:11} {s['entities']}"
                  f"{prog} :: {s['query']!r}")

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
            "subtasks": [{"domain": "admissions", "shape": "lookup",
                          "entities": detect_entities(raw),
                          "query": raw}],
            "profile": dict(state.get("profile") or {}),
            "partials": None,
        }


# same persona and safety rules as an agent answer, since this is what the
# student hears. Keep in step with the baseline's SYSTEM_INSTRUCTION.
ASSEMBLE_PROMPT = PERSONA + (
    "\nWrite ONLY the answer itself. No greeting, no list of claims, no source "
    "markers -- this text is read aloud to the student as-is.\n"
    "\nYou are combining answers that specialist colleagues produced for "
    "different parts of one student question, each grounded in its own "
    "sources.\n"
    "- Keep every factual claim exactly as given. Add nothing, soften nothing.\n"
    "- Where a colleague could not find an answer, say so plainly.\n"
    "- Do not repeat the same fact twice.\n"
    "- One coherent reply, not a list of separate answers.\n"
    "- If a CONFLICTS block is present, two authorities disagree. Never quietly "
    "pick one. Give both figures, say which body each comes from, and tell the "
    "student to confirm with the one that will decide their case.\n"
    "- Anything marked 'as of' is a fact that may have moved since it was last "
    "checked. Say the date and say it is worth confirming."
)


def assemble(state: dict) -> dict:
    """Merge the agents' partial answers. A single partial (most questions)
    passes through with no LLM call.
    """
    partials = state.get("partials") or []

    if not partials:
        return {"answer": "I could not find anything relevant to that question.",
                "claims": [], "sources": []}


    # each agent numbers its sources from 1; renumber into the merged list or
    # claims end up citing another agent's chunk
    claims: list[dict] = []
    sources: list[dict] = []
    struck: list[dict] = []
    for p in partials:
        offset = len(sources)
        sources.extend(p.get("sources", []))
        struck.extend(p.get("struck", []))
        for c in p.get("claims", []):
            entity = ""
            if c.get("sources"):
                first = c["sources"][0] + offset
                if 1 <= first <= len(sources):
                    entity = sources[first - 1].get("meta", {}).get("entity_id", "")
            claims.append({
                "text": c["text"],
                "sources": [n + offset for n in c.get("sources", [])],
                "domain": p["domain"],
                "entity": entity,
            })

    questions = V.collect_questions(partials)
    for q in questions:
        print(f"[assemble] asking {q['pivot']}: {q['question']}")

    def _with_history(answer: str, violations: list[str] | None = None) -> dict:
        turn = {"q": state["raw_query"], "a": answer[:ANSWER_SNIPPET],
                "asked": ", ".join(q["pivot"] for q in questions)}
        past = list(state.get("history") or [])[-(HISTORY_TURNS - 1):]
        return {"answer": answer, "claims": claims, "sources": sources,
                "violations": violations or [],
                "readings": V.collect_readings(partials, answer),
                "history": past + [turn]}

    language = state.get("language", "en")

    # check before splitting: one partial can hold two authorities (a
    # scholarship subtask with ['hec_nbs', 'lums']). Needs the judge, so full
    # arm only.
    conflicts = (V.adjudicate_conflicts(V.find_conflicts(claims))
                 if MODE == "full" else [])
    for c in conflicts:
        print(f"[conflict] {c['a'].get('entity')} vs {c['b'].get('entity')}: "
              f"{c.get('disagreement', '')}")

    if len(partials) == 1:
        answer, violations = partials[0]["answer"], []
        if MODE != "off":
            answer, violations = V.policy_gate(answer, struck, language,
                                               judge=MODE == "full",
                                               kept=claims, query=state["raw_query"])
        answer = V.append_hedge(answer, partials[0].get("as_of", ""), language)
        answer += partials[0].get("scope_note", "")
        answer += V.conflict_note(conflicts, language)
        answer += V.question_note(questions)
        for v in violations:
            print(f"[policy] {v}")
        return _with_history(answer, violations)

    blocks = "\n\n".join(
        f"[{p['domain']} - "
        f"{'could not answer' if p.get('abstained') else 'answered'}]\n{p['answer']}"
        for p in partials
    )

    if conflicts:
        detail = "\n".join(
            f"- {c['a'].get('entity','?')}: {c['a']['text']}\n"
            f"  {c['b'].get('entity','?')}: {c['b']['text']}" for c in conflicts)
        blocks += f"\n\nCONFLICTS:\n{detail}"

    asked = state.get("normalized_query") or state["raw_query"]

    try:
        answer = generate(MODEL, f"The student asked: {asked}\n\n{blocks}",
                          system=(ASSEMBLE_PROMPT + today_clause()
                                  + language_clause(language, structured=False))
                          ).text.strip()
        if not answer:
            raise ValueError("empty merge")
    except Exception as exc:
        print(f"[supervisor] assembly failed ({exc}); concatenating instead")
        answer = "\n\n".join(p["answer"] for p in partials)

    # the merge is a model call, so struck facts can come back
    violations = []
    if MODE != "off":
        answer, violations = V.policy_gate(answer, struck, language,
                                           judge=MODE == "full",
                                           kept=claims, query=state["raw_query"])
    oldest = min((p.get("as_of", "") for p in partials if p.get("as_of")), default="")
    answer = V.append_hedge(answer, oldest, language)
    answer += "".join(p.get("scope_note", "") for p in partials)
    answer += V.conflict_note(conflicts, language)
    answer += V.question_note(questions)
    for v in violations:
        print(f"[policy] {v}")
    return _with_history(answer, violations)


if __name__ == "__main__":
    # exercise understanding on its own: no agents, no retrieval, no graph
    import argparse

    ap = argparse.ArgumentParser(description="test the supervisor's understanding step")
    ap.add_argument("query")
    ap.add_argument("--history", nargs="*", default=None)
    args = ap.parse_args()

    out = understand({"raw_query": args.query, "history": args.history})
    print(f"\nsubtasks: {len(out['subtasks'])}")