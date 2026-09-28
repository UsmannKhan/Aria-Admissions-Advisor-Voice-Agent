"""What each agent computes over its own evidence before generating.

Extract and arrange only, so V2 can still check a computed total against the
table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "data" / "ingestion"))
import config as cfg

from advisor.core.retrieval import (RetrievalSpec, entities_by_type,
                                    entities_mentioning, search)
from advisor.multi_agent.verify import display_name, numbers

import datetime as _dt

# skip years ("FY 2026-27", "Fall 2026"): listed next to 45,800, a 2,026
# invites the model to state a year as a fee
_YEAR_MIN, _YEAR_MAX = 1990, _dt.date.today().year + 15


def _is_money(n: float) -> bool:
    return n >= 1000 and not (_YEAR_MIN <= n <= _YEAR_MAX and float(n).is_integer())


def _heading_of(chunk: dict) -> str:
    return str(chunk.get("meta", {}).get("headings", ""))[:70]


def asked_before(history: list[dict] | None) -> str:
    """The ask-first rule again, right after the evidence. In ELICITATION
    alone (a 5,000-char system prompt) the model ignored it and listed all
    five schemes instead of asking. A dodge (student changed the subject)
    isn't visible in evidence or profile, so that comes from history.
    """
    block = ("\nBEFORE ANSWERING, CHECK THE CONDITIONS\n"
             "Read what the sources above require, not just what they offer. "
             "If one of those conditions decides which options apply to this "
             "student, and you cannot settle it from what they have told you, "
             "put that question in `question` and the fact it establishes in "
             "`pivot`, then keep the answer to the shape of the options rather "
             "than every branch. If nothing is genuinely undecided, leave both "
             "empty and answer in full.\n")

    asked = [h.get("asked") for h in (history or [])[-2:] if h.get("asked")]
    if asked:
        block += ("You already asked about: " + ", ".join(asked)
                  + ". They did not answer, so do not ask again -- answer with "
                    "what you have and say which part still depends on it.\n")
    return block


_ACADEMIC_KEYS = ("fsc_marks", "matric_marks", "percentage", "cgpa", "board",
                  "tests_taken", "intended_programme")


def academic_check(profile: dict, results: list[dict]) -> str:
    """Line up what the student said against the thresholds in the sources.
    No conclusion: "you meet the published minimum" is fine, "you will get
    in" is a prediction (the policy gate flags those).
    """
    stated = {k: v for k, v in (profile or {}).items() if k in _ACADEMIC_KEYS}
    if not stated:
        return ""

    thresholds = []
    for r in results:
        for pct in re.findall(r"(\d{1,3})\s*%", r.get("doc", "")):
            if 30 <= int(pct) <= 100:
                thresholds.append((int(pct), _heading_of(r)))
    seen, uniq = set(), []
    for pct, head in thresholds:
        if pct not in seen:
            seen.add(pct)
            uniq.append(f"{pct}% ({head})")

    lines = ["\nACADEMIC PROFILE CHECK",
             "What the student has told you:"]
    lines += [f"- {k}: {v}" for k, v in stated.items()]
    if uniq:
        lines.append("Percentage thresholds present in the sources above:")
        lines += [f"- {u}" for u in uniq[:8]]
        lines.append("Say plainly which of these published minimums the "
                     "student's figures meet and which they do not. Never say "
                     "or imply whether they will be admitted.")
    else:
        lines.append("The sources state no percentage threshold, so say so "
                     "rather than estimating one.")
    return "\n".join(lines) + "\n"


_FINANCIAL_KEYS = ("income_bracket", "bisp_registered", "seat_type",
                   "household_income", "financial_aid_applied")


def cost_summary(profile: dict, results: list[dict]) -> str:
    """Group every figure in the evidence by the body that published it, so a
    university fee and a scholarship award don't get blended into one number.
    """
    by_entity: dict[str, list[str]] = {}
    for r in results:
        amounts = sorted(n for n in numbers(r.get("doc", "")) if _is_money(n))
        if not amounts:
            continue
        eid = r.get("meta", {}).get("entity_id", "?")
        row = (f"{_heading_of(r)}: "
               + ", ".join(f"{a:,.0f}" for a in amounts[:8]))
        by_entity.setdefault(eid, []).append(row)

    lines = []
    if by_entity:
        lines.append("\nFIGURES IN THE SOURCES, BY THE BODY THAT PUBLISHED THEM")
        for eid, rows in by_entity.items():
            lines.append(f"[{display_name(eid)}]")
            lines += [f"- {r}" for r in rows[:5]]
        lines.append("Attribute every figure to its body. Fee bases differ "
                     "between institutions and between a fee and an award, so "
                     "never add or compare figures from different bodies "
                     "unless the sources say they are comparable.")

    stated = {k: v for k, v in (profile or {}).items() if k in _FINANCIAL_KEYS}
    if stated:
        lines.append("\nWhat the student has told you about their finances:")
        lines += [f"- {k}: {v}" for k, v in stated.items()]
        lines.append("Say which stated eligibility rules this meets. Do not "
                     "decide whether they will receive an award.")
    return "\n".join(lines) + "\n" if lines else ""


def named_exam(query: str) -> str | None:
    """The exam the student asked about, matched against the crawler's list so
    the term is one the corpus indexed."""
    low = query.lower()
    best = None
    for e in cfg.KNOWN_EXAMS:
        if re.search(rf"\b{re.escape(e.lower())}\b", low):
            if best is None or len(e) > len(best):
                best = e
    return best


_KNOWN_TERMS = {"exam": cfg.KNOWN_EXAMS,
                "scholarship": cfg.KNOWN_SCHOLARSHIPS,
                "programme": cfg.KNOWN_PROGRAMS}


def named_term(query: str, kind: str) -> str | None:
    """The longest indexed term of this kind named in the query."""
    low = query.lower()
    best = None
    for t in _KNOWN_TERMS.get(kind, []):
        if re.search(rf"\b{re.escape(t.lower())}\b", low):
            if best is None or len(t) > len(best):
                best = t
    return best


def enumerate_entities(query: str, kind: str = "exam",
                       over: list[str] | None = None,
                       k: int = 3) -> tuple[str, list[dict]]:
    """Entities that carry a term, each with a confirming passage. Dense
    search answers "which universities accept the SAT" with the exam body's
    pages, so candidates come from a metadata scan. Recall is limited to the
    crawler's KNOWN_* lists.
    """
    term = named_term(query, kind)
    if not term:
        return "", []

    over = over or ["university"]
    candidates = entities_mentioning(term, kind, entity_types=over)
    if not candidates:
        return "", []

    extra, rows = [], []
    for eid in candidates:
        hits = search(RetrievalSpec(f"{term} {query}", k=k, entities=[eid],
                                    prefer_categories=["eligibility", "test_format",
                                                       "scholarship_info"],
                                    expand=0))
        extra.extend(hits)
        rows.append(f"- {display_name(eid)}: {len(hits)} passage(s) above")

    block = (f"\nENTITIES WHOSE PAGES MENTION {term.upper()}\n"
             + "\n".join(rows)
             + "\nState the terms each one gives. Do NOT say this list is "
               "complete or exhaustive -- that is a fact about the knowledge "
               "base, not about any source, and it is added separately.\n")
    return block, extra


# kept for readability at the exams call site
def acceptance_table(query: str, k: int = 3) -> tuple[str, list[dict]]:
    return enumerate_entities(query, "exam", over=["university"], k=k)


def comparison_blocks(query: str, entities: list[str], k: int = 4,
                      expand_types: list[str] | None = None,
                      prefer_categories: list[str] | None = None,
                      programme: str | None = None
                      ) -> tuple[str, list[dict]]:
    """Retrieve each institution separately and name the ones with nothing.
    Pooled, the bigger site crowds out one with no coverage, and the student
    hears "no requirement" instead of "not published".
    """
    if len(entities) < 2:
        return "", []

    extra, found, missing = [], [], []
    for eid in entities:
        hits = search(RetrievalSpec(query, k=k, entities=[eid],
                                    expand_types=expand_types,
                                    prefer_categories=prefer_categories,
                                    programme=programme, expand=0))
        if hits:
            extra.extend(hits)
            found.append(f"- {display_name(eid)}: {len(hits)} passage(s) above")
        else:
            missing.append(display_name(eid))

    block = "\nCOMPARISON, RETRIEVED PER INSTITUTION\n" + "\n".join(found)
    if missing:
        block += ("\nNothing was found for: " + ", ".join(missing)
                  + "\nSay so for each of these by name. Do not leave one out "
                    "silently -- an omission reads as 'no requirement' when it "
                    "means 'not published in what I have'.")
    block += ("\nAddress each institution separately before drawing any "
              "comparison, and never carry a figure from one to another.\n")
    return block, extra


def gather_evidence(shape: str, query: str, entities: list[str], k: int,
                    mode: str, expand_types: list[str],
                    prefer_categories: list[str],
                    enumerate_kind: str = "exam",
                    enumerate_over: list[str] | None = None,
                    programme: str | None = None
                    ) -> tuple[list[dict], str]:
    """Retrieve by the planner's shape; unknown shapes fall back to lookup."""
    results = search(RetrievalSpec(
        query=query, k=k, entities=entities, expand_types=expand_types,
        prefer_categories=prefer_categories, mode=mode, programme=programme))
    block = ""

    if shape == "enumeration":
        block, extra = enumerate_entities(query, enumerate_kind, enumerate_over)
    elif shape == "comparison":
        block, extra = comparison_blocks(query, entities, k=k,
                                         expand_types=expand_types,
                                         prefer_categories=prefer_categories,
                                         programme=programme)
    else:
        extra = []

    if extra:
        seen = {r["id"] for r in results}
        results.extend(r for r in extra if r["id"] not in seen)
    return results, block


def coverage_note(query: str, kind: str = "exam", language: str = "en") -> str:
    """Caveat that an enumeration only covers what's ingested. Not a claim:
    "LUMS and NUST are the only ones" is true of the corpus but in no chunk,
    so V3 would strike it. Appended after assembly like the staleness hedge.
    """
    term = named_term(query, kind)
    if not term:
        return ""
    covered = [display_name(e) for e in entities_by_type("university")]
    if not covered:
        return ""
    listed = " and ".join(filter(None, [", ".join(covered[:-1]), covered[-1]]))
    if language == "ur":
        return (f" ابھی میرے پاس صرف {listed} کی معلومات ہیں، اس لیے ممکن ہے "
                f"دوسرے ادارے بھی {term} سے متعلق ہوں۔")
    return (f" I should say that I only cover {listed} at the moment, so other "
            f"institutions may be relevant to {term} too.")
