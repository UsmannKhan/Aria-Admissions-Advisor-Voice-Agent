"""Claim verification.

V1 structural: citation resolves; named institution matches the cited chunk.
V2 numeric: every figure is in the evidence, derivable from it, or the student's.
V3 entailment: one batched judge call on VERIFIER_MODEL.

V1/V2 only flag, V3 decides, except a claim with no resolvable citation,
which is struck. Flags alone struck correct claims (the 27 in "FY 2026-27"
was in a heading V2 wasn't reading). V2 catches near-misses the judge calls
close enough: 1,555,000 vs 1,557,200 in the table.

    python advisor/multi_agent/verify.py [--judge]
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):          # running as a script
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Literal

from pydantic import BaseModel

from advisor.core.entities import detect_entities
from advisor.core.persona import language_clause, today_clause
from advisor.core.llm import ANSWER_MODEL, VERIFIER_MODEL, generate

SUPPORTED = "supported"
OVERREACH = "overreach"      # source is narrower or weaker than the claim
UNSUPPORTED = "unsupported"
NEEDS_REVIEW = "needs_review"  # awaiting the judge
STALE = "stale"              # V4: supported, but the evidence has expired

CONDEMNED = (UNSUPPORTED, OVERREACH)
# stale claims are kept, with an "as of" date added to the answer
HEDGED = (STALE,)

# Flags raised by V1/V2 for the judge:
#   no_citation      nothing resolvable to judge against (the only flag that strikes)
#   entity_mismatch  claim names an institution the cited chunk is not
#   figure_absent    a figure in the claim is not in the evidence
#   month_absent     likewise for a month name

def env_mode() -> str:
    """Verification mode from ADVISOR_VERIFY.

    off   no verification layer at all
    lite  V1+V2 only; the flags decide, no judge
    full  V1-V3 plus policy, conflicts and staleness

    '0'/'1' still mean off/full (older scripts).
    """
    raw = os.environ.get("ADVISOR_VERIFY", "full").strip().lower()
    mode = {"0": "off", "1": "full", "": "full"}.get(raw, raw)
    return mode if mode in ("off", "lite", "full") else "full"

# Urdu digits and separators -> ASCII
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٬٫", "01234567890123456789,.")

_NUMBER = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand|lakh|lac|crore|k)?\b",
    re.IGNORECASE)

# answers say "1.5 million" / "8 lakh" where the table has 1,557,200
_SCALE = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lac": 1e5,
          "million": 1e6, "crore": 1e7, "billion": 1e9}

_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")

# a month only counts next to a day or year ("you may need to apply")
_STEM = "|".join(m[:3] for m in _MONTHS)
_MONTH_NEAR_NUMBER = re.compile(
    rf"\b({_STEM})[a-z]*\.?,?\s+\d{{1,4}}|\b\d{{1,2}}(?:st|nd|rd|th)?\s+({_STEM})[a-z]*",
    re.IGNORECASE)

_URDU_MONTHS = {
    "جنوری": "january", "فروری": "february", "مارچ": "march", "اپریل": "april",
    "مئی": "may", "جون": "june", "جولائی": "july", "اگست": "august",
    "ستمبر": "september", "اکتوبر": "october", "نومبر": "november",
    "دسمبر": "december",
}

# 3-number sums only on small pools; combinations blow up on a big fee page
_TRIPLE_LIMIT = 40

# curly quotes, dashes, nbsp, zero-width spaces: the judge's quote won't have
# them, and a literal compare rejected 3 correct quotes in one answer
_TYPOGRAPHY = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "", "‌": "", "‍": "", "﻿": "",
})

# LUMS puts footnote links mid-sentence ("at least [1](https://...)8 subjects"),
# so compare quotes against rendered text, not markdown
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# drop footnote links, don't unwrap them: "at least [1](url)8 subjects" would
# become "at least 18 subjects"
_MD_FOOTNOTE = re.compile(r"\[\d{1,2}\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"\*{1,3}|_{2,3}|`")

def strip_markup(text: str) -> str:
    out = _MD_IMAGE.sub("", text)
    out = _MD_FOOTNOTE.sub("", out)
    out = _MD_LINK.sub(r"\1", out)
    return _MD_EMPHASIS.sub("", out)

def normalise(text: str) -> str:
    # the judge quotes table cells without the pipes
    out = strip_markup(text).replace("|", " ").translate(_DIGITS).translate(_TYPOGRAPHY)
    for ur, en in _URDU_MONTHS.items():
        out = out.replace(ur, en)
    return out

def numbers(text: str) -> set[float]:
    found = set()
    for m in _NUMBER.finditer(normalise(text)):
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        found.add(value * _SCALE[unit] if unit in _SCALE else value)
    return found

def months(text: str) -> set[str]:
    return {(a or b).lower() for a, b in _MONTH_NEAR_NUMBER.findall(normalise(text))}

def _close(a: float, b: float) -> bool:
    """Integers with float noise: 45,800 must not match 45,801."""
    return abs(a - b) < 0.5

def _round_sig(x: float, sig: int) -> float:
    if not x:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + sig - 1)

def derive(target: float, pool: set[float]) -> str | None:
    """Find the figure as a sum, difference, product, % or rounding of chunk
    figures (two semesters added, rate x credits). Returns how, or None."""
    vals = sorted(pool)

    for a, b in itertools.combinations(vals, 2):
        if _close(a + b, target):
            return f"{a:g} + {b:g}"
        if _close(abs(a - b), target):
            return f"{max(a, b):g} - {min(a, b):g}"
        if _close(a * b, target):
            return f"{a:g} x {b:g}"
        if b and _close(a * b / 100, target):
            return f"{b:g}% of {a:g}"
        if a and _close(b * a / 100, target):
            return f"{a:g}% of {b:g}"

    if len(vals) <= _TRIPLE_LIMIT:
        for combo in itertools.combinations(vals, 3):
            if _close(sum(combo), target):
                return " + ".join(f"{c:g}" for c in combo)

    # sig-fig rounding only: 1.56 million passes for 1,557,200, 1,555,000
    # doesn't (a % tolerance would let it through)
    for c in vals:
        for sig in range(1, 7):
            if _close(_round_sig(c, sig), target):
                return f"rounded from {c:g}"

    return None

def chunk_text(chunk: dict) -> str:
    """Heading path + body, as build_context gives the model. Headings carry
    figures too ("Fee Structure ... FY 2026-27")."""
    meta = chunk.get("meta", {})
    return f"{meta.get('headings', '')}\n{chunk.get('doc', '')}"

def cited_chunks(claim: dict, results: list[dict]) -> tuple[list[dict], list[int]]:
    """Resolve 1-based source numbers, reporting any that point nowhere."""
    good, bad = [], []
    for n in claim.get("sources") or []:
        if isinstance(n, int) and 1 <= n <= len(results):
            good.append(results[n - 1])
        else:
            bad.append(n)
    return good, bad

def check_structural(claim: dict, results: list[dict]) -> dict:
    """Strikes only when no citation resolves; everything else is a flag."""
    chunks, out_of_range = cited_chunks(claim, results)
    flags, notes = [], []

    if out_of_range:
        notes.append(f"source numbers out of range: {out_of_range}")
    if not chunks:
        return {"ok": False, "chunks": [], "flags": ["no_citation"],
                "notes": notes + ["no usable citation"]}

    # flag, don't strike, a NUST claim citing a LUMS page. Same entity type
    # only: universities list the tests they accept, so SAT + LUMS page is
    # normal (flagging it cost 3 correct SAT-deadline claims)
    named = set(detect_entities(claim.get("text", "")))
    cited = {c.get("meta", {}).get("entity_id") for c in chunks} - {None}
    cited_types = {entity_type(e) for e in cited}
    rival = {e for e in named - cited if entity_type(e) in cited_types}
    if rival:
        flags.append("entity_mismatch")
        notes.append(f"claim names {sorted(rival)}, cites {sorted(cited)}")

    return {"ok": True, "chunks": chunks, "flags": flags, "notes": notes}

def check_numeric(claim: dict, chunks: list[dict],
                  query: str = "", profile: dict | None = None) -> dict:
    """Each figure: verbatim in the evidence, derived from it, or absent. The
    student's question and profile count as evidence ("you said 85%")."""
    evidence = set()
    for c in chunks:
        evidence |= numbers(chunk_text(c))
    stated = numbers(query) | numbers(" ".join(str(v) for v in (profile or {}).values()))

    evidence_text = " ".join(normalise(chunk_text(c)).lower() for c in chunks)

    verbatim, derived, absent = [], {}, []
    for t in sorted(numbers(claim.get("text", ""))):
        if t in evidence or t in stated:
            verbatim.append(t)
            continue
        how = derive(t, evidence)
        if how:
            derived[t] = how
        else:
            absent.append(t)

    missing_months = [m for m in months(claim.get("text", "")) if m not in evidence_text]

    return {"verbatim": verbatim, "derived": derived, "absent": absent,
            "missing_months": missing_months, "present": sorted(evidence)}

def checks_block(numeric: dict, flags: list[str]) -> str:
    """V1/V2 findings as text for the judge, including the source figures
    closest to an absent one."""
    lines = []
    if numeric["absent"]:
        near = ", ".join(f"{n:,.0f}" for n in
                         sorted(numeric["present"],
                                key=lambda p: abs(p - numeric["absent"][0]))[:5])
        lines.append("figures not found in the source: "
                     + ", ".join(f"{n:,.0f}" for n in numeric["absent"]))
        lines.append(f"figures present in the source: {near}")
        lines.append("none reachable by sum, difference, product, percentage "
                     "or rounding")
    for t, how in numeric["derived"].items():
        lines.append(f"{t:,.0f} not literal but reachable as {how}")
    if numeric["missing_months"]:
        lines.append(f"months not found in the source: "
                     f"{', '.join(numeric['missing_months'])}")
    if "entity_mismatch" in flags:
        lines.append("the institution named in the claim is not the one this "
                     "source belongs to")
    return "\n".join(lines)

def verify_claim(claim: dict, results: list[dict],
                 query: str = "", profile: dict | None = None) -> dict:
    structural = check_structural(claim, results)
    out = {"text": claim.get("text", ""), "sources": claim.get("sources") or [],
           "notes": list(structural["notes"]), "checked": ["structural"],
           "flags": list(structural["flags"]), "numeric": None, "span": "",
           "overruled": False, "as_of": "", "live_verified": False}

    if not structural["ok"]:
        out["verdict"] = UNSUPPORTED       # no citation: struck, no judge
        return out

    numeric = check_numeric(claim, structural["chunks"], query, profile)
    out["numeric"] = numeric
    out["checked"].append("numeric")
    if numeric["absent"]:
        out["flags"].append("figure_absent")
    if numeric["missing_months"]:
        out["flags"].append("month_absent")

    # V1/V2 only flag; the judge decides
    out["verdict"] = NEEDS_REVIEW
    return out

def verify(claims: list[dict], results: list[dict], query: str = "",
           profile: dict | None = None, judge: bool = True) -> list[dict]:
    """V1 and V2, then V3 on whatever they didn't strike. judge=False stops
    after V1/V2 (tests and the lite arm)."""
    verified = [verify_claim(c, results, query, profile) for c in claims]
    reattribute(verified, results, query, profile)   # repair citations first,
    if judge:                                        # so the judge sees one
        adjudicate(verified, results)                # repaired version only
    return verified

JUDGE_PROMPT = """\
You check whether each claim is supported by the source text given with it. \
You are not answering the student's question and you are not judging whether \
the claim is true in the world -- only whether THIS source says it.

For each claim return one verdict:
- supported: the source states this, or states it in different words
- overreach: the source says something narrower, weaker or more conditional \
than the claim. "Required for international applicants" does not support \
"required for all applicants". This is the most important verdict to get right
- unsupported: the source does not state this at all

Also return `span`: the shortest run of text from the SOURCE, copied exactly \
character for character, that carries the support. Leave it empty for \
unsupported. Never paraphrase the span and never write a span of your own -- \
if you cannot copy one from the source, the claim is unsupported.

Sources are often tables or lists. A table cell or a list item is valid text \
to quote: copy the words of the cell(s) that carry the support, without the \
"|" separators. A supported claim from a table still needs a span.

A claim may combine figures the source states -- adding two semesters, \
multiplying a per-credit rate, taking a percentage, rounding for speech. That \
is supported as long as every figure it rests on is in the source and the \
arithmetic holds. The source does not have to state the total itself. Quote \
the figures it does state as your span.

A claim may be written in Urdu while its source is in English. Judge the \
meaning, not the language. The span is still copied from the source in the \
source's own language.

Some claims carry a CHECKS block. That is literal string and arithmetic \
matching done before you, and it can be wrong -- a figure may sit in the \
section heading, be written in words, or be part of a year range like \
"2026-27". The source text is the authority. Treat CHECKS as a pointer to \
look closely, not as a verdict. Where it reports a figure the source does not \
contain and you cannot find it either, that is a real error and the claim is \
unsupported.

If you support a claim that CHECKS flagged, you must fill `overrule_reason` \
with where in the source the check went wrong -- quote it. Leaving it empty \
means the flag stands and the claim is rejected.
"""

class _Judgement(BaseModel):
    claim_index: int
    verdict: Literal["supported", "overreach", "unsupported"]
    span: str
    # required when supporting a flagged claim
    overrule_reason: str = ""

class _Judgements(BaseModel):
    judgements: list[_Judgement]

def _span_present(span: str, chunks: list[dict]) -> bool:
    """Is the judge's quoted span really in the source? A made-up span is
    logged as a verifier failure."""
    needle = re.sub(r"\s+", " ", normalise(span)).strip().lower()
    if not needle:
        return False
    return any(needle in re.sub(r"\s+", " ", normalise(chunk_text(c))).lower()
               for c in chunks)

def adjudicate(verified: list[dict], results: list[dict]) -> None:
    """One judge call for everything V1/V2 left undecided. Mutates `verified`.
    If the call fails, claims stay needs_review (an outage is never a pass)."""
    pending = [(i, v) for i, v in enumerate(verified) if v["verdict"] == NEEDS_REVIEW]
    if not pending:
        return

    blocks = []
    for n, (_, v) in enumerate(pending):
        chunks, _bad = cited_chunks({"sources": v["sources"]}, results)
        sources = "\n".join(chunk_text(c) for c in chunks)
        block = f"CLAIM {n}: {v['text']}\nSOURCE:\n{sources}\n"
        checks = checks_block(v["numeric"] or {}, v["flags"]) if v["numeric"] else ""
        if checks:
            block += f"CHECKS:\n{checks}\n"
        blocks.append(block)

    try:
        parsed = generate(VERIFIER_MODEL, "\n---\n".join(blocks),
                          system=JUDGE_PROMPT, schema=_Judgements).parsed
        if not parsed:
            raise ValueError("empty judgement")
    except Exception as exc:
        for _, v in pending:
            v["notes"].append(f"entailment check failed ({exc})")
        return

    by_index = {j.claim_index: j for j in parsed.judgements}
    for n, (_, v) in enumerate(pending):
        j = by_index.get(n)
        v["checked"].append("entailment")
        if j is None:
            v["notes"].append("judge returned no verdict for this claim")
            continue

        chunks, _bad = cited_chunks({"sources": v["sources"]}, results)
        if j.verdict == "supported" and not _span_present(j.span, chunks):
            # quoted span not in the source -> struck. Blank span: the lite
            # judge often won't quote tables, and striking those cost 21 correct
            # claims in one run, so it passes if V2 found every figure in the
            # cited chunk (struck if there were no figures)
            numeric = v.get("numeric") or {}
            figures_verified = (not j.span.strip() and not v["flags"]
                                and bool(numbers(v["text"]))
                                and not numeric.get("absent"))
            if not figures_verified:
                v["verdict"] = UNSUPPORTED
                v["notes"].append("judge quoted a span not found in the source"
                                  if j.span.strip() else
                                  "judge gave no span and the claim has no "
                                  "figures to verify")
                continue
            v["verdict"] = SUPPORTED
            v["notes"].append("judge gave no span; figures verified against "
                              "the source by V2")
            continue

        v["verdict"] = {"supported": SUPPORTED, "overreach": OVERREACH,
                        "unsupported": UNSUPPORTED}[j.verdict]
        v["span"] = j.span
        if j.verdict == "overreach":
            v["notes"].append(f"source is narrower than the claim: {j.span[:120]!r}")

        # a flag stands unless the judge gives an overrule_reason
        if v["flags"] and v["verdict"] == SUPPORTED:
            if j.overrule_reason.strip():
                v["overruled"] = True
                v["notes"].append(f"judge overruled {', '.join(v['flags'])}: "
                                  f"{j.overrule_reason.strip()}")
            else:
                v["verdict"] = UNSUPPORTED
                v["notes"].append(f"judge supported it but gave no reason to "
                                  f"overrule {', '.join(v['flags'])}")

# loose on purpose: a full re-crawl is cheap, and the cycle rule below does
# most of the work
STALE_AFTER_DAYS = 60
FETCH_TIMEOUT = 8.0            # hard limit: a voice turn can't wait on a hung fetch
STALENESS_LOG = PROJECT_ROOT / "data" / "ingestion" / "staleness_log.txt"

def current_cycle(today: dt.date | None = None) -> int:
    """Admissions cycles turn over mid-year: a 2026 page is current until the
    next intake opens, not until December."""
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1

def chunk_staleness(chunk: dict, today: dt.date | None = None) -> list[str]:
    """Why this chunk's evidence may have expired; empty if it hasn't.

    cycle_bound: the page itself is out of date (HEC's Ehsaas pages still
    describe the 2021 intake); re-fetching won't help, only a hedge.
    time_sensitive: our copy may be behind the live page; a fetch can fix it.
    Uses freshness_class, cycle_year and scrape_date from ingestion.
    """
    today = today or dt.date.today()
    meta = chunk.get("meta", {})
    klass = meta.get("freshness_class", "")
    reasons = []

    if klass == "cycle_bound":
        try:
            year = int(meta.get("cycle_year"))
        except (TypeError, ValueError):
            year = None
        if year and year < current_cycle(today):
            reasons.append(f"describes the {year} cycle, current is "
                           f"{current_cycle(today)}")

    if klass == "time_sensitive":
        stamped = str(meta.get("scrape_date", ""))[:10]
        try:
            age = (today - dt.date.fromisoformat(stamped)).days
        except ValueError:
            return reasons
        if age > STALE_AFTER_DAYS:
            reasons.append(f"dates page last checked {age} days ago")
    return reasons

def _fetch_live(url: str) -> str | None:
    """Fetch one page as text. The crawler is imported lazily, only when a
    turn needs it."""
    try:
        import asyncio

        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

        async def go():
            async with AsyncWebCrawler(config=BrowserConfig(
                    headless=True, verbose=False, text_mode=True, light_mode=True)) as c:
                res = await c.arun(url, config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS, verbose=False,
                    page_timeout=int(FETCH_TIMEOUT * 1000)))
                md = getattr(res, "markdown", None)
                return getattr(md, "raw_markdown", None) or (md if isinstance(md, str) else None)

        return asyncio.run(asyncio.wait_for(go(), timeout=FETCH_TIMEOUT))
    except Exception as exc:
        print(f"[verify] live fetch failed for {url}: {exc}")
        return None

def _log_staleness(url: str, why: str) -> None:
    """Append what expired and when, for the evaluation's staleness counts."""
    try:
        STALENESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with STALENESS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"{dt.date.today().isoformat()}\t{url}\t{why}\n")
    except Exception:
        pass

def _matching_fresh_chunk(page_markdown: str, stale: dict) -> str | None:
    """Section of the fetched page matching the stale chunk's heading path
    (a whole page is far bigger than the chunk it replaces)."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "data" / "ingestion"))
        from crawler import chunk_markdown
    except Exception:
        return None

    leaf = str(stale.get("meta", {}).get("headings", "")).split(",")[-1].strip().lower()
    fresh = chunk_markdown(page_markdown)
    if not fresh:
        return None
    if leaf:
        for ch in fresh:
            if leaf in " ".join(ch.headings).lower():
                return ch.text
    # no heading match: fall back to whatever overlaps the old text most
    old = set(re.findall(r"[a-z0-9]{4,}", stale.get("doc", "").lower()))
    if not old:
        return None
    best = max(fresh, key=lambda c: len(old & set(re.findall(r"[a-z0-9]{4,}", c.text.lower()))))
    return best.text

def refresh_evidence(results: list[dict], live: bool = False,
                     today: dt.date | None = None) -> int:
    """Swap expired chunks for freshly fetched ones before generation, so
    nothing needs regenerating. Unrefreshable chunks stay marked and their
    claims get hedged. Not in core/retrieval.py: the baseline shares that
    module and must not get live fetching.
    """
    fetched: dict[str, str | None] = {}
    stale = 0

    for chunk in results:
        reasons = chunk_staleness(chunk, today)
        if not reasons:
            continue
        stale += 1
        chunk["stale_reasons"] = reasons
        url = chunk.get("meta", {}).get("source_url", "")

        # cycle_bound: out of date at the source, re-fetching won't help
        if not live or not url or any("cycle" in r for r in reasons):
            continue

        if url not in fetched:
            fetched[url] = _fetch_live(url)
        page = fetched[url]
        if page is None:
            continue

        replacement = _matching_fresh_chunk(page, chunk)
        if not replacement:
            continue
        if numbers(replacement) != numbers(chunk.get("doc", "")):
            _log_staleness(url, "figures changed since ingestion")
        chunk["doc"] = replacement
        chunk["meta"] = {**chunk.get("meta", {}),
                         "scrape_date": (today or dt.date.today()).isoformat()}
        chunk.pop("stale_reasons", None)
        chunk["refreshed"] = True

    return stale

def hedge_stale_claims(verified: list[dict], results: list[dict]) -> int:
    """Mark claims resting on expired, unrefreshed evidence, so the answer
    gives an "as of" date."""
    hedged = 0
    for v in verified:
        if v["verdict"] != SUPPORTED:
            continue
        chunks, _ = cited_chunks({"sources": v["sources"]}, results)
        reasons = [r for c in chunks for r in c.get("stale_reasons", [])]
        if not reasons:
            continue
        v["verdict"] = STALE
        v["flags"].append("stale_evidence")
        v["notes"].extend(reasons)
        v["as_of"] = str(chunks[0].get("meta", {}).get("scrape_date", ""))[:10]
        hedged += 1
    return hedged

def reattribute(verified: list[dict], results: list[dict],
                query: str = "", profile: dict | None = None) -> int:
    """Before striking: if a missing figure is in another retrieved chunk,
    repoint the citation instead. No model call."""
    repaired = 0
    for v in verified:
        if not v["flags"]:
            continue
        for i, chunk in enumerate(results, start=1):
            if i in v["sources"]:
                continue
            trial = verify_claim({"text": v["text"], "sources": [i]},
                                 results, query, profile)
            if trial["flags"]:                 # this chunk is no better
                continue
            v["sources"] = [i]
            v["verdict"] = NEEDS_REVIEW        # back to the judge, not struck
            v["numeric"] = trial["numeric"]
            v["flags"] = []
            v["notes"].append(f"re-attributed to source {i} "
                              f"({chunk.get('meta', {}).get('headings', '')})")
            repaired += 1
            break
    return repaired

def partition(verified: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into kept and struck. Stale claims are kept, with as_of."""
    kept = [v for v in verified if v["verdict"] in (SUPPORTED, STALE)]
    struck = [v for v in verified if v["verdict"] in CONDEMNED]
    return kept, struck

def as_of_date(verified: list[dict]) -> str:
    """Oldest check date among claims whose evidence had expired."""
    dates = [v["as_of"] for v in verified
             if v["verdict"] == STALE and v.get("as_of")]
    return min(dates) if dates else ""

def append_hedge(answer: str, as_of: str, language: str = "en") -> str:
    """Appended after assembly, not asked for in the prompt: the merge rewrite
    drops it."""
    if not as_of or not answer.strip():
        return answer
    # counsellor wording ("I haven't seen this year's yet"), no scrape date
    if language == "ur":
        note = (" یہ پچھلے سیشن کی معلومات ہیں اور اس سال کی اپ ڈیٹ ابھی میرے "
                "سامنے نہیں آئی، اس لیے ایک بار آفیشل ویب سائٹ سے ضرور دیکھ لیں۔")
    else:
        note = (" These are from the last cycle though, and I haven't seen this "
                "year's updated yet — worth confirming on the official page.")
    return answer.rstrip() + note

REWRITE_PROMPT = """\
Some of what you said could not be supported by the sources and has been \
removed. Write the answer again from the facts that survived.

The constraint is on what you ASSERT, not on how you speak:
- These are the only facts about institutions, fees, dates, requirements or \
schemes you may state. Add no figure, date or condition that is not here.
- Everything else about you is unchanged. Explain a term the student would not \
know, keep the warmth, and if you were going to ask them something, still ask \
it. A bare list of facts is not an answer from you.
- If what survived no longer answers what they asked, say plainly which part \
you cannot answer.
- Same language as the original. This is read aloud, so no citation markers and \
no mention of verification or of anything having been removed.
"""

def rewrite_prompt(question: str, kept: list[dict], struck: list[dict]) -> str:
    facts = "\n".join(f"- {v['text']}" for v in kept) or "(nothing survived)"
    removed = "\n".join(f"- {v['text']}  [{v['verdict']}]" for v in struck)
    return (f"The student asked: {question}\n\n"
            f"VERIFIED FACTS:\n{facts}\n\n"
            f"REMOVED, do not restate:\n{removed}\n")

def review(claims: list[dict], results: list[dict], question: str,
           profile: dict | None = None, judge: bool = True) -> tuple[list, list, list]:
    """The whole pass an agent runs: check, repair citations, re-judge, split.

    Returns (verified, kept, struck).
    """
    verified = verify(claims, results, question, profile, judge=judge)
    if not judge:
        # lite: no judge, so any flag strikes
        for v in verified:
            if v["verdict"] == NEEDS_REVIEW:
                v["verdict"] = UNSUPPORTED if v["flags"] else SUPPORTED
    kept, struck = partition(verified)
    return verified, kept, struck

def rewrite_answer(question: str, kept: list[dict], struck: list[dict],
                   persona: str, language: str = "en") -> str | None:
    """Rebuild the spoken answer from the surviving claims. ANSWER_MODEL, not
    the verifier's, so student-facing prose comes from the model under test."""
    try:
        return generate(ANSWER_MODEL, rewrite_prompt(question, kept, struck),
                        system=(persona + REWRITE_PROMPT + today_clause()
                                + language_clause(language, structured=False))
                        ).text.strip() or None
    except Exception as exc:
        print(f"[verify] rewrite failed ({exc})")
        return None

# Source markers must never survive into text that gets read aloud.
_MARKERS = re.compile(r"\[\s*\d+\s*\]|\(\s*source\s*\d+\s*\)|\bsource\s+\d+\b",
                      re.IGNORECASE)

# Arabic-script range, for checking the answer came back in the language asked.
_URDU_CHARS = re.compile(r"[؀-ۿ]")

POLICY_PROMPT = """\
You are checking one thing in an admissions answer: whether it predicts or \
assesses an individual student's outcome.

Not allowed: saying or implying the student will be admitted, has a good or \
poor chance, is a strong or weak candidate, or should or should not bother \
applying.

Allowed: stating published requirements, and saying plainly whether a figure \
the student gave meets a published minimum. "The minimum is 60% and you said \
85%, so you meet it" is a fact about a document. "You will probably get in" \
is a prediction.

Return the offending sentences, or an empty list.
"""

class _Policy(BaseModel):
    offending_sentences: list[str]

def unclaimed_figures(answer: str, kept: list[dict],
                      query: str = "") -> list[float]:
    """Figures in the spoken answer that no kept claim accounts for.

    The persona lets the model explain terms, and unsourced facts slip in that
    way (explaining IBCC, then "it usually takes four to six weeks"). V1-V3
    never see them since they aren't claims. Figures only: prose would need a
    judgement call.
    """
    accounted = set()
    for k in kept:
        accounted |= numbers(k["text"])
    accounted |= numbers(query)          # a figure the student themselves gave

    # skip years ("FY 2026-27" parses as 2026) and 1-9 (list markers, "these
    # three tests"). Real misses (durations, fees, deadlines, percentages) are
    # bigger; no institution requires 5%.
    this_year = dt.date.today().year
    return sorted(n for n in numbers(answer)
                  if n not in accounted
                  and n >= 10
                  and not (1990 <= n <= this_year + 15 and float(n).is_integer()))

def policy_gate(answer: str, struck: list[dict], language: str = "en",
                judge: bool = True, kept: list[dict] | None = None,
                query: str = "") -> tuple[str, list[str]]:
    """Checks on the merged prose after assembly. Strips source markers;
    everything else is only reported in violations. Main one: struck figures
    reappearing after the model rewrite."""
    violations: list[str] = []

    # skip years, as in unclaimed_figures ("Fall 2026" is just the current cycle)
    this_year = dt.date.today().year
    struck_figures = {n for s in struck for n in numbers(s["text"])
                      if not (1990 <= n <= this_year + 15 and float(n).is_integer())}
    reintroduced = struck_figures & numbers(answer)
    if reintroduced:
        violations.append("reintroduced struck figures: "
                          + ", ".join(f"{n:,.0f}" for n in sorted(reintroduced)))

    if kept is not None:
        loose = unclaimed_figures(answer, kept, query)
        if loose:
            violations.append("figures in the answer that no claim accounts for: "
                              + ", ".join(f"{n:,.0f}" for n in loose))

    if _MARKERS.search(answer):
        answer = _MARKERS.sub("", answer)
        answer = re.sub(r"\s{2,}", " ", answer).strip()
        violations.append("stripped source markers from spoken text")

    if answer.strip():
        urdu_ratio = len(_URDU_CHARS.findall(answer)) / max(len(answer), 1)
        if language == "ur" and urdu_ratio < 0.2:
            violations.append("Urdu was asked for but the answer is not in Urdu")
        elif language == "en" and urdu_ratio > 0.2:
            violations.append("English was asked for but the answer is not in English")

    if judge and answer.strip():
        try:
            parsed = generate(VERIFIER_MODEL, answer,
                              system=POLICY_PROMPT, schema=_Policy).parsed
            if parsed and parsed.offending_sentences:
                violations.append("predicts or assesses the student's outcome: "
                                  + " / ".join(parsed.offending_sentences[:3]))
        except Exception as exc:
            print(f"[verify] policy check failed ({exc})")

    return answer, violations

_TOPIC_STOPWORDS = {"the", "and", "for", "with", "that", "this", "are", "is",
                    "you", "your", "from", "per", "has", "have", "must", "can",
                    "will", "not", "all", "any", "its", "their", "which"}

def _topic(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", normalise(text).lower())
    return {w for w in words if w not in _TOPIC_STOPWORDS}

MAX_CONFLICT_CANDIDATES = 8

def find_conflicts(claims: list[dict]) -> list[dict]:
    """Claim pairs where two authorities may disagree (HEC's national rule vs
    a university's version of it; no single agent sees both). Pairs without
    figures still count if their topics overlap more ("apply through your
    university" vs "apply on the HEC portal"). The judge decides.
    """
    candidates = []
    for a, b in itertools.combinations(claims, 2):
        ea, eb = a.get("entity", ""), b.get("entity", "")
        if not ea or not eb or ea == eb:
            continue                       # same authority cannot conflict
        shared = _topic(a["text"]) & _topic(b["text"])
        na, nb = numbers(a["text"]), numbers(b["text"])

        if na and nb:
            if na & nb or len(shared) < 2:
                continue                   # figures agree, or different topics
        elif len(shared) < 3:
            continue                       # prose needs a stronger signal

        candidates.append({"a": a, "b": b, "shared": sorted(shared)})
    candidates.sort(key=lambda c: -len(c["shared"]))
    return candidates[:MAX_CONFLICT_CANDIDATES]

CONFLICT_PROMPT = """\
Two statements about Pakistani university admissions, each from a different \
authority. Decide whether they genuinely disagree about the same thing, in a \
way a student would have to resolve before acting.

Not a conflict: they cover different things, different programmes, or \
different applicant groups; one is a general rule and the other is a specific \
case consistent with it; one simply says more than the other.

A conflict: both answer the same question and give incompatible answers -- a \
different figure, a different deadline, a different place to apply, or \
opposite eligibility.

For each pair, say whether it is real, and if so state the disagreement in one \
short sentence a student would understand.
"""

class _ConflictVerdict(BaseModel):
    pair_index: int
    real: bool
    disagreement: str

class _ConflictVerdicts(BaseModel):
    verdicts: list[_ConflictVerdict]

def adjudicate_conflicts(candidates: list[dict]) -> list[dict]:
    """One judge call to confirm candidate pairs, only when there are any."""
    if not candidates:
        return []

    blocks = []
    for i, c in enumerate(candidates):
        blocks.append(f"PAIR {i}\n"
                      f"{c['a'].get('entity','?')}: {c['a']['text']}\n"
                      f"{c['b'].get('entity','?')}: {c['b']['text']}")
    try:
        parsed = generate(VERIFIER_MODEL, "\n\n".join(blocks),
                          system=CONFLICT_PROMPT, schema=_ConflictVerdicts).parsed
        if not parsed:
            return []
    except Exception as exc:
        print(f"[verify] conflict check failed ({exc})")
        return []

    confirmed = []
    for v in parsed.verdicts:
        if not v.real or not (0 <= v.pair_index < len(candidates)):
            continue
        c = candidates[v.pair_index]
        c["disagreement"] = v.disagreement
        for side in ("a", "b"):
            c[side].setdefault("flags", []).append("conflicted")
        confirmed.append(c)
    return confirmed

_display: dict[str, str] | None = None
_types: dict[str, str] | None = None

def _load_entities() -> None:
    """Names and types from the ingestion config, so this module doesn't need
    ChromaDB."""
    global _display, _types
    if _display is not None:
        return
    try:
        import json
        path = PROJECT_ROOT / "data" / "ingestion" / "entities.json"
        data = json.loads(path.read_text(encoding="utf-8")).get("entities", {})
        _display = {k: v.get("display_name", k) for k, v in data.items()}
        _types = {k: v.get("type", "") for k, v in data.items()}
    except Exception:
        _display, _types = {}, {}

def display_name(entity_id: str) -> str:
    """Entity ids end up in spoken text, and TTS can't say "hec_nbs"."""
    _load_entities()
    return _display.get(entity_id, entity_id)

def entity_type(entity_id: str) -> str:
    _load_entities()
    return _types.get(entity_id, "")

def collect_readings(partials: list[dict], answer: str) -> list[dict]:
    """Spoken forms for the figures in the final answer (the Urdu TTS can't
    read bare digits). Readings whose figure was rewritten away by repair or
    assembly are dropped."""
    present = numbers(answer)
    out, seen = [], set()
    for p in partials:
        for r in p.get("readings") or []:
            written, spoken = str(r.get("written", "")), str(r.get("spoken", ""))
            if not written or not spoken or written in seen:
                continue
            if written not in answer and not (numbers(written) & present):
                continue
            seen.add(written)
            out.append({"written": written, "spoken": spoken})
    # longest first, so "2026" is not eaten by a reading for "20"
    return sorted(out, key=lambda r: -len(r["written"]))

def speakable(text: str, readings: list[dict] | None) -> str:
    """The answer as it should be heard rather than read."""
    for r in readings or []:
        text = text.replace(r["written"], r["spoken"])
    return text

MAX_QUESTIONS = 2

def collect_questions(partials: list[dict]) -> list[dict]:
    """Agent questions, one per pivot, at most two (the answer is spoken)."""
    out, seen = [], set()
    for p in partials:
        q = (p.get("question") or "").strip()
        pivot = (p.get("pivot") or "").strip().lower()
        if not q or pivot in seen:
            continue
        seen.add(pivot)
        out.append({"question": q, "pivot": pivot or q[:24]})
    return out[:MAX_QUESTIONS]

def question_note(questions: list[dict]) -> str:
    """Appended after assembly; the merge rewrite drops questions."""
    if not questions:
        return ""
    if len(questions) == 1:
        return " " + questions[0]["question"]
    return " " + " ".join(q["question"] for q in questions)

def conflict_note(conflicts: list[dict], language: str = "en") -> str:
    """Appended after assembly; the merge rewrite smooths disagreements away."""
    if not conflicts:
        return ""
    c = conflicts[0]
    a = display_name(c["a"].get("entity", "")) or "one source"
    b = display_name(c["b"].get("entity", "")) or "another"

    if language == "ur":
        return (f" ایک بات کا خیال رکھیں: {a} اور {b} اس بارے میں مختلف "
                f"بات کہتے ہیں، اس لیے جو ادارہ آپ کا کیس دیکھے گا اُس سے "
                f"تصدیق ضرور کر لیں۔")

    detail = (c.get("disagreement") or "").strip().rstrip(".")
    if detail:
        detail = detail[0].lower() + detail[1:]
        return (f" One thing to watch: {detail}. Confirm with whichever body "
                f"will decide your case.")
    return (f" One thing to watch: {a} and {b} do not say the same thing here. "
            f"Confirm with whichever body will decide your case.")

def calibration(verified: list[dict]) -> dict:
    """How the deterministic layer and the judge agreed.

    confirmed  V2 flagged it and the judge condemned  -> real error
    overruled  V2 flagged it and the judge supported  -> V2 false positive
    judge_only the judge condemned with no flag       -> V2 blind spot
    """
    out = {"confirmed": 0, "overruled": 0, "judge_only": 0}
    for v in verified:
        flagged = bool(v.get("flags"))
        condemned = v["verdict"] in CONDEMNED
        if flagged and condemned:
            out["confirmed"] += 1
        elif v.get("overruled"):
            out["overruled"] += 1
        elif condemned:
            out["judge_only"] += 1
    return out

def summarise(verified: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for v in verified:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    return counts

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # Real LUMS fee table; the last three claims are the kind V2 should catch.
    RESULTS = [
        {"doc": "| Particulars | Fall 2026 | Spring 2027 | Total |\n"
                "| Credit Hours | 18 | 16 | 34 |\n"
                "| Admission Fee | 176,600 | - | 176,600 |\n"
                "| Tuition Fee | 824,400 | 732,800 | 1,557,200 |\n"
                "| Total | 1,258,700 | 923,500 | 2,182,200 |",
         "meta": {"entity_id": "lums", "headings": "BS Computer Science, Fee Structure"}},
        {"doc": "The per credit hour rate for FY 2026-27 is PKR 45,800/-. "
                "Applications close on January 27, 2026.",
         "meta": {"entity_id": "lums", "headings": "BS Computer Science, Important Points"}},
        {"doc": "NUST charges PKR 89,000 per semester for engineering disciplines.",
         "meta": {"entity_id": "nust", "headings": "Fee Structure"}},
        {"doc": "Applicants must have completed A-Levels or an equivalent "
                "qualification. The SAT is required for international applicants.",
         "meta": {"entity_id": "lums", "headings": "BS Computer Science, Eligibility"}},
    ]
    CLAIMS = [
        {"text": "At LUMS, first-year tuition is PKR 1,557,200.", "sources": [1]},
        {"text": "At LUMS, the per credit hour rate is PKR 45,800.", "sources": [2]},
        {"text": "LUMS tuition comes to roughly 1.56 million for the first year.", "sources": [1]},
        {"text": "18 credit hours at 45,800 works out to 824,400.", "sources": [1, 2]},
        {"text": "ایل یو ایم ایس میں پہلے سال کی ٹیوشن ۱٬۵۵۷٬۲۰۰ روپے ہے۔", "sources": [1]},
        {"text": "At LUMS, the application deadline is January 27, 2026.", "sources": [2]},
        {"text": "At LUMS, first-year tuition is PKR 1,555,000.", "sources": [1]},
        {"text": "At NUST, tuition is PKR 89,000 per semester.", "sources": [1]},
        {"text": "The application fee is PKR 12,000.", "sources": []},
        {"text": "At LUMS, the SAT is accepted for admission.", "sources": [2]},
        {"text": "At LUMS, the SAT is required for all applicants.", "sources": [4]},
        {"text": "At LUMS, applicants need A-Levels or an equivalent qualification.", "sources": [4]},
    ]

    QUESTION = "what does BS CS cost at LUMS?"
    verified, kept, struck = review(CLAIMS, RESULTS, QUESTION,
                                    judge="--judge" in sys.argv)
    width = max(len(v["verdict"]) for v in verified)
    for v in verified:
        flags = f"  flags={v['flags']}" if v["flags"] else ""
        print(f"\n[{v['verdict']:<{width}}] {v['text']}{flags}")
        for n in v["notes"]:
            print(f"{'':>{width + 3}} {n}")
    print(f"\n{summarise(verified)}")
    print(f"kept {len(kept)} | struck {len(struck)} | {calibration(verified)}")
