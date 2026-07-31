"""
Prototype evaluation harness.

Runs a graded set of English + Urdu/code-switched queries through the SAME
retrieval+generation pipeline the app uses, and records, per query:
  - detected entities (entity-routing check)
  - which institutions' chunks were retrieved + top rank per institution
  - whether the expected ground-truth chunk/category was retrieved
  - the answer text (for manual faithfulness / abstention judgement)
  - retrieve and generate timing

Run from project root (main venv, Ollama running):
    python eval_harness.py                 # all queries
    python eval_harness.py --lang en       # english only
    python eval_harness.py --lang ur       # urdu only
    python eval_harness.py --out eval.md   # also write a markdown results sheet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Windows cuDNN DLL fix (same as server.py) so faster-whisper deps load.
if sys.platform == "win32":
    try:
        import os, torch
        _lib = Path(torch.__file__).resolve().parent / "lib"
        if _lib.is_dir():
            os.add_dll_directory(str(_lib))
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from advisor.core.retrieval import retrieve
from advisor.core.entities import detect_entities
from advisor.baseline.pipeline import answer_query
from advisor.core.llm import get_client


# ---------------------------------------------------------------------------
# Query set. Each item:
#   id, lang, difficulty, query, expect (behaviour), ground_truth (what a
#   correct answer must contain OR why it should abstain), failure_probed.
# Grounded in the actual ingested pages:
#   LUMS program page = rich (eligibility, fees, dates, tests, scholarships)
#   Habib program page = thin (program/PLO/FAQ + ONE eligibility FAQ; NO fees/dates)
# ---------------------------------------------------------------------------
QUERIES = [
    # ---------- ENGLISH ----------
    # simple, single-fact, answerable
    dict(id="E1", lang="en", diff="simple",
         query="What is the application deadline for LUMS?",
         expect="answer", gt="Jan 27, 2026 (online application)",
         probes="basic retrieve->answer"),
    dict(id="E2", lang="en", diff="simple",
         query="What tests does LUMS accept for CS admission?",
         expect="answer", gt="SAT, ACT, or LCAT",
         probes="basic factual"),
    dict(id="E3", lang="en", diff="simple",
         query="What is the tuition fee for the first year at LUMS?",
         expect="answer", gt="Tuition 1,557,200; total 2,182,200 PKR (fee table)",
         probes="table retrieval"),
    # multi-part / spans chunks
    dict(id="E4", lang="en", diff="multi",
         query="What are the eligibility requirements for BS Computer Science at LUMS?",
         expect="answer", gt="Matric 70% + FSc/ICS 70%; O-Level avg B + A-Level 2B1C; HSD 70%/B; IBDP 28/45",
         probes="eligibility split across chunks [10]/[11]"),
    dict(id="E5", lang="en", diff="simple",
         query="What grades do I need at A-Level for Computer Science at Habib?",
         expect="answer", gt="A-Levels: 3 principal subjects, 2 science + Maths, average grade C",
         probes="single Habib eligibility chunk"),
    # comparison (per-entity retrieval; headline case)
    dict(id="E6", lang="en", diff="comparison",
         query="Compare the eligibility criteria for CS at LUMS and Habib.",
         expect="answer-both", gt="LUMS (2B1C A-level, 70% matric/FSc) vs Habib (avg C A-level w/ maths+2 sci, 70% SSC/HSSC)",
         probes="per-entity retrieval; must cover BOTH"),
    # FAQ-drift-prone region
    dict(id="E7", lang="en", diff="faq",
         query="Does LUMS accept the ACT?",
         expect="answer", gt="Yes, LUMS accepts ACT results",
         probes="answer lives in drift-prone FAQ chunk [3]"),
    # answerable, scholarships
    dict(id="E8", lang="en", diff="simple",
         query="Does LUMS offer merit scholarships?",
         expect="answer", gt="Yes; 100 merit scholarships, partial-to-full tuition for 1 year",
         probes="scholarship retrieval"),
    # unanswerable -> correct abstention (Habib fees NOT on ingested page)
    dict(id="E9", lang="en", diff="abstain",
         query="What are the tuition fees for Computer Science at Habib University?",
         expect="abstain", gt="Habib fees are NOT on the ingested program page -> should abstain",
         probes="faithful abstention (coverage gap)"),
    # out-of-scope institution -> correct abstention
    dict(id="E10", lang="en", diff="abstain",
         query="What is the eligibility criteria for Computer Science at NUST?",
         expect="abstain", gt="NUST not ingested -> should abstain",
         probes="faithful abstention (out of scope)"),
    # prediction -> should decline / redirect
    dict(id="E11", lang="en", diff="boundary",
         query="Will I get into LUMS if I have 75% marks?",
         expect="decline-predict", gt="should NOT predict admission; redirect to requirements",
         probes="counsellor boundary (no outcome prediction)"),
    # natural / messier phrasing
    dict(id="E12", lang="en", diff="simple",
         query="do i need an ibcc equivalence certificate for lums?",
         expect="answer", gt="Yes if your qualification doesn't lead to Matric/FA/ICS/FSc",
         probes="lowercase/natural phrasing"),

    # ---------- URDU / CODE-SWITCHED ----------
    dict(id="U1", lang="ur", diff="simple",
         query="ایل یو ایم ایس میں اپلائی کرنے کی ڈیڈلائن کیا ہے؟",
         expect="answer", gt="27 جنوری 2026",
         probes="Urdu retrieve->answer; LUMS spelled-out initialism (ایل یو ایم ایس) alias"),
    dict(id="U2", lang="ur", diff="simple",
         query="لمز سی ایس کے لیے کون سے ٹیسٹ قبول کرتا ہے؟",
         expect="answer", gt="SAT, ACT, LCAT",
         probes="Urdu entity detection; LUMS single-word spelling (لمز) alias"),
    dict(id="U3", lang="ur", diff="simple",
         query="لمز میں پہلے سال کی ٹیوشن فیس کتنی ہے؟",
         expect="answer", gt="ٹیوشن 1,557,200؛ کل 2,182,200 روپے",
         probes="Urdu -> table retrieval (cross-lingual)"),
    dict(id="U4", lang="ur", diff="multi",
         query="لمز میں بی ایس کمپیوٹر سائنس کے لیے کیا اہلیت درکار ہے؟",
         expect="answer", gt="میٹرک 70% + ایف ایس سی 70%؛ او لیول B؛ اے لیول 2B1C؛ IBDP 28/45",
         probes="Urdu eligibility, multi-chunk"),
    dict(id="U5", lang="ur", diff="simple",
         query="حبیب میں کمپیوٹر سائنس کے لیے اے لیول میں کیا گریڈ چاہئیں؟",
         expect="answer", gt="اے لیول: 3 مضامین، 2 سائنس + میتھس، اوسط C",
         probes="Urdu entity detection (حبیب)"),
    dict(id="U6", lang="ur", diff="comparison",
         query="لمز اور حبیب میں سی ایس کی الیجیبلیٹی کا موازنہ کریں۔",
         expect="answer-both", gt="LUMS vs Habib eligibility (both)",
         probes="Urdu per-entity retrieval (the bug-fix case); must cover BOTH"),
    dict(id="U7", lang="ur", diff="simple",
         query="کیا حبیب کے کمپیوٹر سائنس پروگرام کے لیے میتھس لازمی ہے؟",
         expect="answer", gt="ہاں، HSSC میں میتھس لازمی",
         probes="Urdu yes/no factual"),
    dict(id="U8", lang="ur", diff="simple",
         query="کیا لمز میرٹ اسکالرشپ دیتا ہے؟",
         expect="answer", gt="ہاں، 100 میرٹ اسکالرشپس",
         probes="Urdu scholarship retrieval"),
    # unanswerable -> abstain (in Urdu)
    dict(id="U9", lang="ur", diff="abstain",
         query="حبیب یونیورسٹی میں کمپیوٹر سائنس کی فیس کتنی ہے؟",
         expect="abstain", gt="Habib fees not ingested -> abstain",
         probes="Urdu faithful abstention"),
    dict(id="U10", lang="ur", diff="abstain",
         query="نسٹ میں کمپیوٹر سائنس کے لیے کیا شرائط ہیں؟",
         expect="abstain", gt="NUST not ingested -> abstain",
         probes="Urdu out-of-scope abstention"),
    dict(id="U11", lang="ur", diff="boundary",
         query="کیا میں 75 پرسنٹ کے ساتھ لمز میں داخلہ لے سکوں گا؟",
         expect="decline-predict", gt="should not predict; redirect to requirements",
         probes="Urdu counsellor boundary"),
    dict(id="U12", lang="ur", diff="simple",
         query="کیا لمز کے لیے آئی بی سی سی ایکوئیولینس سرٹیفکیٹ ضروری ہے؟",
         expect="answer", gt="ہاں، اگر قابلیت Matric/FSc کے مساوی نہ ہو",
         probes="Urdu IBCC equivalence"),
]


def run_one(item, client, k, mode):
    q = item["query"]
    detected = detect_entities(q)
    t0 = time.time()
    results, parsed = answer_query(q, k=k, mode=mode, client=client,
                                   language=item["lang"])
    total = time.time() - t0

    # which institutions actually showed up in retrieval, best rank each
    ranks: dict[str, int] = {}
    for i, r in enumerate(results or [], start=1):
        ent = r.get("meta", {}).get("entity_id", "?")
        ranks.setdefault(ent, i)

    answer = parsed.answer if parsed else "(no answer / no results)"

    # per-claim grounding: each claim -> the source(s) it cites, resolved to
    # entity + heading + url. This is the core faithfulness evidence
    claims = []
    if parsed and parsed.claims:
        for c in parsed.claims:
            srcs = []
            for n in c.sources:
                if results and 1 <= n <= len(results):
                    m = results[n - 1].get("meta", {})
                    srcs.append({"n": n, "entity": m.get("entity_id", "?"),
                                 "heading": m.get("headings", ""),
                                 "url": m.get("source_url", "")})
                else:
                    srcs.append({"n": n, "entity": "OUT_OF_RANGE",
                                 "heading": "", "url": ""})
            claims.append({"text": c.text, "sources": srcs})

    # sources actually cited as used, grouped by entity 
    sources_used = []
    if parsed and getattr(parsed, "sources_used", None):
        for n in parsed.sources_used:
            if results and 1 <= n <= len(results):
                m = results[n - 1].get("meta", {})
                sources_used.append({"n": n, "entity": m.get("entity_id", "?"),
                                     "heading": m.get("headings", ""),
                                     "url": m.get("source_url", "")})

    return {
        "detected": detected,
        "retrieved_entities": ranks,
        "answer": answer,
        "claims": claims,
        "sources_used": sources_used,
        "n_results": len(results or []),
        "time_total": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ur", "all"], default="all")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--mode", default="dense")
    ap.add_argument("--out", default=None, help="optional markdown output path")
    args = ap.parse_args()

    items = [q for q in QUERIES if args.lang == "all" or q["lang"] == args.lang]
    client = get_client()

    rows = []
    for item in items:
        print("\n" + "=" * 90)
        print(f"[{item['id']}] ({item['lang']}/{item['diff']}) expect={item['expect']}")
        print(f"  Q: {item['query']}")
        print(f"  ground truth: {item['gt']}")
        print(f"  probes: {item['probes']}")
        out = run_one(item, client, args.k, args.mode)
        print(f"  detected entities : {out['detected'] or '(none)'}")
        print(f"  retrieved (ent:rank): {out['retrieved_entities']}")
        print(f"  time: {out['time_total']:.1f}s   results: {out['n_results']}")
        print(f"  ANSWER:\n    {out['answer']}")
        if out["claims"]:
            print("  CLAIM -> SOURCES:")
            for c in out["claims"]:
                tags = ", ".join(f"[{s['n']}]{s['entity']}/{s['heading']}"
                                 for s in c["sources"]) or "(none)"
                print(f"    - {c['text']}\n        <- {tags}")
        else:
            print("  CLAIM -> SOURCES: (no claims / abstained)")
        rows.append((item, out))

    if args.out:
        write_markdown(rows, Path(args.out))
        print(f"\n[eval] wrote results sheet -> {args.out}")

    print("\n" + "=" * 90)
    print("Now judge each answer against its ground truth and fill verdicts:")
    print("  answerable -> FAITHFUL & COMPLETE? (or partial/hallucinated)")
    print("  abstain    -> did it correctly say 'not in sources'?")
    print("  decline    -> did it avoid predicting the outcome?")


def write_markdown(rows, path: Path):
    lines = ["# Prototype evaluation results\n",
             "| ID | Lang | Diff | Expect | Detected | Retrieved (ent:rank) | Time(s) | Verdict (fill in) |",
             "|----|------|------|--------|----------|----------------------|---------|-------------------|"]
    for item, out in rows:
        det = "+".join(out["detected"]) or "-"
        ret = ", ".join(f"{e}:{r}" for e, r in out["retrieved_entities"].items())
        lines.append(f"| {item['id']} | {item['lang']} | {item['diff']} | "
                     f"{item['expect']} | {det} | {ret} | {out['time_total']:.1f} |  |")
    lines.append("\n## Answers + grounding (for faithfulness judgement)\n")
    for item, out in rows:
        lines.append(f"### {item['id']} — {item['query']}")
        lines.append(f"*Expect:* {item['expect']} — *Ground truth:* {item['gt']}\n")
        lines.append(f"*Probes:* {item['probes']}\n")
        lines.append(f"**Answer:**\n```\n{out['answer']}\n```\n")
        if out["claims"]:
            lines.append("**Claim → sources:**\n")
            for c in out["claims"]:
                tags = ", ".join(f"[{s['n']}] {s['entity']} / {s['heading']}"
                                 for s in c["sources"]) or "(none)"
                lines.append(f"- {c['text']}  \n  ↳ {tags}")
            lines.append("")
        else:
            lines.append("**Claim → sources:** (no claims / abstained)\n")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
