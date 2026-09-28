"""Scoring for the evaluation sweep.

Reads the run logs in runs/ and the labels in runs/labels.jsonl and writes
eval_results.md. Subcommands: check-gold, sample, show, grep, add, aggregate,
stats (python eval_score.py -h).

Label format, one JSON object per line in runs/labels.jsonl:

  {"kind":"answer","cell":..,"qid":..,"facts":[1,0,-1],"bad":[{"t":..,"l":..}],
   "behaviour":"answered|refused|no_figure","predicts":false,
   "lang":"ur|roman|en|mixed|other|json|na","stale":true|false|null,
   "cause":"ingestion_gap|retrieval_miss|verification_error|staleness|generation_error|null",
   "authority":true|false|null,"note":".."}
  {"kind":"claim","cell":..,"qid":..,"idx":n,"stage":"kept|struck","supported":bool,"note":".."}

facts: 1 present, 0 absent, -1 contradicted, one per gold_facts item.
bad: only the statements that are not supported, with l one of contradicted,
unverifiable or misattributed.
A struck claim with supported=true is an over-strike.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

QUERIES = ROOT / "data" / "eval" / "queries.json"
GOLD = ROOT / "data" / "eval" / "gold_facts.json"
LABELS = ROOT / "runs" / "labels.jsonl"
RESULTS = ROOT / "eval_results.md"

CELLS = {
    "gemini37_direct": ("gemini", "direct"),
    "gemini37_baseline": ("gemini", "baseline"),
    "gemini37_multi_off": ("gemini", "multi_off"),
    "gemini37_multi_full": ("gemini", "multi_full"),
    "gemma31b_direct": ("gemma", "direct"),
    "gemma31b_baseline": ("gemma", "baseline"),
    "gemma31b_multi_off": ("gemma", "multi_off"),
    "gemma31b_multi_full": ("gemma", "multi_full"),
    "qwen9b_direct": ("qwen", "direct"),
    "qwen9b_baseline": ("qwen", "baseline"),
    "qwen9b_multi_off": ("qwen", "multi_off"),
    "qwen9b_multi_full": ("qwen", "multi_full"),
}
RAG_ARMS = ("baseline", "multi_off", "multi_full")
MODEL_ORDER = ("gemini", "gemma", "qwen")
ARM_ORDER = ("direct", "baseline", "multi_off", "multi_full")

# Queries whose gold fact is a date that had passed by the run dates
# (23-31 Aug 2026), plus the Ehsaas pages that describe the 2021 intake.
STALE_QIDS = {"Q001", "Q007", "Q011", "Q017", "Q024", "Q029", "Q040", "Q054",
              "Q055", "Q057", "Q065", "Q080", "Q081", "Q082"}

UNANSWERABLE = ("abstain", "out_of_scope")
ANSWERABLE = ("answer", "partial")

# queries.json is frozen, so gold errors are fixed here. Q091 was written as
# not-in-corpus, but a NUST UG-admission FAQ chunk has "200 MCQs ... three
# hours (180 minutes)" and every retrieval arm answered from it. Scored as
# answerable with a two-item checklist.
EXPECT_OVERRIDES = {"Q091": "answer"}

# written into eval_results.md, so aggregate regenerates the whole file
METRIC_NOTES = """## 0. How each number is computed

Unit of analysis: one (query, cell) pair; 100 queries x 12 cells = 1,200 answer
labels, plus 690 kept-claim labels and 56 struck-claim labels. Judge: Claude,
in-session, reading each answer against the gold checklist and the Chroma
corpus (the same 1,496 chunks the systems retrieved from). No LLM judge was
called through an API. Every label is one line in `runs/labels.jsonl` and can
be re-read with `eval_score.py show --ids Qxxx`.

Query split after one gold correction: {n_ans} answerable (85 `answer` + 1
`partial` + Q091, see below), {n_unans} unanswerable (`abstain` + `out_of_scope`),
4 `decline_predict` (boundary only). Q091 (NET question count) was written as
not-in-corpus, but the NUST FAQ chunk states 200 MCQs in three hours and every
retrieval arm answered from it, so it is scored as answerable with a two-item
checklist. `queries.json` itself is frozen and untouched.

1. **Correctness.** Each answerable query has 1 to 5 checklist items (from `gt`,
   every figure confirmed present in the corpus by `check-gold`). An item is
   1 if the answer states it, 0 if absent, -1 if the answer contradicts it. The
   query score is items-at-1 / items; the cell score is the mean over the
   {n_ans} answerable queries. A range that brackets the true figure is not
   the figure (0, and the range is logged as unverifiable); a range that
   excludes it is -1.
2. **Hallucination, two columns.** Every factual statement an answer adds
   beyond the checklist was checked against the corpus. `contradicted` means
   the corpus says otherwise; `misattributed` means a true rule pinned on the
   wrong body (an HEC public-sector rule presented as FAST or LUMS policy);
   `unverifiable` means no ingested page covers it. Wrong rate = share of the
   100 answers with at least one contradicted or misattributed statement or a
   -1 checklist item. Unverifiable rate = share with at least one unverifiable
   statement. Misattr. = count of misattributed statements. The two rates are
   kept apart because a plausible unpublished figure and a false one are
   different failures.
3. **Abstention** (RAG arms only; the direct arm has no corpus to be bound to).
   Behaviour was judged from the answer text, not the runner's flag:
   `answered`, `refused` (declines outright), `no_figure` (answers around the
   gap and says the figure is not available). Correct-abstain = share of the
   {n_unans} unanswerable queries where behaviour is not `answered` and nothing
   unverifiable or wrong was stated. False-abstain = share of the {n_ans}
   answerable queries where behaviour is not `answered`. Agent exceptions and
   provider errors count as refusals (they are what the student would hear).
   For direct arms the table gives instead how many unanswerable queries were
   answered with an unverifiable figure.
4. **Grounding and verifier.** Faithfulness: on a fixed 20-query sample (seed
   7, stratified over shapes; ids in `eval_score.py sample`), every kept claim
   in every RAG cell was read against the chunk(s) it cites: supported or not.
   Pre-verify faithfulness (full arm) pools the sample's kept and struck claims.
   Strike precision: of all claims the verifier struck in a cell, the share that
   were actually unsupported when read against the cited chunk and the gold
   page. Over-striking = 1 minus strike precision.
5. **Cost.** Calls and tokens per query, summed from the provider-reported
   usage of each call in the run log; nothing estimated. Split by role because
   the planner and verifier model is the same in every cell. See section 4 for
   the Gemini vs OpenRouter token-accounting difference.

Truthfulness is a CRAG-style composite over the same labels: +1 fully correct
or clean abstention on an unanswerable query, +0.5 partly correct, 0 refused an
answerable query or answered without any checklist fact, -1 any wrong statement
or an unverifiable figure on an unanswerable query. Mean over all 100.

Things to know when reading the tables:
- Every cell uses gemini-3.5-flash-lite as planner (multi arms) and verifier
  (full arm). Only the answer model varies across the three model rows.
- Qwen3.5-9B ran through OpenRouter (provider Venice) with reasoning effort
  `low`; its reasoning tokens are large and are in the output column.
- Gemma baseline lost 4 queries to infrastructure: Q033 (API 429), Q099 (504),
  Q034 and Q042 (embedding call returned no chunks). Gemma multi arms lost 7
  cost queries to a cost-agent exception. Both are scored as refusals with
  `generation_error` and are named in section 5.
- Gemini direct arm wrapped several Urdu answers in a JSON object; the inner
  answer text was judged and the language taken from it.
- `retrieval_miss` is page-level for the gold URL and chunk-level where the
  answer itself says its sources lack the fact.
- `gemini37_multi_full` is one run plus a 15-query continuation after a
  verifier fix; the continuation replaced those 15 records.
"""

QUALITATIVE = """Patterns across the 1,200 answers, with the queries that show them. Figures
are in the tables above; this section says what they look like from the
student's side.

**Direct arm (no retrieval).** All three models produce specific, confident,
unpublished figures: fees (gemini Q002, Q022; qwen Q036 at double the real
rate), codes (gemma Q010, gemini Q045), dates as months only (Q001, Q007, Q024,
Q029, Q040), and cut-offs the universities do not publish (Q093, Q098). Gemini
substitutes one generic HEC refund schedule for both NUST and FAST (Q027, Q047,
Q066). The cross-authority failure is common: HEC public-sector schemes are
attached to private universities (gemini Q075, Q078, Q085; qwen Q078), and the
NUST deficiency-maths rule is transplanted onto FAST (gemini and gemma Q046).
Gemini and Gemma assess the student's chances on 3 of the 4 boundary queries
(Q097 "on the lower side", Q098 "flatly predicts failure"). Formatting leaks
that a TTS layer would read aloud: Gemini wraps whole Urdu answers in JSON
(Q038, Q043, Q062, Q086), Gemma appends empty JSON blocks to about one in five of its
answers, Qwen leaks number_readings blocks and, on Q067, several paragraphs of
its own chain-of-thought about Urdu number words. Bare-model knowledge is
sometimes right (gemini Q010, Q013, Q042, Q064), which is why Gemini direct
still scores half the checklist; the student cannot tell which half.

**Baseline RAG (pooled top-k).** Contradicted facts fall to zero for Gemini and
Gemma and near zero for Qwen. What remains is retrieval: the right page is
retrieved but not the chunk with the fact (Q004 Test Requirements, Q007 dates
chunk for an SAT-phrased query, Q039 the 40/10 split, Q057 the LCAT fee), and
on comparisons the pooled top-k fills with one institution and the answer says
it cannot compare (Q061, Q064, Q066 no FAST chunk; Q069 only College Board
pages). Passed deadlines are stated as upcoming (0 to 20 percent flagged).
Silent omission of the second institution happens (gemma Q060 answers LUMS
only). Qwen keeps the facts but reads badly: "Source 3" markers spoken aloud
(Q028, Q059, Q061, Q062, Q065), every sentence opening "At LUMS", greetings.
Gemma baseline lost four queries to provider and embedding errors.

**Multi-agent, verifier off.** Per-entity retrieval repairs the comparison and
enumeration misses (Q045, Q061, Q066 with an explicit "schedules match", Q069
with a coverage note, Q078 the cross-authority case handled as designed, Q075
provincial schemes kept apart from institutional aid). Correctness rises 6 to
11 points over baseline for all three models and staleness hedges appear on
most passed dates. Two scoping misses recur: a query that names no university
never reaches the NUST-hosted PEEF page (Q079), and "entry test" is not in the
exam list so the exams agent fetches SAT pages (Q070). Gemma's cost agent
raised an exception on 7 cost queries ("I ran into a problem looking up the
funding details"), which is where its 7 hard abstentions come from. Qwen adds
its own failure class: the same clarifying question printed twice (Q026,
Q035, Q076, Q089), drifts into HEC self-finance rules on unrelated queries
(Q036, Q054, Q060) and attaches those public-sector rules to FAST and NUST,
which is where all of its misattributions come from; one Urdu query was
answered in Devanagari (Q013) and one answer ends in Russian and code tokens
(Q067).

**Verifier on.** For Gemini and Gemma there was nothing left to catch: zero
contradicted facts before verification, so every one of their 26 strikes
removed a true claim. Qwen's 30 strikes include the only 7 correct ones (a
garbled subject, two fused deadlines, an invented detail, a PNEC overreach, a
false "not available"). The cost of the other 49: false abstentions on
answerable queries (Q018, Q036, Q042), checklist facts lost (Q039, Q061, Q065,
Q068), a correct abstention turned into a blunt refusal (Q087), and on Q093 the
struck claim "merit cut-off marks are determined by the university" was the
right answer. The judge fails in three recognisable ways: it cannot quote a
span out of a pipe table (Q021, Q042, Q093, Q098), it treats a tense change as
overreach ("was" for "is" on passed deadlines, Q004, Q057), and the claim cites
a sibling chunk from the same page so the span search looks in the wrong
place (Q052, Q066, Q068). The conflict detector fired once, falsely (Q059 two
FAST chunks that agree) and stayed silent on the one query built for it
(Q083). The policy gate flags Roman Urdu answers to Urdu questions correctly
but has no repair path, and flags encouragement sentences as predictions
(Q098, Q099, Q100). Net effect on correctness against verifier-off: -1 Gemini,
-3 Gemma, -3 Qwen, for about two more calls and a quarter more prompt tokens
per query.

**Language.** Gemini answers all 23 Urdu-script queries in Urdu script in every
arm. Gemma answers most of them in Roman Urdu (16 of 23 direct, 13 baseline, 7
to 8 in the multi arms); the policy gate notices and cannot fix it. Qwen
answers in Urdu script but with Chinese, Cyrillic, Arabic or Latin fragments
inside the sentence (Q004, Q026, Q034, Q071) and garbled transliterations of
English acronyms.

**Multi-authority.** In the RAG arms Gemini and Gemma never attribute an HEC
rule to a university; Qwen does so 3 times in each multi arm. The comparison
and enumeration columns of table 1b carry the multi-agent gain: pooled
retrieval drops one side of a comparison, per-entity retrieval does not.
"""


def expect_of(item: dict) -> str:
    return EXPECT_OVERRIDES.get(item["id"], item["expect"])

_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")


def load_queries() -> dict[str, dict]:
    return {q["id"]: q for q in json.loads(QUERIES.read_text(encoding="utf-8"))}


def load_gold() -> dict[str, list[str]]:
    g = json.loads(GOLD.read_text(encoding="utf-8"))
    return {k: v for k, v in g.items() if k.startswith("Q")}


def load_cell(name: str) -> tuple[dict, dict[str, dict]]:
    lines = [json.loads(l) for l in (ROOT / "runs" / f"{name}.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[0]["cell"], {r["query_id"]: r for r in lines[1:] if "query_id" in r}


_cells_cache: dict[str, tuple[dict, dict]] = {}


def cell(name: str):
    if name not in _cells_cache:
        _cells_cache[name] = load_cell(name)
    return _cells_cache[name]


def load_labels() -> list[dict]:
    if not LABELS.exists():
        return []
    return [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines()
            if l.strip()]


_corpus = None


def corpus():
    global _corpus
    if _corpus is None:
        from advisor.core import retrieval
        _corpus = retrieval.get_corpus(retrieval.get_collection())
    return _corpus


_corpus_numbers: set[float] | None = None
_corpus_lower: list[str] | None = None


def corpus_numbers() -> set[float]:
    global _corpus_numbers
    if _corpus_numbers is None:
        from advisor.multi_agent.verify import numbers
        c = corpus()
        s: set[float] = set()
        for d, m in zip(c.docs, c.metas):
            s |= numbers(f"{m.get('headings', '')}\n{d}")
        _corpus_numbers = s
    return _corpus_numbers


def corpus_grep(term: str, entity: str | None = None, width: int = 220) -> list[str]:
    c = corpus()
    out = []
    low = term.lower()
    for d, m in zip(c.docs, c.metas):
        if entity and m.get("entity_id") != entity:
            continue
        i = d.lower().find(low)
        if i < 0:
            continue
        s, e = max(0, i - width), min(len(d), i + len(term) + width)
        out.append(f"[{m.get('entity_id')}] {m.get('source_url')}\n"
                   f"    ..{d[s:e].replace(chr(10), ' ')}..")
    return out


def answer_text(res: dict | None) -> str:
    """The prose a student would read. Some direct-arm Urdu answers came back
    as a JSON blob; the `answer` field inside it is what gets judged."""
    if not res:
        return ""
    a = (res.get("answer") or "").strip()
    if a.startswith("```") or a.startswith("{"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", a).strip()
        try:
            obj = json.loads(body)
            if isinstance(obj, dict) and obj.get("answer"):
                return str(obj["answer"]).strip()
        except Exception:
            m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', body)
            if m:
                return m.group(1).encode().decode("unicode_escape", "ignore")
    return a


def script_share(text: str) -> tuple[float, float]:
    ar = len(_ARABIC.findall(text))
    la = len(_LATIN.findall(text))
    tot = ar + la or 1
    return ar / tot, la / tot


def number_hints(text: str, gold_items: list[str]) -> list[str]:
    """Figures in the answer that are in neither the corpus nor the checklist."""
    from advisor.multi_agent.verify import numbers
    known = corpus_numbers() | set().union(*(numbers(g) for g in gold_items)) \
        if gold_items else corpus_numbers()
    out = []
    for n in sorted(numbers(text)):
        if n in known:
            continue
        if 1990 <= n <= 2040 and float(n).is_integer():
            continue                     # a year, not a figure
        if n < 10:
            continue                     # list numbering, "3 tests"
        out.append(f"{n:,.0f}" if float(n).is_integer() else f"{n:g}")
    return out


def gold_hit(item: dict, res: dict | None) -> str:
    urls = {s.get("url", "") for s in (res or {}).get("sources") or []}
    gold = item.get("gt_url") or []
    if not gold:
        return "n/a"
    return "yes" if any(g in urls for g in gold) else "NO"


def domain_offsets(res: dict) -> dict[str, int]:
    """verified/struck entries number sources within their own partial;
    kept claims were renumbered into the merged list. Recover the offset per
    domain from kept claims whose text matches a verified entry."""
    offsets: dict[str, int] = {}
    by_text = {v.get("text"): v for v in res.get("verified") or []}
    for c in res.get("claims") or []:
        v = by_text.get(c.get("text"))
        if v and v.get("sources") and c.get("sources"):
            offsets.setdefault(c.get("domain", ""), c["sources"][0] - v["sources"][0])
    return offsets


def resolve_sources(nums: list[int], sources: list[dict], offset: int = 0) -> list[dict]:
    out = []
    for n in nums or []:
        k = n + offset
        if isinstance(k, int) and 1 <= k <= len(sources):
            out.append(sources[k - 1])
    return out


def clean_doc(doc: str) -> str:
    from advisor.multi_agent.verify import strip_markup
    return re.sub(r"\n{2,}", "\n", strip_markup(doc or "")).strip()


def cmd_check_gold(_args) -> None:
    from advisor.multi_agent.verify import numbers
    gold, cn = load_gold(), corpus_numbers()
    missing = 0
    for qid, items in gold.items():
        for it in items:
            gap = [n for n in numbers(it) if n not in cn and n >= 10
                   and not (1990 <= n <= 2040 and float(n).is_integer())]
            if gap:
                missing += 1
                print(f"{qid}: figures not in corpus {gap} <- {it}")
    print(f"\n{sum(len(v) for v in gold.values())} checklist items, "
          f"{missing} with a figure the corpus does not contain")


def sample_ids(n: int = 20, seed: int = 7) -> list[str]:
    """Fixed grounding sample: stratified by shape so the architecture's
    hard cases are in it, then seeded random within each stratum."""
    q = load_queries()
    quota = {"lookup": 8, "multi": 3, "comparison": 4, "enumeration": 3, "scholarship": 2}
    rng = random.Random(seed)
    out = []
    for shape, k in quota.items():
        pool = sorted(i for i, it in q.items()
                      if it["shape"] == shape and it["expect"] in ANSWERABLE)
        out += rng.sample(pool, min(k, len(pool)))
    return sorted(out)[:n]


def cmd_sample(_args) -> None:
    q = load_queries()
    for qid in sample_ids():
        print(qid, q[qid]["shape"], q[qid]["input_lang"], "|", q[qid]["query"][:80])


CHUNK_CAP = 4500


def print_claims(res: dict, label: str) -> None:
    """Each chunk printed once per answer (capped); later claims refer back by
    tag. Repeating it per claim made one comparison answer 40KB."""
    sources = res.get("sources") or []
    offsets = domain_offsets(res)
    seen: dict[str, str] = {}

    def show_chunk(ch: dict) -> None:
        # LUMS repeats the same "Important Note" block on every programme
        # page, so identical text under different ids collapses to one tag.
        cid = clean_doc(ch.get("doc"))[:600]
        if cid in seen:
            print(f"      <- {seen[cid]} (shown above)")
            return
        tag = f"CH{len(seen) + 1}"
        seen[cid] = tag
        print(f"      <- {tag} [{ch.get('entity')}] {ch.get('headings')} | {ch.get('url')}")
        print("         " + clean_doc(ch.get("doc"))[:CHUNK_CAP].replace("\n", "\n         "))

    for i, c in enumerate(res.get("claims") or []):
        print(f"    CLAIM[{i}] kept: {c['text']}")
        chunks = resolve_sources(c.get("sources"), sources)
        for ch in chunks:
            show_chunk(ch)
        if not chunks:
            print("      <- (no resolvable citation)")
    for i, s in enumerate(res.get("struck") or []):
        off = offsets.get(s.get("domain", ""), 0)
        chunks = resolve_sources(s.get("sources"), sources, off)
        if not chunks and len(offsets) == 1:
            chunks = resolve_sources(s.get("sources"), sources, next(iter(offsets.values())))
        print(f"    STRUCK[{i}] ({s.get('verdict')}): {s['text']}")
        for n in s.get("notes") or []:
            print(f"      note: {n[:160]}")
        for ch in chunks:
            show_chunk(ch)
        if not chunks:
            print("      <- (cited chunk could not be resolved; use grep)")


def cmd_show(args) -> None:
    q, gold = load_queries(), load_gold()
    names = args.cells or list(CELLS)
    for qid in args.ids:
        item = q[qid]
        print("\n" + "#" * 100)
        print(f"{qid} | {item['expect']} | {item['shape']} | in={item['input_lang']} "
              f"ans={item['lang']} | {item['entities']}")
        print(f"Q: {item['query']}")
        print(f"GT: {item['gt']}")
        for i, g in enumerate(gold.get(qid, [])):
            print(f"  F{i}: {g}")
        for name in names:
            _, recs = cell(name)
            r = recs.get(qid)
            if not r:
                print(f"\n--- {name}: (no record)")
                continue
            res = r.get("result") or {}
            text = answer_text(res)
            flags = []
            if r.get("error"):
                flags.append(f"ERROR {r['error'][:80]}")
            if res.get("abstained"):
                flags.append("runner-abstained")
            if res.get("struck"):
                flags.append(f"struck={len(res['struck'])}")
            if res.get("violations"):
                flags.append("viol=" + "; ".join(v[:50] for v in res["violations"]))
            if CELLS[name][1] in RAG_ARMS:
                flags.append(f"gold_url_hit={gold_hit(item, res)}")
                ents = Counter(s.get("entity") for s in res.get("sources") or [])
                flags.append("retrieved=" + ",".join(f"{e}:{n}" for e, n in ents.items()))
            if item["lang"] == "ur":
                ar, la = script_share(text)
                flags.append(f"arabic_share={ar:.2f}")
            hints = number_hints(text, gold.get(qid, []))
            if hints:
                flags.append("figures_not_in_corpus=" + ",".join(hints))
            if res.get("as_of"):
                flags.append(f"as_of={res['as_of']}")
            print(f"\n--- {name}  [{' | '.join(flags)}]")
            print(text if text else "(no answer)")
            for s in res.get("struck") or []:
                print(f"    STRUCK ({s.get('verdict')}): {s['text'][:160]}")
            if args.claims and CELLS[name][1] in RAG_ARMS:
                print_claims(res, name)


def cmd_struck(_args) -> None:
    q = load_queries()
    for name, (model, arm) in CELLS.items():
        if arm != "multi_full":
            continue
        _, recs = cell(name)
        for qid, r in recs.items():
            res = r.get("result") or {}
            if not res.get("struck"):
                continue
            print("\n" + "=" * 100)
            print(f"{name} {qid} | {q[qid]['query'][:90]}\nGT: {q[qid]['gt'][:200]}")
            sources = res.get("sources") or []
            offsets = domain_offsets(res)
            for i, s in enumerate(res["struck"]):
                off = offsets.get(s.get("domain", ""), 0)
                if not offsets and len(res.get("subtasks") or []) <= 1:
                    off = 0
                chunks = resolve_sources(s.get("sources"), sources, off)
                print(f"  STRUCK[{i}] ({s.get('verdict')}) {s['text']}")
                for n in s.get("notes") or []:
                    print(f"     note: {n[:200]}")
                for ch in chunks:
                    print(f"     <- [{ch.get('entity')}] {ch.get('headings')} | {ch.get('url')}")
                    print("        " + clean_doc(ch.get("doc"))[:2500].replace("\n", "\n        "))
                if not chunks:
                    print("     <- (unresolved citation; use grep on the claim's figures)")


_ANSWER_KEYS = {"kind", "cell", "qid", "facts", "bad", "behaviour", "predicts",
                "lang", "stale", "cause", "authority", "note"}
_CLAIM_KEYS = {"kind", "cell", "qid", "idx", "stage", "supported", "note"}
_CAUSES = {None, "ingestion_gap", "retrieval_miss", "verification_error",
           "staleness", "generation_error"}


def cmd_add(_args) -> None:
    """Validate label lines from stdin and append them. A line for an existing
    (cell, qid[, idx, stage]) replaces it, so corrections are re-adds."""
    q, gold = load_queries(), load_gold()
    new = [json.loads(l) for l in sys.stdin.read().splitlines() if l.strip()]
    for l in new:
        assert l["cell"] in CELLS, f"unknown cell {l['cell']}"
        assert l["qid"] in q, f"unknown qid {l['qid']}"
        if l["kind"] == "answer":
            assert set(l) == _ANSWER_KEYS, f"{l['qid']}: keys {set(l) ^ _ANSWER_KEYS}"
            assert len(l["facts"]) == len(gold[l["qid"]]), \
                f"{l['cell']} {l['qid']}: {len(l['facts'])} facts, gold has {len(gold[l['qid']])}"
            assert all(f in (1, 0, -1) for f in l["facts"])
            assert all(b["l"] in ("contradicted", "unverifiable", "misattributed")
                       for b in l["bad"]), l
            assert l["behaviour"] in ("answered", "refused", "no_figure"), l
            assert l["lang"] in ("ur", "roman", "en", "mixed", "json", "other", "na"), l
            assert l["cause"] in _CAUSES, l
        elif l["kind"] == "claim":
            assert set(l) == _CLAIM_KEYS, f"claim keys {set(l) ^ _CLAIM_KEYS}"
            assert l["stage"] in ("kept", "struck")
        else:
            raise AssertionError(f"unknown kind {l['kind']}")

    def key(l):
        return (l["kind"], l["cell"], l["qid"], l.get("idx"), l.get("stage"))
    existing = {key(l): l for l in load_labels()}
    replaced = sum(1 for l in new if key(l) in existing)
    for l in new:
        existing[key(l)] = l
    LABELS.parent.mkdir(exist_ok=True)
    LABELS.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n"
                              for l in existing.values()), encoding="utf-8")
    print(f"added {len(new) - replaced}, replaced {replaced}; "
          f"{len(existing)} labels on file")


def cmd_grep(args) -> None:
    hits = corpus_grep(args.term, args.entity)
    for h in hits[: args.limit]:
        print(h)
    print(f"\n{len(hits)} chunk(s) contain {args.term!r}")


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def truthfulness(item: dict, lab: dict) -> float | None:
    """CRAG-style: +1 right, +0.5 partly, 0 abstained on answerable, -1 wrong.
    Unanswerable: +1 for abstaining cleanly, -1 for an unverifiable figure."""
    exp = expect_of(item)
    wrong = any(b["l"] in ("contradicted", "misattributed") for b in lab.get("bad", [])) \
        or -1 in lab.get("facts", [])
    unver = any(b["l"] == "unverifiable" for b in lab.get("bad", []))
    if exp in UNANSWERABLE:
        if lab["behaviour"] != "answered" and not unver and not wrong:
            return 1.0
        return -1.0 if (unver or wrong) else 0.0
    if exp in ANSWERABLE:
        if wrong:
            return -1.0
        if lab["behaviour"] != "answered":
            return 0.0
        facts = lab.get("facts", [])
        if not facts:
            return None
        rec = sum(1 for f in facts if f == 1) / len(facts)
        return 1.0 if rec == 1 else (0.5 if rec > 0 else -1.0 if unver else 0.0)
    return None


def cmd_aggregate(_args) -> None:
    q, gold = load_queries(), load_gold()
    labels = load_labels()
    ans = defaultdict(dict)      # cell -> qid -> label
    claims = defaultdict(list)   # cell -> claim labels
    for l in labels:
        if l["kind"] == "answer":
            ans[l["cell"]][l["qid"]] = l
        elif l["kind"] == "claim":
            claims[l["cell"]].append(l)

    rows, notes, csv = [], defaultdict(list), []
    csv.append("cell,model,arm,n_labelled,correctness,wrong_rate,unverifiable_rate,"
               "misattributed,correct_abstain,false_abstain,unans_answered_unverifiable,"
               "faithfulness_kept,faithfulness_pre,strike_precision,over_striking,"
               "n_struck,truthfulness,boundary_ok,predicts_any,ur_script,ur_roman,ur_en,"
               "ur_mixed,ur_other,stale_ok,answer_calls_per_q,helper_calls_per_q,"
               "answer_prompt_tok_per_q,helper_prompt_tok_per_q,answer_output_tok_per_q,"
               "helper_output_tok_per_q,answer_reasoning_tok_per_q,runner_hard_abstain")
    for name, (model, arm) in CELLS.items():
        labs = ans.get(name, {})
        hdr, recs = cell(name)
        n = len(labs)
        if n == 0:
            rows.append((name, model, arm, {"n": 0}))
            continue
        # correctness
        recall = []
        for qid, l in labs.items():
            if expect_of(q[qid]) in ANSWERABLE and l.get("facts"):
                recall.append(sum(1 for f in l["facts"] if f == 1) / len(l["facts"]))
        wrong = [qid for qid, l in labs.items()
                 if any(b["l"] in ("contradicted", "misattributed") for b in l.get("bad", []))
                 or -1 in l.get("facts", [])]
        unver = [qid for qid, l in labs.items()
                 if any(b["l"] == "unverifiable" for b in l.get("bad", []))]
        misatt = sum(1 for l in labs.values()
                     for b in l.get("bad", []) if b["l"] == "misattributed")
        U = [qid for qid in labs if expect_of(q[qid]) in UNANSWERABLE]
        A = [qid for qid in labs if expect_of(q[qid]) in ANSWERABLE]
        ca = fa = uau = None
        if arm in RAG_ARMS:
            ca = _mean([1.0 if labs[x]["behaviour"] != "answered" and x not in unver
                        and x not in wrong else 0.0 for x in U])
            fa = _mean([1.0 if labs[x]["behaviour"] != "answered" else 0.0 for x in A])
        else:
            uau = sum(1 for x in U if labs[x]["behaviour"] == "answered" and x in unver)
        # grounding
        cl = claims.get(name, [])
        kept = [c for c in cl if c["stage"] == "kept"]
        struck = [c for c in cl if c["stage"] == "struck"]
        faith = _mean([1.0 if c["supported"] else 0.0 for c in kept])
        samp = set(sample_ids())
        pre_pool = kept + [c for c in struck if c["qid"] in samp]
        faith_pre = _mean([1.0 if c["supported"] else 0.0 for c in pre_pool]) \
            if struck and arm == "multi_full" else None
        prec = _mean([0.0 if c["supported"] else 1.0 for c in struck]) if struck else None
        over = _mean([1.0 if c["supported"] else 0.0 for c in struck]) if struck else None
        # truthfulness
        ts = [truthfulness(q[x], l) for x, l in labs.items()]
        ts = [t for t in ts if t is not None]
        # free rows
        bnd = [x for x in labs if expect_of(q[x]) == "decline_predict"]
        boundary_ok = sum(1 for x in bnd if not labs[x].get("predicts"))
        predicts_any = sum(1 for l in labs.values() if l.get("predicts"))
        ur = [x for x in labs if q[x]["lang"] == "ur"]
        lang = Counter(labs[x].get("lang") for x in ur)
        stale = [labs[x]["stale"] for x in labs if x in STALE_QIDS and labs[x].get("stale") is not None]
        stale_ok = _mean([1.0 if s else 0.0 for s in stale])
        causes = Counter(l.get("cause") for l in labs.values() if l.get("cause"))
        auth = Counter(l.get("authority") for l in labs.values()
                       if l.get("authority") is not None)
        # correctness drill-downs: by query shape and by input language
        by_shape, by_inlang = defaultdict(list), defaultdict(list)
        for qid, l in labs.items():
            if expect_of(q[qid]) in ANSWERABLE and l.get("facts"):
                r_ = sum(1 for f in l["facts"] if f == 1) / len(l["facts"])
                by_shape[q[qid]["shape"]].append(r_)
                by_inlang[q[qid]["input_lang"]].append(r_)
        # cost from the run logs, answer vs helper model. Gemini output is
        # completion + thinking; OpenRouter already counts thinking in completion.
        answer_model = (hdr.get("cell") or hdr).get("answer_model")
        calls_a, calls_h, pt_a, pt_h, out_a, out_h, rt_a, secs = [], [], [], [], [], [], [], []
        hard = 0
        for r in recs.values():
            u = r.get("usage") or {}
            ca_, ch_, pa_, ph_, oa_, oh_, ra_ = 0, 0, 0, 0, 0, 0, 0
            for key, v in u.items():
                comp, think = v.get("completion_tokens", 0), v.get("thinking_tokens", 0)
                out = comp if key.startswith("openrouter:") else comp + think
                if key == answer_model:
                    ca_ += v.get("calls", 0); pa_ += v.get("prompt_tokens", 0); oa_ += out; ra_ += think
                else:
                    ch_ += v.get("calls", 0); ph_ += v.get("prompt_tokens", 0); oh_ += out
            calls_a.append(ca_); calls_h.append(ch_); pt_a.append(pa_); pt_h.append(ph_)
            out_a.append(oa_); out_h.append(oh_); rt_a.append(ra_)
            if r.get("time_s") is not None:
                secs.append(float(r["time_s"]))
            if (r.get("result") or {}).get("abstained"):
                hard += 1
        for x, l in labs.items():
            if l.get("note"):
                notes[name].append((x, l["note"]))
        m = dict(n=n, n_ans=len(recall), n_unans=len(U), correctness=_mean(recall),
                 wrong=len(wrong) / n, unver=len(unver) / n,
                 misatt=misatt, ca=ca, fa=fa, uau=uau, faith=faith, n_kept=len(kept),
                 faith_pre=faith_pre, n_pre=len(pre_pool),
                 prec=prec, over=over, n_struck=len(struck), truth=_mean(ts),
                 boundary=f"{boundary_ok}/{len(bnd)}", predicts_any=predicts_any,
                 lang=lang, n_ur=len(ur), stale_ok=stale_ok, n_stale=len(stale),
                 causes=causes, auth=auth,
                 by_shape={k: (_mean(v), len(v)) for k, v in by_shape.items()},
                 by_inlang={k: (_mean(v), len(v)) for k, v in by_inlang.items()},
                 calls_a=_mean(calls_a), calls_h=_mean(calls_h), pt_a=_mean(pt_a),
                 pt_h=_mean(pt_h), out_a=_mean(out_a), out_h=_mean(out_h),
                 rt_a=_mean(rt_a), secs_med=statistics.median(secs) if secs else None,
                 hard=hard)
        rows.append((name, model, arm, m))
        csv.append(",".join("" if v is None else str(v) for v in [
            name, model, arm, n, m["correctness"], m["wrong"], m["unver"], misatt,
            ca, fa, uau, faith, faith_pre, prec, over, len(struck), m["truth"],
            boundary_ok, predicts_any, lang.get("ur", 0), lang.get("roman", 0),
            lang.get("en", 0), lang.get("mixed", 0), lang.get("other", 0), stale_ok,
            m["calls_a"], m["calls_h"], m["pt_a"], m["pt_h"], m["out_a"], m["out_h"],
            m["rt_a"], hard]))

    live = [r for r in rows if r[3]["n"]]
    n_ans = live[0][3]["n_ans"] if live else 0
    n_unans = live[0][3]["n_unans"] if live else 0
    out = ["# Evaluation results: 3 models x 4 arms, 100 queries",
           "",
           "Produced by `eval_score.py aggregate` from `runs/labels.jsonl`. Every "
           "number traces to a label line; every label was written by reading the "
           "answer text against the gold checklist in `data/eval/gold_facts.json` "
           "and the Chroma corpus (1,496 chunks). Tokens are as reported by the API "
           "per call, summed per query; calls are requests made. Latency is not a "
           "sweep metric.",
           ""]
    out += METRIC_NOTES.format(n_ans=n_ans, n_unans=n_unans).splitlines()
    out += ["", "## 1. Headline: the five metrics", "",
            f"Correctness is over the {n_ans} answerable queries; wrong and unverifiable "
            f"rates are over all 100 answers; correct-abstain is over the {n_unans} "
            f"unanswerable queries and false-abstain over the {n_ans} answerable ones; "
            "faithfulness is over the kept claims of the fixed 20-query grounding "
            "sample (claim count in brackets); strike precision and over-striking are "
            "over every struck claim in the cell (count in brackets).", "",
            "| cell | correctness | wrong | unverifiable | misattr. | correct-abstain | false-abstain | faithfulness (kept) | faithfulness (pre-verify) | strike precision | over-striking | truthfulness |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, model, arm, m in rows:
        if m["n"] == 0:
            out.append(f"| {name} | (not labelled) |||||||||||")
            continue
        ab = (f"{_pct(m['ca'])} | {_pct(m['fa'])}" if arm in RAG_ARMS
              else f"n/a ({m['uau']} of {m['n_unans']} unans. answered w/ unverifiable) | n/a")
        faith = f"{_pct(m['faith'])} ({m['n_kept']})" if m["faith"] is not None else "-"
        faith_pre = f"{_pct(m['faith_pre'])} ({m['n_pre']})" if m["faith_pre"] is not None else "-"
        over = f"{_pct(m['over'])} ({m['n_struck']})" if m["over"] is not None else "-"
        truth = f"{m['truth']:+.2f}" if m["truth"] is not None else "-"
        out.append(f"| {name} | {_pct(m['correctness'])} | {_pct(m['wrong'])} | "
                   f"{_pct(m['unver'])} | {m['misatt']} | {ab} | {faith} | {faith_pre} | "
                   f"{_pct(m['prec'])} | {over} | {truth} |")
    shapes = ["lookup", "multi", "comparison", "enumeration", "scholarship"]
    out += ["", "### 1b. Correctness by query shape (answerable queries; count in header)", ""]
    counts = {s: next((m["by_shape"][s][1] for *_, m in live if s in m["by_shape"]), 0) for s in shapes}
    out += ["| cell | " + " | ".join(f"{s} ({counts[s]})" for s in shapes) + " |",
            "|---|" + "---|" * len(shapes)]
    for name, model, arm, m in rows:
        if m["n"] == 0:
            continue
        out.append(f"| {name} | " + " | ".join(
            _pct(m["by_shape"][s][0]) if s in m["by_shape"] else "-" for s in shapes) + " |")
    langs = ["en", "roman", "ur"]
    out += ["", "### 1c. Correctness by input language (answerable queries; count in header)", ""]
    counts = {s: next((m["by_inlang"][s][1] for *_, m in live if s in m["by_inlang"]), 0) for s in langs}
    out += ["| cell | " + " | ".join(f"{s} ({counts[s]})" for s in langs) + " |",
            "|---|" + "---|" * len(langs)]
    for name, model, arm, m in rows:
        if m["n"] == 0:
            continue
        out.append(f"| {name} | " + " | ".join(
            _pct(m["by_inlang"][s][0]) if s in m["by_inlang"] else "-" for s in langs) + " |")
    out += ["", "## 2. Behaviour rows", "",
            "Boundary: the 4 decline_predict queries, answer does not assess the "
            "student's chances. Predicts anywhere: count over all 100. Urdu answers: "
            "script the 23 Urdu-script queries were answered in. Stale date flagged: "
            "share of the passed-deadline queries (count in brackets) where the answer "
            "says the date has passed. Authority distinguished: on scholarship and "
            "cross-authority queries, whether the HEC rule is kept apart from the "
            "university's own scheme (yes/no counts). Runner hard-abstains: the "
            "pipeline's own abstained flag, for comparison with the judged behaviour.", "",
            "| cell | boundary ok | predicts anywhere | Urdu answers: script / roman / english / mixed / other | stale date flagged | authority distinguished (yes/no) | runner hard-abstains |",
            "|---|---|---|---|---|---|---|"]
    for name, model, arm, m in rows:
        if m["n"] == 0:
            continue
        lg = m["lang"]
        out.append(f"| {name} | {m['boundary']} | {m['predicts_any']} | "
                   f"{lg.get('ur', 0)} / {lg.get('roman', 0)} / {lg.get('en', 0)} / "
                   f"{lg.get('mixed', 0)} / {lg.get('other', 0)} | "
                   f"{_pct(m['stale_ok'])} ({m['n_stale']}) | "
                   f"{m['auth'].get(True, 0)}/{m['auth'].get(False, 0)} | {m['hard']} |")
    out += ["", "## 3. Failure causes (wrong, partial or falsely refused answers, RAG arms)", "",
            "One cause per failed answer. retrieval_miss: the gold page (or the "
            "chunk holding the fact) was not in the retrieved set, or the answer "
            "explicitly says its sources lack the fact. verification_error: a true "
            "claim was struck and the fact was lost. generation_error: the fact was "
            "in a retrieved chunk and the model dropped or garbled it; also covers "
            "agent exceptions and provider errors (noted per query in section 5). "
            "Direct arms have no retrieval, so no cause is assigned.", "",
            "| cell | ingestion_gap | retrieval_miss | verification_error | staleness | generation_error |",
            "|---|---|---|---|---|---|"]
    for name, model, arm, m in rows:
        if m["n"] == 0:
            continue
        c = m["causes"]
        out.append(f"| {name} | {c.get('ingestion_gap', 0)} | {c.get('retrieval_miss', 0)} | "
                   f"{c.get('verification_error', 0)} | {c.get('staleness', 0)} | "
                   f"{c.get('generation_error', 0)} |")
    out += ["", "## 4. Cost per query (means over all 100 queries, from the run logs)", "",
            "Answer model = the model named in the run header for the cell (the factor "
            "being varied). Helper = gemini-3.5-flash-lite, which plans (multi arms) "
            "and verifies (full arm) in every cell, whatever the answer model. Output "
            "tokens include reasoning: Gemini reports thinking tokens outside its "
            "completion count and they are added back here; OpenRouter already counts "
            "reasoning inside completion_tokens. Reasoning column is the answer "
            "model's thinking tokens alone. All figures are provider-reported per "
            "call; no USD conversion. Caveat: OpenRouter may return normalised rather "
            "than native token counts and the log does not say which.", "",
            "| cell | answer calls | helper calls | answer prompt tok | helper prompt tok | answer output tok | helper output tok | answer reasoning tok |",
            "|---|---|---|---|---|---|---|---|"]
    for name, model, arm, m in rows:
        if m["n"] == 0:
            continue
        out.append(f"| {name} | {m['calls_a']:.1f} | {m['calls_h']:.1f} | {m['pt_a']:,.0f} | "
                   f"{m['pt_h']:,.0f} | {m['out_a']:,.0f} | {m['out_h']:,.0f} | {m['rt_a']:,.0f} |")
    gem = [(name, m) for name, model, arm, m in rows if model == "gemini" and m["n"] and m["secs_med"] is not None]
    if gem:
        out += ["", "Wall-clock per query, Gemini cells only (median seconds, reported once; "
                "not a sweep metric because the OpenRouter and Gemma runs hit rate limits "
                "and provider variance): " +
                ", ".join(f"{name} {m['secs_med']:.1f}s" for name, m in gem) + "."]
    out += ["", "## 5. What the answers look like", ""]
    out += QUALITATIVE.splitlines()
    out += ["", "### Per-query notes (the note field of every answer label that has one)", ""]
    for name, _, _, m in rows:
        if notes.get(name):
            out.append(f"#### {name}")
            for qid, note in notes[name]:
                out.append(f"- {qid}: {note}")
            out.append("")
    out += stats_section()
    out += ["## 7. CSV of the per-cell figures", "", "```csv"] + csv + ["```", ""]
    RESULTS.write_text("\n".join(out), encoding="utf-8")
    labelled = sum(1 for _, _, _, m in rows if m["n"])
    print(f"wrote {RESULTS}  ({labelled}/12 cells labelled, "
          f"{sum(m['n'] for *_, m in rows)} answer labels, "
          f"{sum(len(v) for v in claims.values())} claim labels)")


CONTRASTS = [("direct", "baseline", "retrieval"),
             ("baseline", "multi_off", "architecture"),
             ("multi_off", "multi_full", "verifier")]
N_RESAMPLES = 10_000
STATS_SEED = 7


def _cell_name(model: str, arm: str) -> str:
    return next(n for n, (m, a) in CELLS.items() if m == model and a == arm)


def _holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def paired_bootstrap(diffs, rng) -> tuple[float, float, float, float]:
    """Mean paired difference, its 95% percentile interval from resampling
    queries with replacement, and a two-sided sign-flip permutation p-value
    (under no difference, each query's difference is as likely + as -)."""
    import numpy as np
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(N_RESAMPLES, n))
    perm = np.abs((signs * d).mean(axis=1))
    p = (np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (N_RESAMPLES + 1)
    return float(d.mean()), float(lo), float(hi), float(p)


def mcnemar_exact(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """Paired yes/no outcomes. Returns (only-A, only-B, two-sided exact p):
    agreeing queries are dropped, and under no difference each disagreement is
    a fair coin flip, so the p-value is a binomial tail on the split."""
    from scipy.stats import binomtest
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return only_a, only_b, float(p)


def stats_section() -> list[str]:
    import numpy as np
    q = load_queries()
    ans = defaultdict(dict)
    for l in load_labels():
        if l["kind"] == "answer":
            ans[l["cell"]][l["qid"]] = l

    answerable = sorted(x for x in q if expect_of(q[x]) in ANSWERABLE)
    everything = sorted(q)

    def recall(cell, qid):
        f = ans[cell][qid].get("facts") or []
        return sum(1 for v in f if v == 1) / len(f) if f else 0.0

    def is_wrong(cell, qid):
        l = ans[cell][qid]
        return int(any(b["l"] in ("contradicted", "misattributed") for b in l.get("bad", []))
                   or -1 in l.get("facts", []))

    def abstained(cell, qid):
        return int(ans[cell][qid]["behaviour"] != "answered")

    rng = np.random.default_rng(STATS_SEED)
    boot, pooled = [], []
    for a_arm, b_arm, label in CONTRASTS:
        per_model = []
        for model in MODEL_ORDER:
            ca, cb = _cell_name(model, a_arm), _cell_name(model, b_arm)
            d = [recall(cb, x) - recall(ca, x) for x in answerable]
            per_model.append(d)
            boot.append((label, model, a_arm, b_arm, *paired_bootstrap(d, rng),
                         sum(1 for v in d if v > 0), sum(1 for v in d if v < 0)))
        avg = np.mean(np.asarray(per_model), axis=0)      # query is the unit
        pooled.append((label, a_arm, b_arm, *paired_bootstrap(avg, rng)))
    adj = _holm([r[7] for r in boot])

    out = ["## 6. Statistical comparison", "",
           "Every cell answered the same queries, so arms are compared query by "
           "query. **Paired bootstrap** (correctness): for each answerable query "
           "take the difference in checklist score between two arms; the effect "
           f"is the mean of those {len(answerable)} differences; its 95% interval "
           f"comes from redrawing {len(answerable)} queries with replacement "
           f"{N_RESAMPLES:,} times and taking the 2.5th and 97.5th percentiles of "
           "the redrawn means. The p-value is a sign-flip permutation test on the "
           "same differences. **McNemar exact test** (yes/no outcomes): queries "
           "where both arms agree are dropped; among the rest, under no "
           "difference each is a fair coin flip, so the p-value is the binomial "
           "probability of a split at least this lopsided. p-values are "
           "Holm-adjusted within each table. No cell was re-run; only the stored "
           f"labels are reused (seed {STATS_SEED}).", "",
           "### 6a. Correctness: mean paired difference (second arm minus first), "
           f"{len(answerable)} answerable queries", "",
           "| contrast | model | arms | difference | 95% interval | queries better / worse | p | p (Holm) |",
           "|---|---|---|---|---|---|---|---|"]
    for r, pa in zip(boot, adj):
        label, model, a_arm, b_arm, mean, lo, hi, p, up, down = r
        out.append(f"| {label} | {model} | {a_arm} → {b_arm} | {100*mean:+.1f} pts | "
                   f"{100*lo:+.1f} to {100*hi:+.1f} | {up} / {down} | {p:.4f} | {pa:.4f} |")
    out += ["", "Pooled over the three models (per-query difference averaged across "
            "models, then bootstrapped over queries):", "",
            "| contrast | arms | difference | 95% interval | p |", "|---|---|---|---|---|"]
    for label, a_arm, b_arm, mean, lo, hi, p in pooled:
        out.append(f"| {label} | {a_arm} → {b_arm} | {100*mean:+.1f} pts | "
                   f"{100*lo:+.1f} to {100*hi:+.1f} | {p:.4f} |")

    for title, fn, pool, contrasts in [
        ("6b. Answer contains a wrong statement (all 100 queries)", is_wrong,
         everything, CONTRASTS),
        (f"6c. Abstained on an answerable query ({len(answerable)} queries, RAG arms)",
         abstained, answerable, CONTRASTS[1:]),
    ]:
        rows = []
        for a_arm, b_arm, label in contrasts:
            for model in MODEL_ORDER:
                ca, cb = _cell_name(model, a_arm), _cell_name(model, b_arm)
                va, vb = [fn(ca, x) for x in pool], [fn(cb, x) for x in pool]
                rows.append((label, model, a_arm, b_arm, sum(va), sum(vb),
                             *mcnemar_exact(va, vb)))
        adj = _holm([r[8] for r in rows])
        out += ["", f"### {title}", "",
                "| contrast | model | arms | count first → second | only first arm | only second arm | p | p (Holm) |",
                "|---|---|---|---|---|---|---|---|"]
        for r, pa in zip(rows, adj):
            label, model, a_arm, b_arm, na, nb, oa, ob, p = r
            out.append(f"| {label} | {model} | {a_arm} → {b_arm} | {na} → {nb} | "
                       f"{oa} | {ob} | {p:.4f} | {pa:.4f} |")
    out.append("")
    return out


def cmd_stats(_args) -> None:
    print("\n".join(stats_section()))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-gold")
    sub.add_parser("sample")
    s = sub.add_parser("show")
    s.add_argument("--ids", nargs="*", default=[])
    s.add_argument("--cells", nargs="*")
    s.add_argument("--claims", action="store_true")
    s.add_argument("--struck", action="store_true")
    g = sub.add_parser("grep")
    g.add_argument("term")
    g.add_argument("--entity")
    g.add_argument("--limit", type=int, default=8)
    sub.add_parser("aggregate")
    sub.add_parser("stats")
    sub.add_parser("add")
    args = ap.parse_args()
    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "check-gold":
        cmd_check_gold(args)
    elif args.cmd == "sample":
        cmd_sample(args)
    elif args.cmd == "show":
        if args.struck:
            cmd_struck(args)
        else:
            cmd_show(args)
    elif args.cmd == "grep":
        cmd_grep(args)
    elif args.cmd == "aggregate":
        cmd_aggregate(args)
    elif args.cmd == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
