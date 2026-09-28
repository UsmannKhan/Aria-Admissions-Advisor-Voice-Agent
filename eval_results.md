# Evaluation results: 3 models x 4 arms, 100 queries

Produced by `eval_score.py aggregate` from `runs/labels.jsonl`. Every number traces to a label line; every label was written by reading the answer text against the gold checklist in `data/eval/gold_facts.json` and the Chroma corpus (1,496 chunks). Tokens are as reported by the API per call, summed per query; calls are requests made. Latency is not a sweep metric.

## 0. How each number is computed

Unit of analysis: one (query, cell) pair; 100 queries x 12 cells = 1,200 answer
labels, plus 690 kept-claim labels and 56 struck-claim labels. Judge: Claude,
in-session, reading each answer against the gold checklist and the Chroma
corpus (the same 1,496 chunks the systems retrieved from). No LLM judge was
called through an API. Every label is one line in `runs/labels.jsonl` and can
be re-read with `eval_score.py show --ids Qxxx`.

Query split after one gold correction: 87 answerable (85 `answer` + 1
`partial` + Q091, see below), 9 unanswerable (`abstain` + `out_of_scope`),
4 `decline_predict` (boundary only). Q091 (NET question count) was written as
not-in-corpus, but the NUST FAQ chunk states 200 MCQs in three hours and every
retrieval arm answered from it, so it is scored as answerable with a two-item
checklist. `queries.json` itself is frozen and untouched.

1. **Correctness.** Each answerable query has 1 to 5 checklist items (from `gt`,
   every figure confirmed present in the corpus by `check-gold`). An item is
   1 if the answer states it, 0 if absent, -1 if the answer contradicts it. The
   query score is items-at-1 / items; the cell score is the mean over the
   87 answerable queries. A range that brackets the true figure is not
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
   9 unanswerable queries where behaviour is not `answered` and nothing
   unverifiable or wrong was stated. False-abstain = share of the 87
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

## 1. Headline: the five metrics

Correctness is over the 87 answerable queries; wrong and unverifiable rates are over all 100 answers; correct-abstain is over the 9 unanswerable queries and false-abstain over the 87 answerable ones; faithfulness is over the kept claims of the fixed 20-query grounding sample (claim count in brackets); strike precision and over-striking are over every struck claim in the cell (count in brackets).

| cell | correctness | wrong | unverifiable | misattr. | correct-abstain | false-abstain | faithfulness (kept) | faithfulness (pre-verify) | strike precision | over-striking | truthfulness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini37_direct | 52% | 35% | 56% | 3 | n/a (9 of 9 unans. answered w/ unverifiable) | n/a | - | - | - | - | -0.17 |
| gemini37_baseline | 76% | 0% | 0% | 0 | 100% | 10% | 100% (65) | - | - | - | +0.79 |
| gemini37_multi_off | 87% | 0% | 0% | 0 | 100% | 6% | 100% (84) | - | - | - | +0.88 |
| gemini37_multi_full | 86% | 0% | 0% | 0 | 100% | 6% | 100% (69) | 100% (77) | 0% | 100% (13) | +0.88 |
| gemma31b_direct | 27% | 31% | 28% | 1 | n/a (3 of 9 unans. answered w/ unverifiable) | n/a | - | - | - | - | -0.21 |
| gemma31b_baseline | 69% | 0% | 0% | 0 | 100% | 15% | 99% (70) | - | - | - | +0.72 |
| gemma31b_multi_off | 77% | 1% | 0% | 0 | 100% | 15% | 100% (47) | - | - | - | +0.78 |
| gemma31b_multi_full | 74% | 0% | 0% | 0 | 100% | 15% | 100% (52) | 100% (55) | 0% | 100% (13) | +0.76 |
| qwen9b_direct | 16% | 54% | 37% | 2 | n/a (8 of 9 unans. answered w/ unverifiable) | n/a | - | - | - | - | -0.57 |
| qwen9b_baseline | 76% | 2% | 1% | 0 | 100% | 11% | 97% (92) | - | - | - | +0.76 |
| qwen9b_multi_off | 82% | 6% | 4% | 3 | 89% | 6% | 99% (108) | - | - | - | +0.71 |
| qwen9b_multi_full | 79% | 7% | 5% | 3 | 89% | 10% | 97% (103) | 96% (112) | 23% | 77% (30) | +0.65 |

### 1b. Correctness by query shape (answerable queries; count in header)

| cell | lookup (52) | multi (8) | comparison (9) | enumeration (7) | scholarship (11) |
|---|---|---|---|---|---|
| gemini37_direct | 47% | 49% | 51% | 81% | 61% |
| gemini37_baseline | 84% | 58% | 67% | 57% | 75% |
| gemini37_multi_off | 90% | 65% | 93% | 90% | 86% |
| gemini37_multi_full | 90% | 69% | 87% | 79% | 86% |
| gemma31b_direct | 24% | 10% | 26% | 69% | 25% |
| gemma31b_baseline | 75% | 58% | 58% | 57% | 66% |
| gemma31b_multi_off | 85% | 69% | 56% | 71% | 68% |
| gemma31b_multi_full | 81% | 58% | 44% | 71% | 77% |
| qwen9b_direct | 13% | 6% | 15% | 34% | 23% |
| qwen9b_baseline | 84% | 50% | 67% | 57% | 75% |
| qwen9b_multi_off | 86% | 65% | 93% | 79% | 70% |
| qwen9b_multi_full | 82% | 71% | 68% | 79% | 77% |

### 1c. Correctness by input language (answerable queries; count in header)

| cell | en (43) | roman (24) | ur (20) |
|---|---|---|---|
| gemini37_direct | 48% | 51% | 62% |
| gemini37_baseline | 77% | 77% | 74% |
| gemini37_multi_off | 87% | 85% | 90% |
| gemini37_multi_full | 85% | 88% | 88% |
| gemma31b_direct | 24% | 27% | 32% |
| gemma31b_baseline | 69% | 73% | 64% |
| gemma31b_multi_off | 73% | 75% | 88% |
| gemma31b_multi_full | 68% | 72% | 88% |
| qwen9b_direct | 10% | 20% | 22% |
| qwen9b_baseline | 78% | 76% | 70% |
| qwen9b_multi_off | 83% | 79% | 85% |
| qwen9b_multi_full | 78% | 80% | 79% |

## 2. Behaviour rows

Boundary: the 4 decline_predict queries, answer does not assess the student's chances. Predicts anywhere: count over all 100. Urdu answers: script the 23 Urdu-script queries were answered in. Stale date flagged: share of the passed-deadline queries (count in brackets) where the answer says the date has passed. Authority distinguished: on scholarship and cross-authority queries, whether the HEC rule is kept apart from the university's own scheme (yes/no counts). Runner hard-abstains: the pipeline's own abstained flag, for comparison with the judged behaviour.

| cell | boundary ok | predicts anywhere | Urdu answers: script / roman / english / mixed / other | stale date flagged | authority distinguished (yes/no) | runner hard-abstains |
|---|---|---|---|---|---|---|
| gemini37_direct | 1/4 | 3 | 23 / 0 / 0 / 0 / 0 | 77% (13) | 4/4 | 0 |
| gemini37_baseline | 4/4 | 0 | 23 / 0 / 0 / 0 / 0 | 0% (11) | 10/0 | 1 |
| gemini37_multi_off | 4/4 | 0 | 23 / 0 / 0 / 0 / 0 | 67% (9) | 10/0 | 0 |
| gemini37_multi_full | 4/4 | 0 | 23 / 0 / 0 / 0 / 0 | 89% (9) | 9/0 | 2 |
| gemma31b_direct | 1/4 | 3 | 7 / 16 / 0 / 0 / 0 | 70% (10) | 4/2 | 0 |
| gemma31b_baseline | 4/4 | 0 | 7 / 13 / 0 / 1 / 0 | 20% (10) | 10/0 | 3 |
| gemma31b_multi_off | 4/4 | 0 | 13 / 7 / 0 / 3 / 0 | 80% (10) | 6/0 | 7 |
| gemma31b_multi_full | 4/4 | 0 | 10 / 8 / 0 / 5 / 0 | 78% (9) | 7/0 | 8 |
| qwen9b_direct | 3/4 | 1 | 17 / 3 / 0 / 3 / 0 | 62% (8) | 3/1 | 0 |
| qwen9b_baseline | 4/4 | 0 | 15 / 8 / 0 / 0 / 0 | 9% (11) | 10/0 | 1 |
| qwen9b_multi_off | 4/4 | 0 | 20 / 2 / 0 / 0 / 1 | 78% (9) | 11/3 | 0 |
| qwen9b_multi_full | 4/4 | 0 | 14 / 5 / 1 / 3 / 0 | 100% (11) | 16/2 | 2 |

## 3. Failure causes (wrong, partial or falsely refused answers, RAG arms)

One cause per failed answer. retrieval_miss: the gold page (or the chunk holding the fact) was not in the retrieved set, or the answer explicitly says its sources lack the fact. verification_error: a true claim was struck and the fact was lost. generation_error: the fact was in a retrieved chunk and the model dropped or garbled it; also covers agent exceptions and provider errors (noted per query in section 5). Direct arms have no retrieval, so no cause is assigned.

| cell | ingestion_gap | retrieval_miss | verification_error | staleness | generation_error |
|---|---|---|---|---|---|
| gemini37_direct | 0 | 0 | 0 | 0 | 0 |
| gemini37_baseline | 0 | 19 | 0 | 0 | 10 |
| gemini37_multi_off | 0 | 10 | 0 | 0 | 7 |
| gemini37_multi_full | 0 | 10 | 2 | 0 | 5 |
| gemma31b_direct | 0 | 0 | 0 | 0 | 0 |
| gemma31b_baseline | 0 | 20 | 0 | 0 | 17 |
| gemma31b_multi_off | 0 | 9 | 0 | 0 | 16 |
| gemma31b_multi_full | 0 | 9 | 7 | 0 | 15 |
| qwen9b_direct | 0 | 0 | 0 | 0 | 0 |
| qwen9b_baseline | 0 | 18 | 0 | 0 | 10 |
| qwen9b_multi_off | 0 | 9 | 0 | 0 | 14 |
| qwen9b_multi_full | 0 | 7 | 10 | 0 | 11 |

## 4. Cost per query (means over all 100 queries, from the run logs)

Answer model = the model named in the run header for the cell (the factor being varied). Helper = gemini-3.5-flash-lite, which plans (multi arms) and verifies (full arm) in every cell, whatever the answer model. Output tokens include reasoning: Gemini reports thinking tokens outside its completion count and they are added back here; OpenRouter already counts reasoning inside completion_tokens. Reasoning column is the answer model's thinking tokens alone. All figures are provider-reported per call; no USD conversion. Caveat: OpenRouter may return normalised rather than native token counts and the log does not say which.

| cell | answer calls | helper calls | answer prompt tok | helper prompt tok | answer output tok | helper output tok | answer reasoning tok |
|---|---|---|---|---|---|---|---|
| gemini37_direct | 1.0 | 0.0 | 249 | 0 | 292 | 0 | 149 |
| gemini37_baseline | 1.0 | 0.0 | 5,126 | 0 | 322 | 0 | 53 |
| gemini37_multi_off | 1.1 | 1.1 | 11,363 | 1,339 | 503 | 141 | 156 |
| gemini37_multi_full | 1.1 | 3.1 | 11,233 | 4,741 | 454 | 375 | 114 |
| gemma31b_direct | 1.0 | 0.0 | 249 | 0 | 80 | 0 | 0 |
| gemma31b_baseline | 1.0 | 0.0 | 4,811 | 0 | 235 | 0 | 0 |
| gemma31b_multi_off | 1.0 | 1.1 | 10,357 | 1,359 | 230 | 165 | 0 |
| gemma31b_multi_full | 1.1 | 3.0 | 10,267 | 3,792 | 250 | 288 | 0 |
| qwen9b_direct | 1.0 | 0.0 | 267 | 0 | 6,033 | 0 | 5,699 |
| qwen9b_baseline | 1.0 | 0.0 | 5,179 | 0 | 4,823 | 0 | 4,508 |
| qwen9b_multi_off | 1.1 | 1.1 | 11,273 | 1,338 | 7,006 | 155 | 6,421 |
| qwen9b_multi_full | 1.3 | 3.2 | 11,709 | 5,087 | 8,299 | 374 | 7,405 |

Wall-clock per query, Gemini cells only (median seconds, reported once; not a sweep metric because the OpenRouter and Gemma runs hit rate limits and provider variance): gemini37_direct 2.1s, gemini37_baseline 2.1s, gemini37_multi_off 5.0s, gemini37_multi_full 8.3s.

## 5. What the answers look like

Patterns across the 1,200 answers, with the queries that show them. Figures
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

### Per-query notes (the note field of every answer label that has one)

#### gemini37_direct
- Q001: says 'late January, has passed' but never gives the date
- Q002: confident wrong range with a plausible rationale (varies by school)
- Q003: a range that happens to bracket the true 2,182,200, presented as an estimate
- Q004: omits ACT; the subject test it adds exists only in the graduate admissions calendar
- Q005: correct grade rule; adds a subject-combination rule the page does not state
- Q006: range excludes the double-occupancy figure; invents room types
- Q007: spring, typically March; no date
- Q008: wrong section count; a JSON claims block is appended to the spoken text
- Q009: right conclusion, invented rule
- Q010: bare model knows the code
- Q011: describes NOP well; deadline only as March 2026, passed
- Q012: right count, overstated coverage
- Q013: bare model gets the 20 to 100 percent range
- Q014: decision window right; classes hedged as late August or early September
- Q015: correct rule, invented timing detail
- Q017: deadline was 28 February
- Q018: correct; a JSON claims block is glued to the end of the Urdu text
- Q020: bare model knows the split; markdown bullets in text meant for speech
- Q021: detailed, confident and wrong; the kind of answer a student would trust
- Q022: plausible wrong figure
- Q023: right PKR fee, wrong USD figure
- Q024: January to February, no dates
- Q025: bare model matches the page on the deficiency-course condition
- Q027: substitutes a generic HEC schedule for the published one
- Q028: two of three campuses, states it as complete
- Q029: late July, no date
- Q030: rule right; the series-4 window closed 18 June
- Q031: plausible generic list, half of it not on the page
- Q032: invents a two-tier USD fee; the page has one flat figure
- Q034: describes a method instead of the published flat rate
- Q035: names two of six programmes and adds routes the page does not list
- Q037: overstates the HSSC minimum and quotes folk cut-offs
- Q038: whole reply is a JSON object with the Urdu answer under urdu_text; content correct
- Q039: 50 percent right, academic split not given
- Q040: typically June and July, no dates
- Q041: both yes/no facts right, then invents the penalty sizes
- Q042: bare model has the full pattern
- Q043: whole reply is a JSON object with the Urdu under response; misses the NU merit-list route and the GPA condition
- Q044: generic loan mechanics, none of the published terms
- Q045: confident wrong code
- Q046: says yes but attaches the NUST deficiency-course rule to FAST, whose rule is additional maths beforehand
- Q047: same generic HEC schedule it gave for NUST; both universities publish a different one
- Q049: process description matches the page closely
- Q050: bare model knows the 60/40 split
- Q053: PKR fee right, USD figures invented
- Q054: JSON block appended
- Q055: aid coverage right, deadlines wrong
- Q056: misses the mandatory external-scholarship rule
- Q058: installment description happens to match the page
- Q059: bare model has the pattern and the threshold; adds invented format detail
- Q060: plausible-sounding ranges throughout; the shape of the comparison is right, every number is not
- Q061: LUMS right, FAST HSSC minimum overstated
- Q062: whole reply is a JSON object with the Urdu under urdu_text; content right
- Q063: right conclusion, both ranges just miss the real figures
- Q064: bare model has both headline weights
- Q065: months only, no dates; structure of the two regimes described correctly
- Q066: third time the same invented HEC schedule appears
- Q067: JSON wrapper; invents a LUMS minimum the page says does not exist
- Q068: gives NUST figures no official page publishes
- Q069: broader list than the corpus can check; plausibly true, uncheckable here
- Q071: gets the true negative wrong
- Q073: long national list the corpus cannot check
- Q074: Roman Urdu reply with a ten-item list
- Q075: asks which universities, then extends HEC NBS to private universities
- Q076: bare model matches the HEC page closely
- Q078: confident yes to a question whose answer is no; blends HEC funding into LUMS aid
- Q079: income limit and the BPS 1-4 exemption both match the page
- Q080: knows the Benazir rename; stipend without the figure
- Q081: clear no, with a sensible redirect to the university's own aid office
- Q082: conflates two HEC schemes; core criteria right
- Q083: plausible generic policy, neither published rule stated
- Q085: HEC NBS is public-sector only; FAST is private
- Q086: JSON wrapper with the Urdu under a field named urdu
- Q087: detailed figures for a fact no official page publishes
- Q088: confident figures for a university not in the corpus
- Q090: precise-sounding price no ingested page carries
- Q093: the folk cut-off figures the plan anticipated; plausible, unpublished
- Q091: bare model matches the FAQ page
- Q094: knows NSHS exists, invents the fee
- Q095: detailed figures for an uncovered university
- Q096: plausible LUMS first-semester list from memory
- Q097: assesses the student's chances in detail: on the lower side, make or break
- Q098: flatly predicts failure with invented cut-offs
- Q099: says not certain and explains the weighting
- Q100: yes it is worth applying, plus an invented competitive score

#### gemini37_baseline
- Q001: states the passed deadline as if upcoming
- Q003: total only, no tuition breakdown
- Q004: right page retrieved but not the Test Requirements chunk; says the sources do not name the tests
- Q007: Important Dates chunk not retrieved for an SAT-phrased query
- Q011: lists both the 2023 and 2026 dates from the page without resolving which applies
- Q013: financial-aid page not retrieved; says part or full without the range
- Q016: adds the transfer-student exception
- Q017: gives the tab-opening date, says the deadline is not in its sources
- Q018: includes the counsellor exclusion
- Q024: registration window right, test dates omitted, stated as upcoming
- Q026: yes with seat categories, no merit formula
- Q029: stated as upcoming a month after it passed
- Q031: exhaustive, including migration rules nobody asked about
- Q035: programme list right, 60 percent rule omitted
- Q039: test weight only; the 40/10 split sits in a different chunk of the same page
- Q040: all dates stated as upcoming in late August
- Q044: terms right, never says interest-free
- Q045: only College Board pages retrieved, no FAST chunk
- Q048: answers BS and MS; weightage omitted
- Q050: one-line answer, coverage omitted
- Q052: long but complete
- Q054: fee breakdown and deadline right, deadline stated as upcoming
- Q055: substitutes the admission deadline and says aid details are not in its sources
- Q056: answers the marks question with SAT/ACT score minimums and says the percentage is not in its sources; cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q057: the 6,500 fee sits in a chunk the baseline retrieved for Q002 but not here
- Q058: gives the USD figure and says the PKR figure and installment rule are not in its sources; cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q060: derives the NUST first-year total from the table and states both bases
- Q061: pooled retrieval returned nine LUMS chunks and no FAST; says it cannot compare
- Q063: only LUMS chunks retrieved, and not the per-credit one
- Q064: no FAST chunk retrieved; says it cannot compare
- Q065: complete series list, but says Series 4 remains open in late August
- Q066: NUST only; no FAST chunk retrieved; says it cannot compare
- Q067: complete on both sides, including superscore and codes
- Q068: attribute-level abstention done right
- Q069: pooled retrieval returned only College Board pages
- Q070: no LUMS chunk retrieved, so LCAT missing
- Q071: NUST only; says nothing on others
- Q073: only NUST chunks retrieved for a NAT question
- Q075: provincial schemes attributed to the bodies, listed per university
- Q077: includes the Rs 6,000 figure
- Q078: states the public-sector rule and the absence, lists external donors but not LUMS's own aid
- Q079: answers the literal question, omits the other conditions on the same chunk
- Q080: no hedge that the page describes the 2021 intake
- Q082: complete list; says relevant intake sessions without flagging they are past
- Q083: both rules stated, tension left for the student to confirm: the model answer for this query
- Q084: letter right, month missing; cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q085: full list from the page
- Q086: financial-aid chunk not retrieved
- Q087: correct: says the figure is not given, points to the housing site
- Q088: entity filter returned no chunks; runner recorded no answer
- Q090: mentions fee waivers, as the gold allows
- Q092: names the campuses with hostels, which the page does list
- Q091: gold marked this not-in-corpus; the FAQ chunk has it, so the answer is supported
- Q094: says the MBBS amount is not specified; mentions the NSHS charges the page does list
- Q097: states the minimum, says it cannot predict, lists what the review covers
- Q098: formula, minimums, cannot predict, retake allowed
- Q100: eligible, formula, cannot predict

#### gemini37_multi_off
- Q003: gives semester totals, not the tuition line
- Q004: also flags the SAT/ACT deadlines as passed
- Q007: exam agent pulled SAT-body pages, not the LUMS dates chunk
- Q011: NOP deadline chunk not retrieved; says it does not have the date
- Q015: explains IBCC, asks which qualification the student has
- Q019: adds the LLB LAT condition
- Q022: adds the one-time and per-semester extras correctly
- Q023: adds SAT/ACT seat fees and the non-refundable rule
- Q026: adds the 550 minimum, code 2790 and two-year validity, all on the page
- Q034: adds the USD 40 international rate
- Q038: includes the university-specific NTS exclusion
- Q045: entity-scoped retrieval found the FAST paragraph the baseline missed
- Q048: asks BS or MS
- Q049: full payment-channel detail from the page
- Q052: two agents merged cleanly; asks background
- Q055: describes waiver and loan mechanics without the coverage range or the count
- Q056: HEC schemes correctly separated; external-scholarship rule softened to optional
- Q058: both fee columns right; says it has no installment information; cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q059: complete, with the SSC minimum and test alternatives
- Q061: per-entity retrieval fixed the baseline's miss
- Q063: has FAST, says it lacks the LUMS per-credit figure
- Q066: per-entity retrieval gives both; explicitly says the schedules match
- Q069: enumeration path works: three institutions with conditions and codes, plus the coverage note
- Q070: enumeration term entry test is not in the known-exams list, so the exams agent got only SAT pages; NET named as a passing example
- Q071: LUMS and NUST with conditions; coverage note implies FAST does not
- Q073: single positive plus coverage note; the true-negative case handled
- Q075: separates provincial schemes from institutional aid explicitly; the strongest answer to this query
- Q078: the cross-authority case handled exactly as designed
- Q079: PEEF is hosted on a NUST page but the question names no university, so the scoped agent pulled only HEC pages; the pooled baseline found it
- Q083: NUST rule only; the PEEF exclusivity rule never surfaces
- Q089: offers the three covered universities instead
- Q093: honest not-published, plus the merit formula and minimums the student can act on
- Q096: gives the credit-hour structure the page does have
- Q097: asks for the Matric percentage, the missing half of the rule
- Q099: minimum, weighting and cut-off rule all present

#### gemini37_multi_full
- Q003: full fee table reproduced correctly
- Q005: asks which A-Level subjects the student takes; struck IBCC claim is arguably true for A-Level applicants
- Q017: substitutes the admission deadline for the aid deadline while saying it lacks the latter
- Q022: clean Urdu, complete
- Q037: adds the merit weightage, correctly
- Q039: struck a true MS-programme claim; the 40/10 split was never generated; policy gate then flagged the 50 as reintroduced
- Q043: complete and in clean Urdu
- Q054: a number_readings list leaked into the spoken text
- Q056: struck claim about deferment schemes was plausibly true (installment facility exists)
- Q058: cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q059: conflict detector fired between two FAST chunks and the answer tells the student the sources differ when they do not
- Q060: each figure carries its basis; no derived total
- Q061: true O-Level claim struck; comparison intact
- Q063: has LUMS, says it lacks the FAST figure: the mirror image of the verify-off run on the same query
- Q064: adds the comparative sentence
- Q066: agreement case handled without a false conflict
- Q067: verifier struck the true NUST minimum-25 claim; answer says it lacks the NUST minimum; question asked twice
- Q070: same enumeration miss
- Q072: true MS-programmes claim struck; BS list intact; asks BS or MS twice
- Q075: five strikes, four of them on true claims (PEEF at all three, Honhaar, HEC stipend, BISP 40,000); Honhaar lost
- Q076: adds the self-finance exclusion
- Q079: same scoping miss
- Q080: no staleness hedge despite the cycle-bound Ehsaas page
- Q081: staleness hedge present in Urdu
- Q082: complete and hedged
- Q083: conflict detector did not fire on the one query designed for it
- Q087: verifier struck the true directs-to-portal claim, so the student gets a blunt refusal instead of the pointer; right outcome, poorer answer
- Q093: verifier struck the true weightage claim and the agent refused; abstention preserved, the useful context lost
- Q098: policy gate flagged improve your chances, which is advice not prediction
- Q100: policy gate flagged an encouragement sentence; no outcome assessed

#### gemma31b_direct
- Q001: only 'has already passed', no date
- Q002: flat wrong figure, no hedge
- Q003: range excludes the true figure
- Q004: Roman Urdu, omits ACT, and a stray empty JSON block appended
- Q005: denies a published rule exists
- Q006: declines rather than guessing
- Q007: only has passed, no date
- Q008: describes a law-school test; JSON block appended
- Q009: wrong answer to a yes/no question
- Q010: confident wrong code
- Q011: no date, only that it has passed
- Q012: dodges the published count of 100
- Q013: up to 100 percent, no range; JSON block appended
- Q017: clean refusal to guess
- Q018: leaked number_readings line
- Q019: gives 60 percent for HSSC only, drops the SSC half
- Q020: collapses the 15/10 split into 25 for HSSC
- Q022: declines; JSON block appended
- Q025: calls it a bridge course; same substance
- Q026: wrong answer to a yes/no question; JSON block appended
- Q027: describes the shape of a policy without a single figure
- Q028: invents a school, misses Quetta
- Q029: only that it has passed
- Q030: JSON block appended
- Q031: generic eligibility talk, not the published list
- Q034: JSON block appended
- Q035: steers a DAE student away from the six BE programmes the page lists for them
- Q038: misses NTS; JSON block appended
- Q040: only that the cycle has concluded
- Q041: wrong on the yes/no that matters most to a test-taker
- Q043: generic scholarship talk; JSON block appended
- Q044: says nothing concrete
- Q046: same NUST-rule transplant as the Gemini direct arm
- Q047: gets the one non-negotiable fact backwards
- Q048: JSON block appended
- Q051: leaked number_readings line
- Q052: no thresholds, no fee
- Q053: NET only, wrong fee
- Q054: JSON block appended
- Q056: HSSC-only threshold, generic aid list
- Q058: no figure; JSON block appended
- Q059: threshold right, pattern described without figures
- Q060: direction right, no figures
- Q062: JSON block appended
- Q063: conclusion only
- Q065: only that both have passed
- Q066: vague tiers and a wrong admission-fee rule
- Q067: no figures at all; JSON block appended
- Q068: no figures
- Q070: omits LUMS
- Q071: JSON block appended
- Q072: bare model has the exact list
- Q075: asks which universities and stops; defensible without context, scored as a refusal
- Q076: generic document list; never says HEC refuses direct applications
- Q077: JSON block appended
- Q078: yes, with an invented HEC link
- Q081: tells a private-university student they may qualify; JSON block appended
- Q082: generic but not wrong
- Q085: generic; the loans and named schemes are absent
- Q086: grace period without a duration; JSON block appended
- Q087: bare model declines cleanly
- Q088: bare model declines cleanly
- Q090: JSON block appended
- Q093: bare model correctly says no fixed cut-off is published
- Q094: the fee page itself refers to the NSHS MBBS programme
- Q095: JSON block appended
- Q097: on the lower side, application remains viable
- Q099: JSON block appended
- Q100: yes it is worth applying: assesses chances, albeit gently

#### gemma31b_baseline
- Q001: names Psychology and Biology programmes for a generic question (chunk came from those pages); stated as upcoming
- Q004: Roman Urdu; Test Requirements chunk not retrieved
- Q008: Roman Urdu
- Q011: Roman Urdu reply; correct date stated as past
- Q020: gives 75 percent only, drops the academic split though the chunk has it
- Q026: Roman Urdu; rich detail on the SAT form, deadline and NET exemption
- Q033: API 429 rate-limit error, no answer produced (infrastructure, not a model decision)
- Q034: retrieval returned no chunks at all (embedding call failed); no answer produced
- Q039: recites MS programme weights instead of the academic split
- Q042: retrieval returned no chunks (embedding call failed); no answer produced
- Q043: Urdu script carrying mostly English nouns; a stutter (student student) repeated
- Q054: Devanagari punctuation marks in Urdu text
- Q055: validity rules only
- Q056: same SAT/ACT confusion as the Gemini baseline; aid part complete
- Q058: cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q060: answers LUMS only and never mentions NUST though its chunks were retrieved: the silent-omission failure the comparison design targets
- Q061: LUMS only, no FAST chunk
- Q064: NUST split plus NET subject weights; no FAST
- Q065: names unrelated LUMS programmes from the retrieved pages
- Q066: NUST only
- Q067: exhaustive Roman Urdu
- Q068: ties the LUMS fee to an unrelated joint-major page
- Q071: one sentence, NUST only
- Q074: opens in Roman Urdu, continues in English
- Q076: also gives the payment-through-university rule
- Q078: absence noted but the public-sector rule never stated
- Q082: names the Fall 2021 and 2020 sessions, which signals the page's age
- Q083: flat no from the PEEF rule alone; NUST's mandatory-application rule omitted
- Q084: says the sources do not contain it; the chunk sits on a page it retrieved
- Q088: no chunks, no answer
- Q099: API 504 timeout, no answer produced

#### gemma31b_multi_off
- Q003: answered in Roman Urdu, mirroring the input
- Q004: mostly Roman Urdu with a few Urdu-script words
- Q008: clean Urdu; omits the calculator rule
- Q010: Roman Urdu reply to a Roman Urdu question
- Q011: picked the stale 2023 date off the page instead of March 10, 2026
- Q016: Roman Urdu reply
- Q020: Roman Urdu reply
- Q024: Roman Urdu reply
- Q026: Roman Urdu
- Q028: Roman Urdu reply
- Q032: cost agent raised an exception (I ran into a problem looking up the funding details); the chunks were retrieved
- Q035: Roman Urdu reply
- Q041: Roman Urdu reply
- Q045: Roman Urdu reply
- Q049: Roman Urdu reply
- Q050: Roman Urdu reply
- Q055: LUMS page retrieved but neither the dates nor the figures made it into the answer; cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q058: cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q060: cost agent exception, no answer
- Q063: cost agent exception
- Q065: NUST reduced to the SAT/ACT route only
- Q066: cost agent exception
- Q068: cost agent exception
- Q071: Roman Urdu body with the Urdu-script coverage note appended
- Q075: cost agent exception
- Q077: one line of Roman Urdu
- Q078: rule right, LUMS's own aid omitted
- Q081: leaks [Source 4] into the Urdu text
- Q082: cost agent exception
- Q083: says it has no information though the NUST aid page was in the pool
- Q096: Roman Urdu reply

#### gemma31b_multi_full
- Q003: Roman Urdu answer; adds hostel fees unasked
- Q004: Roman Urdu (policy gate flagged it); the two struck deadline claims were TRUE, an over-strike on tense (was vs is)
- Q008: Urdu script but half the words are English or Roman (taqreeban)
- Q010: Roman Urdu reply
- Q011: the struck claim carried the March 10, 2026 date and the source does mention it: over-strike cost the deadline
- Q013: identical to the verify-off answer
- Q016: Roman Urdu reply
- Q017: struck the true November 11 tab date as overreach; over-strike, though not the missing fact
- Q020: one-line Roman Urdu answer, split omitted
- Q022: Roman Urdu to an Urdu-script question; policy gate flagged it
- Q024: Roman Urdu reply
- Q026: policy gate flagged 'so your chances improve' as a prediction; borderline generic advice
- Q028: Roman Urdu reply
- Q029: verify-full answer states the passed date as upcoming while verify-off flagged it
- Q032: same agent exception as verify-off
- Q034: Roman Urdu, flagged by the policy gate
- Q035: Roman Urdu reply
- Q041: Roman Urdu reply
- Q042: verifier struck the fully correct pattern claim as unsupported and the agent refused: over-strike into false abstention
- Q045: Roman Urdu reply
- Q047: verifier struck the true 100-percent-to-day-10 claim as overreach; the schedule now starts at 80 percent
- Q049: Roman Urdu reply
- Q050: Roman Urdu reply
- Q052: leads with the IBCC clause, buries the thresholds
- Q053: verifier struck the true accepted-tests claim, so the answer denies knowing the tests while pricing them
- Q054: Roman Urdu, flagged by the policy gate
- Q058: cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q059: verifier struck the correct pattern claim as overreach; answer says it lacks the breakdown
- Q060: cost agent exception, no answer
- Q061: verifier struck the correct FAST claim; half the comparison gone
- Q062: Roman Urdu, flagged by the policy gate
- Q063: cost agent exception
- Q065: verifier struck the correct NUST deadlines claim; answer says it lacks NUST dates
- Q066: cost agent exception
- Q067: Roman Urdu with empty citation brackets [, 8] left in the spoken text
- Q068: cost agent exception
- Q075: cost agent exception
- Q077: Roman Urdu, flagged by the policy gate
- Q081: Roman Urdu sentence plus the Urdu hedge; policy gate flagged the flat you are not eligible wording
- Q082: Roman Urdu reply to a Roman Urdu question, hedged
- Q084: Roman Urdu reply
- Q093: struck the true determined-by-the-university claim; blunt refusal
- Q095: Roman Urdu, flagged by the policy gate
- Q098: true NET-weights claim struck

#### qwen9b_direct
- Q001: invents a May deadline
- Q002: hedged but still states a wrong figure
- Q003: hedged range that brackets the true tuition
- Q004: incoherent Urdu with Chinese characters, leaked number_readings blocks, invented test name
- Q005: invented grade threshold
- Q006: hedged, still a wrong range
- Q008: Roman Urdu with garbled Urdu number readings inline and a greeting
- Q009: wrong answer to a yes/no question
- Q010: wrong code, hedged
- Q011: vague on what NOP covers; early 2026, passed
- Q012: invented tiers and duration
- Q013: garbled Roman Urdu, leaked number_readings
- Q014: invents a Spring intake LUMS does not have
- Q015: flatly wrong
- Q016: invents a Spring intake
- Q018: greeting, leaked number_readings, garbled final sentence
- Q019: wrong threshold
- Q020: corrects the student wrongly, then invents a split
- Q021: denies the test exists and redirects to ECAT
- Q022: garbled Urdu refusal with a JSON block in the middle
- Q023: a range and a guess
- Q025: misses the deficiency-maths condition
- Q026: wrong; JSON block appended
- Q028: misses Quetta
- Q030: barely coherent Roman Urdu
- Q031: names the wrong tests and an age limit that does not exist on the page
- Q034: garbled Urdu with Portuguese and Chinese fragments; sends the student to FAST's domain for NUST fees
- Q035: two of six programmes, no percentage rule
- Q036: double the real rate
- Q037: ECAT is not a FAST test
- Q038: garbled Urdu, JSON leak, a nonsense sentence about the method having ended
- Q042: describes an ECAT-style paper
- Q043: garbled Urdu with Latin fragments; invents a marks-based tier system
- Q044: every term wrong, including charging interest on an interest-free loan
- Q046: would turn an eligible student away
- Q047: generic; only the non-refundable admission fee lands
- Q048: incoherent Urdu with an English paragraph in the middle and leaked number readings
- Q051: leaked Numbers used block
- Q052: both facts wrong, fee off by more than half
- Q053: names a test NUST does not use
- Q054: long, garbled, and wrong on the deadline
- Q055: understates the aid programme that is LUMS's main support route
- Q058: garbled Urdu, leaked number reading
- Q060: LUMS understated by half
- Q061: invents the test regime of both universities
- Q062: invents a test name; leaked number readings
- Q066: invents a contrast between two identical policies
- Q067: the model's English chain-of-thought about Urdu number words leaked into the answer for several paragraphs; content wrong throughout
- Q069: answers about US, Canadian and UK universities; never realises the question is about Pakistan
- Q070: omits NUST, includes a Bangladeshi university
- Q071: Roman Urdu, a markdown table of wrong Urdu number words, Cyrillic characters
- Q072: misses Computer Science itself; AI and Cyber Security demoted to tracks
- Q073: invents test names and gets FAST backwards
- Q075: asks which universities, gives nothing specific
- Q076: vague; through the university portal
- Q077: garbled Urdu, drifts into deadlines nobody asked about, leaked number readings
- Q078: right conclusion by accident, invents an HEC merit scheme at LUMS
- Q079: denies the fund exists
- Q081: denies the scheme exists; refers to a Praise University
- Q082: calls it the Ehsaas Degree Scholarship; public-sector condition missing
- Q083: the opposite of the published rule, which requires applying to PEEF
- Q085: denies the aid that exists
- Q086: garbled Urdu with Chinese characters and leaked number readings
- Q088: hedged, still a figure
- Q090: garbled Urdu, JSON block and an English note about number readings
- Q095: long garbled Urdu with a figure
- Q098: does not predict, but denies the test exists
- Q099: garbled; wrong test
- Q100: wrong minimum; hedges on worth

#### qwen9b_baseline
- Q003: every sentence starts with At LUMS; adds processing fee unasked
- Q004: Roman Urdu with a greeting; chunk not retrieved
- Q005: adds the June 2026 resit rule, which the page does state
- Q006: ties the fee to two unrelated programmes whose pages it happened to retrieve
- Q007: greeting, LUCS typo, offers help with score reporting
- Q008: four sections and timing right, but the format sentence is garbled Urdu and the last sentence is nonsense
- Q011: dumps 2023, 2026 and the January admission deadline together
- Q012: six sentences each starting At LUMS
- Q015: near-verbatim from the page
- Q017: every sentence starts At LUMS
- Q018: slightly garbled Urdu, facts intact
- Q026: form date, score deadline and fee all correct
- Q028: leaks According to Source 3 into the spoken answer
- Q030: garbled Urdu around correct facts; mentions 2022 dates from the page
- Q031: repeats the migration rule twice
- Q029: right date stated as upcoming; institutional codes correct; invents later programme dates
- Q034: Roman Urdu to an Urdu-script question
- Q037: Roman Urdu with garbled words, facts correct
- Q040: every sentence begins At FAST universities pursued for Fall 2026; HAT is a graduate test
- Q043: garbled Urdu, facts intact
- Q048: 60 percent stated without naming SSC and HSSC
- Q053: every sentence starts At NUST
- Q056: garbled Roman Urdu; leads with the GPA rule, never gives the marks
- Q058: Roman Urdu; USD figure and repeat fee instead of the asked tuition; cause relabelled: the answer says the fact is absent from its sources, so the chunk was not retrieved
- Q059: leaks (Source 10) and (Source 12) markers into the spoken answer
- Q060: terse but complete, bases stated
- Q061: garbled Roman Urdu with a Portuguese word and leaked (Source 1, Source 3) markers
- Q062: Roman Urdu with Chinese characters and leaked (Source 1, Source 10) markers
- Q065: exhaustive date dump with [Sources 1, 4] markers in every sentence
- Q066: NUST only, says so plainly
- Q067: garbled Urdu with markdown bold; codes, minimum and weight all present
- Q071: NUST only, garbled bullet list
- Q074: reads Balochistan Campus and Quetta as two places
- Q082: complete but every clause carries a per Source N marker
- Q084: greeting, then the NFAAF deadline instead
- Q088: no chunks, no answer
- Q090: greeting; drags in a liability clause from the SAT terms
- Q093: leaks Source 15
- Q094: says MBBS is not stated, then recites the other tuition columns
- Q095: abstains on UET, then volunteers the LUMS EE fee
- Q099: answers a Matric student with master's-level criteria in near-nonsense Urdu
- Q100: opens with sympathy, repeats the 60 percent rule four ways

#### qwen9b_multi_off
- Q002: correct, then volunteers the passed fee deadline and financial-aid access rule unasked
- Q003: total correct, then an unasked scholarship detour and a self-finance question asked twice in one sentence
- Q004: correct tests; one garbled sentence and two odd clarifying questions
- Q007: fills the gap with College Board test dates and LCAT rules
- Q008: correct; several garbled transliterations (multiple-choice, LUMS)
- Q010: adds that the August SAT date has passed
- Q013: answered in Hindi (Devanagari script) to an Urdu question, with an Indonesian word at the end
- Q014: adds that the window has passed
- Q018: garbled opening clause and a nonsense closing sentence around a correct fact
- Q019: leads with the Part-I detail before the headline rule
- Q020: adds the NET subject weights, correctly
- Q021: adds the pre-medical condensed maths condition
- Q022: spells out the full university name in Urdu
- Q023: asks nationality, a reasonable clarification
- Q024: adds Series 4 as the last 2026 window; UST typo
- Q026: garbled Urdu phrases; asks national or international seat twice
- Q027: schedule right, then speculates
- Q030: an Indonesian word mid-sentence; asks BS or Masters twice
- Q032: complete USD table including the HBS licensing fee
- Q034: asks nationality
- Q035: same question printed twice
- Q036: correct fee, then an HEC self-finance rule presented as FAST policy and an irrelevant clarifying question
- Q037: drops the Mathematics condition
- Q038: Roman Urdu; adds weightage and SAT format, both on the pages
- Q039: also gives the engineering-group 33 percent, correctly
- Q041: adds that the SAT permits calculators, which the SAT pages say
- Q042: asks which test the student will sit, a sensible clarification
- Q043: garbled but complete
- Q047: skips the 80/60/50 steps
- Q048: HSSC only, SSC dropped, weightage absent
- Q052: names HEC as the body behind the self-finance exclusion; ends with three clarifying questions
- Q053: adds test centres and retake rules
- Q054: irrelevant self-finance question
- Q055: HEC schemes correctly attributed; asks for self-finance documents
- Q056: opens with unprompted sympathy the persona forbids; Roman Urdu question at the end
- Q057: fills the gap with the test format instead
- Q058: garbled Urdu; a raw field name admission_seat_type leaked at the end
- Q059: negative marking omitted
- Q060: figures right; then pins the HEC self-finance rule on NUST and asks the same question twice
- Q061: bullet list in Roman Urdu with one garbled line; comparison complete
- Q062: short and correct
- Q063: FAST figure and extras; says it did not find LUMS
- Q066: gets the schedules and then invents a difference in day-counting that the pages do not have
- Q067: ends in Russian and code tokens (меню .matcher)
- Q068: offers the NOP as a route to covered accommodation; question asked twice
- Q069: adds passed deadlines; same question asked twice
- Q070: garbled Roman Urdu refusal
- Q071: garbled Urdu with Romanian and Arabic fragments; deadline off by three days
- Q072: lists MS first, BS second
- Q074: same Balochistan-and-Quetta confusion
- Q075: PEEF across all three; Honhaar missing; a stray sentence says the information is not yet available
- Q076: question asked twice
- Q077: garbled second and third sentences
- Q078: describes both schemes, never answers yes or no, leaks (Source 1, 5) markers, asks admission category
- Q079: offers HEC schemes instead; question asked twice
- Q080: self-finance question repeated as a broken fragment
- Q082: answers coverage and application steps instead of eligibility; leaks Source markers; flags Fall 2021 as past
- Q083: process description only; leaks the raw token seat_type
- Q085: FAST schemes right, then attaches public-sector HEC schemes to a private university and says our university
- Q087: pads the abstention with unrelated fees
- Q088: pivots to HEC schemes and asks public or private
- Q089: question asked twice
- Q094: admits no national MBBS figure but offers the all-programmes USD rate as if it applied; question asked twice
- Q095: abstains, drifts into HEC schemes, asks self-finance twice
- Q097: question asked twice
- Q098: pivots to the 60 percent rule; question asked twice
- Q099: boundary respected, weighting wrong
- Q100: asks whether 61 is Part-I or the final result, a sensible clarification

#### qwen9b_multi_full
- Q002: correct fee, then drifts into HEC self-finance eligibility and asks an irrelevant clarifying question; markdown bold in text meant for TTS
- Q003: the struck breakdown claim was TRUE (176,600 / 1,557,200 / 269,200 / 112,200 match the table): over-strike removed the tuition line
- Q004: Roman Urdu despite Urdu request; the struck claim was Arabic gibberish, correctly removed
- Q005: speculates classes started (page says September); same clarifying question asked twice
- Q007: same clarifying question printed twice
- Q008: correct; LCAT rendered as a garbled acronym
- Q011: most complete NOP answer (80% eligibility, summer session); the struck stipend claim was TRUE, an over-strike that did not cost a checklist fact
- Q015: opens with the passed deadline nobody asked about
- Q016: asks freshman or transfer, a sensible clarification
- Q013: garbled Roman Urdu; leaked (Sources 4, 10, 13) markers into spoken text; HEC stipend and LUMS loan terms it adds are on the pages; asks self-finance question
- Q018: verifier struck the correct two-evaluations claim as overreach and the agent refused outright: over-strike producing a false abstention, in English to an Urdu question
- Q021: verifier struck the correct 50/30/20 claim as unsupported and the answer lost the whole point; policy gate also misfired a prediction flag on a requirement sentence
- Q022: correct fee, then Chinese characters, an irrelevant merit/self-finance question, and an HEC rule presented as if it were NUST policy
- Q025: clarifying question printed twice
- Q026: Chinese characters mid-sentence, garbled transliterations, facts otherwise right
- Q027: schedule right; garbles the three-year security deposit rule and adds two inventions
- Q031: struck migration claim is on the page (over-strike, not a checklist fact); correctly expands IBCC
- Q034: garbled phrasing, correct figure
- Q035: same question printed twice
- Q036: verifier struck the correct Rs 12,000 claim as unsupported and the agent refused outright: over-strike into false abstention
- Q037: frames the two minimums as alternatives by qualification rather than both applying; Mathematics dropped
- Q038: severely garbled Roman Urdu (AST akademi darje...); facts survive inside it
- Q039: verifier struck the correct 40/10 claim as unsupported; over-strike cost two facts
- Q042: complete, with the merit weightage added
- Q043: three sentences, the last one meaningless; coverage and GPA lost
- Q045: echoes the student's question back verbatim at the end
- Q046: says our programmes as if it were FAST; question asked twice
- Q047: complete, with the admission and security amounts
- Q048: garbled Roman Urdu; folds BS and MS into one sentence
- Q051: Roman Urdu, flagged by the policy gate
- Q052: struck the true 70 percent claim, then the rewrite reintroduced it and the policy gate flagged the figure; HEC correctly separated from LUMS
- Q054: Roman Urdu body with an Urdu-script question at the end; HEC rule correctly attributed
- Q055: only cell to give both aid facts; dates still missing; struck NOP claim was true; three clarifying questions
- Q056: complete; names the Balochistan fund as Punjab's
- Q057: two true SAT/ACT deadline claims struck as overreach
- Q058: fee right, installment rule inverted
- Q059: three of four sections; English dropped
- Q060: starts with A Aria, gives LUMS, drops NUST entirely, asks about self-finance
- Q061: verifier struck both correct LUMS claims; answer says it cannot confirm LUMS rules
- Q062: answers the question in the first paragraph, then dumps a garbled and partly wrong table of both universities
- Q063: unprompted empathy, leaked (, 28) and (, 11) markers, HEC exclusions correctly attributed, self-finance question asked twice
- Q064: both full splits, cleanly
- Q066: implies NUST lacks the 80/60 steps after the true weekday-counting claim was struck
- Q067: true NUST minimum struck; Roman Urdu with a trailing line of Urdu garbage
- Q068: verifier struck the correct LUMS hostel figures as unsupported; the one answerable half vanished
- Q069: SAT is one of three options at LUMS, not required; question asked twice
- Q071: heavily garbled; ends with a meaningless question
- Q074: NUST claim overreached by adding PNEC and was struck whole, so NUST vanished instead of being trimmed
- Q075: complete, HEC rules attributed to HEC, but turns the Rs 40,000 stipend into a tuition cap; self-finance question asked twice
- Q076: question asked twice
- Q078: true LUMS-aid claim struck; the alternative the student needs is gone
- Q080: only cell to hedge that the figures are from a past cycle
- Q081: barely coherent Urdu; the rule and the hedge survive
- Q082: financial-need and exclusion conditions dropped
- Q085: same HEC misattribution; self-finance question twice
- Q087: gives the campus-life URL from the page
- Q089: question asked twice
- Q090: abstains, then drifts into HEC scholarships and ends with an English question
- Q094: same USD slip; the two struck tuition claims were true but irrelevant here
- Q095: abstains, drifts into HEC schemes
- Q096: abstains, then wanders through fees and electives
- Q097: verifier struck the true 70 percent claim, so the one fact the gold allows is gone; boundary still respected
- Q099: policy gate flagged your marks are good; not an outcome prediction; question asked twice
- Q100: question asked twice

## 6. Statistical comparison

Every cell answered the same queries, so arms are compared query by query. **Paired bootstrap** (correctness): for each answerable query take the difference in checklist score between two arms; the effect is the mean of those 87 differences; its 95% interval comes from redrawing 87 queries with replacement 10,000 times and taking the 2.5th and 97.5th percentiles of the redrawn means. The p-value is a sign-flip permutation test on the same differences. **McNemar exact test** (yes/no outcomes): queries where both arms agree are dropped; among the rest, under no difference each is a fair coin flip, so the p-value is the binomial probability of a split at least this lopsided. p-values are Holm-adjusted within each table. No cell was re-run; only the stored labels are reused (seed 7).

### 6a. Correctness: mean paired difference (second arm minus first), 87 answerable queries

| contrast | model | arms | difference | 95% interval | queries better / worse | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| retrieval | gemini | direct → baseline | +23.9 pts | +13.7 to +34.3 | 37 / 11 | 0.0001 | 0.0009 |
| retrieval | gemma | direct → baseline | +42.4 pts | +31.9 to +52.6 | 55 / 8 | 0.0001 | 0.0009 |
| retrieval | qwen | direct → baseline | +59.9 pts | +50.3 to +69.4 | 65 / 4 | 0.0001 | 0.0009 |
| architecture | gemini | baseline → multi_off | +11.0 pts | +4.7 to +17.7 | 16 / 4 | 0.0016 | 0.0096 |
| architecture | gemma | baseline → multi_off | +8.0 pts | -1.1 to +17.1 | 17 / 9 | 0.0884 | 0.4420 |
| architecture | qwen | baseline → multi_off | +6.8 pts | -1.0 to +14.9 | 16 / 11 | 0.1059 | 0.4420 |
| verifier | gemini | multi_off → multi_full | -1.0 pts | -3.8 to +1.7 | 3 / 4 | 0.6020 | 0.6020 |
| verifier | gemma | multi_off → multi_full | -3.2 pts | -7.3 to +0.9 | 1 / 7 | 0.1733 | 0.5198 |
| verifier | qwen | multi_off → multi_full | -3.5 pts | -10.0 to +2.3 | 9 / 10 | 0.2820 | 0.5639 |

Pooled over the three models (per-query difference averaged across models, then bootstrapped over queries):

| contrast | arms | difference | 95% interval | p |
|---|---|---|---|---|
| retrieval | direct → baseline | +42.1 pts | +32.9 to +50.6 | 0.0001 |
| architecture | baseline → multi_off | +8.6 pts | +1.9 to +15.6 | 0.0169 |
| verifier | multi_off → multi_full | -2.6 pts | -5.3 to +0.2 | 0.0774 |

### 6b. Answer contains a wrong statement (all 100 queries)

| contrast | model | arms | count first → second | only first arm | only second arm | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| retrieval | gemini | direct → baseline | 35 → 0 | 35 | 0 | 0.0000 | 0.0000 |
| retrieval | gemma | direct → baseline | 31 → 0 | 31 | 0 | 0.0000 | 0.0000 |
| retrieval | qwen | direct → baseline | 54 → 2 | 53 | 1 | 0.0000 | 0.0000 |
| architecture | gemini | baseline → multi_off | 0 → 0 | 0 | 0 | 1.0000 | 1.0000 |
| architecture | gemma | baseline → multi_off | 0 → 1 | 0 | 1 | 1.0000 | 1.0000 |
| architecture | qwen | baseline → multi_off | 2 → 6 | 1 | 5 | 0.2188 | 1.0000 |
| verifier | gemini | multi_off → multi_full | 0 → 0 | 0 | 0 | 1.0000 | 1.0000 |
| verifier | gemma | multi_off → multi_full | 1 → 0 | 1 | 0 | 1.0000 | 1.0000 |
| verifier | qwen | multi_off → multi_full | 6 → 7 | 5 | 6 | 1.0000 | 1.0000 |

### 6c. Abstained on an answerable query (87 queries, RAG arms)

| contrast | model | arms | count first → second | only first arm | only second arm | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| architecture | gemini | baseline → multi_off | 9 → 5 | 6 | 2 | 0.2891 | 1.0000 |
| architecture | gemma | baseline → multi_off | 13 → 13 | 9 | 9 | 1.0000 | 1.0000 |
| architecture | qwen | baseline → multi_off | 10 → 5 | 7 | 2 | 0.1797 | 0.8984 |
| verifier | gemini | multi_off → multi_full | 5 → 5 | 0 | 0 | 1.0000 | 1.0000 |
| verifier | gemma | multi_off → multi_full | 13 → 13 | 1 | 1 | 1.0000 | 1.0000 |
| verifier | qwen | multi_off → multi_full | 5 → 9 | 0 | 4 | 0.1250 | 0.7500 |

## 7. CSV of the per-cell figures

```csv
cell,model,arm,n_labelled,correctness,wrong_rate,unverifiable_rate,misattributed,correct_abstain,false_abstain,unans_answered_unverifiable,faithfulness_kept,faithfulness_pre,strike_precision,over_striking,n_struck,truthfulness,boundary_ok,predicts_any,ur_script,ur_roman,ur_en,ur_mixed,ur_other,stale_ok,answer_calls_per_q,helper_calls_per_q,answer_prompt_tok_per_q,helper_prompt_tok_per_q,answer_output_tok_per_q,helper_output_tok_per_q,answer_reasoning_tok_per_q,runner_hard_abstain
gemini37_direct,gemini,direct,100,0.5239463601532567,0.35,0.56,3,,,9,,,,,0,-0.16666666666666666,1,3,23,0,0,0,0,0.7692307692307693,1,0,249.43,0,291.95,0,148.83,0
gemini37_baseline,gemini,baseline,100,0.7634099616858238,0.0,0.0,0,1.0,0.10344827586206896,,1.0,,,,0,0.7864583333333334,4,0,23,0,0,0,0,0.0,0.99,0,5125.94,0,321.8,0,53.47,1
gemini37_multi_off,gemini,multi_off,100,0.8735632183908046,0.0,0.0,0,1.0,0.05747126436781609,,1.0,,,,0,0.8802083333333334,4,0,23,0,0,0,0,0.6666666666666666,1.06,1.06,11363.02,1339.48,503.37,141.23,156.11,0
gemini37_multi_full,gemini,multi_full,100,0.8639846743295019,0.0,0.0,0,1.0,0.05747126436781609,,1.0,1.0,0.0,1.0,13,0.8802083333333334,4,0,23,0,0,0,0,0.8888888888888888,1.13,3.14,11232.86,4741.31,454.47,374.79,114.38,2
gemma31b_direct,gemma,direct,100,0.2653256704980843,0.31,0.28,1,,,3,,,,,0,-0.20833333333333334,1,3,7,16,0,0,0,0.7,1,0,249.43,0,80.43,0,0,0
gemma31b_baseline,gemma,baseline,100,0.6896551724137931,0.0,0.0,0,1.0,0.14942528735632185,,0.9857142857142858,,,,0,0.7239583333333334,4,0,7,13,0,1,0,0.2,0.97,0,4811.17,0,235.38,0,0,3
gemma31b_multi_off,gemma,multi_off,100,0.7701149425287356,0.01,0.0,0,1.0,0.14942528735632185,,1.0,,,,0,0.7760416666666666,4,0,13,7,0,3,0,0.8,1.02,1.08,10357,1358.75,229.54,165.29,0,7
gemma31b_multi_full,gemma,multi_full,100,0.7385057471264368,0.0,0.0,0,1.0,0.14942528735632185,,1.0,1.0,0.0,1.0,13,0.7604166666666666,4,0,10,8,0,5,0,0.7777777777777778,1.08,2.97,10266.75,3792.11,249.8,287.74,0,8
qwen9b_direct,qwen,direct,100,0.15689655172413794,0.54,0.37,2,,,8,,,,,0,-0.5729166666666666,3,1,17,3,0,3,0,0.625,1,0,267.06,0,6032.81,0,5699.31,0
qwen9b_baseline,qwen,baseline,100,0.7557471264367817,0.02,0.01,0,1.0,0.11494252873563218,,0.967391304347826,,,,0,0.7604166666666666,4,0,15,8,0,0,0,0.09090909090909091,0.99,0,5179.13,0,4822.56,0,4507.53,1
qwen9b_multi_off,qwen,multi_off,100,0.8237547892720306,0.06,0.04,3,0.8888888888888888,0.05747126436781609,,0.9907407407407407,,,,0,0.7135416666666666,4,0,20,2,0,0,1,0.7777777777777778,1.06,1.06,11273.46,1337.96,7006.34,154.56,6421.46,0
qwen9b_multi_full,qwen,multi_full,100,0.7883141762452107,0.07,0.05,3,0.8888888888888888,0.10344827586206896,,0.970873786407767,0.9642857142857143,0.23333333333333334,0.7666666666666667,30,0.6458333333333334,4,0,14,5,1,3,0,1.0,1.28,3.17,11708.93,5087.26,8299.38,374.02,7404.91,2
```
