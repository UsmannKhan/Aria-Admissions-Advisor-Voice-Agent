# Aria: a voice-first admissions advisor for Pakistani applicants

Final year project, BSc Computer Science, University of London.

Aria answers undergraduate admissions questions about LUMS, NUST, FAST, SAT exams plus
the HEC Need-Based and Ehsaas scholarship schemes, in English, Urdu or the
mix of the two that people actually speak. Every answer is built from official
pages that were crawled into a local knowledge base, and every claim in an
answer points back to the page and section it came from. If the pages do not
have the answer, Aria says so instead of guessing.

The research question behind it: does a multi-agent retrieval system with a
verification stage give more truthful answers than a plain single-agent RAG
baseline, on questions that cut across several institutions and authorities?
The evaluation in `eval_results.md` answers that over 100 fixed queries, three
answer models and four system configurations. Retrieval removes nearly all
wrong answers, per-institution multi-agent retrieval adds 6 to 11 points of
correctness on top, and the verifier as built did not help.

## How a question moves through the system

1. The browser records the question and sends it to the server. Whisper
   large-v3 transcribes it. The language toggle is passed along because
   Whisper's own detection tends to read Urdu as Hindi.
2. A planner model (Gemini 3.5 Flash-Lite) rewrites the transcript into clean
   English, repairs garbled names against a catalogue of known institutions and
   programmes, splits the question by domain (admissions, cost, exams), and
   notes anything the student has said about themselves. A deterministic alias
   matcher cross-checks the institutions it picked.
3. One agent per domain retrieves from ChromaDB, scoped to the institutions
   named. Comparisons retrieve each institution separately so the one with
   more pages cannot crowd the other out. Embeddings are BGE-M3 through Ollama.
4. Each agent writes an answer as a list of claims with source numbers. With
   verification on, every claim is checked: does the citation resolve, are the
   figures in the cited text, and does a judge model agree the source says it.
   Struck claims are removed and the answer rewritten from what survived.
5. The partial answers are merged, passed dates are flagged, and the result is
   read back with Kokoro (English) or OmniVoice (Urdu), streamed sentence by
   sentence so the first audio starts within a few seconds.

The baseline arm is `advisor/baseline/pipeline.py`: one retrieval over the
whole collection, one model call, no planner, no verifier. It is deliberately
left alone so the comparison stays clean.

## Setting up

You need Python 3.12, an NVIDIA GPU with about 8 GB of memory, and
[Ollama](https://ollama.com) installed.

```
python -m venv venv
venv\Scripts\activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
crawl4ai-setup
ollama pull bge-m3
```

Torch goes in first so the speech models get the CUDA build. `crawl4ai-setup`
downloads the headless browser; only the crawler and the optional live page
refresh use it.

Create a `.env` in the project root with three keys:

```
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
HF_TOKEN=...
```

Gemini runs the planner, the verifier and the default answer model, and it
also served the Gemma cells. OpenRouter is only needed for the Qwen cells.
The Hugging Face token is for downloading the speech models.

Urdu speech synthesis runs in its own environment because OmniVoice's
dependencies clash with the main stack:

```
python -m venv omni-venv
omni-venv\Scripts\activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-omni.txt
```

The Urdu voice is cloned from `data/audio/sample/urdu_sample.wav`, which is in
the repository. Recordings and synthesised answers go to `data/audio/input`
and `data/audio/output`, which the server creates on start.

The knowledge base lives in `data/chroma_db/`. It is not in git. Either unpack
the snapshot supplied with the submission into that folder, or rebuild it
with `python data/ingestion/crawler.py` (this crawls the live sites, so the
result will differ from the snapshot the evaluation used).

## Running it

Three processes, three terminals:

```
ollama serve
```

```
omni-venv\Scripts\activate
uvicorn omni_service:app --host 127.0.0.1 --port 8800
```

```
venv\Scripts\activate
uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000. The first start takes a minute while Whisper,
BGE-M3 and Kokoro load. The English-only path works without the OmniVoice
service; Urdu answers will show as text but not play.

To ask a question from the command line without the voice layer:

```
python advisor/multi_agent/graph.py "compare CS eligibility at LUMS and NUST"
python advisor/baseline/pipeline.py "compare CS eligibility at LUMS and NUST"
```

Two environment variables control the configuration. `ADVISOR_VERIFY` is
`off`, `lite` or `full` (default `full`). `ADVISOR_ANSWER_MODEL` picks the
answer model, for example `openrouter:qwen/qwen3.5-9b`.

## Reproducing the evaluation

The scored results are in `eval_results.md`. Everything in it is derived from
`runs/labels.jsonl`, and the file is regenerated with:

```
python eval_score.py aggregate
```

That reads only the labels and the run logs, so it works without Ollama or
the database. `python eval_score.py stats` prints just the statistical
tables. `python eval_score.py show --ids Q042 --claims` prints every
cell's answer to one query with each claim beside the chunk it cites, which
is how the labels were written and how any of them can be checked; that one
does need the knowledge base, since it greps the corpus.

Running a cell from scratch:

```
python eval_runner.py --arm baseline
python eval_runner.py --arm multi --verify off
python eval_runner.py --arm multi --verify full --model openrouter:qwen/qwen3.5-9b
```

Each run writes one JSONL file under `runs/` with the answer, every claim,
what the verifier struck and why, the full text of every retrieved chunk, and
the token usage per call. The twelve cell files used in the report are in
`runs/` already.

## Tests

```
python -m pytest tests -v
```

Sixteen tests over the parts that are deterministic: entity detection
including the Urdu aliases, the numeric checks in the verifier, the policy
gate, source renumbering when agents are merged, and the statistics. They need
no API key, GPU or database. Several of them guard against bugs that showed up
during the evaluation runs, such as source markers being read aloud.

## Layout

```
advisor/core/         retrieval, entity aliases, LLM client, the persona prompt
advisor/multi_agent/  supervisor (rewrite and split), the three agents, tools, verify, graph
advisor/baseline/     the single-agent comparison arm
advisor/stt.py        Whisper
advisor/tts.py        Kokoro
omni_service.py       OmniVoice Urdu TTS, separate venv
server.py             FastAPI backend
static/index.html     the web page
data/ingestion/       crawler, chunk metadata schema, entity registry, corpus inspector
data/eval/            the 100 queries and the gold checklist
eval_runner.py        runs one evaluation cell
eval_score.py         judging views, label storage, aggregation, statistics
eval_results.md       the results
runs/                 run logs and labels
tests/                unit tests
```

`data/ingestion/inspect_stored.py` prints what is in the knowledge base:
chunks per institution, freshness and category counts, and a grep for
finding which chunk holds a given figure.

## Things to know

The knowledge base is a snapshot from August 2026. Admissions pages change
every cycle, and several of the HEC pages describe older intakes. The system
flags dates that have passed and hedges facts from cycle-bound pages, but it
does not fetch live pages by default.

Habib University is in the entity list but not in the corpus, because its site
blocks crawlers. Asking about it produces an honest refusal rather than an
answer from someone else's pages.

Nothing a student says is stored beyond the browser session. The server keeps
conversation state in `advisor/multi_agent/sessions.db` so follow-up questions
work, and the New Chat button deletes it.
