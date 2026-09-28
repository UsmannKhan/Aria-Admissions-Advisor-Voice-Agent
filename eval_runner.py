"""One evaluation cell (arm x answer model x verify mode), everything logged.

    python eval_runner.py --arm baseline
    python eval_runner.py --arm multi --verify off
    python eval_runner.py --arm multi --verify full --model openrouter:qwen/qwen3-32b
    python eval_runner.py --arm multi --limit 3

Flags are set as env vars before any advisor import (modules read config at
import), so one process per cell.

One JSONL line per query with everything scoring needs: answer, claims with
resolved sources, what was struck and why, full chunk text (so judging
survives corpus changes), timings, per-model token usage.

--queries defaults to data/eval/queries.json (the frozen 100-query set).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def slim_chunk(r: dict) -> dict:
    """A retrieved chunk as logged: ids, provenance and full text (what the
    judge scores against)."""
    meta = r.get("meta", {})
    return {
        "id": r.get("id"),
        "entity": meta.get("entity_id"),
        "headings": meta.get("headings"),
        "url": meta.get("source_url"),
        "category": meta.get("content_category"),
        "freshness": meta.get("freshness_class"),
        # set at query time by verify.refresh_evidence, not stored metadata
        "stale_reasons": r.get("stale_reasons", []),
        "refreshed": bool(r.get("refreshed")),
        "doc": r.get("doc", ""),
    }


def slim_verified(v: dict) -> dict:
    """A struck/verified claim without the bulky numeric internals."""
    return {"text": v.get("text"), "verdict": v.get("verdict"),
            "sources": v.get("sources"), "flags": v.get("flags"),
            "notes": v.get("notes"), "span": v.get("span"),
            "overruled": v.get("overruled")}


def run_baseline(item: dict, k: int) -> dict:
    from advisor.baseline.pipeline import answer_query
    results, parsed = answer_query(item["query"], k=k, mode="dense",
                                   language=item.get("lang", "en"))
    if not results or parsed is None:
        return {"answer": None, "claims": [], "sources": [], "abstained": True}
    return {
        "answer": parsed.answer,
        "claims": [{"text": c.text, "sources": c.sources} for c in parsed.claims],
        "sources": [slim_chunk(r) for r in results],
        "abstained": False,
    }


DIRECT_SYSTEM = (
    "You are an admissions counsellor for Pakistani undergraduate applicants. "
    "Answer the student's question as accurately as you can from your own "
    "knowledge. Be concise; the answer is read aloud. If you do not know, say so."
)


def run_direct(item: dict) -> dict:
    """A0: the answer model alone, no retrieval or source rules. No claims, so
    no faithfulness score."""
    from advisor.core.persona import language_clause, today_clause
    text = llm.generate(llm.ANSWER_MODEL, item["query"],
                        system=DIRECT_SYSTEM + today_clause()
                        + language_clause(item.get("lang", "en"))).text.strip()
    return {"answer": text or None, "claims": [], "sources": [],
            "abstained": not text}


def run_web(item: dict) -> dict:
    """A1: model + Google Search grounding, no RAG. Logs web citation URLs
    instead of chunks; no claims."""
    from google.genai import types as gt
    from advisor.core.persona import language_clause, today_clause
    resp = llm.get_client().models.generate_content(
        model=llm.ANSWER_MODEL,
        contents=item["query"],
        config=gt.GenerateContentConfig(
            system_instruction=DIRECT_SYSTEM + today_clause()
            + language_clause(item.get("lang", "en"), structured=False),
            tools=[{"google_search": {}}],
        ),
    )
    meta = getattr(resp, "usage_metadata", None)
    llm._account(llm.ANSWER_MODEL, {
        "prompt_tokens": getattr(meta, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(meta, "candidates_token_count", 0) or 0,
        "thinking_tokens": getattr(meta, "thoughts_token_count", 0) or 0})
    cites = []
    try:
        for ch in resp.candidates[0].grounding_metadata.grounding_chunks or []:
            if getattr(ch, "web", None):
                cites.append({"title": ch.web.title, "url": ch.web.uri})
    except Exception:
        pass
    text = (resp.text or "").strip()
    return {"answer": text or None, "claims": [], "sources": cites,
            "abstained": not text}


def run_multi(item: dict, k: int, session: str) -> dict:
    from advisor.multi_agent import graph
    out = graph.ask(item["query"], k=k, session=session,
                    language=item.get("lang", "en"))
    partials = out.get("partials") or []
    return {
        "answer": out.get("answer"),
        "claims": out.get("claims") or [],
        "sources": [slim_chunk(r) for r in out.get("sources") or []],
        "violations": out.get("violations") or [],
        "struck": [slim_verified(s) for p in partials
                   for s in p.get("struck") or []],
        "verified": [{**slim_verified(v), "domain": p["domain"]}
                     for p in partials for v in p.get("verified") or []],
        "calibration": {k: sum(p.get("calibration", {}).get(k, 0) for p in partials)
                        for k in ("confirmed", "overruled", "judge_only")},
        "timings": out.get("timings") or {},
        "abstained": all(p.get("abstained") for p in partials) if partials else True,
        "abstained_domains": [p["domain"] for p in partials if p.get("abstained")],
        "subtasks": out.get("subtasks") or [],
        "normalized_query": out.get("normalized_query"),
        "corrections": out.get("corrections") or [],
        "entity_notes": out.get("entity_notes") or [],
        "profile": out.get("profile") or {},
        "as_of": [p["as_of"] for p in partials if p.get("as_of")],
    }


def usage_delta(before: dict, after: dict) -> dict:
    """Per-model token spend of one query, from the process-wide counters."""
    out = {}
    for model, row in after.items():
        prev = before.get(model, {})
        diff = {key: row[key] - prev.get(key, 0) for key in row
                if not isinstance(row[key], dict)}
        prov = {p: n - prev.get("providers", {}).get(p, 0)
                for p, n in row.get("providers", {}).items()}
        diff["providers"] = {p: n for p, n in prov.items() if n}
        if any(v for k, v in diff.items() if k != "providers"):
            out[model] = diff
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one evaluation cell")
    ap.add_argument("--arm", choices=["direct", "web", "baseline", "multi"], required=True,
                    help="direct = model alone (A0); web = model + Google Search (A1); "
                         "baseline = single-agent RAG; multi = the multi-agent graph")
    ap.add_argument("--queries", help="JSON file of query items "
                                      "(default: data/eval/queries.json)")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--limit", type=int, help="only the first N queries (smoke)")
    ap.add_argument("--ids", help="comma-separated query ids to run (partial re-run)")
    ap.add_argument("--out", help="JSONL path (default: runs/<cell>_<ts>.jsonl)")
    ap.add_argument("--verify", choices=["off", "lite", "full"],
                    help="verification mode for the multi arm (default: full)")
    ap.add_argument("--model", help="answer model id, e.g. openrouter:qwen/qwen3-32b "
                                    "(default: the project default)")
    ap.add_argument("--provider", help="pin the OpenRouter provider, e.g. "
                                       "'Google AI Studio' (no fallbacks)")
    args = ap.parse_args()

    # Must precede every advisor import: the agents read ADVISOR_VERIFY and
    # llm.py reads ADVISOR_ANSWER_MODEL when their modules load.
    import os
    if args.verify:
        os.environ["ADVISOR_VERIFY"] = args.verify
    if args.model:
        os.environ["ADVISOR_ANSWER_MODEL"] = args.model
    if args.provider:
        os.environ["ADVISOR_OPENROUTER_PROVIDER"] = args.provider
    global llm, detect_entities, V
    from advisor.core import llm
    from advisor.core.entities import detect_entities
    from advisor.multi_agent import verify as V

    qpath = Path(args.queries) if args.queries else PROJECT_ROOT / "data" / "eval" / "queries.json"
    items = json.loads(qpath.read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [x for x in items if x.get("id") in wanted]
    if args.limit:
        items = items[:args.limit]

    verify_mode = V.env_mode() if args.arm == "multi" else "n/a"
    run_id = uuid.uuid4().hex[:8]
    cell = {
        "run_id": run_id,
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "arm": args.arm,
        "verify_mode": verify_mode,
        "answer_model": llm.ANSWER_MODEL,
        # only the multi arm has a planner; only multi/full and multi/lite
        # run verification; direct never retrieves
        "planner_model": llm.PLANNER_MODEL if args.arm == "multi" else None,
        "web_search": args.arm == "web",
        "verifier_model": (llm.VERIFIER_MODEL
                           if args.arm == "multi" and verify_mode == "full" else None),
        "openrouter_provider_pin": llm.OPENROUTER_PROVIDER or None,
        "k": None if args.arm in ("direct", "web") else args.k,
        "n_queries": len(items),
    }

    # corpus identity, so results can't be scored against the wrong data;
    # direct/web never touch the corpus or need Ollama
    if args.arm not in ("direct", "web"):
        from advisor.core import retrieval
        collection = retrieval.get_collection()
        cell["corpus_chunks"] = collection.count()

    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "runs" /
        f"{args.arm}_{verify_mode}_{run_id}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # warm up first, like the server, so query 1 isn't charged the cold loads
    if args.arm not in ("direct", "web"):
        retrieval.warm_embedder()
        retrieval.get_corpus(collection)
    if args.arm == "multi":
        from advisor.multi_agent import graph
        graph.warm_graph()

    print(f"[runner] cell {cell['arm']}/{verify_mode} "
          f"answer={cell['answer_model']} -> {out_path}")

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"cell": cell}, ensure_ascii=False) + "\n")

        for i, item in enumerate(items, start=1):
            before = llm.usage_summary()
            t0 = time.perf_counter()
            try:
                # fresh session per query: no profile/history carried over
                if args.arm == "direct":
                    result = run_direct(item)
                elif args.arm == "web":
                    result = run_web(item)
                elif args.arm == "baseline":
                    result = run_baseline(item, args.k)
                else:
                    result = run_multi(item, args.k,
                                       session=f"eval_{run_id}_{item['id']}")
                error = None
            except Exception as exc:
                result, error = None, f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - t0

            record = {
                "run_id": run_id,
                "query_id": item.get("id", f"q{i}"),
                "item": item,
                "detected_entities": detect_entities(item["query"]),
                "result": result,
                "error": error,
                "time_s": round(elapsed, 2),
                "usage": usage_delta(before, llm.usage_summary()),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

            status = "ERROR" if error else (
                "abstain" if result.get("abstained") else "answer")
            print(f"[runner] {i:>3}/{len(items)} {item.get('id','?'):>4} "
                  f"{elapsed:6.1f}s  {status}")

    totals = llm.usage_summary()
    print(f"\n[runner] done -> {out_path}")
    for model, row in totals.items():
        print(f"[runner] {model}: {row}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
