"""Graph wiring.

    START -> understand -> admissions_agent    -> assemble -> END
                           cost_agent
                           exams_agent

understand() splits the question into per-domain subtasks, dispatch() sends
each to its agent (concurrently), the reducer collects the partials and
assemble() merges them. Only the agents a question needs are run.

    python advisor/multi_agent/graph.py "your question"
"""

from __future__ import annotations

import time
from typing import Annotated, TypedDict

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):          # running as a script
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

SESSIONS_DB = Path(__file__).resolve().parent / "sessions.db"

from advisor.multi_agent.admission_agent import run as admissions_run
from advisor.multi_agent.exam_agent import run as exams_run
from advisor.multi_agent.cost_agent import run as cost_run
from advisor.multi_agent.supervisor import assemble, understand

AGENTS = {
    "admissions_agent": admissions_run,
    "cost_agent": cost_run,
    "exams_agent": exams_run,
}


def timed(name: str, fn):
    """Wrap a node to report its wall-clock. Agents overlap, so the times sum
    to more than the turn."""
    def wrapped(state):
        t0 = time.perf_counter()
        out = fn(state)
        elapsed = time.perf_counter() - t0
        print(f"[timing] {name:20} {elapsed:6.2f}s")
        out["timings"] = {name: round(elapsed, 2)}
        return out
    return wrapped


def merge_timings(left: dict | None, right: dict | None) -> dict:
    """Merge per-node timings from concurrent agents; a later turn overwrites."""
    return {**(left or {}), **(right or {})}


def merge_partials(left: list | None, right: list | None) -> list:
    """Concatenate partials from concurrent agents. None clears the list
    (understand() does this each turn)."""
    if right is None:
        return []
    return (left or []) + list(right)


class State(TypedDict, total=False):
    raw_query: str
    history: list[dict]        # [{"q": ..., "a": ...}], windowed by the supervisor
    profile: dict              # facts the student has stated, e.g. seat_type
    k: int

    normalized_query: str
    corrections: list[str]
    entity_notes: list[str]
    subtasks: list[dict]

    partials: Annotated[list[dict], merge_partials]
    timings: Annotated[dict, merge_timings]

    answer: str
    claims: list[dict]
    sources: list[dict]
    violations: list[str]
    language: str
    readings: list[dict]


def dispatch(state: State) -> list[Send]:
    return [
        Send(f"{s['domain']}_agent", {
            "subtask": s,
            "k": state.get("k", 5),
            "profile": state.get("profile", {}),
            "language": state.get("language", "en"),
            "history": state.get("history") or [],
            # the rewrite drops the student ("what can I get" -> "what is
            # available"), so the agent also gets the original
            "raw_query": state.get("raw_query", ""),
        })
        for s in state.get("subtasks", [])
    ]


def make_checkpointer() -> SqliteSaver:
    """check_same_thread=False is required: LangGraph runs the agents on
    separate threads, and sqlite3 connections are single-thread by default."""
    conn = sqlite3.connect(str(SESSIONS_DB), check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(checkpointer=None):
    b = StateGraph(State)
    b.add_node("understand", timed("understand", understand))
    for name, fn in AGENTS.items():
        b.add_node(name, timed(name, fn))
    b.add_node("assemble", timed("assemble", assemble))

    b.add_edge(START, "understand")
    b.add_conditional_edges("understand", dispatch, list(AGENTS))
    for name in AGENTS:
        b.add_edge(name, "assemble")
    b.add_edge("assemble", END)

    return b.compile(checkpointer=checkpointer or make_checkpointer())


_graph = None


def warm() -> None:
    """Build the corpus and open Chroma up front, so the first CLI query's
    time isn't inflated."""
    from advisor.core import retrieval
    t0 = time.perf_counter()
    retrieval.get_corpus(retrieval.get_collection())
    print(f"[timing] {'corpus+chroma':20} {time.perf_counter() - t0:6.2f}s (startup)")


def warm_graph():
    """Compile into the global ask() uses, so the server's first request
    doesn't rebuild it."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str, k: int = 5, session: str = "cli",
        language: str = "en") -> dict:
    """History and profile come from the checkpointer under `session`; passing
    them in would overwrite what was restored. `language` is the answer's only:
    understand() always rewrites to English, the corpus language.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph.invoke(
        {"raw_query": question, "k": k, "language": language},
        config={"configurable": {"thread_id": session}},
    )


if __name__ == "__main__":
    import argparse

    # redirected stdout on Windows uses the locale encoding, which can't hold Urdu
    sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--session", default="cli")
    ap.add_argument("--language", choices=["en", "ur"], default="en")
    ap.add_argument("--warm", action="store_true",
                    help="build the corpus before timing, as a server would")
    args = ap.parse_args()

    if args.warm:
        warm()

    t_graph = time.perf_counter()
    _graph = build_graph()
    print(f"[timing] {'graph build':20} {time.perf_counter() - t_graph:6.2f}s (startup)")

    start = time.perf_counter()
    out = ask(args.query, k=args.k, session=args.session,
              language=args.language)
    elapsed = time.perf_counter() - start

    print("\n" + "=" * 70)
    print(out.get("answer", "(no answer)"))
    print("=" * 70)
    for c in out.get("claims", []):
        print(f"  {c['sources']} {c['text']}")
    if out.get("profile"):
        print(f"\nprofile: {out['profile']}")
    print(f"\nsubtasks={len(out.get('subtasks', []))}  "
          f"claims={len(out.get('claims', []))}  "
          f"sources={len(out.get('sources', []))}  "
          f"{elapsed:.1f}s query"
          + ("" if args.warm else "  (includes corpus build; use --warm to exclude)"))