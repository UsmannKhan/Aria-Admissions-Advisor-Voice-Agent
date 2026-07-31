"""Graph wiring.

    START -> understand -> admissions_agent    -> assemble -> END
                           scholarships_agent
                           exams_agent

understand() splits the question into subtasks, one per domain. dispatch()
sends each subtask to its agent; LangGraph runs those concurrently. Each agent
returns one partial, the reducer collects them, assemble() merges.

Only the agents a question actually needs are run: a fee question fires
admissions alone, a three-part question fires all three.

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
from advisor.multi_agent.scholarship_agent import run as scholarships_run
from advisor.multi_agent.supervisor import assemble, understand

AGENTS = {
    "admissions_agent": admissions_run,
    "scholarships_agent": scholarships_run,
    "exams_agent": exams_run,
}


def timed(name: str, fn):
    """Wrap a node so it reports its own wall-clock. Agents run concurrently,
    so their times overlap -- the sum will exceed the turn total."""
    def wrapped(state):
        t0 = time.perf_counter()
        out = fn(state)
        print(f"[timing] {name:20} {time.perf_counter() - t0:6.2f}s")
        return out
    return wrapped


def merge_partials(left: list | None, right: list | None) -> list:
    """Agents write concurrently, so partials are concatenated rather than
    overwriting each other. Writing None clears the list, which understand()
    does each turn so a session does not accumulate the previous turn's
    results."""
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

    answer: str
    claims: list[dict]
    sources: list[dict]


def dispatch(state: State) -> list[Send]:
    return [
        Send(f"{s['domain']}_agent", {
            "subtask": s,
            "k": state.get("k", 5),
            "profile": state.get("profile", {}),
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
    """Build the corpus and open Chroma before timing a query. In a server this
    happens once at startup; on the CLI it lands inside the first query and
    makes the turn look slower than it is."""
    from advisor.core import retrieval
    t0 = time.perf_counter()
    retrieval.get_corpus(retrieval.get_collection())
    print(f"[timing] {'corpus+chroma':20} {time.perf_counter() - t0:6.2f}s (startup)")


def ask(question: str, k: int = 5, session: str = "cli") -> dict:
    """History and profile come from the checkpointer under `session` -- they
    are deliberately not passed in, since supplying a key overwrites whatever
    was restored."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph.invoke(
        {"raw_query": question, "k": k},
        config={"configurable": {"thread_id": session}},
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--session", default="cli")
    ap.add_argument("--warm", action="store_true",
                    help="build the corpus before timing, as a server would")
    args = ap.parse_args()

    if args.warm:
        warm()

    t_graph = time.perf_counter()
    _graph = build_graph()
    print(f"[timing] {'graph build':20} {time.perf_counter() - t_graph:6.2f}s (startup)")

    start = time.perf_counter()
    out = ask(args.query, k=args.k, session=args.session)
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