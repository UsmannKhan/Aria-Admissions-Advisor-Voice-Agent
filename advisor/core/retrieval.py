"""
Retrieval: dense (bge-m3 via Ollama + ChromaDB), BM25 over the same chunks,
or both fused with RRF. Multi-institution queries retrieve per entity.

    python advisor/core/retrieval.py "what grades do I need for CS at LUMS?"
    python advisor/core/retrieval.py "compare CS eligibility at LUMS and Habib" --k 4
    python advisor/core/retrieval.py "fee structure" --entity lums
    python advisor/core/retrieval.py "..." --mode dense
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):          # running as a script
    sys.path.insert(0, str(PROJECT_ROOT))

INGESTION_DIR = PROJECT_ROOT / "data" / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))
import config as cfg

import chromadb
import ollama
from rank_bm25 import BM25Okapi

from advisor.core.entities import ENTITY_ALIASES, detect_entities

RRF_K = 60  # conventional default; higher = flatter fusion


def embed_query(text: str, client: ollama.Client) -> list[float]:
    resp = client.embed(model=cfg.EMBEDDING_MODEL, input=[text])
    return resp["embeddings"][0]


def warm_embedder() -> None:
    """Load bge-m3 into Ollama now so the first query doesn't pay the
    cold-load. Retries because Ollama's CUDA init can fail transiently
    when it races other GPU init at startup."""
    import time
    client = ollama.Client(host=cfg.OLLAMA_HOST)
    for attempt in range(3):
        try:
            client.embed(model=cfg.EMBEDDING_MODEL, input=["warmup"], keep_alive=5)
            print(f"[retrieve] embedder warmed ({cfg.EMBEDDING_MODEL})")
            return
        except Exception as e:
            if attempt < 2:
                time.sleep(2.0)
            else:
                print(f"[retrieve] embedder warmup failed (non-fatal): {e}")


_collection = None
_ollama = None
_init_lock = threading.Lock()


def get_collection() -> chromadb.Collection:
    """One client per process, behind a lock. Agents run concurrently, and
    building a PersistentClient per call made three threads race to initialise
    the same store, which fails with a tenant error."""
    global _collection
    if _collection is None:
        with _init_lock:
            if _collection is None:
                client = chromadb.PersistentClient(path=str(cfg.CHROMA_DB_PATH))
                _collection = client.get_collection(cfg.CHROMA_COLLECTION_NAME)
    return _collection


def get_ollama() -> ollama.Client:
    global _ollama
    if _ollama is None:
        with _init_lock:
            if _ollama is None:
                _ollama = ollama.Client(host=cfg.OLLAMA_HOST)
    return _ollama


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# stopwords + comparison/question filler that dilute a BM25 query.
# "cs" is in nearly every chunk of a CS page, so it's not discriminating.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at",
    "is", "are", "do", "does", "i", "need", "what", "which", "how", "compare",
    "comparison", "between", "vs", "versus", "me", "my", "with", "about",
    "cs",
}


def clean_for_bm25(query: str, detected_entities: list[str]) -> str:
    """Strip stopwords and entity names so BM25 scores on content terms.
    Dense still gets the raw query."""
    alias_words = set()
    for eid in detected_entities:
        for alias in ENTITY_ALIASES.get(eid, []):
            alias_words.update(alias.split())
        alias_words.add(eid)
    kept = [
        t for t in _tokenize(query)
        if t not in _STOPWORDS and t not in alias_words
    ]
    return " ".join(kept) if kept else query


class Corpus:
    """All chunks pulled from ChromaDB once + a BM25 index over them.
    In-memory is fine at this KB size."""

    def __init__(self, collection: chromadb.Collection):
        got = collection.get(include=["documents", "metadatas"])
        self.ids = got["ids"]
        self.docs = got["documents"]
        self.metas = got["metadatas"]
        self.pos_by_id = {cid: i for i, cid in enumerate(self.ids)}
        self.by_page = {
            (m.get("source_url", ""), m.get("chunk_position", -1)): i
            for i, m in enumerate(self.metas)
        }
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])

    def bm25_ranking(self, query: str, entity_id: str | None) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_ids = []
        for i in order:
            if entity_id and self.metas[i].get("entity_id") != entity_id:
                continue
            if scores[i] <= 0:  # no lexical overlap at all
                continue
            ranked_ids.append(self.ids[i])
        return ranked_ids


_corpus: "Corpus | None" = None


def get_corpus(collection: chromadb.Collection) -> "Corpus":
    """Build the BM25 index once per process. Rebuilding it per call meant
    every query re-read and re-tokenised the whole corpus, and three parallel
    agents did it three times at once."""
    global _corpus
    if _corpus is None:
        with _init_lock:
            if _corpus is None:
                _corpus = Corpus(collection)
    return _corpus


def _dense_ranking(collection, qvec, entity_id, n) -> list[str]:
    res = collection.query(
        query_embeddings=[qvec],
        n_results=n,
        where={"entity_id": entity_id} if entity_id else None,
    )
    return res["ids"][0]


def _rrf_fuse(rankings: list[list[str]], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    fused = sorted(scores, key=lambda c: scores[c], reverse=True)
    return fused[:k]


_PROG_STOPWORDS = {"bs", "bsc", "ba", "bba", "hons", "honours", "programme",
                   "program", "major", "joint", "the", "and", "for", "with"}

_DEGREE_SLUG = re.compile(r"^(bs|bsc|ba|bba|be|ms|msc|mphil|phd|llb|ba-ll)\b")


def _slug_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _PROG_STOPWORDS and len(w) > 2}


def _page_programme(source_url: str) -> set[str]:
    """Programme tokens for a programme page, empty for anything else.

    The URL slug is the one unambiguous name a page has -- headings vary in
    wording and contain commas that break parsing, but
    /programmes/bs-computer-science identifies exactly one degree.
    """
    if not source_url:
        return set()
    slug = source_url.rstrip("/").rsplit("/", 1)[-1].lower()
    looks_like_programme = (
        "/programmes/" in source_url
        or "/department" in source_url
        or bool(_DEGREE_SLUG.match(slug))
    )
    return _slug_tokens(slug) if looks_like_programme else set()


def _query_programme(query: str) -> set[str]:
    """Distinctive tokens of the longest known programme named in the query.

    Also tries each name without its degree prefix, since the supervisor's
    rewrite often drops it -- "LUMS ka CS ka fee" comes back as "the Computer
    Science program at LUMS", which would not match "BS Computer Science".
    """
    low = query.lower()
    best = ""
    for p in cfg.KNOWN_PROGRAMS:
        full = p.lower()
        stripped = _DEGREE_SLUG.sub("", full).strip()
        for form in (full, stripped):
            if form and form in low and len(form) > len(best):
                best = form
    return _slug_tokens(best) if best else set()


def scope_to_programme(query: str, results: list[dict]) -> list[dict]:
    """Reorder so the programme the student named comes first.

    LUMS publishes one page per programme, each repeating similarly worded fee
    and admission sections, so a fee question returns several programmes' worth
    of near-identical prose that the embedding cannot separate. Matching is by
    token subset, not overlap, so a Computer Science query does not pull in
    Computer Engineering. Nothing is dropped: pages naming another programme
    are demoted, pages naming none stay put.
    """
    wanted = _query_programme(query)
    if not wanted:
        return results

    match, neutral, other = [], [], []
    for r in results:
        owner = _page_programme(r.get("meta", {}).get("source_url", ""))
        if not owner:
            neutral.append(r)
        elif wanted <= owner:
            match.append(r)
        else:
            other.append(r)
    return match + neutral + other


def _dedupe_identical(results: list[dict]) -> list[dict]:
    """Drop chunks whose text is identical to one already kept. Pages that
    share a block (the same deadlines table on every programme page) otherwise
    fill the context with copies. Near-identical content is left alone -- fee
    tables differ between programmes even where they look alike."""
    seen: set[int] = set()
    out = []
    for r in results:
        h = hash(r.get("doc", "").strip())
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def expand_neighbours(corpus, results: list[dict], window: int = 1,
                      top: int = 3) -> list[dict]:
    """Pull each hit's adjacent chunks from the same page.

    Structure-aware chunking splits a section from the note that explains it --
    a fee table ends up in one chunk and "the above breakdown is for Year 1" in
    the next. The note retrieves well (prose, fee vocabulary) and the table does
    not (digits and pipes), so the note wins and the numbers are never seen.
    Expanding by position recovers the pair. This is the parent window from the
    design, built at query time from chunk_position rather than stored.
    """
    seen = {r["id"] for r in results}
    out: list[dict] = []

    for rank, r in enumerate(results):
        out.append(r)
        if rank >= top:          # only the strongest hits pull neighbours
            continue
        meta = r.get("meta", {})
        url = meta.get("source_url", "")
        pos = meta.get("chunk_position")
        if pos is None:
            continue
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            i = corpus.by_page.get((url, pos + delta))
            if i is None:
                continue
            cid = corpus.ids[i]
            if cid in seen:
                continue
            seen.add(cid)
            out.append({
                "id": cid,
                "doc": corpus.docs[i],
                "meta": corpus.metas[i],
                "in_dense": False,
                "in_bm25": False,
                "neighbour_of": r["id"],
            })
    return _dedupe_identical(out)


def hybrid_one(corpus, collection, qvec, bm25_query, k, entity_id, mode):
    """Top-k for one entity (or all) via dense / bm25 / hybrid."""
    pool = max(k * 3, 15)
    dense_ids = _dense_ranking(collection, qvec, entity_id, pool)
    # skip BM25 scoring entirely in dense mode -- it was being computed over
    # the whole corpus and thrown away
    bm25_ids = ([] if mode == "dense"
                else corpus.bm25_ranking(bm25_query, entity_id)[:pool])

    # matched_by should reflect each retriever's OWN top-k, not the pool
    dense_top = set(dense_ids[:k])
    bm25_top = set(bm25_ids[:k])

    if mode == "dense":
        chosen = dense_ids[:k]
    elif mode == "bm25":
        chosen = bm25_ids[:k]
    else:
        chosen = _rrf_fuse([dense_ids, bm25_ids], k)

    out = []
    for cid in chosen:
        i = corpus.pos_by_id.get(cid)
        if i is None:
            continue
        out.append({
            "id": cid,
            "doc": corpus.docs[i],
            "meta": corpus.metas[i],
            "in_dense": cid in dense_top,
            "in_bm25": cid in bm25_top,
        })
    return out


def retrieve(query: str, k: int, entity: str | None, mode: str,
             expand: int = 1) -> list[dict]:
    ollama_client = get_ollama()
    collection = get_collection()
    corpus = get_corpus(collection)
    qvec = embed_query(query, ollama_client)

    if entity:
        label = f"single entity filter={entity}"
        bm25_q = clean_for_bm25(query, [entity])
        results = hybrid_one(corpus, collection, qvec, bm25_q, k, entity, mode)
    else:
        detected = detect_entities(query)
        bm25_q = clean_for_bm25(query, detected)
        if len(detected) >= 2:
            label = f"per-entity (detected: {', '.join(detected)}); k={k} each"
            results = []
            for eid in detected:
                results.extend(
                    hybrid_one(corpus, collection, qvec, bm25_q, k, eid, mode)
                )
        elif len(detected) == 1:
            label = f"single entity (detected: {detected[0]})"
            results = hybrid_one(corpus, collection, qvec, bm25_q, k, detected[0], mode)
        else:
            label = "no entity detected (whole collection)"
            results = hybrid_one(corpus, collection, qvec, bm25_q, k, None, mode)

    results = scope_to_programme(query, results)
    if expand:
        results = expand_neighbours(corpus, results, window=expand)

    print(f"\nQuery: {query!r}")
    print(f"BM25 query (cleaned): {bm25_q!r}")
    print(f"Collection count: {collection.count()}  |  retrieval={mode}  |  {label}")
    print("=" * 80)

    if not results:
        print("No results.")
        return results

    for rank, r in enumerate(results, start=1):
        meta = r["meta"]
        src = []
        if r.get("neighbour_of"):
            src.append("neighbour")
        if r.get("in_dense"):
            src.append("dense")
        if r.get("in_bm25"):
            src.append("bm25")
        print(f"\n[{rank}] entity={meta.get('entity_id')}  "
              f"category={meta.get('content_category')}  "
              f"freshness={meta.get('freshness_class')}  "
              f"matched_by={'+'.join(src) or '?'}")
        print(f"    headings: {meta.get('headings', '')}")
        print(f"    {' '.join(r['doc'].split())[:220]}")

    return results


def retrieve_for_subtask(
    query: str,
    k: int,
    entities: list[str],
    mode: str = "hybrid",
    expand: int = 1,
) -> list[dict]:
    """Quiet retrieval for orchestration: entities come from the supervisor's
    decomposition, so no re-detection and no printing. `retrieve()` above is
    left untouched -- it is the CLI/baseline path."""
    ollama_client = get_ollama()
    collection = get_collection()
    corpus = get_corpus(collection)
    qvec = embed_query(query, ollama_client)
    bm25_q = clean_for_bm25(query, entities)

    if not entities:
        results = hybrid_one(corpus, collection, qvec, bm25_q, k, None, mode)
    else:
        results = []
        for eid in entities:
            results.extend(hybrid_one(corpus, collection, qvec, bm25_q, k, eid, mode))

    results = scope_to_programme(query, results)
    if expand:
        results = expand_neighbours(corpus, results, window=expand)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval test (dense / bm25 / hybrid)")
    p.add_argument("query", help="The query string")
    p.add_argument("--k", type=int, default=5,
                   help="top-k (per entity when comparing; default 5)")
    p.add_argument("--entity",
                   help="Force a single entity_id filter (e.g. lums, habib)")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid",
                   help="retrieval mode (default hybrid)")
    p.add_argument("--expand", type=int, default=1,
                   help="pull N adjacent chunks per hit from the same page (0 = off)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retrieve(args.query, args.k, args.entity, args.mode, expand=args.expand)


if __name__ == "__main__":
    main()