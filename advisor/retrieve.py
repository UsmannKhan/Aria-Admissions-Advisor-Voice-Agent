"""
Retrieval-only test for the query side.

Hybrid retrieval: dense (bge-m3 via Ollama + ChromaDB) fused with lexical
(BM25 over chunk texts) using Reciprocal Rank Fusion. If the
query names more than one institution, retrieves per-entity and combines.

    python retrieve.py "what grades do I need for CS at LUMS?"
    python retrieve.py "compare CS eligibility at LUMS and Habib" --k 4
    python retrieve.py "fee structure" --entity lums
    python retrieve.py "..." --mode dense     # dense-only, for comparison
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Shared constants from the ingestion config
INGESTION_DIR = Path(__file__).resolve().parents[1] / "data" / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))
import config as cfg

import chromadb
import ollama
from rank_bm25 import BM25Okapi


# Entity aliases for keyword detection, matched case-insensitively as
# substrings of the query. Urdu-script spellings (and their variants) are
# included so Urdu queries route to the right entity.
ENTITY_ALIASES: dict[str, list[str]] = {
    "lums": ["lums", "lahore university of management", "لمز", "لُمز", "ایل یو ایم ایس"],
    "habib": ["habib", "حبیب", "حبیب یونیورسٹی", "حبيب"],
}

# RRF constant. 60 is the conventional default; higher = flatter fusion.
RRF_K = 60


def embed_query(text: str, client: ollama.Client) -> list[float]:
    """Embed a single query string with the same model used at ingestion."""
    resp = client.embed(model=cfg.EMBEDDING_MODEL, input=[text])
    return resp["embeddings"][0]


def warm_embedder() -> None:
    """Load the embedding model now so the first query doesn't pay the cold
    load. Retries because Ollama's CUDA init can transiently fail when it
    collides with other GPU initialisation at startup."""
    import time
    client = ollama.Client(host=cfg.OLLAMA_HOST)
    for attempt in range(3):
        try:
            client.embed(model=cfg.EMBEDDING_MODEL, input=["warmup"], keep_alive=5)
            print(f"[retrieve] embedder warmed ({cfg.EMBEDDING_MODEL})")
            return
        except Exception as e:
            if attempt < 2:
                time.sleep(2.0)  # let the GPU settle
            else:
                print(f"[retrieve] embedder warmup failed (non-fatal): {e}")


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DB_PATH))
    return client.get_collection(cfg.CHROMA_COLLECTION_NAME)


def detect_entities(query: str) -> list[str]:
    """Return entity_ids whose alias appears in the query (keyword match)."""
    q = query.lower()
    return [
        eid for eid, aliases in ENTITY_ALIASES.items()
        if any(alias in q for alias in aliases)
    ]


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25. Strips markdown punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


# Common words plus comparison/question filler that dilute a BM25 query.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at",
    "is", "are", "do", "does", "i", "need", "what", "which", "how", "compare",
    "comparison", "between", "vs", "versus", "me", "my", "with", "about",
    "cs",  # appears in nearly every chunk of a CS page; not discriminating
}


def clean_for_bm25(query: str, detected_entities: list[str]) -> str:
    """Strip stopwords and detected entity names so BM25 scores on content
    terms (e.g. 'eligibility criteria') rather than filler and institution
    names that appear in every chunk. Dense gets the raw query."""
    alias_words = set()
    for eid in detected_entities:
        for alias in ENTITY_ALIASES.get(eid, []):
            alias_words.update(alias.split())
        alias_words.add(eid)
    kept = [
        t for t in _tokenize(query)
        if t not in _STOPWORDS and t not in alias_words
    ]
    return " ".join(kept) if kept else query  # fall back to raw if all stripped


class Corpus:
    """All chunks pulled from ChromaDB once, with a BM25 index over texts.

    Dense (via Chroma) and lexical (via BM25) then run over the same chunk
    set. Fine for a small KB; Phase 2 would persist the index.
    """

    def __init__(self, collection: chromadb.Collection):
        got = collection.get(include=["documents", "metadatas"])
        self.ids = got["ids"]
        self.docs = got["documents"]
        self.metas = got["metadatas"]
        # id -> position, so dense results can map into this corpus
        self.pos_by_id = {cid: i for i, cid in enumerate(self.ids)}
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])

    def bm25_ranking(self, query: str, entity_id: str | None) -> list[str]:
        """Return chunk ids ranked by BM25, optionally filtered to one entity."""
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_ids = []
        for i in order:
            if entity_id and self.metas[i].get("entity_id") != entity_id:
                continue
            if scores[i] <= 0:
                continue  # no lexical overlap at all
            ranked_ids.append(self.ids[i])
        return ranked_ids


def _dense_ranking(collection, qvec, entity_id, n) -> list[str]:
    """Return chunk ids ranked by dense similarity (best first)."""
    res = collection.query(
        query_embeddings=[qvec],
        n_results=n,
        where={"entity_id": entity_id} if entity_id else None,
    )
    return res["ids"][0]


def _rrf_fuse(rankings: list[list[str]], k: int) -> list[str]:
    """Reciprocal Rank Fusion. Each id scores sum of 1/(RRF_K + rank)
    across rankings; return top-k ids by fused score."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    fused = sorted(scores, key=lambda c: scores[c], reverse=True)
    return fused[:k]


def hybrid_one(corpus, collection, qvec, bm25_query, k, entity_id, mode):
    """Retrieve top-k for one entity (or all) via dense, bm25, or hybrid.

    dense uses the embedded raw query; bm25 uses the cleaned lexical query.
    matched_by reflects which retriever's own top-k selected each chunk.
    """
    pool = max(k * 3, 15)
    dense_ids = _dense_ranking(collection, qvec, entity_id, pool)
    bm25_ids = corpus.bm25_ranking(bm25_query, entity_id)[:pool]

    dense_top = set(dense_ids[:k])
    bm25_top = set(bm25_ids[:k])

    if mode == "dense":
        chosen = dense_ids[:k]
    elif mode == "bm25":
        chosen = bm25_ids[:k]
    else:  # hybrid
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


def retrieve(query: str, k: int, entity: str | None, mode: str) -> list[dict]:
    ollama_client = ollama.Client(host=cfg.OLLAMA_HOST)
    collection = get_collection()
    corpus = Corpus(collection)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval test (dense / bm25 / hybrid)")
    p.add_argument("query", help="The query string")
    p.add_argument("--k", type=int, default=5,
                   help="top-k (per entity when comparing; default 5)")
    p.add_argument("--entity",
                   help="Force a single entity_id filter (e.g. lums, habib)")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid",
                   help="retrieval mode (default hybrid)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retrieve(args.query, args.k, args.entity, args.mode)


if __name__ == "__main__":
    main()