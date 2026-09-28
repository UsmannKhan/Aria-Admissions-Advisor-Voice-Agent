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
from dataclasses import dataclass
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
    """Load bge-m3 into Ollama at startup so the first query skips the cold
    load. Retries: Ollama's CUDA init sometimes fails racing other GPU init."""
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
    """One client per process, behind a lock. A PersistentClient per call had
    3 agent threads racing to initialise the store (tenant error)."""
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
        # programme tokens -> [(entity_id, page url)], from URL slugs. Grows
        # with ingestion, unlike the hand-written KNOWN_PROGRAMS (went stale).
        self.programme_pages: dict[frozenset, list[tuple[str, str]]] = {}
        for url, i in {m.get("source_url", ""): i
                       for i, m in enumerate(self.metas)}.items():
            owner = _page_programme(url)
            if owner:
                self.programme_pages.setdefault(frozenset(owner), []).append(
                    (self.metas[i].get("entity_id", ""), url))
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])

    def bm25_ranking_urls(self, query: str, urls: set[str]) -> list[str]:
        """BM25 over one programme's pages (a few dozen chunks), on heading +
        body. Inside one page the heading separates sections ("Test
        Requirements" for a tests query). The global index stays body-only:
        corpus-wide, boilerplate headings outscore content."""
        idxs = [i for i, m in enumerate(self.metas)
                if m.get("source_url") in urls]
        if not idxs:
            return []
        texts = [f"{self.metas[i].get('headings', '')} {self.docs[i]}"
                 for i in idxs]
        mini = BM25Okapi([_tokenize(t) for t in texts])
        scores = mini.get_scores(_tokenize(query))
        order = sorted(range(len(idxs)), key=lambda j: scores[j], reverse=True)
        return [self.ids[idxs[j]] for j in order if scores[j] > 0]

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
    """BM25 index, built once per process. Per-call rebuilds re-tokenised the
    whole corpus on every query, 3 times over with parallel agents."""
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
                   "program", "major", "joint", "the", "and", "for", "with",
                   # degree words and abbreviations that appear in slugs like
                   # bachelors-of-science-in-computer-science-bscs
                   "bachelor", "bachelors", "bscs", "bsai", "bsds", "bsse",
                   "bscy", "bece", "bee"}

_DEGREE_SLUG = re.compile(r"^(bs|bsc|ba|bba|be|ms|msc|mphil|phd|llb|ba-ll)\b")


def _slug_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _PROG_STOPWORDS and len(w) > 2}


# SEECS slugs carry the intake (-for-fall-2025-on-wards) and the spelled-out
# degree (bachelor-of-science-in-); strip both or tokens never match a query
_SLUG_INTAKE = re.compile(r"-(?:for-)?(?:fall|spring)-?\d{4}.*$|-\d{4}(?:-.*)?$")
_SLUG_DEGREE = re.compile(r"^bachelors?-of-(?:science-in-|arts-in-|science-|arts-)?")


def _page_programme(source_url: str) -> set[str]:
    """Programme tokens for a programme page from its URL slug, empty for
    anything else. Slug formats differ per site: LUMS /programmes/bs-computer-
    science, FAST /Program/BS(CS), SEECS /program/bachelor-of-science-in-
    artificial-intelligence-for-fall-2025-on-wards. Listing pages
    (/departments/, /Degree-Programs) stay neutral."""
    if not source_url:
        return set()
    low = source_url.lower()
    slug = low.rstrip("/").rsplit("/", 1)[-1]
    looks_like_programme = (
        "/programmes/" in low
        or "/programme/" in low
        or "/program/" in low
        or "/department-page/" in low
        or bool(_DEGREE_SLUG.match(slug))
    )
    if not looks_like_programme:
        return set()
    canon = _PROG_CANON.get(slug)      # FAST-style abbreviation slugs: bs(cs)
    if canon:
        return _slug_tokens(canon)
    slug = _SLUG_INTAKE.sub("", slug)
    slug = _SLUG_DEGREE.sub("", slug)
    return _slug_tokens(slug)


# _slug_tokens drops the degree prefix and anything under 3 chars, so "BS CS"
# gave an empty set and scoping silently turned off. Map short forms to long.
_PROG_CANON = {
    "bscs": "bs computer science", "bs cs": "bs computer science",
    "bs(cs)": "bs computer science", "bs(ai)": "bs artificial intelligence",
    "bs ai": "bs artificial intelligence", "bs(ds)": "bs data science",
    "bs(se)": "bs software engineering", "bs(cy)": "bs cyber security",
    "bs(ee)": "bs electrical engineering", "bs(ce)": "bs computer engineering",
    "bs(cv)": "bs civil engineering", "bs(me)": "bs mechanical engineering",
    "bs(af)": "bs accounting and finance", "bs(ba)": "bs business analytics",
    "bs(fintech)": "bs financial technology",
}

# short forms students type, for retrieval only (KNOWN_PROGRAMS also feeds the
# crawler's mentioned_programs). Word-boundary matched ("cs" not in "physics").
# No "se"/"ds": they collide with romanised Urdu.
_PROG_ALIASES = {
    "cs": "bs computer science", "comp sci": "bs computer science",
    "compsci": "bs computer science", "ai": "bs artificial intelligence",
}


def _query_programme(query: str) -> set[str]:
    """Distinctive tokens of the longest programme named in the query.

    Known-list pass also tries names without the degree prefix (the rewrite
    turns "LUMS ka CS ka fee" into "the Computer Science program at LUMS").
    Corpus pass matches any ingested programme whose tokens are all in the
    query.
    """
    low = query.lower()
    best = ""
    candidates = [(p.lower(), _PROG_CANON.get(p.lower(), p.lower()))
                  for p in cfg.KNOWN_PROGRAMS]
    candidates += list(_PROG_ALIASES.items())
    for form0, full in candidates:
        stripped = _DEGREE_SLUG.sub("", full).strip()
        for form, matched in ((form0, full), (stripped, stripped)):
            if (form and len(matched) > len(best)
                    and re.search(rf"\b{re.escape(form)}\b", low)):
                best = matched
    known = _slug_tokens(best) if best else set()

    qtokens = _slug_tokens(low)
    corpus_best: frozenset = frozenset()
    for tokens in _corpus_programme_sets():
        if tokens <= qtokens and len(tokens) > len(corpus_best):
            corpus_best = tokens
    return set(corpus_best) if len(corpus_best) > len(known) else known


_prog_sets: list[frozenset] | None = None


def _corpus_programme_sets() -> list[frozenset]:
    global _prog_sets
    if _prog_sets is None:
        try:
            _prog_sets = list(get_corpus(get_collection()).programme_pages)
        except Exception:
            return []          # no corpus (unit tests); known-list pass stands
    return _prog_sets


def detect_programme(text: str) -> str:
    """Canonical display form (the page slug) of the programme the text names,
    or ''. A string for the supervisor's cross-check; _query_programme turns it
    back into the same tokens."""
    tokens = frozenset(_query_programme(text))
    if not tokens:
        return ""
    pages = get_corpus(get_collection()).programme_pages.get(tokens)
    if pages:
        slug = pages[0][1].rstrip("/").rsplit("/", 1)[-1]
        return slug.replace("-", " ")
    return " ".join(sorted(tokens))


def programme_catalogue() -> list[str]:
    """Every ingested programme and the institutions that teach it, for the
    supervisor's prompt. From the corpus, not a hand list."""
    corpus = get_corpus(get_collection())
    out = []
    for tokens, pages in sorted(corpus.programme_pages.items(),
                                key=lambda kv: sorted(kv[0])):
        slug = pages[0][1].rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
        ents = sorted({e for e, _ in pages if e})
        out.append(f"{slug} ({', '.join(ents)})")
    return out


def _programme_hits(corpus, collection, qvec, bm25_q: str, wanted: frozenset,
                    k: int, entity_id: str | None) -> list[dict]:
    """Hybrid top-k over the named programme's own pages only.

    A couple of dozen programme pages share near-identical criteria/dates
    blocks, so the named programme's copy can miss the general pool. Hybrid
    because dense ranked "Important Note" above "Test Requirements" for a tests
    query. Exact token set: a joint major is a superset with its own fee table.
    """
    pages = corpus.programme_pages.get(wanted, [])
    urls = [u for e, u in pages if not entity_id or e == entity_id]
    if not urls:
        return []
    try:
        res = collection.query(query_embeddings=[qvec], n_results=k,
                               where={"source_url": {"$in": urls}})
        dense_ids = res["ids"][0]
    except Exception as exc:
        print(f"[retrieve] programme-scoped query failed: {exc}")
        return []
    bm25_ids = corpus.bm25_ranking_urls(bm25_q, set(urls))[:k]

    out = []
    for cid in _rrf_fuse([dense_ids, bm25_ids], k):
        i = corpus.pos_by_id.get(cid)
        if i is None:
            continue
        out.append({"id": cid, "doc": corpus.docs[i], "meta": corpus.metas[i],
                    "in_dense": cid in dense_ids, "in_bm25": cid in bm25_ids,
                    "programme_scoped": True})
    return out


def scope_to_programme(query: str, results: list[dict],
                       drop_other: bool = False) -> list[dict]:
    """Reorder so the named programme comes first.

    LUMS has one page per programme with near-identical fee/admission sections
    the embedding can't tell apart. Exact token-set match: subset matching let
    /bsc-honours-economics-data-and-computer-science (a superset) bring its fee
    table into a CS query.

    drop_other removes pages naming a different programme; pages naming none
    stay (general fee policy, shared deadlines). Off by default so the
    baseline's retrieve() is unchanged.
    """
    wanted = _query_programme(query)
    if not wanted:
        return results

    match, neutral, other = [], [], []
    for r in results:
        owner = _page_programme(r.get("meta", {}).get("source_url", ""))
        if not owner:
            neutral.append(r)
        elif (wanted == owner) if drop_other else (wanted <= owner):
            match.append(r)
        else:
            other.append(r)
    return match + neutral + ([] if drop_other else other)


def _dedupe_identical(results: list[dict]) -> list[dict]:
    """Drop exact duplicate chunk text (e.g. the same deadlines table on every
    programme page). Near-duplicates stay: fee tables that look alike differ."""
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
    """Pull adjacent chunks (by chunk_position) for the top hits.

    A fee table and its note ("the above breakdown is for Year 1") can land in
    separate chunks; the note ranks, the table (digits, pipes) doesn't. A
    parent window built at query time.
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
    # dense mode: don't score BM25 over the whole corpus just to discard it
    bm25_ids = ([] if mode == "dense"
                else corpus.bm25_ranking(bm25_query, entity_id)[:pool])

    # matched_by should reflect each retriever's own top-k, not the pool
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
        scope_ents = [entity]
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
        scope_ents = detected or [None]

    # named programme: pull its pages directly, since reordering can't add
    # chunks the pool never had
    wanted = frozenset(_query_programme(query))
    if wanted:
        seen = {r["id"] for r in results}
        extra = []
        for eid in scope_ents:
            for h in _programme_hits(corpus, collection, qvec, bm25_q,
                                     wanted, k, eid):
                if h["id"] not in seen:
                    seen.add(h["id"])
                    extra.append(h)
        results = extra + results
        label += f" | programme={'-'.join(sorted(wanted))} (+{len(extra)} scoped)"

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


# Scoped retrieval for the multi-agent arm; retrieve() above stays the
# baseline's. Scopes add entities, never filter: a university+scholarship
# scope would hide SAT registration fees (on the exam body's page).

CATEGORY_BOOST = 3      # ranks a preferred-category chunk climbs, at most


@dataclass
class RetrievalSpec:
    query: str
    k: int = 5
    entities: list[str] | None = None            # from the planner
    expand_types: list[str] | None = None        # agent's authority expansion
    programme: str | None = None                 # None = infer from the query
    prefer_categories: list[str] | None = None   # ranking prior, never a filter
    mode: str = "dense"
    expand: int = 1


_by_type: dict[str, list[str]] | None = None


def entities_by_type(entity_type: str) -> list[str]:
    """Entity ids of a given type, from the corpus. Not entities.json's
    linked_entity_ids: every university lists ['sat', 'hec_nbs'] and none
    list hec_ehsaas, so Ehsaas questions never reached its pages."""
    global _by_type
    if _by_type is None:
        corpus = get_corpus(get_collection())
        _by_type = {}
        for m in corpus.metas:
            _by_type.setdefault(m.get("entity_type", ""), [])
            eid = m.get("entity_id")
            if eid and eid not in _by_type[m.get("entity_type", "")]:
                _by_type[m.get("entity_type", "")].append(eid)
    return list(_by_type.get(entity_type, []))


def _boost_categories(results: list[dict], prefer: list[str]) -> list[dict]:
    """Move preferred-category chunks up, at most CATEGORY_BOOST ranks. Not a
    filter: content_category matches the text only 45% of the time for
    eligibility, 32% for test_format."""
    if not prefer:
        return results
    wanted = set(prefer)
    ranked = [
        (rank - (CATEGORY_BOOST if r.get("meta", {}).get("content_category") in wanted else 0),
         rank, r)
        for rank, r in enumerate(results)
    ]
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in ranked]


def search(spec: RetrievalSpec) -> list[dict]:
    """Scoped retrieval for a specialist agent."""
    ollama_client = get_ollama()
    collection = get_collection()
    corpus = get_corpus(collection)
    qvec = embed_query(spec.query, ollama_client)

    targets = list(spec.entities or [])
    for t in spec.expand_types or []:
        for eid in entities_by_type(t):
            if eid not in targets:
                targets.append(eid)

    # over-retrieve before the programme filter, then trim: filtering k hits
    # can leave nothing (a "BS CS fee" query whose top hits are all BS Biology)
    wanted = frozenset(_query_programme(spec.programme or spec.query))
    scoping = bool(wanted)
    pool_k = spec.k * 4 if scoping else spec.k

    bm25_q = clean_for_bm25(spec.query, targets)
    if targets:
        results = []
        for eid in targets:
            results.extend(
                hybrid_one(corpus, collection, qvec, bm25_q, pool_k, eid, spec.mode))
    else:
        results = hybrid_one(corpus, collection, qvec, bm25_q, pool_k, None, spec.mode)

    # fetch the named programme's pages directly and prepend them, so the
    # per-entity trim keeps them
    if wanted:
        seen = {r["id"] for r in results}
        extra = []
        for eid in (targets or [None]):
            for h in _programme_hits(corpus, collection, qvec, bm25_q,
                                     wanted, spec.k, eid):
                if h["id"] not in seen:
                    seen.add(h["id"])
                    extra.append(h)
        results = extra + results

    results = _boost_categories(results, spec.prefer_categories or [])
    results = scope_to_programme(spec.programme or spec.query, results, drop_other=True)

    if scoping:                     # keep k per entity, not k overall
        # cap programme and neutral pages separately: FAST's fees are on a page
        # naming no programme, and with one cap BS CS curriculum chunks pushed
        # the fee table out
        per_group: dict[tuple[str, bool], int] = {}
        trimmed = []
        for r in results:
            meta = r.get("meta", {})
            key = (meta.get("entity_id", "?"),
                   bool(_page_programme(meta.get("source_url", ""))))
            if per_group.get(key, 0) >= spec.k:
                continue
            per_group[key] = per_group.get(key, 0) + 1
            trimmed.append(r)
        results = trimmed

    if spec.expand:
        results = expand_neighbours(corpus, results, window=spec.expand)
    return results


def scan(entity_types: list[str] | None = None,
         entities: list[str] | None = None,
         categories: list[str] | None = None,
         mentions_exam: str | None = None,
         mentions_scholarship: str | None = None,
         mentions_programme: str | None = None,
         limit: int | None = None) -> list[dict]:
    """Metadata-only lookup for enumerations ("which universities accept the
    SAT"); dense search answers those with the exam body's own pages.
    Filtered in Python (Chroma can't substring-match the mentioned_* fields).
    Recall is limited to the crawler's KNOWN_* lists, so callers confirm each
    hit with search()."""
    corpus = get_corpus(get_collection())
    out = []
    for i, meta in enumerate(corpus.metas):
        if entity_types and meta.get("entity_type") not in entity_types:
            continue
        if entities and meta.get("entity_id") not in entities:
            continue
        if categories and meta.get("content_category") not in categories:
            continue
        for term, field in ((mentions_exam, "mentioned_exams"),
                            (mentions_scholarship, "mentioned_scholarships"),
                            (mentions_programme, "mentioned_programs")):
            if term and term.lower() not in str(meta.get(field, "")).lower():
                break
        else:
            out.append({"id": corpus.ids[i], "doc": corpus.docs[i], "meta": meta,
                        "in_dense": False, "in_bm25": False})
            if limit and len(out) >= limit:
                break
    return out


def entities_mentioning(field_term: str, kind: str = "exam",
                        entity_types: list[str] | None = None) -> list[str]:
    """Entities that mention a term, most mentions first. The candidate step
    of an enumeration."""
    kwargs = {f"mentions_{kind}": field_term, "entity_types": entity_types}
    counts: dict[str, int] = {}
    for c in scan(**kwargs):
        eid = c["meta"].get("entity_id")
        if eid:
            counts[eid] = counts.get(eid, 0) + 1
    return sorted(counts, key=lambda e: -counts[e])


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