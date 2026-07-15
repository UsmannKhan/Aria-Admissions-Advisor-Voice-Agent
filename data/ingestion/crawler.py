"""
Main ingestion script.

Reads entities.json, crawls each entity's URLs with Crawl4AI, cleans
content to Markdown, chunks semantically (respecting headings),
classifies freshness, extracts mentioned entities, embeds via Ollama,
and upserts into ChromaDB.

Idempotent: re-running overwrites existing chunks for the same source_url
+ chunk_position. Safe to re-run after partial failures.

Usage:
    python crawler.py                          # Run all entities
    python crawler.py --entity fast            # One entity (for debugging)
    python crawler.py --entity fast --limit 5  # Cap pages per entity
    python crawler.py --dry-run                # Skip embedding/DB writes
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import chromadb
import ollama
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import (
    DomainFilter,
    FilterChain,
    URLPatternFilter,
)

import config as cfg


# =============================================================================
# Logging
# =============================================================================


def setup_logging() -> logging.Logger:
    """Configure logging to both stderr and a timestamped log file."""
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = cfg.LOG_DIR / f"crawl_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("crawler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-runs

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    logger.addHandler(stderr_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Log file: %s", log_file)
    return logger


log = logging.getLogger("crawler")


# =============================================================================
# Data classes for in-memory chunk representation
# =============================================================================


@dataclass
class Chunk:
    """A chunk of content ready for embedding."""
    text: str
    headings: list[str] = field(default_factory=list)
    chunk_position: int = 0
    total_chunks_in_page: int = 0


@dataclass
class CrawledPage:
    """Output of crawling a single URL."""
    url: str
    title: str
    markdown: str
    outbound_urls: list[str] = field(default_factory=list)


# =============================================================================
# URL filtering (per-entity exclusions)
# =============================================================================


def url_is_excluded(url: str, excluded_paths: list[str]) -> bool:
    """True if any excluded path substring matches this URL."""
    path = urlparse(url).path.lower()
    for ex in excluded_paths:
        if ex.lower() in path:
            return True
    return False


def url_is_secondary(url: str, secondary_paths: list[str]) -> bool:
    """True if URL matches a secondary-content path pattern."""
    path = urlparse(url).path.lower()
    return any(s.lower() in path for s in secondary_paths)


# =============================================================================
# Markdown chunker (semantic + size-bounded)
# =============================================================================


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split a too-large block by paragraphs, then sentences if needed."""
    if len(text) <= max_chars:
        return [text]

    # Split on paragraph boundaries first.
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            if buffer:
                chunks.append(buffer.strip())
            if len(para) <= max_chars:
                buffer = para
            else:
                # Paragraph itself is oversize then split on sentences.
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub_buf = ""
                for sent in sentences:
                    if len(sub_buf) + len(sent) + 1 <= max_chars:
                        sub_buf = f"{sub_buf} {sent}" if sub_buf else sent
                    else:
                        if sub_buf:
                            chunks.append(sub_buf.strip())
                        sub_buf = sent
                if sub_buf:
                    chunks.append(sub_buf.strip())
                buffer = ""
    if buffer:
        chunks.append(buffer.strip())
    return chunks

# =============================================================================
# Chunk cleanup: nav-boilerplate filtering + deduplication
# =============================================================================

_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")  # markdown links [text](url)


def _is_nav_boilerplate(text: str, headings: list[str]) -> bool:
    """Nav boilerplate = no heading path and link-dominated."""
    if headings:           # anything under a heading is treated as content
        return False

    stripped = text.strip()
    if not stripped:
        return True        # genuinely empty

    links = _LINK_RE.findall(stripped)
    link_chars = sum(len(m) for m in links)
    link_ratio = link_chars / max(len(stripped), 1)

    # No heading, plus either link-heavy text or just lots of links -> nav.
    return link_ratio > 0.4 or len(links) >= 8

def _content_key(text: str) -> str:
    """Key for near-duplicate detection: first 300 non-whitespace chars.
    Page-duplicated content shares a long identical opening even when
    chunk boundaries differ slightly at the tail."""
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    t = re.sub(r"\s+", "", t).lower()
    return t[:300]


def _dedupe_and_clean(chunks: list["Chunk"]) -> list["Chunk"]:
    seen_hashes: set[str] = set()
    cleaned: list[Chunk] = []
    dropped_nav = 0
    dropped_dup = 0

    for ch in chunks:
        if _is_nav_boilerplate(ch.text, ch.headings):
            dropped_nav += 1
            log.info("    DROP[nav]: headings=%s | %s", ch.headings, ch.text[:80].replace("\n", " "))
            continue
        h = _content_key(ch.text)
        if h in seen_hashes:
            dropped_dup += 1
            log.info("    DROP[dup]: headings=%s | %s", ch.headings, ch.text[:80].replace("\n", " "))
            continue
        seen_hashes.add(h)
        cleaned.append(ch)

    if dropped_nav or dropped_dup:
        log.info("  cleanup: dropped %d nav + %d duplicate chunks (%d -> %d)",
                 dropped_nav, dropped_dup, len(chunks), len(cleaned))
    return cleaned


def chunk_markdown(markdown: str) -> list[Chunk]:
    """
    Chunk Markdown text semantically.

    Strategy:
      1. Walk the document, tracking heading hierarchy (H1, H2, H3).
      2. Group content under headings into "sections".
      3. Emit each section as a chunk, splitting oversized sections by
         paragraph (and then sentence) boundaries.
      4. Tag each chunk with the heading path it belongs to.
    """
    if not markdown or not markdown.strip():
        return []

    lines = markdown.splitlines()
    sections: list[tuple[list[str], list[str]]] = []  # (heading_path, content_lines)
    heading_stack: list[tuple[int, str]] = []  # [(level, text), ...]
    current_content: list[str] = []

    def flush_section() -> None:
        if current_content and any(line.strip() for line in current_content):
            sections.append(
                ([h[1] for h in heading_stack], list(current_content))
            )

    for line in lines:
        heading_match = HEADING_RE.match(line)
        if heading_match:
            # Save what we have under the previous heading path.
            flush_section()
            current_content = []

            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()

            # Pop deeper-or-equal headings, then push.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
        else:
            current_content.append(line)
    flush_section()

    # Convert sections to chunks, splitting oversized ones.
    chunks: list[Chunk] = []
    for heading_path, content_lines in sections:
        text = "\n".join(content_lines).strip()
        if not text:
            continue

        parts = _split_oversized(text, cfg.MAX_CHUNK_CHARS)
        for part in parts:
            if len(part) < cfg.MIN_CHUNK_CHARS and chunks:
                prev = chunks[-1]
                same_path = prev.headings == heading_path
                # Same parent: share all heading levels except the last leaf.
                same_parent = (
                    len(heading_path) >= 2
                    and len(prev.headings) >= 2
                    and prev.headings[:-1] == heading_path[:-1]
                )
                if (same_path or same_parent) and (
                    len(prev.text) + len(part) + 2 <= cfg.MAX_CHUNK_CHARS
                ):
                    prev.text = f"{prev.text}\n\n{part}"
                    continue
            chunks.append(Chunk(text=part, headings=heading_path))

    # Remove nav boilerplate and duplicates before numbering, so positions and totals reflect the final chunk set.
    chunks = _dedupe_and_clean(chunks)

    # Fill in chunk_position / total_chunks_in_page.
    for i, ch in enumerate(chunks):
        ch.chunk_position = i
        ch.total_chunks_in_page = len(chunks)
    return chunks


# =============================================================================
# Freshness classification
# =============================================================================


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def classify_freshness(
    url: str,
    text: str,
    universal: dict,
) -> tuple[cfg.FreshnessClass, int | None]:
    """
    Return (freshness_class, cycle_year).

    When uncertain, bias toward time_sensitive: better to re-check a date
    than serve a stale one.
    """
    cycle_year: int | None = None

    # URL path -> time_sensitive?
    path = urlparse(url).path.lower()
    for pat in universal.get("time_sensitive_path_patterns", []):
        if pat.lower() in path:
            return "time_sensitive", cycle_year

    # Cycle markers in URL?
    for pat_str in universal.get("cycle_markers", {}).get("url_patterns", []):
        m = re.search(pat_str, url)
        if m:
            ym = re.search(r"\d{4}", m.group(0))
            if ym:
                cycle_year = int(ym.group(0))
            return "cycle_bound", cycle_year

    # Cycle markers in content?
    for pat_str in universal.get("cycle_markers", {}).get("content_patterns", []):
        m = re.search(pat_str, text, re.IGNORECASE)
        if m:
            ym = re.search(r"\d{4}", m.group(0))
            if ym:
                cycle_year = int(ym.group(0))
            return "cycle_bound", cycle_year

    # Date-heavy content -> time_sensitive
    date_count = len(re.findall(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|"
        r"Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}",
        text,
    ))
    if date_count >= 3:
        return "time_sensitive", cycle_year

    return "slow_changing", cycle_year


# =============================================================================
# Content category (lightweight heuristic)
# =============================================================================


def classify_content_category(url: str, headings: list[str]) -> str:
    """Categorise a chunk based on URL path + heading hierarchy."""
    path = urlparse(url).path.lower()
    heading_text = " ".join(headings).lower()
    haystack = f"{path} {heading_text}"

    categories = [
        ("eligibility", ["eligibility", "criteria", "eligible", "requirement"]),
        ("fees", ["fee", "tuition", "cost", "payment"]),
        ("scholarship_info", ["scholarship", "financial-aid", "financial aid", "loan", "stipend"]),
        ("test_format", ["test pattern", "test-pattern", "syllabus", "what's on", "structure", "calculator"]),
        ("schedule", ["dates", "schedule", "deadline", "calendar"]),
        ("application_process", ["how to apply", "apply", "how-to-apply", "application", "registration"]),
        ("programs", ["program", "programme", "degree", "course", "majors", "offered"]),
        ("scoring", ["scoring", "scores", "grading", "merit"]),
        ("faq", ["faq", "frequently asked"]),
        ("documents", ["documents", "checklist", "supporting"]),
        ("test_rules", ["testing-rules", "what to bring", "id-requirements", "prohibited"]),
    ]
    for cat, kws in categories:
        if any(kw in haystack for kw in kws):
            return cat
    return "general"


# =============================================================================
# Entity extraction (regex over known lists)
# =============================================================================


def _build_term_regex(terms: list[str]) -> re.Pattern:
    """Build a single case-insensitive regex for a list of terms."""
    escaped = sorted({re.escape(t) for t in terms}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


_PROGRAM_RE = _build_term_regex(cfg.KNOWN_PROGRAMS)
_SCHOLARSHIP_RE = _build_term_regex(cfg.KNOWN_SCHOLARSHIPS)
_EXAM_RE = _build_term_regex(cfg.KNOWN_EXAMS)

_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+\d{1,2}(?:,?\s+\d{4})?\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(
    r"(?:PKR|Rs\.?|USD|\$)\s*[\d,]+(?:\.\d+)?"
    r"|\b\d{1,3}(?:,\d{3})+\s*(?:PKR|Rs)\b"
    r"|\b\d{1,3}%\b",
    re.IGNORECASE,
)


def extract_mentions(text: str) -> dict[str, list[str]]:
    """Pull out programs/scholarships/exams/dates/amounts mentioned in text."""
    def _dedupe_caseless(matches: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for m in matches:
            key = m.lower()
            if key not in seen:
                seen.add(key)
                out.append(m.strip())
        return out

    return {
        "programs": _dedupe_caseless(_PROGRAM_RE.findall(text)),
        "scholarships": _dedupe_caseless(_SCHOLARSHIP_RE.findall(text)),
        "exams": _dedupe_caseless(_EXAM_RE.findall(text)),
        "dates": _dedupe_caseless(_DATE_RE.findall(text)),
        "amounts": _dedupe_caseless(_AMOUNT_RE.findall(text)),
    }


# =============================================================================
# Crawl4AI integration
# =============================================================================


def build_browser_config(ssl_verify: bool) -> BrowserConfig:
    """Per-entity browser config. HEC needs ssl_verify=False."""
    return BrowserConfig(
        headless=True,
        verbose=False,
        ignore_https_errors=(not ssl_verify),
        user_agent=(
            "Mozilla/5.0 (compatible; PakAdmissionsBot/0.1; "
            "research; FYP University of London)"
        ),
    )


def build_run_config(entity_crawl: dict, universal: dict, depth: int) -> CrawlerRunConfig:
    """Per-entity run config for deep crawling."""
    excluded = (
        universal.get("universal_excluded_paths", [])
        + entity_crawl.get("additional_excluded_paths", [])
    )
    allowed_domains = entity_crawl.get("allowed_domains", [])

    filter_chain = FilterChain([
        DomainFilter(allowed_domains=allowed_domains),
        URLPatternFilter(
            patterns=[f"*{p}*" for p in excluded],
            reverse=True,  # block these patterns
        ),
    ])

    deep_strategy = BFSDeepCrawlStrategy(
        max_depth=depth,
        include_external=False,
        filter_chain=filter_chain,
    )

    return CrawlerRunConfig(
        deep_crawl_strategy=deep_strategy,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        page_timeout=cfg.DEFAULT_REQUEST_TIMEOUT * 1000,  # ms
        wait_until="domcontentloaded",
    )


async def crawl_entity(entity_id: str, entity: dict, universal: dict,
                       limit: int | None = None) -> list[CrawledPage]:
    """Crawl all URLs for one entity. Returns deduplicated CrawledPage list."""
    entity_crawl = entity["crawl"]
    root_urls = entity_crawl.get("root_urls", [])
    if not root_urls:
        log.info("[%s] no root URLs configured, skipping crawl", entity_id)
        return []

    ssl_verify = entity_crawl.get("ssl_verify", True)
    depth = entity_crawl.get("depth", 2)
    browser_cfg = build_browser_config(ssl_verify)
    run_cfg = build_run_config(entity_crawl, universal, depth)

    seen_urls: set[str] = set()
    pages: list[CrawledPage] = []

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for root_url in root_urls:
            if limit is not None and len(pages) >= limit:
                break
            log.info("[%s] crawling root: %s", entity_id, root_url)
            try:
                results = await crawler.arun(url=root_url, config=run_cfg)
            except Exception as exc:
                log.error("[%s] crawl failed for %s: %s", entity_id, root_url, exc)
                continue

            # arun may return a single result or list depending on deep_crawl
            result_list = results if isinstance(results, list) else [results]

            for result in result_list:
                if limit is not None and len(pages) >= limit:
                    break
                if not getattr(result, "success", False):
                    log.warning("[%s] fetch failed: %s", entity_id, getattr(result, "url", "?"))
                    continue
                url = result.url
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # crawl4ai may expose markdown as str OR object with raw_markdown.
                md_obj = getattr(result, "markdown", None)
                if md_obj is None:
                    continue
                markdown = (
                    md_obj.raw_markdown
                    if hasattr(md_obj, "raw_markdown")
                    else str(md_obj)
                )
                if not markdown or len(markdown.strip()) < 100:
                    log.debug("[%s] skipping near-empty page: %s", entity_id, url)
                    continue

                title = (result.metadata or {}).get("title", "") if hasattr(result, "metadata") else ""
                links = getattr(result, "links", {}) or {}
                outbound = [
                    l.get("href", "") if isinstance(l, dict) else str(l)
                    for l in (links.get("external", []) or [])
                ][:50]  # cap to keep metadata reasonable

                pages.append(CrawledPage(
                    url=url,
                    title=title,
                    markdown=markdown,
                    outbound_urls=outbound,
                ))

    log.info("[%s] crawled %d unique pages", entity_id, len(pages))
    return pages


# =============================================================================
# Embedding via Ollama
# =============================================================================


def embed_batch(texts: list[str], client: ollama.Client) -> list[list[float]]:
    """Embed a batch of texts. Retries on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(cfg.DEFAULT_MAX_RETRIES):
        try:
            resp = client.embed(model=cfg.EMBEDDING_MODEL, input=texts)
            return resp["embeddings"]
        except Exception as exc:
            last_exc = exc
            sleep_for = cfg.DEFAULT_RETRY_BACKOFF * (2 ** attempt)
            log.warning(
                "embed attempt %d/%d failed (%s); sleeping %ds",
                attempt + 1, cfg.DEFAULT_MAX_RETRIES, exc, sleep_for,
            )
            time.sleep(sleep_for)
    raise RuntimeError(f"embedding failed after retries: {last_exc}")


# =============================================================================
# ChromaDB
# =============================================================================


def get_chroma_collection() -> chromadb.Collection:
    cfg.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DB_PATH))
    return client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_id_for(source_url: str, chunk_position: int) -> str:
    """Deterministic chunk ID -- re-runs overwrite same content."""
    raw = f"{source_url}#{chunk_position}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


# =============================================================================
# Page processing pipeline
# =============================================================================


def process_page(
    page: CrawledPage,
    entity_id: str,
    entity: dict,
    universal: dict,
) -> list[tuple[str, str, str, dict]]:
    """
    Turn one crawled page into (id, store_text, embed_text, metadata) tuples.
    store_text = clean body (stored); embed_text = entity+headings+body (embedded).
    """
    chunks = chunk_markdown(page.markdown)
    if not chunks:
        return []

    domain = urlparse(page.url).netloc
    is_secondary = url_is_secondary(
        page.url, universal.get("secondary_content_paths", [])
    )
    freshness, cycle_year = classify_freshness(page.url, page.markdown, universal)
    now_iso = datetime.now(timezone.utc).isoformat()
    entity_name = entity.get("display_name", entity_id)

    results: list[tuple[str, str, str, dict]] = []
    for chunk in chunks:
        mentions = extract_mentions(chunk.text)
        content_category = classify_content_category(page.url, chunk.headings)

        meta = cfg.ChunkMetadata(
            source_url=page.url,
            source_domain=domain,
            scrape_date=now_iso,
            page_title=page.title or page.url,
            crawler_version=cfg.CRAWLER_VERSION,
            entity_id=entity_id,
            entity_type=entity["type"],
            content_category=content_category,
            freshness_class=freshness,
            inclusion_category="secondary" if is_secondary else "primary",
            page_type="webpage",
            cycle_year=cycle_year,
            headings=chunk.headings,
            chunk_position=chunk.chunk_position,
            total_chunks_in_page=chunk.total_chunks_in_page,
            outbound_urls=page.outbound_urls[:20],
            linked_entity_ids=entity.get("linked_entity_ids", []),
            mentioned_programs=mentions["programs"],
            mentioned_scholarships=mentions["scholarships"],
            mentioned_exams=mentions["exams"],
            mentioned_dates=mentions["dates"],
            mentioned_amounts=mentions["amounts"],
        )

        # embed entity + heading path + body so context words enter the vector
        heading_ctx = " > ".join(chunk.headings) if chunk.headings else ""
        prefix = f"{entity_name} | {heading_ctx}".strip(" |")
        embed_text = f"{prefix}\n\n{chunk.text}" if prefix else chunk.text

        chunk_id = chunk_id_for(page.url, chunk.chunk_position)
        results.append((chunk_id, chunk.text, embed_text, meta.to_chroma_metadata()))
    return results


# =============================================================================
# Per-entity orchestration
# =============================================================================


async def ingest_entity(
    entity_id: str,
    entity: dict,
    universal: dict,
    ollama_client: ollama.Client | None,
    collection: chromadb.Collection | None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    End-to-end ingestion for one entity.
    Returns counters: {"pages": N, "chunks": M, "embedded": K}.
    """
    counters = {"pages": 0, "chunks": 0, "embedded": 0, "errors": 0}

    pages = await crawl_entity(entity_id, entity, universal, limit=limit)
    counters["pages"] = len(pages)
    if not pages:
        return counters

    # Process pages to chunks
    all_records: list[tuple[str, str, str, dict]] = []
    for page in pages:
        try:
            recs = process_page(page, entity_id, entity, universal)
            all_records.extend(recs)
        except Exception as exc:
            counters["errors"] += 1
            log.error("[%s] processing failed for %s: %s", entity_id, page.url, exc)
    counters["chunks"] = len(all_records)
    log.info("[%s] produced %d chunks from %d pages", entity_id, len(all_records), len(pages))

    if dry_run or not all_records:
        return counters

    # embed embed_text (enriched) but store store_text (clean body)
    assert ollama_client is not None and collection is not None
    for i in range(0, len(all_records), cfg.EMBEDDING_BATCH_SIZE):
        batch = all_records[i:i + cfg.EMBEDDING_BATCH_SIZE]
        ids = [r[0] for r in batch]
        store_texts = [r[1] for r in batch]   # clean body -> stored document
        embed_texts = [r[2] for r in batch]   # entity+headings+body -> embedded
        metas = [r[3] for r in batch]
        try:
            vectors = embed_batch(embed_texts, ollama_client)
        except Exception as exc:
            counters["errors"] += 1
            log.error("[%s] embedding batch failed: %s", entity_id, exc)
            continue
        try:
            collection.upsert(
                ids=ids,
                documents=store_texts,
                metadatas=metas,
                embeddings=vectors,
            )
            counters["embedded"] += len(batch)
        except Exception as exc:
            counters["errors"] += 1
            log.error("[%s] chroma upsert failed: %s", entity_id, exc)
    log.info("[%s] embedded+stored %d chunks", entity_id, counters["embedded"])
    return counters


# =============================================================================
# Main
# =============================================================================


def load_entities_config() -> dict:
    with open(cfg.ENTITIES_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    log.info("crawler version %s starting", cfg.CRAWLER_VERSION)

    config_data = load_entities_config()
    universal = config_data.get("universal_settings", {})
    entities = config_data.get("entities", {})

    if args.entity:
        if args.entity not in entities:
            log.error("entity '%s' not found in config. available: %s",
                      args.entity, list(entities.keys()))
            return 1
        entities = {args.entity: entities[args.entity]}

    ollama_client: ollama.Client | None = None
    collection: chromadb.Collection | None = None
    if not args.dry_run:
        log.info("ollama host: %s", cfg.OLLAMA_HOST)
        log.info("chroma path: %s", cfg.CHROMA_DB_PATH)
        ollama_client = ollama.Client(host=cfg.OLLAMA_HOST)
        # Sanity check: model is pulled
        try:
            ollama_client.show(cfg.EMBEDDING_MODEL)
        except Exception as exc:
            log.error(
                "ollama model '%s' not available (run: ollama pull %s). %s",
                cfg.EMBEDDING_MODEL, cfg.EMBEDDING_MODEL, exc,
            )
            return 2
        collection = get_chroma_collection()
        log.info("chroma collection '%s' ready (current count: %d)",
                 cfg.CHROMA_COLLECTION_NAME, collection.count())

    totals = {"pages": 0, "chunks": 0, "embedded": 0, "errors": 0}
    for eid, entity in entities.items():
        log.info("=" * 60)
        log.info("ENTITY: %s (%s)", eid, entity.get("display_name", ""))
        log.info("=" * 60)
        try:
            counters = await ingest_entity(
                eid, entity, universal,
                ollama_client, collection,
                limit=args.limit, dry_run=args.dry_run,
            )
            for k, v in counters.items():
                totals[k] = totals.get(k, 0) + v
        except Exception as exc:
            log.exception("[%s] entity-level failure: %s", eid, exc)
            totals["errors"] += 1

    log.info("=" * 60)
    log.info("RUN COMPLETE. pages=%d chunks=%d embedded=%d errors=%d",
             totals["pages"], totals["chunks"], totals["embedded"], totals["errors"])
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pak admissions ingestion pipeline")
    p.add_argument(
        "--entity",
        help="Run only this entity (e.g. 'fast'). Default: all entities.",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Max pages per entity (debugging).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Crawl and chunk but skip embedding/DB writes.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()