"""Ingestion pipeline config: paths, endpoints, chunk metadata schema."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Paths
# =============================================================================

# Project root: two levels up from this file (data/ingestion/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where ChromaDB persists its data.
CHROMA_DB_PATH = Path(os.environ.get(
    "CHROMA_DB_PATH",
    str(PROJECT_ROOT / "data" / "chroma_db")
))

# Where to write the entities config. Default is alongside this file.
ENTITIES_CONFIG_PATH = Path(__file__).parent / "entities.json"

# Where to write per-run logs
LOG_DIR = PROJECT_ROOT / "data" / "ingestion" / "logs"


# =============================================================================
# ChromaDB
# =============================================================================

CHROMA_COLLECTION_NAME = "pak_admissions_v1"


# =============================================================================
# Ollama (embedding service)
# =============================================================================

# Default Ollama host
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Embedding model name
EMBEDDING_MODEL = "bge-m3"

# Vector dimensionality
EMBEDDING_DIMENSIONS = 1024

# How many chunks to embed per Ollama call.
EMBEDDING_BATCH_SIZE = 16


# =============================================================================
# Chunking
# =============================================================================

# Target chunk size in characters.
TARGET_CHUNK_CHARS = 2000

# Hard upper bound. Chunks larger than this get force-split.
MAX_CHUNK_CHARS = 3500

# Minimum chunk size. Smaller fragments get merged with neighbours unless they're a complete heading section.
MIN_CHUNK_CHARS = 300


# =============================================================================
# Crawler defaults (per-entity overrides live in entities.json)
# =============================================================================

DEFAULT_RATE_LIMIT_SECONDS = 1.5
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_REQUEST_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 5


# =============================================================================
# Versioning
# =============================================================================

# Used in ChunkMetadata to identify which crawler version produced a given chunk.
CRAWLER_VERSION = "0.1.0"


# =============================================================================
# Chunk metadata schema
# =============================================================================

EntityType = Literal["university", "scholarship", "exam"]
FreshnessClass = Literal["stable", "slow_changing", "time_sensitive", "cycle_bound"]
PageType = Literal["webpage", "pdf", "faq_entry"]
InclusionCategory = Literal["primary", "secondary"]


class ChunkMetadata(BaseModel):
    """Metadata stored with each chunk. Shared with the PDF pipeline later."""

    # --- Identity / provenance ---
    source_url: str
    source_domain: str
    scrape_date: str  # ISO 8601 timestamp
    page_title: str
    crawler_version: str

    # --- Routing ---
    entity_id: str
    entity_type: EntityType
    content_category: str  # e.g. "eligibility", "fees", "scholarship_rules"
    freshness_class: FreshnessClass
    inclusion_category: InclusionCategory
    page_type: PageType
    cycle_year: Optional[int] = None

    # --- Structure ---
    headings: list[str] = Field(default_factory=list)
    chunk_position: int
    total_chunks_in_page: int
    outbound_urls: list[str] = Field(default_factory=list)

    # --- Cross-source linking ---
    linked_entity_ids: list[str] = Field(default_factory=list)

    # --- Cheap entity extraction (regex-based) ---
    mentioned_programs: list[str] = Field(default_factory=list)
    mentioned_scholarships: list[str] = Field(default_factory=list)
    mentioned_exams: list[str] = Field(default_factory=list)
    mentioned_dates: list[str] = Field(default_factory=list)
    mentioned_amounts: list[str] = Field(default_factory=list)

    def to_chroma_metadata(self) -> dict:
        """Flatten to scalar values for Chroma; lists become comma-joined strings."""
        d = self.model_dump()
        for key, value in list(d.items()):
            if isinstance(value, list):
                d[key] = ", ".join(value) if value else ""
            elif value is None:
                d[key] = ""
        return d


# =============================================================================
# Known entities for content extraction (regex-based)
# =============================================================================

KNOWN_PROGRAMS = [
    "BSCS", "BS CS", "BS(CS)", "BS Computer Science",
    "BS(AI)", "BS AI", "BS Artificial Intelligence",
    "BS(DS)", "BS Data Science",
    "BS(SE)", "BS Software Engineering",
    "BS(CY)", "BS Cyber Security",
    "BS(EE)", "BS Electrical Engineering",
    "BS(CE)", "BS Computer Engineering",
    "BS(CV)", "BS Civil Engineering",
    "BS(ME)", "BS Mechanical Engineering",
    "BBA", "BS(AF)", "BS Accounting and Finance",
    "BS(FinTech)", "BS Financial Technology",
    "BS(BA)", "BS Business Analytics",
    "BS Biology", "BS Chemistry", "BS Physics", "BS Mathematics",
    "BS Economics", "BS Psychology", "BS Sociology",
    "BS Bioinformatics", "BS Biotechnology",
    "BS Environmental Science", "BS Agriculture",
    "BS Food Science", "BS Liberal Arts",
    "BS Public Administration", "BS Mass Communication",
    "BS Tourism and Hospitality",
    "MBBS", "BDS",
    "B Architecture", "B Industrial Design",
    "BA-LLB", "LLB",
    "BS Applied Linguistics",
]

KNOWN_SCHOLARSHIPS = [
    "Ehsaas", "Benazir Undergraduate Scholarship", "BUS",
    "HEC Need-Based", "Need-Based Scholarship", "NBS",
    "PEEF", "Punjab Educational Endowment Fund",
    "NOP", "National Outreach Programme",
    "Yohsin", "HU TOPS",
    "Honhaar",
    "Akhuwat", "Ihsan Trust", "Qarz-e-Hasna",
    "Merit Scholarship",
]

KNOWN_EXAMS = [
    "MDCAT", "NUST NET", "NET", "SAT", "ACT",
    "ECAT", "LCAT", "GAT",
    "SBASSE Subject Test", "LGAT", "LAT",
    "BCAT", "FAST Entry Test", "FAST NU Entry Test",
    "GIKI Admissions Test", "GCAT",
    "GMAT", "GRE",
    "KMU CAT", "STS", "SIBA",
]