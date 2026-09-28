"""Ingestion pipeline config: paths, endpoints, chunk metadata schema."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DB_PATH = Path(os.environ.get(
    "CHROMA_DB_PATH",
    str(PROJECT_ROOT / "data" / "chroma_db")
))

ENTITIES_CONFIG_PATH = Path(__file__).parent / "entities.json"

LOG_DIR = PROJECT_ROOT / "data" / "ingestion" / "logs"


CHROMA_COLLECTION_NAME = "pak_admissions_v2"  # v1 = 2-page prototype corpus


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_BATCH_SIZE = 16


MAX_CHUNK_CHARS = 3500   # force-split above this
MIN_CHUNK_CHARS = 300    # merge smaller fragments into neighbours


DEFAULT_RATE_LIMIT_SECONDS = 1.5
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_REQUEST_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 5


CRAWLER_VERSION = "0.2.0"


EntityType = Literal["university", "scholarship", "exam"]
FreshnessClass = Literal["stable", "slow_changing", "time_sensitive", "cycle_bound"]
PageType = Literal["webpage", "pdf", "faq_entry"]
InclusionCategory = Literal["primary", "secondary"]


class ChunkMetadata(BaseModel):
    """Metadata stored with each chunk."""

    # identity / provenance
    source_url: str
    source_domain: str
    scrape_date: str  # ISO 8601
    page_title: str
    crawler_version: str

    # routing
    entity_id: str
    entity_type: EntityType
    content_category: str  # "eligibility", "fees", ...
    freshness_class: FreshnessClass
    inclusion_category: InclusionCategory
    page_type: PageType
    cycle_year: Optional[int] = None

    # structure
    headings: list[str] = Field(default_factory=list)
    chunk_position: int
    total_chunks_in_page: int
    outbound_urls: list[str] = Field(default_factory=list)

    # cross-source linking
    linked_entity_ids: list[str] = Field(default_factory=list)

    # regex-extracted mentions
    mentioned_programs: list[str] = Field(default_factory=list)
    mentioned_scholarships: list[str] = Field(default_factory=list)
    mentioned_exams: list[str] = Field(default_factory=list)
    mentioned_dates: list[str] = Field(default_factory=list)
    mentioned_amounts: list[str] = Field(default_factory=list)

    def to_chroma_metadata(self) -> dict:
        """Chroma only takes scalars: lists become comma-joined strings."""
        d = self.model_dump()
        for key, value in list(d.items()):
            if isinstance(value, list):
                d[key] = ", ".join(value) if value else ""
            elif value is None:
                d[key] = ""
        return d


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
    "BSc Management Science", "BSc (Honours) Management Science",
    "BSc Accounting and Finance", "BSc (Honours) Accounting and Finance",
    "BA (Honours) Communication and Design", "Communication and Design",
    "BSc (Honours) Social Development and Policy", "Social Development and Policy",
    "BA (Honours) Comparative Humanities", "Comparative Humanities",
]

KNOWN_SCHOLARSHIPS = [
    "Ehsaas", "Benazir Undergraduate Scholarship", "BUS",
    "HEC Need-Based", "Need-Based Scholarship", "NBS",
    "PEEF", "Punjab Educational Endowment Fund",
    "NOP", "National Outreach Programme",
    "Yohsin", "HU TOPS", "HUTOPS", "HU EOP", "HUEOPS",
    "Excellence Scholarship", "NFAAF",
    "Honhaar",
    "Akhuwat", "Ihsan Trust", "Qarz-e-Hasna",
    "Merit Scholarship",
]

KNOWN_EXAMS = [
    "MDCAT", "NUST NET", "NET", "SAT", "ACT",
    "ECAT", "LCAT", "GAT",
    "NTS NAT", "NAT-IE", "NAT-ICS",
    "SBASSE Subject Test", "LGAT", "LAT",
    "BCAT", "FAST Entry Test", "FAST NU Entry Test",
    "GIKI Admissions Test", "GCAT",
    "GMAT", "GRE",
    "KMU CAT", "STS", "SIBA",
]