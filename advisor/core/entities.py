"""Deterministic entity detection.

Separate from retrieval so name-matching a string does not pull in ChromaDB,
Ollama and BM25.

Stage 2 of disambiguation: the LLM normalises the query, then this matcher
extracts entities from the clean text. Deterministic so routing stays
inspectable. Brittle against raw STT noise ("Habib" -> "hadees") without the
LLM layer in front.
"""

from __future__ import annotations

import re

ENTITY_ALIASES: dict[str, list[str]] = {
    "lums": ["lums", "lahore university of management",
             "لمز", "لُمز", "ایل یو ایم ایس"],
    "habib": ["habib", "habib university",
              "حبیب", "حبیب یونیورسٹی", "حبيب"],
    "nust": ["nust", "national university of sciences",
             "نست", "نسٹ"],
    "sat": ["sat", "scholastic aptitude test", "college board",
            "ایس اے ٹی", "کالج بورڈ"],
    "hec_nbs": ["hec need-based", "hec need based", "need-based scholarship",
                "need based scholarship", "nbs scholarship",
                "ایچ ای سی", "ضرورت کی بنیاد"],
    "hec_ehsaas": ["ehsaas", "ehsas", "benazir undergraduate",
                   "bisp scholarship", "احساس", "بینظیر"],
}

# Short aliases that are substrings of common words, so they MUST match on a
# word boundary, not as a bare substring:
#   "sat" in "satisfy", "net" in "internet", "nbs" in ...  -> false positives.
_WORD_BOUNDARY_ALIASES = {"sat", "net", "nbs", "ehsas", "nust", "lums", "habib"}


def _alias_matches(alias: str, query_lc: str) -> bool:
    if alias in _WORD_BOUNDARY_ALIASES:
        # \b works for ASCII aliases; Urdu aliases fall through to substring
        return re.search(rf"\b{re.escape(alias)}\b", query_lc) is not None
    return alias in query_lc


def detect_entities(query: str) -> list[str]:
    q = query.lower()
    detected = [
        eid for eid, aliases in ENTITY_ALIASES.items()
        if any(_alias_matches(a, q) for a in aliases)
    ]
    # "NET" (the NUST entry test) is a strong nust signal on its own
    if "nust" not in detected and re.search(r"\bnet\b", q):
        detected.append("nust")
    return detected
