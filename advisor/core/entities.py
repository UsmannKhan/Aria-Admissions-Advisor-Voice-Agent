"""Deterministic entity detection.

Separate from retrieval so it doesn't import ChromaDB, Ollama and BM25. Runs
after the supervisor's rewrite: raw STT misses names ("Habib" heard as
"hadees").
"""

from __future__ import annotations

import re

ENTITY_ALIASES: dict[str, list[str]] = {
    "lums": ["lums", "lahore university of management",
             "لمز", "لُمز", "ایل یو ایم ایس"],
    # Habib isn't in the corpus (site blocks the crawler). The alias scopes
    # retrieval to an entity with no chunks, so the answer says it isn't
    # covered instead of using another university's pages.
    "habib": ["habib", "habib university",
              "حبیب", "حبیب یونیورسٹی", "حبيب"],
    "nust": ["nust", "national university of sciences",
             "نست", "نسٹ"],
    # no bare "fast": it's an ordinary word ("how fast do they reply"). The
    # supervisor's rewrite resolves FAST from context.
    "fast": ["fast nuces", "fast-nuces", "fast university", "fast nu",
             "fast-nu", "nuces", "national university of computer",
             "فاسٹ"],
    "sat": ["sat", "scholastic aptitude test", "college board",
            "ایس اے ٹی", "کالج بورڈ"],
    "hec_nbs": ["hec need-based", "hec need based", "need-based scholarship",
                "need based scholarship", "nbs scholarship",
                "ایچ ای سی", "ضرورت کی بنیاد"],
    "hec_ehsaas": ["ehsaas", "ehsas", "benazir undergraduate",
                   "bisp scholarship", "احساس", "بینظیر"],
}

# Short aliases that are substrings of common words, so they must match on a
# word boundary, not as a bare substring:
#   "sat" in "satisfy", "net" in "internet", "nbs" in ...  -> false positives.
_WORD_BOUNDARY_ALIASES = {"sat", "net", "nbs", "ehsas", "nust", "lums", "habib",
                          "nuces", "fast nu"}


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
