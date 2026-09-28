"""Unit tests for the deterministic parts of the advisor and the evaluation.

No API key, GPU, Ollama or Chroma needed: everything here is pure Python.

    python -m pytest tests -v
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["ADVISOR_VERIFY"] = "off"     # read at import; keeps assemble() offline

import pytest

from advisor.core.entities import detect_entities
from advisor.multi_agent import supervisor
from advisor.multi_agent import verify as V
import eval_score as E

FEE_TABLE = {"doc": "| Particulars | Fall 2026 | Spring 2027 | Total |\n"
                    "| Tuition Fee | 824,400 | 732,800 | 1,557,200 |\n"
                    "| Total | 1,258,700 | 923,500 | 2,182,200 |",
             "meta": {"entity_id": "lums", "headings": "Fee Structure"}}


# entity detection

def test_detects_english_and_urdu_aliases():
    assert detect_entities("compare CS at LUMS and NUST") == ["lums", "nust"]
    assert detect_entities("لمز کی فیس کتنی ہے؟") == ["lums"]


def test_short_aliases_need_a_word_boundary():
    # "sat" inside "satisfy" and "net" inside "internet" are not entities
    assert detect_entities("will this satisfy the internet requirement") == []
    assert "nust" in detect_entities("what is the NET fee")


def test_bare_fast_is_not_an_institution():
    assert detect_entities("how fast do they reply") == []
    assert detect_entities("fees at FAST NUCES") == ["fast"]


# numeric verification (V2)

def test_numbers_reads_urdu_digits_and_spoken_scales():
    assert V.numbers("ٹیوشن ۱٬۵۵۷٬۲۰۰ روپے") == {1557200.0}
    assert V.numbers("about 8 lakh") == {800000.0}


def test_figure_verbatim_in_evidence_passes():
    claim = {"text": "First-year tuition is PKR 1,557,200.", "sources": [1]}
    out = V.check_numeric(claim, [FEE_TABLE])
    assert out["absent"] == [] and 1557200.0 in out["verbatim"]


def test_sum_of_two_semesters_is_derived_not_absent():
    claim = {"text": "Both semesters come to 2,182,200.", "sources": [1]}
    pool = {1258700.0, 923500.0}
    assert V.derive(2182200.0, pool) is not None
    assert V.check_numeric(claim, [FEE_TABLE])["absent"] == []


def test_near_miss_figure_is_flagged():
    # the case the judge tends to miss: 1,555,000 against a table saying 1,557,200
    claim = {"text": "Tuition is PKR 1,555,000.", "sources": [1]}
    assert V.check_numeric(claim, [FEE_TABLE])["absent"] == [1555000.0]


def test_claim_without_resolvable_citation_is_unsupported():
    out = V.verify_claim({"text": "LUMS charges a fee.", "sources": [9]}, [FEE_TABLE])
    assert out["verdict"] == V.UNSUPPORTED and "no_citation" in out["flags"]


# policy gate and speech

def test_policy_gate_strips_source_markers_from_spoken_text():
    answer, violations = V.policy_gate("The fee is 45,800 (Source 3).", [],
                                       "en", judge=False)
    assert "Source" not in answer
    assert any("stripped source markers" in v for v in violations)


def test_policy_gate_catches_reintroduced_struck_figure():
    struck = [{"text": "The deposit is 99,000."}]
    _, violations = V.policy_gate("You also pay a deposit of 99,000.", struck,
                                  "en", judge=False)
    assert any("reintroduced" in v for v in violations)


def test_policy_gate_flags_wrong_language():
    _, violations = V.policy_gate("LUMS ki fee 45,800 hai.", [], "ur", judge=False)
    assert any("not in Urdu" in v for v in violations)


def test_speakable_swaps_digits_for_readings():
    out = V.speakable("فیس 70% ہے", [{"written": "70%", "spoken": "ستر فیصد"}])
    assert out == "فیس ستر فیصد ہے"


# assembly

def test_assemble_renumbers_sources_across_agents(monkeypatch):
    def offline(*_a, **_k):
        raise RuntimeError("no network in unit tests")
    monkeypatch.setattr(supervisor, "generate", offline)   # falls back to concat

    a = {"domain": "admissions", "answer": "A.", "abstained": False,
         "claims": [{"text": "LUMS needs 70%.", "sources": [1]}],
         "sources": [{"id": "l1", "meta": {"entity_id": "lums"}}]}
    b = {"domain": "cost", "answer": "B.", "abstained": False,
         "claims": [{"text": "NUST charges 216,750.", "sources": [1]}],
         "sources": [{"id": "n1", "meta": {"entity_id": "nust"}}]}
    out = supervisor.assemble({"raw_query": "q", "partials": [a, b]})

    assert [c["sources"] for c in out["claims"]] == [[1], [2]]
    assert [c["entity"] for c in out["claims"]] == ["lums", "nust"]
    assert len(out["sources"]) == 2


# evaluation statistics

def test_mcnemar_matches_hand_count():
    # 6 disagreements split 5 to 1: 14 of 64 equally likely sequences
    a = [1, 0, 0, 0, 0, 0, 1, 0]
    b = [0, 1, 1, 1, 1, 1, 1, 0]
    only_a, only_b, p = E.mcnemar_exact(a, b)
    assert (only_a, only_b) == (1, 5)
    assert p == pytest.approx(14 / 64)


def test_holm_adjustment():
    assert E._holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_interval_brackets_the_mean_and_detects_a_real_effect():
    import numpy as np
    rng = np.random.default_rng(0)
    mean, lo, hi, p = E.paired_bootstrap([0.5] * 20 + [0.0] * 20, rng)
    assert mean == pytest.approx(0.25) and lo < mean < hi and lo > 0 and p < 0.01
    _, lo0, hi0, p0 = E.paired_bootstrap([0.5, -0.5] * 20, rng)
    assert lo0 < 0 < hi0 and p0 > 0.5
