"""Unit tests for the LongBench-v2 harness (pillar 2, hard test) — pure logic, no LLM/network."""

import pytest

import run_longbench_qa as lb


# --- parse_mc_answer -------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("A", "A"),
    ("Answer: C", "C"),
    ("The answer is B.", "B"),
    ("D) Faisal's mother ...", "D"),
    ("(C)", "C"),
    ("<think>maybe B or D</think>\nAnswer: D", "D"),
    ("I think the correct option is (C).", "C"),
])
def test_parse_mc_answer_extracts_letter(text, expected):
    assert lb.parse_mc_answer(text) == expected


def test_parse_mc_answer_none_on_no_letter():
    assert lb.parse_mc_answer("Sorry, cannot determine.") is None
    assert lb.parse_mc_answer("") is None


# --- chunk_text ------------------------------------------------------------
def test_chunk_text_sizes_and_coverage():
    chunks = lb.chunk_text("a" * 2500, size=1000)
    assert [len(c) for c in chunks] == [1000, 1000, 500]
    assert "".join(chunks) == "a" * 2500


def test_chunk_text_empty():
    assert lb.chunk_text("", size=1000) == []


def test_chunk_text_smaller_than_size():
    assert lb.chunk_text("abc", size=1000) == ["abc"]


# --- mc_accuracy -----------------------------------------------------------
def test_mc_accuracy():
    rows = [
        {"gold": "A", "semvec_pred": "A"},
        {"gold": "B", "semvec_pred": "C"},
        {"gold": "C", "semvec_pred": "C"},
        {"gold": "D", "semvec_pred": None},
    ]
    assert lb.mc_accuracy(rows, "semvec_pred") == pytest.approx(0.5)


def test_mc_accuracy_skips_missing_pred_key():
    # rows where the prediction key is absent (e.g. naive infeasible) are excluded
    rows = [
        {"gold": "A", "naive_pred": "A"},
        {"gold": "B"},                       # naive infeasible — no key
        {"gold": "C", "naive_pred": "C"},
    ]
    assert lb.mc_accuracy(rows, "naive_pred") == pytest.approx(1.0)
    assert lb.mc_accuracy_n(rows, "naive_pred") == 2
