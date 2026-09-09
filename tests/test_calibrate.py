"""Unit tests for the vLLM calibration helper (pillar 1) — pure throughput math, no network."""

import pytest

import calibrate_vllm as cal


# --- tokens_per_s ----------------------------------------------------------
def test_tokens_per_s_basic():
    assert cal.tokens_per_s(1000, 2.0) == pytest.approx(500.0)


def test_tokens_per_s_per_gpu_divides():
    assert cal.tokens_per_s(1000, 2.0, num_gpus=2) == pytest.approx(250.0)


def test_tokens_per_s_zero_wall_raises():
    with pytest.raises(ValueError):
        cal.tokens_per_s(1000, 0.0)


def test_tokens_per_s_zero_gpus_raises():
    with pytest.raises(ValueError):
        cal.tokens_per_s(1000, 2.0, num_gpus=0)


# --- summarize_calibration -------------------------------------------------
def test_summarize_calibration():
    out = cal.summarize_calibration(
        prefill_tokens=12000, prefill_wall=1.0,
        decode_tokens=3000, decode_wall=1.0, num_gpus=1,
    )
    assert out["prefill_tokens_per_s_per_gpu"] == pytest.approx(12000)
    assert out["decode_tokens_per_s_per_gpu"] == pytest.approx(3000)


def test_summarize_calibration_two_gpus_halves():
    out = cal.summarize_calibration(
        prefill_tokens=12000, prefill_wall=1.0,
        decode_tokens=3000, decode_wall=1.0, num_gpus=2,
    )
    assert out["prefill_tokens_per_s_per_gpu"] == pytest.approx(6000)
    assert out["decode_tokens_per_s_per_gpu"] == pytest.approx(1500)
