"""Unit tests for the sizing model (pillar 1) — pure calculation functions, no I/O."""

import math
from pathlib import Path

import pytest
import yaml

import sizing_model as sm

CONFIG = Path(__file__).resolve().parent.parent / "sizing" / "configs" / "reference_2b.yaml"


@pytest.fixture
def cfg():
    return yaml.safe_load(CONFIG.read_text())


# --- model_weights_gib -----------------------------------------------------
def test_weights_fp16_70b():
    # 70e9 parameters × 2 bytes / 2^30 ≈ 130.4 GiB
    assert sm.model_weights_gib(70.0, "fp16") == pytest.approx(70e9 * 2 / sm.GIB, rel=1e-9)
    assert sm.model_weights_gib(70.0, "fp16") == pytest.approx(130.385, abs=0.01)


def test_weights_fp8_is_half_of_fp16():
    assert sm.model_weights_gib(70.0, "fp8") == pytest.approx(sm.model_weights_gib(70.0, "fp16") / 2)


def test_weights_int4_is_quarter():
    assert sm.model_weights_gib(70.0, "int4") == pytest.approx(sm.model_weights_gib(70.0, "fp16") / 4)


def test_weights_unknown_quant_raises():
    with pytest.raises(ValueError, match="quantization"):
        sm.model_weights_gib(70.0, "fp3")


# --- kv_cache_per_request_gib ---------------------------------------------
def test_kv_cache_known_value():
    infra = {"num_layers": 80, "hidden_size": 8192, "num_attn_heads": 64,
             "num_kv_heads": 8, "kv_dtype_bytes": 1}
    # 2 × 80 × 8192 × (8/64) × 1000 × 1 byte = 163_840_000 bytes
    expected = 2 * 80 * 8192 * 0.125 * 1000 * 1 / sm.GIB
    assert sm.kv_cache_per_request_gib(infra, 1000) == pytest.approx(expected, rel=1e-9)


def test_kv_cache_scales_linearly_with_seqlen():
    infra = {"num_layers": 80, "hidden_size": 8192, "num_attn_heads": 64,
             "num_kv_heads": 8, "kv_dtype_bytes": 1}
    a = sm.kv_cache_per_request_gib(infra, 1000)
    b = sm.kv_cache_per_request_gib(infra, 2000)
    assert b == pytest.approx(2 * a)


def test_kv_cache_gqa_ratio_matters():
    base = {"num_layers": 80, "hidden_size": 8192, "num_attn_heads": 64, "kv_dtype_bytes": 2}
    mha = sm.kv_cache_per_request_gib({**base, "num_kv_heads": 64}, 1000)
    gqa = sm.kv_cache_per_request_gib({**base, "num_kv_heads": 8}, 1000)
    assert gqa == pytest.approx(mha / 8)


# --- work_seconds_per_month ------------------------------------------------
def test_work_seconds():
    assert sm.work_seconds_per_month({"work_days_per_month": 22, "work_hours_per_day": 8}) == 22 * 8 * 3600


# --- build_scenarios -------------------------------------------------------
def test_build_scenarios_semvec_needs_fewer_gpus(cfg):
    naive, semvec, meta = sm.build_scenarios(cfg)
    assert semvec.gpus_required <= naive.gpus_required
    assert meta["input_reduction_pct"] > 0


def test_input_output_split_matches_ratio(cfg):
    naive, semvec, meta = sm.build_scenarios(cfg)
    total = meta["total_tokens_month"]
    ratio = cfg["workload"]["in_out_ratio"]
    # output share = 1/(ratio+1)
    assert meta["output_month"] == pytest.approx(total / (ratio + 1.0))
    assert meta["naive_input_month"] == pytest.approx(total * ratio / (ratio + 1.0))


def test_output_unchanged_between_scenarios(cfg):
    naive, semvec, _ = sm.build_scenarios(cfg)
    assert naive.output_tokens_month == pytest.approx(semvec.output_tokens_month)


def test_semvec_input_reduced_vs_naive(cfg):
    naive, semvec, _ = sm.build_scenarios(cfg)
    assert semvec.input_tokens_month < naive.input_tokens_month


def test_reduction_pct_matches_turn_ratio(cfg):
    naive, semvec, meta = sm.build_scenarios(cfg)
    wl = cfg["workload"]
    expected = (1 - wl["semvec_input_tokens_per_turn"] / wl["naive_mean_input_tokens_per_turn"]) * 100
    assert meta["input_reduction_pct"] == pytest.approx(expected, rel=1e-6)


def test_total_tokens_plausi_2b(cfg):
    # 1500 users × 1m = 1.5 bn (sanity check against the 2b request, same order of magnitude)
    _, _, meta = sm.build_scenarios(cfg)
    assert meta["total_tokens_month"] == 1500 * 1_000_000


# --- ScenarioResult properties ---------------------------------------------
def test_binding_is_max_bound():
    bounds = [sm.Bound("prefill", 10.0, ""), sm.Bound("decode", 3.0, ""), sm.Bound("kv-mem", 25.0, "")]
    r = sm.ScenarioResult("x", 0, 0, bounds, 0, 0, 0)
    assert r.binding.name == "kv-mem"
    assert r.gpus_required == 25


def test_gpus_required_rounds_up_and_min_one():
    bounds = [sm.Bound("prefill", 0.2, ""), sm.Bound("decode", 0.1, ""), sm.Bound("kv-mem", 0.3, "")]
    r = sm.ScenarioResult("x", 0, 0, bounds, 0, 0, 0)
    assert r.gpus_required == 1  # ceil(0.3)=1, min 1


def test_weights_exceeding_mem_marks_kv_infinite(cfg):
    # model that does not fit on a single GPU → kv-mem bound is infinite
    cfg["infra"]["gpu_mem_gb"] = 24.0  # < 65 GiB fp8 weights
    naive, semvec, _ = sm.build_scenarios(cfg)
    kv = {b.name: b for b in naive.bounds}["kv-mem"]
    assert math.isinf(kv.gpus)
    assert "does not fit" in kv.detail
