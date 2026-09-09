#!/usr/bin/env python3
"""Token-to-hardware sizing model — naive (full-context replay) vs. Semvec.

Derives the GPU demand and rough cost from workload + SLA + infra and compares
two scenarios:

  * "naive"  — the conversation history is resent on every turn; the input/turn
               grows with the conversation length.
  * "semvec" — Semvec keeps a fixed-size state in front of the LLM; the
               input/turn stays constant, independent of the conversation length.

For each scenario the GPU count is determined as the maximum of three bounds
(prefill-bound, decode-bound, kv-cache-mem-bound) and the respective *binding*
constraint is reported.

All inputs come from a YAML config (see sizing/configs/); nothing is hardcoded.
Runs fully offline/air-gapped (stdlib + PyYAML only).

Example:
    python sizing/sizing_model.py --config sizing/configs/reference_2b.yaml
    python sizing/sizing_model.py --config ... --sweep naive_mean_input_tokens_per_turn:1000:12000:1000
    python sizing/sizing_model.py --config ... --semvec-input-tokens 1620   # from pillar 2

IMPORTANT — calibration: prefill_tokens_per_s_per_gpu and
decode_tokens_per_s_per_gpu are calibratable constants from a real
micro-benchmark (vLLM/TensorRT-LLM on the target GPU). As long as
infra.calibrated == false, the output loudly marks them as "UNCALIBRATED".
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML missing — `pip install pyyaml` (or via semvec[...]).")

GIB = 1024 ** 3
QUANT_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Bound:
    """One of the three sizing bounds."""

    name: str           # "prefill" | "decode" | "kv-mem"
    gpus: float         # required GPU count (before rounding up)
    detail: str         # human-readable derivation


@dataclass
class ScenarioResult:
    label: str
    input_tokens_month: float
    output_tokens_month: float
    bounds: list[Bound]
    concurrency_per_gpu: float
    kv_per_request_gib: float
    weights_gib: float

    @property
    def binding(self) -> Bound:
        return max(self.bounds, key=lambda b: b.gpus)

    @property
    def gpus_required(self) -> int:
        return max(1, math.ceil(self.binding.gpus))


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------
def work_seconds_per_month(rt: dict[str, Any]) -> float:
    return rt["work_days_per_month"] * rt["work_hours_per_day"] * 3600.0


def model_weights_gib(params_b: float, quant: str) -> float:
    bytes_per_param = QUANT_BYTES.get(quant.lower())
    if bytes_per_param is None:
        raise ValueError(f"Unknown quantization '{quant}' (allowed: {list(QUANT_BYTES)})")
    return params_b * 1e9 * bytes_per_param / GIB


def kv_cache_per_request_gib(infra: dict[str, Any], seq_len: float) -> float:
    """KV cache/request ≈ 2 × layers × hidden × KV-head-ratio × seq-len × bytes.

    The factor 2 = key + value. KV-head-ratio = num_kv_heads/num_attn_heads
    captures GQA (for Llama-3.1-70B 8/64 → 0.125)."""
    kv_head_ratio = infra["num_kv_heads"] / infra["num_attn_heads"]
    bytes_per_req = (
        2
        * infra["num_layers"]
        * infra["hidden_size"]
        * kv_head_ratio
        * seq_len
        * infra["kv_dtype_bytes"]
    )
    return bytes_per_req / GIB


def compute_scenario(
    label: str,
    *,
    input_tokens_month: float,
    output_tokens_month: float,
    mean_input_tokens_per_turn: float,
    seq_len_for_kv: float,
    cfg: dict[str, Any],
) -> ScenarioResult:
    wl, sla, infra, rt = cfg["workload"], cfg["sla"], cfg["infra"], cfg["runtime"]
    work_s = work_seconds_per_month(rt)

    # --- decode-bound -------------------------------------------------------
    # max(sustained, peak). Sustained = output/month ÷ work seconds.
    # Peak = peak concurrency × required per-stream TPS (p50 as sustained
    # target to hold per stream).
    decode_per_gpu = infra["decode_tokens_per_s_per_gpu"]
    sustained_decode = output_tokens_month / work_s
    peak_decode = wl["concurrency_peak"] * sla["tps_p50"]
    decode_demand = max(sustained_decode, peak_decode)
    decode_gpus = decode_demand / decode_per_gpu
    decode_detail = (
        f"max(sustained {sustained_decode:,.0f} t/s, peak {peak_decode:,.0f} t/s) "
        f"÷ {decode_per_gpu:,.0f} t/s/GPU"
    )

    # --- prefill-bound ------------------------------------------------------
    # Sustained = input/month ÷ work seconds.
    # Peak (arrival-driven) = peak-concurrency arrival RATE of turns × input tokens/turn.
    #   The arrival rate = concurrency_peak ÷ peak_turn_interval_s, where the turn
    #   interval is how often each active user sends a turn (LLM generation + read
    #   time, ~10 s) — NOT the TTFT budget. Using the TTFT budget (1 s) here would
    #   model a "thundering herd" (all users hitting enter in the same second) and
    #   over-counts prefill GPUs ~10×. TTFT remains a per-request LATENCY check, not
    #   the throughput driver.
    prefill_per_gpu = infra["prefill_tokens_per_s_per_gpu"]
    turn_interval_s = rt.get("peak_turn_interval_s", 10.0)
    arrival_rate = wl["concurrency_peak"] / turn_interval_s          # turns/s at peak
    sustained_prefill = input_tokens_month / work_s
    peak_prefill = arrival_rate * mean_input_tokens_per_turn
    prefill_demand = max(sustained_prefill, peak_prefill)
    prefill_gpus = prefill_demand / prefill_per_gpu
    prefill_detail = (
        f"max(sustained {sustained_prefill:,.0f} t/s, "
        f"peak {peak_prefill:,.0f} t/s @ {arrival_rate:,.0f} turns/s "
        f"({wl['concurrency_peak']} users ÷ {turn_interval_s:.0f}s) "
        f"× {mean_input_tokens_per_turn:,.0f} tok/turn) ÷ {prefill_per_gpu:,.0f} t/s/GPU"
    )

    # --- kv-cache-mem-bound -------------------------------------------------
    # Concurrency/GPU = (GPU mem − weights) ÷ KV/request. GPU count =
    # peak concurrency ÷ concurrency/GPU.
    weights_gib = model_weights_gib(infra["model_params_b"], infra["quantization"])
    kv_req_gib = kv_cache_per_request_gib(infra, seq_len_for_kv)
    mem_avail = infra["gpu_mem_gb"]  # interpreted as GiB (VRAM figures are usually in GiB)
    usable = mem_avail - weights_gib
    if usable <= 0:
        conc_per_gpu = 0.0
        kv_gpus = float("inf")
        kv_detail = (
            f"weights {weights_gib:.1f} GiB > GPU mem {mem_avail:.0f} GiB — "
            f"model does not fit on a single GPU (tensor parallel required)"
        )
    else:
        conc_per_gpu = usable / kv_req_gib if kv_req_gib > 0 else float("inf")
        kv_gpus = wl["concurrency_peak"] / conc_per_gpu if conc_per_gpu > 0 else float("inf")
        kv_detail = (
            f"peak {wl['concurrency_peak']} ÷ (({mem_avail:.0f}−{weights_gib:.1f}) GiB "
            f"÷ {kv_req_gib*1024:.1f} MiB/req = {conc_per_gpu:.1f} req/GPU) "
            f"@ seq_len {seq_len_for_kv:,.0f}"
        )

    return ScenarioResult(
        label=label,
        input_tokens_month=input_tokens_month,
        output_tokens_month=output_tokens_month,
        bounds=[
            Bound("prefill", prefill_gpus, prefill_detail),
            Bound("decode", decode_gpus, decode_detail),
            Bound("kv-mem", kv_gpus, kv_detail),
        ],
        concurrency_per_gpu=conc_per_gpu,
        kv_per_request_gib=kv_req_gib,
        weights_gib=weights_gib,
    )


def build_scenarios(cfg: dict[str, Any]) -> tuple[ScenarioResult, ScenarioResult, dict]:
    wl = cfg["workload"]
    total_month = wl["tokens_per_user_per_month"] * wl["num_users"]
    ratio = wl["in_out_ratio"]
    input_share = ratio / (ratio + 1.0)
    naive_input_month = total_month * input_share
    output_month = total_month * (1.0 - input_share)

    # Semvec lowers ONLY the input. The ratio of bound:naive input/turn
    # determines the reduction factor; the output side stays untouched.
    naive_in_turn = wl["naive_mean_input_tokens_per_turn"]
    semvec_in_turn = wl["semvec_input_tokens_per_turn"]
    reduction = 1.0 - (semvec_in_turn / naive_in_turn) if naive_in_turn > 0 else 0.0
    semvec_input_month = naive_input_month * (semvec_in_turn / naive_in_turn)

    # Seq-len for KV: naive = mean naive input + output; semvec = bound input
    # + output. The smaller KV footprint of the Semvec path is the second
    # lever (more concurrency/GPU).
    out_turn = wl["output_tokens_per_turn"]
    naive = compute_scenario(
        "naive (full-context)",
        input_tokens_month=naive_input_month,
        output_tokens_month=output_month,
        mean_input_tokens_per_turn=naive_in_turn,
        seq_len_for_kv=naive_in_turn + out_turn,
        cfg=cfg,
    )
    semvec = compute_scenario(
        "with Semvec",
        input_tokens_month=semvec_input_month,
        output_tokens_month=output_month,
        mean_input_tokens_per_turn=semvec_in_turn,
        seq_len_for_kv=semvec_in_turn + out_turn,
        cfg=cfg,
    )
    meta = {
        "total_tokens_month": total_month,
        "naive_input_month": naive_input_month,
        "semvec_input_month": semvec_input_month,
        "output_month": output_month,
        "input_reduction_pct": reduction * 100.0,
    }
    return naive, semvec, meta


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _fmt_gpus(s: ScenarioResult, cost_eur: float, gpus_per_node: int) -> str:
    nodes = math.ceil(s.gpus_required / gpus_per_node)
    cost = s.gpus_required * cost_eur
    return f"{s.gpus_required} GPU(s) / {nodes} node(s) / ~{cost:,.0f} €"


def print_report(naive: ScenarioResult, semvec: ScenarioResult, meta: dict, cfg: dict) -> None:
    infra = cfg["infra"]
    wl = cfg["workload"]
    calib = infra.get("calibrated", False)
    flag = "" if calib else "  ⚠️  UNCALIBRATED"

    line = "=" * 78
    print(line)
    print("TOKEN-→-HARDWARE SIZING  ·  naive vs. Semvec")
    print(line)
    print(f"Model:    {infra['model_name']}  @ {infra['quantization']}  "
          f"({infra['model_params_b']:.0f}B, weights {naive.weights_gib:.1f} GiB)")
    print(f"GPU:      {infra['gpu_type']}  ({infra['gpu_mem_gb']:.0f} GiB, "
          f"{infra['gpus_per_node']}/node)")
    print(f"Throughput/GPU: prefill {infra['prefill_tokens_per_s_per_gpu']:,.0f} t/s, "
          f"decode {infra['decode_tokens_per_s_per_gpu']:,.0f} t/s{flag}")
    print()
    print(f"Workload: {wl['num_users']:,} users × {wl['tokens_per_user_per_month']:,} tok/month "
          f"= {meta['total_tokens_month']/1e9:.2f} bn tok/month  (in:out {wl['in_out_ratio']:.0f}:1)")
    print(f"          Concurrency avg {wl['concurrency_avg']} / peak {wl['concurrency_peak']}")
    print(f"  Input/month:  naive {meta['naive_input_month']/1e9:.3f} bn  →  "
          f"Semvec {meta['semvec_input_month']/1e9:.3f} bn  "
          f"(−{meta['input_reduction_pct']:.1f} % input)")
    print(f"  Output/month: {meta['output_month']/1e9:.3f} bn  (unchanged)")
    print()

    # Comparison table
    print(f"{'Bound':<12}{'naive (GPUs)':>16}{'Semvec (GPUs)':>16}")
    print("-" * 44)
    nb = {b.name: b for b in naive.bounds}
    sb = {b.name: b for b in semvec.bounds}
    for name in ("prefill", "decode", "kv-mem"):
        nval = nb[name].gpus
        sval = sb[name].gpus
        nmark = " ←" if naive.binding.name == name else "  "
        smark = " ←" if semvec.binding.name == name else "  "
        ns = "∞" if math.isinf(nval) else f"{nval:,.1f}"
        ss = "∞" if math.isinf(sval) else f"{sval:,.1f}"
        print(f"{name:<12}{ns:>14}{nmark}{ss:>14}{smark}")
    print("-" * 44)
    cost = infra["gpu_cost_eur"]
    npn = infra["gpus_per_node"]
    print(f"{'→ required':<12}{naive.gpus_required:>14}  {semvec.gpus_required:>14}")
    print()
    print(f"  naive:  {_fmt_gpus(naive, cost, npn)}  · binding: {naive.binding.name}")
    print(f"          {naive.binding.detail}")
    print(f"  Semvec: {_fmt_gpus(semvec, cost, npn)}  · binding: {semvec.binding.name}")
    print(f"          {semvec.binding.detail}")
    print()
    if naive.gpus_required > 0:
        saved = naive.gpus_required - semvec.gpus_required
        pct = saved / naive.gpus_required * 100.0
        eur = saved * cost
        print(f"  ⇒ Savings: {saved} GPU(s)  (−{pct:.0f} %, ~{eur:,.0f} € CapEx)")
    if not calib:
        print()
        print("  ⚠️  prefill/decode throughput UNCALIBRATED — default reference values.")
        print("     Before reliable figures: vLLM/TensorRT-LLM micro-benchmark on target GPU,")
        print("     enter the values into infra.* and set infra.calibrated=true.")
    print(line)


def run_sweep(cfg: dict, spec: str) -> None:
    """spec = 'param:start:stop:step' over workload.<param>."""
    try:
        param, start, stop, step = spec.split(":")
        start, stop, step = float(start), float(stop), float(step)
    except ValueError:
        sys.exit(f"--sweep expects 'param:start:stop:step', got: {spec!r}")
    if param not in cfg["workload"]:
        sys.exit(f"Sweep parameter '{param}' not in workload.*")

    print("=" * 78)
    print(f"SENSITIVITY SWEEP over workload.{param}")
    print("=" * 78)
    print(f"{param:>28}{'naive GPUs':>12}{'Semvec GPUs':>14}{'savings':>14}")
    print("-" * 78)
    v = start
    while v <= stop + 1e-9:
        cfg["workload"][param] = v
        naive, semvec, _ = build_scenarios(cfg)
        saved = naive.gpus_required - semvec.gpus_required
        print(f"{v:>28,.0f}{naive.gpus_required:>12}{semvec.gpus_required:>14}"
              f"{saved:>12} GPU")
        v += step
    print("-" * 78)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="YAML config (see sizing/configs/)")
    p.add_argument("--semvec-input-tokens", type=float,
                   help="override for workload.semvec_input_tokens_per_turn (MEASURED from pillar 2)")
    p.add_argument("--prefill-tps", type=float, help="override infra.prefill_tokens_per_s_per_gpu (calibrated)")
    p.add_argument("--decode-tps", type=float, help="override infra.decode_tokens_per_s_per_gpu (calibrated)")
    p.add_argument("--calibrated", action="store_true", help="marks the throughput values as calibrated")
    p.add_argument("--sweep", help="sensitivity sweep 'param:start:stop:step' over workload.<param>")
    args = p.parse_args()

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")
    cfg = yaml.safe_load(args.config.read_text())

    if args.semvec_input_tokens is not None:
        cfg["workload"]["semvec_input_tokens_per_turn"] = args.semvec_input_tokens
    if args.prefill_tps is not None:
        cfg["infra"]["prefill_tokens_per_s_per_gpu"] = args.prefill_tps
    if args.decode_tps is not None:
        cfg["infra"]["decode_tokens_per_s_per_gpu"] = args.decode_tps
    if args.calibrated:
        cfg["infra"]["calibrated"] = True

    naive, semvec, meta = build_scenarios(cfg)
    print_report(naive, semvec, meta, cfg)

    if args.sweep:
        print()
        run_sweep(cfg, args.sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
