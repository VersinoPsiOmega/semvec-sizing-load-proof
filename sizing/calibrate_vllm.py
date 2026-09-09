#!/usr/bin/env python3
"""Calibrate the sizing model's prefill/decode constants against a live vLLM endpoint.

The sizing model (sizing_model.py) divides the prefill- and decode-token demand by
two per-GPU throughput constants. Guessed defaults are flagged UNCALIBRATED. This
tool measures the real numbers by benchmarking an OpenAI-compatible LLM endpoint
(vLLM, SGLang, TensorRT-LLM, …) on the target GPU and writing them back.

It measures *serving* throughput, which is what the sizing model needs:

  * prefill: N concurrent requests with a large prompt and max_tokens=1; the
    aggregate prompt-tokens/second is the prefill throughput the GPU sustains.
  * decode:  N concurrent requests generating many tokens; the aggregate
    completion-tokens/second is the decode throughput.

Per-GPU = aggregate / num_gpus (the tensor-parallel size the model is sharded over;
1 for a model that fits on a single GPU).

Examples:
    # measure only (prints the two constants)
    python sizing/calibrate_vllm.py --base-url https://host:port/v1 --model Qwen/Qwen3.6-27B --num-gpus 1

    # measure and write into a sizing config (sets calibrated: true)
    python sizing/calibrate_vllm.py --base-url ... --model ... --num-gpus 1 \
        --write sizing/configs/qwen3_27b_h100.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Pure throughput math (unit-tested)
# ---------------------------------------------------------------------------
def tokens_per_s(total_tokens: float, wall_s: float, num_gpus: int = 1) -> float:
    """Aggregate tokens/second divided across the tensor-parallel GPU count."""
    if wall_s <= 0:
        raise ValueError("wall_s must be > 0")
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    return total_tokens / wall_s / num_gpus


def summarize_calibration(*, prefill_tokens: float, prefill_wall: float,
                          decode_tokens: float, decode_wall: float,
                          num_gpus: int) -> dict[str, float]:
    """Turn measured (tokens, wall) pairs into the two per-GPU sizing constants."""
    return {
        "prefill_tokens_per_s_per_gpu": round(tokens_per_s(prefill_tokens, prefill_wall, num_gpus)),
        "decode_tokens_per_s_per_gpu": round(tokens_per_s(decode_tokens, decode_wall, num_gpus)),
    }


# ---------------------------------------------------------------------------
# Live benchmark (I/O — validated against a real endpoint, not unit-tested)
# ---------------------------------------------------------------------------
def _chat(base_url: str, api_key: str, model: str, content: str, max_tokens: int,
          temperature: float, timeout: float) -> dict[str, Any]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body, headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _run_phase(base_url: str, api_key: str, model: str, *, content: str,
               max_tokens: float, temperature: float, concurrency: int,
               token_field: str, timeout: float) -> tuple[float, float]:
    """Fire `concurrency` requests at once; return (total_tokens, wall_seconds)."""
    def one(_: int) -> int:
        r = _chat(base_url, api_key, model, content, int(max_tokens), temperature, timeout)
        return int(r.get("usage", {}).get(token_field, 0))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        counts = list(ex.map(one, range(concurrency)))
    wall = time.perf_counter() - t0
    return float(sum(counts)), wall


def benchmark(base_url: str, api_key: str, model: str, *, num_gpus: int,
              concurrency: int, prompt_tokens: int, gen_tokens: int,
              rounds: int, timeout: float) -> dict[str, Any]:
    # A ~prompt_tokens-sized prompt (1 word ≈ 1 token, roughly).
    big_prompt = "Summarize this business report. " + (
        "request summary invoice quantity site status value revision process note. "
        * max(1, prompt_tokens // 10)
    )
    best_prefill = (0.0, 1.0)   # (tokens, wall) maximising tokens/wall
    best_decode = (0.0, 1.0)
    for i in range(rounds):
        p_tok, p_wall = _run_phase(base_url, api_key, model, content=big_prompt,
                                   max_tokens=1, temperature=0.0, concurrency=concurrency,
                                   token_field="prompt_tokens", timeout=timeout)
        d_tok, d_wall = _run_phase(base_url, api_key, model,
                                   content="Write a detailed note about enterprise workflows.",
                                   max_tokens=gen_tokens, temperature=0.7, concurrency=concurrency,
                                   token_field="completion_tokens", timeout=timeout)
        if p_tok / p_wall > best_prefill[0] / best_prefill[1]:
            best_prefill = (p_tok, p_wall)
        if d_tok / d_wall > best_decode[0] / best_decode[1]:
            best_decode = (d_tok, d_wall)
        print(f"  round {i+1}/{rounds}: prefill {p_tok/p_wall:,.0f} t/s · decode {d_tok/d_wall:,.0f} t/s",
              file=sys.stderr)

    consts = summarize_calibration(
        prefill_tokens=best_prefill[0], prefill_wall=best_prefill[1],
        decode_tokens=best_decode[0], decode_wall=best_decode[1], num_gpus=num_gpus)
    return {
        "model": model, "num_gpus": num_gpus, "concurrency": concurrency,
        "prompt_tokens_each": prompt_tokens, "gen_tokens_each": gen_tokens, **consts,
    }


def write_into_config(path: Path, consts: dict[str, float], model: str) -> None:
    """Patch the two constants + calibrated: true into a sizing YAML (text-level,
    to preserve comments)."""
    text = path.read_text()
    import re

    def sub(key: str, val: float, s: str) -> str:
        return re.sub(rf"(^\s*{key}:\s*)[\d_.]+", rf"\g<1>{val:g}", s, count=1, flags=re.M)

    text = sub("prefill_tokens_per_s_per_gpu", consts["prefill_tokens_per_s_per_gpu"], text)
    text = sub("decode_tokens_per_s_per_gpu", consts["decode_tokens_per_s_per_gpu"], text)
    text = re.sub(r"(^\s*calibrated:\s*)\w+", r"\g<1>true", text, count=1, flags=re.M)
    path.write_text(text)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base URL (…/v1)")
    p.add_argument("--api-key", default="", help="bearer token (omit if the endpoint is open)")
    p.add_argument("--model", required=True)
    p.add_argument("--num-gpus", type=int, default=1, help="tensor-parallel size the model is sharded over")
    p.add_argument("--concurrency", type=int, default=16, help="concurrent requests per phase")
    p.add_argument("--prompt-tokens", type=int, default=2000, help="approx prompt size for the prefill phase")
    p.add_argument("--gen-tokens", type=int, default=256, help="tokens generated per request in the decode phase")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--write", type=Path, help="sizing config YAML to patch with the measured constants")
    args = p.parse_args()

    print(f"=== calibrating against {args.base_url} · {args.model} · "
          f"{args.num_gpus} GPU(s) · concurrency {args.concurrency} ===", file=sys.stderr)
    res = benchmark(args.base_url, args.api_key, args.model, num_gpus=args.num_gpus,
                    concurrency=args.concurrency, prompt_tokens=args.prompt_tokens,
                    gen_tokens=args.gen_tokens, rounds=args.rounds, timeout=args.timeout)

    print(json.dumps(res, indent=2))
    print(f"\nprefill_tokens_per_s_per_gpu: {res['prefill_tokens_per_s_per_gpu']:,}")
    print(f"decode_tokens_per_s_per_gpu:  {res['decode_tokens_per_s_per_gpu']:,}")

    if args.write:
        write_into_config(args.write, res, args.model)
        print(f"\n→ patched {args.write} (calibrated: true)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
