#!/usr/bin/env python3
"""LongBench-v2 hard test — naive full-context vs. Semvec on very long contexts.

LongBench-v2 is a hard multiple-choice benchmark over very long contexts (documents
of 25k–500k+ tokens). It stresses Semvec where it matters most:

  * **short/medium** items: the full context fits the model window → compare
    accuracy + input tokens, naive vs Semvec.
  * **long** items: the full context **exceeds** the model window → the naive
    approach is *infeasible*, and only Semvec (chunk → retrieve → compress) can
    answer at all.

Per item: the context is chunked, stored in a Semvec state, and the tuned retrieval
(hybrid BM25 + cross-encoder rerank) builds a compact context for the question. The
answer is one of A/B/C/D — scored by exact match (objective, no LLM judge needed).

Real LLM (OpenAI-compatible, e.g. H100 Qwen3.6-27B), real mpnet/768 embedding, real
cross-encoder rerank. Dataset: THUDM/LongBench-v2 (cache a sample to JSON first).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_profile import MpnetEmbedder, make_token_counter, MPNET_DIM  # noqa: E402
from run_locomo_qa import build_state, tuned_context, _Reranker, _build_bm25  # noqa: E402


def build_state_batched(chunks: list[str], embedder, batch: int = 256):
    """Like run_locomo_qa.build_state but batch-encodes the chunks (essential for
    documents with thousands of chunks — per-chunk encoding would be far too slow)."""
    import numpy as np
    from semvec import SemvecConfig, SemvecState

    chunks = [c for c in chunks if c.strip()]
    state = SemvecState(config=SemvecConfig(dimension=MPNET_DIM))
    if not chunks:
        return state
    embs = embedder._model.encode(chunks, batch_size=batch, normalize_embeddings=True,
                                  show_progress_bar=False)
    for t, e in zip(chunks, embs):
        state.update(np.asarray(e, dtype="float32"), t[:500])
    return state


# ---------------------------------------------------------------------------
# Pure logic (unit-tested)
# ---------------------------------------------------------------------------
def parse_mc_answer(text: str) -> str | None:
    """Extract the chosen option letter (A/B/C/D) from an LLM reply."""
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I).strip()
    m = re.search(r"answer[^A-Da-d]{0,12}([ABCD])\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.match(r"\s*\(?([ABCD])[\).:\s]", t) or re.match(r"\s*\(?([ABCD])\)?\s*$", t)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", t)
    return m.group(1).upper() if m else None


def chunk_text(text: str, size: int = 500) -> list[str]:
    """Split text into fixed-size character chunks (no overlap)."""
    return [text[i:i + size] for i in range(0, len(text), size)] if text else []


def mc_accuracy(rows: list[dict], pred_key: str) -> float:
    scored = [r for r in rows if pred_key in r]
    if not scored:
        return 0.0
    return sum(1 for r in scored if r.get(pred_key) == r["gold"]) / len(scored)


def mc_accuracy_n(rows: list[dict], pred_key: str) -> int:
    return sum(1 for r in rows if pred_key in r)


# ---------------------------------------------------------------------------
# LLM (multiple-choice)
# ---------------------------------------------------------------------------
_MC_SYSTEM = ("Read the context and answer the multiple-choice question. "
              "Respond with ONLY the letter of the correct option: A, B, C or D.")


def _mc_user(context: str, item: dict) -> str:
    return (f"Context:\n{context}\n\nQuestion: {item['question']}\n"
            f"A) {item['choice_A']}\nB) {item['choice_B']}\n"
            f"C) {item['choice_C']}\nD) {item['choice_D']}\n\nAnswer:")


def ask_mc(base_url: str, api_key: str, model: str, context: str, item: dict,
           max_tokens: int, timeout: float, disable_thinking: bool = True) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": _MC_SYSTEM},
                     {"role": "user", "content": _mc_user(context, item)}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 json.dumps(payload).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(items: list[dict], *, base_url: str, api_key: str, model: str, count_tokens,
        naive_cap_tokens: int, chunk_chars: int, rerank_model: str | None,
        rerank_fetch_k: int, rerank_top_k: int, budget_chars: int, use_hybrid: bool,
        max_answer_tokens: int, timeout: float, device: str | None,
        disable_thinking: bool) -> dict[str, Any]:
    embedder = MpnetEmbedder(device=device)
    reranker = _Reranker(rerank_model, device=device) if rerank_model else None
    rows: list[dict] = []

    for n, item in enumerate(items):
      try:
        ctx = item["context"]
        chunks = chunk_text(ctx, chunk_chars)
        state = build_state_batched(chunks, embedder)
        bm25 = _build_bm25(chunks) if (reranker and use_hybrid and chunks) else None
        sv_ctx = tuned_context(state, chunks, embedder, reranker, item["question"],
                               fetch_k=rerank_fetch_k, top_k=rerank_top_k,
                               budget_chars=budget_chars, bm25_index=bm25)
        sv_ans = ask_mc(base_url, api_key, model, sv_ctx, item, max_answer_tokens, timeout, disable_thinking)
        row = {
            "id": item.get("_id"), "domain": item.get("domain"), "length": item.get("length"),
            "gold": item["answer"],
            "context_tokens": count_tokens(ctx),
            "semvec_input_tokens": count_tokens(_MC_SYSTEM + _mc_user(sv_ctx, item)),
            "semvec_pred": parse_mc_answer(sv_ans),
        }
        # naive only if the full context fits the model window
        if row["context_tokens"] <= naive_cap_tokens:
            naive_ans = ask_mc(base_url, api_key, model, ctx, item, max_answer_tokens, timeout, disable_thinking)
            row["naive_input_tokens"] = count_tokens(_MC_SYSTEM + _mc_user(ctx, item))
            row["naive_pred"] = parse_mc_answer(naive_ans)
        else:
            row["naive_infeasible"] = True
        rows.append(row)
        print(f"  [{n+1}/{len(items)}] {item.get('length'):6} ctx~{row['context_tokens']//1000}k "
              f"naive={row.get('naive_pred', 'INFEASIBLE')} semvec={row['semvec_pred']} gold={row['gold']}",
              file=sys.stderr)
      except Exception as e:  # noqa: BLE001 — one bad item must not lose the whole run
        print(f"  [{n+1}/{len(items)}] SKIPPED: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
    return summarize(rows, count_tokens)


def summarize(rows: list[dict], count_tokens) -> dict[str, Any]:
    feasible = [r for r in rows if "naive_pred" in r]
    infeasible = [r for r in rows if r.get("naive_infeasible")]
    tot_naive = sum(r["naive_input_tokens"] for r in feasible)
    tot_sv = sum(r["semvec_input_tokens"] for r in feasible)

    per_len = defaultdict(lambda: {"n": 0, "naive_n": 0})
    for r in rows:
        b = per_len[r["length"]]
        b["n"] += 1
        b["naive_n"] += 1 if "naive_pred" in r else 0
    per_length = {}
    for L, b in per_len.items():
        lr = [r for r in rows if r["length"] == L]
        per_length[L] = {
            "n": b["n"], "naive_feasible": b["naive_n"],
            "naive_acc": round(mc_accuracy(lr, "naive_pred"), 3) if b["naive_n"] else None,
            "semvec_acc": round(mc_accuracy(lr, "semvec_pred"), 3),
        }
    return {
        "summary": {
            "items": len(rows),
            "naive_feasible": len(feasible),
            "naive_infeasible": len(infeasible),
            "naive_acc_feasible": round(mc_accuracy(feasible, "naive_pred"), 3) if feasible else None,
            "semvec_acc_feasible": round(mc_accuracy(feasible, "semvec_pred"), 3) if feasible else None,
            "semvec_acc_all": round(mc_accuracy(rows, "semvec_pred"), 3),
            "naive_input_tokens_mean": round(tot_naive / len(feasible)) if feasible else None,
            "semvec_input_tokens_mean": round(tot_sv / len(feasible)) if feasible else None,
            "input_reduction_pct_feasible": round((1 - tot_sv / tot_naive) * 100, 1) if tot_naive else None,
            "context_tokens_mean": round(statistics.mean(r["context_tokens"] for r in rows)),
            "per_length": per_length,
        },
        "rows": rows,
    }


def print_report(rep: dict) -> None:
    s = rep["summary"]
    line = "=" * 78
    print(line)
    print("LONGBENCH-v2 HARD TEST — naive (full context) vs. Semvec (chunk+retrieve+rerank)")
    print(line)
    print(f"Items: {s['items']}  ·  mean context {s['context_tokens_mean']:,} tokens")
    print(f"  naive feasible (fits window): {s['naive_feasible']}  ·  "
          f"naive INFEASIBLE (context overflow → only Semvec works): {s['naive_infeasible']}")
    if s["naive_feasible"]:
        print(f"  Input tokens/item (feasible): naive {s['naive_input_tokens_mean']:,} → "
              f"Semvec {s['semvec_input_tokens_mean']:,}  (−{s['input_reduction_pct_feasible']:.1f} %)")
        print(f"  Accuracy (feasible):  naive {s['naive_acc_feasible']:.3f}  vs  Semvec {s['semvec_acc_feasible']:.3f}")
    print(f"  Accuracy (ALL items, Semvec): {s['semvec_acc_all']:.3f}  "
          f"(naive can't score the long ones at all)")
    print("  per length:")
    for L, v in s["per_length"].items():
        na = f"{v['naive_acc']:.3f}" if v["naive_acc"] is not None else "—(infeasible)"
        print(f"    {L:6}: n={v['n']:<3} naive {na}  Semvec {v['semvec_acc']:.3f}")
    print(line)


def _env(name: str, default: str = "") -> str:
    """Read an env var, stripped.

    A .env authored on Windows carries CRLF line endings, and `set -a; . .env`
    keeps the trailing CR inside the value. An URL then ends in "\r" and every
    request fails with "URL rejected: Malformed input to a URL function" while
    the endpoint is perfectly reachable — an opaque failure worth one strip().
    """
    return (os.environ.get(name) or default).strip()


def main() -> int:
    import os
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="cached LongBench-v2 sample JSON (with context)")
    p.add_argument("--base-url", default=_env("OPENAI_BASE_URL"))
    p.add_argument("--api-key", default=_env("OPENAI_API_KEY"))
    p.add_argument("--model", default=_env("OPENAI_MODEL"))
    p.add_argument("--limit", type=int, default=0, help="cap items (0 = all)")
    p.add_argument("--naive-cap-tokens", type=int, default=200000, help="max context tokens the naive path attempts")
    p.add_argument("--chunk-chars", type=int, default=500)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--rerank-fetch-k", type=int, default=50)
    p.add_argument("--rerank-top-k", type=int, default=15)
    p.add_argument("--context-budget-chars", type=int, default=10000)
    p.add_argument("--no-hybrid", action="store_true")
    p.add_argument("--max-answer-tokens", type=int, default=16)
    p.add_argument("--thinking", choices=("off", "on"), default="off")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--device", default=None)
    p.add_argument("-o", "--output", type=Path)
    args = p.parse_args()
    if not args.base_url or not args.model:
        sys.exit("--base-url and --model required (or OPENAI_BASE_URL/OPENAI_MODEL)")

    items = json.loads(args.dataset.read_text())
    if args.limit:
        items = items[:args.limit]
    count_tokens = make_token_counter()
    print(f"loaded {len(items)} LongBench-v2 items; reader {args.model} @ {args.base_url}", file=sys.stderr)

    t0 = time.perf_counter()
    rep = run(items, base_url=args.base_url, api_key=args.api_key, model=args.model,
              count_tokens=count_tokens, naive_cap_tokens=args.naive_cap_tokens,
              chunk_chars=args.chunk_chars,
              rerank_model=(None if args.no_rerank else args.rerank_model),
              rerank_fetch_k=args.rerank_fetch_k, rerank_top_k=args.rerank_top_k,
              budget_chars=args.context_budget_chars, use_hybrid=(not args.no_hybrid),
              max_answer_tokens=args.max_answer_tokens, timeout=args.timeout,
              device=args.device, disable_thinking=(args.thinking == "off"))
    rep["summary"]["wall_seconds"] = round(time.perf_counter() - t0, 1)
    print_report(rep)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"\nReport → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
