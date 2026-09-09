#!/usr/bin/env python3
"""LOCOMO QA harness (pillar 2, real dataset) — naive vs. Semvec answer quality with a real LLM.

Replays real LOCOMO conversations into a Semvec state and answers each gold QA pair
twice with a real LLM:

  * naive  — the full conversation history is put in the prompt (full-context replay).
  * semvec — only the Semvec-serialized context (constant size) is put in the prompt.

For every question it records the input-token cost of both prompts AND scores both
answers against the LOCOMO gold answer (word-overlap F1; category 5 = adversarial →
the correct behaviour is to abstain). The output is the real "answer quality at
−X % input tokens" evidence with an actual LLM in the loop.

Embedder: mpnet/768 (local). Reader LLM: any OpenAI-compatible endpoint (default: env).

Example (bounded — keeps load on a shared endpoint moderate):
    python harness/run_locomo_qa.py \
        --dataset /path/to/locomo10.json \
        --base-url https://h100-host:port/v1 --model Qwen/Qwen3.6-27B \
        --conv-limit 1 --limit-qa 40 -o results/locomo_qa.json
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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_profile import MpnetEmbedder, make_token_counter, MPNET_DIM  # noqa: E402

_ARTICLES = {"a", "an", "the"}
_ABSTAIN = (
    "no information", "not mentioned", "don't know", "do not know",
    "no relevant information", "not available", "cannot answer", "can't answer",
    "no record", "isn't mentioned", "is not mentioned", "not in the conversation",
)


# ---------------------------------------------------------------------------
# Scoring (pure, unit-tested)
# ---------------------------------------------------------------------------
def _normalize(s: str) -> list[str]:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return [w for w in s.split() if w not in _ARTICLES]


def f1_score(pred: str, gold: str) -> float:
    """SQuAD-style word-overlap F1 between prediction and gold answer."""
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def is_abstention(pred: str) -> bool:
    low = pred.lower()
    return any(kw in low for kw in _ABSTAIN)


def score_qa(pred: str, gold: str, category: Any) -> float:
    """Category-5 (adversarial) is correct iff the model abstains; else word F1."""
    if str(category) == "5":
        return 1.0 if is_abstention(pred) else 0.0
    return f1_score(pred, gold)


def parse_judge_verdict(text: str) -> bool:
    """Parse a CORRECT/WRONG-style LLM judge reply into a boolean."""
    t = text.strip().lower()
    for neg in ("wrong", "incorrect", "no", "false"):
        if t.startswith(neg):
            return False
    for pos in ("correct", "yes", "true"):
        if t.startswith(pos):
            return True
    return ("correct" in t) and ("incorrect" not in t)


# ---------------------------------------------------------------------------
# Dataset (pure, unit-tested)
# ---------------------------------------------------------------------------
def flatten_conversation(conv: dict) -> list[str]:
    """Flatten LOCOMO session_1..N (in order) into "Speaker: text" turns."""
    idx = sorted(
        int(k.split("_")[1]) for k in conv
        if re.fullmatch(r"session_\d+", k)
    )
    turns: list[str] = []
    for i in idx:
        for t in conv[f"session_{i}"]:
            turns.append(f"{t.get('speaker','')}: {t.get('text','')}".strip())
    return turns


def load_locomo(path: Path, conv_limit: int | None) -> list[dict]:
    data = json.loads(path.read_text())
    out = []
    for cid, c in enumerate(data):
        if conv_limit is not None and cid >= conv_limit:
            break
        out.append({"cid": cid, "turns": flatten_conversation(c["conversation"]), "qa": c["qa"]})
    return out


# ---------------------------------------------------------------------------
# LLM + Semvec state (I/O — validated against the live endpoint)
# ---------------------------------------------------------------------------
_SYSTEM = ("You answer questions about a conversation using ONLY the provided context. "
           "Answer concisely with just the fact. If the context does not contain the "
           "answer, reply exactly: No information available.")


def ask(base_url: str, api_key: str, model: str, context: str, question: str,
        max_tokens: int, timeout: float, disable_thinking: bool = True) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"{_SYSTEM}\n\nContext:\n{context}"},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens, "temperature": 0.0,
    }
    if disable_thinking:
        # Reasoning models (e.g. Qwen3) otherwise spend the whole budget on
        # hidden <think> tokens and return empty content.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body, headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


_JUDGE_SYSTEM = ("You are a strict grader. Given a question, a reference answer and a candidate "
                 "answer, reply with exactly one word: CORRECT if the candidate conveys the "
                 "reference answer, otherwise WRONG.")


def judge(base_url: str, api_key: str, model: str, question: str, gold: str, pred: str,
          timeout: float, disable_thinking: bool = True) -> bool:
    content = (f"Question: {question}\nReference answer: {gold}\n"
               f"Candidate answer: {pred}\nVerdict (CORRECT or WRONG):")
    payload = {"model": model, "messages": [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": content}], "max_tokens": 8, "temperature": 0.0}
    if disable_thinking:
        # Reasoning judges (e.g. Qwen3.6) otherwise spend the whole budget on
        # hidden <think> tokens and return empty content -> every verdict WRONG.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body, headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return parse_judge_verdict((d.get("choices", [{}])[0].get("message", {}).get("content") or ""))


def build_state(turns: list[str], embedder: MpnetEmbedder):
    from semvec import SemvecConfig, SemvecState

    state = SemvecState(config=SemvecConfig(dimension=MPNET_DIM))
    for t in turns:
        if not t.strip():
            continue
        emb = embedder.get_embedding(t)
        state.update(emb, t[:500])
    return state


def semvec_context(state, serializer, embedder, question: str) -> str:
    """Serialize the state for this query WITHOUT storing the question (no pollution)."""
    return serializer.serialize(state, query_embedding=embedder.get_embedding(question))


# --- Tuned retrieval: hybrid (dense + BM25/RRF) + cross-encoder rerank ------
def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion of several ranked lists; deduped, score-sorted."""
    scores: dict[str, float] = {}
    seen: dict[str, int] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            seen.setdefault(item, len(seen))
    return sorted(scores, key=lambda it: (-scores[it], seen[it]))


class _Reranker:
    """Lazy cross-encoder reranker (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_name: str, device: str | None = None, batch: int = 64) -> None:
        from sentence_transformers import CrossEncoder
        self._m = CrossEncoder(model_name, device=device)
        self._batch = batch

    def rerank(self, query: str, texts: list[str], top_k: int) -> list[str]:
        if not texts:
            return []
        scores = self._m.predict([(query, t) for t in texts], batch_size=self._batch)
        return [t for t, _ in sorted(zip(texts, scores), key=lambda x: -float(x[1]))][:top_k]


def _build_bm25(turns: list[str]):
    import bm25s
    toks = bm25s.tokenize(turns, stopwords="en", show_progress=False)
    r = bm25s.BM25()
    r.index(toks, show_progress=False)
    return r


def _bm25_rank(retriever, turns: list[str], query: str, k: int) -> list[str]:
    import bm25s
    q = bm25s.tokenize(query, stopwords="en", show_progress=False)
    res, _ = retriever.retrieve(q, corpus=turns, k=min(k, len(turns)), show_progress=False)
    return list(res[0])


def tuned_context(state, turns, embedder, reranker, question: str, *,
                  fetch_k: int, top_k: int, budget_chars: int, bm25_index=None) -> str:
    """Documented tuned retrieval: dense (+optional BM25/RRF) candidate pool →
    cross-encoder rerank → top_k, capped at budget_chars."""
    q_emb = embedder.get_embedding(question)
    dense = [m.text for m in state.memory.get_relevant_memories(q_emb, top_k=fetch_k)]
    pool = dense
    if bm25_index is not None:
        try:
            lex = _bm25_rank(bm25_index, turns, question, fetch_k)
            pool = rrf_fuse([dense, lex])[:fetch_k]
        except Exception:  # noqa: BLE001 — BM25 is a best-effort signal
            pool = dense
    ranked = reranker.rerank(question, pool, top_k) if reranker else pool[:top_k]
    out, total = [], 0
    for t in ranked:
        if total + len(t) > budget_chars:
            break
        out.append(t)
        total += len(t) + 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(convs: list[dict], *, base_url: str, api_key: str, model: str, count_tokens: Callable[[str], int],
        limit_qa: int | None, max_answer_tokens: int, concurrency: int, timeout: float,
        device: str | None, disable_thinking: bool = True,
        judge_cfg: dict | None = None, rerank_model: str | None = None,
        rerank_fetch_k: int = 50, rerank_top_k: int = 15, budget_chars: int = 10000,
        use_hybrid: bool = True) -> dict[str, Any]:
    from semvec.token_reduction import SemvecStateSerializer, SerializerConfig

    embedder = MpnetEmbedder(device=device)
    serializer = SemvecStateSerializer(SerializerConfig(top_k=30, max_memory_chars=600,
                                                        max_last_response_chars=2000))
    reranker = _Reranker(rerank_model, device=device) if rerank_model else None
    if reranker:
        print(f"tuned retrieval: dense fetch_k={rerank_fetch_k} + "
              f"{'BM25/RRF + ' if use_hybrid else ''}rerank({rerank_model}) → top_k={rerank_top_k}, "
              f"budget {budget_chars} chars", file=sys.stderr)
    rows: list[dict] = []

    for conv in convs:
        print(f"conv {conv['cid']}: building state from {len(conv['turns'])} turns …", file=sys.stderr)
        state = build_state(conv["turns"], embedder)
        full_context = "\n".join(conv["turns"])
        bm25_index = _build_bm25(conv["turns"]) if (reranker and use_hybrid) else None
        qas = conv["qa"][:limit_qa] if limit_qa else conv["qa"]

        def one(qa: dict) -> dict:
            q = qa["question"]
            gold = str(qa.get("answer", ""))
            cat = qa.get("category")
            if reranker is not None:
                sv_ctx = tuned_context(state, conv["turns"], embedder, reranker, q,
                                       fetch_k=rerank_fetch_k, top_k=rerank_top_k,
                                       budget_chars=budget_chars, bm25_index=bm25_index)
            else:
                sv_ctx = semvec_context(state, serializer, embedder, q)
            naive_prompt = f"{_SYSTEM}\n\nContext:\n{full_context}\n{q}"
            sv_prompt = f"{_SYSTEM}\n\nContext:\n{sv_ctx}\n{q}"
            naive_ans = ask(base_url, api_key, model, full_context, q, max_answer_tokens, timeout, disable_thinking)
            sv_ans = ask(base_url, api_key, model, sv_ctx, q, max_answer_tokens, timeout, disable_thinking)
            row = {
                "cid": conv["cid"], "category": str(cat), "question": q, "gold": gold,
                "naive_input_tokens": count_tokens(naive_prompt),
                "semvec_input_tokens": count_tokens(sv_prompt),
                "naive_f1": score_qa(naive_ans, gold, cat),
                "semvec_f1": score_qa(sv_ans, gold, cat),
            }
            if judge_cfg is not None:
                def _correct(ans: str) -> float:
                    if str(cat) == "5":
                        return 1.0 if is_abstention(ans) else 0.0
                    return 1.0 if judge(judge_cfg["base_url"], judge_cfg["api_key"],
                                        judge_cfg["model"], q, gold, ans, timeout) else 0.0
                row["naive_judge"] = _correct(naive_ans)
                row["semvec_judge"] = _correct(sv_ans)
            return row

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(one, qa) for qa in qas]
            conv_rows, failed = [], 0
            for f in futs:
                try:
                    conv_rows.append(f.result())
                except Exception:  # noqa: BLE001 — one bad QA must not abort the suite
                    failed += 1
        rows.extend(conv_rows)
        print(f"  conv {conv['cid']}: {len(conv_rows)} QA answered ({failed} skipped)", file=sys.stderr)

    return summarize(rows)


def summarize(rows: list[dict]) -> dict[str, Any]:
    def mean(xs):
        return round(statistics.mean(xs), 4) if xs else None

    naive_tok = [r["naive_input_tokens"] for r in rows]
    sv_tok = [r["semvec_input_tokens"] for r in rows]
    tot_n, tot_s = sum(naive_tok), sum(sv_tok)
    by_cat = defaultdict(lambda: {"n": 0, "naive_f1": [], "semvec_f1": []})
    for r in rows:
        c = by_cat[r["category"]]
        c["n"] += 1
        c["naive_f1"].append(r["naive_f1"])
        c["semvec_f1"].append(r["semvec_f1"])
    per_cat = {k: {"n": v["n"], "naive_f1": mean(v["naive_f1"]), "semvec_f1": mean(v["semvec_f1"])}
               for k, v in sorted(by_cat.items())}
    judged = [r for r in rows if "naive_judge" in r]
    return {
        "summary": {
            "qa_count": len(rows),
            "naive_f1": mean([r["naive_f1"] for r in rows]),
            "semvec_f1": mean([r["semvec_f1"] for r in rows]),
            "naive_judge_acc": mean([r["naive_judge"] for r in judged]) if judged else None,
            "semvec_judge_acc": mean([r["semvec_judge"] for r in judged]) if judged else None,
            "naive_input_tokens_mean": mean(naive_tok),
            "semvec_input_tokens_mean": mean(sv_tok),
            "input_reduction_pct": round((1 - tot_s / tot_n) * 100, 1) if tot_n else None,
            "per_category": per_cat,
        },
        "rows": rows,
    }


def print_report(rep: dict) -> None:
    s = rep["summary"]
    line = "=" * 78
    print(line)
    print("LOCOMO QA — real LLM answer quality  ·  naive (full-context) vs. Semvec")
    print(line)
    print(f"QA answered: {s['qa_count']}")
    print(f"  Input tokens/QA:  naive {s['naive_input_tokens_mean']:,.0f}  →  "
          f"Semvec {s['semvec_input_tokens_mean']:,.0f}   (−{s['input_reduction_pct']:.1f} %)")
    print(f"  Answer F1 (word): naive {s['naive_f1']:.3f}  vs  Semvec {s['semvec_f1']:.3f}")
    if s.get("naive_judge_acc") is not None:
        print(f"  LLM-judged acc:   naive {s['naive_judge_acc']:.3f}  vs  Semvec {s['semvec_judge_acc']:.3f}  "
              f"(LOCOMO-J style)")
    print(f"  → Semvec keeps {s['semvec_f1']/s['naive_f1']*100:.0f} % of naive quality "
          f"at {100 - s['input_reduction_pct']:.0f} % of the input cost" if s["naive_f1"] else "")
    print("  per category (1=multi-hop 2=temporal 3=open 4=single-hop 5=adversarial):")
    for cat, v in s["per_category"].items():
        print(f"    cat {cat}: n={v['n']:<4} naive F1 {v['naive_f1']:.3f}  Semvec F1 {v['semvec_f1']:.3f}")
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
    p.add_argument("--dataset", type=Path, required=True, help="path to locomo10.json")
    p.add_argument("--base-url", default=_env("OPENAI_BASE_URL"), help="reader LLM endpoint")
    p.add_argument("--api-key", default=_env("OPENAI_API_KEY"))
    p.add_argument("--model", default=_env("OPENAI_MODEL"))
    p.add_argument("--conv-limit", type=int, default=1, help="number of conversations (bounds load)")
    p.add_argument("--limit-qa", type=int, default=40, help="max QA pairs per conversation (bounds load)")
    p.add_argument("--max-answer-tokens", type=int, default=96)
    p.add_argument("--thinking", choices=("off","on"), default="off",
                   help="reasoning models (Qwen3) burn the token budget on <think>; default off")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--device", default=None)
    p.add_argument("--no-rerank", action="store_true", help="bare serializer instead of the tuned rerank stack")
    p.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--rerank-fetch-k", type=int, default=50)
    p.add_argument("--rerank-top-k", type=int, default=15)
    p.add_argument("--context-budget-chars", type=int, default=10000)
    p.add_argument("--no-hybrid", action="store_true", help="dense-only candidate pool (skip BM25/RRF)")
    p.add_argument("--judge", action="store_true", help="add an LLM judge (LOCOMO-J style correctness)")
    p.add_argument("--judge-base-url", default=_env("JUDGE_OPENAI_BASE_URL") or _env("OPENAI_BASE_URL"))
    p.add_argument("--judge-model", default=_env("JUDGE_OPENAI_MODEL") or _env("OPENAI_MODEL"))
    p.add_argument("--judge-key", default=_env("JUDGE_OPENAI_API_KEY") or _env("OPENAI_API_KEY"))
    p.add_argument("-o", "--output", type=Path)
    args = p.parse_args()

    if not args.base_url or not args.model:
        sys.exit("--base-url and --model required (or set OPENAI_BASE_URL/OPENAI_MODEL)")

    count_tokens = make_token_counter()
    convs = load_locomo(args.dataset, args.conv_limit)
    print(f"loaded {len(convs)} conversation(s); reader: {args.model} @ {args.base_url}", file=sys.stderr)

    t0 = time.perf_counter()
    rep = run(convs, base_url=args.base_url, api_key=args.api_key, model=args.model,
              count_tokens=count_tokens, limit_qa=args.limit_qa,
              max_answer_tokens=args.max_answer_tokens, concurrency=args.concurrency,
              timeout=args.timeout, device=args.device, disable_thinking=(args.thinking=='off'),
              judge_cfg=({"base_url": args.judge_base_url, "model": args.judge_model,
                          "api_key": args.judge_key} if args.judge else None),
              rerank_model=(None if args.no_rerank else args.rerank_model),
              rerank_fetch_k=args.rerank_fetch_k, rerank_top_k=args.rerank_top_k,
              budget_chars=args.context_budget_chars, use_hybrid=(not args.no_hybrid))
    rep["summary"]["wall_seconds"] = round(time.perf_counter() - t0, 1)
    print_report(rep)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"\nReport → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
