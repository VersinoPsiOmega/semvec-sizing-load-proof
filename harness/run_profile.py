#!/usr/bin/env python3
"""Measurement harness (pillar 2) — input tokens/turn with vs. without Semvec on a conversation profile.

Feeds in a conversation profile (synthetically generated from a YAML, OR
supplied by the operator as JSONL) and measures per turn:

  * baseline_input_tokens — naive full-context replay: system/preamble + the
    entire history so far + current message (grows linearly with the turn count).
  * semvec_input_tokens   — the context block built by `SemvecChatProxy`
    (constant size, independent of the conversation length).

Plus a quality proxy ("needles"): facts set early are queried later; we measure
whether the fact survives in the Semvec context (retrieval proxy, offline) or —
with --llm — whether the LLM answers it correctly.

Embedder is mpnet throughout (paraphrase-multilingual-mpnet-base-v2, dim 768).
Token counting is exact via tiktoken (cl100k_base), the same encoder for both paths.

Examples:
    # offline, deterministic, no LLM needed (air-gapped):
    python harness/run_profile.py --profile harness/profiles/reference_synthetic.yaml \
        -o results/harness_synthetic.json

    # with a real LLM quality judgment (uses OPENAI_* / JUDGE_OPENAI_* from .env):
    python harness/run_profile.py --profile ... --llm -o results/harness_llm.json

    # your own conversation profile:
    python harness/run_profile.py --custom-profile my_profile.jsonl -o results/...
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML missing — `pip install pyyaml`.")

MPNET_MODEL = "paraphrase-multilingual-mpnet-base-v2"
MPNET_DIM = 768


# ---------------------------------------------------------------------------
# Embedder (mpnet/768) — own embedding_service, as recommended in the docs
# ---------------------------------------------------------------------------
class MpnetEmbedder:
    """Any-object embedder: get_embedding(text) + get_dimension()."""

    def __init__(self, model: str = MPNET_MODEL, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device=device)
        # get_sentence_embedding_dimension() was renamed in sentence-transformers
        # 6.x; keep both paths so the pinned lower bound stays honest.
        getter = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        self._dim = getter()

    def get_embedding(self, text: str):
        import numpy as np

        v = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype="float32")

    def get_dimension(self) -> int:
        return int(self._dim)


def make_token_counter() -> Callable[[str], int]:
    """Exact tiktoken counter; heuristic fallback only when tiktoken is missing."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        # disallowed_special=() so texts containing tokens like <|endoftext|>
        # are counted as plain text instead of raising (LongBench contexts hit this).
        return lambda s: len(enc.encode(s, disallowed_special=()))
    except Exception:  # noqa: BLE001
        sys.stderr.write("tiktoken not available - chars/4 heuristic (inexact)\n")
        # Empty text must count as 0, matching the tiktoken path exactly.
        return lambda s: (max(1, len(s) // 4) if s else 0)


# ---------------------------------------------------------------------------
# Synthetic profile
# ---------------------------------------------------------------------------
_WORDS = (
    "request ticket schedule invoice quantity site status value revision "
    "process history follow-up agreement approval policy delivery-date budget "
    "question answer note context system department colleague shift handover"
).split()


@dataclass
class Needle:
    code: str          # unique code that must survive in the context
    fact: str          # the fact that is set (user turn)
    probe: str         # the later question
    set_at_turn: int
    probe_at_turn: int


@dataclass
class Conversation:
    cid: int
    turns: list[dict[str, Any]]   # [{role, content, kind, needle?}]
    needles: list[Needle]


def _filler(rng: random.Random, target_tokens: int, count_tokens: Callable[[str], int]) -> str:
    """Deterministic filler text of ~ target_tokens size."""
    out: list[str] = []
    while count_tokens(" ".join(out)) < target_tokens:
        out.append(rng.choice(_WORDS))
        if len(out) > target_tokens * 3:  # safety
            break
    return " ".join(out)


def _triangular_int(rng: random.Random, lo: int, mode: int, hi: int) -> int:
    return max(lo, min(hi, round(rng.triangular(lo, hi, mode))))


def generate_conversations(profile: dict, count_tokens: Callable[[str], int]) -> list[Conversation]:
    rng = random.Random(profile.get("seed", 0))
    conv_cfg = profile["conversations"]
    ctx = profile["context"]
    ndl = profile["needles"]

    convs: list[Conversation] = []
    for cid in range(conv_cfg["count"]):
        n_turns = _triangular_int(
            rng, conv_cfg["turns_min"], conv_cfg["turns_median"], conv_cfg["turns_max"]
        )
        # Set needles early, query them later (>= probe_after_turns).
        needles: list[Needle] = []
        n_needles = min(ndl["count_per_conversation"], max(0, n_turns - ndl["probe_after_turns"] - 1))
        for k in range(n_needles):
            set_at = 1 + k
            probe_at = min(n_turns - 1, set_at + ndl["probe_after_turns"] + rng.randint(0, 3))
            code = f"RX{rng.randint(1000, 9999)}-{chr(65 + k)}"
            fact = (
                f"Important reminder: the case code for case {chr(65 + k)} is {code}. "
                f"Please remember it for later."
            )
            probe = f"What was the case code for case {chr(65 + k)} again?"
            needles.append(Needle(code, fact, probe, set_at, probe_at))

        set_map = {nd.set_at_turn: nd for nd in needles}
        probe_map = {nd.probe_at_turn: nd for nd in needles}

        turns: list[dict[str, Any]] = []
        for t in range(n_turns):
            if t in set_map:
                nd = set_map[t]
                content = nd.fact
                kind = "needle_set"
                meta_needle = nd.code
            elif t in probe_map:
                nd = probe_map[t]
                content = nd.probe
                kind = "needle_probe"
                meta_needle = nd.code
            else:
                u_tok = max(20, round(rng.gauss(ctx["user_msg_tokens_mean"], ctx["user_msg_tokens_jitter"])))
                # Filler draws from its OWN rng, derived from (seed, cid, turn).
                # _filler loops until the token budget is met, so its number of
                # draws depends on the tokeniser and the word pool. Sharing `rng`
                # would let either of those shift the conversation structure of
                # every later conversation — see
                # test_conversation_structure_independent_of_token_counter.
                filler_rng = random.Random(f"{profile.get('seed', 0)}-{cid}-{t}")
                content = f"Question {t}: " + _filler(filler_rng, u_tok, count_tokens)
                kind = "filler"
                meta_needle = None
            turns.append({"role": "user", "content": content, "kind": kind, "needle": meta_needle, "turn": t})
        convs.append(Conversation(cid, turns, needles))
    return convs


def load_custom_profile(path: Path) -> list[Conversation]:
    """JSONL: one line per conversation, {"turns": [{"role","content"}, ...]}.

    Needle quality is only evaluated here if turns carry a field
    {"kind": "needle_set"|"needle_probe", "needle": "<code>"}."""
    convs: list[Conversation] = []
    for cid, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        raw = obj["turns"]
        turns = []
        for t, m in enumerate(raw):
            turns.append({
                "role": m.get("role", "user"),
                "content": m["content"],
                "kind": m.get("kind", "filler"),
                "needle": m.get("needle"),
                "turn": t,
            })
        convs.append(Conversation(cid, turns, []))
    return convs


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure(
    convs: list[Conversation],
    *,
    static_preamble_tokens: int,
    count_tokens: Callable[[str], int],
    use_llm: bool,
    device: str | None,
    serializer_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from semvec import SemvecConfig
    from semvec.token_reduction import ChatMessage, SemvecChatProxy, SerializerConfig

    embedder = MpnetEmbedder(device=device)
    # Serializer aligned with the LOCOMO production config (top_k 30, large budget).
    sc = serializer_cfg or {}
    ser = SerializerConfig(
        top_k=sc.get("top_k", 30),
        max_memory_chars=sc.get("max_memory_chars", 600),
        max_last_response_chars=sc.get("max_last_response_chars", 2000),
    )

    # Static preamble (system prompt + optional RAG docs) that the NAIVE path
    # resends on every turn. Deterministic filler text of the target size.
    preamble = _filler(random.Random(1), static_preamble_tokens, count_tokens)
    system_prompt = "You are a helpful enterprise assistant.\n" + preamble

    # LLM only in --llm mode; otherwise the proxy's built-in echo mock.
    llm_call = None
    if use_llm:
        from semvec.token_reduction import create_llm_client

        llm_call = create_llm_client("openai")

    per_turn_rows: list[dict[str, Any]] = []
    needle_results: list[dict[str, Any]] = []

    for conv in convs:
        proxy = SemvecChatProxy(
            llm_call=llm_call,
            system_prompt=system_prompt,
            pss_config=SemvecConfig(dimension=MPNET_DIM),
            serializer_config=ser,
            embedding_service=embedder,
        )
        # Naive history as a ChatMessage list, starting with system/preamble.
        history: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]

        for turn in conv.turns:
            if turn["role"] != "user":
                continue
            user_msg = turn["content"]

            # --- naive input: system/preamble + history + current message
            naive_tokens = naive_input_tokens([m.content for m in history], user_msg, count_tokens)

            # --- Semvec turn: system prompt + context block + current message
            result = proxy.chat(user_msg)
            semvec_tokens = semvec_input_tokens(system_prompt, result.pss_prompt,
                                                user_msg, count_tokens)

            per_turn_rows.append({
                "cid": conv.cid,
                "turn": turn["turn"],
                "kind": turn["kind"],
                "naive_input_tokens": naive_tokens,
                "semvec_input_tokens": semvec_tokens,
                "phase": result.phase,
            })

            # --- needle quality
            if turn["kind"] == "needle_probe" and turn["needle"]:
                code = turn["needle"]
                retained = code in result.pss_prompt
                row = {"cid": conv.cid, "turn": turn["turn"], "code": code, "retained_in_context": retained}
                if use_llm and result.response:
                    row["llm_answer_correct"] = code in result.response
                needle_results.append(row)

            # advance the naive history (including the assistant response)
            history.append(ChatMessage(role="user", content=user_msg))
            history.append(ChatMessage(role="assistant", content=result.response or "(ok)"))

    return summarize(per_turn_rows, needle_results, use_llm)


def naive_input_tokens(history: list[str], user_msg: str,
                       count_tokens: Callable[[str], int]) -> int:
    """What the naive path sends this turn: everything, resent."""
    return sum(count_tokens(h) for h in history) + count_tokens(user_msg)


def semvec_input_tokens(system_prompt: str, context_block: str, user_msg: str,
                        count_tokens: Callable[[str], int]) -> int:
    """What the Semvec path sends this turn.

    The memory layer replaces the *history* with a bounded context block. It
    does not replace the system prompt — Semvec is the conversation compressor,
    not the RAG store — and it obviously does not replace the user's question.
    Both therefore stay on the bill, exactly as on the naive side, so the two
    numbers differ only by history-vs-context. Counting the context block alone
    would inflate the reported reduction by the size of the system prompt.
    """
    return count_tokens(system_prompt) + count_tokens(context_block) + count_tokens(user_msg)


MIN_TURN_SAMPLES = 3
"""Conversations below which a turn index is too thinly sampled to judge.

Turn counts are triangular, so the highest indices are reached by one or two
conversations. Letting a single row decide the break-even would make the metric
hostage to one outlier.
"""


def break_even_turn(rows: list[dict], min_samples: int = MIN_TURN_SAMPLES) -> int | None:
    """First turn index from which the Semvec path is cheaper and stays cheaper.

    Derived from the measurements rather than assumed: a turn threshold written
    into the code would be an assumption dressed up as a result. Turn indices
    reached by fewer than `min_samples` conversations neither win nor end the
    run — they are too thin to carry the decision. Returns None if the Semvec
    path never wins for good.
    """
    if not rows:
        return None
    by_turn: dict[int, list[dict]] = {}
    for r in rows:
        by_turn.setdefault(r["turn"], []).append(r)

    winner: int | None = None
    for t in sorted(by_turn, reverse=True):
        group = by_turn[t]
        if len(group) < min_samples:
            continue
        semvec = statistics.median(r["semvec_input_tokens"] for r in group)
        naive = statistics.median(r["naive_input_tokens"] for r in group)
        if semvec < naive:
            winner = t
        else:
            break
    return winner


def summarize(rows: list[dict], needles: list[dict], use_llm: bool) -> dict[str, Any]:
    def stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {}
        s = sorted(vals)
        return {
            "mean": round(statistics.mean(s), 1),
            "median": round(statistics.median(s), 1),
            "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 1),
            "max": round(max(s), 1),
        }

    naive = [r["naive_input_tokens"] for r in rows]
    semvec = [r["semvec_input_tokens"] for r in rows]
    total_naive = sum(naive)
    total_semvec = sum(semvec)
    reduction = (1 - total_semvec / total_naive) * 100 if total_naive else 0.0

    breakeven = break_even_turn(rows)

    n_retained = sum(1 for n in needles if n["retained_in_context"])
    needle_summary = {
        "total_probes": len(needles),
        "retained_in_context": n_retained,
        "retention_pct": round(100 * n_retained / len(needles), 1) if needles else None,
    }
    if use_llm:
        n_correct = sum(1 for n in needles if n.get("llm_answer_correct"))
        needle_summary["llm_correct"] = n_correct
        needle_summary["llm_accuracy_pct"] = round(100 * n_correct / len(needles), 1) if needles else None

    return {
        "summary": {
            "turns_measured": len(rows),
            "naive_input_tokens_per_turn": stats(naive),
            "semvec_input_tokens_per_turn": stats(semvec),
            "total_naive_input_tokens": total_naive,
            "total_semvec_input_tokens": total_semvec,
            "input_reduction_pct_overall": round(reduction, 1),
            "break_even_turn": breakeven,
            "needle_quality": needle_summary,
        },
        "per_turn": rows,
        "needles": needles,
    }


def print_report(report: dict) -> None:
    s = report["summary"]
    line = "=" * 78
    print(line)
    print("MEASUREMENT HARNESS  ·  input tokens/turn  naive vs. Semvec  (embedder: mpnet/768)")
    print(line)
    print(f"Turns measured: {s['turns_measured']}")
    n, v = s["naive_input_tokens_per_turn"], s["semvec_input_tokens_per_turn"]
    print(f"  naive  input/turn:  mean {n['mean']:,.0f}  median {n['median']:,.0f}  "
          f"p90 {n['p90']:,.0f}  max {n['max']:,.0f}")
    print(f"  Semvec input/turn:  mean {v['mean']:,.0f}  median {v['median']:,.0f}  "
          f"p90 {v['p90']:,.0f}  max {v['max']:,.0f}")
    print()
    print(f"  Input reduction overall:     −{s['input_reduction_pct_overall']:.1f} %")
    be = s.get("break_even_turn")
    if be is not None:
        print(f"  Break-even from turn {be} on (Semvec cheaper and staying cheaper)")
    else:
        print("  Break-even: never reached on this profile")
    q = s["needle_quality"]
    print()
    print("  Quality proxy (needles):")
    if q["retention_pct"] is not None:
        print(f"    Fact retained in Semvec context: {q['retained_in_context']}/{q['total_probes']} "
              f"({q['retention_pct']:.0f} %)")
    if "llm_accuracy_pct" in q and q["llm_accuracy_pct"] is not None:
        print(f"    LLM answer correct:              {q['llm_correct']}/{q['total_probes']} "
              f"({q['llm_accuracy_pct']:.0f} %)")
    print(line)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--profile", type=Path, help="synthetic profile (YAML)")
    g.add_argument("--custom-profile", type=Path, help="your own conversation profile (JSONL)")
    p.add_argument("--llm", action="store_true", help="real LLM quality judgment (OPENAI_* from .env)")
    p.add_argument("--device", default=None, help="embedder device (cpu|cuda); default: auto")
    p.add_argument("-o", "--output", type=Path, help="JSON report path")
    args = p.parse_args()

    count_tokens = make_token_counter()

    serializer_cfg: dict[str, Any] = {}
    if args.profile:
        profile = yaml.safe_load(args.profile.read_text())
        convs = generate_conversations(profile, count_tokens)
        preamble_tokens = profile["context"]["static_preamble_tokens"]
        serializer_cfg = profile.get("serializer", {})
    else:
        convs = load_custom_profile(args.custom_profile)
        preamble_tokens = 800

    print(f"Profile: {len(convs)} conversations, "
          f"{sum(len(c.turns) for c in convs)} turns total. Embedder loading …", file=sys.stderr)

    report = measure(
        convs,
        static_preamble_tokens=preamble_tokens,
        count_tokens=count_tokens,
        use_llm=args.llm,
        device=args.device,
        serializer_cfg=serializer_cfg,
    )
    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nReport → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
