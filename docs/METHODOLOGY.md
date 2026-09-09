# Methodology

How the numbers in this repository are produced, which of them are measured, and exactly what
would change them. If you want to challenge a result, this page tells you where to push.

Results: [RESULTS.md](RESULTS.md) · Executive summary: [RESULTS-EXECUTIVE.md](RESULTS-EXECUTIVE.md)

---

## 1. The question being answered

A conversational assistant sends the growing conversation history to the LLM on every turn
(*full-context replay*). Input per turn therefore grows with conversation length, O(n). Because
time-to-first-token is essentially prefill time, and prefill scales with input length, the
TTFT target at peak concurrency is what sets the GPU count — not the user count and not the
volume of generated text.

A memory layer that holds a **fixed-size state** in front of the LLM makes input per turn
**constant**, O(1). The question this repository answers with measurements rather than claims:

1. How many GPUs does that actually save for a realistic enterprise workload?
2. What does it cost in answer quality?
3. What does the memory layer itself cost in hardware?

One pillar per question.

## 2. Pillar 1 — token→hardware sizing (`sizing/`)

A parameterised model deriving GPU demand from workload, SLA and infrastructure inputs for
**two scenarios** — naive full-context replay vs. constant-input Semvec — and comparing them.
Nothing is hardcoded: every input comes from a YAML config or a CLI flag.

**Inputs.** Workload (tokens/user/month, user count, in/out ratio, avg/peak concurrency,
turns/session, mean and peak naive input tokens/turn, bounded Semvec input/turn, output
tokens/turn) · SLA (TTFT p50/p90, TPS p50/p90) · Infrastructure (model size, quantisation, GPU
type, GPUs/node, effective work seconds/month).

**Logic.**

- Monthly total tokens = tokens/user × users; input/output split via the ratio.
- Sustained decode rate = output tokens/month ÷ work seconds.
- Peak aggregate decode = peak concurrency × required per-stream TPS.
- Prefill load: sustained = input tokens/month ÷ work seconds; **peak is arrival-driven** ∝
  peak concurrency × mean input tokens/turn ÷ turn interval.
- KV cache per request ≈ 2 × layers × hidden × kv-head-ratio × sequence length × bytes;
  concurrency per GPU = (GPU memory − weights) ÷ KV per request.
- **GPU count = max(prefill-bound, decode-bound, kv-memory-bound)** per scenario, and the
  output names which constraint binds.

**Calibration is not optional.** Per-GPU prefill and decode throughput for the chosen model are
**calibratable constants** that must come from a real micro-benchmark (vLLM / TensorRT-LLM on
the target GPU) — never a guessed hardcode. Uncalibrated constants are printed with a loud
`UNCALIBRATED` marker, and `sizing/calibrate_vllm.py` writes measured values into a config.

**On the arrival model.** Peak prefill is driven by how often each active user actually sends a
turn, not by the TTFT budget. Using the 1 s TTFT budget as the arrival interval would model a
thundering herd where all 400 users press enter in the same second. The default assumes ~10 s
between turns (LLM generation plus reading), which is the realistic case; it is a documented,
configurable assumption (`peak_turn_interval_s`).

## 3. Pillar 2 — token reduction and answer quality (`harness/`)

Three harnesses, increasing in cost and rigour:

| Harness | What it does | Needs an LLM? |
|---|---|---|
| `run_profile.py` | Input tokens/turn with vs. without Semvec on a conversation profile, plus a needle-retention proxy (facts set early, queried later) | No — air-gapped |
| `run_locomo_qa.py` | The full LOCOMO suite answered twice by a real reader, scored by word F1 and an LLM judge | Yes |
| `run_longbench_qa.py` | LongBench-v2 multiple choice over very long documents, scored exactly | Yes |

Token counting is exact via tiktoken, with the **same encoder for both paths**, so the
reduction figure is not an artefact of two different tokenisers. Both paths in the QA harnesses
see the same questions and the same reader; only the context construction differs.

**The needle proxy.** `run_profile.py` runs without an LLM, which makes it verifiable inside an
air-gapped environment — but it can then only prove that a fact *survives compression and is
retrieved*, not that a model answers correctly from it. That is why the LOCOMO and LongBench-v2
harnesses exist: they close the gap with a real reader and are the basis for every quality claim
in [RESULTS.md](RESULTS.md).

**Fairness rules applied.** Reasoning/"thinking" is disabled for both paths in the LongBench-v2
run (enabling it would raise both). Naive is counted as infeasible — not as wrong — when the
context exceeds the model window, and those items are reported separately rather than folded
into an average that would flatter Semvec.

## 4. Pillar 3 — load demonstration (`load/`)

Drives the Semvec REST API at the concurrency of the reference workload and reports what it
costs in hardware: `/v1/run` latency percentiles, sustained throughput and the GPU/CPU/RAM
footprint. Deployment shape: N batched embedder daemons plus M lightweight stateless API
workers in sidecar mode, driven by k6 with one persistent session per virtual user.

Each VU is pinned to one worker, which models the **sticky routing** the default `memory` backend
requires (a session must reach the worker that holds its state; a version CAS guards against
clobbering). That is deliberately the constrained case. Since 0.8.x the `redis` and `mongo`
backends make the store authoritative and lift the requirement entirely — a deployment on either
would be *less* constrained than what pillar 3 measured, not more.

## 5. Scope — what this is not

- **Not** a replacement for a RAG vector store. Semvec compresses the *conversation*; document
  retrieval is a separate concern and stays wherever it is.
- **No** change to output token volume. The savings are entirely on the input side.
- **No** final hardware or price commitment. That requires calibration on the target GPU plus
  the open inputs below.
- **Not** a production gateway integration.

## 6. Open inputs

These are the inputs that would move the numbers. They are the reason this repository reports an
order of magnitude and a robust *relative* saving rather than a committed GPU count.

1. **Mean and peak input tokens per turn** in the naive path. The single most sensitive input —
   run the sensitivity sweep to see its effect directly:
   `sizing/sizing_model.py --config … --sweep naive_mean_input_tokens_per_turn:2000:12000:2000`
2. **Turns per session** / the session length distribution. This is what sets the size of the
   saving: a constant-cost state can only save what the history would have cost. Measured
   break-even is turn 1, but the magnitude scales — −34 % over ~16-turn conversations against
   −92 % over ~588-turn ones.
3. **Target model and quantisation.** Measured here on Qwen3.6-27B; a 70B target needs a re-run.
4. **Target GPU** — specifically the one actually being priced.
5. **Whether reasoning models are in scope.** They change output volume substantially, which is
   both SLA- and sizing-relevant.

## 6a. Measurement hygiene

A benchmark result is only as good as the harness that produced it, so a few rules are enforced
in this repository rather than left to care:

- **Child processes never write into an unread pipe.** A full pipe buffer blocks the child, and
  its in-flight requests then hang to the client timeout while the GPU sits idle. Children log to
  files instead, which also survive for post-mortem.
- **Entropy for structure is separate from entropy for content.** The synthetic profile's filler
  text draws from its own rng, derived from `(seed, conversation, turn)`. Sharing one rng would
  let the tokeniser or the word pool change turns-per-conversation, and `seed:` would stop meaning
  "reproducible run".
- **Environment values are stripped.** A trailing carriage return inside a URL — which is what a
  CRLF `.env` produces — fails with an opaque parse error against a reachable endpoint.
- **No absolute paths in artefacts.** k6 embeds whatever path it is handed; reports keep it
  repo-relative so committed artefacts stay portable.
- **Capacity is the plateau of the configuration under test, not the largest load the harness
  accepts.** Throughput peaks at 200 concurrent sessions here and halves by 1,000 — but at that
  point the GPU is at 82 % and the host CPU has more than doubled, so the number describes the
  host, not the memory layer. State the configuration with the figure, and say which tier ran out.
- **Documented figures are checked against the artefacts.** `tools/check_claims.py` runs in CI.

## 7. What "measured" means here

Every figure in this repository carries one of three labels, and they are used strictly:

- **MEASURED** — produced by a run in this repository, with the artefact committed under
  `results/`.
- **ASSUMPTION** — our default, documented and configurable. Replaceable with one config edit.
- **REFERENCE** — a published third-party value used for comparison.

[RESULTS.md](RESULTS.md) §10 additionally records which dependency version and which hardware
each committed artefact was produced with, so a result is never silently re-labelled when a
dependency moves.

## 8. References

- Semvec documentation: <https://semvec-docs.pages.dev/> — in particular
  `api-reference/token-reduction`
- PyPI package: <https://pypi.org/project/semvec/>
- LOCOMO benchmark: <https://github.com/snap-research/locomo>
- LongBench-v2: <https://github.com/THUDM/LongBench>
