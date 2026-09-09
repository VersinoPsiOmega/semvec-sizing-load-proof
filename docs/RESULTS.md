# Sizing & Load Proof — Technical Results Report

**Headline:** the reference chatbot workload can be served with a Semvec memory layer in
front of the LLM at **near-parity answer quality using 74 % fewer LLM GPUs**
(H100-NVL class) — on-prem/air-gapped, and independently verifiable by whoever deploys it.
Every number below is labelled **MEASURED**, **ASSUMPTION** or **REFERENCE**. All three
pillars were measured with the **same tuned configuration** (mpnet/768 + hybrid BM25 +
cross-encoder rerank).

Executive summary: [RESULTS-EXECUTIVE.md](RESULTS-EXECUTIVE.md) ·
Method and open inputs: [METHODOLOGY.md](METHODOLOGY.md) ·
Semvec-layer infrastructure: [INFRA-SIZING.md](INFRA-SIZING.md) ·
**Environment and artefact index: [MEASUREMENT-SETUP.md](MEASUREMENT-SETUP.md)**

---

## 1. The reference workload

A deliberately ordinary large-enterprise chatbot rollout, used here as the fixed target all
three pillars are measured against. It is *not* a vendor benchmark configuration — it is the
kind of requirement set that makes conventional sizing expensive.

| Quantity | Value |
|---|---|
| Users | ~2,000 staff / ~1,500 FTE |
| Tokens/month | 1,000,000 per user → **2 bn total** (see note) |
| In/out ratio | **4:1** → ~1.6 bn input, ~0.4 bn output tokens/month |
| Concurrency | avg 200, **peak 400** |
| TTFT | p50 < 500 ms, p90 < 1000 ms |
| TPS (output) | p50 > 25 t/s, p90 > 10 t/s |
| Data protection | no data to third parties → on-prem / air-gapped |

> **Note on the token basis.** The workload is stated as 2 bn tokens/month (≈2,000 employees ×
> 1m). The sizing configs deliberately compute on the ~1,500 FTE actually generating load, i.e.
> **1.5 bn tokens/month**. That is the conservative direction: a smaller input volume makes the
> *naive* scenario look better, so it understates rather than inflates the saving attributed to
> the memory layer. Both scenarios use the same basis, so the relative comparison is unaffected.

**Cost driver.** TTFT at peak concurrency is what makes this expensive. TTFT ≈ prefill time,
and prefill scales with input length; at a 4:1 ratio, **input is the lever**. A conventional
full-context sizing estimate for this workload lands in the region of 30 H100-class GPUs
(pillar 1 reproduces that number from first principles as the *naive* scenario).

## 2. Approach

**Semvec** is a conversation compressor placed **in front of** the LLM — not a RAG vector
store. It holds a fixed-size state, so the **input per turn stays constant** (O(1) instead of
O(n) as the conversation grows). That lowers prefill (→ TTFT), the KV cache (→ more users per
GPU) and total prefill FLOPs simultaneously.

## 3. Assumed vs. measured inputs (the basis of every calculation)

| Parameter | Value | Status | Note |
|---|---|---|---|
| Users / tokens / ratio / concurrency / SLA | see §1 | **REFERENCE** | the fixed target |
| Turns per session | **12** | ASSUMPTION | confirm against a real deployment profile |
| Mean input tokens/turn (naive) | **6,000** | ASSUMPTION | **the main lever** on the GPU count |
| Peak input tokens/turn (naive) | 26,000 | REFERENCE | gpt-4-turbo-128K / LOCOMO comparison |
| Output tokens/turn | 220 | ASSUMPTION | derived |
| Semvec input tokens/turn | **1,424** | **MEASURED** | tuned LOCOMO full suite (§4, pillar 2) |
| Working time | 22 days × 8 h | ASSUMPTION | effective seconds/month |
| Turn interval (peak arrival rate) | **~10 s** | ASSUMPTION | LLM generation + reading; 400 users ÷ 10 s = 40 turns/s |
| Target model | **Qwen3.6-27B** | MEASURED/deployed | final choice open (70B → re-run) |
| Target GPU | **H100 NVL (94 GiB)** | MEASURED | a real, live-benchmarked card |
| Prefill/decode per GPU | **7,810 / 1,970 t/s** | **MEASURED** | micro-benchmark against live vLLM |

> "Turns/session = 12" does **not** drive the sizing number directly — that depends on the
> **naive input/turn (6,000)** plus concurrency and the TTFT budget. "6,000" is the most
> sensitive assumption in the whole model; pillar 1's sensitivity sweep quantifies its effect.

## 4. Method and results per pillar

### Pillar 1 — Token→hardware sizing (`sizing/`)

**Method.** GPU count = max(prefill-bound, decode-bound, kv-memory-bound). The prefill
requirement is **peak arrival rate** (400 users ÷ ~10 s turn interval ≈ **40 turns/s**) ×
input tokens/turn, divided by the **actually measured** prefill throughput (7,810 t/s/GPU,
via `calibrate_vllm.py` against a live vLLM serving Qwen3.6-27B).

| Scenario | GPUs (H100 NVL, 94 GiB) | Rough CapEx | Binding constraint |
|---|---|---|---|
| **naive** | **31** | ~€0.99M | prefill |
| **with Semvec** | **8** | ~€0.26M | prefill |
| **Saving** | **−23 GPUs (−74 %)** | **~€0.74M** | |

> The saving is robust because both scenarios divide by the same measured constant.
> The absolute count scales with the naive input/turn (assumption: 6,000). CapEx figures are
> illustrative: they come from the configurable `gpu_cost_eur` knob (default ~€30k/GPU list
> price order of magnitude), not from any vendor quote.

### Pillar 2 — Token reduction & answer quality (`harness/`)

**Method.** The real **LOCOMO suite** (10 conversations, **1,985 QA**, 369–689 turns each,
~22 per session). Every question is answered twice by a real LLM: **naive** (the whole
conversation in the prompt) vs. **Semvec** (tuned retrieval: dense + BM25/RRF + cross-encoder
rerank → top_k 15, 10k-character budget). Reader **Qwen3.6-27B**; scoring by word-level F1
plus an LLM judge (LOCOMO-J, gpt-4o-mini); category 5 is adversarial (abstaining is correct).

| Metric | naive | with Semvec |
|---|---|---|
| Input tokens/QA | 18,513 | **1,424 (−92.3 %)** |
| Answer F1 (word) | 0.527 | **0.482 (keeps 91 %)** |
| LLM judge | 0.576 | **0.506 (keeps 88 %)** |

→ **Near-parity at −92 % input.** 0.482 F1 / 0.506 judged sits at published levels
(F1 ~0.495 / J ~0.605); on adversarial questions (category 5, where abstaining is correct)
Semvec's word F1 is **0.966 vs. naive 0.933** — it is *better* than full context at knowing when
not to answer. Evidence: `results/locomo_qa_tuned_full.json` (1,984 QA, live).

> **A note on the judge.** The reader model (Qwen3.6-27B) also acts as its own judge here.
> Word-F1 is unaffected by that; for the LLM-judged figures a separate judge model would be
> preferable, so read them with that caveat.

**Air-gapped baseline (`run_profile.py`, no LLM).** On the shipped synthetic profile
(20 conversations, 324 turns): input/turn **naive 2,639 → Semvec 1,741 (−34.0 %)**, cheaper from
**turn 1** on and staying cheaper, needle retention **42/42 (100 %)**. Evidence:
`results/harness_synthetic.json`. Both paths are counted as a deployment sends them — system
prompt + context + user message — so the memory layer is credited only for replacing the history.

The −34 % sits well below the −92.3 % measured on LOCOMO, and the reason is conversation length:
a constant-cost state can only save what the history would have cost. These conversations run a
median of ~16 turns; the LOCOMO dialogues run 369–689. The break-even is early (turn 1); the
magnitude grows from there. This run needs no endpoint, no licence key and no GPU, which makes it
the one figure here anybody can reproduce in minutes — but it is a retrieval proxy, not an
answer-quality measurement. The two QA suites supply the latter.

**Hard test — LongBench-v2 (cross-validation, very long documents, 60 items).** A second,
harder benchmark as an independent counter-check to LOCOMO: multiple choice over contexts
from 25k to >1M tokens. 20 short / 20 medium / 20 long, same reader (Qwen3.6-27B), embedder
mpnet/768, tuned retrieval. Per item: chunk the context → Semvec state → retrieval + rerank →
compact context → A/B/C/D (scored exactly). Naive = the whole context in the prompt, **only
when it fits the 256k window**. Mean context **249,498 tokens**. Harness:
`harness/run_longbench_qa.py`.

| Metric | naive (full context) | with Semvec |
|---|---|---|
| Feasible | **38/60** (context fits) | **60/60** |
| naive IMPOSSIBLE (context > window) | **22/60** → no answer at all | Semvec answers |
| Input tokens/item (feasible) | 71,832 | **1,895 (−97.4 %)** |
| Accuracy (feasible) | 0.553 | 0.447 |
| Accuracy, "long" subset (n=20) | **0.00** — only 1 of 20 fits the window at all, and that one was answered wrongly | **0.40** (chance 0.25) |

Per length (naive / Semvec, naive feasible items in brackets): short 0.550 / 0.350 (20/20) ·
medium 0.588 / 0.500 (17/20) · long 0.000 / 0.400 (1/20).

→ **Key finding: Semvec makes possible what is naively impossible.** On 22 of 60 items the
context exceeds the model window, so only Semvec answers at all (long 0.400 ≫ 0.25 chance). At
−97.4 % input Semvec keeps **81 %** of naive accuracy on the feasible items (medium 85 %); on
*short* contexts naive is stronger — short contexts fit the window anyway, so compressing buys
nothing there.
**Semvec's advantage grows with context length.** Reasoning/"thinking" was disabled for both
paths (keeping the comparison fair; both would score higher with it). Evidence:
`results/longbench_v2_qwen27b.json`.

**Two benchmarks, same picture (cross-validation).** LOCOMO (1,984 QA, long dialogues):
−92 % tokens at 88–91 % of quality. LongBench-v2 (60 items, very long documents): −97 %
tokens at 81 % on feasible items **and answers what is otherwise unanswerable**. The effect is
therefore **not benchmark-specific** but holds across two domains. Both used the same reader, so
reader-independence is not claimed here.

### Pillar 3 — Load demonstration (`load/`), 8 GB GPU, tuned stack (rerank on)

**Method.** k6 drives the Semvec REST API; each VU holds one persistent session (pinned to a
worker via `__VU`, which is what the default `memory` backend requires); sidecar embedder
(mpnet/768) plus `/v1/run` with hybrid BM25 and cross-encoder rerank. Hardware: RTX 5060 8 GB,
24 cores, 16 GB RAM. semvec 0.8.8. All figures below were measured live for this report — see
[MEASUREMENT-SETUP.md](MEASUREMENT-SETUP.md).

**Capacity.** Driven to saturation (`think=0`) at rising concurrency, the tuned stack peaks and
then degrades on **this** configuration. The last row uses a different one (24 workers, no
reranker) and is included to show where the *host*, not the memory layer, runs out:

| Concurrency | Throughput | `/v1/run` med | p90 | GPU util | VRAM peak | Errors |
|---|---|---|---|---|---|---|
| 200 VU | **172 embeds/s** | 727 ms | 1,475 ms | 92 % | 4.53 GiB | 0 / 24,186 |
| 400 VU | 147 embeds/s | 2,223 ms | 3,318 ms | 97 % | 4.18 GiB | 0 / 21,889 |
| 1,000 VU (no rerank, 24 workers) | 85 embeds/s | 7,139 ms | 15,834 ms | 82 % | 3.52 GiB | 0 / 13,059 |

At 1,000 VU the GPU sat at **82 %** with 3.5 of 8 GiB VRAM used — it was not the constraint. CPU
peaked at 691 % against 278 % on the 200-VU run, with twice the API workers on a 24-core laptop.
That is a saturated host, not a saturated memory layer.

**Embedding capacity is a tier you size yourself.** `--daemons N` starts N batched embedder
daemons here; in a deployment the daemon is an ordinary process
(`python -m semvec.embedder --listen …`) that the API finds through `SEMVEC_EMBEDDER_URL`, and a
`tcp://` endpoint lets one GPU node serve an auto-scaled API tier on CPU instances. The
172 embeds/s below is what **two daemons on one 8 GB laptop GPU** delivered, not a ceiling of the
architecture. Sizing that tier for your own load is a configuration decision this repository does
not measure for you.

**Latency at realistic pacing** (10 s think time — how a human actually uses a chatbot):

| Configuration | median | p90 | p95 | p99 | GPU | Errors | Evidence |
|---|---|---|---|---|---|---|---|
| Tuned stack (rerank) | **35.6 ms** | 61.9 ms | 66.0 ms | 92.1 ms | 14 % | 0 / 629 | `load_realistic_rerank.json` |
| No rerank | 35.7 ms | 57.8 ms | 66.0 ms | — | 12 % | 0 / 645 | `load_realistic_60vu.json` |

→ **The reranker is free on both axes.** 35.6 ms with it vs. 35.7 ms without at realistic pacing,
and absorbed on throughput too. At ~36 ms the memory layer takes **~7 % of a 500 ms TTFT budget**.

**Headroom.** 400 peak users at ~10 s/turn ≈ **40 embeds/s** = **23 % of the 172 embeds/s
plateau, a 4.3× reserve**, on one 8 GB laptop GPU, with zero errors in every run above.

> **Not claimed here:** the ONNX/GPU embedder path. It needs a Rust binary and an ONNX export
> outside the PyPI-only baseline, so no ONNX throughput figure is given.

## 5. Configuration consistency

All three pillars use the same tuned stack (mpnet/768 + hybrid BM25 + cross-encoder rerank,
`configs/locomo_aligned.env`): pillar 1 consumes the Semvec input measured in pillar 2
(1,424 tokens/turn), and pillar 3 measures that same rerank path under load.

## 6. SLA reconciliation

| Requirement | Result | Status |
|---|---|---|
| TTFT p50 < 500 ms | prefill input ~6,000 → ~1.5k/turn; H100 ≈ ~180 ms instead of ~730 ms, + **35.6 ms** measured memory-layer overhead on the tuned stack at realistic pacing | ✅ |
| TPS p50 > 25 t/s | measured decode 1,970 t/s per GPU at concurrency 48 → **~41 t/s per stream** — and ~50 streams/GPU is exactly the target density (400 peak ÷ 8 GPUs) | ✅ |
| Concurrency 400 | the memory layer carries it at **23 %** utilisation of one 8 GB card (4.3× reserve, 0 errors) | ✅ |
| Cost | **−74 % LLM GPUs** | ✅ |
| on-prem / air-gapped | in-process Rust core + a state store inside the boundary (Postgres, Redis or MongoDB; no external ANN database) | ✅ |

## 7. Established vs. still open

**Established (measured — see [MEASUREMENT-SETUP.md](MEASUREMENT-SETUP.md)).** The H100
calibration; **−74 % LLM GPUs**; −92.3 % input at near-parity over the full 1,984-QA LOCOMO
suite; the LongBench-v2 result including the 22 items no full-context path can answer;
172 embeds/s capacity with zero failed requests across four load levels; 35.6 ms memory-layer
overhead on the tuned stack at realistic pacing; the state-blob size; the infrastructure
footprint.

**Not established here.** Throughput on a datacenter GPU (all load figures come from one 8 GB
laptop GPU); a sized embedder tier (every run uses two daemons on that one GPU); the Kubernetes
deployment, which is described but not measured; a judge model distinct from the reader; a second
reader model, so the effect is not claimed to be reader-independent; the ONNX/GPU embedder path;
any decode-bound target.

## 8. Infrastructure of the Semvec layer

Small, and strictly separate from the LLM GPU pool: **1 small GPU (8 GB slice) + 1 app VM
(8 vCPU / 16 GB) + 1 small state store** carry the full 400 peak users with headroom. A real
deployment runs this on Kubernetes, where the API and embedder tiers are separate Deployments with
separate autoscalers — see [INFRA-SIZING.md](INFRA-SIZING.md) §4, variant C. Measured
state size is 132–278 KiB per session (median ~174 KiB), so ~0.34 GB at 2,000 sessions. Details:
[INFRA-SIZING.md](INFRA-SIZING.md).

## 9. Reproducibility

One command per pillar, offline/air-gapped where noted: `sizing/sizing_model.py` ·
`sizing/calibrate_vllm.py` · `harness/run_locomo_qa.py --judge` · `load/run_load.py --rerank`.
99 automated tests, all green.

## 10. Measurement setup

Every figure above was measured against semvec 0.8.8 with the reader, embedder, retrieval
configuration and hardware recorded in [MEASUREMENT-SETUP.md](MEASUREMENT-SETUP.md), which also
indexes which artefact backs which claim. `tools/check_claims.py` fails CI if a documented figure
stops matching its artefact.

## 11. Third-party datasets

Neither dataset is bundled. Pass your own copy via `--dataset`.

- **LOCOMO** — <https://github.com/snap-research/locomo> (`locomo10.json`). Per-item question
  and gold-answer text is **not** redistributed in `results/`; metrics, token counts,
  conversation id and category are retained, so every aggregate stays auditable.
- **LongBench-v2** — <https://github.com/THUDM/LongBench>. `results/longbench_v2_*.json`
  carries only official item ids, the domain/length label, the gold letter and the two
  predicted letters.
