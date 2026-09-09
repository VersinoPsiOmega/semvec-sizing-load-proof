```
   _____ _________   _____   ________   ____  ____  ____  ____  ______
  / ___//  _/__  /  /  _/ | / / ____/  / __ \/ __ \/ __ \/ __ \/ ____/
  \__ \ / /   / /   / //  |/ / / __   / /_/ / /_/ / / / / / / / /_
 ___/ // /   / /___/ // /|  / /_/ /  / ____/ _, _/ /_/ / /_/ / __/
/____/___/  /____/___/_/ |_/\____/  /_/   /_/ |_|\____/\____/_/

        m e a s u r e d   ·   c a l i b r a t e d   ·   a u d i t a b l e
                  what constant-cost memory costs in GPUs

  workload ──▶ sizing model ──▶ ⟨prefill | decode | kv-mem⟩ ──▶ GPUs · 31 → 8
```

# Sizing & Load Proof: Cutting LLM GPUs with a Constant-Cost Memory Layer

**A reproducible measurement of how many GPUs you stop needing when a fixed-size memory layer
replaces full-context replay in front of your LLM — plus the harnesses to measure it on your own
workload.**

Serving a conversational assistant on your own hardware, the expensive part is rarely the
answers. It is that every turn re-sends the whole conversation history to the model. Input per
turn grows with conversation length (O(n)), time-to-first-token is essentially prefill time,
prefill scales with input length — so your **time-to-first-token target at peak concurrency sets
your GPU count.**

Hold a **fixed-size state** in front of the model instead and input per turn becomes **constant**
(O(1)), independent of how long the conversation gets. That hits all three hardware drivers at
once: shorter prefill → lower TTFT · smaller KV cache → more concurrent users per GPU · fewer
total prefill FLOPs.

This repository measures the size of that effect end to end and hands you the tooling to check it
against your own numbers instead of trusting ours.

## Results at a glance

For a reference enterprise workload — ~2,000 staff, 2 bn tokens/month, 4:1 input:output, peak
concurrency 400, TTFT p50 < 500 ms, strictly on-prem:

| | naive (full-context replay) | with a constant-cost memory layer |
|---|---|---|
| **LLM GPUs** (H100 NVL, measured calibration) | 31 | **8 (−74 %)** |
| **Input tokens/QA** (LOCOMO, 1,984 QA, real reader) | 18,513 | **1,424 (−92.3 %)** |
| **Answer quality** (LOCOMO, LLM-judged) | 0.576 | 0.506 — **keeps 88 %** |
| **Very long documents** (LongBench-v2, mean 249k-token contexts) | **22 of 60 items impossible** (context > window) | **60/60 answerable** at −97.4 % input |
| **Cost of the memory layer itself** | — | **one 8 GB GPU** at ~23 % utilisation for all 400 peak users |

Every number in this repository was measured against **semvec 0.8.8** on the hardware named in
[docs/MEASUREMENT-SETUP.md](docs/MEASUREMENT-SETUP.md), and the artefact behind each one is
committed under `results/`. Figures are labelled MEASURED, ASSUMPTION or REFERENCE throughout;
`tools/check_claims.py` fails CI if a documented figure stops matching its artefact.

Full detail: **[docs/RESULTS.md](docs/RESULTS.md)** · Method and open inputs:
**[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**

## What you can do with this

- **Size your own deployment before buying hardware.** Pillar 1 is a standalone,
  dependency-light model — no GPU, no network, no licence. Put in your workload and SLA, get a
  GPU count per scenario with the binding constraint named, and sweep the assumption you trust
  least.
- **Measure the token reduction on your own conversations.** Pillar 2 runs air-gapped without an
  LLM, so it works inside environments that will never call an external API.
- **See what the memory layer costs in hardware** before you allocate any (pillar 3).
- **Check our numbers.** One command per pillar; results are committed so you can diff.

---

## The three pillars

### Pillar 1 — Token→hardware sizing (`sizing/`)

Derives the GPU count from workload + SLA + infrastructure and compares **naive vs. memory
layer**, each with its binding constraint (prefill / decode / kv-memory).

```bash
.venv/bin/python sizing/sizing_model.py --config sizing/configs/reference_2b.yaml
# sensitivity: mean input tokens/turn → GPU curve
.venv/bin/python sizing/sizing_model.py --config sizing/configs/reference_2b.yaml \
    --sweep naive_mean_input_tokens_per_turn:2000:12000:2000
```

**Calibrated against a live H100 NVL.** `sizing/calibrate_vllm.py` benchmarks an
OpenAI-compatible vLLM endpoint and writes the real per-GPU throughput into a config. Measured
on an H100-NVL-94GB running **Qwen3.6-27B**: prefill **7,810 t/s**, decode **1,970 t/s** per GPU
(measured across a WAN link, so the round-trip is included and the figures are conservative).
With those, `sizing/configs/qwen3_27b_h100.yaml` gives **naive 31 → 8 GPUs (−74 %)**,
prefill-bound.

```bash
python sizing/calibrate_vllm.py --base-url https://<vllm-host>/v1 --model Qwen/Qwen3.6-27B \
    --num-gpus 1 --concurrency 48 --write sizing/configs/qwen3_27b_h100.yaml
python sizing/sizing_model.py --config sizing/configs/qwen3_27b_h100.yaml
```

> ⚠️ Run with the shipped defaults in `reference_2b.yaml` and the throughput constants are
> **UNCALIBRATED** ballparks, and the output says so. Calibrate on your target GPU before quoting
> any number.

The absolute GPU count depends on the workload assumptions (mean input tokens/turn, arrival
model). The **saving is the robust part**, because both scenarios divide by the same measured
constant.

### Pillar 2 — Measurement harness (`harness/`)

Measures **input tokens/turn with vs. without the memory layer** on a conversation profile, plus
a quality proxy ("needles": facts set early, queried later). In-process via `SemvecChatProxy`,
embedder **mpnet/768**, exact token counting via tiktoken with the same encoder on both paths —
**no LLM required** (air-gapped). `--llm` adds a real LLM quality judgment.

```bash
.venv/bin/python harness/run_profile.py \
    --profile harness/profiles/reference_synthetic.yaml -o results/harness_synthetic.json
# your own conversations (JSONL, one conversation per line):
.venv/bin/python harness/run_profile.py --custom-profile my_profile.jsonl -o results/...
```

**Result on the shipped synthetic profile** (20 conversations, 324 turns, no LLM — reproducible
in a couple of minutes with no endpoint and no licence key): input/turn **naive 2,639 → Semvec
1,741 (−34.0 %)**, cheaper **from turn 1 onward** and staying cheaper, with **needle retention
42/42 (100 %)** — every fact planted early was still retrievable later. Evidence:
`results/harness_synthetic.json`.

Both paths are counted the way a deployment sends them: system prompt + context + the user's
message. The memory layer replaces the *history*, not the system prompt and not the question, so
neither is netted out of the comparison.

**Why −34 % here and −92 % on LOCOMO below.** These conversations are short — a median of about
16 turns, a maximum of 40. A constant-cost state can only save what the history would have cost,
so the saving tracks conversation length: it is small on short sessions and grows from there. The
LOCOMO dialogues run 369–689 turns, which is where the effect is fully visible. Take the −34 % as
what this profile shows, not as the ceiling.

This is also a *proxy*: without an LLM it shows that a planted fact survives compression and is
retrieved, not that a model answers correctly from it. The two QA suites below close that gap
with a real reader.

**LOCOMO full suite** (1,984 QA, reader Qwen3.6-27B, tuned retrieval: hybrid BM25 +
cross-encoder rerank, LLM judge — `harness/run_locomo_qa.py`): input tokens/QA **18,513 → 1,424
(−92.3 %)**; answer F1 **0.527 → 0.482 (keeps 91 %)**; LLM-judged accuracy **0.576 → 0.506
(keeps 88 %)** → near-parity at −92 % input. On adversarial questions the memory layer is
*better* than full context (F1 **0.966 vs. 0.933**) — it knows when not to answer. Evidence:
`results/locomo_qa_tuned_full.json`.

The LOCOMO conversations are long multi-session dialogues (369–689 turns each, ~588 on average
over 19–32 sessions), which is where a constant-cost state is fully visible. The measured
break-even on the synthetic profile is **turn 1** — the memory layer is cheaper from the second
turn on and stays cheaper — but *how much* cheaper scales with how much history it replaces:
−34 % over ~16-turn conversations, −92 % over ~588-turn ones.

**LongBench-v2** (60 items, mean context **249,498 tokens**, `harness/run_longbench_qa.py`)
reproduces the effect in a different domain — and on **22 of 60 items the naive path cannot
answer at all**, because the context exceeds the model window:

| Metric | naive (full context) | with the memory layer |
|---|---|---|
| Feasible | 38/60 (context fits) | **60/60** |
| naive IMPOSSIBLE (context > window) | **22/60** → no answer at all | answers |
| Input tokens/item (feasible) | 71,832 | **1,895 (−97.4 %)** |
| Accuracy (feasible) | 0.553 | 0.447 |
| Accuracy, "long" subset (n=20) | **0.000** — only 1 of 20 fits the window, and it was answered wrongly | **0.400** (chance 0.25) |

Per length (naive / memory layer): short 0.550 / 0.350 · medium 0.588 / 0.500 · long 0.000 /
0.400. **The advantage grows with context length**, and on short contexts — which fit the window
anyway — the naive path is stronger. Reasoning was disabled for both paths, keeping the
comparison fair.

Neither dataset is bundled (third-party); pass your own copy:

```bash
python harness/run_locomo_qa.py --dataset /path/to/locomo10.json \
    --base-url $OPENAI_BASE_URL --model $OPENAI_MODEL --conv-limit 10 --limit-qa 0 --judge
```

### Pillar 3 — Load demonstration (`load/`)

Drives the Semvec REST API under load at the scale of the reference workload, via the sidecar
embedder with parallel batched daemons (mpnet/768) and many lightweight API workers. Measures
`/v1/run` latency, throughput and the **GPU/CPU/RAM footprint**, so the answer is
hardware-anchored rather than theoretical.

```bash
# prerequisites: SEMVEC_LICENSE_KEY in the environment, k6 on PATH
.venv/bin/python load/run_load.py --rerank --workers 12 --daemons 2 --batch-max 64 \
    --batch-wait-ms 50 --peak-vus 200 --avg-vus 100 --think-ms 0 \
    -o results/load_rerank_200vu.json                                   # capacity
.venv/bin/python load/run_load.py --rerank --workers 6 --daemons 1 \
    --peak-vus 60 --think-ms 10000 -o results/load_realistic_rerank.json  # human-paced
```

**Hardware:** NVIDIA RTX 5060 Laptop **8 GB**, 24 cores, 16 GB RAM.

**Capacity curve** — driven to saturation (`think=0`) at rising concurrency. Read this as *one
configuration on one small GPU*, not as a ceiling of the architecture; the last row deliberately
uses a different one:

| Concurrency | Workers / daemons | Throughput | `/v1/run` median | GPU util | VRAM peak | CPU peak | Errors |
|---|---|---|---|---|---|---|---|
| 200 VU (`--rerank`) | 12 / 2 | **172 embeds/s** | 727 ms | 92 % | 4.53 GiB | 278 % | **0 / 24,186** |
| 400 VU (`--rerank`) | 12 / 2 | 147 embeds/s | 2,223 ms | 97 % | 4.18 GiB | 294 % | **0 / 21,889** |
| 1,000 VU (no rerank) | 24 / 2 | 85 embeds/s | 7,139 ms | 82 % | 3.52 GiB | 691 % | **0 / 13,059** |

**On this configuration throughput peaks at ~172 embeds/s around 200 concurrent sessions**, and
at 1,000 VUs it drops to 85. Note what the last row does *not* say: the GPU sat at **82 %** — it
was not the limit. VRAM used 3.5 of 8 GiB, so the card had room to spare. What did move was the
host: CPU peaked at 691 % against 278 % on the 200-VU run, with 24 API workers instead of 12 on a
24-core laptop. That row measures a saturated *host*, not a saturated memory layer, and it is the
one row run without the reranker — so it is not directly comparable to the two above it either.

**You control the embedder tier separately, and that is the point.** Embedding capacity is not a
fixed property of the memory layer; it is however many daemons you choose to run:

- **In this harness:** `--daemons N` starts N batched embedder daemons and spreads the API
  workers across them. The runs above use two, because a single Python daemon's batcher — not the
  GPU — becomes the limit first.
- **In a deployment:** the daemon is an ordinary process you run yourself
  (`python -m semvec.embedder --listen …`), as many as you like, and the API finds them through
  `SEMVEC_EMBEDDER_URL`. With a `tcp://` endpoint they do not even share a host: one GPU node can
  serve an auto-scaled API tier on CPU instances.
- **In production we deploy on Kubernetes, and scaling is Kubernetes' job from there.** Semvec
  ships the manifests (`deploy/k8s/`) and a Helm chart (`deploy/helm/semvec/`). The embedder is
  its own Deployment, Service and **HorizontalPodAutoscaler**, separate from the API's — so the
  two tiers scale on their own metrics rather than in lockstep. The API dials the embedder over
  the cluster service (`SEMVEC_EMBEDDER_URL=tcp://<release>-embedder:7071`), readiness hangs on
  `/v1/readyz` and liveness on `/v1/health`. Adding embedder capacity is then a replica count,
  not a redeployment.

So the honest reading of the curve is: **~172 embeds/s is what two daemons on one 8 GB laptop GPU
delivered** — a single-box measurement, deliberately so, because it is the configuration anyone
can reproduce. The 1,000-VU row is what happens when you overload the *API* side of that one small
host. Neither is a ceiling you inherit: in a Kubernetes deployment both tiers are horizontally
autoscaled, and sizing them for your own load is a configuration decision this repository does not
measure for you.

**Per-request latency at realistic pacing** — 10 s think time between turns, which is how humans
actually use a chatbot:

| Configuration | median | p90 | p99 | GPU util | Errors |
|---|---|---|---|---|---|
| **Tuned stack (`--rerank`)**, 60 VU | **35.6 ms** | 61.9 ms | 92.1 ms | 14 % | **0 / 629** |
| No rerank, 60 VU | 35.7 ms | 57.8 ms | — | 12 % | **0 / 645** |

**The cross-encoder reranker is free on both axes.** It costs 35.6 ms vs. 35.7 ms without it at
realistic pacing, and it is absorbed on the throughput side too. At ~36 ms the whole memory layer
consumes **~7 % of a 500 ms TTFT budget**.

**Headroom for the reference workload.** 400 peak users at ~10 s/turn need ≈ **40 embeds/s**.
Against the measured 172 embeds/s plateau that is **23 % utilisation — a 4.3× reserve** — on a
single 8 GB laptop GPU, with 0 errors across every run above.

Levers to go further: a lighter embedder (MiniLM with short messages reaches several thousand
QPS), or a datacenter GPU. `run_load.py --onnx` routes the embedder to a Rust ONNX daemon, which
needs the `semvec-embedder` binary and a one-time ONNX export outside the PyPI-only baseline;
that path is not measured here and no ONNX figure is claimed.

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# k6 for pillar 3 (static binary, can be staged offline): https://k6.io/docs/get-started/installation/
# LLM access + licence: copy .env.example → .env, fill in your values, then:
set -a; . .env; set +a
```

> ⚠️ **Write `.env` with Unix line endings.** A file saved with CRLF (Windows editors, or a
> checkout on a Windows filesystem) leaves a trailing carriage return inside every value.
> `set -a; . .env` keeps it, so a URL silently becomes `https://host:port/v1\r` and every request
> fails with `URL rejected: Malformed input to a URL function` — while the endpoint is perfectly
> reachable. The harnesses strip values defensively, but fix the file: `sed -i 's/\r$//' .env`.

- **Embedder:** `paraphrase-multilingual-mpnet-base-v2` (dim 768) throughout — the LOCOMO
  default. Retrieval preset: `configs/locomo_aligned.env`.
- **What needs what:** pillar 1 needs only PyYAML. Pillar 2 needs the `semvec` package (no
  licence key) and, for the `--llm`/`--judge` paths, an OpenAI-compatible endpoint. Pillar 3
  needs a `SEMVEC_LICENSE_KEY` for the REST API plus k6.

### Licensing — read this before you plan around it

This repository — the harnesses, the sizing model, the configs and the results — is
**Apache-2.0** (see [LICENSE](LICENSE)). You can fork it, run it and adapt it freely.

**Semvec itself is commercial, proprietary software** and is not included here; pillar 3
additionally requires a licence key. Pillar 1 stands entirely on its own, so the sizing model is
useful to you regardless. See [NOTICE.md](NOTICE.md).

## Tests

```bash
.venv/bin/pip install -r requirements-test.txt   # ~4 small packages, no torch
.venv/bin/python -m pytest tests/ -q             # 99 tests, ~10 s
tools/sanitize_check.sh                          # publication gate
.venv/bin/python tools/check_claims.py           # documented figures vs. artefacts
```

No GPU, no network, no LLM and no licence key required — and no need for the ~6 GB ML stack
either. All four run in CI on every push.

## Persistence and scale-out (inside the trust boundary)

The memory layer needs **no** external ANN database (Milvus and friends): retrieval and dedup run
**within-state, in-process** in the deterministic Rust core, scoped to one session — there is no
cross-session similarity search in this role.

Where the session state *lives* is a separate decision. `SEMVEC_SESSION_BACKEND` selects among
three shipped backends, and an installed package can contribute a fourth through the
`semvec.backends` entry-point group (a pgvector adapter is the worked example in the docs):

| `SEMVEC_SESSION_BACKEND` | System of record | Sticky routing | Durability model |
|---|---|---|---|
| `memory` (default) | the worker process; **Postgres** when `SEMVEC_STATE_PERSIST=1` | **required** | write-behind: periodic flush + one on SIGTERM |
| `redis` | Redis — nothing behind it | **not needed** | write-through CAS, synchronous |
| `mongo` | MongoDB — nothing behind it | **not needed** | write-through CAS, synchronous |

**The sticky-routing constraint is a property of the default backend, not of Semvec.** With
`redis` or `mongo` the store is authoritative, pods are stateless, any replica serves any session,
and a plain round-robin load balancer is enough. Correctness is held by a version-validated
per-pod cache plus a write-through compare-and-set: a racing write is rejected loudly and retried,
never silently last-writer-wins. Rollback is one flag.

Choosing between them, for an on-prem deployment:

- **`memory` + Postgres** (`SEMVEC_STATE_PERSIST=1`, `DATABASE_URL`) keeps the hot path off the
  database: turns stay in memory, a periodic flush and a SIGTERM flush persist them, and state is
  reloaded lazily on first access after a restart. The trade-off is the recovery point — a
  *graceful* restart resumes bit-exact, but a hard crash can lose up to **one flush interval** of
  turns. It also requires **sticky routing by `session_id`**: there is no distributed lock, only
  an optimistic version that refuses a stale write. `SEMVEC_STATE_DB_SHARDS` spreads the
  state-blob table across several Postgres primaries (rendezvous-hashed) if one becomes the write
  ceiling. Not needed at this workload's scale.
- **`redis`** removes sticky routing but **must be planned as a database, not a cache**: there is
  no second copy in Postgres, so a key lost to eviction is a lost session. Semvec refuses to
  start against a Redis configured to evict or not to persist, naming the setting — run it with
  `appendonly yes`, `maxmemory-policy noeviction` and a backup policy. Rate limiting is then
  enforced fleet-wide rather than per pod.
- **`mongo`** is the same stateless model (`pip install "semvec[mongo]"`, `SEMVEC_MONGO_URI`, no
  default). MongoDB persists by default, so there is no startup durability gate. It additionally
  offers **server-side retrieval** via `SEMVEC_MONGO_ITEM_SYNC=1` with Atlas Vector Search — worth
  knowing that this is a *retrieval* decision, not just a latency one: it fuses semantic, lexical
  and a recency signal that the in-process pipeline has no equivalent for, so the same query can
  return a different set of memories. Every measurement in this repository uses in-process
  retrieval.

`DATABASE_URL` is read **only** by `memory` + `SEMVEC_STATE_PERSIST=1`; the `redis` and `mongo`
backends ignore it, and it has no default — asking for SQL persistence without setting it fails at
startup rather than quietly creating a file.

**State size** (measured, mpnet/768, `SemvecState.to_bytes(compress=True)` over the shipped
synthetic profile — reproduce with `tools/measure_state_size.py`): **132–278 KiB per session,
median ~174 KiB** for conversations of 8–34 turns, roughly 20–30 % below the uncompressed
encoding. The blob grows with conversation length, so size it against your own turn counts. At
this workload's scale: ~2,000 sessions × ~174 KiB ≈ **~0.34 GB** (×10 retention is still only a
few GB), and write load at peak 400 with write-behind stays well under 1 MB/s. A small Postgres,
Redis or MongoDB instance covers it either way.

Whichever backend you pick, on-prem/air-gapped holds: the store sits inside your trust boundary
and data never leaves it. Wire the orchestrator's readiness probe to `GET /v1/readyz`, which
reports whether the backend, the database and the embedder are actually serviceable, and keep
`/v1/health` on liveness.

**For a real deployment this runs on Kubernetes.** Semvec ships ready-to-adapt manifests
(`deploy/k8s/`: deployment, service, configmap, HPA, secret example) and a Helm chart
(`deploy/helm/semvec/`) that additionally provisions Redis and PostgreSQL. API pods and embedder
pods are separate Deployments with **separate HorizontalPodAutoscalers**, so each tier scales on
its own load. Combined with a `redis` or `mongo` session backend the API pods are stateless and
need no session affinity, which is what makes plain horizontal autoscaling work: any pod serves
any session, and capacity is a replica count.

**What the load demo measured.** Pillar 3 runs the default `memory` backend and pins each virtual
user to one worker — i.e. it deliberately measures the *sticky-routing* case, which is the
constrained one. A `redis`- or `mongo`-backed deployment removes that constraint.

Compliance features (append-only event store, deterministic replay, deletion certificates) are an
Enterprise-tier capability of Semvec and are written into whichever store the active backend
keeps. They are out of scope for the measurements here.

## When this does *not* help

The saving is entirely on the **input** side, so it disappears whenever input is not what
constrains your deployment. Pillar 1 prints the binding constraint for exactly this reason:

- **If your sizing is decode-bound, a conversation compressor will not save you GPUs.** Decode
  depends on output volume, which is untouched. Put a low decode throughput into a config and the
  model will correctly report a 0 % saving with `decode` as the binding constraint — the input
  side still falls by 75 %, and it changes nothing. Check which constraint binds before assuming
  a saving.
- The saving scales with conversation length: on short sessions there is little history to
  replace, so expect tens of percent rather than the −92 % measured on long dialogues.
- Contexts that already fit the model window comfortably gain little; on LongBench-v2's *short*
  split the naive path scores better. The advantage grows with context length.

## Scope and honesty

- **Not** a replacement for a RAG vector store, and **no** change to output token volume. The
  saving is entirely on the input side.
- **No** final hardware or price commitments — those need calibration (pillar 1) plus the open
  inputs in [docs/METHODOLOGY.md](docs/METHODOLOGY.md) §6.
- The real saving depends on the **multi-turn share** of your actual traffic. Measure it with
  pillar 2 on your own data before committing to numbers.
- **The load figures come from a single 8 GB laptop GPU.** They are internally consistent (0
  errors across every run), but a datacenter GPU on a quiet host would be the stronger evidence.
- **The LOCOMO judge is the reader model itself.** Word-F1 is unaffected; a separate judge model
  would be preferable for the judged figures.
- CapEx figures are illustrative: they come from the configurable `gpu_cost_eur` knob (default
  ~€30k/GPU list-price order of magnitude), not from any vendor quote.

## Documentation

| Document | Contents |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Technical results report — assumptions, method, every measurement |
| [docs/RESULTS-EXECUTIVE.md](docs/RESULTS-EXECUTIVE.md) | Short, business-focused summary |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | How the numbers are produced, fairness rules, open inputs |
| [docs/MEASUREMENT-SETUP.md](docs/MEASUREMENT-SETUP.md) | Exact environment, versions and hardware behind every figure |
| [docs/INFRA-SIZING.md](docs/INFRA-SIZING.md) | VMs/cores/RAM/VRAM for the memory layer itself |
| `configs/locomo_aligned.env` | The tuned retrieval preset used across all three pillars |

## References

- Semvec documentation: <https://semvec-docs.pages.dev/> — esp. `api-reference/token-reduction`
- PyPI: <https://pypi.org/project/semvec/>
- LOCOMO: <https://github.com/snap-research/locomo> · LongBench-v2: <https://github.com/THUDM/LongBench>

---

Maintained by [Versino PsiOmega GmbH](https://www.versino.de). Semvec is a Versino PsiOmega
GmbH product — <https://www.semvec.io>.
