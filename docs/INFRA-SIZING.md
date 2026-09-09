# Infrastructure Sizing of the Semvec Layer

**Headline:** the Semvec memory layer for the reference workload (400 peak users) is **small**
and runs on **one small GPU, one app VM and one small state store** (Postgres, Redis or
MongoDB — see the backend choice below). It is **strictly separate**
from the separately sized LLM GPUs (pillar 1: 8 H100-NVL GPUs) and must not be taken out of
that pool. All figures are measured on the **tuned, deployed stack** (mpnet/768 + hybrid BM25 +
cross-encoder rerank).

> This sizing covers **Semvec only** — the conversation compressor in front of the LLM, not the
> LLM serving itself.

---

## 1. Data basis (measured, tuned stack)

Measured on an RTX 5060 8 GB / 24 cores / 16 GB RAM, mpnet/768, **with reranking enabled**
(hybrid BM25 + cross-encoder in the `/v1/run` path):

| Quantity | Value (rerank on, measured live) |
|---|---|
| Throughput capacity | **172 embeds/s** at 200 concurrent sessions, 0 errors, GPU-bound (92 %) |
| Behaviour past the plateau | 400 VU → 147/s. At 1,000 VU (24 workers, no rerank) → 85/s with the GPU at only 82 % — the host saturates, not the memory layer. |
| VRAM under load | **4.5 GiB** peak (2 embedder daemons + cross-encoder) |
| Stack RAM (RSS) at full load | **~5.0 GB** (12 workers + 2 daemons) |
| `/v1/run` latency, human-paced (10 s think) | **35.6 ms** median, p90 62 ms, p99 92 ms, 0 errors — **with** rerank |
| Persistence | **132–278 KiB/state** (median ~174 KiB, 8–34 turns; `tools/measure_state_size.py`) → ~0.34 GB at 2,000 sessions; write load < 1 MB/s |

> **The reranker is free on both axes.** 35.6 ms with it vs. 35.7 ms without at realistic pacing,
> and absorbed on the throughput side. The embedder remains the GPU bottleneck.
>
> Sizing below uses the measured 172 embeds/s plateau — what **two embedder daemons on one 8 GB
> laptop GPU** delivered. That tier is sized independently: `--daemons N` here, and in a
> deployment as many standalone `python -m semvec.embedder` processes as you need, reachable over
> `tcp://` from a separate GPU node. More embedder capacity is a configuration decision, not a
> rebuild. Environment and hardware: [MEASUREMENT-SETUP.md](MEASUREMENT-SETUP.md).

**Extrapolated** (kept clearly separate): the real load rate at human pacing, and the retention
projection. Both carry enough headroom that the topology stays valid either way.

## 2. Real load rate on the Semvec layer

One chat turn causes **one** `/v1/run` operation. At realistic pacing a turn takes ~10 s
(LLM generation plus reading and typing):

```
400 peak users ÷ 10 s/turn ≈ 40 embeds/s
```

Against the measured capacity of **one** small GPU (tuned stack, 172 embeds/s plateau):
**40/s out of 172/s ≈ 23 % utilisation → ~4.3× headroom**. → **A single small GPU suffices** for
the full peak load; a second one is optional, for high availability rather than capacity.

## 3. Component sizing

- **GPU/VRAM:** embedder daemons plus cross-encoder rerank occupy **~4.5 GiB** at full load → an
  **8 GB card or MIG slice** is enough, with headroom.
- **API workers / CPU:** the compute load is tiny — 40 req/s × ~36 ms ≈ **~1.4 cores net**. The
  **16–24 workers** exist for concurrency, and — on the default `memory` backend — for sticky
  routing (session → worker), not for compute. On a `redis`/`mongo` backend the workers are fully
  stateless and the sticky-routing requirement disappears. Each worker loads a small cross-encoder (ms-marco-MiniLM-L-6-v2, ~90 MB), a
  minor RAM addition. **8 vCPU** leaves room for workers, reranker, daemon batchers and the OS.
- **RAM:** the stack uses ~5–6 GB at full load → **16 GB** is comfortable for the app VM.
- **State store:** ~0.34 GB of data (×10 retention → a few GB) and < 1 MB/s of writes →
  **2 vCPU / 4–8 GB / ~50 GB SSD**, whether that is Postgres (`memory` backend with
  `SEMVEC_STATE_PERSIST=1`), Redis or MongoDB. No sharding required — `SEMVEC_STATE_DB_SHARDS`
  is an opt-in write-throughput lever for far larger deployments.
  Note for Redis: it is the system of record with nothing behind it, so it must be run as a
  database (`appendonly yes`, `maxmemory-policy noeviction`, backups) — Semvec refuses to start
  otherwise.

## 4. VM topology

### Variant A — minimal

| VM | Role | vCPU | RAM | GPU/VRAM |
|---|---|---|---|---|
| semvec-app-1 | 16–24 API workers (stateless, incl. cross-encoder rerank) + embedder daemon(s) | 8 | 16 GB | 1 small GPU / MIG slice ~8 GB (4.3 GB used) |
| semvec-db-1 | State store: Postgres (state BYTEA), or Redis/MongoDB | 2 | 8 GB | — (50 GB SSD) |

> Collapsible onto **one** VM (~10 vCPU / 24 GB / 1 GPU slice) if a co-located database is
> acceptable.

### Variant B — with high availability

| VM | Role | vCPU | RAM | GPU/VRAM |
|---|---|---|---|---|
| semvec-app-1 | API workers + rerank + embedder daemon (active) | 8 | 16 GB | 1 GPU slice ~8 GB |
| semvec-app-2 | API workers + rerank + embedder daemon (active, behind LB) | 8 | 16 GB | 1 GPU slice ~8 GB |
| semvec-lb-1 | Load balancer with session stickiness (`session_id` hash) | 2 | 4 GB | — |
| semvec-db-1 | Postgres primary | 4 | 8 GB | — (50–100 GB SSD) |
| semvec-db-2 | Postgres standby (streaming replication, failover) | 4 | 8 GB | — (50–100 GB SSD) |

In variant B **each** app VM carries the peak load on its own (40/s vs. 172/s measured), so the
SLA survives losing one.

**Whether the load balancer needs sticky routing depends on the backend.** On the default
`memory` backend it must route by `session_id` (a version CAS additionally guards against
clobbering) — that is the topology above, and the one pillar 3 measured. With
`SEMVEC_SESSION_BACKEND=redis` or `=mongo` the store is authoritative, both app VMs are stateless,
and a plain round-robin balancer suffices; the `semvec-lb-1` row then needs no session affinity.
For a two-node HA deployment that is usually the simpler choice.

### Variant C — Kubernetes (how a real deployment runs)

The VM topologies above describe the shape; in production this is deployed on Kubernetes, and
scaling is the cluster's job from there. Semvec ships the manifests (`deploy/k8s/`) and a Helm
chart (`deploy/helm/semvec/`).

| Workload | Kubernetes object | Scales on |
|---|---|---|
| API | Deployment + Service + HPA | its own CPU/latency target |
| Embedder | **separate** Deployment + Service + **its own HPA** | embedder load, independently of the API |
| Session store | Redis or MongoDB (chart provisions Redis/PostgreSQL) | operated as a database, not a cache |

The API reaches the embedder over the cluster service
(`SEMVEC_EMBEDDER_URL=tcp://<release>-embedder:7071`), readiness hangs on `/v1/readyz` and
liveness on `/v1/health`. With a `redis` or `mongo` backend the API pods are stateless and need no
session affinity — which is precisely what lets a HorizontalPodAutoscaler do its job: any pod
serves any session, so capacity is a replica count rather than a re-architecture. The single-GPU
figures measured in this repository size *one* embedder pod; the cluster decides how many of them
run.

## 5. Scope

This is **only** the Semvec layer. LLM serving is sized separately (see [RESULTS.md](RESULTS.md)):
**naive 31 → with Semvec 8 GPUs (−74 %)** on NVIDIA H100 NVL (94 GiB). The Semvec
infrastructure described here is negligibly small by comparison.

**Why the layer stays small.** Semvec keeps the **input per turn constant** (~1.5–1.9k tokens)
regardless of conversation or document length. Pillar 2 establishes that across two benchmarks
— LOCOMO (dialogues up to 689 turns) and LongBench-v2 (mean 249k-token documents, 22 of 60
items beyond the model window). Because the input processed per turn does not grow with the
history, neither LLM prefill nor embedder load scales with conversation length, and the
topology above stays valid.
