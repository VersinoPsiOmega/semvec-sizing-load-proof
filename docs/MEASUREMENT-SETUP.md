# Measurement Setup

Exactly what produced the figures in this repository: the environment, the hardware, which
artefact backs which claim, and what is deliberately *not* claimed.

---

## 1. Environment

Every measurement in this repository was taken against **semvec 0.8.8**.

| | |
|---|---|
| semvec | **0.8.8** (from PyPI) |
| Python | 3.12 on Linux |
| ML stack | torch 2.14.0 · sentence-transformers 6.0.1 · transformers 5.16.1 · tiktoken 0.14.0 |
| Embedder | `paraphrase-multilingual-mpnet-base-v2`, dim 768 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Retrieval | dense + BM25/RRF + cross-encoder rerank, top_k 15, 10k-character budget |
| Reader LLM | **Qwen3.6-27B** on an H100-NVL-94GB via vLLM (OpenAI-compatible) |
| Load hardware | NVIDIA RTX 5060 Laptop, 8 GB · 24 cores · 16 GB RAM |
| Load driver | k6 v1.3.0 |

`requirements.txt` pins lower bounds (sentence-transformers ≥ 5.6.0, transformers ≥ 4.57.6) and
the measurements ran on 6.0.1 / 5.16.1 — major versions above those bounds, with the full test
suite green. The bounds are deliberately permissive.

## 2. Artefact index — which file backs which claim

Nothing in `results/` is unexplained.

| Artefact | Backs |
|---|---|
| `calibration_h100_qwen27b.json` | Per-GPU throughput constants: prefill 7,810 t/s, decode 1,970 t/s — the basis of the −74 % GPU saving |
| `harness_synthetic.json` | Air-gapped input reduction −34.0 % over 324 turns (median ~16-turn conversations), break-even at turn 1; needle retention 42/42 |
| `locomo_qa_tuned_full.json` | LOCOMO full suite, 1,984 QA: −92.3 % input, F1 0.527 → 0.482, judged 0.576 → 0.506 |
| `longbench_v2_qwen27b.json` | LongBench-v2, 60 items: 22 naively impossible, −97.4 % input, mean context 249,498 tokens |
| `load_smoke.json` | Harness sanity at low load: 62 turns/s, 0 / 3,044 errors |
| `load_realistic_60vu.json` | Latency at human pacing **without** rerank: 35.7 ms median |
| `load_realistic_rerank.json` | Latency at human pacing **with** rerank: 35.6 ms median — the reranker is free |
| `load_rerank_200vu.json` | Capacity of two daemons on one 8 GB GPU: **172 embeds/s**, GPU 92 %, 0 / 24,186 errors |
| `load_rerank_400vu.json` | Past the plateau: 147 embeds/s, latency rises instead of throughput |
| `load_throughput.json` | 1,000 VU with 24 workers and no rerank: 85 embeds/s at only 82 % GPU — a saturated host, not a saturated memory layer |

Third-party dataset text is not redistributed. The LOCOMO artefact carries metrics, token counts,
conversation id and category but no question or gold-answer text; the LongBench-v2 artefact
carries official item ids, the domain and length labels, the gold letter and the two predicted
letters. Every aggregate stays auditable without shipping the datasets.

## 3. What is not claimed

Stated plainly, because the gaps matter as much as the results:

- **Datacenter-GPU load figures.** Every load number comes from one 8 GB laptop GPU. The figures
  are internally consistent — zero failed requests across four load levels — but a quiet
  datacenter host is the measurement worth having.
- **A sized embedder tier.** All load runs use two embedder daemons on that one GPU. How far the
  tier scales with more daemons, more GPUs or a `tcp://` node of its own is a configuration
  decision, and it is not measured here.
- **The Kubernetes deployment.** Production runs on Kubernetes with separate autoscalers for the
  API and the embedder tiers; this repository *describes* that topology but measures a single
  box. No cluster-level scaling figures are claimed.
- **A judge model distinct from the reader.** Qwen3.6-27B served both roles, so the LLM-judged
  figures carry that caveat. Word-F1 is unaffected.
- **A second reader model.** All quality figures come from one reader, so this repository does not
  claim the effect is reader-independent — only that it holds for the reader measured.
- **The ONNX/GPU embedder path.** It needs a Rust binary and an ONNX export outside the PyPI-only
  baseline, so no ONNX throughput figure is given.
- **Decode-bound deployments.** No calibration of a decode-limited target is included; see the
  README's "When this does *not* help" for why the saving disappears in that case.

## 4. Keeping the documentation honest

`tools/check_claims.py` re-derives every headline figure from `results/` and fails if a document
disagrees. It runs in CI on every push, so an updated artefact cannot leave a stale number
behind in the prose.

It anchors each check on a **context pattern**, not a bare number — searching for "92" alone also
matches "GPU 92 %". Percentages get an *absolute* tolerance (0.1 points) rather than a relative
one, since 2 % of "92.0 %" is 1.8 points, wide enough to wave through a materially different
claim. The script is itself tested by deliberately falsifying figures and confirming it objects.

```bash
python tools/check_claims.py     # exit 0 = every documented headline matches its artefact
tools/sanitize_check.sh          # no identifying data, no secrets, valid JSON in results/
```

## 5. Reproduce it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-test.txt
.venv/bin/python -m pytest tests/ -q                    # 99 tests, ~10 s, no GPU
.venv/bin/python sizing/sizing_model.py --config sizing/configs/qwen3_27b_h100.yaml
tools/sanitize_check.sh
.venv/bin/python tools/check_claims.py

# pillars 2 and 3 need the full install
.venv/bin/pip install -r requirements.txt
.venv/bin/python harness/run_profile.py \
    --profile harness/profiles/reference_synthetic.yaml -o results/harness_synthetic.json
.venv/bin/python tools/measure_state_size.py

# with your own endpoint / licence key and k6 on PATH
.venv/bin/python sizing/calibrate_vllm.py --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" \
    --num-gpus 1 --concurrency 48
.venv/bin/python harness/run_locomo_qa.py --dataset /path/to/locomo10.json \
    --base-url "$OPENAI_BASE_URL" --model "$OPENAI_MODEL" --conv-limit 10 --limit-qa 0 --judge
.venv/bin/python load/run_load.py --rerank --workers 12 --daemons 2 --batch-max 64 \
    --batch-wait-ms 50 --peak-vus 200 --avg-vus 100 --think-ms 0 -o results/load_rerank_200vu.json
```

Write `.env` with Unix line endings (`sed -i 's/\r$//' .env`); a trailing carriage return inside a
URL fails with an opaque parse error against a perfectly reachable endpoint.
