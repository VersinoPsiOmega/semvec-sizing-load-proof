# Repository guide

Measurement repository. Three independent pillars quantify what a constant-cost memory layer in
front of an LLM saves in hardware, plus the harnesses to measure it on another workload.

## Layout

| Path | Purpose |
|---|---|
| `sizing/` | Pillar 1 — token→hardware sizing model + vLLM calibration. Pure computation: no GPU, no network, no `semvec`. |
| `harness/` | Pillar 2 — input-tokens/turn and answer-quality harnesses (synthetic profile, LOCOMO, LongBench-v2). |
| `load/` | Pillar 3 — k6-driven load demo against the Semvec REST API, with GPU/CPU/RAM sampling. |
| `results/` | Committed measurement artefacts. Treat as evidence, not as scratch space. |
| `docs/` | Results report, executive summary, methodology, measurement setup, infrastructure sizing. |
| `tests/` | 99 unit tests over the compute and aggregation logic. No GPU/network/LLM/licence, and no torch — see `requirements-test.txt`. |

## Commands

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-test.txt   # tests + pillar 1 (light, no torch)
.venv/bin/pip install -r requirements-dev.txt    # only to run pillars 2/3 (pulls torch)

.venv/bin/python -m pytest tests/ -q                                    # 99 tests
.venv/bin/python sizing/sizing_model.py --config sizing/configs/reference_2b.yaml
.venv/bin/python harness/run_profile.py \
    --profile harness/profiles/reference_synthetic.yaml -o results/harness_synthetic.json
tools/sanitize_check.sh                                                 # publication gate
.venv/bin/python tools/check_claims.py                                  # docs vs. artefacts
```

## Conventions that matter here

- **Every number carries a provenance label.** `MEASURED` / `ASSUMPTION` / `REFERENCE` in
  configs and docs, and `UNCALIBRATED` in the sizing output until real throughput constants are
  fed in. Never present an assumption as a measurement, and never quietly upgrade a label.
- **Never invent a measurement.** If a figure is not backed by an artefact in `results/`, say so.
  When a dependency version changes, do not re-label existing results — record what they were
  produced with (see `docs/RESULTS.md` §10).
- **No identifying data.** This is a public repository. No customer names, no personal names, no
  internal hostnames, no absolute local paths (k6 writes them into its output — check),
  no licence keys. `tools/sanitize_check.sh` enforces a deny-list; run it before publishing.
- **English only** in all code, comments, configs and docs.
- **Domain-neutral vocabulary.** The synthetic word pools in `harness/run_profile.py`,
  `load/k6_chat_scenario.js` and `sizing/calibrate_vllm.py` are deliberately generic. Keep them
  that way, and keep token lengths comparable if you change them — the committed load artefacts
  depend on that length characteristic.
- **Tests before code.** Path constants and CLI flag names are asserted in `tests/`; change the
  test first, watch it fail, then change the code.
- **Keep the test suite dependency-light.** It must pass with only PyYAML and pytest installed.
  Optional imports (`tiktoken`, `psutil`) belong behind a try/except with an equivalent fallback,
  not as hard requirements of a pure-compute function — otherwise CI needs the whole ML stack.

### Measurement hygiene

Each of these is enforced because violating it silently corrupts a published number.

- **Never give a child process `stderr=subprocess.PIPE` unless something drains it.** The pipe
  buffer is 64 KiB; when it fills, the child blocks in `write()` and its in-flight requests hang
  until the client timeout while the GPU sits idle. Children log to files (`_child_log`), and a
  test asserts `stderr=subprocess.PIPE` is absent.
- **Never write an absolute path into a report.** k6 embeds whatever path it is handed;
  `_report_config` keeps every config field free of absolute paths. The publication gate fails
  otherwise.
- **Strip environment values.** A `.env` with CRLF endings puts a trailing `\r` in every value and
  a URL then fails with an opaque parse error against a reachable endpoint. Use `_env()`.
- **Never let entropy for structure share an rng with entropy for content.** The synthetic
  profile's filler uses its own rng derived from `(seed, cid, turn)`; sharing one would let the
  tokeniser or the word pool change turns-per-conversation, and `seed:` would stop meaning
  "reproducible".
- **Don't trust a log's last line as the current position.** Python buffers `print()` into a file,
  and `| tail -N` in a shell pipeline emits nothing until EOF. A run that looks stuck at an early
  line is often working fine; check CPU and GPU activity before concluding it hung.
- **Report the plateau, not the largest load the harness accepts.** Throughput peaks at 200 VU
  and halves by 1,000 VU. Quoting the highest concurrency measures thrashing, not capacity.
- **Re-run the claim checker after touching a figure.** `tools/check_claims.py` enforces
  docs-vs-artefacts in CI. If it fails, the documentation is stale, not the script.
- **A check that never fails is not a check.** Falsify a figure deliberately and confirm the
  tooling objects.
- **Keep `semvec` internals at arm's length.** Do not monkey-patch its private members; pin a
  version requirement instead.

## Fairness rules for the quality comparisons

If you touch the QA harnesses, preserve these — they are what makes the results defensible:

- Identical token encoder on both paths (tiktoken, `cl100k_base`).
- Identical reader and identical questions for naive and Semvec.
- Reasoning/"thinking" disabled for both paths, or enabled for both.
- When the naive context exceeds the model window, the item is **infeasible**, not wrong, and is
  reported separately rather than averaged in.
