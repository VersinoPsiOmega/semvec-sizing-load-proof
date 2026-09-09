# Contributing

Contributions are welcome, particularly measurements on hardware or workloads we do not have.

## Ground rules

1. **Evidence over assertion.** A changed number needs a committed artefact under `results/` and
   a note on the hardware, model and `semvec` version it came from. Results are never re-labelled
   for a dependency bump — record provenance instead (`docs/RESULTS.md` §10).
2. **Keep provenance labels honest.** `MEASURED`, `ASSUMPTION`, `REFERENCE` and `UNCALIBRATED`
   mean specific things. Do not promote an assumption to a measurement.
3. **Nothing identifying.** This repository is public: no customer or personal names, no internal
   hostnames, no absolute local paths, no keys. Run `tools/sanitize_check.sh` before opening a PR
   — CI runs it too. Note that k6 writes absolute paths into its JSON output.
4. **English only**, in code, comments, configs and docs.
5. **Tests first.** `pytest tests/ -q` must stay green; new logic needs a test.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-test.txt   # tests only: 4 small packages, no torch
.venv/bin/python -m pytest tests/ -q             #  tests, ~10 s
tools/sanitize_check.sh                          # no identifying data, no secrets
.venv/bin/python tools/check_claims.py           # docs still match the artefacts

# only if you need to actually run pillars 2/3 (pulls torch, ~6 GB):
.venv/bin/pip install -r requirements-dev.txt
```

The test suite is pure compute and aggregation logic: no GPU, no network, no LLM and no Semvec
licence key. Pillars 2 and 3 need the `semvec` package; pillar 3 additionally needs a licence key
and k6, so those runs cannot be reproduced in CI.

## Especially useful contributions

- **Calibration data** for other GPUs (`sizing/calibrate_vllm.py --write …`) — the sizing model
  is only as good as its throughput constants.
- **Load results** on other hardware, particularly datacenter GPUs and other embedders.
- **A second opinion on the fairness rules** in `docs/METHODOLOGY.md` §3. If a comparison here
  flatters the memory layer, we want to know.
