# Notice — licensing and third-party components

## This repository

Copyright 2026 Versino PsiOmega GmbH.

The contents of this repository — the sizing model, the measurement harnesses, the load driver,
the configurations, the tests and the committed results — are licensed under the
**Apache License, Version 2.0**. See [LICENSE](LICENSE).

## Semvec is not covered by that licence

**Semvec is commercial, proprietary software and is not part of this repository.** It is
distributed separately (<https://pypi.org/project/semvec/>) under its own proprietary licence
terms, and the Apache-2.0 grant above does not extend to it.

What that means in practice:

| Pillar | Needs Semvec? | Needs a licence key? |
|---|---|---|
| 1 — sizing model (`sizing/`) | No | No |
| 2 — measurement harness (`harness/`) | Yes (the `semvec` package) | No |
| 3 — load demonstration (`load/`) | Yes (the `semvec` package) | **Yes** — `SEMVEC_LICENSE_KEY` |

Pillar 1 is self-contained: it needs only PyYAML and no Semvec at all, so the sizing model is
usable independently of any Semvec entitlement.

Semvec is a Versino PsiOmega GmbH product — <https://www.semvec.io>, <https://www.versino.de>.

## Third-party datasets

Neither dataset is bundled with this repository. Supply your own copy via `--dataset`.

- **LOCOMO** — <https://github.com/snap-research/locomo>. Per-item question and gold-answer text
  is **not** redistributed in `results/`: those fields were removed and only metrics, token
  counts, conversation id and category remain, so aggregates stay auditable without
  redistributing the dataset. Consult the upstream repository for its licence terms before
  redistributing any part of it yourself.
- **LongBench-v2** — <https://github.com/THUDM/LongBench>. `results/longbench_v2_*.json` carries
  only official item ids, the domain and length labels, the gold letter and the two predicted
  letters.

## Third-party models

The measurements use publicly available models, each under its own licence:

- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (embedder, dim 768)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker)

Reader models (Qwen3.6-27B, DeepSeek-V4-Flash) were accessed over an OpenAI-compatible endpoint
and are not distributed here.

## Load driver

Pillar 3 requires **k6** (<https://k6.io>), which is licensed under AGPL-3.0 and is not
distributed with this repository — install it separately as a static binary.
