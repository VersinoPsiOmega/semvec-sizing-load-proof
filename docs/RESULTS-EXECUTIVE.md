# A Chatbot at a Fraction of the Hardware Cost — Executive Summary

## The situation

A large-enterprise chatbot rollout for ~2,000 staff, **strictly on-prem, no data to third
parties**. Conventional hardware sizing for that requirement is **expensive**: the demand for
fast responses at high concurrency is what drives up the number of costly GPUs — not the
number of users, and not the volume of answers.

## The solution in one sentence

**Semvec** is a lean "memory layer" that continuously **compresses** the conversation before
it reaches the language model. The model receives a **small, constant** amount of text per
question instead of the complete and ever-growing history — and that history is the single
most expensive item in the bill.

## The result (measured, not estimated)

- **74 % fewer LLM GPUs** — from 31 down to **8** (NVIDIA H100 NVL, 94 GiB).
- **Same answer quality.** On a recognised conversational benchmark (LOCOMO, ~2,000 questions,
  checked with a real language model) Semvec delivers **near-parity** — with **92 % less**
  input text. On adversarial questions it is *better* than full context — it knows when not to
  answer.
- **Runs on small, standard hardware.** The memory layer itself carries the **400 concurrent
  peak users** on **a single small GPU** at roughly 23 % utilisation — a 4.3× reserve, with zero
  errors across four measured load levels on an 8 GB laptop GPU.
- **Achieves what the naive approach cannot.** In a second, harder test (LongBench-v2, very
  long documents) the naive approach exceeds the language model's limit in **more than a third**
  of cases and cannot answer at all — Semvec can, with **97 % less** input text. Two
  independent benchmarks show the same picture.
- **100 % on-prem / air-gapped** — data never leaves the data centre (satisfying GDPR and
  regulated-industry constraints), and every number here can be **re-verified independently**.

## Why this is convincing

Equal or better quality, **substantially cheaper**, entirely inside your own data centre — and
backed by reproducible evidence rather than promises. Every figure in this repository can be
regenerated with one command per pillar.

## What a binding commitment still needs

A handful of workload facts from the target deployment: typical **conversation length / tokens
per request**, **turns per session**, the **final language model**, and the **actually priced
GPU**. Those turn a demonstrated order of magnitude into committed quantities and prices.

> **To be straight about it:** the −74 % is the defensible order of magnitude on documented
> assumptions. The **relative** advantage (substantially less hardware at equal quality) is
> robust; the exact GPU count and price will shift with real workload data. What is *not* proven
> here: throughput on datacenter hardware — every load figure comes from one 8 GB laptop GPU.
> Cheap to close on real target hardware.

*Technical detail, method and all measurements: [RESULTS.md](RESULTS.md) · Semvec-layer
infrastructure: [INFRA-SIZING.md](INFRA-SIZING.md)*
