#!/usr/bin/env python3
"""Lightweight sidecar API worker for the load demo.

Drop-in ASGI worker that deliberately does NOT import torch /
sentence-transformers: in sidecar mode the embedder daemon owns the model, so
each worker stays RAM-cheap and many workers fit in memory. The heavy
``semvec serve`` worker would otherwise cap concurrency well below peak load.

History: earlier revisions of this file monkey-patched
``SidecarEmbedderClient._ensure_open`` to force a reconnect when the reader task
had died after an idle period (a worker sitting idle through a long client think
time). semvec adopted that fix upstream in **0.7.2**: ``_ensure_open`` now
verifies the reader task is alive and reconnects transparently, and an embed
whose connection dies mid-flight is retried once. This repo therefore targets
semvec >= 0.7.2 and carries no patch -- reaching into private client internals
would only risk breaking against a future release.

Env (same as the benchmark launcher):
  SEMVEC_BENCH_PORT     port to bind
  SEMVEC_EMBEDDER_URL   sidecar daemon URL (unix://... or tcp://...)
  SEMVEC_EMBEDDER_DIM   embedding dimension
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    sidecar_url = (os.environ.get("SEMVEC_EMBEDDER_URL") or "").strip()
    if not sidecar_url:
        sys.exit("_sidecar_worker.py requires SEMVEC_EMBEDDER_URL (sidecar mode)")

    from semvec.api import create_app

    app = create_app()  # lifespan injects the SidecarEmbedderClient from the env URL

    import uvicorn

    port = int((os.environ.get("SEMVEC_BENCH_PORT") or "18739").strip())
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
