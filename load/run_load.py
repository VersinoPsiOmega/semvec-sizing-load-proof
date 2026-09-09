#!/usr/bin/env python3
"""Load demo (pillar 3) — Semvec REST under request concurrency, sidecar + batches.

Ramps the Semvec REST API up under load in the order of magnitude of the
reference workload (default peak 400 parallel chat clients) and delivers a
HARDWARE-ANCHORED statement: "on <GPU> / <CPU> / <RAM>, Semvec sustains
<concurrency> sessions at /v1/run p90 = <X> ms and <Y> turns/s; VRAM peak <Z> MiB."

Deployment form (all documented entry points):
  * N embedder daemons  — `python -m semvec.embedder --listen unix://… --model mpnet
                          --dimension 768 --batch-max … --batch-wait-ms …`
                          (mpnet/768, batched parallel encodes on the GPU)
  * M API workers       — lightweight ASGI launcher per port, each routed to a
                          daemon via SEMVEC_EMBEDDER_URL (stateless, no model
                          loaded in-worker → RAM-cheap, so many workers fit)
  * k6                  — drives up to peak VUs; each VU one persistent session,
                          pinned to a worker port via __VU (session affinity)

Measures GPU (nvidia-smi) + process CPU/RSS during the run and combines
everything with the k6 summary into one JSON report.

Prerequisite: SEMVEC_LICENSE_KEY in the environment (REST auth). k6 on PATH.

Example (smoke):
    python load/run_load.py --workers 2 --daemons 1 --peak-vus 40 --avg-vus 20 \
        --peak-hold 20s --plateau 20s --ramp 10s -o results/load_smoke.json

Example (request scale):
    python load/run_load.py --workers 6 --daemons 2 --peak-vus 400 --avg-vus 200 \
        -o results/load_peak400.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from _gpu_sampler import GpuSampler, gpu_info  # noqa: E402

MPNET_MODEL = "paraphrase-multilingual-mpnet-base-v2"
MPNET_DIM = 768

# Lightweight sidecar ASGI worker. Does NOT import torch / sentence-transformers
# (the daemon owns the model) → each worker is RAM-cheap, so many workers fit in
# memory (the heavy `semvec serve` worker would otherwise cap concurrency well
# below peak load). Human-paced traffic with long think times needs the
# reader-aware sidecar reconnect that semvec ships since 0.7.2.
WORKER_LAUNCHER = HERE / "_sidecar_worker.py"


def _report_scenario_path(scenario: str | Path) -> str:
    """Repo-relative scenario path for the report.

    k6's own summary carries whatever path it was given, and this repository is
    public: an absolute path leaks the author's directory layout. Keep it
    relative to the repository root so a committed artefact is portable and the
    publication gate stays green.
    """
    path = Path(scenario)
    root = Path(__file__).resolve().parent.parent
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return path.name if path.is_absolute() else str(path)


def _report_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Strip absolute paths out of the config block of a report.

    This repository is public and its artefacts are committed, so no field may
    carry the author's directory layout. `scenario` gets a repo-relative path;
    any other absolute-looking string keeps only its basename. Non-path values
    pass through untouched.
    """
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if key == "scenario":
            out[key] = _report_scenario_path(value)
        elif isinstance(value, (str, Path)) and str(value).startswith("/"):
            out[key] = Path(str(value)).name
        else:
            out[key] = value
    return out


def _child_log(name: str, directory: Path | None = None) -> tuple[Any, Path]:
    """Open a log file for a child process's stderr.

    Children must NOT write into an unread `subprocess.PIPE`: the pipe buffer
    (64 KiB on Linux) fills, the child blocks in write(), and its in-flight
    /v1/run requests hang until the client timeout — a p95 pinned at 60 s while
    the GPU sits idle. Nothing drains those pipes during a run, so stderr goes
    to a file, which also means the log survives for post-mortem.
    """
    directory = directory or Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"semvec-load-{name}-{uuid.uuid4().hex[:8]}.log"
    return open(path, "wb"), path


def _require_license() -> str:
    key = os.environ.get("SEMVEC_LICENSE_KEY", "").strip()
    if not key:
        sys.exit(
            "SEMVEC_LICENSE_KEY missing. Load a source, e.g.:\n"
            "  set -a; . /path/to/.env; set +a\n"
            "or export your own license key."
        )
    return key


def _check_launcher() -> None:
    if not WORKER_LAUNCHER.exists():
        sys.exit(f"worker launcher not found: {WORKER_LAUNCHER}")


def _wait_health(port: int, timeout: float) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
            pass
        time.sleep(0.3)
    return False


def _sample_proc_resources(pids: list[int], interval: float, stop: threading.Event,
                           samples: list[dict], t0: float) -> None:
    import psutil

    procs: dict[int, psutil.Process] = {}
    for pid in pids:
        try:
            procs[pid] = psutil.Process(pid)
        except psutil.NoSuchProcess:
            pass
    for p in procs.values():
        try:
            p.cpu_percent(None)
        except psutil.NoSuchProcess:
            pass
    while not stop.is_set():
        stop.wait(interval)
        rss = 0.0
        cpu = 0.0
        for pid, p in list(procs.items()):
            try:
                for proc in [p, *p.children(recursive=True)]:
                    with proc.oneshot():
                        rss += proc.memory_info().rss / (1024 * 1024)
                        cpu += proc.cpu_percent(None)
            except psutil.NoSuchProcess:
                procs.pop(pid, None)
        samples.append({"t": round(time.monotonic() - t0, 2),
                        "rss_mb": round(rss, 1), "cpu_pct": round(cpu, 1)})


def _summarise_proc(samples: list[dict]) -> dict[str, Any]:
    if not samples:
        return {}
    rss = [s["rss_mb"] for s in samples]
    cpu = [s["cpu_pct"] for s in samples[1:]] or [0.0]
    try:  # psutil is only needed for the core count; keep this pure-compute
        import psutil

        cores = psutil.cpu_count(logical=True) or 1
    except ImportError:
        cores = os.cpu_count() or 1
    return {
        "rss_mb": {"peak": max(rss), "mean": round(sum(rss) / len(rss), 1)},
        "cpu_pct": {"peak": max(cpu), "mean": round(sum(cpu) / len(cpu), 1), "cores": cores},
    }


def daemon_spawn_kwargs(*, model: str, dimension: int, batch_max: int,
                        batch_wait_ms: float, onnx: bool,
                        rust_binary: Path | None = None,
                        onnx_paths: tuple[Path, Path] | None = None) -> dict[str, Any]:
    """Build the kwargs handed to ``build_embedder_argv`` for one daemon.

    Default (onnx=False): the PyTorch Python daemon (``semvec.embedder``) — no
    extra files. With onnx=True: the Rust ``semvec-embedder`` daemon, which is
    the only ONNX route in semvec and requires the compiled binary plus an
    exported ONNX model (tokenizer.json + onnx/model.onnx). Missing prerequisites
    fail fast with an actionable message instead of a cryptic spawn error."""
    kw: dict[str, Any] = dict(model=model, dimension=dimension,
                              batch_max=batch_max, batch_wait_ms=batch_wait_ms)
    if not onnx:
        return kw
    if rust_binary is None:
        sys.exit(
            "--onnx needs the Rust semvec-embedder binary (not found). The PyTorch\n"
            "Python daemon has no ONNX backend; ONNX is the Rust route. Build it:\n"
            "  cargo build --release --features embedder-daemon --bin semvec-embedder"
        )
    if onnx_paths is None:
        sys.exit(
            "--onnx needs an exported ONNX model (onnx/model.onnx not found in the\n"
            "HF cache). Populate it once via sentence-transformers' onnx backend:\n"
            "  SentenceTransformer(model, backend='onnx')   # requires optimum[onnxruntime-gpu]"
        )
    tokenizer_path, model_path = onnx_paths
    kw.update(executable=rust_binary, tokenizer_path=tokenizer_path, model_path=model_path)
    return kw


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=4, help="number of API workers (own port per worker)")
    p.add_argument("--daemons", type=int, default=1, help="number of embedder daemons (parallel batch streams)")
    p.add_argument("--model", default=MPNET_MODEL, help=f"embedder model (default {MPNET_MODEL})")
    p.add_argument("--dim", type=int, default=MPNET_DIM)
    p.add_argument("--batch-max", type=int, default=32, help="max batch size per encode")
    p.add_argument("--batch-wait-ms", type=float, default=5.0, help="batch collection window")
    p.add_argument("--rerank", action="store_true",
                   help="enable server-side tuned retrieval in /v1/run (hybrid BM25 + cross-encoder "
                        "rerank, configs/locomo_aligned.env settings) — measures the tuned deployment")
    p.add_argument("--onnx", action="store_true",
                   help="run the embedder on the Rust ONNX daemon (higher throughput; "
                        "needs the semvec-embedder binary + an exported ONNX model)")
    p.add_argument("--peak-vus", type=int, default=400)
    p.add_argument("--avg-vus", type=int, default=200)
    p.add_argument("--turns-min", type=int, default=30)
    p.add_argument("--turns-max", type=int, default=120)
    p.add_argument("--think-ms", type=int, default=250)
    p.add_argument("--msg-tokens", type=int, default=200)
    p.add_argument("--ramp", default="30s")
    p.add_argument("--plateau", default="60s")
    p.add_argument("--peak-hold", default="60s")
    p.add_argument("--base-port", type=int, default=18739)
    p.add_argument("--sample-every", type=float, default=1.0)
    p.add_argument("--scenario", default=str(HERE / "k6_chat_scenario.js"))
    p.add_argument("-o", "--output", type=Path)
    args = p.parse_args()

    if shutil.which("k6") is None:
        sys.exit("k6 not on PATH — install/place the binary.")
    token = _require_license()
    _check_launcher()

    from semvec.api.supervisor import build_embedder_argv, wait_for_ready

    # Decide PyTorch (default) vs Rust ONNX daemon once, up front, so a missing
    # prerequisite fails before we spawn anything.
    if args.onnx:
        from semvec.api.supervisor import find_embedder_binary, resolve_model_paths
        rust_binary = find_embedder_binary()
        try:
            onnx_paths = resolve_model_paths(args.model)
        except FileNotFoundError:
            onnx_paths = None
        daemon_kw = daemon_spawn_kwargs(
            model=args.model, dimension=args.dim, batch_max=args.batch_max,
            batch_wait_ms=args.batch_wait_ms, onnx=True,
            rust_binary=rust_binary, onnx_paths=onnx_paths)
    else:
        daemon_kw = daemon_spawn_kwargs(
            model=args.model, dimension=args.dim, batch_max=args.batch_max,
            batch_wait_ms=args.batch_wait_ms, onnx=False)

    base_env = {**os.environ, "SEMVEC_LICENSE_KEY": token}
    ports = [args.base_port + i for i in range(args.workers)]
    daemon_procs: list[subprocess.Popen] = []
    worker_procs: list[subprocess.Popen] = []
    child_logs: list[tuple[Any, Path]] = []
    socks: list[str] = []
    cleanup: list[str] = []

    print(f"=== Load demo — {args.workers} workers / {args.daemons} daemon(s) / "
          f"peak {args.peak_vus} VUs / embedder {args.model} dim{args.dim} ===")
    ginfo = gpu_info()
    if ginfo.get("gpus"):
        g = ginfo["gpus"][0]
        print(f"GPU: {g['name']}  {g['mem_total_mib']:.0f} MiB  driver {g['driver']}")

    try:
        # 1) start embedder daemons (mpnet/768, batched)
        for d in range(args.daemons):
            sock = f"/tmp/semvec-load-{uuid.uuid4().hex[:10]}.sock"
            url = f"unix://{sock}"
            socks.append(url)
            cleanup.append(sock)
            rfd, wfd = os.pipe()
            os.set_inheritable(wfd, True)
            argv = build_embedder_argv(listen_url=url, ready_fd=wfd, **daemon_kw)
            print(f"  Daemon {d} → {url} ({'ONNX/Rust' if args.onnx else 'PyTorch'}, model loading …)")
            dlog, dlog_path = _child_log(f"daemon{d}")
            child_logs.append((dlog, dlog_path))
            cleanup.append(str(dlog_path))
            proc = subprocess.Popen(argv, env=base_env, stdout=subprocess.DEVNULL,
                                    stderr=dlog, pass_fds=(wfd,))
            os.close(wfd)
            try:
                wait_for_ready(rfd, timeout=300.0)
            finally:
                os.close(rfd)
            print(f"  Daemon {d} READY (pid {proc.pid})")
            daemon_procs.append(proc)

        # 2) start API workers (stateless, each its own port, sidecar routing)
        # Each worker gets its OWN state DB (DATABASE_URL), otherwise multiple
        # workers collide when creating the schema on the shared default
        # `semvec.db` (table already exists). Session state is in-memory anyway
        # (SEMVEC_STATE_PERSIST off) and pinned to one worker via __VU.
        for i, port in enumerate(ports):
            url = socks[i % len(socks)]
            db_path = f"/tmp/semvec-load-w{port}-{uuid.uuid4().hex[:8]}.db"
            cleanup.append(db_path)
            wenv = {**base_env,
                    "SEMVEC_BENCH_PORT": str(port),
                    "SEMVEC_EMBEDDER_URL": url,          # sidecar: lifespan injects client
                    "SEMVEC_EMBEDDER_DIM": str(args.dim),
                    "DATABASE_URL": f"sqlite:///{db_path}"}
            if args.rerank:
                # server-side tuned retrieval (configs/locomo_aligned.env values)
                wenv.update({"SEMVEC_RUN_TOP_K": "15", "SEMVEC_CONTEXT_BUDGET_CHARS": "10000",
                             "SEMVEC_RERANK_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                             "SEMVEC_RERANK_FETCH_K": "50", "SEMVEC_RERANK_BATCH": "64",
                             "SEMVEC_HYBRID_BM25": "1", "SEMVEC_BM25_FETCH_K": "50"})
            argv = [sys.executable, str(WORKER_LAUNCHER)]
            wlog, wlog_path = _child_log(f"worker{port}")
            child_logs.append((wlog, wlog_path))
            cleanup.append(str(wlog_path))
            proc = subprocess.Popen(argv, env=wenv, stdout=subprocess.DEVNULL, stderr=wlog)
            worker_procs.append(proc)
        for port, proc in zip(ports, worker_procs):
            if not _wait_health(port, timeout=120.0):
                log_path = next((lp for lh, lp in child_logs if f"worker{port}-" in lp.name), None)
                err = log_path.read_text("utf-8", "replace")[-1500:] if log_path else ""
                raise SystemExit(f"Worker on port {port} not ready.\n{err}")
        print(f"  {args.workers} workers ready on ports {ports[0]}..{ports[-1]}")

        # 3) start sampler
        all_pids = [pp.pid for pp in (*daemon_procs, *worker_procs)]
        proc_samples: list[dict] = []
        stop = threading.Event()
        t0 = time.monotonic()
        sampler = threading.Thread(target=_sample_proc_resources,
                                   args=(all_pids, args.sample_every, stop, proc_samples, t0), daemon=True)
        sampler.start()
        gpu = GpuSampler(interval=args.sample_every)
        gpu.start()

        # 4) k6
        k6_summary = HERE.parent / "results" / ".k6_load.tmp.json"
        k6_summary.parent.mkdir(parents=True, exist_ok=True)
        base_urls = ",".join(f"http://127.0.0.1:{p}" for p in ports)
        k6_cmd = [
            "k6", "run", "--summary-export", str(k6_summary),
            "-e", f"BASE_URLS={base_urls}", "-e", f"LICENSE_KEY={token}",
            "-e", f"PEAK_VUS={args.peak_vus}", "-e", f"AVG_VUS={args.avg_vus}",
            "-e", f"TURNS_MIN={args.turns_min}", "-e", f"TURNS_MAX={args.turns_max}",
            "-e", f"THINK_MS={args.think_ms}", "-e", f"MSG_TOKENS={args.msg_tokens}",
            "-e", f"DIM={args.dim}", "-e", f"RAMP={args.ramp}",
            "-e", f"PLATEAU={args.plateau}", "-e", f"PEAK_HOLD={args.peak_hold}",
            args.scenario,
        ]
        k6_rc = subprocess.run(k6_cmd, check=False).returncode

        stop.set()
        sampler.join(timeout=5)
        gpu_report = gpu.stop()
        proc_report = _summarise_proc(proc_samples)

    finally:
        for pr in worker_procs:
            pr.send_signal(signal.SIGTERM)
        for pr in worker_procs:
            try:
                pr.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pr.kill()
        for pr in daemon_procs:
            pr.send_signal(signal.SIGTERM)
        for pr in daemon_procs:
            try:
                pr.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pr.kill()
        # Close the child stderr log handles only after every child has exited,
        # so a child writing during shutdown still has a valid fd.
        for lh, _lp in child_logs:
            try:
                lh.close()
            except OSError:
                pass
        for s in cleanup:
            try:
                os.unlink(s)
            except OSError:
                pass

    # 5) Report
    k6_data = json.loads(k6_summary.read_text()) if k6_summary.exists() else {}
    if k6_summary.exists():
        k6_summary.unlink()
    metrics = k6_data.get("metrics", {})
    run = metrics.get("action_run_ms", {})
    turns = metrics.get("completed_turns", {}).get("count", 0)
    sessions = metrics.get("completed_sessions", {}).get("count", 0)

    print()
    print("=" * 78)
    print("LOAD DEMO — RESULT (hardware-anchored)")
    print("=" * 78)
    if ginfo.get("gpus"):
        g = ginfo["gpus"][0]
        print(f"GPU:  {g['name']}  ({g['mem_total_mib']:.0f} MiB)")
    if gpu_report:
        print(f"      VRAM used: peak {gpu_report['mem_used_mib']['peak']:.0f} MiB "
              f"(mean {gpu_report['mem_used_mib']['mean']:.0f})  ·  "
              f"GPU util mean {gpu_report['util_pct']['mean']:.0f}% peak {gpu_report['util_pct']['peak']:.0f}%  ·  "
              f"power peak {gpu_report['power_w']['peak']:.0f} W")
    if proc_report:
        print(f"CPU:  process CPU mean {proc_report['cpu_pct']['mean']:.0f}% "
              f"peak {proc_report['cpu_pct']['peak']:.0f}% (of {proc_report['cpu_pct']['cores']*100}%)")
        print(f"RAM:  process RSS peak {proc_report['rss_mb']['peak']:.0f} MB")
    print(f"Load: peak {args.peak_vus} VUs · {args.workers} workers · {args.daemons} daemon(s) "
          f"(batch_max {args.batch_max})")
    if run:
        print(f"/v1/run latency (Semvec overhead BEFORE the LLM):")
        print(f"      med {run.get('med',0):.1f} ms · p90 {run.get('p(90)',0):.1f} ms · "
              f"p95 {run.get('p(95)',0):.1f} ms · p99 {run.get('p(99)',0):.1f} ms")
    dur_s = max(1.0, (metrics.get("iteration_duration", {}) or {}).get("count", 0) or 0)
    print(f"Throughput: {turns:,} turns / {sessions:,} sessions completed")
    fail = metrics.get("http_req_failed", {}).get("value")
    if fail is not None:
        print(f"Error rate: {fail*100:.3f}%")
    print("=" * 78)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "config": _report_config(vars(args)),
            "gpu_info": ginfo, "gpu": gpu_report, "resources": proc_report,
            "k6": k6_data,
        }, indent=2, default=str))
        print(f"Report → {args.output}")
    return k6_rc


if __name__ == "__main__":
    sys.exit(main())
