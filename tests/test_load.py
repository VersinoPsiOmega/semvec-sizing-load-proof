"""Unit tests for the load demo (pillar 3) — aggregation logic, no subprocess/GPU/k6."""

from pathlib import Path

import pytest

import _gpu_sampler as gs
import run_load as rl


# --- daemon_spawn_kwargs: PyTorch (default) vs ONNX/Rust selection ----------
def test_daemon_kwargs_pytorch_default():
    kw = rl.daemon_spawn_kwargs(
        model="m", dimension=768, batch_max=32, batch_wait_ms=5.0, onnx=False
    )
    assert kw["model"] == "m" and kw["dimension"] == 768
    # the PyTorch python daemon path passes no rust executable/model files
    assert "executable" not in kw and "model_path" not in kw


def test_daemon_kwargs_onnx_passes_rust_paths():
    kw = rl.daemon_spawn_kwargs(
        model="m", dimension=768, batch_max=32, batch_wait_ms=5.0, onnx=True,
        rust_binary=Path("/bin/semvec-embedder"),
        onnx_paths=(Path("/t/tokenizer.json"), Path("/t/onnx/model.onnx")),
    )
    assert kw["executable"] == Path("/bin/semvec-embedder")
    assert kw["tokenizer_path"] == Path("/t/tokenizer.json")
    assert kw["model_path"] == Path("/t/onnx/model.onnx")


def test_daemon_kwargs_onnx_without_binary_errors():
    with pytest.raises(SystemExit, match="(?i)rust|cargo build"):
        rl.daemon_spawn_kwargs(
            model="m", dimension=768, batch_max=32, batch_wait_ms=5.0, onnx=True,
            rust_binary=None, onnx_paths=(Path("t"), Path("m")),
        )


def test_daemon_kwargs_onnx_without_model_errors():
    with pytest.raises(SystemExit, match="(?i)onnx|backend"):
        rl.daemon_spawn_kwargs(
            model="m", dimension=768, batch_max=32, batch_wait_ms=5.0, onnx=True,
            rust_binary=Path("/bin/semvec-embedder"), onnx_paths=None,
        )


# --- GpuSampler aggregation (without a real GPU) ---------------------------
def test_gpu_sampler_aggregation():
    s = gs.GpuSampler(interval=0.01)
    s.samples = [
        {"t": 1, "util_pct": 10.0, "mem_used_mib": 1000.0, "power_w": 50.0},
        {"t": 2, "util_pct": 90.0, "mem_used_mib": 3000.0, "power_w": 110.0},
    ]
    rep = s.stop()
    assert rep["util_pct"]["peak"] == 90.0
    assert rep["util_pct"]["mean"] == pytest.approx(50.0)
    assert rep["mem_used_mib"]["peak"] == 3000.0
    assert rep["power_w"]["peak"] == 110.0


def test_gpu_sampler_empty_returns_empty():
    s = gs.GpuSampler()
    assert s.stop() == {}


def test_gpu_info_structure():
    info = gs.gpu_info()
    # On hardware with nvidia-smi: dict with a 'gpus' list; otherwise an empty dict.
    assert isinstance(info, dict)
    if info:
        assert "gpus" in info
        for g in info["gpus"]:
            assert {"name", "mem_total_mib", "driver"} <= set(g)


def test_gpu_available_is_bool():
    assert isinstance(gs.gpu_available(), bool)


# --- _summarise_proc -------------------------------------------------------
def test_summarise_proc():
    samples = [
        {"t": 0, "rss_mb": 100.0, "cpu_pct": 0.0},   # prime sample (skipped)
        {"t": 1, "rss_mb": 200.0, "cpu_pct": 150.0},
        {"t": 2, "rss_mb": 400.0, "cpu_pct": 250.0},
    ]
    rep = rl._summarise_proc(samples)
    assert rep["rss_mb"]["peak"] == 400.0
    assert rep["cpu_pct"]["peak"] == 250.0
    # cpu averages over samples[1:] → (150+250)/2 = 200
    assert rep["cpu_pct"]["mean"] == pytest.approx(200.0)
    assert rep["cpu_pct"]["cores"] >= 1


def test_summarise_proc_empty():
    assert rl._summarise_proc([]) == {}


# --- reports must not leak absolute local paths ----------------------------
def test_report_scenario_path_is_repo_relative():
    """k6 writes its scenario path into the report; it must not be absolute.

    This repository is public, so an absolute path leaks the author's directory
    layout (and previously the internal project name). The publication gate
    rejects it, which means every load run would otherwise need manual cleanup.
    """
    # The repository's own scenario resolves to a portable, repo-relative path.
    here = Path(__file__).resolve().parent.parent / "load" / "k6_chat_scenario.js"
    assert rl._report_scenario_path(str(here)) == "load/k6_chat_scenario.js"
    # An already-relative path is passed through untouched.
    assert rl._report_scenario_path("load/k6_chat_scenario.js") == "load/k6_chat_scenario.js"
    # Any absolute path outside the repository is reduced to its basename: the
    # directory layout must not reach a published artefact.
    for outside in ("/somewhere/checkout/load/k6_chat_scenario.js", "/tmp/custom.js"):
        got = rl._report_scenario_path(outside)
        assert not got.startswith("/"), got
        assert "/" not in got, got


def test_report_config_scrubs_every_path_field():
    """No config field may carry an absolute path into a committed artefact.

    `scenario` was fixed first; `output` then leaked the caller's -o path. Scrub
    the whole config rather than chasing fields one at a time.
    """
    cfg = rl._report_config({
        "scenario": "/anywhere/load/k6_chat_scenario.js",
        "output": "/tmp/elsewhere/run.json",
        "workers": 12,
        "rerank": True,
        "think_ms": 0,
    })
    assert cfg["workers"] == 12 and cfg["rerank"] is True and cfg["think_ms"] == 0
    for key, value in cfg.items():
        assert not (isinstance(value, str) and value.startswith("/")), (key, value)


# --- child process stderr must never be an unread pipe ---------------------
def test_child_stderr_is_not_an_unread_pipe():
    """Child stderr must not go to subprocess.PIPE.

    Worker and daemon stderr was piped but only read on a startup failure. Once
    a child wrote ~64 KiB of warnings the pipe buffer filled, the child blocked
    in write(), and its in-flight /v1/run requests hung until the client
    timeout — showing up as a p95 pinned at 60 s with the GPU nearly idle.
    Newer ML libraries emit more warnings, so this got worse over time.
    Children must log to a file instead.
    """
    src = (Path(__file__).resolve().parent.parent / "load" / "run_load.py").read_text()
    assert "stderr=subprocess.PIPE" not in src, (
        "child stderr goes to an unread pipe; it will deadlock the child once full"
    )


def test_child_log_helper_creates_a_readable_file(tmp_path):
    """The helper must hand back a writable handle plus a readable path."""
    handle, path = rl._child_log("worker-test", tmp_path)
    try:
        handle.write(b"boom\n")
        handle.flush()
    finally:
        handle.close()
    assert path.exists()
    assert "boom" in path.read_text()


# --- constants / embedder default ------------------------------------------
def test_mpnet_defaults():
    assert rl.MPNET_MODEL == "paraphrase-multilingual-mpnet-base-v2"
    assert rl.MPNET_DIM == 768
