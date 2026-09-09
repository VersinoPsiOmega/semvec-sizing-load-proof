#!/usr/bin/env python3
"""Verify that the headline numbers in the docs still match the artefacts.

Documentation drifts from evidence quietly: an artefact gets re-measured, a
figure in the README does not, and nobody notices until a reader checks. This
script re-derives each headline claim from `results/` and fails if the docs
disagree, so drift surfaces in CI rather than in front of a reviewer.

Each check anchors on a *context pattern*, not a bare number. Searching for
"92" alone is useless -- it also matches "GPU 92 %" -- so every rule pins the
figure to the sentence or table cell that carries it.

    python tools/check_claims.py        # exit 0 = docs and artefacts agree
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Documents round; artefacts do not. But a *relative* tolerance is wrong for a
# figure that is itself a percentage: 2 % of "92.0 %" is 1.8 points, wide enough
# to wave through a materially different claim. Percentages get an absolute
# tolerance instead.
REL_TOLERANCE = 0.005   # 0.5 % for counts and token figures
PCT_TOLERANCE = 0.1     # 0.1 percentage points for figures that are percentages


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text())


def documents() -> list[tuple[str, str]]:
    paths = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").glob("*.md"))
    return [(p.name, p.read_text()) for p in paths]


class Checker:
    def __init__(self) -> None:
        self.docs = documents()
        self.failures: list[str] = []
        self.checks = 0
        self.matches = 0

    def expect(self, label: str, pattern: str, expected: float, *, group: int = 1,
               required: bool = True, is_pct: bool = False) -> None:
        """Every occurrence of `pattern` must carry a value close to `expected`."""
        self.checks += 1
        rx = re.compile(pattern)
        tol = PCT_TOLERANCE if is_pct else max(abs(expected) * REL_TOLERANCE, 0.005)
        seen = 0
        for name, text in self.docs:
            for m in rx.finditer(text):
                seen += 1
                got = float(m.group(group).replace(",", "").replace("−", "-"))
                if abs(got - expected) > tol:
                    self.failures.append(
                        f"{label}: {name} says {got}, artefact says {expected:.4g} "
                        f"(tolerance {tol:.4g})")
        self.matches += seen
        if required and seen == 0:
            self.failures.append(f"{label}: pattern never matched — claim missing or reworded")


def main() -> int:
    c = Checker()

    # --- LOCOMO full suite ----------------------------------------------
    s = load("locomo_qa_tuned_full.json")["summary"]
    c.expect("LOCOMO reduction", r"Input tokens/QA \| [\d,]+ \| \*\*[\d,]+ \(−([\d.]+) %\)",
             s["input_reduction_pct"], is_pct=True)
    c.expect("LOCOMO naive input", r"Input tokens/QA \| ([\d,]+) \|",
             s["naive_input_tokens_mean"])
    c.expect("LOCOMO semvec input", r"Input tokens/QA \| [\d,]+ \| \*\*([\d,]+) ",
             s["semvec_input_tokens_mean"])
    c.expect("LOCOMO naive F1", r"Answer F1 \(word\) \| ([\d.]+) \|", s["naive_f1"])
    c.expect("LOCOMO semvec F1", r"Answer F1 \(word\) \| [\d.]+ \| \*\*([\d.]+) ", s["semvec_f1"])
    c.expect("LOCOMO QA count", r"\*\*([\d,]+) QA\*\*", s["qa_count"])

    # --- LongBench-v2 ----------------------------------------------------
    s = load("longbench_v2_qwen27b.json")["summary"]
    c.expect("LongBench reduction", r"\(feasible\) \| [\d,]+ \| \*\*([\d,]+) \(−[\d.]+ %\)",
             s["semvec_input_tokens_mean"])
    c.expect("LongBench naive input", r"Input tokens/item \(feasible\) \| ([\d,]+) \|",
             s["naive_input_tokens_mean"])
    c.expect("LongBench mean context", r"[Mm]ean context \*\*([\d,]+) tokens\*\*",
             s["context_tokens_mean"])
    c.expect("LongBench infeasible", r"\*\*([\d,]+) of 60 items impossible\*\*",
             s["naive_infeasible"], required=False)

    # --- synthetic profile ----------------------------------------------
    s = load("harness_synthetic.json")["summary"]
    c.expect("synthetic reduction", r"Semvec\s+([\d,]+) \(−[\d.]+ %\)\*\*",
             s["semvec_input_tokens_per_turn"]["mean"])
    c.expect("synthetic reduction pct", r"Semvec\s+[\d,]+ \(−([\d.]+) %\)\*\*",
             s["input_reduction_pct_overall"], is_pct=True)
    c.checks += 1
    if s["needle_quality"]["retention_pct"] != 100.0:
        c.failures.append("needle retention is no longer 100 %")

    # --- load: capacity and latency --------------------------------------
    cap = load("load_rerank_200vu.json")["k6"]["metrics"]["completed_turns"]["rate"]
    c.expect("capacity", r"\*\*([\d,]+) embeds/s\*\* at 200 concurrent", cap)
    lat = load("load_realistic_rerank.json")["k6"]["metrics"]["action_run_ms"]["med"]
    c.expect("tuned latency", r"\*\*([\d.]+) ms\*\* median", lat)

    # --- the documented test count must match the suite ------------------
    c.checks += 1
    try:
        import subprocess
        # pytest.ini already sets -q, so a second -q switches the collect output
        # to "<file>: <count>" lines rather than a summary. Sum those.
        out = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests"),
                              "--collect-only", "-q"], capture_output=True, text=True,
                             cwd=ROOT, timeout=300).stdout
        counts = [int(m) for m in re.findall(r"^\S+\.py: (\d+)$", out, re.M)]
        collected = sum(counts)
        if not collected:
            m = re.search(r"^(\d+) tests? collected", out, re.M)
            collected = int(m.group(1)) if m else 0
    except Exception:
        collected = 0
    if collected:
        c.expect("documented test count", r"\b(\d+) (?:unit )?tests\b", collected,
                 required=False)

    # --- every load artefact must stay error-free ------------------------
    for p in sorted((ROOT / "results").glob("load_*.json")):
        c.checks += 1
        m = json.loads(p.read_text())["k6"]["metrics"].get("http_req_failed", {})
        failed, ok = m.get("passes", 0), m.get("fails", 0)
        if failed + ok and failed / (failed + ok) > 0.001:
            c.failures.append(f"{p.name}: error rate {100*failed/(failed+ok):.2f} % exceeds 0.1 %")

    print(f"checked {c.checks} claims, {c.matches} document occurrences")
    if c.failures:
        print("\nFAILED — docs and artefacts disagree:")
        for f in c.failures:
            print(f"  - {f}")
        return 1
    print("all headline claims match their artefacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
