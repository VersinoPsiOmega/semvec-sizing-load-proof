"""Unit tests for the measurement harness (pillar 2) — generation/aggregation, no LLM/embedder."""

import json
import random
from pathlib import Path

import pytest
import yaml

import run_profile as rp

PROFILE = Path(__file__).resolve().parent.parent / "harness" / "profiles" / "reference_synthetic.yaml"


@pytest.fixture
def profile():
    return yaml.safe_load(PROFILE.read_text())


@pytest.fixture
def count_tokens():
    return rp.make_token_counter()


# --- token counter ---------------------------------------------------------
def test_token_counter_positive(count_tokens):
    assert count_tokens("hello world") > 0


def test_token_counter_monotonic(count_tokens):
    assert count_tokens("a b c d e") > count_tokens("a b")


def test_token_counter_empty(count_tokens):
    assert count_tokens("") == 0


# --- the report must render whatever summarize() produces ------------------
def test_print_report_renders_a_current_summary(capsys):
    """print_report must consume summarize()'s own output.

    It reached into a key summarize() no longer emits, so a finished
    measurement crashed *after* the run — the expensive part was done and the
    result was lost to a KeyError. Feed it the real thing.
    """
    rows = [_row(0, t, "filler", 1000 + 100 * t, 900) for t in range(3)]
    report = rp.summarize(rows, [], use_llm=False)
    rp.print_report(report)
    out = capsys.readouterr().out
    assert "Turns measured" in out
    assert "break-even" in out.lower()


# --- both paths must be counted the same way -------------------------------
def test_semvec_input_counts_system_prompt_and_user_message():
    """Both paths must count what a deployment actually sends.

    The naive path carries the system prompt, the history and the current user
    message. The Semvec path replaces the *history* with a context block — it
    does not remove the system prompt (Semvec is not the RAG store; the README
    says so repeatedly) and it certainly does not remove the user's question.
    Counting only the context block on one side and everything on the other
    inflates the reported reduction by the size of the system prompt.
    """
    count = len  # 1 token per character keeps the arithmetic obvious
    system, context, message = "SYS", "CTX-BLOCK", "USER-ASKS"
    got = rp.semvec_input_tokens(system, context, message, count)
    assert got == len(system) + len(context) + len(message)


def test_naive_input_counts_history_and_user_message():
    """The naive side stays what it was: everything resent on this turn."""
    count = len
    history = ["SYS", "earlier-turn"]
    message = "USER-ASKS"
    got = rp.naive_input_tokens(history, message, count)
    assert got == sum(map(len, history)) + len(message)


def test_the_two_paths_differ_only_by_history_vs_context():
    """With an empty context block the Semvec path cannot beat the naive one.

    On the very first turn there is no history to compress, so a fair count has
    the Semvec path at system + message and the naive path at system + message
    too. Any reported saving there would be an artefact of the counting.
    """
    count = len
    system, message = "SYSTEM-PROMPT", "QUESTION"
    naive = rp.naive_input_tokens([system], message, count)
    semvec = rp.semvec_input_tokens(system, "", message, count)
    assert semvec == naive


# --- break-even is derived from the data, not asserted ---------------------
def test_break_even_turn_is_the_first_turn_semvec_wins_and_keeps_winning():
    """`break_even_turn` must come from the measurements, not from a constant.

    Rows below: Semvec loses on turn 0, wins on 1, loses again on 2, then wins
    for good from turn 3. The break-even is 3 — the first turn from which it
    never loses again — not 1.
    """
    rows = [
        {"turn": 0, "naive_input_tokens": 100, "semvec_input_tokens": 120},
        {"turn": 1, "naive_input_tokens": 200, "semvec_input_tokens": 150},
        {"turn": 2, "naive_input_tokens": 300, "semvec_input_tokens": 310},
        {"turn": 3, "naive_input_tokens": 400, "semvec_input_tokens": 200},
        {"turn": 4, "naive_input_tokens": 500, "semvec_input_tokens": 210},
    ]
    assert rp.break_even_turn(rows, min_samples=1) == 3


def test_break_even_turn_ignores_thinly_sampled_tail_turns():
    """A turn reached by one conversation must not decide the whole metric.

    Turn counts are triangular, so the highest indices are reached by a single
    conversation. Letting one such row end the "stays cheaper" run would report
    no break-even at all while hundreds of turns below it win clearly.
    """
    rows = [{"turn": t, "naive_input_tokens": 1000 + 100 * t, "semvec_input_tokens": 500}
            for t in range(10) for _ in range(5)]          # 5 samples per turn, Semvec wins
    rows.append({"turn": 10, "naive_input_tokens": 400, "semvec_input_tokens": 900})  # n=1 outlier
    assert rp.break_even_turn(rows) == 0


def test_break_even_turn_still_respects_a_well_sampled_loss():
    """A turn with enough samples does end the run, thin tails or not."""
    rows = [{"turn": t, "naive_input_tokens": 1000, "semvec_input_tokens": 500}
            for t in range(3) for _ in range(5)]
    rows += [{"turn": 3, "naive_input_tokens": 400, "semvec_input_tokens": 900} for _ in range(5)]
    rows += [{"turn": 4, "naive_input_tokens": 1000, "semvec_input_tokens": 500} for _ in range(5)]
    assert rp.break_even_turn(rows) == 4


def test_break_even_turn_is_zero_when_semvec_wins_throughout():
    rows = [
        {"turn": 0, "naive_input_tokens": 100, "semvec_input_tokens": 40},
        {"turn": 1, "naive_input_tokens": 200, "semvec_input_tokens": 60},
    ]
    assert rp.break_even_turn(rows, min_samples=1) == 0


def test_break_even_turn_is_none_when_semvec_never_wins():
    rows = [
        {"turn": 0, "naive_input_tokens": 100, "semvec_input_tokens": 140},
        {"turn": 1, "naive_input_tokens": 200, "semvec_input_tokens": 260},
    ]
    assert rp.break_even_turn(rows, min_samples=1) is None


def test_summary_reports_break_even_not_an_assumed_turn_threshold():
    """The summary must not carry a hardcoded turn-10 cut."""
    rows = [
        {"cid": c, "turn": t, "kind": "filler",
         "naive_input_tokens": 100 + 50 * t, "semvec_input_tokens": 90}
        for t in range(12) for c in range(3)
    ]
    out = rp.summarize(rows, [], use_llm=False)["summary"]
    assert "input_reduction_pct_turn10plus" not in out, (
        "a fixed turn-10 threshold implies a break-even the data has to show"
    )
    assert out["break_even_turn"] == 0   # Semvec is cheaper from the very first turn
    assert "input_reduction_pct_overall" in out


# --- structural determinism ------------------------------------------------
def test_conversation_structure_independent_of_token_counter():
    """Turn counts must not shift when the token counter or word pool changes.

    The filler text must not consume entropy from the same rng that decides the
    conversation structure: otherwise a different tokeniser (or a reworded word
    pool) silently changes turns-per-conversation, and `seed:` stops meaning
    "reproducible run".
    """
    profile = yaml.safe_load(PROFILE.read_text())
    coarse = rp.generate_conversations(profile, lambda s: max(1, len(s) // 4))
    fine = rp.generate_conversations(profile, lambda s: max(1, len(s) // 7))
    assert [len(c.turns) for c in coarse] == [len(c.turns) for c in fine]
    assert [[n.code for n in c.needles] for c in coarse] == \
           [[n.code for n in c.needles] for c in fine]


# --- _triangular_int -------------------------------------------------------
def test_triangular_within_bounds():
    rng = random.Random(1)
    for _ in range(200):
        v = rp._triangular_int(rng, 4, 12, 40)
        assert 4 <= v <= 40


def test_triangular_deterministic():
    a = [rp._triangular_int(random.Random(7), 4, 12, 40) for _ in range(5)]
    b = [rp._triangular_int(random.Random(7), 4, 12, 40) for _ in range(5)]
    assert a == b


# --- generate_conversations ------------------------------------------------
def test_generate_count(profile, count_tokens):
    convs = rp.generate_conversations(profile, count_tokens)
    assert len(convs) == profile["conversations"]["count"]


def test_generate_deterministic(profile, count_tokens):
    a = rp.generate_conversations(profile, count_tokens)
    b = rp.generate_conversations(profile, count_tokens)
    assert [len(c.turns) for c in a] == [len(c.turns) for c in b]
    assert [[n.code for n in c.needles] for c in a] == [[n.code for n in c.needles] for c in b]


def test_seed_changes_output(profile, count_tokens):
    a = rp.generate_conversations(profile, count_tokens)
    p2 = {**profile, "seed": profile["seed"] + 1}
    b = rp.generate_conversations(p2, count_tokens)
    assert [len(c.turns) for c in a] != [len(c.turns) for c in b] or \
        [[n.code for n in c.needles] for c in a] != [[n.code for n in c.needles] for c in b]


def test_needle_set_and_probe_share_code(profile, count_tokens):
    convs = rp.generate_conversations(profile, count_tokens)
    for c in convs:
        codes_set = {t["needle"] for t in c.turns if t["kind"] == "needle_set"}
        codes_probe = {t["needle"] for t in c.turns if t["kind"] == "needle_probe"}
        # every probe references a set code
        assert codes_probe <= codes_set


def test_needle_set_contains_its_code(profile, count_tokens):
    convs = rp.generate_conversations(profile, count_tokens)
    for c in convs:
        for t in c.turns:
            if t["kind"] == "needle_set":
                assert t["needle"] in t["content"]


def test_turn_bounds_respected(profile, count_tokens):
    convs = rp.generate_conversations(profile, count_tokens)
    lo, hi = profile["conversations"]["turns_min"], profile["conversations"]["turns_max"]
    for c in convs:
        assert lo <= len(c.turns) <= hi


# --- load_custom_profile -------------------------------------------------
def test_load_custom_profile(tmp_path):
    f = tmp_path / "profile.jsonl"
    f.write_text(
        json.dumps({"turns": [{"role": "user", "content": "question 1"},
                              {"role": "user", "content": "question 2"}]}) + "\n"
        + json.dumps({"turns": [{"role": "user", "content": "x", "kind": "needle_set", "needle": "RX1-A"}]}) + "\n"
    )
    convs = rp.load_custom_profile(f)
    assert len(convs) == 2
    assert len(convs[0].turns) == 2
    assert convs[1].turns[0]["needle"] == "RX1-A"


def test_load_custom_profile_skips_blank_lines(tmp_path):
    f = tmp_path / "k.jsonl"
    f.write_text(json.dumps({"turns": [{"content": "a"}]}) + "\n\n  \n")
    assert len(rp.load_custom_profile(f)) == 1


# --- summarize -------------------------------------------------------------
def _row(cid, turn, kind, naive, semvec):
    return {"cid": cid, "turn": turn, "kind": kind,
            "naive_input_tokens": naive, "semvec_input_tokens": semvec, "phase": "x"}


def test_summarize_reduction_math():
    rows = [_row(0, 0, "filler", 1000, 250), _row(0, 1, "filler", 2000, 250)]
    rep = rp.summarize(rows, [], use_llm=False)
    s = rep["summary"]
    assert s["total_naive_input_tokens"] == 3000
    assert s["total_semvec_input_tokens"] == 500
    assert s["input_reduction_pct_overall"] == pytest.approx(83.3, abs=0.1)


def test_summarize_reports_a_measured_break_even():
    """The summary reports where Semvec starts winning, measured, not assumed."""
    rows = ([_row(c, 0, "filler", 1000, 1200) for c in range(3)]    # naive cheaper on turn 0
            + [_row(c, 1, "filler", 2000, 900) for c in range(3)]   # Semvec wins from turn 1
            + [_row(c, 2, "filler", 3000, 950) for c in range(3)])
    rep = rp.summarize(rows, [], use_llm=False)
    assert rep["summary"]["break_even_turn"] == 1


def test_summarize_needle_retention():
    needles = [{"cid": 0, "turn": 8, "code": "A", "retained_in_context": True},
               {"cid": 0, "turn": 9, "code": "B", "retained_in_context": False}]
    rep = rp.summarize([_row(0, 0, "filler", 10, 5)], needles, use_llm=False)
    q = rep["summary"]["needle_quality"]
    assert q["total_probes"] == 2
    assert q["retained_in_context"] == 1
    assert q["retention_pct"] == pytest.approx(50.0)


def test_summarize_empty_rows():
    rep = rp.summarize([], [], use_llm=False)
    assert rep["summary"]["turns_measured"] == 0
    assert rep["summary"]["needle_quality"]["retention_pct"] is None


def test_token_counter_handles_special_tokens(count_tokens):
    # tiktoken raises on <|endoftext|> by default; the counter must not.
    assert count_tokens("hello <|endoftext|> world") > 0
