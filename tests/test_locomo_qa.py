"""Unit tests for the LOCOMO QA harness (pillar 2, real-dataset) — scoring/flatten, no LLM."""

import pytest

import run_locomo_qa as lq


# --- f1_score --------------------------------------------------------------
def test_f1_exact_match():
    assert lq.f1_score("7 May 2023", "7 May 2023") == pytest.approx(1.0)


def test_f1_case_and_punct_insensitive():
    assert lq.f1_score("Psychology, Counseling.", "psychology counseling") == pytest.approx(1.0)


def test_f1_partial_overlap():
    # pred has 2 tokens, gold has 3; overlap 2 → F1 = 2*2/(2+3) = 0.8
    assert lq.f1_score("psychology counseling", "psychology counseling certification") == pytest.approx(0.8)


def test_f1_no_overlap():
    assert lq.f1_score("banana", "psychology") == pytest.approx(0.0)


def test_f1_empty():
    assert lq.f1_score("", "anything") == pytest.approx(0.0)


# --- is_abstention ---------------------------------------------------------
def test_abstention_detected():
    for s in ["No information available.", "This is not mentioned in the conversation.",
              "I don't know.", "There is no relevant information."]:
        assert lq.is_abstention(s), s


def test_abstention_negative():
    assert not lq.is_abstention("Caroline went on 7 May 2023.")


# --- score_qa (category-aware) --------------------------------------------
def test_score_adversarial_category5_rewards_abstention():
    assert lq.score_qa("There is no information about that.", "Not mentioned", "5") == pytest.approx(1.0)
    assert lq.score_qa("It was on 7 May.", "Not mentioned", "5") == pytest.approx(0.0)


def test_score_normal_category_uses_f1():
    assert lq.score_qa("7 May 2023", "7 May 2023", "2") == pytest.approx(1.0)


def test_score_category_int_or_str():
    # category may arrive as int 5 or str "5"
    assert lq.score_qa("no information", "x", 5) == pytest.approx(1.0)


# --- flatten_conversation --------------------------------------------------
def test_flatten_orders_sessions_and_formats_turns():
    conv = {
        "speaker_a": "Caroline", "speaker_b": "Melanie",
        "session_1_date_time": "1 Jan", "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hi"},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "Hello"},
        ],
        "session_2_date_time": "2 Jan", "session_2": [
            {"speaker": "Caroline", "dia_id": "D2:1", "text": "Bye"},
        ],
    }
    turns = lq.flatten_conversation(conv)
    assert turns == ["Caroline: Hi", "Melanie: Hello", "Caroline: Bye"]


def test_flatten_handles_missing_sessions():
    assert lq.flatten_conversation({"speaker_a": "A", "speaker_b": "B"}) == []


# --- parse_judge_verdict (LLM-judge output parsing) ------------------------
def test_judge_verdict_correct():
    for s in ["CORRECT", "correct", "Yes, correct", "yes", "true"]:
        assert lq.parse_judge_verdict(s) is True, s


def test_judge_verdict_wrong():
    for s in ["WRONG", "Incorrect.", "incorrect", "no", "No, wrong", "false"]:
        assert lq.parse_judge_verdict(s) is False, s


# --- rrf_fuse (hybrid dense+BM25 fusion) -----------------------------------
def test_rrf_fuse_single_list_preserves_order():
    assert lq.rrf_fuse([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_fuse_rewards_top_of_both_lists():
    # 'a' is rank 1 in both lists → must come first
    out = lq.rrf_fuse([["a", "b", "c"], ["a", "c"]], k=60)
    assert out[0] == "a"
    # 'c' (ranks 3 & 2) outranks 'b' (rank 2, absent in list 2)
    assert out.index("c") < out.index("b")


def test_rrf_fuse_dedupes():
    out = lq.rrf_fuse([["a", "b"], ["b", "a"]])
    assert sorted(out) == ["a", "b"]


def test_judge_disables_thinking_for_reasoning_judge(monkeypatch):
    """Reasoning judges (e.g. Qwen3.6) must get enable_thinking=False, else they
    spend the tiny token budget on hidden <think> and return empty -> always WRONG."""
    import json as _json
    import urllib.request
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"CORRECT"}}]}'

    def fake_urlopen(req, timeout=None):
        captured["body"] = _json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    verdict = lq.judge("http://x/v1", "k", "Qwen/Qwen3.6-27B", "q", "gold", "pred", timeout=5)
    assert verdict is True
    assert captured["body"].get("chat_template_kwargs") == {"enable_thinking": False}
    assert captured["body"]["max_tokens"] >= 8

# --- environment hygiene ----------------------------------------------------
def test_env_strips_crlf_and_whitespace(monkeypatch):
    """A .env written on Windows leaves a trailing CR in every value.

    `set -a; . .env` keeps the CR, so OPENAI_BASE_URL becomes
    "https://host:port/v1\r" and every request dies with an opaque
    "URL rejected: Malformed input to a URL function" — with the endpoint
    perfectly reachable. Reading env values must strip.
    """
    monkeypatch.setenv("SEMVEC_TEST_URL", "https://host:32195/v1\r")
    assert lq._env("SEMVEC_TEST_URL") == "https://host:32195/v1"
    monkeypatch.setenv("SEMVEC_TEST_URL", "  padded-model  ")
    assert lq._env("SEMVEC_TEST_URL") == "padded-model"
    monkeypatch.delenv("SEMVEC_TEST_URL", raising=False)
    assert lq._env("SEMVEC_TEST_URL", "fallback") == "fallback"
