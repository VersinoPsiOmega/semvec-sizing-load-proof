#!/usr/bin/env python3
"""Measure the persisted state-blob size, for capacity planning.

Builds Semvec states from the shipped synthetic profile and reports the size of
`SemvecState.to_bytes(compress=True)` -- the exact blob a durable backend stores
per session. The blob grows with conversation length, so the output is a range
keyed to turn counts rather than a single number.

    python tools/measure_state_size.py

Needs the pillar-2 dependencies (`pip install -r requirements.txt`); no licence
key, no LLM and no network beyond the embedder download.
"""
import sys, pathlib, statistics
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
import run_profile as rp
import yaml

profile = yaml.safe_load((ROOT / "harness/profiles/reference_synthetic.yaml").read_text())
count = rp.make_token_counter()
convs = rp.generate_conversations(profile, count)
emb = rp.MpnetEmbedder()

from semvec import SemvecConfig
from semvec.token_reduction import ChatMessage, SemvecChatProxy

def find_state(proxy):
    seen = set()
    stack = [proxy]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if hasattr(o, "to_bytes") and not isinstance(o, (bytes, bytearray, str, int)):
            return o
        for a in vars(o) if hasattr(o, "__dict__") else []:
            v = getattr(o, a, None)
            if v is not None and hasattr(v, "__dict__") or hasattr(v, "to_bytes"):
                stack.append(v)
    return None

rows = []
for conv in convs[:8]:
    proxy = SemvecChatProxy(
        llm_call=lambda *a, **k: "(ok)",
        system_prompt="You are a helpful enterprise assistant.",
        pss_config=SemvecConfig(dimension=rp.MPNET_DIM),
        serializer_config=None,
        embedding_service=emb,
    )
    n = 0
    for turn in conv.turns:
        if turn["role"] != "user":
            continue
        proxy.chat(turn["content"])
        n += 1
    st = find_state(proxy)
    if st is None:
        print("NOTE attrs:", [a for a in vars(proxy)][:30])
        break
    try:
        blob = st.to_bytes(compress=True)
    except TypeError:
        blob = st.to_bytes()
    raw = None
    try:
        raw = len(st.to_bytes(compress=False))
    except Exception:
        pass
    rows.append((n, len(blob), raw))
    print(f"RESULT turns={n:3d} compressed={len(blob)/1024:7.1f} KiB"
          + (f"  uncompressed={raw/1024:8.1f} KiB" if raw else ""))
if rows:
    c = [r[1] for r in rows]
    print(f"RESULT SUMMARY n={len(rows)} state type={type(st).__name__} "
          f"min={min(c)/1024:.1f} median={statistics.median(c)/1024:.1f} max={max(c)/1024:.1f} KiB")
