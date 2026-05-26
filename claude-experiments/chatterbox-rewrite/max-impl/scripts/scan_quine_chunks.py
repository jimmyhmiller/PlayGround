"""Render every Quine chunk and flag the ones that produce loud thumps.

Uses the same seed (0xDEADBEEF, no per-chunk variance) the audiobook pipeline uses.
"""
import sys, os, json
sys.path.insert(0, '/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/paper-audiobooks/src')
sys.path.insert(0, '.')

os.environ['CHATTERBOX_BF16'] = '1'
os.environ['CHATTERBOX_CFM_STEPS'] = '5'
os.environ['CHATTERBOX_T3_FUSE_QKV'] = '1'
os.environ['CHATTERBOX_T3_FUSE_MLP'] = '1'

import numpy as np

with open('/tmp/quine_mojo/Quine-Epistemology-Naturalized.chapters.json') as f:
    chapters = json.load(f)

# Inline naive chunker so we don't import paper-audiobooks (no soundfile etc).
import re
def chunk_text(text, max_chars=250):
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    cur = ""
    for s in sents:
        if not s.strip(): continue
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip() if cur else s
    if cur:
        chunks.append(cur.strip())
    return chunks

# This may not match the pipeline's chunker exactly, but for finding ROUGH-thump-chunks it's enough.
body = chapters[0]['body']
chunks = chunk_text(body, max_chars=250)
print(f"[scan] total chunks: {len(chunks)}", flush=True)

from chatterbox_mojo import ChatterboxTTS
tts = ChatterboxTTS.from_pretrained(use_bf16=True)
tts.prepare_conditionals('/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav', exaggeration=0.5)

bad_chunks = []
for i, text in enumerate(chunks):
    wav = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95, min_p=0.05,
                       repetition_penalty=1.2, exaggeration=0.5, rng_seed=0xDEADBEEF)
    audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
    diff = np.abs(np.diff(audio))
    big = np.flatnonzero(diff > 0.8)
    big_huge = int(np.sum(diff > 1.5))
    if len(big) >= 3:
        bad_chunks.append((i, len(text), len(big), big_huge, text[:80]))
        print(f"[bad] chunk {i} ({len(text)} chars, audio={audio.size/24000:.1f}s): {len(big)} thumps, {big_huge} huge", flush=True)
        np.savez(f'/tmp/quine_mojo/bad_chunk_{i:03d}.npz',
                 audio=audio, text=np.array([text]), idx=np.array([i]))

print(f"\n[scan] {len(bad_chunks)} bad chunks out of {len(chunks)}")
for ci, nchars, nt, nh, text80 in bad_chunks:
    print(f"  chunk {ci} ({nchars} chars): {nt} thumps, {nh} huge — {text80!r}")
