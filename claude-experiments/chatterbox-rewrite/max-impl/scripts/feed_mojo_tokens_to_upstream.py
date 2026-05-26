"""Take Mojo's bad-seed token sequence and feed it into upstream's CFM.
If upstream also thumps on these tokens → tokens themselves are pathological.
If upstream is clean → our flow encoder has a bug.
"""
import os, sys, types
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "chatterbox" / "src"))
os.environ['CHATTERBOX_BF16'] = '1'
os.environ['CHATTERBOX_CFM_STEPS'] = '5'
os.environ['CHATTERBOX_T3_FUSE_QKV'] = '1'
os.environ['CHATTERBOX_T3_FUSE_MLP'] = '1'

REF = '/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav'
TEXT = ("What then could have motivated Carnap's heroic efforts on the conceptual "
        "side of epistemology, when hope of certainty on the doctrinal side was "
        "abandoned? There were two good reasons still.")

# Step 1: get Mojo tokens for the bad seed.
from chatterbox_mojo import ChatterboxTTS
import op_t3, op_text_tokenize
from chatterbox_mojo.tts import punc_norm

mojo = ChatterboxTTS.from_pretrained(use_bf16=True)
mojo.prepare_conditionals(REF, exaggeration=0.5)
text_n = punc_norm(TEXT)
text_ids = op_text_tokenize.tokenize(mojo._tok_h, text_n)
text_ids_full = [255] + list(text_ids) + [0]
raw = op_t3.generate(mojo._t3_h, mojo.conds.speaker_emb_256, mojo.conds.cond_prompt_tok,
                     text_ids_full, {'emotion': 0.5, 'cfg_weight': 0.5, 'temperature': 0.8,
                     'top_p': 0.95, 'rep_penalty': 1.2, 'min_p': 0.05, 'max_new': 1000,
                     'rng_seed': 0xDEADBEEF + 53})
EOS = 6562
mojo_tokens = [int(t) for t in raw if t != EOS and t < 6561]
print(f"mojo bad-seed tokens: {len(mojo_tokens)}", flush=True)
del mojo

# Step 2: feed into upstream's s3gen.
import importlib.metadata as _im
_orig_version = _im.version
def _fake_version(name):
    try: return _orig_version(name)
    except _im.PackageNotFoundError: return "0.0.0"
_im.version = _fake_version
from chatterbox.tts import ChatterboxTTS as UpstreamTTS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ups = UpstreamTTS.from_pretrained(device=device)
ups.prepare_conditionals(REF, exaggeration=0.5)

speech_tokens = torch.tensor(mojo_tokens, dtype=torch.long, device=device).unsqueeze(0)
print(f"upstream s3gen device={device}, tokens shape={speech_tokens.shape}", flush=True)

with torch.inference_mode():
    wav, _ = ups.s3gen.inference(
        speech_tokens=speech_tokens,
        ref_dict=ups.conds.gen,
    )
    audio = wav.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

print(f"upstream s3gen on mojo's tokens: {audio.size/24000:.2f}s", flush=True)
diff = np.abs(np.diff(audio))
big = np.flatnonzero(diff > 0.8)
if len(big):
    gaps = np.diff(big) if len(big) > 1 else np.array([])
    starts = np.concatenate([[0], np.flatnonzero(gaps > 24000//20) + 1]) if len(big) > 1 else [0]
    n_clusters = len(starts)
else:
    n_clusters = 0
print(f"upstream LOUD THUMPS on mojo's bad-seed tokens: {n_clusters}", flush=True)

import wave
pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
out = "/tmp/cfm_diag/upstream_on_mojo_tokens.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes(pcm.tobytes())
print(f"saved {out}", flush=True)
