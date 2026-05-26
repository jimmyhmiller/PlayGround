"""Run upstream s3gen on Mojo's bad-seed tokens (loaded from file)."""
import sys, importlib.metadata as _im
sys.path.insert(0, '/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src')
_orig = _im.version
def _v(name):
    try: return _orig(name)
    except _im.PackageNotFoundError: return "0.0.0"
_im.version = _v

import numpy as np, torch, wave
from chatterbox.tts import ChatterboxTTS as UpstreamTTS

REF = '/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav'
mojo_tokens = np.load('/tmp/cfm_diag/mojo_bad_tokens.npy').tolist()
print(f"loaded mojo tokens: {len(mojo_tokens)}", flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}", flush=True)
ups = UpstreamTTS.from_pretrained(device=device)
ups.prepare_conditionals(REF, exaggeration=0.5)

speech_tokens = torch.tensor(mojo_tokens, dtype=torch.long, device=device).unsqueeze(0)

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
print(f"UPSTREAM CFM LOUD THUMPS on mojo's bad-seed tokens: {n_clusters}", flush=True)

pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
out = "/tmp/cfm_diag/upstream_on_mojo_tokens.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes(pcm.tobytes())
print(f"saved {out}", flush=True)
