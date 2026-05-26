"""Run upstream chatterbox with T3 cast to bf16, see if the thump appears."""
import sys, importlib.metadata as _im
sys.path.insert(0, '/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src')
_orig = _im.version
def _v(name):
    try: return _orig(name)
    except _im.PackageNotFoundError: return "0.0.0"
_im.version = _v

import os, torch, numpy as np, wave
torch.manual_seed(0xDEADBEEF + 25)

REF = '/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav'
TEXT = ('By his identification of bodies with impressions he did succeed in construing '
        'some singular statements about bodies as indubitable truths, yes; as truths '
        'about impressions, directly known.')

from chatterbox.tts import ChatterboxTTS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Run 1: f32 T3 (default).
print("=== f32 T3 (default) ===", flush=True)
torch.manual_seed(0xDEADBEEF + 25)
tts = ChatterboxTTS.from_pretrained(device=device)
tts.prepare_conditionals(REF, exaggeration=0.5)
wav_f32 = tts.generate(TEXT, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                       min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)
audio_f32 = wav_f32.squeeze(0).cpu().numpy().astype(np.float32)
diff = np.abs(np.diff(audio_f32))
print(f"f32 audio={audio_f32.size/24000:.2f}s, loud thumps (>0.8) = {int(np.sum(diff>0.8))}", flush=True)
with wave.open('/tmp/cfm_diag/upstream_f32.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes((np.clip(audio_f32, -1, 1) * 32767).astype(np.int16).tobytes())

# Run 2: cast T3 backbone to bf16 (the actual transformer layers).
print("=== bf16 T3 backbone ===", flush=True)
torch.manual_seed(0xDEADBEEF + 25)
tts = ChatterboxTTS.from_pretrained(device=device)
tts.prepare_conditionals(REF, exaggeration=0.5)
# Cast just the llama backbone to bf16, keep heads / embeds / norms in f32.
tts.t3.tfmr = tts.t3.tfmr.to(dtype=torch.bfloat16)
# Embeddings produce f32 still — need cast at the boundary. Easiest: monkey-patch
# input_embeds to cast right before tfmr call. But T3.inference passes inputs_embeds
# directly, so we need to cast there. Cleanest: wrap tfmr.forward.
orig_tfmr_forward = tts.t3.tfmr.forward
def bf16_tfmr_forward(inputs_embeds=None, **kw):
    if inputs_embeds is not None:
        inputs_embeds = inputs_embeds.to(torch.bfloat16)
    out = orig_tfmr_forward(inputs_embeds=inputs_embeds, **kw)
    # Cast last_hidden_state back to f32 for downstream f32 heads.
    if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
        out.last_hidden_state = out.last_hidden_state.float()
    if hasattr(out, 'hidden_states') and out.hidden_states is not None:
        out.hidden_states = tuple(h.float() if h is not None else h for h in out.hidden_states)
    return out
tts.t3.tfmr.forward = bf16_tfmr_forward

wav_bf16 = tts.generate(TEXT, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                        min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)
audio_bf16 = wav_bf16.squeeze(0).cpu().numpy().astype(np.float32)
diff = np.abs(np.diff(audio_bf16))
print(f"bf16 audio={audio_bf16.size/24000:.2f}s, loud thumps (>0.8) = {int(np.sum(diff>0.8))}", flush=True)
with wave.open('/tmp/cfm_diag/upstream_bf16.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes((np.clip(audio_bf16, -1, 1) * 32767).astype(np.int16).tobytes())

print("saved /tmp/cfm_diag/upstream_{f32,bf16}.wav for A/B")
