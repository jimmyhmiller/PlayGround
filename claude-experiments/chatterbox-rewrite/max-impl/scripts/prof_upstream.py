"""Profile upstream chatterbox per-stage timing."""
import time, torch
from chatterbox.tts import ChatterboxTTS
import chatterbox.tts as tts_mod
from chatterbox.models.s3gen import S3Gen

orig_inference = S3Gen.inference
def timed_inference(self, *args, **kw):
    t0 = time.perf_counter()
    out = orig_inference(self, *args, **kw)
    dt = time.perf_counter() - t0
    print(f"[prof_up]   s3gen.inference={dt*1000:.1f}ms", flush=True)
    return out
S3Gen.inference = timed_inference

# Patch T3.inference.
import chatterbox.models.t3.t3 as t3_mod
orig_t3 = t3_mod.T3.inference
def timed_t3(self, *args, **kw):
    t0 = time.perf_counter()
    out = orig_t3(self, *args, **kw)
    dt = time.perf_counter() - t0
    print(f"[prof_up]   t3.inference({out.shape})={dt*1000:.1f}ms", flush=True)
    return out
t3_mod.T3.inference = timed_t3

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = ChatterboxTTS.from_pretrained(device=device)
tts.prepare_conditionals("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav", exaggeration=0.5)

texts = [
    "Various attempts have been made in recent years to state necessary and sufficient conditions for someones knowing a given proposition.",
    "I shall argue that the standard analysis of knowledge as justified true belief is false.",
    "Suppose that Smith and Jones have applied for a certain job.",
    "And suppose that Smith has strong evidence for the conjunctive proposition that Jones is the man who will get the job, and Jones has ten coins in his pocket.",
]
for t in texts:
    t0 = time.perf_counter()
    w = tts.generate(t, cfg_weight=0.5, temperature=0.8, top_p=0.95, min_p=0.05, repetition_penalty=1.2, exaggeration=0.5)
    dt = time.perf_counter() - t0
    audio_s = w.shape[-1] / tts.sr
    print(f"[prof_up] total={dt*1000:.1f}ms  audio={audio_s:.2f}s\n", flush=True)
