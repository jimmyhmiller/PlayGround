"""Render chunk 53 with different CFM step counts; measure mel oscillation
amplitude and thump count for each."""
import os, sys, wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["CHATTERBOX_BF16"] = "1"
os.environ["CHATTERBOX_T3_FUSE_QKV"] = "1"
os.environ["CHATTERBOX_T3_FUSE_MLP"] = "1"

import numpy as np
from chatterbox_mojo import ChatterboxTTS
import chatterbox_mojo.tts as tts_mod

REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
TEXT = ("What then could have motivated Carnap's heroic efforts on the conceptual "
        "side of epistemology, when hope of certainty on the doctrinal side was "
        "abandoned? There were two good reasons still.")

captured = {"mel": None}

def patched_decode(self, speech_tokens, cfg_rate=0.7, n_steps=0, noise_seed=0xC0FFEE):
    import os as _os
    if n_steps == 0:
        n_steps = int(_os.environ.get("CHATTERBOX_CFM_STEPS", 5))
    from max.driver import Buffer
    from max.dtype import DType
    gpu, cpu = self._gpu, self._cpu
    c = self.conds
    T_GEN = len(speech_tokens)
    T_TOTAL_TOKEN = 250 + T_GEN
    T_TOTAL_MEL = 2 * T_TOTAL_TOKEN
    T_OUT_MEL = T_TOTAL_MEL - 500
    prompt_token_host = c.prompt_token.to(cpu).to_numpy().reshape(-1)
    tok_combined = np.concatenate([
        prompt_token_host.astype(np.int64),
        np.array(speech_tokens, dtype=np.int64),
    ]).reshape(1, T_TOTAL_TOKEN)
    tok_buf_i64 = Buffer.from_numpy(tok_combined).to(gpu)
    mel_out = Buffer(shape=(1, 80, T_OUT_MEL), dtype=DType.float32, device=gpu)
    flow_config = {
        "B": 1, "T_token": T_TOTAL_TOKEN, "T_prompt_mel": 500,
        "T_total_mel": T_TOTAL_MEL, "T_out_mel": T_OUT_MEL,
        "n_steps": int(n_steps), "cfg_rate": float(cfg_rate),
        "noise_seed": int(noise_seed),
    }
    import op_flow, op_hift
    op_flow.forward(self._flow_h, tok_buf_i64, c.spks, c.prompt_feat, mel_out, flow_config)
    captured["mel"] = mel_out.to(cpu).to_numpy().reshape(80, T_OUT_MEL).copy()
    T_HIFT = T_OUT_MEL * 120 + 1
    T_AUDIO = (T_HIFT - 1) * 4
    audio_out = Buffer(shape=(1, T_AUDIO), dtype=DType.float32, device=gpu)
    op_hift.forward(self._hift_h, mel_out, audio_out, 1, T_OUT_MEL)
    return audio_out.to(cpu).to_numpy().reshape(-1).copy()

tts_mod.ChatterboxTTS._s3gen_decode = patched_decode


def stats(audio, mel):
    diff = np.abs(np.diff(audio.astype(np.float32)))
    big = np.flatnonzero(diff > 0.8)
    n_loud = len(big)
    deltas = np.linalg.norm(mel[:, 1:] - mel[:, :-1], axis=0)
    median_delta = np.median(deltas)
    p99_delta = np.percentile(deltas, 99)
    max_delta = deltas.max()
    return n_loud, median_delta, p99_delta, max_delta


def main():
    tts = ChatterboxTTS.from_pretrained(use_bf16=True)
    tts.prepare_conditionals(REF, exaggeration=0.5)

    print(f"{'cfm_steps':>10} {'audio (s)':>10} {'thumps':>7} {'med_Δmel':>10} {'p99_Δmel':>10} {'max_Δmel':>10}")
    for n in [5, 7, 10, 15, 20]:
        os.environ["CHATTERBOX_CFM_STEPS"] = str(n)
        captured["mel"] = None
        wav = tts.generate(TEXT, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                          min_p=0.05, repetition_penalty=1.2, exaggeration=0.5,
                          rng_seed=0xDEADBEEF + 53)
        audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
        mel = captured["mel"]
        n_loud, med, p99, mx = stats(audio, mel)
        print(f"{n:>10} {audio.size/24000:>10.2f} {n_loud:>7} {med:>10.2f} {p99:>10.2f} {mx:>10.2f}")
        # Save audio for each.
        pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(f"/tmp/loud_thumps/chunk53_cfm{n}.wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(pcm16.tobytes())


if __name__ == "__main__":
    sys.exit(main() or 0)
