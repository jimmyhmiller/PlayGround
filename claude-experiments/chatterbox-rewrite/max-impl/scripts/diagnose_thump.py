"""Render one chunk, capture mel + audio, locate thumps, inspect mel around them.

This tells us whether the thump comes from:
  (a) a discontinuity in the mel CFM produces  → CFM bug
  (b) a clean mel that HiFT decodes badly      → HiFT bug
"""
import os, sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from chatterbox_mojo import ChatterboxTTS
import chatterbox_mojo.tts as tts_mod

# Make sure we use our usual fast-path
os.environ["CHATTERBOX_BF16"] = "1"
os.environ["CHATTERBOX_CFM_STEPS"] = "5"
os.environ["CHATTERBOX_T3_FUSE_QKV"] = "1"
os.environ["CHATTERBOX_T3_FUSE_MLP"] = "1"

REF = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

# Sample text that I'd expect to produce thumps (long sentence from Quine).
CHUNKS = [
    "Conceived thus broadly, epistemology includes the study of the foundations of mathematics as one of its departments.",
    "Specialists at the turn of the century thought that their efforts in this particular department were achieving notable success.",
    "Mathematics seemed to reduce altogether to logic.",
]

# Monkey-patch _s3gen_decode to also save mel_out.
orig_decode = tts_mod.ChatterboxTTS._s3gen_decode
captured = {"mel": None, "T_out_mel": None}

def patched_decode(self, speech_tokens, cfg_rate=0.7, n_steps=0, noise_seed=0xC0FFEE):
    # Copy of the original but save the mel.
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
    # Capture the mel.
    mel_host = mel_out.to(cpu).to_numpy().reshape(80, T_OUT_MEL).copy()
    captured["mel"] = mel_host
    captured["T_out_mel"] = T_OUT_MEL

    T_HIFT = T_OUT_MEL * 120 + 1
    T_AUDIO = (T_HIFT - 1) * 4
    audio_out = Buffer(shape=(1, T_AUDIO), dtype=DType.float32, device=gpu)
    op_hift.forward(self._hift_h, mel_out, audio_out, 1, T_OUT_MEL)
    return audio_out.to(cpu).to_numpy().reshape(-1).copy()

tts_mod.ChatterboxTTS._s3gen_decode = patched_decode


def find_thumps(audio, sr=24000, thresh=0.3, cluster_ms=50):
    diff = np.abs(np.diff(audio.astype(np.float32)))
    big = np.flatnonzero(diff > thresh)
    if len(big) == 0:
        return []
    gaps = np.diff(big)
    starts = np.concatenate([[0], np.flatnonzero(gaps > sr * cluster_ms // 1000) + 1])
    events = [big[s] for s in starts]
    return events


def main():
    tts = ChatterboxTTS.from_pretrained(use_bf16=True)
    tts.prepare_conditionals(REF, exaggeration=0.5)

    for i, text in enumerate(CHUNKS):
        print(f"\n=== chunk {i}: {text!r} ===")
        for trial in range(3):
            captured["mel"] = None
            wav = tts.generate(text, cfg_weight=0.5, temperature=0.8, top_p=0.95,
                               min_p=0.05, repetition_penalty=1.2, exaggeration=0.5,
                               rng_seed=0xDEADBEEF + i * 100 + trial)
            audio = wav.squeeze(0).cpu().numpy().astype(np.float32)
            mel = captured["mel"]
            thumps = find_thumps(audio)
            print(f"  trial {trial}: audio={audio.size/24000:.2f}s mel.shape={mel.shape} thumps={len(thumps)}")
            if thumps:
                # Save for the first thumpy trial.
                t_sample = thumps[0]
                t_sec = t_sample / 24000
                # Map audio sample → mel frame. T_audio = T_out_mel * 480.
                mel_frame = int(t_sample / 480)
                pre_audio = audio[max(0, t_sample - 200):t_sample]
                post_audio = audio[t_sample:t_sample + 200]
                print(f"     first thump at sample {t_sample} (t={t_sec:.3f}s, mel_frame={mel_frame})")
                print(f"     pre-200samp peak={np.abs(pre_audio).max():.3f}  post-200samp peak={np.abs(post_audio).max():.3f}")
                print(f"     step itself: audio[{t_sample}]={audio[t_sample]:.3f} → audio[{t_sample+1}]={audio[t_sample+1]:.3f}")
                # Inspect mel around that frame.
                f0 = max(0, mel_frame - 5)
                f1 = min(mel.shape[1], mel_frame + 6)
                # Per-frame mel energy.
                frame_energy = np.linalg.norm(mel[:, f0:f1], axis=0)
                print(f"     mel frame energy around frame {mel_frame} (±5):")
                for fi, e in zip(range(f0, f1), frame_energy):
                    marker = "  *" if fi == mel_frame else ""
                    print(f"       frame {fi}: energy={e:.3f}{marker}")
                # Inter-frame mel deltas
                if f1 - f0 > 1:
                    deltas = np.linalg.norm(mel[:, f0+1:f1] - mel[:, f0:f1-1], axis=0)
                    print(f"     inter-frame mel deltas:")
                    for fi, d in zip(range(f0, f1-1), deltas):
                        marker = "  *" if fi == mel_frame else ""
                        print(f"       Δ frame {fi}→{fi+1}: {d:.3f}{marker}")
                # Save the mel + audio for offline inspection
                np.savez(f"/tmp/thump_diag_chunk{i}_trial{trial}.npz",
                         mel=mel, audio=audio, thumps=np.array(thumps),
                         text=np.array([text]))
                print(f"     saved to /tmp/thump_diag_chunk{i}_trial{trial}.npz")
                break  # only need first thumpy trial per chunk


if __name__ == "__main__":
    sys.exit(main() or 0)
