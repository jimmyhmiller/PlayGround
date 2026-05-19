"""Dump T3CondEnc inputs (speaker_emb, cond_prompt_tokens, emotion_adv) and output cond_emb."""
import os, struct
import numpy as np
import torch


def write_tensor(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def write_i64(path, arr):
    arr = np.ascontiguousarray(arr.astype(np.int64))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 2))
        f.write(arr.tobytes())


def main():
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    m = ChatterboxTTS.from_pretrained("cpu")
    m.prepare_conditionals("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav")
    t3_cond = m.conds.t3
    t3 = m.t3
    t3.eval()

    OUT = "weights/s3gen_prompt/cond_enc_diag"
    os.makedirs(OUT, exist_ok=True)

    print("t3_cond:")
    print(f"  speaker_emb: shape={t3_cond.speaker_emb.shape} mean-abs={t3_cond.speaker_emb.abs().mean().item():.4f}")
    print(f"  cond_prompt_speech_tokens: shape={t3_cond.cond_prompt_speech_tokens.shape}")
    print(f"  emotion_adv: shape={t3_cond.emotion_adv.shape} value={t3_cond.emotion_adv.flatten()[0].item()}")

    write_tensor(f"{OUT}/speaker_emb.bin", t3_cond.speaker_emb.cpu().numpy())
    write_i64(f"{OUT}/cond_prompt_speech_tokens.bin", t3_cond.cond_prompt_speech_tokens.cpu().numpy())
    write_tensor(f"{OUT}/emotion_adv.bin", t3_cond.emotion_adv.cpu().numpy())

    with torch.inference_mode():
        cond_emb = t3.prepare_conditioning(t3_cond)
        # Also dump intermediates.
        # 1. speech_emb(cond_tokens) + speech_pos_emb(cond_tokens)
        cs_emb_only = t3.speech_emb(t3_cond.cond_prompt_speech_tokens)
        cs_emb_with_pos = cs_emb_only + t3.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        write_tensor(f"{OUT}/cs_emb_only.bin", cs_emb_only.cpu().numpy())
        write_tensor(f"{OUT}/cs_emb_with_pos.bin", cs_emb_with_pos.cpu().numpy())
        # 2. Perceiver output.
        perc_out = t3.cond_enc.perceiver(cs_emb_with_pos)
        write_tensor(f"{OUT}/perceiver_out.bin", perc_out.cpu().numpy())
        # 3. Speaker proj.
        spk_proj = t3.cond_enc.spkr_enc(t3_cond.speaker_emb)
        write_tensor(f"{OUT}/spkr_proj.bin", spk_proj.cpu().numpy())
        # 4. Emotion proj.
        emo_proj = t3.cond_enc.emotion_adv_fc(t3_cond.emotion_adv.view(-1, 1, 1))
        write_tensor(f"{OUT}/emo_proj.bin", emo_proj.cpu().numpy())
        print(f"  cs_emb_only: shape={cs_emb_only.shape}")
        print(f"  cs_emb_with_pos: shape={cs_emb_with_pos.shape}")
        print(f"  perceiver_out: shape={perc_out.shape}")
        print(f"  spkr_proj: shape={spk_proj.shape}")
        print(f"  emo_proj: shape={emo_proj.shape}")

    print(f"cond_emb: shape={cond_emb.shape} mean-abs={cond_emb.abs().mean().item():.4f}")
    write_tensor(f"{OUT}/cond_emb.bin", cond_emb.cpu().numpy())

    print(f"Wrote to {OUT}/")


if __name__ == "__main__":
    main()
