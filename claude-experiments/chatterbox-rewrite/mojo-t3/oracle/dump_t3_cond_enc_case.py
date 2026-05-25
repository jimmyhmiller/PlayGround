"""Dump T3CondEnc fixture: speaker_emb + speech tokens → cond_emb."""
import os, struct, sys, importlib.util
import numpy as np
import torch
import torch.nn as nn


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "t3_cond_enc")
os.makedirs(OUT_DIR, exist_ok=True)


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def save(name, t):
    if isinstance(t, torch.Tensor): t = t.detach().cpu().numpy()
    write_tensor(os.path.join(OUT_DIR, name), t)


def main():
    spec = importlib.util.spec_from_file_location(
        "p", "/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src/chatterbox/models/t3/modules/perceiver.py",
    )
    pmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(pmod)

    torch.manual_seed(0)
    # T3 config (real values from t3_config.py).
    SPEAKER_EMB = 256
    N_CHANNELS = 1024
    COND_PROMPT_LEN = 150
    SPEECH_VOCAB = 8194

    # Build pieces.
    spkr_enc = nn.Linear(SPEAKER_EMB, N_CHANNELS).eval()
    emotion_fc = nn.Linear(1, N_CHANNELS, bias=False).eval()
    perceiver = pmod.Perceiver().eval()    # default: 32 queries × 1024
    speech_emb_table = nn.Embedding(SPEECH_VOCAB, N_CHANNELS).eval()

    # Inputs.
    speaker_emb = torch.randn(1, SPEAKER_EMB)
    cond_speech_tokens = torch.randint(0, SPEECH_VOCAB, (1, COND_PROMPT_LEN))
    emotion_adv = torch.tensor([[[0.5]]])    # (1, 1, 1)

    # Forward.
    with torch.inference_mode():
        cond_spkr = spkr_enc(speaker_emb)[:, None]    # (1, 1, 1024)
        cond_speech_emb = speech_emb_table(cond_speech_tokens)   # (1, 150, 1024)
        cond_prompt_emb = perceiver(cond_speech_emb)  # (1, 32, 1024)
        cond_emotion = emotion_fc(emotion_adv.view(-1, 1, 1))    # (1, 1, 1024)
        # Concat speaker | clap (empty) | prompt_emb | emotion → (1, 1+0+32+1, 1024) = (1, 34, 1024).
        cond_emb = torch.cat([cond_spkr, cond_prompt_emb, cond_emotion], dim=1)

    print("cond_emb shape:", cond_emb.shape)
    print("cond_emb[0, 0, :4]:", cond_emb[0, 0, :4].tolist())

    save("speaker_emb.bin", speaker_emb)
    save("cond_speech_tokens.bin", cond_speech_tokens.float())
    save("emotion_adv.bin", emotion_adv)
    save("cond_emb.bin", cond_emb)
    save("spkr_enc_w.bin", spkr_enc.weight)
    save("spkr_enc_b.bin", spkr_enc.bias)
    save("emotion_fc_w.bin", emotion_fc.weight)
    save("speech_emb_w.bin", speech_emb_table.weight)
    save("pre_attention_query.bin", perceiver.pre_attention_query)
    save("attn_norm_w.bin", perceiver.attn.norm.weight)
    save("attn_norm_b.bin", perceiver.attn.norm.bias)
    save("to_q_w.bin", perceiver.attn.to_q.weight)
    save("to_q_b.bin", perceiver.attn.to_q.bias)
    save("to_k_w.bin", perceiver.attn.to_k.weight)
    save("to_k_b.bin", perceiver.attn.to_k.bias)
    save("to_v_w.bin", perceiver.attn.to_v.weight)
    save("to_v_b.bin", perceiver.attn.to_v.bias)
    save("proj_out_w.bin", perceiver.attn.proj_out.weight)
    save("proj_out_b.bin", perceiver.attn.proj_out.bias)
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
