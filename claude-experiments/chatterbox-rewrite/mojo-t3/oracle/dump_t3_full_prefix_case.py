"""Dump the complete T3 input-prefix fixture.

Replicates exactly what T3.prepare_input_embeds does end-to-end:
  cond_emb = T3CondEnc(t3_cond)                  # (1, 34, 1024)
  text_emb = text_emb_table[text_tokens] + text_pos_emb[arange]  # (1, T_text, 1024)
  speech_emb = speech_emb_table[start_id] + speech_pos_emb[0]    # (1, 1, 1024)
  embeds = cat([cond_emb, text_emb, speech_emb], dim=1)          # (1, 34+T_text+1, 1024)
"""
import os, struct, importlib.util
import numpy as np
import torch
import torch.nn as nn


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "t3_full_prefix")
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


spec_perc = importlib.util.spec_from_file_location(
    "p", "/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src/chatterbox/models/t3/modules/perceiver.py",
)
pmod = importlib.util.module_from_spec(spec_perc); spec_perc.loader.exec_module(pmod)

spec_lpe = importlib.util.spec_from_file_location(
    "lpe", "/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/chatterbox-rewrite/chatterbox/src/chatterbox/models/t3/modules/learned_pos_emb.py",
)
lpe_mod = importlib.util.module_from_spec(spec_lpe); spec_lpe.loader.exec_module(lpe_mod)


def main():
    torch.manual_seed(0)

    # T3 config.
    SPEAKER_EMB = 256
    D = 1024
    VOCAB_TEXT = 704
    VOCAB_SPEECH = 8194
    MAX_TEXT = 2050
    MAX_SPEECH = 4096 + 2
    COND_PROMPT_LEN = 150
    START_SPEECH = 6561
    T_TEXT = 17    # number of text tokens after BPE
    SQ = 32        # perceiver output queries

    # ---- T3CondEnc pieces ----
    spkr_enc = nn.Linear(SPEAKER_EMB, D).eval()
    emotion_fc = nn.Linear(1, D, bias=False).eval()
    perceiver = pmod.Perceiver().eval()
    speech_emb_table = nn.Embedding(VOCAB_SPEECH, D).eval()

    # ---- T3 piece: text emb and pos emb ----
    text_emb_table = nn.Embedding(VOCAB_TEXT, D).eval()
    text_pos = lpe_mod.LearnedPositionEmbeddings(MAX_TEXT, D).eval()
    speech_pos = lpe_mod.LearnedPositionEmbeddings(MAX_SPEECH, D).eval()

    # ---- Inputs ----
    speaker_emb = torch.randn(1, SPEAKER_EMB)
    cond_speech_tokens = torch.randint(0, VOCAB_SPEECH, (1, COND_PROMPT_LEN))
    emotion_adv = torch.tensor([[[0.5]]])
    text_tokens = torch.randint(0, VOCAB_TEXT, (1, T_TEXT))
    start_speech_tokens = torch.tensor([[START_SPEECH]], dtype=torch.long)   # (1, 1)

    with torch.inference_mode():
        # T3CondEnc forward.
        cond_spkr = spkr_enc(speaker_emb)[:, None]
        cond_speech_emb = speech_emb_table(cond_speech_tokens)
        cond_prompt_emb = perceiver(cond_speech_emb)
        cond_emotion = emotion_fc(emotion_adv.view(-1, 1, 1))
        cond_emb = torch.cat([cond_spkr, cond_prompt_emb, cond_emotion], dim=1)  # (1, 34, D)

        # text_emb + pos_emb.
        text_e = text_emb_table(text_tokens) + text_pos(text_tokens)             # (1, T_text, D)

        # speech_emb + pos_emb for start token.
        speech_e = speech_emb_table(start_speech_tokens) + speech_pos(start_speech_tokens)  # (1, 1, D)

        # Concat.
        embeds = torch.cat([cond_emb, text_e, speech_e], dim=1)                  # (1, 34+T_text+1, D)

    print("embeds shape:", embeds.shape)
    print("embeds[0, 0, :4]:", embeds[0, 0, :4].tolist())
    print("embeds[0, 34, :4]:", embeds[0, 34, :4].tolist())
    print("embeds[0, -1, :4]:", embeds[0, -1, :4].tolist())

    # Save inputs.
    save("speaker_emb.bin", speaker_emb)
    save("cond_speech_tokens.bin", cond_speech_tokens.float())
    save("emotion_adv.bin", emotion_adv)
    save("text_tokens.bin", text_tokens.float())
    save("start_speech_tokens.bin", start_speech_tokens.float())
    save("embeds.bin", embeds)

    # Save weights.
    save("spkr_enc_w.bin", spkr_enc.weight); save("spkr_enc_b.bin", spkr_enc.bias)
    save("emotion_fc_w.bin", emotion_fc.weight)
    save("speech_emb_w.bin", speech_emb_table.weight)
    save("text_emb_w.bin", text_emb_table.weight)
    save("text_pos_w.bin", text_pos.emb.weight)
    save("speech_pos_w.bin", speech_pos.emb.weight)
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

    with open(os.path.join(OUT_DIR, "meta.txt"), "w") as f:
        f.write(f"B=1\nT_text={T_TEXT}\nT_cond=34\nT_speech=1\nT_total={34+T_TEXT+1}\nD={D}\n")
    print("dumped to", OUT_DIR)


if __name__ == "__main__":
    main()
