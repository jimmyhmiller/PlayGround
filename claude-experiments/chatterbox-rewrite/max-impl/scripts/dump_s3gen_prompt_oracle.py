"""Dump upstream s3gen prompt-side ingredients + the full mel produced
by 10-step CFM inside `flow.inference`. Used by Mojo to wire the prompt
prefix into the flow encoder / CFM chain, matching what s3gen was trained
to expect.

Writes:
  s3gen_prompt/prompt_token.bin       (1, T_prompt_token)        int64
  s3gen_prompt/prompt_token_len.bin   (1,)                       int64
  s3gen_prompt/prompt_feat.bin        (1, T_prompt_feat, 80)     float32
  s3gen_prompt/prompt_feat_len.bin    (1,)                       int64  (may be -1 sentinel)
  s3gen_prompt/embedding.bin          (1, 192)                   float32
  s3gen_prompt/embedding_normed_affine.bin  (1, 80)              float32  (after F.normalize + spk_embed_affine)
  s3gen_prompt/expected_mel.bin       (1, 80, T_mel_out)         float32  (post-CFM, post-trim)
  s3gen_prompt/expected_audio.bin     (1, T_audio)               float32  (post-HiFT)
  s3gen_prompt/text.txt               the text used
"""
import os, struct
import numpy as np
import torch
import torch.nn.functional as F


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def write_i64(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.int64))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 2))
        f.write(arr.tobytes())


def main():
    import random
    from chatterbox.tts import ChatterboxTTS

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    OUT = "weights/s3gen_prompt"
    os.makedirs(OUT, exist_ok=True)

    text = os.environ.get(
        "DUMP_TEXT",
        "She sells seashells by the seashore. The waves crashed against the rocky cliffs.",
    )
    ref_path = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"

    print("Loading Chatterbox...")
    m = ChatterboxTTS.from_pretrained("cpu")
    m.prepare_conditionals(ref_path)
    s3gen_ref_dict = m.conds.gen
    print("s3gen_ref keys:", list(s3gen_ref_dict.keys()))
    for k, v in s3gen_ref_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Dump the prompt-side artifacts.
    pt = s3gen_ref_dict["prompt_token"]
    ptl = s3gen_ref_dict["prompt_token_len"]
    pf = s3gen_ref_dict["prompt_feat"]
    pfl = s3gen_ref_dict.get("prompt_feat_len", None)
    emb = s3gen_ref_dict["embedding"]
    write_i64(f"{OUT}/prompt_token.bin", pt.detach().cpu().numpy())
    write_i64(f"{OUT}/prompt_token_len.bin", ptl.detach().cpu().numpy() if isinstance(ptl, torch.Tensor) else np.array([int(ptl)]))
    write_tensor(f"{OUT}/prompt_feat.bin", pf.detach().cpu().numpy())
    if pfl is None:
        write_i64(f"{OUT}/prompt_feat_len.bin", np.array([-1], dtype=np.int64))
    else:
        arr = pfl.detach().cpu().numpy() if isinstance(pfl, torch.Tensor) else np.array([int(pfl)])
        write_i64(f"{OUT}/prompt_feat_len.bin", arr)
    write_tensor(f"{OUT}/embedding.bin", emb.detach().cpu().numpy())

    # Also dump the post-normalize/affine 80-d vector (what flow.inference uses internally as spks).
    embedding_normed = F.normalize(torch.atleast_2d(emb), dim=1)
    embedding_normed_affine = m.s3gen.flow.spk_embed_affine_layer(embedding_normed)
    print(f"  embedding_normed_affine shape: {embedding_normed_affine.shape}")
    write_tensor(f"{OUT}/embedding_normed_affine.bin", embedding_normed_affine.detach().cpu().numpy())

    # Now generate actual TTS via upstream end-to-end and capture intermediates.
    # We need: the speech_tokens that T3 produced, AND the mel that CFM produced
    # (so Mojo can validate just-the-CFM-chain against torch).
    print("Generating audio via upstream end-to-end...")
    captures = {}
    def cap(name):
        def hook(mod, inputs, output):
            captures[name] = output
        return hook
    h_decoder = m.s3gen.flow.decoder.register_forward_hook(cap("cfm_velocity_estimator_last_output"))
    # The decoder hook captures the LAST velocity field — not super useful.
    # Better: hook `flow_inference` output indirectly by monkey-patching s3gen.flow.inference
    # to capture mel before HiFT.

    orig_flow_inference = m.s3gen.flow_inference
    captured_mel = {}
    captured_tokens = {}
    captured_noise = {}
    def my_flow_inference(speech_tokens, *args, **kwargs):
        captured_tokens["speech_tokens"] = speech_tokens.detach().clone()
        out = orig_flow_inference(speech_tokens, *args, **kwargs)
        captured_mel["mel"] = out.detach().clone()
        return out
    m.s3gen.flow_inference = my_flow_inference

    # Monkey-patch torch.randn_like inside CausalConditionalCFM.forward so we capture
    # the exact noise tensor the upstream CFM sees.
    import torch as _torch
    _orig_randn_like = _torch.randn_like
    def _capturing_randn_like(t, *a, **kw):
        z = _orig_randn_like(t, *a, **kw)
        # Capture only the s3gen CFM call: it's (1, 80, T_mel_total).
        if z.dim() == 3 and z.shape[1] == 80 and "z" not in captured_noise:
            captured_noise["z"] = z.detach().clone()
        return z
    _torch.randn_like = _capturing_randn_like

    wav = m.generate(text, audio_prompt_path=ref_path, exaggeration=0.5, cfg_weight=0.5, temperature=0.8)
    if isinstance(wav, tuple): wav = wav[0]
    wav = wav.detach().cpu()
    print(f"audio: shape={wav.shape}  max-abs={wav.abs().max().item():.3f}  mean-abs={wav.abs().mean().item():.4f}")

    mel = captured_mel["mel"]
    print(f"CFM mel (post-trim): shape={mel.shape}  mean-abs={mel.abs().mean().item():.3f}")

    h_decoder.remove()
    _torch.randn_like = _orig_randn_like

    write_tensor(f"{OUT}/expected_mel.bin", mel.numpy())
    write_tensor(f"{OUT}/expected_audio.bin", wav.numpy())
    write_i64(f"{OUT}/speech_tokens.bin", captured_tokens["speech_tokens"].detach().cpu().numpy())
    if "z" in captured_noise:
        z = captured_noise["z"]
        print(f"CFM noise z: shape={z.shape}  mean={z.mean().item():.4f}  std={z.std().item():.4f}")
        write_tensor(f"{OUT}/cfm_noise_z.bin", z.detach().cpu().numpy())
    else:
        print("WARNING: did not capture CFM noise z")
    st = captured_tokens["speech_tokens"]
    print(f"speech_tokens (input to flow_inference): shape={st.shape}")
    with open(f"{OUT}/text.txt", "w") as f:
        f.write(text + "\n")

    # Save the audio as a WAV so we can listen to upstream's output for the
    # same prompt+text combo (apples-to-apples comparison vs our Mojo output).
    # Use scipy.io.wavfile rather than torchaudio (which now requires torchcodec).
    from scipy.io import wavfile
    pcm = (wav.squeeze(0).numpy() * 32767.0).clip(-32768, 32767).astype(np.int16)
    wavfile.write(f"{OUT}/upstream.wav", 24000, pcm)
    print(f"wrote upstream.wav  ({pcm.shape[0]} samples = {pcm.shape[0]/24000:.2f}s)")
    print(f"all written to {OUT}/")


if __name__ == "__main__":
    main()
