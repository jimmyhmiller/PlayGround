"""
Capture every intermediate from a real S3Gen flow.inference() call so we have
parity targets for the Mojo port at every sub-stage.

Run with the paper-audiobooks chatterbox venv:
  /home/jimmyhmiller/.cache/paper-audiobooks/venvs/chatterbox/bin/python \
      oracle/dump_s3gen_intermediates.py

Captures (all saved to tests/fixtures/s3gen/):
  speech_tokens.bin           (1, T_tok)  int64   input speech tokens after T3
  embedding_raw.bin           (1, 192)            xvector (CAMPPlus output)
  embedding_normed.bin        (1, 192)            after F.normalize
  embedding_affine.bin        (1, 80)             after spk_embed_affine_layer

  flow_token_in.bin           (1, T_full)         token (after concat with prompt_token)
  flow_token_emb.bin          (1, T_full, 512)    after input_embedding

  encoder_h.bin               (1, T_h, 512)       encoder output
  encoder_proj_h.bin          (1, T_h, 80)        after encoder_proj

  conds.bin                   (1, T_h, 80)        the prompt-conditioning tensor

  cfm_z_init.bin              (1, 80, T_h)        initial noise z (random; we record what was sampled)
  cfm_step_0.bin..N.bin       per Euler step      z after each ODE step
  cfm_mel_out.bin             (1, 80, T_h)        final mel (also = decoder output)

  # Weights for each module (so Mojo can load them).
  spk_embed_affine_layer.{weight,bias}.bin
  flow.input_embedding.weight.bin
  flow.encoder_proj.{weight,bias}.bin
  # plus the full encoder + decoder state_dicts in their own subdirs.
"""
from __future__ import annotations

import struct
import os
from pathlib import Path

import numpy as np
import torch

# Match paper-audiobooks env.
os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

from chatterbox.tts import ChatterboxTTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "s3gen"
OUT.mkdir(parents=True, exist_ok=True)
W_DIR = OUT / "weights"
W_DIR.mkdir(parents=True, exist_ok=True)

REF_VOICE = Path("/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav")
PROMPT = "Hello, this is a Mojo-based Chatterbox vocoder running on AMD."


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag, raw = 0, arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag, raw = 1, arr.astype(np.uint16, copy=False).tobytes()
    elif arr.dtype == np.int64:
        tag, raw = 2, arr.astype(np.int64, copy=False).tobytes()
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def save_module_state(prefix: str, mod):
    n = 0
    for k, v in mod.state_dict().items():
        fname = (prefix + "." + k).replace(".", "__") + ".bin"
        write_tensor(W_DIR / fname, v.detach().cpu().numpy().astype(np.float32))
        n += 1
    return n


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[s3gen] device={device}")
    model = ChatterboxTTS.from_pretrained(device=device)
    s3gen = model.s3gen

    # Strip weight_norm parametrizations everywhere.
    import torch.nn.utils.parametrize as parametrize
    def _strip(mod):
        if hasattr(mod, "parametrizations"):
            for name in list(mod.parametrizations.keys()):
                parametrize.remove_parametrizations(mod, name, leave_parametrized=True)
        for child in mod.children():
            _strip(child)
    _strip(s3gen)

    # Hook into flow.inference to capture intermediates.
    flow = s3gen.flow
    captured = {}

    orig_emb = flow.input_embedding.forward
    def hook_emb(token):
        out = orig_emb(token)
        captured["flow_token_emb"] = out.detach()
        captured["flow_token_in"] = token.detach()
        return out
    flow.input_embedding.forward = hook_emb

    orig_enc = flow.encoder.forward
    def hook_enc(*args, **kwargs):
        out = orig_enc(*args, **kwargs)
        # encoder returns (h, h_masks). h shape: (B, T_h, C).
        if isinstance(out, tuple):
            captured["encoder_h"] = out[0].detach()
        return out
    flow.encoder.forward = hook_enc

    orig_proj = flow.encoder_proj.forward
    def hook_proj(x):
        out = orig_proj(x)
        captured["encoder_proj_h"] = out.detach()
        return out
    flow.encoder_proj.forward = hook_proj

    # Capture the decoder (CFM) input + each Euler step output.
    cfm = flow.decoder
    orig_solve = cfm.solve_euler
    def hook_solve(x, t_span, mu, mask, spks, cond, *args, **kwargs):
        captured["cfm_z_init"] = x.detach()
        captured["cfm_mu"] = mu.detach()
        captured["cfm_mask"] = mask.detach()
        captured["cfm_spks"] = spks.detach()
        captured["cfm_cond"] = cond.detach()
        captured["cfm_t_span"] = t_span.detach()
        steps = []
        step_inputs = []   # (x, mask, mu, t, spks, cond) per step
        step_outputs = []  # estimator output per step (the dxdt before CFG split)
        orig_forward = cfm.estimator.forward
        def estimator_forward(*fargs, **fkwargs):
            inputs = {}
            if fargs:
                steps.append(fargs[0].detach())
            elif "x" in fkwargs:
                steps.append(fkwargs["x"].detach())
            # Snapshot ALL named inputs so we can replay the estimator
            # against this exact state later.
            for name in ("x", "mask", "mu", "t", "spks", "cond"):
                if name in fkwargs and fkwargs[name] is not None:
                    inputs[name] = fkwargs[name].detach()
            step_inputs.append(inputs)
            out = orig_forward(*fargs, **fkwargs)
            step_outputs.append(out.detach())
            return out
        cfm.estimator.forward = estimator_forward
        try:
            result = orig_solve(x, t_span, mu, mask, spks, cond, *args, **kwargs)
        finally:
            cfm.estimator.forward = orig_forward
        captured["cfm_step_zs"] = steps
        captured["cfm_step_inputs"] = step_inputs   # list of dicts
        captured["cfm_step_outputs"] = step_outputs # list of estimator outputs
        captured["cfm_mel_out"] = result.detach() if isinstance(result, torch.Tensor) else None
        return result
    cfm.solve_euler = hook_solve

    # Also capture the spk_embed_affine_layer output.
    orig_spk = flow.spk_embed_affine_layer.forward
    def hook_spk(x):
        captured["embedding_normed"] = x.detach()
        out = orig_spk(x)
        captured["embedding_affine"] = out.detach()
        return out
    flow.spk_embed_affine_layer.forward = hook_spk

    # Capture s3gen.embed_ref output to grab the raw xvector embedding.
    orig_embed_ref = s3gen.embed_ref
    def hook_embed_ref(*args, **kwargs):
        ref_dict = orig_embed_ref(*args, **kwargs)
        captured["embedding_raw"] = ref_dict["embedding"].detach()
        captured["prompt_token"] = ref_dict["prompt_token"].detach()
        captured["prompt_feat"] = ref_dict["prompt_feat"].detach()
        return ref_dict
    s3gen.embed_ref = hook_embed_ref

    # Also capture the speech_tokens that T3 produces.
    orig_t3_inf = model.t3.inference
    def hook_t3(*args, **kwargs):
        tokens = orig_t3_inf(*args, **kwargs)
        captured["speech_tokens"] = tokens.detach()
        return tokens
    model.t3.inference = hook_t3

    with torch.inference_mode():
        wav = model.generate(text=PROMPT, audio_prompt_path=str(REF_VOICE))
    print(f"[s3gen] wav: {tuple(wav.shape)}")

    # Save captured intermediates.
    for k, v in captured.items():
        if k == "cfm_step_zs":
            for i, z in enumerate(v):
                write_tensor(OUT / f"cfm_step_{i:02d}.bin",
                             z.cpu().numpy().astype(np.float32))
            print(f"  cfm_step_zs: {len(v)} steps, shape {tuple(v[0].shape)}")
        elif k == "cfm_step_inputs":
            for i, inp in enumerate(v):
                for name, tensor in inp.items():
                    write_tensor(OUT / f"cfm_step_{i:02d}_input_{name}.bin",
                                 tensor.cpu().numpy().astype(np.float32))
            print(f"  cfm_step_inputs: {len(v)} steps with named inputs")
        elif k == "cfm_step_outputs":
            for i, out in enumerate(v):
                write_tensor(OUT / f"cfm_step_{i:02d}_output.bin",
                             out.cpu().numpy().astype(np.float32))
            print(f"  cfm_step_outputs: {len(v)} steps, shape {tuple(v[0].shape)}")
        elif isinstance(v, torch.Tensor):
            dtype = v.dtype
            if dtype == torch.int64 or dtype == torch.int32:
                arr = v.cpu().numpy().astype(np.int64)
            else:
                arr = v.cpu().numpy().astype(np.float32)
            write_tensor(OUT / (k + ".bin"), arr)
            print(f"  {k}: {tuple(v.shape)} {dtype}")

    # Save module weights for the components Mojo needs.
    print("[s3gen] saving module weights...")
    n_aff = save_module_state("spk_embed_affine_layer", flow.spk_embed_affine_layer)
    n_inp = save_module_state("flow.input_embedding", flow.input_embedding)
    n_proj = save_module_state("flow.encoder_proj", flow.encoder_proj)
    n_enc = save_module_state("flow.encoder", flow.encoder)
    n_dec = save_module_state("flow.decoder", flow.decoder)
    print(f"  affine={n_aff}  input_emb={n_inp}  proj={n_proj}  encoder={n_enc}  decoder={n_dec}")

    print("[s3gen] DONE")


if __name__ == "__main__":
    main()
