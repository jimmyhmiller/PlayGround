"""
HiFiGAN (HiFTGenerator) parity case using the real Chatterbox s3gen checkpoint.

We can't `import chatterbox` here (it has heavy deps we haven't pinned in
pixi). Instead we add the vendored source directory to sys.path and import
the HiFTGenerator class directly from there.

Loads the s3gen checkpoint, builds a HiFTGenerator with the production config,
calls `remove_weight_norm()` to materialize plain conv weights, then runs
`inference()` on a small synthetic mel input to produce the reference waveform.

We dump:
  mel.bin              (1, 80, T_MEL) input
  expected_wav.bin     (1, T_WAV)    upstream HiFTGenerator output
  (lots of weight tensors, named after the upstream attribute path)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# We don't want to evaluate chatterbox/__init__.py (it requires the package
# to be pip-installed). Load the specific source files directly with
# importlib.util so we get just the HiFiGAN class.
VENDORED = ROOT.parent / "chatterbox" / "src" / "chatterbox" / "models" / "s3gen"
import importlib.util


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_hifigan_mod = _load_module("hifigan_src", VENDORED / "hifigan.py")
_f0_mod = _load_module("f0_predictor_src", VENDORED / "f0_predictor.py")
HiFTGenerator = _hifigan_mod.HiFTGenerator
ConvRNNF0Predictor = _f0_mod.ConvRNNF0Predictor

OUT = ROOT / "tests" / "fixtures" / "hifigan"
OUT.mkdir(parents=True, exist_ok=True)

S3GEN_SR = 24000


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


def build_hifigan() -> HiFTGenerator:
    f0 = ConvRNNF0Predictor()
    return HiFTGenerator(
        sampling_rate=S3GEN_SR,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0,
    )


def load_checkpoint(hift: HiFTGenerator) -> None:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # The full s3gen checkpoint has keys like "mel2wav.<...>" — HiFTGenerator
    # lives under the `mel2wav` attribute on S3Token2Wav. We grab just that
    # prefix.
    ckpt_path = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="s3gen.safetensors")
    state = load_file(ckpt_path)
    # Filter and strip prefix.
    hift_state = {}
    for k, v in state.items():
        if k.startswith("mel2wav."):
            hift_state[k[len("mel2wav."):]] = v
    if not hift_state:
        raise RuntimeError("no mel2wav.* keys found in s3gen checkpoint")
    missing, unexpected = hift.load_state_dict(hift_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:5]}...")
    # Some bookkeeping keys (like buffers) may legitimately be missing.
    if missing:
        # Surface a warning but don't crash — common for stft_window etc.
        bad = [k for k in missing if "stft_window" not in k and "trim_fade" not in k]
        if bad:
            print(f"WARNING missing keys: {bad[:5]}...")


def main() -> None:
    torch.manual_seed(0)
    hift = build_hifigan().eval()
    load_checkpoint(hift)
    # Newer torch uses the parametrize API; upstream's remove_weight_norm()
    # relies on the legacy hook API. Bypass and call the parametrize remove
    # ourselves on every conv/conv-transpose that has parametrizations.
    import torch.nn.utils.parametrize as parametrize

    def _strip_all(mod):
        if hasattr(mod, "parametrizations"):
            for name in list(mod.parametrizations.keys()):
                parametrize.remove_parametrizations(mod, name, leave_parametrized=True)
        for child in mod.children():
            _strip_all(child)

    _strip_all(hift)

    # Small synthetic mel input — 80 channels (matches s3gen mel dim), modest length.
    T_MEL = 32
    g = torch.Generator().manual_seed(0xC0FFEE)
    mel = torch.randn(1, 80, T_MEL, generator=g, dtype=torch.float32) * 0.5

    with torch.inference_mode():
        wav, _ = hift.inference(speech_feat=mel)
    print(f"[hifigan] mel{tuple(mel.shape)} -> wav{tuple(wav.shape)}")

    write_tensor(OUT / "mel.bin", mel.numpy().astype(np.float32))
    write_tensor(OUT / "expected_wav.bin", wav.numpy().astype(np.float32))
    write_tensor(OUT / "meta.bin", np.array(
        [1, 80, T_MEL, wav.shape[1]], dtype=np.int64))

    # Dump the materialized state_dict so a future Mojo driver can pick up
    # individual layers by name.
    sd = hift.state_dict()
    print(f"[hifigan] dumping {len(sd)} weight tensors")
    for k, v in sd.items():
        # Map dotted keys to filenames; replace dots with __ for cleanliness.
        fname = k.replace(".", "__") + ".bin"
        write_tensor(OUT / "weights" / fname, v.cpu().numpy().astype(np.float32))

    # Also dump a listing.
    with (OUT / "weights_manifest.txt").open("w") as f:
        for k in sorted(sd.keys()):
            shape = tuple(sd[k].shape)
            f.write(f"{k}\t{shape}\n")
    print(f"[hifigan] manifest written")


if __name__ == "__main__":
    main()
