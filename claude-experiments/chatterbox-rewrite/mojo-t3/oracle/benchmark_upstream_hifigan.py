"""
Time upstream HiFTGenerator.inference on the same input we ran through Mojo,
so we have an apples-to-apples comparison.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

# Load HiFTGenerator from our vendored source (no chatterbox-tts install needed).
VENDORED = Path("../chatterbox/src/chatterbox/models/s3gen").resolve()
import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load f0_predictor first since hifigan imports from it via class_utils etc.
_load("scipy_get_window_stub", VENDORED / "f0_predictor.py")  # ensure deps loaded
hift_mod = _load("hifigan_src", VENDORED / "hifigan.py")
f0_mod = _load("f0_predictor_src", VENDORED / "f0_predictor.py")
HiFTGenerator = hift_mod.HiFTGenerator
ConvRNNF0Predictor = f0_mod.ConvRNNF0Predictor


def build_and_load():
    f0 = ConvRNNF0Predictor()
    hift = HiFTGenerator(
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0,
    )
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    ckpt = hf_hub_download(repo_id="ResembleAI/chatterbox", filename="s3gen.safetensors")
    state = load_file(ckpt)
    hift_state = {k[len("mel2wav."):]: v for k, v in state.items() if k.startswith("mel2wav.")}
    hift.load_state_dict(hift_state, strict=False)
    import torch.nn.utils.parametrize as parametrize
    def _strip(mod):
        if hasattr(mod, "parametrizations"):
            for n in list(mod.parametrizations.keys()):
                parametrize.remove_parametrizations(mod, n, leave_parametrized=True)
        for c in mod.children():
            _strip(c)
    _strip(hift)
    return hift.cuda().eval()


import struct
def read_fp32(p):
    with open(p, "rb") as f:
        rank = struct.unpack("<q", f.read(8))[0]
        sh = struct.unpack(f"<{rank}q", f.read(8 * rank))
        tag = struct.unpack("<i", f.read(4))[0]
        return np.frombuffer(f.read(), dtype=np.float32).reshape(sh)


def main():
    hift = build_and_load()
    mel = torch.from_numpy(read_fp32("tests/fixtures/real/real_mel.bin")).cuda()
    print(f"mel shape: {tuple(mel.shape)}")

    # Warmup.
    with torch.inference_mode():
        _ = hift.inference(speech_feat=mel)
    torch.cuda.synchronize()

    # Benchmark N runs.
    N = 5
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N):
            wav, _ = hift.inference(speech_feat=mel)
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    per_call = (t1 - t0) / N
    print(f"upstream torch HiFTGenerator: {per_call*1000:.0f} ms/call ({N} runs)")
    print(f"  wav shape: {tuple(wav.shape)}  duration={wav.shape[1]/24000:.2f}s")
    print(f"  realtime factor: {wav.shape[1]/24000/per_call:.2f}x")


if __name__ == "__main__":
    main()
