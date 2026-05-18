"""Convert upstream Chatterbox safetensors checkpoints into max-impl binaries.

Reuses the dump-fixture format (header: rank | shape | dtype tag | payload)
that `src/fixture.mojo` reads. Maps upstream weight names to the names our
Mojo layers expect.

This is the non-NN "easy reuse" allowed by the goal — pure Python I/O,
no neural network ops.
"""
import os, struct, sys
import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("Need: pip install safetensors", file=sys.stderr)
    sys.exit(1)


def write_tensor(path, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", 0))
        f.write(arr.tobytes())


def convert(in_safetensors, out_dir, key_map):
    """Convert weights from safetensors into out_dir, applying key_map for renames."""
    with safe_open(in_safetensors, framework="pt") as f:
        for upstream_key in f.keys():
            arr = f.get_tensor(upstream_key).cpu().numpy()
            mapped = key_map(upstream_key)
            if mapped is None:
                continue
            out_path = os.path.join(out_dir, mapped + ".bin")
            write_tensor(out_path, arr)
            print(f"  {upstream_key} → {mapped} {tuple(arr.shape)}")


def voice_encoder_map(k):
    if k.startswith("lstm.weight_ih_l"):
        return "ve/" + k.split(".")[-1]
    elif k.startswith("lstm.weight_hh_l"):
        return "ve/" + k.split(".")[-1]
    elif k.startswith("lstm.bias_ih_l"):
        return "ve/" + k.split(".")[-1]
    elif k.startswith("lstm.bias_hh_l"):
        return "ve/" + k.split(".")[-1]
    elif k == "proj.weight":
        return "ve/proj_w"
    elif k == "proj.bias":
        return "ve/proj_b"
    return None


# Add more mappings (s3tokenizer, T3 backbone, s3gen) as we run end-to-end.


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_weights.py <in_safetensors> <out_dir>", file=sys.stderr)
        sys.exit(2)
    convert(sys.argv[1], sys.argv[2], voice_encoder_map)
