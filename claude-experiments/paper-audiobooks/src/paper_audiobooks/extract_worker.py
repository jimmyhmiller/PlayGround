"""Subprocess worker for marker PDF extraction.

Reads JSON {"pdf_path": str, "out_path": str} on stdin, writes the extracted
markdown to out_path, prints "ok" on success.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# Reduce HIP allocator fragmentation. Strix Halo unified memory shares VRAM
# with system RAM and other GPU consumers (llama-server on Vulkan, any other
# torch process), so the allocator's default behaviour leaves big holes that
# OOM marker's batched conv kernels even when total free memory is plenty.
# expandable_segments grows allocations dynamically instead of reserving fixed
# chunks. The OOM error message itself recommends this exact setting.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> int:
    try:
        req = json.loads(sys.stdin.read())
        from paper_audiobooks.extract import extract_markdown

        markdown = extract_markdown(Path(req["source_path"]), page_range=req.get("page_range"))
        out = Path(req["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        print("ok")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
