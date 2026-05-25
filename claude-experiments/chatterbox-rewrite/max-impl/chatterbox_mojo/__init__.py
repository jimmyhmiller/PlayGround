"""Chatterbox Mojo Python orchestrator.

Loads Mojo .so ops via mojo.importer and chains them on GPU. Python performs
ZERO compute — only orchestration (loop counters, buffer allocation, op call
sequencing). All math lives in the Mojo .so files under ../ops/.
"""
import os
import sys
from pathlib import Path

# Make the ops/ directory importable. Each subdir contains one Mojo source
# file; mojo.importer compiles it to a .so on first import and caches it.
_OPS_ROOT = Path(__file__).resolve().parent.parent / "ops"
for op_dir in _OPS_ROOT.iterdir():
    if op_dir.is_dir() and (op_dir / f"{op_dir.name}.mojo").exists():
        sys.path.insert(0, str(op_dir))


def __getattr__(name):
    if name in ("ChatterboxTTS", "Conditionals", "punc_norm"):
        from . import tts
        return getattr(tts, name)
    raise AttributeError(name)
