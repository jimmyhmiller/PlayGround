"""Compute transitive Mojo deps and symlink them into an op's directory.

Usage: python scripts/setup_op_deps.py <op_dir> <entry_module1> [entry_module2 ...]

The entry modules are the Mojo files (without .mojo) that the op_*.mojo
imports from. Their transitive imports of local src/*.mojo files are
chased and symlinked.

Excludes 'weights' (it imports too much). Each op must inline its own
needed loaders into a weights_<op>.mojo (see ops/op_campplus/ for the
canonical example).
"""
import os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
LOCAL_MODS = {f.stem for f in SRC.glob("*.mojo")}
EXCLUDED = {"weights"}  # too big — each op gets its own slimmed loader


def imports_of(mojo_path: Path) -> set[str]:
    out = set()
    with open(mojo_path) as f:
        for line in f:
            m = re.match(r"(?:from|import)\s+([a-z_][a-z_0-9]*)", line)
            if m and m.group(1) in LOCAL_MODS and m.group(1) not in EXCLUDED:
                out.add(m.group(1))
    return out


def closure(seeds: list[str]) -> set[str]:
    seen, queue = set(), list(seeds)
    while queue:
        cur = queue.pop()
        if cur in seen or cur in EXCLUDED:
            continue
        seen.add(cur)
        if (SRC / f"{cur}.mojo").exists():
            queue.extend(imports_of(SRC / f"{cur}.mojo"))
    return seen


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    op_dir = ROOT / sys.argv[1]
    seeds = sys.argv[2:]
    deps = closure(seeds)
    print(f"[setup_op_deps] {op_dir.name}: {len(deps)} deps")
    for d in sorted(deps):
        target = op_dir / f"{d}.mojo"
        # Remove existing symlink/file (don't overwrite real files in the op dir).
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            print(f"  SKIP {d}.mojo (real file exists, not symlinking)")
            continue
        src = SRC / f"{d}.mojo"
        rel = os.path.relpath(src, op_dir)
        target.symlink_to(rel)
        print(f"  link {d}.mojo -> {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
