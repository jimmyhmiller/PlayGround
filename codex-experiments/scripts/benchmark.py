#!/usr/bin/env python3
"""
Benchmark helper that compares git-history-visualizer with the upstream Python oracle.

The script:
1. Builds the Rust binary in release mode (unless --bin-path is provided).
2. Runs both analyzers against the supplied repository, capturing wall-clock time.
3. Validates that all JSON outputs are byte-for-byte equivalent after parsing.
4. Prints a concise timing summary.

Usage:
    python scripts/benchmark.py --repo /path/to/repo
        [--python-cmd git-of-theseus-analyze]
        [--bin-path target/release/git-history-visualizer]
        [--jobs 0]  # passes through to the Rust analyzer
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

JSON_FILES = [
    "cohorts.json",
    "exts.json",
    "authors.json",
    "dirs.json",
    "domains.json",
    "survival.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark git-history-visualizer against git-of-theseus."
    )
    parser.add_argument(
        "--repo",
        required=True,
        type=Path,
        help="Path to the repository to analyze",
    )
    parser.add_argument(
        "--python-cmd",
        default="git-of-theseus-analyze",
        help=(
            "Command used to invoke the Python analyzer. "
            "Defaults to 'git-of-theseus-analyze'; try 'python3 -m git_of_theseus.analyze'."
        ),
    )
    parser.add_argument(
        "--bin-path",
        type=Path,
        help="Path to an existing git-history-visualizer binary. "
        "If omitted, the script builds target/release/git-history-visualizer.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel jobs to pass to the Rust analyzer (optional)",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep the generated output directories for inspection (defaults to cleanup)",
    )
    return parser.parse_args()


def ensure_rust_binary(bin_path: Path | None) -> Path:
    if bin_path is not None:
        if not bin_path.exists():
            raise FileNotFoundError(
                f"Provided binary path {bin_path} does not exist."
            )
        return bin_path

    print("Building git-history-visualizer in release mode...", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "--release", "--quiet"],
        check=True,
    )
    release_bin = Path("target/release/git-history-visualizer")
    if not release_bin.exists():
        raise FileNotFoundError(
            f"Expected release binary at {release_bin}, but it does not exist."
        )
    return release_bin


def run_command(cmd: Iterable[str], cwd: Path = None) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=cwd)
    return time.perf_counter() - start


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compare_outputs(oracle_dir: Path, rust_dir: Path) -> Dict[str, Tuple[bool, str]]:
    results: Dict[str, Tuple[bool, str]] = {}
    for name in JSON_FILES:
        oracle_path = oracle_dir / name
        rust_path = rust_dir / name
        if not oracle_path.exists() or not rust_path.exists():
            results[name] = (False, "missing file")
            continue
        oracle_obj = load_json(oracle_path)
        rust_obj = load_json(rust_path)
        if oracle_obj == rust_obj:
            results[name] = (True, "match")
        else:
            results[name] = (False, "JSON mismatch")
    return results


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    if not repo.exists():
        print(f"Repository path {repo} does not exist.", file=sys.stderr)
        sys.exit(1)
    if not (repo / ".git").exists():
        print(
            f"{repo} does not look like a git repository (missing .git directory).",
            file=sys.stderr,
        )
        print("Pass --repo pointing at the root of a git repository.", file=sys.stderr)
        sys.exit(1)

    rust_bin = ensure_rust_binary(args.bin_path)

    tmp_root = Path(tempfile.mkdtemp(prefix="git-hist-benchmark-"))
    oracle_out = tmp_root / "oracle"
    rust_out = tmp_root / "rust"
    oracle_out.mkdir()
    rust_out.mkdir()

    python_cmd: List[str] = shlex.split(args.python_cmd)
    print(f"Running Python oracle ({' '.join(python_cmd)})...", file=sys.stderr)
    python_time = run_command(
        [
            *python_cmd,
            str(repo),
            "--outdir",
            str(oracle_out),
            "--quiet",
        ]
    )
    print(f"oracle_out {oracle_out}")
    print(f"rust_out {rust_out}")

    rust_cmd = [
        str(rust_bin),
        "analyze",
        str(repo),
        "--outdir",
        str(rust_out),
        "--quiet",
    ]
    if args.jobs is not None:
        rust_cmd.extend(["--jobs", str(args.jobs)])

    print("Running git-history-visualizer...", file=sys.stderr)
    rust_time = run_command(rust_cmd)

    comparisons = compare_outputs(oracle_out, rust_out)

    mismatches = {name: reason for name, (ok, reason) in comparisons.items() if not ok}
    if mismatches:
        print("\nOutput comparison failed:", file=sys.stderr)
        for name, reason in mismatches.items():
            print(f"  {name}: {reason}", file=sys.stderr)
    else :
        print("\n✅ JSON outputs match exactly.")
        print(f"Python ({args.python_cmd}): {python_time:.3f} s")
        print(f"Rust   ({rust_bin}): {rust_time:.3f} s")

    if python_time > 0:
        speedup = python_time / rust_time if rust_time > 0 else float("inf")
        print(f"Speedup (Python / Rust): {speedup:.2f}x")





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.", file=sys.stderr)
        sys.exit(130)
