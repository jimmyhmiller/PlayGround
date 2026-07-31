#!/usr/bin/env python3
"""Generate and verify Coil compiler snapshots without per-stage shell wrappers."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "tests/compiler/oracle"
TARGET_X86 = "x86_64-apple-macosx11.0.0"
STAGES = ("read", "ast", "load", "resolved", "checked", "expand", "mono", "ir", "diag", "x86", "full")
COMMAND = {
    "read": "dump-read", "ast": "dump-ast", "load": "dump-load",
    "resolved": "dump-resolved", "checked": "dump-checked",
    "expand": "dump-expand", "mono": "dump-mono", "ir": "dump-ir",
    "full": "emit-ir", "x86": "emit-ir",
}

IR_SEEDS = """
src/examples/allocation.coil src/examples/allocators.coil src/examples/args.coil
src/examples/bitfields.coil src/examples/closure.coil src/examples/explicit-layout.coil
src/examples/extern.coil src/examples/fib.coil src/examples/generics.coil
src/examples/inference.coil src/examples/io.coil src/examples/layout.coil
src/examples/lockfree.coil src/examples/mem.coil src/examples/references.coil
src/examples/structs.coil src/examples/sums.coil src/examples/threads.coil
src/examples/vector.coil src/examples/widths.coil src/apps/chip8/objc.coil
src/stdlib/alloc.coil src/stdlib/arraylist.coil src/stdlib/atomic.coil
src/stdlib/closure.coil src/stdlib/control.coil src/stdlib/derive.coil
src/stdlib/dyn.coil src/stdlib/fmt.coil src/stdlib/hashmap.coil src/stdlib/match.coil
src/stdlib/mem.coil src/stdlib/mmio.coil src/stdlib/print.coil src/stdlib/result.coil
src/stdlib/slice.coil src/stdlib/thread.coil src/stdlib/try.coil
""".split()

FULL_EXTRA = """
src/examples/calc.coil src/examples/json.coil src/examples/hashmap.coil
src/examples/dyn_write.coil src/examples/simd.coil
tests/compiler/oracle/features/meta_stage3.coil
tests/compiler/oracle/features/export_c.coil
tests/compiler/oracle/features/x86_sysv_abi.coil
tests/compiler/oracle/features/fs_lib.coil
src/examples/conventions.coil src/examples/per-arch.coil
src/examples/shim.coil src/examples/everything.coil
""".split()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def mangle(path: str) -> str:
    return path.replace("/", "_")


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def write_list(path: Path, entries: list[str]) -> None:
    path.write_text("".join(f"{entry}\n" for entry in sorted(entries)))


def real_sources() -> list[str]:
    files: list[Path] = []
    for directory in (ROOT / "src/examples", ROOT / "src/stdlib", ROOT / "src/apps"):
        files.extend(directory.rglob("*.coil"))
    return sorted(rel(path) for path in files)


def run(compiler: Path, command: str, source: str, *extra: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([str(compiler), command, source, *extra], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def reset(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)


def snapshot_simple(compiler: Path, stage: str, inputs: list[str], *, command: str | None = None,
                    extra: tuple[str, ...] = (), suffix: str = ".dump", allow_fail: bool = False) -> list[str]:
    base = ORACLE if stage == "read" else ORACLE / stage
    reference = base / "reference"
    corpus = base / "corpus.txt"
    reset(reference)
    accepted: list[str] = []
    for source in inputs:
        result = run(compiler, command or COMMAND[stage], source, *extra)
        if result.returncode and allow_fail:
            first = result.stderr.decode(errors="replace").splitlines()[:1]
            print(f"SKIP {source}: {first[0] if first else 'compiler failed'}")
            continue
        if result.returncode:
            sys.stderr.buffer.write(result.stderr)
            raise SystemExit(f"snapshot {stage} failed: {source}")
        (reference / f"{mangle(source)}{suffix}").write_bytes(result.stdout)
        accepted.append(source)
    write_list(corpus, accepted)
    return accepted


def snapshot_expanded(compiler: Path, stage: str) -> list[str]:
    base = ORACLE / stage
    corpus_dir = base / "corpus"
    reference = base / "reference"
    exclusions = base / "EXCLUDED.txt"
    reset(corpus_dir)
    reset(reference)
    accepted: list[str] = []
    excluded: list[str] = []
    for source in real_sources():
        expanded_name = mangle(source)
        expanded = corpus_dir / expanded_name
        result = run(compiler, "expand", source)
        if result.returncode:
            first = result.stderr.decode(errors="replace").splitlines()[:1]
            excluded.append(f"{source} : {first[0] if first else 'expand failed'}")
            continue
        expanded.write_bytes(result.stdout)
        expanded_rel = rel(expanded)
        dumped = run(compiler, COMMAND[stage], expanded_rel)
        if dumped.returncode:
            sys.stderr.buffer.write(dumped.stderr)
            raise SystemExit(f"snapshot {stage} failed: {expanded_rel}")
        (reference / f"{mangle(expanded_rel)}.dump").write_bytes(dumped.stdout)
        accepted.append(expanded_rel)
    fixture_dir = base / ("negative" if stage == "ast" else "fixtures")
    for fixture in sorted(fixture_dir.glob("*.coil")):
        source = rel(fixture)
        dumped = run(compiler, COMMAND[stage], source)
        if dumped.returncode:
            sys.stderr.buffer.write(dumped.stderr)
            raise SystemExit(f"snapshot {stage} failed: {source}")
        (reference / f"{mangle(source)}.dump").write_bytes(dumped.stdout)
        accepted.append(source)
    write_list(base / "corpus.txt", accepted)
    exclusions.write_text("".join(f"{item}\n" for item in excluded))
    return accepted


def snapshot_filtering(compiler: Path, stage: str) -> list[str]:
    base = ORACLE / stage
    reference = base / "reference"
    reset(reference)
    accepted: list[str] = []
    excluded: list[str] = []
    inputs = real_sources() + [rel(path) for path in sorted((base / "fixtures").glob("*.coil"))]
    for source in inputs:
        result = run(compiler, COMMAND[stage], source)
        if result.returncode:
            first = result.stderr.decode(errors="replace").splitlines()[:1]
            excluded.append(f"{source} : {first[0] if first else 'compiler failed'}")
            continue
        (reference / f"{mangle(source)}.dump").write_bytes(result.stdout)
        accepted.append(source)
    write_list(base / "corpus.txt", accepted)
    (base / "EXCLUDED.txt").write_text("".join(f"{item}\n" for item in excluded))
    return accepted


def snapshot_reusing(compiler: Path, stage: str, prior: str) -> list[str]:
    base = ORACLE / stage
    inputs = read_list(ORACLE / prior / "corpus.txt")
    inputs += [rel(path) for path in sorted((base / "fixtures").glob("*.coil"))]
    return snapshot_simple(compiler, stage, inputs)


def snapshot_diag(compiler: Path) -> int:
    base = ORACLE / "diag"
    reference = base / "reference"
    reset(reference)
    inputs = [rel(path) for path in sorted((base / "inputs").glob("*.coil"))]
    write_list(base / "corpus.txt", inputs)
    root_prefix = f"{ROOT}/".encode()
    for source in inputs:
        result = run(compiler, "emit-ir", source)
        (reference / f"{mangle(source)}.diag").write_bytes((result.stdout + result.stderr).replace(root_prefix, b""))
    build_inputs = [rel(path) for path in sorted((base / "build-inputs").glob("*.coil"))]
    write_list(base / "build-corpus.txt", build_inputs)
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        for source in build_inputs:
            output = temp_path / Path(source).stem
            result = subprocess.run([str(compiler), "build", source, "-o", str(output)], cwd=ROOT,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            normalized = result.stdout.replace(root_prefix, b"").replace(f"{temp}/".encode(), b"")
            stem = reference / f"{mangle(source)}"
            Path(f"{stem}.diag").write_bytes(normalized)
            Path(f"{stem}.exit").write_text(f"{result.returncode}\n")
    print(f"snapshot diag: {len(inputs)} diagnostics, {len(build_inputs)} build diagnostics")
    return 0


def snapshot(compiler: Path, stage: str) -> int:
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise SystemExit(f"reference compiler is not executable: {compiler}")
    if stage == "all":
        for item in STAGES:
            snapshot(compiler, item)
        return 0
    if stage == "diag":
        return snapshot_diag(compiler)
    if stage == "read":
        inputs = real_sources() + [rel(path) for path in sorted((ORACLE / "negative").glob("*.coil"))]
        accepted = snapshot_simple(compiler, stage, inputs)
    elif stage in ("ast", "resolved"):
        accepted = snapshot_expanded(compiler, stage)
    elif stage in ("load", "expand"):
        accepted = snapshot_filtering(compiler, stage)
    elif stage == "checked":
        accepted = snapshot_reusing(compiler, stage, "resolved")
    elif stage == "mono":
        accepted = snapshot_reusing(compiler, stage, "checked")
    elif stage == "ir":
        inputs = [rel(path) for path in sorted((ORACLE / "ir/fixtures").glob("*.coil"))] + IR_SEEDS
        accepted = snapshot_simple(compiler, stage, inputs, allow_fail=True)
    elif stage == "full":
        inputs = [rel(path) for path in sorted((ORACLE / "ir/fixtures").glob("*.coil"))]
        inputs += [rel(path) for path in sorted((ROOT / "src/stdlib").glob("*.coil"))]
        inputs += IR_SEEDS[:21] + FULL_EXTRA
        accepted = snapshot_simple(compiler, stage, inputs, command=os.environ.get("COIL_IR_CMD", "emit-ir"), allow_fail=True)
    elif stage == "x86":
        inputs = [rel(path) for path in sorted((ORACLE / "features").glob("*x86*.coil"))]
        accepted = snapshot_simple(compiler, stage, inputs, extra=("--target", TARGET_X86))
    else:
        raise SystemExit(f"unknown snapshot stage: {stage}")
    print(f"snapshot {stage}: {len(accepted)} files")
    return 0


def gate_diag(compiler: Path, verbose: bool) -> int:
    base = ORACLE / "diag"
    reference = base / "reference"
    failures: list[str] = []
    root_prefix = f"{ROOT}/".encode()
    for source in read_list(base / "corpus.txt"):
        result = run(compiler, "emit-ir", source)
        got = (result.stdout + result.stderr).replace(root_prefix, b"")
        want = (reference / f"{mangle(source)}.diag").read_bytes()
        if got != want:
            failures.append(source)
            if verbose:
                print(f"FAIL diag: {source}")
    with tempfile.TemporaryDirectory() as temp:
        for source in read_list(base / "build-corpus.txt"):
            output = Path(temp) / Path(source).stem
            result = subprocess.run([str(compiler), "build", source, "-o", str(output)], cwd=ROOT,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            got = result.stdout.replace(root_prefix, b"").replace(f"{temp}/".encode(), b"")
            stem = reference / mangle(source)
            want = Path(f"{stem}.diag").read_bytes()
            want_code = int(Path(f"{stem}.exit").read_text())
            if got != want or result.returncode != want_code:
                failures.append(source)
                if verbose:
                    print(f"FAIL build diagnostic: {source}")
    print(f"gate diag: {'PASS' if not failures else f'{len(failures)} failed'}")
    return 1 if failures else 0


def gate(compiler: Path, stage: str, verbose: bool) -> int:
    if stage == "all":
        for item in STAGES:
            if gate(compiler, item, verbose):
                return 1
        return 0
    if stage == "diag":
        return gate_diag(compiler, verbose)
    base = ORACLE if stage == "read" else ORACLE / stage
    suffix = ".dump"
    extra = ["--target", TARGET_X86] if stage == "x86" else shlex.split(os.environ.get("COIL_SELF_ARGS", ""))
    failures: list[str] = []
    passed = 0
    for source in read_list(base / "corpus.txt"):
        result = run(compiler, COMMAND[stage], source, *extra)
        reference = base / "reference" / f"{mangle(source)}{suffix}"
        if result.returncode == 0 and reference.is_file() and result.stdout.rstrip(b"\n") == reference.read_bytes().rstrip(b"\n"):
            passed += 1
            continue
        failures.append(source)
        if verbose:
            reason = result.stderr.decode(errors="replace").splitlines()[:1]
            print(f"FAIL {stage}: {source}: {reason[0] if reason else 'output mismatch'}")
    print(f"gate {stage}: {passed} passed, {len(failures)} failed")
    return 1 if failures else 0


def runtime(compiler: Path, action: str, platform: str, verbose: bool) -> int:
    source_platform = "arm64" if platform == "linux" else platform
    base = ORACLE / source_platform
    reference = base / "reference"
    excluded = set(read_list(ORACLE / "linux/arm64-only.txt")) if platform == "linux" else set()
    failures = 0
    passed = 0
    for line in read_list(base / "corpus.txt"):
        parts = shlex.split(line)
        rust_reference = parts[0] == "R"
        if rust_reference:
            parts.pop(0)
        source, *program_args = parts
        identity = source.replace("/", "_").replace(".", "_")
        fixed_prefix = "coil-arm64" if platform in ("arm64", "linux") else "coil-x64"
        executable = Path("/tmp") / f"{fixed_prefix}-fixed-{identity}"
        build = [str(compiler), "build", source, "-o", str(executable)]
        if action == "gate" and platform != "linux":
            build += ["--backend", platform]
        elif action == "snapshot" and rust_reference and platform == "arm64":
            build += ["--backend", "arm64"]
        result = subprocess.run(build, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
        if platform == "linux" and source in excluded:
            if result.returncode and b"not a general-purpose register on the target architecture" in result.stderr:
                passed += 1
            else:
                failures += 1
                print(f"FAIL architecture diagnostic: {source}")
            continue
        if result.returncode:
            failures += 1
            print(f"FAIL build: {source}")
            if verbose:
                print(result.stderr.decode(errors="replace").splitlines()[:3])
            continue
        ran = subprocess.run([str(executable), *program_args], stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        stdout_file = reference / f"{identity}.stdout"
        stderr_file = reference / f"{identity}.stderr"
        exit_file = reference / f"{identity}.exit"
        if action == "snapshot":
            stdout_file.write_bytes(ran.stdout)
            stderr_file.write_bytes(ran.stderr)
            exit_file.write_text(f"{ran.returncode}\n")
            passed += 1
        elif stdout_file.is_file() and ran.stdout == stdout_file.read_bytes() and ran.returncode == int(exit_file.read_text()):
            passed += 1
        else:
            failures += 1
            print(f"FAIL run: {source} exit={ran.returncode}")
    print(f"runtime {action} {platform}: {passed} passed, {failures} failed")
    return 1 if failures else 0


def coverage() -> int:
    corpus = read_list(ORACLE / "full/corpus.txt")
    excluded = set(read_list(ORACLE / "linux/arm64-only.txt"))
    mac = ORACLE / "full/reference"
    linux = ORACLE / "linux/full-reference"
    expected_mac = {f"{mangle(source)}.dump" for source in corpus}
    expected_linux = {f"{mangle(source)}.dump" for source in corpus if source not in excluded}
    actual_mac = {path.name for path in mac.glob("*.dump")}
    actual_linux = {path.name for path in linux.glob("*.dump")}
    problems = []
    for label, expected, actual in (("macOS", expected_mac, actual_mac), ("Linux", expected_linux, actual_linux)):
        problems += [f"MISSING {label}: {name}" for name in sorted(expected - actual)]
        problems += [f"ORPHAN {label}: {name}" for name in sorted(actual - expected)]
    if problems:
        print("\n".join(problems))
        print(f"coverage: {len(problems)} problem(s)")
        return 1
    print(f"coverage: PASS ({len(corpus)} shared full-pipeline entries)")
    return 0


def linux_ir(compiler: Path, action: str, verbose: bool) -> int:
    corpus = read_list(ORACLE / "full/corpus.txt")
    excluded = set(read_list(ORACLE / "linux/arm64-only.txt"))
    reference = ORACLE / "linux/full-reference"
    reference.mkdir(parents=True, exist_ok=True)
    failures = 0
    passed = 0
    for source in corpus:
        result = run(compiler, "emit-ir", source, *shlex.split(os.environ.get("COIL_SELF_ARGS", "")))
        if source in excluded:
            if result.returncode and b"not a general-purpose register on the target architecture" in result.stderr:
                passed += 1
            else:
                failures += 1
                print(f"FAIL architecture diagnostic: {source}")
            continue
        output = reference / f"{mangle(source)}.dump"
        if action == "snapshot":
            if result.returncode:
                failures += 1
                print(f"FAIL snapshot: {source}")
            else:
                output.write_bytes(result.stdout)
                passed += 1
        elif result.returncode == 0 and output.is_file() and result.stdout.rstrip(b"\n") == output.read_bytes().rstrip(b"\n"):
            passed += 1
        else:
            failures += 1
            print(f"FAIL Linux IR: {source}")
            if verbose and result.stderr:
                print(result.stderr.decode(errors="replace").splitlines()[0])
    print(f"Linux IR {action}: {passed} passed, {failures} failed")
    return 1 if failures else 0


def interpreter(compiler: Path, live: bool, verbose: bool) -> int:
    base = ORACLE / "arm64"
    reference = base / "reference"
    failures = 0
    passed = 0
    for line in read_list(base / "corpus.txt"):
        parts = shlex.split(line)
        special_backend = parts[0] == "R"
        if special_backend:
            parts.pop(0)
        source, *program_args = parts
        identity = source.replace("/", "_").replace(".", "_")
        interpreted = subprocess.run([str(compiler), "interp", source, *program_args], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, timeout=60)
        if live:
            executable = Path("/tmp") / f"coil-interp-compiled-{identity}"
            build = [str(compiler), "build", source, "-o", str(executable)]
            if special_backend:
                build += ["--backend", "arm64"]
            built = subprocess.run(build, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
            if built.returncode:
                failures += 1
                print(f"FAIL interpreter comparison build: {source}")
                continue
            compiled = subprocess.run([source, *program_args], executable=str(executable), cwd=ROOT,
                                      stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, timeout=30)
            want_stdout, want_code = compiled.stdout, compiled.returncode
        else:
            want_stdout = (reference / f"{identity}.stdout").read_bytes()
            want_code = int((reference / f"{identity}.exit").read_text())
        if interpreted.stdout == want_stdout and interpreted.returncode == want_code:
            passed += 1
        else:
            failures += 1
            print(f"FAIL interpreter: {source} exit={interpreted.returncode} want={want_code}")
            if verbose:
                print(f"stdout bytes: got={len(interpreted.stdout)} want={len(want_stdout)}")
    label = "live" if live else "snapshot"
    print(f"interpreter {label}: {passed} passed, {failures} failed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("gate", "snapshot"):
        command = sub.add_parser(action)
        command.add_argument("stage", choices=("all", *STAGES))
        command.add_argument("--compiler", default=os.environ.get("COIL_REF_BIN", "build/bin/coil"))
        if action == "gate":
            command.add_argument("--verbose", action="store_true", default=os.environ.get("VERBOSE") == "1")
    command = sub.add_parser("runtime")
    command.add_argument("operation", choices=("gate", "snapshot"))
    command.add_argument("platform", choices=("arm64", "x64", "linux"))
    command.add_argument("--compiler", default="build/bin/coil")
    command.add_argument("--verbose", action="store_true")
    command = sub.add_parser("linux-ir")
    command.add_argument("operation", choices=("gate", "snapshot"))
    command.add_argument("--compiler", default="build/bin/coil")
    command.add_argument("--verbose", action="store_true")
    command = sub.add_parser("interpreter")
    command.add_argument("mode", choices=("snapshot", "live"), nargs="?", default="snapshot")
    command.add_argument("--compiler", default="build/bin/coil")
    command.add_argument("--verbose", action="store_true")
    sub.add_parser("coverage")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "coverage":
        return coverage()
    compiler = Path(args.compiler)
    if not compiler.is_absolute():
        compiler = ROOT / compiler
    if args.action == "snapshot":
        return snapshot(compiler, args.stage)
    if args.action == "runtime":
        return runtime(compiler, args.operation, args.platform, args.verbose)
    if args.action == "linux-ir":
        return linux_ir(compiler, args.operation, args.verbose)
    if args.action == "interpreter":
        return interpreter(compiler, args.mode == "live", args.verbose)
    return gate(compiler, args.stage, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
