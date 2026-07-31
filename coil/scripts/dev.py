#!/usr/bin/env python3
"""Single developer entry point for building, testing, snapshotting, and examples."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def execute(*command: str, env: dict[str, str] | None = None, cwd: Path = ROOT) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build(args: argparse.Namespace) -> None:
    scripts = {
        "full": "scripts/compiler/rebootstrap.sh",
        "nollvm": "scripts/compiler/rebootstrap-nollvm.sh",
        "linux": "scripts/compiler/rebootstrap-linux.sh",
        "nollvm-linux": "scripts/compiler/rebootstrap-nollvm-linux.sh",
        "x64": "scripts/compiler/bootstrap-x64.sh",
    }
    command = [scripts[args.variant]]
    if args.output:
        command.append(args.output)
    execute(*command)


def test(args: argparse.Namespace) -> None:
    compiler = args.compiler
    if args.suite == "all":
        execute("scripts/compiler/rebootstrap.sh")
    elif args.suite == "snapshots":
        execute(sys.executable, "scripts/oracle.py", "gate", "all", "--compiler", compiler,
                *(["--verbose"] if args.verbose else []))
    elif args.suite == "cli":
        execute("scripts/compiler/oracle/gate-cli.sh", compiler)
    elif args.suite == "runtime":
        execute(sys.executable, "scripts/oracle.py", "runtime", "gate", "arm64", "--compiler", compiler)
    elif args.suite == "wasm":
        test_wasm(compiler)
    elif args.suite == "meta":
        test_meta(compiler)
    elif args.suite == "metaprogramming":
        execute("scripts/tests/metaprogramming/compile-and-run/run.sh", compiler)
    elif args.suite == "interpreter":
        execute(sys.executable, "scripts/oracle.py", "interpreter", "live", "--compiler", compiler,
                *(["--verbose"] if args.verbose else []))


def test_meta(compiler: str) -> None:
    interpreted = os.environ.copy()
    interpreted["COIL_META_INTERP"] = "1"
    execute(sys.executable, "scripts/oracle.py", "runtime", "gate", "arm64", "--compiler", compiler,
            env=interpreted)
    compiled = Path("/tmp/coil-meta-compiled")
    interp = Path("/tmp/coil-meta-interp")
    execute(compiler, "build", "src/compiler/main_a64.coil", "--backend", "arm64", "-o", str(compiled))
    execute(compiler, "build", "src/compiler/main_a64.coil", "--backend", "arm64", "-o", str(interp), env=interpreted)
    left = subprocess.run(["otool", "-X", "-s", "__TEXT", "__text", str(compiled)], stdout=subprocess.PIPE, check=True).stdout
    right = subprocess.run(["otool", "-X", "-s", "__TEXT", "__text", str(interp)], stdout=subprocess.PIPE, check=True).stdout
    if hashlib.sha256(left).digest() != hashlib.sha256(right).digest():
        raise SystemExit("compiled and interpreted metaprogram engines produced different compilers")
    print("metaprogram engines: PASS")


def test_wasm(compiler: str) -> None:
    if not shutil.which("node") or not shutil.which("wasm-tools"):
        print("wasm gate: SKIP (requires node and wasm-tools)")
        return
    wasm = "/tmp/gate-wasm-coilc.wasm"
    execute(compiler, "build", "src/compiler/main_wasm.coil", "--target", "wasm64-unknown-unknown",
            "--wasm-stack-size=64", "-o", wasm)
    execute("wasm-tools", "validate", "--features=memory64", wasm)
    printed = subprocess.run(["wasm-tools", "print", wasm], text=True, stdout=subprocess.PIPE, check=True).stdout
    if sum(line.startswith("(module") for line in printed.splitlines()) != 1:
        raise SystemExit("wasm compiler is not a single static module")
    env = os.environ.copy()
    env["COIL_WASM_META_TRACE"] = "1"
    result = subprocess.run(["node", "src/tooling/wasm-host/run-coil-wasm.mjs", wasm,
                             "check", "src/compiler/main_a64.coil"], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode or b"meta_run_wasm" in result.stderr or b"WALL1" in result.stderr:
        sys.stderr.buffer.write(result.stderr)
        raise SystemExit("wasm compiler self-check failed")
    print("wasm gate: PASS")


def snapshot(args: argparse.Namespace) -> None:
    execute(sys.executable, "scripts/oracle.py", "snapshot", args.stage, "--compiler", args.compiler)


def llvm_flags(mode: str) -> list[str]:
    result = subprocess.run(["scripts/compiler/llvm-link-flags.sh", mode], cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, check=True)
    return shlex.split(result.stdout)


def bootstrap_c(args: argparse.Namespace) -> None:
    source = ROOT / "src/bootstrap"
    output = ROOT / "build/bootstrap/c"
    output.mkdir(parents=True, exist_ok=True)
    wasm = Path(args.wasm).resolve() if args.wasm else ROOT / "bootstrap/seeds/wasm/coilc.wasm"
    cc = os.environ.get("CC", "cc")
    opt = shlex.split(os.environ.get("OPT", "-O1"))
    execute(cc, "-O2", "-o", str(output / "wasm2c"), str(source / "wasm2c.c"))
    execute(str(output / "wasm2c"), str(wasm), str(output / "coilc.c"), "little")
    execute(cc, *opt, "-w", "-o", str(output / "coil-bootstrap"),
            str(output / "coilc.c"), str(source / "runtime.c"), "-lm")
    print(f"built {output / 'coil-bootstrap'}")


def bootstrap_wasm32(args: argparse.Namespace) -> None:
    source = ROOT / "src/bootstrap"
    output = ROOT / "build/bootstrap/wasm32"
    output.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC", "cc")
    opt = shlex.split(os.environ.get("OPT", "-O1"))
    compiler = Path(os.environ.get("COIL", "build/bin/coil"))
    provided = os.environ.get("COIL_SEED32")
    seed = Path(provided) if provided and os.access(provided, os.X_OK) else output / "coil-seed32"
    if seed == output / "coil-seed32":
        execute(str(compiler), "build", "src/compiler/main.coil", "-o", str(seed), *llvm_flags("dynamic"))
    execute(str(seed), "build", "src/compiler/main_wasm.coil", "--target", "wasm32-unknown-unknown",
            "--wasm-stack-size=64", "-o", str(output / "coilc32.wasm"))
    execute(cc, "-O2", "-o", str(output / "wasm2c"), str(source / "wasm2c.c"))
    execute(str(output / "wasm2c"), str(output / "coilc32.wasm"), str(output / "coilc32.c"), "little")
    execute(cc, *opt, "-w", "-o", str(output / "coil-bootstrap32"),
            str(output / "coilc32.c"), str(source / "runtime32.c"), "-lm")
    print(f"built {output / 'coil-bootstrap32'}")


def bootstrap(args: argparse.Namespace) -> None:
    bootstrap_c(args) if args.variant == "c" else bootstrap_wasm32(args)


def benchmark(args: argparse.Namespace) -> None:
    script = "python3 scripts/dev.py benchmark runtime" if args.kind == "runtime" else "python3 scripts/dev.py benchmark compile-scale"
    execute(script, *args.names)


def example(args: argparse.Namespace) -> None:
    if args.name == "mini-scheme":
        output = ROOT / "build/examples/mini-scheme"
        output.parent.mkdir(parents=True, exist_ok=True)
        execute(args.compiler, "build", "src/apps/mini-scheme/scheme.coil", "-o", str(output))
        print(f"built {output}")
    elif args.name == "freestanding":
        program = args.program
        build_dir = ROOT / "build/examples/freestanding"
        build_dir.mkdir(parents=True, exist_ok=True)
        obj, boot, elf = (build_dir / f"{program}.{suffix}" for suffix in ("o", "boot.o", "elf"))
        execute(args.compiler, "emit-obj", f"src/examples/freestanding/{program}.coil", "-o", str(obj),
                "--target", "aarch64-unknown-none")
        execute("clang", "-target", "aarch64-unknown-none", "-c", "src/examples/freestanding/start.s", "-o", str(boot))
        execute("ld.lld", "--gc-sections", "-T", "src/examples/freestanding/virt.ld", str(boot), str(obj), "-o", str(elf))
        if not args.build_only:
            execute("qemu-system-aarch64", "-M", "virt", "-cpu", "cortex-a57", "-nographic", "-kernel", str(elf))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    command = commands.add_parser("build", help="rebuild and verify the compiler")
    command.add_argument("variant", choices=("full", "nollvm", "linux", "nollvm-linux", "x64"), nargs="?", default="full")
    command.add_argument("--output")
    command.set_defaults(func=build)

    command = commands.add_parser("test", help="run a test suite")
    command.add_argument("suite", choices=("all", "snapshots", "cli", "runtime", "wasm", "meta", "interpreter", "metaprogramming"), nargs="?", default="all")
    command.add_argument("--compiler", default="build/bin/coil")
    command.add_argument("--verbose", action="store_true")
    command.set_defaults(func=test)

    command = commands.add_parser("snapshot", help="regenerate compiler snapshots")
    command.add_argument("stage", choices=("all", *(__import__("oracle").STAGES)), nargs="?", default="all")
    command.add_argument("--compiler", default="build/bin/coil")
    command.set_defaults(func=snapshot)

    command = commands.add_parser("bootstrap", help="build the portable C bootstrap")
    command.add_argument("variant", choices=("c", "wasm32"), nargs="?", default="c")
    command.add_argument("--wasm")
    command.set_defaults(func=bootstrap)

    command = commands.add_parser("benchmark", help="run benchmarks")
    command.add_argument("kind", choices=("runtime", "compile-scale"), nargs="?", default="runtime")
    command.add_argument("names", nargs="*")
    command.set_defaults(func=benchmark)

    command = commands.add_parser("example", help="build an example application")
    command.add_argument("name", choices=("mini-scheme", "freestanding"))
    command.add_argument("program", nargs="?", default="hello")
    command.add_argument("--compiler", default="build/bin/coil")
    command.add_argument("--build-only", action="store_true")
    command.set_defaults(func=example)
    return result


def main() -> int:
    os.chdir(ROOT)
    args = parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
