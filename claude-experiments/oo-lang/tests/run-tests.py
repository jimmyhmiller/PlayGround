#!/usr/bin/env python3
"""Golden test harness for `scry parse-dump` (Phase 1 front end).

Modeled on the clox harness. For each tests/parse/NAME.scry:

  * If NAME.out exists  -> a SUCCESS test: run `scry parse-dump NAME.scry`,
    expect exit 0 and stdout exactly equal to NAME.out.
  * If NAME.err exists  -> an ERROR test: expect a NONZERO exit code, and every
    non-blank line of NAME.err must appear (as a substring) somewhere in stderr.

Usage:
  ./run-tests.py [path-to-scry-binary] [--filter SUBSTR] [-v] [--bless]

Defaults: binary = ../scry relative to this file. --bless (re)writes NAME.out
for every SUCCESS-style test (a .scry with no .err sibling) from current output;
use only after eyeballing a diff. Exits nonzero if any test fails.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARSE_DIR = os.path.join(HERE, "parse")
CHECK_DIR = os.path.join(HERE, "check")


def run(binary, subcmd, scry):
    try:
        p = subprocess.run([binary, subcmd, scry],
                           capture_output=True, text=True, timeout=60)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return None, "", "TIMEOUT"


def main():
    args = list(sys.argv[1:])
    verbose = "-v" in args
    if verbose:
        args.remove("-v")
    bless = "--bless" in args
    if bless:
        args.remove("--bless")
    filt = None
    if "--filter" in args:
        i = args.index("--filter"); filt = args[i + 1]; del args[i:i + 2]
    binary = os.path.abspath(args[0]) if args else os.path.join(HERE, "..", "scry")
    binary = os.path.abspath(binary)
    if not os.path.exists(binary):
        print(f"scry binary not found: {binary} (run `coil build` first)")
        sys.exit(2)

    passed = failed = 0
    fails = []

    def run_dir(directory, subcmd, golden):
        nonlocal passed, failed
        if not os.path.isdir(directory):
            return
        for f in sorted(fn for fn in os.listdir(directory) if fn.endswith(".scry")):
            name = f[:-5]
            if filt and filt not in name:
                continue
            scry = os.path.join(directory, f)
            out_path = os.path.join(directory, name + ".out")
            err_path = os.path.join(directory, name + ".err")
            code, out, err = run(binary, subcmd, scry)
            problems = []
            if os.path.exists(err_path):
                # error test: nonzero exit + each nonblank .err line is a stderr substring
                if code == 0 or code is None:
                    problems.append(f"expected nonzero exit, got {code}")
                with open(err_path) as fh:
                    for line in fh.read().splitlines():
                        if line.strip() and line not in err:
                            problems.append(f"missing expected diagnostic substring: {line!r}\n       actual stderr: {err.strip()!r}")
            elif golden:
                # golden-stdout success test (parse-dump)
                if bless:
                    with open(out_path, "w") as fh:
                        fh.write(out)
                if code != 0:
                    problems.append(f"expected exit 0, got {code}; stderr: {err.strip()!r}")
                expected = ""
                if os.path.exists(out_path):
                    with open(out_path) as fh:
                        expected = fh.read()
                elif not bless:
                    problems.append(f"no golden file {name}.out (run with --bless to create)")
                if not bless and out != expected:
                    problems.append("stdout != golden " + name + ".out")
                    if verbose:
                        problems.append(_diff(expected, out))
            else:
                # exit-0 success test (check): just require a clean exit
                if code != 0:
                    problems.append(f"expected exit 0, got {code}; stderr: {err.strip()!r}")
            if problems:
                failed += 1
                fails.append(name)
                print(f"FAIL {name}")
                for pr in problems:
                    print("     " + pr)
            else:
                passed += 1

    run_dir(PARSE_DIR, "parse-dump", True)
    run_dir(CHECK_DIR, "check", False)

    print(f"\n{passed} passed, {failed} failed ({passed + failed} tests)")
    if fails and not verbose:
        print("Failed: " + ", ".join(fails))
    sys.exit(1 if failed else 0)


def _diff(expected, actual):
    import difflib
    d = difflib.unified_diff(expected.splitlines(), actual.splitlines(),
                             "golden", "actual", lineterm="")
    return "\n".join("     " + l for l in d)


if __name__ == "__main__":
    main()
