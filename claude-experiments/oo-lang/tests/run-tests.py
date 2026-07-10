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
RUN_DIR = os.path.join(HERE, "run")
RUNERR_DIR = os.path.join(HERE, "run-err")
ARENAS_DIR = os.path.join(HERE, "run-arenas")


def run(binary, subargs, scry):
    try:
        p = subprocess.run([binary] + subargs + [scry],
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

    def run_dir(directory, subargs, golden):
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
            code, out, err = run(binary, subargs, scry)
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

    run_dir(PARSE_DIR, ["parse-dump"], True)
    run_dir(CHECK_DIR, ["check"], False)
    # `scry run` now starts the eval server; --no-viewer runs main() to completion
    # and exits (identical stdout to the pre-Phase-4 `run`).
    run_dir(RUN_DIR, ["run", "--no-viewer"], True)
    run_dir(RUNERR_DIR, ["run", "--no-viewer"], True)
    run_dir(ARENAS_DIR, ["run", "--dump-arenas"], True)

    # Phase 4: eval golden tests (tests/eval/*.t) + the server smoke test.
    ep, ef, efails = run_eval_tests(binary, filt)
    passed += ep; failed += ef; fails += efails
    sp, sf = run_smoke_test(binary, filt)
    passed += sp; failed += sf
    if sf: fails.append("smoke")

    print(f"\n{passed} passed, {failed} failed ({passed + failed} tests)")
    if fails and not verbose:
        print("Failed: " + ", ".join(fails))
    sys.exit(1 if failed else 0)


EVAL_DIR = os.path.join(HERE, "eval")


def parse_eval_t(path):
    """A .t file: `key: value` lines. Keys: file, expr, readonly, contains (repeat),
    notcontains (repeat). `expr` may span until the next top-level `key:` line."""
    spec = {"file": None, "expr": None, "readonly": False, "contains": [], "notcontains": []}
    cur_key = None
    for raw in open(path).read().splitlines():
        if raw.startswith("file:"):
            spec["file"] = raw[5:].strip(); cur_key = None
        elif raw.startswith("readonly:"):
            spec["readonly"] = raw[9:].strip().lower() in ("yes", "true", "1"); cur_key = None
        elif raw.startswith("contains:"):
            spec["contains"].append(raw[9:].strip()); cur_key = None
        elif raw.startswith("notcontains:"):
            spec["notcontains"].append(raw[12:].strip()); cur_key = None
        elif raw.startswith("expr:"):
            spec["expr"] = raw[5:]; cur_key = "expr"
        elif cur_key == "expr":
            spec["expr"] += "\n" + raw
    if spec["expr"] is not None:
        spec["expr"] = spec["expr"].strip("\n")
    return spec


def run_eval_tests(binary, filt):
    passed = failed = 0
    fails = []
    if not os.path.isdir(EVAL_DIR):
        return passed, failed, fails
    for f in sorted(fn for fn in os.listdir(EVAL_DIR) if fn.endswith(".t")):
        name = "eval/" + f[:-2]
        if filt and filt not in name:
            continue
        spec = parse_eval_t(os.path.join(EVAL_DIR, f))
        scry = os.path.join(HERE, "..", spec["file"])
        sub = ["eval", os.path.abspath(scry), "-e", spec["expr"]]
        if spec["readonly"]:
            sub.append("--readonly")
        try:
            p = subprocess.run([binary] + sub, capture_output=True, text=True, timeout=60)
            out = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        except subprocess.TimeoutExpired:
            out = "TIMEOUT"
        problems = []
        for c in spec["contains"]:
            if c not in out:
                problems.append(f"missing {c!r}\n       actual: {out!r}")
        for c in spec["notcontains"]:
            if c in out:
                problems.append(f"unexpected {c!r}")
        if problems:
            failed += 1; fails.append(name)
            print(f"FAIL {name}")
            for pr in problems:
                print("     " + pr)
        else:
            passed += 1
    return passed, failed, fails


def run_smoke_test(binary, filt):
    """Start `scry run` (with server), curl POST /eval + GET /, assert, kill."""
    if filt and "smoke" not in filt:
        return 0, 0
    import time, json, urllib.request
    demo = os.path.abspath(os.path.join(HERE, "..", "examples", "demo-mini.scry"))
    proc = subprocess.Popen([binary, "run", demo], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    port = None
    try:
        # read the "viewer: http://localhost:PORT" line
        for _ in range(200):
            line = proc.stdout.readline()
            if not line:
                break
            if "viewer: http://localhost:" in line:
                port = int(line.strip().split(":")[-1]); break
        if port is None:
            print("FAIL smoke\n     never printed viewer URL")
            return 0, 1

        def post_eval(src):
            body = json.dumps({"id": "e1", "source": src}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/eval", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=10).read())

        problems = []
        time.sleep(0.2)
        r = post_eval("types()")
        if r.get("id") != "e1":
            problems.append(f"types() id mismatch: {r}")
        items = r.get("value", {}).get("items", [])
        names = [t.get("name") for t in items]
        if "Agent" not in names:
            problems.append(f"types() missing Agent: {names}")
        r2 = post_eval("Agent.instances()")
        if r2.get("value", {}).get("length") != 3:
            problems.append(f"expected 3 agents, got {r2.get('value', {}).get('length')}")
        # GET / returns HTML
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().decode()
        if "<" not in html or "scry" not in html.lower():
            problems.append("GET / did not return viewer HTML")
        if problems:
            print("FAIL smoke")
            for pr in problems:
                print("     " + pr)
            return 0, 1
        return 1, 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _diff(expected, actual):
    import difflib
    d = difflib.unified_diff(expected.splitlines(), actual.splitlines(),
                             "golden", "actual", lineterm="")
    return "\n".join("     " + l for l in d)


if __name__ == "__main__":
    main()
