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


def run(binary, subargs, scry, stdin=None):
    try:
        p = subprocess.run([binary] + subargs + [scry],
                           capture_output=True, text=True, timeout=60,
                           input=stdin)
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
            # Phase 7: an optional NAME.stdin file is piped to the program's stdin, so a
            # Console.readLine()-driven interactive program can be golden-tested by feeding
            # it a scripted transcript. Absent => empty stdin (immediate EOF).
            stdin_path = os.path.join(directory, name + ".stdin")
            stdin = open(stdin_path).read() if os.path.exists(stdin_path) else None
            code, out, err = run(binary, subargs, scry, stdin=stdin)
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
    # Phase 7: interactive-app goldens (tests/app/*.t) — scripted stdin -> stdout substrings.
    ap, af, afails = run_app_tests(binary, filt)
    passed += ap; failed += af; fails += afails
    sp, sf = run_smoke_test(binary, filt)
    passed += sp; failed += sf
    if sf: fails.append("smoke")
    lp, lf = run_liveness_test(binary, filt)
    passed += lp; failed += lf
    if lf: fails.append("liveness")
    xp, xf = run_liveedit_test(binary, filt)
    passed += xp; failed += xf
    if xf: fails.append("liveedit")

    print(f"\n{passed} passed, {failed} failed ({passed + failed} tests)")
    if fails and not verbose:
        print("Failed: " + ", ".join(fails))
    sys.exit(1 if failed else 0)


EVAL_DIR = os.path.join(HERE, "eval")


def parse_eval_t(path):
    """A .t file: `key: value` lines. Keys: file, expr (may REPEAT — each is a separate
    `-e`, run in sequence against one process for live-change tests), readonly,
    contains (repeat), notcontains (repeat). An `expr` spans until the next `key:` line."""
    spec = {"file": None, "exprs": [], "readonly": False, "contains": [], "notcontains": [], "stdin": None}
    cur_key = None
    for raw in open(path).read().splitlines():
        if raw.startswith("file:"):
            spec["file"] = raw[5:].strip(); cur_key = None
        elif raw.startswith("stdin:"):
            # a scripted stdin transcript fed to the program's main() before the -e evals run;
            # \n is a literal newline escape so a transcript fits one line.
            spec["stdin"] = raw[6:].strip().replace("\\n", "\n"); cur_key = None
        elif raw.startswith("readonly:"):
            spec["readonly"] = raw[9:].strip().lower() in ("yes", "true", "1"); cur_key = None
        elif raw.startswith("contains:"):
            spec["contains"].append(raw[9:].strip()); cur_key = None
        elif raw.startswith("notcontains:"):
            spec["notcontains"].append(raw[12:].strip()); cur_key = None
        elif raw.startswith("expr:"):
            spec["exprs"].append(raw[5:]); cur_key = "expr"
        elif cur_key == "expr":
            spec["exprs"][-1] += "\n" + raw
    spec["exprs"] = [e.strip("\n") for e in spec["exprs"]]
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
        sub = ["eval", os.path.abspath(scry)]
        for e in spec["exprs"]:
            sub += ["-e", e]
        if spec["readonly"]:
            sub.append("--readonly")
        try:
            p = subprocess.run([binary] + sub, capture_output=True, text=True, timeout=60,
                               input=spec["stdin"])
            # match against the eval RESULT lines only (JSON), never the program's stdout;
            # joined so a multi-expr sequence can be asserted with distinct substrings.
            evlines = [ln for ln in p.stdout.splitlines()
                       if ln.startswith('{"value"') or ln.startswith('{"error"')]
            out = "\n".join(evlines) if evlines else (p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "")
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


APP_DIR = os.path.join(HERE, "app")


def run_app_tests(binary, filt):
    """Phase 7 app goldens: run examples/assistant.scry (the flagship interactive app) with a
    scripted stdin transcript (`scry run --no-viewer`) and assert substrings on its terminal
    stdout. Substring (not exact) matching keeps the deterministic assertions (prompt/response,
    help, EOF/exit goodbye, post-join aggregate) robust to the intentionally-nondeterministic
    interleaving of background sub-agent lines. A .t file: `stdin:` (a \\n-escaped transcript),
    `contains:`/`notcontains:` (repeatable). `file:` defaults to examples/assistant.scry."""
    passed = failed = 0
    fails = []
    if not os.path.isdir(APP_DIR):
        return passed, failed, fails
    for f in sorted(fn for fn in os.listdir(APP_DIR) if fn.endswith(".t")):
        name = "app/" + f[:-2]
        if filt and filt not in name and "app" not in (filt or ""):
            continue
        spec = parse_eval_t(os.path.join(APP_DIR, f))
        rel = spec["file"] or "examples/assistant.scry"
        scry = os.path.abspath(os.path.join(HERE, "..", rel))
        try:
            p = subprocess.run([binary, "run", "--no-viewer", scry],
                               capture_output=True, text=True, timeout=60,
                               input=spec["stdin"])
            out = p.stdout
        except subprocess.TimeoutExpired:
            out = "TIMEOUT"
        problems = []
        for c in spec["contains"]:
            if c not in out:
                problems.append(f"missing {c!r}\n       actual: {out!r}")
        for c in spec["notcontains"]:
            if c in out:
                problems.append(f"unexpected {c!r}\n       actual: {out!r}")
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
            return json.loads(urllib.request.urlopen(req, timeout=30).read())

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


def run_liveness_test(binary, filt):
    """THE Phase-5 demo beat: start examples/agents.scry (3 agents on 3 OS threads), and
    while they run, POST evals through the viewer channel. Assert (1) types() counts and the
    agent instances are visible, (2) Conversation sizes climb between two polls (agents work
    concurrently), (3) invoking pause() on one agent freezes ITS conversation while the
    others keep growing, and (4) resume() restarts it. Every eval runs under a full STW."""
    if filt and "liveness" not in filt:
        return 0, 0
    import time, json, threading, urllib.request
    demo = os.path.abspath(os.path.join(HERE, "..", "examples", "agents.scry"))
    proc = subprocess.Popen([binary, "run", demo], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    # Drain stdout continuously in a thread: an agent's Console.log write must never block on
    # a full pipe, or it can't reach a safepoint and would stall request-global-stop's STW.
    lines = []
    threading.Thread(target=lambda: [lines.append(ln) for ln in proc.stdout], daemon=True).start()
    port = None
    try:
        for _ in range(400):
            for ln in lines:
                if "viewer: http://localhost:" in ln:
                    port = int(ln.strip().split(":")[-1]); break
            if port is not None:
                break
            time.sleep(0.05)
        if port is None:
            print("FAIL liveness\n     never printed viewer URL")
            return 0, 1

        def ev(src):
            body = json.dumps({"id": "L", "source": src}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/eval", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())

        problems = []
        time.sleep(1.0)
        # (1) types() and instances visible
        names = [t["name"] for t in ev("types()")["value"]["items"]]
        for want in ("Agent", "Conversation", "Message", "ScriptedModel", "AgentWorker"):
            if want not in names:
                problems.append(f"types() missing {want}: {names}")
        insts = ev("Agent.instances()")["value"]["items"]
        slot = {it["fields"]["name"]["value"]: int(it["ref"].split("#")[1]) for it in insts}
        if not all(a in slot for a in ("researcher", "coder", "reviewer")):
            problems.append(f"missing agents: {slot}")
            print("FAIL liveness");  [print("     " + p) for p in problems]
            return 0, 1
        r, c = slot["researcher"], slot["coder"]

        def size(s):
            return ev(f"Agent.instance({s}).conversation.size()")["value"]["value"]

        # (2) both climb while running
        r1, c1 = size(r), size(c)
        time.sleep(1.8)
        r2, c2 = size(r), size(c)
        if not (r2 > r1):
            problems.append(f"researcher did not climb while running: {r1}->{r2}")
        if not (c2 > c1):
            problems.append(f"coder did not climb while running: {c1}->{c2}")

        # (3) pause researcher -> it freezes, coder keeps climbing
        ev(f"Agent.instance({r}).pause()")
        time.sleep(1.8)                 # let any in-flight step settle
        ra, ca = size(r), size(c)
        time.sleep(1.6)
        rb, cb = size(r), size(c)
        if rb != ra:
            problems.append(f"paused researcher still grew: {ra}->{rb}")
        if not (cb > ca):
            problems.append(f"coder stalled while researcher paused: {ca}->{cb}")

        # (4) resume researcher -> it climbs again
        ev(f"Agent.instance({r}).resume()")
        time.sleep(1.8)
        rc = size(r)
        if not (rc > rb):
            problems.append(f"resumed researcher did not climb: {rb}->{rc}")

        if problems:
            print("FAIL liveness")
            for p in problems:
                print("     " + p)
            return 0, 1
        return 1, 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def run_liveedit_test(binary, filt):
    """THE Phase-6 demo beat: start examples/agents.scry (3 agents on 3 OS threads) and,
    WHILE they print to the terminal, POST a redefinition of ScriptedModel.complete that
    changes the printed text. Assert (1) the terminal output visibly changes within a couple
    of turns, (2) the process keeps running and instance identity/counts persist, then
    (3) POST a deliberately-bad edit (type error) and assert it is rejected AND the behavior
    is unchanged (the running program is untouched by a rejected edit)."""
    if filt and "liveedit" not in filt:
        return 0, 0
    import time, json, threading, urllib.request
    demo = os.path.abspath(os.path.join(HERE, "..", "examples", "agents.scry"))
    proc = subprocess.Popen([binary, "run", demo], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    def drain():
        for ln in proc.stdout:
            lines.append(ln)
    t = threading.Thread(target=drain, daemon=True); t.start()

    def port():
        for ln in lines:
            if "viewer: http://localhost:" in ln:
                return int(ln.strip().split(":")[-1])
        return None
    try:
        p = None
        for _ in range(200):
            p = port()
            if p is not None:
                break
            time.sleep(0.05)
        if p is None:
            print("FAIL liveedit\n     never printed viewer URL")
            return 0, 1

        def ev(src):
            body = json.dumps({"id": "X", "source": src}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{p}/eval", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())

        problems = []
        time.sleep(1.5)
        # (0) baseline: the old ScriptedModel.complete prints "<reply> re: task <n>"
        base = [ln for ln in lines if "re: task" in ln]
        if not base:
            problems.append("no baseline agent output containing 're: task'")

        # (1) live-redefine ScriptedModel.complete to change the printed text
        r = ev('class ScriptedModel {\n  reply: String\n'
                '  fn complete(prompt: String) -> String { "PATCHED[" + self.reply + "] " + prompt }\n}')
        v = r.get("value", {})
        if v.get("type") != "defined" or v.get("defined") != "ScriptedModel":
            problems.append(f"redefinition not accepted: {r}")
        gen_after = v.get("gen")
        if gen_after != 1:
            problems.append(f"expected gen 1 after first edit, got {gen_after}")

        # (2) the terminal output visibly changes within a couple of turns
        mark = len(lines)
        patched = None
        for _ in range(30):
            time.sleep(0.3)
            if any("PATCHED[" in ln for ln in lines[mark:]):
                patched = True
                break
        if not patched:
            problems.append("terminal output never showed the redefined 'PATCHED[' text")

        # (3) process still running + instance identity/counts persist across the edit
        if proc.poll() is not None:
            problems.append("process exited during the live edit")
        names = [tt["name"] for tt in ev("types()")["value"]["items"]]
        for want in ("Agent", "ScriptedModel"):
            if want not in names:
                problems.append(f"types() missing {want} after edit: {names}")
        nsm = ev("ScriptedModel.instances()")["value"]["length"]
        nag = ev("Agent.instances()")["value"]["length"]
        if nsm != 3:
            problems.append(f"ScriptedModel count changed across edit: {nsm}")
        if nag != 3:
            problems.append(f"Agent count changed across edit: {nag}")
        agents = ev("Agent.instances()")["value"]["items"]
        anames = sorted(a["fields"]["name"]["value"] for a in agents)
        if anames != ["coder", "researcher", "reviewer"]:
            problems.append(f"agent identities changed: {anames}")

        # (4) a deliberately-bad edit is REJECTED and behavior is unchanged
        rb = ev('class ScriptedModel {\n  reply: String\n'
                '  fn complete(prompt: String) -> String { self.reply + 5 }\n}')
        if "error" not in rb or rb["error"].get("kind") != "TypeError":
            problems.append(f"bad edit not rejected as TypeError: {rb}")
        if ev("generation()")["value"]["value"] != 1:
            problems.append("generation changed on a rejected edit (must be a no-op)")
        mark2 = len(lines)
        still_patched = None
        for _ in range(20):
            time.sleep(0.3)
            new = lines[mark2:]
            if new and all(("PATCHED[" in ln) for ln in new if "msg#" in ln):
                still_patched = True
                break
        if still_patched is None:
            # tolerate: as long as no post-reject line reverted to the old "re: task" form
            reverted = any(("re: task" in ln and "PATCHED[" not in ln) for ln in lines[mark2:])
            if reverted:
                problems.append("rejected edit changed behavior (output reverted)")

        if problems:
            print("FAIL liveedit")
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
