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
# LLM keys scrubbed from os.environ at startup for hermetic determinism (see main()); the
# online agent test re-injects from this saved copy. Module-level default so it always exists.
SAVED_LLM_ENV = {}
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

    # Hermetic env: the deterministic golden tests spawn examples/assistant.scry, whose
    # chooseBrain() picks the LIVE model when an LLM key is present in the environment.
    # Scrub the keys so EVERY test subprocess gets the deterministic ScriptedModel regardless
    # of the developer's shell; run_agent_online_test re-injects the saved key explicitly.
    global SAVED_LLM_ENV
    SAVED_LLM_ENV = {k: os.environ.pop(k) for k in
                     ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY", "ANTHROPIC_API_KEY") if k in os.environ}

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
    ip, if_ = run_inspect_test(binary, filt)
    passed += ip; failed += if_
    if if_: fails.append("inspect")
    iip, iif_ = run_inspect_instances_test(binary, filt)
    passed += iip; failed += iif_
    if iif_: fails.append("inspect_instances")
    lp, lf = run_liveness_test(binary, filt)
    passed += lp; failed += lf
    if lf: fails.append("liveness")
    xp, xf = run_liveedit_test(binary, filt)
    passed += xp; failed += xf
    if xf: fails.append("liveedit")
    e7p, e7f = run_assistant_e2e(binary, filt)
    passed += e7p; failed += e7f
    if e7f: fails.append("assistant_e2e")
    up, uf = run_ui_smoke_test(binary, filt)
    passed += up; failed += uf
    if uf: fails.append("ui_smoke")
    # Phase 10: the reverse-proxy portal (registry + proxy + reaping). Gated on curl present.
    pp, pf = run_portal_test(binary, filt)
    passed += pp; failed += pf
    if pf: fails.append("portal")
    # Phase 8a: real HTTP(S) client via libcurl.
    hnp, hnf, hnfails = run_http_network_test(binary, filt)
    passed += hnp; failed += hnf; fails += hnfails
    hsp, hsf, hsfails = run_http_stw_test(binary, filt)
    passed += hsp; failed += hsf; fails += hsfails
    # Phase 8b: Env.get reads a variable the harness injects into the child's environment.
    evp, evf = run_env_roundtrip_test(binary, filt)
    passed += evp; failed += evf
    if evf: fails.append("env_roundtrip")
    # Phase 8c: the real agent loop, live inspection (always, scripted) + online (gated on key).
    alp, alf = run_agent_liveness_test(binary, filt)
    passed += alp; failed += alf
    if alf: fails.append("agent_liveness")
    aop, aof = run_agent_online_test(binary, filt)
    passed += aop; failed += aof
    if aof: fails.append("agent_online")

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


def run_inspect_test(binary, filt):
    """Phase 9 gate: `scry inspect examples/assistant.scry` serves the STATIC schema WITHOUT
    running main(). Asserts: (1) the viewer URL + the schema-only note print, (2) main() never
    ran (no `you> ` prompt, no agent/'delegating' output on stdout), (3) types() returns the
    full class set with liveCount 0, (4) the new schema() op returns class + interface + enum
    nodes (Agent class, Tool interface, AgentStatus enum) all at count 0, with a resolved
    Agent->Conversation field ref, and (5) GET / serves the viewer HTML + app.js contains the
    graph view. Then kills it."""
    if filt and "inspect" not in filt:
        return 0, 0
    import time, json, threading, urllib.request
    demo = os.path.abspath(os.path.join(HERE, "..", "examples", "assistant.scry"))
    proc = subprocess.Popen([binary, "inspect", demo], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    threading.Thread(target=lambda: [lines.append(l) for l in proc.stdout], daemon=True).start()
    port = None
    try:
        for _ in range(200):
            for l in lines:
                if "viewer: http://localhost:" in l:
                    port = int(l.strip().split(":")[-1]); break
            if port is not None:
                break
            time.sleep(0.05)
        if port is None:
            print("FAIL inspect\n     never printed viewer URL"); return 0, 1

        def ev(src):
            body = json.dumps({"id": "I", "source": src}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/eval", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=15).read())

        problems = []
        time.sleep(0.4)
        # (1) schema-only note printed
        if not any("inspect: schema only" in l for l in lines):
            problems.append("did not print the 'inspect: schema only' note")
        # (2) main() never ran — no interactive prompt / agent output
        joined = "".join(lines)
        for forbidden in ("you> ", "delegating", "goodbye"):
            if forbidden in joined:
                problems.append(f"main() appears to have run (stdout contains {forbidden!r})")
        # (3) types() -> full class set, all liveCount 0
        titems = ev("types()")["value"]["items"]
        tnames = [t["name"] for t in titems]
        for want in ("Agent", "Conversation", "Message", "ScriptedModel", "Orchestrator"):
            if want not in tnames:
                problems.append(f"types() missing {want}: {tnames}")
        nonzero = [t["name"] for t in titems if t["liveCount"] != 0]
        if nonzero:
            problems.append(f"inspect mode has nonzero liveCounts (main must not have run): {nonzero}")
        # (4) schema() -> class + interface + enum nodes at count 0, resolved edge
        nodes = ev("schema()")["value"]["nodes"]
        by = {n["name"]: n for n in nodes}
        if by.get("Agent", {}).get("kind") != "class":
            problems.append(f"schema() Agent not a class node: {by.get('Agent')}")
        if by.get("Tool", {}).get("kind") != "interface":
            problems.append(f"schema() missing Tool interface node")
        if by.get("AgentStatus", {}).get("kind") != "enum":
            problems.append(f"schema() missing AgentStatus enum node")
        if any(n["liveCount"] != 0 for n in nodes):
            problems.append("schema() has nonzero liveCount in inspect mode")
        agent = by.get("Agent", {})
        conv_ref = any("Conversation" in (f.get("refTypes") or []) for f in agent.get("fields", []))
        if not conv_ref:
            problems.append("schema() Agent has no field refTypes edge to Conversation")
        variants = [v["name"] for v in by.get("AgentStatus", {}).get("variants", [])]
        if "Idle" not in variants:
            problems.append(f"AgentStatus enum missing variants: {variants}")
        # (5) viewer HTML + graph-bearing app.js
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().decode()
        if "<" not in html or "scry" not in html.lower():
            problems.append("GET / did not return viewer HTML")
        appjs = urllib.request.urlopen(f"http://127.0.0.1:{port}/app.js", timeout=10).read().decode()
        if "GraphPane" not in appjs or "schema()" not in appjs:
            problems.append("app.js does not contain the graph view")
        if problems:
            print("FAIL inspect")
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


def run_inspect_instances_test(binary, filt):
    """Regression test for a real segfault: `scry inspect` starts the eval server but never
    calls vm-run (main() must never execute), so `(vm).program` — the shared VM handle every
    Program-touching opcode reads directly — was left null. `1 + 1` and `types()`/`schema()`
    happened to work anyway (the latter two are intercepted before compiling to bytecode and
    read the server's OWN `srv-program` global instead), but `SomeClass.instances()` compiles
    to OP_ARENA_INSTANCES, which runs on the VM and dereferenced the null `(vm).program` ->
    EXC_BAD_ACCESS, killing the whole server. Fixed by `vm-bind-program` (src/vm.coil), called
    from both `vm-run` (scry run) and `scry-inspect` (src/server.coil) so the VM handle is
    always bound before the first eval, independent of whether main() ran.

    Asserts: `Agent.instances()`, `Message.instances()`, and a filtered/paginated
    `Agent.instances(filter:, offset:, limit:)` each come back as a clean, empty list (never
    a crash, never an error — an entity type with a never-allocated arena legitimately has 0
    live instances), and the server is still alive and answering afterward."""
    if filt and "inspect" not in filt and "instances" not in filt:
        return 0, 0
    import time, json, threading, urllib.request
    demo = os.path.abspath(os.path.join(HERE, "..", "examples", "assistant.scry"))
    proc = subprocess.Popen([binary, "inspect", demo], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    threading.Thread(target=lambda: [lines.append(l) for l in proc.stdout], daemon=True).start()
    port = None
    try:
        for _ in range(200):
            for l in lines:
                if "viewer: http://localhost:" in l:
                    port = int(l.strip().split(":")[-1]); break
            if port is not None:
                break
            time.sleep(0.05)
        if port is None:
            print("FAIL inspect_instances\n     never printed viewer URL"); return 0, 1

        def ev(src):
            body = json.dumps({"id": "R", "source": src}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/eval", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=15).read())

        problems = []
        time.sleep(0.3)
        for expr, want_type in (
            ("Agent.instances()", "Agent"),
            ("Message.instances()", "Message"),
            ('Agent.instances(filter: "name == \\"x\\"", offset: 0, limit: 10)', "Agent"),
        ):
            try:
                r = ev(expr)
            except Exception as e:
                problems.append(f"`{expr}` crashed the inspect-mode server: {e}")
                continue
            v = r.get("value")
            if (v is None or v.get("type") != "list" or v.get("length") != 0
                    or v.get("elementType") != want_type or v.get("items") != []):
                problems.append(f"`{expr}` expected an empty {want_type} list, got: {r}")
        # the server must still be alive and answering correctly afterward
        if not any(p.startswith("`") and "crashed" in p for p in problems):
            try:
                alive = ev("1 + 1")
                if alive.get("value") != {"type": "Int", "value": 2}:
                    problems.append(f"server not answering correctly after instances() calls: {alive}")
            except Exception as e:
                problems.append(f"server died after instances() calls: {e}")
        if problems:
            print("FAIL inspect_instances")
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


def run_assistant_e2e(binary, filt):
    """THE Phase-7 gate: drive examples/assistant.scry interactively (stdin pipe) while POSTing
    evals over HTTP. Beats:
      (b) WHILE the app sits blocked at the initial `you> ` prompt inside Console.readLine(),
          POST types()/instance-read/a MUTATION and assert they answer promptly — proving the
          readLine is safepoint-cooperative (an STW lands while the user 'thinks').
      (a) type 'research quantum computing' -> orchestrator reply + per-agent sub-agent lines
          appear in the terminal, and a types()/instances() eval DURING the interaction shows
          the Agent count grew 1 -> 3 and the Message count climbing.
      (c) POST the exact Session.suggest redefinition from the file header, type another input,
          and assert the suggestions box now appears under the prompt (live two-way edit).
      (d) type 'exit' -> clean goodbye + the process is still serving evals.
    Paced with generous sleeps against the app's canned Clock.sleep timing (no flakiness)."""
    if filt and "assistant_e2e" not in filt:
        return 0, 0
    import time, json, threading, urllib.request
    demo = os.path.abspath(os.path.join(HERE, "..", "examples", "assistant.scry"))
    proc = subprocess.Popen([binary, "run", demo], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines = []
    threading.Thread(target=lambda: [lines.append(l) for l in proc.stdout], daemon=True).start()

    def send(s):
        proc.stdin.write(s + "\n"); proc.stdin.flush()

    port = None
    try:
        for _ in range(400):
            for l in lines:
                if "viewer: http://localhost:" in l:
                    port = int(l.strip().split(":")[-1]); break
            if port is not None:
                break
            time.sleep(0.05)
        if port is None:
            print("FAIL assistant_e2e\n     never printed viewer URL")
            return 0, 1

        def ev(src):
            body = json.dumps({"id": "E", "source": src}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/eval", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())

        problems = []
        time.sleep(0.9)   # app is now blocked at the first `you> ` prompt inside readLine

        # (b) STW answers WHILE the main thread sits in readLine
        names = [t["name"] for t in ev("types()")["value"]["items"]]
        for want in ("Agent", "Conversation", "Message", "ScriptedModel", "SubAgentWorker", "Session", "Orchestrator"):
            if want not in names:
                problems.append(f"types() (during readLine) missing {want}: {names}")
        a1 = ev("Agent.instances()")["value"]["length"]
        if a1 != 1:
            problems.append(f"expected 1 Agent before research, got {a1}")
        # a MUTATION eval during readLine: append to the orchestrator's conversation, read it back
        before = ev("Agent.instance(0).conversation.size()")["value"]["value"]
        ev('Agent.instance(0).say("probe question?")')
        after = ev("Agent.instance(0).conversation.size()")["value"]["value"]
        if not (after == before + 2):
            problems.append(f"mutation during readLine did not take effect: {before}->{after}")

        # (a) research spawns 2 sub-agents on real threads; count 1->3, messages climb
        send("research quantum computing")
        time.sleep(0.35)
        a2 = ev("Agent.instances()")["value"]["length"]
        if a2 != 3:
            problems.append(f"Agent count did not grow to 3 after research: {a2}")
        m0 = ev("Message.instances()")["value"]["length"]
        climbed = False
        for _ in range(20):          # watch messages climb while the sub-agents work
            time.sleep(0.15)
            if ev("Message.instances()")["value"]["length"] > m0:
                climbed = True; break
        if not climbed:
            problems.append(f"Message count never climbed while sub-agents worked (start {m0})")
        # orchestrator reply + per-agent sub-agent lines are in the terminal
        if not any("delegating" in l for l in lines):
            problems.append("no orchestrator 'delegating' reply in terminal")
        time.sleep(1.4)              # let all sub-agent turns finish printing
        if not any("researcher" in l for l in lines):
            problems.append("no 'researcher' sub-agent line in terminal")
        if not any("summarizer" in l for l in lines):
            problems.append("no 'summarizer' sub-agent line in terminal")

        # (c) live-redefine Session.suggest (the EXACT snippet from the file header) -> box pops up
        snippet = ('class Session {\n'
                   '  history: List<String>\n'
                   '  fn init() { self.history = List<String>() }\n'
                   '  fn renderPrompt() -> String { "you> " }\n'
                   '  fn suggest(input: String) -> String {\n'
                   '    if self.history.len() == 0 { "" }\n'
                   '    else { "  [suggestions: help | research <topic> | exit  (last: " + self.history.get(self.history.len() - 1) + ")]" }\n'
                   '  }\n'
                   '}')
        r = ev(snippet)
        v = r.get("value", {})
        if v.get("type") != "defined" or v.get("defined") != "Session":
            problems.append(f"Session.suggest redefinition not accepted: {r}")
        mark = len(lines)
        send("hello again")
        boxed = False
        for _ in range(20):
            time.sleep(0.15)
            if any("[suggestions:" in l for l in lines[mark:]):
                boxed = True; break
        if not boxed:
            problems.append("suggestions box never appeared after redefining Session.suggest")

        # (d) exit -> clean goodbye, process still serving evals
        send("exit")
        gone = False
        for _ in range(20):
            time.sleep(0.15)
            if any("goodbye" in l for l in lines):
                gone = True; break
        if not gone:
            problems.append("no goodbye after exit")
        if proc.poll() is not None:
            problems.append("process exited after 'exit' (should stay alive serving evals)")
        elif ev("types()").get("value") is None:
            problems.append("process stopped serving evals after exit")

        if problems:
            print("FAIL assistant_e2e")
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


def run_ui_smoke_test(binary, filt):
    """Real headless-browser test of the React viewer (tests/ui-smoke.mjs). Zero npm deps —
    drives system Chrome over the DevTools Protocol via Node's built-in WebSocket. Verifies the
    click-through (rail -> table -> detail -> method invoke) AND, crucially, that an open method
    card + typed arg text SURVIVE two poll refreshes (the whole point of the React rewrite).
    Gated: SKIPPED (not failed) if `node` or a Chrome binary is absent."""
    if filt and "ui" not in filt and "smoke" not in filt:
        return 0, 0
    import shutil
    if shutil.which("node") is None:
        print("SKIPPED ui_smoke: node not found on PATH")
        return 0, 0
    script = os.path.join(HERE, "ui-smoke.mjs")
    if not os.path.exists(script):
        return 0, 0
    try:
        p = subprocess.run(["node", script], capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        print("FAIL ui_smoke\n     timed out")
        return 0, 1
    # the script prints SKIPPED + exits 0 when no Chrome is present
    if "SKIPPED" in p.stdout:
        print(p.stdout.strip().splitlines()[0])
        return 0, 0
    if p.returncode != 0:
        print("FAIL ui_smoke")
        for ln in (p.stdout + p.stderr).strip().splitlines():
            print("     " + ln)
        return 0, 1
    return 1, 0


def run_portal_test(binary, filt):
    """Phase 10 gate: the reverse-proxy PORTAL. Starts `scry portal` (fixed :7357), then:
      (1) GET /api/programs is [] before any program registers;
      (2) `scry run examples/demo-mini.scry` in the background APPEARS in /api/programs within
          ~2s with mode 'run' and an ephemeral port (>= 7400);
      (3) POST /p/<id>/eval {source:'types()'} proxies to that program and returns the real
          schema JSON (proves the reverse proxy) — with the SAME id echoed back;
      (4) `scry inspect examples/agents.scry` gives a SECOND entry with mode 'inspect';
      (5) killing one program greys it to status 'exited' within the reap window;
      (6) GET / serves the viewer HTML.
    SKIPPED LOUDLY (not failed) if :7357 is already occupied (e.g. a developer's own portal)."""
    if filt and "portal" not in filt:
        return 0, 0
    import time, json, socket as _socket, threading, urllib.request
    # if something already holds :7357, skip rather than clobber it.
    try:
        s = _socket.create_connection(("127.0.0.1", 7357), timeout=0.4); s.close()
        print("SKIPPED portal: 127.0.0.1:7357 already in use (a portal is already running?)")
        return 0, 0
    except OSError:
        pass  # free — good

    demo = os.path.abspath(os.path.join(HERE, "..", "examples", "demo-mini.scry"))
    other = os.path.abspath(os.path.join(HERE, "..", "examples", "agents.scry"))
    portal = subprocess.Popen([binary, "portal"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1)
    plines = []
    threading.Thread(target=lambda: [plines.append(l) for l in portal.stdout], daemon=True).start()
    kids = []

    def api():
        return json.loads(urllib.request.urlopen("http://127.0.0.1:7357/api/programs", timeout=10).read())

    def proxy(pid, src):
        body = json.dumps({"id": "P", "source": src}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:7357/p/{pid}/eval", data=body,
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    try:
        # wait for the portal to come up (or bail if it couldn't bind)
        up = False
        for _ in range(100):
            if any("portal: http://localhost:7357" in l for l in plines):
                up = True; break
            if any("could not bind" in l for l in plines):
                print("SKIPPED portal: portal could not bind :7357"); return 0, 0
            time.sleep(0.05)
        if not up:
            print("FAIL portal\n     portal never printed its startup line"); return 0, 1

        problems = []
        # (1) empty registry
        if api() != []:
            problems.append(f"expected [] before any program, got {api()}")

        # (2) launch a `run` program; it should pop up
        kids.append(subprocess.Popen([binary, "run", demo], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL))
        run_entry = None
        for _ in range(40):        # ~4s
            time.sleep(0.1)
            progs = api()
            run_entry = next((p for p in progs if p["mode"] == "run"), None)
            if run_entry:
                break
        if not run_entry:
            problems.append("run program never appeared in /api/programs")
            print("FAIL portal"); [print("     " + p) for p in problems]; return 0, 1
        if run_entry["port"] < 7400:
            problems.append(f"run program port not ephemeral (>=7400): {run_entry['port']}")
        if run_entry["name"] != "demo-mini.scry":
            problems.append(f"unexpected program name: {run_entry['name']}")

        # (3) reverse-proxy an eval to it
        r = proxy(run_entry["id"], "types()")
        if r.get("id") != "P":
            problems.append(f"proxied eval id not echoed: {r}")
        names = [t.get("name") for t in r.get("value", {}).get("items", [])]
        if "Agent" not in names:
            problems.append(f"proxied types() missing Agent: {names}")

        # (4) a second program, this time `inspect` -> two entries, correct modes
        kids.append(subprocess.Popen([binary, "inspect", other], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        insp_entry = None
        for _ in range(40):
            time.sleep(0.1)
            progs = api()
            insp_entry = next((p for p in progs if p["mode"] == "inspect"), None)
            if insp_entry and len([p for p in progs if p["status"] == "running"]) >= 2:
                break
        if not insp_entry:
            problems.append("inspect program never appeared with mode 'inspect'")
        else:
            ri = proxy(insp_entry["id"], "types()")
            if "Agent" not in [t.get("name") for t in ri.get("value", {}).get("items", [])]:
                problems.append(f"proxied types() to inspect program failed: {ri}")

        # (5) kill the run program -> it greys (status 'exited') within the reap window
        kids[0].kill()
        greyed = False
        for _ in range(30):        # ~3s (reap runs on each /api/programs poll)
            time.sleep(0.1)
            e = next((p for p in api() if p["id"] == run_entry["id"]), None)
            if e and e["status"] == "exited":
                greyed = True; break
        if not greyed:
            problems.append("killed run program never greyed to status 'exited'")

        # (6) GET / serves the viewer HTML
        html = urllib.request.urlopen("http://127.0.0.1:7357/", timeout=10).read().decode()
        if "<" not in html or "scry" not in html.lower():
            problems.append("portal GET / did not return viewer HTML")

        if problems:
            print("FAIL portal")
            for p in problems:
                print("     " + p)
            return 0, 1
        return 1, 0
    finally:
        for k in kids:
            try: k.kill()
            except Exception: pass
        portal.terminate()
        try:
            portal.wait(timeout=5)
        except Exception:
            portal.kill()


HTTP_DIR = os.path.join(HERE, "http")


def _net_reachable(host="api.anthropic.com", port=443, timeout=5):
    import socket
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except Exception:
        return False


def run_env_roundtrip_test(binary, filt):
    """Phase 8b: prove Env.get actually reads the process environment (not just HOME/None, which
    the env_get golden covers). The harness injects SCRY_TEST_VAR into the child's env and runs a
    throwaway program that echoes Env.get("SCRY_TEST_VAR"); we assert the injected value comes back
    and that a sibling unset name is None. Deleted immediately."""
    if filt and "env" not in filt:
        return 0, 0
    prog = os.path.join(HERE, "run", "_env_roundtrip_tmp.scry")
    with open(prog, "w") as fh:
        fh.write('fn main() {\n'
                 '  match Env.get("SCRY_TEST_VAR") { Some(v) -> Console.log("got=" + v) None -> Console.log("got=None") }\n'
                 '  match Env.get("SCRY_STILL_UNSET_QQ") { Some(v) -> Console.log("other=Some") None -> Console.log("other=None") }\n'
                 '}\n')
    try:
        env = dict(os.environ)
        env["SCRY_TEST_VAR"] = "hello-from-harness"
        env.pop("SCRY_STILL_UNSET_QQ", None)
        p = subprocess.run([binary, "run", "--no-viewer", prog],
                           capture_output=True, text=True, timeout=60, env=env)
        out = p.stdout
        problems = []
        if "got=hello-from-harness" not in out:
            problems.append(f"Env.get did not read injected var; stdout: {out!r}")
        if "other=None" not in out:
            problems.append(f"unset var not None; stdout: {out!r}")
        if problems:
            print("FAIL env_roundtrip")
            for pr in problems:
                print("     " + pr)
            return 0, 1
        return 1, 0
    finally:
        try:
            os.remove(prog)
        except OSError:
            pass


def run_http_network_test(binary, filt):
    """Phase 8a real-network gate. Runs tests/http/get.scry (a real HTTPS GET to the Anthropic
    API root through libcurl) and asserts it printed a genuine HTTP status (200/401/404 all prove
    DNS+TLS+parse). Then, ONLY if ANTHROPIC_API_KEY is set, POSTs a minimal /v1/messages request
    and asserts status 200 + body contains "content". SKIPPED LOUDLY (not failed) when offline.
    The key is read from the environment at test time and written to a throwaway temp file that is
    deleted immediately — never hardcoded, never committed."""
    if filt and "http" not in filt:
        return 0, 0, []
    prog = os.path.join(HTTP_DIR, "get.scry")
    if not os.path.exists(prog):
        return 0, 0, []
    if not _net_reachable():
        print("SKIPPED http_network: api.anthropic.com:443 unreachable (offline)")
        return 0, 0, []
    passed = failed = 0
    fails = []
    # (1) plain GET -> a real HTTP status code
    code, out, err = run(binary, ["run", "--no-viewer"], os.path.abspath(prog))
    status = None
    for ln in out.splitlines():
        if ln.startswith("status="):
            try:
                status = int(ln.split("=", 1)[1])
            except ValueError:
                pass
    if code != 0 or status not in (200, 401, 403, 404):
        print(f"FAIL http_network/get\n     expected a real HTTP status, got status={status} "
              f"exit={code} stderr={err.strip()!r}")
        failed += 1
        fails.append("http_network/get")
    else:
        passed += 1

    # (2) authenticated POST — only when a real key is present in the environment
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        import json as _json
        body = _json.dumps({"model": "claude-haiku-4-5-20251001", "max_tokens": 16,
                            "messages": [{"role": "user", "content": "hi"}]})
        # Scry has no env access yet (that's Phase 8b), so embed the key + body in a throwaway
        # temp program. NEVER committed; deleted in finally.
        import tempfile
        def sq(s):  # Scry string literal escaping for embedding JSON/headers
            return s.replace("\\", "\\\\").replace('"', '\\"')
        prog_src = (
            'fn main() {\n'
            '  let h = List<String>()\n'
            f'  h.push("x-api-key: {sq(key)}")\n'
            '  h.push("anthropic-version: 2023-06-01")\n'
            '  h.push("content-type: application/json")\n'
            f'  let r = Http.request("POST", "https://api.anthropic.com/v1/messages", h, "{sq(body)}")\n'
            '  Console.log("status=${r.status}")\n'
            '  Console.log("body=${r.body}")\n'
            '}\n'
        )
        tf = tempfile.NamedTemporaryFile("w", suffix=".scry", delete=False)
        try:
            tf.write(prog_src); tf.close()
            code, out, err = run(binary, ["run", "--no-viewer"], tf.name)
            got = None
            for ln in out.splitlines():
                if ln.startswith("status="):
                    got = ln.split("=", 1)[1]
            if got != "200" or '"content"' not in out:
                print(f"FAIL http_network/post\n     expected 200 + content, got status={got} "
                      f"exit={code}\n     out={out.strip()!r} err={err.strip()!r}")
                failed += 1
                fails.append("http_network/post")
            else:
                passed += 1
        finally:
            os.unlink(tf.name)
    else:
        print("SKIPPED http_network/post: ANTHROPIC_API_KEY not set")
    return passed, failed, fails


def run_http_stw_test(binary, filt):
    """Phase 8a cooperative-STW gate. Starts `scry run` (with the viewer server) on
    tests/http/ping_thread.scry — a background OS thread hammering HTTPS requests in a loop — and
    WHILE requests are in flight POSTs types()/the pinger's count through the eval channel,
    asserting each answers in well under a second and that HttpPinger.count climbs. A prompt eval
    reply mid-request proves the HTTP thread parked for the global stop within one 50ms multi-poll
    slice (it did not block the OS thread in curl_easy_perform). SKIPPED LOUDLY when offline.
    Offline structural fallback: assert http-perform's loop body calls safepoint-poll."""
    if filt and "http" not in filt:
        return 0, 0, []
    src = os.path.join(os.path.dirname(HERE), "src", "http.coil")
    # structural invariant (always checked, also the offline proof): the multi loop parks.
    struct_ok = False
    if os.path.exists(src):
        text = open(src).read()
        # crude but robust: safepoint-poll appears inside http-perform before curl_multi_poll.
        hp = text.split("defn http-perform", 1)
        struct_ok = len(hp) == 2 and "safepoint-poll" in hp[1].split("defn ", 1)[0]
    if not struct_ok:
        print("FAIL http_stw/structural\n     http-perform loop does not call safepoint-poll")
        return 0, 1, ["http_stw/structural"]

    prog = os.path.join(HTTP_DIR, "ping_thread.scry")
    if not os.path.exists(prog):
        return 1, 0, []
    if not _net_reachable():
        print("SKIPPED http_stw/live: api.anthropic.com:443 unreachable (offline); "
              "structural safepoint-poll check passed")
        return 1, 0, []

    import time, json, threading, urllib.request
    proc = subprocess.Popen([binary, "run", os.path.abspath(prog)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    threading.Thread(target=lambda: [lines.append(ln) for ln in proc.stdout], daemon=True).start()
    port = None
    passed = failed = 0
    fails = []
    try:
        for _ in range(400):
            for ln in lines:
                if "viewer: http://localhost:" in ln:
                    port = int(ln.strip().split(":")[-1]); break
            if port is not None:
                break
            time.sleep(0.05)
        if port is None:
            print("FAIL http_stw/live\n     never printed viewer URL")
            return passed, 1, ["http_stw/live"]

        def ev(src_):
            body = json.dumps({"id": "H", "source": src_}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/eval", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=10).read())

        problems = []
        # POST several evals DURING the ping loop; each must answer fast (proves mid-request park).
        counts = []
        for _ in range(4):
            t0 = time.time()
            r = ev("types()")
            dt = time.time() - t0
            names = [t.get("name") for t in r.get("value", {}).get("items", [])]
            if "HttpPinger" not in names:
                problems.append(f"types() missing HttpPinger: {names}")
                break
            if dt > 1.0:
                problems.append(f"eval mid-request took {dt:.2f}s (>1s: HTTP thread did not park)")
            c = ev("HttpPinger.instances()")["value"]["items"]
            if c:
                counts.append(int(c[0]["fields"]["count"]["value"]))
            time.sleep(0.4)
        if counts and not (counts[-1] > counts[0]):
            problems.append(f"HttpPinger.count did not climb while running: {counts}")
        if problems:
            print("FAIL http_stw/live")
            for pr in problems:
                print("     " + pr)
            failed += 1
            fails.append("http_stw/live")
        else:
            passed += 1
        return passed, failed, fails
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def run_agent_liveness_test(binary, filt):
    """Phase 8c LIVE inspection of the real agent loop — always runs (scripted brain, ZERO network).
    Starts `scry run` (viewer server) on examples/assistant.scry, types `loop weather in Tokyo` so a
    LoopWorker runs a REPEATING agent loop (model->tool->result->answer) on a background OS thread,
    then over the eval channel: (1) asserts Message.instances() climbs while it runs, (2) pause()s the
    looper Agent and asserts its Message count FREEZES (the loop visibly stalls), (3) resume()s and
    asserts it climbs again. Same STW-cooperative liveness pattern as Phase 5/7, now over the loop."""
    if filt and "agent" not in filt and "liveness" not in filt:
        return 0, 0
    import time, json, threading, urllib.request
    demo = os.path.abspath(os.path.join(HERE, "..", "examples", "assistant.scry"))
    proc = subprocess.Popen([binary, "run", demo], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    threading.Thread(target=lambda: [lines.append(l) for l in proc.stdout], daemon=True).start()

    def send(s):
        proc.stdin.write(s + "\n"); proc.stdin.flush()

    port = None
    try:
        for _ in range(400):
            for l in lines:
                if "viewer: http://localhost:" in l:
                    port = int(l.strip().split(":")[-1]); break
            if port is not None:
                break
            time.sleep(0.05)
        if port is None:
            print("FAIL agent_liveness\n     never printed viewer URL")
            return 0, 1

        def ev(src):
            body = json.dumps({"id": "AL", "source": src}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/eval", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=30).read())

        problems = []
        time.sleep(0.6)  # app is at the first prompt inside readLine
        send("loop weather in Tokyo")
        time.sleep(1.0)

        def mcount():
            return ev("Message.instances()")["value"]["length"]

        # (1) Message climbs while the loop runs
        m1 = mcount()
        time.sleep(1.6)
        m2 = mcount()
        if not (m2 > m1):
            problems.append(f"Message count did not climb while the loop ran: {m1}->{m2}")

        # locate the looper Agent's slot
        insts = ev("Agent.instances()")["value"]["items"]
        slot = None
        for it in insts:
            if it["fields"]["name"]["value"] == "looper":
                slot = int(it["ref"].split("#")[1]); break
        if slot is None:
            problems.append(f"no looper Agent found: {[i['fields']['name']['value'] for i in insts]}")
            print("FAIL agent_liveness"); [print("     " + p) for p in problems]
            proc.terminate(); return 0, 1

        # (2) pause -> the loop freezes (let the in-flight round settle first)
        ev(f"Agent.instance({slot}).pause()")
        time.sleep(1.5)
        ma = mcount()
        time.sleep(1.6)
        mb = mcount()
        if mb != ma:
            problems.append(f"paused loop still produced Messages: {ma}->{mb}")

        # (3) resume -> it climbs again
        ev(f"Agent.instance({slot}).resume()")
        time.sleep(1.8)
        mc = mcount()
        if not (mc > mb):
            problems.append(f"resumed loop did not climb: {mb}->{mc}")

        if problems:
            print("FAIL agent_liveness")
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


def run_agent_online_test(binary, filt):
    """Phase 8c ONLINE gate: the REAL Anthropic agent loop. Gated on ANTHROPIC_API_KEY + reachability
    (SKIPPED LOUDLY otherwise). Exports the key into the child, drives examples/assistant.scry with a
    prompt that needs a tool ('what is 17 times 23?'), and asserts the live model went stop_reason
    tool_use -> the `calculate` tool ran -> a final answer contains 391. The key is read from the
    environment only, never written to disk, never committed."""
    if filt and "agent" not in filt and "online" not in filt:
        return 0, 0
    # Default target is DeepSeek's Anthropic-compatible endpoint; chooseBrain resolves the key
    # from DEEPSEEK_API_KEY | DEEPSEEK_KEY | ANTHROPIC_API_KEY and the base from ANTHROPIC_BASE_URL.
    # The keys were scrubbed from os.environ for hermetic determinism; read the saved copy and
    # re-inject it into THIS child only, so the live model is exercised here and nowhere else.
    saved = globals().get("SAVED_LLM_ENV", {})
    key = (saved.get("DEEPSEEK_API_KEY") or saved.get("DEEPSEEK_KEY")
           or saved.get("ANTHROPIC_API_KEY"))
    if not key:
        print("SKIPPED agent_online: no DEEPSEEK_API_KEY / DEEPSEEK_KEY / ANTHROPIC_API_KEY set")
        return 0, 0
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    host = base.split("://", 1)[-1].split("/", 1)[0]
    if not _net_reachable(host=host):
        print(f"SKIPPED agent_online: {host}:443 unreachable (offline)")
        return 0, 0
    child_env = dict(os.environ, **saved)
    demo = os.path.abspath(os.path.join(HERE, "..", "examples", "assistant.scry"))
    try:
        p = subprocess.run([binary, "run", "--no-viewer", demo],
                           input="what is 17 times 23?\nexit\n",
                           capture_output=True, text=True, timeout=120, env=child_env)
        out = p.stdout
    except subprocess.TimeoutExpired:
        print("FAIL agent_online\n     timed out talking to the live API")
        return 0, 1
    problems = []
    if "AnthropicModel" not in out:
        problems.append("did not announce the AnthropicModel brain (key not seen?)")
    if "[agent] -> tool_use: calculate" not in out:
        problems.append(f"live model never emitted a tool_use for calculate; out={out.strip()!r}")
    if "391" not in out:
        problems.append(f"final answer did not contain 391; out={out.strip()!r}")
    if problems:
        print("FAIL agent_online")
        for pr in problems:
            print("     " + pr)
        return 0, 1
    return 1, 0


def _diff(expected, actual):
    import difflib
    d = difflib.unified_diff(expected.splitlines(), actual.splitlines(),
                             "golden", "actual", lineterm="")
    return "\n".join("     " + l for l in d)


if __name__ == "__main__":
    main()
