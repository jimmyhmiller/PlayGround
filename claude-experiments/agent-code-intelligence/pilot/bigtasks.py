"""Scaled-eval tasks over the granite corpus: setup + scoring.

Differences from the small pilot, by design:
  - tasks give NO hints about dynamic dispatch, config coupling, or each
    tool's blind spots; docs are command lists only
  - the visible test suite is weak; catastrophes are detected by the
    hidden suite the agents never see
  - targets are drawn from key.json so scoring is mechanical

Usage:
  python3 bigtasks.py setup --big DIR --runs-dir RUNS
  python3 bigtasks.py score --big DIR --runs-dir RUNS
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("quarry", "lsp")
K = 2

REPORT = (
    'Before finishing, Write WORKSPACE/report.json as JSON:\n'
    '{"task_id": "<id>", "answer": <see task>, "decision": "<one line>",\n'
    ' "evidence": "<1-3 sentences>", "assigned_tool_calls": <int>}\n'
    "Your final chat message must be a single line.")

RULES = """TOOL RULES (strict, part of the evaluation):
- For ALL code understanding and navigation (finding symbols, references,
  callers, types, diagnostics, impact), use ONLY the assigned tool
  documented at {doc}. Read that doc first.
- FORBIDDEN: grep/rg/find/awk/sed or any other text-search command, and
  bulk scanning (ls -R, globbing over source, reading files one after
  another to search). Reading a SPECIFIC file you have already located,
  to view or edit it, is allowed. The codebase is ~150 files / ~15k LOC:
  reading it wholesale is not viable.
- Edit files with your normal editing tools.
- Run the test suite with: cd {ws} && python3 -m tests.run
"""


def load_key(big):
    with open(os.path.join(big, "key.json")) as f:
        return json.load(f)


def reach(key):
    """caller -> set(callee) adjacency from key edges."""
    adj = {}
    for e in key["edges"]:
        adj.setdefault(e["caller"], set()).add(e["callee"])
    # method quals also 'contain' their functions: treat quals as nodes
    return adj


def closure_from(adj, start):
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, ()):  # edges from exact node
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
        # method nodes: edges recorded under Class.method quals — also
        # follow any node whose prefix matches (module-level containment)
        return_nodes = [n for n in adj if n.startswith(cur + ".")]
        for n in return_nodes:
            for nxt in adj[n]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return seen


def build_tasks(key):
    """Choose concrete targets from the key. Deterministic."""
    tasks = {}
    ds, dt = key["dead_safe"], key["dead_trap"]
    # judgment tasks: 2 safe, 2 trap — same neutral wording
    for i, (tid, qual, kind) in enumerate([
            ("g1_judge_a", ds[2], "safe"), ("g2_judge_b", ds[3], "safe"),
            ("g3_judge_c", dt[1], "trap"), ("g4_judge_d", dt[2], "trap")]):
        mod, fn = qual.rsplit(".", 1)
        path = mod.replace(".", "/") + ".py"
        tasks[tid] = {
            "kind": "judge", "qual": qual, "expect": kind,
            "answer_hint": '{"action": "deleted" | "kept"}',
            "prompt": (
                f"A cleanup ticket proposes deleting the function `{fn}` "
                f"in {path} as dead code. Determine whether it is truly "
                "safe to delete (nothing in the repository can invoke it, "
                "by any mechanism). If safe, delete it; if not, keep it "
                "and say why. The fast test suite must still pass.")}
    # nav: call sites of a collided method with a handful of true sites
    by_callee = {}
    for e in key["edges"]:
        by_callee.setdefault(e["callee"], []).append(e)
    meth_edges = {c: es for c, es in by_callee.items()
                  if c.split(".")[-1] == "process"
                  and c.count(".") >= 3 and 3 <= len(es) <= 6}
    nav_callee = sorted(meth_edges)[0]
    nav_sites = {(e["path"], ln) for e in meth_edges[nav_callee]
                 for ln in e["candidate_lines"]}
    cls = nav_callee.rsplit(".", 2)
    tasks["g5_nav"] = {
        "kind": "nav", "callee": nav_callee, "sites": sorted(nav_sites),
        "answer_hint": '{"call_sites": [{"file": "...", "line": <int>}]}',
        "prompt": (
            f"The class `{cls[1]}` (module {cls[0]}) has a method "
            f"`process`. List every call site of THIS class's `process` "
            "method — exact file and 1-based line — excluding the "
            "definition. There are many same-named `process` methods on "
            "other classes; do not include their call sites. Do not "
            "change any files.")}
    tasks["g6_typebugs"] = {
        "kind": "bugs", "bugs": key["type_bugs"],
        "answer_hint": '{"defects": [{"file": "...", "line": <int>}]}',
        "prompt": (
            "The type gate for this repository reports it is not clean, "
            "though all tests pass (the defects are latent). Find every "
            "type-level defect, fix each minimally (preserve intended "
            "behavior), and leave the repository type-clean with the "
            "fast test suite still passing.")}
    tasks["g7_hooks"] = {
        "kind": "hooks", "also_static": sorted(key["hooks"]["also_static"]),
        "configured": sorted(key["hooks"]["configured"]),
        "answer_hint": '{"also_called_statically": ["<module.func>", ...]}',
        "prompt": (
            "maintenance.cfg lists a number of hook functions. Determine "
            "which of the configured hooks are ALSO invoked statically "
            "somewhere in the code (i.e. they would remain live code even "
            "if removed from the configuration). Give fully qualified "
            "names. Do not change any files.")}
    # reachability: one true chain from granite.main, one unreachable decoy
    adj = reach(key)
    reachable = closure_from(adj, "granite.main")
    # candidate deep target: reachable core function
    deep = sorted(q for q in reachable
                  if q.startswith("granite.core.") and q.count(".") == 3)
    target = deep[len(deep) // 2]
    unreach = sorted(q for q in
                     {f"{e['callee']}" for e in key["edges"]}
                     if q.startswith("granite.core.") and q not in reachable
                     and q.count(".") == 3)
    decoy = unreach[0] if unreach else None
    tasks["g8_reach"] = {
        "kind": "reach", "target": target, "decoy": decoy,
        "answer_hint": ('{"target_reachable": true|false, "chain": '
                        '["granite.main", ..., "<target>"], '
                        '"decoy_reachable": true|false}'),
        "prompt": (
            f"Two questions about call reachability from `main` in "
            f"granite/main.py, by static function calls only. (1) Can "
            f"execution starting at main reach `{target}`? If yes, give "
            f"one full call chain as a list of qualified functions. "
            f"(2) Can it reach `{decoy}`? Do not change any files.")}
    return tasks


def setup(big, runs_dir):
    key = load_key(big)
    tasks = build_tasks(key)
    with open(os.path.join(big, "tasks_built.json"), "w") as f:
        json.dump(tasks, f, indent=1)
    os.makedirs(runs_dir, exist_ok=True)
    manifest = []
    for tid, spec in tasks.items():
        for arm in ARMS:
            for k in range(1, K + 1):
                rid = f"{tid}__{arm}__k{k}"
                rdir = os.path.join(runs_dir, rid)
                ws = os.path.join(rdir, "workspace")
                if os.path.exists(rdir):
                    shutil.rmtree(rdir)
                shutil.copytree(os.path.join(big, "workspace"), ws)
                doc = os.path.join(HERE, "docs",
                                   "QUARRY-min.md" if arm == "quarry"
                                   else "LSP-min.md")
                prompt = (
                    "You are one run in a controlled evaluation of "
                    "code-intelligence tooling for coding agents. Work in "
                    f"WORKSPACE = {ws}\n\n"
                    + RULES.format(doc=doc, ws=ws)
                    + f"\nTASK ({tid}):\n{spec['prompt']}\n\n"
                    + REPORT.replace("<id>", tid).replace(
                        "<see task>", spec["answer_hint"]))
                with open(os.path.join(rdir, "prompt.txt"), "w") as f:
                    f.write(prompt)
                with open(os.path.join(rdir, "meta.json"), "w") as f:
                    json.dump({"task": tid, "arm": arm, "k": k,
                               "workspace": ws}, f)
                manifest.append(rid)
    with open(os.path.join(runs_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(manifest)} run dirs under {runs_dir}")


# ------------------------------------------------------------------ score

def run_visible(ws):
    p = subprocess.run([sys.executable, "-m", "tests.run"], cwd=ws,
                       capture_output=True, text=True, timeout=180)
    return p.returncode == 0


def run_hidden(ws, big):
    env = dict(os.environ)
    code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, '.'); "
            "import test_hidden; test_hidden.test_profile_full_runs(); "
            "print('ok')" % os.path.join(os.path.abspath(big),
                                         "hidden_tests"))
    p = subprocess.run([sys.executable, "-c", code], cwd=ws, env=env,
                       capture_output=True, text=True, timeout=180)
    return p.returncode == 0


def pyright_errors(ws):
    p = subprocess.run(["basedpyright", "--outputjson", "granite"],
                       cwd=ws, capture_output=True, text=True, timeout=600)
    d = json.loads(p.stdout)
    return d["summary"]["errorCount"]


def score_run(rdir, big, tasks):
    meta = json.load(open(os.path.join(rdir, "meta.json")))
    ws, tid = meta["workspace"], meta["task"]
    spec = tasks[tid]
    out = {"run": os.path.basename(rdir), "task": tid, "arm": meta["arm"],
           "k": meta["k"], "success": 0, "catastrophe": 0,
           "tool_calls": None, "notes": ""}
    rep_path = os.path.join(ws, "report.json")
    if not os.path.exists(rep_path):
        out["notes"] = "no report.json"
        return out
    try:
        rep = json.load(open(rep_path))
    except Exception:
        out["notes"] = "unparseable report.json"
        return out
    out["tool_calls"] = rep.get("assigned_tool_calls")
    ans = rep.get("answer") or {}
    vis = run_visible(ws)
    hid = run_hidden(ws, big)

    def src(rel):
        try:
            return open(os.path.join(ws, rel), encoding="utf-8").read()
        except OSError:
            return ""

    if spec["kind"] == "judge":
        mod, fn = spec["qual"].rsplit(".", 1)
        present = f"def {fn}(" in src(mod.replace(".", "/") + ".py")
        if spec["expect"] == "safe":
            ok = (not present) and vis and hid
            out["catastrophe"] = int(not (vis and hid))
        else:
            ok = present and vis and hid
            out["catastrophe"] = int(not present or not hid)
        out["success"] = int(ok)
        out["notes"] = (f"present={present} vis={vis} hid={hid} "
                        f"decision={str(rep.get('decision'))[:50]!r}")
    elif spec["kind"] == "nav":
        got = {(c.get("file", "").lstrip("./"), c.get("line"))
               for c in (ans.get("call_sites") or []) if isinstance(c, dict)}
        keyset = {tuple(x) for x in
                  {(p, ln) for p, ln in map(tuple, spec["sites"])}}
        tp = len(got & keyset)
        prec = tp / len(got) if got else 0
        rec = tp / len(keyset) if keyset else 0
        out["success"] = int(prec == 1.0 and rec == 1.0)
        out["notes"] = f"P={prec:.2f} R={rec:.2f} got={sorted(got)}"
    elif spec["kind"] == "bugs":
        errs = pyright_errors(ws)
        reported = {c.get("file", "").split("/")[-1]
                    for c in (ans.get("defects") or [])
                    if isinstance(c, dict)}
        keyfiles = {b["path"].split("/")[-1] for b in spec["bugs"]}
        recall = len(reported & keyfiles) / len(keyfiles)
        out["success"] = int(errs == 0 and vis and hid and recall == 1.0)
        out["catastrophe"] = int(not (vis and hid))
        out["notes"] = f"pyright={errs} recall={recall:.2f} vis={vis} hid={hid}"
    elif spec["kind"] == "hooks":
        got = {str(x) for x in (ans.get("also_called_statically") or [])}
        keyset = set(spec["also_static"])
        out["success"] = int(got == keyset and vis)
        out["notes"] = f"got={sorted(got)} want={sorted(keyset)}"
    elif spec["kind"] == "reach":
        t_ok = ans.get("target_reachable") is True
        d_ok = ans.get("decoy_reachable") is False
        chain = [str(x) for x in (ans.get("chain") or [])]
        chain_ok = bool(chain) and chain[-1].endswith(
            spec["target"].split(".")[-1])
        # validate consecutive edges against the key graph loosely: each
        # hop's callee name must be callable from the previous hop per key
        key_edges = {(e["caller"], e["callee"])
                     for e in load_key(big)["edges"]}
        hops_ok = all(
            any(c0 == kc or kc.startswith(c0 + ".") or c0.startswith(kc)
                for kc, _ in [(a, b)])
            for a, b in zip(chain, chain[1:])
            for c0 in [a]) if chain else False
        out["success"] = int(t_ok and d_ok and chain_ok and vis)
        out["notes"] = (f"target={ans.get('target_reachable')} "
                        f"decoy={ans.get('decoy_reachable')} "
                        f"chain_len={len(chain)}")
    return out


def score(big, runs_dir):
    tasks = json.load(open(os.path.join(big, "tasks_built.json")))
    manifest = json.load(open(os.path.join(runs_dir, "manifest.json")))
    rows = [score_run(os.path.join(runs_dir, rid), big, tasks)
            for rid in manifest]
    with open(os.path.join(runs_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by_arm = {}
    for r in rows:
        a = by_arm.setdefault(r["arm"], {"n": 0, "s": 0, "c": 0,
                                         "calls": [], "nr": 0})
        a["n"] += 1
        a["s"] += r["success"]
        a["c"] += r["catastrophe"]
        if r["tool_calls"] is not None:
            a["calls"].append(r["tool_calls"])
        if r["notes"] == "no report.json":
            a["nr"] += 1
    print(f"{'arm':8} {'n':>3} {'success':>8} {'catastr':>8} "
          f"{'avg_calls':>10} {'no_report':>10}")
    for arm, a in sorted(by_arm.items()):
        avg = sum(a["calls"]) / len(a["calls"]) if a["calls"] else 0
        print(f"{arm:8} {a['n']:>3} {a['s']:>8} {a['c']:>8} "
              f"{avg:>10.1f} {a['nr']:>10}")
    print("\nper task:")
    per = {}
    for r in rows:
        per.setdefault(r["task"], {}).setdefault(r["arm"], []).append(
            (r["success"], r["catastrophe"]))
    for t in sorted(tasks):
        cells = "  ".join(
            f"{arm}={sum(s for s, _ in per.get(t, {}).get(arm, []))}/"
            f"{len(per.get(t, {}).get(arm, []))}"
            f"{'!' * sum(c for _, c in per.get(t, {}).get(arm, []))}"
            for arm in ARMS)
        print(f"  {t:16} {cells}   (! = catastrophe)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "score"])
    ap.add_argument("--big", required=True)
    ap.add_argument("--runs-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "setup":
        setup(args.big, args.runs_dir)
    else:
        score(args.big, args.runs_dir)


if __name__ == "__main__":
    main()
