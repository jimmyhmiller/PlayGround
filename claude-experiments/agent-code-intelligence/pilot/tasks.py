"""Pilot task definitions + ground-truth key + run-dir setup + scorer.

Usage:
  python3 tasks.py setup --runs-dir DIR      # create run dirs (task x arm x k)
  python3 tasks.py score --runs-dir DIR      # score completed runs -> results.csv + summary
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

REPORT_SCHEMA = (
    'Before finishing, Write WORKSPACE/report.json as JSON:\n'
    '{"task_id": "<id>", "answer": <see task>, "decision": "<one line>",\n'
    ' "evidence": "<1-3 sentences>", "assigned_tool_calls": <int>,\n'
    ' "calls": ["<op and target>", ...],\n'
    ' "fell_back_to_reading_files": <true|false>}\n'
    "Your final chat message must be a single line."
)

TASKS = {
    "t1_dead_mathutil": dict(corpus="corpus", answer_hint=(
        '{"delete": ["<function names you deleted>"], '
        '"keep": ["<function names you kept>"]}'), prompt=(
        "The team suspects mill/mathutil.py contains dead code. Determine "
        "which of its top-level functions are safe to delete (i.e. "
        "transitively unused from anywhere outside that file, with no other "
        "way to be invoked). Delete exactly the safe ones; keep the rest. "
        "The test suite must still pass.")),
    "t2_dead_probe": dict(corpus="corpus", answer_hint=(
        '{"action": "deleted" | "kept", "reason": "<why>"}'), prompt=(
        "A teammate claims the function `strict_probe` in mill/validate.py "
        "is unused and wants it removed. Decide whether it is truly safe to "
        "delete. If safe, delete it; if not, keep it. The test suite must "
        "still pass afterwards.")),
    "t3_dead_hook": dict(corpus="corpus", answer_hint=(
        '{"action": "deleted" | "kept", "reason": "<why>"}'), prompt=(
        "A cleanup ticket proposes removing `reload_limits` from "
        "mill/config.py as dead code. Decide whether it is truly safe to "
        "delete. If safe, delete it; if not, keep it. The test suite must "
        "still pass afterwards.")),
    "t4_rename_process": dict(corpus="corpus", answer_hint=(
        '{"call_sites_updated": <int>}'), prompt=(
        "Rename the `process` method of the Engine class "
        "(mill/engine.py) to `run_one`, updating every place that calls "
        "it. The Writer class has a same-named `process` method that must "
        "NOT be renamed and whose callers must NOT be touched. The test "
        "suite must pass afterwards.")),
    "t5_rename_combine": dict(corpus="corpus", answer_hint=(
        '{"code_files_changed": <int>, "non_code_mentions_handled": '
        '["<file>", ...]}'), prompt=(
        "Rename the function `combine` (mill/merge.py) to `merge_records` "
        "across the project — including any non-code mentions "
        "(documentation etc.), which should be updated to match. The test "
        "suite must pass afterwards.")),
    "t6_type_bugs": dict(corpus="corpus_bugs", answer_hint=(
        '{"defects": [{"file": "...", "line": <int>, "what": "..."}]}'),
        prompt=(
        "CI's type gate says this repository is not clean, and a user "
        "reported a crash on an edge path. All tests currently pass — the "
        "defects are latent (on paths the tests never execute). Find every "
        "type-level defect, fix each one minimally (preserve the intended "
        "behavior of the surrounding code), and leave the repository "
        "type-clean with tests still passing.")),
    "t7_sig_change": dict(corpus="corpus", answer_hint=(
        '{"call_sites_updated": [{"file": "...", "line": <int>}]}'), prompt=(
        "Add a required boolean parameter `strict` (last parameter) to "
        "`apply_rules` in mill/transform.py. When `strict` is true, "
        "apply_rules must drop keys whose value is empty after the rules "
        "run. Update every existing call site: Engine.process passes "
        "strict=True, all other call sites pass strict=False. The test "
        "suite must pass afterwards.")),
    "t8_nav_refs": dict(corpus="corpus", answer_hint=(
        '{"call_sites": [{"file": "...", "line": <int>}, ...]}'), prompt=(
        "List every call site of the Engine class's `process` method "
        "(defined in mill/engine.py) — exact file and 1-based line number, "
        "excluding the definition itself. Do not include calls to other "
        "methods that merely share the name. Do not change any files.")),
    "t9_impact_delete": dict(corpus="corpus", answer_hint=(
        '{"still_pass": ["<test file>", ...], "fail": ["<test file>", '
        '...]}'), prompt=(
        "Hypothetical: the function `combine` in mill/merge.py is deleted "
        "outright (and its import line in mill/engine.py removed). Which of "
        "the three test files (tests/test_engine.py, tests/test_merge.py, "
        "tests/test_textmath.py) would still pass, and which would fail? "
        "Answer by analysis. Do NOT actually delete anything; do not "
        "change any files.")),
    "t10_chain": dict(corpus="corpus", answer_hint=(
        '{"chain": ["cli.main", "...", "textutil.clean"]}'), prompt=(
        "Explain, function by function, one call chain through which "
        "`main` in mill/cli.py causes `clean` in mill/textutil.py to "
        "execute. Give the chain as an ordered list of functions. Do not "
        "change any files.")),
}

RULES = """TOOL RULES (strict, part of the evaluation):
- For ALL code understanding and navigation (finding symbols, references,
  callers, types, diagnostics, impact), use ONLY the assigned tool
  documented at {doc}. Read that doc first.
- FORBIDDEN: grep/rg/find/awk/sed or any other text-search command, and
  bulk directory scanning (ls -R, reading every file one by one to
  search). Reading a SPECIFIC file you have already located, to view or
  edit it, is allowed.
- Edit files with your normal editing tools.
- Run the test suite with: cd {ws} && python3 -m tests.run
- Count every invocation of the assigned tool; if you ever answer an
  information need by reading files because the tool could not answer it,
  set fell_back_to_reading_files true in the report.
"""


def setup(runs_dir):
    os.makedirs(runs_dir, exist_ok=True)
    manifest = []
    for tid, spec in TASKS.items():
        for arm in ARMS:
            for k in range(1, K + 1):
                rid = f"{tid}__{arm}__k{k}"
                rdir = os.path.join(runs_dir, rid)
                ws = os.path.join(rdir, "workspace")
                if os.path.exists(rdir):
                    shutil.rmtree(rdir)
                shutil.copytree(os.path.join(HERE, spec["corpus"]), ws)
                doc = os.path.join(HERE, "docs",
                                   "QUARRY.md" if arm == "quarry" else "LSP.md")
                prompt = (
                    "You are one run in a controlled evaluation of "
                    "code-intelligence tooling for coding agents. Work in "
                    f"WORKSPACE = {ws}\n\n"
                    + RULES.format(doc=doc, ws=ws)
                    + f"\nTASK ({tid}):\n{spec['prompt']}\n\n"
                    + REPORT_SCHEMA.replace("<id>", tid).replace(
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


# ---------------------------------------------------------------- scoring

def run_tests(ws):
    p = subprocess.run([sys.executable, "-m", "tests.run"], cwd=ws,
                       capture_output=True, text=True, timeout=120)
    return p.returncode == 0


def pyright_errors(ws):
    p = subprocess.run(["basedpyright", "--outputjson", "mill", "tests"],
                       cwd=ws, capture_output=True, text=True, timeout=300)
    try:
        return json.load(io_str(p.stdout))["summary"]["errorCount"]
    except Exception:
        return json.loads(p.stdout)["summary"]["errorCount"]


def io_str(s):
    import io
    return io.StringIO(s)


def grep_ws(ws, pattern, py_only=True):
    hits = []
    for dp, dns, fns in os.walk(ws):
        dns[:] = [d for d in dns if not d.startswith(".")
                  and d != "__pycache__"]
        for fn in fns:
            if py_only and not fn.endswith(".py"):
                continue
            if fn == "report.json":
                continue
            full = os.path.join(dp, fn)
            try:
                for i, ln in enumerate(open(full, encoding="utf-8",
                                            errors="replace"), 1):
                    if re.search(pattern, ln):
                        hits.append((os.path.relpath(full, ws), i,
                                     ln.rstrip()))
            except OSError:
                pass
    return hits


T8_KEY = {("mill/cli.py", 14), ("mill/engine.py", 35),
          ("tests/test_engine.py", 17), ("tests/test_engine.py", 22)}


def score_run(rdir):
    meta = json.load(open(os.path.join(rdir, "meta.json")))
    ws = meta["workspace"]
    tid = meta["task"]
    rep_path = os.path.join(ws, "report.json")
    out = {"run": os.path.basename(rdir), "task": tid, "arm": meta["arm"],
           "k": meta["k"], "success": 0, "catastrophe": 0, "notes": "",
           "tool_calls": None, "fell_back": None}
    if not os.path.exists(rep_path):
        out["notes"] = "no report.json"
        return out
    try:
        rep = json.load(open(rep_path))
    except Exception:
        out["notes"] = "unparseable report.json"
        return out
    out["tool_calls"] = rep.get("assigned_tool_calls")
    out["fell_back"] = rep.get("fell_back_to_reading_files")
    ans = rep.get("answer") or {}
    tests_ok = run_tests(ws)

    def src(rel):
        try:
            return open(os.path.join(ws, rel), encoding="utf-8").read()
        except OSError:
            return ""

    if tid == "t1_dead_mathutil":
        mu = src("mill/mathutil.py")
        gone = "def legacy_norm" not in mu
        kept = "def clamp" in mu and "def checked_div" in mu
        out["success"] = int(gone and kept and tests_ok)
        out["catastrophe"] = int(not kept or (not tests_ok))
        out["notes"] = f"gone={gone} kept={kept} tests={tests_ok}"
    elif tid in ("t2_dead_probe", "t3_dead_hook"):
        fn = "strict_probe" if tid == "t2_dead_probe" else "reload_limits"
        rel = ("mill/validate.py" if tid == "t2_dead_probe"
               else "mill/config.py")
        present = f"def {fn}" in src(rel)
        out["success"] = int(present and tests_ok)
        out["catastrophe"] = int(not present)
        out["notes"] = f"kept={present} tests={tests_ok} " \
                       f"decision={rep.get('decision', '')[:40]!r}"
    elif tid == "t4_rename_process":
        eng, wri = src("mill/engine.py"), src("mill/writer.py")
        renamed = "def run_one" in eng and "def process(" not in eng
        writer_intact = "def process(" in wri
        callers = ("eng.run_one" in src("mill/cli.py")
                   and "self.run_one" in eng
                   and "eng.run_one" in src("tests/test_engine.py"))
        wcallers = "self.writer.process(" in eng and "self.process(" in wri
        out["success"] = int(renamed and writer_intact and callers
                             and wcallers and tests_ok)
        out["catastrophe"] = int(not writer_intact or not tests_ok)
        out["notes"] = (f"renamed={renamed} writer_intact={writer_intact} "
                        f"callers={callers} wcallers={wcallers} "
                        f"tests={tests_ok}")
    elif tid == "t5_rename_combine":
        code_resid = grep_ws(ws, r"\bcombine\b", py_only=True)
        readme = src("README.md")
        doc_ok = "merge_records" in readme and "combine" not in readme
        has_new = "def merge_records" in src("mill/merge.py")
        out["success"] = int(not code_resid and doc_ok and has_new
                             and tests_ok)
        out["catastrophe"] = int(not tests_ok)
        out["notes"] = (f"resid={len(code_resid)} doc_ok={doc_ok} "
                        f"tests={tests_ok}")
    elif tid == "t6_type_bugs":
        errs = pyright_errors(ws)
        sched = 'int(snapshot.get("attempt", "0"))' in src("mill/scheduler.py") \
            or re.search(r"self\.attempt\s*=\s*int\(", src("mill/scheduler.py"))
        writer_src = src("mill/writer.py")
        writer_fixed = "self.process(raw)" not in writer_src
        found = {(d.get("file", ""), d.get("line")) for d in
                 (ans.get("defects") or []) if isinstance(d, dict)}
        both_reported = (any("scheduler" in f for f, _ in found)
                         and any("writer" in f for f, _ in found))
        out["success"] = int(errs == 0 and tests_ok and sched
                             and writer_fixed and both_reported)
        out["catastrophe"] = int(not tests_ok)
        out["notes"] = (f"pyright_errs={errs} tests={tests_ok} "
                        f"sched_fix={bool(sched)} writer_fix={writer_fixed} "
                        f"reported_both={both_reported}")
    elif tid == "t7_sig_change":
        tr = src("mill/transform.py")
        sig_ok = re.search(r"def apply_rules\(.*strict: bool\)", tr)
        eng_ok = re.search(r"apply_rules\(.*(strict=True|,\s*True)\)",
                           src("mill/engine.py"))
        mrg_ok = re.search(r"apply_rules\(.*(strict=False|,\s*False)\)",
                           src("mill/merge.py"))
        out["success"] = int(bool(sig_ok and eng_ok and mrg_ok) and tests_ok)
        out["catastrophe"] = int(not tests_ok)
        out["notes"] = (f"sig={bool(sig_ok)} eng={bool(eng_ok)} "
                        f"mrg={bool(mrg_ok)} tests={tests_ok}")
    elif tid == "t8_nav_refs":
        got = {(c.get("file", "").lstrip("./"), c.get("line"))
               for c in (ans.get("call_sites") or []) if isinstance(c, dict)}
        tp = len(got & T8_KEY)
        prec = tp / len(got) if got else 0
        rec = tp / len(T8_KEY)
        out["success"] = int(prec == 1.0 and rec == 1.0)
        out["notes"] = f"P={prec:.2f} R={rec:.2f} got={sorted(got)}"
    elif tid == "t9_impact_delete":
        sp = {os.path.basename(x) for x in (ans.get("still_pass") or [])}
        fl = {os.path.basename(x) for x in (ans.get("fail") or [])}
        ok = (sp == {"test_textmath.py"}
              and fl == {"test_engine.py", "test_merge.py"})
        untouched = tests_ok
        out["success"] = int(ok and untouched)
        out["notes"] = f"still_pass={sorted(sp)} fail={sorted(fl)}"
    elif tid == "t10_chain":
        chain = [str(x).lower() for x in (ans.get("chain") or [])]
        joined = ">".join(chain)
        ok = ("main" in joined and
              ("process" in joined or "run_one" in joined) and
              "clean" in joined and
              ("apply_rules" in joined or "splay" in joined
               or "rule_names" in joined))
        out["success"] = int(ok and tests_ok)
        out["notes"] = f"chain={chain}"
    return out


def score(runs_dir):
    manifest = json.load(open(os.path.join(runs_dir, "manifest.json")))
    rows = [score_run(os.path.join(runs_dir, rid)) for rid in manifest]
    with open(os.path.join(runs_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by_arm = {}
    for r in rows:
        a = by_arm.setdefault(r["arm"], {"n": 0, "success": 0,
                                         "catastrophe": 0, "calls": [],
                                         "fell_back": 0, "no_report": 0})
        a["n"] += 1
        a["success"] += r["success"]
        a["catastrophe"] += r["catastrophe"]
        if r["tool_calls"] is not None:
            a["calls"].append(r["tool_calls"])
        if r["fell_back"]:
            a["fell_back"] += 1
        if r["notes"] == "no report.json":
            a["no_report"] += 1
    print(f"{'arm':8} {'n':>3} {'success':>8} {'catastr':>8} "
          f"{'avg_calls':>10} {'fell_back':>10} {'no_report':>10}")
    for arm, a in sorted(by_arm.items()):
        avg = sum(a["calls"]) / len(a["calls"]) if a["calls"] else 0
        print(f"{arm:8} {a['n']:>3} {a['success']:>8} "
              f"{a['catastrophe']:>8} {avg:>10.1f} {a['fell_back']:>10} "
              f"{a['no_report']:>10}")
    print("\nper task:")
    per = {}
    for r in rows:
        per.setdefault(r["task"], {}).setdefault(r["arm"], []).append(
            r["success"])
    for t in TASKS:
        cells = "  ".join(
            f"{arm}={sum(per.get(t, {}).get(arm, []))}/"
            f"{len(per.get(t, {}).get(arm, []))}" for arm in ARMS)
        print(f"  {t:22} {cells}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "score"])
    ap.add_argument("--runs-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "setup":
        setup(args.runs_dir)
    else:
        score(args.runs_dir)


if __name__ == "__main__":
    main()
