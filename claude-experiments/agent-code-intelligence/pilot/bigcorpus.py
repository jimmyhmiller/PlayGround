#!/usr/bin/env python3
"""Generate the 'granite' corpus: a large, layered Python codebase with
ground truth by construction, for the scaled quarry-vs-LSP eval.

Design goals (what the small pilot lacked):
  - too large to read wholesale (~100 modules, ~20k+ LOC)
  - heavy same-name method collisions (small method-name pool over many
    classes) so receiver resolution actually matters
  - dynamic dispatch via config + getattr, where the VISIBLE test suite
    does not exercise the dynamic paths (weak tests on purpose)
  - a HIDDEN test suite (emitted next to, not inside, the workspace) that
    does exercise them — the scorer's catastrophe detector
  - planted truly-dead functions, trap-dead functions (config/string
    referenced only), and latent type bugs on paths tests never run
  - an answer key: call sites, graph edges, dead lists, bug locations

Deterministic for a given seed. Usage:
  python3 bigcorpus.py --out DIR [--seed 7]
Emits DIR/workspace (the repo agents see), DIR/hidden_tests/,
DIR/key.json.
"""

import argparse
import json
import os
import random
import shutil

LAYERS = ["core", "store", "wire", "ops", "svc", "app"]
MODS_PER_LAYER = [32, 30, 26, 22, 16, 10]
METHOD_POOL = ["process", "handle", "resolve", "flush", "advance",
               "collect", "emit", "probe"]
VERBS = ["merge", "split", "pack", "scan", "fold", "trim", "route", "bind",
         "weigh", "stamp", "sift", "braid", "notch", "glaze", "tally"]
NOUNS = ["frame", "ledger", "chunk", "berth", "crate", "spool", "gasket",
         "quorum", "billet", "carton", "hopper", "sluice", "mantle",
         "pivot", "vellum"]
CLASS_NOUNS = ["Basin", "Crucible", "Gantry", "Lathe", "Mortar", "Pallet",
               "Quarry", "Rotor", "Silo", "Truss", "Vault", "Winch",
               "Anvil", "Bellows", "Chisel"]


class Gen:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.modules = {}            # rel -> dict(layer, name, funcs, classes)
        self.funcs = []              # (rel, qual, name) callable functions
        self.methods = []            # (rel, class, method)
        self.edges = []              # {caller, callee, path, line} filled late
        self.key = {"dead_safe": [], "dead_trap": [], "type_bugs": [],
                    "hooks": {"configured": [], "also_static": [],
                              "dynamic_only": []},
                    "collision_sites": {}, "chains": [], "impact": {}}
        self.used_names = set()

    def fresh_name(self, kind="func"):
        for attempt in range(1000):
            if kind == "func":
                n = f"{self.rng.choice(VERBS)}_{self.rng.choice(NOUNS)}"
                if attempt > 5:      # pool thinning: widen the space
                    n += f"_{self.rng.choice(NOUNS)}"
                if attempt > 50:
                    n += f"{self.rng.randint(2, 99)}"
            else:
                n = (self.rng.choice(CLASS_NOUNS)
                     + self.rng.choice(CLASS_NOUNS))
                if attempt > 50:
                    n += str(self.rng.randint(2, 99))
            if n not in self.used_names:
                self.used_names.add(n)
                return n
        raise RuntimeError("name space exhausted")

    # ------------------------------------------------------------ planning
    def plan(self):
        for li, layer in enumerate(LAYERS):
            for mi in range(MODS_PER_LAYER[li]):
                mod_name = f"{self.fresh_name('func')}"
                rel = f"granite/{layer}/{mod_name}.py"
                nfuncs = self.rng.randint(5, 8)
                nclasses = self.rng.randint(1, 3) if li > 0 else 1
                funcs = [self.fresh_name() for _ in range(nfuncs)]
                classes = []
                for _ in range(nclasses):
                    cname = self.fresh_name("class")
                    methods = self.rng.sample(METHOD_POOL,
                                              self.rng.randint(3, 4))
                    classes.append((cname, methods))
                self.modules[rel] = {"layer": li, "name": mod_name,
                                     "funcs": funcs, "classes": classes,
                                     "mod": rel[:-3].replace("/", ".")}
                for f in funcs:
                    self.funcs.append((rel, f))
                for cname, methods in classes:
                    for m in methods:
                        self.methods.append((rel, cname, m))

    def lower_mods(self, li):
        return [r for r, m in self.modules.items() if m["layer"] < li]

    # ------------------------------------------------------------ emission
    def emit(self, out):
        ws = os.path.join(out, "workspace")
        os.makedirs(ws, exist_ok=True)
        for layer in LAYERS:
            os.makedirs(os.path.join(ws, "granite", layer), exist_ok=True)
            open(os.path.join(ws, "granite", layer, "__init__.py"),
                 "w").close()
        open(os.path.join(ws, "granite", "__init__.py"), "w").close()

        rng = self.rng
        # choose special roles first
        all_func_refs = list(self.funcs)
        rng.shuffle(all_func_refs)
        dead_safe = all_func_refs[:10]
        dead_trap = all_func_refs[10:16]
        bug_hosts = all_func_refs[16:21]
        self.key["dead_safe"] = [f"{self.modules[r]['mod']}.{f}"
                                 for r, f in dead_safe]
        self.key["dead_trap"] = [f"{self.modules[r]['mod']}.{f}"
                                 for r, f in dead_trap]
        special = {(r, f) for r, f in dead_safe + dead_trap}
        callable_funcs = [(r, f) for r, f in self.funcs
                          if (r, f) not in special]

        # write每 module
        for rel, mod in sorted(self.modules.items()):
            li = mod["layer"]
            lines = [f'"""{mod["name"]} — layer {LAYERS[li]} utilities."""',
                     ""]
            lower = self.lower_mods(li)
            imports, imported_funcs, imported_classes = [], [], []
            if lower:
                picks = rng.sample(lower, min(len(lower),
                                              rng.randint(2, 4)))
                for prel in picks:
                    pm = self.modules[prel]
                    fs = [f for f in pm["funcs"]
                          if (prel, f) in set(callable_funcs)]
                    cs = [c for c, _ in pm["classes"]]
                    take_f = rng.sample(fs, min(len(fs), 2)) if fs else []
                    take_c = [rng.choice(cs)] if cs and rng.random() < .5 \
                        else []
                    if take_f or take_c:
                        names = ", ".join(sorted(take_f + take_c))
                        imports.append(f"from {pm['mod']} import {names}")
                        imported_funcs += [(prel, f) for f in take_f]
                        imported_classes += [(prel, c) for c in take_c]
            lines += imports + [""] if imports else []

            def body_calls(caller_qual, indent="    "):
                out_lines, val = [], "seed"
                targets = []
                if imported_funcs:
                    targets = rng.sample(
                        imported_funcs,
                        min(len(imported_funcs), rng.randint(1, 2)))
                for prel, fn in targets:
                    out_lines.append(f"{indent}{'val = '}{fn}(val)")
                    self.edges.append({"caller": caller_qual,
                                       "callee":
                                       f"{self.modules[prel]['mod']}.{fn}",
                                       "path": rel})
                if imported_classes and rng.random() < .6:
                    prel, cls = rng.choice(imported_classes)
                    pm = self.modules[prel]
                    meth = rng.choice([m for c, ms in pm["classes"]
                                       if c == cls for m in ms])
                    out_lines.append(f"{indent}box = {cls}()")
                    out_lines.append(f"{indent}val = box.{meth}(val)")
                    self.edges.append({"caller": caller_qual,
                                       "callee": f"{pm['mod']}.{cls}.{meth}",
                                       "path": rel})
                return out_lines

            # classes
            for cname, methods in mod["classes"]:
                lines.append(f"class {cname}:")
                lines.append(f"    def __init__(self) -> None:")
                lines.append(f"        self.gauge: int = {rng.randint(2, 9)}")
                lines.append(f"        self.label: str = \"{cname.lower()}\"")
                lines.append("")
                for meth in methods:
                    qual = f"{mod['mod']}.{cname}.{meth}"
                    lines.append(
                        f"    def {meth}(self, val: int) -> int:")
                    lines += body_calls(qual, "        ")
                    for _ in range(rng.randint(1, 3)):
                        a, b = rng.randint(2, 9), rng.randint(1, 30)
                        lines.append(f"        val = (val * {a}) % {b + 40}")
                    lines.append(
                        f"        return val * self.gauge + "
                        f"{rng.randint(1, 7)}")
                    lines.append("")
            # functions
            for fn in mod["funcs"]:
                qual = f"{mod['mod']}.{fn}"
                is_dead_safe = (rel, fn) in set(dead_safe)
                is_dead_trap = (rel, fn) in set(dead_trap)
                is_bug = (rel, fn) in set(bug_hosts)
                lines.append(f"def {fn}(val: int) -> int:")
                if is_dead_safe:
                    lines.append('    """Legacy pathway kept for '
                                 'reference."""')
                if is_dead_trap:
                    lines.append('    """Maintenance hook (see profile '
                                 'configuration)."""')
                lines += body_calls(qual)
                for _ in range(rng.randint(2, 5)):
                    a, b = rng.randint(2, 9), rng.randint(1, 30)
                    op = rng.choice(["+", "*", "-", "%%"]).replace("%%", "%")
                    lines.append(f"    val = (val {op} {a}) + {b}")
                if is_bug:
                    # latent type bug: str assigned into int flow on a
                    # branch ordinary values never take
                    lines.append(f"    if val == -987654:")
                    lines.append(f"        val = \"drained\"")
                    self.key["type_bugs"].append(
                        {"module": mod["mod"], "function": fn, "path": rel})
                lines.append(f"    return val + {rng.randint(1, 9)}")
                lines.append("")
            with open(os.path.join(ws, rel), "w") as f:
                f.write("\n".join(lines).rstrip() + "\n")

        self._emit_registry(ws, dead_trap, callable_funcs, rng)
        self._emit_entry(ws, rng)
        self._emit_tests(ws, out)
        self._emit_docs(ws)
        self._locate_key(ws)
        with open(os.path.join(out, "key.json"), "w") as f:
            json.dump(self.key, f, indent=1)

    # ------------------------------------------------------- special files
    def _emit_registry(self, ws, dead_trap, callable_funcs, rng):
        # hooks: trap-dead funcs are configured (dynamic-only); plus a few
        # normal funcs configured AND statically called elsewhere
        also_static = rng.sample(callable_funcs, 4)
        hooks = [(r, f) for r, f in dead_trap] + also_static
        self.key["hooks"]["configured"] = \
            [f"{self.modules[r]['mod']}.{f}" for r, f in hooks]
        self.key["hooks"]["dynamic_only"] = \
            [f"{self.modules[r]['mod']}.{f}" for r, f in dead_trap]
        self.key["hooks"]["also_static"] = \
            [f"{self.modules[r]['mod']}.{f}" for r, f in also_static]
        lines = ['"""Profile-driven maintenance hook dispatch."""',
                 "", "import importlib", "",
                 "def run_profile_hooks(settings_path: str, val: int) -> int:",
                 "    names = []",
                 "    with open(settings_path, encoding=\"utf-8\") as f:",
                 "        for line in f:",
                 "            line = line.strip()",
                 "            if line.startswith(\"hook:\"):",
                 "                names.append(line.split(\":\", 1)[1].strip())",
                 "    for name in names:",
                 "        mod_name, _, fn_name = name.rpartition(\".\")",
                 "        mod = importlib.import_module(mod_name)",
                 "        val = getattr(mod, fn_name)(val)",
                 "    return val", ""]
        with open(os.path.join(ws, "granite", "app", "profile_hooks.py"),
                  "w") as f:
            f.write("\n".join(lines))
        cfg = ["# maintenance profile — executed with --profile full only"]
        for r, fn in hooks:
            cfg.append(f"hook: {self.modules[r]['mod']}.{fn}")
        with open(os.path.join(ws, "maintenance.cfg"), "w") as f:
            f.write("\n".join(cfg) + "\n")

    def _emit_entry(self, ws, rng):
        # main entry chains through the top app modules; also a deep chain
        top = [r for r, m in self.modules.items() if m["layer"] == 5]
        chain_mods = rng.sample(top, 3)
        calls = []
        for r in chain_mods:
            m = self.modules[r]
            fn = m["funcs"][0]
            calls.append((m["mod"], fn))
        lines = ['"""granite entrypoint."""', "", "import sys", ""]
        for mod, fn in calls:
            lines.append(f"from {mod} import {fn}")
        lines.append("from granite.app.profile_hooks import run_profile_hooks")
        lines += ["", "",
                  "def main(argv: list[str]) -> int:",
                  "    val = int(argv[0]) if argv else 3"]
        for mod, fn in calls:
            lines.append(f"    val = {fn}(val)")
            self.edges.append({"caller": "granite.main", "callee":
                               f"{mod}.{fn}", "path": "granite/main.py"})
        lines += ["    if \"--profile\" in argv and \"full\" in argv:",
                  "        val = run_profile_hooks(\"maintenance.cfg\", val)",
                  "    return val % 97", "", "",
                  "if __name__ == \"__main__\":",
                  "    print(main(sys.argv[1:]))", ""]
        with open(os.path.join(ws, "granite", "main.py"), "w") as f:
            f.write("\n".join(lines))

    def _emit_tests(self, ws, out):
        # visible tests: static paths only, NO profile hooks
        vis = ['"""Smoke tests (fast suite)."""',
               "from granite.main import main", "", "",
               "def test_main_default():",
               "    assert isinstance(main([]), int)", "",
               "def test_main_seeded():",
               "    assert main([\"5\"]) == main([\"5\"])", ""]
        os.makedirs(os.path.join(ws, "tests"), exist_ok=True)
        open(os.path.join(ws, "tests", "__init__.py"), "w").close()
        with open(os.path.join(ws, "tests", "test_smoke.py"), "w") as f:
            f.write("\n".join(vis))
        runner = ["import sys, traceback",
                  "from tests import test_smoke",
                  "mods = [test_smoke]",
                  "passed, failed = 0, []",
                  "for m in mods:",
                  "    for n in sorted(dir(m)):",
                  "        if n.startswith('test_'):",
                  "            try:",
                  "                getattr(m, n)(); passed += 1",
                  "            except Exception:",
                  "                failed.append(f'{m.__name__}.{n}')",
                  "                traceback.print_exc()",
                  "print(f'{passed} passed, {len(failed)} failed')",
                  "sys.exit(1 if failed else 0)"]
        with open(os.path.join(ws, "tests", "run.py"), "w") as f:
            f.write("\n".join(runner) + "\n")
        # hidden suite: exercises the profile hooks (catastrophe detector)
        hid = ['"""HIDDEN scoring suite — not shipped to agents."""',
               "from granite.main import main", "", "",
               "def test_profile_full_runs():",
               "    assert isinstance(main([\"4\", \"--profile\", "
               "\"full\"]), int)", ""]
        os.makedirs(os.path.join(out, "hidden_tests"), exist_ok=True)
        with open(os.path.join(out, "hidden_tests", "test_hidden.py"),
                  "w") as f:
            f.write("\n".join(hid))

    def _emit_docs(self, ws):
        # README references a handful of function names (rename collateral)
        mention = self.key["dead_trap"][0].split(".")[-1]
        with open(os.path.join(ws, "README.md"), "w") as f:
            f.write(
                "# granite\n\nA layered batch-processing library "
                "(core -> store -> wire -> ops -> svc -> app).\n\n"
                "Run `python3 -m granite.main 5`. The fast test suite is "
                "`python3 -m tests.run`.\nMaintenance profiles configure "
                f"hooks such as `{mention}` in `maintenance.cfg`; they run "
                "only with `--profile full`.\n")

    # ------------------------------------------------------------- key
    def _locate_key(self, ws):
        """Resolve edges to real line numbers and record collision sites."""
        line_index = {}
        for rel in list(self.modules) + ["granite/main.py"]:
            with open(os.path.join(ws, rel), encoding="utf-8") as f:
                line_index[rel] = f.read().splitlines()
        located = []
        for e in self.edges:
            name = e["callee"].split(".")[-1]
            hits = [i + 1 for i, ln in enumerate(line_index[e["path"]])
                    if f"{name}(val)" in ln or f".{name}(val)" in ln
                    or f"= {name}(val)" in ln]
            located.append(dict(e, candidate_lines=hits))
        self.key["edges"] = located
        # collision stats + chosen nav targets
        by_method = {}
        for rel, cname, meth in self.methods:
            by_method.setdefault(meth, []).append(
                f"{self.modules[rel]['mod']}.{cname}")
        self.key["collision_counts"] = {m: len(v)
                                        for m, v in by_method.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    g = Gen(args.seed)
    g.plan()
    g.emit(args.out)
    n_files = sum(len(fs) for _, _, fs in os.walk(
        os.path.join(args.out, "workspace")))
    loc = 0
    for dp, _, fns in os.walk(os.path.join(args.out, "workspace")):
        for fn in fns:
            if fn.endswith(".py"):
                loc += sum(1 for _ in open(os.path.join(dp, fn)))
    print(f"emitted {n_files} files, {loc} LOC python -> {args.out}")


if __name__ == "__main__":
    main()
