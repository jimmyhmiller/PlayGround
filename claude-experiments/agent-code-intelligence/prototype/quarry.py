#!/usr/bin/env python3
"""quarry — prototype of the Quarry code intelligence protocol (SPEC.md v0.1).

Stdlib-only. Implements for Python codebases:
  L0  search:regex over all text files
  L1  resolve/describe/refs/graph/search:symbol from an ast-built index
  +   check (CPython compile + optional pyflakes), impact, overlay.*,
      edit.rename (diff-returning, conflict-surfacing), capabilities/doctor,
      batch, budgets with deterministic elision, anchored locators.

Every response is wrapped in the mandatory epistemics envelope (SPEC §2.5).
Honesty rules: this indexer is *syntactic* — reference matching is by name,
so envelopes report confidence "syntactic" and never claim "compiler" except
for CPython parse diagnostics in `check`.
"""

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import keyword
import os
import re
import select
import subprocess
import sys
import time
import tokenize as py_tokenize
from difflib import unified_diff

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".quarry", "target",
             "venv", ".venv", "dist", "build", ".mypy_cache", ".pytest_cache"}
TEXT_EXT = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".json", ".cfg",
            ".ini", ".rst", ".sh", ".js", ".ts", ".rs", ".go", ".html", ".css"}
MAX_FILE_BYTES = 1_000_000
PROTO_VERSION = "0.1-prototype"


class QuarryError(Exception):
    def __init__(self, code, detail=None, **extra):
        self.code = code
        self.detail = detail
        self.extra = extra
        super().__init__(f"{code}: {detail}")


# ---------------------------------------------------------------- workspace

class Workspace:
    def __init__(self, root, overlay_id=None):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise QuarryError("WORKSPACE_NOT_FOUND",
                              f"no such directory: {self.root}")
        self.overlay = {}          # relpath -> content (str)
        self.overlay_id = overlay_id
        self.overlay_base = None
        self.contents = None       # atomic snapshot of disk (rel -> str)
        self.unstable = False      # True if disk kept changing during load
        if overlay_id:
            path = self._overlay_path(overlay_id)
            if not os.path.exists(path):
                raise QuarryError("SNAPSHOT_GONE",
                                  f"overlay {overlay_id} not found")
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
            self.overlay = blob["files"]
            self.overlay_base = blob.get("base")

    def _overlay_path(self, oid):
        return os.path.join(self.root, ".quarry", "overlays",
                            oid.replace(":", "_") + ".json")

    def walk(self):
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for fn in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, fn), self.root)
                ext = os.path.splitext(fn)[1].lower()
                if ext in TEXT_EXT or fn in ("Makefile", "Dockerfile"):
                    full = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(full) <= MAX_FILE_BYTES:
                            out.append(rel)
                    except OSError:
                        pass
        # overlay may introduce files that don't exist on disk
        for rel in self.overlay:
            if rel not in out:
                out.append(rel)
        return sorted(out)

    def _read_disk(self, rel):
        full = os.path.join(self.root, rel)
        with open(full, encoding="utf-8", errors="replace") as f:
            return f.read()

    def ensure_loaded(self):
        """Atomic snapshot load: read every file, then re-read to verify no
        file changed during the scan. A concurrent editor can otherwise
        produce a 'torn' cross-file view served with full confidence (the
        0.4% stale hit in the stress hammer). Instability after retries is
        disclosed in the snapshot, never silent."""
        if self.contents is not None:
            return
        files = [r for r in self.walk() if r not in self.overlay]
        contents = {}
        for _ in range(3):
            contents = {}
            for rel in files:
                try:
                    contents[rel] = self._read_disk(rel)
                except OSError:
                    pass
            stable = True
            for rel in list(contents):
                try:
                    if self._read_disk(rel) != contents[rel]:
                        stable = False
                        break
                except OSError:
                    stable = False
                    break
            if stable:
                self.contents = contents
                return
        self.contents = contents
        self.unstable = True

    def read(self, rel):
        if rel in self.overlay:
            return self.overlay[rel]
        self.ensure_loaded()
        if rel in self.contents:
            return self.contents[rel]
        return self._read_disk(rel)

    def exists(self, rel):
        if rel in self.overlay:
            return True
        self.ensure_loaded()
        return rel in self.contents or \
            os.path.exists(os.path.join(self.root, rel))

    def materialize(self, tmpdir):
        """Write the workspace view (disk + overlay) into tmpdir so batch
        tools that only read disk (mypy) can see hypothetical worlds."""
        for rel in self.walk():
            dst = os.path.join(tmpdir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                f.write(self.read(rel))

    def snapshot(self):
        self.ensure_loaded()
        h = hashlib.sha256()
        files = self.walk()
        for rel in files:
            try:
                h.update(rel.encode())
                h.update(hashlib.sha256(self.read(rel).encode()).digest())
            except OSError:
                pass
        sid = "snap:" + h.hexdigest()[:12]
        info = {"id": sid, "file_count": len(files)}
        if self.unstable:
            info["unstable"] = True
        try:
            rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 cwd=self.root, capture_output=True, text=True,
                                 timeout=5)
            if rev.returncode == 0:
                info["vcs"] = "git"
                info["rev"] = rev.stdout.strip()
        except Exception:
            pass
        if self.overlay_id:
            info["overlay"] = self.overlay_id
        return info


# ---------------------------------------------------------------- indexing

def module_name(rel):
    mod = rel[:-3] if rel.endswith(".py") else rel
    mod = mod.replace(os.sep, ".").replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


class _Indexer(ast.NodeVisitor):
    """One file. Collects symbols and occurrences with enclosing qualnames.

    Receiver inference (the L2 upgrade): a global class registry plus
    local type tracking — `v = ClassName()`, annotated params/vars,
    `self`, and direct `ClassName().meth()` chains — lets attribute
    occurrences carry `recv` (the receiver's class qualname) when
    provable. Unknown receivers stay None and are disclosed downstream.
    """

    def __init__(self, rel, source, index, registry=None):
        self.rel = rel
        self.mod = module_name(rel)
        self.lines = source.splitlines()
        self.index = index
        self.stack = [self.mod]           # qualname stack
        self.class_stack = []             # ClassDef qualnames
        self.call_marks = {}              # id(func_node) -> n_args
        self.registry = registry or {}    # module -> {classname: qualname}
        self.class_env = dict(self.registry.get(self.mod, {}))
        self.type_envs = [{}]             # var name -> class qualname

    # -- receiver-inference helpers
    def _ann_class(self, ann):
        if isinstance(ann, ast.Name):
            return self.class_env.get(ann.id)
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            return self.class_env.get(ann.value)
        return None

    def _value_class(self, value):
        """Class qualname of an expression's value, when provable."""
        if isinstance(value, ast.Name):
            for env in reversed(self.type_envs):
                if value.id in env:
                    return env[value.id]
            return None
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            return self.class_env.get(value.func.id)
        return None

    # -- helpers
    def qual(self, name):
        return ".".join(self.stack + [name])

    def line_text(self, lineno):
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].rstrip()
        return ""

    def add_symbol(self, name, kind, node, signature=None, doc=None):
        q = self.qual(name)
        if q in self.index["symbols"]:
            return q
        self.index["symbols"][q] = {
            "sym": "py:" + q,
            "display": ".".join(q.split(".")[-2:]),
            "qualname": q, "kind": kind, "lang": "python",
            "def": {"path": self.rel, "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "text": self.line_text(node.lineno)},
            "signature": signature or self.line_text(node.lineno).strip(),
            "doc": doc,
            "visibility": "private" if name.startswith("_") else "public",
        }
        return q

    def add_occ(self, name, node, role, n_args=None, recv=None):
        occ = {"name": name, "path": self.rel, "line": node.lineno,
               "col": node.col_offset, "role": role,
               "enclosing": ".".join(self.stack),
               "text": self.line_text(node.lineno)}
        if n_args is not None:
            occ["n_args"] = n_args
        if recv is not None:
            occ["recv"] = recv
        self.index["occurrences"].append(occ)

    # -- visitors
    def visit_Call(self, node):
        self.call_marks[id(node.func)] = (
            len(node.args), len(node.keywords))
        self.generic_visit(node)

    def _def(self, node, kind):
        doc = ast.get_docstring(node)
        sig = self.line_text(node.lineno).strip()
        # decorators shift lineno of the name line in py>=3.8? lineno is the
        # def/class line itself, decorators have their own lines. Fine.
        q = self.add_symbol(node.name, kind, node, signature=sig, doc=doc)
        # definition occurrence must point at the identifier, not the
        # def/class keyword, or col-verified rewrites of the def line fail
        line = self.line_text(node.lineno)
        m = re.search(r"\b(?:def|class)\s+(" + re.escape(node.name) + r")\b",
                      line)
        fake = ast.Name(id=node.name, ctx=ast.Load())
        fake.lineno = node.lineno
        fake.col_offset = (len(line[:m.start(1)].encode("utf-8"))
                           if m else node.col_offset)
        self.add_occ(node.name, fake, "definition")
        self.stack.append(node.name)
        if kind == "class":
            self.class_stack.append(q)
        if kind != "class":
            env = {}
            args = getattr(node, "args", None)
            if args is not None:
                for a in (args.posonlyargs + args.args + args.kwonlyargs):
                    cls = self._ann_class(a.annotation) if a.annotation \
                        else None
                    if cls:
                        env[a.arg] = cls
                if self.class_stack and (args.posonlyargs + args.args):
                    first = (args.posonlyargs + args.args)[0].arg
                    if first in ("self", "cls"):
                        env[first] = self.class_stack[-1]
            self.type_envs.append(env)
        self.generic_visit(node)
        if kind != "class":
            self.type_envs.pop()
        if kind == "class":
            self.class_stack.pop()
        self.stack.pop()

    def visit_ClassDef(self, node):
        self._def(node, "class")

    def visit_FunctionDef(self, node):
        kind = "method" if (self.class_stack and
                            ".".join(self.stack) == self.class_stack[-1]) \
            else "function"
        self._def(node, kind)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            cls = self._ann_class(node.annotation)
            if cls:
                self.type_envs[-1][node.target.id] = cls
        self.generic_visit(node)

    def visit_Assign(self, node):
        # receiver inference: v = ClassName(...) / v = other_typed_var
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            cls = self._value_class(node.value)
            tname = node.targets[0].id
            if cls:
                self.type_envs[-1][tname] = cls
            else:
                # reassignment to something unprovable kills the binding
                self.type_envs[-1].pop(tname, None)
        # self.X = ... inside a method -> attribute symbol on the class
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self" and self.class_stack):
                cls_q = self.class_stack[-1]
                q = cls_q + "." + tgt.attr
                if q not in self.index["symbols"]:
                    self.index["symbols"][q] = {
                        "sym": "py:" + q,
                        "display": ".".join(q.split(".")[-2:]),
                        "qualname": q, "kind": "attribute", "lang": "python",
                        "def": {"path": self.rel, "line": tgt.lineno,
                                "end_line": tgt.lineno,
                                "text": self.line_text(tgt.lineno)},
                        "signature": self.line_text(tgt.lineno).strip(),
                        "doc": None,
                        "visibility": "private" if tgt.attr.startswith("_")
                        else "public",
                    }
        self.generic_visit(node)

    def visit_Name(self, node):
        role = "read"
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            role = "write"
        mark = self.call_marks.get(id(node))
        if mark:
            self.add_occ(node.id, node, "call", n_args=sum(mark))
        else:
            self.add_occ(node.id, node, role)

    def visit_Attribute(self, node):
        role = "read"
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            role = "write"
        mark = self.call_marks.get(id(node))
        recv = self._value_class(node.value)
        # col of the attribute name itself: end of node minus len(attr)
        fake = ast.Name(id=node.attr, ctx=node.ctx)
        fake.lineno = node.end_lineno
        fake.col_offset = node.end_col_offset - len(node.attr)
        if mark:
            self.add_occ(node.attr, fake, "call", n_args=sum(mark),
                         recv=recv)
        else:
            self.add_occ(node.attr, fake, role, recv=recv)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.index["imports"].append(
                {"path": self.rel, "module": alias.name, "line": node.lineno})
            # the alias node carries the imported name's own position;
            # using the statement's col would break col-verified rewrites
            fake = ast.Name(id=alias.name.split(".")[-1], ctx=ast.Load())
            fake.lineno = getattr(alias, "lineno", node.lineno)
            fake.col_offset = getattr(alias, "col_offset", node.col_offset)
            self.add_occ(alias.name.split(".")[-1], fake, "import")

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        if node.level:      # relative import: resolve against this module
            base = self.mod.split(".")
            base = base[: len(base) - node.level]
            mod = ".".join(base + ([mod] if mod else []))
        self.index["imports"].append(
            {"path": self.rel, "module": mod, "line": node.lineno})
        for alias in node.names:
            # imported classes join the receiver-inference environment
            if alias.name in self.registry.get(mod, {}):
                self.class_env[alias.asname or alias.name] = \
                    self.registry[mod][alias.name]
            fake = ast.Name(id=alias.name, ctx=ast.Load())
            fake.lineno = getattr(alias, "lineno", node.lineno)
            fake.col_offset = getattr(alias, "col_offset", node.col_offset)
            self.add_occ(alias.name, fake, "import")


def build_index(ws):
    t0 = time.time()
    index = {"symbols": {}, "occurrences": [], "imports": [],
             "files": [], "errored": [], "excluded": 0}
    # pass 1: global class registry so receiver inference can resolve
    # imported class names across files
    registry, trees = {}, {}
    for rel in ws.walk():
        if not rel.endswith(".py"):
            continue
        try:
            trees[rel] = ast.parse(ws.read(rel))
        except SyntaxError:
            continue
        mod = module_name(rel)
        for node in ast.walk(trees[rel]):
            if isinstance(node, ast.ClassDef):
                registry.setdefault(mod, {})[node.name] = \
                    f"{mod}.{node.name}"
    for rel in ws.walk():
        if not rel.endswith(".py"):
            continue
        index["files"].append(rel)
        source = ws.read(rel)
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            index["errored"].append({"path": rel, "line": e.lineno,
                                     "message": str(e.msg)})
            continue
        mod = module_name(rel)
        node = ast.Module(body=[], type_ignores=[])
        node.lineno, node.end_lineno = 1, max(1, len(source.splitlines()))
        ix = _Indexer(rel, source, index, registry=registry)
        # module symbol
        index["symbols"].setdefault(mod, {
            "sym": "py:" + mod, "display": mod, "qualname": mod,
            "kind": "module", "lang": "python",
            "def": {"path": rel, "line": 1,
                    "end_line": node.end_lineno, "text": ""},
            "signature": mod, "doc": ast.get_docstring(tree),
            "visibility": "public"})
        ix.visit(tree)
    index["build_ms"] = int((time.time() - t0) * 1000)
    return index


def load_index(ws):
    """Cache-as-daemon: reuse a cached index when the snapshot id matches."""
    snap = ws.snapshot()
    cache_dir = os.path.join(ws.root, ".quarry", "cache")
    cache_file = os.path.join(cache_dir, snap["id"].replace(":", "_") + ".json")
    if not ws.overlay_id and os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as f:
                idx = json.load(f)
            idx["cache"] = "warm"
            return snap, idx
        except Exception:
            pass
    idx = build_index(ws)
    idx["cache"] = "cold"
    if not ws.overlay_id:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(idx, f)
        except OSError:
            pass
    return snap, idx


# ---------------------------------------------------------------- envelope

def make_envelope(snap, idx, result, status="complete", confidence="syntactic",
                  elided=None, notes=None):
    env = {
        "status": status,
        "confidence": confidence,
        "index": {"state": "fresh", "age_s": 0,
                  "built_pct": 100, "cache": idx.get("cache", "cold"),
                  "build_ms": idx.get("build_ms", 0)},
        "coverage": {"files_indexed": len(idx["files"]),
                     "files_errored": [e["path"] for e in idx["errored"]]},
        "snapshot": snap,
        "result": result,
    }
    if idx["errored"]:
        env["status"] = "partial" if status == "complete" else status
        env["coverage"]["errored_detail"] = idx["errored"]
    if snap.get("unstable"):
        # the workspace kept changing during the atomic load retries
        env["status"] = "partial" if env["status"] == "complete" \
            else env["status"]
        notes = list(notes or [])
        notes.append("workspace was being modified during snapshot load "
                     "(3 stability retries failed): answers reflect a "
                     "possibly-torn view — re-query when writes settle")
    if elided and elided.get("count"):
        env["elided"] = elided
    if notes:
        env["notes"] = notes
    return env


def error_envelope(code, detail=None, **extra):
    err = {"code": code}
    if detail:
        err["detail"] = detail
    err.update(extra)
    return {"status": "error", "confidence": "textual",
            "index": {"state": "fresh"}, "error": err}


# ---------------------------------------------------------------- resolving

def resolve_symbols(idx, name=None, sym=None, kinds=None, match="auto"):
    """Returns symbol dict copies annotated with how they matched
    ("match": exact|suffix|fuzzy) so callers can disclose non-exact
    resolution instead of silently substituting a different symbol."""
    symbols = idx["symbols"]
    if sym:
        q = sym[3:] if sym.startswith("py:") else sym
        hit = symbols.get(q)
        return [dict(hit, match="exact")] if hit else []
    if not name:
        return []
    exact = [s for s in symbols.values() if s["qualname"] == name]
    if exact and match in ("auto", "exact"):
        out = [dict(s, match="exact") for s in exact]
    else:
        suffix = [s for s in symbols.values()
                  if s["qualname"].endswith("." + name)
                  or s["qualname"].split(".")[-1] == name]
        if suffix and match in ("auto", "suffix", "exact"):
            out = [dict(s, match="suffix") for s in suffix]
        elif match in ("auto", "fuzzy"):
            low = name.lower()
            out = [dict(s, match="fuzzy") for s in symbols.values()
                   if low in s["qualname"].lower()]
        else:
            out = []
    if kinds:
        out = [s for s in out if s["kind"] in kinds]
    return sorted(out, key=lambda s: s["qualname"])


def resolve_unique(ws, idx, req):
    at = req.get("at")
    if at and (req.get("name") or req.get("sym")):
        raise QuarryError("BAD_REQUEST",
                          "pass exactly one of at | name | sym")
    if at:
        return symbol_at(ws, idx, at)
    cands = resolve_symbols(idx, req.get("name"), req.get("sym"),
                            req.get("kinds"), req.get("match", "auto"))
    if not cands:
        raise QuarryError("NOT_FOUND",
                          f"no symbol matches {req.get('name') or req.get('sym')!r}")
    # a lone fuzzy hit is a DIFFERENT name than asked for: never silently
    # substitute it under a completeness promise — surface it instead
    if cands[0].get("match") == "fuzzy" and req.get("match") != "fuzzy":
        raise QuarryError(
            "NOT_FOUND",
            f"no exact or suffix match for {req.get('name')!r}",
            nearest=[c["qualname"] for c in cands[:5]],
            hint="pass \"match\": \"fuzzy\" to accept a fuzzy match")
    if len(cands) > 1:
        raise QuarryError(
            "AMBIGUOUS", f"{len(cands)} symbols match",
            candidates=[{"sym": c["sym"], "kind": c["kind"],
                         "def": c["def"]} for c in cands])
    return cands[0]


def verify_anchor(ws, at):
    path, line = at.get("path"), at.get("line")
    anchor = at.get("anchor")
    window = at.get("window", 0)
    if not ws.exists(path):
        raise QuarryError("NOT_FOUND", f"no such file: {path}")
    lines = ws.read(path).splitlines()
    if not (1 <= line <= len(lines)):
        raise QuarryError("ANCHOR_MISMATCH",
                          f"{path} has {len(lines)} lines, requested {line}")
    if anchor:
        lo = max(0, line - 1 - window)
        hi = min(len(lines), line + window)
        for i in range(lo, hi):
            if anchor in lines[i]:
                return i + 1, lines[i]
        raise QuarryError(
            "ANCHOR_MISMATCH",
            f"anchor not found at {path}:{line}",
            actual_line_text=lines[line - 1].rstrip())
    return line, lines[line - 1]


def symbol_at(ws, idx, at):
    # verify the anchor first: a stale locator must error, never lie
    line, _ = verify_anchor(ws, at)
    anchor = at.get("anchor", "")
    occs = [o for o in idx["occurrences"]
            if o["path"] == at["path"] and o["line"] == line]
    if not occs:
        raise QuarryError("NOT_FOUND",
                          f"no symbol occurrence at {at['path']}:{line}")
    if anchor:
        # the anchor must bind to a symbol: never fall back to "whatever
        # is on the line" when it doesn't
        bound = [o for o in occs if o["name"] in anchor]
        if not bound:
            raise QuarryError(
                "NOT_FOUND",
                f"anchor names no symbol occurrence at {at['path']}:{line}",
                occurrences_on_line=[o["name"] for o in occs])
        occs = bound
    nth = at.get("occurrence", 1)
    if not isinstance(nth, int) or nth < 1 or nth > len(occs):
        raise QuarryError(
            "BAD_REQUEST",
            f"occurrence must be 1..{len(occs)} at {at['path']}:{line}, "
            f"got {nth!r}",
            occurrences_on_line=[o["name"] for o in occs])
    occ = occs[nth - 1]
    cands = resolve_symbols(idx, occ["name"])
    if not cands:
        raise QuarryError("NOT_FOUND",
                          f"occurrence {occ['name']!r} resolves to no indexed symbol")
    if len(cands) > 1:
        raise QuarryError(
            "AMBIGUOUS",
            f"occurrence {occ['name']!r} resolves to {len(cands)} symbols "
            "(syntactic resolution cannot pick a receiver)",
            candidates=[{"sym": c["sym"], "kind": c["kind"],
                         "def": c["def"]} for c in cands])
    return cands[0]


def refs_for(idx, symbol, roles=None):
    """Returns (occurrences, receiver_meta).

    For class members, occurrences are partitioned by receiver inference:
    provably this class -> kept as receiver 'proven'; provably ANOTHER
    class -> excluded (counted); unprovable -> kept as 'unknown'. For
    plain functions receiver_meta is None.
    """
    last = symbol["qualname"].split(".")[-1]
    occs = [o for o in idx["occurrences"] if o["name"] == last]
    # constructor trap: calls to Klass(...) ARE references to __init__
    parts = symbol["qualname"].split(".")
    if last == "__init__" and len(parts) >= 2:
        cls = parts[-2]
        occs = occs + [o for o in idx["occurrences"]
                       if o["name"] == cls and o["role"] == "call"]
    cls_qual = ".".join(parts[:-1])
    meta = None
    if idx["symbols"].get(cls_qual, {}).get("kind") == "class":
        kept, excluded, proven, unknown = [], 0, 0, 0
        for o in occs:
            recv = o.get("recv")
            # a definition's receiver is its enclosing class, always
            if o["role"] == "definition" and idx["symbols"].get(
                    o.get("enclosing", ""), {}).get("kind") == "class":
                recv = o["enclosing"]
            if recv == cls_qual:
                kept.append(dict(o, receiver="proven"))
                proven += 1
            elif recv is None:
                kept.append(dict(o, receiver="unknown"))
                unknown += 1
            else:
                excluded += 1
        occs = kept
        meta = {"receiver_proven": proven, "receiver_unknown": unknown,
                "excluded_other_receivers": excluded}
    if roles:
        occs = [o for o in occs if o["role"] in roles]
    return occs, meta


def apply_item_budget(items, budget, cursor=None):
    """Deterministic truncation. Returns (kept, elided_dict)."""
    offset = 0
    if cursor is not None:
        m = re.fullmatch(r"offset:(\d+)", str(cursor))
        if not m:
            raise QuarryError("BAD_REQUEST",
                              f"unrecognized cursor {cursor!r} "
                              "(expected offset:<n>)")
        offset = int(m.group(1))
    items = items[offset:]
    limit = None
    if budget:
        limit = budget.get("items")
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise QuarryError("BAD_REQUEST",
                              f"budget.items must be a non-negative int, "
                              f"got {limit!r}")
        if budget.get("tokens") is not None and limit is None:
            tokens = budget["tokens"]
            if not isinstance(tokens, int) or tokens < 0:
                raise QuarryError("BAD_REQUEST",
                                  f"budget.tokens must be a non-negative "
                                  f"int, got {tokens!r}")
            if tokens < 60:
                raise QuarryError("BUDGET_TOO_SMALL",
                                  f"budget.tokens={tokens} cannot fit one "
                                  "result item (~60 tokens)")
            # crude: assume ~60 tokens per item
            limit = max(1, tokens // 60)
    if limit is not None and len(items) > limit:
        kept = items[:limit]
        return kept, {"count": len(items) - limit,
                      "cursor": f"offset:{offset + limit}"}
    return items, None


# ---------------------------------------------------------------- verbs

def verb_resolve(ws, snap, idx, req):
    cands = resolve_symbols(idx, req.get("name"), req.get("sym"),
                            req.get("kinds"), req.get("match", "auto"))
    kept, elided = apply_item_budget(cands, req.get("budget"),
                                    req.get("cursor"))
    return make_envelope(snap, idx, {"symbols": kept, "total": len(cands)},
                         elided=elided)


def _counts(idx, symbol):
    occs, _rm = refs_for(idx, symbol)
    non_def = [o for o in occs if o["role"] != "definition"]
    callers = {o["enclosing"] for o in occs if o["role"] == "call"}
    tests = {o["path"] for o in non_def if "test" in o["path"].lower()}
    return {"refs": len(non_def), "callers": len(callers),
            "tests_touching": len(tests)}


def verb_describe(ws, snap, idx, req):
    symbol = resolve_unique(ws, idx, req)
    result = dict(symbol)
    result["counts"] = _counts(idx, symbol)
    notes = ["reference counts are name-matched (syntactic confidence): "
             "same-named symbols in other scopes are included"]
    return make_envelope(snap, idx, result, notes=notes)


def verb_refs(ws, snap, idx, req):
    symbol = resolve_unique(ws, idx, req)
    roles = req.get("role")
    if isinstance(roles, str):
        roles = [roles]
    occs, recv_meta = refs_for(idx, symbol, roles)
    if not req.get("include_definitions"):
        occs = [o for o in occs if o["role"] != "definition"] \
            if roles is None else occs
    total = len(occs)
    ctx_n = req.get("context_lines", 0)
    kept, elided = apply_item_budget(occs, req.get("budget"), req.get("cursor"))
    out = []
    for o in kept:
        item = dict(o)
        if ctx_n:
            lines = ws.read(o["path"]).splitlines()
            lo = max(0, o["line"] - 1 - ctx_n)
            hi = min(len(lines), o["line"] + ctx_n)
            item["context"] = [ln.rstrip() for ln in lines[lo:hi]]
        out.append(item)
    if req.get("group_by") == "file":
        groups = {}
        for item in out:
            groups.setdefault(item["path"], []).append(item)
        result = {"total": total, "symbol": symbol["sym"],
                  "groups": [{"path": p, "occurrences": v}
                             for p, v in sorted(groups.items())]}
    else:
        result = {"total": total, "symbol": symbol["sym"], "occurrences": out}
    confidence = "syntactic"
    if recv_meta is not None:
        result["receivers"] = recv_meta
        if recv_meta["receiver_unknown"] == 0:
            confidence = "index"
            notes = [f"every occurrence's receiver is proven to be this "
                     f"class; {recv_meta['excluded_other_receivers']} "
                     "same-named occurrence(s) on other classes excluded"]
        else:
            notes = [f"receiver-partitioned: "
                     f"{recv_meta['receiver_proven']} proven, "
                     f"{recv_meta['receiver_unknown']} unknown (marked per "
                     f"occurrence), "
                     f"{recv_meta['excluded_other_receivers']} excluded as "
                     "provably other classes — verify 'unknown' ones "
                     "before destructive action"]
    else:
        notes = ["matches are by name (syntactic): verify receivers before "
                 "destructive action"]
    # §7.1 L1 blind-spot disclosure: string/comment/config mentions are
    # outside the symbol graph (possible getattr-style dynamic access) —
    # a zero here without this count is a green light off a cliff
    last = symbol["qualname"].split(".")[-1]
    mentions = _textual_mentions(ws, last, exclude_py=False)
    result["textual_mentions"] = {"count": len(mentions),
                                  "sample": mentions[:5]}
    if mentions:
        notes.append(f"{len(mentions)} textual mention(s) exist in strings/"
                     "comments/config outside the symbol graph — possible "
                     "dynamic access (getattr, config keys): inspect before "
                     "acting on the occurrence count alone")
    return make_envelope(snap, idx, result, confidence=confidence,
                         elided=elided, notes=notes)


def _callers_of(idx, qualname):
    last = qualname.split(".")[-1]
    parent = ".".join(qualname.split(".")[:-1])
    is_member = idx["symbols"].get(parent, {}).get("kind") == "class"
    out = set()
    for o in idx["occurrences"]:
        if o["name"] != last or o["role"] != "call":
            continue
        if is_member and o.get("recv") is not None \
                and o["recv"] != parent:
            continue        # provably another class's method
        out.add(o["enclosing"])
    return sorted(out)


def verb_graph(ws, snap, idx, req):
    symbol = resolve_unique(ws, idx, req)
    edge = req.get("edge", "callers")
    depth = int(req.get("depth", 1))
    nodes, edges = {symbol["qualname"]: symbol["sym"]}, []

    if edge == "callers":
        frontier = [symbol["qualname"]]
        for _ in range(depth):
            nxt = []
            for q in frontier:
                for caller in _callers_of(idx, q):
                    edges.append({"from": caller, "to": q, "edge": "calls"})
                    if caller not in nodes:
                        nodes[caller] = "py:" + caller
                        nxt.append(caller)
            frontier = nxt
    elif edge in ("callees", "path"):
        def resolve_call(o):
            """Callee candidates for a call occurrence — receiver-exact
            when inference proved the class, suffix-matched otherwise."""
            recv = o.get("recv")
            if recv is not None:
                exact = idx["symbols"].get(recv + "." + o["name"])
                return [dict(exact, match="receiver")] if exact else []
            return resolve_symbols(idx, o["name"], match="suffix")

        def callees_of(qualname):
            s = idx["symbols"].get(qualname)
            if not s:
                return []
            d = s["def"]
            return [o for o in idx["occurrences"]
                    if o["path"] == d["path"] and o["role"] == "call"
                    and d["line"] <= o["line"] <= d["end_line"]]

        if edge == "path":
            to_name = req.get("to")
            if not to_name:
                raise QuarryError("BAD_REQUEST",
                                  "edge=path requires 'to' (target symbol)")
            t_cands = resolve_symbols(idx, to_name)
            t_cands = [c for c in t_cands if c.get("match") != "fuzzy"]
            if not t_cands:
                raise QuarryError("NOT_FOUND",
                                  f"no symbol matches target {to_name!r}")
            targets = {c["qualname"] for c in t_cands}
            max_depth = int(req.get("depth", 30))
            parent, hop = {}, {}
            frontier, seen = [symbol["qualname"]], {symbol["qualname"]}
            found = None
            for _ in range(max_depth):
                if found or not frontier:
                    break
                nxt = []
                for q in frontier:
                    for o in callees_of(q):
                        cands = resolve_call(o)
                        for cand in cands:
                            cq = cand["qualname"]
                            if cq in seen:
                                continue
                            seen.add(cq)
                            parent[cq] = q
                            hop[cq] = {"path": o["path"], "line": o["line"],
                                       "receiver_resolved":
                                       o.get("recv") is not None
                                       or len(cands) == 1}
                            if cq in targets:
                                found = cq
                                break
                            nxt.append(cq)
                        if found:
                            break
                    if found:
                        break
                frontier = nxt
            t_last = to_name.split(".")[-1]
            t_mentions = _textual_mentions(ws, t_last, exclude_py=False)
            notes_p = ["static call edges only — dynamic dispatch "
                       "(getattr/importlib) is NOT traversed"]
            if t_mentions:
                notes_p.append(
                    f"{len(t_mentions)} textual mention(s) of the target "
                    "exist (config/strings): dynamic reachability is "
                    "possible even if statically unreachable")
            if found:
                chain, hops, c = [found], [], found
                while c != symbol["qualname"]:
                    hops.append(dict(hop[c], callee=c, caller=parent[c]))
                    c = parent[c]
                    chain.append(c)
                chain.reverse()
                hops.reverse()
                conf = "index" if all(h["receiver_resolved"]
                                      for h in hops) else "syntactic"
                return make_envelope(
                    snap, idx,
                    {"from": symbol["sym"], "to": found, "reachable": True,
                     "chain": chain, "hops": hops,
                     "nodes_explored": len(seen),
                     "textual_mentions_of_target": len(t_mentions)},
                    confidence=conf, notes=notes_p)
            return make_envelope(
                snap, idx,
                {"from": symbol["sym"], "to": sorted(targets),
                 "reachable": False, "nodes_explored": len(seen),
                 "textual_mentions_of_target": len(t_mentions)},
                confidence="syntactic", notes=notes_p)

        frontier, visited = [symbol["qualname"]], {symbol["qualname"]}
        for _ in range(depth):
            nxt = []
            for q in frontier:
                for o in callees_of(q):
                    for cand in resolve_call(o):
                        edges.append({"from": q, "to": cand["qualname"],
                                      "edge": "calls",
                                      "receiver_resolved":
                                      o.get("recv") is not None,
                                      "at": {"path": o["path"],
                                             "line": o["line"]}})
                        nodes[cand["qualname"]] = cand["sym"]
                        if cand["qualname"] not in visited:
                            visited.add(cand["qualname"])
                            nxt.append(cand["qualname"])
            frontier = nxt
    elif edge == "importers":
        mod = symbol["qualname"]
        for imp in idx["imports"]:
            if imp["module"] == mod or imp["module"].endswith("." + mod.split(".")[-1]) \
               or mod.endswith("." + imp["module"]) or mod == imp["module"]:
                edges.append({"from": imp["path"], "to": mod,
                              "edge": "imports", "line": imp["line"]})
    elif edge == "imports":
        d = symbol["def"]["path"]
        for imp in idx["imports"]:
            if imp["path"] == d:
                edges.append({"from": d, "to": imp["module"],
                              "edge": "imports", "line": imp["line"]})
    else:
        raise QuarryError("UNSUPPORTED",
                          f"edge {edge!r} not implemented at this level",
                          supported=["callers", "callees", "path",
                                     "importers", "imports"])
    kept, elided = apply_item_budget(edges, req.get("budget"),
                                    req.get("cursor"))
    return make_envelope(snap, idx,
                         {"root": symbol["sym"], "edge": edge,
                          "nodes": nodes, "edges": kept, "total": len(edges)},
                         elided=elided,
                         notes=["caller/callee edges are name-matched "
                                "(syntactic)",
                                "callees of a class include calls made "
                                "anywhere in its body, methods included"])


def _enclosing_for(idx, path, line):
    best = None
    for s in idx["symbols"].values():
        d = s["def"]
        if d["path"] == path and d["line"] <= line <= d["end_line"] \
                and s["kind"] in ("function", "method", "class"):
            if best is None or d["line"] > best["def"]["line"]:
                best = s
    return best["qualname"] if best else module_name(path)


def verb_search(ws, snap, idx, req):
    pattern = req.get("pattern") or {}
    where = req.get("where") or {}
    if pattern.get("ast"):
        raise QuarryError("UNSUPPORTED",
                          "search:ast is not implemented in this prototype",
                          fallback="regex")
    if pattern.get("symbol"):
        pred = pattern["symbol"]
        hits = [s for s in idx["symbols"].values()
                if (not pred.get("kind") or s["kind"] == pred["kind"])
                and (not pred.get("visibility")
                     or s["visibility"] == pred["visibility"])
                and (not pred.get("name_regex")
                     or re.search(pred["name_regex"], s["qualname"]))]
        if where.get("path"):
            hits = [s for s in hits
                    if fnmatch.fnmatch(s["def"]["path"], where["path"])]
        kept, elided = apply_item_budget(hits, req.get("budget"),
                                        req.get("cursor"))
        return make_envelope(snap, idx, {"symbols": kept, "total": len(hits)},
                             elided=elided)
    regex = pattern.get("regex") or req.get("regex")
    if not regex:
        raise QuarryError("BAD_REQUEST",
                          "pattern.regex, pattern.symbol, or pattern.ast required")
    try:
        rx = re.compile(regex)
    except re.error as e:
        raise QuarryError("BAD_REQUEST", f"bad regex: {e}")
    path_glob = where.get("path")
    matches, files_searched = [], 0
    for rel in ws.walk():
        if path_glob and not fnmatch.fnmatch(rel, path_glob):
            continue
        files_searched += 1
        for i, ln in enumerate(ws.read(rel).splitlines(), 1):
            if rx.search(ln):
                m = {"path": rel, "line": i, "text": ln.rstrip()}
                if rel.endswith(".py"):
                    m["enclosing"] = _enclosing_for(idx, rel, i)
                matches.append(m)
    if where.get("enclosing_kind"):
        want = where["enclosing_kind"]
        matches = [m for m in matches
                   if "enclosing" in m
                   and idx["symbols"].get(m["enclosing"], {}).get("kind") == want]
    kept, elided = apply_item_budget(matches, req.get("budget"),
                                    req.get("cursor"))
    env = make_envelope(snap, idx, {"matches": kept, "total": len(matches)},
                        confidence="textual", elided=elided)
    # for L0 textual queries, query coverage != index coverage: say which
    env["coverage"]["files_searched"] = files_searched
    return env


def _have_mypy():
    try:
        import mypy  # noqa: F401
        return True
    except ImportError:
        return False


_MYPY_LINE = re.compile(r"^(.+?):(\d+):(?:(\d+):)?\s*(error|warning|note):\s*(.*)$")


def _mypy_run(ws, files, timeout_s=60):
    """Run mypy against a materialized view of the workspace.

    Returns (diagnostics, notes_by_location, meta). Raises QuarryError on
    setup problems; timeouts and crashes come back in meta so callers can
    degrade honestly instead of guessing.
    """
    import tempfile
    with tempfile.TemporaryDirectory(prefix="quarry-mypy-") as tmp:
        ws.materialize(tmp)
        cmd = [sys.executable, "-m", "mypy",
               "--no-error-summary", "--no-color-output",
               "--show-column-numbers", "--check-untyped-defs",
               "--ignore-missing-imports",
               "--cache-dir", os.path.join(tmp, ".mypy_cache")]
        cmd += files
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True,
                                  text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return [], {}, {"backend": "mypy", "timeout": True}
        if proc.returncode not in (0, 1):
            return [], {}, {"backend": "mypy", "crashed": True,
                            "stderr": proc.stderr.strip()[:400]}
        diags, notes = [], {}
        for line in proc.stdout.splitlines():
            m = _MYPY_LINE.match(line)
            if not m:
                continue
            path, lineno, col, sev, msg = m.groups()
            path = path.replace(os.sep, "/")
            if sev == "note":
                notes.setdefault((path, int(lineno)), []).append(msg)
                continue
            diags.append({"severity": "error" if sev == "error" else "warning",
                          "path": path, "line": int(lineno),
                          "col": int(col) if col else None,
                          "message": msg, "source": "mypy"})
        return diags, notes, {"backend": "mypy"}


def _pyflakes_check(source, rel):
    try:
        from pyflakes.api import check as pf_check
        from pyflakes.reporter import Reporter
    except ImportError:
        return None
    out, err = io.StringIO(), io.StringIO()
    pf_check(source, rel, Reporter(out, err))
    diags = []
    for line in out.getvalue().splitlines():
        m = re.match(r"^(.*?):(\d+):(\d+):?\s*(.*)$", line)
        if m:
            msg = m.group(4)
            sev = "error" if ("undefined" in msg or "used before" in msg) \
                else "warning"
            diags.append({"severity": sev, "path": rel,
                          "line": int(m.group(2)), "message": msg,
                          "source": "pyflakes"})
    return diags


def _dependents(idx, paths):
    mods = {module_name(p) for p in paths}
    extra = set()
    for imp in idx["imports"]:
        for m in mods:
            if imp["module"] == m or m.endswith("." + imp["module"]) \
                    or imp["module"].endswith("." + m.split(".")[-1]):
                extra.add(imp["path"])
    return extra


def verb_check(ws, snap, idx, req):
    scope = req.get("scope") or {}
    paths = scope.get("paths")
    files = [p for p in (paths or idx["files"]) if p.endswith(".py")]
    skipped = [p for p in (paths or []) if not p.endswith(".py")]
    missing = [p for p in files if not ws.exists(p)]
    if missing:
        raise QuarryError("NOT_FOUND", f"no such files: {missing}")
    if paths and not files:
        raise QuarryError("UNSUPPORTED",
                          "no checkable files in scope: this prototype "
                          "checks Python only",
                          skipped=skipped)
    if paths and scope.get("plus_dependents"):
        files = sorted(set(files) | _dependents(idx, files))
    diagnostics, syntax_broken = [], []
    for rel in files:
        try:
            compile(ws.read(rel), rel, "exec")
        except SyntaxError as e:
            diagnostics.append({"severity": "error", "path": rel,
                                "line": e.lineno or 0,
                                "message": f"SyntaxError: {e.msg}",
                                "source": "cpython"})
            syntax_broken.append(rel)
    status, notes, backend = "complete", [], "cpython-parse"
    if _have_mypy():
        checkable = [f for f in files if f not in syntax_broken]
        if checkable:
            m_diags, _, meta = _mypy_run(
                ws, checkable, timeout_s=req.get("max_wait_s", 60))
            if meta.get("timeout"):
                status = "partial"
                notes.append(f"mypy exceeded max_wait_s="
                             f"{req.get('max_wait_s', 60)}: type checking "
                             "did NOT complete; clean covers syntax only")
            elif meta.get("crashed"):
                status = "degraded"
                notes.append("mypy crashed — type checking did not run: "
                             + meta.get("stderr", ""))
            else:
                backend = "cpython-parse+mypy"
                diagnostics.extend(m_diags)
                notes.append("clean covers syntax, name, attribute, and "
                             "type errors (mypy --check-untyped-defs)")
    else:
        have_pyflakes = True
        for rel in files:
            if rel in syntax_broken:
                continue
            pf = _pyflakes_check(ws.read(rel), rel)
            if pf is None:
                have_pyflakes = False
            else:
                diagnostics.extend(pf)
        if have_pyflakes:
            backend = "cpython-parse+pyflakes"
            notes.append("clean covers syntax and name errors only — "
                         "attribute and type errors are NOT detected "
                         "(pip install mypy for L3 checking)")
        else:
            status = "degraded"
            notes.append("neither mypy nor pyflakes installed: only CPython "
                         "syntax checking ran; clean=true means 'parses', "
                         "not 'correct'")
    errors = [d for d in diagnostics if d["severity"] == "error"]
    result = {"diagnostics": diagnostics,
              "checked": {"files": len(files), "backend": backend},
              "clean": not errors if status != "partial" else None}
    if status == "partial":
        result["clean_syntax_only"] = not errors
    if skipped:
        # requested scope was narrowed: clean must not overpromise
        result["skipped"] = skipped
        status = "degraded" if status == "complete" else status
        notes.append(f"{len(skipped)} requested file(s) were not checkable "
                     "(non-Python) and are NOT covered by clean")
    return make_envelope(snap, idx, result, status=status,
                         confidence="compiler", notes=notes)


def verb_type_of(ws, snap, idx, req):
    at = req.get("at")
    if not at or not at.get("path") or not at.get("line"):
        raise QuarryError("BAD_REQUEST",
                          "type-of requires at:{path, line, anchor}")
    if not _have_mypy():
        raise QuarryError("UNSUPPORTED",
                          "type-of needs the mypy backend "
                          "(pip install mypy)", fallback="describe")
    if not at["path"].endswith(".py"):
        raise QuarryError("UNSUPPORTED", "type-of is Python-only here")
    line_no, _ = verify_anchor(ws, at)
    expr = req.get("expr") or at.get("anchor")
    if not expr:
        raise QuarryError("BAD_REQUEST",
                          "pass expr (or an anchor that IS the expression)")
    path = at["path"]
    lines = ws.read(path).splitlines()
    target = lines[line_no - 1]
    indent = re.match(r"[ \t]*", target).group(0)
    probe = indent + f"reveal_type({expr})"
    # a probe after return/raise/etc. is unreachable and mypy skips it;
    # a probe after a block opener (line ending ':') would empty the block
    # and break the parse — both cases probe BEFORE the line instead
    if re.match(r"(return|raise|break|continue)\b", target.strip()) \
            or target.rstrip().endswith(":"):
        probe_line = line_no
        new_lines = lines[:line_no - 1] + [probe] + lines[line_no - 1:]
    else:
        probe_line = line_no + 1
        new_lines = lines[:line_no] + [probe] + lines[line_no:]
    probed = Workspace(ws.root)
    probed.overlay = dict(ws.overlay)
    probed.overlay[path] = "\n".join(new_lines) + "\n"
    diags, notes_map, meta = _mypy_run(probed, [path],
                                       timeout_s=req.get("max_wait_s", 60))
    if meta.get("timeout") or meta.get("crashed"):
        raise QuarryError("INTERNAL", "mypy did not complete for the probe",
                          mypy=meta)
    # errors AT the probe line mean the expression itself is bad there
    # (undefined name, syntax): mypy still "reveals" Any for those, and
    # reporting that Any as a type would be confidently wrong
    probe_errors = [d for d in diags
                    if d["path"] == path and d["line"] == probe_line]
    if probe_errors:
        raise QuarryError(
            "NOT_FOUND",
            f"mypy could not type {expr!r} at {path}:{line_no}",
            mypy_errors=[d["message"] for d in probe_errors][:5])
    revealed = [n for n in notes_map.get((path, probe_line), [])
                if n.startswith("Revealed type is")]
    if not revealed:
        raise QuarryError(
            "NOT_FOUND",
            f"mypy produced no type for {expr!r} at {path}:{line_no} "
            "(unreachable code, or probe insertion failed)",
            mypy_errors=[d["message"] for d in diags][:5])
    m = re.search(r'Revealed type is "(.*)"', revealed[0])
    return make_envelope(
        snap, idx,
        {"expr": expr, "at": {"path": path, "line": line_no},
         "type": m.group(1) if m else revealed[0]},
        confidence="compiler",
        notes=["computed by mypy via a reveal_type probe in a "
               "hypothetical overlay world"])


def _count_params(sig_text):
    """Parse 'def f(a, b, c=1)' -> (required, total). None on failure."""
    try:
        m = re.search(r"def\s+\w+\s*\((.*)\)", sig_text, re.S)
        if not m:
            return None
        tree = ast.parse(f"def _f({m.group(1)}): pass")
        args = tree.body[0].args
        pos = list(args.posonlyargs) + list(args.args)
        pos = [a for a in pos if a.arg != "self"]
        required = len(pos) - len(args.defaults)
        total = len(pos) + (1000 if args.vararg else 0)
        return required, total
    except SyntaxError:
        return None


def verb_impact(ws, snap, idx, req):
    change = req.get("change") or {}
    kind = change.get("kind")
    if kind != "patch":
        symbol = resolve_unique(ws, idx, req)
    if kind == "delete":
        del_occs, _rm = refs_for(idx, symbol)
        breaks = [dict(o, reason="reference to deleted symbol")
                  for o in del_occs
                  if o["role"] != "definition"]
        # string/comment/config mentions can be live dynamic access
        # (getattr, registries): a delete verdict that omits them is a
        # confident green light for a breaking action
        breaks += [dict(m, reason="textual/string mention — possible "
                                  "dynamic access; not provable at this "
                                  "level")
                   for m in _textual_mentions(
                       ws, symbol["qualname"].split(".")[-1],
                       exclude_py=False)]
        reason = "reference to deleted symbol"
    elif kind == "signature":
        new_counts = _count_params(change.get("to", ""))
        old_counts = _count_params(symbol.get("signature", ""))
        breaks, reason = [], "call arity incompatible with new signature"
        sig_occs, _rm = refs_for(idx, symbol)
        calls = [o for o in sig_occs if o["role"] == "call"]
        if new_counts is None:
            raise QuarryError("BAD_REQUEST",
                              "could not parse change.to as a def signature")
        req_n, total_n = new_counts
        for o in calls:
            n = o.get("n_args")
            if n is None:
                continue
            if n < req_n or n > total_n:
                breaks.append(o)
    elif kind == "rename":
        # breaks = nothing semantically if done via edit.rename; report
        # textual mentions that a rename will NOT rewrite
        breaks = _textual_mentions(ws, symbol["qualname"].split(".")[-1],
                                   exclude_py=False)
        reason = "textual mention a rename will not rewrite"
    elif kind == "patch":
        if not _have_mypy():
            raise QuarryError("UNSUPPORTED",
                              "impact:patch needs the mypy backend",
                              fallback="overlay.create + check")
        files = _files_from_patch(ws, change.get("patch") or {})
        affected = sorted(set([p for p in files if p.endswith(".py")])
                          | _dependents(idx, list(files)))
        if not affected:
            raise QuarryError("UNSUPPORTED",
                              "patch touches no checkable Python files")
        patched = Workspace(ws.root)
        patched.overlay = dict(ws.overlay); patched.overlay.update(files)
        base_d, _, meta_b = _mypy_run(ws, affected)
        new_d, _, meta_n = _mypy_run(patched, affected)
        if any(m.get("timeout") or m.get("crashed")
               for m in (meta_b, meta_n)):
            raise QuarryError("INTERNAL", "mypy did not complete",
                              mypy=[meta_b, meta_n])
        seen_base = {(d["path"], d["message"]) for d in base_d}
        breaks = [d for d in new_d
                  if (d["path"], d["message"]) not in seen_base]
        reason = "new diagnostic introduced by the patch"
        tests = sorted({b["path"] for b in breaks
                        if "test" in b["path"].lower()})
        kept, elided = apply_item_budget(breaks, req.get("budget"),
                                        req.get("cursor"))
        for b in kept:
            b.setdefault("reason", reason)
        return make_envelope(
            snap, idx,
            {"change": "patch", "breaks": kept, "count": len(breaks),
             "tests_affected": tests, "files_checked": affected},
            confidence="compiler", elided=elided,
            notes=["breaks = mypy diagnostics present in the patched world "
                   "but not the base (compared by path+message)"])
    else:
        raise QuarryError("BAD_REQUEST",
                          "change.kind must be delete|signature|rename|patch")
    tests = sorted({o["path"] for o in breaks
                    if "test" in o.get("path", "").lower()})
    kept, elided = apply_item_budget(breaks, req.get("budget"),
                                    req.get("cursor"))
    for b in kept:
        b.setdefault("reason", reason)
    return make_envelope(
        snap, idx,
        {"symbol": symbol["sym"], "change": kind, "breaks": kept,
         "count": len(breaks), "tests_affected": tests},
        elided=elided,
        notes=["impact is computed syntactically (name-matched call sites); "
               "run check on an overlay for compiler-grade confirmation",
               "tests_affected lists files with DIRECT breaking call sites "
               "only; use graph --edge callers for transitive test exposure"])


def verb_brief(ws, snap, idx, req):
    symbol = resolve_unique(ws, idx, req)
    d = symbol["def"]
    lines = ws.read(d["path"]).splitlines()
    body = "\n".join(lines[d["line"] - 1:d["end_line"]])
    brief_occs, _rm = refs_for(idx, symbol)
    calls = [o for o in brief_occs if o["role"] == "call"]
    calls.sort(key=lambda o: (0 if "test" in o["path"].lower() else 1,
                              o["path"], o["line"]))
    kept, elided = apply_item_budget(
        calls, req.get("budget") or {"items": 10}, req.get("cursor"))
    result = {
        "symbol": symbol["sym"], "facts": {k: symbol[k] for k in
                                           ("kind", "signature", "doc",
                                            "visibility", "def")},
        "counts": _counts(idx, symbol),
        "definition_body": body,
        "call_sites": [{"path": o["path"], "line": o["line"],
                        "text": o["text"], "enclosing": o["enclosing"]}
                       for o in kept],
    }
    return make_envelope(snap, idx, result, elided=elided)


# ------------------------------------------------------------ overlays/edit

def _apply_unified_diff(base_text, diff_text, path):
    """Minimal hunk applier with exact context matching."""
    lines = base_text.splitlines(keepends=True)
    out, idx_line = [], 0
    hunks = re.split(r"^(@@ .*@@.*)$", diff_text, flags=re.M)
    # hunks[0] is headers; then alternating header/body
    if len(hunks) < 2:
        # a diff section with no parseable hunk is a malformed patch, not
        # a no-op: applying it silently would "verify" the unpatched world
        raise QuarryError("PATCH_FAILED",
                          f"no @@ hunks found for {path}",
                          unconsumed=diff_text.strip()[:200] or "<empty>")
    it = iter(range(1, len(hunks), 2))
    for h in it:
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@",
                     hunks[h])
        if not m:
            raise QuarryError("PATCH_FAILED", f"bad hunk header in {path}")
        start = int(m.group(1)) - 1
        if start < idx_line:
            raise QuarryError("PATCH_FAILED", f"overlapping hunks in {path}")
        out.extend(lines[idx_line:start])
        idx_line = start
        for raw in hunks[h + 1].splitlines()[0:]:
            if not raw:
                continue
            tag, content = raw[0], raw[1:]
            if tag == " ":
                if idx_line >= len(lines) or lines[idx_line].rstrip("\n") != content:
                    raise QuarryError(
                        "PATCH_FAILED",
                        f"context mismatch at {path}:{idx_line + 1}",
                        expected=content,
                        actual=(lines[idx_line].rstrip("\n")
                                if idx_line < len(lines) else "<eof>"))
                out.append(lines[idx_line]); idx_line += 1
            elif tag == "-":
                if idx_line >= len(lines) or lines[idx_line].rstrip("\n") != content:
                    raise QuarryError(
                        "PATCH_FAILED",
                        f"delete mismatch at {path}:{idx_line + 1}",
                        expected=content,
                        actual=(lines[idx_line].rstrip("\n")
                                if idx_line < len(lines) else "<eof>"))
                idx_line += 1
            elif tag == "+":
                out.append(content + "\n")
            elif tag == "\\":
                pass
            else:
                raise QuarryError("PATCH_FAILED",
                                  f"unparseable patch line in {path}: "
                                  f"{raw[:80]!r}")
    out.extend(lines[idx_line:])
    return "".join(out)


def _files_from_patch(ws, patch):
    fmt = patch.get("format", "files")
    files = {}
    if fmt == "files":
        files = dict(patch.get("files") or {})
        if not files:
            raise QuarryError("BAD_REQUEST",
                              'patch.files {path: content} required')
    elif fmt == "unified-diff":
        content = patch.get("content", "")
        # split per-file sections
        sections = re.split(r"^--- ", content, flags=re.M)
        if len(sections) < 2:
            raise QuarryError("PATCH_FAILED",
                              "no '--- <file>' sections found in diff",
                              unconsumed=content.strip()[:200] or "<empty>")
        for sec in sections[1:]:
            m = re.match(r"(?:a/)?(\S+)\n\+\+\+ (?:b/)?(\S+)\n", sec)
            if not m:
                raise QuarryError("PATCH_FAILED", "bad file header in diff")
            path = m.group(2)
            body = sec[m.end():]
            # stop this file's body at the next '--- ' already handled by split
            base = ws.read(path) if ws.exists(path) else ""
            files[path] = _apply_unified_diff(base, body, path)
    else:
        raise QuarryError("BAD_REQUEST",
                          "patch.format must be files|unified-diff")
    for p in files:
        if os.path.isabs(p) or ".." in p.split("/"):
            raise QuarryError("BAD_REQUEST", f"path escapes workspace: {p}")
    return files


def verb_overlay_create(ws, snap, idx, req):
    files = _files_from_patch(ws, req.get("patch") or {})
    blob = json.dumps(files, sort_keys=True)
    oid = "ovl:" + hashlib.sha256(blob.encode()).hexdigest()[:8]
    path = os.path.join(ws.root, ".quarry", "overlays",
                        oid.replace(":", "_") + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"base": snap["id"], "files": files}, f)
    return make_envelope(snap, idx,
                         {"overlay_id": oid, "files": sorted(files)})


def _textual_mentions(ws, name, exclude_py=True, rewritten=None):
    """\\b<name>\\b matches not covered by identifier rewrites."""
    rx = re.compile(r"\b" + re.escape(name) + r"\b")
    rewritten = rewritten or set()
    out = []
    for rel in ws.walk():
        is_py = rel.endswith(".py")
        if exclude_py and is_py:
            continue
        for i, ln in enumerate(ws.read(rel).splitlines(), 1):
            for m in rx.finditer(ln):
                if (rel, i, m.start()) in rewritten:
                    continue
                if is_py:
                    stripped = ln.strip()
                    in_comment = "#" in ln[:m.start()]
                    in_string = ln[:m.start()].count('"') % 2 == 1 or \
                        ln[:m.start()].count("'") % 2 == 1
                    if not (in_comment or in_string or
                            stripped.startswith('"""')):
                        continue
                    reason = "mention in comment/string — not rewritten"
                else:
                    reason = f"textual mention in non-Python file — not rewritten"
                out.append({"path": rel, "line": i,
                            "text": ln.rstrip(), "reason": reason})
    return out


def verb_edit_rename(ws, snap, idx, req):
    symbol = resolve_unique(ws, idx, req)
    new = req.get("to")
    if not new or not new.isidentifier() or keyword.iskeyword(new):
        raise QuarryError("BAD_REQUEST",
                          "--to must be a valid, non-keyword identifier")
    mode = req.get("mode", "diff")
    if mode == "apply" and ws.overlay_id:
        raise QuarryError("BAD_REQUEST",
                          "cannot apply a transform computed in an overlay "
                          "world to disk — run without --overlay, or keep "
                          "mode=diff")
    old = symbol["qualname"].split(".")[-1]
    ren_occs, ren_meta = refs_for(idx, symbol)
    occs = ren_occs + \
        [o for o in idx["occurrences"]
         if o["name"] == old and o["role"] == "definition"]
    # dedupe
    seen, targets = set(), []
    for o in occs:
        key = (o["path"], o["line"], o["col"])
        if key not in seen:
            seen.add(key)
            targets.append(o)
    by_file = {}
    for o in targets:
        by_file.setdefault(o["path"], []).append(o)
    rewritten, new_contents, mismatches = set(), {}, []
    for rel, file_occs in sorted(by_file.items()):
        lines = ws.read(rel).splitlines(keepends=True)
        file_hits = 0
        for o in sorted(file_occs, key=lambda x: (-x["line"], -x["col"])):
            li = o["line"] - 1
            if li >= len(lines):
                mismatches.append({"path": rel, "line": o["line"],
                                   "reason": "line vanished — not rewritten"})
                continue
            raw = lines[li].encode("utf-8")
            col = o["col"]
            actual = raw[col:col + len(old)].decode("utf-8", "replace")
            if actual != old:
                mismatches.append({"path": rel, "line": o["line"],
                                   "col": col,
                                   "reason": "occurrence text mismatch — "
                                             "not rewritten",
                                   "expected": old, "actual": actual,
                                   "text": lines[li].rstrip()})
                continue
            lines[li] = (raw[:col] + new.encode("utf-8")
                         + raw[col + len(old):]).decode("utf-8")
            rewritten.add((rel, o["line"], col))
            file_hits += 1
        if file_hits:
            new_contents[rel] = "".join(lines)
    # same-named symbols elsewhere: their occurrences were rewritten too
    # (name-matching cannot tell receivers apart) — surface each as a
    # conflict, not just a generic note
    same_named = [
        {"path": s["def"]["path"], "line": s["def"]["line"],
         "text": s["def"]["text"],
         "reason": f"same-named symbol {s['qualname']} also exists — "
                   "occurrences with PROVEN receivers on it were excluded "
                   "from the rewrite; unknown-receiver ones were rewritten "
                   "and are listed separately"}
        for s in idx["symbols"].values()
        if s["qualname"].split(".")[-1] == old
        and s["qualname"] != symbol["qualname"]]
    unknown_recv = [
        {"path": o["path"], "line": o["line"], "text": o["text"],
         "reason": "receiver unproven — rewritten; verify this occurrence "
                   "belongs to the intended class"}
        for o in ren_occs if o.get("receiver") == "unknown"]
    conflicts = mismatches + same_named + unknown_recv + _textual_mentions(
        ws, old, exclude_py=False, rewritten=rewritten)
    diff_parts = []
    for rel, content in sorted(new_contents.items()):
        base = ws.read(rel)
        diff_parts.extend(unified_diff(
            base.splitlines(keepends=True), content.splitlines(keepends=True),
            fromfile="a/" + rel, tofile="b/" + rel))
    diff_text = "".join(diff_parts)
    # post-check on an overlay of the new contents
    if not new_contents:
        post = {"clean": None, "note": "nothing rewritten — no check ran"}
    else:
        try:
            ov = Workspace(ws.root)
            ov.overlay = dict(new_contents)
            pc_diags = []
            for rel in new_contents:
                try:
                    compile(ov.read(rel), rel, "exec")
                except SyntaxError as e:
                    pc_diags.append({"path": rel, "line": e.lineno,
                                     "message": str(e.msg)})
            post = {"clean": not pc_diags, "diagnostics": pc_diags,
                    "confidence": "compiler(syntax-only)"}
        except Exception as e:
            post = {"clean": None, "error": str(e)}
    applied = False
    if mode == "apply" and post.get("clean") is False and not req.get("force"):
        raise QuarryError(
            "TRANSFORM_UNSAFE",
            "post-check found diagnostics in the transformed world; "
            "nothing was written (pass \"force\": true to apply anyway)",
            diagnostics=post.get("diagnostics"))
    if mode == "apply" and new_contents:
        for rel, content in new_contents.items():
            full = os.path.join(ws.root, rel)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
        applied = True
    kept_conf, elided = apply_item_budget(conflicts, req.get("budget"),
                                         req.get("cursor"))
    # truth-telling: a rename that rewrote nothing is not a success
    status = "complete" if rewritten else "partial"
    return make_envelope(
        snap, idx,
        {"diff": diff_text, "applied": applied,
         "files_changed": len(new_contents),
         "occurrences_rewritten": len(rewritten),
         "conflicts": kept_conf, "conflict_total": len(conflicts),
         "post_check": post},
        status=status, elided=elided,
        notes=["rename matched occurrences by name (syntactic): same-named "
               "attributes on other classes are also rewritten — review the "
               "diff and conflicts"])


def verb_capabilities(ws, snap, idx, req):
    try:
        import pyflakes  # noqa: F401
        pf = True
    except ImportError:
        pf = False
    my = _have_mypy()
    if my:
        backend = "cpython-parse+mypy"
    elif pf:
        backend = "cpython-parse+pyflakes"
    else:
        backend = "cpython-parse"
    verbs = ["resolve", "describe", "refs", "graph", "graph:path",
             "search:regex", "search:symbol", "check", "impact",
             "brief", "overlay.create", "edit.rename",
             "capabilities", "doctor"]
    missing = ["search:ast", "edit.move", "edit.change-signature"]
    extras = ["cross-file refs/graph", "receiver-resolved member refs "
              "(constructor/annotation/self inference; unknowns "
              "disclosed)", "graph path reachability", "impact",
              "overlay", "edit.rename"]
    if my:
        verbs += ["type-of", "impact:patch"]
        extras += ["check at L3 (mypy: syntax+name+attribute+type)",
                   "type-of (mypy reveal probe)"]
    else:
        missing += ["type-of", "impact:patch"]
        extras += ["check (syntax+names)"]
    return make_envelope(snap, idx, {
        "protocol": PROTO_VERSION,
        "langs": {
            "python": {
                # SPEC §7/§7.1: receiver inference + the global class
                # registry give cross-file identity for the provable
                # subset (disclosed partition), meeting the L2 bar;
                # unknown receivers degrade honestly to syntactic
                "level": 2, "confidence": "index",
                "extras_beyond_level": extras,
                "verbs": verbs,
                "missing": missing,
                "check_backend": backend,
            },
            "*": {"level": 0, "confidence": "textual",
                  "verbs": ["search:regex"]},
        },
        "ranking": {"brief.call_sites": "tests first, then path,line",
                    "truncation": "deterministic prefix after ranking"},
    })


def verb_doctor(ws, snap, idx, req):
    checks = []
    checks.append({"check": "python", "ok": True,
                   "detail": sys.version.split()[0]})
    if _have_mypy():
        checks.append({"check": "mypy", "ok": True,
                       "detail": "check runs at L3 (types+attributes); "
                                 "type-of and impact:patch available"})
    else:
        checks.append({"check": "mypy", "ok": False,
                       "remedy": "pip install mypy",
                       "consequence": "check degrades to name errors at "
                                      "best; type-of and impact:patch "
                                      "UNSUPPORTED"})
    try:
        import pyflakes  # noqa: F401
        checks.append({"check": "pyflakes", "ok": True})
    except ImportError:
        checks.append({"check": "pyflakes", "ok": False,
                       "remedy": "pip install pyflakes",
                       "consequence": "fallback when mypy absent"})
    checks.append({"check": "index", "ok": not idx["errored"],
                   "detail": f"{len(idx['files'])} files, "
                             f"{len(idx['symbols'])} symbols, "
                             f"{len(idx['occurrences'])} occurrences, "
                             f"built in {idx.get('build_ms')}ms",
                   "errored": idx["errored"] or None})
    return make_envelope(snap, idx, {"checks": checks})


def verb_warm(ws, snap, idx, req):
    return make_envelope(snap, idx, {"warmed": True,
                                     "cache": idx.get("cache")})


VERBS = {
    "resolve": verb_resolve,
    "describe": verb_describe,
    "refs": verb_refs,
    "graph": verb_graph,
    "search": verb_search,
    "check": verb_check,
    "type-of": verb_type_of,
    "impact": verb_impact,
    "brief": verb_brief,
    "overlay.create": verb_overlay_create,
    "edit.rename": verb_edit_rename,
    "capabilities": verb_capabilities,
    "doctor": verb_doctor,
    "warm": verb_warm,
}


def run_one(root, req):
    verb = req.get("verb")
    if verb not in VERBS and verb != "batch":
        return error_envelope("UNSUPPORTED", f"unknown verb {verb!r}",
                              supported=sorted(VERBS) + ["batch"])
    idx = None
    try:
        ws = Workspace(root, overlay_id=req.get("overlay"))
        if ws.overlay_id and ws.overlay_base:
            # an overlay is a world built on a specific base snapshot; if
            # the disk moved underneath, that world no longer exists
            disk_id = Workspace(root).snapshot()["id"]
            if disk_id != ws.overlay_base:
                raise QuarryError(
                    "SNAPSHOT_GONE",
                    f"overlay {ws.overlay_id} was built on "
                    f"{ws.overlay_base}, but the workspace is now "
                    f"{disk_id} — recreate the overlay")
        snap, idx = load_index(ws)
        if verb == "batch":
            out = []
            for sub in req.get("queries", []):
                sub_v = sub.get("verb")
                if sub_v not in VERBS:
                    out.append({"id": sub.get("id"),
                                **error_envelope("UNSUPPORTED",
                                                 f"unknown verb {sub_v!r}")})
                    continue
                try:
                    sws = Workspace(root, overlay_id=sub.get("overlay",
                                                             req.get("overlay")))
                    if sws.overlay_id != ws.overlay_id:
                        ssnap, sidx = load_index(sws)
                    else:
                        sws, ssnap, sidx = ws, snap, idx
                    env = VERBS[sub_v](sws, ssnap, sidx, sub)
                except QuarryError as e:
                    env = error_envelope(e.code, e.detail, **e.extra)
                env["id"] = sub.get("id")
                out.append(env)
            # the batch wrapper is a response too: it carries the envelope
            worst = "error" if all(r.get("status") == "error" for r in out) \
                else ("partial" if any(r.get("status") == "error"
                                       for r in out) else "complete")
            wrapper = make_envelope(snap, idx, None, status=worst)
            del wrapper["result"]
            wrapper["results"] = out
            return wrapper
        return VERBS[verb](ws, snap, idx, req)
    except QuarryError as e:
        env = error_envelope(e.code, e.detail, **e.extra)
        if idx is not None:
            # errors keep their epistemics: an agent told NOT_FOUND must
            # still see whether the defining file failed to index
            env["coverage"] = {
                "files_indexed": len(idx["files"]),
                "files_errored": [x["path"] for x in idx["errored"]]}
            if idx["errored"]:
                env["coverage"]["errored_detail"] = idx["errored"]
        return env
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        # malformed request shapes are the caller's problem, not INTERNAL
        return error_envelope("BAD_REQUEST",
                              f"malformed request: {type(e).__name__}: {e}")
    except Exception as e:  # never a stack trace on stdout: structured error
        return error_envelope("INTERNAL", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- CLI

def build_request(argv):
    p = argparse.ArgumentParser(prog="quarry", add_help=True,
                                description=__doc__.splitlines()[0])
    p.add_argument("verb", help="verb name, or 'batch'")
    p.add_argument("--root", default=".", help="workspace root")
    p.add_argument("--json", help="full request as JSON (merged over flags)")
    p.add_argument("--name"); p.add_argument("--sym"); p.add_argument("--to")
    p.add_argument("--mode"); p.add_argument("--edge")
    p.add_argument("--depth", type=int); p.add_argument("--role")
    p.add_argument("--regex"); p.add_argument("--overlay")
    p.add_argument("--cursor"); p.add_argument("--match")
    p.add_argument("--group-by", dest="group_by")
    p.add_argument("--context-lines", dest="context_lines", type=int)
    p.add_argument("--budget-items", dest="budget_items", type=int)
    p.add_argument("--budget-tokens", dest="budget_tokens", type=int)
    p.add_argument("--path", help="scope/where path (glob for search, "
                                  "file for check)")
    p.add_argument("--plus-dependents", action="store_true")
    args = p.parse_args(argv)
    req = {"verb": args.verb}
    for k in ("name", "sym", "to", "mode", "edge", "depth", "role", "regex",
              "overlay", "cursor", "match", "group_by", "context_lines"):
        v = getattr(args, k)
        if v is not None:
            req[k] = v
    if args.budget_items is not None or args.budget_tokens is not None:
        req["budget"] = {}
        if args.budget_items is not None:
            req["budget"]["items"] = args.budget_items
        if args.budget_tokens is not None:
            req["budget"]["tokens"] = args.budget_tokens
    if args.path:
        if args.verb == "search":
            req.setdefault("where", {})["path"] = args.path
        elif args.verb == "check":
            req["scope"] = {"paths": [args.path],
                            "plus_dependents": args.plus_dependents}
    if args.regex and args.verb == "search":
        req["pattern"] = {"regex": args.regex}
    if args.json:
        req.update(json.loads(args.json))
    elif not sys.stdin.isatty():
        # never hang on an open-but-silent pipe (the normal harness/CI
        # condition): only read stdin if data is already available
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        except (OSError, ValueError):
            ready = []
        if ready:
            data = sys.stdin.read().strip()
            if data:
                req.update(json.loads(data))
    return args.root, req


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    try:
        root, req = build_request(argv)
    except json.JSONDecodeError as e:
        print(json.dumps(error_envelope("BAD_REQUEST", f"bad JSON: {e}")))
        return 0
    env = run_one(root, req)
    print(json.dumps(env, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
