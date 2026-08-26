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
import os
import re
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
        self.overlay = {}          # relpath -> content (str)
        self.overlay_id = overlay_id
        if overlay_id:
            path = self._overlay_path(overlay_id)
            if not os.path.exists(path):
                raise QuarryError("SNAPSHOT_GONE",
                                  f"overlay {overlay_id} not found")
            with open(path, encoding="utf-8") as f:
                self.overlay = json.load(f)["files"]

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

    def read(self, rel):
        if rel in self.overlay:
            return self.overlay[rel]
        full = os.path.join(self.root, rel)
        with open(full, encoding="utf-8", errors="replace") as f:
            return f.read()

    def exists(self, rel):
        return rel in self.overlay or os.path.exists(os.path.join(self.root, rel))

    def snapshot(self):
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
    """One file. Collects symbols and occurrences with enclosing qualnames."""

    def __init__(self, rel, source, index):
        self.rel = rel
        self.mod = module_name(rel)
        self.lines = source.splitlines()
        self.index = index
        self.stack = [self.mod]           # qualname stack
        self.class_stack = []             # ClassDef qualnames
        self.call_marks = {}              # id(func_node) -> n_args

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

    def add_occ(self, name, node, role, n_args=None):
        occ = {"name": name, "path": self.rel, "line": node.lineno,
               "col": node.col_offset, "role": role,
               "enclosing": ".".join(self.stack),
               "text": self.line_text(node.lineno)}
        if n_args is not None:
            occ["n_args"] = n_args
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
        self.add_occ(node.name, node, "definition")
        self.stack.append(node.name)
        if kind == "class":
            self.class_stack.append(q)
        self.generic_visit(node)
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

    def visit_Assign(self, node):
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
        # col of the attribute name itself: end of node minus len(attr)
        fake = ast.Name(id=node.attr, ctx=node.ctx)
        fake.lineno = node.end_lineno
        fake.col_offset = node.end_col_offset - len(node.attr)
        if mark:
            self.add_occ(node.attr, fake, "call", n_args=sum(mark))
        else:
            self.add_occ(node.attr, fake, role)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.index["imports"].append(
                {"path": self.rel, "module": alias.name, "line": node.lineno})
            fake = ast.Name(id=alias.name.split(".")[-1], ctx=ast.Load())
            fake.lineno, fake.col_offset = node.lineno, node.col_offset
            self.add_occ(alias.name.split(".")[-1], fake, "import")

    def visit_ImportFrom(self, node):
        mod = node.module or ""
        self.index["imports"].append(
            {"path": self.rel, "module": mod, "line": node.lineno})
        for alias in node.names:
            fake = ast.Name(id=alias.name, ctx=ast.Load())
            fake.lineno, fake.col_offset = node.lineno, node.col_offset
            self.add_occ(alias.name, fake, "import")


def build_index(ws):
    t0 = time.time()
    index = {"symbols": {}, "occurrences": [], "imports": [],
             "files": [], "errored": [], "excluded": 0}
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
        ix = _Indexer(rel, source, index)
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
    symbols = idx["symbols"]
    if sym:
        q = sym[3:] if sym.startswith("py:") else sym
        hit = symbols.get(q)
        return [hit] if hit else []
    if not name:
        return []
    exact = [s for s in symbols.values() if s["qualname"] == name]
    if exact and match in ("auto", "exact"):
        out = exact
    else:
        suffix = [s for s in symbols.values()
                  if s["qualname"].endswith("." + name)
                  or s["qualname"].split(".")[-1] == name]
        if suffix and match in ("auto", "suffix", "exact"):
            out = suffix
        elif match in ("auto", "fuzzy"):
            low = name.lower()
            out = [s for s in symbols.values()
                   if low in s["qualname"].lower()]
        else:
            out = []
    if kinds:
        out = [s for s in out if s["kind"] in kinds]
    return sorted(out, key=lambda s: s["qualname"])


def resolve_unique(ws, idx, req):
    at = req.get("at")
    if at:
        return symbol_at(ws, idx, at)
    cands = resolve_symbols(idx, req.get("name"), req.get("sym"),
                            req.get("kinds"))
    if not cands:
        raise QuarryError("NOT_FOUND",
                          f"no symbol matches {req.get('name') or req.get('sym')!r}")
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
    if anchor:
        occs = [o for o in occs if o["name"] in anchor] or occs
    nth = at.get("occurrence", 1)
    if not occs:
        raise QuarryError("NOT_FOUND",
                          f"no symbol occurrence at {at['path']}:{line}")
    occ = occs[min(nth, len(occs)) - 1]
    cands = resolve_symbols(idx, occ["name"])
    if not cands:
        raise QuarryError("NOT_FOUND",
                          f"occurrence {occ['name']!r} resolves to no indexed symbol")
    return cands[0]


def refs_for(idx, symbol, roles=None):
    last = symbol["qualname"].split(".")[-1]
    occs = [o for o in idx["occurrences"] if o["name"] == last]
    if roles:
        occs = [o for o in occs if o["role"] in roles]
    return occs


def apply_item_budget(items, budget, cursor=None):
    """Deterministic truncation. Returns (kept, elided_dict)."""
    offset = 0
    if cursor and cursor.startswith("offset:"):
        offset = int(cursor.split(":", 1)[1])
    items = items[offset:]
    limit = None
    if budget:
        limit = budget.get("items")
        if budget.get("tokens") and limit is None:
            # crude: assume ~60 tokens per item
            limit = max(1, budget["tokens"] // 60)
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
    occs = refs_for(idx, symbol)
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
    occs = refs_for(idx, symbol, roles)
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
    notes = ["matches are by name (syntactic): verify receivers before "
             "destructive action"]
    return make_envelope(snap, idx, result, elided=elided, notes=notes)


def _callers_of(idx, qualname):
    last = qualname.split(".")[-1]
    return sorted({o["enclosing"] for o in idx["occurrences"]
                   if o["name"] == last and o["role"] == "call"})


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
    elif edge == "callees":
        d = symbol["def"]
        span = [o for o in idx["occurrences"]
                if o["path"] == d["path"] and o["role"] == "call"
                and d["line"] <= o["line"] <= d["end_line"]]
        for o in span:
            for cand in resolve_symbols(idx, o["name"]):
                edges.append({"from": symbol["qualname"],
                              "to": cand["qualname"], "edge": "calls",
                              "at": {"path": o["path"], "line": o["line"]}})
                nodes[cand["qualname"]] = cand["sym"]
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
                          supported=["callers", "callees", "importers",
                                     "imports"])
    kept, elided = apply_item_budget(edges, req.get("budget"),
                                    req.get("cursor"))
    return make_envelope(snap, idx,
                         {"root": symbol["sym"], "edge": edge,
                          "nodes": nodes, "edges": kept, "total": len(edges)},
                         elided=elided,
                         notes=["caller/callee edges are name-matched "
                                "(syntactic)"])


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
    matches = []
    for rel in ws.walk():
        if path_glob and not fnmatch.fnmatch(rel, path_glob):
            continue
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
    return make_envelope(snap, idx, {"matches": kept, "total": len(matches)},
                         confidence="textual", elided=elided)


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
    missing = [p for p in files if not ws.exists(p)]
    if missing:
        raise QuarryError("NOT_FOUND", f"no such files: {missing}")
    if paths and scope.get("plus_dependents"):
        files = sorted(set(files) | _dependents(idx, files))
    diagnostics, have_pyflakes = [], True
    for rel in files:
        source = ws.read(rel)
        try:
            compile(source, rel, "exec")
        except SyntaxError as e:
            diagnostics.append({"severity": "error", "path": rel,
                                "line": e.lineno or 0,
                                "message": f"SyntaxError: {e.msg}",
                                "source": "cpython"})
            continue
        pf = _pyflakes_check(source, rel)
        if pf is None:
            have_pyflakes = False
        else:
            diagnostics.extend(pf)
    errors = [d for d in diagnostics if d["severity"] == "error"]
    result = {"diagnostics": diagnostics,
              "checked": {"files": len(files)},
              "clean": not errors}
    notes = ["clean covers syntax and name errors only — attribute and type "
             "errors are NOT detected at this level (needs L3 / mypy)"]
    status, confidence = "complete", "compiler"
    if not have_pyflakes:
        status = "degraded"
        notes.append("pyflakes not installed: only CPython syntax checking "
                     "ran; name errors are NOT detected. clean=true means "
                     "'parses', not 'correct'. (pip install pyflakes)")
    return make_envelope(snap, idx, result, status=status,
                         confidence=confidence, notes=notes)


def verb_type_of(ws, snap, idx, req):
    raise QuarryError("UNSUPPORTED",
                      "type-of requires level 3 (semantic); this prototype "
                      "implements python at level 2",
                      fallback="describe")


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
    symbol = resolve_unique(ws, idx, req)
    change = req.get("change") or {}
    kind = change.get("kind")
    if kind == "delete":
        breaks = [o for o in refs_for(idx, symbol)
                  if o["role"] != "definition"]
        reason = "reference to deleted symbol"
    elif kind == "signature":
        new_counts = _count_params(change.get("to", ""))
        old_counts = _count_params(symbol.get("signature", ""))
        breaks, reason = [], "call arity incompatible with new signature"
        calls = [o for o in refs_for(idx, symbol) if o["role"] == "call"]
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
    else:
        raise QuarryError("BAD_REQUEST",
                          "change.kind must be delete|signature|rename")
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
               "run check on an overlay for compiler-grade confirmation"])


def verb_brief(ws, snap, idx, req):
    symbol = resolve_unique(ws, idx, req)
    d = symbol["def"]
    lines = ws.read(d["path"]).splitlines()
    body = "\n".join(lines[d["line"] - 1:d["end_line"]])
    calls = [o for o in refs_for(idx, symbol) if o["role"] == "call"]
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
    out.extend(lines[idx_line:])
    return "".join(out)


def verb_overlay_create(ws, snap, idx, req):
    patch = req.get("patch") or {}
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
    if not new or not new.isidentifier():
        raise QuarryError("BAD_REQUEST", "--to must be a valid identifier")
    mode = req.get("mode", "diff")
    old = symbol["qualname"].split(".")[-1]
    occs = refs_for(idx, symbol) + \
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
        for o in sorted(file_occs, key=lambda x: (-x["line"], -x["col"])):
            li = o["line"] - 1
            if li >= len(lines):
                mismatches.append({"path": rel, "line": o["line"],
                                   "reason": "line vanished — not rewritten"})
                continue
            raw = lines[li].encode("utf-8")
            col = o["col"]
            if raw[col:col + len(old)].decode("utf-8", "replace") != old:
                mismatches.append({"path": rel, "line": o["line"],
                                   "reason": "occurrence text mismatch — "
                                             "not rewritten",
                                   "text": lines[li].rstrip()})
                continue
            lines[li] = (raw[:col] + new.encode("utf-8")
                         + raw[col + len(old):]).decode("utf-8")
            rewritten.add((rel, o["line"], col))
        new_contents[rel] = "".join(lines)
    conflicts = mismatches + _textual_mentions(
        ws, old, exclude_py=False, rewritten=rewritten)
    diff_parts = []
    for rel, content in sorted(new_contents.items()):
        base = ws.read(rel)
        diff_parts.extend(unified_diff(
            base.splitlines(keepends=True), content.splitlines(keepends=True),
            fromfile="a/" + rel, tofile="b/" + rel))
    diff_text = "".join(diff_parts)
    # post-check on an overlay of the new contents
    post = {"clean": None}
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
    if mode == "apply":
        for rel, content in new_contents.items():
            full = os.path.join(ws.root, rel)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
        applied = True
    kept_conf, elided = apply_item_budget(conflicts, req.get("budget"),
                                         req.get("cursor"))
    return make_envelope(
        snap, idx,
        {"diff": diff_text, "applied": applied,
         "files_changed": len(new_contents),
         "occurrences_rewritten": len(rewritten),
         "conflicts": kept_conf, "conflict_total": len(conflicts),
         "post_check": post},
        elided=elided,
        notes=["rename matched occurrences by name (syntactic): same-named "
               "attributes on other classes are also rewritten — review the "
               "diff and conflicts"])


def verb_capabilities(ws, snap, idx, req):
    try:
        import pyflakes  # noqa: F401
        pf = True
    except ImportError:
        pf = False
    return make_envelope(snap, idx, {
        "protocol": PROTO_VERSION,
        "langs": {
            "python": {
                "level": 2, "confidence": "syntactic",
                "verbs": ["resolve", "describe", "refs", "graph",
                          "search:regex", "search:symbol", "check",
                          "impact", "brief", "overlay.create",
                          "edit.rename", "capabilities", "doctor"],
                "missing": ["type-of", "search:ast", "edit.move",
                            "edit.change-signature"],
                "check_backend": "cpython-parse" + ("+pyflakes" if pf else ""),
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
    try:
        import pyflakes  # noqa: F401
        checks.append({"check": "pyflakes", "ok": True})
    except ImportError:
        checks.append({"check": "pyflakes", "ok": False,
                       "remedy": "pip install pyflakes",
                       "consequence": "check() degrades to syntax-only"})
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
    try:
        ws = Workspace(root, overlay_id=req.get("overlay"))
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
            return {"status": "complete", "snapshot": snap, "results": out}
        return VERBS[verb](ws, snap, idx, req)
    except QuarryError as e:
        return error_envelope(e.code, e.detail, **e.extra)
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
    if args.budget_items or args.budget_tokens:
        req["budget"] = {}
        if args.budget_items:
            req["budget"]["items"] = args.budget_items
        if args.budget_tokens:
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
