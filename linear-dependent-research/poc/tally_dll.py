#!/usr/bin/env python3
"""
lambda-Tally POC v1 -- the intrusive doubly-linked list, checked and running.

This is the operation safe Rust cannot express: O(1) removal of a node given a
*handle* to it, with no GC, no Rc/RefCell, no `unsafe` -- and fully checked.

The mechanism (see ../docs/07-implementation-guide.md and ../poc/tally_poc.py):
  * a `List` is a LINEAR value owning a ghost REGION -- the bundle of one Perm
    per node, plus the next/prev link order.  Being linear, it cannot be dropped
    implicitly; you must empty it and `drop_empty` it (so no leaks).
  * a `Cursor` is a COPYABLE address plus an ERASED membership proof.  Holding a
    cursor does NOT borrow the list -- which is exactly why `remove(cursor)` can
    be O(1) and is impossible in safe Rust.
  * the checker is a SYMBOLIC EXECUTOR over the region: it tracks the exact set
    and order of live nodes, so for a closed program every membership /
    liveness / disjointness obligation is decided by lookup -- no inductive
    theorem proving.  (Proving an abstract `remove<T>` for ALL lists is the
    separation-logic proof deferred to the kernel.)

Observable, like v0:
  * ACCEPTS safe list code (build, traverse by cursor, O(1) remove, drain, drop)
    and RUNS it leak-free under the runtime safety monitor;
  * REJECTS double-remove, use-after-remove, cross-list remove, and leak (drop a
    non-empty list / forget to drop) -- with precise errors;
  * DIFFERENTIAL: bypass the checker, run the rejected program, and the monitor
    fires on the exact heap bug;
  * FUZZ: random op-sequences; every accepted program runs clean, 0 violations.

Run:  python3 tally_dll.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re
import random

from tally_poc import Heap, MonitorError      # the runtime heap + safety monitor


# ===========================================================================
# 1. Lexer / parser  (small surface: let-bindings, associated + method calls)
# ===========================================================================

TOKEN = re.compile(r"""
    \s+
  | (?P<int>\d+)
  | (?P<dcolon>::)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<punc>[(){};,.=])
""", re.VERBOSE)


@dataclass
class Tok: kind: str; val: str


def lex(src: str) -> list[Tok]:
    toks, i = [], 0
    while i < len(src):
        m = TOKEN.match(src, i)
        if not m:
            raise SyntaxError(f"bad char {src[i]!r}")
        i = m.end()
        if m.lastgroup is None:
            continue
        toks.append(Tok(m.lastgroup, m.group()))
    toks.append(Tok("eof", ""))
    return toks


# AST -------------------------------------------------------------------------
@dataclass
class Int:    v: int
@dataclass
class Null:   pass
@dataclass
class Var:    name: str
@dataclass
class AssocCall: ty: str; fn: str; args: list      # List::new()
@dataclass
class MethodCall: recv: object; meth: str; args: list   # e.method(args)
@dataclass
class Let:    name: str; rhs: object
@dataclass
class ExprS:  e: object


class Parser:
    def __init__(self, toks): self.toks, self.i = toks, 0
    def peek(self): return self.toks[self.i]
    def nxt(self):
        t = self.toks[self.i]; self.i += 1; return t
    def eat(self, kind, val=None):
        t = self.nxt()
        if t.kind != kind or (val is not None and t.val != val):
            raise SyntaxError(f"expected {val or kind}, got {t.kind}:{t.val!r}")
        return t
    def is_p(self, v): t = self.peek(); return t.kind == "punc" and t.val == v

    def block(self):
        out = []
        while self.peek().kind != "eof":
            out.append(self.stmt())
        return out

    def stmt(self):
        if self.peek().kind == "id" and self.peek().val == "let":
            self.nxt(); name = self.eat("id").val
            self.eat("punc", "="); rhs = self.expr(); self.eat("punc", ";")
            return Let(name, rhs)
        e = self.expr(); self.eat("punc", ";"); return ExprS(e)

    def expr(self):
        e = self.primary()
        while self.is_p("."):                      # postfix method calls
            self.nxt(); meth = self.eat("id").val
            self.eat("punc", "("); args = self.args(); self.eat("punc", ")")
            e = MethodCall(e, meth, args)
        return e

    def args(self):
        a = []
        while not self.is_p(")"):
            a.append(self.expr())
            if self.is_p(","): self.nxt()
        return a

    def primary(self):
        t = self.peek()
        if t.kind == "int": self.nxt(); return Int(int(t.val))
        if t.kind == "id" and t.val == "null": self.nxt(); return Null()
        if t.kind == "id":
            self.nxt()
            if self.peek().kind == "dcolon":       # Ty::fn(args)
                self.nxt(); fn = self.eat("id").val
                self.eat("punc", "("); args = self.args(); self.eat("punc", ")")
                return AssocCall(t.val, fn, args)
            return Var(t.val)
        if self.is_p("("):
            self.nxt(); e = self.expr(); self.eat("punc", ")"); return e
        raise SyntaxError(f"unexpected {t.kind}:{t.val!r}")


def parse(src): return Parser(lex(src)).block()


# ===========================================================================
# 2. The checker -- a symbolic executor over the ghost region
# ===========================================================================
# Symbolic values:
#   ('int',)            copyable
#   ('null',)
#   ('list', lid)       LINEAR handle to region `lid`
#   ('cursor', lid, k)  COPYABLE; k = node-id or None (end cursor)
#   ('consumed',)       a list value that was moved/dropped away
# regions[lid] = ordered list of node-ids (the live nodes & their order), or
#                None once the list has been dropped.

class Checker:
    def __init__(self):
        self.env: dict[str, tuple] = {}
        self.regions: dict[int, Optional[list]] = {}
        self.errors: list[str] = []
        self.lid = 0
        self.nid = 0

    def err(self, m): self.errors.append(m)

    def fresh_list(self):
        self.lid += 1; self.regions[self.lid] = []; return self.lid

    def fresh_node(self):
        self.nid += 1; return self.nid

    # evaluate an expression symbolically (borrowing: never consumes the receiver)
    def ev(self, e) -> tuple:
        if isinstance(e, Int):  return ("int",)
        if isinstance(e, Null): return ("null",)
        if isinstance(e, Var):
            v = self.env.get(e.name)
            if v is None: self.err(f"{e.name}: unbound"); return ("int",)
            if v == ("consumed",):
                self.err(f"{e.name}: use after move/drop"); return ("int",)
            return v
        if isinstance(e, AssocCall):
            if (e.ty, e.fn) == ("List", "new"):
                return ("list", self.fresh_list())
            self.err(f"unknown {e.ty}::{e.fn}"); return ("int",)
        if isinstance(e, MethodCall):
            return self.method(e)
        raise RuntimeError(e)

    def region_of(self, recv) -> Optional[int]:
        v = self.ev(recv)
        if not (isinstance(v, tuple) and v and v[0] == "list"):
            self.err("receiver is not a list"); return None
        lid = v[1]
        if self.regions.get(lid) is None:
            self.err("use of a dropped list"); return None
        return lid

    def method(self, e: MethodCall) -> tuple:
        m = e.meth
        # ---- list methods ------------------------------------------------
        if m in ("push_front", "push_back", "pop_front", "cursor_front",
                 "remove", "is_empty", "len", "drop_empty"):
            lid = self.region_of(e.recv)
            if lid is None: return ("int",)
            nodes = self.regions[lid]
            if m in ("push_front", "push_back"):
                self.ev(e.args[0])                       # element (copyable here)
                k = self.fresh_node()
                (nodes.insert(0, k) if m == "push_front" else nodes.append(k))
                return ("null",)
            if m == "pop_front":
                if not nodes: self.err("pop_front on an empty list")
                else: nodes.pop(0)
                return ("int",)
            if m == "cursor_front":
                return ("cursor", lid, nodes[0] if nodes else None)
            if m == "is_empty": return ("int",)
            if m == "len":      return ("int",)
            if m == "remove":
                c = self.ev(e.args[0])
                if not (isinstance(c, tuple) and c and c[0] == "cursor"):
                    self.err("remove: argument is not a cursor"); return ("int",)
                _, clid, k = c
                if clid != lid:
                    self.err("remove: cursor belongs to a DIFFERENT list")
                elif k is None:
                    self.err("remove: cursor is past-the-end (null)")
                elif k not in nodes:
                    self.err("remove: STALE cursor (its node was already removed)")
                else:
                    nodes.remove(k)
                return ("int",)
            if m == "drop_empty":
                if nodes:
                    self.err(f"drop_empty: list still has {len(nodes)} node(s) "
                             f"(would leak); drain it first")
                else:
                    self.regions[lid] = None
                    if isinstance(e.recv, Var):          # consume the variable
                        self.env[e.recv.name] = ("consumed",)
                return ("null",)
        # ---- cursor methods ---------------------------------------------
        if m in ("next", "get"):
            c = self.ev(e.recv)
            if not (isinstance(c, tuple) and c and c[0] == "cursor"):
                self.err(f".{m}: receiver is not a cursor"); return ("int",)
            _, lid, k = c
            if self.regions.get(lid) is None:
                self.err(f".{m}: cursor into a dropped list"); return ("int",)
            nodes = self.regions[lid]
            if k is None:
                self.err(f".{m}: cursor is past-the-end (null)"); return ("int",)
            if k not in nodes:
                self.err(f".{m}: STALE cursor (its node was removed)"); return ("int",)
            if m == "get": return ("int",)
            i = nodes.index(k)
            return ("cursor", lid, nodes[i + 1] if i + 1 < len(nodes) else None)
        self.err(f"unknown method {m}"); return ("int",)

    def stmt(self, s):
        if isinstance(s, Let):
            if self.env.get(s.name) and self.env[s.name][0] == "list" \
                    and self.regions.get(self.env[s.name][1]) is not None:
                self.err(f"let {s.name}: rebinding drops a live list (leak)")
            # move semantics: binding a list-valued variable consumes the source
            if isinstance(s.rhs, Var) and self.env.get(s.rhs.name, ("",))[0] == "list":
                v = self.env[s.rhs.name]; self.env[s.rhs.name] = ("consumed",)
                self.env[s.name] = v
            else:
                self.env[s.name] = self.ev(s.rhs)
        elif isinstance(s, ExprS):
            self.ev(s.e)

    def check(self, prog) -> list[str]:
        for s in prog:
            self.stmt(s)
        for name, v in self.env.items():            # leak check
            if v[0] == "list" and self.regions.get(v[1]) is not None:
                self.err(f"leak: list `{name}` is live at end of scope "
                         f"({len(self.regions[v[1]])} node(s)); must drop_empty it")
        return self.errors


def check(src): return Checker().check(parse(src))


# ===========================================================================
# 3. The interpreter -- real intrusive nodes on the monitored heap
# ===========================================================================

class RList:
    """Runtime list: two raw pointers; nodes are intrusive cells on the heap."""
    def __init__(self): self.head = None; self.tail = None

    def push_back(self, h, x):
        n = h.alloc({"next": None, "prev": self.tail, "elem": x})
        if self.tail is not None: h.write(self.tail, "next", n)
        else: self.head = n
        self.tail = n

    def push_front(self, h, x):
        n = h.alloc({"next": self.head, "prev": None, "elem": x})
        if self.head is not None: h.write(self.head, "prev", n)
        else: self.tail = n
        self.head = n

    def pop_front(self, h):
        n = self.head
        if n is None: raise MonitorError("pop_front on empty list")
        nxt = h.read(n, "next"); elem = h.read(n, "elem"); h.free(n)
        self.head = nxt
        if nxt is None: self.tail = None
        else: h.write(nxt, "prev", None)
        return elem

    def remove(self, h, node):
        prev = h.read(node, "prev"); nxt = h.read(node, "next")
        elem = h.read(node, "elem")
        if prev is not None: h.write(prev, "next", nxt)
        else: self.head = nxt
        if nxt is not None: h.write(nxt, "prev", prev)
        else: self.tail = prev
        h.free(node)
        return elem


class Interp:
    def __init__(self): self.heap = Heap(); self.env = {}

    def ev(self, e):
        if isinstance(e, Int):  return e.v
        if isinstance(e, Null): return None
        if isinstance(e, Var):  return self.env[e.name]
        if isinstance(e, AssocCall):
            if (e.ty, e.fn) == ("List", "new"): return RList()
            raise RuntimeError(e)
        if isinstance(e, MethodCall): return self.method(e)
        raise RuntimeError(e)

    def method(self, e):
        recv = self.ev(e.recv); h = self.heap; m = e.meth
        if isinstance(recv, RList):
            if m == "push_front": recv.push_front(h, self.ev(e.args[0])); return None
            if m == "push_back":  recv.push_back(h, self.ev(e.args[0]));  return None
            if m == "pop_front":  return recv.pop_front(h)
            if m == "cursor_front": return ("cursor", recv.head)
            if m == "remove":
                c = self.ev(e.args[0])                      # ('cursor', node_ptr)
                return recv.remove(h, c[1])
            if m == "drop_empty": return None               # leak shows up at run-end
            if m == "is_empty":   return recv.head is None
        if isinstance(recv, tuple) and recv and recv[0] == "cursor":
            node = recv[1]
            if m == "next": return ("cursor", h.read(node, "next"))
            if m == "get":  return h.read(node, "elem")
        raise RuntimeError(f"bad method {m} on {recv}")

    def stmt(self, s):
        if isinstance(s, Let):  self.env[s.name] = self.ev(s.rhs)
        elif isinstance(s, ExprS): self.ev(s.e)

    def run(self, prog):
        for s in prog: self.stmt(s)
        return self.heap.leaks()


def run(src):
    it = Interp()
    try:
        return it.run(parse(src)), None
    except MonitorError as e:
        return it.heap.leaks(), str(e)


# ===========================================================================
# 4. Examples + differential demo
# ===========================================================================

GOOD = {
"build, O(1) remove the middle by cursor, drain, drop": """
    let l = List::new();
    l.push_back(1);
    l.push_back(2);
    l.push_back(3);
    let c  = l.cursor_front();
    let c2 = c.next();
    let x  = l.remove(c2);
    l.pop_front();
    l.pop_front();
    l.drop_empty();
""",

"remove head then the only remaining node (edge cases)": """
    let l = List::new();
    l.push_back(1);
    l.push_back(2);
    let c = l.cursor_front();
    let a = l.remove(c);
    let d = l.cursor_front();
    let b = l.remove(d);
    l.drop_empty();
""",

"two independent lists": """
    let p = List::new();
    let q = List::new();
    p.push_front(1);
    q.push_front(2);
    p.pop_front();
    q.pop_front();
    p.drop_empty();
    q.drop_empty();
""",
}

BAD = {
"double remove (remove the same cursor twice)": """
    let l = List::new();
    l.push_back(1); l.push_back(2); l.push_back(3);
    let c = l.cursor_front();
    let c2 = c.next();
    let x = l.remove(c2);
    let y = l.remove(c2);
    l.pop_front(); l.pop_front(); l.drop_empty();
""",

"use after remove (advance a cursor whose node was removed)": """
    let l = List::new();
    l.push_back(1); l.push_back(2); l.push_back(3);
    let c = l.cursor_front();
    let c2 = c.next();
    let x = l.remove(c2);
    let c3 = c2.next();
    l.pop_front(); l.pop_front(); l.drop_empty();
""",

"cross-list remove (cursor from p, removed via q)": """
    let p = List::new();
    let q = List::new();
    p.push_back(1); p.push_back(2);
    q.push_back(3);
    let cp = p.cursor_front();
    let bad = q.remove(cp);
    p.pop_front(); p.pop_front(); q.pop_front();
    p.drop_empty(); q.drop_empty();
""",

"leak: drop a non-empty list": """
    let l = List::new();
    l.push_back(1); l.push_back(2);
    l.drop_empty();
""",

"leak: forget to drop the list (nodes never freed)": """
    let l = List::new();
    l.push_back(1);
    l.push_back(2);
""",
}


def banner(s): print("\n" + "=" * 74 + f"\n{s}\n" + "=" * 74)


def main():
    banner("ACCEPTED: checker silent, runs leak-free under the safety monitor")
    for name, src in GOOD.items():
        errs = check(src)
        if errs:
            print(f"  [UNEXPECTED REJECT] {name}: {errs}"); continue
        leaks, mon = run(src)
        ok = "OK" if (mon is None and not leaks) else f"!! {mon} leaks={leaks}"
        print(f"  ✓ accept + run clean   {name:52} -> {ok}")

    banner("REJECTED: the checker explains the bug before it can happen")
    for name, src in BAD.items():
        errs = check(src)
        print(f"  ✗ {'rejected' if errs else '!! ACCEPTED (CHECKER BUG)':9} {name}")
        for e in errs[:1]:
            print(f"        - {e}")

    banner("DIFFERENTIAL: bypass the checker, run the rejected programs,\n"
           "and the runtime monitor fires on the exact heap bug")
    for name, src in BAD.items():
        leaks, mon = run(src)
        if mon:      print(f"  monitor FIRED  {name:50} -> {mon}")
        elif leaks:  print(f"  monitor: LEAK  {name:50} -> live cells {leaks}")
        else:        print(f"  (ran clean)    {name}")

    banner("FUZZ: 20000 random list programs; every ACCEPTED one must run with\n"
           "the monitor silent and no leak (0 violations = sound here)")
    fuzz()


# ===========================================================================
# 5. Fuzzer -- random op sequences over the list API
# ===========================================================================

def _smart_prog(rng, length):
    """Linearity/region-aware generator: tracks the symbolic list so it emits
    mostly-valid programs; drains and drops at the end."""
    nodes = 0                       # how many nodes currently in the list
    cursors = []                    # cursor var names that are currently valid
    cidx = {}                       # cursor var -> position index (0-based) it points at
    lines = ["let l = List::new();"]
    cv = 0
    for _ in range(length):
        r = rng.random()
        if r < 0.4:
            lines.append(f"l.push_back({rng.randint(0,9)});"); nodes += 1
            cidx = {c: i + 1 for c, i in cidx.items()}     # push_back shifts nothing at front
            cidx = {c: i for c, i in cidx.items()}
        elif nodes > 0 and r < 0.6:
            # make a cursor to the front
            name = f"c{cv}"; cv += 1
            lines.append(f"let {name} = l.cursor_front();")
            cursors.append(name); cidx[name] = 0
        elif cursors and r < 0.78:
            # advance a valid cursor that is not at the last node
            cand = [c for c in cursors if cidx[c] < nodes - 1]
            if cand:
                c = rng.choice(cand); name = f"c{cv}"; cv += 1
                lines.append(f"let {name} = {c}.next();")
                cursors.append(name); cidx[name] = cidx[c] + 1
        elif cursors and nodes > 0 and r < 0.9:
            # remove a node via a valid cursor; invalidate cursors at that index
            cand = [c for c in cursors if 0 <= cidx[c] < nodes]
            if cand:
                c = rng.choice(cand); pos = cidx[c]
                lines.append(f"let _r = l.remove({c});"); nodes -= 1
                # cursors after pos shift left by 1; cursor(s) at pos become stale
                new_cursors, new_idx = [], {}
                for cc in cursors:
                    if cidx[cc] == pos:        # stale -> drop from valid set
                        continue
                    new_idx[cc] = cidx[cc] - 1 if cidx[cc] > pos else cidx[cc]
                    new_cursors.append(cc)
                cursors, cidx = new_cursors, new_idx
        elif nodes > 0:
            lines.append("let _p = l.pop_front();"); nodes -= 1
            cursors, cidx = [], {}             # be conservative: pop invalidates cursors
    for _ in range(nodes):
        lines.append("let _q = l.pop_front();")
    lines.append("l.drop_empty();")
    return "\n".join(lines)


def fuzz(trials=20000, length=10, seed=0):
    rng = random.Random(seed)
    accepted = clean = violations = 0
    for _ in range(trials):
        src = _smart_prog(rng, rng.randint(1, length))
        try:
            errs = check(src)
        except Exception:
            continue
        if errs:
            continue
        accepted += 1
        leaks, mon = run(src)
        if mon:
            violations += 1; print(f"  !! ACCEPTED-BUT-UNSAFE ({mon}):\n{src}\n")
        elif leaks:
            violations += 1; print(f"  !! ACCEPTED-BUT-LEAKS (cells {leaks}):\n{src}\n")
        else:
            clean += 1
    print(f"  generated {trials}, accepted {accepted}, ran clean {clean}, "
          f"SAFETY VIOLATIONS among accepted: {violations}")
    return violations


if __name__ == "__main__":
    main()
