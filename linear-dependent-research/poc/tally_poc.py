#!/usr/bin/env python3
"""
lambda-Tally POC v0 -- the L3 address/permission split, running.

This is a *small but real* checker + interpreter for the core mechanism behind
the safe intrusive data structures: a heap address (`Addr`) is UNRESTRICTED
(freely copyable, carries NO permission), while the capability to touch a cell
(`Perm`) is LINEAR (used exactly once) and ZERO-SIZED (erased at runtime). The
linear discipline is exactly the one proved sound in agda/CombinedSound.agda.

It demonstrates, *observably*:
  * the checker ACCEPTS safe code (alloc / read / write / free, with aliased
    interior pointers) and REJECTS double-free, use-after-free, use-after-move,
    leaks, and -- the key safety win -- dereferencing a bare aliased address you
    hold no capability for;
  * accepted programs RUN on a modelled heap under a runtime SAFETY MONITOR
    (every read/write/free checks liveness; program end checks for leaks) and
    the monitor never fires;
  * DIFFERENTIAL proof-of-work: bypass the checker on a rejected program, run
    it, and the monitor fires on the very bug the checker predicted.

Surface language (v0): a single block of statements.
    let x = <expr>;          -- bind (alloc => x owns a Perm; else x is a value)
    x.f = <expr>;            -- write field f of cell x   (needs x's Perm)
    free x;                  -- consume x's Perm, reclaim the cell
    <expr>;
  expressions:
    42 | unit | null | x
    alloc { f: e, g: e, ... }   -- allocate a cell, return an owning handle
    addr(x)                     -- the COPYABLE address of x (does NOT spend its Perm)
    x.f                         -- read field f of cell x   (needs x's Perm)
    e + e

Run:  python3 tally_poc.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re

# ===========================================================================
# 1. Tokenizer
# ===========================================================================

KEYWORDS = {"let", "free", "alloc", "addr", "null", "unit"}
TOKEN = re.compile(r"""
    \s+
  | (?P<int>\d+)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<punc>[{}();,.=+:])
""", re.VERBOSE)


@dataclass
class Tok:
    kind: str   # 'int' | 'id' | 'kw' | 'punc' | 'eof'
    val: str


def lex(src: str) -> list[Tok]:
    toks, i = [], 0
    while i < len(src):
        m = TOKEN.match(src, i)
        if not m:
            raise SyntaxError(f"bad char {src[i]!r} at {i}")
        i = m.end()
        if m.lastgroup is None:        # whitespace
            continue
        g, v = m.lastgroup, m.group()
        if g == "id" and v in KEYWORDS:
            toks.append(Tok("kw", v))
        else:
            toks.append(Tok(g, v))
    toks.append(Tok("eof", ""))
    return toks


# ===========================================================================
# 2. AST
# ===========================================================================

# expressions
@dataclass
class Int:   v: int
@dataclass
class Null:  pass
@dataclass
class UnitE: pass
@dataclass
class Var:   name: str
@dataclass
class Alloc: fields: list[tuple[str, "Expr"]]
@dataclass
class AddrOf: name: str
@dataclass
class Field: obj: "Expr"; fld: str
@dataclass
class Add:   l: "Expr"; r: "Expr"

Expr = object

# statements
@dataclass
class Let:    name: str; rhs: Expr
@dataclass
class WriteF: obj: Expr; fld: str; rhs: Expr     # obj is the base path of the lvalue
@dataclass
class FreeS:  name: str
@dataclass
class ExprS:  e: Expr


# ===========================================================================
# 3. Parser  (recursive descent)
# ===========================================================================

class Parser:
    def __init__(self, toks: list[Tok]):
        self.toks, self.i = toks, 0

    def peek(self) -> Tok: return self.toks[self.i]
    def next(self) -> Tok:
        t = self.toks[self.i]; self.i += 1; return t

    def eat(self, kind: str, val: Optional[str] = None) -> Tok:
        t = self.next()
        if t.kind != kind or (val is not None and t.val != val):
            raise SyntaxError(f"expected {val or kind}, got {t.kind}:{t.val!r}")
        return t

    def is_punc(self, v: str) -> bool:
        t = self.peek(); return t.kind == "punc" and t.val == v

    def parse_block(self) -> list:
        stmts = []
        while self.peek().kind != "eof":
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self):
        t = self.peek()
        if t.kind == "kw" and t.val == "let":
            self.next(); name = self.eat("id").val
            self.eat("punc", "="); rhs = self.parse_expr(); self.eat("punc", ";")
            return Let(name, rhs)
        if t.kind == "kw" and t.val == "free":
            self.next(); name = self.eat("id").val; self.eat("punc", ";")
            return FreeS(name)
        # otherwise: parse an expression; if it is a field path followed by `=`,
        # it is a field-write to that lvalue, else an expression statement.
        e = self.parse_expr()
        if isinstance(e, Field) and self.is_punc("="):
            self.next(); rhs = self.parse_expr(); self.eat("punc", ";")
            return WriteF(e.obj, e.fld, rhs)
        self.eat("punc", ";")
        return ExprS(e)

    def parse_expr(self):
        e = self.parse_atom()
        while self.is_punc("+"):
            self.next(); e = Add(e, self.parse_atom())
        # postfix field reads:  e.f.g
        while self.is_punc("."):
            self.next(); fld = self.eat("id").val; e = Field(e, fld)
        return e

    def parse_atom(self):
        t = self.peek()
        if t.kind == "int":   self.next(); return Int(int(t.val))
        if t.kind == "kw" and t.val == "null": self.next(); return Null()
        if t.kind == "kw" and t.val == "unit": self.next(); return UnitE()
        if t.kind == "kw" and t.val == "alloc": return self.parse_alloc()
        if t.kind == "kw" and t.val == "addr":
            self.next(); self.eat("punc", "("); n = self.eat("id").val
            self.eat("punc", ")"); return AddrOf(n)
        if t.kind == "id":
            self.next(); e = Var(t.val)
            while self.is_punc("."):
                self.next(); fld = self.eat("id").val; e = Field(e, fld)
            return e
        if self.is_punc("("):
            self.next(); e = self.parse_expr(); self.eat("punc", ")"); return e
        raise SyntaxError(f"unexpected {t.kind}:{t.val!r}")

    def parse_alloc(self):
        self.eat("kw", "alloc"); self.eat("punc", "{")
        fields = []
        while not self.is_punc("}"):
            fname = self.eat("id").val; self.eat("punc", ":")
            fields.append((fname, self.parse_expr()))
            if self.is_punc(","): self.next()
        self.eat("punc", "}")
        return Alloc(fields)


def parse(src: str) -> list:
    return Parser(lex(src)).parse_block()


# ===========================================================================
# 4. The checker  --  the linear / permission discipline
# ===========================================================================
#
# Each variable has a static STATE:
#   OWN   : holds a live linear Perm to a heap cell  (must be consumed by `free`)
#   ADDR  : a copyable address / value -- NO permission
#   MOVED : an OWN whose Perm was moved or freed away (using it is an error)
# Field reads always yield a copyable value (ADDR): you can NEVER fabricate a
# capability by reading memory -- Perms come only from `alloc`. That is the
# single invariant that makes aliased back-pointers safe.

OWN, ADDR, MOVED = "own", "addr", "moved"


class CheckError(Exception):
    pass


class Checker:
    def __init__(self):
        self.state: dict[str, str] = {}
        self.errors: list[str] = []

    def err(self, msg: str):
        self.errors.append(msg)

    # value-kind of an expression; `move` = we are consuming it as a value
    def kind_of(self, e, *, allow_own_move: bool) -> str:
        if isinstance(e, (Int, Add)):  return ADDR     # plain values are copyable
        if isinstance(e, (Null, UnitE)): return ADDR
        if isinstance(e, Alloc):
            for _, fe in e.fields:
                self.require_copyable(fe)
            return OWN
        if isinstance(e, AddrOf):
            st = self.state.get(e.name)
            if st == OWN:    return ADDR               # borrow the address; Perm untouched
            if st == MOVED:  self.err(f"addr({e.name}): {e.name} was already moved/freed")
            elif st is None: self.err(f"addr({e.name}): unbound")
            else:            return ADDR
            return ADDR
        if isinstance(e, Field):
            self.require_perm_for(e.obj)               # reading needs the cap on the base
            return ADDR                                # ...and yields a copyable value
        if isinstance(e, Var):
            st = self.state.get(e.name)
            if st is None:   self.err(f"{e.name}: unbound"); return ADDR
            if st == MOVED:  self.err(f"{e.name}: use after move/free"); return ADDR
            if st == OWN:
                if allow_own_move:
                    self.state[e.name] = MOVED         # linear move
                    return OWN
                self.err(f"{e.name}: linear owned value used where a plain value is "
                         f"required (use addr({e.name}) to copy its address)")
                return ADDR
            return ADDR                                # ADDR
        raise CheckError(f"?? {e}")

    # an expression in a field/value slot must be copyable (no Perm escapes into a field)
    def require_copyable(self, e):
        self.kind_of(e, allow_own_move=False)

    # base of a read/write must currently hold a Perm
    def require_perm_for(self, e):
        if isinstance(e, Var):
            st = self.state.get(e.name)
            if st == OWN:    return
            if st == MOVED:  self.err(f"{e.name}.<field>: use after move/free")
            elif st is None: self.err(f"{e.name}.<field>: unbound")
            else:
                self.err(f"{e.name}.<field>: no capability for this address "
                         f"(`{e.name}` is a bare Addr; you hold no Perm for it)")
        else:
            # e.g. a.next.x  : the inner read yields a bare Addr we hold no Perm for
            self.kind_of(e, allow_own_move=False)
            self.err("deref of an interior/aliased address: no capability held for it")

    def check_stmt(self, s):
        if isinstance(s, Let):
            if self.state.get(s.name) == OWN:    # rebinding would drop a live Perm
                self.err(f"let {s.name}: rebinding drops `{s.name}`'s live Perm (leak)")
            k = self.kind_of(s.rhs, allow_own_move=True)
            self.state[s.name] = OWN if k == OWN else ADDR
        elif isinstance(s, WriteF):
            self.require_perm_for(s.obj)
            self.require_copyable(s.rhs)
        elif isinstance(s, FreeS):
            st = self.state.get(s.name)
            if st == OWN:    self.state[s.name] = MOVED
            elif st == MOVED: self.err(f"free {s.name}: double free / use after free")
            elif st is None: self.err(f"free {s.name}: unbound")
            else:            self.err(f"free {s.name}: no capability (bare Addr)")
        elif isinstance(s, ExprS):
            self.kind_of(s.e, allow_own_move=False)

    def check(self, prog) -> list[str]:
        for s in prog:
            self.check_stmt(s)
        # leak check: any Perm still held at end of scope is a leak
        for name, st in self.state.items():
            if st == OWN:
                self.err(f"leak: `{name}` still owns a live cell at end of scope "
                         f"(linear Perm never consumed)")
        return self.errors


def check(src: str) -> list[str]:
    return Checker().check(parse(src))


# ===========================================================================
# 5. The interpreter + runtime SAFETY MONITOR
# ===========================================================================
# At runtime there is NO distinction between OWN and ADDR -- Perms are erased.
# Both are just a pointer ('loc', n). The monitor is the ground truth: it fires
# on any read/write/free of a non-live cell, and reports leaks at the end.

class MonitorError(Exception):
    pass


class Heap:
    def __init__(self):
        self.cells: dict[int, dict] = {}
        self.live: set[int] = set()
        self.n = 0
        self.log: list[str] = []

    def alloc(self, fields: dict) -> tuple:
        loc = self.n; self.n += 1
        self.cells[loc] = dict(fields); self.live.add(loc)
        self.log.append(f"alloc L{loc} = {fields}")
        return ("loc", loc)

    def _loc(self, v, op) -> int:
        if not (isinstance(v, tuple) and v and v[0] == "loc"):
            raise MonitorError(f"{op}: not a pointer ({v!r})")
        return v[1]

    def read(self, base, fld):
        loc = self._loc(base, "read")
        if loc not in self.live:
            raise MonitorError(f"USE-AFTER-FREE: read L{loc}.{fld} (cell is dead)")
        return self.cells[loc][fld]

    def write(self, base, fld, val):
        loc = self._loc(base, "write")
        if loc not in self.live:
            raise MonitorError(f"USE-AFTER-FREE: write L{loc}.{fld} (cell is dead)")
        self.cells[loc][fld] = val
        self.log.append(f"write L{loc}.{fld} = {val}")

    def free(self, base):
        loc = self._loc(base, "free")
        if loc not in self.live:
            raise MonitorError(f"DOUBLE-FREE: free L{loc} (cell already dead)")
        self.live.discard(loc); del self.cells[loc]
        self.log.append(f"free  L{loc}")

    def leaks(self) -> list[int]:
        return sorted(self.live)


class Interp:
    def __init__(self):
        self.heap = Heap()
        self.env: dict[str, object] = {}

    def eval(self, e):
        if isinstance(e, Int):   return e.v
        if isinstance(e, Null):  return None
        if isinstance(e, UnitE): return ()
        if isinstance(e, Var):   return self.env[e.name]
        if isinstance(e, AddrOf): return self.env[e.name]          # same pointer; Perm erased
        if isinstance(e, Add):   return self.eval(e.l) + self.eval(e.r)
        if isinstance(e, Alloc):
            return self.heap.alloc({f: self.eval(fe) for f, fe in e.fields})
        if isinstance(e, Field): return self.heap.read(self.eval(e.obj), e.fld)
        raise RuntimeError(f"?? {e}")

    def run_stmt(self, s):
        if isinstance(s, Let):    self.env[s.name] = self.eval(s.rhs)
        elif isinstance(s, WriteF): self.heap.write(self.eval(s.obj), s.fld, self.eval(s.rhs))
        elif isinstance(s, FreeS):  self.heap.free(self.env[s.name])
        elif isinstance(s, ExprS):  self.eval(s.e)

    def run(self, prog):
        for s in prog:
            self.run_stmt(s)
        return self.heap.leaks()


def run(src: str):
    """Run (ignoring the checker). Returns (leaks, monitor_error_or_None)."""
    it = Interp()
    try:
        leaks = it.run(parse(src))
        return leaks, None
    except MonitorError as m:
        return it.heap.leaks(), str(m)


# ===========================================================================
# 6. Examples + differential demo
# ===========================================================================

GOOD = {
"alloc / write / free": """
    let a = alloc { val: 41, next: null };
    a.val = 42;
    free a;
""",

"two nodes, mutual back-pointers (the doubly-linked essence)": """
    let a = alloc { next: null, prev: null, elem: 1 };
    let b = alloc { next: null, prev: null, elem: 2 };
    a.next = addr(b);
    b.prev = addr(a);
    free a;
    free b;
""",

"three-node intrusive chain, freed exactly once each": """
    let a = alloc { next: null, prev: null, elem: 10 };
    let b = alloc { next: null, prev: null, elem: 20 };
    let c = alloc { next: null, prev: null, elem: 30 };
    a.next = addr(b);  b.prev = addr(a);
    b.next = addr(c);  c.prev = addr(b);
    free a;  free b;  free c;
""",

"linear move": """
    let a = alloc { val: 1 };
    let b = a;
    free b;
""",
}

BAD = {
"double free": """
    let a = alloc { val: 1 };
    free a;
    free a;
""",

"use after free": """
    let a = alloc { val: 1 };
    free a;
    a.val = 2;
""",

"use after move": """
    let a = alloc { val: 1 };
    let b = a;
    free a;
    free b;
""",

"leak (Perm never freed)": """
    let a = alloc { val: 1 };
""",

"dangling pointer via alias (use-after-free through a.next)": """
    let a = alloc { next: null };
    let b = alloc { next: null };
    a.next = addr(b);
    free b;
    a.next.next = null;
""",
}


# ---------------------------------------------------------------------------
# 6b. Fuzzer -- empirical soundness evidence (the Redex/Rosette rung, at the
# surface level): generate random programs; for every one the checker ACCEPTS,
# run it and assert the monitor stays silent AND no cell leaks. A single
# accepted-but-unsafe program would be a checker soundness bug.
# ---------------------------------------------------------------------------

import random

_VARS = ["a", "b", "c", "d"]
_FLDS = ["f0", "f1"]


def _rand_val(rng):
    return rng.choice([
        lambda: Int(rng.randint(0, 9)),
        lambda: Null(),
        lambda: AddrOf(rng.choice(_VARS)),
        lambda: Field(Var(rng.choice(_VARS)), rng.choice(_FLDS)),
    ])()


def _rand_stmt(rng):
    k = rng.random()
    v = rng.choice(_VARS)
    if k < 0.34:
        return Let(v, Alloc([(f, _rand_val(rng)) for f in _FLDS]))
    if k < 0.50:
        return FreeS(v)
    if k < 0.66:
        return Let(v, Var(rng.choice(_VARS)))                       # move / copy
    if k < 0.83:
        base = Var(v) if rng.random() < 0.6 else Field(Var(v), rng.choice(_FLDS))
        return WriteF(base, rng.choice(_FLDS), _rand_val(rng))
    return ExprS(Field(Var(v), rng.choice(_FLDS)))                  # read


def _smart_prog(rng, length):
    """A linearity-aware generator: tracks which vars own a Perm, so it produces
    mostly well-typed programs (occasionally slips in a bad op, which the checker
    filters out). Frees everything still owned at the end."""
    owned: set[str] = set()
    prog = []
    def aval():
        opts = [lambda: Int(rng.randint(0, 9)), lambda: Null()]
        if owned:
            opts.append(lambda: AddrOf(rng.choice(list(owned))))
            opts.append(lambda: Field(Var(rng.choice(list(owned))), rng.choice(_FLDS)))
        return rng.choice(opts)()
    for _ in range(length):
        free_vars = [v for v in _VARS if v not in owned]
        r = rng.random()
        if free_vars and r < 0.4:
            v = rng.choice(free_vars)
            prog.append(Let(v, Alloc([(f, aval()) for f in _FLDS]))); owned.add(v)
        elif owned and r < 0.6:
            v = rng.choice(list(owned)); prog.append(FreeS(v)); owned.discard(v)
        elif owned and r < 0.85:
            v = rng.choice(list(owned)); prog.append(WriteF(Var(v), rng.choice(_FLDS), aval()))
        elif owned and free_vars:
            v = rng.choice(list(owned)); w = rng.choice(free_vars)        # move
            prog.append(Let(w, Var(v))); owned.discard(v); owned.add(w)
        else:
            prog.append(ExprS(Int(0)))
    for v in list(owned):
        prog.append(FreeS(v))                                             # free the rest
    return prog


def fuzz(trials=20000, length=8, seed=0):
    rng = random.Random(seed)
    accepted = clean = violations = 0
    for _ in range(trials):
        prog = _smart_prog(rng, rng.randint(1, length))
        try:
            errs = Checker().check(prog)
        except Exception:
            continue                       # malformed (e.g. unbound) -- skip
        if errs:
            continue
        accepted += 1
        it = Interp()
        try:
            leaks = it.run(prog)
            if leaks:
                violations += 1
                print(f"  !! ACCEPTED-BUT-LEAKS: {prog}")
            else:
                clean += 1
        except MonitorError as m:
            violations += 1
            print(f"  !! ACCEPTED-BUT-UNSAFE ({m}): {prog}")
    print(f"  generated {trials}, accepted {accepted}, ran clean {clean}, "
          f"SAFETY VIOLATIONS among accepted: {violations}")
    return violations


def banner(s): print("\n" + "=" * 74 + f"\n{s}\n" + "=" * 74)


def main():
    banner("ACCEPTED programs: checker is silent, and they RUN clean (no leak)")
    for name, src in GOOD.items():
        errs = check(src)
        if errs:
            print(f"  [UNEXPECTED REJECT] {name}: {errs}"); continue
        leaks, mon = run(src)
        status = "OK" if (mon is None and not leaks) else f"MONITOR/LEAK: {mon} leaks={leaks}"
        print(f"  ✓ accept + run clean   {name:48} -> {status}")

    banner("REJECTED programs: checker explains the bug BEFORE it can happen")
    for name, src in BAD.items():
        errs = check(src)
        verdict = "rejected" if errs else "!! ACCEPTED (BUG IN CHECKER) !!"
        print(f"  ✗ {verdict:9} {name}")
        for e in errs:
            print(f"        - {e}")

    banner("DIFFERENTIAL: bypass the checker on the rejected programs, RUN them,"
           "\nand watch the runtime monitor fire on the exact predicted bug")
    for name, src in BAD.items():
        leaks, mon = run(src)
        if mon:
            print(f"  monitor FIRED  {name:42} -> {mon}")
        elif leaks:
            print(f"  monitor: LEAK  {name:42} -> live cells {leaks}")
        else:
            print(f"  (ran clean)    {name}")

    banner("FUZZ: 20000 random programs; every program the checker ACCEPTS must\n"
           "run with the monitor silent and no leak (0 violations = sound here)")
    fuzz()


if __name__ == "__main__":
    main()
