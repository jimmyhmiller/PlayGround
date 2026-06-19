#!/usr/bin/env python3
"""
lambda-Tally POC: a SEPARATION-LOGIC verifier (the "flavor 3" / ghost
separating-bundle discipline).

Where the linear/region type systems (flavors 1-2) are decidable and automatic
but bounded in what they can express, separation logic is a *program logic*:
strictly more expressive (arbitrary inductive heap invariants), at the cost of
discharging proof obligations. This is a small, sound symbolic executor in the
Smallfoot/VeriFast style. It demonstrates the defining ingredients:

  * a symbolic HEAP = a separating conjunction of chunks
        p |-> {f: v, ...}     (a single owned cell)        and
        pred(args)            (an inductive predicate instance);
  * the FRAME rule: an operation touches only its footprint; the rest is carried
    through untouched (so a recursive call's spec frames the caller's cells);
  * inductive PREDICATES with fold / unfold;
  * functions verified against `requires` / `ensures` contracts;
  * memory safety falls out: read/write/free need the points-to chunk, so
    use-after-free / double-free are "no permission" errors, and a leftover
    chunk at the end is a leak (the postcondition is not established).

Run:  python3 seplogic.py

This file (part 1): the symbolic state (heap + pure facts via union-find) and
the primitive heap operations, validated on straight-line programs. Predicates
(fold/unfold) and contracts are added on top.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import copy as _copy
import itertools

NULL = "null"
_gensym = itertools.count()


def fresh(prefix="_t"):
    return f"{prefix}{next(_gensym)}"


# ===========================================================================
# Symbolic state:  pure facts (equalities / disequalities) + a heap of chunks
# ===========================================================================

# a chunk is one of:
#   ("pt",   addr_term, {field: term})     -- addr |-> {...}
#   ("pred", name, (arg_term, ...))         -- an inductive predicate instance


class VerifyError(Exception):
    pass


@dataclass
class State:
    parent: dict = field(default_factory=dict)        # union-find over terms
    diseq: list = field(default_factory=list)         # list of (a, b) known a != b
    heap: list = field(default_factory=list)          # the separating conjunction

    # ---- pure: union-find with disequality ----
    def find(self, t):
        self.parent.setdefault(t, t)
        root = t
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[t] != root:
            self.parent[t], t = root, self.parent[t]
        return root

    def eq(self, a, b) -> bool:
        return self.find(a) == self.find(b)

    def neq(self, a, b) -> bool:
        fa, fb = self.find(a), self.find(b)
        if fa == fb:
            return False
        for (x, y) in self.diseq:
            fx, fy = self.find(x), self.find(y)
            if {fx, fy} == {fa, fb}:
                return True
        return False

    def assume_eq(self, a, b) -> bool:
        """add a == b; return False if it contradicts a known disequality."""
        if self.neq(a, b):
            return False
        self.parent[self.find(a)] = self.find(b)
        return True

    def assume_neq(self, a, b) -> bool:
        """add a != b; return False if it contradicts a known equality."""
        if self.eq(a, b):
            return False
        self.diseq.append((a, b))
        return True

    def copy(self) -> "State":
        return _copy.deepcopy(self)

    # ---- heap: find / consume points-to chunks ----
    def find_pt(self, addr) -> Optional[int]:
        for i, ch in enumerate(self.heap):
            if ch[0] == "pt" and self.eq(ch[1], addr):
                return i
        return None


# ===========================================================================
# Primitive heap operations (the points-to fragment) over a single State
# ===========================================================================

def op_alloc(st: State, x: str, fields: dict):
    # a fresh allocation is a brand-new, separate cell (the * keeps it disjoint)
    st.heap.append(("pt", x, dict(fields)))


def op_read(st: State, y: str, p, f: str):
    i = st.find_pt(p)
    if i is None:
        raise VerifyError(f"read {p}.{f}: no points-to for {p} "
                          f"(out of footprint / possible use-after-free)")
    val = st.heap[i][2].get(f)
    if val is None:
        raise VerifyError(f"read {p}.{f}: cell has no field {f}")
    st.assume_eq(y, val)   # bind program var y to the stored term


def op_write(st: State, p, f: str, v):
    i = st.find_pt(p)
    if i is None:
        raise VerifyError(f"write {p}.{f}: no points-to for {p} "
                          f"(out of footprint / possible use-after-free)")
    st.heap[i][2][f] = v


def op_free(st: State, p):
    i = st.find_pt(p)
    if i is None:
        raise VerifyError(f"free {p}: no points-to for {p} (double free / use-after-free)")
    del st.heap[i]


# ===========================================================================
# Inductive predicates, fold / unfold, contracts, and the symbolic executor
# ===========================================================================
#
# A predicate case is (params, locals, guard, chunks):
#   lseg(x, y) :=  (x = y  &  emp)
#              \/  (x != y &  x |-> {next: z}  *  lseg(z, y))     -- z local
#
# PREDS:  name -> [case, ...]
# FUNCS:  name -> (params, pre, post)            (contract, for calls)
# BODIES: name -> (params, pre, post, body)      (to verify)
# pre / post are  (pure_atoms, chunks).  An atom is ("eq"|"neq", a, b).

PREDS: dict = {}
FUNCS: dict = {}
BODIES: dict = {}


def define_pred(name, cases):
    PREDS[name] = cases


def define_fn(name, params, pre, post, body):
    FUNCS[name] = (params, pre, post)
    BODIES[name] = (params, pre, post, body)


def _inst_term(t, m):
    return m.get(t, t)


def _inst_atom(atom, m):
    return (atom[0], _inst_term(atom[1], m), _inst_term(atom[2], m))


def _inst_chunk(ch, m):
    if ch[0] == "pt":
        return ("pt", _inst_term(ch[1], m), {f: _inst_term(v, m) for f, v in ch[2].items()})
    return ("pred", ch[1], tuple(_inst_term(a, m) for a in ch[2]))


# ---- matching chunks against the heap (with existential `holes`) ----

def _match_term(req, got, st, binds, holes):
    if req in holes:
        if req in binds:
            return st.eq(binds[req], got)
        binds[req] = got
        return True
    return st.eq(req, got)


def _match_chunk(req, ch, st, binds, holes):
    if req[0] != ch[0]:
        return False
    if req[0] == "pt":
        if not _match_term(req[1], ch[1], st, binds, holes):
            return False
        for f, v in req[2].items():
            if f not in ch[2] or not _match_term(v, ch[2][f], st, binds, holes):
                return False
        return True
    return (req[1] == ch[1] and len(req[2]) == len(ch[2])
            and all(_match_term(a, b, st, binds, holes) for a, b in zip(req[2], ch[2])))


def consume(st: State, reqs, holes):
    """Match & remove each required chunk from st.heap (frame = what's left).
    Returns the existential bindings, or None if some chunk can't be matched."""
    binds = {}
    heap = list(st.heap)
    for req in reqs:
        hit = None
        for i, ch in enumerate(heap):
            trial = dict(binds)
            if _match_chunk(req, ch, st, trial, holes):
                hit, binds = i, trial
                break
        if hit is None:
            return None
        heap.pop(hit)
    st.heap = heap
    return binds


def _assume(st, atom):
    return st.assume_eq(atom[1], atom[2]) if atom[0] == "eq" else st.assume_neq(atom[1], atom[2])


def _entailed(st, atom):
    return st.eq(atom[1], atom[2]) if atom[0] == "eq" else st.neq(atom[1], atom[2])


def unfold(st: State, name, args):
    idx = next((i for i, ch in enumerate(st.heap)
                if ch[0] == "pred" and ch[1] == name and len(ch[2]) == len(args)
                and all(st.eq(a, b) for a, b in zip(ch[2], args))), None)
    if idx is None:
        raise VerifyError(f"unfold {name}{tuple(args)}: no such predicate instance")
    out = []
    for (params, locals_, guard, chunks) in PREDS[name]:
        st2 = st.copy()
        del st2.heap[idx]
        m = dict(zip(params, args))
        for L in locals_:
            m[L] = fresh("z")
        if all(_assume(st2, _inst_atom(g, m)) for g in guard):  # prune infeasible cases
            st2.heap += [_inst_chunk(ch, m) for ch in chunks]
            out.append(st2)
    return out


def fold(st: State, name, args):
    for (params, locals_, guard, chunks) in PREDS[name]:
        st2 = st.copy()
        m = dict(zip(params, args))
        holes = set()
        for L in locals_:
            h = fresh("h")
            m[L], _ = h, holes.add(h)
        if not all(_entailed(st2, _inst_atom(g, m)) for g in guard):
            continue
        if consume(st2, [_inst_chunk(ch, m) for ch in chunks], holes) is not None:
            st2.heap.append(("pred", name, tuple(args)))
            return [st2]
    raise VerifyError(f"fold {name}{tuple(args)}: no predicate case matches the heap")


def call(st: State, fname, args):
    (params, (pre_pure, pre_chunks), (post_pure, post_chunks)) = FUNCS[fname]
    m = dict(zip(params, args))
    for g in pre_pure:
        if not _entailed(st, _inst_atom(g, m)):
            raise VerifyError(f"call {fname}: precondition {_inst_atom(g, m)} not entailed")
    if consume(st, [_inst_chunk(c, m) for c in pre_chunks], set()) is None:
        raise VerifyError(f"call {fname}: precondition heap not satisfied")
    # (st.heap now holds the FRAME — untouched by the call)
    for g in post_pure:
        _assume(st, _inst_atom(g, m))
    st.heap += [_inst_chunk(c, m) for c in post_chunks]
    return [st]


def exec_stmt(st: State, s):
    op = s[0]
    if op == "alloc":
        op_alloc(st, s[1], s[2]); return [st]
    if op == "read":
        op_read(st, s[1], s[2], s[3]); return [st]
    if op == "write":
        op_write(st, s[1], s[2], s[3]); return [st]
    if op == "free":
        op_free(st, s[1]); return [st]
    if op == "assume":
        return [st] if _assume(st, s[1]) else []
    if op == "unfold":
        return unfold(st, s[1], s[2])
    if op == "fold":
        return fold(st, s[1], s[2])
    if op == "call":
        return call(st, s[1], s[2])
    if op == "return":
        st.assume_eq("ret", s[1]); return [st]
    if op == "if":
        cond = s[1]
        neg = ("neq", cond[1], cond[2]) if cond[0] == "eq" else ("eq", cond[1], cond[2])
        out = []
        a = st.copy()
        if _assume(a, cond):
            out += exec_stmts([a], s[2])
        b = st.copy()
        if _assume(b, neg):
            out += exec_stmts([b], s[3])
        return out
    raise VerifyError(f"unknown statement {op}")


def exec_stmts(states, stmts):
    cur = states
    for s in stmts:
        cur = [r for st in cur for r in exec_stmt(st, s)]
    return cur


def verify_fn(name):
    (params, (pre_pure, pre_chunks), (post_pure, post_chunks), body) = BODIES[name]
    st = State()
    for g in pre_pure:
        _assume(st, g)
    st.heap += [_inst_chunk(c, {}) for c in pre_chunks]
    errs = []
    try:
        finals = exec_stmts([st], body)
    except VerifyError as e:
        return [str(e)]
    for fst in finals:
        s2 = fst.copy()
        if consume(s2, [_inst_chunk(c, {}) for c in post_chunks], set()) is None:
            errs.append(f"postcondition heap not established (final heap: {fst.heap})")
            continue
        if s2.heap:
            errs.append(f"leak: leftover heap not covered by the postcondition: {s2.heap}")
            continue
        for g in post_pure:
            if not _entailed(s2, g):
                errs.append(f"postcondition {g} not entailed")
    return errs


# ---------------------------------------------------------------------------
# quick self-check of the points-to core (part 1)
# ---------------------------------------------------------------------------

def _selfcheck():
    # alloc; write; read; free  -- should all succeed, heap empty at the end
    st = State()
    op_alloc(st, "a", {"val": NULL})
    op_write(st, "a", "val", "x")
    op_read(st, "y", "a", "val")     # y == x
    assert st.eq("y", "x")
    op_free(st, "a")
    assert st.heap == [], st.heap

    # double free is caught
    st = State()
    op_alloc(st, "a", {"val": NULL})
    op_free(st, "a")
    try:
        op_free(st, "a")
        assert False, "double free not caught"
    except VerifyError:
        pass

    # use-after-free is caught
    st = State()
    op_alloc(st, "a", {"next": NULL})
    op_free(st, "a")
    try:
        op_read(st, "z", "a", "next")
        assert False, "use-after-free not caught"
    except VerifyError:
        pass

    print("part 1 (points-to core): ok")


# ===========================================================================
# Demo: a singly-linked list segment, verified by separation logic
# ===========================================================================

def banner(s):
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)


def setup_lseg():
    # lseg(x, y) :=  (x = y & emp)  \/  (x != y & x|->{next:z} * lseg(z, y))
    define_pred("lseg", [
        (["x", "y"], [], [("eq", "x", "y")], []),
        (["x", "y"], ["z"], [("neq", "x", "y")],
         [("pt", "x", {"next": "z"}), ("pred", "lseg", ("z", "y"))]),
    ])


def demo():
    _selfcheck()
    setup_lseg()

    # ---- GOOD: recursively free an entire list segment ----
    #   dispose(x)  requires lseg(x, null)  ensures emp
    define_fn(
        "dispose", ["x"],
        pre=([], [("pred", "lseg", ("x", NULL))]),
        post=([], []),
        body=[
            ("unfold", "lseg", ["x", NULL]),
            ("if", ("eq", "x", NULL),
             [],  # base: x = null, heap is emp
             [("read", "z", "x", "next"),
              ("call", "dispose", ["z"]),     # consumes lseg(z,null); FRAMES x|->{next:z}
              ("free", "x")]),                # then frees the framed cell
        ],
    )

    # ---- GOOD: prepend a node (uses fold to re-establish the invariant) ----
    #   push(x, v)  requires lseg(x, null)  ensures lseg(ret, null)
    define_fn(
        "push", ["x", "v"],
        pre=([], [("pred", "lseg", ("x", NULL))]),
        post=([], [("pred", "lseg", ("ret", NULL))]),
        body=[
            ("alloc", "n", {"next": "x", "val": "v"}),
            ("assume", ("neq", "n", NULL)),   # a fresh allocation is non-null
            ("fold", "lseg", ["n", NULL]),    # n|->{next:x} * lseg(x,null)  ==>  lseg(n,null)
            ("return", "n"),
        ],
    )

    # ---- BAD: forget to free the head (leak) ----
    define_fn(
        "dispose_leak", ["x"],
        pre=([], [("pred", "lseg", ("x", NULL))]),
        post=([], []),
        body=[
            ("unfold", "lseg", ["x", NULL]),
            ("if", ("eq", "x", NULL), [],
             [("read", "z", "x", "next"),
              ("call", "dispose", ["z"])]),    # missing `free x`
        ],
    )

    # ---- BAD: double free ----
    define_fn(
        "dispose_double", ["x"],
        pre=([], [("pred", "lseg", ("x", NULL))]),
        post=([], []),
        body=[
            ("unfold", "lseg", ["x", NULL]),
            ("if", ("eq", "x", NULL), [],
             [("read", "z", "x", "next"),
              ("call", "dispose", ["z"]),
              ("free", "x"),
              ("free", "x")]),                 # double free of the head
        ],
    )

    # ---- BAD: build a node but forget to fold the invariant ----
    define_fn(
        "push_nofold", ["x", "v"],
        pre=([], [("pred", "lseg", ("x", NULL))]),
        post=([], [("pred", "lseg", ("ret", NULL))]),
        body=[
            ("alloc", "n", {"next": "x", "val": "v"}),
            ("return", "n"),                   # never folded lseg(n, null)
        ],
    )

    good = ["dispose", "push"]
    bad = ["dispose_leak", "dispose_double", "push_nofold"]

    banner("VERIFIED: separation-logic contracts hold (frame rule, fold/unfold)")
    for f in good:
        errs = verify_fn(f)
        print(f"  {'✓ verified' if not errs else '✗ FAILED  '}  {f}")
        for e in errs:
            print(f"        - {e}")

    banner("REJECTED: the proof obligation cannot be discharged")
    for f in bad:
        errs = verify_fn(f)
        print(f"  {'✗ rejected' if errs else '!! ACCEPTED (UNSOUND)'}  {f}")
        for e in errs[:1]:
            print(f"        - {e}")


if __name__ == "__main__":
    demo()
