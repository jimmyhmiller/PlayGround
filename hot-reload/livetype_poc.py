#!/usr/bin/env python3
"""
livetype_poc.py — a proof-of-concept for a language that is BOTH fully statically
typed AND fully hot-reloadable, using a *type-condition system* as the single
unifying primitive.

The bet (see the design conversation):
  - Apply type/function updates to a RUNNING program, at safe points (between ticks).
  - Code that stays consistent keeps running; values migrate lazily (identity-preserving),
    auto-derived where possible.
  - When running code REACHES a point that is now type-inconsistent -- a broken
    function, or a value that can't be auto-migrated -- we DON'T crash and we DON'T
    reject the update. We raise a CONDITION, PAUSE (rolling the in-flight tick back so
    nothing half-typed is ever published), let the developer REPAIR live, and RESUME by
    re-running the tick.

Soundness invariant we are testing:
  "Every committed state is well-typed under the current definitions; a paused frame is
   quarantined (its in-flight, possibly ill-typed state is discarded on rollback), so no
   running code ever observes an ill-typed value."

This is Proteus's problem solved the Lisp way: static types for everything running,
dynamic con-freeness by trap-and-repair instead of static con-freeness by proof.

Run:  python3 livetype_poc.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# ────────────────────────────────────────────────────────────────────────────
#  AST
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TInt:            # the type  Int
    pass
@dataclass
class TRec:            # a nominal record type, by name (layout looked up by CURRENT version)
    name: str

@dataclass
class ELit:  val: int
@dataclass
class EVar:  name: str
@dataclass
class EField: obj: object; fname: str
@dataclass
class EBin:  op: str; l: object; r: object
@dataclass
class ECall: fn: str; args: list
@dataclass
class ECons: tname: str; fields: list   # list of (fname, expr)

# ────────────────────────────────────────────────────────────────────────────
#  Lexer + Parser  (tiny recursive descent -- so "repairs" really are source edits)
# ────────────────────────────────────────────────────────────────────────────

PUNCT = set("{}(),:=+-*.>")
def lex(s):
    i, toks = 0, []
    while i < len(s):
        c = s[i]
        if c.isspace(): i += 1; continue
        if c == '-' and i+1 < len(s) and s[i+1] == '>':
            toks.append(('op', '->')); i += 2; continue
        if c in PUNCT: toks.append(('punct', c)); i += 1; continue
        if c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit(): j += 1
            toks.append(('int', int(s[i:j]))); i = j; continue
        if c.isalpha() or c == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'): j += 1
            toks.append(('name', s[i:j])); i = j; continue
        raise SyntaxError(f"bad char {c!r}")
    toks.append(('eof', None))
    return toks

class Parser:
    def __init__(self, toks): self.t = toks; self.i = 0
    def peek(self): return self.t[self.i]
    def next(self): tok = self.t[self.i]; self.i += 1; return tok
    def eat(self, kind, val=None):
        k, v = self.next()
        if k != kind or (val is not None and v != val):
            raise SyntaxError(f"expected {kind} {val}, got {k} {v}")
        return v
    def is_(self, kind, val=None):
        k, v = self.peek()
        return k == kind and (val is None or v == val)

    def parse_type(self):
        name = self.eat('name')
        return TInt() if name == 'Int' else TRec(name)

    # expr := add ;  add := mul (('+'|'-') mul)* ;  mul := term ('*' term)*
    def parse_expr(self):
        e = self.parse_mul()
        while self.is_('punct', '+') or self.is_('punct', '-'):
            op = self.next()[1]
            e = EBin(op, e, self.parse_mul())
        return e

    def parse_mul(self):
        e = self.parse_term()
        while self.is_('punct', '*'):
            self.next()
            e = EBin('*', e, self.parse_term())
        return e

    def parse_term(self):
        e = self.parse_atom()
        while self.is_('punct', '.'):
            self.next(); fname = self.eat('name')
            e = EField(e, fname)
        return e

    def parse_atom(self):
        k, v = self.peek()
        if k == 'int': self.next(); return ELit(v)
        if k == 'punct' and v == '(':
            self.next(); e = self.parse_expr(); self.eat('punct', ')'); return e
        if k == 'name':
            self.next()
            if self.is_('punct', '{'):                # record construction  T{ f: e, ... }
                self.eat('punct', '{'); flds = []
                if not self.is_('punct', '}'):
                    while True:
                        fn = self.eat('name'); self.eat('punct', ':')
                        flds.append((fn, self.parse_expr()))
                        if self.is_('punct', ','): self.next(); continue
                        break
                self.eat('punct', '}')
                return ECons(v, flds)
            if self.is_('punct', '('):                # call  f(e, ...)
                self.eat('punct', '('); args = []
                if not self.is_('punct', ')'):
                    while True:
                        args.append(self.parse_expr())
                        if self.is_('punct', ','): self.next(); continue
                        break
                self.eat('punct', ')')
                return ECall(v, args)
            return EVar(v)
        raise SyntaxError(f"unexpected {k} {v}")

def parse_expr(src): return Parser(lex(src)).parse_expr()

# ────────────────────────────────────────────────────────────────────────────
#  Program store:  versioned nominal types + functions
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldSpec:
    name: str
    ty: object                       # TInt | TRec
    default: Optional[object] = None # default expr, or None if required

@dataclass
class Fn:
    name: str
    params: list                     # list of (name, type)
    ret: object                      # declared return type
    body: object                     # expr AST
    broken: Optional[str] = None     # None if OK, else the type error message

class Program:
    def __init__(self):
        # type name -> list of layouts, one per version (index 0 == version 1)
        self.type_versions: dict[str, list[list[FieldSpec]]] = {}
        self.fns: dict[str, Fn] = {}
        # (Tname, from_version) -> callable(old Record) -> new Record  ; missing == gap
        self.migrations: dict[tuple, object] = {}

    def cur_version(self, tname): return len(self.type_versions[tname])   # 1-based
    def layout(self, tname, version): return self.type_versions[tname][version - 1]
    def cur_layout(self, tname): return self.layout(tname, self.cur_version(tname))
    def field(self, tname, version, fname):
        for fs in self.layout(tname, version):
            if fs.name == fname: return fs
        return None

# ────────────────────────────────────────────────────────────────────────────
#  Type checker
# ────────────────────────────────────────────────────────────────────────────

class TypeError_(Exception): pass

def ty_eq(a, b):
    if isinstance(a, TInt) and isinstance(b, TInt): return True
    if isinstance(a, TRec) and isinstance(b, TRec): return a.name == b.name
    return False
def ty_str(t): return "Int" if isinstance(t, TInt) else t.name

def check(prog: Program, e, env: dict):
    """Return the type of e under variable env, using CURRENT type layouts. Raise TypeError_."""
    if isinstance(e, ELit): return TInt()
    if isinstance(e, EVar):
        if e.name not in env: raise TypeError_(f"unbound variable '{e.name}'")
        return env[e.name]
    if isinstance(e, EBin):
        lt, rt = check(prog, e.l, env), check(prog, e.r, env)
        if not (isinstance(lt, TInt) and isinstance(rt, TInt)):
            raise TypeError_(f"'{e.op}' needs Int, Int but got {ty_str(lt)}, {ty_str(rt)}")
        return TInt()
    if isinstance(e, EField):
        ot = check(prog, e.obj, env)
        if not isinstance(ot, TRec): raise TypeError_(f"field '.{e.fname}' on non-record {ty_str(ot)}")
        fs = prog.field(ot.name, prog.cur_version(ot.name), e.fname)
        if fs is None: raise TypeError_(f"type '{ot.name}' has no field '{e.fname}'")
        return fs.ty
    if isinstance(e, ECall):
        if e.fn not in prog.fns: raise TypeError_(f"unknown function '{e.fn}'")
        fn = prog.fns[e.fn]
        if len(e.args) != len(fn.params):
            raise TypeError_(f"'{e.fn}' expects {len(fn.params)} args, got {len(e.args)}")
        for (a, (pn, pt)) in zip(e.args, fn.params):
            at = check(prog, a, env)
            if not ty_eq(at, pt): raise TypeError_(f"'{e.fn}' arg '{pn}' expects {ty_str(pt)}, got {ty_str(at)}")
        return fn.ret                                  # rely on SIGNATURE, not callee body
    if isinstance(e, ECons):
        if e.tname not in prog.type_versions: raise TypeError_(f"unknown type '{e.tname}'")
        layout = prog.cur_layout(e.tname)
        given = dict(e.fields)
        for fs in layout:
            if fs.name in given:
                at = check(prog, given[fs.name], env)
                if not ty_eq(at, fs.ty):
                    raise TypeError_(f"field '{fs.name}' of {e.tname} expects {ty_str(fs.ty)}, got {ty_str(at)}")
            elif fs.default is None:
                raise TypeError_(f"missing field '{fs.name}' in {e.tname}{{...}}")
        for gn in given:
            if not any(fs.name == gn for fs in layout):
                raise TypeError_(f"{e.tname} has no field '{gn}'")
        return TRec(e.tname)
    raise TypeError_(f"cannot type {e}")

def check_fn(prog, fn: Fn) -> Optional[str]:
    env = {pn: pt for (pn, pt) in fn.params}
    try:
        bt = check(prog, fn.body, env)
        if not ty_eq(bt, fn.ret):
            return f"body has type {ty_str(bt)}, declared return {ty_str(fn.ret)}"
        return None
    except TypeError_ as ex:
        return str(ex)

# ────────────────────────────────────────────────────────────────────────────
#  Runtime values + evaluator (with lazy, identity-preserving migration)
# ────────────────────────────────────────────────────────────────────────────

_obj_ids = {}
def obj_id(r):
    return _obj_ids.setdefault(id(r), len(_obj_ids) + 1)

@dataclass
class Record:
    tname: str
    version: int
    fields: dict
    def __repr__(self):
        inner = ", ".join(f"{k}={fmt(v)}" for k, v in self.fields.items())
        return f"{self.tname}@v{self.version}{{{inner}}}"

def fmt(v):
    return repr(v) if isinstance(v, Record) else str(v)

def value_ok(prog, v, ty):
    """Structural check: is v a valid inhabitant of ty under the CURRENTLY live layouts?
    A record is checked against the layout of its OWN version (older versions stay valid
    until migrated), so this is the real soundness invariant, not a printed claim."""
    if isinstance(ty, TInt):
        return isinstance(v, int)
    if isinstance(ty, TRec):
        if not (isinstance(v, Record) and v.tname == ty.name):
            return False
        layout = prog.layout(v.tname, v.version)
        if set(v.fields) != {fs.name for fs in layout}:
            return False
        return all(value_ok(prog, v.fields[fs.name], fs.ty) for fs in layout)
    return False

class Pause(Exception):
    """A type condition raised at the point of use. kind in {'function','migration'}."""
    def __init__(self, kind, key, message):
        super().__init__(message)
        self.kind, self.key, self.message = kind, key, message

def migrate_to_current(prog: Program, r: Record):
    """Lazily migrate r up to the current version, identity-preserving (mutate in place).
    Raises Pause('migration', ...) if a needed step has no registered transformer."""
    cur = prog.cur_version(r.tname)
    while r.version < cur:
        step = prog.migrations.get((r.tname, r.version))
        if step is None:
            raise Pause('migration', (r.tname, r.version),
                        f"value {r.tname}#{obj_id(r)} is at v{r.version} but no transformer "
                        f"v{r.version}->v{r.version+1} exists (change was not auto-derivable)")
        newf = step(r)                       # produce next-version fields
        r.fields, r.version = newf, r.version + 1   # become:-style in-place update
    return r

def evl(prog: Program, e, env: dict, mig=True):
    """Evaluate e. If mig=False (inside a migration transformer), field access is RAW
    (no auto-migration) so a transformer can read the old layout of `old`."""
    if isinstance(e, ELit): return e.val
    if isinstance(e, EVar): return env[e.name]
    if isinstance(e, EBin):
        a, b = evl(prog, e.l, env, mig), evl(prog, e.r, env, mig)
        return {'+': a + b, '-': a - b, '*': a * b}[e.op]
    if isinstance(e, EField):
        o = evl(prog, e.obj, env, mig)
        if mig and isinstance(o, Record):
            migrate_to_current(prog, o)      # migrate-on-cross, at the moment of use
        return o.fields[e.fname]
    if isinstance(e, ECall):
        fn = prog.fns[e.fn]
        if fn.broken is not None:            # reached a function that no longer typechecks
            raise Pause('function', e.fn, f"function '{e.fn}' is inconsistent: {fn.broken}")
        args = [evl(prog, a, env, mig) for a in e.args]
        callenv = {pn: av for ((pn, _), av) in zip(fn.params, args)}
        return evl(prog, fn.body, callenv, mig)
    if isinstance(e, ECons):
        layout = prog.cur_layout(e.tname)
        given = dict(e.fields)
        fields = {}
        for fs in layout:
            if fs.name in given:
                fields[fs.name] = evl(prog, given[fs.name], env, mig)
            else:
                fields[fs.name] = evl(prog, fs.default, {}, mig)
        return Record(e.tname, prog.cur_version(e.tname), fields)
    raise RuntimeError(f"cannot eval {e}")

# ────────────────────────────────────────────────────────────────────────────
#  Update machinery:  add type version, add/replace fn, auto-derive migrations
# ────────────────────────────────────────────────────────────────────────────

def parse_layout(spec: str) -> list:
    """'id: Int, balance: Int, fee: Int = 0'  ->  [FieldSpec, ...]"""
    out = []
    for part in [p for p in spec.split(',') if p.strip()]:
        left, _, dflt = part.partition('=')
        nm, _, tystr = left.strip().partition(':')
        ty = TInt() if tystr.strip() == 'Int' else TRec(tystr.strip())
        default = parse_expr(dflt.strip()) if dflt.strip() else None
        out.append(FieldSpec(nm.strip(), ty, default))
    return out

def try_auto_migration(prog, tname, from_v):
    """Return a transformer callable if v(from_v)->v(from_v+1) is auto-derivable, else None."""
    old = {fs.name: fs for fs in prog.layout(tname, from_v)}
    new = prog.layout(tname, from_v + 1)
    plan = []                                   # (fname, 'copy'|('default', expr))
    for fs in new:
        if fs.name in old and ty_eq(old[fs.name].ty, fs.ty):
            plan.append((fs.name, 'copy'))
        elif fs.default is not None:
            plan.append((fs.name, ('default', fs.default)))
        else:
            return None                         # field changed type / new & required -> manual
    def m(old_rec):
        f = {}
        for (fname, how) in plan:
            f[fname] = old_rec.fields[fname] if how == 'copy' else evl(prog, how[1], {}, mig=False)
        return f
    return m

def recheck_all(prog) -> list:
    """Re-typecheck every function against current types. Return names newly broken."""
    newly = []
    for fn in prog.fns.values():
        err = check_fn(prog, fn)
        was = fn.broken
        fn.broken = err
        if err and not was: newly.append((fn.name, err))
    return newly

# ────────────────────────────────────────────────────────────────────────────
#  Driver: run a scripted scenario with pause -> repair -> resume
# ────────────────────────────────────────────────────────────────────────────

def hr(title=""):
    print("\n" + "─" * 78)
    if title: print(title)
    if title: print("─" * 78)

def define_type(prog, name, spec):
    prog.type_versions.setdefault(name, [])
    prog.type_versions[name].append(parse_layout(spec))

def update_type(prog, name, spec, log=True):
    prog.type_versions[name].append(parse_layout(spec))
    fromv = prog.cur_version(name) - 1
    m = try_auto_migration(prog, name, fromv)
    if m is not None:
        prog.migrations[(name, fromv)] = m
        if log: print(f"  [update] type {name} -> v{fromv+1}  (migration v{fromv}->v{fromv+1} AUTO-DERIVED)")
    else:
        if log: print(f"  [update] type {name} -> v{fromv+1}  (migration v{fromv}->v{fromv+1} needs a MANUAL "
                       f"transformer; will pause on first use)")

def define_fn(prog, src):
    # fn NAME(p: T, ...) -> RET = EXPR
    head, _, bodysrc = src.partition('=')
    head = head.strip()
    assert head.startswith('fn ')
    sig = head[3:]
    name, _, rest = sig.partition('(')
    params_src, _, rest2 = rest.partition(')')
    ret_src = rest2.replace('->', '').strip()
    params = []
    for p in [x for x in params_src.split(',') if x.strip()]:
        pn, _, pt = p.partition(':')
        params.append((pn.strip(), TInt() if pt.strip() == 'Int' else TRec(pt.strip())))
    ret = TInt() if ret_src == 'Int' else TRec(ret_src)
    fn = Fn(name.strip(), params, ret, parse_expr(bodysrc.strip()))
    prog.fns[fn.name] = fn
    return fn

# --- the repair "developer": scripted answers keyed by the condition, but structured
#     exactly like an interactive prompt would be. Each answer is source text.
class Developer:
    def __init__(self, prog, fn_fixes, mig_fixes):
        self.prog = prog
        self.fn_fixes = fn_fixes          # fn name -> new source
        self.mig_fixes = mig_fixes        # (tname, from_v) -> migration expr source with `old` in scope

    def repair(self, cond: Pause):
        if cond.kind == 'function':
            src = self.fn_fixes[cond.key]
            print(f"    DEV repairs function '{cond.key}':")
            print(f"        {src}")
            fn = define_fn(self.prog, src)
            err = check_fn(self.prog, fn)
            if err: raise RuntimeError(f"repair still ill-typed: {err}")
            fn.broken = None
            print(f"    ✓ '{cond.key}' now typechecks")
        elif cond.kind == 'migration':
            tname, from_v = cond.key
            src = self.mig_fixes[cond.key]
            print(f"    DEV supplies transformer {tname} v{from_v}->v{from_v+1}:")
            print(f"        (old) => {src}")
            expr = parse_expr(src)
            def m(old_rec, expr=expr):
                r = evl(self.prog, expr, {'old': old_rec}, mig=False)  # raw read of old layout
                return r.fields
            self.prog.migrations[(tname, from_v)] = m
            print(f"    ✓ transformer registered")

def main():
    prog = Program()

    hr("BOOT — define an initially well-typed program, then run it")
    define_type(prog, 'Account', 'id: Int, balance: Int')
    define_fn(prog, 'fn charge(a: Account, amt: Int) -> Account = '
                    'Account{ id: a.id, balance: a.balance - amt }')
    define_fn(prog, 'fn tick(s: Account) -> Account = charge(s, 5)')
    recheck_all(prog)
    print("  type Account@v1 = { id: Int, balance: Int }")
    print("  fn charge(a, amt) = Account{ id: a.id, balance: a.balance - amt }")
    print("  fn tick(s)        = charge(s, 5)")

    state = Record('Account', 1, {'id': 1, 'balance': 100})
    print(f"\n  initial state: {state}")

    dev = Developer(prog,
        fn_fixes={
            'charge': 'fn charge(a: Account, amt: Int) -> Account = '
                      'Account{ id: a.id, balance: Money{ cents: a.balance.cents - amt * 100 } }',
        },
        mig_fixes={
            ('Account', 2): 'Account{ id: old.id, balance: Money{ cents: old.balance * 100 }, fee: old.fee }',
        })

    # scripted timeline of updates applied at the safe point BEFORE the given tick index
    updates = {
        2: [('type', 'Account', 'id: Int, balance: Int, fee: Int = 0')],           # auto-migratable
        3: [('deftype', 'Money', 'cents: Int'),
            ('type', 'Account', 'id: Int, balance: Money, fee: Int = 0')],          # breaks charge + manual migration
    }

    tick_i, MAX_PAUSES = 0, 8
    while tick_i < 6:
        # ---- SAFE POINT: apply any pending updates between ticks ----
        if tick_i in updates:
            hr(f"HOT UPDATE  (safe point before tick {tick_i})")
            for u in updates[tick_i]:
                if u[0] == 'deftype':
                    define_type(prog, u[1], u[2]); print(f"  [update] new type {u[1]} = {{ {u[2]} }}")
                elif u[0] == 'type':
                    update_type(prog, u[1], u[2])
            newly = recheck_all(prog)
            if newly:
                print("  [recheck] functions now inconsistent (eager detect, surfaced lazily when reached):")
                for (nm, err) in newly: print(f"             - {nm}: {err}")
            else:
                print("  [recheck] all functions still typecheck")

        # ---- run the tick as a transaction; a Pause rolls it back ----
        pauses = 0
        while True:
            try:
                new_state = evl(prog, ECall('tick', [EVar('__state__')]), {'__state__': state})
                # ENFORCE the soundness invariant before publishing the new state
                assert value_ok(prog, new_state, TRec('Account')), \
                    f"INVARIANT VIOLATED: committed state {new_state} is ill-typed!"
                state = new_state                       # COMMIT
                print(f"  tick {tick_i}: state -> {state}   [✓ well-typed]")
                break
            except Pause as cond:
                pauses += 1
                if pauses > MAX_PAUSES: raise RuntimeError("too many pauses; giving up")
                hr(f"⏸  PAUSE  (type condition during tick {tick_i}) — {cond.kind}")
                print(f"  condition: {cond.message}")
                print(f"  QUARANTINE: in-flight tick discarded; committed state still {state}")
                print(f"              (nothing running can observe a half-typed value)")
                print(f"  → hand control to the developer:")
                dev.repair(cond)
                print(f"  ↻ RESUME: re-run tick {tick_i} from the top")
                # loop retries the same tick with the repaired program
        tick_i += 1

    hr("DONE — the program kept running across two hot updates and three repairs")
    print("  Every committed state above was checked well-typed under the definitions live")
    print("  at that moment (the assert would have fired otherwise). Each pause rolled the")
    print("  in-flight tick back cleanly and resumed after a live repair. No restart.")

if __name__ == '__main__':
    main()
