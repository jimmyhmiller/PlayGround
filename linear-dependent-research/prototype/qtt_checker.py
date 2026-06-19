#!/usr/bin/env python3
"""
lambda-Tally, Stage A: an executable bidirectional type checker for the
Quantitative Type Theory (QTT) core.

This is the "get a handle on it" artifact for docs/01-qtt-core.md. It is a
*semi-formal executable specification*: small enough to read as a definition,
runnable so you can watch the multiplicity ledger accept and reject programs.

What it demonstrates (see the test suite at the bottom / run this file):
  * dependent functions (Pi) with per-argument multiplicities {0, 1, w};
  * the *cohabitation* of linear and dependent typing: a variable can appear
    in TYPES freely (multiplicity 0) yet be spent linearly at runtime;
  * erasure: type-level use costs 0, so the dependent layer is runtime-free;
  * linearity errors (use-twice, drop-a-linear, use-an-erased-value).

Design (kept deliberately standard so it ports to Haskell/pi-forall cleanly):
  * de Bruijn *indices* for terms; de Bruijn *levels* for runtime values;
  * Normalization by Evaluation (NbE) for definitional equality / conversion;
  * a "resourced" bidirectional algorithm: check/infer return a *usage vector*
    (one multiplicity per context variable). Contexts are combined with the rig
    operations `+` (addU) and `*` (scaleU). This is the algorithmic reflection
    of QTT's multiplicity-annotated contexts.

NOT covered here (on purpose; that is Stage B / later):
  * the memory primitives alloc/read/write/free and the heap operational
    semantics  -> those live in the PLT Redex model (memory model.rkt);
  * a universe hierarchy (we use Type : Type, fine for exploration);
  * fully dependent let-pair elimination (motive); ours is non-dependent.

Run:  python3 "qtt checker.py"
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. The rig (semiring of multiplicities)  R = {0, 1, w}
# ---------------------------------------------------------------------------
# We use the strings "0", "1", "w". See docs/01-qtt-core.md S1.
ZERO, ONE, MANY = "0", "1", "w"

def m_plus(a: str, b: str) -> str:
    if a == ZERO: return b
    if b == ZERO: return a
    return MANY                      # 1+1 = 1+w = w+_ = w

def m_times(a: str, b: str) -> str:
    if a == ZERO or b == ZERO: return ZERO
    if a == ONE: return b
    if b == ONE: return a
    return MANY                      # w*w = w

def m_leq(used: str, allowed: str) -> bool:
    """Is a computed usage `used` permitted by the declared budget `allowed`?
    Order: x <= x, and x <= w for all x. 0 and 1 are INCOMPARABLE.
    Consequences:
      * declared 1 (linear): only used==1 is ok  => "exactly once".
      * declared 0 (erased): only used==0 is ok   => no runtime use.
      * declared w (unrestricted): anything ok.
    """
    if allowed == MANY: return True
    return used == allowed

# ---------------------------------------------------------------------------
# 2. Syntax (de Bruijn indices). Binders introduce one variable each, except
#    LetPair which introduces two (x then y).
# ---------------------------------------------------------------------------
@dataclass(eq=True)
class Term: pass

@dataclass(eq=True)
class Var(Term):   i: int                      # de Bruijn index (0 = innermost)
@dataclass(eq=True)
class Univ(Term):  pass                         # Type : Type
@dataclass(eq=True)
class Pi(Term):    m: str; dom: Term; cod: Term # (x :^m dom) -> cod
@dataclass(eq=True)
class Lam(Term):   body: Term                   # \x. body
@dataclass(eq=True)
class App(Term):   fn: Term; arg: Term
@dataclass(eq=True)
class Ann(Term):   tm: Term; ty: Term           # (tm : ty)
@dataclass(eq=True)
class Sig(Term):   m: str; dom: Term; cod: Term # (x :^m dom) (x) B   multiplicative pair type
@dataclass(eq=True)
class Pair(Term):  fst: Term; snd: Term
@dataclass(eq=True)
class LetPair(Term): scrut: Term; body: Term    # let (x,y) = scrut in body   (body under x then y)
@dataclass(eq=True)
class TUnit(Term): pass
@dataclass(eq=True)
class TT(Term):    pass

# ---------------------------------------------------------------------------
# 3. Semantic values for NbE (closures are Python functions; vars are LEVELS).
# ---------------------------------------------------------------------------
class Val: pass
class Neu: pass

@dataclass
class VNeutral(Val): n: "Neu"
@dataclass
class VUniv(Val): pass
@dataclass
class VPi(Val):  m: str; dom: Val; cod: Callable[[Val], Val]
@dataclass
class VLam(Val): body: Callable[[Val], Val]
@dataclass
class VSig(Val): m: str; dom: Val; cod: Callable[[Val], Val]
@dataclass
class VPair(Val): fst: Val; snd: Val
@dataclass
class VUnitT(Val): pass
@dataclass
class VTT(Val): pass

@dataclass
class NVar(Neu): lvl: int
@dataclass
class NApp(Neu): fn: "Neu"; arg: Val
@dataclass
class NLetPair(Neu): scrut: "Neu"; k: Callable[[Val, Val], Val]

# ---------------------------------------------------------------------------
# 4. Evaluation (Term -> Val) and quotation (Val -> Term), i.e. NbE.
#    env is innermost-first: env[i] is the value of de Bruijn index i.
# ---------------------------------------------------------------------------
def evaluate(env: List[Val], t: Term) -> Val:
    if isinstance(t, Var):     return env[t.i]
    if isinstance(t, Univ):    return VUniv()
    if isinstance(t, Pi):      return VPi(t.m, evaluate(env, t.dom),
                                          lambda v: evaluate([v] + env, t.cod))
    if isinstance(t, Lam):     return VLam(lambda v: evaluate([v] + env, t.body))
    if isinstance(t, App):     return v_app(evaluate(env, t.fn), evaluate(env, t.arg))
    if isinstance(t, Ann):     return evaluate(env, t.tm)
    if isinstance(t, Sig):     return VSig(t.m, evaluate(env, t.dom),
                                           lambda v: evaluate([v] + env, t.cod))
    if isinstance(t, Pair):    return VPair(evaluate(env, t.fst), evaluate(env, t.snd))
    if isinstance(t, LetPair): return v_letpair(evaluate(env, t.scrut),
                                                lambda vx, vy: evaluate([vy, vx] + env, t.body))
    if isinstance(t, TUnit):   return VUnitT()
    if isinstance(t, TT):      return VTT()
    raise Exception(f"eval: unknown term {t}")

def v_app(f: Val, a: Val) -> Val:
    if isinstance(f, VLam):     return f.body(a)
    if isinstance(f, VNeutral): return VNeutral(NApp(f.n, a))
    raise Exception("v_app: not a function")

def v_letpair(s: Val, k: Callable[[Val, Val], Val]) -> Val:
    if isinstance(s, VPair):    return k(s.fst, s.snd)
    if isinstance(s, VNeutral): return VNeutral(NLetPair(s.n, k))
    raise Exception("v_letpair: not a pair")

def quote(lvl: int, v: Val) -> Term:
    if isinstance(v, VUniv):  return Univ()
    if isinstance(v, VPi):    return Pi(v.m, quote(lvl, v.dom),
                                        quote(lvl + 1, v.cod(VNeutral(NVar(lvl)))))
    if isinstance(v, VLam):   return Lam(quote(lvl + 1, v.body(VNeutral(NVar(lvl)))))
    if isinstance(v, VSig):   return Sig(v.m, quote(lvl, v.dom),
                                         quote(lvl + 1, v.cod(VNeutral(NVar(lvl)))))
    if isinstance(v, VPair):  return Pair(quote(lvl, v.fst), quote(lvl, v.snd))
    if isinstance(v, VUnitT): return TUnit()
    if isinstance(v, VTT):    return TT()
    if isinstance(v, VNeutral): return quote_neu(lvl, v.n)
    raise Exception(f"quote: unknown value {v}")

def quote_neu(lvl: int, n: Neu) -> Term:
    if isinstance(n, NVar):     return Var(lvl - n.lvl - 1)     # level -> index
    if isinstance(n, NApp):     return App(quote_neu(lvl, n.fn), quote(lvl, n.arg))
    if isinstance(n, NLetPair):
        body = n.k(VNeutral(NVar(lvl)), VNeutral(NVar(lvl + 1)))
        return LetPair(quote_neu(lvl, n.scrut), quote(lvl + 2, body))
    raise Exception(f"quote_neu: unknown neutral {n}")

def conv(lvl: int, a: Val, b: Val) -> bool:
    """Definitional equality = alpha-equality of normal forms."""
    return quote(lvl, a) == quote(lvl, b)

# ---------------------------------------------------------------------------
# 5. Contexts and usage vectors.
# ---------------------------------------------------------------------------
class Ctx:
    """Parallel innermost-first lists, plus the current de Bruijn LEVEL."""
    def __init__(self, types: List[Val], mults: List[str], env: List[Val], level: int):
        self.types, self.mults, self.env, self.level = types, mults, env, level

    def extend(self, ty: Val, mult: str) -> "Ctx":
        var = VNeutral(NVar(self.level))            # the fresh variable's value
        return Ctx([ty] + self.types, [mult] + self.mults, [var] + self.env, self.level + 1)

EMPTY = Ctx([], [], [], 0)

# Usage vector: one multiplicity per in-scope variable, innermost-first.
def zeroU(n: int) -> List[str]:               return [ZERO] * n
def addU(u: List[str], v: List[str]) -> List[str]:   return [m_plus(a, b) for a, b in zip(u, v)]
def scaleU(m: str, u: List[str]) -> List[str]:       return [m_times(m, a) for a in u]

class TCError(Exception): pass

# ---------------------------------------------------------------------------
# 6. The bidirectional, resourced type checker.
#    sigma in {0, 1} is the modal index (docs/01 S3): 1 = runtime-relevant,
#    0 = erased/type-formation position. Recurse into TYPES at sigma = 0, which
#    is exactly how "type-level use costs 0" / erasure is realized.
# ---------------------------------------------------------------------------
def infer(ctx: Ctx, sigma: str, t: Term) -> Tuple[Val, List[str]]:
    if isinstance(t, Var):
        u = zeroU(ctx.level); u[t.i] = sigma          # using x spends `sigma` of x
        return ctx.types[t.i], u

    if isinstance(t, Univ):
        return VUniv(), zeroU(ctx.level)               # Type : Type

    if isinstance(t, (Pi, Sig)):                       # type FORMATION, done at sigma=0
        check(ctx, ZERO, t.dom, VUniv())
        domv = evaluate(ctx.env, t.dom)
        check(ctx.extend(domv, ZERO), ZERO, t.cod, VUniv())
        return VUniv(), zeroU(ctx.level)

    if isinstance(t, App):
        fty, uF = infer(ctx, sigma, t.fn)
        if not isinstance(fty, VPi):
            raise TCError(f"application of a non-function: {show(quote(ctx.level, fty))}")
        arg_sigma = ZERO if fty.m == ZERO else sigma   # erased args checked at 0
        uA = check(ctx, arg_sigma, t.arg, fty.dom)
        argv = evaluate(ctx.env, t.arg)
        return fty.cod(argv), addU(uF, scaleU(fty.m, uA))   # uF + m * uA

    if isinstance(t, Ann):
        check(ctx, ZERO, t.ty, VUniv())
        tyv = evaluate(ctx.env, t.ty)
        uTm = check(ctx, sigma, t.tm, tyv)
        return tyv, uTm

    if isinstance(t, TUnit): return VUniv(), zeroU(ctx.level)
    if isinstance(t, TT):    return VUnitT(), zeroU(ctx.level)

    raise TCError(f"cannot infer a type for {show(t)} (try an annotation)")

def check(ctx: Ctx, sigma: str, t: Term, want: Val) -> List[str]:
    # Lam checks against a Pi: this is where a binder's budget is verified.
    if isinstance(t, Lam):
        if not isinstance(want, VPi):
            raise TCError(f"lambda checked against non-function type {show(quote(ctx.level, want))}")
        ctx2 = ctx.extend(want.dom, want.m)
        body_want = want.cod(VNeutral(NVar(ctx.level)))
        u_body = check(ctx2, sigma, t.body, body_want)
        used_here = u_body[0]
        required = m_times(sigma, want.m)
        if not m_leq(used_here, required):
            raise TCError(_budget_msg("function argument", used_here, want.m, sigma))
        return u_body[1:]

    if isinstance(t, Pair):
        if not isinstance(want, VSig):
            raise TCError(f"pair checked against non-pair type {show(quote(ctx.level, want))}")
        fst_sigma = ZERO if want.m == ZERO else sigma
        uF = check(ctx, fst_sigma, t.fst, want.dom)
        fstv = evaluate(ctx.env, t.fst)
        uS = check(ctx, sigma, t.snd, want.cod(fstv))
        return addU(scaleU(want.m, uF), uS)            # m * uF + uS

    if isinstance(t, LetPair):
        sty, uScrut = infer(ctx, sigma, t.scrut)
        if not isinstance(sty, VSig):
            raise TCError("let-pair on a non-pair")
        cx = ctx.extend(sty.dom, sty.m)                # x (outer binder)
        xv = VNeutral(NVar(ctx.level))
        cxy = cx.extend(sty.cod(xv), ONE)              # y (inner binder), linear
        # NOTE: non-dependent elimination -- `want` must not mention (x,y).
        # NbE uses absolute levels, so `want` needs no weakening here.
        u_body = check(cxy, sigma, t.body, want)
        used_y, used_x = u_body[0], u_body[1]
        if not m_leq(used_x, m_times(sigma, sty.m)):
            raise TCError(_budget_msg("first pair component", used_x, sty.m, sigma))
        if not m_leq(used_y, m_times(sigma, ONE)):
            raise TCError(_budget_msg("second pair component", used_y, ONE, sigma))
        return addU(uScrut, u_body[2:])

    # Fallback: infer, then demand the inferred type converts to `want`.
    got, u = infer(ctx, sigma, t)
    if not conv(ctx.level, got, want):
        raise TCError(f"type mismatch:\n   expected {show(quote(ctx.level, want))}"
                      f"\n   inferred {show(quote(ctx.level, got))}")
    return u

def _budget_msg(what: str, used: str, declared: str, sigma: str) -> str:
    if sigma == ONE and declared == ONE and used == ZERO:
        return f"linearity: {what} declared 1 (linear) but used 0 times (must be exactly once)"
    if sigma == ONE and declared == ONE and used == MANY:
        return f"linearity: {what} declared 1 (linear) but used more than once"
    if declared == ZERO:
        return f"erasure: {what} declared 0 (erased) but used at runtime"
    return f"budget: {what} declared {declared} but used {used}"

# ---------------------------------------------------------------------------
# 7. A tiny named front-end so tests are readable (names -> de Bruijn).
# ---------------------------------------------------------------------------
@dataclass
class N: pass
@dataclass
class NVarN(N):  x: str
@dataclass
class NU(N):     pass
@dataclass
class NPi(N):    m: str; x: str; dom: N; cod: N
@dataclass
class NLam(N):   x: str; body: N
@dataclass
class NApp(N):   fn: N; arg: N
@dataclass
class NAnn(N):   tm: N; ty: N
@dataclass
class NSig(N):   m: str; x: str; dom: N; cod: N
@dataclass
class NPair(N):  fst: N; snd: N
@dataclass
class NLet(N):   x: str; y: str; scrut: N; body: N
@dataclass
class NUnit(N):  pass
@dataclass
class NTT(N):    pass

def to_db(t: N, scope: List[str]) -> Term:
    """scope is a list of names; the LAST element is the innermost binder."""
    if isinstance(t, NVarN):
        for k in range(len(scope) - 1, -1, -1):        # rightmost (innermost) match
            if scope[k] == t.x:
                return Var(len(scope) - 1 - k)
        raise TCError(f"unbound variable {t.x}")
    if isinstance(t, NU):    return Univ()
    if isinstance(t, NPi):   return Pi(t.m, to_db(t.dom, scope), to_db(t.cod, scope + [t.x]))
    if isinstance(t, NLam):  return Lam(to_db(t.body, scope + [t.x]))
    if isinstance(t, NApp):  return App(to_db(t.fn, scope), to_db(t.arg, scope))
    if isinstance(t, NAnn):  return Ann(to_db(t.tm, scope), to_db(t.ty, scope))
    if isinstance(t, NSig):  return Sig(t.m, to_db(t.dom, scope), to_db(t.cod, scope + [t.x]))
    if isinstance(t, NPair): return Pair(to_db(t.fst, scope), to_db(t.snd, scope))
    if isinstance(t, NLet):  return LetPair(to_db(t.scrut, scope), to_db(t.body, scope + [t.x, t.y]))
    if isinstance(t, NUnit): return TUnit()
    if isinstance(t, NTT):   return TT()
    raise Exception(f"to_db: {t}")

# sugar
def var(x):                return NVarN(x)
U = NU()
def pi(m, x, A, B):        return NPi(m, x, A, B)
def lam(x, b):             return NLam(x, b)
def app(f, *args):
    e = f
    for a in args: e = NApp(e, a)
    return e
def ann(t, ty):            return NAnn(t, ty)
def sig(m, x, A, B):       return NSig(m, x, A, B)
def pair(a, b):            return NPair(a, b)
def letp(x, y, s, b):      return NLet(x, y, s, b)
Unit = NUnit()
tt = NTT()
def arrow(m, A, B):        return NPi(m, "_", A, B)   # non-dependent (x:^m A)->B

# ---------------------------------------------------------------------------
# 8. Pretty-printer for de Bruijn terms (generates names by depth).
# ---------------------------------------------------------------------------
def show(t: Term, d: int = 0) -> str:
    nm = lambda k: f"x{d - k - 1}"
    if isinstance(t, Var):   return nm(t.i) if 0 <= t.i < d else f"#{t.i}"
    if isinstance(t, Univ):  return "Type"
    if isinstance(t, Pi):    return f"(x{d} :^{t.m} {show(t.dom, d)}) -> {show(t.cod, d + 1)}"
    if isinstance(t, Lam):   return f"\\x{d}. {show(t.body, d + 1)}"
    if isinstance(t, App):   return f"({show(t.fn, d)} {show(t.arg, d)})"
    if isinstance(t, Ann):   return f"({show(t.tm, d)} : {show(t.ty, d)})"
    if isinstance(t, Sig):   return f"(x{d} :^{t.m} {show(t.dom, d)}) * {show(t.cod, d + 1)}"
    if isinstance(t, Pair):  return f"({show(t.fst, d)}, {show(t.snd, d)})"
    if isinstance(t, LetPair): return f"let (x{d},x{d+1}) = {show(t.scrut, d)} in {show(t.body, d + 2)}"
    if isinstance(t, TUnit): return "Unit"
    if isinstance(t, TT):    return "tt"
    return str(t)

# ---------------------------------------------------------------------------
# 9. Test harness.
# ---------------------------------------------------------------------------
def typecheck(term: N, typ: N):
    tm = to_db(term, [])
    ty = to_db(typ, [])
    check(EMPTY, ZERO, ty, VUniv())          # the type must be a type
    tyv = evaluate([], ty)
    check(EMPTY, ONE, tm, tyv)               # the term must inhabit it (at runtime, sigma=1)

PASS, FAIL = "should typecheck", "should be REJECTED"
RESULTS = []

def expect(name: str, outcome: str, term: N, typ: N):
    try:
        typecheck(term, typ)
        ok = (outcome == PASS)
        RESULTS.append((ok, name, outcome, "accepted" if ok else "accepted (UNEXPECTED)"))
    except TCError as e:
        ok = (outcome == FAIL)
        msg = str(e).split("\n")[0]
        RESULTS.append((ok, name, outcome, f"rejected: {msg}"))

# ----- the type abbreviations used below --------------------------------------
# linId  : (A :^0 Type) -> (x :^1 A) -> A     -- A erased, x linear
# manyId : (A :^0 Type) -> (x :^w A) -> A
def tyId(mx):  return pi("0", "A", U, arrow(mx, var("A"), var("A")))

# ----- the demos --------------------------------------------------------------
# (1) unrestricted identity: x used once, allowed w  -> OK
expect("identity (arg w)", PASS,
       lam("A", lam("x", var("x"))), tyId("w"))

# (2) THE cohabitation demo: linear identity. `A` appears in TYPES (type of x,
#     and the return type) but is declared 0 and contributes 0 usage; `x` is
#     linear and used exactly once. Linear + dependent coexist.
expect("linear identity  (A used in types, x used once)", PASS,
       lam("A", lam("x", var("x"))), tyId("1"))

# (2b) contrast: declare `A :^1`. It is only ever used in TYPES (= 0 usage), so
#      the linear obligation on A is unmet -> rejected. Type-uses don't count.
expect("A declared linear but only used in types", FAIL,
       lam("A", lam("x", var("x"))),
       pi("1", "A", U, arrow("1", var("A"), var("A"))))

# (3) duplicate a linear value: x used twice (in a pair) -> rejected.
expect("duplicate a linear value (x,x)", FAIL,
       lam("A", lam("x", pair(var("x"), var("x")))),
       pi("0", "A", U, arrow("1", var("A"),
          sig("1", "_", var("A"), var("A")))))

# (4) drop a linear value: y unused, declared 1 -> rejected.
expect("drop a linear value", FAIL,
       lam("A", lam("x", lam("y", var("x")))),
       pi("0", "A", U, arrow("1", var("A"), arrow("1", var("A"), var("A")))))

# (5) drop an unrestricted value: y unused, declared w -> OK.
expect("drop an unrestricted value", PASS,
       lam("A", lam("x", lam("y", var("x")))),
       pi("0", "A", U, arrow("w", var("A"), arrow("w", var("A"), var("A")))))

# (6) erasure: use an erased (0) argument at runtime -> rejected.
expect("use an erased argument at runtime", FAIL,
       lam("A", lam("x", var("x"))),
       pi("0", "A", U, arrow("0", var("A"), var("A"))))

# (7) swap a linear pair: both components used exactly once -> OK. This is the
#     shape of "pointer (x) view" from docs/02 (a linear tensor).
swapTy = pi("0", "A", U, pi("0", "B", U,
          arrow("1", sig("1", "_", var("A"), var("B")),
                     sig("1", "_", var("B"), var("A")))))
expect("swap a linear pair", PASS,
       lam("A", lam("B", lam("p",
           letp("x", "y", var("p"), pair(var("y"), var("x")))))),
       swapTy)

# (8) swap but drop a component: y unused -> rejected.
expect("linear pair, drop a component", FAIL,
       lam("A", lam("B", lam("p",
           letp("x", "y", var("p"), pair(var("x"), var("x")))))),
       pi("0", "A", U, pi("0", "B", U,
          arrow("1", sig("1", "_", var("A"), var("B")),
                     sig("1", "_", var("A"), var("A"))))))

# ---------------------------------------------------------------------------
def main():
    # quick NbE sanity: ((\A.\x. x) Unit tt) normalizes to tt
    nf = quote(0, evaluate([], to_db(app(lam("A", lam("x", var("x"))), Unit, tt), [])))
    print("NbE check:  (\\A.\\x.x) Unit tt   ~~>   " + show(nf))
    print()
    width = max(len(n) for _, n, _, _ in RESULTS)
    allok = True
    for ok, name, outcome, detail in RESULTS:
        allok &= ok
        mark = "ok  " if ok else "FAIL"
        print(f"[{mark}] {name.ljust(width)}  | {outcome:18} | {detail}")
    print()
    print("ALL EXPECTATIONS MET" if allok else "SOME EXPECTATIONS VIOLATED")
    raise SystemExit(0 if allok else 1)

if __name__ == "__main__":
    main()
