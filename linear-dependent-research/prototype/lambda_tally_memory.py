#!/usr/bin/env python3
"""
lambda-Tally, Stage C: UNIFYING the dependent kernel (Stage A) with the memory
discipline (Stage B), in a single quantitative type checker.

This is the artifact for the open seam in docs/04 (Phases 2 & 4): Stage A
(qtt_checker.py) had the dependent multiplicity kernel but no heap; Stage B
(memory-model.rkt) had the heap and linear capabilities but no dependency. Here
the memory layer of docs/02 is expressed *inside* the QTT kernel of docs/01,
with NO new typing rules in the checker -- we only POSTULATE:

    Loc   : Type                          -- locations: erased (used at mult 0)
    Ptr   : Loc -> Type                   -- pointers: unrestricted (mult w)
    View  : Type -> Loc -> Type           -- A @ l : the LINEAR capability (mult 1)

    alloc : (A :^0 Type) -> (i :^1 A) -> Sigma(l:^0 Loc). Sigma(p:^w Ptr l). (View A l)
    read  : (l:^0 Loc)(A:^0 Type)(p:^w Ptr l)(v:^1 View A l) -> Sigma(_:^w A). View A l
    write : (l:^0 Loc)(A:^0 Type)(B:^0 Type)(p:^w Ptr l)(v:^1 View A l)(n:^1 B)
                                                          -> Sigma(_:^w Unit). View B l
    free  : (l:^0 Loc)(A:^0 Type)(p:^w Ptr l)(v:^1 View A l) -> Unit

The point: the pointer/view split of L3/ATS is *just two multiplicities in a
dependent tensor*. `p` is bound at w (copy it freely); the view `v` is bound at
1 (linear). The EXISTING multiplicity accounting of Stage A then enforces, with
no extra machinery:
  * use-after-free  -> a view used twice  -> linearity error;
  * double-free     -> a view used twice  -> linearity error;
  * leak            -> a view never used  -> "unused linear" error;
  * STRONG UPDATE   -> `write` returns View B l from View A l: the type stored
    at l CHANGES (Unit -> Byte), and it is sound precisely because the view is
    linear (no stale View A l can survive).

This is the static half of the safety story; the operational half (that the
discipline actually keeps the heap consistent at runtime) is memory-model.rkt.

Run:  python3 lambda_tally_memory.py
"""

from qtt_checker import (
    # named surface syntax
    var, U, pi, lam, app, sig, pair, letp, Unit, tt, arrow, to_db,
    # core checker
    Ctx, EMPTY, evaluate, check, infer, quote, show,
    VUniv, VNeutral, NVar, ZERO, ONE, MANY, TCError,
)

# ---------------------------------------------------------------------------
# 1. Build a prelude context of postulated constants.
#    Each declaration's type is checked & evaluated in the context built so far,
#    so later postulates may refer to earlier ones (Ptr/View refer to Loc, etc).
# ---------------------------------------------------------------------------
def build_prelude(decls):
    ctx, scope = EMPTY, []
    for name, named_ty, mult in decls:
        ty = to_db(named_ty, scope)
        check(ctx, ZERO, ty, VUniv())            # the postulated type must be a type
        ctx = ctx.extend(evaluate(ctx.env, ty), mult)
        scope.append(name)
    return ctx, scope

# abbreviations for the postulate types
def Ptr_(l):       return app(var("Ptr"), l)                 # Ptr l
def View_(A, l):   return app(var("View"), A, l)             # View A l  (= A @ l)

alloc_ty = pi("0", "A", U, pi("1", "i", var("A"),
             sig("0", "l", var("Loc"),
               sig("w", "p", Ptr_(var("l")),
                 View_(var("A"), var("l"))))))

read_ty  = pi("0", "l", var("Loc"), pi("0", "A", U,
             pi("w", "p", Ptr_(var("l")),
               pi("1", "v", View_(var("A"), var("l")),
                 sig("w", "_", var("A"), View_(var("A"), var("l")))))))

write_ty = pi("0", "l", var("Loc"), pi("0", "A", U, pi("0", "B", U,
             pi("w", "p", Ptr_(var("l")),
               pi("1", "v", View_(var("A"), var("l")),
                 pi("1", "n", var("B"),
                   sig("w", "_", Unit, View_(var("B"), var("l")))))))))

free_ty  = pi("0", "l", var("Loc"), pi("0", "A", U,
             pi("w", "p", Ptr_(var("l")),
               pi("1", "v", View_(var("A"), var("l")), Unit))))

PRELUDE_DECLS = [
    ("Loc",   U,                                          MANY),  # Loc : Type
    ("Ptr",   arrow("0", var("Loc"), U),                  MANY),  # Ptr : Loc -> Type
    ("View",  pi("0", "A", U, arrow("0", var("Loc"), U)), MANY),  # View : Type -> Loc -> Type
    ("Byte",  U,                                          MANY),  # a second base type
    ("b0",    var("Byte"),                                MANY),  # a Byte value
    ("seq",   arrow("w", Unit, arrow("w", Unit, Unit)),   MANY),  # seq : Unit->Unit->Unit
    ("alloc", alloc_ty,                                   MANY),
    ("read",  read_ty,                                    MANY),
    ("write", write_ty,                                   MANY),
    ("free",  free_ty,                                    MANY),
]

PRELUDE, SCOPE = build_prelude(PRELUDE_DECLS)

# ---------------------------------------------------------------------------
# 2. Check a program against a type in the prelude context.
# ---------------------------------------------------------------------------
def typecheck_in_prelude(term, typ):
    tm, ty = to_db(term, SCOPE), to_db(typ, SCOPE)
    check(PRELUDE, ZERO, ty, VUniv())
    check(PRELUDE, ONE, tm, evaluate(PRELUDE.env, ty))

# ---------------------------------------------------------------------------
# 3. Programs (all have result type Unit).
# ---------------------------------------------------------------------------
# (good) allocate, then free.  v is the linear view; used exactly once (in free).
p_alloc_free = letp("l", "r", app(var("alloc"), Unit, tt),
                 letp("p", "v", var("r"),
                   app(var("free"), var("l"), Unit, var("p"), var("v"))))

# (good) allocate, read (consumes v, yields v2), then free v2.
p_read = letp("l", "r", app(var("alloc"), Unit, tt),
           letp("p", "v", var("r"),
             letp("u", "v2", app(var("read"), var("l"), Unit, var("p"), var("v")),
               app(var("free"), var("l"), Unit, var("p"), var("v2")))))

# (good) STRONG UPDATE: store Unit, overwrite with a Byte (view type Unit -> Byte),
#        then free at the NEW type.  Sound because the view is linear.
p_strong_update = letp("l", "r", app(var("alloc"), Unit, tt),
                    letp("p", "v", var("r"),
                      letp("u", "v2",
                           app(var("write"), var("l"), Unit, var("Byte"),
                               var("p"), var("v"), var("b0")),
                        app(var("free"), var("l"), var("Byte"), var("p"), var("v2")))))

# (BAD) leak: allocate but never free -> the linear view v is dropped.
p_leak = letp("l", "r", app(var("alloc"), Unit, tt),
           letp("p", "v", var("r"), tt))

# (BAD) double-free: free twice -> v used twice.
p_double_free = letp("l", "r", app(var("alloc"), Unit, tt),
                  letp("p", "v", var("r"),
                    app(var("seq"),
                        app(var("free"), var("l"), Unit, var("p"), var("v")),
                        app(var("free"), var("l"), Unit, var("p"), var("v")))))

# (BAD) use-after-free: read consumes v (giving v2); we correctly free v2 but
#        ALSO reuse the already-consumed v -> v is used twice.
p_use_after_free = letp("l", "r", app(var("alloc"), Unit, tt),
                     letp("p", "v", var("r"),
                       letp("u", "v2", app(var("read"), var("l"), Unit, var("p"), var("v")),
                         app(var("seq"),
                             app(var("free"), var("l"), Unit, var("p"), var("v2")),
                             app(var("free"), var("l"), Unit, var("p"), var("v"))))))

# ---------------------------------------------------------------------------
# 4. Test harness.
# ---------------------------------------------------------------------------
PASS, FAIL = "should typecheck", "should be REJECTED"
RESULTS = []

def expect(name, outcome, term, typ=Unit):
    try:
        typecheck_in_prelude(term, typ)
        RESULTS.append((outcome == PASS, name, outcome, "accepted"))
    except TCError as e:
        RESULTS.append((outcome == FAIL, name, outcome, "rejected: " + str(e).split("\n")[0]))

expect("alloc then free",                 PASS, p_alloc_free)
expect("alloc, read, free",               PASS, p_read)
expect("STRONG UPDATE (Unit -> Byte)",    PASS, p_strong_update)
expect("leak: never free the view",       FAIL, p_leak)
expect("double free",                     FAIL, p_double_free)
expect("use-after-free (reuse view)",     FAIL, p_use_after_free)

def main():
    # Show that the strong-update program really does change the stored type:
    # normalize the type of the view handed to `free` at the end.
    print("Memory primitives expressed purely as QTT functions (no new kernel rules).")
    print("The pointer is bound at multiplicity w; the view at multiplicity 1.\n")
    width = max(len(n) for _, n, _, _ in RESULTS)
    allok = True
    for ok, name, outcome, detail in RESULTS:
        allok &= ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {name.ljust(width)}  | {outcome:18} | {detail}")
    print("\n" + ("ALL EXPECTATIONS MET" if allok else "SOME EXPECTATIONS VIOLATED"))
    raise SystemExit(0 if allok else 1)

if __name__ == "__main__":
    main()
