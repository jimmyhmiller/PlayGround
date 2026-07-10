//! A nanopass optimizer over `Ir`.
//!
//! Each pass is a small, single-purpose `Ir -> Ir` rewrite. Because `Ir` is the
//! neutral, backend-independent form (§`code.rs`), an optimization here benefits
//! *every* `CodeSpace` — the tree-walker, the bytecode VM, and the JIT alike —
//! and composes with the backend wrappers (`Traced`, `Tiered`) that sit below
//! it. This is the "optimization axis": intelligence in a pass pipeline over the
//! IR, the way a real nanopass compiler (Chez's `cp0`/`cptypes`) puts its
//! cleverness *upstream* of codegen rather than inside a smart backend.
//!
//! The passes:
//!   * [`simplify`] — constant-`if` folding and singleton-`do` collapse.
//!   * [`specialize_fixnums`] — the borrow from Chez's `cptypes`: prove a
//!     lambda's parameters are immediate fixnums with ONE guard at entry, then
//!     let every arithmetic use of them skip its own per-op tag check.
//!
//! Delivered as the [`Optimized`] `CodeSpace` wrapper so it is opt-in and
//! composable. The transform is IDEMPOTENT (re-optimizing an already-optimized
//! tree is a no-op), which is what keeps it correct under a backend that
//! re-enters `eval_ir` per node (the interpreter tiers); a compiling backend
//! (the JIT) never re-enters, so it pays the transform exactly once per form.

use std::rc::Rc;

use crate::code::CodeSpace;
use crate::ir::{Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::runtime::Runtime;
use crate::value::{Cat, Locals, Val};

/// The immediate-fixnum range: 61-bit two's complement (3 tag/box bits), the
/// same window `LowBit`/`HighBit` use and the JIT's fast path range-checks to.
const FIX_MIN: i128 = -(1 << 60);
const FIX_MAX: i128 = (1 << 60) - 1;

/// Is the constant-pool value `id` an *immediate* fixnum under model `M`? False
/// under a model with no immediate integers (NaN-boxing boxes them), so the
/// specializer naturally produces no fixnum ops there.
fn const_is_fixnum<M: ValueModel>(rt: &Runtime<M>, id: u32) -> bool {
    M::R::is_immediate(Cat::Int)
        && matches!(rt.decode(rt.get_const(id)), Val::Int(v) if (FIX_MIN..=FIX_MAX).contains(&v))
}

/// Immediate children of a node, in evaluation order — the shared spine every
/// pass recurses over.
fn children(ir: &Ir) -> Vec<&Ir> {
    match ir {
        Ir::Const(_) | Ir::Quote(_) | Ir::Local { .. } | Ir::Global(_) => vec![],
        Ir::SetLocal { val, .. } | Ir::SetGlobal { val, .. } => vec![val],
        Ir::If(c, t, e) => vec![c, t, e],
        Ir::Do(xs) => xs.iter().collect(),
        Ir::Def { init, .. } => vec![init],
        Ir::Let(inits, body) => {
            let mut v: Vec<&Ir> = inits.iter().collect();
            v.push(body);
            v
        }
        Ir::Lambda { body, .. } => vec![body],
        Ir::Call(f, args) => {
            let mut v = vec![f.as_ref()];
            v.extend(args.iter());
            v
        }
        Ir::Prim(_, args) => args.iter().collect(),
        Ir::DefMethod { imp, .. } => vec![imp],
        Ir::Dispatch { args, .. } => args.iter().collect(),
    }
}

/// Rebuild `ir` with `f` applied to each immediate child. The structural spine
/// a post-order pass rides.
fn map_children(ir: &Ir, f: &mut impl FnMut(&Ir) -> Ir) -> Ir {
    match ir {
        Ir::Const(_) | Ir::Quote(_) | Ir::Local { .. } | Ir::Global(_) => ir.clone(),
        Ir::SetLocal { up, idx, val } => Ir::SetLocal { up: *up, idx: *idx, val: Box::new(f(val)) },
        Ir::SetGlobal { name, val } => Ir::SetGlobal { name: *name, val: Box::new(f(val)) },
        Ir::If(c, t, e) => Ir::If(Box::new(f(c)), Box::new(f(t)), Box::new(f(e))),
        Ir::Do(xs) => Ir::Do(xs.iter().map(|x| f(x)).collect()),
        Ir::Def { name, init } => Ir::Def { name: *name, init: Box::new(f(init)) },
        Ir::Let(inits, body) => Ir::Let(inits.iter().map(|x| f(x)).collect(), Box::new(f(body))),
        Ir::Lambda { nparams, variadic, body } => {
            Ir::Lambda { nparams: *nparams, variadic: *variadic, body: Rc::new(f(body)) }
        }
        Ir::Call(g, args) => Ir::Call(Box::new(f(g)), args.iter().map(|x| f(x)).collect()),
        Ir::Prim(p, args) => Ir::Prim(*p, args.iter().map(|x| f(x)).collect()),
        Ir::DefMethod { name, ty, imp } => Ir::DefMethod { name: *name, ty: *ty, imp: Box::new(f(imp)) },
        Ir::Dispatch { site, method, args } => {
            Ir::Dispatch { site: *site, method: *method, args: args.iter().map(|x| f(x)).collect() }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Pass -1 — inline: beta-reduce a directly-applied lambda into a `let`.
//   `((fn (p..) body) a..)`  ==>  `(let (a'..) body)`
// The lambda's frame and the `let`'s frame parent the SAME call-site scope and
// bind the SAME slots, so `body` is unchanged; the args, now evaluated one frame
// deeper (inside the growing `let`), have their free de Bruijn indices lifted by
// one. This is Chez's `cp0` procedure integration in its safest form: an
// anonymous, exactly-applied lambda, so no recursion and no name capture.
// ─────────────────────────────────────────────────────────────────────────

/// Lift by `by` the free lexical indices of `ir` — every `Local`/`SetLocal`
/// whose `up >= cutoff` (i.e. that escapes the binders inside `ir`). `cutoff`
/// rises through `let`/`lambda`, so a variable bound *within* `ir` is untouched
/// (capture-avoiding).
fn shift(ir: &Ir, by: u16, cutoff: u16) -> Ir {
    match ir {
        Ir::Local { up, idx } => {
            Ir::Local { up: if *up >= cutoff { *up + by } else { *up }, idx: *idx }
        }
        Ir::SetLocal { up, idx, val } => Ir::SetLocal {
            up: if *up >= cutoff { *up + by } else { *up },
            idx: *idx,
            val: Box::new(shift(val, by, cutoff)),
        },
        Ir::Let(inits, body) => Ir::Let(
            inits.iter().map(|i| shift(i, by, cutoff + 1)).collect(),
            Box::new(shift(body, by, cutoff + 1)),
        ),
        Ir::Lambda { nparams, variadic, body } => Ir::Lambda {
            nparams: *nparams,
            variadic: *variadic,
            body: Rc::new(shift(body, by, cutoff + 1)),
        },
        other => map_children(other, &mut |c| shift(c, by, cutoff)),
    }
}

/// An atom is a variable or literal: side-effect-free AND free to duplicate, so
/// it can be substituted for a parameter at every use with no cost. But a
/// *variable* atom is only sound to substitute if the body never MUTATES it —
/// binding captures the value at call time, whereas substitution re-reads it at
/// each use, and a `set!` in between would diverge. So `Const`/`Quote`
/// (immutable) are always substitutable; `Local`/`Global` only when the body has
/// no assignment (see [`body_has_assignment`]).
fn is_literal(ir: &Ir) -> bool {
    matches!(ir, Ir::Const(_) | Ir::Quote(_))
}
fn is_var(ir: &Ir) -> bool {
    matches!(ir, Ir::Local { .. } | Ir::Global(_))
}

/// Does `ir` perform ANY assignment (`set!` on a local or global), anywhere —
/// including inside a nested lambda that could escape and run later? A `true`
/// here means variable substitution across it is unsafe.
fn body_has_assignment(ir: &Ir) -> bool {
    match ir {
        Ir::SetLocal { .. } | Ir::SetGlobal { .. } => true,
        other => children(other).iter().any(|c| body_has_assignment(c)),
    }
}

/// Does `ir` assign one of the enclosing lambda's `nparams` params (a `set!` at
/// the param frame, `up == depth`)? If so it needs a real slot — substitution
/// would have nowhere to put the assignment.
fn assigns_param(ir: &Ir, depth: u16, nparams: usize) -> bool {
    match ir {
        Ir::SetLocal { up, idx, val } => {
            (*up == depth && (*idx as usize) < nparams) || assigns_param(val, depth, nparams)
        }
        Ir::Let(inits, body) => {
            inits.iter().any(|i| assigns_param(i, depth + 1, nparams))
                || assigns_param(body, depth + 1, nparams)
        }
        Ir::Lambda { body, .. } => assigns_param(body, depth + 1, nparams),
        other => children(other).iter().any(|c| assigns_param(c, depth, nparams)),
    }
}

/// Substitute atom `args` for the param references in `body`, ELIMINATING the
/// param frame. At frame `depth` (rising through binders): `up == depth` is a
/// param → its arg, lifted `depth` frames to the use site; `up > depth` is an
/// outer scope that loses the now-gone param frame (→ `up - 1`); `up < depth` is
/// an inner binder, untouched.
fn subst_params(body: &Ir, depth: u16, args: &[Ir]) -> Ir {
    match body {
        Ir::Local { up, idx } => {
            if *up == depth {
                shift(&args[*idx as usize], depth, 0)
            } else if *up > depth {
                Ir::Local { up: up - 1, idx: *idx }
            } else {
                body.clone()
            }
        }
        Ir::SetLocal { up, idx, val } => Ir::SetLocal {
            // `up == depth` (assigning a param) is excluded before we get here.
            up: if *up > depth { up - 1 } else { *up },
            idx: *idx,
            val: Box::new(subst_params(val, depth, args)),
        },
        Ir::Let(inits, b) => Ir::Let(
            inits.iter().map(|i| subst_params(i, depth + 1, args)).collect(),
            Box::new(subst_params(b, depth + 1, args)),
        ),
        Ir::Lambda { nparams, variadic, body: b } => Ir::Lambda {
            nparams: *nparams,
            variadic: *variadic,
            body: Rc::new(subst_params(b, depth + 1, args)),
        },
        other => map_children(other, &mut |c| subst_params(c, depth, args)),
    }
}

/// Beta-reduce every directly-applied, exactly-arity, non-variadic lambda.
/// Bottom-up and terminating (the callee becomes a `let` or is substituted
/// away — never a fresh redex). Two strategies:
///   * ALL args atoms and no param is assigned → full SUBSTITUTION: the param
///     frame vanishes entirely (best — a `let` frame would otherwise defeat the
///     JIT's register self-tail loop). Atoms are pure and free to duplicate.
///   * otherwise → `let`-based beta: a complex arg keeps a slot so it is
///     evaluated exactly once.
pub fn inline(ir: &Ir) -> Ir {
    let n = map_children(ir, &mut |c| inline(c));
    if let Ir::Call(f, args) = &n {
        if let Ir::Lambda { nparams, variadic: false, body } = f.as_ref() {
            if *nparams == args.len() {
                // A variable arg is only substitutable if the body never mutates
                // (see `is_var`); literals always are. And no param may be `set!`
                // (substitution leaves its assignment homeless).
                let mutates = body_has_assignment(body);
                let substitutable = args
                    .iter()
                    .all(|a| is_literal(a) || (is_var(a) && !mutates));
                if substitutable && !assigns_param(body, 0, *nparams) {
                    return subst_params(body, 0, args);
                }
                let inits: Vec<Ir> = args.iter().map(|a| shift(a, 1, 0)).collect();
                return Ir::Let(inits, Box::new((**body).clone()));
            }
        }
    }
    n
}

// ─────────────────────────────────────────────────────────────────────────
// Pass 0 — fold_const: evaluate pure prims over literal operands at compile
// time. We fold by running the very same `rt.prim` the interpreter would, so
// the result is correct by construction (including bignum promotion). Only
// side-effect-free, value-deterministic ops qualify.
// ─────────────────────────────────────────────────────────────────────────

fn is_foldable_pure(op: Prim) -> bool {
    matches!(op, Prim::Add | Prim::Sub | Prim::Mul | Prim::Lt | Prim::Eq | Prim::Identical)
}

/// Fold `(op c0 c1 ...)` to a constant when `op` is pure and every operand is
/// already a constant. Bottom-up, so `(+ (+ 1 2) 3)` folds fully. Idempotent (a
/// folded tree has no foldable node left).
pub fn fold_const<M: ValueModel>(rt: &mut Runtime<M>, ir: &Ir) -> Ir {
    let n = map_children(ir, &mut |c| fold_const::<M>(rt, c));
    if let Ir::Prim(op, args) = &n {
        if is_foldable_pure(*op) && !args.is_empty() && args.iter().all(|a| matches!(a, Ir::Const(_))) {
            let argv: Vec<u64> = args
                .iter()
                .map(|a| match a {
                    Ir::Const(id) => rt.get_const(*id),
                    _ => unreachable!("guarded by all(Const)"),
                })
                .collect();
            let folded = rt.prim(*op, &argv);
            return Ir::Const(rt.intern_const(folded));
        }
    }
    n
}

// ─────────────────────────────────────────────────────────────────────────
// Pass 0.5 — eliminate_dead_lets: drop pure `let` bindings that are never
// referenced, renumbering the survivors' slots.
// ─────────────────────────────────────────────────────────────────────────

/// Is `ir` free of observable side effects (so an unused binding of it can be
/// dropped)? Conservative: calls, dispatch, assignment, `def`, and the effectful
/// prims are impure; value-producing prims (including allocating ones like
/// `cons`/`vector` — dropping unused garbage is fine) are pure.
fn is_pure(ir: &Ir) -> bool {
    match ir {
        Ir::Const(_) | Ir::Quote(_) | Ir::Local { .. } | Ir::Global(_) | Ir::Lambda { .. } => true,
        Ir::If(c, t, e) => is_pure(c) && is_pure(t) && is_pure(e),
        Ir::Do(xs) => xs.iter().all(is_pure),
        Ir::Let(inits, body) => inits.iter().all(is_pure) && is_pure(body),
        Ir::Prim(op, args) => {
            let pure_op = matches!(
                op,
                Prim::Add | Prim::Sub | Prim::Mul | Prim::Lt | Prim::Eq | Prim::Identical
                    | Prim::List | Prim::Cons | Prim::First | Prim::Rest | Prim::IsNil
                    | Prim::StrLen | Prim::CharToInt | Prim::IntToChar
                    | Prim::Vector | Prim::VectorRef | Prim::VectorLen
                    | Prim::Field | Prim::Record | Prim::Values | Prim::ValuesToList
            );
            pure_op && args.iter().all(is_pure)
        }
        // Call/Dispatch/Set*/Def*/effectful prims (Gc/Println/VectorSet/Apply/
        // continuations/CallEc): conservatively impure.
        _ => false,
    }
}

/// Does a reference to slot `idx` of the frame `depth` frames out appear in `ir`?
/// `depth` rises through the binders (`let`/`lambda`) so a same-named inner slot
/// is not confused with ours.
fn uses_slot(ir: &Ir, depth: u16, idx: u16) -> bool {
    match ir {
        Ir::Local { up, idx: i } => *up == depth && *i == idx,
        Ir::SetLocal { up, idx: i, val } => {
            (*up == depth && *i == idx) || uses_slot(val, depth, idx)
        }
        Ir::Let(inits, body) => {
            inits.iter().any(|x| uses_slot(x, depth + 1, idx)) || uses_slot(body, depth + 1, idx)
        }
        Ir::Lambda { body, .. } => uses_slot(body, depth + 1, idx),
        other => children(other).iter().any(|c| uses_slot(c, depth, idx)),
    }
}

/// Remap slot indices of the frame `depth` frames out via `map` (old idx → new
/// idx). Only kept slots are ever referenced, so `map[old]` is always `Some`.
fn renumber_frame(ir: &Ir, depth: u16, map: &[Option<u16>]) -> Ir {
    match ir {
        Ir::Local { up, idx } => {
            if *up == depth {
                Ir::Local { up: *up, idx: map[*idx as usize].expect("dead slot referenced") }
            } else {
                ir.clone()
            }
        }
        Ir::SetLocal { up, idx, val } => {
            let nidx = if *up == depth { map[*idx as usize].expect("dead slot assigned") } else { *idx };
            Ir::SetLocal { up: *up, idx: nidx, val: Box::new(renumber_frame(val, depth, map)) }
        }
        Ir::Let(inits, body) => Ir::Let(
            inits.iter().map(|x| renumber_frame(x, depth + 1, map)).collect(),
            Box::new(renumber_frame(body, depth + 1, map)),
        ),
        Ir::Lambda { nparams, variadic, body } => Ir::Lambda {
            nparams: *nparams,
            variadic: *variadic,
            body: Rc::new(renumber_frame(body, depth + 1, map)),
        },
        other => map_children(other, &mut |c| renumber_frame(c, depth, map)),
    }
}

/// Drop `let` bindings whose slot is never used AND whose init is pure,
/// renumbering the survivors. Bottom-up. (Sequential `let`: a binding is used if
/// referenced in the body or in ANY init — a later init can read an earlier
/// slot.)
pub fn eliminate_dead_lets(ir: &Ir) -> Ir {
    let n = map_children(ir, &mut |c| eliminate_dead_lets(c));
    let Ir::Let(inits, body) = &n else { return n };
    let k = inits.len();
    let live: Vec<bool> = (0..k)
        .map(|i| {
            uses_slot(body, 0, i as u16) || inits.iter().any(|init| uses_slot(init, 0, i as u16))
        })
        .collect();
    // Keep a slot unless it is dead AND its init is pure.
    let keep: Vec<bool> = (0..k).map(|i| live[i] || !is_pure(&inits[i])).collect();
    if keep.iter().all(|&b| b) {
        return n;
    }
    // old idx -> new idx for kept slots (in order).
    let mut map = vec![None; k];
    let mut next = 0u16;
    for i in 0..k {
        if keep[i] {
            map[i] = Some(next);
            next += 1;
        }
    }
    let new_inits: Vec<Ir> = (0..k)
        .filter(|&i| keep[i])
        .map(|i| renumber_frame(&inits[i], 0, &map))
        .collect();
    let new_body = renumber_frame(body, 0, &map);
    Ir::Let(new_inits, Box::new(new_body))
}

// ─────────────────────────────────────────────────────────────────────────
// Pass 0.7 — propagate_copies: substitute a `let`-bound literal into the body,
// leaving the (now dead) binding for `eliminate_dead_lets` to remove. Only
// `Const`/`Quote` (immutable) — a variable copy has the same mutation hazard as
// inline substitution. This is what lets `(let (x 3) (+ x x))` reach the folder
// as `(+ 3 3)` -> `6` across a fixpoint.
// ─────────────────────────────────────────────────────────────────────────

/// Is slot `idx` of the frame `depth` out ever the target of a `set!`?
fn slot_is_set(ir: &Ir, depth: u16, idx: u16) -> bool {
    match ir {
        Ir::SetLocal { up, idx: i, val } => {
            (*up == depth && *i == idx) || slot_is_set(val, depth, idx)
        }
        Ir::Let(inits, body) => {
            inits.iter().any(|x| slot_is_set(x, depth + 1, idx)) || slot_is_set(body, depth + 1, idx)
        }
        Ir::Lambda { body, .. } => slot_is_set(body, depth + 1, idx),
        other => children(other).iter().any(|c| slot_is_set(c, depth, idx)),
    }
}

/// Replace references to the frame `depth` out whose slot has a value in `vals`
/// with that value (lifted to the use depth). Non-target slots are untouched.
fn subst_slots(ir: &Ir, depth: u16, vals: &[Option<Ir>]) -> Ir {
    match ir {
        Ir::Local { up, idx } if *up == depth && vals[*idx as usize].is_some() => {
            shift(vals[*idx as usize].as_ref().unwrap(), depth, 0)
        }
        Ir::SetLocal { up, idx, val } => Ir::SetLocal {
            up: *up,
            idx: *idx,
            val: Box::new(subst_slots(val, depth, vals)),
        },
        Ir::Let(inits, body) => Ir::Let(
            inits.iter().map(|i| subst_slots(i, depth + 1, vals)).collect(),
            Box::new(subst_slots(body, depth + 1, vals)),
        ),
        Ir::Lambda { nparams, variadic, body } => Ir::Lambda {
            nparams: *nparams,
            variadic: *variadic,
            body: Rc::new(subst_slots(body, depth + 1, vals)),
        },
        other => map_children(other, &mut |c| subst_slots(c, depth, vals)),
    }
}

/// Propagate `let`-bound literals into the body (and later inits). Bottom-up.
pub fn propagate_copies(ir: &Ir) -> Ir {
    let n = map_children(ir, &mut |c| propagate_copies(c));
    if let Ir::Let(inits, body) = &n {
        let k = inits.len();
        let vals: Vec<Option<Ir>> = (0..k)
            .map(|i| {
                let set = slot_is_set(body, 0, i as u16)
                    || inits.iter().any(|x| slot_is_set(x, 0, i as u16));
                if is_literal(&inits[i]) && !set {
                    Some(inits[i].clone())
                } else {
                    None
                }
            })
            .collect();
        if vals.iter().any(|v| v.is_some()) {
            let new_inits = inits.iter().map(|x| subst_slots(x, 0, &vals)).collect();
            let new_body = subst_slots(body, 0, &vals);
            return Ir::Let(new_inits, Box::new(new_body));
        }
    }
    n
}

// ─────────────────────────────────────────────────────────────────────────
// Pass 1 — simplify: fold constant `if`, collapse a singleton `do`.
// ─────────────────────────────────────────────────────────────────────────

/// Fold branches whose condition is a literal, and unwrap a one-expression
/// `do`. Bottom-up, so an inner fold exposes an outer one. Idempotent.
pub fn simplify<M: ValueModel>(rt: &Runtime<M>, ir: &Ir) -> Ir {
    let n = map_children(ir, &mut |c| simplify::<M>(rt, c));
    match n {
        Ir::If(c, t, e) => {
            if let Ir::Const(id) = c.as_ref() {
                return if M::truthy(rt.decode(rt.get_const(*id))) { *t } else { *e };
            }
            Ir::If(c, t, e)
        }
        Ir::Do(mut xs) if xs.len() == 1 => xs.pop().unwrap(),
        other => other,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Pass 2 — specialize_fixnums: guard once at entry, then skip per-op checks.
// ─────────────────────────────────────────────────────────────────────────

fn is_arith(op: Prim) -> bool {
    matches!(op, Prim::Add | Prim::Sub | Prim::Mul | Prim::Lt | Prim::Eq)
}

fn to_fx(op: Prim) -> Prim {
    match op {
        Prim::Add => Prim::FxAdd,
        Prim::Sub => Prim::FxSub,
        Prim::Mul => Prim::FxMul,
        Prim::Lt => Prim::FxLt,
        Prim::Eq => Prim::FxEq,
        other => other,
    }
}

/// For each `Lambda`, wrap its body in a fixnum guard when profitable. Recurses
/// first, so nested lambdas are specialized independently. Idempotent: a body
/// already wrapped in an `AllFixnum` guard is left alone.
pub fn specialize_fixnums<M: ValueModel>(rt: &Runtime<M>, ir: &Ir) -> Ir {
    let n = map_children(ir, &mut |c| specialize_fixnums::<M>(rt, c));
    match n {
        Ir::Lambda { nparams, variadic, body } => {
            let guarded = guard_lambda::<M>(rt, nparams, variadic, &body);
            Ir::Lambda { nparams, variadic, body: Rc::new(guarded) }
        }
        other => other,
    }
}

/// Already fixnum-guarded? (Idempotency check — the guard is an optimizer-only
/// prim, so this shape is unambiguous.)
fn already_guarded(body: &Ir) -> bool {
    matches!(body, Ir::If(c, _, _) if matches!(c.as_ref(), Ir::Prim(Prim::AllFixnum, _)))
}

/// Tail/non-tail call census of a body: `(has_tail_call, has_nontail_call)`.
/// Standard tail positions: an `if`'s branches (if the `if` is tail), a `do`'s
/// last form, a `let`'s body; call/prim arguments are never tail.
fn call_census(ir: &Ir, tail: bool) -> (bool, bool) {
    let mut ht = false; // a call in tail position
    let mut hn = false; // a call in non-tail position
    match ir {
        Ir::Call(f, args) => {
            if tail {
                ht = true;
            } else {
                hn = true;
            }
            // The callee and every argument are evaluated in non-tail position.
            for c in std::iter::once(f.as_ref()).chain(args.iter()) {
                let (a, b) = call_census(c, false);
                ht |= a;
                hn |= b;
            }
        }
        Ir::Dispatch { args, .. } => {
            if tail {
                ht = true;
            } else {
                hn = true;
            }
            for c in args {
                let (a, b) = call_census(c, false);
                ht |= a;
                hn |= b;
            }
        }
        Ir::If(c, t, e) => {
            for (child, ctail) in [(c.as_ref(), false), (t.as_ref(), tail), (e.as_ref(), tail)] {
                let (a, b) = call_census(child, ctail);
                ht |= a;
                hn |= b;
            }
        }
        Ir::Do(xs) => {
            let last = xs.len().saturating_sub(1);
            for (i, x) in xs.iter().enumerate() {
                let (a, b) = call_census(x, tail && i == last);
                ht |= a;
                hn |= b;
            }
        }
        Ir::Let(inits, body) => {
            for i in inits {
                let (a, b) = call_census(i, false);
                ht |= a;
                hn |= b;
            }
            let (a, b) = call_census(body, tail);
            ht |= a;
            hn |= b;
        }
        // A nested lambda's own calls belong to ITS body, not this one.
        Ir::Lambda { .. } => {}
        other => {
            for c in children(other) {
                let (a, b) = call_census(c, false);
                ht |= a;
                hn |= b;
            }
        }
    }
    (ht, hn)
}

/// Build `(if (all-fixnum? p..) FAST BODY)` where `FAST` is `body` with every
/// arithmetic op over guaranteed-fixnum operands specialized. Returns `body`
/// unchanged when nothing qualifies.
///
/// PROFITABILITY: only a *pure self-tail loop* (a tail call, no non-tail calls)
/// is specialized. There the guard hoists to the loop header and amortizes over
/// the per-op tag checks it removes — a real win. On call-bound tree recursion
/// (fib/ack/tak) the same guard is per-call overhead plus code bloat for no
/// benefit — Cranelift already makes the per-op checks nearly free — so we leave
/// those alone. (Measured: loops +~8%, tree recursion neutral-to-negative.)
fn guard_lambda<M: ValueModel>(rt: &Runtime<M>, nparams: usize, variadic: bool, body: &Ir) -> Ir {
    if variadic || nparams == 0 || !M::R::is_immediate(Cat::Int) || already_guarded(body) {
        return body.clone();
    }
    let (has_tail, has_nontail) = call_census(body, true);
    if !has_tail || has_nontail {
        return body.clone();
    }
    // Params used as a direct operand of a depth-0 arithmetic op, minus any that
    // are ever reassigned (a `set!` would break the "constant fixnum" proof).
    let mut candidate = vec![false; nparams];
    let mut reassigned = vec![false; nparams];
    scan(body, 0, nparams, &mut candidate, &mut reassigned);
    let guarded: Vec<usize> =
        (0..nparams).filter(|&i| candidate[i] && !reassigned[i]).collect();
    if guarded.is_empty() {
        return body.clone();
    }

    let fast = mark_fast::<M>(rt, body, &guarded);
    let guard_args: Vec<Ir> =
        guarded.iter().map(|&i| Ir::Local { up: 0, idx: i as u16 }).collect();
    let guard = Ir::Prim(Prim::AllFixnum, guard_args);
    Ir::If(Box::new(guard), Box::new(fast), Box::new(body.clone()))
}

/// Find, at frame `depth` (0 = this lambda's own frame), which of the lambda's
/// `nparams` params are used as a direct operand of a depth-0 arithmetic op
/// (`candidate`) and which are ever assigned (`reassigned`). Depth rises through
/// the binding forms (`let`, nested `lambda`); a `Local`/`SetLocal` names one of
/// our params iff its `up` equals the current depth.
fn scan(ir: &Ir, depth: usize, nparams: usize, candidate: &mut [bool], reassigned: &mut [bool]) {
    match ir {
        Ir::SetLocal { up, idx, val } => {
            if *up as usize == depth && (*idx as usize) < nparams {
                reassigned[*idx as usize] = true;
            }
            scan(val, depth, nparams, candidate, reassigned);
        }
        Ir::Prim(op, args) if is_arith(*op) && args.len() == 2 => {
            if depth == 0 {
                for a in args {
                    if let Ir::Local { up: 0, idx } = a {
                        if (*idx as usize) < nparams {
                            candidate[*idx as usize] = true;
                        }
                    }
                }
            }
            for a in args {
                scan(a, depth, nparams, candidate, reassigned);
            }
        }
        Ir::Let(inits, body) => {
            for i in inits {
                scan(i, depth + 1, nparams, candidate, reassigned);
            }
            scan(body, depth + 1, nparams, candidate, reassigned);
        }
        Ir::Lambda { body, .. } => scan(body, depth + 1, nparams, candidate, reassigned),
        other => {
            for c in children(other) {
                scan(c, depth, nparams, candidate, reassigned);
            }
        }
    }
}

/// An operand is a *known* immediate fixnum iff it is a fixnum literal or a
/// direct (depth-0, `up:0`) reference to a guarded param. (Deliberately NOT the
/// result of another op — that can overflow to a bignum.)
fn operand_is_fixnum<M: ValueModel>(rt: &Runtime<M>, operand: &Ir, guarded: &[usize]) -> bool {
    match operand {
        Ir::Local { up: 0, idx } => guarded.contains(&(*idx as usize)),
        Ir::Const(id) => const_is_fixnum::<M>(rt, *id),
        _ => false,
    }
}

/// Rewrite depth-0 arithmetic ops whose BOTH operands are known fixnums into
/// their `Fx*` form. Stops at the first binding boundary (`let`/`lambda`): past
/// it, `up:0` no longer names our params, so the guard doesn't cover it.
fn mark_fast<M: ValueModel>(rt: &Runtime<M>, ir: &Ir, guarded: &[usize]) -> Ir {
    match ir {
        Ir::Prim(op, args) if is_arith(*op) && args.len() == 2 => {
            let a = &args[0];
            let b = &args[1];
            let marked_op = if operand_is_fixnum::<M>(rt, a, guarded)
                && operand_is_fixnum::<M>(rt, b, guarded)
            {
                to_fx(*op)
            } else {
                *op
            };
            // Recurse into operands regardless: a non-markable parent can still
            // contain a markable sub-op, e.g. `(< (- n 1) 5)`.
            let na = mark_fast::<M>(rt, a, guarded);
            let nb = mark_fast::<M>(rt, b, guarded);
            Ir::Prim(marked_op, vec![na, nb])
        }
        // Binding boundary: our params move to `up>0` here, so stop specializing.
        Ir::Let(..) | Ir::Lambda { .. } => ir.clone(),
        other => map_children(other, &mut |c| mark_fast::<M>(rt, c, guarded)),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The Optimized wrapper.
// ─────────────────────────────────────────────────────────────────────────

/// One cleanup round: inline, fold, propagate copies, simplify, DCE. Each pass
/// exposes work for the next (inlining exposes constant args → folding →
/// propagation → dead bindings), so the round is iterated to a fixpoint.
fn cleanup_round<M: ValueModel>(rt: &mut Runtime<M>, ir: &Ir) -> Ir {
    let a = inline(ir);
    let b = fold_const::<M>(rt, &a);
    let c = propagate_copies(&b);
    let d = simplify::<M>(rt, &c);
    eliminate_dead_lets(&d)
}

/// Run the pipeline to a fixpoint, then specialize profitable fixnum loops once
/// (specialization is codegen-prep, not a source cleanup — no reason to iterate
/// it). The loop is bounded; real programs converge in a couple of rounds.
fn optimize<M: ValueModel>(rt: &mut Runtime<M>, ir: &Ir) -> Ir {
    let mut cur = cleanup_round::<M>(rt, ir);
    for _ in 0..7 {
        let next = cleanup_round::<M>(rt, &cur);
        if next == cur {
            break;
        }
        cur = next;
    }
    specialize_fixnums::<M>(rt, &cur)
}

/// A composable backend that optimizes each form (nanopass pipeline over `Ir`)
/// before handing it to an inner `CodeSpace`. Opt-in: `Optimized::new(jit)`.
///
/// Crucially it delegates to the inner backend as the inner's OWN `top`, NOT by
/// staying in the open-recursion chain. The optimizer is a compile-time code
/// transform, not a call observer — so it must not do what `Traced` does. If it
/// threaded itself as `top`, a compiling backend would see `top != self`, flip
/// off its native (direct) calling convention, and route every call back
/// through the shim — turning the optimization into a net loss on call-heavy
/// code. Handing the transformed tree to `inner` as `inner`'s top keeps the
/// native path on (`direct == true` for a bare JIT) while still running the
/// specialized code. (Composition is preserved for the useful direction: an
/// observer BELOW the optimizer — `Optimized(Traced(jit))` — still sees every
/// call, because whether calls are native is `inner`'s decision, not ours.)
///
/// A second benefit: because the optimizer leaves the recursion chain, an
/// interpreter tier's per-node `eval_ir` re-entry lands on `inner`, never back
/// here — so each form (and each nested lambda body) is transformed exactly
/// once, for every backend. (The transform is idempotent regardless.)
pub struct Optimized<M: ValueModel> {
    inner: Box<dyn CodeSpace<M>>,
}

impl<M: ValueModel> Optimized<M> {
    pub fn new(inner: impl CodeSpace<M> + 'static) -> Self {
        Optimized { inner: Box::new(inner) }
    }
}

impl<M: ValueModel> CodeSpace<M> for Optimized<M> {
    fn eval_ir(&self, _top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, ir: &Ir, locals: &Locals) -> u64 {
        let opt = optimize::<M>(rt, ir);
        // Run on `inner` as its OWN top (see the type doc): keeps a compiling
        // backend's native calling convention on. `opt` owns the transformed
        // tree for this call; closures `Rc::clone` their specialized bodies, so
        // those survive independently.
        self.inner.eval_ir(self.inner.as_ref(), rt, &opt, locals)
    }

    fn invoke(&self, _top: &dyn CodeSpace<M>, rt: &mut Runtime<M>, callee: u64, args: &[u64]) -> u64 {
        self.inner.invoke(self.inner.as_ref(), rt, callee, args)
    }
}
