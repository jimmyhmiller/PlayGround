# Traits — structured, definition-checked, monomorphized (design)

A Rust/Carbon-style trait system: declare a trait, implement it per type, bound a
generic on it, and have method calls resolve to the right impl — with bounds
**checked where the generic is defined**, not at instantiation. Coil is
whole-program and monomorphizing, so we drop everything Rust needs for separate
compilation: no coherence/orphan rules, no runtime dictionaries, no vtables
(static dispatch only — `dyn` is a later, separate feature).

## Surface

```lisp
(deftrait Eq [Self]
  (eq [(a Self) (b Self)] (-> bool)))

(impl Eq Point
  (eq [(a Point) (b Point)] (-> bool)
    (and (icmp-eq (load (field a x)) (load (field b x)))
         (icmp-eq (load (field a y)) (load (field b y))))))

; a bounded type parameter: `(T Eq)` = T must implement Eq (bare `T` = no bound)
(defn all-eq [(T Eq)] [(xs (ptr (ArrayList T))) (x T)] (-> bool)
  (let [(mut i) 0 (mut ok) true n (al-len [T] (load xs))]
    (loop (if (icmp-ge (load i) n) (break)
            (do (if (eq (al-get [T] (load xs) (load i)) x) 0 (store! ok false))
                (store! i (iadd (load i) 1)))))
    (load ok)))
```

`derive` stays a macro on top: `(derive Eq Point)` expands to the `impl` by
reflecting fields (today's `derive-eq`, registered as an impl).

## Representation (AST)

- `TraitDef { name, self_param, methods: Vec<TraitMethod> }`,
  `TraitMethod { name, params, ret }` — param/ret types may mention `Self`
  (represented as the type name `"Self"`).
- `ImplDef { trait_name, for_type, methods: Vec<Func> }` — each method is lowered
  to an ordinary `Func` named `<Trait>$<Type>$<method>` (e.g. `Eq$Point$eq`), with
  `Self` substituted to `for_type`. So codegen/mono see plain functions.
- `Func.bounds: Vec<(String, Vec<String>)>` — type param → required traits.
  (`type_params` stays `Vec<String>`; bounds are parallel, minimal churn.)
- `ExprKind::TraitCall { trait_name, method, self_tp, args }` — a *deferred* call
  inside a bounded generic (Self is a type parameter). Produced by the checker,
  resolved away by monomorphization; codegen never sees it.
- `Program.traits`, `Program.impls`.

## Resolution

Tables built in `check`:
- `methods: name -> (trait, self_index)` (v1: trait-method names are globally
  unique — a collision is a hard error).
- `impls: (trait, type) -> { method -> mangled fn name }`.

Checking a call `(m args…)` where `m` is a trait method of trait `Tr` (Self at
`self_index`):
1. `self_ty` = type of the Self-position argument.
2. **concrete** `self_ty = C`: look up `impls[(Tr, C)]`; rewrite to a normal call
   of the mangled impl fn. Missing impl → error “C does not implement Tr”.
3. **type parameter** `self_ty = P`: it must be that `Tr ∈ bounds[P]`
   (**definition-time bound check** — the body is checked once against the
   declared bound). Then emit `TraitCall{Tr, m, P, args}` (deferred). If `P` is
   not bounded by `Tr` → error “P is not bounded by Tr”.
The method’s declared signature (with `Self := self_ty`) types the call either way.

Instantiation-site bound check: when a bounded generic is called with `T = C`
(solved in `solve_type_args`), verify `impls[(Tr, C)]` exists for each `Tr ∈
bounds[T]` → else error “C does not implement Tr (required by …)”.

Monomorphization: `resolve_expr` turns `TraitCall{Tr, m, P, args}` with `P := C`
into a normal call of `impls[(Tr, C)].m`. The impl functions are already ordinary
(possibly generic-instantiated) funcs, so nothing else changes.

## Phasing

- **Phase 1 (DONE):** `deftrait`/`impl`, bound list per param `(T A B)`,
  definition-time bound checking, concrete + deferred resolution, monomorphized
  static dispatch, impl↔trait signature conformance, all-methods-implemented
  check. Tests in `tests/traits.rs`; full suite 370 green.
- Later: `derive Trait Type` integration, supertraits, multiple-Self / associated
  types, generic impls (`impl Eq for (Pair T T)`), and `dyn Trait` (vtables).

## v1 idiom: pass `Self` by pointer

A trait method should take `Self` **by pointer** — `(eq [(a (ptr Self)) …] …)` —
not by value. Reason: a monomorphized generic with an aggregate type-arg passes
that aggregate *by value* (the type param was unconstrained at the generic's
definition, so the reference model didn't ref it), but an impl method over a
struct takes it *by reference* (the reference model refs concrete aggregate
params). A by-value `(a Self)` therefore mismatches the impl's `(ptr Point)` at a
deferred (generic) call site — an LLVM-verifier error. Passing `(ptr Self)` makes
every argument a pointer (a scalar), so the ABI is uniform across concrete and
generic calls. (Concrete by-value `(a Self)` calls *do* work — `coerce_arg` loads
a place to a value — but the generic case does not, so prefer `(ptr Self)`.)
Scalar `Self` (e.g. an `Ord` over `i64`) also works by value. Making by-value
aggregate `Self` work in generics needs the deeper fix: monomorphized generics
re-deriving by-reference ABI for aggregate type-args (a known follow-up).

## Implementation map

- `ast.rs`: `TraitDef`/`TraitMethod`/`ImplDef`, `Func.bounds`,
  `ExprKind::TraitCall`, `Program.{traits,impls}`, `trait_method_fn` (mangling).
- `parse.rs`: `deftrait`/`impl`; `(T Trait…)` bound entries in the type-param vector.
- `resolve.rs`: qualifies types in trait/impl sigs (Self stays); flat trait namespace.
- `check.rs`: trait/method/impl tables; lowers impl methods to `Trait$Type$method`
  funcs; conformance check; trait-call resolution (concrete → direct, type-param →
  `TraitCall` after the bound check); instantiation-site bound check.
- `mono.rs`: resolves `TraitCall` to a direct call of the implementing function.
- `codegen.rs`: never sees `TraitCall` (a guard errors if one slips through).
