# Microlisp + Extensible JIT — Complete Plan

A small Lisp built directly on the dynamic-toolkit stack
(`dynir` / `dynlower` / `dynlang`), designed around **fully expressive,
same-image macros**. This document covers the toolkit changes required to
support incremental compilation, the language design, the macro system, and
a worked test suite proving the macro system is genuinely expressive.

## Why microlisp

The existing frontends (beagle, lox, lua2dynir, contlang) are all batch
compilers: parse the whole program, lower in one pass, JIT once, run.
That's adequate for benchmark-style programs but precludes:

- A REPL.
- Compile-time code execution (Lisp-style macros, partial evaluation).
- Self-modifying programs / `eval`.
- Any frontend that wants to interleave compilation and execution.

Microlisp is the smallest language that exercises every one of these. If the
toolkit can host a Lisp REPL with full macros, every batch frontend already
benefits, and the door is open for richer ones.

## Non-goals

- **Not a CL or Scheme clone.** Where they conflict, we pick whichever rule
  is simpler to implement on top of dynlang.
- **No hygienic macros (no `syntax-rules` / `syntax-case`).** `gensym` is the
  only hygiene tool. CL-style.
- **No reader macros.** The reader is fixed.
- **No tail-call optimization beyond what dynlower already gives us.** Tail
  calls in `if` / `cond` arms are fine; explicit `loop` macro covers
  iteration when we exhaust call frames.
- **No continuations / dynamic-wind in v1.** dynlower supports them but
  we don't need them for the macro story.

## Part 1 — Toolkit changes (the prerequisites)

The microlisp REPL needs the JIT to be incrementally extensible. That's
"Option C" from the design discussion: same image, no per-form fragmentation.
Five concrete changes, all confined to `dynlower` and `dynlang`.

### 1.1 Stable call table

**Problem.** `JitModule::compile_with_regalloc`
(`crates/dynlower/src/lib.rs:1015`) builds a `Vec<*const u8>` and bakes its
heap pointer into every emitted call site as an immediate
(`lib.rs:3260-3270`). Pushing past the Vec's capacity reallocates and
invalidates all previously emitted code.

**Fix.** Replace the Vec with a pointer-stable container:

```rust
pub struct CallTable {
    slots: Box<[Cell<*const u8>]>,   // fixed allocation, stable address
    len: AtomicUsize,                // next free slot
    capacity: usize,
}

impl CallTable {
    pub fn new(capacity: usize) -> Self { ... }
    pub fn base(&self) -> *const Cell<*const u8> { self.slots.as_ptr() }
    pub fn push(&self, ptr: *const u8) -> usize { /* fetch_add then write */ }
    pub fn set(&self, idx: usize, ptr: *const u8) { /* for late-patched internals */ }
}
```

`call_table_base` becomes `table.base() as u64`. Codegen unchanged: still
`movabs x27, base; ldr x28, [x27, #idx*8]; blr x28`.

**Capacity choice.** 64 K slots = 512 KB. Plenty for any plausible REPL
session. Promote to per-function indirection cells (one stable `Box` per
`FuncRef`, address-of-cell as the immediate) only if the cap ever bites.

**Touch points.** All four `compile_*` paths (`compile_fast_with_regalloc`,
`compile_batch`, `compile_with_regalloc`, callers in `crates/dynlang/src/gc.rs`).

### 1.2 Snapshotting `DynModule`

**Problem.** `DynModule::build(self) -> BuiltModule`
(`crates/dynlang/src/lib.rs:695`) consumes by value. After build you can't
declare more functions / types.

**Fix.** Add `pub fn snapshot(&self) -> Module` that clones the underlying
`dynir::Module` without consuming. `build()` stays for batch callers. Same
for `mb` (the `ModuleBuilder`): walk its function table and clone.
Everything else (`func_refs`, `auto_externs`, `obj_types`) is already
keyed by stable IDs that survive across snapshots.

```rust
impl DynModule {
    /// Take a snapshot of currently-declared/defined functions and types.
    /// The DynModule remains live and can be extended further.
    pub fn snapshot(&self) -> Module { self.mb.snapshot() }
}
```

The `dynir::ModuleBuilder` needs a corresponding `snapshot(&self) -> Module`
method that's a deep clone of its current state.

### 1.3 `JitModule::extend(...)`

**Problem.** `JitModule` is built once. No way to add functions later.

**Fix.** Split `compile_with_regalloc` into two phases:

```rust
impl JitModule {
    pub fn new_empty<Cfg, B, R>(
        capacity: usize,
        externs: &ExternEnv,
        safepoint_handler: Option<u64>,
        call_mode: CallMode,
    ) -> Self;

    pub fn extend<Cfg, B, R>(
        &mut self,
        module: &Module,
        new_func_idx_range: Range<usize>,
        externs: &ExternEnv,
    ) -> Vec<FuncRef>;   // refs of newly compiled funcs
}
```

`extend` does what the existing batch loop does, just for a slice:

1. For any `func_table` slots beyond `self.call_table.len()`, append: extern
   pointers go in directly; internals are pushed as null.
2. For each function in the range:
   - Build the `Lowerer` (same as today).
   - `memory.push(code)` — `PagedCodeMemory` already supports incremental
     pushes after a previous `finalize()`.
   - Append to `function_entry_offsets`, `function_safepoints`,
     `function_suspend_records`, `function_frame_reify_records`.
3. `memory.finalize()` — only re-protects newly written pages.
4. Patch newly written internal slots in the call table with their addresses.
5. `register_jit_code(start, end, safepoints)` for the new range.
6. Append to perf-map entries.

Cross-batch calls work for free: slots filled in extend N stay valid forever,
and extend N+1 emits `ldr + blr` against the same stable table base.

### 1.4 Call-mode pinning

**Problem.** `compile_with_regalloc:1037-1072` computes module-wide flags
(`uses_continuations`, `func_has_deopts`, `any_invoke`) before lowering, to
decide which calls need control-aware suspend/resume bookkeeping. In an
incremental setting we can't see future functions.

**Fix.** Pin the choice at JitModule creation:

```rust
pub enum CallMode {
    /// All internal calls are control-aware. Required if any function may
    /// use continuations, deopts, or Invoke terminators. ~2x call-site
    /// code size, small constant runtime overhead. Default for REPLs.
    ControlAware,
    /// All internal calls are plain call/return. Faster but rejects
    /// continuations and Invoke terminators at compile time.
    FastCall,
}
```

`extend` respects the mode and asserts violations (e.g. emitting an Invoke
in FastCall mode panics with a clear message).

### 1.5 Growable IC slots

**Problem.** `PropertyIc::finalize()` allocates the IC slot array once, on
the same one-shot model as the call table.

**Fix.** Same as 1.1 — back the IC array with a pointer-stable
`Box<[Cell<IcSlot>]>` of fixed capacity (16 K slots = 1 MB), with
`mint_slot(&self) -> usize` returning the next free index. Already-emitted
guards keep the same base address forever.

### 1.6 What stays the same

- `PagedCodeMemory` (`crates/dynasm/src/code_memory.rs`) is already
  incremental. No changes.
- `DynGcRuntime` and its heap. Lifetime is already designed to outlive any
  number of `compile_jit`/`run_jit` cycles.
- `auto_externs`, `slow_paths`, host context (`dynlang::host`). All grow-only
  by design.
- `StringPool`. Already grow-only.

### Resulting API

```rust
// One-time setup
let mut dm = DynModule::new(gc_config, tags);
let gc = DynGcRuntime::new(&gc_config, &tags, &dm.obj_types);
let mut jit = JitModule::new_empty::<Cfg, B, R>(64*1024, &externs, ..., CallMode::ControlAware);

// Per top-level form
fn handle_form(form: ConsCell, dm: &mut DynModule, jit: &mut JitModule, ...) {
    let expanded = macroexpand(form, &macro_env);
    let funcs_added = compile_form(expanded, dm);   // mutates dm
    let snap = dm.snapshot();
    let new_refs = jit.extend(&snap, funcs_added, &externs);
    if let Some(entry) = entry_func(&new_refs) {
        gc.run_jit(&jit, entry, &[]);
    }
}
```

No `build()` is ever called. The same `DynModule` and `JitModule` live for
the entire REPL session.

## Part 2 — Language design

### 2.1 Values

NanBox layout (already provided by `dynvalue`):

| Tag | Type | Notes |
|-----|------|-------|
| float | Number | f64 inline |
| 0 | nil | the empty list |
| 1 | bool | true/false |
| 2 | ptr | GC-allocated heap object (cons, symbol, string, vector, hashmap, function) |
| 3 | char | unicode scalar value |

Heap object types (registered as dynlang `ObjType`s):

- `cons` — two slots (`car`, `cdr`)
- `symbol` — interned identifier; one slot for the name string + a numeric ID
- `string` — varlen byte array
- `vector` — varlen value array (mutable)
- `hashmap` — open-addressed table on top of vector
- `closure` — captured env + `FuncRef`
- `macro` — same shape as closure but tagged for the expander

Symbols are interned via a `SymbolTable` host extern (Rust-side
`HashMap<String, NanBox>`) so `eq?` on symbols is pointer comparison.

### 2.2 Syntax (the reader)

The reader is ~250 lines of Rust. Tokens:

```
( )  [ ]              ; brackets are list sugar (Clojure-style optional)
'expr                 ; (quote expr)
`expr                 ; (quasiquote expr)
,expr                 ; (unquote expr)
,@expr                ; (unquote-splicing expr)
"string"              ; string literal with \n \t \\ \" escapes
123 -45 3.14 1e10     ; numbers
#t #f                 ; booleans
nil                   ; the empty list
#\c                   ; character literal
identifier            ; symbol
;...                  ; line comment
#|...|#               ; block comment
```

The reader produces cons-cell trees. `[a b c]` reads identically to
`(a b c)` (purely cosmetic). `'x` reads as `(quote x)`. Quasiquote and
friends read as `(quasiquote x)` / `(unquote x)` / `(unquote-splicing x)`.

### 2.3 Special forms

Recognized by symbol identity in head position. Anything else is a function
call.

| Form | Shape | Notes |
|------|-------|-------|
| `quote` | `(quote x)` | returns `x` literally |
| `if` | `(if c t e)` | `e` defaults to `nil` |
| `lambda` | `(lambda (params) body...)` | closes over lexical env |
| `let` | `(let ((x v)...) body...)` | parallel binding |
| `let*` | macro → nested `let` | |
| `letrec` | `(letrec ((f v)...) body...)` | mutual recursion |
| `define` | `(define name expr)` or `(define (name . args) body)` | top-level + internal |
| `defmacro` | `(defmacro name (params) body)` | binds in macro env |
| `set!` | `(set! name expr)` | mutation |
| `begin` | `(begin e1 e2...)` | sequencing; returns last |
| `quasiquote` | `` `tmpl `` | reader sugar; expander handles |
| `unquote` | `,e` | only valid inside quasiquote |
| `unquote-splicing` | `,@e` | only valid inside quasiquote |

`cond`, `when`, `unless`, `and`, `or`, `case`, `do`, `for`, `while` are
**all** macros. None are special forms. This is the test of expressiveness.

### 2.4 Lambda lists

`defmacro` and `lambda` accept the same parameter syntax:

```
(name1 name2 ... &rest tail)
(name1 &optional (n2 default) ... &rest tail)
((destructure-pat) ... &rest tail)        ; destructuring
```

Destructuring patterns recurse into cons structure. `&rest` collects the
remaining args into a list. `&optional` defaults to `nil` when omitted, or
to its default expression evaluated in the param env.

This is required for `defmacro` to feel like CL: macros routinely
destructure their arg list (e.g. `(defmacro for ((var start end) . body)
...)`).

### 2.5 The compiler

A pure Rust function:

```rust
pub fn compile_top_form(
    form: NanBox,
    dm: &mut DynModule,
    macro_env: &MacroEnv,
) -> Vec<FuncRef> {
    let expanded = macroexpand_all(form, macro_env);
    match head_symbol(expanded) {
        Some("define") => compile_define(expanded, dm),
        Some("defmacro") => compile_defmacro(expanded, dm, macro_env),
        Some("begin") => compile_begin(expanded, dm),
        _ => compile_eval_expr(expanded, dm),  // anonymous "main"
    }
}
```

The body of a `define` or `defmacro` is lowered just like beagle lowers a
function body: walk the cons cells, emit `dynir` instructions via
`DynFunc`. Closures over lexical bindings allocate environment cons cells
explicitly (no escape analysis in v1).

Special forms have direct emitters. Function calls go through the call
table. Built-in primitives (`car`, `cdr`, `cons`, arithmetic, `eq?`, etc.)
are externs registered at `DynModule` construction.

## Part 3 — The macro system

### 3.1 The macro environment

```rust
pub struct MacroEnv {
    /// Symbol → FuncRef of the JIT-compiled macro body.
    /// FuncRef points into the SHARED JitModule — same image as runtime.
    macros: HashMap<SymbolId, FuncRef>,
}
```

The same `JitModule` holds both runtime and compile-time code. When the
expander needs to call macro `foo`, it just runs `gc.run_jit(jit, fref,
&args)` against the live JitModule — same heap, same IC, same everything.

### 3.2 `defmacro`

Compilation of `(defmacro name (params) body)`:

1. Create a new function in `dm` with `name` and one param per `params`.
2. Compile the body just like any other function — full language available
   (cons, car, cdr, all primitives, all previously-defined functions and
   macros).
3. After `dm.snapshot()` + `jit.extend(...)`, look up the new `FuncRef`.
4. Insert `(name → FuncRef)` into `macro_env`.

Crucially, macro bodies are **regular compiled code** in the **same
JitModule**. There is no separate "compile-time interpreter." When the
expander calls the macro, it's a plain JIT call.

### 3.3 The expander

```rust
fn macroexpand_1(form: NanBox, env: &MacroEnv) -> (NanBox, bool) {
    let Some(head) = head_symbol(form) else { return (form, false); };
    let Some(&fref) = env.macros.get(&head) else { return (form, false); };
    let args = list_tail(form);   // (head a1 a2 ...) → (a1 a2 ...)
    let result = run_macro(fref, args);   // JIT call
    (result, true)
}

fn macroexpand_all(form: NanBox, env: &MacroEnv) -> NanBox {
    let mut current = form;
    loop {
        let (next, expanded) = macroexpand_1(current, env);
        if !expanded { current = next; break; }
        current = next;
    }
    // Recurse into sub-forms after head expansion has settled.
    walk_subforms(current, |sub| macroexpand_all(sub, env))
}
```

Subform walking respects special forms: it does NOT recurse into the
body of `quote`, only into the *evaluated* positions of `lambda`/`let`/etc.

### 3.4 Quote, quasiquote, unquote, unquote-splicing

`'x` is trivial — the compiler emits a literal cons-cell tree.

Quasiquote is implemented in the **expander**, not as a runtime construct,
by translating templates into explicit `cons`/`list`/`append` calls:

```
`x                → 'x
`,e               → e
`(a b ,c ,@d)     → (cons 'a (cons 'b (cons c (append d '()))))
`(a . ,b)         → (cons 'a b)
`#(a ,b)          → (vector 'a b)            ; vectors too
```

This is a pure cons-tree → cons-tree rewrite, ~80 lines. Done before
`compile_form`, after `macroexpand_all`.

### 3.5 `gensym`

```
(gensym)        → #:g1024
(gensym "tag")  → #:tag1025
```

Manual hygiene: macros that bind names use `(let ((g (gensym))) `(let ((,g ...)) ...))`
to avoid capturing user-supplied names. Standard CL discipline.

Implementation: a Rust extern bumping an atomic counter, returning a
freshly-interned uninternable symbol (the `#:` prefix is purely for
printing; identity is by symbol-id).

### 3.6 `eval` and `macroexpand` at runtime

Both are externs that re-enter the compiler:

```rust
extern "C" fn lisp_eval(form: NanBox) -> NanBox {
    let h = host::<MicrolispHost>();
    let mut dm = h.dm.borrow_mut();
    let mut jit = h.jit.borrow_mut();
    let funcs = compile_top_form(form, &mut dm, &h.macro_env);
    let snap = dm.snapshot();
    jit.extend(&snap, funcs.range, &h.externs);
    h.gc.run_jit(&jit, funcs.entry, &[])
}

extern "C" fn lisp_macroexpand(form: NanBox) -> NanBox {
    let h = host::<MicrolispHost>();
    macroexpand_all(form, &h.macro_env)
}
```

Both are callable from any Lisp code. `eval` is the clearest demonstration
that macro and runtime share the image: the expander, the compiler, and
the JIT are all reachable from running code.

### 3.7 Macroexpansion order — why same-image matters

Consider:

```lisp
(define (helper xs)            ; 1: regular function
  (map list xs xs))

(defmacro pair-up (xs)          ; 2: macro that calls helper
  `(quote ,(helper xs)))

(define result (pair-up (1 2 3)))   ; 3: at compile time, calls helper
```

Form 1 compiles → `helper` lives in the JitModule.
Form 2 compiles → `pair-up` lives in the JitModule and `macro_env`.
Form 3:
- `macroexpand_all` sees `(pair-up (1 2 3))`.
- Looks up `pair-up` in `macro_env`, finds its `FuncRef`.
- `run_jit(jit, fref, [list(1,2,3)])`.
- Inside the macro body, `(helper xs)` is a regular call — compiled to a
  call-table indirect call, lands on `helper`'s entry.
- Returns `(quote ((1 1) (2 2) (3 3)))`.
- Expander returns `(quote ((1 1) (2 2) (3 3)))`.
- Compiler emits a literal.

This only works because there is one image. With phase-isolated macros
you'd have to redefine `helper` in the macro phase, or import it
"for-syntax" — neither feels like Lisp.

## Part 4 — Macros that must work (the expressiveness suite)

If any of these can't be expressed, the macro system isn't fully expressive.
Each is a test case in `crates/microlisp/tests/`.

### 4.1 The basics

```lisp
(defmacro when (cond . body)
  `(if ,cond (begin ,@body) nil))

(defmacro unless (cond . body)
  `(if ,cond nil (begin ,@body)))
```

Tests: `,@body` splicing; `&rest` in lambda lists.

### 4.2 Recursive expansion (`cond`)

```lisp
(defmacro cond clauses
  (if (null? clauses)
      'nil
      (let ((first (car clauses))
            (rest (cdr clauses)))
        (if (eq? (car first) 'else)
            `(begin ,@(cdr first))
            `(if ,(car first)
                 (begin ,@(cdr first))
                 (cond ,@rest))))))
```

Tests: macro calling itself in its expansion (the inner `(cond ,@rest)`
re-expands recursively); `else` keyword detection.

### 4.3 Sequential binding (`let*`)

```lisp
(defmacro let* (bindings . body)
  (if (null? bindings)
      `(begin ,@body)
      `(let (,(car bindings))
         (let* ,(cdr bindings) ,@body))))
```

Tests: a macro that produces another use of itself, requiring re-expansion
to fixed point.

### 4.4 Short-circuit (`and`, `or`)

```lisp
(defmacro and args
  (cond ((null? args) #t)
        ((null? (cdr args)) (car args))
        (else `(if ,(car args) (and ,@(cdr args)) #f))))

(defmacro or args
  (cond ((null? args) #f)
        ((null? (cdr args)) (car args))
        (else (let ((g (gensym)))
                `(let ((,g ,(car args)))
                   (if ,g ,g (or ,@(cdr args))))))))
```

Tests: macros calling **other macros** during expansion (`and` body uses
`cond`, which is itself a macro); `gensym` for hygiene in `or`.

### 4.5 Iteration (`for`, `while`)

```lisp
(defmacro while (cond . body)
  (let ((loop (gensym "loop")))
    `(letrec ((,loop (lambda ()
                       (when ,cond
                         ,@body
                         (,loop)))))
       (,loop))))

(defmacro for ((var start end) . body)
  (let ((s (gensym)) (e (gensym)))
    `(let ((,s ,start) (,e ,end))
       (let loop ((,var ,s))
         (when (< ,var ,e)
           ,@body
           (loop (+ ,var 1)))))))
```

Tests: destructuring lambda lists (`((var start end) . body)`); multiple
gensyms in one macro; macro using another macro (`while` uses `when`).

### 4.6 Pattern matching (`match`)

```lisp
(defmacro match (expr . clauses)
  (let ((g (gensym)))
    `(let ((,g ,expr))
       ,(expand-match-clauses g clauses))))

(define (expand-match-clauses scrutinee clauses)
  (if (null? clauses)
      '(error "no match")
      (let ((c (car clauses)))
        `(if ,(compile-pattern (car c) scrutinee)
             (begin ,@(cdr c))
             ,(expand-match-clauses scrutinee (cdr clauses))))))

(define (compile-pattern pat scrutinee) ...)
```

Tests: macros calling **non-trivial helper functions** (`expand-match-clauses`,
`compile-pattern`); recursive helper that walks pattern structure; helpers
that themselves use macros (`when`, `cond`).

### 4.7 Macros generating macros (`defstruct`)

```lisp
(defmacro defstruct (name . fields)
  (let ((ctor (symbol-append 'make- name))
        (pred (symbol-append name '?)))
    `(begin
       (define (,ctor ,@fields)
         (vector ',name ,@fields))
       (define (,pred x)
         (and (vector? x) (eq? (vector-ref x 0) ',name)))
       ,@(map-indexed
          (lambda (i field)
            `(define (,(symbol-append name '- field)) (s)
               (vector-ref s ,(+ i 1))))
          fields))))
```

Tests: emitting multiple top-level definitions from one macro;
runtime-time `symbol-append` callable from macro context; `map-indexed`
helper used by macro.

### 4.8 Macroexpand-driven self-test

```lisp
(define (test-expand input expected)
  (let ((actual (macroexpand input)))
    (when (not (equal? actual expected))
      (error "expansion mismatch" input actual expected))))

(test-expand
  '(when (> x 0) (print x))
  '(if (> x 0) (begin (print x)) nil))

(test-expand
  '(let* ((a 1) (b (+ a 1))) b)
  '(let ((a 1)) (let ((b (+ a 1))) (begin b))))
```

Tests: `macroexpand` exposed to user code; runtime use of the same expander
the compiler uses. Self-checking suite.

### 4.9 `eval` round-trip

```lisp
(define program
  '(begin
     (define (fact n)
       (if (<= n 1) 1 (* n (fact (- n 1)))))
     (fact 10)))

(assert (= (eval program) 3628800))

;; A macro defined via eval
(eval '(defmacro twice (e) `(begin ,e ,e)))
(define counter 0)
(twice (set! counter (+ counter 1)))
(assert (= counter 2))
```

Tests: data → executable code; macro defined at runtime is visible to
subsequent compilation. Same-image is the whole point of this case
working.

### 4.10 The "anaphoric if"

```lisp
(defmacro aif (test then . else)
  `(let ((it ,test))
     (if it ,then ,@else)))

(aif (lookup "key" db)
     (process it)
     (default-value))
```

Tests: intentional capture (anti-hygiene). Proves we don't accidentally
gensym away user-visible bindings.

### 4.11 The torture test (`once-only`, then `do`)

```lisp
;; once-only: helper macro used by other macros
(defmacro once-only (vars . body)
  (let ((gensyms (map (lambda (v) (gensym (symbol->string v))) vars)))
    `(list 'let
       (list ,@(map (lambda (g v) `(list ',g ,v)) gensyms vars))
       ,(let ((body-substituted
               (subst-vars body (map cons vars gensyms))))
          `(list ,@body-substituted)))))

;; do: the CL iteration macro, built on once-only
(defmacro do (bindings (test . result) . body)
  (let ((loop (gensym "do-loop")))
    `(letrec ((,loop
               (lambda ,(map car bindings)
                 (if ,test
                     (begin ,@result)
                     (begin
                       ,@body
                       (,loop ,@(map (lambda (b)
                                       (if (null? (cddr b))
                                           (car b)
                                           (caddr b)))
                                     bindings)))))))
       (,loop ,@(map cadr bindings)))))
```

If both of these work, the system is genuinely expressive. They use:

- Higher-order helpers (`map`, `subst-vars`) from macros.
- Macros that build other macro definitions.
- Multiple gensyms with named tags.
- Destructuring of arbitrary depth in lambda lists.

## Part 5 — Implementation order

Each step has a passing test before the next begins.

### Step A — toolkit changes (Part 1)

A1. Stable `CallTable` replacing `Vec<*const u8>`. All existing tests still pass.
A2. `dynir::ModuleBuilder::snapshot()`. `DynModule::snapshot()`.
A3. `JitModule::new_empty` + `extend`. New test: compile module with `f`,
    run; extend with `g` that calls `f`, run.
A4. `CallMode` flag.
A5. Growable IC slots (only needed once microlisp uses them — defer).

**Done when:** beagle's existing benchmarks pass on top of the new API
(call sites use `CallMode::ControlAware` to match today's behavior); a
new dynlower integration test exercises incremental extend.

### Step B — microlisp value model & reader

B1. `crates/microlisp` crate scaffold. Cons / symbol / string / vector
    obj-types registered. Symbol intern table.
B2. Reader producing cons-cell trees. Tests for every token class plus
    quasiquote desugaring.
B3. `printer` for round-tripping values to text. Used by tests.

**Done when:** `read_then_print(s)` is a stable round-trip for every token
class.

### Step C — non-macro compiler

C1. Compile `quote`, `if`, `lambda`, `let`, `letrec`, `define`, `set!`,
    `begin` directly to `dynir`. No macros yet.
C2. Built-in primitives as externs: arithmetic, comparisons, `cons`,
    `car`, `cdr`, `eq?`, `null?`, `pair?`, `vector*`, `print`, `error`.
C3. Closures: emit explicit env cons cells; `lambda` allocates and
    captures.
C4. REPL loop: read → compile_top_form → snapshot → extend → run.

**Done when:** can run a non-macro program end-to-end including recursion,
closures, mutation. Test: `fact`, `fib`, `map`/`filter` written in
microlisp itself.

### Step D — macros

D1. `defmacro` special form: compile the body like a regular function,
    register `(symbol → FuncRef)` in `macro_env`.
D2. `macroexpand_1` and `macroexpand_all`. Subform walking that respects
    special-form shape.
D3. Quasiquote → cons/list/append rewrite, run before `compile_form`.
D4. Lambda-list destructuring (required + rest + optional + nested).
D5. `gensym` extern.
D6. `eval` and `macroexpand` exposed as Lisp-callable externs.

**Done when:** every example in Part 4 expands correctly and runs to the
expected result.

### Step E — REPL + tooling

E1. Multi-line input.
E2. `(load "file.lisp")` — repeated compile_top_form against a file.
E3. Error messages with source positions (carry token spans through the
    reader → expander → compiler).
E4. `(time expr)` macro (uses `eval`, demonstrates compile-time access to
    runtime).

**Done when:** can paste the entire Part 4 test suite into the REPL
without restart and have it pass.

## Part 6 — Risks and open questions

### 6.1 Recursion depth in the expander

`macroexpand_all` is itself recursive over arbitrary Lisp structure. For
deeply nested user code the host stack can overflow. Mitigate by writing
the expander iteratively with an explicit work stack, or by spawning a
larger-stack thread for compilation (beagle already does this in
`main.rs:116-121`). Pick after measurement.

### 6.2 GC during compilation

Macro bodies allocate. The compiler itself allocates (it's reading and
producing cons cells). Both run on the same heap as user code. Need to
verify: while we're walking a cons-cell program inside the compiler, the
JIT-callable macro body allocates and triggers GC — does our walker hold
roots that survive? The `DynGcRuntime` API already takes
`extra_roots: &[&dyn TraceRoots]`; the compiler must thread its
in-flight cons handles through. Plan: every `Lowerer` visit method that
holds a `NanBox` across an allocation pushes it to a thread-local root
vec, pops on return. Same discipline as
`docs/toolkit-improvements/02-gc-rooted-temps.md`.

### 6.3 Forward references across `define`s

Form 1: `(define (f x) (g x))` — `g` not yet defined.
Form 2: `(define (g x) (* x 2))`.

Solution: each top-level `define` does (a) declare a `FuncRef` for the
name immediately, before compiling the body, and (b) insert the FuncRef
into a global name table. References to undefined globals compile to a
deferred-resolution stub: a load from a global cell that's null until
the symbol is defined, plus a runtime check on each call. After the
referenced define lands, the cell points at its entry. This matches how
CL handles "function not yet defined" warnings.

This means user code that references undefined names compiles cleanly,
fails at first call until resolved, and works thereafter. Slightly
different from beagle's batch model where everything must be resolved
before any compilation. Fine for a REPL.

### 6.4 Macro recompilation when a helper changes

A user redefines `helper`. Macros that called `helper` at expansion time
already produced their expansion — no recompile of *those* macros is
needed (their output is just data). But macros that haven't yet expanded
will see the new `helper`. This matches CL: a redefined helper takes
effect for future macro expansions, not past ones. Document and move on.

### 6.5 Macro vs function name collisions

If `foo` is both a macro and a function, which wins? Pick: macros shadow
functions in head position. Function position only consults the macro
env. This is the CL convention. Implement by checking `macro_env` first
in `macroexpand_1` before looking up a function binding.

### 6.6 Tail calls and `loop` macros

The `while` macro above expands to a `letrec` + recursive call. Without
TCO this stack-overflows. `dynlower` already supports tail-call
emission for direct internal calls in tail position; need to verify that
the microlisp compiler emits the right `Inst::TailCall` for self-calls
inside `letrec`. If it doesn't, the loop macro is a source-level lie.
This is a real risk and worth a dedicated test before the macro test
suite.

### 6.7 Hygiene by gensym only

Manual gensym is error-prone. The user must remember to gensym every
binding their macro introduces. If we ever want `syntax-rules` /
`syntax-case` later, the expander needs to grow scope-tracking. Out of
scope for v1; document explicitly as a limitation.

## Summary of moving parts

| Component | Where | LOC est. |
|-----------|-------|----------|
| Stable `CallTable` | `crates/dynlower/src/lib.rs` | ~80 |
| `ModuleBuilder::snapshot` | `crates/dynir/` | ~50 |
| `DynModule::snapshot` | `crates/dynlang/src/lib.rs` | ~10 |
| `JitModule::new_empty` + `extend` | `crates/dynlower/src/lib.rs` | ~250 (mostly refactor) |
| Growable IC slots | `crates/dynlang/src/ic.rs` | ~40 |
| Microlisp value model | `crates/microlisp/src/value.rs` | ~300 |
| Reader / printer | `crates/microlisp/src/reader.rs` | ~350 |
| Compiler (special forms + calls) | `crates/microlisp/src/compile.rs` | ~800 |
| Closures + env | `crates/microlisp/src/compile.rs` | ~150 |
| Primitives (externs) | `crates/microlisp/src/prims.rs` | ~400 |
| Macroexpander | `crates/microlisp/src/expand.rs` | ~250 |
| Quasiquote rewrite | `crates/microlisp/src/expand.rs` | ~80 |
| Lambda-list destructure | `crates/microlisp/src/lambda_list.rs` | ~150 |
| `eval` / `macroexpand` externs | `crates/microlisp/src/host.rs` | ~80 |
| REPL driver | `crates/microlisp/src/main.rs` | ~150 |
| Test suite (Part 4) | `crates/microlisp/tests/` | ~500 |
| **Total new code** | | **~3.6 K LOC** |

The toolkit work is ~430 LOC, the microlisp itself is ~3.2 K LOC. Modest.

Once the toolkit work lands, every existing frontend (beagle, lox, lua,
contlang) gets the same option to switch from batch to incremental at
zero additional cost.
