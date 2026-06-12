# funct — implementation spec (v0.1 draft)

> Name: **funct** (a nod to *functional* / *function*). A small, functional, embeddable language built around `|>`, real pattern
> matching, a stack-based bytecode VM whose state is fully reified (so it can be paused/stepped/inspected
> between any two instructions), hot reloading, and frictionless Rust interop.
>
> This is the *build* document — the companion `funct-language-sketch.html` is the design/feedback artifact.
> Where the two disagree, this file wins; update it as decisions change.

---

## 0. Decisions locked so far

| Area | Decision |
|------|----------|
| Paradigm | Expression-oriented, functional-leaning, dynamically dispatched |
| Types | **Gradual** — annotations optional, checked at compile/reload where present |
| Lambda syntax | `x => x + 1`, `(x, y) => …` |
| Pattern matching | Full: destructuring, guards, ranges, or-patterns, `as`-binding, list `..rest` |
| Pipe | `|>` first-class; unified with method calls via UFCS |
| Top-level mutability | **None.** No `var`, no top-level `let mut`. |
| Local mutability | `let mut` inside a fn/block only; cannot escape its scope |
| Escaping/shared state | **Atoms only** — `atom(v)`, `@a` deref, `swap!`, `reset!` |
| Errors | `Result` + `?` + `match`. **No** algebraic effects. |
| VM | **Stack-based** bytecode; reified `VmState`; step/suspend/resume |
| Hot reload | Name-keyed function table, atom-state survives, atomic swap at instruction boundary |

Open: final name; atom deref syntax (`@a` vs `a.value`); `!`-suffix naming convention; reload semantics for
in-flight continuations and changed data shapes.

---

## 1. Lexical grammar

UTF-8 source. Newlines are significant as statement terminators (see §3.1). Indentation is **not** significant.

```
comment      ::= "//" .* (newline | EOF)
ident        ::= (alpha | "_") (alnum | "_")*          // 'swap!' / 'reset!': trailing '!' allowed (see note)
typename     ::= upper (alnum | "_")*                  // Capitalized = type/constructor
int          ::= digit (digit | "_")*
float        ::= digit (digit|"_")* "." digit (digit|"_")*
string       ::= '"' (char | escape | interp)* '"'
interp       ::= "{" expr "}"                          // inside a string literal
escape       ::= "\" ( "n" | "t" | "\" | '"' | "{" | "}" | "0" )
```

Notes:
- A trailing `!` is permitted in identifiers solely to support the `swap!`/`reset!` mutation-marker convention
  (open question — may be dropped). The lexer treats `name!` as a single ident token.
- String interpolation `"{expr}"` lexes to a sequence of string-chunk + embedded-expr tokens; the parser
  reassembles it into a concat expression.

### Keywords
```
let  mut  fn  match  if  else  while  for  in  type  import  export  return
true  false  and  or  not  as
```
`atom`, `swap!`, `reset!`, `map`, `filter`, etc. are **not** keywords — they are ordinary functions in the
prelude. `match`, `if`, `fn` are keywords because they introduce special forms.

### Operators / punctuation
```
|>            pipe
=> ->         lambda/arm body ; (-> reserved, maybe ranges/types)
+ - * / % **  arithmetic
== != < <= > >=   comparison
and or not    logical (keywords, short-circuit)
= += -= *= /= %=   assignment / compound (compound only valid on `let mut` locals & atoms via swap!)
.. ..=        exclusive / inclusive range
.             field / method access
?             try-propagate (postfix)
@             atom deref (prefix)   [open]
_             hole / wildcard
( ) [ ] { }   grouping / list / block-or-record
, ; :         separators
.. (prefix)   spread in record/list update & rest-patterns
```

Precedence, lowest → highest:
```
1  |>                       (left-assoc)
2  or
3  and
4  == != < <= > >=          (non-assoc)
5  .. ..=
6  + -
7  * / %
8  **                       (right-assoc)
9  unary: not  -  @
10 postfix: f(args)  x.field  x[i]  x?
```

---

## 2. Values & runtime types

```
Unit            ()                       the empty value
Bool            true | false
Int             i64
Float           f64
Char            'x'                       (optional in v0.1)
Str             immutable UTF-8 string
List            persistent vector         [1,2,3]
Tuple           fixed heterogeneous       (1, "a")
Map / Record    persistent hash map       { x: 1, y: 2 }
Variant         tag + payload             Some(3), Circle { radius: 2.0 }
Fn              closure (code ptr + captured env)
Atom            shared mutable cell holding one Value
Native          opaque host (Rust) value, registered via interop
```

All non-Atom values are **immutable** and use structural sharing. `Atom` is the *only* mutable runtime
object. Equality is structural for everything except `Atom`/`Native`, which are by identity.

---

## 3. Syntactic grammar (EBNF sketch)

```
program     ::= item*
item        ::= fn_def | type_def | import | export | let_top | expr_stmt
let_top     ::= "let" pattern "=" expr                 // top-level: NO 'mut' allowed (enforced in §3.4)

fn_def      ::= "fn" ident "(" params? ")" ( "=" expr | block )
params      ::= pattern ("," pattern)*
type_def    ::= "type" typename generics? "=" type_body
type_body   ::= record_ty | variant_ty
variant_ty  ::= "|"? variant ("|" variant)*
variant     ::= typename record_ty?                    // Circle { radius: Float } | None

block       ::= "{" stmt* expr? "}"                    // last expr is the block's value
stmt        ::= let_local | expr_stmt | while | for | return
let_local   ::= "let" "mut"? pattern "=" expr          // 'mut' legal here only
while       ::= "while" expr block
for         ::= "for" pattern "in" expr block

expr        ::= pipe
pipe        ::= or ( "|>" call_tail )*
or          ::= and ("or" and)*
and         ::= cmp ("and" cmp)*
cmp         ::= range (cmp_op range)?
range       ::= add (("..="|"..") add)?
add         ::= mul (("+"|"-") mul)*
mul         ::= pow (("*"|"/"|"%") pow)*
pow         ::= unary ("**" pow)?
unary       ::= ("not"|"-"|"@") unary | postfix
postfix     ::= primary ( call | field | index | "?" )*
call        ::= "(" args? ")"
field       ::= "." ident
index       ::= "[" expr "]"
primary     ::= literal | ident | typename ctor? | lambda | if | match
              | block | list | tuple | record | "(" expr ")" | "_"

lambda      ::= ident "=>" expr
              | "(" params? ")" "=>" ( expr | block )
if          ::= "if" expr block ("else" (if | block))?
match       ::= "match" expr? "{" arm ("," arm)* ","? "}"   // subject optional ⇒ matching fn
arm         ::= pattern ("if" expr)? "=>" expr

list        ::= "[" (expr ("," expr)*)? "]"
tuple       ::= "(" expr ("," expr)+ ")"
record      ::= "{" (".." expr ",")? (field_init ("," field_init)*)? "}"
field_init  ::= ident (":" expr)?                      // { x } shorthand for { x: x }
```

### 3.1 Statement termination
A statement ends at a newline unless the line ends with an operator, an open bracket, or `|>` — in which
case the statement continues. This lets pipelines wrap:
```
xs
  |> map(f)        // continues because previous line is mid-`|>` chain
  |> sum
```

### 3.4 Static rule: no top-level `mut`
The parser/checker rejects `mut` on any binding whose enclosing scope is the program (module) scope. The
*only* way to introduce mutable top-level state is to bind (immutably) an atom: `let counter = atom(0)`.

---

## 4. Semantics

### 4.1 Pipe & UFCS
- `x |> f(a, b)` desugars to `f(x, a, b)`.
- `x |> f` (no parens) desugars to `f(x)`.
- A `_` hole in the call positions the piped value: `x |> g(a, _)` ⇒ `g(a, x)`.
- `x.f(a)` desugars to `f(x, a)` (uniform function call syntax) **iff** no record field named `f` shadows it;
  field access `x.f` wins when `f` is a field of the record. Method-vs-field is resolved at compile time when
  the type is known (gradual), else at runtime.
- Consequence: every registered Rust function is automatically usable as `x.f(..)`, `x |> f(..)`, and `f(x,..)`.

### 4.2 Subjectless match = matching function
`match { p1 => e1, p2 => e2 }` (no subject) compiles to an anonymous one-arg function
`__a => match __a { p1 => e1, p2 => e2 }`. This is why `x |> match { … }` works with no special rule.

### 4.3 Binding & scope
- `let p = e` binds immutably; re-`let` of the same name **shadows** (new slot), it does not mutate.
- `let mut x = e` (local only) creates a mutable slot; `x = …`, `x += …` update it. Captured-by-closure
  `let mut` is allowed and shares the slot (closures see updates).
- Blocks are expressions; the trailing expression is the value (or `()` if none).

### 4.4 Atoms (the only escaping mutability)
```
atom(v)            -> Atom            // construct
@a        / deref(a) -> Value          // read current value
swap!(a, f)        -> Value           // a := f(@a); returns new value; atomic (CAS retry)
reset!(a, v)       -> Value           // a := v; returns v
a |> swap!(_, f)                       // pipes like anything else
```
- `swap!` must be pure in `f` modulo the atom; on contention it re-reads and re-applies (compare-and-set loop).
- Atoms may carry **watchers**: `watch(a, key, fn(old, new))`; fired after a successful swap/reset. The host
  editor uses this to re-render on state change.
- The set of all live atoms is the program's mutable root set — see §7 externalization.

### 4.5 Errors
- `Result a e = Ok(a) | Err(e)`; `Option a = Some(a) | None` are prelude variants.
- Postfix `?`: `e?` evaluates `e`; if `Ok(v)`/`Some(v)` it yields `v`; if `Err(_)`/`None` it returns that
  from the *enclosing function* immediately. Legal only in a fn whose return type unifies with the carrier.
- No exceptions in user code. Host/native functions may raise a VM **fault** (e.g. divide-by-zero, type
  error) which unwinds to the host as `Err`-like `Fault`; it is not catchable in-script in v0.1.

---

## 5. Pattern matching

Patterns:
```
_                       wildcard
ident                   bind
literal                 int/float/str/bool/unit, matched by ==
typename(p, …)          tuple-style variant: Some(x)
typename { f: p, …}     record-style variant / record: Circle { radius }
typename { f, .. }      shorthand bind f; `..` ignores remaining fields
(p, …)                  tuple
[p, …]                  list, exact length
[p, …, ..rest]          list with rest-binding (rest is a List)
a .. b   /  a ..= b     range pattern (numeric/char)
p1 | p2                 or-pattern (both sides must bind the same names)
p as ident              bind whole matched value to ident
```
Semantics:
- Arms tried top-to-bottom; first matching arm (whose guard, if any, is `true`) wins.
- Bindings from the chosen pattern are in scope in the guard and the arm body.
- **Exhaustiveness (gradual):** when the scrutinee's type is statically a known closed variant/record set,
  the checker requires the arms to cover it (or include `_`); a gap is a compile/reload error. With unknown
  (dynamic) type, a runtime `no-match` fault is raised if nothing matches.

---

## 6. Bytecode & VM

### 6.1 Why stack-based + reified
The host call stack is **never** used for script frames. All script execution state lives in a heap
`VmState`. That is what makes "stop between any two instructions, inspect, resume" possible — the editor
calls `step()` as many times as it wants and owns the state in between.

### 6.2 VmState
```rust
struct VmState {
    code:   Arc<FnTable>,        // name-keyed, hot-swappable (§8)
    frames: Vec<Frame>,          // explicit call stack
    stack:  Vec<Value>,          // operand stack
    status: Status,              // Running | Paused | Done(Value) | Fault(Fault)
    fuel:   Option<u64>,         // optional per-instruction quota
}
struct Frame {
    func:   FnId,                // index into FnTable (by name → id)
    ip:     usize,               // instruction pointer within func
    base:   usize,               // operand-stack base for this frame's locals
    locals: Vec<Value>,          // mutable slots (let mut)
}
```
`VmState` is plain data → cloneable for snapshots, serializable for save/restore.

### 6.3 Instruction set (initial, stack-based)
```
// constants & locals
CONST k            push constants[k]
UNIT               push ()
TRUE / FALSE
LOAD n             push frame.locals[n]
STORE n            frame.locals[n] = pop()         // let mut
GETUP n            push captured upvalue n
// data
LIST  c            pop c values → List, push
TUPLE c            pop c values → Tuple, push
RECORD c keys      pop c values → Record with keys, push
GETFIELD key       push (pop).key
VARIANT tag c      pop c → Variant(tag, payload), push
// operators (could also be CALL to prelude; inline for speed)
ADD SUB MUL DIV MOD POW
EQ NE LT LE GT GE
NEG NOT
// control
JUMP off
JUMP_IF_FALSE off
// pattern matching support
MATCH_TAG tag off  if top is Variant(tag,..) fall through else JUMP off  (non-consuming)
DESTRUCT …         push bound sub-values per a pattern descriptor
// functions
CLOSURE fnid ups   build closure capturing `ups` upvalues, push
CALL argc          call top-(argc+1); pushes Frame (trampolined, NOT host recursion)
TAILCALL argc      reuse current frame
RET                pop frame, push result
// atoms
ATOM               push atom(pop())
DEREF              push @(pop())
SWAP               a=pop2; f=pop1; CAS-loop f over a; push new
RESET              a,v=pop2; set; push v
// errors
TRY off            if top is Err/None → unwind to RET with it, else unwrap   (impl of `?`)
// host
NATIVE id argc     call registered Rust fn id with argc args
// misc
POP
HALT
```
`CALL` pushes a `Frame` onto `vm.frames` and continues the same `step()` loop — there is no Rust recursion,
so suspension is possible at every instruction including across calls.

### 6.4 The step loop & control API
```rust
impl Run {
    fn step(&mut self) -> Step;              // execute exactly one instruction
    fn run_until(&mut self, stop: StopWhen) -> Step;  // run, checking stop each instruction
    fn state(&self) -> &VmState;             // inspect frames/locals/stack while paused
    fn snapshot(&self) -> VmState;           // clone for time-travel
    fn stop(&mut self);                      // request halt
}
enum Step { Paused(Cause), Done(Value), Fault(Fault) }
enum Cause { Breakpoint(SrcSpan), Step, FuelExhausted, HostStop }
enum StopWhen { Never, NextLine, Breakpoints(Set<SrcSpan>), Fuel(u64) }
```
- **Pause/cancel** = the editor stops calling `step()`, or calls `stop()`. Because state is owned data, there
  is nothing to "interrupt" — it's already at an instruction boundary.
- **Step / step-over / step-into** = `run_until(NextLine)` with frame-depth tracking via a line table.
- **Breakpoints** = `run_until(Breakpoints(..))`; a `SrcSpan` line table maps `(FnId, ip) → source span`.
- **Fuel** = decrement `fuel` per instruction → deterministic limits (precise, unlike rhai's sampled
  `num_operations`).

---

## 7. State externalization (atoms ⇄ host)

Because §4.4 guarantees atoms are the *only* escaping mutable state, the runtime holds the complete root set:
```rust
vm.capture_atoms() -> AtomSnapshot      // serialize every live atom (+ watchers metadata)
vm.restore_atoms(snapshot)              // rehydrate; identities re-established by stable AtomId
```
Use cases: persist on editor close, restore on reopen; carry state across a hot reload; checkpoint a session.
Serialization format: stable `AtomId` → serialized `Value`. `Native` values need a host-provided
serializer or are marked non-persistable (then capture errors loudly — never silently drops, per project
rule on stubs).

---

## 8. Hot reload

Model:
1. **Compilation unit = top-level item.** Each `fn`/`type`/`let` compiles independently to a `FnTable` entry
   keyed by name.
2. **Calls resolve through the table by name → FnId.** Swapping a name's bytecode updates all live callers
   and closures immediately (closures hold `FnId`, not raw code pointers).
3. **Atomic swap at an instruction boundary.** Since every boundary is a safe point (§6), the reload thread
   swaps `vm.code = Arc::new(new_table)` between two `step()`s. No call is ever mid-instruction.
4. **State survives** because state is atoms (§7), which are independent of the code table.
5. **Incremental.** Only changed items recompile; in gradual mode, unannotated items reload with no re-check.

Open semantics to decide:
- In-flight continuation sitting in old `foo` when `foo` is replaced → finish on old code, next call uses new
  (proposed default).
- A `type`'s field set changes → run an optional `migrate(old) -> new` hook over existing instances, else
  leave old instances as-is.

---

## 9. Rust interop / embedding API

Mirrors rhai's ergonomics; UFCS removes the method/getter/operator boilerplate.
```rust
let mut vm = Funct::new();

vm.register("double", |x: i64| x * 2);                 // auto type-convert
vm.register("read_file", |p: String| -> Result<String, Error> { … });   // Result ⇒ script Result

#[derive(Funct, Clone)]
struct Player { name: String, hp: i32 }
impl Player { fn damage(&mut self, n: i32) { self.hp -= n; } }
vm.register_type::<Player>();                          // fields ⇒ getters/setters, methods ⇒ UFCS fns

#[funct::module] mod math { pub fn lerp(a: f64, b: f64, t: f64) -> f64 { a + (b-a)*t } }
vm.register_module("math", math::module());

let dmg: i64 = vm.call("compute_damage", (weapon, target))?;   // Rust → script
```
Conversions handled by a `FromValue`/`ToValue` pair (the `Dynamic`-equivalent is `Value`). A `derive(Funct)`
generates `FromValue`/`ToValue`, field accessors, and UFCS method registrations.

---

## 10. Implementation roadmap

Milestones, each independently runnable:

- **M0 — Skeleton.** `Value` enum, `VmState`, the `step()` loop, a hand-assembled bytecode program runs and
  returns a value. No parser yet. Proves the reified-VM core + suspend/resume.
- **M1 — Front end.** Lexer + parser → AST for the §3 grammar; AST → bytecode compiler. Literals, `let`,
  `fn`, calls, arithmetic, `if`, blocks. `funct run file.ft` works.
- **M2 — Functional core.** Closures/upvalues, `|>` + UFCS desugar, lists/tuples/records, `match` (incl.
  subjectless), `Result`/`Option` + `?`.
- **M3 — State.** Atoms, `swap!`/`reset!`/`@`, watchers, `capture_atoms`/`restore_atoms`.
- **M4 — Control surface.** Breakpoints, step/step-over/step-into, line table, fuel; a tiny DAP-ish adapter
  for the editor plugin.
- **M5 — Hot reload.** Name-keyed table swap at boundaries; incremental recompile; reload-while-paused.
- **M6 — Interop.** `Value` conversions, `register`/`register_type`/`register_module`, `derive(Funct)`,
  `vm.call`.
- **M7 — Gradual types.** Optional annotations, `match` exhaustiveness check, editor hints.

Recommended first cut to validate the thesis: **M0 → M3 → M4**, because pause/step/inspect over real mutable
state is the differentiator. Parser polish and types can trail.

---

## 11. Things still undecided (track here)

- Final name (10 candidates in the sketch; my lean: Sluice or Flume).
- Atom deref syntax: `@a` vs `a.value` vs `a |> deref`.
- `!`-suffix naming convention for mutating fns (`swap!`) — keep or drop.
- Char type in v0.1 or defer.
- Whether `for` loops exist or everything is `map`/`fold`/recursion.
- Module/import resolution model (file-based like rhai's `FileModuleResolver`?).
- Standalone language vs. the embedding layer for **Beagle** (this design fits Beagle's pause/resume goals).
