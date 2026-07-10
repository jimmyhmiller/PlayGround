# 06 — Implementation (living doc)

Phase 1 is the **front end**: lexer, AST, recursive-descent parser, and the
`scry parse-dump` CLI. **Phase 2 is the static typechecker** (`src/types.coil`,
`src/typecheck.coil`, `scry check`), a nominal two-pass checker over the Phase 1
AST. Everything is written in Coil (`run coil guide`). No bytecode or VM yet —
that is Phase 3. This doc records how each layer is built so later phases can
stack on it without re-deriving it. (Phase 2 additions are at the bottom, under
"Phase 2 — the typechecker".)

## Build & run

```
coil build                     # project (Coil.toml) -> ./scry
./scry parse-dump f.scry       # parse and print the AST (stable golden format)
./scry check f.scry            # typecheck (follows imports); prints "ok" or diagnostics, exit 65
./scry dump-types f.scry       # debug: print the mono type table + itables (Phase 3 input)
./scry run f.scry              # HARD ERROR: "not implemented until Phase 3", exit 2
python3 tests/run-tests.py     # golden suite (tests/parse + tests/check); nonzero on any failure
```

`coil build` writes the binary named after the package (`scry`). It is gitignored.

## Module layout (`src/`)

| File | Module | Responsibility |
|---|---|---|
| `ioutil.coil` | `ioutil` | **All** libc externs (write, snprintf, strtod, fopen…, strcmp) + `w-cstr/w-int/w-num/w-bytes/xalloc`. Declared **once** here — the Coil guide warns externs are not deduped across modules — and `:use *`-imported everywhere else. |
| `lexer.coil` | `lexer` | Token kinds (`TK_*`), `Token` struct, the scanner. Produces a flat `(ArrayList Token)`. Owns string-interpolation lexing (below). |
| `ast.coil` | `ast` | The `Node` record, node kinds (`NK_*`), constructors, and the `parse-dump` pretty printer. |
| `parser.coil` | `parser` | Recursive-descent parser over the token vector → a `NK_PROGRAM` root. Fail-fast diagnostics. |
| `main.coil` | `app` | The `scry` CLI entry: reads the file, drives lex → parse → dump. |

**Import-path gotcha (Coil):** imports resolve relative to *different* bases for
the entry vs. non-entry files. `main.coil` (the Coil.toml entry) resolves relative
to the **CWD** (project root), so it imports `"src/lexer.coil"`. Every other file
resolves relative to **its own directory**, so they import siblings bare
(`"lexer.coil"`). Keep this in mind when adding files.

## Tokens (`lexer.coil`)

Every token carries `kind`, `start` (ptr into source, or into a decoded buffer for
string chunks), `len`, and **`line`/`col` spans** (1-based; columns are byte
columns). Integer/float tokens also carry a decoded `ival`/`fval`.

Kinds: `EOF IDENT INT FLOAT`; string parts `STR_BEGIN STR_CHUNK INTERP_BEGIN
INTERP_END STR_END`; keywords `class interface implements enum object fn let var
if else while for in match return migrate import module self true false as`;
punctuation `( ) { } [ ] -> => :: : , ; . ?  <  <=  >  >=  ==  !=  =  + - * / %
&& || !`; and `ERROR`.

Notes:
- `<`/`>` are always single tokens (there is **no** `<<`/`>>`), so nested generics
  `Inventory<List<Tool>>` lex as four angle tokens — no `>>`-splitting needed.
- Comments: `//` line and `/* */` block (block comments are not in `01-language.md`
  but are a standard, harmless extension; noted here).
- **String interpolation is "lexed as parts."** A string becomes the token run
  `STR_BEGIN (STR_CHUNK | INTERP_BEGIN <expr tokens…> INTERP_END)* STR_END`. The
  lexer keeps a small brace-depth stack (`interp`) so `}` inside an interpolation's
  own block vs. the `}` that closes the interpolation are told apart, and nested
  strings inside `${…}` work (`"a ${ f("b ${x}") }"`).
- Escapes decoded in `STR_CHUNK`: `\n \t \\ \" \xHH` (per `01-language.md` §3.1).
  Any other escape is a hard error.

## AST (`ast.coil`)

The AST is a **single wide record** `Node` with a `kind` tag rather than a family
of sum types — one code path for construction and for the dumper, and trivially
extensible for Phase 2. Slots:

```
kind line col          ; tag + span (every node has a span)
s-name s-len           ; a name/text payload (ptr+len; may point into source or a decoded buffer)
ival fval              ; int literal / bool flag / operator token-kind / unit flag ; float literal
a b c d                ; up to four fixed child pointers
kids                   ; ArrayList of child pointers (variable-arity children)
```

Each node self-identifies by its dump head, so optional/absent children are simply
omitted (no positional placeholders). Per-kind slot conventions:

| Kind | s-name | ival/fval | a / b / c / d | kids |
|---|---|---|---|---|
| `PROGRAM` | | | | top-level items |
| `MODULE` | dotted name | | | |
| `IMPORT` | dotted path | | a=alias ident? | imported name idents (brace form) |
| `CLASS` | name | | a=typeparams? b=implements? | members (FIELD/FN) |
| `INTERFACE` | name | | a=typeparams? | method sigs (FN, no body) |
| `ENUM` | name | | a=typeparams? | VARIANTs |
| `OBJECT` | name | | | members |
| `FN` | name | | a=typeparams? b=params c=rettype? d=body-block? | |
| `MIGRATE` | type name | | a=from-ident b=to-ident | MIGRATE_ENTRYs |
| `MIGRATE_ENTRY` | field | | a=param-ident? (`(old)=>`) b=value | |
| `FIELD` | name | | a=type | |
| `VARIANT` | name | | | payload type nodes |
| `PARAM` | name | | a=type | |
| `PARAMS`/`TYPEPARAMS`/`IMPLEMENTS` | | | | children |
| `TYPEPARAM` | name | | a=bound-type? | |
| `TYPE` | name | ival=1 → unit `()` | | type args (TYPE) |
| `BLOCK` | | | | statements |
| `LET` | name | ival: 0=let,1=var | a=decl-type? b=init | |
| `ASSIGN` | | | a=target b=value | |
| `EXPRSTMT` | | | a=expr | |
| `IF` | | | a=cond b=then-block c=else? | |
| `WHILE` | | | a=cond b=body | |
| `FOR` | loop var | | a=iterable b=body | |
| `RETURN` | | | a=expr? | |
| `INT`/`FLOAT`/`BOOL` | | ival or fval | | |
| `IDENT` | name | | | |
| `SELF` / `UNIT` | | | | |
| `STRING` | | | | parts (STRCHUNK or expr) |
| `STRCHUNK` | decoded bytes | | | |
| `BINARY`/`UNARY` | | ival=operator token-kind | a=lhs (b=rhs) | |
| `CALL` | | | a=callee | ARG nodes |
| `ARG` | name? (named arg) | | a=value | |
| `INDEX` | | | a=obj b=index | |
| `FIELDACC` | field name | | a=receiver | |
| `TERNARY` | | | a=cond b=then c=else | |
| `TRY` | | | a=expr (postfix `?`) | |
| `MATCH` | | | a=scrutinee | ARM nodes |
| `ARM` | | | a=pattern b=body(expr/block) | |
| `PAT_ENUM` | variant | | | binding idents (`_` allowed) |
| `PAT_LIT` | | | a=literal expr | |
| `PAT_WILD` | | | | |

Method calls / enum-qualified values are **not** special nodes: `a.b(c)` is
`CALL(callee=FIELDACC(b, a), …)` and `ToolError.Failed(x)` is
`CALL(callee=FIELDACC(Failed, ident ToolError), …)`. Generic construction
`List<Tool>()` is `CALL(callee=TYPE List<Tool>, …)` — the callee is a `TYPE` node.

### Dump format
`parse-dump` prints a nested, indented s-expression (2 spaces/level). Leaves and
`TYPE`/pattern nodes render inline; compound nodes go multi-line. Spans are **not**
printed (keeps goldens stable). The format is the golden-test contract — if you
change it, re-bless (`python3 tests/run-tests.py --bless`) and eyeball the diff.

## Parser (`parser.coil`)

Recursive descent. Precedence (loosest→tightest): ternary `?:` → `||` → `&&` →
`== !=` → `< <= > >=` → `+ -` → `* / %` → unary `! -` → postfix (`.` call `[]`
`?`) → primary.

Decisions worth knowing:
- **Generic vs. less-than (`<`)** is resolved the standard way: in expression
  position an identifier followed by `<` is a generic construction **only** when a
  speculative scan finds a balanced `<…>` (of type-ish tokens) whose closing `>` is
  immediately followed by `(`. Otherwise `<` is comparison. (`scan-is-generic-ctor`.)
- **`?` propagation vs. ternary** is resolved by adjacency: a `?` with no
  whitespace before it (`Int.parse(s)?`) is Result-propagation (`TRY`); a `?` with
  a space before it inside a conditional (`cond ? a : b`) is the ternary. This
  supports both `01-language.md` §1.5 and the §1.8 migrate example.
- **Lambdas/closures are a hard parse error** everywhere except the migrate-entry
  `(old) => expr` derived form, which is recognized by a 4-token lookahead before
  general expression parsing ever sees `=>`.
- **Error handling is fail-fast:** the first lex/parse error prints
  `file:line:col: error: <msg> (found "<token>")` to stderr and exits nonzero
  (parse errors exit 65, lex errors exit 1). No multi-error recovery — the task
  permits fail-fast; each error golden exercises one bad construct.

## Conventions (please keep)

- **Hard-error stubs, never silent ones** (owner's iron rule). `scry run` and any
  not-yet-implemented path must print a clear message and exit nonzero.
- **Spans on every node.** They are threaded from tokens into `Node.line/col`; do
  not drop them — Phase 2 diagnostics and the viewer's ids depend on them.
- **`.coil` files are s-expressions.** After editing, sanity-check parens; note
  that `paredit-like balance` mis-parses Coil `c"…"` strings containing `\"`/`)`,
  so trust the Coil compiler over paredit for those lines.
- Coil gotchas hit during Phase 1: `and`/`or` are strictly binary (nest them); the
  clean `=` is i64/bool only (cast `i32` results from libc); `(if c (store! …) 0)`
  type-mismatches — wrap effectful branches in `(do … 0)`; a value param
  (`(ArrayList Token)`) is an immutable ref — `(let [l p] (field l …))` to read it.

## What Phase 2 (typechecker) builds on

- Walk the `NK_PROGRAM` root's `kids` for declarations. `Node.kind` is the
  discriminant; the table above is the schema. Add a `resolved-type`/symbol slot to
  `Node` (or a side table keyed by node identity) rather than mutating the shape.
- The nominal type system (`01-language.md` §2) needs: a class/interface/enum
  symbol table, `implements` conformance checking (every interface method present
  with an exact signature), definite-assignment on `init` (§1.1), exhaustive
  `match` (§1.3), and monomorphization identity for generic **classes** (§2.4).
  All the surface needed for these already parses and is in the AST.
- Interface method sigs are `FN` nodes with a null body slot (`d`); a body on one
  is already rejected at parse time.
- `migrate` blocks parse but Phase 2/PoC must hard-error on them
  (`01-language.md` §1.8: "migrate blocks are not implemented in this build").

## Known deviations from `01-language.md` (Phase 1)

- `01-language.md` uses `var` (not `mut`) for mutable locals; the parser implements
  `let`/`var` accordingly. (The original brief said "mut?"; the doc wins.)
- Block comments `/* */` are accepted though only `//` appears in the doc.
- No genuine syntax typos were found in §6 — both example files parse verbatim.

---

# Phase 2 — the typechecker

A nominal, fully-static checker over the Phase 1 AST. Two files:

| File | Module | Responsibility |
|---|---|---|
| `src/types.coil` | `types` | The resolved `Type` universe, the symbol/`Decl` table, the **monomorphization table**, built-in-type registration hooks, type resolution from AST `TYPE` nodes, nominal equality/assignability, and the diagnostics buffer. |
| `src/typecheck.coil` | `typecheck` | The two-pass driver, expression/statement/body checking, interface conformance, definite assignment, `match` exhaustiveness, `?`, the built-in-type definitions, module/import loading, and the `check`/`dump-types` entry points. |

**No AST shape change** (per Phase 1's guidance): the checker keeps its own side
tables keyed by `Decl`/`Node` identity rather than adding slots to `Node`. Types
are separate `Type` records; the AST is read-only input.

## Type representation (`Type`)

`Type` is a small tagged record: `kind` (one of `TY_INT/FLOAT/BOOL/STRING/VOID/
UNIT/NAMED/VAR/ERROR`), a `name`, a `decl` pointer (for `TY_NAMED`), and an
`args` list of child `Type`s (type arguments). `TY_VAR` is an unsubstituted
generic parameter; `TY_ERROR` propagates and suppresses cascade errors (assignable
to/from anything). Equality is **nominal**: same `decl` pointer + pairwise-equal
type args. Assignability adds exactly one subtyping rule — a `class`/`object` whose
`implements` list names interface `I` is assignable to `I` (`class-implements`,
resolved by name to the interface `Decl`). No numeric coercion, no `Void`/`()`
interchange.

## Symbol table (`Decl`)

One `Decl` per declaration (`class`/`interface`/`enum`/`object`/top-level `fn`,
plus built-ins), holding `kind`, name, the AST `node`, a `tparams` list (name
nodes), a `members` list (the `FIELD`/`FN`/`VARIANT` child nodes), and the source
`file` (for diagnostics). Unqualified enum-variant names (`Some`, `Ok`, `Circle`)
are indexed separately (`VariantRef`) so `Ok(x)` resolves without a receiver.

## Monomorphization table — what Phase 3 consumes (DECISIONS #6)

Every **distinct** concrete instantiation is interned once into `ctx.monos` as a
`MonoType` and assigned a stable integer `id` (its **type-id**), keyed by a
canonical mangled name: `Inventory<Tool>`, `Inventory<Widget>`, `List<Message>`,
`Map<String,String>`, `Result<String,ToolError>`, nested `Box<List<Int>>`, and
every plain non-generic class/enum/object too. Interning happens the moment a
concrete `TY_NAMED` is built (`mk-named` → `intern-mono`), so it fires from field
types, param/return types, local annotations, and construction sites uniformly —
built-in generics (`List`/`Map`/`Mutex`/`Option`/`Result`) go through the exact
same path as user generics. `MonoType` carries:

- `id` (type-id), canonical `name`, `base` `Decl`, resolved `args`;
- `is-entity` — true for `class`/`object` (gets a per-type arena in Phase 3),
  false for enums, `List`/`Map`/etc. (values, no arena);
- `fields` — for entities, the **substituted field layout in declaration order**
  (`MonoField` = name + resolved concrete `Type`). Cycle-safe: the entry is
  registered before its fields are populated, so a self- or mutually-referential
  field type finds the in-progress entry instead of recursing forever.

`./scry dump-types f.scry` prints this table plus the **itables** (per
`class × interface`, the interface's methods numbered in declaration order → the
implementing class's method) — i.e. exactly `02-runtime.md` §5's `ITable` slot
assignment, and the field layout `TypeInfo`/`ShapeInfo` wants. That dump is the
concrete hand-off to Phase 3 codegen: mono type-ids + ordered field layouts +
itable slots. Resolved per-node types are recomputed on demand (the checker is
side-effect-light); Phase 3 can re-run resolution or read this table.

## Diagnostics

Errors carry `file:line:col` (span threaded from the offending `Node`) + a
message, printed to stderr as `file:line:col: error: <msg>`. Type mismatches use
the doc's `expected <X>, found <Y>` shape (`00-vision.md`). Recovery is
**per-declaration**: a failed declaration doesn't stop the others, and inside an
expression a type error yields `TY_ERROR` to suppress cascades, so one run reports
many independent errors. Any error ⇒ exit 65.

## Two passes + module loading

1. **Collect** — register every declaration (dup-name check; `migrate` →
   hard error; a generic **bound** like `<T: Comparable>` → hard error, matching
   `05-milestones.md`'s "what we fake" #3).
2. **Signatures + conformance** (all loaded modules): resolve field types (interns
   monos, reports unknown types), intern each non-generic class/enum/object,
   verify `implements` conformance (every interface method present with an
   exactly-matching signature — each missing/mismatched method reported with its
   span; `implements` of a non-interface rejected; interface method bodies
   rejected).
3. **Bodies** (entry module only, see below): definite assignment on `init`,
   statement/expression checking, `self` typing, return-path checking, `match`
   exhaustiveness, `?`, generic-function type-argument inference.

**Imports / multi-file (`scry check` follows imports).** `import a.b.{X,Y}` (and
`import a.b.X as Z`) resolves the module `a.b` to a file **relative to the
importing file's directory**, trying `a/b.scry`, then `a-b.scry`, then the last
segment `b.scry`. Found modules are parsed + collected (recursively, dedup by
path); each imported name is validated to exist. Names that resolve to a built-in
(e.g. `std.collections.{List}`) are accepted as no-ops even with no file. This is a
**separate-compilation boundary**: the entry file's bodies are deep-checked, but
imported modules contribute only their *declarations* (fields resolved + interface
conformance checked) — their method **bodies** are not walked. So
`scry check examples/main.scry` pulls `Agent`/`Tool`/… signatures from
`agents.core` (and `Tui` from a small `examples/ui-tui.scry` created for the
demo's `ui.tui` import) and typechecks green, without depending on symbols that
only appear inside `agents.core`'s method bodies.

## Built-ins

`Int/Float/Bool/String/Void/()` are primitive kinds. `List<T>`(push/get/len,
iterable), `Map<K,V>`(set/get→`Option<V>`/containsKey/len), `Mutex<T>`(lock/unlock/
get/set), `Option<T>`, `Result<T,E>`, `Runnable`(interface), `Thread`(spawn→
`ThreadHandle`), `ThreadHandle`(join), and `Clock`/`Console` are registered as
synthetic `Decl`s with synthetic `FN`/`VARIANT` member nodes, so method/variant
resolution is uniform with user types. `print(x)` is a built-in free function.
`String` methods (`len`, `slice`) are special-cased. `List<Int>()`/`Map<K,V>()`
take no ctor args; `Mutex<T>(v)` is positional; user classes construct by **named**
args (all fields, definite assignment).

## Doc deviations / gaps found in Phase 2

- **§6's `agents/core.scry` is not self-contained.** Its method bodies reference
  `Llm.complete(...)`, a return type `AgentError`, and `__builtin_*` intrinsics
  that are declared **nowhere** in the doc set (they are M4/M5 placeholders). So
  `scry check` on `agents-core.scry` *by itself* reports those as unknown — that
  is correct, and it is why the separate-compilation boundary above matters: the
  canonical gate is `scry check examples/main.scry`, which only needs
  `agents.core`'s public signatures (none of which mention `Llm`/`AgentError`) and
  passes clean.
- **`Clock`/`Console` are both "built-ins" (task) and user-defined (§6).** Resolved
  by letting a user declaration of a built-in-named `object` **override** the
  built-in (no duplicate error), so programs that define `Clock` (the demo) and
  programs that just use it both work.
- **`ui.tui` / `std.collections` are referenced by `main.scry` but never defined
  in the doc.** `std.collections.{List}` is satisfied by the built-in `List`; a
  minimal `examples/ui-tui.scry` (`object Tui { fn render(agents: List<Agent>) }`)
  was added so the canonical multi-file program resolves — noted here rather than
  silently faking `Tui`.
- **Generic *functions*** are supported (minimal, per §2.4 "erased/boxed"): type
  args are inferred from argument types at the call site (no explicit turbofish).
  Generic **bounds** are rejected (fake #3). Definite-assignment is the pragmatic
  "every field assigned on some path" (not full per-path "exactly once") — noted
  as a deliberate simplification; the demo's straight-line `init`s satisfy it.

## Coil friction hit in Phase 2 (add to Phase 1's list)

- `nl` name-collides with `ast.coil`'s newline helper across `:use *` — renamed the
  node-length accessor to `nlen`. Watch for helper-name clashes when two modules
  are both `:use *`-imported.
- `cond`/`if` branches must have the **same type**; an effectful branch like
  `(store! x v)` returns the stored value's type, not `i64` — wrap in `(do … 0)`.
  A `cond` treats a trailing `true`-guarded clause as a normal arm and appends an
  implicit `0` else, mismatching a `(ptr …)` result — make the last clause the bare
  default instead.
- `al-push!` needs a **mutable place** (`(mut x)` / `(mut (field p f))`); a
  let-bound `ArrayList` must be bound `(mut xs)` and read back with `(load xs)`. To
  share/grow a list through a pointer, wrap it in a one-field struct and push via
  `(mut (field p items))` (used for `PList`); a plain byte string-builder is easier
  as a hand-rolled `StrBuf` than an `(ArrayList u8)`.

---

# Phase 3 — bytecode compiler, VM, and per-type arenas (M0)

Phase 3 turns the type-checked AST into a running program: a bytecode compiler
(`src/compile.coil`), a stack VM (`src/vm.coil`), the per-type slab allocator
(`src/arena.coil`), the instruction set + program model (`src/bytecode.coil`), and
the built-in value runtime (`src/builtins.coil`). `02-runtime.md` is the spec; where
M0 deviates it is because M0 is **single-threaded by mandate** (05 M0 OUT) — the
thread-safety machinery (magazines, atomics, safepoints) is M1.

## Module layout (`src/`, added this phase)

| File | Module | Responsibility |
|---|---|---|
| `bytecode.coil` | `bytecode` | Opcode + builtin-id + field-kind constants; `Chunk` (growable code buffer + i64 constant pool); the program model (`MethodInfo`, `FieldInfo`, `ITable`, `TypeInfo`, `Program`); the disassembler (`dump-bytecode`). |
| `arena.coil` | `arena` | `InstanceHeader`, `TypeArena`, slab bump-allocation + free-list, `arena-for-each-live` enumeration, `OutOfArenaSpace`. Leaf (imports only `ioutil`). |
| `builtins.coil` | `builtins` | `StringObj`/`ListObj`/`MapObj`, enum boxes (Option/Result + every user enum), `print`/`Console.log`, `Clock`. Leaf; kept behind functions so Phase 5 magazines/ropes can replace them. |
| `vm.coil` | `vm` | `VM`/`CallFrame`, the value stack, the dispatch loop (`run`), `vm-run` (allocate object singletons, invoke `main`). |
| `compile.coil` | `compile` | AST→bytecode; builds `TypeInfo`+arenas+itables; the `run`/`dump-bytecode` CLI drivers. |

Import DAG (no cycles): `arena`←`bytecode`; `{arena,builtins,bytecode}`←`vm`;
`{...,typecheck,types,vm}`←`compile`←`main`. clox uses one file to dodge its
value↔object↔table cycles; Scry's layers are acyclic, so it stays split.

## Values: untagged slots, typed opcodes (§1)

Every stack slot and every field is a raw 8-byte `i64`. The compiler knows each
value's static type, so it picks a **typed opcode** and the VM never tag-checks:
`Int`→`i64`, `Float`→the `f64` bit pattern (round-tripped through memory, never
`(cast f64 i64)`), `Bool`→0/1, `String`/class/`List`/`Map`/enum→a raw pointer. There
is no NaN-boxing and no runtime `Value` union.

## Opcode inventory (`bytecode.coil`)

Operands: `u16` big-endian for const-idx / local-slot / field-offset / jump /
type-id / iface-id / method-idx; `u8` for argc / enum-tag / builtin-id / enum-arg-idx.

- **Literals/stack:** `CONST`, `TRUE`, `FALSE`, `UNIT`, `POP`, `DUP`.
- **Locals:** `LOAD_LOCAL`, `STORE_LOCAL` (slot = offset from frame base).
- **Fields (typed):** `LOAD_FIELD_{I64,F64,REF}`, `STORE_FIELD_{I64,F64,REF}` — operand
  is a **compile-time byte offset** that already includes the header (a pointer-add +
  a move; no hashing, unlike clox's field `Table`). The three families are distinct
  for GC/metadata intent; in M0 all three are the same 8-byte move.
- **Arithmetic:** `{ADD,SUB,MUL,DIV,NEG}_{I64,F64}`, `REM_I64`.
- **Comparison:** `{LT,LE,GT,GE,EQ,NE}_{I64,F64}`, `{EQ,NE}_BOOL`, `NOT`,
  `{EQ,NE}_REF` (pointer identity), `{EQ,NE}_ENUM` (structural), `STR_{EQ,NE}`.
- **Strings:** `CONCAT`, `{I64,F64,BOOL}_TO_STR` (interpolation coercions).
- **Control:** `JUMP`, `JUMP_IF_FALSE` (pops), `LOOP` (backward).
- **Calls:** `CALL_STATIC <method-idx> <argc>`, `CALL_VIRTUAL <iface-id> <slot> <argc>`,
  `CALL_BUILTIN <builtin-id>` (fixed arity), `RETURN`.
- **Objects/enums:** `NEW <type-id>`, `LOAD_SINGLETON <type-id>` (objects),
  `MAKE_ENUM <tag> <argc>`, `ENUM_TAG` (pop box→tag), `ENUM_GET <idx>` (pop box→payload).

Built-in ids (`CALL_BUILTIN`): `PRINT_{I64,F64,BOOL,STR}`, `CONSOLE_LOG`,
`LIST_{NEW,PUSH,GET,LEN}`, `MAP_{NEW,SET,GET,HAS,LEN}`, `STR_{LEN,SLICE}`,
`CLOCK_{NOW,SLEEP}`, `THREAD_{SPAWN,JOIN}` (hard-error until Phase 5).

## Chunk + function-table format (Phase 4 hook)

A `Chunk` is `{code:(ptr u8), consts:(ptr i64)}`, both malloc/realloc-grown. String/
float constants are materialized **at compile time** (a `StringObj*` / the `f64` bit
pattern) and stored directly in the pool. A `Program` holds a **flat method table**
(`methods[]`) plus `types[]` (indexed by mono type-id) and `entry` (index of `main`).
**Every call is through a method index, never a raw code pointer** — precisely so
Phase 4's live-swap can rebuild `methods[]`/`types[]` and atomically repoint without
touching call sites (03-live-semantics' function-table indirection).

## Calling convention

Stack VM à la clox. A call site pushes receiver-then-args; `CALL_STATIC` sets
`base = stack-top - argc`, reserves+zeroes `[base+argc, base+local_count)` (so
unwritten ref slots read null — the §2 zeroing invariant), and `stack-top =
base+local_count`. Slot 0 is `self` for methods (`OP_LOAD_LOCAL 0`); params follow;
locals/temporaries above. `RETURN` pops the result, sets `stack-top = base`, and pushes
the result onto the caller — the callee's args are consumed, replaced by one value.
Frames don't overlap because a nested call's base is the caller's live `stack-top`.
`local_count` = the method's peak slot high-water (params + locals + scoped bindings +
compiler temps); operand-stack depth is naturally covered because temporaries live above
`local_count` in the shared stack. `void` methods (and `init`) push `UNIT` before
`RETURN`; an unreachable trailing `RETURN` after a fully-returning body is dead code the
VM never executes.

Dispatch paths (§5): a call on a **concretely-typed** receiver (class/object) is
`CALL_STATIC` to the baked-in method index. A call on an **interface-typed** receiver is
`CALL_VIRTUAL iface-id slot argc`: load `type-id` from the receiver's header →
`types[type-id].itables[iface-id].slots[slot]` → method index → same push-frame path.
itable slots are interface-declaration-order; itables are built per (class, interface)
at compile time. Built-in methods (List/Map/String/Clock/Console) compile to
`CALL_BUILTIN`.

## Object layout + arenas (§3/§4)

`InstanceHeader` = `{type-id, slot-index, generation, flags}` (M0 uses four `i64`
fields — spec's `u32` packing is deferred; correctness-equivalent). Fields follow in
declaration order at fixed 8-byte offsets; `field-offset = sizeof(InstanceHeader) +
field-index*8`. One `TypeArena` per **entity** mono type-id (`class`/`object`, incl.
each generic instantiation — `Inventory<Tool>` and `Inventory<Message>` get separate
arenas). Enums, `List`, `Map`, `Option`, `Result` are values (no arena). Arena params:
`slot-size = align8(header + fields)`; `slab-cap = max(8, 65536/slot-size)` (≈64 KiB
slabs); slabs `calloc`'d (fresh slots read `flags=0`); a flat `bump-cursor` frontier +
a `free-head` free-list (list present but unused until GC — no frees happen in M0).
Stable identity = `(type-id, slot-index, generation)`. `OP_NEW` calls `arena-alloc`,
which stamps the header. Exceeding `max-slots` (100000) is the loud
`OutOfArenaSpace: <Type> arena full at N slots ... GC not implemented in this build`.

### M0 deviations from §4 (all because M0 is single-threaded)
- **No per-thread magazines, no atomics.** The shared-arena API (bump + free-list) is
  the layer M1's magazines sit on; its shape is here, without the atomics. §4's
  `arena-free-push`/`arena-bump-batch` collapse to plain bump/free.
- **slot→(slab,local) is div/mod**, not shift/mask (slab-cap needn't be 2^k).
- **`shape-id` is not in the header yet** (Phase 4 migration adds it).

## Enums, match, `?`, strings

Enum values are heap boxes `[tag, argc, arg0…]` (`builtins.coil`); `MAKE_ENUM` builds
one, `ENUM_TAG`/`ENUM_GET` read it. Variant tag = declaration order (Option: None=0,
Some=1; Result: Ok=0, Err=1). `match` stores the scrutinee in a temp local, then per
arm compares the tag (or a literal for primitive match) and binds payloads into fresh
locals — a clean design that avoids stack juggling; the exhaustiveness the checker
proved means the fall-through past the last arm is unreachable. `?` stores the `Result`
in a temp, `RETURN`s the box early on `Err` (representation-identical across `Result<_,E>`
so no rebuild), else extracts the `Ok` payload. `String` is `{len, data}`; concat/slice
malloc; interpolation coerces each non-string part with `*_TO_STR` then `CONCAT`s.

## `Clock.sleep` note (revisit in Phase 5)

`Clock.sleep(ms)` is a real `nanosleep`. In M0 (single-threaded, no safepoints) that is
fine. In Phase 5 a sleeping thread must poll its safepoint between short slices so it
never delays a stop-the-world by more than one slice (01-language.md §1.7); this
implementation must change then.

## What Phase 4 (eval server) hooks into

- **Compile-in-context + call.** `build-program` produces the `Program`; a live eval
  compiles an expression/definition against the *same* `ctx` (types/monos) and appends
  to `Program.methods`, then invokes it via the existing `push-frame`+`run` path — no
  new call machinery. Because calls go through method **indices**, a new generation can
  swap `methods[]`/`types[]` atomically (03-live-semantics) without rewriting call sites.
- **Safepoint hook.** M1 will insert `safepoint-poll` at the **top of `run`'s dispatch
  loop** (one branch, almost always false) — the single place every instruction passes
  through. The loop is already parameter-shaped to take a `VMThread` (M0 uses one
  implicit global `VM`); M1 threads each pthread's own frames/stack through `run`.
- **Enumeration.** `arena-for-each-live(arena, fn)` is the viewer's foundation:
  `dump-arenas-of` already walks every entity `TypeInfo`'s arena for `live-count` /
  high-water / slab-count. Handle resolution will be `arena-slot-ptr(type-id, slot-index)`
  + a `generation` compare (both fields already stamped in the header).

## Coil friction hit in Phase 3 (adds to Phase 1/2's list)

- **`field-index` is a reserved builtin form** ("expects `(field-index TYPE name)`") —
  a plain `defn` of that name shadows it and errors; renamed to `fld-index`. (Same class
  of clash as Phase 1's `call`/`block`.)
- **`and`/`or` are strictly binary** (already known) — an 8-way `(or …)` over node kinds
  had to become a `cond`-based `stmt-like?` helper.
- **`(field (load place) f)` vs `(field place f)`** — to truncate a let-bound
  `ArrayList`'s `len`, the place is `(field cs locals)` (a `(ptr ArrayList)`), **not**
  `(load (field cs locals))` (the ArrayList value); `field`/`index` need the pointer.
- **A let-bound growable `ArrayList` must be `(mut x) (al-new …)`** and pushed via
  `(al-push! (mut x) …)` / read via `(load x)`; you cannot `(mut y) x`-alias an
  immutable let binding of it.
- **`case` arms take exactly one expression** — the giant VM dispatch `case` needs each
  opcode's whole handler wrapped in one `(let …)`/`(do …)`.

---

# Phase 4 — the eval server + the browser viewer (M2 + M3)

Phase 4 is the reason the project exists: the running program embeds a server whose
**only** wire operation is `eval` (`04-viewer.md` §4, `DECISIONS.md` #8). Every viewer
pane is sugar over `POST /eval {id, source} → {id, value|error}`; refresh is re-eval on an
interval/focus/after-action; nothing is ever pushed. This is a REPL into the live process,
not a message feed.

## Module layout (`src/`, added this phase)

| File | Module | Responsibility |
|---|---|---|
| `safepoint.coil` | `safepoint` | The one poll hook (`safepoint-poll`, called at the top of the VM dispatch loop) + a `stop-flag` + a registered drain callback. A leaf so `vm` can call it without importing `server` (which imports `vm`). Phase 5's STW protocol generalizes exactly here. |
| `evalrt.coil` | `evalrt` | Eval-time error handling: an `eval-active` flag, a `setjmp` landing pad + `eval-panic` (records the error and `longjmp`s — never kills the process), and stderr capture (redirect fd 2 to a temp file so the real compiler diagnostic can be read back into the JSON `message`). |
| `json.coil` | `json` | A growable byte buffer (`JBuf`, reallocs — unlike `types.coil`'s 4 KiB `StrBuf`) with JSON emit helpers (string escaping, ints, floats, bools). |
| `serialize.coil` | `serialize` | Value → tagged JSON (§4.1) with the depth rule; the reflection responses `types()`/`fields()`/`methods()`. The only place the wire format lives. |
| `reflect.coil` | `reflect` | Runtime handle resolution (`arena-at` with generation compare, `arena-instance`, `arena-instances`) + the filter-predicate evaluator. What the reflection opcodes call. |
| `server.coil` | `server` | Sockets + minimal HTTP/1.1, the single-slot mailbox, the eval executor (`eval-core`), the safepoint drain, the post-main service loop, and the `scry run`/`scry eval` entries. |

Import DAG (no cycles): `vm → {safepoint, reflect}`; `server → {vm, compile, serialize, reflect, …}`. The `vm`↔`server` cycle is broken by `safepoint` (a leaf holding a drain **fnptr** the server registers).

## The eval pipeline (`eval-core` in `server.coil`)

One request runs, in order, on the **mutator thread** at a safepoint:

1. Arm the landing pad: `eval-set-active 1`, `diag-capture-begin` (redirect fd 2), `setjmp`.
2. `eval-exec`: `lex-file` → classify the leading token. A **definition** (`class`/`fn`/`enum`/`interface`/`object`/`import`/`module`/`migrate`) is the live-code-change seam — it hard-errors `NotImplemented: live code change: not implemented until Phase 6` (Phase 6 replaces this branch).
3. `parse-eval-block` (a `{…}` block, or a statement sequence wrapped as a `BLOCK`). `try-reflection` intercepts a bare `types()`/`fields("X")`/`methods("X")` and emits JSON directly (these return *metadata*, not VM values).
4. Otherwise: `block-value` typechecks the block in the live `ctx` (interns any new monos), producing the result `Type`; a nonzero `err-count` delta → `TypeError` (message = the captured real diagnostic). `compile-eval-method` compiles it into a standalone nullary `MethodInfo`; if `--readonly`, `chunk-mutates` scans the compiled bytecode transitively (store-field / `NEW` / list-push/map-set / any virtual call) and rejects.
5. `vm-eval-invoke` runs it via the **exact** `push-frame` + `run-to <caller-depth>` path bytecode uses; `serialize-value` turns the result into JSON at depth 0.
6. On any `longjmp` (syntax error via the parser, `StaleReference`/`OutOfArenaSpace`/list-OOB via `eval-panic`), the landing pad writes the error envelope. `diag-capture-end`, `eval-set-active 0`.

**`at`/`instance`/`instances` are compiler-synthesized reflection members** (`04` §4.2), taught to both the typechecker (`infer-reflect-static`) and the compiler (`compile-reflect-static` → `OP_ARENA_AT`/`OP_ARENA_INSTANCE`/`OP_ARENA_INSTANCES`), so `Agent.at(7,3).resume()` composes: `at` yields a real instance pointer of the class's static type, and the method call dispatches normally. `types()` stays a driver-level special case (its `TypeDescriptor` list is metadata, not an arena value).

## JSON format

Per `04` §4.1, uniform depth rule: the directly-returned value(s) are depth 0 (entities expand `fields{}`, list/map elements stay depth 0); anything reached through an entity **field** is depth ≥ 1 and collapses to `{"type":"ref","class":<concrete>,"ref":"Name#slot","generation":g,"summary":…}`. Scalars/strings/bools direct; `Void` = `{"type":"Void"}`; enums (incl. `Option`/`Result`) `{"type":<enum>,"case":…,"payload":[…]}`; lists/maps carry `length`+`truncated` and cap at 100 elements. A ref's `class` is always the **concrete** implementing type (read from the instance header's `type-id`), never the declared interface. Errors: `{"error":{"kind","message"[,"line","col"]}}` with kinds `SyntaxError`/`TypeError`/`RuntimeError`/`StaleReference`/`NotImplemented`/`ReadOnly`/`BadRequest`.

## Server architecture (N=1 mutator; `02` §7 with one thread)

- **One OS thread** (Coil `pthread` via `lib/thread.coil`) runs `server-accept-loop`: BSD-socket externs (`socket`/`bind`/`listen`/`accept`/`read`/`write`/`close`, all in `ioutil.coil`), a minimal HTTP/1.1 parse (request line + `Content-Length`, that's it), serving `POST /eval`, `GET /` and static `viewer/` assets.
- **The mutator thread** (running `main()`, then a post-main service loop) polls `safepoint-poll` at the top of `run`'s dispatch loop. When the server thread has a request it fills a **single-slot mailbox**, sets `req-ready` + `stop-flag` (atomics from `lib/atomic.coil`), and spins on `resp-ready`; the mutator's next poll drains it, runs the eval to completion on the VM, serializes, sets `resp-ready`. Evals are serialized, run-to-completion, and see a consistent heap — `04`'s promise. Because there is one server thread issuing one request at a time, a single slot suffices (documented simplification; Phase 5 makes it a real lock-free queue).
- **After `main()` finishes** the process stays alive in the service loop (`safepoint-poll` + `usleep`), servicing evals until Ctrl-C — a finished program with a live heap is still browsable (printed on exit of main). `--no-viewer` runs `main()` and exits (no server); `--readonly` (default OFF) rejects mutating evals.
- `scry run` binds **7357** (falls back to the next free up to +20) and prints `viewer: http://localhost:7357` at startup. Viewer assets resolve to `<dir of argv[0]>/viewer/`, falling back to `./viewer/` (CWD).
- `scry eval <file.scry> -e '<expr>'` runs the program to completion then evaluates the expression once, prints the JSON, and exits — the browser-free golden-test path through the entire eval stack.

## Viewer (`viewer/index.html` + `app.js` + `style.css`, vanilla, self-contained)

Implements `04`'s IA: a left rail **type index** (re-evals `types()` every 500ms; live counts, client-computed trend arrows, interface implementors grouped/foldable) → **instance table** (columns = schema fields with `id` pinned; a filter box passing the predicate string into `instances(filter:…)`; re-evals ~750ms) → **instance detail** (fields with clickable ref links + breadcrumbs, `implements` line, methods with typed-argument invoke forms, inline results/errors, re-eval ~750ms + immediately after invoke; changed fields flash the accent) → a **REPL dock** (backtick toggle; `self` bound by client-side textual substitution `\bself\b → Type.at(slot,gen)`; scrollback rendered by the shared value renderer) → an **eval-transcript drawer** logging every request/response. Dark-first with a `prefers-color-scheme` light override; one accent hue reserved for liveness; monospace for data, sans for chrome.

## The seams Phase 5 / Phase 6 inherit

- **Phase 6 (live change).** The definition-classification point in `eval-exec` (the `is-def-token` branch that hard-errors) is exactly where a definition-eval will instead typecheck against the live class and, at a full stop, **swap `Program.methods[]`/`types[]`**. All calls already route through method **indices**, so a new generation repoints the tables without touching call sites (`03-live-semantics.md`); `vm-eval-invoke` already enters through `push-frame` so an invoke is indistinguishable from a bytecode call.
- **Phase 5 (threads).** The N=1 safepoint protocol lives entirely in `safepoint.coil`: one `stop-flag`, one poll at the dispatch-loop top, one drain callback. Phase 5 generalizes `stop-flag` to the global stop every `VMThread` polls, replaces the single-slot mailbox + spin with `request-global-stop`/`release-global-stop` parking all N threads and a lock-free queue, and runs the drain on a dedicated eval `VMThread` (`02` §7). `run` is already parameter-shaped (`run-to`) to hand each thread its own frames/stack.

## Coil friction hit in Phase 4 (adds to Phase 1/2/3's list)

- **`c"…"` decodes only `\n \t \\ \"` — NOT `\r` nor `\xHH`.** HTTP CRLFs written via `c"…\r\n…"` came out as the literal letter `r`; `\x0d` came out as literal `x0d`. `curl` tolerated the malformed headers but Python `urllib` returned an empty body. Fix: assemble headers with explicit `\return`(13)/`\newline`(10) **char literals** (which the reader *does* turn into integers), not string escapes.
- **`setjmp`/`longjmp` work** across Coil functions (verified); the safe pattern is to keep every value that must survive the `longjmp` in `alloc-static` globals (the `EvalError` record, the result buffers), sidestepping any returns-twice/optimizer concern. This is what lets the parser's `exit`-on-error and the VM's panic sites become recoverable eval errors.
- **`close()` on a socket sends the response fine**, but the response headers must be well-formed CRLF or strict HTTP clients drop the body (see above) — the bug looked like a socket problem and was a lexer problem.
- **`localhost` resolves to `::1` first on macOS**; we bind IPv4 only, so clients using `localhost` eat a "connection refused" + fallback. Not a bug in the server; tests/tools should hit `127.0.0.1`. `curl`/browsers fall back automatically.
- **A `(fnptr c […] …)` cannot be `(cast i64 …)`-null-checked** — keep a separate `has-drain` i64 flag instead of comparing the fnptr to 0.
- **An `(ArrayList T)` value parameter is an immutable ref** (known from Phase 2/3) — the readonly bytecode-visited set is a heap `VisitSet` struct passed by pointer, not an `ArrayList` param.
