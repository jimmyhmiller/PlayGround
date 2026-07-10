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

---

# Phase 5 — real OS threads + the agents demo (M1 + M5)

Phase 5 makes concurrency real (`DECISIONS.md` #4b): every Scry-level `Thread.spawn` is a
genuine `pthread` running the dispatch loop against its own `VMThread`, sharing the
Program/heap/arenas; a generalized safepoint parks all of them at once for an eval; and
`examples/agents.scry` is the M5 demo — 3 named agents on 3 real threads, a shared task
list under a `Mutex`, `Clock.sleep` pacing, and a viewer-invoked `pause()`/`resume()` that
visibly stops/restarts one agent's terminal activity.

## What changed, by file

| File | Change |
|---|---|
| `bytecode.coil` | New opcodes `OP_THREAD_SPAWN <iface-id> <slot>` / `OP_THREAD_JOIN`; new builtin ids `BLT_MUTEX_{NEW,LOCK,UNLOCK,GET,SET}`; disassembler entries. |
| `safepoint.coil` | **Rewritten** into the N-thread protocol: one global `stop-flag`, a registry of per-thread `parked`-word addresses, a coordinator marker, and `sp-register` / `safepoint-poll(parked)` / `request-global-stop(coord)` / `release-global-stop`. Still a leaf (knows only `(ptr i64)` parked words). |
| `arena.coil` | Thread-safe allocation: `atomic-add` bump frontier + a per-arena `growing` CAS spinlock for slab mapping; `slabs` array pre-sized (never realloc'd). `live-count` is `atomic-add`. |
| `builtins.coil` | `MutexObj` + `mutex-new/get/set` (data + non-blocking accessors); blocking `clock-sleep` removed (moved to `vm.coil`, cooperative). Float bit-cast helper made per-call (thread-safe). |
| `vm.coil` | **Refactored** off the single global `VM`: per-thread state lives in `VMThread` (its own `stack`/`frames`/`stack-top`/`frame-count` + atomic `parked`/`done`); `run-to`/`push-frame`/`vpush`/`vpop`/`vpeek`/`do-builtin` all take the `VMThread`. New: `vm-spawn`, `vm-join`, `vm-sleep`, `vm-mutex-lock/unlock`, `interpreter-thread-main`, main/eval `VMThread` singletons. |
| `compile.coil` | `Thread.spawn` → `OP_THREAD_SPAWN` (Runnable iface-id + `run` slot baked in); `ThreadHandle.join` → `OP_THREAD_JOIN`; `Mutex<T>(v)` and `.lock/.unlock/.get/.set` → the mutex builtins. The Phase-3 "not implemented" hard errors are gone. |
| `server.coil` | Eval flow **inverted**: the server thread is now the eval **coordinator** — `request-global-stop` parks every language thread, `eval-core` runs on the dedicated eval `VMThread`, `release-global-stop` resumes them. The Phase-4 mailbox/drain-on-mutator round-trip is deleted. Post-main loop now polls the *main* `VMThread`'s parked word so a coordinator can stop it too. |

## VMThread / VM (as-built)

```
(defstruct VMThread [(stack (array i64 65536)) (stack-top i64)
                     (frames (array CallFrame 256)) (frame-count i64)
                     (parked i64) (done i64) (thread-id i64) (os-thread Thread)])
(defstruct VM [(program (ptr Program))])            ; the ONLY shared VM state
```
Main and the eval-coordinator each own one `VMThread` **singleton** (`alloc-static`); agent
threads are `malloc`'d at spawn. Program/types/methods/arenas are shared read-only after
`build-program`. `parked`/`done` are the only atomically-touched words on a `VMThread`.

## Safepoint protocol (as-built)

- Each thread that runs the dispatch loop calls `sp-register((field vt parked))` once, at
  startup/spawn, publishing the address of its own `parked` word into a fixed 64-entry
  table. The eval `VMThread` is **not** registered — it is always the coordinator, never a
  waiter.
- `safepoint-poll(parked)` runs at the top of `run-to`'s loop (once per instruction) and
  inside `Clock.sleep` / `Mutex.lock` spins. When `stop-flag==1` and this thread is not the
  coordinator, it publishes `parked=1`, spins until the flag clears, then `parked=0`.
- `request-global-stop(coord)` CAS-acquires `stop-flag` (0→1), records `coord`, and spins
  until every *other* registered thread's `parked==1`. `release-global-stop` stores 0.
- **Dead threads count as parked.** `interpreter-thread-main` sets `parked=1` (and `done=1`)
  when its dispatch loop returns, so a finished agent never makes `request-global-stop` wait
  forever.
- **Blocked foreign calls made cooperative.** `Clock.sleep` sleeps in 2 ms slices, polling
  the safepoint each slice; `ThreadHandle.join` spins on the target's `done` flag while
  polling its *own* safepoint. So neither a sleeping nor a joining thread delays a global
  stop by more than one slice — nothing parks in a blocking syscall while holding the stop
  hostage. (stdout writes are the one remaining non-cooperative foreign call, but they are
  sub-millisecond; `request-global-stop` just waits out the write.)
- **Every eval is full STW** — the deliberately conservative choice (05 M1 builds only the
  full, all-thread stop). Partial/read-only safepoints (02 §7's read-eval fast path) are a
  documented future optimization, not built here. Because there is one server thread taking
  one connection at a time and `request-global-stop` CAS-serializes coordinators, evals
  never overlap, so `eval-core`'s global scratch/`setjmp` state stays single-threaded.

## Alloc strategy chosen (LOUD, per the brief)

**Atomic-bump shared arena + a per-arena `growing` spinlock — NOT yet per-thread
magazines.** 05 M1 lets us take the documented "simpler correct thing" at 3–5 threads; we
did, and shaped it so §4's magazines slot in on top later:
- `arena-alloc` = `atomic-add(bump-cursor, 1)`. The returned old value is this caller's
  **unique** slot — two threads never collide, and there is no CAS loop because the frontier
  only grows. A magazine refill later is exactly the same call with `+MAGAZINE_CAP` instead
  of `+1`, over this identical layer.
- Each thread writes **its own** slot's header; no other thread touches that slot → race-free
  by construction, and the header's `type-id` is written before `OP_NEW` ever publishes the
  pointer (05 M1's construction-visibility ordering note).
- Slab mapping (malloc a 64 KiB block + publish its base) is the one shared mutation; it
  takes the per-arena `growing` CAS spinlock. The common case (slot already in a mapped
  slab) never touches it — just an `atomic-load` of `slab-count`.
- The `slabs` base-pointer array is **pre-sized to the type's whole budget** at `arena-new`
  and never realloc'd, so a concurrent `arena-slot-ptr` can never read a torn/moved array
  base. `live-count` is `atomic-add` (coarse, eventually consistent — §4).

Proven race-free empirically: 3 threads × 5000 and 6 threads × 15000 concurrent `Message`
allocations yield **exactly** 15000 / 90000 live with matching high-water and no
duplicated/lost slots, stable across dozens of runs; 4–8 threads × tens of thousands of
mutex-guarded increments yield the **exact** expected total every time.

## Mutex (as-built) — the parked-holder deadlock, and how it's dodged

`MutexObj = {lock, val}`. `lock` is a `atomic-cas(0→1)` spin that **backs off through the
safepoint** (so a waiting thread still parks for a global stop instead of hanging
`request-global-stop`). `unlock` stores 0; `get`/`set` are single aligned-word accesses.
The one subtlety: an agent can be parked at a safepoint *while holding a mutex*, and a
viewer-invoked method (run by the coordinator during STW) might try to acquire that same
mutex → deadlock. Dodged by making `lock`/`unlock` **no-ops for the coordinator**: during a
global stop the coordinator has exclusive access to the whole heap, so it needs no app-level
lock, and skipping avoids ever waiting on a lock a parked thread holds. Agents never run
during STW, so they never take a lock while stopped; only the coordinator does, and for it
it is a safe no-op.

## agents.scry design notes (race-free by single-writer discipline)

`AgentStatus {Waiting,Running,Paused,Done}`, `Message`, `Conversation`, `Task`,
`interface Tool` (`ShellTool`/`SearchTool`), `ScriptedModel` (honest, in the type list),
`Agent`, `AgentWorker : Runnable`. Concurrency safety without locks on the hot path comes
from **single-writer-per-field**: `Agent.paused` is written *only* by the viewer/eval thread
(`pause`/`resume`) and read by the worker; `Agent.status` and the `Conversation` are written
*only* by that agent's worker. Single-word writes are non-tearing (02 §1), so no torn state
without a lock. The genuinely shared mutable cell — `TaskList.remaining` — is behind a real
`Mutex<Int>`; `take()` hands out a unique task id under it (the demo output shows ids
120,119,118,… never duplicated, which is the mutex working). The TUI is the M5-acceptable
minimal form: a scrolling log with per-agent ANSI color prefixes built via `\xHH` escapes
(`\x1b[…m`), not a cursor-addressed repaint.

## Tests (16 new; 213 total, all green, zero-regression on the 197 single-threaded)

- `tests/run/`: `thread_spawn_join` (deterministic post-join aggregate), `mutex_counter`
  (4×25000 → exactly 100000), `clock_sleep`, `many_threads` (8→8), `shared_list_mutex`
  (3×100 → 300), `two_waves`, `mutex_get_set`, `threads_alloc_sum`,
  `thread_interface_dispatch` (concurrent itable dispatch → 160), `thread_enum_match`
  (enum+match under threads → 300), `thread_conversation`.
- `tests/run-arenas/alloc_race`: 3×5000 concurrent `Message` allocs → exactly 15000 live.
- `tests/eval/40-42`: `scry eval` against `examples/threads-mini.scry` (runs to completion,
  then evaluates the final quiescent heap) — exact `Message` count 3000, type list, fields.
- **`liveness`** (python, in `run-tests.py`): starts `examples/agents.scry`, POSTs evals
  while the 3 agents run — asserts type/instance visibility, that two agents' `Conversation`
  sizes **climb between polls**, that `pause()` on one **freezes** its conversation while the
  others keep climbing, and that `resume()` **restarts** it. This is THE demo beat, verified
  through the real viewer channel under real STW. All concurrency goldens are exact
  (deterministic by design — post-join aggregates and mutex totals, never interleaved
  prints), so a failure is a real race, never rehearsal noise.

## What Phase 6 (live code change) inherits

- **The stop machinery is exactly what a table swap needs.** `request-global-stop` /
  `release-global-stop` already bring *every* `VMThread` to a quiescent stop and run
  coordinator work with the heap frozen — a definition-eval swaps `Program.methods[]` /
  `types[]` at that same stop instead of running an expression. All calls already route
  through method **indices**, so a new generation repoints the tables without touching call
  sites (03-live-semantics).
- **The seam is unchanged from Phase 4**: `eval-exec`'s `is-def-token` branch still
  hard-errors `NotImplemented: live code change`; Phase 6 replaces that branch with
  typecheck-against-live-class + table swap, now already inside a full STW.
- **What Phase 6 must still add for 02 §4's shape migration** (beyond a method-body swap):
  `InstanceHeader.shape-id`, the `old-shape`/`pending-shape` `TypeArena` fields, and the
  per-thread stack-root walk for the pointer-rewrite pass — none of which exist yet. The
  registry in `safepoint.coil` already enumerates every live `VMThread`, which is the list
  that walk will iterate.

## Coil friction hit in Phase 5 (adds to Phase 1-4's list)

- **A shared `alloc-static` cell is a data race across threads.** Phase 3's `to-bits`/
  `to-f64` float bit-cast used one global `alloc-static i64` scratch cell; with N threads
  doing float ops that races. Fix: use a per-call `alloc-stack i64` (freed on return, one
  per invocation) — thread-safe, and LLVM folds the through-memory round-trip to a register
  move anyway.
- **`lib/thread.coil`'s externs dedup fine when declared once and `:use *`'d.** Both `vm`
  and `server` import `thread.coil`; because the `pthread_*` externs live in that one module
  and are only `:use *`-imported (never re-declared), there is no double-declaration link
  error — the guide's "declare each extern in ONE module" advice, confirmed for pthreads.
- **`(field vt parked)` is a clean decoupling seam.** Because `safepoint.coil` only ever
  handles `(ptr i64)` parked-word addresses (never the `VMThread` struct), the safepoint
  module stays a leaf and `vm.coil` calls it with no import cycle — the same trick Phase 4
  used for the drain fnptr, generalized.
- **`;` is not a statement separator in Scry** (re-confirmed while writing tests): statements
  are newline-separated inside a block; `let a = f(); let b = g()` is a parse error. Enum
  variants are likewise newline-separated, and a `match` on an enum needs bare (unqualified)
  variant patterns.

---

# Phase 6 — live code change (M4)

The final demo beat: redefine a method on a *running* program, whole-program-typechecked,
atomically, at a full stop-the-world. This is `03-live-semantics.md`'s "generations, not
diffs" cut down to exactly `05-milestones.md` **M4**: method-**body** swap (same signature),
whole-class body swap, and **additive fields with a static default**. Everything else is a
loud rejection with a real diagnostic. No new wire op — it is `eval` of a `source` that
happens to be a definition (`04-viewer.md` §3.4), over the same `{id, source}` →
`{id, value|error}` channel, and it runs *inside the exact STW section Phase 5 built*
(`server-eval-stw` → `request-global-stop` … `eval-core` … `release-global-stop`), so a
definition-eval swaps tables with every language thread parked and the heap quiescent.

## What changed, by file

| File | Change |
|---|---|
| `parser.coil` | `parse-field` accepts an optional `= <expr>` **static default** (stored in `FIELD.b`). Plain fields are unchanged (b = null), so every existing golden is byte-identical. |
| `ast.coil` | `NK_FIELD` dump emits the default child when present (optional; absent = unchanged dump). |
| `typecheck.coil` | Definite-assignment treats a **defaulted** field as pre-assigned — a field with `= default` needs no `init` store (that's what makes an added field constructible without touching `init`). |
| `compile.coil` | Generation counter (`current-generation`/`bump-generation`); `FIELD_HEADROOM` reserved in every entity arena's slot; the low-level swap mechanics: `repoint-method-node!`, `recompile-method-at!`, `add-field-live!`. |
| `server.coil` | The redefinition engine: `eval-definition` (dispatch), `redefine-class` / `redefine-fn` / `redefine-interface`, shape classification, method-set + signature checks, the rejection helpers, `ev-defined`, and `generation()` reflection. The Phase-4 `is-def-token` hard-error branch now calls `eval-definition`. |
| `main.coil` / `server.coil` | `scry eval` takes **multiple** `-e` run in sequence against one process — the browser-free define-then-observe path for golden tests. |
| `viewer/` | A `#code-panel` drawer (opened by "✎ edit source" in the instance detail) prefilled with the class skeleton from the live schema; POSTs the source, shows `✓ … generation N` or the inline rejection. |

## Generation machinery as-built

One global counter (`compile.coil`, `gen-cell`): the initial `build-program` is **generation
0**; each **accepted** edit calls `bump-generation` and the response reports the new number.
A rejected edit never bumps it — rejection is a strict no-op (`03` §5). `generation()` is a
bare reflection call (like `types()`), intercepted in `try-reflection` → `{"type":"Int",…}`.
An accepted change serializes as `{"type":"defined","defined":"<name>","gen":N,"message":…}`.

The swap itself is `03`'s "repoint one table, keep the index": every call is a `CALL_STATIC
<method-idx>` / itable slot baked at compile time, so a redefinition **keeps the method's
index and stores a fresh `MethodInfo` at it** in `Program.methods[]`. Every existing call
site — and every future one — reaches the new body with zero call-site patching. To make the
*new* body's own self/sibling calls resolve, the compiler's node→index map is repointed
first (`repoint-method-node!` sets `method-nodes[idx] = new-node`), for **all** the class's
methods, *then* each is recompiled (two passes, so cross-references resolve). Itables are
untouched (method indices are stable) for a same-shape / additive edit. **Old chunks are
never freed** — a stale frame may still be executing one — so there is a bounded
per-generation leak, documented and acceptable for the PoC (no GC in this build anyway).

## The acceptance path (per definition kind)

- **`class` redefinition.** Find the live `Decl` (else *unknown class* reject). Classify the
  shape delta against the live field layout: **SAME** (identical names+types+order),
  **ADDITIVE** (old fields are a prefix, each new trailing field carries a static default),
  or **incompatible** (removed / retyped / renamed / reordered → loud reject naming the field
  and the live-instance count). Enforce the M4 method rules: no added method, no removed
  method, and every method's signature matches the live one exactly (a sig change is
  rejected — M4 forbids it). Then swap `d.members` to the new set, typecheck **every** method
  body (and each default) against the now-current field table, and roll the members back on
  any type error (rejection = no-op). On success: repoint + recompile every method at its
  index, run the additive migration, bump the generation.
- **standalone `fn`.** Signature must match the live one; typecheck the new body; repoint +
  recompile at the same index.
- **`interface`.** Temp-install the new members, re-run conformance for **every** implementor,
  roll back. If conformance breaks (an implementor lacks a now-required method) → the captured
  conformance diagnostic. Otherwise an honest *not supported in this build (M4)* — interface
  changes beyond the conformance guard are out of M4 scope.
- **`migrate` / `enum` / `object` / `import` / `module`** → loud *not supported* rejection.

Every rejection carries a real diagnostic (typechecker-produced ones flow through the Phase-4
stderr capture; policy ones are formatted messages) and leaves the running program **exactly**
as it was — proven by tests that assert both the message *and* unchanged behavior/state after.

## Additive field — the M4 cut, exactly

`05` M4 pins field-add as: *a class may gain a new field only if it carries a mandatory
default; on accept, walk the arena once (during the STW pause) and write the default into the
new field's slot for every existing live instance; future constructions include it normally.*
It explicitly does **not** ship the full `02` §4 migration (new-strided slab + graph pointer
rewrite + shape-id + quarantine) — that is deferred past M5.

To honour "write the default into the new slot" **without** re-striding the arena (which would
move instances and dangle every raw pointer into it), **every entity arena reserves
`FIELD_HEADROOM` (32 bytes = 4 i64 fields) of trailing per-slot space at `arena-new`.** An
additive field therefore lands at a fixed trailing offset **inside each slot's existing
reserved space** — no new slab, no instance move, no pointer rewrite. Identity
(`type-id, slot-index, generation`), all field offsets of the old fields (additive only
*appends*, so they never shift — which is also why a stale frame reading old fields stays
correct), and every raw pointer into the arena survive the edit untouched. The migration is
literally: append a `MonoField` (viewer schema) + a `FieldInfo` (VM metadata), then loop the
live slots writing `defval` at the new offset — O(live instances), in place. The default is a
static value: it is compiled as a nullary method and run once on the eval `VMThread` (heap
quiescent), so a shared `""`/`0`/enum default is materialised once and stamped into every
instance. A `self` in a default fails to typecheck (no self in scope) → rejected, so a default
is genuinely static. Exhausting the headroom (a 5th added field) is a **loud** reject, never a
silent overrun. This changed the two `run-arenas` goldens' `slot-size`/`slabs` numbers (bigger
slots); the semantically-meaningful `live`/`high-water` counts are identical and were the only
thing those tests actually guard — re-blessed.

## Stale-frame story as-implemented

Exactly `03`'s per-frame rule, and it falls out of Phase 5's STW for free. A definition-eval
parks every language thread at a safepoint, swaps `Program.methods[]`, bumps the generation,
and resumes. A thread that was **mid-old-body** finishes that frame on the old bytecode (its
code pointer and PC were captured at entry; nothing re-fetches them) — but its *next* `CALL`,
including a recursive self-call, re-reads `methods[idx]` and gets the new body. The live-edit
e2e test exercises this directly: three agents are looping when the swap lands (some frames in
flight during the STW); they keep running and the redefined output appears **within a couple
of turns**, never mid-frame. There is no per-frame versioning and none is needed — the swap is
one atomic store visible at the next dispatch, per thread. A deterministic "prove a specific
frame finished on old code" test would need a `Clock.sleep` planted mid-body plus precise
thread timing; that is inherently racy, so — per the owner's no-flaky-tests rule — it is not a
golden, and the property is instead covered by the (deterministic) e2e "output changes within
a couple turns while the process keeps running and identity/counts persist" assertions.

## Viewer UX added

An instance-detail **"✎ edit source"** button opens a persistent bottom `#code-panel` drawer
(separate from the 750ms-polled detail pane, so it never wipes what you type), prefilled with
the class's skeleton reconstructed from the live `fields()`/`methods()` schema (field types
have their `ref:`/`list:` serialization prefix stripped; bodies are `// edit this body`
stubs). "define" (or ⌘/Ctrl-Enter) POSTs the buffer as an ordinary eval and shows
`✓ <Class> redefined — now at generation N` (accent flash) or `✗ <Kind>: <message>` inline;
after acceptance the detail poll picks up the new behavior on its own. The eval-transcript
drawer logs the definition eval like any other request. Dark/light both styled; the one accent
hue (reserved for liveness) marks the accepted change.

## What post-PoC migration would add (beyond M4)

To reach `03`/`02` §4's full field-change design: `InstanceHeader.shape-id` + per-arena
`old-shape`/`pending-shape`; the two-shape dispatch table (`methods[shape_id][slot]`) that
only exists while a class drains; the `active_frames` per-class quiescence counter gating a
migration on `== 0` rather than the whole-process STW used here; a **new-strided slab** with
the graph pointer-rewrite pass (reusing GC's `TypeInfo`-driven walk over **every** thread's
stack — the `safepoint.coil` registry already enumerates them) so a field **remove/retype**
can move instances safely; user `migrate` functions with per-instance **quarantine** on
failure; and interface **method-add acceptance** (itable rebuild + slot renumbering) rather
than the conformance-guard-only handling here. None of that ships in M4, by design.

## Coil friction hit in Phase 6 (adds to Phase 1-5's list)

- **`al-set!` is the way to overwrite an `ArrayList` element in place** — used to repoint the
  compiler's `method-nodes[idx]` and `methods[idx]` to the new generation's node/`MethodInfo`
  without rebuilding the whole table.
- **`snprintf` with `%.*s` bridges Scry's `(ptr u8, len)` names into C-string messages** —
  names aren't NUL-terminated, so every dynamic rejection message is built with
  `%.*s` (`(cast i32 len)` + `(cast (ptr i8) ptr)`), which is also how a formatted message
  reaches `ev-fail`/`ev-set-msg` (policy rejections use `ev-msgbuf`; real typechecker
  diagnostics come back through the Phase-4 fd-2 capture for `EK_TYPE`).
- **Swap-members-recheck-rollback keeps the retypecheck localized.** Redefinition installs the
  new members on the live `Decl`, runs the *existing* `check-fn-body`/`check-conformance`
  against them, and restores the old members on any error — no separate "trial context", and
  `mk-named`/`intern-mono` are idempotent (a re-intern hits the existing mono), so re-checking
  a live class doesn't duplicate monos.
- **A definition-eval never needs its own STW** — it already runs between
  `request-global-stop`/`release-global-stop` in `server-eval-stw`, so the swap is
  automatically at a full stop; the CLI path (`scry eval`) runs after `main()` returns, so
  there are no other threads to park at all.

---

# Phase 7 — interactive stdin I/O + the Claude-Code-like assistant (M5 flagship)

Phase 7 makes the demo app a *real interactive terminal program*: it prompts, reads a line the
user types, dispatches, replies, and (on "research …") spawns sub-agents on real threads — all
while remaining 100% viewer-unaware (the runtime injects the eval server for any program, NREPL
style). The point the owner wanted: *user interaction at the command line, and we watch it in the
UI, and we can change code live and make new things pop up* — proven by redefining
`Session.suggest` from the viewer and seeing a suggestions box appear under every prompt with no
restart.

## What changed, by file

| File | Change |
|---|---|
| `bytecode.coil` | New builtin ids `BLT_CONSOLE_PRINT 25` (String → stdout, no trailing newline) and `BLT_CONSOLE_READLINE 26` (no args → `(ptr Option<String>)`). |
| `ioutil.coil` | One extern: `poll(struct pollfd*, nfds, timeout-ms)` — the primitive behind the cooperative read. |
| `builtins.coil` | `console-print` (like `console-log` but no `\n`; fd-1 `write` is unbuffered, so a prompt reaches the terminal before the following read). readLine's body lives in `vm.coil` (needs the VMThread to poll). |
| `vm.coil` | `vm-readline(vt)` — the safepoint-cooperative line reader (below); `do-builtin` dispatches the two new ids. |
| `typecheck.coil` | `Console` gains `print(s: String) -> Void` and `readLine() -> Option<String>`. |
| `compile.coil` | `compile-builtin-object`'s `Console` branch is now a `cond` over the method name: `print` → `BLT_CONSOLE_PRINT`, `readLine` → `BLT_CONSOLE_READLINE` (no args compiled), else `log`. |
| `examples/assistant.scry` | The flagship app (below). `agents.scry` is kept. |
| `tests/run-tests.py` | `.stdin` files for `run/` goldens; a `stdin:` key for `eval/` `.t`; a new `app/*.t` runner (scripted stdin → stdout substrings); the `assistant_e2e` gate. |

## `Console.readLine()` — blocking to the program, cooperative to the runtime

The hard requirement: a line read must look blocking to the Scry program, but **never park the OS
thread in a blocking `read()`** — that would freeze every STW/eval while the user thinks, defeating
the whole "inspect the app while it waits for input" premise. So `vm-readline` is shaped **exactly
like Phase 5's `Clock.sleep`**: a loop that each pass calls `safepoint-poll((field vt parked))`,
then `poll(fd 0, POLLIN, 20ms)`. On a 20 ms timeout it loops (having just polled the safepoint); on
readable it `read`s **one byte**. So the longest a coordinator's `request-global-stop` waits for
this thread to park is one 20 ms slice — an eval POSTed while the user sits at the prompt answers
promptly. The `struct pollfd` is built as a single `i64`: `fd=0` in bytes 0–3, `events=POLLIN(1)`
at byte 4 → the literal `(<< 1 32)`; no sub-word struct writes needed.

Byte-at-a-time under `poll` (rather than a bulk `read`) is deliberate: it means we never block
between the bytes of a line even when input arrives split across pipe chunks — each byte waits at
the cooperative `poll`, so a slow/partial writer can't hold the STW hostage. A complete terminal
line (delivered on Enter) is fully buffered, so its bytes come back with zero inter-byte `poll`
waits; we stop at `\newline` before touching the next line.

**EOF is an honest `Option`.** 01 has no null, so `readLine()` returns `Option<String>`:
`Some(line)` for a completed line (newline stripped; a bare Enter is `Some("")`, a real empty line),
and `None` at EOF (Ctrl-D / closed pipe: `read` returns 0 with nothing buffered). The app matches on
it. `Console.print(s)` is the no-newline sibling of `Console.log` for prompts.

### stdin-vs-STW interaction (the crux, and how it's tested)

While the main thread sits "in" `readLine`, it is really spinning the `poll`+`safepoint-poll` loop.
When the server coordinator requests a global stop for an eval, the main thread publishes `parked=1`
within ≤20 ms (same mechanism as a sleeping/joining thread), the eval runs on the eval `VMThread`
with the heap quiescent (including a **mutation** eval — the `assistant_e2e` test appends to the
orchestrator's conversation via `Agent.instance(0).say(...)` while the app waits at the prompt and
reads the new size back), then the main thread resumes its read loop. No new machinery: `readLine`
reuses the exact `parked`-word registered at thread start, so it is just another cooperative foreign
call alongside `Clock.sleep`/`ThreadHandle.join`/`Mutex.lock`.

## `examples/assistant.scry` — structurally real, honestly fake

Entities: `Message`, `Conversation`, `interface Tool` (`ShellTool`/`SearchTool`, canned outputs),
`ScriptedModel` (keyword-triggered canned replies — `hello`/`thanks`/`?`/default), `Agent`
(name/role/color/status enum/model/conversation/tools), `SubAgentWorker : Runnable` (a spawned
sub-agent's thread body), `Session` (REPL owner + history — the live-edit target), `Orchestrator`
(owns the assistant `Agent`, shared tools, and every spawned sub-agent + its `ThreadHandle`).

Main loop: print `renderPrompt()` with `Console.print`, `match Console.readLine()`, dispatch,
optionally print `suggest(line)`, loop; `None` (EOF) or `"exit"` ends the loop → `orch.shutdown()`
joins outstanding sub-agents → `goodbye`; `main` returns and the runtime keeps the process alive and
browsable (its own post-main hint). Keyword routing (`Orchestrator.dispatch`): `help` lists
capabilities; `research <topic>` prints a delegating line and `Thread.spawn`s two `SubAgentWorker`s
(researcher: 4 turns, summarizer: 3 turns) that each append `Message`s to their own `Conversation`
with `Clock.sleep(250)` pacing and print an interleaved per-agent line — **the main loop stays
responsive, so you can type the next input while they work**; any other input gets a direct
`ScriptedModel` reply. Sub-agents are ordinary `Agent` instances, so a `research` grows the live
`Agent` count 1 → 3 and the `Message` count climbs — visible in the viewer *during* the interaction.
Concurrency-safe by single-writer discipline (each sub-agent thread is the only writer of its own
agent's conversation/status), exactly as `agents.scry`.

### The `suggest()` extension-point pattern (the two-way live-edit beat)

`Session` is kept **deliberately tiny** — `history`, `init`, `renderPrompt() -> String` ("you> "),
`suggest(input) -> String` ("") — so the whole class is a compact Phase-6 redefinition target
(M4 requires the *exact* field + method set, so a small class keeps the swap snippet short).
`suggest` ships returning `""` (empty ⇒ main prints nothing). The demo redefines the whole class
live from the viewer's code panel, changing only `suggest`'s body to return a bracketed suggestions
line built from `history`; the very next prompt prints it — a new UI element **popped into a running
program from the outside**. The exact snippet is in the file header (and asserted verbatim by
`assistant_e2e`):

```
class Session {
  history: List<String>
  fn init() { self.history = List<String>() }
  fn renderPrompt() -> String { "you> " }
  fn suggest(input: String) -> String {
    if self.history.len() == 0 { "" }
    else { "  [suggestions: help | research <topic> | exit  (last: " + self.history.get(self.history.len() - 1) + ")]" }
  }
}
```

The source has **zero** viewer/server awareness — `suggest` is just a normal method the app calls;
that the viewer can swap it is a runtime property, not something the app knows.

## Tests (12 new; 248 total)

- **`tests/run/readline_*` (5 golden, `.stdin`-driven, exact stdout):** `readline_echo` (line
  loop), `readline_eof` (empty stdin → `None`), `readline_print` (`print` has no trailing newline,
  `log` does), `readline_empty_line` (blank line is `Some("")`, distinct from `None`),
  `readline_count` (deterministic post-loop aggregate). The harness pipes `NAME.stdin` if present.
- **`tests/app/*.t` (5, scripted stdin → stdout substrings):** `hello`/`help`/`plain` response
  flows, `eof` (EOF reaches goodbye), `research_aggregate` (the deterministic **post-join
  aggregate** — "2 sub-agent(s) finished" — never the interleaved per-turn lines). Substring (not
  exact) matching is what keeps a research golden non-flaky despite background-thread interleaving.
- **`tests/eval/50-assistant-subagents.t` (1, `stdin:` + `-e`):** feed `research quantum`, run to
  completion (which **joins** the sub-agents), then reflect: exactly 3 `Agent`s, 21 `Message`s,
  3 `ScriptedModel`s, sub-agents `status Done` — the aggregate proof, deterministic because it is
  read *after* the join, not off interleaved stdout.
- **`assistant_e2e` (the gate):** drives the app over a stdin pipe while POSTing evals — (b) STW +
  a mutation answer while the app is blocked in `readLine`; (a) `research` grows Agents 1→3 and
  Messages climb *during* the interaction, sub-agent lines appear; (c) the verbatim `Session.suggest`
  redefinition makes the suggestions box appear on the next typed input; (d) `exit` → goodbye and
  the process still serves evals. Paced with generous sleeps against the canned `Clock.sleep`
  timing; stable across repeated runs.

## Coil friction hit in Phase 7 (adds to Phase 1-6's list)

- **A `struct pollfd` is cleanest as one packed `i64` literal.** Rather than sub-word (`i16`) field
  stores for `events`, build the whole `{fd,events,revents}` as a single `alloc-stack i64` and
  `store!` `(<< 1 32)` (fd=0, `events=POLLIN=1` at byte offset 4, revents=0). One store, correct
  little-endian layout, no struct type needed for a one-shot FFI arg.
- **fd-1 `write` is unbuffered**, so `Console.print` needs no explicit flush before a `readLine` —
  the prompt is already on the terminal. (Had `print` gone through a buffered `FILE*`, the prompt
  would have lagged behind the blocking read; the raw `write` syscall sidesteps it.)
- **The safepoint-cooperative recipe generalizes verbatim from `Clock.sleep` to `readLine`.** Any
  "blocking" foreign wait becomes STW-safe by looping `safepoint-poll` + a short-timeout syscall
  (`nanosleep` slice / `poll` timeout) instead of one long blocking call — the single most important
  runtime pattern in this codebase, now applied three times (sleep, join-spin, readLine).
- **A whole-class Phase-6 redefinition must carry the class's *exact* method set** (M4: no
  add/remove). Keeping `Session` to `init`/`renderPrompt`/`suggest` is what makes the live-edit
  snippet short enough to paste — a design constraint the app honors on purpose.

# Phase 8a — a real HTTP(S) client (`Http`), backed by libcurl

Phase 8a gives Scry a genuine network client: `object Http { request(...) }`, performed over real
DNS + TLS through **libcurl via FFI** — no `curl` subprocess, no hand-rolled TLS. It is the
foundation the agent phases build on (8b Env+Json read config/parse responses; 8c AnthropicModel
becomes a *real* `Model` that calls `/v1/messages` instead of the scripted stub).

## The surface (Scry)

```
object Http { fn request(method: String, url: String, headers: List<String>, body: String) -> HttpResponse }
class HttpResponse { status: Int ; body: String }
```

`headers` is a `List<String>` of `"Key: Value"` lines. Usage:

```
let h = List<String>()
h.push("x-api-key: ${key}")
h.push("anthropic-version: 2023-06-01")
h.push("content-type: application/json")
let r = Http.request("POST", "https://api.anthropic.com/v1/messages", h, body)
// r.status : Int   r.body : String
```

### Error model — status 0 is the sentinel (documented choice)

A **transport failure** (DNS/TLS/connect/timeout — anything where no HTTP response was received)
returns `HttpResponse{ status: 0, body: <curl error text> }`. Every real HTTP outcome (200, 401,
404, 5xx…) returns that status with the response body. We do **not** return `Result<HttpResponse,
String>`: status-0 is the safe default the orchestrator asked for (the agent layer just checks
`status == 0`), and it threads through the checker with zero friction since `HttpResponse` is a
plain value with two fields. A cert failure surfaces as a status-0 transport error with curl's
message — **never** a silent bypass (`VERIFYPEER`/`VERIFYHOST` stay at curl's secure defaults, so
the system trust store is authoritative).

## The binding, by file

| file | change |
|------|--------|
| `Coil.toml` | new `[link] libs = ["curl"]` → `-lcurl` (system `/usr/lib/libcurl.4.dylib`, 8.7.1; default linker paths find it, no `-L` needed). |
| `ioutil.coil` | the libcurl externs (the one extern home): `curl_global_init`, `curl_easy_init/setopt/getinfo/cleanup/strerror`, `curl_slist_append/free_all`, `curl_multi_init/add_handle/remove_handle/perform/poll/info_read/cleanup/strerror`. `setopt`/`getinfo` are declared variadic (`...`) — the C option int selects the vararg type; we always pass an 8-byte value. |
| `http.coil` | **new module.** The whole client: growable `HttpBuf` + write callback, `str-to-cstr`, `http-build-headers` (List<String>→`curl_slist`), the cooperative `http-perform` multi-loop, `http-result-code` (drains the transport CURLcode), and `http-request` → `HttpResult{status,body,err}`. Imports only `ioutil`/`builtins`/`safepoint`, and takes the caller's `(field vt parked)` word directly — so it does **not** import `vm.coil` (no cycle). |
| `bytecode.coil` | `BLT_HTTP_REQUEST 27`. |
| `typecheck.coil` | registers `HttpResponse` (a **`DK_CLASS`** builtin with two `NK_FIELD` members — class-kinded so it becomes an arena entity) and `object Http` with the `request` signature. New `bt-field` helper. |
| `compile.coil` | `compile-builtin-object`'s `Http` branch: push the 4 args left-to-right, emit `OP_CALL_BUILTIN BLT_HTTP_REQUEST`. |
| `vm.coil` | imports `http.coil`; `do-builtin` pops `(method,url,headers,body)` and calls `http-request((field vt parked), …)`, then `http-alloc-response` turns the result into a real `HttpResponse` entity; `vm-run` does the one-time `curl_global_init` before any thread can race. |

## The cooperative multi-loop — why an in-flight request stays inspectable

The request is driven through libcurl's **`multi`** interface, never `curl_easy_perform` (which
blocks the whole request and would stall every STW for its full duration). `http-perform` is the
*exact* safepoint-cooperative shape used by `Clock.sleep` / `vm-readline` / the join-spin, now
applied a fourth time — reusing the **same `parked` word** the thread registered at spawn:

```
(loop
  (safepoint-poll parked)                 ; park here for a global stop, every pass
  (curl_multi_perform mh &running)
  (if (= running 0) (break)
    (do (curl_multi_poll mh NULL 0 50 &numfds) (continue))))   ; wait ≤50ms, then re-poll
```

So a coordinator that requests a global stop **while a network round-trip is outstanding** sees
this thread publish `parked=1` within one 50ms poll slice — an eval (`types()`, an instance read,
a mutation) answers promptly mid-request. This is what makes "browse the heap while an agent is
waiting on Claude" real rather than aspirational. `NOSIGNAL=1` is set so libcurl never arms
`SIGALRM` off the main thread; a 30s `TIMEOUT_MS` guards against a hung peer.

curl option/info integers are baked as constants in `http.coil` with `; curl.h:` provenance
comments (verified against the 8.7.1 headers): `URL=10002`, `WRITEFUNCTION=20011`,
`WRITEDATA=10001`, `POSTFIELDS=10015`, `POSTFIELDSIZE=60`, `HTTPHEADER=10023`,
`CUSTOMREQUEST=10036`, `TIMEOUT_MS=155`, `NOSIGNAL=99`, `RESPONSE_CODE=2097154`. The transport
CURLcode is read from the `CURLMsg` returned by `curl_multi_info_read` at the verified layout
offset (`result` @ 16).

## `HttpResponse` is a real arena entity (the delightful part)

`HttpResponse` is **not** a special serializer box — it is a first-class arena-backed entity, the
same machinery as any user `class`. Registered `DK_CLASS`, it gets an entity `MonoType` (fields
`status: Int`, `body: String`), and `build-program` builds it a `TypeInfo` + `TypeArena` exactly
like `Agent` or `Conversation`. `do-builtin` allocates each response with `arena-alloc` (stamping
the header: type-id / slot-index / generation / live) and stores the two fields at their decl
offsets (`status` @ header+0, `body` @ header+8). Consequences:

- `--dump-arenas` shows an `HttpResponse` arena (`live=N`, `slot-size=80`).
- Reflection works: `HttpResponse.instances()` returns each response fully serialized —
  `{ "type":"HttpResponse", "ref":"HttpResponse#0", "fields":{ "status":…, "body":… } }` — so
  **every HTTP response an agent ever received is browsable in the viewer**, with stable identity.
- Because `HttpResponse` is registered unconditionally, its arena is always present (like any
  built-in type) even for programs that never call `Http.request` — two `run-arenas` goldens were
  updated to include the (`live=0`) line.

`vm.coil` finds the `TypeInfo` by name at request time (`find-typeinfo-by-name` scans the
program's types once; a local `vm-name-eq` avoids pulling `types.coil` into `vm.coil`).

## Tests (2 new; 251 total)

- **`http_network` (real-network gate, `tests/http/get.scry`)** — a real HTTPS GET to
  `api.anthropic.com/`; asserts a genuine HTTP status (200/401/403/404 all prove DNS+TLS+parse).
  Verified live: **status 404**. Then, only if `ANTHROPIC_API_KEY` is set, it writes a throwaway
  temp `.scry` (key read from env, file deleted immediately — never committed) that POSTs a minimal
  `/v1/messages` request and asserts **status 200 + body contains `"content"`**. The POST *path*
  (custom headers + request body + response capture) is verified live against the API returning a
  real **401** JSON error with a bogus key. **SKIPPED LOUDLY** when offline / no key.
- **`http_stw` (cooperative-STW gate, `tests/http/ping_thread.scry`)** — a background OS thread
  hammers HTTPS requests in a loop while `main` parks in `join()`. The harness `POST`s `types()` /
  `HttpPinger.instances()` through the eval channel *during* the loop and asserts each round-trips
  in **< 1s** and that `HttpPinger.count` climbs — a prompt reply mid-request proves the HTTP thread
  parked for the STW within one 50ms slice. Always runs a **structural** check (that
  `http-perform`'s loop body calls `safepoint-poll`) as the offline fallback + belt-and-suspenders;
  the live half is SKIPPED LOUDLY when offline.

## What 8b / 8c build on

- **8b Env + Json.** `Env.get("ANTHROPIC_API_KEY")` removes the temp-file dance in tests and lets a
  real app read its key. A `Json` parser turns `HttpResponse.body` into a navigable value (extract
  `content[0].text` from a `/v1/messages` reply). Both are pure additions; `Http` is unchanged.
- **8c AnthropicModel.** Implement the existing `Model` interface (`fn complete(prompt) -> String`)
  by building the headers/body, calling `Http.request`, checking `status` (0 ⇒ transport error;
  non-2xx ⇒ API error surfaced honestly), and `Json`-extracting the completion. Because
  `HttpResponse` is arena-visible, every model call the assistant makes is already browsable in the
  viewer with zero extra work — the agent's actual network history, live.

## Coil / libcurl friction (adds to Phase 1-7's list)

- **Variadic FFI is the clean way to bind `curl_easy_setopt`/`getinfo`.** Declaring them
  `[(ptr i8) i32 ...]` and always passing an 8-byte value (a pointer, or a `long` widened to i64)
  matches curl's `va_arg` dispatch (the option number selects the type) — no per-option wrapper.
- **`curl_multi` over `curl_easy_perform` is non-negotiable for a cooperative runtime.** The blocking
  easy call would defeat the entire STW design; the multi loop is the only shape that keeps a
  network wait as STW-safe as `Clock.sleep`.
- **A builtin type becomes a real entity for free by kinding it `DK_CLASS` with `NK_FIELD` members.**
  The intern/`build-typeinfo`/arena pipeline keys off `is-entity` (= class/object), so
  `HttpResponse` needed no new runtime machinery to be arena-backed and browsable — just the right
  decl kind. This is the reusable recipe for every future built-in value that should be inspectable.
- **`http.coil` takes `(field vt parked)`, not the `VMThread`.** Same decoupling seam Phase 5 noted:
  passing the bare `(ptr i64)` parked word keeps the HTTP module free of a `vm.coil` import, so there
  is no module cycle even though the VM calls into it and it calls back into the safepoint protocol.
