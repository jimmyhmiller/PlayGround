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

# Phase 8b — `Env` builtin + a `Json` library written IN Scry

Phase 8b closes the two remaining gaps between Phase 8a's raw HTTP transport and Phase 8c's real
`AnthropicModel`: reading the API key from the process environment (`Env.get`) and turning an
`HttpResponse.body` string into a navigable value (`Json.parse`). `Env` is a builtin (it needs
libc `getenv`); **`Json` is written entirely in Scry** — dogfood, not a builtin — so it doubles as
the largest Scry program the language runs.

## New builtins (five, same 4-site registration as Phase 8a's `Http`)

| builtin | surface | BLT id | VM handler |
|---------|---------|--------|------------|
| `Env.get` | `object Env { get(name: String) -> Option<String> }` | `BLT_ENV_GET 28` | `vm-env-get`: NUL-terminate a copy of the name, `getenv`, `None` if null else copy the value into an owned buffer → `Some(String)` |
| `String.charCode` | `charCode(i: Int) -> Int` | `BLT_STR_CHARCODE 29` | byte at index `i` as an i64; out of bounds ⇒ `-1` (never traps) |
| `String.toInt` | `toInt() -> Option<Int>` | `BLT_STR_TOINT 30` | strict base-10 signed parse (optional `+/-`, digits, nothing else) |
| `String.toFloat` | `toFloat() -> Option<Float>` | `BLT_STR_TOFLOAT 31` | `strtod` over a NUL-terminated copy; `None` unless the WHOLE string was consumed. Payload is f64 bits |
| `Str.fromCharCode` | `object Str { fromCharCode(code: Int) -> String }` | `BLT_STR_FROMCHARCODE 32` | UTF-8 encode a Unicode codepoint (1–4 bytes) into a fresh `String` |
| `Map.keys` | `keys() -> List<K>` | `BLT_MAP_KEYS 33` | keys in insertion order — the ONLY way to enumerate a `Map` from Scry (the stringifier needs it) |

The four registration sites are identical to `Http`'s: `bytecode.coil` (the `BLT_*` consts),
`typecheck.coil` (`register-builtins` for the `Env`/`Str` objects + the `Map.keys` member; the
`string-method` typechecker for `charCode`/`toInt`/`toFloat`, with a new `ty-option-of` helper that
builds `Option<T>` via `mk-named`), `compile.coil` (`compile-builtin-object`'s `Env`/`Str`
branches, `compile-string-method`, and the `Map` branch of `compile-named-method`), and `vm.coil`
(`do-builtin` + the handler defns). `getenv` is the one new extern, declared in `ioutil.coil`.
Strings are `{len,data}` and **not** NUL-terminated, so anything handed to libc (`getenv`,
`strtod`) gets a malloc'd NUL-terminated copy first.

## The `Json` library — `std/json.scry`

A single Scry module (`module std.json`), imported by relative path (`import
std.json.{Json, JsonValue, jget, ...}`). Scry's import resolver is relative to the *importing*
file's directory, so the one canonical `std/json.scry` at the repo root is reached from
`tests/run/` and `examples/` through committed symlinks (`tests/run/std`, `examples/std` →
`../std`) — no copies.

```
enum JsonValue { JNull ; JBool(Bool) ; JNum(Float) ; JStr(String)
                 ; JArr(List<JsonValue>) ; JObj(Map<String, JsonValue>) }
object Json { fn parse(input: String) -> Result<JsonValue, String>
              fn stringify(v: JsonValue) -> String }
```

`parse` is a recursive-descent parser (a `JsonParser` class holding `src`/`pos`/`n`, driven by the
new `charCode`/`slice`/`toFloat` builtins) over the raw UTF-8 bytes: skip whitespace, then dispatch
on the first byte to object / array / string / number / `true` / `false` / `null`. Strings decode
`\" \\ \/ \b \f \n \r \t` and `\uXXXX` (including UTF-16 surrogate pairs) — runs of ordinary bytes
are `slice`d straight out (preserving raw UTF-8) and escapes are rebuilt with `Str.fromCharCode`.
Numbers become `JNum(Float)` via `String.toFloat` (`strtod`). Every malformed input is an `Err("…
at <byte pos>")` value — **the parser never panics.** `stringify` re-emits canonical JSON (object
members in insertion order via `Map.keys`, whole-valued floats without a fraction).

### Accessor API (what Phase 8c should use)

Free functions in `std/json.scry`, all total (return `Option`/never panic):

| function | signature | purpose |
|----------|-----------|---------|
| `jget` | `(v: JsonValue, key: String) -> Option<JsonValue>` | field of a `JObj` |
| `jindex` | `(v: JsonValue, i: Int) -> Option<JsonValue>` | element of a `JArr` |
| `jstr` | `(v: JsonValue) -> Option<String>` | the `String` of a `JStr` |
| `jint` | `(v: JsonValue) -> Option<Int>` | the truncated `Int` of a `JNum` |
| `jbool` | `(v: JsonValue) -> Option<Bool>` | the `Bool` of a `JBool` |
| `jgetStr` | `(v: JsonValue, key: String) -> Option<String>` | `jget` then `jstr` (the common case) |
| `jgetInt` | `(v: JsonValue, key: String) -> Option<Int>` | `jget` then `jint` (token counts) |

So Phase 8c extracts a completion from a `/v1/messages` body with
`jgetStr(jindex(jget(body,"content")?,0)?, "text")` and reads `jgetStr(body,"stop_reason")`,
`jgetInt(usage,"output_tokens")`, or a `tool_use` block's `name` / `input.location` — exactly the
navigation the `json_anthropic` golden verifies.

## Tests (6 new; 257 total)

`tests/run/` goldens (stdout compared): `env_get` (Some/None structure), `str_numchar`
(`charCode`/`toInt`/`toFloat`/`fromCharCode`), `json_roundtrip` (parse→stringify canonical forms:
scalars, escapes, arrays, nested objects, whitespace-insensitive), `json_errors` (truncated /
trailing-junk / bad-key inputs → `Err` with a position), and **`json_anthropic`** — parses a real
Anthropic messages body and extracts `content[0].text == "hello"`, `stop_reason == "end_turn"`, and
`usage` token counts, plus a `tool_use` body's `name`/`input.location`. A harness function
`run_env_roundtrip_test` injects `SCRY_TEST_VAR` into the child's environment and asserts `Env.get`
reads it back (the golden can't set env vars).

## What Phase 8c consumes

- `Env.get("ANTHROPIC_API_KEY")` → the key, no temp-file dance.
- `Json.parse(response.body)` + the `jget*` accessors → the completion text, stop reason, and usage
  from a real `/v1/messages` reply; the same accessors read `tool_use` blocks for tool-calling.
- All of it is plain Scry values threading through the checker with zero friction.

## Coil / Scry friction (adds to Phase 1–8a's list)

- **The import resolver is relative-only (no project root).** `import std.json` resolves to
  `<importer-dir>/std/json.scry`, so a shared library needs to sit beside each consumer. Committed
  relative symlinks (`tests/run/std`, `examples/std` → `../std`) give the single canonical
  `std/json.scry` one source of truth without copies; a real include path would be the clean fix.
- **Scry statements are newline-separated — no `;`, and assignment is a statement, not an
  expression.** A `match` arm whose body assigns (`Some(v) -> out = …`) must wrap the body in a
  block (`Some(v) -> { out = … }`); a bare `out` parses as the arm and the `=` starts a "new
  pattern". Method calls also don't chain off a call result (`f().len()`), so intermediate values
  bind to locals.
- **A class with an `init` constructs by init-parameter name** (`JsonParser(src: input)`), not by
  field name — the memberwise form is only for classes without an `init`.
- **`Str.fromCharCode` is the missing primitive for building strings from codepoints.** Scry can
  read bytes (`charCode`) but had no way to *construct* a character, so JSON escape handling (and
  any future text synthesis) needed one UTF-8-encoding builtin. Slicing handles the raw-byte runs;
  `fromCharCode` handles the decoded escapes — together they preserve multibyte UTF-8 exactly.
- **`Map` had no enumeration.** `set`/`get`/`containsKey`/`len` can't drive a serializer; `keys() ->
  List<K>` (insertion order) is the minimal addition that lets a Scry program walk a map.

---

# Phase 8c — a TRUE agent loop: a `Model` interface + a real `AnthropicModel`

Phase 8c turns the assistant's canned brain into a **real, model-driven tool-use loop**. A `Model`
interface makes brains interchangeable; `AnthropicModel` does live tool use against `/v1/messages`
(over 8a `Http` + 8b `Json`); `ScriptedModel` is a deterministic offline brain that drives the
**identical** loop so the whole thing is testable with zero network. Everything is written IN Scry,
in the shared module `std/agent.scry`, imported by both `examples/assistant.scry` (the live app)
and `tests/run/agent_loop.scry` (the offline golden).

## The shapes (`std/agent.scry`)

```
interface Model {
  fn complete(prompt: String) -> String                       // one-shot chat (legacy say path)
  fn respond(convo: Conversation, tools: List<Tool>) -> ModelResponse   // one agentic turn
}
interface Tool {
  fn name() -> String ; fn description() -> String
  fn inputSchema() -> String                                  // a JSON object string (input_schema)
  fn run(argsJson: String) -> String                          // receives the tool_use input JSON
}
class ModelResponse { text: String ; toolCalls: List<ToolCall> ; stopReason: String ; contentJson: String }
class ToolCall      { id: String ; name: String ; inputJson: String }
class Message       { role: String ; content: String ; contentJson: String }
class Conversation  { messages: List<Message> ; count: Int }
```

`ModelResponse` is a **class, not an enum**, because Anthropic can return text AND `tool_use`
together in one turn — a flat record captures "some text, plus N tool calls, plus a stop_reason"
without contortions. `contentJson` on both `Message` and `ModelResponse` is the load-bearing field:
it holds the exact JSON for the API `content` field (a quoted string for a user turn, a block array
for an assistant/tool_result turn) so a turn can be **replayed verbatim** on the next request with
`tool_use`/`tool_result` ids preserved.

## THE loop (`runAgentLoop`)

```
append user(task) to convo
loop (bounded by maxIters — hard stop, never a silent infinite loop):
  resp = brain.respond(convo, tools)          # one API round-trip (or one scripted decision)
  append assistant(resp.contentJson) to convo
  if resp.stopReason == "tool_use":
    for each ToolCall: find the Tool by name, run(inputJson), collect a tool_result block
    append user([tool_result blocks]) to convo # ids match the tool_use ids
    continue
  else:
    return resp.text                            # end_turn: the assistant's text is the answer
```

The same function runs whether `brain` is `ScriptedModel` or `AnthropicModel` — that is the whole
point of the `Model` seam. `Agent.runLoop(task)` is a thin wrapper that runs it over the agent's own
`conversation` + `tools`; every `Message`/`ToolCall` is an arena entity, so the loop is
**browsable/pausable/hot-swappable live** in the viewer with zero extra machinery.

## `AnthropicModel` — the real client

`respond` hand-builds the request body (no serializer dependency — `jq` JSON-quotes each string and
tool `inputSchema()`/message `contentJson` splice in directly), sets the three headers
(`x-api-key`, `anthropic-version: 2023-06-01`, `content-type`), `Http.request`-POSTs to
`https://api.anthropic.com/v1/messages`, and:

- `status == 0` → **transport error** surfaced (`stopReason:"error"`, text = curl message).
- non-2xx → **API error** surfaced with the response body (verified live: a bogus key returns a real
  **401** `authentication_error` with a `request_id`; the request body is valid JSON the API parses).
- 2xx → `Json.parse` the body and walk `content[]`: `text` blocks concatenate into `.text`,
  `tool_use` blocks become `ToolCall{id,name,inputJson=Json.stringify(input)}`; `stop_reason` and the
  re-serialized content array (`Json.stringify(content)`) are captured. **Never silent** — every
  failure becomes a visible `stopReason:"error"` message.

Because 8a's `HttpResponse` is itself an arena entity and the multi-loop is STW-cooperative, an agent
thread parked mid-API-call stays live-inspectable for free.

## `ScriptedModel` — the deterministic offline brain

`respond` inspects the last message: a `tool_result` → a final answer that **incorporates the tool
output**; otherwise it keyword-matches the user text and emits a REAL `tool_use` decision (extracting
the integers for `calculate`, the city for `get_weather`) or answers directly via `complete()`. It
emits the same `contentJson` block shapes as the real model, so the loop is exercised end-to-end with
zero network and byte-deterministic output.

## Tools

`CalcTool` (the arg-taking proof: genuine integer `a op b` — the answer `391` only exists if `run`
actually ran on the model-chosen args) and `WeatherTool` (canned per-location) are the two the model
must pick and fill; `ShellTool`/`SearchTool` are canned-but-honest and tolerate a bare-string arg
(the legacy sub-agent path). Each exposes a full `input_schema`, so a real model sees a proper tool
catalog.

## Wiring (`examples/assistant.scry`)

`chooseBrain()` reads `Env.get("ANTHROPIC_API_KEY")` → `AnthropicModel("claude-sonnet-5", …)` when
present, else `ScriptedModel`; the active brain is announced at startup (`brain: …`). Plain input
routes through `Agent.runLoop` (one real agentic turn — the model may call a tool); `research`
keeps the Phase-7 sub-agents on real threads; `loop <task>` spawns a `LoopWorker` that runs a
repeating loop on a background thread with `pause()`/`resume()` for live inspection. The
`Session.suggest` live-edit hook is unchanged; a tool's `run` body can be hot-swapped mid-loop too.

## Tests (3 new; 260 total)

- **Offline golden (`tests/run/agent_loop.scry`, exact stdout).** A `ScriptedModel` + `CalcTool`
  runs the full loop: the model requests `calculate({"a":17,"b":23,"op":"mul"})`, the tool computes
  `391`, the result feeds back, and the final answer contains it. Proves model-driven dispatch +
  result feedback with zero network, deterministically. `tests/app/agent_loop.t` proves the same
  through the interactive app wiring.
- **Live inspection (`agent_liveness`, always runs, scripted).** While a `LoopWorker` runs a
  repeating agent loop on a background thread, the harness POSTs evals: `Message.instances()` climbs;
  `pause()` on the looper `Agent` **freezes** the count; `resume()` restarts it — the Phase-5/7
  STW-cooperative liveness pattern, now over the loop.
- **Online (`agent_online`, gated on `ANTHROPIC_API_KEY` + reachability, SKIPPED LOUDLY otherwise).**
  Drives the app with "what is 17 times 23?"; asserts the live model went `stop_reason: tool_use` →
  `calculate` ran → a final answer contains `391`. The key is read from the environment only, never
  written to disk or committed.

## Coil / Scry friction (adds to Phase 1–8b's list)

- **A bare interface-typed field works** (`model: Model`), not just `List<Tool>` — so an `Agent` can
  hold whichever brain, and one `runLoop` serves both. Interface methods may take/return user classes
  (`respond(Conversation, List<Tool>) -> ModelResponse`), which is what makes the `Model` seam clean.
- **A class only satisfies an interface with an explicit `implements`** — structural conformance is
  not inferred, so `class CalcTool implements Tool` is required even though the methods all match.
- **No `break`/`continue`.** Loops that scan to a sentinel (walk a `JArr` via `jindex` until `None`,
  collect digit runs) use a `var more = true` flag; early exit inside a `fn` uses `return`.
- **Building JSON by hand beats round-tripping through a value tree.** With no cross-module access to
  the `JStr` constructor, a 12-line `jq` (quote + escape `" \ \n \t \r`) plus direct string splicing
  of already-valid JSON (`contentJson`, `inputSchema()`) is simpler and faster than constructing a
  `JsonValue` just to `stringify` it; parsing still uses the full `Json` + `jget*` accessor surface.

# Phase 9 — static schema views: the class graph before it runs (`scry inspect` + graph view)

Per DECISIONS #14. The typechecker already resolves the whole static structure (field-type refs,
`implements`, generic instantiations, enums, method sigs). Phase 9 exposes that structure two ways
that **unify into one view**: a `scry inspect` command that serves a program's schema *without
running it*, and a node-link **class-relationship graph** in the viewer that is the landing view in
both the inspected (pre-run) and the running state — the same nodes simply gain live instance counts
and become drillable once `main()` populates the arenas.

## `scry inspect <file>` — see the code before it runs

`scry-inspect` (src/server.coil) is `scry-serve` **minus `vm-run p` and minus the main-thread
safepoint loop**: it `ctx-init`s, typechecks, `build-program`s (so arenas and the type table exist,
**empty**), starts the eval server, prints the viewer URL plus `(inspect: schema only, program not
running …)`, and then just keeps the process alive serving evals. It never runs `main()`.

- **No language thread is ever registered.** `request-global-stop` therefore acquires the global
  lock instantly (its wait loop iterates over zero registered parked-words) — every eval still runs
  under the same full-STW coordinator path as `scry run`, it just has nothing to park.
- **The whole eval channel works against the empty heap.** `types()`/`schema()`/`fields()`/
  `methods()` return the full static schema with `liveCount: 0`; `Agent.instances()` returns an empty
  list. Mutating/definition evals are *allowed* (there is no running program to disturb) but not the
  point of the mode.
- `scry run` is **unchanged**; `--no-viewer` is accepted by `inspect` too (used by nothing yet, kept
  symmetric).

CLI: `cmd-inspect` in src/main.coil, dispatched on `cstr-eq sub c"inspect"`; usage text updated.

## Schema enrichment — a new `schema()` reflection op (server-resolved edges)

The graph needs precise edges (which field types point at which entity/interface/enum node, through
`List<T>`/`Map<K,V>` element types too), plus interface and enum **nodes** that `types()` never
emitted. Rather than client-side string-parsing of the `ref:`/`list:` type strings — fragile — the
edges are **resolved on the server against the real type table** and shipped as data.

**`types()` is left byte-for-byte unchanged** (the type rail and every existing golden still consume
it). A new `schema()` op (src/serialize.coil `reflect-schema`, wired in src/server.coil
`try-reflection`) returns the full graph in one payload:

```
{ "type":"Schema", "nodes":[
  { "name":"Agent", "kind":"class", "builtin":false, "liveCount":1,
    "implements":[...],
    "fields":[ {"name":"model","type":"ref:Model","refTypes":["Model"]},
               {"name":"tools","type":"list:Tool","refTypes":["Tool"]}, ... ],
    "methods":[ {name,params,returns}, ... ] },
  { "name":"Tool", "kind":"interface", "builtin":true, "liveCount":0,
    "methods":[...], "implementors":["ShellTool","SearchTool","CalcTool","WeatherTool"] },
  { "name":"JsonValue", "kind":"enum", "builtin":false, "liveCount":0,
    "variants":[ {"name":"JArr","payload":["List<JsonValue>"],"refTypes":["JsonValue"]}, ... ] } ] }
```

Additions over `types()`:
- **`kind`** — `class` / `object` / `interface` / `enum` (distinct visual treatment).
- **`refTypes` per field (and per enum variant)** — the graph-node names this type references.
  Computed by `collect-ref-types`, which walks the `Type`: `List`/`Map` are transparent wrappers
  (recurse into element types, don't emit the wrapper), scalars contribute nothing, and any
  class/object/interface/enum named type is emitted (recursing into generic args too, so
  `Inventory<Tool>` → `Tool` and `Map<String,JsonValue>` → `JsonValue`). Edges are therefore exact.
- **interface nodes** (from `DK_INTERFACE`/`DK_BUILTIN_IFACE` decls) with their `methods` and a
  server-computed **`implementors`** array (scan of entity monos whose decl `implements` it).
- **enum nodes** (from `DK_ENUM` decls) with `variants` (name + payload type strings + `refTypes`).
- **`builtin`** (decl has no AST node) — lets the client hide unreferenced stdlib nodes.

`liveCount` reuses the arena live-count (same source as `types()`), so it climbs live in `scry run`
and is 0 in `scry inspect`.

## The graph view (viewer/app.js + style.css, hand-rolled SVG, no new vendor lib)

A new top-level `GraphPane`, selected by a **Graph | List** toggle in the top bar, **defaulting to
Graph** (the better first impression). List is the original rail + table + detail, untouched.

- **Nodes** — one per class/object/interface/enum. `deriveGraph()` builds them + typed edges from
  `schema()`. Distinct treatment per kind: class = blue rounded rect, object = gold (thicker), interface
  = dashed teal stadium (italic label), enum = purple. Each carries a **live-count badge** (dim `0`
  pre-run, accent when climbing). `schema()` is polled every 800 ms exactly like the rail polls
  `types()`.
- **Edges** — `field` (solid, → the field's entity type, incl. through `List`/`Map`/generic element
  types), `implements` (dashed teal), `generic` (dotted, parsed from a mono's own `Name<…>`).
  Arrowheads via SVG markers, clipped to the target node's box (AABB) so the head lands on the border.
- **Layout — deterministic and stable.** `computeLayout()` is a hand-rolled force sim with **zero
  RNG**: seed positions from the node index (golden-angle spread) + a per-kind vertical band
  (interfaces up, classes centre, enums down), then a **fixed 600 iterations** of Fruchterman-Reingold
  (repulsion + edge springs + band gravity) followed each iteration by a **hard AABB overlap-resolution
  pass** (separate any two boxes that still intersect along the axis of least penetration). Same input
  shape ⇒ byte-identical output ⇒ no jitter across reloads. Crucially the layout is `useMemo`'d on a
  **structural signature** (sorted node names/kinds + edges) — *not* on `liveCount` — so positions
  never move when counts tick. Pan (drag) + zoom (wheel), auto-fit once per new structure.
- **Interaction / the unification.** Hover highlights a node's incident edges and dims the rest.
  **Click**: a class/object with `liveCount > 0` drills straight into its existing instance **table**
  (`onOpenType` → switch to List mode + `openTable`, reusing `NavContext`); a node with **no live
  instances** (the inspect state) or an interface/enum shows a **static `NodeCard`** in place — fields,
  methods, variants, implementors, all straight from the `schema()` payload, and, when live, a
  `browse →` button. One graph, two states: that is the point.

Dark-first with a `prefers-color-scheme` light path, reusing the existing viewer CSS variables.

## What the portal phase (DECISIONS #13) wraps — NOW BUILT in Phase 10 (see below)

The portal is a reverse-proxy hub where each `scry run`/`scry inspect` registers `{name, pid, port,
mode, …}` and appears as a card. Phase 9 was built to slot straight in, and Phase 10 realizes it:

- **`schema()` is the registration payload.** A program can hand the portal its `schema()` result at
  register time; the portal can render the class graph as the card's preview **before the program is
  even running** (an `inspect`-registered program is exactly a graph with all-zero badges).
- **The graph is already state-agnostic and self-fetching.** `GraphPane` only needs an eval endpoint
  that answers `schema()`; the portal routes that per-program (same one wire op it proxies for
  everything else). Mounting it per program is: point its `evalSource` at the proxied port. No layout
  recompute on reconnect (structural-signature memo), so a card can poll cheaply for climbing badges.
- Suggested route: the portal owns `/{program}/` and proxies `/{program}/eval`; the viewer's Graph
  view becomes each card's default, and clicking a live node deep-links into that program's List view.

## Tests (1 new py gate + graph beats in ui-smoke; 261 total)

- **`inspect` gate (`run_inspect_test`).** Boots `scry inspect examples/assistant.scry`; asserts the
  schema-only note prints and **`main()` never ran** (no `you> ` prompt, no `delegating`/`goodbye` on
  stdout); `types()` returns the full class set with **every `liveCount` 0**; `schema()` returns
  Agent (class) + Tool (interface) + AgentStatus (enum) nodes at count 0 with a resolved
  Agent→Conversation field `refTypes` edge and the enum's variants; `GET /` serves the viewer HTML and
  `app.js` contains the graph view. Then kills it.
- **`ui-smoke` graph beats (headless Chrome, gated/SKIPPED if absent).** On the landing Graph view,
  asserts nodes for Agent + Conversation, ≥1 interface node and ≥1 enum node, an
  Agent→Conversation field edge (`data-from`/`data-to` on the SVG `<line>`), and that **clicking the
  live Agent node navigates to its instance table** — then falls through into the original
  rail→table→detail→method click-through, still green.
- All 260 prior tests unchanged and green.

## Coil / JS friction (adds to Phase 1–8c's list)

- **`refTypes` had to be server-side, and it was easy** — `collect-ref-types` reuses the exact same
  `Type` walk / `List`/`Map`-by-name special-case as `ty-schema`; passing a `(mut rf)` i64 pointer as
  the "first element?" flag is the same by-reference-local trick used for `al-push!` args.
- **Enum variant payload types resolve through the decl.** `resolve-type (kid-at variant j) d` (the
  same call `enum-payload-type` uses) gives the payload `Type`, so `List<JsonValue>` self-edges fall
  straight out — no special casing for recursive enums.
- **Deterministic layout means NO `Math.random()` anywhere** — the golden-angle seed + fixed
  iteration count + AABB separation give a settled, non-overlapping, reload-stable layout; memoizing on
  a structural signature (never on counts) is what keeps it from jittering when badges tick live.


# Phase 10 — the reverse-proxy PORTAL: one hub, programs pop up as cards (DECISIONS #13)

The UI model changes: instead of juggling a per-program URL, you launch ONE persistent **portal**
(`scry portal`, fixed `http://localhost:7357`) and sit at it. Each `scry run`/`scry inspect` binds an
EPHEMERAL port and REGISTERS with the portal; programs appear as cards when they start and grey out
when they stop. The portal serves the viewer shell and REVERSE-PROXIES the eval channel to the right
program (`POST /p/<id>/eval`). One origin, no port juggling. The eval channel itself is unchanged —
still the only wire op; the portal just routes it. **Crucially the portal is ADDITIVE, never
required:** registration is best-effort, so `scry run` still serves its own viewer standalone exactly
as before if no portal is up.

## The registry + the hub (`src/portal.coil`)

The portal runs **NO VM**. It is pure socket transport + a blocking libcurl, so its accept loop is
single-threaded and serial ⇒ the registry needs **no lock** (all mutation happens on the one accept
thread). The registry is a fixed array of `ProgEntry {id, name, mode, port, pid, status, start-time}`
(mode 0=run / 1=inspect; status 0=running / 1=exited). Routes on the accept thread:

- `GET /api/programs` → **reap-then-serialize**. Before answering it health-probes every *running*
  entry with a fast blocking `GET /` to `127.0.0.1:<port>` (400 ms budget); a dead port (curl status
  0 = connection refused) flips the entry to `status:"exited"` so it visibly greys. Then it emits the
  JSON array `[{id,name,mode,port,pid,status,startTime}, …]`.
- `POST /register {name,pid,port,mode}` → assigns an id, adds the entry, returns `{"id":N}`. The
  portal timestamps `startTime` (its own clock) on receipt.
- `POST /deregister {id}` / `POST /heartbeat {id}` → mark exited / un-grey (both optional; the health
  probe is the real liveness source, so deregister is just a courtesy on clean exit).
- `POST /p/<id>/eval` → **the reverse proxy.** Look the entry up by id, `pc-request POST` the body
  verbatim to `http://127.0.0.1:<port>/eval`, return the program's response verbatim (status + body).
  A dead peer (status 0) greys the entry and returns `502`. `path-prog-id` parses the `<id>` digits
  out of `/p/<id>/eval`.
- `GET` anything else → `serve-static` (the same viewer shell/assets server.coil already serves).

The portal reuses server.coil's HTTP primitives (`send-response`, `serve-static`, `parse-req`,
`find-body-start`, `header-content-length`, `make-sockaddr`) and json.coil's `JBuf`. `portal-bind`
binds **exactly** 7357 (no `+20` scan — the portal must own that port or fail loudly).

## The blocking curl client (`src/portalclient.coil`)

A tiny leaf module (imports only `ioutil`) shared by two callers that both lack a VM: the portal's
proxy and a program's registration. Unlike http.coil's cooperative `curl_multi` loop (which parks on
safepoints because it runs *inside* a mutator), here a plain **blocking `curl_easy_perform`** is
exactly right (added one extern, `curl_easy_perform`, to ioutil.coil). `pc-request method url body
blen timeout-ms outbuf` returns the HTTP status (0 = transport failure) and leaves the body in a
growable `PcBuf`. `portal-register name pid port mode` POSTs to `/register` and parses `{"id":N}`
back; `portal-deregister id` is fire-and-forget. No import cycle: server.coil and portal.coil both
import portalclient; it imports neither.

## Ephemeral ports + best-effort registration (`src/server.coil`)

`scry-serve`/`scry-inspect` now bind starting at **7400** (`bind-server 7400`, +20 scan) so they never
grab the portal's 7357. After the accept thread is up and the direct viewer URL is printed, each calls
`srv-register path port mode`: it POSTs `{name:<basename>, pid:getpid(), port, mode}` to
`127.0.0.1:7357/register`. **If the portal is not running the POST fails and we silently continue** —
the program still serves its own viewer on its ephemeral port. On success it additionally prints
`portal: http://localhost:7357`, so BOTH the direct URL and the portal entry point work.

## Viewer dual-mode (`viewer/app.js` + `style.css`)

One app, two modes, decided once at boot by a `Root` component:

- A module-level `evalBase` (default `""`) is prepended to every `evalSource` POST → `${evalBase}/eval`.
  Standalone leaves it `""` (today's `/eval`); clicking into a program through the portal sets it to
  `/p/<id>` so **every existing pane works unchanged through the proxy** — nothing below `evalSource`
  knows the difference.
- `Root` probes `GET /api/programs` once: **200 ⇒ portal mode** (show the landing grid); **404 ⇒
  standalone** (a program serving its own viewer directly) ⇒ render `<App/>` immediately with
  `evalBase=""`, exactly as before.
- The **`Landing`** grid polls `/api/programs` every ~1 s so cards POP UP when a program launches and
  grey when it exits (`.pcard.exited`). Each `ProgramCard` shows the name, a mode badge
  (running/inspect), a live/exited status dot, the port, start time, and cheap live stats (one
  `types()` through the proxy → instance + type counts). A subtle `pcard-in` enter animation makes new
  cards feel alive.
- Clicking a card sets `evalBase="/p/<id>"` and mounts `<App key=<id> onBack=… programName=…/>` — the
  existing inspector, Graph view default (Phase 9), now driven entirely through the proxy. `App` grows
  an optional `← portal` back button (in `TopBar`) that clears `evalBase` and returns to the grid.

## Tests (1 new py gate + a portal scenario in ui-smoke; 262 total)

- **`run_portal_test`.** Starts `scry portal`; asserts `GET /api/programs` is `[]`; launches
  `scry run examples/demo-mini.scry` and asserts it appears within ~4 s with `mode:"run"` and an
  ephemeral port ≥ 7400; `POST /p/<id>/eval {source:"types()"}` **proxies and returns the real schema
  JSON** with the id echoed (proves the proxy); launches `scry inspect examples/agents.scry` for a
  SECOND entry with `mode:"inspect"`; **kills** the run program and asserts it greys to
  `status:"exited"` within the reap window; `GET /` serves the viewer HTML. SKIPPED LOUDLY if :7357 is
  already in use (a developer's own portal), never clobbering it.
- **`ui-smoke` portal beats (headless Chrome, gated).** After the standalone graph/click-through
  beats, boots a portal + a program, navigates the page to `http://localhost:7357/`, asserts a program
  **card renders**, clicks it, asserts the **inspector graph loads through the proxy** (Agent node
  present), and that the `← portal` back button returns to the grid.
- All 261 prior tests unchanged and green (they hit programs directly on their ephemeral ports, not
  through the portal — standalone stays first-class).

## Coil / JS friction (adds to Phase 1–9's list)

- **A single-threaded accept loop dissolves the "mutex-guarded registry" requirement.** The portal
  runs no mutator, accepts one connection at a time, and does all registry reads/writes/reaps on that
  one thread — so the obvious concurrency worry simply isn't there. The lock the brief suggested would
  have guarded nothing.
- **Health-probe-on-poll beats a heartbeat timer.** Because the viewer already polls `/api/programs`
  every second, hanging the reap off that GET (probe each running port) needs no background thread and
  no clock skew — a dead program greys on the very next poll. Deregister/heartbeat exist but are
  belt-and-suspenders.
- **`bind-server 7357`→`7400` was the whole "port juggling" fix** — one constant. Because
  `bind-server` already scans `+20`, two programs on one machine transparently land on 7400/7401/… and
  each registers its real bound port, so the proxy always targets the right process.

---

# Phase V1 — the bespoke NESTED-CONTAINMENT view (DECISIONS #15a)

The off-the-shelf force-directed instance graph is **rejected** (DECISIONS #15, emphatic). Phase V1
replaces the default live view with a hand-built, deterministic **nested-containment** instrument
panel where **ownership becomes nesting** and **live instance count becomes mass**, ported from the
signed-off mockup (`scratchpad/scry-bespoke-views.html`). The old `GraphPane`/`deriveGraph`/
`computeLayout` (the Fruchterman-Reingold sim) are **deleted**. Two modes remain: **Map** (this view,
the default landing) and **List** (the rail + table/detail browse). The static class graph of Phase 9
is superseded as the landing view by this.

## The `graph()` reflection op (`src/serialize.coil` `reflect-graph`, wired in `src/server.coil`)

One new eval op powers the whole view. It walks **every entity arena** and emits one compact record
per LIVE instance so the client can compute ownership vs sharing entirely on its own:

```json
{ "type":"Graph", "instances":[
  { "ref":"Agent#0", "type":"Agent", "generation":0,
    "scalars": { "name":{"type":"String","value":"assistant"},
                 "role":{"type":"String","value":"orchestrator"},
                 "status":{"type":"AgentStatus","case":"Idle"} },
    "refs": [ {"field":"conversation","list":false,"ids":["Conversation#0"]},
              {"field":"model","list":false,"ids":["ScriptedModel#0"]},
              {"field":"tools","list":true,"ids":["CalcTool#0","WeatherTool#0","ShellTool#0","SearchTool#0"]} ] } ] }
```

- **`scalars`** — only scalar/enum LEAF fields (Int/Bool/Float/String + enums), inlined via the
  existing `serialize-value` at depth 1. `List`/`Map`-of-scalar and entity fields are excluded here.
- **`refs`** — one entry per entity-typed field: a single entity ref (`list:false`) or a
  `List<Entity>` (`list:true`). Every id is the **concrete runtime `Type#slot`** recovered from the
  *target's own* `InstanceHeader`, so an interface- or `List<T>`-typed field resolves to real
  implementors (`Agent.tools : List<Tool>` → `["CalcTool#0", …]`).
- Registered exactly like `schema()`: a `graph()` branch in `try-reflection` + `reflect-graph` in
  serialize.coil. `types()`/`schema()`/`instances()` and their goldens are byte-for-byte untouched.

**Why one op, not cascading `T.instances()`:** a single `graph()` is one round-trip that carries the
whole live reference structure with stable ids; the alternative (one `instances()` per type, then
re-resolving refs client-side) is N chatty round-trips per poll and re-derives what the server already
knows. The client still ALSO polls `schema()` (unchanged) for the static SHAPE — interface
implementors, per-field `refTypes`, and climbing per-type `liveCount` — which `graph()` deliberately
does not carry. So: `schema()` = static skeleton + counts, `graph()` = live instance graph. Two small
evals per 800 ms poll; the census/mass reads `schema().liveCount` (climbs live), the nesting reads
`graph()`.

## Ownership / sharing / infrastructure rules (all client-side, `computeNested` in `viewer/app.js`)

Pure and deterministic — same program → same layout, positions derive from the stable structure, not
from counts, so climbing counts grow stacks/bars **in place** and never reflow.

1. **Static domain-type set.** Build the type-reference graph from `schema()` (`refTypes` per field),
   expanding each interface to its `implementors`. The **primary domain root type** is the
   *unreferenced container* whose reachable set is largest; ties broken by total live instances in
   that reach (so a real, populated container like `Orchestrator` beats an empty, same-shaped worker,
   and a small side-tree like `ModelResponse→ToolCall` can never win). **Worker types** — those
   implementing a *builtin* interface (`Runnable`, the thread-body contract) — are excluded as root
   candidates: they hold a domain object to *run* it, they are not the domain container. `domainTypes`
   = the reach set of the chosen root.
2. **Ownership = nesting.** Among domain instances only, count distinct owners per target (an entity
   ref edge owner→target). Restricting to domain owners is what makes a sub-agent held by BOTH the
   `Orchestrator` (`subs`) and its (infra) `SubAgentWorker` still nest under the Orchestrator instead
   of counting as shared. A target with **exactly one** domain owner **nests** inside it.
3. **Size = mass.** The census ribbon bars use the mockup's `pow(n/max, 0.72)` compression (big masses
   dominate, small ones stay visible); a Conversation's message stack renders **one row per Message**,
   so its height literally IS its mass — 29 Messages visibly dominate 3 Agents.
4. **Shared = identity color, not nesting.** A domain instance with **≥2 distinct domain owners** (the
   4 tools, held by the Orchestrator and every Agent) is never nested; it gets a stable identity slot
   (`id → sorted-index mod 8`, palette `--id-0..--id-7`) and renders as that colored chip **everywhere**
   it is referenced, plus in the legend. Hovering any chip adds `id-active` to the view and `id-hi` to
   *every* appearance of that exact id (imperative toggle, survives poll re-renders). The optional
   **show links** toggle draws the mockup's hub-and-spoke faded SVG connectors between appearances.
5. **Infrastructure recedes.** Entity types **not** in `domainTypes` (and not singleton-objects) with a
   live count — `ModelResponse`, `ToolCall`, `HttpResponse`, `JsonParser`, the `Runnable` workers — sit
   in a faded, collapsible **infrastructure strip** with live counts; click a util type to browse it in
   List mode. The rule is principled (reachability from the domain root, not a class-name allowlist);
   **V2's `view`/hidden construct will let the program override it.**
6. **Singleton objects.** An unreferenced leaf with no entity fields and exactly 1 live instance
   (`Session`, the std `Json` object) renders as a small `obj` node in the primary root's header, not
   the infra strip.
7. **Interaction / live.** Clicking a region or chip opens the existing detail (switches to browse mode,
   reuses `NavContext.openDetail`). Fresh Messages get the mockup's soft-teal pulse; live census rows
   flag `▲` when a count climbs; everything respects `prefers-reduced-motion`. Dark-first and fully
   theme-aware — `prefers-color-scheme` AND an explicit `data-theme` toggle both switch base + identity
   palettes in both directions (the viewer's `data-theme` blocks were extended to restate the base
   palette so an explicit toggle can always win over the OS media query).

## What V2's `view` construct will layer on (DECISIONS #15b)

The default view above is the substrate. V2 adds the first-class program-declared `view Name for T {
title; size; badge; section "…" { field as timeline|chips|rows|card|heat } }`: reflection surfaces the
spec + the fields it needs, and a per-view bespoke renderer honors it (the mockup's Act III `AgentBoard`
and Act IV `Pulse`). A `hidden`/`view`-level override will also let a program *reclassify* a type the
heuristic put in the infra strip (or force something out of the default nesting) — the reachability rule
here is the sensible default it overrides, not a hard wall.

## Tests

- `tests/eval/60-graph.t` — a `graph()` golden on `examples/demo-mini.scry`: asserts the `Graph`
  payload, an `Agent`'s `conversation`/`model`/`tools` ref ids, a `Conversation`→`Inventory<Message>`
  nesting, the shared `["ShellTool#0","SearchTool#0"]` tool list appearing under multiple owners, and a
  `Message`'s inlined `role` enum scalar.
- `tests/ui-smoke.mjs` — the graph beats are replaced by nested-view beats (Chrome-gated, skips loud
  if absent): the census renders; an `Agent` region nests a `Conversation` that renders message rows;
  a shared tool chip appears across ≥2 owner regions with **one** stable identity color; hovering it
  lights up all appearances (`id-active` + ≥2 `id-hi`); the infra strip renders and expands; clicking a
  region drills into detail. The portal + inspect scenarios now assert the proxied/empty **nested** view.
- All 264 prior tests remain green (265 total with the new golden); `schema()`/`types()`/`instances()`
  goldens unchanged.

## Coil / JS friction (adds to Phase 1–10's list)

- **`graph()` is ~90 lines of straight arena-walking Coil, zero new machinery.** It reuses the exact
  live-slot scan pattern from `reflect.coil`'s `arena-instances` (`bump-cursor` high-water + `FLAG_LIVE`)
  and the `serialize-value`/`emit-ref-id`/`mono-by-id` helpers already in serialize.coil — the only new
  logic is classifying a field's declared type (scalar-leaf vs single-entity vs `List<entity>`), a
  three-line predicate over `ty-kind`/decl-kind.
- **Real live data is messier than the mockup, and the fix was a heuristic, not a special case.** Two
  places bit: (1) `Agent` has its own `role` field, so a naïve "has a `role` scalar ⇒ it's a Message"
  test swallowed Agents into message stacks — fixed by requiring a *body* scalar (`content`/`text`)
  too; (2) `research` spawns `SubAgentWorker`s that hold the sub-`Agent`s, so those agents have two
  owners and the workers are unreferenced containers that tie the real `Orchestrator` on reach —
  fixed by excluding builtin-interface (`Runnable`) workers from domain roots, which drops both the
  worker and its second-owner edge, restoring clean nesting. Both fixes are data-driven signals
  (field shape, builtin-interface conformance), not class-name allowlists.
- **React `onMouseEnter` fires on native `mouseover`.** The headless-Chrome hover beat had to dispatch
  `mouseover` (not `mouseenter`) to trip React's synthetic enter handler.

# Phase V2 — program-declared `view` construct (DECISIONS #15b)

V2 makes "the program speaks to how it ought to be visualized" real: a first-class `view` toplevel
declaration, checked against its target type, exposed by reflection, and honored by a bespoke renderer
that the viewer flips to via a per-instance DEFAULT↔CUSTOM toggle. Built on V1's nested default (the
substrate) — the mockup's Act III `AgentBoard` and Act IV `Pulse`.

## Grammar (as implemented)
```
view <Name> for <Type> {
  <clause>*
}
```
where a clause is either a **keyed clause** `<key>: <item>` (e.g. `title: name`, `size: byCount`,
`badge: status`, `key: role`, `stream: all as heat(...)`) or a **section** `section "<label>" { <item> }`.
An **item** is a dotted field-path with an optional representation: `<field-path> [ as <repr>(<arg>*) ]`,
and an **arg** is `<name>: <ident>` (e.g. `order: createdAt`, `by: role`). Representations: `timeline`,
`chips`, `rows`, `card`, `heat`. Two special single-hop paths: `all` (all instances of the target type,
top-level clauses only) and `byCount` (a count, `size:` only). `view` is the only new reserved keyword;
`section`/`title`/`size`/`badge`/`timeline`/… stay contextual identifiers (`for`/`as` already existed),
so no existing program breaks. `parse-dump` renders views as `(view N (for (type T)) (clause …)
(section "…" (item (path a b) (as repr) (arg …))))`.

- **Lexer** (`lexer.coil`): one keyword, `TK_VIEW`.
- **AST** (`ast.coil`): `NK_VIEW` (s-name = name; slot a = `NK_TYPE` target; kids = clauses),
  `NK_VIEW_CLAUSE` (s-name = key; a = item), `NK_VIEW_SECTION` (s-name = label; a = item),
  `NK_VIEW_ITEM` (a = `NK_VIEW_PATH`; s-name = representation, empty if none; kids = `NK_ARG`),
  `NK_VIEW_PATH` (kids = `NK_IDENT` hops). A view node also stashes its source file pointer in
  `ival` (cast) so `check-view` can set the diagnostic file.
- **Parser** (`parser.coil`): `parse-view` + `parse-toplevel` dispatch on `TK_VIEW`. Sections are
  detected by a contextual `section` ident followed by a string (`cur-is-section?`), so a clause
  literally named `section:` is still possible.

## Typecheck rules (`typecheck.coil`, pass 3 `check-all-views`)
Views collected (entry module only, `deep`) into `Ctx.views`, checked AFTER signatures+bodies so monos
exist. Per view: the target `T` must be a declared entity **class/object** (interface/enum/unknown =
error). Then per clause/section: the field-path resolves hop-by-hop from `T` (each intermediate hop must
be an entity to descend; unknown field or descend-into-non-entity = span'd error). `title:` must resolve
to `String`; `badge:` to `String` or an enum; `size:` is `byCount` or any resolvable field. A
representation is validated against the resolved type: `timeline`/`rows`/`chips`/`heat` require a
`List<…>`, `card` a single entity, anything else = "unknown representation". `order:`/`by:` args must
name a field on the list's element type. Every failure carries a file:line:col span. Check goldens:
`tests/check/50-view-ok` (success) + `e39`–`e45` (unknown type, unknown field, bad repr, timeline on a
non-list, non-String title, bad order field, interface target).

## Reflection `views()` (`serialize.coil` `reflect-views`, wired in `server.coil` `try-reflection`)
Static (from the parsed+checked program, read off `Ctx.views`); registered exactly like `schema()`/
`graph()`. Payload: `{ "type":"Views", "views":[ { "name", "target", "clauses":[ … ] } ] }`, one clause
entry per top-level clause OR section, each carrying its dotted `path`, `representation` (or `null`), and
`args:[{name,value}]`:
```json
{"kind":"clause","key":"title","path":"name","representation":null,"args":[]}
{"kind":"section","label":"conversation","path":"conversation.messages","representation":"timeline","args":[]}
{"kind":"clause","key":"stream","path":"all","representation":"heat","args":[{"name":"by","value":"role"}]}
```
Eval golden: `tests/eval/72-views` on `examples/assistant.scry`.

## Viewer: declared-view rendering + the toggle (`viewer/app.js`, `viewer/style.css`)
`NestedView` now polls `views()` alongside `graph()`+`schema()` (three evals per 800 ms) and builds
`viewsByType` (first view per target). A `Region` whose instance type has a declared view shows a
`▤ cell / ▧ board` toggle (per-instance mode in `viewMode`, keyed by instance id); DEFAULT stays the V1
nested cell, CUSTOM renders `BoardView`. `BoardView` reads the spec's `title`/`badge` scalars for the
header and renders each section via `<Representation>`, which resolves the item's instances by
**following the field-path over the same `graph()` records** (`followPath`: hop → each instance's
`refs[field].ids`) — or, for `all`, every instance of the target type. Representations:
`timeline` = ordered role-tinted message rows (reuses V1's `roleClass`/`--m-*`), `chips` = identity
chips (shared → V1 `IdChip`, else a solo chip; both open detail on click), `rows` = compact list,
`card` = a scalar field grid, `heat` = role-keyed colored cells with a legend. Live (re-eval), fully
theme-aware (reuses V1's tokens/palette). `ui-smoke.mjs` gained beats: the Agent region shows the
toggle; toggling renders the AgentBoard timeline rows + tool chips; toggling back restores the nested
cell. Verified in real headless Chrome against `examples/assistant.scry` — the three agent cells flip to
titled boards with role-colored timelines and identity-colored tool chips, matching the mockup's Act III.

## Example views (`examples/assistant.scry`)
`view AgentBoard for Agent { title: name; size: byCount; badge: status;
section "conversation" { conversation.messages as timeline }; section "tools" { tools as chips } }` and
`view Pulse for Message { stream: all as heat(by: role); key: role }`. **Ordering decision:** `Message`
has no `createdAt`/timestamp and a `Conversation.messages` is an ordered `List`, so `timeline` uses
natural insertion order (no `order:` arg) rather than synthesizing a fake ordinal — the honest, clean
choice. (`order:` remains supported + typechecked; the parse golden `tests/parse/60-view` exercises it.)

# Phase V3 — the STATIC type-level view (bespoke skeleton + declared-view templates)

`scry inspect <file>` typechecks + serves the schema WITHOUT running `main()`, so every arena
exists but is EMPTY (0 instances). V1/V2 already had all the static structure in hand — `schema()`
carries the full type-reference graph (fields/`refTypes`/`implementors`) and `views()` the declared
specs — but the Map view only *drew* them from live instances, so inspect hit a dead-end ("no live
instances yet — run the program"). That was backwards: per DECISIONS #14, the static view is the
TEMPLATE that live data fills in, and per #15 the custom views ("do the custom views not handle the
static view as well?") should honor it too. V3 makes the bespoke nested view AND the program-declared
`view`s render **statically, from TYPES**, in `scry inspect`.

## Integration approach chosen: 0-instance fallback (NOT the unified base-skeleton)

The brief offered two integrations and PREFERRED the unified one (NestedView always renders a
type-skeleton base that live instances populate). **I chose the 0-instance fallback** and it is the
principled call, not merely the safe one:

- The live V1 view is fundamentally **instance-level** and is genuinely better for live data: an
  `Agent "assistant"` region owning a `Conversation` whose height IS its 29-Message mass, with tools
  as per-*instance* identity chips (`ShellTool#0`) hover-linked across owners. Rebuilding that as
  "Agent-type cell ▸ [Agent#0] ▸ Conversation-type cell ▸ [Conversation#0] ▸ Message-type cell ▸ 29
  rows" would **regress the V1 appearance** (the explicit bar: V1 screenshots). Doing the unified
  version *without* that regression is a large, risky refactor of the exact code the demo depends on.
- So: `NestedView` computes BOTH models each poll — `computeNested(instances, schema)` (live, V1/V2,
  untouched) and `computeTypeSkeleton(schema)` (new, type-level). It renders the skeleton **iff
  `instances.length === 0` and the skeleton has a domain root**; otherwise the live path renders
  exactly as before. This still realizes "static is the template live fills in": at `scry run`
  startup there is a split-second where `graph()` is empty and you literally see the template, which
  then fills as `main()` populates the arenas — the same view at two fill levels, with zero live
  regression. The live/skeleton switch is a single `showSkeleton` boolean; the live branch is
  byte-for-byte the old code.

## The type-containment skeleton model (`computeTypeSkeleton` in `viewer/app.js`)

Pure, deterministic, `schema()`-only — mirrors `computeNested`'s static rules at the TYPE level:

1. **Static reference graph + domain root.** Same as V1: build `refOut` (type → entity types it
   references, interfaces expanded to `implementors`), pick the primary domain-root TYPE as the
   unreferenced non-`Runnable`-worker container with the largest reachable set (ties → name, since
   there are no live counts to break them). `domainTypes` = its reach. For `assistant.scry`:
   `Orchestrator` → `{Agent, Conversation, Message, ScriptedModel, AnthropicModel, Shell/Search/Calc/WeatherTool}`.
2. **Type-level ownership = nesting.** `owners(U)` = domain types whose `refOut` contains `U`. A type
   with **exactly one** domain owner nests inside it (`Orchestrator ▸ Agent ▸ {Conversation ▸ Message,
   the two Model implementors}`). A **message-like** child type (has a `role` scalar AND a
   `content`/`text`/`body` scalar — same shape rule as V1's `isMsgLike`) renders as a labeled
   **placeholder message stack** (ghost `.mrow`s), not a sub-region — so `Conversation ▸ Message`
   reads exactly like the live conversation, minus the data.
3. **Shared TYPES = identity chips.** A domain type with **≥2 distinct owner types** (the 4 tools,
   referenced by both `Agent.tools` and `Orchestrator.tools`) is never nested; it gets a stable
   identity color (`sorted-type-name → index mod 8`, palette `--id-0..7`) and renders as that colored
   chip everywhere it is referenced + in the legend. Reuses the `.chip[data-identity]` contract, so
   V1's imperative hover-highlight lights up every appearance of the same TYPE unchanged.
4. **Singletons + infrastructure.** `object`-kind entity types (the std `Json`) render as `obj`
   chips in the header — a class can't be *known* a singleton statically (that's a runtime count of
   1), so classes like `Session` recede to the faded **infrastructure strip** alongside the other
   non-domain entity types (`ModelResponse`, `ToolCall`, `HttpResponse`, `JsonParser`, the workers).
5. **Census with 0 counts.** The mass ribbon lists every entity type (domain first, then singletons,
   then infra) with a **dashed empty track** and `×—` instead of bars — "types present, mass fills at
   runtime". A "SCHEMA · NOT RUNNING" affordance sits in the header (echoing the inspect banner).

`computeTypeSkeleton` returns the same *shape* of model the renderers expect (`roots`, `sharedTypes`,
`idColor`, `infraTypes`, `census`, `singletonTypes`, plus `byName`/`expand`/`isMsgType` for the
template path-follower). Parallel renderers `TypeRegion`/`TypeChip` reuse V1's CSS vocabulary
(`.region`, `.conv`, `.mstack`, `.refs`, `.chip`) with a `tmpl` modifier for the muted/dashed ghost
styling.

## Declared views as static templates (`TypeBoardView`/`TypeRepresentation`)

The `▤ cell / ▧ board` toggle works in inspect too (per-type mode, keyed by type name — no collision
with the live per-instance-id mode). Toggling an `Agent` type-cell to board renders `AgentBoard` as a
**template against the TYPE**:

- **Header wiring.** `title`/`badge` show their FIELD-NAME source, not a value: `Agent  title ⟵ name`
  and a ghost `badge ⟵ status` — so before running you SEE how an Agent will be presented.
- **Sections resolve the element TYPE** via `followTypePath` (the type-graph analog of V2's
  `followPath`: hop → each type's field `refTypes`, interfaces expanded). `conversation.messages as
  timeline` resolves to `Message` and renders its shape (`Message ⟵ role, content, contentJson`) plus
  a few role-tinted placeholder rows (`role · content ⟵ Message`); `tools as chips` resolves to the
  4 Tool implementor TYPES and renders them as identity chips; `card`/`rows`/`heat` likewise render
  against the element type's shape. When instances exist, V2's live `BoardView` renders real data,
  entirely unchanged.

## Tests + screenshots

- `tests/ui-smoke.mjs` — the inspect scenario was rewritten from "assert `.stage-empty` dead-end" to
  assert the SKELETON: `#nested.skeleton` renders; `Agent` type-cell nests `Conversation` nests a
  Message placeholder stack; shared Tool type chips appear; the infra strip renders; the `AgentBoard`
  toggle flips to a template with title/badge field-name wiring + a timeline placeholder + 4
  tool-type chips. The run-mode + portal beats are unchanged and still green.
- All **275** tests pass; build clean (viewer is a no-build JS/CSS + one test + this doc — no `scry`
  recompile, no `schema()`/`views()`/`graph()` payload change; V3 needed **no runtime change**, the
  static payloads already carried everything).
- Screenshots: `scratchpad/probe-inspect-nested.png` (type skeleton) and
  `scratchpad/probe-inspect-board.png` (AgentBoard template) — both match the mockup's aesthetic,
  drawn from types instead of instances.

## Friction / notes

- **No runtime change needed.** `schema()` already carries `kind`/`fields`/`refTypes`/`implementors`/
  `implements`/`builtin` and `views()` the full clause specs — everything the type view needs. The
  whole phase is client-side (`computeTypeSkeleton` + `TypeRegion`/`TypeBoardView`/`TypeRepresentation`
  + the `showSkeleton` switch in `NestedView`), plus `tmpl` CSS.
- **Statically un-knowable singletons.** `Session` is a 1-of at runtime but a `class` in the schema;
  without counts there's no honest way to call it a singleton, so it sits in infra in inspect and
  moves to the header singleton row only once the live view sees `liveCount === 1`. Called out rather
  than faked.
- **The two Model implementors** (`ScriptedModel`, `AnthropicModel`) both nest as leaf cells under
  `Agent` (each has one owner type, `Agent.model`), following the same ≥2-owners rule as everything
  else rather than a special case for interface-typed fields — consistent with the live algorithm.

# Phase V4 — drill into ALL details from WITHIN the Map (the in-map inspector)

Before V4 the only way to see an instance's full detail from the Map was `onNestOpenDetail`, which
did `setMode("browse")` — it **threw you out of the Map** into the List/detail screen, losing the map
and your place. And the dense message-stack rows in a Conversation cell were purely decorative — you
could not click an individual `Message` to inspect it. V4 makes every entity in the Map drillable
**without ever leaving the Map**.

## The UX: a right-docked inspector column (not an overlay drawer)

Clicking any entity opens a **docked inspector column** to the right of the nested stage — the
"IDE inspector" choice — rather than an overlay drawer. Reasoning: the whole point is to keep the
map *and* the detail visible at once; an overlay would cover exactly the map you're trying to keep
your place in. When the inspector is open, `#layout` (class `nested-layout has-inspector`) becomes a
flex row: the map stage scrolls on the left (`.nested-stage-col`, `#nested` max-width tightened to
980px), the inspector is a fixed `flex: 0 0 460px` column on the right with its own scroll and a
sticky header (breadcrumb + back `←` + close `✕`). Closing returns the map to full width.

## What became drillable (all of it → the inspector, never browse)

- **Region head** (instance cell) — `Region`'s `onOpen` now routes to the inspector.
- **Message-stack rows** (`.mrow.drill` in a Conversation cell) — previously inert; each row is now a
  real click target to *that specific* `Message` instance (`refParts(c.id)` + its generation).
- **Board-view (declared `view`) rows** — `timeline` rows (`.tl.drill`) and `heat` cells
  (`.hcell.drill`) are now clickable to their underlying `Message`; `chips`/`rows` reps already
  opened detail and now target the inspector too.
- **Identity chips & tool chips** (shared instances) and **singleton obj-chips** — inspector.
- **Infra utility buttons** — open that TYPE's static detail in the inspector (in-map, graceful),
  instead of jumping to the browse table.
- **STATIC / inspect mode** (V3 type templates, no instances) — clicking a TYPE cell/chip opens that
  TYPE's static detail (`TypeStaticDetail`: fields, method signatures, implementors) in the same
  inspector. No instance to poll, so it shows the shape (`⟵ runtime` field values) — never an error.

## Reuse (nothing reinvented)

- **`DetailPane` verbatim.** The instance branch of the inspector renders the *exact* `DetailPane`
  the browse screen uses — same `at(slot,gen)` 750ms poll, same field flash-diffing, same
  `detailBus` refresh-after-invoke, same `✎ edit source` (`openCodePanel`). No refactor of its body
  was needed; it already took `cls/slot/gen/schema/onEditSource` as props.
- **`MethodCard` / invoke** verbatim inside that `DetailPane` — signatures, inline arg forms,
  results/errors, and the immediate `bumpDetail` read-back all work unchanged in the inspector.
- **Reference navigation is an in-inspector back-stack.** The inspector holds a stack of targets
  (`App.inspect = { stack: [...] }`). It wraps its body in a `NavContext.Provider` whose
  `navigateRef` **pushes** onto the stack instead of switching to browse — so every `RefLink`
  (field values, method results) navigates *within* the inspector and the breadcrumb grows; crumbs
  are clickable to truncate the stack, `←` pops.
- **The instance-scoped REPL reuses the existing bottom `ReplDock`.** While the inspector shows an
  instance in Map mode, `App` feeds `ReplDock` a `replRoute` of `{view:"detail", ref:…}` bound to the
  inspector's current target — so `self` = the open instance and post-eval `bumpDetail` refreshes the
  inspector, with zero changes to the dock component.
- **Selection highlight reuses the identity mechanism.** A `selectedId` effect in `NestedView` rings
  every appearance of the inspected instance in the map (`.sel-inspect` on its `[data-region]`,
  `[data-mrow]`/`[data-hcell]`, and `.chip[data-identity]`), mirroring the hover `id-hi` effect.

## Structure

`NestedView` no longer takes browse-routing props; it takes `onInspect(target)` + `selectedId` and
translates region/chip/mrow/type clicks into targets. `App` owns the `inspect` stack and the
`openInspect`/`inspectPush`/`inspectGoto`/`inspectBack`/`inspectClose` handlers; switching to List
(`changeMode`) closes the inspector (it belongs to the Map). Two new components:
`InspectorPanel` (header + nav-override + dispatch to `DetailPane`/`TypeStaticDetail`) and
`TypeStaticDetail` (static type card).

## Tests + screenshot

- `tests/ui-smoke.mjs` gained V4 beats (live): clicking an Agent region opens the docked inspector
  **while staying in the Map** (`.nested-layout.has-inspector #inspector` present, `#nested` still
  there); the inspector shows the Agent's fields + methods; the instance is ringed in the map
  (`.sel-inspect`); a 0-arg invoke from the inspector produces a result; a reference field grows the
  inspector breadcrumb; a `.mrow.drill` message row opens that `Message`'s detail; closing restores
  full width. Plus an inspect-mode beat: clicking a TYPE cell opens its static detail (fields) with
  the skeleton map still visible, no error.
- All **275** tests pass (List/browse + `DetailPane` + REPL unchanged; run + inspect ui-smoke green).
  Viewer is no-build JS/CSS — no `scry` recompile, no reflection-payload change.
- Screenshot: `docs/v4-inspector.png` — the Map with the Agent inspector docked open (fields incl.
  clickable `model`/`conversation` refs and tool chips, methods with invoke buttons) and the Agent
  region ringed in the still-visible map.

## Friction / notes

- **No runtime change needed** — V4 is entirely `viewer/app.js` + `viewer/style.css`. Every payload
  the inspector needs (`at(slot,gen)`, `schema()` fields/methods/implementors, `graph()` owned
  message ids) already existed from V1–V3.
- **htm/React context is the clean seam for ref-nav.** Because `RefLink` reads `navigateRef` from
  `NavContext`, redirecting refs from "switch to browse" to "push onto the inspector stack" was a
  one-line context override around the panel body — no prop-drilling through `DetailPane`/`ValueView`.


# Phase V6 — program-declared `action` construct (DECISIONS #17)

V6 is the MIRROR of V2's `view`: where `view` declares how an entity should be SEEN, `action`
declares what a user can DO to it — named, typed, parameterized operations that change state / do
side effects, surfaced as prominent BUTTONS in the instance detail inspector (above the raw method
list, since curated affordances are primary). The whole thing is ~90% assembly of existing
machinery: `view` is the structural template, a METHOD is the body/compile/invoke template, and the
desugar reuses the entire method pipeline with ZERO new runtime.

## Grammar (as implemented)
```
action "<Label>" for <Type> [(<param>: <T>, …)] { <body> }
```
The body is full Scry with `self` bound to the target instance (call methods, mutate fields, spawn
threads, etc.). It may return a value or Void; V6 always reports `Void` and discards any yielded
value (actions are side-effecting). `action` is the only new reserved keyword (`for` already
existed). `parse-dump` renders it as `(action "Label" (for (type T)) (params (param p (type T)))
(block …))`.

- **Lexer** (`lexer.coil`): one keyword, `TK_ACTION` (33).
- **AST** (`ast.coil`): `NK_ACTION` (52). Its slot layout deliberately MIRRORS an `NK_FN`
  (`b`=`NK_PARAMS`, `c`=return type (null ⇒ Void), `d`=`NK_BLOCK` body) so `check-fn-body` and
  `compile-method` drive an action body **directly**, with no adapter. `s-name`=label, `a`=`NK_TYPE`
  target, `ival`=(cast) source-file pointer for diagnostics.
- **Parser** (`parser.coil`): `parse-action` + `parse-toplevel` dispatch on `TK_ACTION`. Reuses
  `parse-string` (label), `parse-type` (target), `parse-params` (the optional `(…)`), `parse-block`
  (body) — the same helpers `parse-fn`/`parse-view` use.

## Typecheck rules (`typecheck.coil`, pass 3 `check-all-actions`)
Actions are collected (entry module only, `deep`) into `Ctx.actions`, checked AFTER
signatures+bodies+views so monos exist — exactly like `check-all-views`. Per action `check-action`:
the target `T` must be a declared entity **class/object** (interface/enum/unknown ⇒ span'd error),
then the body is checked by handing the action node straight to `check-fn-body(T, action, self:T)`
(valid because the node shares the `NK_FN` slot layout). Params are typed and bound; an unknown
field/method/type/param in the body is a clean `file:line:col` diagnostic. Check goldens:
`tests/check/51-action-ok` (success — mutate a field, call a method, take a param) + `e46`–`e50`
(unknown target type, unknown method, unknown field, interface target, bad param type).

## Desugar → hidden synthetic method (`compile.coil` `inject-actions`, in `build-program`)
Each `action "L" for T (params) { body }` at registry index `i` becomes a HIDDEN synthetic method
`__action_<i>` on `T`: a fresh `NK_FN` node sharing the action's params (`b`) / null return (`c`) /
body (`d`), pushed onto `T`'s member list BEFORE `register-methods`. From there it flows through the
UNCHANGED pipeline — `register-methods` gives it a method-table index, `compile-all-methods` compiles
it with `self:T`, and it resolves on the normal method-invoke path — so the viewer invokes it as
`T.at(slot,gen).__action_<i>(args)` with no new opcode. Injection is guarded to run once (a
build-program can be called once per process). The synthetic methods are EXCLUDED from every shown
method list by their `__action_` name prefix (`serialize.emit-methods-array`'s `is-action-method`),
so `methods()`/`schema()` never leak them.

## Reflection `actions()` (`serialize.coil` `reflect-actions`, wired in `server.coil` `try-reflection`)
Registered exactly like `views()`; read off `Ctx.actions`. Payload:
```json
{ "type":"Actions", "actions":[
  {"label":"Pause","target":"Agent","invoke":"__action_0","params":[],"returns":"Void"},
  {"label":"Ask","target":"Agent","invoke":"__action_2",
   "params":[{"name":"question","type":"String"}],"returns":"Void"} ] }
```
`invoke` is the mangled synthetic-method name (index `i` mirrors `inject-actions`' `__action_<i>`);
`params` reuses the same `{name,type}` shape as `methods()`. Eval goldens: `tests/eval/60-actions`
(payload), `61-action-invoke` (a live Pause flips `paused` false→true, returns Void),
`62-action-methods-hidden` (`methods("Agent")` never contains `__action_`).

## Viewer: the ACTIONS section (`viewer/app.js`, `viewer/style.css`)
`DetailPane` fetches `actions()` once per type (they're static) and renders a new **ACTIONS** section
ABOVE the methods section. Each declared action is an `ActionCard`: a prominent button (its own warm
"action" green hue — distinct from methods AND from the blue liveness accent). A 0-arg action invokes
on click; a param action opens an inline arg form (reusing `literalFor`/`literalHint`, same as
`MethodCard`) with a `run <Label>` button. Invocation evals
`<Type>.at(slot,gen).<invoke>(<args>)` through the existing invoke path, shows the result/error
inline, then bumps `detailBus` so the instance re-reads and the mutation is visible LIVE. Since it
runs through the safepoint eval channel, `--readonly` rejects the mutating ones automatically. Because
`DetailPane` is shared, the section appears in BOTH the browse detail and the V4 in-map inspector.
`ui-smoke.mjs` gained V6 beats (live, real headless Chrome on `examples/assistant.scry`): the Agent
inspector renders Pause/Resume/Ask buttons above the method list, the param action (Ask) reveals an
inline arg form, and clicking the 0-arg Pause action flips the Agent's `paused` field false→true LIVE
in the inspector. Screenshot: `docs/v6-actions-inspector.png`.

## Example actions (`examples/assistant.scry`)
```
action "Pause"  for Agent { self.pause() }
action "Resume" for Agent { self.resume() }
action "Ask"    for Agent (question: String) { self.runLoop(question) }
action "Spawn researcher" for Orchestrator (topic: String) { self.research(topic) }
```
The assistant still runs + typechecks + passes its e2e app goldens with these added.

## Friction / notes
- **Almost pure reuse.** The one design decision that unlocked it: give `NK_ACTION` the `NK_FN` slot
  layout (`b`/`c`/`d`), so `check-fn-body` and `compile-method` accept an action node verbatim, and
  the desugar is a 4-field `NK_FN` shim into the target's member list. No new typecheck path, no new
  compile path, no new opcode, no runtime change.
- **Index alignment is the contract** between `inject-actions` and `reflect-actions`: both use the
  `Ctx.actions` registry index `i` for `__action_<i>`, so the reflected `invoke` name always matches
  the compiled method. In a well-typed program every action has a valid target (else typecheck fails
  and it never runs), so every action is injected and indices stay aligned.
- **Theme-aware, one new hue.** The action green (`--action`) has explicit light/dark values plus
  `prefers-color-scheme` + `data-theme` overrides, keeping the blue accent reserved for liveness.
