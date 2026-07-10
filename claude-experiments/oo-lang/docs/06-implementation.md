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
