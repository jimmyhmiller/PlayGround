# 06 — Implementation (living doc)

Phase 1 is the **front end**: lexer, AST, recursive-descent parser, and the
`scry parse-dump` CLI, all written in Coil (`run coil guide`). No typechecker, no
bytecode, no VM yet — those are Phase 2+. This doc records how the front end is
built so later phases can stack on it without re-deriving it.

## Build & run

```
coil build                     # project (Coil.toml) -> ./scry
./scry parse-dump f.scry       # parse and print the AST (stable golden format)
./scry run f.scry              # HARD ERROR: "not implemented until Phase 3", exit 2
python3 tests/run-tests.py     # golden suite; exits nonzero on any failure
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
