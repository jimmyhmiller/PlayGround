# Scry — Module System Design

**Status: RATIFIED (Jimmy ruling, 2026-07-13) — pinned as DECISIONS.md #18. BUILT
2026-07-14 (all three phases, 321 tests green).**
Wildcard imports are ruled out permanently (not open). Supersedes `01-language.md` §1.6
(which sketched the same direction but was never implemented and contained one
inconsistency, fixed there now).

**As-built deviations (flagged for ruling, not silently decided):**
1. **Focus URL is a hash route (`#m=<dotted.module>`), not the path form `/p/<id>/m/…`**
   this doc specified. The server/portal have no SPA-fallback routing — path form means
   hand-rolled wildcard parsers in both `server.coil` and `portal.coil` for zero
   functional gain (the hash is shareable and back/forward-navigable). Upgradeable later.
2. **§6's claim that `agents.tools.shell.ShellTool.instances()` works is not accurate as
   built.** Qualified names work as value refs, constructor calls, and function calls,
   but the qualified fast path is one level deep (`<module>.<member>(...)`) — a
   *multi-segment receiver* for reflect-static forms (`<mod>.<Class>.instances()`) and
   enum-variant access (`<mod>.<Enum>.Variant`) falls through to "unknown identifier".
   The working mechanism everywhere (and what the viewer uses) is the eval module header
   (`module a.b.c` prefix, §6). OPEN follow-up: extend `compile-dotted-call` to resolve
   the longest module prefix so #18's "fully-qualified refs work everywhere" is literal.
3. **Unfocused map rings bucket by each ownership tree's root's module.** Types from an
   imported module that live entirely inside another module's owned tree (e.g.
   `agent.core` types owned by `assistant.Agent`) ride inside the owning ring with
   per-row module tags, and get standalone roots when focused. Unconditional
   same-module-only nesting would have broken natural cross-module ownership
   (`Agent owns Conversation`); nesting is unrestricted unfocused, same-module while
   focused. Module chrome renders only when ≥2 non-std top-level modules exist, so
   single-module programs look exactly as before.

## 0. Why modules, and why they look different in Scry

Scry needs modules for the ordinary reason — a real program does not fit in one file, and
today two files cannot even declare the same class name (one flat global namespace,
`typecheck.coil` `collect-decl` duplicate error). But it also needs them for a Scry-specific
reason: **the viewer needs a unit of focus**. "Show me just the billing part of this system"
is a question the map view cannot answer today because the language has no word for
"the billing part."

One thing Scry modules deliberately are **not**: a visibility fence. There is no
`public`/`private`/`internal`, no export lists, and no convention (like `_underscore`) that
carries semantics. This is a thesis decision, not an omission: Scry's product is that
*nothing in a running system is hidden* — the viewer shows every field of every instance.
A language construct whose job is to hide declarations from other code would fight the
product. Modules here are **addresses and lenses**: an address so every declaration has a
unique stable name, a lens so the UI can scope itself to a context.

Good news from the code survey: `module a.b.c` headers and
`import a.b.{X, Y}` / `import a.b.X as Z` already lex, parse, and drive file loading
(`parser.coil` `parse-module`/`parse-import`, `typecheck.coil` `load-file`/`resolve-import`).
What's missing is semantics: the header is parsed and discarded, all declarations land in
one flat table, canonical names are bare identifiers everywhere (mono keys, arenas,
reflection JSON, live-redefinition), and import resolution is relative to the importing
file (hence the committed `std` symlinks). This design gives the existing syntax its
meaning.

## 1. The model in one paragraph

**One file is one module.** A module's name is its file path from the project root, dotted,
with the file stem as the last segment: `agents/tools/shell.scry` is module
`agents.tools.shell`. Directories are not modules themselves, but they form the hierarchy
(`agents.tools.*`) that the viewer's focus tree and qualified names hang off. Every
top-level declaration's canonical name is `<module>.<Name>` —
`agents.tools.shell.ShellTool` — and that qualified name is what the mono table, the
per-type arenas, reflection JSON, and live redefinition key on. Within a file you use bare
names for your own module's declarations and for anything you `import`; fully-qualified
names work everywhere without an import. Nothing is private. Import cycles are legal
because imports are pure name-scoping over a single whole-program compilation unit — they
never execute anything.

## 2. Module identity

- **Path-derived, not header-declared.** The module name is a pure function of the file's
  path relative to the project root. `std/json.scry` → `std.json`. `main.scry` at the
  root → `main`. This keeps the import resolver what it already is (dotted path → file
  path) with no manifest, no scan, no registry.
- **The `module` header is optional, checked documentation.** If present, it must equal
  the derived name or typecheck fails with a clear error ("file agents/tools/shell.scry
  declares module agents.tools — expected agents.tools.shell; did the file move?"). This
  catches moved/copied files instead of silently re-addressing every type in them.
  (Existing files already conform: `std/json.scry` declares `module std.json`.)
- **Fix to `01-language.md` §1.6:** its example puts `module agents.tools` in
  `agents/tools/shell.scry`. Under this design that header is a typecheck error; the
  example becomes `module agents.tools.shell`. §1.6 gets rewritten to point here once
  this doc is ratified.
- **Renaming/moving a file renames the module** and therefore every qualified name in it.
  That is the honest consequence of path-derived identity and it is fine: names resolve at
  compile time, and a *running* process is never affected (you can't move a file into a
  running program; live eval addresses the already-loaded module table).

## 3. Project root and file resolution

Two roots, checked in order:

1. **Project root** = the directory containing the entry file, unless a marker file
   (`scry.toml` — new, trivial, may be empty) is found in an ancestor directory, in which
   case the nearest marker directory is the root. `import agents.tools.shell` resolves to
   `<root>/agents/tools/shell.scry`.
2. **The std root** — `import std.*` additionally resolves against the Scry
   installation's `std/` directory (located relative to the `scry` binary, overridable
   with `SCRY_STD`). Project root wins on conflict, so a project can shadow a std module.

This kills the committed-symlink workaround (`examples/std -> ../std` etc.) documented as
friction in `06-implementation.md`. The existing relative-to-importing-file candidates can
stay as a deprecated fallback during migration, then be removed.

Loading is unchanged in shape: transitive closure of imports from the entry file, deduped
by absolute path (`load-file` already does this), all files typechecked together as one
program. Modules are **not** compilation units — whole-program compilation and
monomorphization stay exactly as they are.

## 4. Names and resolution

Canonical name of every top-level declaration (class, interface, enum, object, fn, view,
action) = `<module>.<Name>`. `Ctx.decls` becomes keyed by (module, name); the duplicate
error only fires within one module, so `agents.tools.shell.Shell` and `ui.panels.Shell`
coexist — that is the point.

A bare identifier in a file resolves in this order:

1. Locals / params / fields via `self` (unchanged).
2. The file's own module's declarations.
3. Names brought in by this file's `import` statements (including `as` aliases).
4. **Builtins** (`List`, `Map`, `String`, `Console`, `Thread`, `Mutex`, `Result`, …).
   Builtins remain a bare-name prelude — they are the language, not a library, and forcing
   `std.core.List` everywhere would be noise. (They keep their existing one-time
   override-ability.)
5. Otherwise: error. If the name exists in other modules, the error must say so and print
   the exact import line to add — the compiler knows, so it tells you.

**Fully-qualified references work everywhere with no import**: `std.json.parse(text)`,
`let t: agents.tools.shell.ShellTool`. Qualified names are the canonical addresses the
whole system speaks (viewer links, REPL, reflection JSON, error messages), so source code
must accept them too. Parsing stays unambiguous the way it already is for field access —
`a.b.c` parses as a postfix chain, and the *typechecker* resolves the head: if the leading
segment is a local/param, it's a field access; otherwise if the longest dotted prefix names
a known module, it's a qualified reference. Locals shadow module roots (locals win; the
compiler warns when a local actually shadows a module path that the rest of the chain
would have resolved against).

Generic canon keys become fully qualified: `Inventory<agents.tools.shell.ShellTool>` in
the mono table; display names shorten to bare when unambiguous (see §7).

## 5. Imports

Existing surface, now with real semantics — no new syntax:

```
import std.json.{parse, JsonValue}
import agents.tools.shell.{ShellTool}
import agents.tools.search.SearchTool as Finder
```

- An import brings *names into this file's scope*. Nothing else. No initialization, no
  side effects (there is no module-level statement in Scry), no re-export, no visibility
  gate to negotiate.
- `validate-imports` keeps doing what it does — every imported name must exist in the
  target module — but against the per-module table instead of the flat one.
- **Cycles are legal.** `a` importing `b` importing `a` is fine: both files join the same
  whole-program closure and typecheck together. There is no initialization order to
  corrupt. The loader's existing dedup already terminates the recursion.
- **No wildcard import** (`import m.*`): DECIDED, no — Jimmy ruling, permanent. Explicit
  lists keep a file's context legible — you can read the top of a file and know its
  vocabulary — and the compiler's "add this import" errors plus qualified names remove
  most of the annoyance. The parser should reject `.*` with a message saying so.
- **No whole-module import** (`import agents.tools.shell` then `shell.ShellTool`): OPEN,
  leaning no for the PoC. Fully-qualified names already cover the "I don't want to list
  everything" case; a second, abbreviated qualification form adds resolution rules for
  little gain.

## 6. Runtime and live semantics

- **Arenas and mono table key on the qualified canon name.** Per-type arenas are otherwise
  untouched; `agents.tools.shell.ShellTool.instances()` is the same slab walk.
- **The eval wire op stays `{id, source} → {id, value|error}`** (DECISIONS #8, untouched).
  Module context rides *inside the source*, exactly like a file: an eval may begin with a
  `module a.b.c` header (today this is rejected — `server.coil:677`), which sets the
  resolution context for that eval. Bare names resolve as if the source were appended to
  that module's file; a redefinition (`fn step(...) { ... }`) targets `a.b.c.step`. With
  no header, context defaults to the entry module. The viewer's REPL dock grows a module
  picker whose entire implementation is "prepend the header."
- `import` inside a live eval stays rejected — use qualified names; an eval is a probe,
  not a file.
- **Redefinition addressing** (`redefine-class`/`redefine-fn`/`redefine-interface`) looks
  up (module, name) instead of bare name. The generation counter stays a single global —
  generations order *changes to the program*, not changes per module, and every consumer
  of `generation()` wants exactly that.
- Everything in the allowed/rejected live-change matrix (STATUS.md) is unchanged; it just
  applies per qualified name.

## 7. Reflection and the viewer: modules as focus contexts

Wire/reflection additions (all still just evals):

- Every node in `schema()` / `types()` items gains `"module": "agents.tools.shell"`;
  `"name"` stays the bare declaration name (display), and `"qualified"` carries the canon
  name (identity). All `refTypes` / links use qualified names.
- New reflection op `modules()` → the module tree with per-module aggregates:
  `{name, path, children[], typeCount, fnCount, liveCount}`. This is the data for the
  focus tree, and it works statically (schema-json) as well as live — so `scry inspect`
  and the portal's static project view get the same contexts for free.

Viewer behavior (the actual point):

- **Map view: module = the outermost containment ring.** The bespoke nested-containment
  layout (DECISIONS #15a) gains one more level above ownership: modules are the top-level
  regions, deterministic layout as always, region size ∝ aggregate live mass. `std.*` and
  other utility modules recede to the faded periphery by default — the existing
  utility-fade concept, now with a principled boundary to fade at.
- **Cross-module references use the existing shared-entity treatment**: an instance
  referenced across a module boundary renders as its identity-color chip at the boundary;
  hover highlights all appearances. No new visual vocabulary needed.
- **Focus mode — the headline feature.** Click a module (map region, breadcrumb, or the
  type rail's module group header) and the *entire viewer* scopes to that subtree: census
  ribbon, map, instance lists, functions, search. Focus is URL-addressable
  (`/p/<id>/m/agents.tools`) so a context is a link you can send someone. Breadcrumbs
  (`agents ▸ tools`) zoom out; Esc clears. While focused, instances reachable from outside
  the focus show as boundary chips with an "expand focus to include <module>" affordance.
- **Type rail groups by module** (collapsible), with the existing interface groups nested
  inside each module group. The type-filter box and global search default to the current
  focus, with an "everywhere" toggle.
- **`view` and `action` follow the type, not the declaring module**: a
  `view Board for kanban.model.Card` declared in `kanban.ui` still renders wherever a
  `Card` appears. No privacy means no restriction on declaring views/actions for another
  module's types — that's a feature (instrumentation modules that decorate a system they
  don't own).

## 8. Migration and compatibility

- **A single-file program is one module and notices nothing.** Entry file `foo.scry` with
  no header = module `foo`; every name is bare within it; all existing examples and the
  ~301-test suite keep passing unmodified (the duplicate-decl golden `e23` stays valid —
  same module).
- `std` imports keep working via the std root; the `std` symlinks are deleted once the
  fallback resolver retires.
- Reflection consumers: the viewer is versioned with the runtime (served from the same
  process), so adding `"module"`/`"qualified"` fields is not a wire-compat event.

## 9. Build order (proposed cut lines)

1. **Compiler namespacing** — (module, name) decl table, qualified canon names in mono
   keys/arenas, path-derived module identity + header check, project/std root resolution,
   qualified references in source, import validation per module, error messages that print
   the missing import. Golden tests: cross-module same-name coexistence, qualified refs,
   header mismatch, cycle, suggested-import error text.
2. **Reflection + live eval** — `"module"`/`"qualified"` in schema/types, `modules()` op,
   eval module-header context, redefinition per qualified name. Eval-suite goldens.
3. **Viewer focus** — module ring in the map, focus mode + URL, type-rail grouping,
   scoped search, boundary chips. This phase is pure viewer work on data phase 2 provides.

Each phase lands green on the full suite before the next starts.

## 10. Open questions (flagged, not decided)

- ~~Wildcard import `import m.*`~~ — **DECIDED no, permanently** (Jimmy ruling 2026-07-13).
- Whole-module import enabling short-qualified `shell.ShellTool` — lean **no** for PoC (§5).
- Re-exports — lean **never**: with no visibility there is nothing to re-export; a name is
  always importable from where it lives. Listed only so nobody adds it by reflex.
- `scry.toml` contents beyond being a root marker (name? std pin?) — out of scope here.
- Should the shadowing warning (local over module root, §4) be an error instead? Lean
  warning.
