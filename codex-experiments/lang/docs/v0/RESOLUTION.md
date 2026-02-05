# Name Resolution v0 (Draft)

This defines module layout, visibility, and how names are resolved.

## 1. Module layout

- A package root is a folder.
- Entry file: `main.lang` (or `lib.lang` for libraries).
- `module a::b::c;` declares the logical module path for that file.
- File path must match module path: `a/b/c.lang`.

## 2. Visibility

- `pub` items are visible outside their defining module.
- Non-`pub` items are module-private.

Applies to:

- `struct`, `enum`, `trait`, `fn` declarations.

## 3. Imports

Syntax:

```
use a::b::c;
use a::b::{c, d, e};
```

Rules:

- Imports are resolved relative to the package root.
- `use` binds the last path segment in the current module scope.
- No aliasing in v0.

## 4. Resolution order

When resolving an unqualified identifier:

1. Local bindings (parameters, locals, let-bound names).
2. Items declared in the current module.
3. `use` imports.
4. Builtins.

Qualified paths (`a::b::c`) bypass local bindings and resolve as item paths.

## 5. Shadowing

- Locals may shadow earlier locals in the same block.
- Locals may shadow module items.
- Shadowing imported names is allowed.

## 6. Trait and impl lookup (v0)

- No generic or advanced coherence checks yet.
- Lookup is by full path and exact type match.
- If multiple impls match, it is a compile error.

