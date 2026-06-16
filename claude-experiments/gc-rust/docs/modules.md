# Modules (Phase 4)

gc-rust has a Rust-style module system: inline modules, multi-file modules,
`pub` visibility, and `use` imports.

## Declaring modules

```rust
// Inline module — items are qualified as `math::square`.
mod math {
    pub fn square(x: i64) -> i64 { x * x }
    fn helper(x: i64) -> i64 { x + 1 }   // private to `math`
}

// File module — loads a sibling file.
mod geometry;        // loads ./geometry.gcr  OR  ./geometry/mod.gcr
```

File resolution follows modern Rust: `mod foo;` in a file living in directory
`dir` loads `dir/foo.gcr` if present, otherwise `dir/foo/mod.gcr`. A `mod bar;`
declared *inside* `foo` resolves against `dir/foo/` (so `foo/bar.gcr`).

## Visibility

Items are **private by default**; `pub` makes them visible to other modules.
A module-qualified reference (`a::b::c`) to a private item from outside its
module is a compile error:

```
error: `lib::priv_fn` is private and not accessible from here (mark it `pub`)
```

The rule (close to Rust): an item is visible from a module `M` if it is `pub`,
or it lives in `M` itself or an ancestor of `M` (and everything can see the
crate root). Enum variants inherit their enum's visibility.

## `use`

```rust
use geometry::Point;     // bring `Point` into scope as a short name
use math::*;             // glob: every item directly in `math`
```

`use a::b::c;` aliases the short name `c` to the full path `a::b::c`. This also
**disambiguates** when two modules export the same last segment:

```rust
mod a { pub fn val() -> i64 { 1 } }
mod b { pub fn val() -> i64 { 2 } }
use b::val;              // now `val()` means `b::val`
```

A glob `use a::b::*;` imports every item declared directly in `a::b` (not items
of nested submodules).

## The prelude

The standard library (`Option`/`Result`, `Vec`, `String` ops, `MapStr`,
iterators, the `Eq`/`Ord` traits) is injected at the crate root, so its names
are available unqualified in every program. A user program that declares its own
`Option`/`Vec`/etc. shadows the prelude's version.

## Pipeline

`compile::parse_file_with_prelude(path)` reads the entry file, then
`load_file_modules` recursively parses each `mod foo;` sibling file and splices
its items in as an inline module. Resolution (`resolve.rs`) then collects every
item under its fully-qualified module path, applies `use` aliases, validates
types, and enforces visibility.
