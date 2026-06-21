//! The front-to-back compile pipeline with prelude injection.
//!
//! `parse_with_prelude` parses the embedded `prelude.gcr` and prepends every
//! prelude item whose name the user program does NOT already declare. This lets
//! programs use `Option`/`Result` and the prelude helpers without declaring
//! them, while existing programs that DO declare their own (e.g. the examples)
//! keep working — the user's declaration shadows the prelude's.

use crate::ast::{Item, ItemKind, Module};
use crate::core::SourceEntry;
use crate::lexer::{lex, lex_with_source, SourceId};
use crate::parser::{ParseError, parse_module};
use std::path::{Path, PathBuf};

const PRELUDE_SRC: &str = include_str!("prelude.gcr");

/// The SourceMap built during prelude/module merging: one [`SourceEntry`] per
/// [`SourceId`] (0 = user, then `mod` files, then the prelude). Returned by the
/// parse entries and attached to `CoreProgram.sources` so spans resolve against
/// their real source (multi-source span resolution — debugger foundation).
pub type SourceMap = Vec<SourceEntry>;

/// Register a source, returning its [`SourceId`] (= its index in the map). Fails
/// loudly rather than silently wrapping if a program somehow exceeds the
/// `SourceId` (u16) capacity — the `id == map index` invariant must hold.
fn add_source(sources: &mut SourceMap, path: String, text: String) -> SourceId {
    let id = SourceId::try_from(sources.len())
        .expect("source count exceeds SourceId (u16) capacity");
    sources.push(SourceEntry { path, text });
    id
}

/// Resolve a `mod foo;` declaration to its file. Following modern Rust, a module
/// `foo` declared in a file living in `dir` is loaded from either `dir/foo.gcr`
/// or `dir/foo/mod.gcr`. Returns the chosen path (and the directory that nested
/// `mod` declarations inside it should resolve against).
fn module_file(dir: &Path, name: &str) -> Result<(PathBuf, PathBuf), ParseError> {
    let flat = dir.join(format!("{}.gcr", name));
    let nested = dir.join(name).join("mod.gcr");
    if flat.exists() {
        // Nested mods inside `foo.gcr` resolve against `dir/foo/`.
        Ok((flat, dir.join(name)))
    } else if nested.exists() {
        Ok((nested.clone(), dir.join(name)))
    } else {
        Err(ParseError {
            msg: format!(
                "cannot find module `{}`: expected `{}` or `{}`",
                name, flat.display(), nested.display()
            ),
            span: crate::lexer::Span::new(0, 0),
        })
    }
}

/// Recursively load file modules (`mod foo;`) referenced by `items`, parsing the
/// sibling files and replacing each declaration with an inline module holding the
/// loaded items. `dir` is the directory that this set of items' file modules
/// resolve against.
fn load_file_modules(items: &mut Vec<Item>, dir: &Path, sources: &mut SourceMap) -> Result<(), ParseError> {
    for item in items.iter_mut() {
        if let ItemKind::Mod(m) = &mut item.kind {
            if m.inline {
                // Inline module: its own nested file mods resolve against
                // `dir/<modname>/` (matching the nested-directory convention).
                let sub = dir.join(&m.name);
                load_file_modules(&mut m.items, &sub, sources)?;
            } else {
                let (file, subdir) = module_file(dir, &m.name)?;
                let src = std::fs::read_to_string(&file).map_err(|e| ParseError {
                    msg: format!("cannot read module file `{}`: {}", file.display(), e),
                    span: m.span,
                })?;
                // Each module is its own source: register it + lex with its id so
                // its spans resolve against its own text, not the user file.
                let sid = add_source(sources, file.display().to_string(), src.clone());
                let toks = lex_with_source(&src, sid).map_err(|e| ParseError { msg: e.msg, span: e.span })?;
                let mut parsed = parse_module(&toks)?;
                // Recurse into this file's own `mod foo;` declarations.
                load_file_modules(&mut parsed.items, &subdir, sources)?;
                m.items = parsed.items;
                m.inline = true;
            }
        }
    }
    Ok(())
}

/// The top-level name an item declares (for dedup against the prelude).
fn item_name(item: &Item) -> Option<&str> {
    match &item.kind {
        ItemKind::Fn(f) => Some(&f.name),
        ItemKind::Struct(s) => Some(&s.name),
        ItemKind::Enum(e) => Some(&e.name),
        ItemKind::Trait(t) => Some(&t.name),
        ItemKind::TypeAlias(a) => Some(&a.name),
        ItemKind::Const(c) => Some(&c.name),
        ItemKind::Mod(m) => Some(&m.name),
        ItemKind::Impl(_) | ItemKind::Use(_) => None,
    }
}

/// Prepend the prelude to a user module's items (dropping prelude items the user
/// redeclares), producing the final module.
fn inject_prelude(user: Module, sources: &mut SourceMap) -> Module {
    let user_names: std::collections::HashSet<String> = user
        .items
        .iter()
        .filter_map(|i| item_name(i).map(|s| s.to_string()))
        .collect();

    // The prelude is its own source ("<std>"): register it + lex with its id so
    // prelude spans resolve against PRELUDE_SRC, never a fabricated user line.
    let sid = add_source(sources, "<std>".to_string(), PRELUDE_SRC.to_string());
    let prelude_tokens = lex_with_source(PRELUDE_SRC, sid).expect("prelude must lex");
    let prelude = parse_module(&prelude_tokens).expect("prelude must parse");

    let mut items: Vec<Item> = prelude
        .items
        .into_iter()
        .filter(|i| match item_name(i) {
            Some(n) => !user_names.contains(n),
            None => true, // impls/uses always kept
        })
        .collect();
    items.extend(user.items);
    Module { items }
}

/// Parse a user program (a single source string, no file modules) and inject the
/// prelude. Used for tests and the in-memory REPL-style path.
pub fn parse_with_prelude(user_src: &str) -> Result<(Module, SourceMap), ParseError> {
    let mut sources: SourceMap = Vec::new();
    add_source(&mut sources, "<input>".to_string(), user_src.to_string()); // user = id 0
    let user_tokens = lex(user_src).map_err(|e| ParseError { msg: e.msg, span: e.span })?;
    let user = parse_module(&user_tokens)?;
    let module = inject_prelude(user, &mut sources);
    Ok((module, sources))
}

/// Parse a user program from a file, recursively loading any `mod foo;` file
/// modules relative to the file's directory, then inject the prelude. This is the
/// driver entry point for multi-file projects.
pub fn parse_file_with_prelude(path: &Path) -> Result<(Module, SourceMap), ParseError> {
    let src = std::fs::read_to_string(path).map_err(|e| ParseError {
        msg: format!("cannot read `{}`: {}", path.display(), e),
        span: crate::lexer::Span::new(0, 0),
    })?;
    let mut sources: SourceMap = Vec::new();
    add_source(&mut sources, path.display().to_string(), src.clone()); // user = id 0
    let tokens = lex(&src).map_err(|e| ParseError { msg: e.msg, span: e.span })?;
    let mut user = parse_module(&tokens)?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    load_file_modules(&mut user.items, dir, &mut sources)?; // mods get ids 1..
    let module = inject_prelude(user, &mut sources); // prelude gets the next id
    Ok((module, sources))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::lower_program;
    use crate::resolve::resolve_module;
    use crate::codegen::jit_run_i64_gc;

    fn run(src: &str) -> i64 {
        let (m, _) = parse_with_prelude(src).unwrap();
        let r = resolve_module(m).unwrap();
        let prog = lower_program(&r.globals).unwrap();
        jit_run_i64_gc(&prog, false).unwrap()
    }

    #[test]
    fn prelude_option_available_without_declaration() {
        // No `enum Option` declared by the user — it comes from the prelude.
        // `opt_unwrap_or(Option::None, 8)` infers `T=i64` from the `8` argument
        // (two-pass argument inference).
        let src = "fn main() -> i64 { opt_unwrap_or(Option::Some(42), 0) + opt_unwrap_or(Option::None, 8) }";
        assert_eq!(run(src), 50);
    }

    #[test]
    fn prelude_numeric_helpers() {
        assert_eq!(run("fn main() -> i64 { max_i64(3, 9) + min_i64(3, 9) }"), 12);
        assert_eq!(run("fn main() -> i64 { pow_i64(2, 10) }"), 1024);
        assert_eq!(run("fn main() -> i64 { gcd_i64(48, 36) }"), 12);
        assert_eq!(run("fn main() -> i64 { abs_i64(0 - 7) }"), 7);
    }

    #[test]
    fn prelude_vec_grows() {
        // The prelude Vec<T> reallocates past its initial capacity of 4.
        let src = "fn main() -> i64 { \
                     let mut v: Vec<i64> = vec_new(); \
                     let mut i = 0; \
                     while i < 30 { v = vec_push(v, i * i); i = i + 1; } \
                     vec_get(v, 5) + vec_get(v, 29) + vec_len(v) \
                   }";
        // 25 + 841 + 30 = 896
        assert_eq!(run(src), 896);
    }

    #[test]
    fn prelude_vec_with_capacity_and_set() {
        let src = "fn main() -> i64 { \
                     let mut v: Vec<i64> = vec_with_capacity(8); \
                     let mut i = 0; \
                     while i < 8 { v = vec_push(v, i); i = i + 1; } \
                     v = vec_set(v, 0, 100); \
                     vec_get(v, 0) + vec_get(v, 7) + vec_len(v) \
                   }";
        // 100 + 7 + 8 = 115
        assert_eq!(run(src), 115);
    }

    #[test]
    fn user_declaration_shadows_prelude() {
        // The user declares their own Option; it must win (no duplicate error).
        let src = "enum Option<T> { None, Some(T) } \
                   fn main() -> i64 { match Option::Some(5) { Option::Some(x) => x, Option::None => 0 } }";
        assert_eq!(run(src), 5);
    }

    // ---- string stdlib (written in gc-rust, over the runtime primitives) ----

    #[test]
    fn prelude_str_trim_and_prefix() {
        let src = "fn main() -> i64 { \
                     let t = str_trim(\"  hi there  \"); \
                     let p = if str_starts_with(t, \"hi\") { 1 } else { 0 }; \
                     let e = if str_ends_with(t, \"there\") { 1 } else { 0 }; \
                     str_len(t) * 10 + p + e \
                   }";
        // trimmed \"hi there\" len 8 -> 80 + 1 + 1 = 82
        assert_eq!(run(src), 82);
    }

    #[test]
    fn prelude_parse_int() {
        assert_eq!(run("fn main() -> i64 { opt_unwrap_or(parse_int(\"12345\"), 0) }"), 12345);
        assert_eq!(run("fn main() -> i64 { opt_unwrap_or(parse_int(\"-42\"), 0) }"), 0 - 42);
        // Malformed -> None -> the default.
        assert_eq!(run("fn main() -> i64 { opt_unwrap_or(parse_int(\"12x\"), 0 - 1) }"), 0 - 1);
        assert_eq!(run("fn main() -> i64 { opt_unwrap_or(parse_int(\"\"), 0 - 7) }"), 0 - 7);
    }

    #[test]
    fn prelude_split_and_join_roundtrip() {
        // Split on ',' then join with '|' — count pieces and total length.
        let src = "fn main() -> i64 { \
                     let parts = str_split_byte(\"a,bb,ccc\", 44); \
                     let joined = str_join(parts, \"|\"); \
                     vec_len(parts) * 100 + str_len(joined) \
                   }";
        // 3 pieces; \"a|bb|ccc\" len 8 -> 308
        assert_eq!(run(src), 308);
    }

    #[test]
    fn prelude_replace_byte() {
        let src = "fn main() -> i64 { str_len(str_replace_byte(\"a-b-c\", 45, \"__\")) }";
        // \"a__b__c\" -> length 7
        assert_eq!(run(src), 7);
    }

    // ---- integer overflow semantics (docs/overflow.md) --------------------

    #[test]
    fn overflow_wraps() {
        // i64::MAX + 1 wraps to i64::MIN — a defined behavior, not UB.
        let src = "fn main() -> i64 { let m = 9223372036854775807; m + 1 }";
        assert_eq!(run(src), i64::MIN);
    }

    #[test]
    fn checked_add_detects_overflow() {
        // checked_add_i64 returns None at the boundary, Some otherwise.
        let over = "fn main() -> i64 { if opt_is_some(checked_add_i64(9223372036854775807, 1)) { 1 } else { 0 } }";
        assert_eq!(run(over), 0);
        let ok = "fn main() -> i64 { opt_unwrap_or(checked_add_i64(100, 200), 0) }";
        assert_eq!(run(ok), 300);
    }

    #[test]
    fn checked_mul_detects_overflow() {
        let over = "fn main() -> i64 { if opt_is_some(checked_mul_i64(9223372036854775807, 2)) { 1 } else { 0 } }";
        assert_eq!(run(over), 0);
        let ok = "fn main() -> i64 { opt_unwrap_or(checked_mul_i64(6, 7), 0) }";
        assert_eq!(run(ok), 42);
    }

    // ---- Vec completion ----------------------------------------------------

    #[test]
    fn prelude_vec_sort_and_search() {
        let src = "fn main() -> i64 { \
                     let mut v: Vec<i64> = vec_new(); \
                     v = vec_push(v, 5); v = vec_push(v, 3); v = vec_push(v, 8); v = vec_push(v, 1); \
                     let s = vec_sort_i64(vec_copy(v)); \
                     vec_get(s, 0) * 1000 + vec_get(s, 3) * 100 + vec_index_of_i64(v, 8) * 10 + vec_sum_i64(v) \
                   }";
        // sorted[0]=1, sorted[3]=8, index_of(8)=2, sum=17 -> 1000+800+20+17 = 1837
        assert_eq!(run(src), 1837);
    }

    #[test]
    fn prelude_vec_pop_last_reverse() {
        let src = "fn main() -> i64 { \
                     let mut v: Vec<i64> = vec_new(); \
                     v = vec_push(v, 1); v = vec_push(v, 2); v = vec_push(v, 3); \
                     let last = opt_unwrap_or(vec_last(v), 0); \
                     let rev = vec_reverse(v); \
                     let popped = vec_pop(v); \
                     last * 100 + vec_get(rev, 0) * 10 + vec_len(popped) \
                   }";
        // last=3, rev[0]=3, pop len=2 -> 300 + 30 + 2 = 332
        assert_eq!(run(src), 332);
    }

    // ---- MapStr (String-keyed hash map) ------------------------------------

    #[test]
    fn prelude_mapstr_basic() {
        let src = "fn main() -> i64 { \
                     let mut m: MapStr<i64> = mapstr_new(); \
                     m = mapstr_insert(m, \"one\", 1); \
                     m = mapstr_insert(m, \"two\", 2); \
                     m = mapstr_insert(m, \"two\", 22); \
                     let g = opt_unwrap_or(mapstr_get(m, \"two\"), 0); \
                     let c = if mapstr_contains(m, \"one\") { 1 } else { 0 }; \
                     let miss = if mapstr_contains(m, \"x\") { 1 } else { 0 }; \
                     g * 100 + mapstr_len(m) * 10 + c + miss \
                   }";
        // get(two)=22, len=2 (one+two), contains(one)=1, miss=0 -> 2200+20+1 = 2221
        assert_eq!(run(src), 2221);
    }

    #[test]
    fn prelude_iterator_combinators() {
        let src = "fn main() -> i64 { \
                     let v = vec_range(1, 6); \
                     let doubled = vec_map(v, |x: i64| x * 2); \
                     let evens = vec_filter(v, |x: i64| x % 2 == 0); \
                     let total = vec_fold(v, 0, |acc: i64, x: i64| acc + x); \
                     vec_sum_i64(doubled) * 1000 + vec_sum_i64(evens) * 100 + total * 1 + vec_count(v, |x: i64| x > 3) \
                   }";
        // doubled sum=30, evens sum=6, fold total=15, count(>3)=2 -> 30000+600+15+2 = 30617
        assert_eq!(run(src), 30617);
    }

    #[test]
    fn prelude_generic_trait_ops() {
        // vec_contains / vec_sort / vec_max via the Eq/Ord traits on i64.
        let src = "fn main() -> i64 { \
                     let mut v: Vec<i64> = vec_new(); \
                     v = vec_push(v, 30); v = vec_push(v, 10); v = vec_push(v, 20); \
                     let s = vec_sort(vec_copy(v)); \
                     let c = if vec_contains(v, 20) { 1 } else { 0 }; \
                     vec_get(s, 0) * 100 + vec_get(s, 2) + opt_unwrap_or(vec_max(v), 0) + c \
                   }";
        // sorted[0]=10, sorted[2]=30, max=30, contains=1 -> 1000+30+30+1 = 1061
        assert_eq!(run(src), 1061);
    }

    #[test]
    fn prelude_user_type_implements_ord() {
        // A user struct implementing Ord is sorted by the generic vec_sort.
        let src = "struct P { level: i64 } \
                   impl Ord for P { \
                     fn cmp(self, other: P) -> i64 { \
                       if self.level < other.level { 0 - 1 } else { if self.level > other.level { 1 } else { 0 } } \
                     } \
                   } \
                   fn main() -> i64 { \
                     let mut v: Vec<P> = vec_new(); \
                     v = vec_push(v, P { level: 3 }); v = vec_push(v, P { level: 1 }); v = vec_push(v, P { level: 2 }); \
                     let s = vec_sort(v); \
                     vec_get(s, 0).level * 100 + vec_get(s, 1).level * 10 + vec_get(s, 2).level \
                   }";
        // sorted levels: 1, 2, 3 -> 123
        assert_eq!(run(src), 123);
    }

    #[test]
    fn prelude_iterator_any_all() {
        let src = "fn main() -> i64 { \
                     let v = vec_range(1, 5); \
                     let a = if vec_any(v, |x: i64| x > 3) { 1 } else { 0 }; \
                     let b = if vec_all(v, |x: i64| x > 0) { 1 } else { 0 }; \
                     let c = if vec_all(v, |x: i64| x > 2) { 1 } else { 0 }; \
                     a * 100 + b * 10 + c \
                   }";
        // any(>3)=1, all(>0)=1, all(>2)=0 -> 110
        assert_eq!(run(src), 110);
    }

    #[test]
    fn prelude_mapstr_grows_and_rehashes() {
        // Insert past the initial capacity (8) to force at least two rehashes;
        // every key must remain findable, including the earliest insert.
        let src = "fn main() -> i64 { \
                     let mut m: MapStr<i64> = mapstr_new(); \
                     m = mapstr_insert(m, \"first\", 7); \
                     let mut i = 0; \
                     while i < 60 { m = mapstr_insert(m, to_string(i), i * 10); i = i + 1; } \
                     opt_unwrap_or(mapstr_get(m, \"k\"), 0) + \
                       opt_unwrap_or(mapstr_get(m, \"first\"), 0) * 1 + \
                       opt_unwrap_or(mapstr_get(m, \"37\"), 0) + \
                       mapstr_len(m) \
                   }";
        // get(k)=0(absent), first=7, get(37)=370, len=61 -> 0+7+370+61 = 438
        assert_eq!(run(src), 438);
    }
}
