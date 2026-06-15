//! The front-to-back compile pipeline with prelude injection.
//!
//! `parse_with_prelude` parses the embedded `prelude.gcr` and prepends every
//! prelude item whose name the user program does NOT already declare. This lets
//! programs use `Option`/`Result` and the prelude helpers without declaring
//! them, while existing programs that DO declare their own (e.g. the examples)
//! keep working — the user's declaration shadows the prelude's.

use crate::ast::{Item, ItemKind, Module};
use crate::lexer::lex;
use crate::parser::{ParseError, parse_module};

const PRELUDE_SRC: &str = include_str!("prelude.gcr");

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

/// Parse a user program and inject the prelude. Prelude items whose name the
/// user already declares are dropped (the user's version wins).
pub fn parse_with_prelude(user_src: &str) -> Result<Module, ParseError> {
    let user_tokens = lex(user_src).map_err(|e| ParseError { msg: e.msg, span: e.span })?;
    let user = parse_module(&user_tokens)?;

    let user_names: std::collections::HashSet<String> = user
        .items
        .iter()
        .filter_map(|i| item_name(i).map(|s| s.to_string()))
        .collect();

    let prelude_tokens = lex(PRELUDE_SRC).expect("prelude must lex");
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
    Ok(Module { items })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::lower_program;
    use crate::resolve::resolve_module;
    use crate::codegen::jit_run_i64_gc;

    fn run(src: &str) -> i64 {
        let m = parse_with_prelude(src).unwrap();
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
}
