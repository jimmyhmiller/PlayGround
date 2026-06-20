//! Regression tests for the prelude stdlib round-out: generic `HashMap<K, V>`,
//! the `Display` trait + generic print helpers, and the `Vec`/string API
//! additions. Each program ends in an `i64` that encodes its checks, so a wrong
//! answer (not just a crash) fails the test. Run under GC-stress so the new
//! generic code is also exercised by the collector.

use gcrust::codegen::jit_run_i64_gc;
use gcrust::compile::parse_with_prelude;
use gcrust::lower::lower_program;
use gcrust::resolve::resolve_module;

/// Compile `src` (with the prelude) and JIT-run it, returning `main`'s i64.
fn run(src: &str) -> i64 {
    let module = parse_with_prelude(src).expect("parse");
    let resolved = resolve_module(module).expect("resolve");
    let prog = lower_program(&resolved.globals).expect("lower");
    // `true` = collect on every allocation, so the new generic heap shapes
    // (HashMap key/val arrays, etc.) are traced under stress.
    jit_run_i64_gc(&prog, true).expect("run")
}

#[test]
fn hashmap_string_keys_insert_get_overwrite() {
    let r = run(r#"
        fn main() -> i64 {
            let mut m: HashMap<String, i64> = hashmap_new();
            m = hashmap_insert(m, "alpha", 10);
            m = hashmap_insert(m, "beta", 20);
            m = hashmap_insert(m, "alpha", 11);            // overwrite, not a new entry
            let a = hashmap_get_or(m, "alpha", 0 - 1);     // 11
            let b = hashmap_get_or(m, "beta", 0 - 1);      // 20
            let miss = hashmap_get_or(m, "gamma", 999);    // 999
            let len = hashmap_len(m);                      // 2 (overwrite didn't grow it)
            a + b + miss + len                             // 11 + 20 + 999 + 2 = 1032
        }
    "#);
    assert_eq!(r, 1032);
}

#[test]
fn hashmap_int_keys_force_grow() {
    // 50 entries forces several grows past the 0.7 load factor; the int hash
    // must spread sequential keys so probing stays correct across rehashes.
    let r = run(r#"
        fn main() -> i64 {
            let mut n: HashMap<i64, i64> = hashmap_new();
            let mut i = 0;
            while i < 50 { n = hashmap_insert(n, i, i * i); i = i + 1; }
            let sq = hashmap_get_or(n, 7, 0 - 1);          // 49
            let len = hashmap_len(n);                       // 50
            let has = if hashmap_contains(n, 49) { 1 } else { 0 };   // 1
            let nope = if hashmap_contains(n, 50) { 1 } else { 0 };  // 0
            sq + len + has + nope                           // 49 + 50 + 1 + 0 = 100
        }
    "#);
    assert_eq!(r, 100);
}

#[test]
fn hashmap_keys_vals_roundtrip() {
    let r = run(r#"
        fn main() -> i64 {
            let mut m: HashMap<i64, i64> = hashmap_new();
            m = hashmap_insert(m, 1, 100);
            m = hashmap_insert(m, 2, 200);
            m = hashmap_insert(m, 3, 300);
            let ks = hashmap_keys(m);   // {1,2,3} in some order
            let vs = hashmap_vals(m);   // {100,200,300}
            vec_sum_i64(ks) + vec_sum_i64(vs)   // 6 + 600 = 606
        }
    "#);
    assert_eq!(r, 606);
}

#[test]
fn display_and_str_of() {
    // `bool` renders as true/false (length 4/5); ints/floats via to_string.
    let r = run(r#"
        fn main() -> i64 {
            let t = str_of(true);       // "true"  len 4
            let f = str_of(false);      // "false" len 5
            let n = str_of(42);         // "42"    len 2
            str_len(t) + str_len(f) + str_len(n)   // 4 + 5 + 2 = 11
        }
    "#);
    assert_eq!(r, 11);
}

#[test]
fn string_formatting_helpers() {
    let r = run(r#"
        fn main() -> i64 {
            let rep = str_repeat("ab", 3);            // "ababab" len 6
            let pl = str_pad_left("42", 5, "0");      // "00042" len 5
            let pr = str_pad_right("x", 4, ".");      // "x..." len 4
            let rev = str_reverse("abcdef");          // "fedcba"
            let rev_ok = if str_eq(rev, "fedcba") { 1 } else { 0 };
            str_len(rep) + str_len(pl) + str_len(pr) + rev_ok   // 6 + 5 + 4 + 1 = 16
        }
    "#);
    assert_eq!(r, 16);
}

#[test]
fn vec_api_roundout() {
    let r = run(r#"
        fn main() -> i64 {
            let mut a: Vec<i64> = vec_new();
            a = vec_push(a, 10); a = vec_push(a, 20); a = vec_push(a, 30); a = vec_push(a, 40);
            let mut b: Vec<i64> = vec_new();
            b = vec_push(b, 50); b = vec_push(b, 60);

            let f = opt_unwrap_or(vec_first(a), 0 - 1);        // 10
            let oob = opt_unwrap_or(vec_get_opt(a, 9), 0 - 1); // -1
            let c = vec_concat(a, b);                          // len 6
            let sl = vec_sum_i64(vec_slice(c, 1, 4));          // 20+30+40 = 90
            let tk = vec_sum_i64(vec_take(c, 2));              // 10+20 = 30
            let dr = vec_sum_i64(vec_drop(c, 4));              // 50+60 = 110
            let sw = opt_unwrap_or(vec_first(vec_swap(vec_copy(a), 0, 3)), 0); // 40
            let pos = vec_position(a, |x: i64| x - (x / 2) * 2 == 0);          // 0
            let found = opt_unwrap_or(vec_find(a, |x: i64| x > 25), 0 - 1);    // 30

            f + oob + vec_len(c) + sl + tk + dr + sw + pos + found
            // 10 + (-1) + 6 + 90 + 30 + 110 + 40 + 0 + 30 = 315
        }
    "#);
    assert_eq!(r, 315);
}

#[test]
fn unicode_codepoint_api() {
    // "café" is 5 bytes but 4 code points (é is 2 bytes). The code-point API
    // must see 4 chars, round-trip losslessly, slice without splitting é, and
    // index code points (233 = U+00E9) distinctly from bytes (195 = é's lead byte).
    let r = run(r#"
        fn main() -> i64 {
            let s = "café";
            let bytes = str_len(s);                                            // 5
            let chars = str_len_chars(s);                                      // 4
            let rt = if str_eq(str_from_chars(str_chars(s)), s) { 1 } else { 0 };  // 1
            let cp = char_at(s, 3);                                            // 233
            let byte = str_get(s, 3);                                          // 195
            let sub_ok = if str_eq(str_sub_chars(s, 0, 4), s) { 1 } else { 0 };    // 1 (é intact)
            let emoji_bytes = str_len(char_to_str(128512));                   // 4 (😀)
            bytes + chars + rt + cp + byte + sub_ok + emoji_bytes
            // 5 + 4 + 1 + 233 + 195 + 1 + 4 = 443
        }
    "#);
    assert_eq!(r, 443);
}

#[test]
fn unicode_case_folding_and_invalid_scalar() {
    let r = run(r#"
        fn main() -> i64 {
            // ASCII folds; é (non-ASCII) passes through unchanged either way.
            let up_ok = if str_eq(str_to_upper("Café!"), "CAFé!") { 1 } else { 0 };
            let lo_ok = if str_eq(str_to_lower("HeLLo"), "hello") { 1 } else { 0 };
            let inv = str_len(char_to_str(0 - 5));   // invalid scalar -> U+FFFD = 3 bytes
            up_ok + lo_ok + inv                      // 1 + 1 + 3 = 5
        }
    "#);
    assert_eq!(r, 5);
}

#[test]
fn string_split_and_index_of() {
    let r = run(r#"
        fn main() -> i64 {
            let idx = str_index_of("hello world", "world");   // 6
            let miss = str_index_of("abc", "z");              // -1
            let parts = str_split("a,bb,ccc", ",");           // 3 parts
            let joined = str_join(parts, "|");                // "a|bb|ccc" len 8
            idx + miss + vec_len(parts) + str_len(joined)     // 6 + (-1) + 3 + 8 = 16
        }
    "#);
    assert_eq!(r, 16);
}
