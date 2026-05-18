//! ai-lang standard library, in ai-lang source.
//!
//! This module hosts the canonical stdlib source as a string constant.
//! Tests + harnesses concatenate `SOURCE` ahead of user source before
//! compiling — the stdlib's defs get content-addressed hashes like
//! any other code, so dedup is free across modules.
//!
//! ## What's here
//!
//! - `Option<T>` + a few helpers (is_some, is_none, unwrap_or, ...).
//! - Math helpers on `Int` (abs, min, max, pow, ...). Where useful,
//!   functions are written tail-recursively so deep iteration doesn't
//!   blow the native stack — the TCO pass turns the recursion into
//!   a loop.
//! - Lightweight string helpers built on `string_len` / `string_eq` /
//!   `string_concat`. A real string stdlib (byte_at, slice, split,
//!   etc.) needs new primitives — flagged in code comments.
//!
//! Things NOT here yet:
//!
//! - `List<T>` / `Vec<T>` / `Map<K, V>` — these need recursive type
//!   definitions (a struct or enum referencing itself), which the
//!   resolver doesn't support yet. See task #37.
//! - `Result<T, E>` ergonomics (?-operator, `map_err`, ...) — basic
//!   `Result` exists for the `at()` builtin; helpers can be added
//!   here once we decide on naming.
//! - `Bytes` ops — String covers UTF-8; raw bytes would need a
//!   separate heap shape.

/// The canonical stdlib source. Concatenate ahead of user source
/// before compiling. Names declared here are visible to user code.
///
/// **Stability:** every change here invalidates the hashes of every
/// def that transitively depends on it. Add freely; rename / remove
/// only when prepared for a full re-hash.
pub const SOURCE: &str = r#"
// ---- Option ----
//
// Generic optional value. Used as the return type for partial
// operations (a "Some" carries the result; a "None" signals absence).
enum Option<T> { Some(T), None }

def opt_is_some<T>(opt: Option<T>) -> Int =
    match opt {
        Some(_) => 1,
        None => 0,
    }

def opt_is_none<T>(opt: Option<T>) -> Int =
    match opt {
        Some(_) => 0,
        None => 1,
    }

def opt_unwrap_or<T>(opt: Option<T>, default: T) -> T =
    match opt {
        Some(v) => v,
        None => default,
    }

// First Some wins; if `a` is None, fall back to `b`. Strict (both args
// evaluated). No lazy variant yet — needs higher-order generic.
def opt_or<T>(a: Option<T>, b: Option<T>) -> Option<T> =
    match a {
        Some(v) => Some(v),
        None => b,
    }

// ---- Math / Int ----
//
// All Int ops. Tail-recursive where they iterate so deep N runs
// without native-stack growth.

def abs(x: Int) -> Int =
    if x < 0 { 0 - x } else { x }

def max(a: Int, b: Int) -> Int =
    if a > b { a } else { b }

def min(a: Int, b: Int) -> Int =
    if a < b { a } else { b }

def clamp(x: Int, lo: Int, hi: Int) -> Int =
    if x < lo { lo } else { if x > hi { hi } else { x } }

def is_even(x: Int) -> Int =
    if x == 0 { 1 } else { if x < 0 { is_even(0 - x) } else { is_odd(x - 1) } }

def is_odd(x: Int) -> Int =
    if x == 0 { 0 } else { if x < 0 { is_odd(0 - x) } else { is_even(x - 1) } }

// pow(base, exp) — tail-recursive accumulator form.
def pow(base: Int, exp: Int) -> Int = pow_acc(base, exp, 1)

def pow_acc(base: Int, exp: Int, acc: Int) -> Int =
    if exp <= 0 { acc } else { pow_acc(base, exp - 1, acc * base) }

// Greatest common divisor (Euclid's). Tail-rec.
def gcd(a: Int, b: Int) -> Int =
    if b == 0 { abs(a) } else { gcd(b, a - (a / b) * b) }

// Sum of integers from 1..=n, inclusive. Closed-form is n*(n+1)/2,
// but the recursive accumulator form exercises tail-rec at depth.
def sum_to(n: Int) -> Int = sum_to_acc(n, 0)

def sum_to_acc(n: Int, acc: Int) -> Int =
    if n <= 0 { acc } else { sum_to_acc(n - 1, acc + n) }

// n! as a tail-rec accumulator. Negative n is treated as 0! = 1.
def factorial(n: Int) -> Int = factorial_acc(n, 1)

def factorial_acc(n: Int, acc: Int) -> Int =
    if n <= 1 { acc } else { factorial_acc(n - 1, acc * n) }

// Fibonacci, tail-rec via a sliding (a, b) state. fib(0)=0, fib(1)=1.
def fib(n: Int) -> Int = fib_acc(n, 0, 1)

def fib_acc(n: Int, a: Int, b: Int) -> Int =
    if n <= 0 { a } else { fib_acc(n - 1, b, a + b) }

// Sign function: -1 if x < 0, 0 if x == 0, 1 if x > 0.
def sign(x: Int) -> Int =
    if x == 0 { 0 } else { if x < 0 { 0 - 1 } else { 1 } }

// Least common multiple. lcm(a, 0) = 0 (degenerate but well-defined).
def lcm(a: Int, b: Int) -> Int =
    if a == 0 { 0 } else { if b == 0 { 0 } else { abs(a * b) / gcd(a, b) } }

// ---- String ----
//
// What we can build on len/eq/concat. Other ops (byte_at, slice,
// split) need new primitives.

def string_is_empty(s: String) -> Int =
    if string_len(s) == 0 { 1 } else { 0 }

// Repeat a string `n` times. Tail-rec with accumulator built up via
// concat — performance is O(n^2) due to repeated concatenation;
// fine for small n.
def string_repeat(s: String, n: Int) -> String = string_repeat_acc(s, n, "")

def string_repeat_acc(s: String, n: Int, acc: String) -> String =
    if n <= 0 { acc } else { string_repeat_acc(s, n - 1, string_concat(acc, s)) }

// ---- IntList ----
//
// A singly-linked list of Int. Non-generic for now — full generic
// `List<T>` requires user-side construction of generic enums (`Cons`
// for `List<Int>`) to thread the type argument through, which the
// resolver doesn't infer at construction sites yet.
//
// `Cell` is self-referential (via `IntList`) and `IntList` references
// `Cell` — that's a 2-member type SCC, handled by the new combined
// pass + Tarjan SCC.
struct IntListCell { head: Int, tail: IntList }
enum IntList { ICons(IntListCell), INil }

def intlist_length(xs: IntList) -> Int = intlist_length_acc(xs, 0)

def intlist_length_acc(xs: IntList, acc: Int) -> Int =
    match xs {
        ICons(cell) => intlist_length_acc(cell.tail, acc + 1),
        INil => acc,
    }

def intlist_sum(xs: IntList) -> Int = intlist_sum_acc(xs, 0)

def intlist_sum_acc(xs: IntList, acc: Int) -> Int =
    match xs {
        ICons(cell) => intlist_sum_acc(cell.tail, acc + cell.head),
        INil => acc,
    }

// Build a list `[lo, lo+1, ..., hi-1]` (right-exclusive). Tail-rec
// reverse-cons accumulator → reverse pass.
def intlist_range(lo: Int, hi: Int) -> IntList =
    intlist_reverse(intlist_range_acc(lo, hi, INil))

def intlist_range_acc(lo: Int, hi: Int, acc: IntList) -> IntList =
    if lo >= hi { acc } else {
        intlist_range_acc(lo + 1, hi, ICons(IntListCell { head: lo, tail: acc }))
    }

def intlist_reverse(xs: IntList) -> IntList = intlist_reverse_acc(xs, INil)

def intlist_reverse_acc(xs: IntList, acc: IntList) -> IntList =
    match xs {
        ICons(cell) => intlist_reverse_acc(
            cell.tail,
            ICons(IntListCell { head: cell.head, tail: acc }),
        ),
        INil => acc,
    }

// ---- Generic List<T> ----
//
// Generic singly-linked list. Construction relies on the resolver's
// bottom-up inference at variant/struct construction sites: given
// `Cons(ListCell { head: 1, tail: Nil })`, the field `head: 1` pins
// `T = Int`, which flows up through `ListCell<T>` and `List<T>` so
// the result type carries `Apply(List, [Int])`. Codegen sees the
// instantiation and (un)boxes Int payloads through the uniform
// pointer-typed slots.
//
// `list_length<T>` is fully generic — it never touches values of type
// T, only the spine. `list_sum` specialises to `List<Int>` because it
// reads `cell.head` as Int.
struct ListCell<T> { head: T, tail: List<T> }
enum List<T> { Cons(ListCell<T>), Nil }

def list_length<T>(xs: List<T>) -> Int = list_length_acc(xs, 0)

def list_length_acc<T>(xs: List<T>, acc: Int) -> Int =
    match xs {
        Cons(cell) => list_length_acc(cell.tail, acc + 1),
        Nil => acc,
    }

def list_sum(xs: List<Int>) -> Int = list_sum_acc(xs, 0)

def list_sum_acc(xs: List<Int>, acc: Int) -> Int =
    match xs {
        Cons(cell) => list_sum_acc(cell.tail, acc + cell.head),
        Nil => acc,
    }

def int_list_range(lo: Int, hi: Int) -> List<Int> =
    int_list_reverse(int_list_range_acc(lo, hi, Nil))

def int_list_range_acc(lo: Int, hi: Int, acc: List<Int>) -> List<Int> =
    if lo >= hi { acc } else {
        int_list_range_acc(lo + 1, hi, Cons(ListCell { head: lo, tail: acc }))
    }

def int_list_reverse(xs: List<Int>) -> List<Int> =
    int_list_reverse_acc(xs, Nil)

def int_list_reverse_acc(xs: List<Int>, acc: List<Int>) -> List<Int> =
    match xs {
        Cons(cell) => int_list_reverse_acc(
            cell.tail,
            Cons(ListCell { head: cell.head, tail: acc }),
        ),
        Nil => acc,
    }

// Empty-list test on a generic list. Touches only the spine, not T.
def list_is_empty<T>(xs: List<T>) -> Int =
    match xs {
        Nil => 1,
        Cons(_) => 0,
    }

// Generic reverse. Walks the spine, conses each head onto an
// accumulator, never touches T as a value.
def list_reverse<T>(xs: List<T>) -> List<T> = list_reverse_acc(xs, Nil)

def list_reverse_acc<T>(xs: List<T>, acc: List<T>) -> List<T> =
    match xs {
        Nil => acc,
        Cons(c) => list_reverse_acc(
            c.tail,
            Cons(ListCell { head: c.head, tail: acc }),
        ),
    }

// Generic index lookup. `None` when `i` is out of range (negative or
// past the end). Linear-time — we have no random-access lists.
def list_at<T>(xs: List<T>, i: Int) -> Option<T> =
    match xs {
        Nil => None,
        Cons(c) =>
            if i == 0 { Some(c.head) } else { list_at(c.tail, i - 1) },
    }

// Append two lists. `xs ++ ys` semantics. Tail-rec on the reverse of
// xs, conses ys, then reverses back — O(|xs|) but lazily two-pass.
// Specialised to `List<Int>` because the reverse step touches Int
// payloads (the unbox path); a fully-generic `list_append<T>` would
// need codegen to thread T through the temporary reversed accumulator
// without unboxing.
def list_append_int(xs: List<Int>, ys: List<Int>) -> List<Int> =
    list_append_int_acc(int_list_reverse(xs), ys)

def list_append_int_acc(rev_xs: List<Int>, acc: List<Int>) -> List<Int> =
    match rev_xs {
        Nil => acc,
        Cons(cell) => list_append_int_acc(
            cell.tail,
            Cons(ListCell { head: cell.head, tail: acc }),
        ),
    }

// Membership: 1 if `target` is in xs, 0 otherwise.
def list_contains_int(xs: List<Int>, target: Int) -> Int =
    match xs {
        Nil => 0,
        Cons(cell) =>
            if cell.head == target { 1 } else { list_contains_int(cell.tail, target) },
    }

// Index-based lookup. Returns Some(elem) if idx is in range, None if
// out of range (or negative).
def list_at_int(xs: List<Int>, idx: Int) -> Option<Int> =
    match xs {
        Nil => None,
        Cons(cell) =>
            if idx == 0 { Some(cell.head) } else { list_at_int(cell.tail, idx - 1) },
    }

// Max element. Returns None for an empty list.
def list_max_int(xs: List<Int>) -> Option<Int> =
    match xs {
        Nil => None,
        Cons(cell) => Some(list_max_int_acc(cell.tail, cell.head)),
    }

def list_max_int_acc(xs: List<Int>, cur: Int) -> Int =
    match xs {
        Nil => cur,
        Cons(cell) => list_max_int_acc(cell.tail, max(cur, cell.head)),
    }

// Min element. Returns None for an empty list.
def list_min_int(xs: List<Int>) -> Option<Int> =
    match xs {
        Nil => None,
        Cons(cell) => Some(list_min_int_acc(cell.tail, cell.head)),
    }

def list_min_int_acc(xs: List<Int>, cur: Int) -> Int =
    match xs {
        Nil => cur,
        Cons(cell) => list_min_int_acc(cell.tail, min(cur, cell.head)),
    }

// ---- Higher-order ops ----
//
// Both Int-specialised and fully generic versions are provided. The
// generic versions rely on the uniform closure ABI: every lifted
// lambda has LLVM signature `(*Thread, *Closure, ptr × N) -> ptr` and
// the body unboxes Int params at entry / boxes Int returns at exit.
// Indirect call sites box/unbox based on the closure's declared
// FnType. This lets `list_map<T, U>(xs: List<T>, f: fn(T)->U)` work
// for any concrete instantiation including `T = Int`.

def list_map_int(xs: List<Int>, f: fn(Int) -> Int) -> List<Int> =
    match xs {
        Nil => Nil,
        Cons(c) => Cons(ListCell { head: f(c.head), tail: list_map_int(c.tail, f) }),
    }

def list_filter_int(xs: List<Int>, pred: fn(Int) -> Int) -> List<Int> =
    match xs {
        Nil => Nil,
        Cons(c) =>
            if pred(c.head) == 0 {
                list_filter_int(c.tail, pred)
            } else {
                Cons(ListCell { head: c.head, tail: list_filter_int(c.tail, pred) })
            },
    }

def list_foldl_int(xs: List<Int>, init: Int, f: fn(Int, Int) -> Int) -> Int =
    match xs {
        Nil => init,
        Cons(c) => list_foldl_int(c.tail, f(init, c.head), f),
    }

// `list_any_int(xs, p) = 1` iff there exists an element where p returns
// non-zero; `0` otherwise.
def list_any_int(xs: List<Int>, p: fn(Int) -> Int) -> Int =
    match xs {
        Nil => 0,
        Cons(c) =>
            if p(c.head) == 0 { list_any_int(c.tail, p) } else { 1 },
    }

def list_all_int(xs: List<Int>, p: fn(Int) -> Int) -> Int =
    match xs {
        Nil => 1,
        Cons(c) =>
            if p(c.head) == 0 { 0 } else { list_all_int(c.tail, p) },
    }

// Fully generic versions — work for any T, U via the uniform closure ABI.

def list_map<T, U>(xs: List<T>, f: fn(T) -> U) -> List<U> =
    match xs {
        Nil => Nil,
        Cons(c) => Cons(ListCell { head: f(c.head), tail: list_map(c.tail, f) }),
    }

def list_foldl<T, U>(xs: List<T>, init: U, f: fn(U, T) -> U) -> U =
    match xs {
        Nil => init,
        Cons(c) => list_foldl(c.tail, f(init, c.head), f),
    }

def list_filter<T>(xs: List<T>, pred: fn(T) -> Int) -> List<T> =
    match xs {
        Nil => Nil,
        Cons(c) =>
            if pred(c.head) == 0 {
                list_filter(c.tail, pred)
            } else {
                Cons(ListCell { head: c.head, tail: list_filter(c.tail, pred) })
            },
    }

def opt_map<T, U>(opt: Option<T>, f: fn(T) -> U) -> Option<U> =
    match opt {
        Some(v) => Some(f(v)),
        None => None,
    }

// ---- I/O ----
//
// These bind to Rust functions registered in the global FFI registry
// by `io_externs::register_io_externs()` (called automatically by
// `init_native_target`). All ai-lang programs running under the
// standard JIT setup can call these directly — no manual extern
// registration required.

extern fn print_int(n: Int) -> Int
extern fn print_string(s: String) -> Int
extern fn println(s: String) -> Int
extern fn read_line() -> String
extern fn int_to_string(n: Int) -> String
extern fn string_to_int(s: String) -> Int
extern fn string_is_int(s: String) -> Int

// Sleep for the given number of milliseconds. Returns 0.
extern fn sleep_ms(ms: Int) -> Int

// Command-line args passed to the runner after `--`. `arg_count()` is
// the number of args; `get_arg(i)` returns the i-th (empty string on
// out-of-range).
extern fn arg_count() -> Int
extern fn get_arg(i: Int) -> String

// Worker nodes spawned by the runner via `--nodes=N`. The runner
// populates the port table before invoking `main()`. `node_count()` is
// the number of workers; `get_node_port(i)` is the i-th worker's TCP
// port on 127.0.0.1 (or -1 if out of range).
extern fn node_count() -> Int
extern fn get_node_port(i: Int) -> Int

// `println_int(n)` — print an Int followed by newline. Reuses the
// existing string-formatting and println paths.
def println_int(n: Int) -> Int = println(int_to_string(n))

// Parse-or-default: convert a String to Int, or return `default` if
// the string isn't a valid integer. Uses the `string_is_int` probe to
// avoid bare 0 ambiguity.
def parse_int_or(s: String, default: Int) -> Int =
    if string_is_int(s) == 0 { default } else { string_to_int(s) }
"#;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CompiledModule, Jit, def_symbol, init_native_target};
    use crate::hash::Hash;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;
    use crate::runtime::{Runtime, Thread};
    use inkwell::context::Context;
    use std::collections::HashMap;
    use std::sync::Once;

    static INIT: Once = Once::new();
    fn init() {
        INIT.call_once(|| {
            init_native_target().expect("init native target");
        });
    }

    /// Build the stdlib + user source as one module. Returns the
    /// JIT-ready Runtime, Jit, and name→hash map. The stdlib's
    /// content-addressed hashes flow through naturally — every test
    /// that uses the same stdlib version will agree on them.
    fn build_with_stdlib<'ctx>(
        ctx: &'ctx Context,
        user_src: &str,
    ) -> (Runtime, Jit<'ctx>, HashMap<String, Hash>) {
        let combined = format!("{}\n{}", SOURCE, user_src);
        let m = parse_module(&combined).expect("parse stdlib + user");
        let r = resolve_module(&m).expect("resolve");
        // Typecheck the whole program before codegen. Catches bugs
        // that codegen would silently accept. If this fails, the
        // panic message names the offending def + error.
        let mut tc_cache = crate::typecheck::TypeCache::new();
        crate::typecheck::typecheck_module(&r, &mut tc_cache)
            .expect("typecheck stdlib + user");
        let names: HashMap<String, Hash> =
            r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
        let cm = CompiledModule::build(ctx, &r).expect("build module");
        let rt = Runtime::new_with_metadata(
            cm.closure_type_infos.clone(),
            cm.shape_registry.clone(),
            cm.shape_meta.clone(),
            cm.shape_by_type_id.clone(),
        );
        let jit = Jit::new(cm, &rt).expect("init Jit");
        (rt, jit, names)
    }

    #[test]
    fn stdlib_parses_and_resolves() {
        init();
        let m = parse_module(SOURCE).expect("stdlib must parse");
        resolve_module(&m).expect("stdlib must resolve");
    }

    #[test]
    fn math_abs_min_max_clamp() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_abs(x: Int) -> Int = abs(x)
             def t_min(a: Int, b: Int) -> Int = min(a, b)
             def t_max(a: Int, b: Int) -> Int = max(a, b)
             def t_clamp(x: Int, lo: Int, hi: Int) -> Int = clamp(x, lo, hi)",
        );
        unsafe {
            let f = jit.get_fn1(&names["t_abs"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), -7), 7);
            assert_eq!(f.call(rt.thread_ptr(), 0), 0);
            assert_eq!(f.call(rt.thread_ptr(), 42), 42);

            let f = jit.get_fn2(&names["t_min"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 3, 5), 3);
            assert_eq!(f.call(rt.thread_ptr(), 9, 1), 1);

            let f = jit.get_fn2(&names["t_max"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 3, 5), 5);

            let f = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64, i64) -> i64>(
                    &def_symbol(&names["t_clamp"]),
                )
                .unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 5, 0, 10), 5);
            assert_eq!(f.call(rt.thread_ptr(), -3, 0, 10), 0);
            assert_eq!(f.call(rt.thread_ptr(), 99, 0, 10), 10);
        }
    }

    #[test]
    fn math_pow_via_tail_rec() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) =
            build_with_stdlib(&ctx, "def t_pow(b: Int, e: Int) -> Int = pow(b, e)");
        unsafe {
            let f = jit.get_fn2(&names["t_pow"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 2, 0), 1);
            assert_eq!(f.call(rt.thread_ptr(), 2, 1), 2);
            assert_eq!(f.call(rt.thread_ptr(), 2, 10), 1024);
            assert_eq!(f.call(rt.thread_ptr(), 3, 4), 81);
        }
    }

    #[test]
    fn math_gcd() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) =
            build_with_stdlib(&ctx, "def t_gcd(a: Int, b: Int) -> Int = gcd(a, b)");
        unsafe {
            let f = jit.get_fn2(&names["t_gcd"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 12, 8), 4);
            assert_eq!(f.call(rt.thread_ptr(), 17, 5), 1);
            assert_eq!(f.call(rt.thread_ptr(), 100, 75), 25);
        }
    }

    /// `sum_to(N)` uses a tail-rec accumulator; should run at depth
    /// well past any reasonable native-stack budget.
    #[test]
    fn math_sum_to_deep() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) =
            build_with_stdlib(&ctx, "def t_sum(n: Int) -> Int = sum_to(n)");
        unsafe {
            let f = jit.get_fn1(&names["t_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 100), 5050);
            // 100k iterations via tail-rec.
            assert_eq!(
                f.call(rt.thread_ptr(), 100_000),
                100_000_i64 * 100_001 / 2
            );
        }
    }

    /// Mutually-recursive is_even / is_odd. Mutual TCO isn't done yet,
    /// so this only runs at sub-stack-overflow depth.
    #[test]
    fn math_is_even_shallow() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) =
            build_with_stdlib(&ctx, "def t_even(n: Int) -> Int = is_even(n)");
        unsafe {
            let f = jit.get_fn1(&names["t_even"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 1);
            assert_eq!(f.call(rt.thread_ptr(), 1), 0);
            assert_eq!(f.call(rt.thread_ptr(), 100), 1);
            assert_eq!(f.call(rt.thread_ptr(), 1001), 0);
        }
    }

    #[test]
    fn option_helpers() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_some_is_some() -> Int = opt_is_some(Some(42))
             def t_none_is_some() -> Int = opt_is_some(None)
             def t_unwrap_some() -> Int = opt_unwrap_or(Some(7), 99)
             def t_unwrap_none() -> Int = opt_unwrap_or(None, 99)",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_some_is_some"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["t_none_is_some"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
            let f = jit.get_fn0(&names["t_unwrap_some"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 7);
            let f = jit.get_fn0(&names["t_unwrap_none"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 99);
        }
    }

    #[test]
    fn string_is_empty_works() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_empty() -> Int = string_is_empty(\"\")
             def t_nonempty() -> Int = string_is_empty(\"x\")",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_empty"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["t_nonempty"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
    }

    #[test]
    fn intlist_length_and_sum() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def make_3() -> IntList =
                 ICons(IntListCell { head: 10,
                   tail: ICons(IntListCell { head: 20,
                     tail: ICons(IntListCell { head: 30, tail: INil }) }) })
             def t_len() -> Int = intlist_length(make_3())
             def t_sum() -> Int = intlist_sum(make_3())
             def t_empty_len() -> Int = intlist_length(INil)",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_len"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 3);
            let f = jit.get_fn0(&names["t_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 60);
            let f = jit.get_fn0(&names["t_empty_len"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
    }

    #[test]
    fn intlist_range_and_reverse() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_range_5_len() -> Int = intlist_length(intlist_range(0, 5))
             def t_range_5_sum() -> Int = intlist_sum(intlist_range(0, 5))
             def t_range_100_sum() -> Int = intlist_sum(intlist_range(0, 100))
             // Reversing twice should give the same list — verify by sum.
             def t_rev_rev_sum() -> Int =
                intlist_sum(intlist_reverse(intlist_reverse(intlist_range(1, 11))))",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_range_5_len"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 5);
            let f = jit.get_fn0(&names["t_range_5_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0 + 1 + 2 + 3 + 4);
            let f = jit.get_fn0(&names["t_range_100_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), (0..100).sum::<i64>());
            let f = jit.get_fn0(&names["t_rev_rev_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), (1..=10).sum::<i64>());
        }
    }

    #[test]
    fn generic_list_int_basic() {
        // Build a 3-element `List<Int>` directly using the generic
        // `Cons` / `ListCell` / `Nil`. The resolver must infer
        // `T = Int` bottom-up from `head: 1` etc., and codegen must
        // box Int values into the TypeVar-typed fields, then unbox
        // them on `cell.head` access inside `list_sum`.
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def make_3() -> List<Int> =
                 Cons(ListCell { head: 10,
                   tail: Cons(ListCell { head: 20,
                     tail: Cons(ListCell { head: 30, tail: Nil }) }) })
             def t_len() -> Int = list_length(make_3())
             def t_sum() -> Int = list_sum(make_3())
             def t_empty_len() -> Int = list_length(Nil)",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_len"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 3);
            let f = jit.get_fn0(&names["t_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 60);
            let f = jit.get_fn0(&names["t_empty_len"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
    }

    #[test]
    fn generic_list_range_and_reverse() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_range_5_len() -> Int = list_length(int_list_range(0, 5))
             def t_range_5_sum() -> Int = list_sum(int_list_range(0, 5))
             def t_range_100_sum() -> Int = list_sum(int_list_range(0, 100))
             def t_rev_rev_sum() -> Int =
                list_sum(int_list_reverse(int_list_reverse(int_list_range(1, 11))))",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_range_5_len"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 5);
            let f = jit.get_fn0(&names["t_range_5_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0 + 1 + 2 + 3 + 4);
            let f = jit.get_fn0(&names["t_range_100_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), (0..100).sum::<i64>());
            let f = jit.get_fn0(&names["t_rev_rev_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), (1..=10).sum::<i64>());
        }
    }

    #[test]
    fn string_repeat_works() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_repeat_3() -> Int =
                string_len(string_repeat(\"ab\", 3))
             def t_repeat_zero() -> Int =
                string_len(string_repeat(\"ab\", 0))
             def t_repeat_eq() -> Int =
                string_eq(string_repeat(\"ab\", 3), \"ababab\")",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_repeat_3"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 6);
            let f = jit.get_fn0(&names["t_repeat_zero"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
            let f = jit.get_fn0(&names["t_repeat_eq"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
        }
    }

    #[test]
    fn math_factorial_and_fib() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_fact(n: Int) -> Int = factorial(n)
             def t_fib(n: Int) -> Int = fib(n)",
        );
        unsafe {
            let f = jit.get_fn1(&names["t_fact"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 1);
            assert_eq!(f.call(rt.thread_ptr(), 1), 1);
            assert_eq!(f.call(rt.thread_ptr(), 5), 120);
            assert_eq!(f.call(rt.thread_ptr(), 10), 3_628_800);

            let f = jit.get_fn1(&names["t_fib"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 0);
            assert_eq!(f.call(rt.thread_ptr(), 1), 1);
            assert_eq!(f.call(rt.thread_ptr(), 2), 1);
            assert_eq!(f.call(rt.thread_ptr(), 10), 55);
            assert_eq!(f.call(rt.thread_ptr(), 20), 6765);
            // Deep — tail-rec eats the loop.
            assert_eq!(f.call(rt.thread_ptr(), 90), 2_880_067_194_370_816_120);
        }
    }

    #[test]
    fn math_sign_and_lcm() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_sign(x: Int) -> Int = sign(x)
             def t_lcm(a: Int, b: Int) -> Int = lcm(a, b)",
        );
        unsafe {
            let f = jit.get_fn1(&names["t_sign"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), -7), -1);
            assert_eq!(f.call(rt.thread_ptr(), 0), 0);
            assert_eq!(f.call(rt.thread_ptr(), 42), 1);

            let f = jit.get_fn2(&names["t_lcm"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 4, 6), 12);
            assert_eq!(f.call(rt.thread_ptr(), 21, 6), 42);
            assert_eq!(f.call(rt.thread_ptr(), 7, 11), 77);
            assert_eq!(f.call(rt.thread_ptr(), 0, 5), 0);
        }
    }

    #[test]
    fn option_or_short_circuit() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_some_or_some() -> Int = opt_unwrap_or(opt_or(Some(1), Some(2)), 99)
             def t_none_or_some() -> Int = opt_unwrap_or(opt_or(None, Some(7)), 99)
             def t_none_or_none() -> Int = opt_unwrap_or(opt_or(None, None), 99)",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_some_or_some"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["t_none_or_some"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 7);
            let f = jit.get_fn0(&names["t_none_or_none"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 99);
        }
    }

    #[test]
    fn generic_list_is_empty() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_empty() -> Int = list_is_empty(int_list_range(0, 0))
             def t_three() -> Int = list_is_empty(int_list_range(0, 3))",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_empty"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["t_three"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
    }

    #[test]
    fn list_int_contains_and_at() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "// xs = [0, 1, 2, 3, 4]
             def t_has_3() -> Int = list_contains_int(int_list_range(0, 5), 3)
             def t_has_99() -> Int = list_contains_int(int_list_range(0, 5), 99)
             def t_at_0() -> Int = opt_unwrap_or(list_at_int(int_list_range(0, 5), 0), -1)
             def t_at_4() -> Int = opt_unwrap_or(list_at_int(int_list_range(0, 5), 4), -1)
             def t_at_oob() -> Int = opt_unwrap_or(list_at_int(int_list_range(0, 5), 99), -1)
             def t_at_neg() -> Int = opt_unwrap_or(list_at_int(int_list_range(0, 5), 0 - 1), -1)",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_has_3"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["t_has_99"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
            let f = jit.get_fn0(&names["t_at_0"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
            let f = jit.get_fn0(&names["t_at_4"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 4);
            let f = jit.get_fn0(&names["t_at_oob"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), -1);
            let f = jit.get_fn0(&names["t_at_neg"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), -1);
        }
    }

    #[test]
    fn list_int_max_min() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def mk5() -> List<Int> =
                 Cons(ListCell { head: 3,
                   tail: Cons(ListCell { head: 1,
                     tail: Cons(ListCell { head: 4,
                       tail: Cons(ListCell { head: 1,
                         tail: Cons(ListCell { head: 5, tail: Nil }) }) }) }) })
             def t_max() -> Int = opt_unwrap_or(list_max_int(mk5()), -1)
             def t_min() -> Int = opt_unwrap_or(list_min_int(mk5()), -1)
             def t_max_empty() -> Int = opt_unwrap_or(list_max_int(Nil), -42)
             def t_min_empty() -> Int = opt_unwrap_or(list_min_int(Nil), -42)",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_max"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 5);
            let f = jit.get_fn0(&names["t_min"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["t_max_empty"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), -42);
            let f = jit.get_fn0(&names["t_min_empty"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), -42);
        }
    }

    #[test]
    fn generic_hof_list_map_int_to_int() {
        // Fully generic `list_map<T, U>` instantiated with T=U=Int.
        // Closure boxing kicks in at the indirect-call site.
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t() -> Int =
                 list_sum(list_map(int_list_range(1, 6), |x: Int| x * 10))",
        );
        unsafe {
            let f = jit.get_fn0(&names["t"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), (1 + 2 + 3 + 4 + 5) * 10);
        }
    }

    #[test]
    fn generic_hof_list_foldl_int() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_sum() -> Int =
                 list_foldl(int_list_range(1, 6), 0, |acc: Int, x: Int| acc + x)
             def t_product() -> Int =
                 list_foldl(int_list_range(1, 6), 1, |acc: Int, x: Int| acc * x)",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1 + 2 + 3 + 4 + 5);
            let f = jit.get_fn0(&names["t_product"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 120);
        }
    }

    #[test]
    fn generic_hof_list_filter_int() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_evens_sum() -> Int =
                 list_sum(list_filter(int_list_range(1, 11),
                     |x: Int| if x - (x / 2) * 2 == 0 { 1 } else { 0 }))",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_evens_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 2 + 4 + 6 + 8 + 10);
        }
    }

    #[test]
    fn generic_hof_opt_map() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_some() -> Int =
                 opt_unwrap_or(opt_map(Some(21), |x: Int| x * 2), -1)
             def t_none() -> Int =
                 opt_unwrap_or(opt_map(None, |x: Int| x * 2), 99)",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_some"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 42);
            let f = jit.get_fn0(&names["t_none"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 99);
        }
    }

    #[test]
    fn io_int_string_conversions() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_int_to_int(n: Int) -> Int = string_to_int(int_to_string(n))
             def t_parse_good() -> Int = parse_int_or(\"123\", -1)
             def t_parse_bad() -> Int = parse_int_or(\"oops\", -1)
             def t_parse_empty() -> Int = parse_int_or(\"\", 0)
             def t_parse_negative() -> Int = parse_int_or(\"-42\", 0)
             def t_is_int_yes() -> Int = string_is_int(\"42\")
             def t_is_int_no() -> Int = string_is_int(\"4x2\")",
        );
        unsafe {
            let f = jit.get_fn1(&names["t_int_to_int"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 0);
            assert_eq!(f.call(rt.thread_ptr(), 12345), 12345);
            assert_eq!(f.call(rt.thread_ptr(), -7), -7);
            let f = jit.get_fn0(&names["t_parse_good"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 123);
            let f = jit.get_fn0(&names["t_parse_bad"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), -1);
            let f = jit.get_fn0(&names["t_parse_empty"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
            let f = jit.get_fn0(&names["t_parse_negative"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), -42);
            let f = jit.get_fn0(&names["t_is_int_yes"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["t_is_int_no"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
    }

    #[test]
    fn io_print_via_redirect() {
        // We can't easily intercept the real stdout from a test (it
        // depends on the test runner's redirection). Verify the
        // print-side externs at least don't crash — coverage of the
        // ABI path (heap String pointers, no GC corruption around the
        // call).
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t() -> Int = {
                 let _a = print_string(\"\");
                 let _b = println_int(0);
                 let _c = print_int(0);
                 print_string(\"\")
             }",
        );
        unsafe {
            let f = jit.get_fn0(&names["t"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
    }

    #[test]
    fn list_int_higher_order() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_map_sum() -> Int =
                 list_sum(list_map_int(int_list_range(1, 6), |x: Int| x * 10))
             def t_filter_even_sum() -> Int =
                 list_sum(list_filter_int(int_list_range(1, 11), |x: Int|
                     if x - (x / 2) * 2 == 0 { 1 } else { 0 }))
             def t_fold_sum() -> Int =
                 list_foldl_int(int_list_range(1, 6), 0, |acc: Int, x: Int| acc + x)
             def t_fold_product() -> Int =
                 list_foldl_int(int_list_range(1, 6), 1, |acc: Int, x: Int| acc * x)
             def t_any_gt_3() -> Int =
                 list_any_int(int_list_range(0, 5), |x: Int| if x > 3 { 1 } else { 0 })
             def t_all_lt_99() -> Int =
                 list_all_int(int_list_range(0, 5), |x: Int| if x < 99 { 1 } else { 0 })
             def t_none_gt_99() -> Int =
                 list_any_int(int_list_range(0, 5), |x: Int| if x > 99 { 1 } else { 0 })",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_map_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), (1 + 2 + 3 + 4 + 5) * 10);
            let f = jit.get_fn0(&names["t_filter_even_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 2 + 4 + 6 + 8 + 10);
            let f = jit.get_fn0(&names["t_fold_sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1 + 2 + 3 + 4 + 5);
            let f = jit.get_fn0(&names["t_fold_product"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 120);
            let f = jit.get_fn0(&names["t_any_gt_3"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["t_all_lt_99"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["t_none_gt_99"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
    }

    #[test]
    fn list_int_append() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_append_lens() -> Int =
                 list_length(list_append_int(int_list_range(0, 3), int_list_range(10, 15)))
             def t_append_sum() -> Int =
                 list_sum(list_append_int(int_list_range(0, 5), int_list_range(10, 13)))
             def t_append_empty_left() -> Int =
                 list_sum(list_append_int(Nil, int_list_range(1, 5)))
             def t_append_empty_right() -> Int =
                 list_sum(list_append_int(int_list_range(1, 5), Nil))",
        );
        unsafe {
            let f = jit.get_fn0(&names["t_append_lens"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 3 + 5);
            let f = jit.get_fn0(&names["t_append_sum"]).unwrap();
            // [0..5) + [10..13) = 0+1+2+3+4 + 10+11+12 = 10 + 33 = 43
            assert_eq!(f.call(rt.thread_ptr()), 43);
            let f = jit.get_fn0(&names["t_append_empty_left"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1 + 2 + 3 + 4);
            let f = jit.get_fn0(&names["t_append_empty_right"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1 + 2 + 3 + 4);
        }
    }
}
