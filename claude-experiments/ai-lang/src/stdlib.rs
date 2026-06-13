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
        Option::Some(_) => 1,
        Option::None => 0,
    }

def opt_is_none<T>(opt: Option<T>) -> Int =
    match opt {
        Option::Some(_) => 0,
        Option::None => 1,
    }

def opt_unwrap_or<T>(opt: Option<T>, default: T) -> T =
    match opt {
        Option::Some(v) => v,
        Option::None => default,
    }

// First Some wins; if `a` is None, fall back to `b`. Strict (both args
// evaluated). No lazy variant yet — needs higher-order generic.
def opt_or<T>(a: Option<T>, b: Option<T>) -> Option<T> =
    match a {
        Option::Some(v) => Option::Some(v),
        Option::None => b,
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
        IntList::ICons(cell) => intlist_length_acc(cell.tail, acc + 1),
        IntList::INil => acc,
    }

def intlist_sum(xs: IntList) -> Int = intlist_sum_acc(xs, 0)

def intlist_sum_acc(xs: IntList, acc: Int) -> Int =
    match xs {
        IntList::ICons(cell) => intlist_sum_acc(cell.tail, acc + cell.head),
        IntList::INil => acc,
    }

// Build a list `[lo, lo+1, ..., hi-1]` (right-exclusive). Tail-rec
// reverse-cons accumulator → reverse pass.
def intlist_range(lo: Int, hi: Int) -> IntList =
    intlist_reverse(intlist_range_acc(lo, hi, IntList::INil))

def intlist_range_acc(lo: Int, hi: Int, acc: IntList) -> IntList =
    if lo >= hi { acc } else {
        intlist_range_acc(lo + 1, hi, IntList::ICons(IntListCell { head: lo, tail: acc }))
    }

def intlist_reverse(xs: IntList) -> IntList = intlist_reverse_acc(xs, IntList::INil)

def intlist_reverse_acc(xs: IntList, acc: IntList) -> IntList =
    match xs {
        IntList::ICons(cell) => intlist_reverse_acc(
            cell.tail,
            IntList::ICons(IntListCell { head: cell.head, tail: acc }),
        ),
        IntList::INil => acc,
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
        List::Cons(cell) => list_length_acc(cell.tail, acc + 1),
        List::Nil => acc,
    }

def list_sum(xs: List<Int>) -> Int = list_sum_acc(xs, 0)

def list_sum_acc(xs: List<Int>, acc: Int) -> Int =
    match xs {
        List::Cons(cell) => list_sum_acc(cell.tail, acc + cell.head),
        List::Nil => acc,
    }

def int_list_range(lo: Int, hi: Int) -> List<Int> =
    int_list_reverse(int_list_range_acc(lo, hi, List::Nil))

def int_list_range_acc(lo: Int, hi: Int, acc: List<Int>) -> List<Int> =
    if lo >= hi { acc } else {
        int_list_range_acc(lo + 1, hi, List::Cons(ListCell { head: lo, tail: acc }))
    }

def int_list_reverse(xs: List<Int>) -> List<Int> =
    int_list_reverse_acc(xs, List::Nil)

def int_list_reverse_acc(xs: List<Int>, acc: List<Int>) -> List<Int> =
    match xs {
        List::Cons(cell) => int_list_reverse_acc(
            cell.tail,
            List::Cons(ListCell { head: cell.head, tail: acc }),
        ),
        List::Nil => acc,
    }

// Empty-list test on a generic list. Touches only the spine, not T.
def list_is_empty<T>(xs: List<T>) -> Int =
    match xs {
        List::Nil => 1,
        List::Cons(_) => 0,
    }

// Generic reverse. Walks the spine, conses each head onto an
// accumulator, never touches T as a value.
def list_reverse<T>(xs: List<T>) -> List<T> = list_reverse_acc(xs, List::Nil)

def list_reverse_acc<T>(xs: List<T>, acc: List<T>) -> List<T> =
    match xs {
        List::Nil => acc,
        List::Cons(c) => list_reverse_acc(
            c.tail,
            List::Cons(ListCell { head: c.head, tail: acc }),
        ),
    }

// Generic index lookup. `None` when `i` is out of range (negative or
// past the end). Linear-time — we have no random-access lists.
def list_at<T>(xs: List<T>, i: Int) -> Option<T> =
    match xs {
        List::Nil => Option::None,
        List::Cons(c) =>
            if i == 0 { Option::Some(c.head) } else { list_at(c.tail, i - 1) },
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
        List::Nil => acc,
        List::Cons(cell) => list_append_int_acc(
            cell.tail,
            List::Cons(ListCell { head: cell.head, tail: acc }),
        ),
    }

// Membership: 1 if `target` is in xs, 0 otherwise.
def list_contains_int(xs: List<Int>, target: Int) -> Int =
    match xs {
        List::Nil => 0,
        List::Cons(cell) =>
            if cell.head == target { 1 } else { list_contains_int(cell.tail, target) },
    }

// Index-based lookup. Returns Some(elem) if idx is in range, None if
// out of range (or negative).
def list_at_int(xs: List<Int>, idx: Int) -> Option<Int> =
    match xs {
        List::Nil => Option::None,
        List::Cons(cell) =>
            if idx == 0 { Option::Some(cell.head) } else { list_at_int(cell.tail, idx - 1) },
    }

// Max element. Returns None for an empty list.
def list_max_int(xs: List<Int>) -> Option<Int> =
    match xs {
        List::Nil => Option::None,
        List::Cons(cell) => Option::Some(list_max_int_acc(cell.tail, cell.head)),
    }

def list_max_int_acc(xs: List<Int>, cur: Int) -> Int =
    match xs {
        List::Nil => cur,
        List::Cons(cell) => list_max_int_acc(cell.tail, max(cur, cell.head)),
    }

// Min element. Returns None for an empty list.
def list_min_int(xs: List<Int>) -> Option<Int> =
    match xs {
        List::Nil => Option::None,
        List::Cons(cell) => Option::Some(list_min_int_acc(cell.tail, cell.head)),
    }

def list_min_int_acc(xs: List<Int>, cur: Int) -> Int =
    match xs {
        List::Nil => cur,
        List::Cons(cell) => list_min_int_acc(cell.tail, min(cur, cell.head)),
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
        List::Nil => List::Nil,
        List::Cons(c) => List::Cons(ListCell { head: f(c.head), tail: list_map_int(c.tail, f) }),
    }

def list_filter_int(xs: List<Int>, pred: fn(Int) -> Int) -> List<Int> =
    match xs {
        List::Nil => List::Nil,
        List::Cons(c) =>
            if pred(c.head) == 0 {
                list_filter_int(c.tail, pred)
            } else {
                List::Cons(ListCell { head: c.head, tail: list_filter_int(c.tail, pred) })
            },
    }

def list_foldl_int(xs: List<Int>, init: Int, f: fn(Int, Int) -> Int) -> Int =
    match xs {
        List::Nil => init,
        List::Cons(c) => list_foldl_int(c.tail, f(init, c.head), f),
    }

// `list_any_int(xs, p) = 1` iff there exists an element where p returns
// non-zero; `0` otherwise.
def list_any_int(xs: List<Int>, p: fn(Int) -> Int) -> Int =
    match xs {
        List::Nil => 0,
        List::Cons(c) =>
            if p(c.head) == 0 { list_any_int(c.tail, p) } else { 1 },
    }

def list_all_int(xs: List<Int>, p: fn(Int) -> Int) -> Int =
    match xs {
        List::Nil => 1,
        List::Cons(c) =>
            if p(c.head) == 0 { 0 } else { list_all_int(c.tail, p) },
    }

// Fully generic versions — work for any T, U via the uniform closure ABI.

def list_map<T, U>(xs: List<T>, f: fn(T) -> U) -> List<U> =
    match xs {
        List::Nil => List::Nil,
        List::Cons(c) => List::Cons(ListCell { head: f(c.head), tail: list_map(c.tail, f) }),
    }

def list_foldl<T, U>(xs: List<T>, init: U, f: fn(U, T) -> U) -> U =
    match xs {
        List::Nil => init,
        List::Cons(c) => list_foldl(c.tail, f(init, c.head), f),
    }

def list_filter<T>(xs: List<T>, pred: fn(T) -> Int) -> List<T> =
    match xs {
        List::Nil => List::Nil,
        List::Cons(c) =>
            if pred(c.head) == 0 {
                list_filter(c.tail, pred)
            } else {
                List::Cons(ListCell { head: c.head, tail: list_filter(c.tail, pred) })
            },
    }

def opt_map<T, U>(opt: Option<T>, f: fn(T) -> U) -> Option<U> =
    match opt {
        Option::Some(v) => Option::Some(f(v)),
        Option::None => Option::None,
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

// ---- C string marshaling (built on the raw-pointer intrinsics) ----
//
// The generic C FFI passes `Ptr` (a raw i64 address). To hand a String
// to a C function we copy its bytes into a malloc'd, NUL-terminated
// buffer (`cstr`); to read one back we walk to the NUL and copy into a
// GC String (`cstr_to_string`). These two are the bridge between the
// language's heap strings and C's `char*`.

extern "C" lib "c" {
    fn malloc(size: Int) -> Ptr
    fn free(p: Ptr) -> Int
    fn open_memstream(bufp: Ptr, sizep: Ptr) -> Ptr
    fn fflush(stream: Ptr) -> Int
    fn fclose(stream: Ptr) -> Int
    // OS surface (used by the env/clock/filesystem stdlib below).
    fn getenv(name: Ptr) -> Ptr
    fn time(tloc: Ptr) -> Int
    fn clock_gettime(clk_id: Int, tp: Ptr) -> Int
    fn fopen(path: Ptr, mode: Ptr) -> Ptr
    fn fread(ptr: Ptr, size: Int, nmemb: Int, stream: Ptr) -> Int
    fn fwrite(ptr: Ptr, size: Int, nmemb: Int, stream: Ptr) -> Int
    fn fseek(stream: Ptr, offset: Int, whence: Int) -> Int
    fn ftell(stream: Ptr) -> Int
    fn access(path: Ptr, mode: Int) -> Int
    // Calendar time (used by the date formatting below, e.g. AWS SigV4).
    fn gmtime_r(timep: Ptr, result: Ptr) -> Ptr
    fn strftime(s: Ptr, maxsize: Int, format: Ptr, timeptr: Ptr) -> Int
}

// Copy `s` into a fresh malloc'd, NUL-terminated C buffer. The caller
// owns the buffer and must `free` it.
def cstr(s: String) -> Result<Ptr, IndexError> = {
    let b = bytes_from_string(s);
    let n = bytes_len(b);
    let p = malloc(n + 1);
    cstr_fill(p, b, 0, n)
}

def cstr_fill(p: Ptr, b: Bytes, i: Int, n: Int) -> Result<Ptr, IndexError> =
    if i == n {
        let t = ptr_write_u8(p, n, 0);
        Result::Ok(p)
    } else {
        let w = ptr_write_u8(p, i, bytes_get(b, i)?);
        cstr_fill(p, b, i + 1, n)
    }

// Read a NUL-terminated C string at `p` into a fresh GC String.
def cstr_to_string(p: Ptr) -> Result<String, IndexError> = {
    let n = cstr_len(p, 0);
    let b = bytes_new(n);
    let filled = cstr_copy_out(p, b, 0, n)?;
    Result::Ok(string_from_bytes(filled))
}

def cstr_len(p: Ptr, i: Int) -> Int =
    if ptr_read_u8(p, i) == 0 { i } else { cstr_len(p, i + 1) }

def cstr_copy_out(p: Ptr, b: Bytes, i: Int, n: Int) -> Result<Bytes, IndexError> =
    if i == n {
        Result::Ok(b)
    } else {
        let w = bytes_set(b, i, ptr_read_u8(p, i))?;
        cstr_copy_out(p, b, i + 1, n)
    }

// Copy exactly `len` raw bytes at `p` into a fresh GC String. Unlike
// `cstr_to_string` this does not stop at a NUL, so it preserves binary
// content (used when the length is known, e.g. a file's size).
def ptr_to_string(p: Ptr, len: Int) -> Result<String, IndexError> = {
    let b = bytes_new(len);
    let filled = cstr_copy_out(p, b, 0, len)?;
    Result::Ok(string_from_bytes(filled))
}

// ---- HTTP client — libcurl, called entirely through the C FFI ----
//
// No Rust glue: these declarations resolve real libcurl symbols via
// dlopen/dlsym at JIT-link time, and the request logic below is ordinary
// ai-lang driving the libcurl "easy" API. `curl_easy_setopt` /
// `curl_easy_getinfo` are variadic — the trailing value's C type depends
// on the option, so they are declared with `...` and the variadic ABI
// carries each call's value correctly.
//
// The response body is captured without a write callback by pointing
// CURLOPT_WRITEDATA at a FILE* from `open_memstream`; libcurl's default
// writer fwrites into it, and we read the resulting buffer afterward.
extern "C" lib "curl" {
    fn curl_easy_init() -> Ptr
    fn curl_easy_setopt(handle: Ptr, option: Int, ...) -> Int
    fn curl_easy_perform(handle: Ptr) -> Int
    fn curl_easy_getinfo(handle: Ptr, info: Int, ...) -> Int
    fn curl_easy_cleanup(handle: Ptr) -> Int
    fn curl_slist_append(list: Ptr, item: Ptr) -> Ptr
    fn curl_slist_free_all(list: Ptr) -> Int
    // The "multi" interface: drive many easy handles concurrently in one
    // thread (the basis for fanning out N Lambda invocations at once).
    fn curl_multi_init() -> Ptr
    fn curl_multi_add_handle(multi: Ptr, easy: Ptr) -> Int
    fn curl_multi_remove_handle(multi: Ptr, easy: Ptr) -> Int
    fn curl_multi_perform(multi: Ptr, running: Ptr) -> Int
    fn curl_multi_poll(multi: Ptr, fds: Ptr, nfds: Int, timeout_ms: Int, numfds: Ptr) -> Int
    fn curl_multi_cleanup(multi: Ptr) -> Int
}

// libcurl option / info constants (from curl.h).
def curlopt_url() -> Int = 10002
def curlopt_writedata() -> Int = 10001
def curlopt_followlocation() -> Int = 52
def curlopt_customrequest() -> Int = 10036
def curlopt_httpheader() -> Int = 10023
def curlopt_postfields() -> Int = 10015
def curlopt_postfieldsize() -> Int = 60
def curlopt_headerdata() -> Int = 10029
def curlinfo_response_code() -> Int = 2097154

// Build a curl_slist from newline-separated "Key: Value" header lines.
// Returns Ok(null Ptr) when `headers` is empty (no headers to set).
def http_build_headers(headers: String) -> Result<Ptr, IndexError> = {
    let b = bytes_from_string(headers);
    http_headers_scan(b, 0, 0, bytes_len(b), ptr_null())
}

def http_headers_scan(b: Bytes, i: Int, start: Int, len: Int, list: Ptr) -> Result<Ptr, IndexError> =
    if i == len {
        http_slist_add_line(b, start, len, list)
    } else {
        if bytes_get(b, i)? == 10 {
            let list2 = http_slist_add_line(b, start, i, list)?;
            http_headers_scan(b, i + 1, i + 1, len, list2)
        } else {
            http_headers_scan(b, i + 1, start, len, list)
        }
    }

// Append bytes[start..end) as one header line, skipping empty lines.
def http_slist_add_line(b: Bytes, start: Int, end: Int, list: Ptr) -> Result<Ptr, IndexError> =
    if end == start {
        Result::Ok(list)
    } else {
        let line = string_from_bytes(bytes_slice(b, start, end - start));
        let linec = cstr(line)?;
        let list2 = curl_slist_append(list, linec);
        let f = free(linec);
        Result::Ok(list2)
    }

def http_maybe_set_headers(h: Ptr, list: Ptr) -> Int =
    if ptr_is_null(list) == 1 {
        0
    } else {
        curl_easy_setopt(h, curlopt_httpheader(), list)
    }

def http_maybe_set_body(h: Ptr, body: String, bodyc: Ptr) -> Int =
    if string_is_empty(body) == 1 {
        0
    } else {
        let s1 = curl_easy_setopt(h, curlopt_postfields(), bodyc);
        curl_easy_setopt(h, curlopt_postfieldsize(), string_len(body))
    }

def http_free_slist(list: Ptr) -> Int =
    if ptr_is_null(list) == 1 {
        0
    } else {
        curl_slist_free_all(list)
    }

// A full HTTP response: the status code, the raw response headers (one
// "Key: Value" per line, CRLF-stripped), and the body, regardless of
// whether the status was a success. Needed for APIs (e.g. AWS, the Lambda
// Runtime API) that return structured error bodies on non-2xx, or carry
// data in response headers, and where the caller wants the status either
// way.
struct HttpResponse { status: Int, headers: String, body: String }

// Perform one HTTP(S) request and return the FULL response (status +
// headers + body), erroring only on a transport-level failure (curl
// rc != 0). The caller decides what a non-2xx status means. `method` is
// the verb; `headers` is newline-separated "Key: Value" lines (empty for
// none); `body` is the request body (empty for none).
def http_request_full(method: String, url: String, headers: String, body: String) -> Result<HttpResponse, String> = {
    // Each resource is released with `defer` right where it is acquired,
    // so every exit path (including any `?`) frees it deterministically,
    // in LIFO order, with no GC finalizer involved. `fflush`/`fclose` stay
    // inline because the response buffers are only valid once the streams
    // are closed, and they are read below.
    let bufp = malloc(8);
    defer free(bufp);
    let sizep = malloc(8);
    defer free(sizep);
    let fp = open_memstream(bufp, sizep);
    let hbufp = malloc(8);
    defer free(hbufp);
    let hsizep = malloc(8);
    defer free(hsizep);
    let hfp = open_memstream(hbufp, hsizep);
    let h = curl_easy_init();
    defer curl_easy_cleanup(h);
    let urlc = ix(cstr(url))?;
    defer free(urlc);
    let methodc = ix(cstr(method))?;
    defer free(methodc);
    let bodyc = ix(cstr(body))?;
    defer free(bodyc);
    let s_url = curl_easy_setopt(h, curlopt_url(), urlc);
    let s_wd = curl_easy_setopt(h, curlopt_writedata(), fp);
    let s_hd = curl_easy_setopt(h, curlopt_headerdata(), hfp);
    let s_fl = curl_easy_setopt(h, curlopt_followlocation(), 1);
    let s_cr = curl_easy_setopt(h, curlopt_customrequest(), methodc);
    let hdr_list = ix(http_build_headers(headers))?;
    defer http_free_slist(hdr_list);
    let s_hdr = http_maybe_set_headers(h, hdr_list);
    let s_body = http_maybe_set_body(h, body, bodyc);
    let rc = curl_easy_perform(h);
    let codep = malloc(8);
    defer free(codep);
    let g = curl_easy_getinfo(h, curlinfo_response_code(), codep);
    let code = ptr_read_i64(codep, 0);
    let flushed = fflush(fp);
    let closed = fclose(fp);
    let hflushed = fflush(hfp);
    let hclosed = fclose(hfp);
    let bufptr = ptr_read_ptr(bufp, 0);
    defer free(bufptr);
    let hbufptr = ptr_read_ptr(hbufp, 0);
    defer free(hbufptr);
    let resp_headers = ix(cstr_to_string(hbufptr))?;
    let resp_body = ix(cstr_to_string(bufptr))?;
    if rc == 0 {
        Result::Ok(HttpResponse {
            status: code,
            headers: resp_headers,
            body: resp_body
        })
    } else {
        Result::Err(string_concat("curl error ", int_to_string(rc)))
    }
}

// Extract a response header value by (case-insensitive) name, or Ok("")
// if absent. `headers` is the raw block from `HttpResponse.headers`.
def http_header(headers: String, name: String) -> Result<String, IndexError> = {
    let b = bytes_from_string(headers);
    let want = string_lower(name)?;
    http_header_lines(b, 0, bytes_len(b), want)
}

def http_header_lines(b: Bytes, i: Int, len: Int, want: String) -> Result<String, IndexError> =
    if i >= len {
        Result::Ok("")
    } else {
        let nl = http_find_byte(b, i, len, 10)?;
        let line_end = if nl > i { if bytes_get(b, nl - 1)? == 13 { nl - 1 } else { nl } } else { nl };
        let v = http_header_match(string_from_bytes(bytes_slice(b, i, line_end - i)), want)?;
        if string_is_empty(v) == 1 {
            if nl >= len { Result::Ok("") } else { http_header_lines(b, nl + 1, len, want) }
        } else {
            Result::Ok(v)
        }
    }

def http_header_match(line: String, want: String) -> Result<String, IndexError> = {
    let b = bytes_from_string(line);
    let n = bytes_len(b);
    let colon = http_find_byte(b, 0, n, 58)?;
    if colon >= n {
        Result::Ok("")
    } else {
        let key_raw = string_trim(string_from_bytes(bytes_slice(b, 0, colon)))?;
        let key = string_lower(key_raw)?;
        if string_eq(key, want) == 1 {
            string_trim(string_from_bytes(bytes_slice(b, colon + 1, n - colon - 1)))
        } else {
            Result::Ok("")
        }
    }
}

def http_find_byte(b: Bytes, i: Int, len: Int, target: Int) -> Result<Int, IndexError> =
    if i >= len { Result::Ok(len) } else { if bytes_get(b, i)? == target { Result::Ok(i) } else { http_find_byte(b, i + 1, len, target) } }

// ---- Concurrent requests (libcurl multi interface) ----
//
// Fire N HTTP requests AT ONCE from a single thread and collect all N
// responses (in input order). This is what lets one process fan out many
// Lambda invocations in parallel (Lambda then auto-scales to N concurrent
// execution environments). Each request gets its own easy handle + output
// buffer; the multi handle drives them all concurrently.

struct HttpReq { method: String, url: String, headers: String, body: String }

// Per-request resources, threaded from setup through collect to cleanup.
struct ReqState {
    easy: Ptr, bufp: Ptr, sizep: Ptr, fp: Ptr,
    urlc: Ptr, methodc: Ptr, bodyc: Ptr, slist: Ptr
}

def http_request_many(reqs: List<HttpReq>) -> Result<List<HttpResponse>, IndexError> = {
    let multi = curl_multi_init();
    let states = hrm_setup(reqs, multi)?;
    let runningp = malloc(8);
    let drive = hrm_drive(multi, runningp);
    let fr = free(runningp);
    let rev = hrm_collect(states, List::Nil)?;
    let cleanup = hrm_cleanup(multi, states);
    let fm = curl_multi_cleanup(multi);
    Result::Ok(list_reverse(rev))
}

// Build one easy handle for `req`, configure it, add it to `multi`.
def hrm_one(req: HttpReq, multi: Ptr) -> Result<ReqState, IndexError> = {
    let bufp = malloc(8);
    let sizep = malloc(8);
    let fp = open_memstream(bufp, sizep);
    let h = curl_easy_init();
    let urlc = cstr(req.url)?;
    let methodc = cstr(req.method)?;
    let bodyc = cstr(req.body)?;
    let su = curl_easy_setopt(h, curlopt_url(), urlc);
    let sw = curl_easy_setopt(h, curlopt_writedata(), fp);
    let sf = curl_easy_setopt(h, curlopt_followlocation(), 1);
    let sc = curl_easy_setopt(h, curlopt_customrequest(), methodc);
    let slist = http_build_headers(req.headers)?;
    let sh = http_maybe_set_headers(h, slist);
    let sb = http_maybe_set_body(h, req.body, bodyc);
    let add = curl_multi_add_handle(multi, h);
    Result::Ok(ReqState { easy: h, bufp: bufp, sizep: sizep, fp: fp,
               urlc: urlc, methodc: methodc, bodyc: bodyc, slist: slist })
}

def hrm_setup(reqs: List<HttpReq>, multi: Ptr) -> Result<List<ReqState>, IndexError> =
    match reqs {
        List::Nil => Result::Ok(List::Nil),
        List::Cons(cell) => {
            let head_state = hrm_one(cell.head, multi)?;
            let tail_states = hrm_setup(cell.tail, multi)?;
            Result::Ok(List::Cons(ListCell {
                head: head_state,
                tail: tail_states
            }))
        }
    }

// Drive all transfers to completion: perform, and while any are still
// running, poll for activity and perform again. Tail-recursive loop.
def hrm_drive(multi: Ptr, runningp: Ptr) -> Int = {
    let p = curl_multi_perform(multi, runningp);
    // `running_handles` is a C `int` (4 bytes); mask off the high 4 bytes
    // of the i64 read so uninitialized buffer garbage can't keep us looping.
    let running = bit_and(ptr_read_i64(runningp, 0), 4294967295);
    if running == 0 {
        0
    } else {
        let w = curl_multi_poll(multi, ptr_null(), 0, 1000, ptr_null());
        hrm_drive(multi, runningp)
    }
}

// Read each response (status + body) in order, prepending to `acc`.
def hrm_collect(states: List<ReqState>, acc: List<HttpResponse>) -> Result<List<HttpResponse>, IndexError> =
    match states {
        List::Nil => Result::Ok(acc),
        List::Cons(cell) => {
            let st = cell.head;
            let fl = fflush(st.fp);
            let fc = fclose(st.fp);
            let bufptr = ptr_read_ptr(st.bufp, 0);
            let codep = malloc(8);
            let g = curl_easy_getinfo(st.easy, curlinfo_response_code(), codep);
            let code = ptr_read_i64(codep, 0);
            let fcp = free(codep);
            let body_str = cstr_to_string(bufptr)?;
            let resp = HttpResponse { status: code, headers: "", body: body_str };
            hrm_collect(cell.tail, List::Cons(ListCell { head: resp, tail: acc }))
        }
    }

def hrm_cleanup(multi: Ptr, states: List<ReqState>) -> Int =
    match states {
        List::Nil => 0,
        List::Cons(cell) => {
            let st = cell.head;
            let rm = curl_multi_remove_handle(multi, st.easy);
            let cl = curl_easy_cleanup(st.easy);
            let f1 = free(st.urlc);
            let f2 = free(st.methodc);
            let f3 = free(st.bodyc);
            let fs = http_free_slist(st.slist);
            let bufptr = ptr_read_ptr(st.bufp, 0);
            let fb = free(bufptr);
            let f4 = free(st.bufp);
            let f5 = free(st.sizep);
            hrm_cleanup(multi, cell.tail)
        }
    }

// Perform one HTTP(S) request and return the body on success, collapsing
// a non-2xx status to `Err("HTTP <code>")`. The simple ergonomic wrapper
// over `http_request_full` for callers that just want the body.
def http_do(method: String, url: String, headers: String, body: String) -> Result<String, String> =
    match http_request_full(method, url, headers, body) {
        Result::Err(e) => Result::Err(e),
        Result::Ok(resp) =>
            if resp.status < 300 {
                Result::Ok(resp.body)
            } else {
                Result::Err(string_concat("HTTP ", int_to_string(resp.status)))
            }
    }

// Perform an HTTP(S) GET. Pair with `?` to propagate failures.
def http_get(url: String) -> Result<String, String> =
    http_do("GET", url, "", "")

// GET with caller-supplied headers ("Key: Value" lines, newline-separated).
def http_get_with_headers(url: String, headers: String) -> Result<String, String> =
    http_do("GET", url, headers, "")

// Perform an HTTP(S) POST with a request body and explicit headers
// (e.g. "Content-Type: application/json").
def http_post(url: String, headers: String, body: String) -> Result<String, String> =
    http_do("POST", url, headers, body)

// ---- JSON — a real recursive-descent parser, written in ai-lang ----
//
// No external library and no host shim: this parses JSON text into a
// `Json` value tree, then navigates it by dot-path. Numbers are stored
// as integers (any fractional/exponent part is consumed but truncated);
// strings decode the standard escapes including `\uXXXX` (BMP) into
// UTF-8. The value tree uses non-generic recursive types so enum-payload
// matching unboxes cleanly.

enum Json {
    JNull,
    JBool(Int),
    JNumber(Int),
    JString(String),
    JArray(JArr),
    JObject(JObj)
}

enum JArr { JANil, JACons(JArrCell) }
struct JArrCell { head: Json, tail: JArr }

enum JObj { JONil, JOCons(JObjCell) }
struct JObjCell { key: String, val: Json, rest: JObj }

// Parser result carriers (value/array/object/string + cursor + ok flag).
// The `ok` flag is the parser's own "invalid JSON" signal; the parser
// functions additionally return Result<P*, IndexError> so out-of-bounds
// indexing surfaces as a value, never an abort.
struct PJson { val: Json, pos: Int, ok: Int }
struct PArr { arr: JArr, pos: Int, ok: Int }
struct PObj { obj: JObj, pos: Int, ok: Int }
struct PStr { str: String, pos: Int, ok: Int }

def pjson_ok(val: Json, pos: Int) -> PJson = PJson { val: val, pos: pos, ok: 1 }
def pjson_fail() -> PJson = PJson { val: Json::JNull, pos: 0, ok: 0 }
def parr_ok(arr: JArr, pos: Int) -> PArr = PArr { arr: arr, pos: pos, ok: 1 }
def parr_fail() -> PArr = PArr { arr: JArr::JANil, pos: 0, ok: 0 }
def pobj_ok(obj: JObj, pos: Int) -> PObj = PObj { obj: obj, pos: pos, ok: 1 }
def pobj_fail() -> PObj = PObj { obj: JObj::JONil, pos: 0, ok: 0 }
def pstr_ok(str: String, pos: Int) -> PStr = PStr { str: str, pos: pos, ok: 1 }
def pstr_fail() -> PStr = PStr { str: "", pos: 0, ok: 0 }

// Parse a complete JSON document.
def parse_json(text: String) -> Result<Json, String> = {
    let b = bytes_from_string(text);
    let len = bytes_len(b);
    let i = ix(json_skip_ws(b, 0, len))?;
    let r = ix(json_parse_value(b, i, len))?;
    if r.ok == 1 {
        let j = ix(json_skip_ws(b, r.pos, len))?;
        if j == len { Result::Ok(r.val) } else { Result::Err("invalid JSON") }
    } else {
        Result::Err("invalid JSON")
    }
}

def json_skip_ws(b: Bytes, i: Int, len: Int) -> Result<Int, IndexError> =
    if i >= len {
        Result::Ok(i)
    } else {
        let c = bytes_get(b, i)?;
        if c == 32 {
            json_skip_ws(b, i + 1, len)
        } else {
            if c == 9 {
                json_skip_ws(b, i + 1, len)
            } else {
                if c == 10 {
                    json_skip_ws(b, i + 1, len)
                } else {
                    if c == 13 { json_skip_ws(b, i + 1, len) } else { Result::Ok(i) }
                }
            }
        }
    }

def json_parse_value(b: Bytes, i: Int, len: Int) -> Result<PJson, IndexError> =
    if i >= len {
        Result::Ok(pjson_fail())
    } else {
        let c = bytes_get(b, i)?;
        if c == 34 {
            json_parse_string_val(b, i, len)
        } else {
            if c == 123 {
                json_parse_object(b, i, len)
            } else {
                if c == 91 {
                    json_parse_array(b, i, len)
                } else {
                    if c == 116 {
                        json_parse_lit(b, i, len, "true", Json::JBool(1))
                    } else {
                        if c == 102 {
                            json_parse_lit(b, i, len, "false", Json::JBool(0))
                        } else {
                            if c == 110 {
                                json_parse_lit(b, i, len, "null", Json::JNull)
                            } else {
                                if json_is_num_start(c) == 1 {
                                    json_parse_number(b, i, len)
                                } else {
                                    Result::Ok(pjson_fail())
                                }
                            }
                        }
                    }
                }
            }
        }
    }

def json_is_num_start(c: Int) -> Int =
    if c == 45 { 1 } else { if c >= 48 { if c <= 57 { 1 } else { 0 } } else { 0 } }

def json_parse_lit(b: Bytes, i: Int, len: Int, lit: String, val: Json) -> Result<PJson, IndexError> = {
    let lb = bytes_from_string(lit);
    let ll = bytes_len(lb);
    let m = json_match_lit(b, i, len, lb, 0, ll)?;
    if m == 1 {
        Result::Ok(pjson_ok(val, i + ll))
    } else {
        Result::Ok(pjson_fail())
    }
}

def json_match_lit(b: Bytes, i: Int, len: Int, lb: Bytes, k: Int, ll: Int) -> Result<Int, IndexError> =
    if k == ll {
        Result::Ok(1)
    } else {
        if i + k >= len {
            Result::Ok(0)
        } else {
            let c1 = bytes_get(b, i + k)?;
            let c2 = bytes_get(lb, k)?;
            if c1 == c2 {
                json_match_lit(b, i, len, lb, k + 1, ll)
            } else {
                Result::Ok(0)
            }
        }
    }

// Integer-valued numbers: parse the leading integer, then consume (and
// ignore) any fractional/exponent tail so the document stays valid.
def json_parse_number(b: Bytes, i: Int, len: Int) -> Result<PJson, IndexError> = {
    let c0 = bytes_get(b, i)?;
    let neg = if c0 == 45 { 1 } else { 0 };
    let start = i + neg;
    let endpos = json_scan_digits(b, start, len)?;
    if endpos == start {
        Result::Ok(pjson_fail())
    } else {
        let mag = json_digits_to_int(b, start, endpos, 0)?;
        let v = if neg == 1 { 0 - mag } else { mag };
        let after = json_skip_number_tail(b, endpos, len)?;
        Result::Ok(pjson_ok(Json::JNumber(v), after))
    }
}

def json_scan_digits(b: Bytes, i: Int, len: Int) -> Result<Int, IndexError> =
    if i >= len {
        Result::Ok(i)
    } else {
        let c = bytes_get(b, i)?;
        if c >= 48 {
            if c <= 57 { json_scan_digits(b, i + 1, len) } else { Result::Ok(i) }
        } else {
            Result::Ok(i)
        }
    }

def json_digits_to_int(b: Bytes, i: Int, end: Int, acc: Int) -> Result<Int, IndexError> =
    if i >= end {
        Result::Ok(acc)
    } else {
        let d = bytes_get(b, i)?;
        json_digits_to_int(b, i + 1, end, acc * 10 + (d - 48))
    }

def json_skip_number_tail(b: Bytes, i: Int, len: Int) -> Result<Int, IndexError> =
    if i >= len {
        Result::Ok(i)
    } else {
        let c = bytes_get(b, i)?;
        if json_is_num_tail(c) == 1 {
            json_skip_number_tail(b, i + 1, len)
        } else {
            Result::Ok(i)
        }
    }

def json_is_num_tail(c: Int) -> Int =
    if c == 46 {
        1
    } else {
        if c == 101 {
            1
        } else {
            if c == 69 {
                1
            } else {
                if c == 43 {
                    1
                } else {
                    if c == 45 {
                        1
                    } else {
                        if c >= 48 { if c <= 57 { 1 } else { 0 } } else { 0 }
                    }
                }
            }
        }
    }

def json_parse_array(b: Bytes, i: Int, len: Int) -> Result<PJson, IndexError> = {
    let j = json_skip_ws(b, i + 1, len)?;
    if j >= len {
        Result::Ok(pjson_fail())
    } else {
        let c = bytes_get(b, j)?;
        if c == 93 {
            Result::Ok(pjson_ok(Json::JArray(JArr::JANil), j + 1))
        } else {
            let r = json_parse_array_elems(b, j, len)?;
            if r.ok == 1 { Result::Ok(pjson_ok(Json::JArray(r.arr), r.pos)) } else { Result::Ok(pjson_fail()) }
        }
    }
}

def json_parse_array_elems(b: Bytes, i: Int, len: Int) -> Result<PArr, IndexError> = {
    let r = json_parse_value(b, i, len)?;
    if r.ok == 0 {
        Result::Ok(parr_fail())
    } else {
        let j = json_skip_ws(b, r.pos, len)?;
        if j >= len {
            Result::Ok(parr_fail())
        } else {
            let c = bytes_get(b, j)?;
            if c == 44 {
                let k = json_skip_ws(b, j + 1, len)?;
                let rest = json_parse_array_elems(b, k, len)?;
                if rest.ok == 0 {
                    Result::Ok(parr_fail())
                } else {
                    Result::Ok(parr_ok(JArr::JACons(JArrCell { head: r.val, tail: rest.arr }), rest.pos))
                }
            } else {
                if c == 93 {
                    Result::Ok(parr_ok(JArr::JACons(JArrCell { head: r.val, tail: JArr::JANil }), j + 1))
                } else {
                    Result::Ok(parr_fail())
                }
            }
        }
    }
}

def json_parse_object(b: Bytes, i: Int, len: Int) -> Result<PJson, IndexError> = {
    let j = json_skip_ws(b, i + 1, len)?;
    if j >= len {
        Result::Ok(pjson_fail())
    } else {
        let c = bytes_get(b, j)?;
        if c == 125 {
            Result::Ok(pjson_ok(Json::JObject(JObj::JONil), j + 1))
        } else {
            let r = json_parse_object_members(b, j, len)?;
            if r.ok == 1 { Result::Ok(pjson_ok(Json::JObject(r.obj), r.pos)) } else { Result::Ok(pjson_fail()) }
        }
    }
}

def json_parse_object_members(b: Bytes, i: Int, len: Int) -> Result<PObj, IndexError> = {
    let ks = json_parse_string_raw(b, i, len)?;
    if ks.ok == 0 {
        Result::Ok(pobj_fail())
    } else {
        let j = json_skip_ws(b, ks.pos, len)?;
        if j >= len {
            Result::Ok(pobj_fail())
        } else {
            let c = bytes_get(b, j)?;
            if c == 58 {
                let k = json_skip_ws(b, j + 1, len)?;
                let v = json_parse_value(b, k, len)?;
                if v.ok == 0 {
                    Result::Ok(pobj_fail())
                } else {
                    json_object_after_value(b, len, ks.str, v)
                }
            } else {
                Result::Ok(pobj_fail())
            }
        }
    }
}

def json_object_after_value(b: Bytes, len: Int, key: String, v: PJson) -> Result<PObj, IndexError> = {
    let m = json_skip_ws(b, v.pos, len)?;
    if m >= len {
        Result::Ok(pobj_fail())
    } else {
        let c = bytes_get(b, m)?;
        if c == 44 {
            let n = json_skip_ws(b, m + 1, len)?;
            let rest = json_parse_object_members(b, n, len)?;
            if rest.ok == 0 {
                Result::Ok(pobj_fail())
            } else {
                Result::Ok(pobj_ok(JObj::JOCons(JObjCell { key: key, val: v.val, rest: rest.obj }), rest.pos))
            }
        } else {
            if c == 125 {
                Result::Ok(pobj_ok(JObj::JOCons(JObjCell { key: key, val: v.val, rest: JObj::JONil }), m + 1))
            } else {
                Result::Ok(pobj_fail())
            }
        }
    }
}

def json_parse_string_val(b: Bytes, i: Int, len: Int) -> Result<PJson, IndexError> = {
    let r = json_parse_string_raw(b, i, len)?;
    if r.ok == 1 { Result::Ok(pjson_ok(Json::JString(r.str), r.pos)) } else { Result::Ok(pjson_fail()) }
}

// `i` points at the opening quote. Scan to the closing quote (respecting
// backslash escapes), then decode the content into a fresh String.
def json_parse_string_raw(b: Bytes, i: Int, len: Int) -> Result<PStr, IndexError> =
    if i >= len {
        Result::Ok(pstr_fail())
    } else {
        let c = bytes_get(b, i)?;
        if c == 34 {
            let start = i + 1;
            let endq = json_str_end(b, start, len)?;
            if endq < 0 {
                Result::Ok(pstr_fail())
            } else {
                let cap = endq - start;
                let buf = bytes_new(cap);
                let w = json_decode_str(b, start, endq, buf, 0)?;
                Result::Ok(pstr_ok(string_from_bytes(bytes_slice(buf, 0, w)), endq + 1))
            }
        } else {
            Result::Ok(pstr_fail())
        }
    }

def json_str_end(b: Bytes, i: Int, len: Int) -> Result<Int, IndexError> =
    if i >= len {
        Result::Ok(0 - 1)
    } else {
        let c = bytes_get(b, i)?;
        if c == 34 {
            Result::Ok(i)
        } else {
            if c == 92 {
                if i + 1 >= len { Result::Ok(0 - 1) } else { json_str_end(b, i + 2, len) }
            } else {
                json_str_end(b, i + 1, len)
            }
        }
    }

def json_decode_str(b: Bytes, i: Int, end: Int, buf: Bytes, w: Int) -> Result<Int, IndexError> =
    if i >= end {
        Result::Ok(w)
    } else {
        let c = bytes_get(b, i)?;
        if c == 92 {
            json_decode_escape(b, i + 1, end, buf, w)
        } else {
            let x = bytes_set(buf, w, c)?;
            json_decode_str(b, i + 1, end, buf, w + 1)
        }
    }

def json_decode_escape(b: Bytes, i: Int, end: Int, buf: Bytes, w: Int) -> Result<Int, IndexError> = {
    let e = bytes_get(b, i)?;
    if e == 117 {
        let cp = json_hex4(b, i + 1)?;
        let w2 = json_utf8_encode(buf, w, cp)?;
        json_decode_str(b, i + 5, end, buf, w2)
    } else {
        let x = bytes_set(buf, w, json_escape_byte(e))?;
        json_decode_str(b, i + 1, end, buf, w + 1)
    }
}

def json_escape_byte(e: Int) -> Int =
    if e == 110 {
        10
    } else {
        if e == 116 {
            9
        } else {
            if e == 114 {
                13
            } else {
                if e == 98 {
                    8
                } else {
                    if e == 102 { 12 } else { e }
                }
            }
        }
    }

def json_hex4(b: Bytes, i: Int) -> Result<Int, IndexError> = {
    let h0 = bytes_get(b, i)?;
    let h1 = bytes_get(b, i + 1)?;
    let h2 = bytes_get(b, i + 2)?;
    let h3 = bytes_get(b, i + 3)?;
    Result::Ok(json_hex(h0) * 4096
        + json_hex(h1) * 256
        + json_hex(h2) * 16
        + json_hex(h3))
}

def json_hex(c: Int) -> Int =
    if c <= 57 { c - 48 } else { if c <= 70 { c - 55 } else { c - 87 } }

def json_utf8_encode(buf: Bytes, w: Int, cp: Int) -> Result<Int, IndexError> =
    if cp < 128 {
        let a = bytes_set(buf, w, cp)?;
        Result::Ok(w + 1)
    } else {
        if cp < 2048 {
            let a = bytes_set(buf, w, 192 + cp / 64)?;
            let b2 = bytes_set(buf, w + 1, 128 + cp - (cp / 64) * 64)?;
            Result::Ok(w + 2)
        } else {
            let a = bytes_set(buf, w, 224 + cp / 4096)?;
            let mid = cp / 64;
            let b2 = bytes_set(buf, w + 1, 128 + mid - (mid / 64) * 64)?;
            let c2 = bytes_set(buf, w + 2, 128 + cp - (cp / 64) * 64)?;
            Result::Ok(w + 3)
        }
    }

// ---- JSON navigation ----

def json_object_find(obj: JObj, key: String) -> Option<Json> =
    match obj {
        JObj::JONil => Option::None,
        JObj::JOCons(cell) =>
            if string_eq(cell.key, key) {
                Option::Some(cell.val)
            } else {
                json_object_find(cell.rest, key)
            }
    }

def json_array_at(arr: JArr, idx: Int) -> Option<Json> =
    match arr {
        JArr::JANil => Option::None,
        JArr::JACons(cell) =>
            if idx == 0 { Option::Some(cell.head) } else { json_array_at(cell.tail, idx - 1) }
    }

def json_arr_len(arr: JArr, acc: Int) -> Int =
    match arr {
        JArr::JANil => acc,
        JArr::JACons(cell) => json_arr_len(cell.tail, acc + 1)
    }

def json_obj_len(obj: JObj, acc: Int) -> Int =
    match obj {
        JObj::JONil => acc,
        JObj::JOCons(cell) => json_obj_len(cell.rest, acc + 1)
    }

// Length of an array or object; 0 for any scalar.
def json_len(j: Json) -> Int =
    match j {
        Json::JArray(arr) => json_arr_len(arr, 0),
        Json::JObject(obj) => json_obj_len(obj, 0),
        _ => 0
    }

// Pull a String / Int out of a leaf value.
def json_string(j: Json) -> Option<String> =
    match j {
        Json::JString(s) => Option::Some(s),
        _ => Option::None
    }

def json_int(j: Json) -> Option<Int> =
    match j {
        Json::JNumber(n) => Option::Some(n),
        _ => Option::None
    }

// Navigate one dot-path segment: an object key, or an array index.
def json_step(j: Json, seg: String) -> Option<Json> =
    match j {
        Json::JObject(obj) => json_object_find(obj, seg),
        Json::JArray(arr) =>
            if string_is_int(seg) == 1 {
                json_array_at(arr, string_to_int(seg))
            } else {
                Option::None
            },
        _ => Option::None
    }

// Navigate a dot-separated path ("meta.version", "tags.0"), returning
// `Ok(Some(value))` when found, `Ok(None)` when the path is missing.
// The empty path yields the value unchanged.
def json_get(j: Json, path: String) -> Result<Option<Json>, IndexError> = {
    let pb = bytes_from_string(path);
    json_get_walk(j, pb, 0, bytes_len(pb))
}

def json_get_walk(j: Json, pb: Bytes, i: Int, len: Int) -> Result<Option<Json>, IndexError> =
    if i >= len {
        Result::Ok(Option::Some(j))
    } else {
        let dot = json_find_dot(pb, i, len)?;
        let seg = string_from_bytes(bytes_slice(pb, i, dot - i));
        match json_step(j, seg) {
            Option::None => Result::Ok(Option::None),
            Option::Some(next) =>
                if dot >= len { Result::Ok(Option::Some(next)) } else { json_get_walk(next, pb, dot + 1, len) }
        }
    }

def json_find_dot(pb: Bytes, i: Int, len: Int) -> Result<Int, IndexError> =
    if i >= len {
        Result::Ok(len)
    } else {
        let c = bytes_get(pb, i)?;
        if c == 46 { Result::Ok(i) } else { json_find_dot(pb, i + 1, len) }
    }

// Parse `text` and read the string value at `path`, returning
// `Ok(value)` or `Err(message)` on a parse error or a missing path.
def json_field(text: String, path: String) -> Result<String, String> =
    match parse_json(text) {
        Result::Err(e) => Result::Err(e),
        Result::Ok(j) =>
            match ix(json_get(j, path))? {
                Option::None => Result::Err(string_concat("no such JSON path: ", path)),
                Option::Some(v) =>
                    match json_string(v) {
                        Option::Some(s) => Result::Ok(s),
                        Option::None => Result::Err("value is not a string")
                    }
            }
    }

// ---- OS surface: environment, clock, filesystem (libc via the FFI) ----
//
// All of these are ordinary ai-lang calling libc directly through the C
// FFI (getenv / clock_gettime / fopen / ...). No Rust glue.

// Value of an environment variable: `Ok("")` if unset.
def env_get(name: String) -> Result<String, IndexError> = {
    let namec = cstr(name)?;
    defer free(namec);
    env_get_value(getenv(namec))
}

def env_get_value(v: Ptr) -> Result<String, IndexError> =
    if ptr_is_null(v) == 1 { Result::Ok("") } else { cstr_to_string(v) }

// Ok(1) if the environment variable is set, else Ok(0).
def env_has(name: String) -> Result<Int, IndexError> = {
    let namec = cstr(name)?;
    defer free(namec);
    if ptr_is_null(getenv(namec)) == 1 { Result::Ok(0) } else { Result::Ok(1) }
}

// Seconds since the Unix epoch (libc `time(NULL)`).
def now_unix() -> Int = time(ptr_null())

// Milliseconds since the Unix epoch via `clock_gettime(CLOCK_REALTIME)`.
// struct timespec is two 8-byte fields (tv_sec, tv_nsec) on 64-bit, so
// both read cleanly with ptr_read_i64. CLOCK_REALTIME = 0.
def now_unix_millis() -> Int = {
    let ts = malloc(16);
    defer free(ts);
    let r = clock_gettime(0, ts);
    let sec = ptr_read_i64(ts, 0);
    let nsec = ptr_read_i64(ts, 8);
    sec * 1000 + nsec / 1000000
}

// Read a file's contents, returning `Ok(contents)` or `Err(message)`.
def read_file(path: String) -> Result<String, String> = {
    let pathc = ix(cstr(path))?;
    defer free(pathc);
    let modec = ix(cstr("rb"))?;
    defer free(modec);
    let fp = fopen(pathc, modec);
    if ptr_is_null(fp) == 1 {
        Result::Err(string_concat("cannot open file: ", path))
    } else {
        ix(read_file_open(fp))
    }
}

// Read an open FILE* to end: seek to end for the size, rewind, read it
// all into a fresh String. SEEK_END = 2, SEEK_SET = 0.
def read_file_open(fp: Ptr) -> Result<String, IndexError> = {
    let s1 = fseek(fp, 0, 2);
    let size = ftell(fp);
    let s2 = fseek(fp, 0, 0);
    let buf = malloc(size + 1);
    defer free(buf);
    defer fclose(fp);
    let nread = fread(buf, 1, size, fp);
    ptr_to_string(buf, nread)
}

// Write `contents` to `path` (truncating). Returns Ok(0) on success,
// Ok(-1) on failure (could not open, or short write).
def fs_write(path: String, contents: String) -> Result<Int, IndexError> = {
    let pathc = cstr(path)?;
    defer free(pathc);
    let modec = cstr("wb")?;
    defer free(modec);
    let fp = fopen(pathc, modec);
    if ptr_is_null(fp) == 1 { Result::Ok(0 - 1) } else { fs_write_open(fp, contents) }
}

def fs_write_open(fp: Ptr, contents: String) -> Result<Int, IndexError> = {
    let b = bytes_from_string(contents);
    let n = bytes_len(b);
    let datac = cstr(contents)?;
    defer free(datac);
    defer fclose(fp);
    let w = fwrite(datac, 1, n, fp);
    if w == n { Result::Ok(0) } else { Result::Ok(0 - 1) }
}

// Ok(1) if the path exists (libc `access(path, F_OK)`; F_OK = 0), else Ok(0).
def fs_exists(path: String) -> Result<Int, IndexError> = {
    let pathc = cstr(path)?;
    defer free(pathc);
    if access(pathc, 0) == 0 { Result::Ok(1) } else { Result::Ok(0) }
}

// ---- Crypto: SHA-256 and HMAC-SHA256 (OpenSSL libcrypto via the FFI) --
//
// No Rust glue: these call OpenSSL's `SHA256` / `HMAC` one-shot functions
// directly through the C FFI (resolved from libcrypto via dlopen/dlsym),
// and the hex rendering is pure ai-lang. `*_hex` return lowercase hex;
// `*_raw` return the raw 32 binary bytes (a byte-buffer String) so
// digests can be chained as the next HMAC key. These are the primitives
// request-signing (e.g. AWS SigV4) needs.
extern "C" lib "crypto" {
    fn SHA256(data: Ptr, len: Int, md: Ptr) -> Ptr
    fn HMAC(evp_md: Ptr, key: Ptr, key_len: Int, data: Ptr, data_len: Int, md: Ptr, md_len: Ptr) -> Ptr
    fn EVP_sha256() -> Ptr
}

// Render a raw byte-buffer String as lowercase hex.
def hex_encode(data: String) -> Result<String, IndexError> = {
    let b = bytes_from_string(data);
    let n = bytes_len(b);
    let out = bytes_new(n * 2);
    let filled = hex_fill(b, out, 0, n)?;
    Result::Ok(string_from_bytes(filled))
}

def hex_fill(b: Bytes, out: Bytes, i: Int, n: Int) -> Result<Bytes, IndexError> =
    if i == n {
        Result::Ok(out)
    } else {
        let byte = bytes_get(b, i)?;
        let hi = bytes_set(out, i * 2, hex_digit(byte / 16))?;
        let lo = bytes_set(out, i * 2 + 1, hex_digit(byte - (byte / 16) * 16))?;
        hex_fill(b, out, i + 1, n)
    }

def hex_digit(v: Int) -> Int =
    if v < 10 { 48 + v } else { 87 + v }

// ---- Base64 (standard alphabet, with padding) ----

def base64_char(v: Int) -> Int =
    if v < 26 {
        65 + v
    } else {
        if v < 52 {
            97 + (v - 26)
        } else {
            if v < 62 {
                48 + (v - 52)
            } else {
                if v == 62 { 43 } else { 47 }
            }
        }
    }

def base64_encode(data: String) -> Result<String, IndexError> = {
    let b = bytes_from_string(data);
    let n = bytes_len(b);
    let groups = (n + 2) / 3;
    let out = bytes_new(groups * 4);
    let filled = base64_fill(b, out, 0, n, 0)?;
    Result::Ok(string_from_bytes(filled))
}

def base64_fill(b: Bytes, out: Bytes, i: Int, n: Int, o: Int) -> Result<Bytes, IndexError> =
    if i >= n {
        Result::Ok(out)
    } else {
        let rem = n - i;
        let b0 = bytes_get(b, i)?;
        let b1 = if rem > 1 { bytes_get(b, i + 1)? } else { 0 };
        let b2 = if rem > 2 { bytes_get(b, i + 2)? } else { 0 };
        let triple = bit_or(bit_or(bit_shl(b0, 16), bit_shl(b1, 8)), b2);
        let c0 = base64_char(bit_and(bit_shr(triple, 18), 63));
        let c1 = base64_char(bit_and(bit_shr(triple, 12), 63));
        let c2 = if rem > 1 { base64_char(bit_and(bit_shr(triple, 6), 63)) } else { 61 };
        let c3 = if rem > 2 { base64_char(bit_and(triple, 63)) } else { 61 };
        let w0 = bytes_set(out, o, c0)?;
        let w1 = bytes_set(out, o + 1, c1)?;
        let w2 = bytes_set(out, o + 2, c2)?;
        let w3 = bytes_set(out, o + 3, c3)?;
        base64_fill(b, out, i + 3, n, o + 4)
    }

// ---- CRC-32 (IEEE, reflected, polynomial 0xEDB88320) ----
// Bit-by-bit (no table). Returns the 32-bit CRC as an Int (Ok-wrapped).

def crc32(data: String) -> Result<Int, IndexError> = {
    let b = bytes_from_string(data);
    let crc = crc32_bytes(b, 0, bytes_len(b), 4294967295)?;
    Result::Ok(bit_xor(crc, 4294967295))
}

def crc32_bytes(b: Bytes, i: Int, n: Int, crc: Int) -> Result<Int, IndexError> =
    if i >= n {
        Result::Ok(crc)
    } else {
        let byte = bytes_get(b, i)?;
        crc32_bytes(b, i + 1, n, crc32_bits(bit_xor(crc, byte), 0))
    }

def crc32_bits(crc: Int, k: Int) -> Int =
    if k >= 8 {
        crc
    } else {
        let next =
            if bit_and(crc, 1) == 1 {
                bit_xor(bit_shr(crc, 1), 3988292384)
            } else {
                bit_shr(crc, 1)
            };
        crc32_bits(next, k + 1)
    }

// ---- ZIP archive (STORED / no compression) ----
//
// Enough of the ZIP format to build a single-file archive (e.g. a Lambda
// custom-runtime `bootstrap`). Little-endian throughout. The single file
// is marked executable (Unix mode 0755) via the external-attributes field.

// Write little-endian u16 / u32 into `b` at `off`. Returns Ok(0).
def put_le16(b: Bytes, off: Int, v: Int) -> Result<Int, IndexError> = {
    let w0 = bytes_set(b, off, bit_and(v, 255))?;
    let w1 = bytes_set(b, off + 1, bit_and(bit_shr(v, 8), 255))?;
    Result::Ok(0)
}
def put_le32(b: Bytes, off: Int, v: Int) -> Result<Int, IndexError> = {
    let w0 = bytes_set(b, off, bit_and(v, 255))?;
    let w1 = bytes_set(b, off + 1, bit_and(bit_shr(v, 8), 255))?;
    let w2 = bytes_set(b, off + 2, bit_and(bit_shr(v, 16), 255))?;
    let w3 = bytes_set(b, off + 3, bit_and(bit_shr(v, 24), 255))?;
    Result::Ok(0)
}

def zip_local_header(crc: Int, size: Int, fnlen: Int) -> Result<Bytes, IndexError> = {
    let h = bytes_new(30);
    let a = put_le32(h, 0, 67324752)?;
    let b = put_le16(h, 4, 20)?;
    let c = put_le16(h, 6, 0)?;
    let d = put_le16(h, 8, 0)?;
    let e = put_le16(h, 10, 0)?;
    let f = put_le16(h, 12, 33)?;
    let g = put_le32(h, 14, crc)?;
    let i = put_le32(h, 18, size)?;
    let j = put_le32(h, 22, size)?;
    let k = put_le16(h, 26, fnlen)?;
    let l = put_le16(h, 28, 0)?;
    Result::Ok(h)
}

def zip_central_header(crc: Int, size: Int, fnlen: Int, offset: Int) -> Result<Bytes, IndexError> = {
    let h = bytes_new(46);
    let a = put_le32(h, 0, 33639248)?;
    let b = put_le16(h, 4, 20)?;
    let c = put_le16(h, 6, 20)?;
    let d = put_le16(h, 8, 0)?;
    let e = put_le16(h, 10, 0)?;
    let f = put_le16(h, 12, 0)?;
    let g = put_le16(h, 14, 33)?;
    let i = put_le32(h, 16, crc)?;
    let j = put_le32(h, 20, size)?;
    let k = put_le32(h, 24, size)?;
    let l = put_le16(h, 28, fnlen)?;
    let m = put_le16(h, 30, 0)?;
    let n = put_le16(h, 32, 0)?;
    let o = put_le16(h, 34, 0)?;
    let p = put_le16(h, 36, 0)?;
    let q = put_le32(h, 38, 2179792896)?;
    let r = put_le32(h, 42, offset)?;
    Result::Ok(h)
}

def zip_eocd(count: Int, cd_size: Int, cd_offset: Int) -> Result<Bytes, IndexError> = {
    let h = bytes_new(22);
    let a = put_le32(h, 0, 101010256)?;
    let b = put_le16(h, 4, 0)?;
    let c = put_le16(h, 6, 0)?;
    let d = put_le16(h, 8, count)?;
    let e = put_le16(h, 10, count)?;
    let f = put_le32(h, 12, cd_size)?;
    let g = put_le32(h, 16, cd_offset)?;
    let i = put_le16(h, 20, 0)?;
    Result::Ok(h)
}

// Build a one-file ZIP archive (filename + content) and return the raw
// archive bytes as a byte-string.
def zip_one(filename: String, content: String) -> Result<String, IndexError> = {
    let fnb = bytes_from_string(filename);
    let datab = bytes_from_string(content);
    let crc = crc32(content)?;
    let size = bytes_len(datab);
    let fnlen = bytes_len(fnb);
    let lhdr = zip_local_header(crc, size, fnlen)?;
    let loc = bytes_concat(lhdr, bytes_concat(fnb, datab));
    let loc_size = bytes_len(loc);
    let chdr = zip_central_header(crc, size, fnlen, 0)?;
    let central = bytes_concat(chdr, fnb);
    let eocd = zip_eocd(1, bytes_len(central), loc_size)?;
    Result::Ok(string_from_bytes(bytes_concat(loc, bytes_concat(central, eocd))))
}

// SHA-256 of `data`, returned as the raw 32 binary bytes.
def sha256_raw(data: String) -> Result<String, IndexError> = {
    let b = bytes_from_string(data);
    let n = bytes_len(b);
    let datac = cstr(data)?;
    defer free(datac);
    let md = malloc(32);
    defer free(md);
    let r = SHA256(datac, n, md);
    ptr_to_string(md, 32)
}

def sha256_hex(data: String) -> Result<String, IndexError> = {
    let raw = sha256_raw(data)?;
    hex_encode(raw)
}

// HMAC-SHA256 of `data` under `key`, as the raw 32 binary bytes.
def hmac_sha256_raw(key: String, data: String) -> Result<String, IndexError> = {
    let kb = bytes_from_string(key);
    let kn = bytes_len(kb);
    let db = bytes_from_string(data);
    let dn = bytes_len(db);
    let keyc = cstr(key)?;
    defer free(keyc);
    let datac = cstr(data)?;
    defer free(datac);
    let md = malloc(32);
    defer free(md);
    let mdlen = malloc(4);
    defer free(mdlen);
    let evp = EVP_sha256();
    let r = HMAC(evp, keyc, kn, datac, dn, md, mdlen);
    ptr_to_string(md, 32)
}

def hmac_sha256_hex(key: String, data: String) -> Result<String, IndexError> = {
    let raw = hmac_sha256_raw(key, data)?;
    hex_encode(raw)
}

// Derive the AWS SigV4 signing key (raw 32 bytes) from a secret access
// key, an 8-digit date stamp (YYYYMMDD), a region, and a service. This is
// the `kSigning = HMAC(HMAC(HMAC(HMAC("AWS4"+secret, date), region),
// service), "aws4_request")` chain — the crux of authenticating to AWS.
def aws_sigv4_signing_key_raw(
    secret: String, date: String, region: String, service: String) -> Result<String, IndexError> = {
    let k_date = hmac_sha256_raw(string_concat("AWS4", secret), date)?;
    let k_region = hmac_sha256_raw(k_date, region)?;
    let k_service = hmac_sha256_raw(k_region, service)?;
    hmac_sha256_raw(k_service, "aws4_request")
}

// The signing key as lowercase hex (the published-test-vector form).
def aws_sigv4_signing_key(
    secret: String, date: String, region: String, service: String) -> Result<String, IndexError> = {
    let raw = aws_sigv4_signing_key_raw(secret, date, region, service)?;
    hex_encode(raw)
}

// ---- String helpers (for request canonicalization) ----

def str3(a: String, b: String, c: String) -> String =
    string_concat(a, string_concat(b, c))
def str4(a: String, b: String, c: String, d: String) -> String =
    string_concat(a, str3(b, c, d))
def str5(a: String, b: String, c: String, d: String, e: String) -> String =
    string_concat(a, str4(b, c, d, e))
def str6(a: String, b: String, c: String, d: String, e: String, f: String) -> String =
    string_concat(a, str5(b, c, d, e, f))
def str7(a: String, b: String, c: String, d: String, e: String, f: String, g: String) -> String =
    string_concat(a, str6(b, c, d, e, f, g))

// Lowercase the ASCII letters in `s`.
def string_lower(s: String) -> Result<String, IndexError> = {
    let b = bytes_from_string(s);
    let n = bytes_len(b);
    let out = bytes_new(n);
    let filled = string_lower_fill(b, out, 0, n)?;
    Result::Ok(string_from_bytes(filled))
}
def string_lower_fill(b: Bytes, out: Bytes, i: Int, n: Int) -> Result<Bytes, IndexError> =
    if i == n {
        Result::Ok(out)
    } else {
        let c = bytes_get(b, i)?;
        let lc = if c >= 65 { if c <= 90 { c + 32 } else { c } } else { c };
        let w = bytes_set(out, i, lc)?;
        string_lower_fill(b, out, i + 1, n)
    }

// Strip leading and trailing ASCII spaces.
def string_trim(s: String) -> Result<String, IndexError> = {
    let b = bytes_from_string(s);
    let n = bytes_len(b);
    let start = string_trim_start(b, 0, n)?;
    let end = string_trim_end(b, n)?;
    if end <= start { Result::Ok("") } else {
        Result::Ok(string_from_bytes(bytes_slice(b, start, end - start)))
    }
}
def string_trim_start(b: Bytes, i: Int, n: Int) -> Result<Int, IndexError> =
    if i >= n { Result::Ok(n) } else {
        if bytes_get(b, i)? == 32 { string_trim_start(b, i + 1, n) } else { Result::Ok(i) }
    }
def string_trim_end(b: Bytes, e: Int) -> Result<Int, IndexError> =
    if e <= 0 { Result::Ok(0) } else {
        if bytes_get(b, e - 1)? == 32 { string_trim_end(b, e - 1) } else { Result::Ok(e) }
    }

// 1 if `a` sorts strictly before `b` byte-lexicographically, else 0.
def string_lt(a: String, b: String) -> Result<Int, IndexError> = {
    let ba = bytes_from_string(a);
    let bb = bytes_from_string(b);
    string_lt_go(ba, bb, 0, bytes_len(ba), bytes_len(bb))
}
def string_lt_go(ba: Bytes, bb: Bytes, i: Int, na: Int, nb: Int) -> Result<Int, IndexError> =
    if i >= na {
        if i >= nb { Result::Ok(0) } else { Result::Ok(1) }
    } else {
        if i >= nb {
            Result::Ok(0)
        } else {
            let ca = bytes_get(ba, i)?;
            let cb = bytes_get(bb, i)?;
            if ca < cb { Result::Ok(1) } else {
                if ca > cb { Result::Ok(0) } else { string_lt_go(ba, bb, i + 1, na, nb) }
            }
        }
    }

// ---- AWS SigV4 request signing ----
//
// A request header to be signed. `RequestHeaders` is an ordinary linked
// list; `sigv4_authorization` sorts/canonicalizes it.
struct SigHeader { name: String, value: String }
enum RequestHeaders { RHCons(RHCell), RHNil }
struct RHCell { head: SigHeader, tail: RequestHeaders }

def sig_header(name: String, value: String) -> SigHeader =
    SigHeader { name: name, value: value }

// Insertion sort the headers by lowercased name.
def sig_sort(xs: RequestHeaders) -> Result<RequestHeaders, IndexError> =
    match xs {
        RequestHeaders::RHNil => Result::Ok(RequestHeaders::RHNil),
        RequestHeaders::RHCons(cell) => {
            let rest = sig_sort(cell.tail)?;
            sig_insert(cell.head, rest)
        },
    }
def sig_insert(x: SigHeader, sorted: RequestHeaders) -> Result<RequestHeaders, IndexError> =
    match sorted {
        RequestHeaders::RHNil => Result::Ok(RequestHeaders::RHCons(RHCell { head: x, tail: RequestHeaders::RHNil })),
        RequestHeaders::RHCons(cell) => {
            let xl = string_lower(x.name)?;
            let cl = string_lower(cell.head.name)?;
            let lt = string_lt(xl, cl)?;
            if lt == 1 {
                Result::Ok(RequestHeaders::RHCons(RHCell { head: x, tail: sorted }))
            } else {
                let rest = sig_insert(x, cell.tail)?;
                Result::Ok(RequestHeaders::RHCons(RHCell { head: cell.head, tail: rest }))
            }
        },
    }

// `lower(name):trim(value)\n` for each header, in order.
def sig_canonical_headers(xs: RequestHeaders) -> Result<String, IndexError> =
    match xs {
        RequestHeaders::RHNil => Result::Ok(""),
        RequestHeaders::RHCons(cell) => {
            let ln = string_lower(cell.head.name)?;
            let tv = string_trim(cell.head.value)?;
            let rest = sig_canonical_headers(cell.tail)?;
            Result::Ok(str5(ln, ":", tv, "\n", rest))
        },
    }

// `lower(name)` joined by ";".
def sig_signed_headers(xs: RequestHeaders) -> Result<String, IndexError> =
    match xs {
        RequestHeaders::RHNil => Result::Ok(""),
        RequestHeaders::RHCons(cell) =>
            match cell.tail {
                RequestHeaders::RHNil => string_lower(cell.head.name),
                RequestHeaders::RHCons(rest) => {
                    let ln = string_lower(cell.head.name)?;
                    let joined = sig_signed_headers(cell.tail)?;
                    Result::Ok(str3(ln, ";", joined))
                },
            },
    }

// Compute the AWS SigV4 `Authorization` header value for a request. The
// caller supplies the (unsorted) headers it intends to sign; this builds
// the canonical request, the string-to-sign, and the signature.
def sigv4_authorization(
    method: String, canonical_uri: String, canonical_query: String,
    headers: RequestHeaders, payload: String,
    access_key: String, secret_key: String,
    region: String, service: String,
    amzdate: String, datestamp: String) -> Result<String, IndexError> = {
    let sorted = sig_sort(headers)?;
    let canon_h = sig_canonical_headers(sorted)?;
    let signed_h = sig_signed_headers(sorted)?;
    let payload_hash = sha256_hex(payload)?;
    let canonical_request =
        str7(
            string_concat(method, "\n"),
            string_concat(canonical_uri, "\n"),
            string_concat(canonical_query, "\n"),
            canon_h,
            "\n",
            string_concat(signed_h, "\n"),
            payload_hash);
    let scope = str5(datestamp, "/", region, "/", string_concat(service, "/aws4_request"));
    let cr_hash = sha256_hex(canonical_request)?;
    let string_to_sign =
        str6("AWS4-HMAC-SHA256\n", amzdate, "\n", scope, "\n", cr_hash);
    let k_signing = aws_sigv4_signing_key_raw(secret_key, datestamp, region, service)?;
    let signature = hmac_sha256_hex(k_signing, string_to_sign)?;
    Result::Ok(str7(
        "AWS4-HMAC-SHA256 Credential=",
        string_concat(access_key, "/"),
        string_concat(scope, ", SignedHeaders="),
        signed_h,
        ", Signature=",
        signature,
        ""))
}

// ---- UTC date formatting (libc gmtime_r + strftime) ----

def sigv4_strftime(epoch: Int, fmt: String) -> Result<String, IndexError> = {
    let tp = malloc(8);
    defer free(tp);
    let w = ptr_write_i64(tp, 0, epoch);
    let tm = malloc(64);
    defer free(tm);
    let g = gmtime_r(tp, tm);
    let fmtc = cstr(fmt)?;
    defer free(fmtc);
    let buf = malloc(40);
    defer free(buf);
    let n = strftime(buf, 40, fmtc, tm);
    ptr_to_string(buf, n)
}

// `YYYYMMDDTHHMMSSZ` (X-Amz-Date) and `YYYYMMDD` (credential-scope date).
def amz_datetime(epoch: Int) -> Result<String, IndexError> = sigv4_strftime(epoch, "%Y%m%dT%H%M%SZ")
def amz_datestamp(epoch: Int) -> Result<String, IndexError> = sigv4_strftime(epoch, "%Y%m%d")

// ---- AWS Lambda client ----
//
// Invoke a Lambda function by name in `region`, signing the request with
// SigV4 using the given credentials (`token` is the session token for
// temporary credentials; pass "" for long-lived keys). Returns the FULL
// HTTP response so the caller sees Lambda's status + (JSON) body, whether
// the invocation succeeded or returned an error.
def lambda_invoke(
    region: String, function_name: String, payload: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let host = str3("lambda.", region, ".amazonaws.com");
    let path = str3("/2015-03-31/functions/", function_name, "/invocations");
    let url = str3("https://", host, path);
    aws_signed_post(url, host, path, region, "lambda", payload, access_key, secret_key, token)
}

// Sign and send a POST to an AWS-style endpoint (a thin wrapper over
// `aws_signed_request`, kept for callers that only POST).
def aws_signed_post(
    url: String, host: String, canonical_uri: String,
    region: String, service: String, payload: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> =
    aws_signed_request(
        "POST", url, host, canonical_uri, region, service, payload,
        access_key, secret_key, token)

// Sign and send any-method request to an AWS endpoint. `host` must be the
// authority the request is sent to (it is part of the signature),
// `canonical_uri` the request path. Factored so the url/host can be
// pointed anywhere (e.g. a local mock) and any verb used.
def aws_signed_request(
    method: String, url: String, host: String, canonical_uri: String,
    region: String, service: String, payload: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let req = ix(aws_signed_req(
        method, url, host, canonical_uri, region, service, payload,
        access_key, secret_key, token))?;
    http_request_full(req.method, req.url, req.headers, req.body)
}

// BUILD (don't send) a SigV4-signed request as an `HttpReq`. This is the
// piece the concurrent path needs: sign N requests, then fire them all at
// once with `http_request_many`.
def aws_signed_req(
    method: String, url: String, host: String, canonical_uri: String,
    region: String, service: String, payload: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpReq, IndexError> = {
    let now = now_unix();
    let amzdate = amz_datetime(now)?;
    let datestamp = amz_datestamp(now)?;
    let payload_hash = sha256_hex(payload)?;
    let base = RequestHeaders::RHCons(RHCell { head: sig_header("host", host), tail:
               RequestHeaders::RHCons(RHCell { head: sig_header("x-amz-content-sha256", payload_hash), tail:
               RequestHeaders::RHCons(RHCell { head: sig_header("x-amz-date", amzdate), tail: RequestHeaders::RHNil }) }) });
    let headers =
        if string_is_empty(token) == 1 {
            base
        } else {
            RequestHeaders::RHCons(RHCell { head: sig_header("x-amz-security-token", token), tail: base })
        };
    let auth = sigv4_authorization(
        method, canonical_uri, "", headers, payload,
        access_key, secret_key, region, service, amzdate, datestamp)?;
    let block = lambda_header_block(amzdate, payload_hash, token, auth);
    Result::Ok(HttpReq { method: method, url: url, headers: block, body: payload })
}

// ---- Parallel Lambda fan-out (map-reduce over Lambda) ----
//
// Invoke a Lambda function CONCURRENTLY with N different payloads. Each
// invoke is independently signed; all fire at once and Lambda auto-scales
// to N parallel execution environments. Returns the N responses in order.
def lambda_invoke_many(
    region: String, name: String, payloads: List<String>,
    access_key: String, secret_key: String, token: String) -> Result<List<HttpResponse>, IndexError> = {
    let reqs = lambda_invoke_reqs(region, name, payloads, access_key, secret_key, token)?;
    http_request_many(reqs)
}

def lambda_invoke_reqs(
    region: String, name: String, payloads: List<String>,
    access_key: String, secret_key: String, token: String) -> Result<List<HttpReq>, IndexError> =
    match payloads {
        List::Nil => Result::Ok(List::Nil),
        List::Cons(cell) => {
            let head = lambda_one_req(region, name, cell.head, access_key, secret_key, token)?;
            let tail = lambda_invoke_reqs(region, name, cell.tail, access_key, secret_key, token)?;
            Result::Ok(List::Cons(ListCell { head: head, tail: tail }))
        },
    }

def lambda_one_req(
    region: String, name: String, payload: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpReq, IndexError> = {
    let host = str3("lambda.", region, ".amazonaws.com");
    let path = str3("/2015-03-31/functions/", name, "/invocations");
    let url = str3("https://", host, path);
    aws_signed_req("POST", url, host, path, region, "lambda", payload,
        access_key, secret_key, token)
}

// ---- Hand code to a generic worker at invoke time ----
//
// The deployed worker (`ai-lang lambda-worker`) has NO baked logic. You
// hand it ai-lang SOURCE (defining `task(input: String) -> String`) plus
// an input in the invoke payload; it compiles and runs it. So one generic
// function serves any computation, decided per-invocation.

def lambda_code_payload(source: String, input: String) -> Result<String, IndexError> = {
    let src64 = base64_encode(source)?;
    let in64 = base64_encode(input)?;
    Result::Ok(str5("{\"src64\":\"", src64, "\",\"in64\":\"", in64, "\"}"))
}

// Run `source`'s `task(input)` on the generic worker. The response body is
// the String the task returned.
def lambda_run_code(
    region: String, name: String, source: String, input: String,
    access_key: String, secret_key: String, token: String) -> Result<String, String> = {
    let payload = ix(lambda_code_payload(source, input))?;
    match lambda_invoke(region, name, payload, access_key, secret_key, token) {
        Result::Ok(resp) => if resp.status == 200 { Result::Ok(resp.body) } else { Result::Err(resp.body) },
        Result::Err(e) => Result::Err(e)
    }
}

// Fan out the SAME code across N inputs, all invocations in parallel.
def lambda_run_code_many(
    region: String, name: String, source: String, inputs: List<String>,
    access_key: String, secret_key: String, token: String) -> Result<List<HttpResponse>, IndexError> = {
    let reqs = lambda_code_reqs(region, name, source, inputs, access_key, secret_key, token)?;
    http_request_many(reqs)
}

def lambda_code_reqs(
    region: String, name: String, source: String, inputs: List<String>,
    access_key: String, secret_key: String, token: String) -> Result<List<HttpReq>, IndexError> =
    match inputs {
        List::Nil => Result::Ok(List::Nil),
        List::Cons(cell) => {
            let payload = lambda_code_payload(source, cell.head)?;
            let head = lambda_one_req(region, name, payload, access_key, secret_key, token)?;
            let tail = lambda_code_reqs(region, name, source, cell.tail, access_key, secret_key, token)?;
            Result::Ok(List::Cons(ListCell { head: head, tail: tail }))
        },
    }

// Create a Lambda function from a deployment zip (custom runtime,
// `provided.al2`, `bootstrap` entry). `role_arn` is the execution role.
// The zip bytes are base64-encoded into the `Code.ZipFile` field.
def lambda_create_function(
    region: String, name: String, role_arn: String, zip_bytes: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let host = str3("lambda.", region, ".amazonaws.com");
    let url = str3("https://", host, "/2015-03-31/functions");
    lambda_create_at(
        url, host, "/2015-03-31/functions", region, name, role_arn, zip_bytes,
        access_key, secret_key, token)
}

// CreateFunction against an explicit endpoint (so it can be pointed at a
// local mock). Builds the JSON body with the base64-encoded zip.
def lambda_create_at(
    url: String, host: String, path: String,
    region: String, name: String, role_arn: String, zip_bytes: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let zip64 = ix(base64_encode(zip_bytes))?;
    let body = str6(
        str3("{\"FunctionName\":\"", name, "\","),
        str3("\"Role\":\"", role_arn, "\","),
        "\"Runtime\":\"provided.al2\",\"Handler\":\"bootstrap\",",
        "\"Code\":{\"ZipFile\":\"",
        zip64,
        "\"}}");
    aws_signed_request("POST", url, host, path, region, "lambda", body,
        access_key, secret_key, token)
}

// Create a Lambda function from a container image (the realistic route
// for a JIT runtime: the ai-lang binary + LLVM + the worker codebase are
// baked into the image, which polls the Runtime API on cold start).
// `image_uri` is an ECR image reference.
def lambda_create_from_image(
    region: String, name: String, role_arn: String, image_uri: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let host = str3("lambda.", region, ".amazonaws.com");
    let url = str3("https://", host, "/2015-03-31/functions");
    let body = str6(
        str3("{\"FunctionName\":\"", name, "\","),
        str3("\"Role\":\"", role_arn, "\","),
        "\"PackageType\":\"Image\",",
        str3("\"Code\":{\"ImageUri\":\"", image_uri, "\"}}"),
        "",
        "");
    aws_signed_request("POST", url, host, "/2015-03-31/functions", region, "lambda", body,
        access_key, secret_key, token)
}

// Delete a Lambda function by name.
def lambda_delete_function(
    region: String, name: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let host = str3("lambda.", region, ".amazonaws.com");
    let path = str3("/2015-03-31/functions/", name, "");
    let url = str3("https://", host, path);
    aws_signed_request(
        "DELETE", url, host, path, region, "lambda", "",
        access_key, secret_key, token)
}

// Create a Lambda function whose code lives in S3 (the route for larger
// packages that can't go inline). `bucket`/`key` point at an already
// uploaded deployment zip.
def lambda_create_from_s3(
    region: String, name: String, role_arn: String, bucket: String, key: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let host = str3("lambda.", region, ".amazonaws.com");
    let url = str3("https://", host, "/2015-03-31/functions");
    let body = str6(
        str3("{\"FunctionName\":\"", name, "\","),
        str3("\"Role\":\"", role_arn, "\","),
        "\"Runtime\":\"provided.al2\",\"Handler\":\"bootstrap\",",
        str4("\"Code\":{\"S3Bucket\":\"", bucket, "\",\"S3Key\":\"", key),
        "\"}}",
        "");
    aws_signed_request("POST", url, host, "/2015-03-31/functions", region, "lambda", body,
        access_key, secret_key, token)
}

// ---- URI encoding (for S3 canonical request paths) ----

def char_str(c: Int) -> Result<String, IndexError> = {
    let b = bytes_new(1);
    let w = bytes_set(b, 0, c)?;
    Result::Ok(string_from_bytes(b))
}

def hex_upper(v: Int) -> Int = if v < 10 { 48 + v } else { 55 + v }

// 1 if `c` is an RFC 3986 unreserved character (A-Za-z0-9-._~).
def uri_unreserved(c: Int) -> Int =
    if c == 45 {
        1
    } else {
        if c == 46 {
            1
        } else {
            if c == 95 {
                1
            } else {
                if c == 126 {
                    1
                } else {
                    if c >= 48 {
                        if c <= 57 {
                            1
                        } else {
                            if c >= 65 {
                                if c <= 90 {
                                    1
                                } else {
                                    if c >= 97 { if c <= 122 { 1 } else { 0 } } else { 0 }
                                }
                            } else {
                                0
                            }
                        }
                    } else {
                        0
                    }
                }
            }
        }
    }

// Percent-encode `s` for an S3 canonical URI, keeping "/" separators.
def uri_encode_path(s: String) -> Result<String, IndexError> = {
    let b = bytes_from_string(s);
    uri_encode_go(b, 0, bytes_len(b), "")
}

def uri_encode_go(b: Bytes, i: Int, n: Int, acc: String) -> Result<String, IndexError> =
    if i >= n {
        Result::Ok(acc)
    } else {
        let c = bytes_get(b, i)?;
        let piece =
            if uri_unreserved(c) == 1 {
                let lit = char_str(c)?;
                lit
            } else {
                if c == 47 {
                    "/"
                } else {
                    let hi = char_str(hex_upper(bit_shr(c, 4)))?;
                    let lo = char_str(hex_upper(bit_and(c, 15)))?;
                    str3("%", hi, lo)
                }
            };
        uri_encode_go(b, i + 1, n, string_concat(acc, piece))
    }

// ---- S3 (virtual-hosted-style) ----
//
// Put / get / delete an object. `content` for put is the object body;
// get returns it in the response body. Built on `aws_signed_request`
// with service "s3" (so x-amz-content-sha256 = SHA-256 of the body,
// which S3 verifies).

def s3_host(region: String, bucket: String) -> String =
    str4(bucket, ".s3.", region, ".amazonaws.com")

def s3_put_object(
    region: String, bucket: String, key: String, content: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let host = s3_host(region, bucket);
    let enc = ix(uri_encode_path(key))?;
    let path = string_concat("/", enc);
    let url = str3("https://", host, path);
    aws_signed_request("PUT", url, host, path, region, "s3", content,
        access_key, secret_key, token)
}

def s3_get_object(
    region: String, bucket: String, key: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let host = s3_host(region, bucket);
    let enc = ix(uri_encode_path(key))?;
    let path = string_concat("/", enc);
    let url = str3("https://", host, path);
    aws_signed_request("GET", url, host, path, region, "s3", "",
        access_key, secret_key, token)
}

def s3_delete_object(
    region: String, bucket: String, key: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpResponse, String> = {
    let host = s3_host(region, bucket);
    let enc = ix(uri_encode_path(key))?;
    let path = string_concat("/", enc);
    let url = str3("https://", host, path);
    aws_signed_request("DELETE", url, host, path, region, "s3", "",
        access_key, secret_key, token)
}

// GET many S3 objects CONCURRENTLY (one signed request each, all in flight
// at once). Returns the responses in key order.
def s3_get_many(
    region: String, bucket: String, keys: List<String>,
    access_key: String, secret_key: String, token: String) -> Result<List<HttpResponse>, IndexError> = {
    let reqs = s3_get_reqs(region, bucket, keys, access_key, secret_key, token)?;
    http_request_many(reqs)
}

def s3_get_reqs(
    region: String, bucket: String, keys: List<String>,
    access_key: String, secret_key: String, token: String) -> Result<List<HttpReq>, IndexError> =
    match keys {
        List::Nil => Result::Ok(List::Nil),
        List::Cons(cell) => {
            let head = s3_one_get_req(region, bucket, cell.head, access_key, secret_key, token)?;
            let tail = s3_get_reqs(region, bucket, cell.tail, access_key, secret_key, token)?;
            Result::Ok(List::Cons(ListCell { head: head, tail: tail }))
        },
    }

def s3_one_get_req(
    region: String, bucket: String, key: String,
    access_key: String, secret_key: String, token: String) -> Result<HttpReq, IndexError> = {
    let host = s3_host(region, bucket);
    let enc = uri_encode_path(key)?;
    let path = string_concat("/", enc);
    let url = str3("https://", host, path);
    aws_signed_req("GET", url, host, path, region, "s3", "",
        access_key, secret_key, token)
}

// The wire header block (newline-separated "Key: Value"). curl supplies
// `Host` from the URL; we send the signed amz headers + Authorization,
// plus an (unsigned) Content-Type so curl does not default it.
def lambda_header_block(amzdate: String, payload_hash: String, token: String, auth: String) -> String = {
    let base = str4(
        str3("X-Amz-Date: ", amzdate, "\n"),
        str3("X-Amz-Content-Sha256: ", payload_hash, "\n"),
        str3("Authorization: ", auth, "\n"),
        "Content-Type: application/json\n");
    if string_is_empty(token) == 1 {
        base
    } else {
        string_concat(str3("X-Amz-Security-Token: ", token, "\n"), base)
    }
}

// ---- AWS Lambda custom runtime (the worker side) ----
//
// This is the other half: code that RUNS as a Lambda. AWS's custom
// runtime is pure HTTP against the Lambda Runtime API. A worker polls for
// the next invocation, runs its handler, and posts the result back — all
// expressible here now that `http_request_full` exposes response headers
// (the invocation carries its request id in `Lambda-Runtime-Aws-Request-Id`).
//
// `handler` maps an event body (a JSON string) to a response body.

// Process exactly one invocation. Returns the POST status on success.
def lambda_run_once(api_base: String, handler: fn(String) -> String) -> Result<Int, String> = {
    let next_url = str3("http://", api_base, "/2018-06-01/runtime/invocation/next");
    let inv = http_request_full("GET", next_url, "", "")?;
    let req_id = ix(http_header(inv.headers, "Lambda-Runtime-Aws-Request-Id"))?;
    let result = handler(inv.body);
    let resp_url = str5("http://", api_base, "/2018-06-01/runtime/invocation/", req_id, "/response");
    let posted = http_request_full("POST", resp_url, "", result)?;
    Result::Ok(posted.status)
}

// Poll-and-process forever (the custom-runtime main loop). Returns only
// on a transport error. Tail-recursive, so it loops without stack growth.
def lambda_run_forever(api_base: String, handler: fn(String) -> String) -> Int =
    match lambda_run_once(api_base, handler) {
        Result::Ok(s) => lambda_run_forever(api_base, handler),
        Result::Err(e) => 0 - 1
    }

// The custom-runtime entry point: read the Runtime API endpoint from the
// environment (AWS sets `AWS_LAMBDA_RUNTIME_API`) and serve forever. A
// deployed worker's `main` is just `lambda_serve(|e| my_handler(e))`.
// (Same convention as `lambda_run_forever`: a transport/environment
// failure surfaces as the -1 return, never a crash.)
def lambda_serve(handler: fn(String) -> String) -> Int =
    match env_get("AWS_LAMBDA_RUNTIME_API") {
        Result::Ok(api) => lambda_run_forever(api, handler),
        Result::Err(_e) => 0 - 1,
    }

// `println_int(n)` — print an Int followed by newline. Reuses the
// existing string-formatting and println paths.
def println_int(n: Int) -> Int = println(int_to_string(n))

// Parse-or-default: convert a String to Int, or return `default` if
// the string isn't a valid integer. Uses the `string_is_int` probe to
// avoid bare 0 ambiguity.
def parse_int_or(s: String, default: Int) -> Int =
    if string_is_int(s) == 0 { default } else { string_to_int(s) }

// ---- Distributed compute ----
//
// `Node`, `Failure`, `Result<T, E>`, and `tcp_node` form the public
// surface for the `at(node, thunk)` builtin. The runtime reads
// `Node`'s fields at fixed offsets, so the field order is
// part of the ABI — do NOT reorder without also updating
// `ai_net_at` in src/net.rs.
//
// User code calls `tcp_node(127, 0, 0, 1, port)` rather than
// constructing a Node directly. Future transports (Unix socket,
// QUIC) will get their own constructors; the field layout will
// gain a tag once we need to distinguish them at runtime.
struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }

enum Failure {
    Unreachable(Node),
    Crashed(Node),
    CodeMissing(Node),
    Cancelled(Node),
}

enum Result<T, E> { Ok(T), Err(E) }

// The error of the CHECKED array/bytes accessors (`array_get`,
// `array_set`, `bytes_get`, `bytes_set`): an out-of-bounds index is a
// `Result::Err(IndexError::OutOfBounds(OobInfo { index, len }))` VALUE
// flowing through `?`/match like any other error — never a hidden
// channel. There is no trusted/unchecked tier and no abort: the language
// is TOTAL — every indexing failure, including a read of an
// uninitialized (never-written) non-scalar slot, is an `Err` the caller
// handles or threads out with `?`.
struct OobInfo { index: Int, len: Int }
enum IndexError { OutOfBounds(OobInfo), Uninitialized(OobInfo) }

// Adapt an indexing Result into a String-error context (read_file, the
// HTTP layer, ... predate IndexError and use Result<_, String>): the
// index/len land in the message, so `let v = ix(bytes_get(b, i))?;`
// composes with their `?` without losing the evidence.
def ix<T>(r: Result<T, IndexError>) -> Result<T, String> =
    match r {
        Result::Ok(v) => Result::Ok(v),
        Result::Err(e) => match e {
            IndexError::OutOfBounds(o) => Result::Err(string_concat(
                "index out of bounds: ", string_concat(int_to_string(o.index),
                string_concat(" (len ", string_concat(int_to_string(o.len), ")"))))),
            IndexError::Uninitialized(o) => Result::Err(string_concat(
                "uninitialized slot: ", int_to_string(o.index))),
        },
    }

// Error returned by the checked, generic `decode::<T>(bytes)`:
// `TypeMismatch` = the bytes held a value of a different type than `T`;
// `Malformed` = the bytes were truncated or otherwise un-decodable.
enum DecodeError { TypeMismatch, Malformed }

def tcp_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
    Node { a: a, b: b, c: c, d: d, port: port }

// ---- Remote pointers (opt-in, explicit) ----
//
// A raw `Ptr` is a LOCAL machine address and is intentionally NOT
// shippable: the typechecker rejects passing one across an `at(...)`
// boundary. A `RemotePtr` is the deliberate way to reference foreign
// memory anyway. It is wire-portable because it carries the address as
// plain data (`addr: Int`) plus the node that owns it — never a live
// `Ptr`. You cannot dereference it locally; every access sends the
// operation back to the owning node (an `at(...)` RPC) and therefore
// returns a `Result<_, Failure>`. Nothing here happens implicitly: you
// mint a RemotePtr on purpose and read/write it on purpose.

struct RemotePtr { node: Node, addr: Int }

// Mint a RemotePtr from a LOCAL `Ptr` you know lives on `node`. This is
// the explicit opt-in that turns a non-shippable address into a portable
// handle. (`ptr_to_int` is the deliberate reinterpret.)
def remote(node: Node, p: Ptr) -> RemotePtr =
    RemotePtr { node: node, addr: ptr_to_int(p) }

// Allocate `size` bytes ON `node` and return a handle to them. The
// allocation happens remotely; only the address (an Int) travels back.
def remote_alloc(node: Node, size: Int) -> Result<RemotePtr, Failure> = {
    let addr = at(node, || ptr_to_int(malloc(size)))?;
    Result::Ok(RemotePtr { node: node, addr: addr })
}

// Free a remote allocation (RPC to the owner).
def remote_free(rp: RemotePtr) -> Result<Int, Failure> = {
    let a = rp.addr;
    at(rp.node, || free(int_to_ptr(a)))
}

// Read 8 bytes at the remote address. The thunk captures only the
// address (an Int) and reconstructs the `Ptr` ON the owning node via
// `int_to_ptr`, so no `Ptr` ever crosses the wire.
def remote_read_i64(rp: RemotePtr) -> Result<Int, Failure> = {
    let a = rp.addr;
    at(rp.node, || ptr_read_i64(int_to_ptr(a), 0))
}

// Write 8 bytes at the remote address (RPC to the owner).
def remote_write_i64(rp: RemotePtr, v: Int) -> Result<Int, Failure> = {
    let a = rp.addr;
    at(rp.node, || ptr_write_i64(int_to_ptr(a), 0, v))
}

// ---- Lazy distributed values ----
//
// `Value<T>` is a *located blueprint*: a node where the computation
// will run plus the closure that produces a T. Operations on Values
// (like `value_map`) build new blueprints WITHOUT evaluating
// anything — composition just nests closures. Only `value_force`
// ships the (potentially deeply nested) recipe to the node and
// returns the actual T to the caller.
//
// The headline property: chaining N maps produces one closure
// shipped in one round trip. Intermediate values never leave the
// worker because the recipe evaluates the whole chain locally.

struct Value<T> { node: Node, recipe: fn() -> T }

def value_pure<T>(node: Node, x: T) -> Value<T> =
    Value { node: node, recipe: || x }

def value_map<T, U>(v: Value<T>, f: fn(T) -> U) -> Value<U> =
    Value { node: v.node, recipe: || f((v.recipe)()) }

def value_force<T>(v: Value<T>) -> Result<T, Failure> =
    at(v.node, v.recipe)

def list_drop<T>(xs: List<T>, n: Int) -> List<T> =
    if n <= 0 { xs } else {
        match xs {
            List::Nil => List::Nil,
            List::Cons(c) => list_drop(c.tail, n - 1),
        }
    }

def list_take<T>(xs: List<T>, n: Int) -> List<T> =
    list_reverse(list_take_acc(xs, n, List::Nil))

def list_take_acc<T>(xs: List<T>, n: Int, acc: List<T>) -> List<T> =
    if n <= 0 { acc } else {
        match xs {
            List::Nil => acc,
            List::Cons(c) => list_take_acc(c.tail, n - 1, List::Cons(ListCell { head: c.head, tail: acc })),
        }
    }

// Append `ys` to `xs`. The reverse trick: reverse xs, then "reverse"
// it again into ys's accumulator — list_reverse_acc walks the
// already-reversed input prepending each head, restoring original
// order and emitting xs ++ ys. Reuses list_reverse_acc rather than
// re-deriving the same body (which would content-hash identical
// and cause symbol-dedup confusion in the JIT).
def list_concat<T>(xs: List<T>, ys: List<T>) -> List<T> =
    list_reverse_acc(list_reverse(xs), ys)

// ---- Dataset<T>: list of remote chunks ----
//
// A `Dataset<T>` is morally `List<Value<List<T>>>` — N chunks, each
// a lazy Value holding a list of T on some Node. Operations like
// `dataset_map` produce new lazy Datasets; no work happens until a
// terminal op like `dataset_collect` or `dataset_reduce` forces.
//
// Stacking lazy combinators composes closures locally on the
// client; only the FINAL forcing trip ships the chain to each
// worker, which evaluates everything in one pass.

// Zip a list of local chunks with a list of nodes, one chunk per
// node. Extra chunks or extra nodes are dropped.
def dataset_from_chunks<T>(chunks: List<List<T>>, nodes: List<Node>) -> List<Value<List<T>>> =
    list_reverse(dataset_from_chunks_acc(chunks, nodes, List::Nil))

def dataset_from_chunks_acc<T>(
    chunks: List<List<T>>, nodes: List<Node>, acc: List<Value<List<T>>>
) -> List<Value<List<T>>> =
    match chunks {
        List::Nil => acc,
        List::Cons(c) => match nodes {
            List::Nil => acc,
            List::Cons(n) => dataset_from_chunks_acc(
                c.tail, n.tail,
                List::Cons(ListCell { head: value_pure(n.head, c.head), tail: acc })
            ),
        },
    }

// Specialized value-level operations on `Value<List<T>>`. Each
// builds a NEW lazy Value whose recipe applies the given list op
// to v's forced data. Defined as top-level fns rather than via
// `value_map(v, |xs| ...)` because nested lambdas aren't supported
// (the inner lambda would capture v's outer scope).
def value_list_map<T, U>(v: Value<List<T>>, f: fn(T) -> U) -> Value<List<U>> =
    Value { node: v.node, recipe: || list_map((v.recipe)(), f) }

def value_list_filter<T>(v: Value<List<T>>, p: fn(T) -> Int) -> Value<List<T>> =
    Value { node: v.node, recipe: || list_filter((v.recipe)(), p) }

def value_list_foldl<T, U>(v: Value<List<T>>, init: U, f: fn(U, T) -> U) -> Value<U> =
    Value { node: v.node, recipe: || list_foldl((v.recipe)(), init, f) }

// Lazy map: each chunk-value gets a value_list_map. No round trips.
def dataset_map<T, U>(d: List<Value<List<T>>>, f: fn(T) -> U) -> List<Value<List<U>>> =
    list_map(d, |v: Value<List<T>>| value_list_map(v, f))

// Lazy filter.
def dataset_filter<T>(d: List<Value<List<T>>>, p: fn(T) -> Int) -> List<Value<List<T>>> =
    list_map(d, |v: Value<List<T>>| value_list_filter(v, p))

// Force every chunk and concatenate. Failed chunks are skipped.
def dataset_collect<T>(d: List<Value<List<T>>>) -> List<T> =
    match d {
        List::Nil => List::Nil,
        List::Cons(c) => match value_force(c.head) {
            Result::Ok(xs) => list_concat(xs, dataset_collect(c.tail)),
            Result::Err(_) => dataset_collect(c.tail),
        },
    }

// Per-chunk reduce on the worker, then client-side combine. Assumes
// `op` is associative and `init` is the identity for it.
def dataset_reduce<T>(d: List<Value<List<T>>>, init: T, op: fn(T, T) -> T) -> T =
    match d {
        List::Nil => init,
        List::Cons(c) =>
            match value_force(value_list_foldl(c.head, init, op)) {
                Result::Ok(partial) => op(partial, dataset_reduce(c.tail, init, op)),
                Result::Err(_) => dataset_reduce(c.tail, init, op),
            },
    }

//
// A `Dataset<T>` is morally `List<Value<List<T>>>` — N chunks, each
// a lazy Value holding a list of T on some Node. Operations like
// `dataset_map` produce new lazy Datasets; no work happens until a
// terminal op like `dataset_collect` or `dataset_reduce` forces.
//
// Stacking lazy combinators composes closures locally on the
// client; only the FINAL forcing trip ships the chain to each
// worker, which evaluates everything in one pass.

// ---- StringMap<V>: open-addressing hash map, string keys ----
//
// Linear probing over an `Array<SBucket<V>>`; every slot holds SEmpty
// or SFull(entry). Load factor > 0.7 triggers a 2x resize + rehash.
//
// Maps are threaded functionally: `insert` returns a new StringMap
// (wrapped in Result, since probing indexes the backing array). The
// backing array is mutated in place, so callers must use the
// returned value (`let m = smap_insert(m, k, v)?`).
//
// NOTE: deletion / tombstones are not implemented yet (probing assumes
// Empty terminates a chain). Add an SDeleted variant + two-phase probe
// when delete is needed.

struct SEntry<V> { skey: String, sval: V }
enum SBucket<V> { SEmpty, SFull(SEntry<V>) }
struct StringMap<V> { sbuckets: Array<SBucket<V>>, scount: Int, scap: Int }

// Bounded polynomial hash over the key bytes. Stays < 1000003 so the
// i64 multiply never overflows, and is always non-negative.
def str_hash(s: String) -> Result<Int, IndexError> = str_hash_acc(bytes_from_string(s), 0, 0)

def str_hash_acc(b: Bytes, i: Int, h: Int) -> Result<Int, IndexError> =
    if i >= bytes_len(b) { Result::Ok(h) }
    else { str_hash_acc(b, i + 1, (h * 31 + bytes_get(b, i)?) % 1000003) }

// Fill slots [i, cap) with SEmpty.
def smap_fill<V>(a: Array<SBucket<V>>, i: Int, cap: Int) -> Result<Array<SBucket<V>>, IndexError> =
    if i >= cap { Result::Ok(a) }
    else {
        let _x = array_set(a, i, SBucket::SEmpty)?;
        smap_fill(a, i + 1, cap)
    }

def smap_with_cap<V>(cap: Int) -> Result<StringMap<V>, IndexError> = {
    let buckets = smap_fill(array_new(cap), 0, cap)?;
    Result::Ok(StringMap { sbuckets: buckets, scount: 0, scap: cap })
}

def smap_new<V>() -> Result<StringMap<V>, IndexError> = smap_with_cap(8)

def smap_size<V>(m: StringMap<V>) -> Int = m.scount

// Probe for `key` from `idx`: index of the slot holding `key`, else
// the first SEmpty slot. `steps` bounds the loop (load < 1 guarantees
// an Empty slot is found first).
def smap_probe<V>(a: Array<SBucket<V>>, cap: Int, key: String, idx: Int, steps: Int) -> Result<Int, IndexError> =
    if steps >= cap { Result::Ok(0 - 1) }
    else {
        match array_get(a, idx)? {
            SBucket::SEmpty => Result::Ok(idx),
            SBucket::SFull(e) =>
                if string_eq(e.skey, key) == 1 { Result::Ok(idx) }
                else { smap_probe(a, cap, key, (idx + 1) % cap, steps + 1) },
        }
    }

def smap_get<V>(m: StringMap<V>, key: String) -> Result<Option<V>, IndexError> = {
    let h = str_hash(key)?;
    smap_get_at(m.sbuckets, m.scap, key, h % m.scap, 0)
}

def smap_get_at<V>(a: Array<SBucket<V>>, cap: Int, key: String, idx: Int, steps: Int) -> Result<Option<V>, IndexError> =
    if steps >= cap { Result::Ok(Option::None) }
    else {
        match array_get(a, idx)? {
            SBucket::SEmpty => Result::Ok(Option::None),
            SBucket::SFull(e) =>
                if string_eq(e.skey, key) == 1 { Result::Ok(Option::Some(e.sval)) }
                else { smap_get_at(a, cap, key, (idx + 1) % cap, steps + 1) },
        }
    }

def smap_contains<V>(m: StringMap<V>, key: String) -> Result<Int, IndexError> = {
    let found = smap_get(m, key)?;
    Result::Ok(opt_is_some(found))
}

// Insert/update. Resizes first if adding would exceed 0.7 load.
def smap_insert<V>(m: StringMap<V>, key: String, val: V) -> Result<StringMap<V>, IndexError> = {
    let grown = smap_maybe_grow(m)?;
    smap_insert_grown(grown, key, val)
}

def smap_maybe_grow<V>(m: StringMap<V>) -> Result<StringMap<V>, IndexError> =
    if (m.scount + 1) * 10 >= m.scap * 7 { smap_resize(m, m.scap * 2) } else { Result::Ok(m) }

def smap_resize<V>(m: StringMap<V>, newcap: Int) -> Result<StringMap<V>, IndexError> = {
    let fresh = smap_with_cap(newcap)?;
    smap_rehash(m.sbuckets, m.scap, 0, fresh)
}

def smap_rehash<V>(old: Array<SBucket<V>>, oldcap: Int, i: Int, dst: StringMap<V>) -> Result<StringMap<V>, IndexError> =
    if i >= oldcap { Result::Ok(dst) }
    else {
        match array_get(old, i)? {
            SBucket::SEmpty => smap_rehash(old, oldcap, i + 1, dst),
            SBucket::SFull(e) => {
                let next = smap_insert_grown(dst, e.skey, e.sval)?;
                smap_rehash(old, oldcap, i + 1, next)
            },
        }
    }

// Insert assuming capacity is sufficient (no resize check).
def smap_insert_grown<V>(m: StringMap<V>, key: String, val: V) -> Result<StringMap<V>, IndexError> = {
    let h = str_hash(key)?;
    let idx = smap_probe(m.sbuckets, m.scap, key, h % m.scap, 0)?;
    smap_place(m, idx, key, val)
}

def smap_place<V>(m: StringMap<V>, idx: Int, key: String, val: V) -> Result<StringMap<V>, IndexError> =
    match array_get(m.sbuckets, idx)? {
        // Overwriting an existing key: count unchanged.
        SBucket::SFull(_) => {
            let _x = array_set(m.sbuckets, idx, SBucket::SFull(SEntry { skey: key, sval: val }))?;
            Result::Ok(m)
        },
        // Fresh slot: count grows.
        SBucket::SEmpty => {
            let _x = array_set(m.sbuckets, idx, SBucket::SFull(SEntry { skey: key, sval: val }))?;
            Result::Ok(StringMap { sbuckets: m.sbuckets, scount: m.scount + 1, scap: m.scap })
        },
    }

// ---- IntMap<V>: open-addressing hash map, Int keys ----
//
// Mirror of StringMap with Int keys (== for equality, a bounded
// multiplicative hash). Same probing / resize / functional-threading
// contract. Deletion not implemented (see StringMap note).

struct IEntry<V> { ikey: Int, ival: V }
enum IBucket<V> { IEmpty, IFull(IEntry<V>) }
struct IntMap<V> { ibuckets: Array<IBucket<V>>, icount: Int, icap: Int }

// Bounded multiplicative mix. Reduce mod a prime first so the multiply
// stays well within i64, and keep the result non-negative.
def int_hash(k: Int) -> Int = (abs(k) % 1000003) * 2654435 % 1000003

def imap_fill<V>(a: Array<IBucket<V>>, i: Int, cap: Int) -> Result<Array<IBucket<V>>, IndexError> =
    if i >= cap { Result::Ok(a) }
    else {
        let _x = array_set(a, i, IBucket::IEmpty)?;
        imap_fill(a, i + 1, cap)
    }

def imap_with_cap<V>(cap: Int) -> Result<IntMap<V>, IndexError> = {
    let buckets = imap_fill(array_new(cap), 0, cap)?;
    Result::Ok(IntMap { ibuckets: buckets, icount: 0, icap: cap })
}

def imap_new<V>() -> Result<IntMap<V>, IndexError> = imap_with_cap(8)

def imap_size<V>(m: IntMap<V>) -> Int = m.icount

def imap_probe<V>(a: Array<IBucket<V>>, cap: Int, key: Int, idx: Int, steps: Int) -> Result<Int, IndexError> =
    if steps >= cap { Result::Ok(0 - 1) }
    else {
        match array_get(a, idx)? {
            IBucket::IEmpty => Result::Ok(idx),
            IBucket::IFull(e) =>
                if e.ikey == key { Result::Ok(idx) }
                else { imap_probe(a, cap, key, (idx + 1) % cap, steps + 1) },
        }
    }

def imap_get<V>(m: IntMap<V>, key: Int) -> Result<Option<V>, IndexError> =
    imap_get_at(m.ibuckets, m.icap, key, int_hash(key) % m.icap, 0)

def imap_get_at<V>(a: Array<IBucket<V>>, cap: Int, key: Int, idx: Int, steps: Int) -> Result<Option<V>, IndexError> =
    if steps >= cap { Result::Ok(Option::None) }
    else {
        match array_get(a, idx)? {
            IBucket::IEmpty => Result::Ok(Option::None),
            IBucket::IFull(e) =>
                if e.ikey == key { Result::Ok(Option::Some(e.ival)) }
                else { imap_get_at(a, cap, key, (idx + 1) % cap, steps + 1) },
        }
    }

def imap_contains<V>(m: IntMap<V>, key: Int) -> Result<Int, IndexError> = {
    let found = imap_get(m, key)?;
    Result::Ok(opt_is_some(found))
}

def imap_insert<V>(m: IntMap<V>, key: Int, val: V) -> Result<IntMap<V>, IndexError> = {
    let grown = imap_maybe_grow(m)?;
    imap_insert_grown(grown, key, val)
}

def imap_maybe_grow<V>(m: IntMap<V>) -> Result<IntMap<V>, IndexError> =
    if (m.icount + 1) * 10 >= m.icap * 7 { imap_resize(m, m.icap * 2) } else { Result::Ok(m) }

def imap_resize<V>(m: IntMap<V>, newcap: Int) -> Result<IntMap<V>, IndexError> = {
    let fresh = imap_with_cap(newcap)?;
    imap_rehash(m.ibuckets, m.icap, 0, fresh)
}

def imap_rehash<V>(old: Array<IBucket<V>>, oldcap: Int, i: Int, dst: IntMap<V>) -> Result<IntMap<V>, IndexError> =
    if i >= oldcap { Result::Ok(dst) }
    else {
        match array_get(old, i)? {
            IBucket::IEmpty => imap_rehash(old, oldcap, i + 1, dst),
            IBucket::IFull(e) => {
                let next = imap_insert_grown(dst, e.ikey, e.ival)?;
                imap_rehash(old, oldcap, i + 1, next)
            },
        }
    }

def imap_insert_grown<V>(m: IntMap<V>, key: Int, val: V) -> Result<IntMap<V>, IndexError> = {
    let idx = imap_probe(m.ibuckets, m.icap, key, int_hash(key) % m.icap, 0)?;
    imap_place(m, idx, key, val)
}

def imap_place<V>(m: IntMap<V>, idx: Int, key: Int, val: V) -> Result<IntMap<V>, IndexError> =
    match array_get(m.ibuckets, idx)? {
        IBucket::IFull(_) => {
            let _x = array_set(m.ibuckets, idx, IBucket::IFull(IEntry { ikey: key, ival: val }))?;
            Result::Ok(m)
        },
        IBucket::IEmpty => {
            let _x = array_set(m.ibuckets, idx, IBucket::IFull(IEntry { ikey: key, ival: val }))?;
            Result::Ok(IntMap { ibuckets: m.ibuckets, icount: m.icount + 1, icap: m.icap })
        },
    }


// ---- HashMap<K, V>: a PERSISTENT (immutable, structurally-shared) HAMT ----
//
// GENERIC keys AND values: keys are hashed and compared STRUCTURALLY via
// `value_hash` / `value_eq` (the language's universal hashability), so any
// type can be a key. `hashmap_assoc` never mutates an existing version: it
// returns a new map sharing all untouched subtrees. The immutable value an
// atom holds so `swap` is a true CAS. 32-way bitmap nodes; a collision
// bucket handles full-hash
// clashes. Ships over the wire (Array + String + struct/enum portable).

// A persistent HAMT keyed by ANY value type `K`. Keys are hashed and
// compared STRUCTURALLY via `value_hash` / `value_eq` (the language's
// universal hashability), so `HashMap<Int, V>`, `HashMap<String, V>`,
// `HashMap<MyStruct, V>` all work with one implementation. 32-way bitmap
// nodes; a collision bucket handles full-hash clashes. Immutable, so it is
// exactly the value an `Atom` holds for lock-free CAS, and it ships over
// the wire (Array + struct/enum portable).
struct HLeaf<K, V> { kh: Int, hkey: K, val: V }
struct HBitmap<K, V> { bitmap: Int, kids: Array<HNode<K, V>> }
struct HColl<K, V> { kh: Int, items: Array<HNode<K, V>> }
enum HNode<K, V> { HEmpty, HL(HLeaf<K, V>), HB(HBitmap<K, V>), HC(HColl<K, V>) }
struct HashMap<K, V> { root: HNode<K, V>, size: Int }
struct AssocOut<K, V> { node: HNode<K, V>, added: Int }

def popcount(x: Int) -> Int =
    if x == 0 { 0 } else { bit_and(x, 1) + popcount(bit_shr(x, 1)) }
def frag(h: Int, shift: Int) -> Int = bit_and(bit_shr(h, shift), 31)
def bitpos(h: Int, shift: Int) -> Int = bit_shl(1, frag(h, shift))
def index_of(bitmap: Int, bit: Int) -> Int = popcount(bit_and(bitmap, bit - 1))

def arr1<K, V>(x: HNode<K, V>) -> Result<Array<HNode<K, V>>, IndexError> = {
    let a = array_new(1);
    let _s = array_set(a, 0, x)?;
    Result::Ok(a)
}
def arr2<K, V>(a: HNode<K, V>, b: HNode<K, V>) -> Result<Array<HNode<K, V>>, IndexError> = {
    let arr = array_new(2);
    let _s0 = array_set(arr, 0, a)?;
    let _s1 = array_set(arr, 1, b)?;
    Result::Ok(arr)
}
def arr_set_copy<K, V>(a: Array<HNode<K, V>>, idx: Int, v: HNode<K, V>) -> Result<Array<HNode<K, V>>, IndexError> = {
    let n = array_len(a);
    let out = array_new(n);
    arr_copy_set(a, out, 0, n, idx, v)
}
def arr_copy_set<K, V>(src: Array<HNode<K, V>>, dst: Array<HNode<K, V>>, i: Int, n: Int, idx: Int, v: HNode<K, V>) -> Result<Array<HNode<K, V>>, IndexError> =
    if i == n { let _s = array_set(dst, idx, v)?; Result::Ok(dst) }
    else {
        let _s = array_set(dst, i, array_get(src, i)?)?;
        arr_copy_set(src, dst, i + 1, n, idx, v)
    }
def arr_insert_copy<K, V>(a: Array<HNode<K, V>>, idx: Int, v: HNode<K, V>) -> Result<Array<HNode<K, V>>, IndexError> = {
    let n = array_len(a);
    let out = array_new(n + 1);
    arr_ins_fill(a, out, 0, n, idx, v)
}
def arr_ins_fill<K, V>(src: Array<HNode<K, V>>, dst: Array<HNode<K, V>>, j: Int, n: Int, idx: Int, v: HNode<K, V>) -> Result<Array<HNode<K, V>>, IndexError> =
    if j == n + 1 { Result::Ok(dst) }
    else {
        let chosen = if j < idx { array_get(src, j)? }
            else { if j == idx { v } else { array_get(src, j - 1)? } };
        let _s = array_set(dst, j, chosen)?;
        arr_ins_fill(src, dst, j + 1, n, idx, v)
    }

def hashmap_empty<K, V>() -> HashMap<K, V> = HashMap { root: HNode::HEmpty, size: 0 }
def hashmap_size<K, V>(m: HashMap<K, V>) -> Int = m.size

def hashmap_get<K, V>(m: HashMap<K, V>, key: K) -> Result<Option<V>, IndexError> =
    node_get(m.root, value_hash(key), 0, key)
def node_get<K, V>(node: HNode<K, V>, h: Int, shift: Int, key: K) -> Result<Option<V>, IndexError> =
    match node {
        HNode::HEmpty => Result::Ok(Option::None),
        HNode::HL(leaf) =>
            if value_eq(leaf.hkey, key) { Result::Ok(Option::Some(leaf.val)) } else { Result::Ok(Option::None) },
        HNode::HC(coll) => coll_get(coll.items, 0, array_len(coll.items), key),
        HNode::HB(bm) => {
            let bit = bitpos(h, shift);
            if bit_and(bm.bitmap, bit) == 0 { Result::Ok(Option::None) }
            else {
                let kid = array_get(bm.kids, index_of(bm.bitmap, bit))?;
                node_get(kid, h, shift + 5, key)
            }
        },
    }
def coll_get<K, V>(items: Array<HNode<K, V>>, i: Int, n: Int, key: K) -> Result<Option<V>, IndexError> =
    if i == n { Result::Ok(Option::None) }
    else {
        match array_get(items, i)? {
            HNode::HL(leaf) =>
                if value_eq(leaf.hkey, key) { Result::Ok(Option::Some(leaf.val)) }
                else { coll_get(items, i + 1, n, key) },
            HNode::HEmpty => coll_get(items, i + 1, n, key),
            HNode::HB(b) => coll_get(items, i + 1, n, key),
            HNode::HC(c) => coll_get(items, i + 1, n, key),
        }
    }

def hashmap_assoc<K, V>(m: HashMap<K, V>, key: K, val: V) -> Result<HashMap<K, V>, IndexError> = {
    let out = node_assoc(m.root, value_hash(key), 0, key, val)?;
    Result::Ok(HashMap { root: out.node, size: m.size + out.added })
}
def node_assoc<K, V>(node: HNode<K, V>, h: Int, shift: Int, key: K, val: V) -> Result<AssocOut<K, V>, IndexError> =
    match node {
        HNode::HEmpty => Result::Ok(AssocOut { node: HNode::HL(HLeaf { kh: h, hkey: key, val: val }), added: 1 }),
        HNode::HL(leaf) =>
            if value_eq(leaf.hkey, key) {
                Result::Ok(AssocOut { node: HNode::HL(HLeaf { kh: h, hkey: key, val: val }), added: 0 })
            } else {
                if leaf.kh == h {
                    let c = make_coll(leaf, h, key, val)?;
                    Result::Ok(AssocOut { node: c, added: 1 })
                } else {
                    let s = split_leaf(leaf, h, key, val, shift)?;
                    Result::Ok(AssocOut { node: s, added: 1 })
                }
            },
        HNode::HC(coll) => coll_assoc(coll, key, val),
        HNode::HB(bm) => {
            let bit = bitpos(h, shift);
            let idx = index_of(bm.bitmap, bit);
            if bit_and(bm.bitmap, bit) == 0 {
                let nk = arr_insert_copy(bm.kids, idx, HNode::HL(HLeaf { kh: h, hkey: key, val: val }))?;
                Result::Ok(AssocOut { node: HNode::HB(HBitmap { bitmap: bit_or(bm.bitmap, bit), kids: nk }), added: 1 })
            } else {
                let sub = node_assoc(array_get(bm.kids, idx)?, h, shift + 5, key, val)?;
                let nk = arr_set_copy(bm.kids, idx, sub.node)?;
                Result::Ok(AssocOut { node: HNode::HB(HBitmap { bitmap: bm.bitmap, kids: nk }), added: sub.added })
            }
        },
    }
def split_leaf<K, V>(leaf: HLeaf<K, V>, h: Int, key: K, val: V, shift: Int) -> Result<HNode<K, V>, IndexError> = {
    let f1 = frag(leaf.kh, shift);
    let f2 = frag(h, shift);
    if f1 == f2 {
        let child = split_leaf(leaf, h, key, val, shift + 5)?;
        let kids = arr1(child)?;
        Result::Ok(HNode::HB(HBitmap { bitmap: bit_shl(1, f1), kids: kids }))
    } else {
        let n1 = HNode::HL(leaf);
        let n2 = HNode::HL(HLeaf { kh: h, hkey: key, val: val });
        let kids = if f1 < f2 { arr2(n1, n2)? } else { arr2(n2, n1)? };
        Result::Ok(HNode::HB(HBitmap { bitmap: bit_or(bit_shl(1, f1), bit_shl(1, f2)), kids: kids }))
    }
}
def make_coll<K, V>(leaf: HLeaf<K, V>, h: Int, key: K, val: V) -> Result<HNode<K, V>, IndexError> = {
    let items = arr2(HNode::HL(leaf), HNode::HL(HLeaf { kh: h, hkey: key, val: val }))?;
    Result::Ok(HNode::HC(HColl { kh: h, items: items }))
}
def coll_assoc<K, V>(coll: HColl<K, V>, key: K, val: V) -> Result<AssocOut<K, V>, IndexError> = {
    let found = coll_find(coll.items, 0, array_len(coll.items), key)?;
    let leaf = HNode::HL(HLeaf { kh: coll.kh, hkey: key, val: val });
    if found < 0 {
        let nk = arr_insert_copy(coll.items, array_len(coll.items), leaf)?;
        Result::Ok(AssocOut { node: HNode::HC(HColl { kh: coll.kh, items: nk }), added: 1 })
    } else {
        let nk = arr_set_copy(coll.items, found, leaf)?;
        Result::Ok(AssocOut { node: HNode::HC(HColl { kh: coll.kh, items: nk }), added: 0 })
    }
}
def coll_find<K, V>(items: Array<HNode<K, V>>, i: Int, n: Int, key: K) -> Result<Int, IndexError> =
    if i == n { Result::Ok(0 - 1) }
    else {
        match array_get(items, i)? {
            HNode::HL(leaf) => if value_eq(leaf.hkey, key) { Result::Ok(i) } else { coll_find(items, i + 1, n, key) },
            HNode::HEmpty => coll_find(items, i + 1, n, key),
            HNode::HB(b) => coll_find(items, i + 1, n, key),
            HNode::HC(c) => coll_find(items, i + 1, n, key),
        }
    }

// ---- HashMap<K, V> operations: remove, fold, keys, contains ----

def hashmap_fold<K, V, A>(m: HashMap<K, V>, init: A, f: fn(A, K, V) -> A) -> Result<A, IndexError> =
    node_fold(m.root, init, f)
def node_fold<K, V, A>(node: HNode<K, V>, acc: A, f: fn(A, K, V) -> A) -> Result<A, IndexError> =
    match node {
        HNode::HEmpty => Result::Ok(acc),
        HNode::HL(leaf) => Result::Ok(f(acc, leaf.hkey, leaf.val)),
        HNode::HC(coll) => arr_fold(coll.items, 0, array_len(coll.items), acc, f),
        HNode::HB(bm) => arr_fold(bm.kids, 0, array_len(bm.kids), acc, f),
    }
def arr_fold<K, V, A>(a: Array<HNode<K, V>>, i: Int, n: Int, acc: A, f: fn(A, K, V) -> A) -> Result<A, IndexError> =
    if i == n { Result::Ok(acc) }
    else {
        let next = node_fold(array_get(a, i)?, acc, f)?;
        arr_fold(a, i + 1, n, next, f)
    }

struct RemoveOut<K, V> { rnode: HNode<K, V>, removed: Int }
def arr_remove_at<K, V>(a: Array<HNode<K, V>>, idx: Int) -> Result<Array<HNode<K, V>>, IndexError> = {
    let n = array_len(a);
    let out = array_new(n - 1);
    arr_rm_fill(a, out, 0, n, idx)
}
def arr_rm_fill<K, V>(src: Array<HNode<K, V>>, dst: Array<HNode<K, V>>, j: Int, n: Int, idx: Int) -> Result<Array<HNode<K, V>>, IndexError> =
    if j == n { Result::Ok(dst) }
    else {
        let _w = if j < idx { array_set(dst, j, array_get(src, j)?)? }
            else { if j == idx { 0 } else { array_set(dst, j - 1, array_get(src, j)?)? } };
        arr_rm_fill(src, dst, j + 1, n, idx)
    }
def hashmap_remove<K, V>(m: HashMap<K, V>, key: K) -> Result<HashMap<K, V>, IndexError> = {
    let out = node_remove(m.root, value_hash(key), 0, key)?;
    Result::Ok(HashMap { root: out.rnode, size: m.size - out.removed })
}
def node_remove<K, V>(node: HNode<K, V>, h: Int, shift: Int, key: K) -> Result<RemoveOut<K, V>, IndexError> =
    match node {
        HNode::HEmpty => Result::Ok(RemoveOut { rnode: HNode::HEmpty, removed: 0 }),
        HNode::HL(leaf) =>
            if value_eq(leaf.hkey, key) { Result::Ok(RemoveOut { rnode: HNode::HEmpty, removed: 1 }) }
            else { Result::Ok(RemoveOut { rnode: HNode::HL(leaf), removed: 0 }) },
        HNode::HC(coll) => coll_remove(coll, key),
        HNode::HB(bm) => {
            let bit = bitpos(h, shift);
            if bit_and(bm.bitmap, bit) == 0 { Result::Ok(RemoveOut { rnode: HNode::HB(bm), removed: 0 }) }
            else {
                let idx = index_of(bm.bitmap, bit);
                let sub = node_remove(array_get(bm.kids, idx)?, h, shift + 5, key)?;
                match sub.rnode {
                    HNode::HEmpty => {
                        let nk = arr_remove_at(bm.kids, idx)?;
                        let nbm = bit_xor(bm.bitmap, bit);
                        if nbm == 0 { Result::Ok(RemoveOut { rnode: HNode::HEmpty, removed: sub.removed }) }
                        else { Result::Ok(RemoveOut { rnode: HNode::HB(HBitmap { bitmap: nbm, kids: nk }), removed: sub.removed }) }
                    },
                    HNode::HL(l) => {
                        let nk = arr_set_copy(bm.kids, idx, HNode::HL(l))?;
                        Result::Ok(RemoveOut { rnode: HNode::HB(HBitmap { bitmap: bm.bitmap, kids: nk }), removed: sub.removed })
                    },
                    HNode::HB(b2) => {
                        let nk = arr_set_copy(bm.kids, idx, HNode::HB(b2))?;
                        Result::Ok(RemoveOut { rnode: HNode::HB(HBitmap { bitmap: bm.bitmap, kids: nk }), removed: sub.removed })
                    },
                    HNode::HC(c2) => {
                        let nk = arr_set_copy(bm.kids, idx, HNode::HC(c2))?;
                        Result::Ok(RemoveOut { rnode: HNode::HB(HBitmap { bitmap: bm.bitmap, kids: nk }), removed: sub.removed })
                    },
                }
            }
        },
    }
def coll_remove<K, V>(coll: HColl<K, V>, key: K) -> Result<RemoveOut<K, V>, IndexError> = {
    let found = coll_find(coll.items, 0, array_len(coll.items), key)?;
    if found < 0 { Result::Ok(RemoveOut { rnode: HNode::HC(coll), removed: 0 }) }
    else {
        let n = array_len(coll.items);
        if n - 1 == 1 {
            let keep_idx = if found == 0 { 1 } else { 0 };
            Result::Ok(RemoveOut { rnode: array_get(coll.items, keep_idx)?, removed: 1 })
        } else {
            let nk = arr_remove_at(coll.items, found)?;
            Result::Ok(RemoveOut { rnode: HNode::HC(HColl { kh: coll.kh, items: nk }), removed: 1 })
        }
    }
}

// hashmap_contains: is the key present?
def hashmap_contains<K, V>(m: HashMap<K, V>, key: K) -> Result<Int, IndexError> =
    match hashmap_get(m, key)? { Option::Some(v) => Result::Ok(1), Option::None => Result::Ok(0) }

// hashmap_keys: all keys as a List<K> (fold collecting into a list).
def hashmap_keys<K, V>(m: HashMap<K, V>) -> Result<List<K>, IndexError> =
    hashmap_fold(m, List::Nil, |acc: List<K>, k: K, v: V| List::Cons(ListCell { head: k, tail: acc }))

// ---- Atom<T>: a local, lock-free atomic reference (Clojure-style) ----
//
// An atom is a mutable identity over an immutable value. Purely LOCAL,
// nothing to do with the network. The escape hatch for shared mutable
// state. Storage is a 1-slot array; the slot holds a pointer to the
// current immutable value.
//
// `swap(a, f)` is the Clojure move and the heart of it: read the current
// value, apply the PURE function f, and lock-free compare-and-set the
// result in, retrying if another writer beat us. NO locks: it is one
// hardware compare-exchange (the `atom_swap` primitive). Correct
// because values are immutable, so identity-CAS on the slot pointer is
// enough. f may run more than once under contention, hence must be pure.
//
// Works for ANY value type with zero user thought (Int, String, PMap,
// structs, ...): the primitive moves raw object pointers; the closure
// handles its own boxing via the uniform ABI.
//
// `Atom<T>` is a dedicated runtime shape (one GC-traced, atomically
// updated pointer cell), NOT a struct wrapping an Array. That makes a
// shared mutable cell recognizable as such to the GC and reflection,
// instead of masquerading as a 1-element immutable array.
def atom<T>(init: T) -> Atom<T> = atom_new(init)
def deref<T>(a: Atom<T>) -> T = atom_load(a)
// swap = the lock-free CAS retry loop, in the runtime primitive.
def swap<T>(a: Atom<T>, f: fn(T) -> T) -> T = atom_swap(a, f)
// reset = a swap that ignores the old value.
def reset<T>(a: Atom<T>, v: T) -> T = swap(a, |_old: T| v)

// ---- OS threads: spawn / join ----
//
// `spawn(|| ...)` runs a zero-arg closure on a fresh OS thread (its own
// shadow-stack + GC ThreadState, sharing this heap), returning a handle.
// `join(h)` blocks until that thread finishes and yields its result. The
// handle is the dedicated `ThreadHandle<T>` runtime shape, like `Atom<T>`.
def spawn<T>(thunk: fn() -> T) -> ThreadHandle<T> = thread_spawn(thunk)
// `spawn_shared` opts OUT of isolation: the worker shares the parent's heap
// objects directly (zero-copy). Use with `Atom`-mediated sharing, or when
// you accept responsibility for races on plain mutable data.
def spawn_shared<T>(thunk: fn() -> T) -> ThreadHandle<T> = thread_spawn_shared(thunk)
def join<T>(h: ThreadHandle<T>) -> T = thread_join(h)

// ---- Networking: POSIX sockets + length-prefixed framing ----
//
// The ENTIRE transport is written here in ai-lang. The only primitives
// are libc socket syscalls (declared as C externs) plus the existing
// `Ptr` memory intrinsics. There is NO Rust networking code behind this:
// bind/accept/recv/send and the frame protocol are all ordinary ai-lang.
//
// This is the foundation for writing a *node* (a server loop) in the
// language itself, rather than relying on a hardcoded Rust serve loop.
//
// File descriptors (listeners and connections) are plain `Int`s, exactly
// as the kernel hands them back. Every fallible op returns
// `Result<_, NetErr>`.

extern "C" lib "c" {
    fn socket(domain: Int, ty: Int, protocol: Int) -> Int
    fn bind(fd: Int, addr: Ptr, addrlen: Int) -> Int
    fn listen(fd: Int, backlog: Int) -> Int
    fn accept(fd: Int, addr: Ptr, addrlen: Ptr) -> Int
    fn connect(fd: Int, addr: Ptr, addrlen: Int) -> Int
    fn read(fd: Int, buf: Ptr, count: Int) -> Int
    fn write(fd: Int, buf: Ptr, count: Int) -> Int
    fn close(fd: Int) -> Int
    fn setsockopt(fd: Int, level: Int, optname: Int, optval: Ptr, optlen: Int) -> Int
    fn unlink(path: Ptr) -> Int
}

enum NetErr {
    SocketFailed,
    BindFailed,
    ListenFailed,
    AcceptFailed,
    ConnectFailed,
    ConnClosed,
    WriteFailed,
    FrameTooLarge,
}

// Largest frame we will read, to bound a corrupt/hostile length prefix.
def net_max_frame() -> Int = 67108864   // 64 MiB

// Zero `n` bytes starting at `p`. (malloc does not zero.)
def net_zero(p: Ptr, i: Int, n: Int) -> Int =
    if i >= n { 0 } else { let _w = ptr_write_u8(p, i, 0); net_zero(p, i + 1, n) }

// Build a `sockaddr_in` (16 bytes) for a.b.c.d:port in a fresh malloc'd
// buffer. macOS layout: sin_len@0, sin_family@1; port + addr are stored
// in network byte order (big-endian) by hand, so no htons is needed.
def net_sockaddr_in(a: Int, b: Int, c: Int, d: Int, port: Int) -> Ptr = {
    let sa = malloc(16);
    let _z = net_zero(sa, 0, 16);
    let _l = ptr_write_u8(sa, 0, 16);                  // sin_len
    let _f = ptr_write_u8(sa, 1, 2);                   // AF_INET
    let _ph = ptr_write_u8(sa, 2, (port / 256) % 256); // port hi
    let _pl = ptr_write_u8(sa, 3, port % 256);         // port lo
    let _a = ptr_write_u8(sa, 4, a);
    let _b = ptr_write_u8(sa, 5, b);
    let _c = ptr_write_u8(sa, 6, c);
    let _d = ptr_write_u8(sa, 7, d);
    sa
}

// SO_REUSEADDR so a server can rebind a recently-used port. (macOS
// SOL_SOCKET = 0xffff, SO_REUSEADDR = 0x0004.)
def net_set_reuseaddr(fd: Int) -> Int = {
    let one = malloc(8);
    let _z = ptr_write_i64(one, 0, 0);
    let _o = ptr_write_u8(one, 0, 1);
    let r = setsockopt(fd, 65535, 4, one, 4);
    let _free = free(one);
    r
}

// Open a listening socket bound to 0.0.0.0:port. Returns the listener fd.
def tcp_listen(port: Int) -> Result<Int, NetErr> = {
    let fd = socket(2, 1, 0);   // AF_INET, SOCK_STREAM, default proto
    if fd < 0 { Result::Err(NetErr::SocketFailed) } else {
        let _r = net_set_reuseaddr(fd);
        let sa = net_sockaddr_in(0, 0, 0, 0, port);
        defer free(sa);
        let br = bind(fd, sa, 16);
        if br < 0 {
            let _c = close(fd);
            Result::Err(NetErr::BindFailed)
        } else {
            let lr = listen(fd, 16);
            if lr < 0 {
                let _c = close(fd);
                Result::Err(NetErr::ListenFailed)
            } else {
                Result::Ok(fd)
            }
        }
    }
}

// Accept one connection. Blocks until a client connects. Returns the
// connection fd. (NULL addr/addrlen: we don't need the peer address.)
def tcp_accept(listener: Int) -> Result<Int, NetErr> = {
    let c = accept(listener, ptr_null(), ptr_null());
    if c < 0 { Result::Err(NetErr::AcceptFailed) } else { Result::Ok(c) }
}

// Connect to a.b.c.d:port. Returns the connection fd.
def tcp_connect(a: Int, b: Int, c: Int, d: Int, port: Int) -> Result<Int, NetErr> = {
    let fd = socket(2, 1, 0);
    if fd < 0 { Result::Err(NetErr::SocketFailed) } else {
        let sa = net_sockaddr_in(a, b, c, d, port);
        defer free(sa);
        let r = connect(fd, sa, 16);
        if r < 0 {
            let _c = close(fd);
            Result::Err(NetErr::ConnectFailed)
        } else {
            Result::Ok(fd)
        }
    }
}

def conn_close(fd: Int) -> Int = close(fd)

// --- Length-prefixed framing (4-byte big-endian length + payload) ---

// Read exactly `n` bytes from `fd` into `buf` starting at `got`,
// retrying on short reads. ConnClosed on EOF (read returns 0) or error.
def net_read_into(fd: Int, buf: Ptr, got: Int, n: Int) -> Result<Int, NetErr> =
    if got >= n { Result::Ok(got) } else {
        let r = read(fd, ptr_add(buf, got), n - got);
        if r <= 0 { Result::Err(NetErr::ConnClosed) } else { net_read_into(fd, buf, got + r, n) }
    }

// Copy `n` raw bytes from `buf` into Bytes `b`.
def net_buf_to_bytes(buf: Ptr, b: Bytes, i: Int, n: Int) -> Result<Bytes, IndexError> =
    if i >= n { Result::Ok(b) } else {
        let _s = bytes_set(b, i, ptr_read_u8(buf, i))?;
        net_buf_to_bytes(buf, b, i + 1, n)
    }

// Receive exactly `n` bytes as a Bytes. A copy IndexError is mapped to
// `NetErr::ConnClosed` (structurally unreachable: `b` is freshly sized
// to `n`), keeping the NetErr boundary callers pattern-match intact.
def net_recv_exact(fd: Int, n: Int) -> Result<Bytes, NetErr> = {
    let buf = malloc(n);
    defer free(buf);
    let _got = net_read_into(fd, buf, 0, n)?;
    let b = bytes_new(n);
    match net_buf_to_bytes(buf, b, 0, n) {
        Result::Ok(filled) => Result::Ok(filled),
        Result::Err(_e) => Result::Err(NetErr::ConnClosed),
    }
}

// Decode a 4-byte big-endian length.
def net_be32(b: Bytes) -> Result<Int, IndexError> = {
    let b0 = bytes_get(b, 0)?;
    let b1 = bytes_get(b, 1)?;
    let b2 = bytes_get(b, 2)?;
    let b3 = bytes_get(b, 3)?;
    Result::Ok(b0 * 16777216 + b1 * 65536 + b2 * 256 + b3)
}

// Receive one length-prefixed frame. A header that fails to decode (a
// short header can never index all four bytes) surfaces as ConnClosed —
// the connection did not deliver a usable frame.
def recv_frame(fd: Int) -> Result<Bytes, NetErr> = {
    let hdr = net_recv_exact(fd, 4)?;
    match net_be32(hdr) {
        Result::Ok(len) =>
            if len > net_max_frame() { Result::Err(NetErr::FrameTooLarge) } else { net_recv_exact(fd, len) },
        Result::Err(_e) => Result::Err(NetErr::ConnClosed),
    }
}

// Copy Bytes `b` into raw buffer `buf`.
def net_bytes_to_buf(b: Bytes, buf: Ptr, i: Int, n: Int) -> Result<Int, IndexError> =
    if i >= n { Result::Ok(0) } else {
        let v = bytes_get(b, i)?;
        let _w = ptr_write_u8(buf, i, v);
        net_bytes_to_buf(b, buf, i + 1, n)
    }

// Write exactly `n` bytes from `buf` (starting at `sent`), retrying on
// short writes.
def net_write_all(fd: Int, buf: Ptr, sent: Int, n: Int) -> Result<Int, NetErr> =
    if sent >= n { Result::Ok(sent) } else {
        let w = write(fd, ptr_add(buf, sent), n - sent);
        if w <= 0 { Result::Err(NetErr::WriteFailed) } else { net_write_all(fd, buf, sent + w, n) }
    }

// Send all bytes of `b` over `fd`. A copy that fails to index `b`
// (impossible while `n` is `b`'s own length) means nothing was written,
// surfaced as WriteFailed.
def net_send_bytes(fd: Int, b: Bytes) -> Result<Int, NetErr> = {
    let n = bytes_len(b);
    let buf = malloc(n);
    defer free(buf);
    match net_bytes_to_buf(b, buf, 0, n) {
        Result::Ok(_c) => net_write_all(fd, buf, 0, n),
        Result::Err(_e) => Result::Err(NetErr::WriteFailed),
    }
}

// Encode a 4-byte big-endian length header.
def net_be32_bytes(n: Int) -> Result<Bytes, IndexError> = {
    let h = bytes_new(4);
    let _0 = bytes_set(h, 0, (n / 16777216) % 256)?;
    let _1 = bytes_set(h, 1, (n / 65536) % 256)?;
    let _2 = bytes_set(h, 2, (n / 256) % 256)?;
    let _3 = bytes_set(h, 3, n % 256)?;
    Result::Ok(h)
}

// Send one length-prefixed frame. A header that fails to encode (the
// four stores into a fresh 4-byte buffer cannot go out of bounds) means
// the frame was never written, surfaced as WriteFailed.
def send_frame(fd: Int, payload: Bytes) -> Result<Int, NetErr> =
    match net_be32_bytes(bytes_len(payload)) {
        Result::Ok(hdr) => net_send_bytes(fd, bytes_concat(hdr, payload)),
        Result::Err(_e) => Result::Err(NetErr::WriteFailed),
    }

// ---- Generic at() handler server ----
//
// `serve` runs node-resident at() handlers. One turn: accept a connection,
// read the shipped closure frame, run it ON THIS NODE (`wire_invoke`
// decodes the closure, invokes it, and encodes the result), ship the result
// back. Handler-agnostic: the closure carries its own code, so `serve`
// never names a handler or a `state` — a node author just `serve`s and
// remote participants ship `|| handler(msg)`. `wire_invoke` does NOT
// memoize, so a stateful handler (one that touches a node `state`) runs
// every time, which is exactly what mutation requires.
//
// Requires the host to have installed the current runtime (so `wire_invoke`
// can reach the node's code table + state). Returns 0 on a handled request,
// negative on accept/recv failure (the loop variants keep going regardless).
def serve_turn(listener: Int) -> Int =
    match tcp_accept(listener) {
        Result::Ok(conn) => match recv_frame(conn) {
            Result::Ok(req) => {
                let resp = wire_invoke(req);
                let _s = send_frame(conn, resp);
                let _c = conn_close(conn);
                0
            },
            Result::Err(_e) => 0 - 1,
        },
        Result::Err(_e) => 0 - 2,
    }

// Handle exactly `n` requests then return (testable; bounded).
def serve_turns(listener: Int, n: Int) -> Int =
    if n <= 0 { 0 } else {
        let _t = serve_turn(listener);
        serve_turns(listener, n - 1)
    }

// Serve forever (tail-recursive loop, no stack growth). The node's main.
def serve(listener: Int) -> Int = {
    let _t = serve_turn(listener);
    serve(listener)
}

// ============================================================
// Transports — `at` over a pluggable carrier (records of closures)
// ============================================================
//
// A `Transport` is the language-level "trait" for reaching a node: a
// record holding ONE closure that turns a request frame (the shipped
// closure's wire bytes) into a reply frame (the encoded result). Any
// carrier — TCP, a Unix socket, an in-process call, later HTTP/Lambda —
// is just a value of this type, built by capturing its own connection
// details. `at_via` drives any of them; nothing in it knows how the
// bytes actually travel.

struct Transport { roundtrip: fn(Bytes) -> Result<Bytes, NetErr> }

// `at`, as an ordinary function over a pluggable transport. Ship the
// thunk's bytes; decode the reply back to T. A decode failure (the reply
// held some other type, or was truncated) folds into ConnClosed — the
// call did not deliver a usable T.
def at_via<T>(t: Transport, thunk: fn() -> T) -> Result<T, NetErr> = {
    let req = wire_encode(thunk);
    let rt = t.roundtrip;
    match rt(req) {
        Result::Ok(resp) => match decode::<T>(resp) {
            Result::Ok(v) => Result::Ok(v),
            Result::Err(_e) => Result::Err(NetErr::ConnClosed),
        },
        Result::Err(e) => Result::Err(e),
    }
}

// One request/response over an already-connected fd, closing it after.
def fd_roundtrip(fd: Int, req: Bytes) -> Result<Bytes, NetErr> =
    match send_frame(fd, req) {
        Result::Ok(_w) => {
            let r = recv_frame(fd);
            let _c = conn_close(fd);
            r
        },
        Result::Err(e) => { let _c = conn_close(fd); Result::Err(e) },
    }

// In-process transport: run the shipped thunk locally via wire_invoke.
// No socket, no copy off-process — the same interface, a different carrier.
def local_transport() -> Transport =
    Transport { roundtrip: |req: Bytes| Result::Ok(wire_invoke(req)) }

// TCP transport: connect per call, frame the request, read the reply.
def tcp_roundtrip(a: Int, b: Int, c: Int, d: Int, port: Int, req: Bytes) -> Result<Bytes, NetErr> =
    match tcp_connect(a, b, c, d, port) {
        Result::Ok(fd) => fd_roundtrip(fd, req),
        Result::Err(e) => Result::Err(e),
    }

def tcp_transport(a: Int, b: Int, c: Int, d: Int, port: Int) -> Transport =
    Transport { roundtrip: |req: Bytes| tcp_roundtrip(a, b, c, d, port, req) }

// ---- Unix-domain-socket transport ----
//
// Only `connect`/`listen` differ from TCP — the framing (send_frame /
// recv_frame), the per-fd roundtrip (fd_roundtrip), and the server loop
// (serve_turns, whose accept takes any listener fd) are all fd-generic
// and reused verbatim. That reuse is the whole point of the abstraction.

// Build a `sockaddr_un` for `path` in a fresh malloc'd buffer. macOS
// layout: sun_len@0, sun_family@1 (AF_UNIX = 1), then the NUL-terminated
// path; the used address length is `len(path) + 3`. (On Linux byte 0 is
// the low byte of the 2-byte family field — this targets the dev host.)
def net_sockaddr_un(path: String) -> Ptr = {
    let pb = bytes_from_string(path);
    let n = bytes_len(pb);
    let total = n + 3;
    let sa = malloc(total);
    let _z = net_zero(sa, 0, total);
    let _l = ptr_write_u8(sa, 0, total);   // sun_len
    let _f = ptr_write_u8(sa, 1, 1);       // AF_UNIX
    let _p = net_bytes_to_buf(pb, ptr_add(sa, 2), 0, n);
    sa
}

def net_sockaddr_un_len(path: String) -> Int = bytes_len(bytes_from_string(path)) + 3

// A NUL-terminated C string in a fresh malloc'd buffer (the C FFI takes
// `Ptr`, not `String`). Caller frees.
def net_cstr(s: String) -> Ptr = {
    let b = bytes_from_string(s);
    let n = bytes_len(b);
    let p = malloc(n + 1);
    let _z = ptr_write_u8(p, n, 0);
    let _c = net_bytes_to_buf(b, p, 0, n);
    p
}

// Connect a SOCK_STREAM AF_UNIX socket to `path`.
def unix_connect(path: String) -> Result<Int, NetErr> = {
    let fd = socket(1, 1, 0);              // AF_UNIX, SOCK_STREAM
    if fd < 0 { Result::Err(NetErr::SocketFailed) } else {
        let sa = net_sockaddr_un(path);
        defer free(sa);
        let r = connect(fd, sa, net_sockaddr_un_len(path));
        if r < 0 {
            let _c = close(fd);
            Result::Err(NetErr::ConnectFailed)
        } else {
            Result::Ok(fd)
        }
    }
}

// Open a listening AF_UNIX socket bound to `path`, removing any stale
// socket file first. Returns the listener fd.
def unix_listen(path: String) -> Result<Int, NetErr> = {
    let cp = net_cstr(path);
    let _u = unlink(cp);
    let _fc = free(cp);
    let fd = socket(1, 1, 0);
    if fd < 0 { Result::Err(NetErr::SocketFailed) } else {
        let sa = net_sockaddr_un(path);
        defer free(sa);
        let r = bind(fd, sa, net_sockaddr_un_len(path));
        if r < 0 {
            let _c = close(fd);
            Result::Err(NetErr::BindFailed)
        } else {
            let lr = listen(fd, 16);
            if lr < 0 {
                let _c = close(fd);
                Result::Err(NetErr::ListenFailed)
            } else {
                Result::Ok(fd)
            }
        }
    }
}

def unix_roundtrip(path: String, req: Bytes) -> Result<Bytes, NetErr> =
    match unix_connect(path) {
        Result::Ok(fd) => fd_roundtrip(fd, req),
        Result::Err(e) => Result::Err(e),
    }

def unix_transport(path: String) -> Transport =
    Transport { roundtrip: |req: Bytes| unix_roundtrip(path, req) }

// ---- Live service slots: deploy / upgrade / rollback as a library ----
//
// A service is node `state` holding a STACK of handler versions — head
// is the current version, the tail is history. Deployment then needs no
// protocol at all: ship the install with the existing at() machinery
// (the handler's code travels by hash; the node JIT-installs it), and
// rollback is popping the stack — the old version's code is still
// resident.
//
//   state svc: Atom<List<fn(Int) -> Int>> = atom_new(svc_slot())
//   def install(h: fn(Int) -> Int) -> Int = svc_install(svc, h)
//   def handle(x: Int) -> Int = { let f = svc_current(svc); f(x) }
//
//   at(node, || install(|x: Int| x * 2))   // deploy a new version
//   at(node, || handle(5))                  // run the node's CURRENT version
//   at(node, || rollback())                 // instant flip back
//
// The slot's type pins the handler interface: an ill-typed handler is a
// TYPECHECK ERROR at the install call site — no runtime check needed.
// All of these touch node state, so at() never memoizes them (the
// stateful-hash cache bypass); repeated installs and calls always run.

def svc_slot<T>() -> List<T> = List::Nil

// Push a new current version. Returns the number of versions live.
def svc_install<T>(a: Atom<List<T>>, h: T) -> Int = {
    let vs = swap(a, |vs: List<T>| List::Cons(ListCell { head: h, tail: vs }));
    list_length(vs)
}

// The current version, or `None` when nothing is installed — a missing
// handler is a VALUE the caller decides about, never a crash and never
// a default.
def svc_current<T>(a: Atom<List<T>>) -> Option<T> =
    match deref(a) {
        List::Cons(cell) => Option::Some(cell.head),
        List::Nil => Option::None,
    }

// Flip back to the previous version (which is still JIT-resident).
// Returns the number of versions remaining. A slot holding zero or one
// version is left UNCHANGED (a service never becomes empty); the
// returned count tells the caller whether anything was popped.
def svc_rollback<T>(a: Atom<List<T>>) -> Int = {
    let vs = swap(a, |vs: List<T>| svc_pop(vs));
    list_length(vs)
}

def svc_pop<T>(vs: List<T>) -> List<T> =
    match vs {
        List::Cons(cell) => match cell.tail {
            List::Cons(_prev) => cell.tail,
            // A single version stays: never pop a service to empty.
            List::Nil => vs,
        },
        List::Nil => List::Nil,
    }

// How many versions a slot holds (current + history).
def svc_versions<T>(a: Atom<List<T>>) -> Int = list_length(deref(a))

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
    fn hamt_basic_assoc_get() {
        init();
        let ctx = Context::create();
        let driver = "
            def chk(r: Result<Int, IndexError>) -> Int =
                match r { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }
            def get_or(m: HashMap<String, Int>, k: String, d: Int) -> Result<Int, IndexError> = {
                let o = hashmap_get(m, k)?;
                Result::Ok(opt_unwrap_or(o, d))
            }
            def build3() -> Result<HashMap<String, Int>, IndexError> = {
                let m1 = hashmap_assoc(hashmap_empty(), \"apple\", 1)?;
                let m2 = hashmap_assoc(m1, \"banana\", 2)?;
                hashmap_assoc(m2, \"cherry\", 3)
            }
            def t_a_go() -> Result<Int, IndexError> = { let m = build3()?; get_or(m, \"apple\", 0 - 1) }
            def t_a() -> Int = chk(t_a_go())
            def t_b_go() -> Result<Int, IndexError> = { let m = build3()?; get_or(m, \"banana\", 0 - 1) }
            def t_b() -> Int = chk(t_b_go())
            def t_c_go() -> Result<Int, IndexError> = { let m = build3()?; get_or(m, \"cherry\", 0 - 1) }
            def t_c() -> Int = chk(t_c_go())
            def t_missing_go() -> Result<Int, IndexError> = { let m = build3()?; get_or(m, \"zebra\", 0 - 1) }
            def t_missing() -> Int = chk(t_missing_go())
            def t_size_go() -> Result<Int, IndexError> = { let m = build3()?; Result::Ok(hashmap_size(m)) }
            def t_size() -> Int = chk(t_size_go())
            def t_replace_go() -> Result<Int, IndexError> = {
                let m0 = build3()?;
                let m = hashmap_assoc(m0, \"apple\", 100)?;
                get_or(m, \"apple\", 0 - 1)
            }
            def t_replace() -> Int = chk(t_replace_go())
            def t_replace_size_go() -> Result<Int, IndexError> = {
                let m0 = build3()?;
                let m = hashmap_assoc(m0, \"apple\", 100)?;
                Result::Ok(hashmap_size(m))
            }
            def t_replace_size() -> Int = chk(t_replace_size_go())
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(
                        &names[n],
                    ))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("t_a"), 1);
            assert_eq!(f("t_b"), 2);
            assert_eq!(f("t_c"), 3);
            assert_eq!(f("t_missing"), -1);
            assert_eq!(f("t_size"), 3);
            assert_eq!(f("t_replace"), 100);
            assert_eq!(f("t_replace_size"), 3, "replace must not grow size");
        }
    }

    #[test]
    fn hamt_immutability_and_sharing() {
        init();
        let ctx = Context::create();
        // assoc on a derived map must NOT mutate the original.
        let driver = "
            def chk(r: Result<Int, IndexError>) -> Int =
                match r { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }
            def get_or(m: HashMap<String, Int>, k: String, d: Int) -> Result<Int, IndexError> = {
                let o = hashmap_get(m, k)?;
                Result::Ok(opt_unwrap_or(o, d))
            }
            def k10() -> Result<HashMap<String, Int>, IndexError> = {
                let m1 = hashmap_assoc(hashmap_empty(), \"alpha\", 1)?;
                let m2 = hashmap_assoc(m1, \"beta\", 2)?;
                let m3 = hashmap_assoc(m2, \"gamma\", 3)?;
                let m4 = hashmap_assoc(m3, \"delta\", 4)?;
                let m5 = hashmap_assoc(m4, \"epsilon\", 5)?;
                let m6 = hashmap_assoc(m5, \"zeta\", 6)?;
                let m7 = hashmap_assoc(m6, \"eta\", 7)?;
                let m8 = hashmap_assoc(m7, \"theta\", 8)?;
                let m9 = hashmap_assoc(m8, \"iota\", 9)?;
                hashmap_assoc(m9, \"kappa\", 10)
            }
            def old_beta_go() -> Result<Int, IndexError> = { let m = k10()?; get_or(m, \"beta\", 0 - 1) }
            def old_beta() -> Int = chk(old_beta_go())
            def new_beta_go() -> Result<Int, IndexError> = {
                let m10 = k10()?;
                let m11 = hashmap_assoc(m10, \"beta\", 999)?;
                get_or(m11, \"beta\", 0 - 1)
            }
            def new_beta() -> Int = chk(new_beta_go())
            def old_after_new_go() -> Result<Int, IndexError> = {
                let m10 = k10()?;
                let m11 = hashmap_assoc(m10, \"beta\", 999)?;
                get_or(m10, \"beta\", 0 - 1)
            }
            def old_after_new() -> Int = chk(old_after_new_go())
            def shared_kappa_go() -> Result<Int, IndexError> = {
                let m10 = k10()?;
                let m11 = hashmap_assoc(m10, \"beta\", 999)?;
                get_or(m11, \"kappa\", 0 - 1)
            }
            def shared_kappa() -> Int = chk(shared_kappa_go())
            def size10_go() -> Result<Int, IndexError> = { let m = k10()?; Result::Ok(hashmap_size(m)) }
            def size10() -> Int = chk(size10_go())
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(
                        &names[n],
                    ))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("size10"), 10, "all ten distinct keys present");
            assert_eq!(f("old_beta"), 2);
            assert_eq!(f("new_beta"), 999);
            assert_eq!(
                f("old_after_new"),
                2,
                "assoc on a derived map must NOT mutate the original (immutability)"
            );
            assert_eq!(f("shared_kappa"), 10, "untouched keys survive in the new version");
        }
    }

    #[test]
    fn hashmap_wire_roundtrip() {
        init();
        let ctx = Context::create();
        let driver = "
            def get_or(m: HashMap<String, Int>, k: String, d: Int) -> Int =
                match hashmap_get(m, k) {
                    Result::Ok(o) => opt_unwrap_or(o, d),
                    Result::Err(_) => 0 - 999
                }
            def build3_go() -> Result<HashMap<String, Int>, IndexError> = {
                let m1 = hashmap_assoc(hashmap_empty(), \"apple\", 1)?;
                let m2 = hashmap_assoc(m1, \"banana\", 2)?;
                hashmap_assoc(m2, \"cherry\", 3)
            }
            // Err fallback is an EMPTY map: every Rust-side lookup assertion
            // then fails loudly (get_or returns the -1 default).
            def build3() -> HashMap<String, Int> =
                match build3_go() { Result::Ok(m) => m, Result::Err(_) => hashmap_empty() }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            // Build a 3-entry persistent map (contains String keys and,
            // once it branches, Array<HNode> nodes).
            let build3 = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> *mut u8>(&def_symbol(
                    &names["build3"],
                ))
                .unwrap();
            let m = build3.call(rt.thread_ptr());

            // Encode to wire bytes, then decode into a FRESH map on the
            // same runtime — the exact path atoms use.
            let mut buf = Vec::new();
            crate::wire::encode_value(&rt, crate::wire::WireValue::Heap(m as *const u8), &mut buf)
                .expect("encode PMap");
            let (decoded, consumed) =
                crate::wire::decode_value(&rt, &buf).expect("decode PMap");
            assert_eq!(consumed, buf.len(), "decoder must consume exactly all bytes");
            let dm = match decoded {
                crate::wire::WireValue::Heap(p) => p as *mut u8,
                _ => panic!("decoded PMap should be a heap value"),
            };

            // Look up every key in the DECODED map. The decoded map `dm` and
            // each freshly-built key are heap pointers held across MORE
            // allocations (building the next key, and each lookup), so they
            // must be rooted — otherwise a moving GC relocates them and we
            // pass stale pointers to `get_or`. The borrow-checked `Alloc`
            // handle layer makes this safe (and makes forgetting a compile
            // error): a `Rooted` re-read via `.get(a)` always tracks the move.
            let get_or = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, *mut u8, i64) -> i64>(
                    &def_symbol(&names["get_or"]),
                )
                .unwrap();
            crate::handle::Alloc::enter(rt.thread_ptr(), |a, scope| {
                let dm = a.adopt(dm).root(scope);
                let apple = a.str_new(b"apple").root(scope);
                let banana = a.str_new(b"banana").root(scope);
                let cherry = a.str_new(b"cherry").root(scope);
                let zebra = a.str_new(b"zebra").root(scope);
                let t = a.thread();
                assert_eq!(get_or.call(t, dm.get(a).ptr(), apple.get(a).ptr(), -1), 1);
                assert_eq!(get_or.call(t, dm.get(a).ptr(), banana.get(a).ptr(), -1), 2);
                assert_eq!(get_or.call(t, dm.get(a).ptr(), cherry.get(a).ptr(), -1), 3);
                assert_eq!(get_or.call(t, dm.get(a).ptr(), zebra.get(a).ptr(), -1), -1);
            });
        }
    }

    /// The whole point: `HashMap<K, V>` keys on ANY type, hashed + compared
    /// STRUCTURALLY via `value_hash`/`value_eq`. Int keys, and STRUCT keys
    /// where a freshly-rebuilt structurally-equal key still hits.
    #[test]
    fn hashmap_nonstring_keys() {
        init();
        let ctx = Context::create();
        let driver = "
            struct Point { x: Int, y: Int }
            def chk(r: Result<Int, IndexError>) -> Int =
                match r { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }

            // Int keys (boxed Ints hashed/compared by content).
            def hm_int_go() -> Result<Int, IndexError> = {
                let m1 = hashmap_assoc(hashmap_empty(), 10, 100)?;
                let m = hashmap_assoc(m1, 20, 200)?;
                let o = hashmap_get(m, 20)?;
                Result::Ok(opt_unwrap_or(o, 0 - 1))
            }
            def hm_int() -> Int = chk(hm_int_go())
            def hm_int_overwrite_go() -> Result<Int, IndexError> = {
                let m1 = hashmap_assoc(hashmap_empty(), 7, 1)?;
                let m = hashmap_assoc(m1, 7, 2)?;
                let o = hashmap_get(m, 7)?;
                Result::Ok(hashmap_size(m) * 1000 + opt_unwrap_or(o, 0 - 1))
            }
            def hm_int_overwrite() -> Int = chk(hm_int_overwrite_go())

            // Struct keys: a DIFFERENT Point object that is structurally
            // equal must resolve to the same entry (value equality, not
            // pointer identity). Struct literals appear as MID call args —
            // unambiguous inside the call's parens.
            def hm_struct_hit_go() -> Result<Int, IndexError> = {
                let m = hashmap_assoc(hashmap_empty(), Point { x: 1, y: 2 }, 42)?;
                let o = hashmap_get(m, Point { x: 1, y: 2 })?;
                Result::Ok(opt_unwrap_or(o, 0 - 1))
            }
            def hm_struct_hit() -> Int = chk(hm_struct_hit_go())
            def hm_struct_miss_go() -> Result<Int, IndexError> = {
                let m = hashmap_assoc(hashmap_empty(), Point { x: 1, y: 2 }, 42)?;
                let o = hashmap_get(m, Point { x: 1, y: 9 })?;
                Result::Ok(opt_unwrap_or(o, 0 - 1))
            }
            def hm_struct_miss() -> Int = chk(hm_struct_miss_go())
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("hm_int"), 200, "Int keys work");
            assert_eq!(f("hm_int_overwrite"), 1002, "same Int key overwrites (size 1, val 2)");
            assert_eq!(f("hm_struct_hit"), 42, "structurally-equal struct key hits");
            assert_eq!(f("hm_struct_miss"), 0 - 1, "different struct key misses");
        }
    }

    #[test]
    fn hashmap_generic_string_values() {
        init();
        let ctx = Context::create();
        // The point of generics: a HashMap<String, String> (string -> string) works
        // with the SAME hashmap_* defs as HashMap<String, Int>. Also a HashMap<String, Pair> to
        // prove struct values ride too.
        let driver = "
            struct Pair { a: Int, b: Int }
            def chk(r: Result<Int, IndexError>) -> Int =
                match r { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }
            def get_or_str(m: HashMap<String, String>, k: String, d: String) -> Result<String, IndexError> = {
                let o = hashmap_get(m, k)?;
                Result::Ok(opt_unwrap_or(o, d))
            }
            def build_strs() -> Result<HashMap<String, String>, IndexError> = {
                let m1 = hashmap_assoc(hashmap_empty(), \"name\", \"ada\")?;
                let m2 = hashmap_assoc(m1, \"lang\", \"ai-lang\")?;
                hashmap_assoc(m2, \"x\", \"y\")
            }
            def t_lang_len_go() -> Result<Int, IndexError> = {
                let m = build_strs()?;
                let s = get_or_str(m, \"lang\", \"?\")?;
                Result::Ok(string_len(s))
            }
            def t_lang_len() -> Int = chk(t_lang_len_go())
            def t_name_len_go() -> Result<Int, IndexError> = {
                let m = build_strs()?;
                let s = get_or_str(m, \"name\", \"?\")?;
                Result::Ok(string_len(s))
            }
            def t_name_len() -> Int = chk(t_name_len_go())
            def t_str_miss_go() -> Result<Int, IndexError> = {
                let m = build_strs()?;
                let s = get_or_str(m, \"nope\", \"MISS\")?;
                Result::Ok(string_len(s))
            }
            def t_str_miss() -> Int = chk(t_str_miss_go())
            def t_str_size_go() -> Result<Int, IndexError> = {
                let m = build_strs()?;
                Result::Ok(hashmap_size(m))
            }
            def t_str_size() -> Int = chk(t_str_size_go())

            def build_pairs() -> Result<HashMap<String, Pair>, IndexError> = {
                let m1 = hashmap_assoc(hashmap_empty(), \"p\", Pair { a: 3, b: 4 })?;
                hashmap_assoc(m1, \"q\", Pair { a: 10, b: 20 })
            }
            def t_pair_sum_go() -> Result<Int, IndexError> = {
                let m = build_pairs()?;
                let o = hashmap_get(m, \"q\")?;
                match o { Option::Some(p) => Result::Ok(p.a + p.b), Option::None => Result::Ok(0 - 1) }
            }
            def t_pair_sum() -> Int = chk(t_pair_sum_go())
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(
                        &names[n],
                    ))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("t_lang_len"), 7, "HashMap<String, String> value \"ai-lang\"");
            assert_eq!(f("t_name_len"), 3, "HashMap<String, String> value \"ada\"");
            assert_eq!(f("t_str_miss"), 4, "missing key returns default \"MISS\"");
            assert_eq!(f("t_str_size"), 3);
            assert_eq!(f("t_pair_sum"), 30, "HashMap<String, Pair> struct value (10+20)");
        }
    }

    #[test]
    fn hashmap_remove_fold_keys() {
        init();
        let ctx = Context::create();
        let driver = "
            def chk(r: Result<Int, IndexError>) -> Int =
                match r { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }
            def build4() -> Result<HashMap<String, Int>, IndexError> = {
                let m1 = hashmap_assoc(hashmap_empty(), \"a\", 1)?;
                let m2 = hashmap_assoc(m1, \"b\", 2)?;
                let m3 = hashmap_assoc(m2, \"c\", 3)?;
                hashmap_assoc(m3, \"d\", 4)
            }

            // remove: size shrinks, key gone, others survive, original intact
            def t_rm_size_go() -> Result<Int, IndexError> = {
                let m = build4()?;
                let m2 = hashmap_remove(m, \"b\")?;
                Result::Ok(hashmap_size(m2))
            }
            def t_rm_size() -> Int = chk(t_rm_size_go())
            def t_rm_gone_go() -> Result<Int, IndexError> = {
                let m = build4()?;
                let m2 = hashmap_remove(m, \"b\")?;
                let o = hashmap_get(m2, \"b\")?;
                match o { Option::Some(v) => Result::Ok(0 - 100), Option::None => Result::Ok(0) }
            }
            def t_rm_gone() -> Int = chk(t_rm_gone_go())
            def t_rm_survivors_go() -> Result<Int, IndexError> = {
                let m = build4()?;
                let m2 = hashmap_remove(m, \"b\")?;
                let oa = hashmap_get(m2, \"a\")?;
                let od = hashmap_get(m2, \"d\")?;
                match oa {
                    Option::Some(va) => match od { Option::Some(vd) => Result::Ok(va + vd), Option::None => Result::Ok(0 - 1) },
                    Option::None => Result::Ok(0 - 2)
                }
            }
            def t_rm_survivors() -> Int = chk(t_rm_survivors_go())
            def t_rm_immutable_go() -> Result<Int, IndexError> = {
                let m = build4()?;
                let _m2 = hashmap_remove(m, \"b\")?;
                let o = hashmap_get(m, \"b\")?;
                Result::Ok(opt_unwrap_or(o, 0 - 9))
            }
            def t_rm_immutable() -> Int = chk(t_rm_immutable_go())
            def t_rm_missing_size_go() -> Result<Int, IndexError> = {
                let m = build4()?;
                let m2 = hashmap_remove(m, \"zzz\")?;
                Result::Ok(hashmap_size(m2))
            }
            def t_rm_missing_size() -> Int = chk(t_rm_missing_size_go())
            def t_rm_then_readd_go() -> Result<Int, IndexError> = {
                let m = build4()?;
                let m1 = hashmap_remove(m, \"b\")?;
                let m2 = hashmap_assoc(m1, \"b\", 99)?;
                let o = hashmap_get(m2, \"b\")?;
                Result::Ok(opt_unwrap_or(o, 0 - 1))
            }
            def t_rm_then_readd() -> Int = chk(t_rm_then_readd_go())

            // contains
            def t_has_go() -> Result<Int, IndexError> = { let m = build4()?; hashmap_contains(m, \"c\") }
            def t_has() -> Int = chk(t_has_go())
            def t_hasnt_go() -> Result<Int, IndexError> = { let m = build4()?; hashmap_contains(m, \"nope\") }
            def t_hasnt() -> Int = chk(t_hasnt_go())

            // fold: sum of all values
            def t_fold_sum_go() -> Result<Int, IndexError> = {
                let m = build4()?;
                hashmap_fold(m, 0, |acc: Int, k: String, v: Int| acc + v)
            }
            def t_fold_sum() -> Int = chk(t_fold_sum_go())
            // fold: total key-length (string V-agnostic over keys)
            def t_fold_keylen_go() -> Result<Int, IndexError> = {
                let m = build4()?;
                hashmap_fold(m, 0, |acc: Int, k: String, v: Int| acc + string_len(k))
            }
            def t_fold_keylen() -> Int = chk(t_fold_keylen_go())

            // keys: count via list length
            def t_key_count_go() -> Result<Int, IndexError> = {
                let m = build4()?;
                let ks = hashmap_keys(m)?;
                Result::Ok(list_length(ks))
            }
            def t_key_count() -> Int = chk(t_key_count_go())
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(
                        &names[n],
                    ))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("t_rm_size"), 3, "remove shrinks size 4->3");
            assert_eq!(f("t_rm_gone"), 0, "removed key absent");
            assert_eq!(f("t_rm_survivors"), 5, "a(1)+d(4) survive");
            assert_eq!(f("t_rm_immutable"), 2, "remove returns new map; original keeps b");
            assert_eq!(f("t_rm_missing_size"), 4, "removing absent key keeps size");
            assert_eq!(f("t_rm_then_readd"), 99, "can re-add after remove");
            assert_eq!(f("t_has"), 1);
            assert_eq!(f("t_hasnt"), 0);
            assert_eq!(f("t_fold_sum"), 10, "1+2+3+4");
            assert_eq!(f("t_fold_keylen"), 4, "four 1-char keys");
            assert_eq!(f("t_key_count"), 4, "hashmap_keys returns all 4");
        }
    }

    /// Node-resident `state`: a top-level singleton `Atom` that handler
    /// `def`s close over. The installer runs once at JIT startup; separate
    /// handler calls share the one live cell (not per-call copies). This is
    /// the local half of the remote-handler model.
    #[test]
    fn node_state_shared_singleton() {
        init();
        let ctx = Context::create();
        let driver = "
            state ncounter: Atom<Int> = atom(0)
            def ns_bump(d: Int) -> Int = swap(ncounter, |n: Int| n + d)
            def ns_read() -> Int = deref(ncounter)
            // A second handler proves DIFFERENT defs reach the SAME cell.
            def ns_reset_to(v: Int) -> Int = reset(ncounter, v)

            // A PMap-valued state, to show it's not Int-specific. Driven
            // entirely inside ail so the test calls a plain `() -> Int`.
            state nstore: Atom<HashMap<String, Int>> = atom(hashmap_empty())
            // Err keeps the old map: the later ns_get then misses (-1) and
            // the Rust assertion fails loudly.
            def ns_assoc(m: HashMap<String, Int>, k: String, v: Int) -> HashMap<String, Int> =
                match hashmap_assoc(m, k, v) { Result::Ok(m2) => m2, Result::Err(_) => m }
            def ns_put(k: String, v: Int) -> Int = {
                let _s = swap(nstore, |m: HashMap<String, Int>| ns_assoc(m, k, v));
                v
            }
            def ns_get(k: String) -> Int =
                match hashmap_get(deref(nstore), k) {
                    Result::Ok(o) => opt_unwrap_or(o, 0 - 1),
                    Result::Err(_) => 0 - 999
                }
            def ns_t_store() -> Int = {
                let _a = ns_put(\"x\", 41);
                let _b = ns_put(\"y\", 1);
                ns_get(\"x\") + ns_get(\"y\")    // 42, both from the shared store
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f0 = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            let f1 = |n: &str, a: i64| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr(), a)
            };
            // Shared counter: two bumps + a read see the accumulated state.
            assert_eq!(f1("ns_bump", 5), 5, "first bump returns new value");
            assert_eq!(f1("ns_bump", 10), 15, "second bump sees the first");
            assert_eq!(f0("ns_read"), 15, "read sees the shared cell");
            assert_eq!(f1("ns_reset_to", 100), 100, "a different handler hits the same cell");
            assert_eq!(f0("ns_read"), 100, "reset is visible through read");

            // PMap-valued state: puts/gets share one map across calls.
            assert_eq!(f0("ns_t_store"), 42, "PMap state shared across put/get");
        }
    }

    #[test]
    fn atom_basic_deref_swap_reset() {
        init();
        let ctx = Context::create();
        // The real local atom: deref / swap / reset over various value
        // types, all through the SAME generic Atom<T> + lock-free swap.
        let driver = "
            def t_int() -> Int = {
                let a = atom(0);
                let _1 = swap(a, |x: Int| x + 5);
                let _2 = swap(a, |x: Int| x * 3);   // (0+5)*3 = 15
                deref(a)
            }
            def t_reset() -> Int = {
                let a = atom(100);
                let _1 = reset(a, 42);
                deref(a)
            }
            def t_swap_returns_new() -> Int = {
                let a = atom(10);
                swap(a, |x: Int| x + 1)             // returns 11
            }
            def empty_int_map() -> HashMap<String, Int> = hashmap_empty()
            // Err keeps the old map, so the final lookup misses (-1) and the
            // Rust assertion fails loudly.
            def assoc_or_self(m: HashMap<String, Int>, k: String, v: Int) -> HashMap<String, Int> =
                match hashmap_assoc(m, k, v) { Result::Ok(m2) => m2, Result::Err(_) => m }
            def t_pmap() -> Int = {
                let a = atom(empty_int_map());
                let _1 = swap(a, |m: HashMap<String, Int>| assoc_or_self(m, \"k\", 7));
                let _2 = swap(a, |m: HashMap<String, Int>| assoc_or_self(m, \"k\", 9));
                let m = deref(a);
                match hashmap_get(m, \"k\") {
                    Result::Ok(o) => opt_unwrap_or(o, 0 - 1),
                    Result::Err(_) => 0 - 999
                }
            }
            def t_string() -> Int = {
                let a = atom(\"hi\");
                let _1 = swap(a, |s: String| string_concat(s, \"!!!\"));
                string_len(deref(a))               // \"hi!!!\" = 5
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("t_int"), 15, "Int atom swaps compose");
            assert_eq!(f("t_reset"), 42, "reset overwrites");
            assert_eq!(f("t_swap_returns_new"), 11, "swap returns the installed value");
            assert_eq!(f("t_pmap"), 9, "PMap atom (reference value) swaps");
            assert_eq!(f("t_string"), 5, "String atom swaps");
        }
    }

    /// End-to-end `spawn`/`join` from ail source: a thunk runs on a fresh
    /// OS thread (its own execution context) and `join` returns its result.
    /// Covers the Int (boxed/unboxed) path, many concurrent threads, and a
    /// non-Int (String) result.
    #[test]
    fn spawn_join_basic() {
        init();
        let ctx = Context::create();
        let driver = "
            def t_basic() -> Int = {
                let h = spawn(|| 21 + 21);
                join(h)
            }
            def t_many() -> Int = {
                let h1 = spawn(|| 10);
                let h2 = spawn(|| 20);
                let h3 = spawn(|| 30);
                join(h1) + join(h2) + join(h3)
            }
            def t_string() -> Int = {
                let h = spawn(|| \"hello\");
                string_len(join(h))
            }
            // Thunk body is a builtin CALL (string_concat) whose return
            // type must flow through the lambda + spawn + join so that
            // string_len sees a String. Exercises the infer_type fix.
            def t_string_call() -> Int = {
                let h = spawn(|| string_concat(\"ab\", \"cde\"));
                string_len(join(h))
            }
            def t_capture() -> Int = {
                let base = 100;
                let h = spawn(|| base + 7);
                join(h)
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("t_basic"), 42, "spawn/join an Int thunk");
            assert_eq!(f("t_many"), 60, "join three concurrent threads");
            assert_eq!(f("t_string"), 5, "spawn/join a String-returning thunk");
            assert_eq!(
                f("t_string_call"),
                5,
                "thunk returning a builtin-call result (String) flows through spawn/join"
            );
            assert_eq!(f("t_capture"), 107, "spawn closure captures an outer local");
        }
    }

    /// The share-nothing guarantee, end to end: a `spawn`ed closure that
    /// mutates a captured Array mutates its OWN deep-copy — the parent's
    /// array is untouched. `spawn_shared` opts out: the same write IS
    /// visible to the parent.
    #[test]
    fn spawn_isolates_mutable_captures() {
        init();
        let ctx = Context::create();
        let driver = "
            def chk(r: Result<Int, IndexError>) -> Int =
                match r { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }
            def set0(b: Bytes, v: Int) -> Int =
                match bytes_set(b, 0, v) { Result::Ok(x) => x, Result::Err(_) => 0 - 999 }
            def isolated_go() -> Result<Int, IndexError> = {
                let b = bytes_new(1);
                let _z = bytes_set(b, 0, 7)?;
                let h = spawn(|| set0(b, 99));
                let _j = join(h);
                bytes_get(b, 0)
            }
            def isolated() -> Int = chk(isolated_go())
            def shared_go() -> Result<Int, IndexError> = {
                let b = bytes_new(1);
                let _z = bytes_set(b, 0, 7)?;
                let h = spawn_shared(|| set0(b, 99));
                let _j = join(h);
                bytes_get(b, 0)
            }
            def shared() -> Int = chk(shared_go())
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(
                f("isolated"),
                7,
                "default spawn deep-copies the captured array; parent is unaffected"
            );
            assert_eq!(
                f("shared"),
                99,
                "spawn_shared shares the array; the worker's write is visible"
            );
        }
    }

    /// A `String` is immutable, so a `spawn`ed closure capturing one shares
    /// it (the deep-copy returns the same pointer) — no copy, and it reads
    /// back correctly on the worker. (`Bytes`, being mutable, IS copied —
    /// see `spawn_isolates_mutable_captures`.)
    #[test]
    fn spawn_shares_immutable_string_capture() {
        init();
        let ctx = Context::create();
        let driver = "
            def strcap() -> Int = {
                let s = \"hello world\";
                let h = spawn(|| string_len(s));
                join(h)
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names["strcap"]))
                .unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 11, "captured String reads back on the worker");
        }
    }

    /// Concurrency stress test: repeatedly fan out workers that each
    /// allocate heavily, forcing collections WHILE multiple workers run —
    /// exercising per-thread GC contexts, the deep-copy on spawn, join, and
    /// stop-the-world coordination under real contention. Correct sums
    /// every iteration mean no lost/relocated roots and no races.
    #[test]
    fn spawn_concurrency_stress() {
        init();
        let ctx = Context::create();
        let driver = "
            // Allocate an Atom each step (immediately garbage) to churn the
            // heap; tail-recursive so it loops rather than growing the stack.
            def churn(n: Int, acc: Int) -> Int =
                if n == 0 { acc } else { let _g = atom(n); churn(n - 1, acc + n) }
            def worker() -> Int = churn(4000, 0)
            def fan() -> Int = {
                let h1 = spawn(|| worker());
                let h2 = spawn(|| worker());
                let h3 = spawn(|| worker());
                let h4 = spawn(|| worker());
                join(h1) + join(h2) + join(h3) + join(h4)
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let fan = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names["fan"]))
                .unwrap();
            // sum 1..4000 = 8_002_000; four workers per fan. Across 60 fans
            // this churns ~30 MB, triggering several stop-the-world
            // collections while the workers run — exercising per-thread GC
            // contexts, spawn's deep-copy, join, and STW coordination.
            let expected = 4 * 8_002_000i64;
            for i in 0..60 {
                assert_eq!(
                    fan.call(rt.thread_ptr()),
                    expected,
                    "fan iteration {i} produced a wrong sum (lost root / race?)"
                );
            }
        }
    }

    /// `spawn`/`join` with allocation + a forced stop-the-world collection
    /// while results sit in the registry. Each worker allocates (atoms) and
    /// returns a heap value; the main thread forces a GC (which now stops
    /// all mutators) before joining. Proves the registry roots the input
    /// closures and the produced results across relocation.
    #[test]
    fn spawn_join_survives_gc() {
        init();
        let ctx = Context::create();
        let driver = "
            // Allocates `n` atoms, summing 1..n through them.
            def work(n: Int, acc: Int) -> Int =
                if n == 0 { acc }
                else {
                    let a = atom(n);
                    work(n - 1, acc + deref(a))
                }
            def t() -> Int = {
                let h1 = spawn(|| work(50, 0));
                let h2 = spawn(|| work(50, 0));
                let h3 = spawn(|| work(50, 0));
                // Force a STW collection while the workers run / their
                // results sit in the registry, then join.
                let _g = gc_collect();
                join(h1) + join(h2) + join(h3)
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names["t"]))
                .unwrap();
            // sum 1..50 = 1275, three workers = 3825.
            assert_eq!(f.call(rt.thread_ptr()), 3825, "results survive a forced GC");
        }
    }

    /// The dedicated `Atom` shape has exactly one GC-traced pointer
    /// field; a forced collection must scan it so the held value
    /// survives and the cell's slot is rewritten to the relocated copy.
    /// Allocate garbage after the atom to make relocation observable,
    /// then deref/swap across the collect.
    #[test]
    fn atom_value_survives_forced_gc() {
        init();
        let ctx = Context::create();
        let driver = "
            def churn(n: Int) -> Int =
                if n == 0 { 0 } else { let _g = atom(n); churn(n - 1) }
            // Hold a String atom across a GC that relocates everything,
            // then keep mutating it. If the single cell field weren't
            // traced, deref would read stale from-space bits.
            def t_str_gc() -> Int = {
                let a = atom(\"hi\");
                let _c = churn(200);
                let _t = gc_collect();
                let _1 = swap(a, |s: String| string_concat(s, \"!!!\"));
                let _t2 = gc_collect();
                string_len(deref(a))               // \"hi!!!\" = 5
            }
            // Same for a PMap (deep reference value) carried through GC.
            def empty_im() -> HashMap<String, Int> = hashmap_empty()
            // Err keeps the old map: the final lookup then misses (-1) and
            // the Rust assertion fails loudly.
            def assoc_or_self(m: HashMap<String, Int>, k: String, v: Int) -> HashMap<String, Int> =
                match hashmap_assoc(m, k, v) { Result::Ok(m2) => m2, Result::Err(_) => m }
            def t_hashmap_gc() -> Int = {
                let a = atom(empty_im());
                let _1 = swap(a, |m: HashMap<String, Int>| assoc_or_self(m, \"k\", 41));
                let _c = churn(200);
                let _t = gc_collect();
                let _2 = swap(a, |m: HashMap<String, Int>| assoc_or_self(m, \"k\", 9));
                let m = deref(a);
                match hashmap_get(m, \"k\") {
                    Result::Ok(o) => opt_unwrap_or(o, 0 - 1),
                    Result::Err(_) => 0 - 999
                }
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        let before = rt.heap.collections();
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("t_str_gc"), 5, "String atom value survives GC relocation");
            assert_eq!(f("t_hashmap_gc"), 9, "PMap atom value survives GC relocation");
        }
        assert!(
            rt.heap.collections() > before,
            "gc_collect() should have actually run collections",
        );
    }

    #[test]
    fn named_function_as_first_class_value() {
        init();
        let ctx = Context::create();
        // A bare reference to a named top-level function used as a VALUE
        // (not as a direct-call callee) is eta-expanded into an adapter
        // closure. This is the `swap(counter, inc)` case: passing `inc`
        // where a `fn(T) -> T` is expected.
        let driver = "
            def inc(n: Int) -> Int = n + 1
            def dbl(n: Int) -> Int = n * 2

            // Passed to the generic higher-order list_map (Int -> Int).
            def t_map() -> Int = {
                let xs = List::Cons(ListCell { head: 1, tail:
                          List::Cons(ListCell { head: 2, tail:
                          List::Cons(ListCell { head: 3, tail: List::Nil }) }) });
                list_foldl(list_map(xs, inc), 0, |acc: Int, x: Int| acc + x)
            }

            // Passed by name to atom swap — the original reported case.
            def t_swap_named() -> Int = {
                let a = atom(0);
                let _1 = swap(a, inc);
                let _2 = swap(a, inc);
                let _3 = swap(a, dbl);   // (0+1+1)*2 = 4
                deref(a)
            }

            // A named function used BOTH directly and as a value in the
            // same body still works (direct call path is unaffected).
            def t_mixed() -> Int = {
                let a = atom(inc(10));   // direct call: 11
                let _1 = swap(a, inc);   // value:        12
                deref(a)
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("t_map"), 9, "map inc over [1,2,3] then sum = 2+3+4");
            assert_eq!(f("t_swap_named"), 4, "named fns passed to swap");
            assert_eq!(f("t_mixed"), 12, "named fn used both directly and as a value");
        }
    }

    #[test]
    fn concrete_builtin_as_first_class_value() {
        init();
        let ctx = Context::create();
        // A concrete core builtin (here `string_len`, a `fn(String) -> Int`)
        // used as a bare VALUE — resolved to a `core/*` BuiltinRef and
        // eta-expanded into an adapter closure, so it can flow through a
        // higher-order fn. (Generic builtins like `array_get` also work as
        // values — see `generic_builtin_as_first_class_value`. Only call-
        // site-special ones like `at` stay unusable as bare values.)
        let driver = "
            def t_map_strlen() -> Int = {
                let xs = List::Cons(ListCell { head: \"a\", tail:
                          List::Cons(ListCell { head: \"bcd\", tail:
                          List::Cons(ListCell { head: \"ef\", tail: List::Nil }) }) });
                // map string_len over [\"a\",\"bcd\",\"ef\"] = [1,3,2], sum = 6
                list_foldl(list_map(xs, string_len), 0, |acc: Int, n: Int| acc + n)
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("t_map_strlen"), 6, "builtin string_len passed by value to map");
        }
    }

    #[test]
    fn generic_builtin_as_first_class_value() {
        init();
        let ctx = Context::create();
        // Generic core builtins (`array_len`, `atom_swap`) used as bare
        // VALUES. Their signatures carry a `TypeVar`, so the adapter closure
        // is polymorphic and composes through the uniform closure ABI; the
        // concrete (un)boxing is settled at the call site that pins the
        // TypeVars — for an indirect call that's `compile_indirect_call`'s
        // instantiation-aware unboxing, for a direct call the `TopRef` path.
        // (The CHECKED accessors `array_get`/`array_set` are by design NOT
        // usable as bare values — see the rejection assertion below.)
        let driver = "
            // array_len passed into a monomorphic higher-order fn.
            def len_of(a: Array<Int>, g: fn(Array<Int>) -> Int) -> Int = g(a)
            def t_hof() -> Int = {
                let arr = array_new(2);
                len_of(arr, array_len)
            }
            // array_len let-bound, then called where the array type pins T=Int.
            def len0(a: Array<Int>) -> Int = {
                let g = array_len;
                g(a)
            }
            def t_letbound() -> Int = {
                let arr = array_new(1);
                len0(arr)
            }
            // atom_swap passed by value (the lock-free CAS primitive).
            def do_swap(a: Atom<Int>, f: fn(Atom<Int>, fn(Int) -> Int) -> Int) -> Int =
                f(a, |x: Int| x + 100)
            def t_atom() -> Int = {
                let a = atom(5);
                do_swap(a, atom_swap)
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let f = |n: &str| {
                jit.engine
                    .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names[n]))
                    .unwrap()
                    .call(rt.thread_ptr())
            };
            assert_eq!(f("t_hof"), 2, "array_len passed by value to a monomorphic HOF");
            assert_eq!(f("t_letbound"), 1, "let-bound array_len called where T pins to Int");
            assert_eq!(f("t_atom"), 105, "atom_swap passed by value (5 + 100)");
        }

        // The checked accessors are NOT bare values: the resolver rejects
        // them with a wrap-it-in-a-lambda suggestion rather than silently
        // producing an unchecked function value.
        let combined = format!(
            "{}\n{}",
            SOURCE, "def bad() -> Int = { let g = array_get; 0 }"
        );
        let m = parse_module(&combined).expect("parse");
        let err = resolve_module(&m);
        assert!(
            err.is_err(),
            "expected bare-value `array_get` to be rejected at resolve time"
        );
    }

    #[test]
    fn atom_enum_op_local() {
        init();
        let ctx = Context::create();
        // The stdlib Atom<T> (local atomic ref) + a typechecked enum
        // "protocol" + `op` as a plain match function. No bytes, no net.
        let driver = "
            enum Op { OGet(String), OPut(Pair), OBump(String) }
            struct Pair { pk: String, pv: Int }
            def hm_get_or(m: HashMap<String, Int>, k: String, d: Int) -> Int =
                match hashmap_get(m, k) {
                    Result::Ok(o) => opt_unwrap_or(o, d),
                    Result::Err(_) => 0 - 999
                }
            // Err keeps the old map: later OGets then miss and the packed
            // result diverges from the Rust assertion.
            def hm_assoc_or_self(m: HashMap<String, Int>, k: String, v: Int) -> HashMap<String, Int> =
                match hashmap_assoc(m, k, v) { Result::Ok(m2) => m2, Result::Err(_) => m }
            def op(store: Atom<HashMap<String, Int>>, msg: Op) -> Int =
                match msg {
                    Op::OGet(k) => hm_get_or(deref(store), k, 0 - 1),
                    Op::OPut(p) => { let _s = swap(store, |m: HashMap<String, Int>| hm_assoc_or_self(m, p.pk, p.pv)); p.pv },
                    Op::OBump(k) => {
                        let cur = hm_get_or(deref(store), k, 0);
                        let _s = swap(store, |m: HashMap<String, Int>| hm_assoc_or_self(m, k, cur + 1));
                        cur + 1
                    },
                }
            def demo() -> Int = {
                let store = atom(hashmap_empty());
                let _1 = op(store, Op::OPut(Pair { pk: \"x\", pv: 10 }));
                let _2 = op(store, Op::OPut(Pair { pk: \"y\", pv: 20 }));
                let _3 = op(store, Op::OBump(\"x\"));
                let gx = op(store, Op::OGet(\"x\"));
                let gy = op(store, Op::OGet(\"y\"));
                let gz = op(store, Op::OGet(\"missing\"));
                gx * 100 + gy + (gz + 1)
            }
        ";
        let (rt, jit, names) = build_with_stdlib(&ctx, driver);
        unsafe {
            let demo = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(&names["demo"]))
                .unwrap();
            assert_eq!(demo.call(rt.thread_ptr()), 1120);
        }
    }

    #[test]
    fn bit_ops_work() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_and(a: Int, b: Int) -> Int = bit_and(a, b)
             def t_or(a: Int, b: Int) -> Int = bit_or(a, b)
             def t_xor(a: Int, b: Int) -> Int = bit_xor(a, b)
             def t_shl(a: Int, b: Int) -> Int = bit_shl(a, b)
             def t_shr(a: Int, b: Int) -> Int = bit_shr(a, b)
             def t_pop(a: Int) -> Int = popcount(a)",
        );
        unsafe {
            let f2 = |n: &str, a: i64, b: i64| {
                jit.get_fn2(&names[n]).unwrap().call(rt.thread_ptr(), a, b)
            };
            let f1 =
                |n: &str, a: i64| jit.get_fn1(&names[n]).unwrap().call(rt.thread_ptr(), a);
            assert_eq!(f2("t_and", 6, 3), 2);
            assert_eq!(f2("t_or", 1, 2), 3);
            assert_eq!(f2("t_xor", 5, 3), 6);
            assert_eq!(f2("t_shl", 1, 5), 32);
            assert_eq!(f2("t_shr", 256, 4), 16);
            assert_eq!(f1("t_pop", 255), 8);
            assert_eq!(f1("t_pop", 0), 0);
        }
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
            "def t_some_is_some() -> Int = opt_is_some(Option::Some(42))
             def t_none_is_some() -> Int = opt_is_some(Option::None)
             def t_unwrap_some() -> Int = opt_unwrap_or(Option::Some(7), 99)
             def t_unwrap_none() -> Int = opt_unwrap_or(Option::None, 99)",
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
                 IntList::ICons(IntListCell { head: 10,
                   tail: IntList::ICons(IntListCell { head: 20,
                     tail: IntList::ICons(IntListCell { head: 30, tail: IntList::INil }) }) })
             def t_len() -> Int = intlist_length(make_3())
             def t_sum() -> Int = intlist_sum(make_3())
             def t_empty_len() -> Int = intlist_length(IntList::INil)",
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
                 List::Cons(ListCell { head: 10,
                   tail: List::Cons(ListCell { head: 20,
                     tail: List::Cons(ListCell { head: 30, tail: List::Nil }) }) })
             def t_len() -> Int = list_length(make_3())
             def t_sum() -> Int = list_sum(make_3())
             def t_empty_len() -> Int = list_length(List::Nil)",
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
            "def t_some_or_some() -> Int = opt_unwrap_or(opt_or(Option::Some(1), Option::Some(2)), 99)
             def t_none_or_some() -> Int = opt_unwrap_or(opt_or(Option::None, Option::Some(7)), 99)
             def t_none_or_none() -> Int = opt_unwrap_or(opt_or(Option::None, Option::None), 99)",
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
                 List::Cons(ListCell { head: 3,
                   tail: List::Cons(ListCell { head: 1,
                     tail: List::Cons(ListCell { head: 4,
                       tail: List::Cons(ListCell { head: 1,
                         tail: List::Cons(ListCell { head: 5, tail: List::Nil }) }) }) }) })
             def t_max() -> Int = opt_unwrap_or(list_max_int(mk5()), -1)
             def t_min() -> Int = opt_unwrap_or(list_min_int(mk5()), -1)
             def t_max_empty() -> Int = opt_unwrap_or(list_max_int(List::Nil), -42)
             def t_min_empty() -> Int = opt_unwrap_or(list_min_int(List::Nil), -42)",
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
                 opt_unwrap_or(opt_map(Option::Some(21), |x: Int| x * 2), -1)
             def t_none() -> Int =
                 opt_unwrap_or(opt_map(Option::None, |x: Int| x * 2), 99)",
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
                 list_sum(list_append_int(List::Nil, int_list_range(1, 5)))
             def t_append_empty_right() -> Int =
                 list_sum(list_append_int(int_list_range(1, 5), List::Nil))",
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

    // Regression: a GC-pointer local introduced *inside* an `if` branch
    // (here, the `GSome(p)` match payload in the `else`) must get a frame
    // root slot. `count_gc_locals` originally skipped `if` branches, so
    // the slot was unallocated and the payload write ran past the roots
    // array — corrupting the value into a bad pointer. This is the exact
    // shape of the StringMap probe functions.
    #[test]
    fn gc_local_inside_if_branch_is_rooted() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "struct GPair<V> { gk: String, gv: V }
             enum GWrap<V> { GNone, GSome(GPair<V>) }
             def fill_one<V>(a: Array<GWrap<V>>, w: GWrap<V>) -> Result<Int, IndexError> =
                array_set(a, 0, w)
             def find_rec<V>(a: Array<GWrap<V>>, i: Int, n: Int) -> Result<Option<V>, IndexError> =
                if i >= n { Result::Ok(Option::None) }
                else {
                    match array_get(a, i)? {
                        GWrap::GNone => find_rec(a, i + 1, n),
                        GWrap::GSome(p) => Result::Ok(Option::Some(p.gv)),
                    }
                }
             def t_go() -> Result<Int, IndexError> = {
                let a = array_new(2);
                let _x = fill_one(a, GWrap::GSome(GPair { gk: \"x\", gv: 5 }))?;
                let o = find_rec(a, 0, 2)?;
                Result::Ok(opt_unwrap_or(o, 0))
             }
             def t() -> Int =
                match t_go() { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }",
        );
        unsafe {
            let f = jit.get_fn0(&names["t"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 5);
        }
    }

    // ---- StringMap ----

    #[test]
    fn stringmap_insert_get_update_miss() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_go() -> Result<Int, IndexError> = {
                let m0 = smap_new()?;
                let m1 = smap_insert(m0, \"alpha\", 10)?;
                let m2 = smap_insert(m1, \"beta\", 20)?;
                let m3 = smap_insert(m2, \"alpha\", 99)?;
                let a = opt_unwrap_or(smap_get(m3, \"alpha\")?, 0);
                let b = opt_unwrap_or(smap_get(m3, \"beta\")?, 0);
                let miss = opt_unwrap_or(smap_get(m3, \"gamma\")?, 0);
                Result::Ok(a * 1000 + b * 10 + miss + smap_size(m3) * 100000)
             }
             def t() -> Int =
                match t_go() { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }",
        );
        unsafe {
            let f = jit.get_fn0(&names["t"]).unwrap();
            // a=99, b=20, miss=0, size=2 → 99000 + 200 + 0 + 200000
            assert_eq!(f.call(rt.thread_ptr()), 299200);
        }
    }

    #[test]
    fn stringmap_resize_preserves_all_entries() {
        init();
        let ctx = Context::create();
        // key_of(i) builds a distinct single-char key (pure stdlib, no
        // externs). Insert i*i under 26 keys 'A'..'Z', forcing several
        // resizes, then sum all looked-up values back.
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def key_of(i: Int) -> Result<String, IndexError> = {
                let b = bytes_new(1);
                let _x = bytes_set(b, 0, 65 + i)?;
                Result::Ok(string_from_bytes(b))
             }
             def build(m: StringMap<Int>, i: Int, n: Int) -> Result<StringMap<Int>, IndexError> =
                if i >= n { Result::Ok(m) }
                else {
                    let k = key_of(i)?;
                    let m2 = smap_insert(m, k, i * i)?;
                    build(m2, i + 1, n)
                }
             def check(m: StringMap<Int>, i: Int, n: Int, acc: Int) -> Result<Int, IndexError> =
                if i >= n { Result::Ok(acc) }
                else {
                    let k = key_of(i)?;
                    let v = opt_unwrap_or(smap_get(m, k)?, 0 - 999);
                    check(m, i + 1, n, acc + v)
                }
             def t_go() -> Result<Int, IndexError> = {
                let m0 = smap_new()?;
                let m = build(m0, 0, 26)?;
                let c = check(m, 0, 26, 0)?;
                Result::Ok(c + smap_size(m) * 1000000)
             }
             def t() -> Int =
                match t_go() { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }",
        );
        unsafe {
            let f = jit.get_fn0(&names["t"]).unwrap();
            // sum of i*i for i in 0..25 = 5525; size 26 → 5525 + 26_000_000
            let expected: i64 = (0..26).map(|i| i * i).sum::<i64>() + 26_000_000;
            assert_eq!(f.call(rt.thread_ptr()), expected);
        }
    }

    // ---- Float in generics (boxed scalar) ----

    #[test]
    fn float_through_option_and_array() {
        init();
        let ctx = Context::create();
        // Float crossing TypeVar boundaries: Option<Float> via opt_unwrap_or,
        // and Array<Float> via get/set. Both box the f64 bits into a
        // BoxedInt and unbox on the way out.
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def make() -> Array<Float> = array_new(3)
             def opt_test() -> Int =
                float_to_int(opt_unwrap_or(Option::Some(2.5), 0.0) * 4.0)
             def arr_go() -> Result<Int, IndexError> = {
                let a = make();
                let _x = array_set(a, 0, 1.25)?;
                let _y = array_set(a, 1, 3.75)?;
                let v0 = array_get(a, 0)?;
                let v1 = array_get(a, 1)?;
                Result::Ok(float_to_int((v0 + v1) * 2.0))
             }
             def arr_test() -> Int =
                match arr_go() { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }",
        );
        unsafe {
            // Some(2.5) -> 2.5 * 4 = 10
            assert_eq!(jit.get_fn0(&names["opt_test"]).unwrap().call(rt.thread_ptr()), 10);
            // (1.25 + 3.75) * 2 = 10
            assert_eq!(jit.get_fn0(&names["arr_test"]).unwrap().call(rt.thread_ptr()), 10);
        }
    }

    // ---- IntMap ----

    #[test]
    fn intmap_insert_get_update_miss() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_go() -> Result<Int, IndexError> = {
                let m0 = imap_new()?;
                let m1 = imap_insert(m0, 100, 10)?;
                let m2 = imap_insert(m1, 200, 20)?;
                let m3 = imap_insert(m2, 100, 99)?;
                let a = opt_unwrap_or(imap_get(m3, 100)?, 0);
                let b = opt_unwrap_or(imap_get(m3, 200)?, 0);
                let miss = opt_unwrap_or(imap_get(m3, 300)?, 0);
                Result::Ok(a * 1000 + b * 10 + miss + imap_size(m3) * 100000)
             }
             def t() -> Int =
                match t_go() { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }",
        );
        unsafe {
            let f = jit.get_fn0(&names["t"]).unwrap();
            // a=99, b=20, miss=0, size=2 → 99000 + 200 + 0 + 200000
            assert_eq!(f.call(rt.thread_ptr()), 299200);
        }
    }

    #[test]
    fn intmap_resize_and_collisions() {
        init();
        let ctx = Context::create();
        // Insert i -> i*2 for 0..40 (forces several resizes; keys that
        // collide mod small caps exercise probing), then sum gets back.
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def build(m: IntMap<Int>, i: Int, n: Int) -> Result<IntMap<Int>, IndexError> =
                if i >= n { Result::Ok(m) }
                else {
                    let m2 = imap_insert(m, i, i * 2)?;
                    build(m2, i + 1, n)
                }
             def check(m: IntMap<Int>, i: Int, n: Int, acc: Int) -> Result<Int, IndexError> =
                if i >= n { Result::Ok(acc) }
                else {
                    let v = opt_unwrap_or(imap_get(m, i)?, 0 - 999);
                    check(m, i + 1, n, acc + v)
                }
             def t_go() -> Result<Int, IndexError> = {
                let m0 = imap_new()?;
                let m = build(m0, 0, 40)?;
                let c = check(m, 0, 40, 0)?;
                Result::Ok(c + imap_size(m) * 1000000)
             }
             def t() -> Int =
                match t_go() { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }",
        );
        unsafe {
            let f = jit.get_fn0(&names["t"]).unwrap();
            // sum of i*2 for 0..39 = 2 * (39*40/2) = 1560; size 40
            let expected: i64 = (0..40).map(|i: i64| i * 2).sum::<i64>() + 40_000_000;
            assert_eq!(f.call(rt.thread_ptr()), expected);
        }
    }

    // ---- checked indexing: Result-returning array/bytes accessors ----

    /// Fused `array_get(..)?` / `array_set(..)?`: the happy path unwraps
    /// raw values; an out-of-bounds access early-returns
    /// `Err(IndexError::OutOfBounds(OobInfo { index, len }))`.
    #[test]
    fn checked_indexing_ok_and_oob_paths() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def mk(n: Int) -> Array<Int> = array_new(n)
             def go(i: Int) -> Result<Int, IndexError> = {
                let a = mk(3);
                let _s = array_set(a, 0, 11)?;
                let _t = array_set(a, 1, 22)?;
                Result::Ok(array_get(a, i)?)
             }
             def run(i: Int) -> Int =
                match go(i) {
                    Result::Ok(v) => v,
                    Result::Err(e) => match e {
                        IndexError::OutOfBounds(o) => 0 - (o.index * 1000 + o.len),
                        IndexError::Uninitialized(o) => 0 - (o.index * 1000 + o.len) - 500000,
                    },
                }",
        );
        unsafe {
            let f = jit.get_fn1(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 11);
            assert_eq!(f.call(rt.thread_ptr(), 1), 22);
            // OOB index 9 on len 3 → Err carrying both numbers.
            assert_eq!(f.call(rt.thread_ptr(), 9), -(9 * 1000 + 3));
            // Negative index too.
            assert_eq!(f.call(rt.thread_ptr(), -2), -((-2) * 1000 + 3));
        }
    }

    /// Non-fused checked access: the Result is an ordinary first-class
    /// value a caller can store and match later.
    #[test]
    fn checked_indexing_result_is_a_value() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def mk(n: Int) -> Array<Int> = array_new(n)
             def run(i: Int) -> Int = {
                let a = mk(2);
                let _s = match array_set(a, 0, 7) {
                    Result::Ok(v) => v,
                    Result::Err(_) => 0 - 999,
                };
                let r = array_get(a, i);
                match r {
                    Result::Ok(v) => v,
                    Result::Err(_) => 0 - 1,
                }
             }",
        );
        unsafe {
            let f = jit.get_fn1(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 7);
            assert_eq!(f.call(rt.thread_ptr(), 5), -1);
        }
    }

    /// Checked bytes indexing follows the same protocol.
    #[test]
    fn checked_bytes_indexing() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def go(i: Int) -> Result<Int, IndexError> = {
                let b = bytes_new(4);
                let _s = bytes_set(b, 0, 200)?;
                Result::Ok(bytes_get(b, i)?)
             }
             def run(i: Int) -> Int =
                match go(i) {
                    Result::Ok(v) => v,
                    Result::Err(_) => 0 - 1,
                }",
        );
        unsafe {
            let f = jit.get_fn1(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 200);
            assert_eq!(f.call(rt.thread_ptr(), 1), 0);
            assert_eq!(f.call(rt.thread_ptr(), 99), -1);
        }
    }

    // ---- the `?` operator over Result<T, E> (Tier 1, step 2) ----

    /// `?` unwraps `Ok` and early-returns `Err` through a chain of
    /// `Result`-returning functions. `add_one` propagates with `?`;
    /// `pipeline` chains two `?`s then wraps the result; `run` matches
    /// the final Result. Covers both the Ok path (value flows through)
    /// and the Err path (short-circuits all the way out).
    #[test]
    fn try_operator_propagates_ok_and_err() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def make_ok(x: Int) -> Result<Int, Int> = Result::Ok(x)
             def make_err(e: Int) -> Result<Int, Int> = Result::Err(e)
             def add_one(r: Result<Int, Int>) -> Result<Int, Int> = Result::Ok(r? + 1)
             def pipeline(start: Result<Int, Int>) -> Result<Int, Int> = {
                let a = add_one(start)?;
                let b = add_one(make_ok(a))?;
                Result::Ok(b + 1000)
             }
             def run(x: Int, is_err: Int) -> Int =
                match pipeline(if is_err > 0 { make_err(x) } else { make_ok(x) }) {
                    Result::Ok(v) => v,
                    Result::Err(e) => 0 - e
                }",
        );
        unsafe {
            let f = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64) -> i64>(
                    &def_symbol(&names["run"]),
                )
                .unwrap();
            // Ok path: 5 -> +1 -> +1 -> +1000 = 1007.
            assert_eq!(f.call(rt.thread_ptr(), 5, 0), 1007);
            assert_eq!(f.call(rt.thread_ptr(), 0, 0), 1002);
            // Err path: Err(42) short-circuits; run negates it.
            assert_eq!(f.call(rt.thread_ptr(), 42, 1), -42);
            assert_eq!(f.call(rt.thread_ptr(), 7, 1), -7);
        }
    }

    /// `?` on a `Result` whose Ok payload is a *pointer* type (String):
    /// the payload is loaded directly (no unbox), exercising the
    /// `payload_is_pointer && !needs_unbox` branch of `compile_try`.
    #[test]
    fn try_operator_pointer_ok_payload() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def get_msg(ok: Int) -> Result<String, Int> =
                if ok > 0 { Result::Ok(\"hello\") } else { Result::Err(1) }
             def greet(ok: Int) -> Result<String, Int> = {
                let m = get_msg(ok)?;
                Result::Ok(string_concat(m, \"!\"))
             }
             def run_str(ok: Int) -> Int =
                match greet(ok) {
                    Result::Ok(s) => string_len(s),
                    Result::Err(e) => 0 - e
                }",
        );
        unsafe {
            let f = jit.get_fn1(&names["run_str"]).unwrap();
            // "hello" -> "hello!" -> len 6.
            assert_eq!(f.call(rt.thread_ptr(), 1), 6);
            // Err(1) short-circuits -> -1.
            assert_eq!(f.call(rt.thread_ptr(), 0), -1);
        }
    }

    /// `?` requires the enclosing function to return a `Result` with the
    /// same error type. Using `?` in a plain `-> Int` function is a
    /// typecheck error (not a silent miscompile).
    #[test]
    fn try_in_non_result_fn_is_a_type_error() {
        init();
        let combined = format!(
            "{}\n{}",
            SOURCE,
            "def bad(r: Result<Int, Int>) -> Int = r? + 1"
        );
        let m = parse_module(&combined).expect("parse");
        let r = resolve_module(&m).expect("resolve");
        let mut tc = crate::typecheck::TypeCache::new();
        let err = crate::typecheck::typecheck_module(&r, &mut tc);
        assert!(
            err.is_err(),
            "expected `?` in a non-Result function to be a type error"
        );
    }

    // ---- HTTP FFI bridge (Tier 2) ----

    /// End-to-end test of the HTTP client against a one-shot localhost
    /// server (plain HTTP, so no external network / TLS dependency).
    /// Exercises the whole path: ai-lang `http_get` → generic C FFI →
    /// real libcurl (variadic `curl_easy_setopt`, body captured via
    /// `open_memstream`) → `cstr_to_string` → ai-lang builds
    /// `Result<String, String>` → match. Returns the body length so the
    /// Rust side only has to check an i64. The port is passed in as an
    /// Int and the URL is assembled in ai-lang.
    #[test]
    fn http_get_against_localhost_server() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        init();

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind localhost");
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                // Drain the request headers (we don't parse them).
                let mut buf = [0u8; 2048];
                let _ = sock.read(&mut buf);
                let body = "hello";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
            }
        });

        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def test_http(port: Int) -> Int = {
                let url = string_concat(
                    string_concat(\"http://127.0.0.1:\", int_to_string(port)), \"/\");
                match http_get(url) {
                    Result::Ok(body) => string_len(body),
                    Result::Err(e) => 0 - string_len(e) - 1000
                }
             }",
        );
        let result = unsafe {
            let f = jit.get_fn1(&names["test_http"]).unwrap();
            f.call(rt.thread_ptr(), port as i64)
        };
        let _ = server.join();
        // Body "hello" has length 5; a negative result means the request
        // failed (Err path).
        assert_eq!(result, 5, "expected body 'hello' (len 5), got {}", result);
    }

    /// `http_post` against a localhost server that verifies the request
    /// line is a POST and the request body arrived intact, then echoes a
    /// fixed body back. Confirms method + headers + body marshaling end
    /// to end (no external network / TLS).
    #[test]
    fn http_post_sends_method_and_body() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::mpsc;
        init();

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind localhost");
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel::<String>();
        let server = std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let n = sock.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).into_owned();
                let body = "received";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
                let _ = tx.send(req);
            }
        });

        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def test_post(port: Int) -> Int = {
                let url = string_concat(
                    string_concat(\"http://127.0.0.1:\", int_to_string(port)), \"/submit\");
                match http_post(url, \"X-Test: ai-lang\", \"payload-123\") {
                    Result::Ok(body) => string_len(body),
                    Result::Err(e) => 0 - string_len(e) - 1000
                }
             }",
        );
        let result = unsafe {
            let f = jit.get_fn1(&names["test_post"]).unwrap();
            f.call(rt.thread_ptr(), port as i64)
        };
        let _ = server.join();
        let req = rx.recv().expect("server captured the request");

        // Response body "received" -> length 8.
        assert_eq!(result, 8, "expected echoed body len 8, got {}", result);
        // The server actually saw a POST carrying our header + body.
        assert!(req.starts_with("POST /submit"), "request line was: {:?}", req.lines().next());
        assert!(req.contains("X-Test: ai-lang"), "custom header missing in: {}", req);
        assert!(req.contains("payload-123"), "request body missing in: {}", req);
    }

    // ---- JSON FFI bridge (Tier 2) ----

    /// Parse a JSON document and read it back by path: nested object
    /// keys, an array length, an integer field, and a string field.
    /// Packs the four results into one Int so the Rust side checks a
    /// single value. Deterministic (no network).
    #[test]
    fn json_parse_and_path_access() {
        init();
        let ctx = Context::create();
        // doc = {"name":"ai-lang","meta":{"version":2},"tags":["a","b","c"]}
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def int_at(j: Json, path: String) -> Int =
                match json_get(j, path) {
                    Result::Ok(o) => match o {
                        Option::Some(v) => match json_int(v) { Option::Some(n) => n, Option::None => 0 - 1 },
                        Option::None => 0 - 1
                    },
                    Result::Err(_) => 0 - 999
                }
             def len_at(j: Json, path: String) -> Int =
                match json_get(j, path) {
                    Result::Ok(o) => match o { Option::Some(v) => json_len(v), Option::None => 0 - 1 },
                    Result::Err(_) => 0 - 999
                }
             def strlen_at(j: Json, path: String) -> Int =
                match json_get(j, path) {
                    Result::Ok(o) => match o {
                        Option::Some(v) => match json_string(v) { Option::Some(s) => string_len(s), Option::None => 0 - 1 },
                        Option::None => 0 - 1
                    },
                    Result::Err(_) => 0 - 999
                }
             def test_json() -> Int =
                match parse_json(\"{\\\"name\\\":\\\"ai-lang\\\",\\\"meta\\\":{\\\"version\\\":2},\\\"tags\\\":[\\\"a\\\",\\\"b\\\",\\\"c\\\"]}\") {
                    Result::Err(e) => 0 - 1,
                    Result::Ok(j) => {
                        let tags = len_at(j, \"tags\");
                        let ver = int_at(j, \"meta.version\");
                        let namelen = strlen_at(j, \"name\");
                        let missing = match json_get(j, \"nope\") {
                            Result::Ok(o) => match o { Option::Some(x) => 1, Option::None => 0 },
                            Result::Err(_) => 0 - 999
                        };
                        tags * 10000 + ver * 1000 + namelen * 10 + missing
                    }
                }
             def tag_one() -> Int =
                match parse_json(\"{\\\"tags\\\":[\\\"a\\\",\\\"bb\\\",\\\"ccc\\\"]}\") {
                    Result::Err(e) => 0 - 1,
                    Result::Ok(j) => strlen_at(j, \"tags.1\")
                }",
        );
        unsafe {
            let f = jit.get_fn0(&names["test_json"]).unwrap();
            // tags=3, ver=2, namelen=7 ("ai-lang"), missing=0
            // -> 30000 + 2000 + 70 + 0 = 32070
            assert_eq!(f.call(rt.thread_ptr()), 32070);
            // tags.1 = "bb" -> length 2 (array-index path navigation).
            assert_eq!(jit.get_fn0(&names["tag_one"]).unwrap().call(rt.thread_ptr()), 2);
        }
    }

    /// JSON string escapes: simple escapes (`\n`, `\t`, `\"`) and a
    /// `\uXXXX` BMP escape decoded to UTF-8. Checks decoded byte lengths
    /// so the escape + UTF-8 encoder paths are actually exercised.
    #[test]
    fn json_string_escapes() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            // {"s":"a\nb\t\"c"}  -> decoded "a<nl>b<tab>\"c" = 6 bytes.
            "def esc_len() -> Int =
                match json_field(\"{\\\"s\\\":\\\"a\\\\nb\\\\t\\\\\\\"c\\\"}\", \"s\") {
                    Result::Ok(v) => string_len(v),
                    Result::Err(e) => 0 - 1
                }
             // {\"s\":\"\\u00e9\"} -> U+00E9 é encodes to 2 UTF-8 bytes.
             def uni_len() -> Int =
                match json_field(\"{\\\"s\\\":\\\"\\\\u00e9\\\"}\", \"s\") {
                    Result::Ok(v) => string_len(v),
                    Result::Err(e) => 0 - 1
                }
             // {\"s\":\"\\u20ac\"} -> U+20AC € encodes to 3 UTF-8 bytes.
             def euro_len() -> Int =
                match json_field(\"{\\\"s\\\":\\\"\\\\u20ac\\\"}\", \"s\") {
                    Result::Ok(v) => string_len(v),
                    Result::Err(e) => 0 - 1
                }",
        );
        unsafe {
            assert_eq!(jit.get_fn0(&names["esc_len"]).unwrap().call(rt.thread_ptr()), 6);
            assert_eq!(jit.get_fn0(&names["uni_len"]).unwrap().call(rt.thread_ptr()), 2);
            assert_eq!(jit.get_fn0(&names["euro_len"]).unwrap().call(rt.thread_ptr()), 3);
        }
    }

    /// `json_field` returns `Ok(value)` for a present path and `Err` for
    /// invalid JSON — exercising the stdlib wrapper that composes the
    /// JSON accessors into a `Result<String, String>`.
    #[test]
    fn json_field_ok_and_error() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def good() -> Int =
                match json_field(\"{\\\"city\\\":\\\"NYC\\\"}\", \"city\") {
                    Result::Ok(v) => string_len(v),
                    Result::Err(e) => 0 - 1
                }
             def bad() -> Int =
                match json_field(\"not json\", \"city\") {
                    Result::Ok(v) => string_len(v),
                    Result::Err(e) => 0 - string_len(e)
                }",
        );
        unsafe {
            // Ok("NYC") -> len 3.
            assert_eq!(jit.get_fn0(&names["good"]).unwrap().call(rt.thread_ptr()), 3);
            // Err("invalid JSON") -> -len("invalid JSON") = -12.
            assert_eq!(jit.get_fn0(&names["bad"]).unwrap().call(rt.thread_ptr()), -12);
        }
    }

    // ---- crypto FFI bridge (Tier 2) ----

    /// SHA-256 and HMAC-SHA256 against published test vectors. The
    /// ai-lang fns compare the hex output to the expected string and
    /// return 1 on a match.
    #[test]
    fn crypto_sha256_and_hmac_vectors() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def test_sha256() -> Int =
                match sha256_hex(\"abc\") {
                    Result::Ok(h) => string_eq(h,
                      \"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\"),
                    Result::Err(_) => 0 - 999
                }
             def test_hmac() -> Int =
                match hmac_sha256_hex(\"key\", \"The quick brown fox jumps over the lazy dog\") {
                    Result::Ok(h) => string_eq(h,
                      \"f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8\"),
                    Result::Err(_) => 0 - 999
                }",
        );
        unsafe {
            assert_eq!(jit.get_fn0(&names["test_sha256"]).unwrap().call(rt.thread_ptr()), 1);
            assert_eq!(jit.get_fn0(&names["test_hmac"]).unwrap().call(rt.thread_ptr()), 1);
        }
    }

    /// The AWS SigV4 signing-key derivation, computed entirely in ai-lang
    /// via the chained `hmac_sha256_raw` + `hex_encode`, checked against
    /// AWS's published test vector. This proves the crypto chain needed
    /// to authenticate to AWS works end to end through the FFI bridge.
    #[test]
    fn aws_sigv4_signing_key_vector() {
        init();
        let ctx = Context::create();
        // AWS docs example: secret "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        // date 20150830, region us-east-1, service iam.
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def sigkey() -> Result<String, IndexError> =
                aws_sigv4_signing_key(
                    \"wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY\",
                    \"20150830\", \"us-east-1\", \"iam\")
             def matches() -> Int =
                match sigkey() {
                    Result::Ok(k) => string_eq(k,
                      \"c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9\"),
                    Result::Err(_) => 0 - 999
                }",
        );
        unsafe {
            assert_eq!(jit.get_fn0(&names["matches"]).unwrap().call(rt.thread_ptr()), 1);
        }
    }

    /// `http_request_many` fires all requests CONCURRENTLY. Proven with a
    /// barrier server: it accepts and reads all N requests before
    /// responding to any. If ai-lang sent them one-at-a-time (waiting for
    /// each response before the next request), the server would block
    /// forever waiting for the rest — so a (fast) pass proves the N
    /// requests were genuinely in flight at once. Each response echoes the
    /// request's index, so we also confirm results come back in order.
    #[test]
    fn http_request_many_is_concurrent() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        init();

        let n: usize = 12;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut conns = Vec::new();
            // BARRIER: read all N requests first.
            for _ in 0..n {
                let (mut sock, _) = listener.accept().unwrap();
                let mut buf = [0u8; 2048];
                let r = sock.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..r]).into_owned();
                let body = req.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
                conns.push((sock, body));
            }
            // Only now respond, each echoing the request's body (its index).
            for (mut sock, body) in conns {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
            }
        });

        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def build_reqs(port: Int, i: Int, n: Int) -> List<HttpReq> =
                if i >= n {
                    List::Nil
                } else {
                    let url = str3(\"http://127.0.0.1:\", int_to_string(port), \"/\");
                    List::Cons(ListCell {
                        head: HttpReq { method: \"POST\", url: url, headers: \"\", body: int_to_string(i) },
                        tail: build_reqs(port, i + 1, n)
                    })
                }
             def check(resps: List<HttpResponse>, i: Int) -> Int =
                match resps {
                    List::Nil => 0,
                    List::Cons(cell) => {
                        let ok = if cell.head.status == 200 {
                            if string_eq(cell.head.body, int_to_string(i)) { 1 } else { 0 }
                        } else { 0 };
                        ok + check(cell.tail, i + 1)
                    }
                }
             def run(port: Int, n: Int) -> Int =
                match http_request_many(build_reqs(port, 0, n)) {
                    Result::Ok(resps) => check(resps, 0),
                    Result::Err(_) => 0 - 999
                }",
        );
        let correct = unsafe {
            jit.engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
                .call(rt.thread_ptr(), port as i64, n as i64)
        };
        server.join().unwrap();
        // All 12 ran at once (barrier passed) and came back in order.
        assert_eq!(correct, n as i64);
    }

    /// Full distributed map-reduce pattern end to end. ai-lang splits a
    /// range into chunks, fans them out as concurrent requests, each
    /// "worker" (the mock) sums its chunk, and ai-lang reduces the
    /// partials. This is exactly `lambda_invoke_many` + reduce, just with
    /// a local server standing in for the Lambda fleet.
    #[test]
    fn distributed_map_reduce_pattern() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        init();

        let total: i64 = 2000;
        let per: i64 = 100; // 20 chunks
        let n = (total / per) as usize;
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut handlers = Vec::new();
            for _ in 0..n {
                let (mut sock, _) = listener.accept().unwrap();
                handlers.push(std::thread::spawn(move || {
                    let mut buf = [0u8; 2048];
                    let r = sock.read(&mut buf).unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..r]).into_owned();
                    let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
                    let fld = |k: &str| -> i64 {
                        let m = format!("\"{}\":", k);
                        let s = &body[body.find(&m).unwrap() + m.len()..];
                        let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
                        s[..end].parse().unwrap()
                    };
                    let (lo, hi) = (fld("lo"), fld("hi"));
                    let sum: i64 = (lo..hi).sum();
                    let out = format!("{{\"sum\":{}}}", sum);
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        out.len(),
                        out
                    );
                    let _ = sock.write_all(resp.as_bytes());
                }));
            }
            for h in handlers {
                let _ = h.join();
            }
        });

        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def chunk_req(port: Int, lo: Int, hi: Int) -> HttpReq = {
                let url = str3(\"http://127.0.0.1:\", int_to_string(port), \"/\");
                HttpReq { method: \"POST\", url: url, headers: \"\",
                          body: str4(\"{\\\"lo\\\":\", int_to_string(lo), \",\\\"hi\\\":\", string_concat(int_to_string(hi), \"}\")) }
             }
             def chunk_reqs(port: Int, start: Int, total: Int, per: Int, acc: List<HttpReq>) -> List<HttpReq> =
                if start >= total {
                    list_reverse(acc)
                } else {
                    let hi = if start + per > total { total } else { start + per };
                    chunk_reqs(port, hi, total, per, List::Cons(ListCell { head: chunk_req(port, start, hi), tail: acc }))
                }
             def partial(resp: HttpResponse) -> Int =
                match parse_json(resp.body) {
                    Result::Ok(j) => match json_get(j, \"sum\") {
                        Result::Ok(o) => match o { Option::Some(v) => match json_int(v) { Option::Some(x) => x, Option::None => 0 }, Option::None => 0 },
                        Result::Err(_) => 0 - 999
                    },
                    Result::Err(e) => 0
                }
             def reduce_sum(resps: List<HttpResponse>, acc: Int) -> Int =
                match resps {
                    List::Nil => acc,
                    List::Cons(cell) => reduce_sum(cell.tail, acc + partial(cell.head))
                }
             def run(port: Int, total: Int, per: Int) -> Int =
                match http_request_many(chunk_reqs(port, 0, total, per, List::Nil)) {
                    Result::Ok(resps) => reduce_sum(resps, 0),
                    Result::Err(_) => 0 - 999
                }",
        );
        let result = unsafe {
            jit.engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64, i64) -> i64>(
                    &crate::codegen::def_symbol(&names["run"]),
                )
                .unwrap()
                .call(rt.thread_ptr(), port as i64, total, per)
        };
        server.join().unwrap();
        // 0 + 1 + ... + 1999 = 1999000, computed across 20 parallel workers.
        assert_eq!(result, 1999000);
    }

    /// URI percent-encoding for S3 canonical paths: unreserved chars and
    /// "/" pass through; everything else is %XX (uppercase).
    #[test]
    fn s3_uri_encoding() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def enc_eq(s: String, want: String) -> Int =
                match uri_encode_path(s) {
                    Result::Ok(e) => string_eq(e, want),
                    Result::Err(_) => 0 - 999
                }
             def t_plain() -> Int = enc_eq(\"ai-lang/test.txt\", \"ai-lang/test.txt\")
             def t_special() -> Int = enc_eq(\"a b/c+d.txt\", \"a%20b/c%2Bd.txt\")",
        );
        unsafe {
            assert_eq!(jit.get_fn0(&names["t_plain"]).unwrap().call(rt.thread_ptr()), 1);
            assert_eq!(jit.get_fn0(&names["t_special"]).unwrap().call(rt.thread_ptr()), 1);
        }
    }

    /// Bitwise ops, base64, and CRC-32 — the foundations for building a
    /// Lambda deployment zip. Checked against reference values.
    #[test]
    fn bitwise_base64_crc32() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def t_and() -> Int = bit_and(12, 10)
             def t_or() -> Int = bit_or(12, 10)
             def t_xor() -> Int = bit_xor(12, 10)
             def t_shl() -> Int = bit_shl(1, 4)
             def t_shr() -> Int = bit_shr(256, 4)
             def b64_eq(s: String, want: String) -> Int =
                match base64_encode(s) {
                    Result::Ok(e) => string_eq(e, want),
                    Result::Err(_) => 0 - 999
                }
             def crc_of(s: String) -> Int =
                match crc32(s) { Result::Ok(v) => v, Result::Err(_) => 0 - 999 }
             def t_b64_foobar() -> Int = b64_eq(\"foobar\", \"Zm9vYmFy\")
             def t_b64_foob() -> Int = b64_eq(\"foob\", \"Zm9vYg==\")
             def t_b64_fooba() -> Int = b64_eq(\"fooba\", \"Zm9vYmE=\")
             def t_b64_empty() -> Int = b64_eq(\"\", \"\")
             def t_crc_foobar() -> Int = crc_of(\"foobar\")
             def t_crc_foo() -> Int = crc_of(\"foo\")
             def t_crc_empty() -> Int = crc_of(\"\")",
        );
        unsafe {
            let g = |n: &str| jit.get_fn0(&names[n]).unwrap().call(rt.thread_ptr());
            assert_eq!(g("t_and"), 8);
            assert_eq!(g("t_or"), 14);
            assert_eq!(g("t_xor"), 6);
            assert_eq!(g("t_shl"), 16);
            assert_eq!(g("t_shr"), 16);
            assert_eq!(g("t_b64_foobar"), 1);
            assert_eq!(g("t_b64_foob"), 1);
            assert_eq!(g("t_b64_fooba"), 1);
            assert_eq!(g("t_b64_empty"), 1);
            assert_eq!(g("t_crc_foobar"), 2666930069);
            assert_eq!(g("t_crc_foo"), 2356372769);
            assert_eq!(g("t_crc_empty"), 0);
        }
    }

    /// Program A provisions Program B: an ai-lang program builds a Lambda
    /// deployment zip, base64-encodes it into a CreateFunction request,
    /// SigV4-signs it, and POSTs it. The mock service receives the signed
    /// request; we then pull the base64 `ZipFile` out of the body, decode
    /// it, and unzip it — proving the whole deployment-package pipeline
    /// (zip → base64 → JSON → SigV4 → HTTP) produced a real, valid
    /// archive carrying the bootstrap.
    #[test]
    fn program_a_creates_function_with_real_zip() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::mpsc;
        init();

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel::<String>();
        let server = std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = vec![0u8; 65536];
                let n = sock.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).into_owned();
                let body = "{\"FunctionArn\":\"arn:aws:lambda:us-east-1:0:function:b\"}";
                let resp = format!(
                    "HTTP/1.1 201 Created\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
                let _ = tx.send(req);
            }
        });

        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def run(port: Int) -> Int = {
                let p = int_to_string(port);
                let host = string_concat(\"127.0.0.1:\", p);
                let path = \"/2015-03-31/functions\";
                let url = str3(\"http://\", host, path);
                match zip_one(\"bootstrap\", \"#!/bin/sh\\nexec ai-lang run main\\n\") {
                    Result::Err(_) => 0 - 999,
                    Result::Ok(zip) =>
                        match lambda_create_at(url, host, path, \"us-east-1\",
                                \"prog-b\", \"arn:aws:iam::0:role/r\", zip,
                                \"AKIDEXAMPLE\", \"SECRETKEY\", \"\") {
                            Result::Ok(resp) => resp.status,
                            Result::Err(e) => 0 - 1
                        }
                }
             }",
        );
        let status =
            unsafe { jit.get_fn1(&names["run"]).unwrap().call(rt.thread_ptr(), port as i64) };
        let _ = server.join();
        let req = rx.recv().expect("service received CreateFunction");

        assert_eq!(status, 201);
        assert!(
            req.starts_with("POST /2015-03-31/functions")
                && req.contains("Authorization: AWS4-HMAC-SHA256")
                && req.contains("\"Runtime\":\"provided.al2\"")
                && req.contains("\"FunctionName\":\"prog-b\""),
            "CreateFunction request malformed:\n{}",
            req.chars().take(600).collect::<String>()
        );

        // Pull the base64 ZipFile out of the JSON body, decode + unzip it.
        let body = req.split("\r\n\r\n").nth(1).expect("request body");
        let marker = "\"ZipFile\":\"";
        let start = body.find(marker).expect("ZipFile field") + marker.len();
        let b64 = &body[start..body[start..].find('"').map(|i| start + i).unwrap()];
        let zip_path = std::env::temp_dir().join("ai_lang_created_fn.zip");
        let decode = std::process::Command::new("python3")
            .arg("-c")
            .arg("import base64,sys; open(sys.argv[1],'wb').write(base64.b64decode(sys.argv[2]))")
            .arg(&zip_path)
            .arg(b64)
            .output()
            .expect("decode base64");
        assert!(decode.status.success(), "base64 decode failed: {:?}", decode);
        let cat = std::process::Command::new("unzip")
            .arg("-p")
            .arg(&zip_path)
            .arg("bootstrap")
            .output()
            .expect("unzip");
        assert!(cat.status.success(), "unzip failed: {:?}", cat);
        assert_eq!(
            String::from_utf8_lossy(&cat.stdout),
            "#!/bin/sh\nexec ai-lang run main\n"
        );
        let _ = std::fs::remove_file(&zip_path);
    }

    /// A ZIP archive built entirely in ai-lang is a valid archive: we
    /// write it to disk and let the system `unzip` extract it, checking
    /// the file name and contents round-trip. This is the deployment
    /// package machinery (a Lambda custom-runtime `bootstrap`).
    #[test]
    fn zip_archive_is_valid() {
        init();
        let path = std::env::temp_dir().join("ai_lang_zip_test.zip");
        let path_str = path.to_string_lossy().replace('\\', "/");
        let _ = std::fs::remove_file(&path);

        let ctx = Context::create();
        let src = format!(
            "def make_zip() -> Int =
                match zip_one(\"bootstrap\", \"#!/bin/sh\\necho hello-from-ai-lang\\n\") {{
                    Result::Err(_) => 0 - 999,
                    Result::Ok(z) => match fs_write(\"{path}\", z) {{
                        Result::Ok(w) => w,
                        Result::Err(_) => 0 - 998
                    }}
                }}",
            path = path_str
        );
        let (rt, jit, names) = build_with_stdlib(&ctx, &src);
        let wrote = unsafe { jit.get_fn0(&names["make_zip"]).unwrap().call(rt.thread_ptr()) };
        assert_eq!(wrote, 0, "fs_write should succeed");

        // List the archive: it must contain exactly `bootstrap`.
        let list = std::process::Command::new("unzip")
            .arg("-l")
            .arg(&path)
            .output()
            .expect("run unzip -l");
        assert!(list.status.success(), "unzip -l failed: {:?}", list);
        let listing = String::from_utf8_lossy(&list.stdout);
        assert!(listing.contains("bootstrap"), "archive listing: {}", listing);

        // Extract the entry and check its contents byte-for-byte.
        let cat = std::process::Command::new("unzip")
            .arg("-p")
            .arg(&path)
            .arg("bootstrap")
            .output()
            .expect("run unzip -p");
        assert!(cat.status.success(), "unzip -p failed: {:?}", cat);
        assert_eq!(
            String::from_utf8_lossy(&cat.stdout),
            "#!/bin/sh\necho hello-from-ai-lang\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Full SigV4 request signing against AWS's published `get-vanilla`
    /// test vector. The headers are passed UNSORTED (x-amz-date before
    /// host) so the canonicalization sort is exercised. The produced
    /// `Authorization` header must match AWS byte for byte.
    #[test]
    fn aws_sigv4_authorization_get_vanilla() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def authz() -> Result<String, IndexError> =
                sigv4_authorization(
                    \"GET\", \"/\", \"\",
                    RequestHeaders::RHCons(RHCell { head: sig_header(\"x-amz-date\", \"20150830T123600Z\"), tail:
                    RequestHeaders::RHCons(RHCell { head: sig_header(\"host\", \"example.amazonaws.com\"), tail: RequestHeaders::RHNil }) }),
                    \"\",
                    \"AKIDEXAMPLE\", \"wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY\",
                    \"us-east-1\", \"service\",
                    \"20150830T123600Z\", \"20150830\")
             def matches() -> Int =
                match authz() {
                    Result::Ok(a) => string_eq(a,
                      \"AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31\"),
                    Result::Err(_) => 0 - 999
                }",
        );
        unsafe {
            assert_eq!(jit.get_fn0(&names["matches"]).unwrap().call(rt.thread_ptr()), 1);
        }
    }

    /// End-to-end AWS request: `aws_signed_post` builds a real SigV4-
    /// signed POST and sends it (via libcurl) to a localhost mock that
    /// stands in for the Lambda endpoint. We assert the program got the
    /// mock's 200 + body back, and that the server actually received a
    /// well-formed signed request (POST, correct path, Authorization +
    /// X-Amz-Date headers, the JSON body). This exercises the whole
    /// client glue: URL/host/path, dates, header assembly, signing, and
    /// the HTTP round-trip — everything but real AWS credentials.
    #[test]
    fn aws_signed_post_reaches_mock_endpoint() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::mpsc;
        init();

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel::<String>();
        let server = std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let n = sock.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).into_owned();
                let body = "{\"ok\":true}";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
                let _ = tx.send(req);
            }
        });

        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def run(port: Int) -> Int = {
                let p = int_to_string(port);
                let host = string_concat(\"127.0.0.1:\", p);
                let path = \"/2015-03-31/functions/myfunc/invocations\";
                let url = str3(\"http://\", host, path);
                match aws_signed_post(url, host, path, \"us-east-1\", \"lambda\",
                                      \"{\\\"x\\\":1}\", \"AKIDEXAMPLE\", \"SECRETKEY\", \"\") {
                    Result::Ok(resp) => resp.status,
                    Result::Err(e) => 0 - 1
                }
             }",
        );
        let status = unsafe {
            jit.get_fn1(&names["run"]).unwrap().call(rt.thread_ptr(), port as i64)
        };
        let _ = server.join();
        let req = rx.recv().expect("server captured the request");

        assert_eq!(status, 200, "mock returned 200");
        assert!(
            req.starts_with("POST /2015-03-31/functions/myfunc/invocations"),
            "request line: {:?}",
            req.lines().next()
        );
        assert!(
            req.contains("Authorization: AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"),
            "missing/invalid Authorization in:\n{}",
            req
        );
        assert!(
            req.contains("/us-east-1/lambda/aws4_request"),
            "credential scope wrong in:\n{}",
            req
        );
        assert!(req.contains("X-Amz-Date:"), "missing X-Amz-Date in:\n{}", req);
        assert!(req.contains("{\"x\":1}"), "missing request body in:\n{}", req);
    }

    /// Program B: an ai-lang AWS Lambda custom runtime. `lambda_run_once`
    /// is driven against a mock of the real Lambda Runtime API: it GETs
    /// the next invocation (reading the request id from a response
    /// header), runs the handler, and POSTs the result to the
    /// per-invocation response endpoint. We assert the worker posted the
    /// handler's output for the right request id.
    #[test]
    fn lambda_runtime_worker_processes_invocation() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::mpsc;
        init();

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel::<String>();
        let server = std::thread::spawn(move || {
            // Connection 1: GET /.../invocation/next -> event + request id.
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf);
                let event = "{\"name\":\"world\"}";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nLambda-Runtime-Aws-Request-Id: req-abc-123\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    event.len(),
                    event
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
            }
            // Connection 2: POST /.../invocation/req-abc-123/response.
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let n = sock.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).into_owned();
                let resp = "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
                let _ = tx.send(req);
            }
        });

        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def handler(event: String) -> String =
                match json_field(event, \"name\") {
                    Result::Ok(name) => str3(\"{\\\"greeting\\\":\\\"hello \", name, \"\\\"}\"),
                    Result::Err(e) => \"{\\\"error\\\":\\\"bad event\\\"}\"
                }
             def run(port: Int) -> Int = {
                let api = string_concat(\"127.0.0.1:\", int_to_string(port));
                match lambda_run_once(api, |e: String| handler(e)) {
                    Result::Ok(s) => s,
                    Result::Err(e) => 0 - 1
                }
             }",
        );
        let status = unsafe {
            jit.get_fn1(&names["run"]).unwrap().call(rt.thread_ptr(), port as i64)
        };
        let _ = server.join();
        let posted = rx.recv().expect("worker posted a response");

        // The worker posted to the correct per-invocation response path
        // (proving it read the request id from the response header)...
        assert_eq!(status, 202, "worker should report the POST status");
        assert!(
            posted.starts_with("POST /2015-03-31/functions") == false
                && posted.contains("/2018-06-01/runtime/invocation/req-abc-123/response"),
            "wrong response path in:\n{}",
            posted
        );
        // ...with the handler's computed body.
        assert!(
            posted.contains("{\"greeting\":\"hello world\"}"),
            "handler output missing in:\n{}",
            posted
        );
    }

    /// Program A and Program B together. B is an ai-lang Lambda handler;
    /// A is an ai-lang client that invokes it with a SigV4-signed request.
    /// A local server stands in for the AWS Lambda service: it receives
    /// A's signed invoke and returns what B's handler computes for the
    /// event. We run B's handler for real (via the JIT) to produce the
    /// response, then assert A's invoke received exactly that. Both halves
    /// are genuine ai-lang programs connected by a signed HTTP invoke.
    #[test]
    fn program_a_invokes_program_b_handler() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::mpsc;
        init();

        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            // B: the handler. A: invoke a localhost endpoint with an event,
            // return the response body it receives.
            "def handler(event: String) -> String =
                match json_field(event, \"name\") {
                    Result::Ok(name) => str3(\"{\\\"greeting\\\":\\\"hello \", name, \"\\\"}\"),
                    Result::Err(e) => \"{\\\"error\\\":\\\"bad event\\\"}\"
                }
             def invoke(port: Int, event: String) -> String = {
                let p = int_to_string(port);
                let host = string_concat(\"127.0.0.1:\", p);
                let path = \"/2015-03-31/functions/echo/invocations\";
                let url = str3(\"http://\", host, path);
                match aws_signed_post(url, host, path, \"us-east-1\", \"lambda\",
                                      event, \"AKIDEXAMPLE\", \"SECRETKEY\", \"\") {
                    Result::Ok(resp) => resp.body,
                    Result::Err(e) => string_concat(\"ERR: \", e)
                }
             }",
        );

        // Run B's handler for real to get the canonical response for the
        // event, and stash it where the mock service can return it.
        let event = "{\"name\":\"alice\"}";
        let b_output: String = unsafe {
            let handler = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["handler"]),
                )
                .unwrap();
            let ev = crate::ffi::owned_str_to_heap(rt.thread_ptr(), event);
            let out = handler.call(rt.thread_ptr(), ev);
            crate::ffi::heap_str_to_owned(out)
        };
        assert_eq!(b_output, "{\"greeting\":\"hello alice\"}");

        // Mock AWS Lambda service: receive A's invoke, return B's output.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel::<String>();
        let b_out_for_server = b_output.clone();
        let server = std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let n = sock.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).into_owned();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    b_out_for_server.len(),
                    b_out_for_server
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
                let _ = tx.send(req);
            }
        });

        // Run A: invoke with the event, get back the response body.
        let a_received: String = unsafe {
            let invoke = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, *mut u8) -> *mut u8>(
                    &crate::codegen::def_symbol(&names["invoke"]),
                )
                .unwrap();
            let ev = crate::ffi::owned_str_to_heap(rt.thread_ptr(), event);
            let out = invoke.call(rt.thread_ptr(), port as i64, ev);
            crate::ffi::heap_str_to_owned(out)
        };
        let _ = server.join();
        let req = rx.recv().expect("service received the invoke");

        // A's signed invoke carried the event...
        assert!(
            req.starts_with("POST /2015-03-31/functions/echo/invocations")
                && req.contains("Authorization: AWS4-HMAC-SHA256")
                && req.contains("{\"name\":\"alice\"}"),
            "A's invoke was not a well-formed signed request carrying the event:\n{}",
            req
        );
        // ...and A received exactly what B's handler computed.
        assert_eq!(a_received, b_output);
    }

    /// UTC date formatting via libc gmtime_r + strftime, against a known
    /// epoch. 1440937560 = 2015-08-30T12:26:00Z.
    #[test]
    fn aws_amz_date_formatting() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def dt() -> Int =
                match amz_datetime(1440937560) {
                    Result::Ok(s) => string_eq(s, \"20150830T122600Z\"),
                    Result::Err(_) => 0 - 999
                }
             def ds() -> Int =
                match amz_datestamp(1440937560) {
                    Result::Ok(s) => string_eq(s, \"20150830\"),
                    Result::Err(_) => 0 - 999
                }",
        );
        unsafe {
            assert_eq!(jit.get_fn0(&names["dt"]).unwrap().call(rt.thread_ptr()), 1);
            assert_eq!(jit.get_fn0(&names["ds"]).unwrap().call(rt.thread_ptr()), 1);
        }
    }

    // ---- OS surface FFI bridge (Tier 2) ----

    /// Filesystem write + `read_file` roundtrip, environment read, and a
    /// sanity bound on the clock. All deterministic.
    #[test]
    fn os_fs_env_clock() {
        init();
        // Set an env var this process can read back (unique name so it
        // can't collide with anything else).
        unsafe {
            std::env::set_var("AI_LANG_TEST_VAR_OSXYZ", "hello42");
        }
        let path = std::env::temp_dir().join("ai_lang_os_test_roundtrip.txt");
        let path_str = path.to_string_lossy().replace('\\', "/");

        let ctx = Context::create();
        let src = format!(
            "def fs_roundtrip() -> Int = {{
                let w = match fs_write(\"{path}\", \"roundtrip!\") {{
                    Result::Ok(v) => v,
                    Result::Err(_) => 0 - 999
                }};
                match read_file(\"{path}\") {{
                    Result::Ok(c) => w + string_len(c),
                    Result::Err(e) => 0 - 1
                }}
             }}
             def env_len() -> Int =
                match env_get(\"AI_LANG_TEST_VAR_OSXYZ\") {{
                    Result::Ok(s) => string_len(s),
                    Result::Err(_) => 0 - 999
                }}
             def env_missing() -> Int =
                match env_has(\"AI_LANG_DEFINITELY_UNSET_VAR_QQQ\") {{
                    Result::Ok(v) => v,
                    Result::Err(_) => 0 - 999
                }}
             def clock_ok() -> Int = if now_unix() > 1700000000 {{ 1 }} else {{ 0 }}
             def clock_millis_ok() -> Int =
                if now_unix_millis() > 1700000000000 {{ 1 }} else {{ 0 }}
             def exists_of(p: String) -> Int =
                match fs_exists(p) {{ Result::Ok(v) => v, Result::Err(_) => 0 - 999 }}
             def fs_exists_ok() -> Int = exists_of(\"{path}\")
             def fs_missing() -> Int = exists_of(\"/no/such/path/qqzz_unlikely\")",
            path = path_str
        );
        let (rt, jit, names) = build_with_stdlib(&ctx, &src);
        unsafe {
            // write returns 0, contents "roundtrip!" len 10 -> 10.
            assert_eq!(jit.get_fn0(&names["fs_roundtrip"]).unwrap().call(rt.thread_ptr()), 10);
            // "hello42" len 7.
            assert_eq!(jit.get_fn0(&names["env_len"]).unwrap().call(rt.thread_ptr()), 7);
            // unset var -> 0.
            assert_eq!(jit.get_fn0(&names["env_missing"]).unwrap().call(rt.thread_ptr()), 0);
            // clock is past 2023.
            assert_eq!(jit.get_fn0(&names["clock_ok"]).unwrap().call(rt.thread_ptr()), 1);
            // millisecond clock is also past 2023.
            assert_eq!(jit.get_fn0(&names["clock_millis_ok"]).unwrap().call(rt.thread_ptr()), 1);
            // the file we just wrote exists; a bogus path does not.
            assert_eq!(jit.get_fn0(&names["fs_exists_ok"]).unwrap().call(rt.thread_ptr()), 1);
            assert_eq!(jit.get_fn0(&names["fs_missing"]).unwrap().call(rt.thread_ptr()), 0);
        }
        let _ = std::fs::remove_file(&path);
    }

    // ---- defer (deterministic cleanup) ----

    /// `defer` runs its cleanup on normal completion of the enclosing
    /// block. The cleanup writes a marker into a malloc'd cell which the
    /// caller reads back after the deferring function returns.
    #[test]
    fn defer_runs_on_normal_exit() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def body_normal(cell: Ptr) -> Int = {
                defer ptr_write_i64(cell, 0, 1);
                99
             }
             def t_normal() -> Int = {
                let cell = malloc(8);
                let z = ptr_write_i64(cell, 0, 0);
                let r = body_normal(cell);
                let v = ptr_read_i64(cell, 0);
                let f = free(cell);
                r + v
             }",
        );
        unsafe {
            // body returns 99, defer set the cell to 1 -> 100.
            assert_eq!(jit.get_fn0(&names["t_normal"]).unwrap().call(rt.thread_ptr()), 100);
        }
    }

    /// Multiple defers run in LIFO order. Each cleanup stamps the current
    /// value of a shared step counter into its own cell; the one declared
    /// last runs first (step 0), the one declared first runs last (step 1).
    #[test]
    fn defer_lifo_order() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def step(counter: Ptr) -> Int = {
                let c = ptr_read_i64(counter, 0);
                let w = ptr_write_i64(counter, 0, c + 1);
                c
             }
             def body_lifo(counter: Ptr, a: Ptr, b: Ptr) -> Int = {
                defer ptr_write_i64(a, 0, step(counter));
                defer ptr_write_i64(b, 0, step(counter));
                0
             }
             def t_lifo() -> Int = {
                let counter = malloc(8);
                let a = malloc(8);
                let b = malloc(8);
                let z = ptr_write_i64(counter, 0, 0);
                let r = body_lifo(counter, a, b);
                let av = ptr_read_i64(a, 0);
                let bv = ptr_read_i64(b, 0);
                let f1 = free(counter);
                let f2 = free(a);
                let f3 = free(b);
                av * 10 + bv
             }",
        );
        unsafe {
            // b's defer runs first (step 0), a's runs second (step 1):
            // av=1, bv=0 -> 1*10 + 0 = 10.
            assert_eq!(jit.get_fn0(&names["t_lifo"]).unwrap().call(rt.thread_ptr()), 10);
        }
    }

    /// `defer` runs even when a `?` early-returns out of the function:
    /// the cleanup fires on the Err path before the Result propagates.
    #[test]
    fn defer_runs_on_try_early_return() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def make_err() -> Result<Int, Int> = Result::Err(0 - 5)
             def body_early(cell: Ptr) -> Result<Int, Int> = {
                defer ptr_write_i64(cell, 0, 1);
                let x = make_err()?;
                Result::Ok(x + 1)
             }
             def t_early() -> Int = {
                let cell = malloc(8);
                let z = ptr_write_i64(cell, 0, 0);
                let r = match body_early(cell) {
                    Result::Ok(v) => v,
                    Result::Err(e) => e
                };
                let v = ptr_read_i64(cell, 0);
                let f = free(cell);
                v * 100 + (0 - r)
             }",
        );
        unsafe {
            // defer ran (v=1) and the Err(-5) propagated (r=-5):
            // 1*100 + 5 = 105.
            assert_eq!(jit.get_fn0(&names["t_early"]).unwrap().call(rt.thread_ptr()), 105);
        }
    }

    /// `defer` happy-path with a `?` that succeeds: the cleanup still runs
    /// once, on normal completion, and the Ok value flows through.
    #[test]
    fn defer_runs_once_on_try_success() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_with_stdlib(
            &ctx,
            "def make_ok() -> Result<Int, Int> = Result::Ok(40)
             def body_ok(counter: Ptr) -> Result<Int, Int> = {
                defer ptr_write_i64(counter, 0, ptr_read_i64(counter, 0) + 1);
                let x = make_ok()?;
                Result::Ok(x + 2)
             }
             def t_ok() -> Int = {
                let counter = malloc(8);
                let z = ptr_write_i64(counter, 0, 0);
                let r = match body_ok(counter) {
                    Result::Ok(v) => v,
                    Result::Err(e) => 0 - 1
                };
                let runs = ptr_read_i64(counter, 0);
                let f = free(counter);
                r * 10 + runs
             }",
        );
        unsafe {
            // Ok(42) flows through, defer ran exactly once: 42*10 + 1 = 421.
            assert_eq!(jit.get_fn0(&names["t_ok"]).unwrap().call(rt.thread_ptr()), 421);
        }
    }
}
