//! Parser/lexer robustness fuzzing. The contract: `lex` and `parse_module` must
//! NEVER panic on any input — they return `Ok` or a clean `Err`, but never crash
//! (no unwrap-on-None, no slice-out-of-bounds, no infinite loop). A compiler
//! front end is a parser of untrusted text; a panic here is a real bug.
//!
//! This is a deterministic, dependency-free fuzzer: a seeded xorshift PRNG drives
//! several input generators (random bytes, token-salad from the real vocabulary,
//! and mutations of valid programs). It runs under plain `cargo test` and any
//! failure prints the exact seed + input to reproduce. Each `parse` is run on a
//! catch_unwind so a panic is reported as a test failure with its input, rather
//! than aborting the whole run.

use gcrust::lexer::lex;
use gcrust::parser::parse_module;

/// A tiny deterministic PRNG (xorshift64*). Reproducible from a seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

/// Lex + parse `src`, catching any panic. Returns `Err(panic message)` if it
/// panicked. We don't care whether parsing SUCCEEDS — only that it doesn't crash
/// (logic panic). Runs on a worker thread with a large stack so this test
/// validates parser LOGIC rather than the test harness's small (2 MB) stack;
/// stack-overflow protection itself is guaranteed by the parser's depth limit
/// and tested separately in `depth_limit_prevents_overflow`.
fn try_parse(src: &str) -> Result<(), String> {
    let src = src.to_string();
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            // Lex may error (fine); only parse a successful lex.
            if let Ok(toks) = lex(&src) {
                let _ = parse_module(&toks);
            }
        })
        .expect("spawn fuzz worker");
    handle.join().map_err(|e| {
        if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "panic (non-string payload)".to_string()
        }
    })
}

/// Random bytes (often invalid UTF-8 paths excluded — we build from chars).
fn gen_random_chars(rng: &mut Rng, len: usize) -> String {
    // A charset weighted toward syntactically-meaningful characters.
    const CHARS: &[char] = &[
        'f', 'n', 'a', 'x', 'i', '6', '4', '0', '1', ' ', '\n', '\t',
        '(', ')', '{', '}', '[', ']', '<', '>', ':', ';', ',', '.',
        '+', '-', '*', '/', '%', '=', '!', '&', '|', '"', '\'', '_',
        ':', ':', // bias toward `::`
    ];
    (0..len).map(|_| *rng.pick(CHARS)).collect()
}

/// "Token salad" — sequences of real keywords/punctuation, which reach far deeper
/// into the parser than random bytes (they lex cleanly, so parse_module sees real
/// token streams).
fn gen_token_salad(rng: &mut Rng, len: usize) -> String {
    const TOKS: &[&str] = &[
        "fn", "let", "mut", "struct", "enum", "impl", "trait", "for", "in",
        "if", "else", "match", "while", "loop", "return", "break", "continue",
        "mod", "use", "pub", "value", "as", "where", "self", "Self",
        "i64", "f64", "bool", "true", "false", "main", "x", "foo", "T",
        "(", ")", "{", "}", "[", "]", "<", ">", "::", ":", ";", ",", ".",
        "->", "=>", "=", "==", "+", "-", "*", "/", "|", "&", "?", "0", "42",
        "\"s\"", "'c'",
    ];
    let mut out = String::new();
    for _ in 0..len {
        out.push_str(*rng.pick(TOKS));
        out.push(' ');
    }
    out
}

/// Mutate a valid program by deleting, duplicating, or corrupting bytes — finds
/// crashes near the boundary of valid syntax (the most bug-prone region).
fn gen_mutation(rng: &mut Rng, base: &str) -> String {
    let mut bytes: Vec<char> = base.chars().collect();
    if bytes.is_empty() {
        return String::new();
    }
    let edits = 1 + rng.below(5);
    for _ in 0..edits {
        if bytes.is_empty() { break; }
        let i = rng.below(bytes.len());
        match rng.below(3) {
            0 => { bytes.remove(i); }                       // delete
            1 => { let c = bytes[i]; bytes.insert(i, c); }  // duplicate
            _ => { bytes[i] = *rng.pick(&['{', '}', '(', ')', '<', '>', ';', ':']); } // corrupt
        }
    }
    bytes.into_iter().collect()
}

const SEEDS: &[&str] = &[
    "fn main() -> i64 { 0 }",
    "struct P { x: i64, y: i64 } fn main() -> i64 { let p = P { x: 1, y: 2 }; p.x }",
    "enum E<T> { A, B(T) } fn f<T>(e: E<T>) -> i64 { match e { E::A => 0, E::B(_) => 1 } }",
    "trait Show { fn show(self) -> i64; } impl Show for i64 { fn show(self) -> i64 { self } }",
    "mod m { pub fn g() -> i64 { 1 } } use m::g; fn main() -> i64 { g() }",
    "fn main() -> i64 { let f = |x: i64| x * 2; f(21) }",
];

/// Run `iters` random inputs from `gen`, failing the test (with the seed +
/// offending input) on the first panic.
fn fuzz_with(mut seed: u64, iters: usize, mut make: impl FnMut(&mut Rng) -> String) {
    for _ in 0..iters {
        let mut rng = Rng(seed);
        let input = make(&mut rng);
        if let Err(panic) = try_parse(&input) {
            panic!(
                "parser panicked!\n  seed: {}\n  panic: {}\n  input: {:?}",
                seed, panic, input
            );
        }
        seed = seed.wrapping_add(0x9E3779B97F4A7C15); // advance the seed stream
    }
}

#[test]
fn fuzz_random_chars() {
    fuzz_with(0xDEADBEEF, 4000, |rng| {
        let len = rng.below(120);
        gen_random_chars(rng, len)
    });
}

#[test]
fn fuzz_token_salad() {
    fuzz_with(0x1234_5678, 4000, |rng| {
        let len = rng.below(40);
        gen_token_salad(rng, len)
    });
}

#[test]
fn fuzz_mutations() {
    let mut which = Rng(0xCAFE);
    fuzz_with(0xABCDEF, 4000, |rng| {
        let base = SEEDS[which.below(SEEDS.len())];
        gen_mutation(rng, base)
    });
}

#[test]
fn depth_limit_prevents_overflow_on_small_stack() {
    // The parser's depth limit must keep deeply-nested input from overflowing a
    // standard 2 MB thread stack (the Rust test-thread default, and a reasonable
    // floor) — it returns a clean Err, never crashes.
    let deep_parens = format!("fn main() -> i64 {{ {} 0 {} }}", "(".repeat(2000), ")".repeat(2000));
    let deep_blocks = format!("fn main() -> i64 {{ {} 0 {} }}", "{".repeat(2000), "}".repeat(2000));
    for src in [deep_parens, deep_blocks] {
        let handle = std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                let toks = lex(&src).unwrap();
                // Must return an Err (too deep), not panic / overflow.
                parse_module(&toks).is_err()
            })
            .unwrap();
        let too_deep = handle.join().expect("parser overflowed a 2 MB stack instead of erroring");
        assert!(too_deep, "expected a 'nested too deeply' error");
    }
}

#[test]
fn fuzz_pathological_shapes() {
    // Deeply nested and degenerate inputs that have historically broken parsers.
    let cases = [
        "(".repeat(500),
        ")".repeat(500),
        "{".repeat(500),
        "<".repeat(500),
        "fn ".repeat(300),
        "fn f<".to_string() + &"T,".repeat(300) + ">() -> i64 { 0 }",
        format!("fn main() -> i64 {{ {} 0 }}", "((".repeat(200)),
        "::".repeat(400),
        String::new(),
        " ".repeat(1000),
        "\0\0\0".to_string(),
    ];
    for c in &cases {
        if let Err(panic) = try_parse(c) {
            panic!("parser panicked on pathological input!\n  panic: {}\n  input: {:?}", panic, c);
        }
    }
}
