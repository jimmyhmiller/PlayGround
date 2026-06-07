//! **Structural flat-tape front end** — text → SIMD lex → a self-contained
//! postorder (RPN) tape, no AST, no interning, no IR.
//!
//! Each node is a fixed 16-byte record: `(kind, flags, nargs, start, end)`.
//! Nodes are emitted in post-order, so a node's `nargs` children are the
//! immediately-preceding completed subtrees on the tape — exactly RPN. That one
//! invariant reconstructs the whole tree (a single stack walk), and the source
//! spans recover every name / literal / operator. It carries as much as a
//! standard AST while building at ~memcpy cost (push a struct into a `Vec`).
//!
//! This is the same grammar subset as `jsir-parse` (literals, identifiers,
//! unary/binary/assignment/update expressions, expression statements, and
//! let/var/const declarations) but driven straight into the tape, so we can
//! bench tape-build vs oxc's AST and vs the heavy columnar JSIR `Module`.
//!
//! Usage:  tape [statements]        (default 20000; also prints a tiny demo)

use std::hint::black_box;
use std::time::{Duration, Instant};

use oxc_allocator::Allocator;
use oxc_parser::Parser as OxcParser;
use oxc_span::SourceType;

use simd_lang::js::{Lexer, TokKind as S};

// ───────────────────────────── tape ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
enum K {
    NumLit,
    StrLit,
    BoolLit,
    NullLit,
    BigIntLit,
    Ident,
    IdentRef,
    Binary,
    Unary,
    Update,
    Assign,
    VarDeclarator,
    VarDecl,
    ExprStmt,
}

/// A flat tape node — 16 bytes, no pointers. For operator nodes (`Binary`,
/// `Unary`, `Update`, `Assign`) `start..end` is the *operator* token span; for
/// leaves it is the leaf's own span. `nargs` is the RPN arity (children are the
/// preceding `nargs` completed subtrees). `flags` bit 0 = prefix.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Node {
    kind: K,
    flags: u16,
    nargs: u32,
    start: u32,
    end: u32,
}

#[derive(Default)]
struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    #[inline]
    fn push(&mut self, kind: K, nargs: u32, start: u32, end: u32, prefix: bool) {
        self.nodes.push(Node { kind, flags: prefix as u16, nargs, start, end });
    }
}

// ───────────────────────────── punctuators ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pk {
    Plus, Minus, Star, Slash, Percent, Lt, Le, Gt, Ge, EqEq, EqEqEq, Ne, NeEq,
    Amp, Pipe, Caret, Shl, Shr, UShr, Bang, Tilde, PlusPlus, MinusMinus,
    Assign, PlusEq, MinusEq, StarEq, SlashEq, PercentEq, ShlEq, ShrEq, UShrEq,
    AmpEq, PipeEq, CaretEq, StarStarEq, Semi, Comma, Dot, Other,
}

/// Classify an operator from its raw bytes by dispatching on (length, bytes) —
/// integer jump tables, no string comparison. The lexer already delimited the
/// span, so this is the only classification work left.
#[inline]
fn classify_punct(s: &[u8]) -> Pk {
    match s.len() {
        1 => match s[0] {
            b'+' => Pk::Plus, b'-' => Pk::Minus, b'*' => Pk::Star, b'/' => Pk::Slash,
            b'%' => Pk::Percent, b'<' => Pk::Lt, b'>' => Pk::Gt, b'=' => Pk::Assign,
            b'&' => Pk::Amp, b'|' => Pk::Pipe, b'^' => Pk::Caret, b'!' => Pk::Bang,
            b'~' => Pk::Tilde, b';' => Pk::Semi, b',' => Pk::Comma, b'.' => Pk::Dot,
            _ => Pk::Other,
        },
        2 => match (s[0], s[1]) {
            (b'<', b'=') => Pk::Le, (b'>', b'=') => Pk::Ge, (b'=', b'=') => Pk::EqEq,
            (b'!', b'=') => Pk::Ne, (b'<', b'<') => Pk::Shl, (b'>', b'>') => Pk::Shr,
            (b'+', b'+') => Pk::PlusPlus, (b'-', b'-') => Pk::MinusMinus,
            (b'+', b'=') => Pk::PlusEq, (b'-', b'=') => Pk::MinusEq, (b'*', b'=') => Pk::StarEq,
            (b'/', b'=') => Pk::SlashEq, (b'%', b'=') => Pk::PercentEq, (b'&', b'=') => Pk::AmpEq,
            (b'|', b'=') => Pk::PipeEq, (b'^', b'=') => Pk::CaretEq,
            _ => Pk::Other,
        },
        3 => match (s[0], s[1], s[2]) {
            (b'=', b'=', b'=') => Pk::EqEqEq, (b'!', b'=', b'=') => Pk::NeEq,
            (b'>', b'>', b'>') => Pk::UShr, (b'<', b'<', b'=') => Pk::ShlEq,
            (b'>', b'>', b'=') => Pk::ShrEq, (b'*', b'*', b'=') => Pk::StarStarEq,
            _ => Pk::Other,
        },
        _ => Pk::Other, // >>>= and friends are rare; the parser treats Other as non-operator
    }
}

#[inline]
fn is_assign_op(p: Pk) -> bool {
    matches!(p, Pk::Assign | Pk::PlusEq | Pk::MinusEq | Pk::StarEq | Pk::SlashEq
        | Pk::PercentEq | Pk::ShlEq | Pk::ShrEq | Pk::UShrEq | Pk::AmpEq | Pk::PipeEq
        | Pk::CaretEq | Pk::StarStarEq)
}

#[inline]
fn binary_bp(p: Pk) -> Option<u8> {
    Some(match p {
        Pk::Pipe => 4, Pk::Caret => 5, Pk::Amp => 6,
        Pk::EqEq | Pk::Ne | Pk::EqEqEq | Pk::NeEq => 7,
        Pk::Lt | Pk::Le | Pk::Gt | Pk::Ge => 8,
        Pk::Shl | Pk::Shr | Pk::UShr => 9,
        Pk::Plus | Pk::Minus => 10,
        Pk::Star | Pk::Slash | Pk::Percent => 11,
        _ => return None,
    })
}

// ───────────────────────────── token + pull lexer ─────────────────────────

#[derive(Clone, Copy)]
struct Tok {
    kind: S,
    punct: Pk,
    start: u32,
    end: u32,
}

/// Non-materializing pull source over the SIMD lexer with 2-token lookahead.
struct Lex<'a> {
    inner: Lexer<'a>,
    src: &'a [u8],
    buf: [Tok; 2],
    len: usize,
}

impl<'a> Lex<'a> {
    fn new(src: &'a [u8], sm: &'a [u64], wm: &'a [u64]) -> Lex<'a> {
        Lex {
            inner: Lexer::new(src, sm, wm),
            src,
            buf: [Tok { kind: S::Punct, punct: Pk::Other, start: 0, end: 0 }; 2],
            len: 0,
        }
    }
    #[inline]
    fn pull(&mut self) -> Tok {
        loop {
            match self.inner.next_tok::<false>() {
                Some(t) => {
                    let punct = if matches!(t.kind, S::Punct) {
                        classify_punct(&self.src[t.start..t.end])
                    } else if t.kind.is_trivia() {
                        continue;
                    } else {
                        Pk::Other
                    };
                    return Tok { kind: t.kind, punct, start: t.start as u32, end: t.end as u32 };
                }
                None => {
                    let n = self.src.len() as u32;
                    return Tok { kind: S::Punct, punct: Pk::Other, start: n, end: n };
                }
            }
        }
    }
    /// Like `pull` but never calls `classify_punct` — for bounding its cost.
    #[inline]
    fn pull_nopk(&mut self) -> Tok {
        loop {
            match self.inner.next_tok::<false>() {
                Some(t) => {
                    if t.kind.is_trivia() { continue; }
                    return Tok { kind: t.kind, punct: Pk::Other, start: t.start as u32, end: t.end as u32 };
                }
                None => {
                    let n = self.src.len() as u32;
                    return Tok { kind: S::Punct, punct: Pk::Other, start: n, end: n };
                }
            }
        }
    }
    #[inline]
    fn fill(&mut self, n: usize) {
        while self.len < n {
            let t = self.pull();
            self.buf[self.len] = t;
            self.len += 1;
        }
    }
    #[inline]
    fn eof(&mut self) -> bool {
        self.fill(1);
        self.buf[0].start as usize >= self.src.len() && matches!(self.buf[0].kind, S::Punct) && self.buf[0].punct == Pk::Other
    }
}

// helper on simd TokKind
trait TriviaExt { fn is_trivia(self) -> bool; }
impl TriviaExt for S {
    #[inline]
    fn is_trivia(self) -> bool { matches!(self, S::LineComment | S::BlockComment) }
}

// ───────────────────── token source abstraction ─────────────────────
//
// The parser is generic over where tokens come from: the live fused SIMD lexer
// (`Lex`) or a pre-materialized `Vec<Tok>` (`VecLex`). Splitting these lets us
// measure how much of the parse is *lexing* vs *tree-building* — and the array
// is exactly the shape a `.simd` kernel can fill in bulk.
trait TokSrc {
    fn peek(&mut self) -> Tok;
    fn peek2(&mut self) -> Tok;
    fn bump(&mut self) -> Tok;
    fn at_eof(&mut self) -> bool;
}

impl<'a> TokSrc for Lex<'a> {
    #[inline]
    fn peek(&mut self) -> Tok { self.fill(1); self.buf[0] }
    #[inline]
    fn peek2(&mut self) -> Tok { self.fill(2); self.buf[1] }
    #[inline]
    fn bump(&mut self) -> Tok {
        self.fill(1);
        let t = self.buf[0];
        self.buf[0] = self.buf[1];
        self.len -= 1;
        t
    }
    #[inline]
    fn at_eof(&mut self) -> bool { self.eof() }
}

/// Token source over a pre-materialized slice (the EOF sentinel is the last elem).
struct VecLex<'a> {
    toks: &'a [Tok],
    i: usize,
}
impl<'a> TokSrc for VecLex<'a> {
    #[inline]
    fn peek(&mut self) -> Tok { self.toks[self.i] }
    #[inline]
    fn peek2(&mut self) -> Tok { self.toks[(self.i + 1).min(self.toks.len() - 1)] }
    #[inline]
    fn bump(&mut self) -> Tok {
        let t = self.toks[self.i];
        if self.i + 1 < self.toks.len() { self.i += 1; }
        t
    }
    #[inline]
    fn at_eof(&mut self) -> bool { self.i + 1 >= self.toks.len() }
}

// ───────────────────────────── parser ─────────────────────────────

struct P<'a, L: TokSrc> {
    src: &'a [u8],
    lx: L,
    t: Tape,
    err: Option<String>,
}

impl<'a, L: TokSrc> P<'a, L> {
    #[inline]
    fn peek(&mut self) -> Tok { self.lx.peek() }
    #[inline]
    fn peek2(&mut self) -> Tok { self.lx.peek2() }
    #[inline]
    fn bump(&mut self) -> Tok { self.lx.bump() }
    #[inline]
    fn eat(&mut self, p: Pk) -> bool {
        if self.peek().punct == p { self.bump(); true } else { false }
    }
    #[inline]
    fn txt(&self, t: Tok) -> &'a str {
        // Bind to `'a` (the source), not `&self`, so callers can hold the &str
        // across `&mut self` calls (e.g. a keyword match whose guard peeks).
        let s: &'a [u8] = self.src;
        unsafe { std::str::from_utf8_unchecked(&s[t.start as usize..t.end as usize]) }
    }

    fn program(&mut self) {
        while !self.lx.at_eof() && self.err.is_none() {
            self.statement();
        }
    }

    fn statement(&mut self) {
        let t = self.peek();
        if matches!(t.kind, S::Ident) {
            match self.txt(t) {
                "let" | "var" | "const" if matches!(self.peek2().kind, S::Ident) => {
                    return self.var_decl();
                }
                _ => {}
            }
        }
        self.assignment();
        self.eat(Pk::Semi);
        self.t.push(K::ExprStmt, 1, 0, 0, false);
    }

    fn var_decl(&mut self) {
        let kt = self.bump(); // let|var|const
        let mut n = 0u32;
        loop {
            let nt = self.peek();
            if !matches!(nt.kind, S::Ident) {
                self.err = Some(format!("expected identifier at {}", nt.start));
                return;
            }
            self.bump();
            self.t.push(K::IdentRef, 0, nt.start, nt.end, false);
            if self.eat(Pk::Assign) {
                self.assignment();
                self.t.push(K::VarDeclarator, 2, 0, 0, false);
            } else {
                self.t.push(K::VarDeclarator, 1, 0, 0, false);
            }
            n += 1;
            if !self.eat(Pk::Comma) { break; }
        }
        self.eat(Pk::Semi);
        self.t.push(K::VarDecl, n, kt.start, kt.end, false);
    }

    fn assignment(&mut self) {
        let t = self.peek();
        if matches!(t.kind, S::Ident) && is_assign_op(self.peek2().punct)
            && !matches!(self.txt(t), "true" | "false" | "null")
        {
            let nt = self.bump();
            let ot = self.bump();
            self.t.push(K::IdentRef, 0, nt.start, nt.end, false);
            self.assignment();
            self.t.push(K::Assign, 2, ot.start, ot.end, false);
            return;
        }
        self.binary(0);
    }

    fn binary(&mut self, min_bp: u8) {
        self.unary();
        loop {
            let t = self.peek();
            let Some(bp) = binary_bp(t.punct) else { break };
            if bp < min_bp { break; }
            let ot = self.bump();
            self.binary(bp + 1);
            self.t.push(K::Binary, 2, ot.start, ot.end, false);
        }
    }

    fn unary(&mut self) {
        let t = self.peek();
        match t.punct {
            Pk::Minus | Pk::Plus | Pk::Bang | Pk::Tilde => {
                self.bump();
                self.unary();
                self.t.push(K::Unary, 1, t.start, t.end, true);
                return;
            }
            Pk::PlusPlus | Pk::MinusMinus => {
                self.bump();
                let nt = self.peek();
                if !matches!(nt.kind, S::Ident) {
                    self.err = Some(format!("expected lvalue at {}", nt.start));
                    return;
                }
                self.bump();
                self.t.push(K::IdentRef, 0, nt.start, nt.end, false);
                self.t.push(K::Update, 1, t.start, t.end, true);
                return;
            }
            _ => {}
        }
        self.primary();
    }

    fn primary(&mut self) {
        let t = self.peek();
        match t.kind {
            S::Number => {
                self.bump();
                let kind = if self.src[t.end as usize - 1] == b'n' { K::BigIntLit } else { K::NumLit };
                self.t.push(kind, 0, t.start, t.end, false);
            }
            S::String => {
                self.bump();
                self.t.push(K::StrLit, 0, t.start, t.end, false);
            }
            S::Ident | S::Keyword => {
                let name = self.txt(t);
                match name {
                    "true" | "false" => { self.bump(); self.t.push(K::BoolLit, 0, t.start, t.end, false); }
                    "null" => { self.bump(); self.t.push(K::NullLit, 0, t.start, t.end, false); }
                    _ => {
                        self.bump();
                        let nx = self.peek().punct;
                        if matches!(nx, Pk::PlusPlus | Pk::MinusMinus) {
                            let ot = self.bump();
                            self.t.push(K::IdentRef, 0, t.start, t.end, false);
                            self.t.push(K::Update, 1, ot.start, ot.end, false);
                        } else {
                            self.t.push(K::Ident, 0, t.start, t.end, false);
                        }
                    }
                }
            }
            _ => {
                self.err = Some(format!("unexpected token at {}", t.start));
                // consume to avoid infinite loop
                self.bump();
            }
        }
    }
}

fn parse_to_tape(src: &str, sm: &[u64], wm: &[u64]) -> Result<Tape, String> {
    let b = src.as_bytes();
    let mut p = P {
        src: b,
        lx: Lex::new(b, sm, wm),
        t: Tape { nodes: Vec::with_capacity(b.len() / 3 + 8) },
        err: None,
    };
    p.program();
    if let Some(e) = p.err.take() { return Err(e); }
    Ok(p.t)
}

/// Materialize the whole token stream into a `Vec<Tok>` (with a trailing EOF
/// sentinel) — the representation a `.simd` kernel would fill in bulk.
fn materialize_tokens(src: &str, sm: &[u64], wm: &[u64]) -> Vec<Tok> {
    let b = src.as_bytes();
    let mut lx = Lex::new(b, sm, wm);
    let mut out = Vec::with_capacity(b.len() / 4 + 8);
    loop {
        let t = lx.pull();
        let is_eof = t.start as usize >= b.len() && matches!(t.kind, S::Punct) && t.punct == Pk::Other;
        out.push(t);
        if is_eof { break; }
    }
    out
}

/// Absolute lexer floor: walk the start bitmap, compute each token's end from
/// the word bitmap (no byte reads, no classify), build a Tok. Bounds how fast a
/// SIMD tokenizer could possibly feed the parser.
fn skeleton_lex(src: &[u8], sm: &[u64], wm: &[u64]) -> usize {
    let n = src.len();
    let mut out = Vec::with_capacity(n / 4 + 8);
    for (ci, &w0) in sm.iter().enumerate() {
        let mut w = w0;
        while w != 0 {
            let bit = w.trailing_zeros() as usize;
            let p = ci * 64 + bit;
            if p >= n { break; }
            // word_end: first 0 in word bitmap at/after p
            let mut chunk = p / 64;
            let mut b = p % 64;
            let end = loop {
                if chunk >= wm.len() { break n; }
                let masked = wm[chunk] | ((1u64 << b) - 1);
                if masked != u64::MAX {
                    break (chunk * 64 + masked.trailing_ones() as usize).min(n);
                }
                chunk += 1;
                b = 0;
            };
            let end = if end > p { end } else { p + 1 };
            out.push(Tok { kind: S::Ident, punct: Pk::Other, start: p as u32, end: end as u32 });
            w &= w - 1;
        }
    }
    out.len()
}

/// First 0-bit in `wm` at/after `p` — token end for word runs (idents/numbers).
#[inline]
fn word_end(wm: &[u64], p: usize, n: usize) -> usize {
    let mut chunk = p / 64;
    let mut b = p % 64;
    loop {
        if chunk >= wm.len() { return n; }
        let masked = wm[chunk] | ((1u64 << b) - 1);
        if masked != u64::MAX {
            return (chunk * 64 + masked.trailing_ones() as usize).min(n);
        }
        chunk += 1;
        b = 0;
    }
}

/// Maximal-munch operator length (1–4) — same table as the shared lexer.
#[inline]
fn op_len(src: &[u8], p: usize) -> usize {
    let n = src.len();
    let b0 = src[p];
    if !matches!(b0, b'>' | b'<' | b'=' | b'!' | b'*' | b'&' | b'|' | b'+' | b'-' | b'%' | b'^') {
        return 1; // includes ( ) [ ] { } ; , . : ~ ? / and any single-char punct
    }
    let g = |o: usize| if p + o < n { src[p + o] } else { 0 };
    let (b1, b2, b3) = (g(1), g(2), g(3));
    match b0 {
        b'>' => match (b1, b2, b3) { (b'>', b'>', b'=') => 4, (b'>', b'>', _) => 3, (b'>', b'=', _) => 3, (b'>', _, _) => 2, (b'=', _, _) => 2, _ => 1 },
        b'<' => match (b1, b2) { (b'<', b'=') => 3, (b'<', _) => 2, (b'=', _) => 2, _ => 1 },
        b'=' => match (b1, b2) { (b'=', b'=') => 3, (b'=', _) => 2, (b'>', _) => 2, _ => 1 },
        b'!' => match (b1, b2) { (b'=', b'=') => 3, (b'=', _) => 2, _ => 1 },
        b'*' => match (b1, b2) { (b'*', b'=') => 3, (b'*', _) => 2, (b'=', _) => 2, _ => 1 },
        b'&' => match b1 { b'&' | b'=' => 2, _ => 1 },
        b'|' => match b1 { b'|' | b'=' => 2, _ => 1 },
        b'+' => match b1 { b'+' | b'=' => 2, _ => 1 },
        b'-' => match b1 { b'-' | b'=' => 2, _ => 1 },
        b'%' => match b1 { b'=' => 2, _ => 1 },
        b'^' => match b1 { b'=' => 2, _ => 1 },
        _ => 1,
    }
}

/// Lean self-contained tokenizer: walk the start bitmap; ends come from the word
/// bitmap (idents/numbers) or `op_len` (operators); strings/comments scalar.
/// Correct for the subset; this is what a `.simd` kernel would accelerate.
fn lex_to_array(src: &[u8], sm: &[u64], wm: &[u64]) -> Vec<Tok> {
    let n = src.len();
    let mut out = Vec::with_capacity(n / 4 + 8);
    for (ci, &w0) in sm.iter().enumerate() {
        let mut w = w0;
        while w != 0 {
            let p = ci * 64 + w.trailing_zeros() as usize;
            w &= w - 1;
            if p >= n { break; }
            let wbit = (wm[p / 64] >> (p % 64)) & 1 == 1;
            if wbit {
                // ident or number — word run end from the bitmap
                let mut end = word_end(wm, p, n);
                let digit = src[p].is_ascii_digit();
                if digit && end < n && src[end] == b'.' {
                    // decimal: extend past the fractional word run
                    let e2 = word_end(wm, end + 1, n);
                    end = e2.max(end + 1);
                }
                let kind = if digit { S::Number } else { S::Ident };
                out.push(Tok { kind, punct: Pk::Other, start: p as u32, end: end as u32 });
            } else {
                let c = src[p];
                match c {
                    b'"' | b'\'' => {
                        let mut i = p + 1;
                        while i < n && src[i] != c { i += if src[i] == b'\\' { 2 } else { 1 }; }
                        i = (i + 1).min(n);
                        out.push(Tok { kind: S::String, punct: Pk::Other, start: p as u32, end: i as u32 });
                    }
                    b'/' if p + 1 < n && src[p + 1] == b'/' => {
                        // line comment — skipped (trivia), don't emit
                        continue;
                    }
                    b'/' if p + 1 < n && src[p + 1] == b'*' => {
                        continue; // block comment trivia
                    }
                    _ => {
                        let len = op_len(src, p);
                        let end = p + len;
                        out.push(Tok { kind: S::Punct, punct: classify_punct(&src[p..end]), start: p as u32, end: end as u32 });
                    }
                }
            }
        }
    }
    let n32 = n as u32;
    out.push(Tok { kind: S::Punct, punct: Pk::Other, start: n32, end: n32 });
    out
}

/// Parse from a pre-materialized token slice (no lexing in the timed region).
fn parse_from_tokens<'a>(src: &'a [u8], toks: &'a [Tok]) -> Result<Tape, String> {
    let mut p = P {
        src,
        lx: VecLex { toks, i: 0 },
        t: Tape { nodes: Vec::with_capacity(toks.len() + 8) },
        err: None,
    };
    p.program();
    if let Some(e) = p.err.take() { return Err(e); }
    Ok(p.t)
}

// ───────────────────── tape walking (proves it's an AST) ─────────────────────

/// Verify RPN balance: walking and popping each node's `nargs` children must
/// leave exactly one root per top-level statement, never underflowing. Returns
/// the statement count.
fn validate(t: &Tape) -> Result<usize, String> {
    let mut stack: i64 = 0;
    let mut roots = 0usize;
    for (i, n) in t.nodes.iter().enumerate() {
        if (n.nargs as i64) > stack {
            return Err(format!("node {i} {:?} wants {} children, stack={stack}", n.kind, n.nargs));
        }
        stack -= n.nargs as i64;
        stack += 1; // this node becomes a value
        if matches!(n.kind, K::ExprStmt | K::VarDecl) {
            stack -= 1; // statements are consumed (not values)
            roots += 1;
        }
    }
    if stack != 0 {
        return Err(format!("unbalanced tape: {stack} dangling values"));
    }
    Ok(roots)
}

/// Pretty-print the tape as a tree (postorder → indented preorder) for a demo.
fn dump_tree(t: &Tape, src: &[u8]) -> String {
    fn span<'a>(src: &'a [u8], s: u32, e: u32) -> &'a str {
        std::str::from_utf8(&src[s as usize..e as usize]).unwrap_or("?")
    }
    // Rebuild children by consuming the RPN stack; each entry is a rendered block.
    let mut stack: Vec<(usize, String)> = Vec::new(); // (start-depth marker unused) text
    let mut out = String::new();
    for n in &t.nodes {
        let k = n.nargs as usize;
        let kids: Vec<String> = stack.split_off(stack.len() - k).into_iter().map(|(_, s)| s).collect();
        let label = match n.kind {
            K::NumLit => format!("Num({})", span(src, n.start, n.end)),
            K::StrLit => format!("Str({})", span(src, n.start, n.end)),
            K::BoolLit => format!("Bool({})", span(src, n.start, n.end)),
            K::NullLit => "Null".into(),
            K::BigIntLit => format!("BigInt({})", span(src, n.start, n.end)),
            K::Ident => format!("Ident({})", span(src, n.start, n.end)),
            K::IdentRef => format!("Ref({})", span(src, n.start, n.end)),
            K::Binary => format!("Binary[{}]", span(src, n.start, n.end)),
            K::Unary => format!("Unary[{}]", span(src, n.start, n.end)),
            K::Update => format!("Update[{}{}]", span(src, n.start, n.end), if n.flags & 1 != 0 { " pre" } else { " post" }),
            K::Assign => format!("Assign[{}]", span(src, n.start, n.end)),
            K::VarDeclarator => "Declarator".into(),
            K::VarDecl => format!("VarDecl({})", span(src, n.start, n.end)),
            K::ExprStmt => "ExprStmt".into(),
        };
        let mut block = label;
        for (i, kid) in kids.iter().enumerate() {
            let last = i + 1 == kids.len();
            let (head, rest) = (if last { "└─ " } else { "├─ " }, if last { "   " } else { "│  " });
            block.push('\n');
            for (j, line) in kid.lines().enumerate() {
                if j == 0 { block.push_str(head); } else { block.push_str(rest); }
                block.push_str(line);
                block.push('\n');
            }
            block.pop();
        }
        stack.push((0, block));
    }
    for (_, s) in stack {
        out.push_str(&s);
        out.push('\n');
    }
    out
}

// ───────────────────────────── bench harness ─────────────────────────────

fn generate(stmts: usize) -> String {
    let mut s = String::with_capacity(stmts * 40);
    for i in 0..stmts {
        match i % 3 {
            0 => s.push_str(&format!("var v{i} = {i} + {a} * 3 - 1;\n", a = i * 2)),
            1 => s.push_str(&format!("v{j} = v{j} + {i} * 2;\n", j = i.saturating_sub(1))),
            _ => s.push_str(&format!("var w{i} = {a} & 7 | {i};\n", a = i + 3)),
        }
    }
    s
}

fn bench(name: &str, bytes: usize, iters: u32, mut f: impl FnMut()) -> Duration {
    for _ in 0..3 { f(); }
    let t = Instant::now();
    for _ in 0..iters { f(); }
    let per = t.elapsed() / iters;
    let mbps = (bytes as f64 / (1 << 20) as f64) / per.as_secs_f64();
    println!("  {name:<34} {per:>10.2?}/iter   {mbps:>8.0} MiB/s");
    per
}

fn main() {
    // Tiny demo first — show the tape reconstructs a real tree.
    let demo = "let x = 1 + 2 * 3; y = -a; z++;";
    let (sm, wm) = simd_lang::stage1::lex(demo.as_bytes());
    let tape = parse_to_tape(demo, &sm, &wm).unwrap();
    println!("demo: {demo}");
    println!("tape: {} nodes ({} bytes), {} statements\n",
        tape.nodes.len(), tape.nodes.len() * std::mem::size_of::<Node>(),
        validate(&tape).unwrap());
    println!("{}", dump_tree(&tape, demo.as_bytes()));

    let stmts: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let src = generate(stmts);
    let bytes = src.len();
    let iters = 50;
    println!("source: {} statements, {:.2} MiB\n", stmts, bytes as f64 / (1 << 20) as f64);

    let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
    let tape = parse_to_tape(&src, &sm, &wm).unwrap();
    let roots = validate(&tape).unwrap();
    println!("tape: {} nodes, {} roots, {:.2} MiB tape ({} B/node) — validated ✓\n",
        tape.nodes.len(), roots,
        (tape.nodes.len() * std::mem::size_of::<Node>()) as f64 / (1 << 20) as f64,
        std::mem::size_of::<Node>());

    println!("text → tree (our SIMD lex + flat tape):");
    let ours = bench("ours: two-pass (lean lex→array→tape)", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        let toks = lex_to_array(src.as_bytes(), &sm, &wm);
        black_box(parse_from_tokens(src.as_bytes(), &toks).unwrap().nodes.len());
    });
    bench("  (fused, for comparison)", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        black_box(parse_to_tape(&src, &sm, &wm).unwrap().nodes.len());
    });

    // ── Split the fused pass: how much is lexing vs tree-building? ────────────
    let toks = materialize_tokens(&src, &sm, &wm);
    println!("\nlex/parse split ({} tokens):", toks.len());
    bench("  stage-1 (SIMD bitmaps)", bytes, iters, || {
        black_box(simd_lang::stage1::lex(src.as_bytes()));
    });
    bench("  materialize token Vec (lex)", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        black_box(materialize_tokens(&src, &sm, &wm).len());
    });
    bench("  parse-from-array → tape", bytes, iters, || {
        black_box(parse_from_tokens(src.as_bytes(), &toks).unwrap().nodes.len());
    });
    // Component costs: re-lex but skip classify_punct (Pk::Other) to bound it.
    bench("  lex w/o classify_punct", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        let b = src.as_bytes();
        let mut lx = Lex::new(b, &sm, &wm);
        let mut n = 0usize;
        loop {
            let t = lx.pull_nopk();
            let eof = t.start as usize >= b.len() && matches!(t.kind, S::Punct);
            n += 1;
            if eof { break; }
        }
        black_box(n);
    });
    // Absolute floor: iterate start bits + word_end + build a Tok, nothing else.
    bench("  skeleton (bitscan+word_end)", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        black_box(skeleton_lex(src.as_bytes(), &sm, &wm));
    });
    bench("  lean lex_to_array (real)", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        black_box(lex_to_array(src.as_bytes(), &sm, &wm).len());
    });

    // Correctness: the lean tokenizer must drive the parser to the same tape.
    let lean_toks = lex_to_array(src.as_bytes(), &sm, &wm);
    let t_fused = parse_to_tape(&src, &sm, &wm).unwrap();
    let t_lean = parse_from_tokens(src.as_bytes(), &lean_toks).unwrap();
    assert_eq!(t_fused.nodes.len(), t_lean.nodes.len(), "lean tokenizer → different tape len");


    println!("\nreference targets:");
    let ir = bench("jsir-parse: → columnar Module", bytes, iters, || {
        let m = jsir_parse::parse_to_module(&src).unwrap();
        black_box(m.op_count());
    });
    let oxc = bench("oxc: lex+parse → AST", bytes, iters, || {
        let alloc = Allocator::default();
        let ret = OxcParser::new(&alloc, &src, SourceType::default()).parse();
        black_box(ret.program.body.len());
    });
    let oxc_sem = bench("oxc: parse + semantic", bytes, iters, || {
        let alloc = Allocator::default();
        let ret = OxcParser::new(&alloc, &src, SourceType::default()).parse();
        let s = oxc_semantic::SemanticBuilder::new().build(&ret.program);
        black_box(s.semantic.nodes().len());
    });

    println!("\nours vs oxc bare AST      : {:.2}x", oxc.as_secs_f64() / ours.as_secs_f64());
    println!("ours vs oxc parse+semantic: {:.2}x", oxc_sem.as_secs_f64() / ours.as_secs_f64());
    println!("ours vs heavy JSIR Module : {:.2}x", ir.as_secs_f64() / ours.as_secs_f64());
}
