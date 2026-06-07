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

// AOT-compiled stage-1 + operator-adjacency kernel (examples/js_stage1_ops.simd).
mod aot_ops {
    #![allow(dead_code)]
    include!("../../generated/js_stage1_ops.rs");
}

/// Run the AOT `js_stage1_ops` kernel: returns (start_masks, word_masks,
/// op_ext_masks). Pads input to a multiple of 64 for whole-chunk reads.
fn aot_stage1_ops(raw: &[u8]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let mut padded = raw.to_vec();
    while padded.len() % 64 != 0 { padded.push(0); }
    let nchunks = padded.len() / 64 + 1;
    let mut sm = vec![0u64; nchunks];
    let mut wm = vec![0u64; nchunks];
    let mut oe = vec![0u64; nchunks];
    aot_ops::js_stage1_ops(&mut padded, &mut sm, &mut wm, &mut oe);
    (sm, wm, oe)
}

// ───────────────────────────── tape ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
    Member,  // a.b      — 1 child (object); span = property name
    Index,   // a[b]     — 2 children (object, index)
    Call,    // f(a,b)   — nargs = 1 callee + argc args
    Array,   // [a,b,c]  — nargs = element count
    // statements / declarations
    Block,   // { … }    — nargs = statement count
    If,      // if/else  — nargs = 2 (cond, then) or 3 (+ else)
    While,   // while    — nargs = 2 (cond, body)
    For,     // for(;;)  — nargs = 4 (init, cond, update, body); Empty fills gaps
    Return,  //          — nargs = 0 or 1
    Break,   //          — nargs = 0
    Continue,//          — nargs = 0
    Func,    // function — nargs = paramcount + 1 (body block); span = name (may be empty)
    Param,   //          — leaf; span = param name
    Empty,   //          — empty statement / missing for-clause
    Cond,    // a ? b : c — 3 children
    New,     // new X(…) — 1 child (the callee/call subtree)
    Object,  // { … }    — nargs = property count
    Property,// k: v     — 1 child (value), or 0 (shorthand); span = key
    RegexLit,// /re/flags — leaf
    Seq,     // a, b, c  — comma/sequence expression; nargs = count
    Switch,  // switch   — nargs = 1 (discriminant) + case count
    Case,    // case/default — nargs = 1 (test; Empty for default) + stmt count
    Throw,   // throw e  — 1 child
    DoWhile, // do … while — nargs = 2 (body, cond)
    Try,     // try/catch/finally — nargs = 2 or 3 (try block, catch block?, finally block?)
    Catch,   // catch(e){…} — nargs = 1 (block); span = param (may be empty)
}

/// A flat tape node — 16 bytes, no pointers. For operator nodes (`Binary`,
/// `Unary`, `Update`, `Assign`) `start..end` is the *operator* token span; for
/// leaves it is the leaf's own span. `nargs` is the RPN arity (children are the
/// preceding `nargs` completed subtrees). `flags` bit 0 = prefix.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Node {
    start: u32,
    end: u32,
    nargs: u16,
    kind: K,
    flags: u8,
}

#[derive(Default)]
struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    #[inline]
    fn push(&mut self, kind: K, nargs: u32, start: u32, end: u32, prefix: bool) {
        self.nodes.push(Node { kind, flags: prefix as u8, nargs: nargs as u16, start, end });
    }
}

// ───────────────────────────── punctuators ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pk {
    Plus, Minus, Star, Slash, Percent, Lt, Le, Gt, Ge, EqEq, EqEqEq, Ne, NeEq,
    Amp, Pipe, Caret, Shl, Shr, UShr, Bang, Tilde, PlusPlus, MinusMinus,
    Assign, PlusEq, MinusEq, StarEq, SlashEq, PercentEq, ShlEq, ShrEq, UShrEq,
    AmpEq, PipeEq, CaretEq, StarStarEq, Semi, Comma, Dot,
    AndAnd, OrOr, Coalesce, StarStar, Arrow, Question,
    OptChain, Spread, AndAndEq, OrOrEq, CoalesceEq,
    LParen, RParen, LBracket, RBracket, LBrace, RBrace, Colon, Other,
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
            b'?' => Pk::Question, b':' => Pk::Colon,
            b'(' => Pk::LParen, b')' => Pk::RParen, b'[' => Pk::LBracket, b']' => Pk::RBracket,
            b'{' => Pk::LBrace, b'}' => Pk::RBrace,
            _ => Pk::Other,
        },
        2 => match (s[0], s[1]) {
            (b'<', b'=') => Pk::Le, (b'>', b'=') => Pk::Ge, (b'=', b'=') => Pk::EqEq,
            (b'!', b'=') => Pk::Ne, (b'<', b'<') => Pk::Shl, (b'>', b'>') => Pk::Shr,
            (b'+', b'+') => Pk::PlusPlus, (b'-', b'-') => Pk::MinusMinus,
            (b'+', b'=') => Pk::PlusEq, (b'-', b'=') => Pk::MinusEq, (b'*', b'=') => Pk::StarEq,
            (b'/', b'=') => Pk::SlashEq, (b'%', b'=') => Pk::PercentEq, (b'&', b'=') => Pk::AmpEq,
            (b'|', b'=') => Pk::PipeEq, (b'^', b'=') => Pk::CaretEq,
            (b'&', b'&') => Pk::AndAnd, (b'|', b'|') => Pk::OrOr, (b'?', b'?') => Pk::Coalesce,
            (b'*', b'*') => Pk::StarStar, (b'=', b'>') => Pk::Arrow, (b'?', b'.') => Pk::OptChain,
            _ => Pk::Other,
        },
        3 => match (s[0], s[1], s[2]) {
            (b'=', b'=', b'=') => Pk::EqEqEq, (b'!', b'=', b'=') => Pk::NeEq,
            (b'>', b'>', b'>') => Pk::UShr, (b'<', b'<', b'=') => Pk::ShlEq,
            (b'>', b'>', b'=') => Pk::ShrEq, (b'*', b'*', b'=') => Pk::StarStarEq,
            (b'.', b'.', b'.') => Pk::Spread,
            (b'&', b'&', b'=') => Pk::AndAndEq, (b'|', b'|', b'=') => Pk::OrOrEq, (b'?', b'?', b'=') => Pk::CoalesceEq,
            _ => Pk::Other,
        },
        4 if s == b">>>=" => Pk::UShrEq,
        _ => Pk::Other,
    }
}

#[inline]
fn is_assign_op(p: Pk) -> bool {
    matches!(p, Pk::Assign | Pk::PlusEq | Pk::MinusEq | Pk::StarEq | Pk::SlashEq
        | Pk::PercentEq | Pk::ShlEq | Pk::ShrEq | Pk::UShrEq | Pk::AmpEq | Pk::PipeEq
        | Pk::CaretEq | Pk::StarStarEq | Pk::AndAndEq | Pk::OrOrEq | Pk::CoalesceEq)
}

#[inline]
fn binary_bp(p: Pk) -> Option<u8> {
    Some(match p {
        Pk::Coalesce | Pk::OrOr => 2,
        Pk::AndAnd => 3,
        Pk::Pipe => 4, Pk::Caret => 5, Pk::Amp => 6,
        Pk::EqEq | Pk::Ne | Pk::EqEqEq | Pk::NeEq => 7,
        Pk::Lt | Pk::Le | Pk::Gt | Pk::Ge => 8,
        Pk::Shl | Pk::Shr | Pk::UShr => 9,
        Pk::Plus | Pk::Minus => 10,
        Pk::Star | Pk::Slash | Pk::Percent => 11,
        Pk::StarStar => 12,
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

/// Fused lean token source: bit-cursor over the start bitmap + lean classify
/// (word_masks-derived ends, inline `op_len`), producing tokens on demand with
/// no intermediate array. Combines the lean tokenizer's speed with fusion (no
/// token-array traffic).
struct LeanLex<'a> {
    src: &'a [u8],
    sm: &'a [u64],
    wm: &'a [u64],
    oe: &'a [u64], // op_ext bitmap; empty → always full match_operator
    ci: usize,
    w: u64,
    cursor: usize, // start bits below this are inside an already-consumed token
    prev_ends_expr: bool, // true ⇒ a `/` here is division, not a regex
    buf: [Tok; 2],
    len: usize,
}

impl<'a> LeanLex<'a> {
    fn new(src: &'a [u8], sm: &'a [u64], wm: &'a [u64], oe: &'a [u64]) -> LeanLex<'a> {
        LeanLex {
            src, sm, wm, oe,
            ci: 0,
            w: if sm.is_empty() { 0 } else { sm[0] },
            cursor: 0,
            prev_ends_expr: false, // start of input → a leading `/` would be regex
            buf: [Tok { kind: S::Punct, punct: Pk::Other, start: 0, end: 0 }; 2],
            len: 0,
        }
    }
    #[inline]
    fn next_start(&mut self) -> Option<usize> {
        let n = self.src.len();
        loop {
            if self.w != 0 {
                let bit = self.w.trailing_zeros() as usize;
                self.w &= self.w - 1;
                let p = self.ci * 64 + bit;
                if p >= n { return None; }
                if p < self.cursor { continue; } // interior of a consumed token
                return Some(p);
            }
            self.ci += 1;
            if self.ci >= self.sm.len() { return None; }
            self.w = self.sm[self.ci];
        }
    }
    #[inline]
    fn pull(&mut self) -> Tok {
        let n = self.src.len();
        loop {
            let Some(p) = self.next_start() else {
                let n32 = n as u32;
                return Tok { kind: S::Punct, punct: Pk::Other, start: n32, end: n32 };
            };
            let wbit = (self.wm[p / 64] >> (p % 64)) & 1 == 1;
            if wbit {
                let digit = self.src[p].is_ascii_digit();
                let end = if digit { num_end(self.src, self.wm, p, n) } else { word_end(self.wm, p, n) };
                self.cursor = end;
                let kind = if digit { S::Number } else { S::Ident };
                // a number, or an identifier/keyword that ends an expression,
                // makes the next `/` division. Keyword operators don't.
                self.prev_ends_expr = digit || !regex_kw_before(&self.src[p..end]);
                return Tok { kind, punct: Pk::Other, start: p as u32, end: end as u32 };
            }
            let c = self.src[p];
            match c {
                b'.' if p + 1 < n && self.src[p + 1].is_ascii_digit() => {
                    // leading-dot number: `.5`, `.5e3`
                    let end = num_end(self.src, self.wm, p + 1, n);
                    self.cursor = end;
                    self.prev_ends_expr = true;
                    return Tok { kind: S::Number, punct: Pk::Other, start: p as u32, end: end as u32 };
                }
                b'"' | b'\'' | b'`' => {
                    let end = scan_string(self.src, p);
                    self.cursor = end;
                    self.prev_ends_expr = true;
                    return Tok { kind: S::String, punct: Pk::Other, start: p as u32, end: end as u32 };
                }
                b'/' if p + 1 < n && self.src[p + 1] == b'/' => {
                    self.cursor = memchr_nl(self.src, p + 2);
                    continue; // line comment trivia (doesn't change prev_ends_expr)
                }
                b'/' if p + 1 < n && self.src[p + 1] == b'*' => {
                    self.cursor = block_comment_end(self.src, p + 2);
                    continue; // block comment trivia
                }
                b'/' if !self.prev_ends_expr => {
                    // regex literal
                    let end = scan_regex(self.src, p);
                    self.cursor = end;
                    self.prev_ends_expr = true;
                    return Tok { kind: S::Regex, punct: Pk::Other, start: p as u32, end: end as u32 };
                }
                _ => {
                    let bit = p % 64;
                    let ext = self.oe.is_empty() || bit == 63 || (self.oe[p / 64] >> bit) & 1 == 1;
                    let end = if ext { p + op_len(self.src, p) } else { p + 1 };
                    self.cursor = end;
                    let pk = classify_punct(&self.src[p..end]);
                    // `)`/`]`/`++`/`--` end an expression → next `/` divides;
                    // everything else (operators, `(`, `,`, `{`, …) allows regex.
                    self.prev_ends_expr = matches!(pk, Pk::RParen | Pk::RBracket | Pk::PlusPlus | Pk::MinusMinus);
                    return Tok { kind: S::Punct, punct: pk, start: p as u32, end: end as u32 };
                }
            }
        }
    }
    #[inline]
    fn fill(&mut self, k: usize) {
        while self.len < k {
            let t = self.pull();
            self.buf[self.len] = t;
            self.len += 1;
        }
    }
}

impl<'a> TokSrc for LeanLex<'a> {
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
    fn at_eof(&mut self) -> bool {
        self.fill(1);
        self.buf[0].start as usize >= self.src.len()
            && matches!(self.buf[0].kind, S::Punct) && self.buf[0].punct == Pk::Other
    }
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
        if t.punct == Pk::LBrace { return self.block(); }
        if t.punct == Pk::Semi { self.bump(); self.t.push(K::Empty, 0, t.start, t.start, false); return; }
        if matches!(t.kind, S::Ident | S::Keyword) {
            match self.txt(t) {
                "let" | "var" | "const" if matches!(self.peek2().kind, S::Ident) => return self.var_decl(),
                "if" => return self.if_stmt(),
                "while" => return self.while_stmt(),
                "for" => return self.for_stmt(),
                "function" => return self.func_decl(),
                "switch" => return self.switch_stmt(),
                "throw" => { let k = self.bump(); self.expression(); self.eat(Pk::Semi); self.t.push(K::Throw, 1, k.start, k.end, false); return; }
                "do" => return self.do_while(),
                "try" => return self.try_stmt(),
                "return" => return self.return_stmt(),
                "break" => { self.bump(); self.eat(Pk::Semi); self.t.push(K::Break, 0, t.start, t.end, false); return; }
                "continue" => { self.bump(); self.eat(Pk::Semi); self.t.push(K::Continue, 0, t.start, t.end, false); return; }
                _ => {}
            }
        }
        self.expression();
        self.eat(Pk::Semi);
        self.t.push(K::ExprStmt, 1, 0, 0, false);
    }

    fn block(&mut self) {
        let lb = self.bump(); // {
        let mut n = 0u32;
        while self.peek().punct != Pk::RBrace && !self.lx.at_eof() && self.err.is_none() {
            self.statement();
            n += 1;
        }
        self.eat(Pk::RBrace);
        self.t.push(K::Block, n, lb.start, lb.start + 1, false);
    }

    fn paren_expr(&mut self) {
        if !self.eat(Pk::LParen) {
            self.err = Some(format!("expected ( at {}", self.peek().start));
            return;
        }
        self.expression();
        if !self.eat(Pk::RParen) {
            self.err = Some(format!("expected ) at {}", self.peek().start));
        }
    }

    fn if_stmt(&mut self) {
        let kt = self.bump(); // if
        self.paren_expr(); // condition
        self.statement(); // then
        let mut nargs = 2;
        let e = self.peek();
        if matches!(e.kind, S::Ident | S::Keyword) && self.txt(e) == "else" {
            self.bump();
            self.statement();
            nargs = 3;
        }
        self.t.push(K::If, nargs, kt.start, kt.end, false);
    }

    fn while_stmt(&mut self) {
        let kt = self.bump();
        self.paren_expr();
        self.statement();
        self.t.push(K::While, 2, kt.start, kt.end, false);
    }

    fn for_stmt(&mut self) {
        let kt = self.bump();
        if !self.eat(Pk::LParen) { self.err = Some(format!("expected ( at {}", self.peek().start)); return; }
        // init
        let it = self.peek();
        if it.punct == Pk::Semi {
            self.bump();
            self.t.push(K::Empty, 0, it.start, it.start, false);
        } else if matches!(it.kind, S::Ident | S::Keyword) && matches!(self.txt(it), "let" | "var" | "const") {
            self.var_decl(); // consumes the first `;`
        } else {
            self.assignment();
            self.eat(Pk::Semi);
            self.t.push(K::ExprStmt, 1, 0, 0, false);
        }
        // cond
        let ct = self.peek();
        if ct.punct == Pk::Semi { self.t.push(K::Empty, 0, ct.start, ct.start, false); } else { self.expression(); }
        self.eat(Pk::Semi);
        // update
        let ut = self.peek();
        if ut.punct == Pk::RParen { self.t.push(K::Empty, 0, ut.start, ut.start, false); } else { self.expression(); }
        if !self.eat(Pk::RParen) { self.err = Some(format!("expected ) at {}", self.peek().start)); return; }
        self.statement(); // body
        self.t.push(K::For, 4, kt.start, kt.end, false);
    }

    fn do_while(&mut self) {
        let kt = self.bump(); // do
        self.statement(); // body
        let w = self.peek();
        if !(matches!(w.kind, S::Ident | S::Keyword) && self.txt(w) == "while") {
            self.err = Some(format!("expected while at {}", w.start));
            return;
        }
        self.bump();
        self.paren_expr();
        self.eat(Pk::Semi);
        self.t.push(K::DoWhile, 2, kt.start, kt.end, false);
    }

    fn try_stmt(&mut self) {
        let kt = self.bump(); // try
        if self.peek().punct != Pk::LBrace { self.err = Some(format!("expected {{ at {}", self.peek().start)); return; }
        self.block();
        let mut nargs = 1u32;
        let c = self.peek();
        if matches!(c.kind, S::Ident | S::Keyword) && self.txt(c) == "catch" {
            self.bump();
            let (ps, pe) = if self.peek().punct == Pk::LParen {
                self.bump();
                let pt = self.peek();
                let span = if matches!(pt.kind, S::Ident | S::Keyword) { self.bump(); (pt.start, pt.end) } else { (pt.start, pt.start) };
                self.eat(Pk::RParen);
                span
            } else { (c.start, c.start) };
            if self.peek().punct != Pk::LBrace { self.err = Some(format!("expected {{ at {}", self.peek().start)); return; }
            self.block();
            self.t.push(K::Catch, 1, ps, pe, false);
            nargs += 1;
        }
        let f = self.peek();
        if matches!(f.kind, S::Ident | S::Keyword) && self.txt(f) == "finally" {
            self.bump();
            if self.peek().punct != Pk::LBrace { self.err = Some(format!("expected {{ at {}", self.peek().start)); return; }
            self.block();
            nargs += 1;
        }
        self.t.push(K::Try, nargs, kt.start, kt.end, false);
    }

    fn switch_stmt(&mut self) {
        let kt = self.bump(); // switch
        self.paren_expr(); // discriminant
        if !self.eat(Pk::LBrace) { self.err = Some(format!("expected {{ at {}", self.peek().start)); return; }
        let mut ncases = 0u32;
        while self.peek().punct != Pk::RBrace && !self.lx.at_eof() && self.err.is_none() {
            let ct = self.peek();
            let is_case = matches!(ct.kind, S::Ident | S::Keyword) && self.txt(ct) == "case";
            let is_default = matches!(ct.kind, S::Ident | S::Keyword) && self.txt(ct) == "default";
            if !is_case && !is_default {
                self.err = Some(format!("expected case/default at {}", ct.start));
                return;
            }
            self.bump(); // case|default
            if is_case { self.expression(); } else { self.t.push(K::Empty, 0, ct.start, ct.start, false); }
            self.eat(Pk::Colon);
            // statements until the next case/default/}
            let mut nstmt = 0u32;
            loop {
                let p = self.peek();
                if p.punct == Pk::RBrace || self.lx.at_eof() || self.err.is_some() { break; }
                if matches!(p.kind, S::Ident | S::Keyword) && matches!(self.txt(p), "case" | "default") { break; }
                self.statement();
                nstmt += 1;
            }
            self.t.push(K::Case, 1 + nstmt, ct.start, ct.end, false);
            ncases += 1;
        }
        self.eat(Pk::RBrace);
        self.t.push(K::Switch, 1 + ncases, kt.start, kt.end, false);
    }

    fn return_stmt(&mut self) {
        let kt = self.bump();
        let t = self.peek();
        if t.punct == Pk::Semi || t.punct == Pk::RBrace || self.lx.at_eof() {
            self.eat(Pk::Semi);
            self.t.push(K::Return, 0, kt.start, kt.end, false);
        } else {
            self.expression();
            self.eat(Pk::Semi);
            self.t.push(K::Return, 1, kt.start, kt.end, false);
        }
    }

    fn func_decl(&mut self) {
        let _kt = self.bump(); // function
        let nt = self.peek();
        let (ns, ne) = if matches!(nt.kind, S::Ident | S::Keyword) { self.bump(); (nt.start, nt.end) } else { (0, 0) };
        let np = self.param_list();
        if self.err.is_some() { return; }
        if self.peek().punct != Pk::LBrace {
            self.err = Some(format!("expected function body {{ at {}", self.peek().start));
            return;
        }
        self.block();
        self.t.push(K::Func, np + 1, ns, ne, false);
    }

    /// Object literal `{ k: v, k2, [c]: v, "s": v, m() {…} }` (basic forms).
    fn object(&mut self) {
        let lb = self.bump(); // {
        let mut n = 0u32;
        while self.peek().punct != Pk::RBrace && !self.lx.at_eof() && self.err.is_none() {
            let kt = self.peek();
            // computed key [expr]
            if kt.punct == Pk::LBracket {
                self.bump();
                self.assignment();
                self.eat(Pk::RBracket);
                if self.eat(Pk::Colon) { self.assignment(); self.t.push(K::Property, 2, kt.start, kt.start + 1, false); }
                else { self.t.push(K::Property, 1, kt.start, kt.start + 1, false); }
            } else {
                // ident / string / number key
                if !matches!(kt.kind, S::Ident | S::Keyword | S::String | S::Number) {
                    self.err = Some(format!("expected property key at {}", kt.start));
                    return;
                }
                self.bump();
                if self.eat(Pk::Colon) {
                    self.assignment();
                    self.t.push(K::Property, 1, kt.start, kt.end, false);
                } else if self.peek().punct == Pk::LParen {
                    // method shorthand: key(params){body}
                    let np = self.param_list();
                    if self.peek().punct == Pk::LBrace { self.block(); } else { self.err = Some("expected method body".into()); return; }
                    self.t.push(K::Func, np + 1, kt.start, kt.end, false);
                    self.t.push(K::Property, 1, kt.start, kt.end, false);
                } else {
                    // shorthand { x }
                    self.t.push(K::Property, 0, kt.start, kt.end, false);
                }
            }
            n += 1;
            if !self.eat(Pk::Comma) { break; }
        }
        self.eat(Pk::RBrace);
        self.t.push(K::Object, n, lb.start, lb.start + 1, false);
    }

    /// Function expression in expression position (`var f = function(a){…}`).
    fn func_expr(&mut self) {
        let kt = self.bump(); // function
        let nt = self.peek();
        let (ns, ne) = if matches!(nt.kind, S::Ident | S::Keyword) && nt.punct == Pk::Other {
            self.bump(); (nt.start, nt.end)
        } else { (kt.start, kt.start) };
        let np = self.param_list();
        if self.err.is_some() { return; }
        if self.peek().punct != Pk::LBrace {
            self.err = Some(format!("expected function body {{ at {}", self.peek().start));
            return;
        }
        self.block();
        self.t.push(K::Func, np + 1, ns, ne, false);
    }

    /// `( ident, ident, … )` — pushes a `Param` leaf per parameter; returns count.
    fn param_list(&mut self) -> u32 {
        if !self.eat(Pk::LParen) {
            self.err = Some(format!("expected ( at {}", self.peek().start));
            return 0;
        }
        let mut np = 0u32;
        if self.peek().punct != Pk::RParen {
            loop {
                let pt = self.peek();
                if !matches!(pt.kind, S::Ident | S::Keyword) {
                    self.err = Some(format!("expected parameter at {}", pt.start));
                    return np;
                }
                self.bump();
                self.t.push(K::Param, 0, pt.start, pt.end, false);
                np += 1;
                if !self.eat(Pk::Comma) { break; }
            }
        }
        if !self.eat(Pk::RParen) {
            self.err = Some(format!("expected ) at {}", self.peek().start));
        }
        np
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

    /// Full expression including the comma/sequence operator. Used where a comma
    /// is *not* a separator (expression statements, grouping, return, for-clauses).
    fn expression(&mut self) {
        self.assignment();
        let mut n = 1u32;
        while self.peek().punct == Pk::Comma {
            // stop at a trailing comma before a closer
            let after = self.peek2().punct;
            if matches!(after, Pk::RParen | Pk::RBracket | Pk::RBrace | Pk::Semi) { break; }
            self.bump();
            self.assignment();
            n += 1;
        }
        if n > 1 {
            self.t.push(K::Seq, n, 0, 0, false);
        }
    }

    fn assignment(&mut self) {
        // LHS (a full expression incl. member/index chains); `=` isn't a binary op.
        self.binary(0);
        let t = self.peek();
        if is_assign_op(t.punct) {
            let ot = self.bump();
            self.assignment(); // right-associative
            self.t.push(K::Assign, 2, ot.start, ot.end, false);
        } else if t.punct == Pk::Question {
            // ternary  cond ? a : b
            let qt = self.bump();
            self.assignment(); // consequent
            if !self.eat(Pk::Colon) {
                self.err = Some(format!("expected : at {}", self.peek().start));
                return;
            }
            self.assignment(); // alternate
            self.t.push(K::Cond, 3, qt.start, qt.start + 1, false);
        }
    }

    fn binary(&mut self, min_bp: u8) {
        self.unary();
        loop {
            let t = self.peek();
            // keyword binary operators `in` / `instanceof` (relational precedence)
            let bp = if let Some(bp) = binary_bp(t.punct) {
                bp
            } else if matches!(t.kind, S::Ident | S::Keyword) && matches!(self.txt(t), "in" | "instanceof") {
                8
            } else {
                break;
            };
            if bp < min_bp { break; }
            let ot = self.bump();
            self.binary(bp + 1);
            self.t.push(K::Binary, 2, ot.start, ot.end, false);
        }
    }

    fn unary(&mut self) {
        let t = self.peek();
        // keyword prefix unary operators
        if matches!(t.kind, S::Ident | S::Keyword) && matches!(self.txt(t), "typeof" | "void" | "delete" | "await") {
            self.bump();
            self.unary();
            self.t.push(K::Unary, 1, t.start, t.end, true);
            return;
        }
        match t.punct {
            Pk::Minus | Pk::Plus | Pk::Bang | Pk::Tilde => {
                self.bump();
                self.unary();
                self.t.push(K::Unary, 1, t.start, t.end, true);
                return;
            }
            Pk::PlusPlus | Pk::MinusMinus => {
                self.bump();
                self.unary();
                self.t.push(K::Update, 1, t.start, t.end, true);
                return;
            }
            _ => {}
        }
        self.postfix();
    }

    /// Postfix chains: `.member`, `[index]`, `(call args)`, `expr++`/`--`.
    fn postfix(&mut self) {
        self.primary();
        loop {
            let t = self.peek();
            match t.punct {
                Pk::Dot => {
                    self.bump();
                    let nt = self.peek();
                    if !matches!(nt.kind, S::Ident | S::Keyword) {
                        self.err = Some(format!("expected property name at {}", nt.start));
                        return;
                    }
                    self.bump();
                    self.t.push(K::Member, 1, nt.start, nt.end, false);
                }
                Pk::OptChain => {
                    // a?.b / a?.[i] / a?.(args) — record optionality in flags bit 1
                    self.bump();
                    match self.peek().punct {
                        Pk::LParen => { let lp = self.bump(); let argc = self.arg_list(Pk::RParen); self.t.push(K::Call, 1 + argc, lp.start, lp.start + 1, false); }
                        Pk::LBracket => { let lb = self.bump(); self.assignment(); self.eat(Pk::RBracket); self.t.push(K::Index, 2, lb.start, lb.start + 1, false); }
                        _ => {
                            let nt = self.peek();
                            if !matches!(nt.kind, S::Ident | S::Keyword) { self.err = Some(format!("expected property after ?. at {}", nt.start)); return; }
                            self.bump();
                            let id = self.t.nodes.len();
                            self.t.push(K::Member, 1, nt.start, nt.end, false);
                            self.t.nodes[id].flags |= 2; // optional
                        }
                    }
                }
                Pk::LBracket => {
                    self.bump();
                    self.assignment();
                    if !self.eat(Pk::RBracket) {
                        self.err = Some(format!("expected ] at {}", self.peek().start));
                        return;
                    }
                    self.t.push(K::Index, 2, t.start, t.start + 1, false);
                }
                Pk::LParen => {
                    self.bump();
                    let argc = self.arg_list(Pk::RParen);
                    self.t.push(K::Call, 1 + argc, t.start, t.start + 1, false);
                }
                Pk::PlusPlus | Pk::MinusMinus => {
                    self.bump();
                    self.t.push(K::Update, 1, t.start, t.end, false);
                }
                _ => break,
            }
            if self.err.is_some() { return; }
        }
    }

    /// Comma-separated `assignment` expressions until `close`; returns the count.
    fn arg_list(&mut self, close: Pk) -> u32 {
        if self.peek().punct == close {
            self.bump();
            return 0;
        }
        let mut n = 0u32;
        loop {
            let sp = self.peek();
            if sp.punct == Pk::Spread {
                self.bump();
                self.assignment();
                self.t.push(K::Unary, 1, sp.start, sp.start + 3, true); // spread as a unary
            } else if sp.punct == Pk::Comma {
                // elision in arrays: `[a, , b]`
                self.t.push(K::Empty, 0, sp.start, sp.start, false);
            } else {
                self.assignment();
            }
            n += 1;
            if self.err.is_some() { return n; }
            if !self.eat(Pk::Comma) { break; }
            if self.peek().punct == close { break; } // trailing comma
        }
        if !self.eat(close) {
            self.err = Some(format!("expected closing delimiter at {}", self.peek().start));
        }
        n
    }

    fn primary(&mut self) {
        let t = self.peek();
        match t.kind {
            S::Number => {
                self.bump();
                let kind = if self.src[t.end as usize - 1] == b'n' { K::BigIntLit } else { K::NumLit };
                self.t.push(kind, 0, t.start, t.end, false);
            }
            S::String | S::TemplateNoSub | S::TemplateHead | S::TemplateMiddle | S::TemplateTail => {
                self.bump();
                self.t.push(K::StrLit, 0, t.start, t.end, false);
            }
            S::Regex => {
                self.bump();
                self.t.push(K::RegexLit, 0, t.start, t.end, false);
            }
            S::Ident | S::Keyword => {
                let name = self.txt(t);
                match name {
                    "true" | "false" => { self.bump(); self.t.push(K::BoolLit, 0, t.start, t.end, false); }
                    "null" => { self.bump(); self.t.push(K::NullLit, 0, t.start, t.end, false); }
                    "function" => self.func_expr(),
                    "new" => { self.bump(); self.postfix(); self.t.push(K::New, 1, t.start, t.end, false); }
                    _ => { self.bump(); self.t.push(K::Ident, 0, t.start, t.end, false); }
                }
            }
            _ => match t.punct {
                Pk::LParen => {
                    // grouping — transparent (precedence already captured)
                    self.bump();
                    self.expression();
                    if !self.eat(Pk::RParen) {
                        self.err = Some(format!("expected ) at {}", self.peek().start));
                    }
                }
                Pk::LBracket => {
                    self.bump();
                    let n = self.arg_list(Pk::RBracket);
                    self.t.push(K::Array, n, t.start, t.start + 1, false);
                }
                Pk::LBrace => self.object(),
                _ => {
                    self.err = Some(format!("unexpected token at {}", t.start));
                    self.bump(); // avoid infinite loop
                }
            },
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

/// Fused parse using the lean bit-cursor tokenizer — no token array, no slow
/// `Lexer` classify. The best of both: lean classification + fusion.
fn parse_to_tape_lean(src: &str, sm: &[u64], wm: &[u64]) -> Result<Tape, String> {
    let b = src.as_bytes();
    let mut p = P {
        src: b,
        lx: LeanLex::new(b, sm, wm, &[]),
        t: Tape { nodes: Vec::with_capacity(b.len() / 3 + 8) },
        err: None,
    };
    p.program();
    if let Some(e) = p.err.take() { return Err(e); }
    Ok(p.t)
}

/// Fused-lean parse using the AOT `op_ext` bitmap to skip `match_operator` for
/// standalone operators — fusion (no array) + structural operator shortcut.
fn parse_to_tape_lean_ops(src: &str, sm: &[u64], wm: &[u64], oe: &[u64]) -> Result<Tape, String> {
    let b = src.as_bytes();
    let mut p = P {
        src: b,
        lx: LeanLex::new(b, sm, wm, oe),
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

/// End of a numeric literal whose digit run starts at `start` (a digit, possibly
/// just past a leading `.`). Uses the word bitmap for the mantissa (so hex/bigint/
/// `1e5` are free) and extends across a decimal point and a signed exponent.
#[inline]
fn num_end(src: &[u8], wm: &[u64], start: usize, n: usize) -> usize {
    let mut end = word_end(wm, start, n);
    if end < n && src[end] == b'.' {
        end = word_end(wm, end + 1, n).max(end + 1);
    }
    if end + 1 < n && matches!(src[end], b'+' | b'-') && end > 0 && matches!(src[end - 1], b'e' | b'E') {
        end = word_end(wm, end + 1, n).max(end + 1);
    }
    end
}

/// Scan a quoted string/template starting at the quote `src[p]`; returns the
/// index just past the closing quote. (Templates with `${}` are treated as a
/// single string for now — a known subset limitation.)
#[inline]
fn scan_string(src: &[u8], p: usize) -> usize {
    let q = src[p];
    let n = src.len();
    let mut i = p + 1;
    while i < n && src[i] != q {
        i += if src[i] == b'\\' { 2 } else { 1 };
    }
    (i + 1).min(n)
}

/// Scan a regex literal starting at `/` at `src[p]`; returns index past flags.
#[inline]
fn scan_regex(src: &[u8], p: usize) -> usize {
    let n = src.len();
    let mut i = p + 1;
    let mut in_class = false;
    while i < n {
        match src[i] {
            b'\\' => { i += 2; continue; }
            b'[' => in_class = true,
            b']' => in_class = false,
            b'/' if !in_class => { i += 1; break; }
            b'\n' => break, // unterminated — bail
            _ => {}
        }
        i += 1;
    }
    while i < n && (src[i].is_ascii_alphanumeric() || src[i] == b'_') { i += 1; } // flags
    i.min(n)
}

/// Keywords after which a `/` begins a regex (not division).
#[inline]
fn regex_kw_before(w: &[u8]) -> bool {
    matches!(w, b"return" | b"typeof" | b"instanceof" | b"in" | b"of" | b"new"
        | b"delete" | b"void" | b"do" | b"else" | b"yield" | b"await" | b"case" | b"throw")
}

/// End of a line comment (the `\n`, or EOF).
#[inline]
fn memchr_nl(src: &[u8], from: usize) -> usize {
    let mut i = from;
    while i < src.len() && src[i] != b'\n' { i += 1; }
    i
}

/// End of a block comment body (just past `*/`, or EOF).
#[inline]
fn block_comment_end(src: &[u8], from: usize) -> usize {
    let n = src.len();
    let mut i = from;
    while i + 1 < n && !(src[i] == b'*' && src[i + 1] == b'/') { i += 1; }
    (i + 2).min(n)
}

/// Maximal-munch operator length (1–4) — same table as the shared lexer.
#[inline]
fn op_len(src: &[u8], p: usize) -> usize {
    let n = src.len();
    let b0 = src[p];
    if !matches!(b0, b'>' | b'<' | b'=' | b'!' | b'*' | b'&' | b'|' | b'+' | b'-' | b'%' | b'^' | b'/' | b'?' | b'.') {
        return 1; // ( ) [ ] { } ; , : ~ and any single-char punct
    }
    let g = |o: usize| if p + o < n { src[p + o] } else { 0 };
    let (b1, b2, b3) = (g(1), g(2), g(3));
    match b0 {
        b'>' => match (b1, b2, b3) { (b'>', b'>', b'=') => 4, (b'>', b'>', _) => 3, (b'>', b'=', _) => 3, (b'>', _, _) => 2, (b'=', _, _) => 2, _ => 1 },
        b'<' => match (b1, b2) { (b'<', b'=') => 3, (b'<', _) => 2, (b'=', _) => 2, _ => 1 },
        b'=' => match (b1, b2) { (b'=', b'=') => 3, (b'=', _) => 2, (b'>', _) => 2, _ => 1 },
        b'!' => match (b1, b2) { (b'=', b'=') => 3, (b'=', _) => 2, _ => 1 },
        b'*' => match (b1, b2) { (b'*', b'=') => 3, (b'*', _) => 2, (b'=', _) => 2, _ => 1 },
        b'&' => match (b1, b2) { (b'&', b'=') => 3, (b'&', _) => 2, (b'=', _) => 2, _ => 1 }, // &&=, &&, &=
        b'|' => match (b1, b2) { (b'|', b'=') => 3, (b'|', _) => 2, (b'=', _) => 2, _ => 1 }, // ||=, ||, |=
        b'?' => match (b1, b2) { (b'?', b'=') => 3, (b'?', _) => 2, (b'.', _) => 2, _ => 1 }, // ??=, ??, ?.
        b'.' => if b1 == b'.' && b2 == b'.' { 3 } else { 1 }, // ...
        b'/' => match b1 { b'=' => 2, _ => 1 }, // /=  (comments/regex handled before op_len)
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
    let mut cursor = 0usize; // skip start bits already inside a consumed token
    for (ci, &w0) in sm.iter().enumerate() {
        let mut w = w0;
        while w != 0 {
            let p = ci * 64 + w.trailing_zeros() as usize;
            w &= w - 1;
            if p >= n { break; }
            if p < cursor { continue; }
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
                cursor = end;
                let kind = if digit { S::Number } else { S::Ident };
                out.push(Tok { kind, punct: Pk::Other, start: p as u32, end: end as u32 });
            } else {
                let c = src[p];
                match c {
                    b'"' | b'\'' | b'`' => {
                        let end = scan_string(src, p);
                        cursor = end;
                        out.push(Tok { kind: S::String, punct: Pk::Other, start: p as u32, end: end as u32 });
                    }
                    b'/' if p + 1 < n && src[p + 1] == b'/' => { cursor = memchr_nl(src, p + 2); continue; }
                    b'/' if p + 1 < n && src[p + 1] == b'*' => { cursor = block_comment_end(src, p + 2); continue; }
                    _ => {
                        let end = p + op_len(src, p);
                        cursor = end;
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

/// Like `lex_to_array` but operator length comes from a precomputed per-byte
/// array (what a `.simd` kernel would emit) instead of the scalar `op_len`.
fn lex_to_array_pre(src: &[u8], sm: &[u64], wm: &[u64], oplen: &[u8]) -> Vec<Tok> {
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
                let mut end = word_end(wm, p, n);
                let digit = src[p].is_ascii_digit();
                if digit && end < n && src[end] == b'.' {
                    end = word_end(wm, end + 1, n).max(end + 1);
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
                    b'/' if p + 1 < n && (src[p + 1] == b'/' || src[p + 1] == b'*') => continue,
                    _ => {
                        let end = p + oplen[p] as usize;
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

/// `lex_to_array` driven by the SIMD `op_ext` bitmap: operator length comes from
/// `match_operator` ONLY where `op_ext` says two op chars are adjacent (or at a
/// 64-byte boundary, where op_ext can't see across); elsewhere a punctuator is a
/// single byte. Produces byte-identical tokens to `lex_to_array`.
fn lex_to_array_ops(src: &[u8], sm: &[u64], wm: &[u64], oe: &[u64]) -> Vec<Tok> {
    let n = src.len();
    let mut out = Vec::with_capacity(n / 4 + 8);
    let mut cursor = 0usize;
    for (ci, &w0) in sm.iter().enumerate() {
        let mut w = w0;
        while w != 0 {
            let bit = w.trailing_zeros() as usize;
            let p = ci * 64 + bit;
            w &= w - 1;
            if p >= n { break; }
            if p < cursor { continue; }
            let wbit = (wm[p / 64] >> (p % 64)) & 1 == 1;
            if wbit {
                let mut end = word_end(wm, p, n);
                let digit = src[p].is_ascii_digit();
                if digit && end < n && src[end] == b'.' {
                    end = word_end(wm, end + 1, n).max(end + 1);
                }
                cursor = end;
                let kind = if digit { S::Number } else { S::Ident };
                out.push(Tok { kind, punct: Pk::Other, start: p as u32, end: end as u32 });
            } else {
                let c = src[p];
                match c {
                    b'"' | b'\'' | b'`' => {
                        let end = scan_string(src, p);
                        cursor = end;
                        out.push(Tok { kind: S::String, punct: Pk::Other, start: p as u32, end: end as u32 });
                    }
                    b'/' if p + 1 < n && src[p + 1] == b'/' => { cursor = memchr_nl(src, p + 2); continue; }
                    b'/' if p + 1 < n && src[p + 1] == b'*' => { cursor = block_comment_end(src, p + 2); continue; }
                    _ => {
                        // op_ext bit set (or chunk-boundary byte) → maybe multi-char.
                        let ext = (oe[p / 64] >> bit) & 1 == 1 || bit == 63;
                        let end = if ext { p + op_len(src, p) } else { p + 1 };
                        cursor = end;
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

/// Diagnostic: same as `lex_to_array` but operators are length-1 with no
/// classification — bounds how much the operator path costs.
fn lex_to_array_noop(src: &[u8], sm: &[u64], wm: &[u64]) -> usize {
    let n = src.len();
    let mut count = 0usize;
    for (ci, &w0) in sm.iter().enumerate() {
        let mut w = w0;
        while w != 0 {
            let p = ci * 64 + w.trailing_zeros() as usize;
            w &= w - 1;
            if p >= n { break; }
            let wbit = (wm[p / 64] >> (p % 64)) & 1 == 1;
            if wbit {
                let _end = word_end(wm, p, n);
                count += 1;
            } else {
                count += 1; // pretend single-char punct, no classify
            }
        }
    }
    count
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

/// Verify RPN balance under the unified model: every node pops `nargs` subtrees
/// and pushes one (so an expression *and* a statement are both single subtrees).
/// A well-formed program ends with one stack item per top-level statement and
/// never underflows. Returns the top-level statement count.
fn validate(t: &Tape) -> Result<usize, String> {
    let mut stack: i64 = 0;
    for (i, n) in t.nodes.iter().enumerate() {
        if (n.nargs as i64) > stack {
            return Err(format!("node {i} {:?} wants {} children, stack={stack}", n.kind, n.nargs));
        }
        stack -= n.nargs as i64;
        stack += 1;
    }
    if stack < 0 {
        return Err(format!("unbalanced tape: stack={stack}"));
    }
    Ok(stack as usize)
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
            K::Member => format!("Member(.{})", span(src, n.start, n.end)),
            K::Index => "Index[]".into(),
            K::Call => format!("Call({} args)", n.nargs.saturating_sub(1)),
            K::Array => format!("Array({} elems)", n.nargs),
            K::Block => format!("Block({} stmts)", n.nargs),
            K::If => if n.nargs == 3 { "If/else".into() } else { "If".into() },
            K::While => "While".into(),
            K::For => "For".into(),
            K::Return => "Return".into(),
            K::Break => "Break".into(),
            K::Continue => "Continue".into(),
            K::Func => format!("Func({})", span(src, n.start, n.end)),
            K::Param => format!("Param({})", span(src, n.start, n.end)),
            K::Empty => "Empty".into(),
            K::Cond => "Cond(?:)".into(),
            K::New => "New".into(),
            K::Object => format!("Object({} props)", n.nargs),
            K::Property => format!("Property({})", span(src, n.start, n.end)),
            K::RegexLit => format!("Regex({})", span(src, n.start, n.end)),
            K::Seq => format!("Seq({})", n.nargs),
            K::Switch => "Switch".into(),
            K::Case => "Case".into(),
            K::Throw => "Throw".into(),
            K::DoWhile => "DoWhile".into(),
            K::Try => "Try".into(),
            K::Catch => format!("Catch({})", span(src, n.start, n.end)),
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
    let mut s = String::with_capacity(stmts * 44);
    for i in 0..stmts {
        let j = i.saturating_sub(1);
        match i % 4 {
            0 => s.push_str(&format!("var v{i} = {i} + {a} * 3 - 1;\n", a = i * 2)),
            1 => s.push_str(&format!("v{j} = v{j} + {i} * 2;\n")),
            2 => s.push_str(&format!("var w{i} = {a} & 7 | {i};\n", a = i + 3)),
            // nested structure: call, member, index, array, grouping
            _ => s.push_str(&format!("var r{i} = f(v{j}, w{j}.x) + arr[{i}] * (v{j} - 1);\n")),
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
    // Lex-check mode: validate the lean SIMD-fed tokenizer against the
    // established (round-trip + oxc-aligned) `simd_lang::js` lexer on a real file.
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("--lex-check") {
        let path = argv.get(2).expect("usage: tape --lex-check <path.js>");
        let src = std::fs::read(path).expect("read");
        // reference: the validated lexer (handles regex/templates), trivia dropped
        let (sm, wm) = simd_lang::stage1::lex(&src);
        let mut refs: Vec<(usize, usize)> = Vec::new();
        simd_lang::js::drive(&src, &sm, &wm, |t| {
            if !matches!(t.kind, S::LineComment | S::BlockComment) {
                refs.push((t.start, t.end));
            }
        });
        // ours: lean tokenizer (drop the EOF sentinel)
        let mut lx = LeanLex::new(&src, &sm, &wm, &[]);
        let mut mine: Vec<(usize, usize)> = Vec::new();
        loop {
            let t = lx.pull();
            if t.start as usize >= src.len() { break; }
            mine.push((t.start as usize, t.end as usize));
        }
        let mut ok = 0usize;
        let mut firstbad = None;
        for i in 0..refs.len().min(mine.len()) {
            if refs[i] == mine[i] { ok += 1; } else { firstbad = Some(i); break; }
        }
        println!("{path}: {} ref tokens, {} ours; {ok} matched", refs.len(), mine.len());
        if let Some(i) = firstbad {
            let (rs, re) = refs[i];
            let (ms, me) = mine[i];
            println!("  first divergence at token {i}: ref [{rs}..{re}]={:?}  ours [{ms}..{me}]={:?}",
                std::str::from_utf8(&src[rs..re.min(src.len())]).unwrap_or("?"),
                std::str::from_utf8(&src[ms..me.min(src.len())]).unwrap_or("?"));
        } else if refs.len() == mine.len() {
            println!("  ✓ identical token stream ({} tokens)", ok);
        }
        return;
    }

    // File mode: `tape --file <path.js>` parses a real file and reports coverage.
    if argv.get(1).map(String::as_str) == Some("--file") {
        let path = argv.get(2).expect("usage: tape --file <path.js>");
        let src = std::fs::read_to_string(path).expect("read");
        let (rsm, rwm) = simd_lang::stage1::lex(src.as_bytes());
        let r_old = parse_to_tape(&src, &rsm, &rwm).map(|t| t.nodes.len());
        let r_lean = parse_to_tape_lean(&src, &rsm, &rwm).map(|t| t.nodes.len());
        println!("[diag] old-lexer: {r_old:?}   lean(no-ops): {r_lean:?}");
        let (sm, wm, oe) = aot_stage1_ops(src.as_bytes());
        match parse_to_tape_lean_ops(&src, &sm, &wm, &oe) {
            Ok(t) => {
                let roots = validate(&t).unwrap_or(0);
                println!("{path}: {} bytes → {} tape nodes, {} top-level stmts ✓", src.len(), t.nodes.len(), roots);
            }
            Err(e) => {
                // show context around the failure
                println!("{path}: {} bytes — parse stopped: {e}", src.len());
                if let Some(byte) = e.rsplit(' ').next().and_then(|s| s.trim_end_matches('.').parse::<usize>().ok()) {
                    let lo = byte.saturating_sub(40);
                    let hi = (byte + 40).min(src.len());
                    println!("  …{}…", &src[lo..hi].replace('\n', "⏎"));
                    println!("  {}^ here", " ".repeat(byte.saturating_sub(lo)));
                }
            }
        }
        return;
    }

    // Torture string: multi-char ops, strings, comments, member/index — must
    // tokenize identically across all three paths (old Lexer / lean / lean+op_ext).
    {
        let tc = "a >>>= b; x === y; p <= q; r !== s; t = \"a+b/c\"; // cmt <=\n u = /* x */ v ** 2; o.m[i](1, 2);";
        let (sm, wm) = simd_lang::stage1::lex(tc.as_bytes());
        let (asm, awm, aoe) = aot_stage1_ops(tc.as_bytes());
        let a = parse_to_tape(tc, &sm, &wm).expect("old").nodes.len();
        let b = parse_to_tape_lean(tc, &sm, &wm).expect("lean").nodes.len();
        let c = parse_to_tape_lean_ops(tc, &asm, &awm, &aoe).expect("lean+ops").nodes.len();
        assert!(a == b && b == c, "torture: tokenizer paths disagree (old={a} lean={b} ops={c})");
    }

    // Tiny demo first — show the tape reconstructs a real tree.
    let demo = "function sum(a, b) { var t = 0; for (var i = 0; i < a; i = i + 1) { t = t + b[i]; } if (t > 10) return t; else return -1; }";
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

    // Correctness: the fused-lean path must match the fused-Lexer tape.
    {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        let a = parse_to_tape(&src, &sm, &wm).unwrap();
        let c = parse_to_tape_lean(&src, &sm, &wm).unwrap();
        assert_eq!(a.nodes.len(), c.nodes.len(), "fused-lean tape len differs");
        let (asm, awm, aoe) = aot_stage1_ops(src.as_bytes());
        let d = parse_to_tape_lean_ops(&src, &asm, &awm, &aoe).unwrap();
        assert_eq!(a.nodes.len(), d.nodes.len(), "fused-lean+op_ext tape len differs");
        for (x, y) in a.nodes.iter().zip(d.nodes.iter()) {
            assert!(x.kind == y.kind && x.start == y.start && x.end == y.end, "op_ext node mismatch");
        }
    }

    // Correctness: AOT op_ext bitmaps must equal pure-Rust stage1, and the
    // op_ext-driven tokenizer must produce byte-identical tokens.
    {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        let (asm, awm, aoe) = aot_stage1_ops(src.as_bytes());
        assert_eq!(&sm[..], &asm[..wm.len()], "AOT start_masks differ");
        assert_eq!(&wm[..], &awm[..wm.len()], "AOT word_masks differ");
        let plain = lex_to_array(src.as_bytes(), &sm, &wm);
        let viaops = lex_to_array_ops(src.as_bytes(), &asm, &awm, &aoe);
        assert_eq!(plain.len(), viaops.len(), "op_ext token count differs");
        for (a, b) in plain.iter().zip(viaops.iter()) {
            assert!(a.start == b.start && a.end == b.end && a.punct == b.punct,
                "op_ext token mismatch at {}", a.start);
        }
        let _ = aoe;
    }

    println!("text → tree (our SIMD lex + flat tape):");
    let ours = bench("ours: two-pass (lean→array→tape)", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        let toks = lex_to_array(src.as_bytes(), &sm, &wm);
        black_box(parse_from_tokens(src.as_bytes(), &toks).unwrap().nodes.len());
    });
    let ours_ops = bench("ours+op_ext: AOT 3-mask → array → tape", bytes, iters, || {
        let (sm, wm, oe) = aot_stage1_ops(src.as_bytes());
        let toks = lex_to_array_ops(src.as_bytes(), &sm, &wm, &oe);
        black_box(parse_from_tokens(src.as_bytes(), &toks).unwrap().nodes.len());
    });
    let ours_lean = bench("  fused-lean (no array)", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        black_box(parse_to_tape_lean(&src, &sm, &wm).unwrap().nodes.len());
    });
    let ours_lean_ops = bench("ours: fused-lean + op_ext (AOT)", bytes, iters, || {
        let (sm, wm, oe) = aot_stage1_ops(src.as_bytes());
        black_box(parse_to_tape_lean_ops(&src, &sm, &wm, &oe).unwrap().nodes.len());
    });
    bench("  fused-Lexer (old)", bytes, iters, || {
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
    bench("  lean w/o op_len+classify_punct", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        black_box(lex_to_array_noop(src.as_bytes(), &sm, &wm));
    });
    // Ceiling if a `.simd` kernel precomputed per-byte op-length (untimed here).
    let mut oplen = vec![0u8; src.len() + 4];
    for p in 0..src.len() { oplen[p] = op_len(src.as_bytes(), p) as u8; }
    bench("  lean w/ precomputed op_len", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        black_box(lex_to_array_pre(src.as_bytes(), &sm, &wm, &oplen).len());
    });

    // Correctness: the lean tokenizer must drive the parser to the same tape.
    let lean_toks = lex_to_array(src.as_bytes(), &sm, &wm);
    let t_fused = parse_to_tape(&src, &sm, &wm).unwrap();
    let t_lean = parse_from_tokens(src.as_bytes(), &lean_toks).unwrap();
    assert_eq!(t_fused.nodes.len(), t_lean.nodes.len(), "lean tokenizer → different tape len");


    println!("\nreference targets:");
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

    // Best of our paths (fusion vs two-pass trade off with grammar depth).
    let best = ours.min(ours_ops).min(ours_lean).min(ours_lean_ops);
    println!("\nours two-pass         vs oxc bare AST : {:.2}x", oxc.as_secs_f64() / ours.as_secs_f64());
    println!("ours two-pass+op_ext  vs oxc bare AST : {:.2}x", oxc.as_secs_f64() / ours_ops.as_secs_f64());
    println!("ours fused-lean       vs oxc bare AST : {:.2}x", oxc.as_secs_f64() / ours_lean.as_secs_f64());
    println!("ours fused-lean+op_ext vs oxc bare AST: {:.2}x", oxc.as_secs_f64() / ours_lean_ops.as_secs_f64());
    println!("BEST                  vs oxc bare AST : {:.2}x", oxc.as_secs_f64() / best.as_secs_f64());
    println!("BEST                  vs oxc+semantic : {:.2}x", oxc_sem.as_secs_f64() / best.as_secs_f64());
}
