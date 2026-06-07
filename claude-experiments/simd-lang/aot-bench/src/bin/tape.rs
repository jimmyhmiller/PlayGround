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

#[inline]
fn classify_punct(s: &str) -> Pk {
    match s {
        "+" => Pk::Plus, "-" => Pk::Minus, "*" => Pk::Star, "/" => Pk::Slash,
        "%" => Pk::Percent, "<" => Pk::Lt, "<=" => Pk::Le, ">" => Pk::Gt, ">=" => Pk::Ge,
        "==" => Pk::EqEq, "===" => Pk::EqEqEq, "!=" => Pk::Ne, "!==" => Pk::NeEq,
        "&" => Pk::Amp, "|" => Pk::Pipe, "^" => Pk::Caret, "<<" => Pk::Shl, ">>" => Pk::Shr,
        ">>>" => Pk::UShr, "!" => Pk::Bang, "~" => Pk::Tilde, "++" => Pk::PlusPlus,
        "--" => Pk::MinusMinus, "=" => Pk::Assign, "+=" => Pk::PlusEq, "-=" => Pk::MinusEq,
        "*=" => Pk::StarEq, "/=" => Pk::SlashEq, "%=" => Pk::PercentEq, "<<=" => Pk::ShlEq,
        ">>=" => Pk::ShrEq, ">>>=" => Pk::UShrEq, "&=" => Pk::AmpEq, "|=" => Pk::PipeEq,
        "^=" => Pk::CaretEq, "**=" => Pk::StarStarEq, ";" => Pk::Semi, "," => Pk::Comma,
        "." => Pk::Dot, _ => Pk::Other,
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
                        // SAFETY: punct tokens are ASCII operator bytes.
                        classify_punct(unsafe {
                            std::str::from_utf8_unchecked(&self.src[t.start..t.end])
                        })
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

// ───────────────────────────── parser ─────────────────────────────

struct P<'a> {
    src: &'a [u8],
    lx: Lex<'a>,
    t: Tape,
    err: Option<String>,
}

impl<'a> P<'a> {
    #[inline]
    fn peek(&mut self) -> Tok { self.lx.fill(1); self.lx.buf[0] }
    #[inline]
    fn peek2(&mut self) -> Tok { self.lx.fill(2); self.lx.buf[1] }
    #[inline]
    fn bump(&mut self) -> Tok {
        self.lx.fill(1);
        let t = self.lx.buf[0];
        self.lx.buf[0] = self.lx.buf[1];
        self.lx.len -= 1;
        t
    }
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
        while !self.lx.eof() && self.err.is_none() {
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
    let ours = bench("ours: SIMD lex → flat tape", bytes, iters, || {
        let (sm, wm) = simd_lang::stage1::lex(src.as_bytes());
        let t = parse_to_tape(&src, &sm, &wm).unwrap();
        black_box(t.nodes.len());
    });

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
