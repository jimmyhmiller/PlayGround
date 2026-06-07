//! A no-AST front end: **text → tokens → JSIR**, straight into the columnar
//! [`jsir_ir::Module`] via [`jsir_ir::IrBuild`]. No syntax tree is ever built;
//! the recursive-descent/Pratt parser's semantic actions emit ops as it climbs,
//! returning each sub-expression's [`ValueId`] up the recursion. SSA value
//! numbering is left to the printer (it renumbers `%N` by traversal), so the
//! parser only has to emit ops in post-order — which it does naturally.
//!
//! Scope: a subset of JS — literals (number/string/bool/null/bigint),
//! identifiers, unary/binary/assignment/update expressions, expression
//! statements, and `let`/`var`/`const` declarations. Constructs that print
//! locations/scopes (member/object/function/import) or use the `jshir` region
//! dialect (logical/conditional) are intentionally excluded; for this subset the
//! byte-exact output carries no trivia at all.
//!
//! The token contract mirrors the simd-lang lexer (`(kind, start, end)` byte
//! spans + `text`), so the real SIMD `Lexer` can be swapped in for [`tokenize`]
//! without touching the parser.

use jsir_ir::{Attr, AttrSpec, BlockId, IrBuild, Module, OpId, ValueId};

// ───────────────────────────── tokens ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokKind {
    Ident,
    Number,
    Str,
    Punct,
    Eof,
}

/// Integer punctuator kind, classified once at tokenize time so the hot parse
/// loop dispatches on an int instead of re-matching operator strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pk {
    None,
    // binary / relational / bitwise
    Plus, Minus, Star, Slash, Percent, Lt, Le, Gt, Ge,
    EqEq, EqEqEq, Ne, NeEq, Amp, Pipe, Caret, Shl, Shr, UShr,
    AndAnd, OrOr, Coalesce,
    // unary / update
    Bang, Tilde, PlusPlus, MinusMinus,
    // assignment
    Assign, PlusEq, MinusEq, StarEq, SlashEq, PercentEq, ShlEq, ShrEq, UShrEq,
    AmpEq, PipeEq, CaretEq, StarStarEq,
    // structural
    Semi, Comma, LParen, RParen, LBrace, RBrace, LBracket, RBracket, Dot,
    Question, Colon,
    Other,
}

#[inline]
fn classify_punct(s: &str) -> Pk {
    match s {
        "+" => Pk::Plus, "-" => Pk::Minus, "*" => Pk::Star, "/" => Pk::Slash,
        "%" => Pk::Percent, "<" => Pk::Lt, "<=" => Pk::Le, ">" => Pk::Gt, ">=" => Pk::Ge,
        "==" => Pk::EqEq, "===" => Pk::EqEqEq, "!=" => Pk::Ne, "!==" => Pk::NeEq,
        "&" => Pk::Amp, "|" => Pk::Pipe, "^" => Pk::Caret, "<<" => Pk::Shl, ">>" => Pk::Shr,
        ">>>" => Pk::UShr, "&&" => Pk::AndAnd, "||" => Pk::OrOr, "??" => Pk::Coalesce,
        "!" => Pk::Bang, "~" => Pk::Tilde, "++" => Pk::PlusPlus, "--" => Pk::MinusMinus,
        "=" => Pk::Assign, "+=" => Pk::PlusEq, "-=" => Pk::MinusEq, "*=" => Pk::StarEq,
        "/=" => Pk::SlashEq, "%=" => Pk::PercentEq, "<<=" => Pk::ShlEq, ">>=" => Pk::ShrEq,
        ">>>=" => Pk::UShrEq, "&=" => Pk::AmpEq, "|=" => Pk::PipeEq, "^=" => Pk::CaretEq,
        "**=" => Pk::StarStarEq, ";" => Pk::Semi, "," => Pk::Comma, "(" => Pk::LParen,
        ")" => Pk::RParen, "{" => Pk::LBrace, "}" => Pk::RBrace, "[" => Pk::LBracket,
        "]" => Pk::RBracket, "." => Pk::Dot, "?" => Pk::Question, ":" => Pk::Colon,
        _ => Pk::Other,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Token {
    pub kind: TokKind,
    pub punct: Pk,
    pub start: usize,
    pub end: usize,
}

impl Token {
    pub fn text<'a>(&self, src: &'a str) -> &'a str {
        &src[self.start..self.end]
    }
}

/// Punctuators, longest first, so the scanner does maximal munch.
const PUNCTS: &[&str] = &[
    ">>>=", "===", "!==", "**=", "<<=", ">>=", ">>>", "...", "==", "!=", "<=", ">=", "&&", "||",
    "??", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "**", "<<", ">>", "(", ")",
    "{", "}", "[", "]", ";", ",", ".", "?", ":", "=", "+", "-", "*", "/", "%", "<", ">", "!", "~",
    "&", "|", "^",
];

/// A token's source byte range as `(start, end): (u32, u32)` — what `AttrSpec`'s
/// source-span values want.
#[inline]
fn o(t: Token) -> (u32, u32) {
    (t.start as u32, t.end as u32)
}

fn is_word_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b == b'$'
}
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// Tokenize with the **SIMD lexer** (`simd-lang`): NEON stage-1 bitmaps feed the
/// stage-2 pull lexer. The result is mapped to this crate's [`Token`] contract
/// (identical shape) and capped at the supported subset (no templates/regex).
/// This is the default path used by [`parse_to_module`].
pub fn tokenize_simd(src: &str) -> Result<Vec<Token>, String> {
    use simd_lang::js::TokKind as S;
    let bytes = src.as_bytes();
    let (start_masks, word_masks) = simd_lang::stage1::lex(bytes);
    // Single pass: `drive` streams tokens into our vec directly — no intermediate
    // `Vec<simd Token>` materialization.
    let mut out = Vec::with_capacity(bytes.len() / 4 + 2);
    let mut err: Option<String> = None;
    // Parser-mode: we classify keywords ourselves from the identifier text, so
    // the lexer shouldn't re-read identifier bodies to label keywords.
    simd_lang::js::drive_pm(bytes, &start_masks, &word_masks, |t| {
        if err.is_some() {
            return;
        }
        let (kind, punct) = match t.kind {
            S::Ident | S::Keyword => (TokKind::Ident, Pk::None),
            S::Number => (TokKind::Number, Pk::None),
            S::String => (TokKind::Str, Pk::None),
            S::Punct => (TokKind::Punct, classify_punct(&src[t.start..t.end])),
            S::LineComment | S::BlockComment => return, // trivia
            other => {
                err = Some(format!("token kind {other:?} is out of the supported subset"));
                return;
            }
        };
        out.push(Token { kind, punct, start: t.start, end: t.end });
    });
    if let Some(e) = err {
        return Err(e);
    }
    out.push(Token { kind: TokKind::Eof, punct: Pk::None, start: bytes.len(), end: bytes.len() });
    Ok(out)
}

/// Scalar reference tokenizer over the subset (kept for comparison/fallback).
/// Returns spans into `src`.
pub fn tokenize(src: &str) -> Result<Vec<Token>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        // whitespace
        if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
            i += 1;
            continue;
        }
        // line / block comments
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
            continue;
        }
        // strings (no escapes needed for the subset, but tolerate \x)
        if c == b'\'' || c == b'"' {
            let start = i;
            i += 1;
            while i < b.len() && b[i] != c {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1; // closing quote
            out.push(Token { kind: TokKind::Str, punct: Pk::None, start, end: i });
            continue;
        }
        // numbers (incl. trailing `n` bigint)
        if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let start = i;
            while i < b.len()
                && (b[i].is_ascii_alphanumeric() || b[i] == b'.' || b[i] == b'_')
            {
                i += 1;
            }
            out.push(Token { kind: TokKind::Number, punct: Pk::None, start, end: i });
            continue;
        }
        // identifiers / keywords
        if is_word_start(c) {
            let start = i;
            while i < b.len() && is_word(b[i]) {
                i += 1;
            }
            out.push(Token { kind: TokKind::Ident, punct: Pk::None, start, end: i });
            continue;
        }
        // punctuators
        if let Some(p) = PUNCTS.iter().find(|p| src[i..].starts_with(**p)) {
            out.push(Token { kind: TokKind::Punct, punct: classify_punct(p), start: i, end: i + p.len() });
            i += p.len();
            continue;
        }
        return Err(format!("unexpected byte {:?} at {}", c as char, i));
    }
    out.push(Token { kind: TokKind::Eof, punct: Pk::None, start: b.len(), end: b.len() });
    Ok(out)
}

// ───────────────────────────── builder ─────────────────────────────

/// Emits ops into a [`Module`] via [`IrBuild`], tracking a fresh-`ValueId`
/// counter and a stack of block-op accumulators (set-once: a block's op list is
/// recorded with `set_block_ops` once its accumulator is complete).
struct Builder {
    m: Module,
    next_val: u32,
    stack: Vec<Vec<OpId>>,
    noemit: bool, // profiling: when set, skip Module writes to isolate parse cost
}

impl Builder {
    fn new() -> Builder {
        Builder {
            m: Module::new(),
            next_val: 0,
            stack: vec![Vec::new()],
            noemit: std::env::var_os("NOEMIT").is_some(),
        }
    }
    fn fresh(&mut self) -> ValueId {
        let v = ValueId(self.next_val);
        self.next_val += 1;
        v
    }
    fn push_op(&mut self, id: OpId) {
        self.stack.last_mut().unwrap().push(id);
    }
    /// Emit an op with one result; pushes it to the current block, returns the
    /// result. Attrs come as zero-allocation [`AttrSpec`]s (keys are `&'static`,
    /// string values borrow the source) — interned straight into the arena.
    fn expr(&mut self, name: &str, operands: &[ValueId], attrs: &[AttrSpec]) -> ValueId {
        if self.noemit {
            return self.fresh();
        }
        let id = self.m.add_op(name);
        self.m.set_operands(id, operands);
        self.m.set_attrs_spec(id, attrs);
        let v = self.fresh();
        self.m.set_results(id, &[v]);
        self.push_op(id);
        v
    }
    /// Emit a result-less op (statement / region terminator).
    fn stmt(&mut self, name: &str, operands: &[ValueId], attrs: &[AttrSpec]) -> OpId {
        if self.noemit {
            return OpId(0);
        }
        let id = self.m.add_op(name);
        self.m.set_operands(id, operands);
        self.m.set_attrs_spec(id, attrs);
        self.m.set_results(id, &[]);
        self.push_op(id);
        id
    }
    /// Build a single-block region from a freshly-accumulated op list.
    fn region_from(&mut self, ops: &[OpId]) -> jsir_ir::traits::RegionId {
        let block = self.m.add_block(BlockId(0));
        self.m.set_block_args(block, &[]);
        self.m.set_block_ops(block, ops);
        let region = self.m.add_region();
        self.m.set_region_blocks(region, &[block]);
        region
    }
}

// ───────────────────────────── pull lexer ─────────────────────────────

/// A pull token source over the SIMD lexer with a 2-token lookahead buffer — so
/// the parser never materializes a `Vec<Token>` (oxc-style: tokens are produced
/// on demand and only the 1–2 the parser is looking at are live).
struct Lex<'a> {
    inner: simd_lang::js::Lexer<'a>,
    src: &'a str,
    buf: [Token; 2],
    len: usize,
    err: Option<String>,
}

impl<'a> Lex<'a> {
    fn new(src: &'a str, start_masks: &'a [u64], word_masks: &'a [u64]) -> Lex<'a> {
        Lex {
            inner: simd_lang::js::Lexer::new(src.as_bytes(), start_masks, word_masks),
            src,
            buf: [Token { kind: TokKind::Eof, punct: Pk::None, start: 0, end: 0 }; 2],
            len: 0,
            err: None,
        }
    }
    /// Pull the next non-comment token, mapped to our `Token` (parser-mode).
    fn pull(&mut self) -> Token {
        use simd_lang::js::TokKind as S;
        loop {
            match self.inner.next_tok::<false>() {
                Some(t) => {
                    let (kind, punct) = match t.kind {
                        S::Ident | S::Keyword => (TokKind::Ident, Pk::None),
                        S::Number => (TokKind::Number, Pk::None),
                        S::String => (TokKind::Str, Pk::None),
                        S::Punct => (TokKind::Punct, classify_punct(&self.src[t.start..t.end])),
                        S::LineComment | S::BlockComment => continue,
                        other => {
                            if self.err.is_none() {
                                self.err = Some(format!("token kind {other:?} out of subset"));
                            }
                            return self.eof();
                        }
                    };
                    return Token { kind, punct, start: t.start, end: t.end };
                }
                None => return self.eof(),
            }
        }
    }
    fn eof(&self) -> Token {
        let n = self.src.len();
        Token { kind: TokKind::Eof, punct: Pk::None, start: n, end: n }
    }
    #[inline]
    fn fill_to(&mut self, n: usize) {
        while self.len < n {
            let t = self.pull();
            self.buf[self.len] = t;
            self.len += 1;
        }
    }
}

// ───────────────────────────── parser ─────────────────────────────

struct Parser<'a> {
    src: &'a str,
    lx: Lex<'a>,
    b: Builder,
}

impl<'a> Parser<'a> {
    fn peek(&mut self) -> Token {
        self.lx.fill_to(1);
        self.lx.buf[0]
    }
    fn peek2(&mut self) -> Token {
        self.lx.fill_to(2);
        self.lx.buf[1]
    }
    fn bump(&mut self) -> Token {
        self.lx.fill_to(1);
        let t = self.lx.buf[0];
        self.lx.buf[0] = self.lx.buf[1];
        self.lx.len -= 1;
        t
    }
    fn txt(&self, t: Token) -> &'a str {
        t.text(self.src)
    }
    fn eat_punct(&mut self, p: Pk) -> bool {
        if self.peek().punct == p {
            self.bump();
            true
        } else {
            false
        }
    }
    fn program(&mut self) -> Result<(), String> {
        while self.peek().kind != TokKind::Eof {
            self.statement()?;
        }
        Ok(())
    }

    fn statement(&mut self) -> Result<(), String> {
        let t = self.peek();
        if t.kind == TokKind::Ident {
            match self.txt(t) {
                "let" | "var" | "const" if self.peek2().kind == TokKind::Ident => {
                    return self.var_decl();
                }
                _ => {}
            }
        }
        // expression statement
        let v = self.assignment()?;
        self.eat_punct(Pk::Semi);
        self.b.stmt("jsir.expression_statement", &[v], &[]);
        Ok(())
    }

    fn var_decl(&mut self) -> Result<(), String> {
        let kt = self.bump(); // let|var|const keyword token
        self.b.stack.push(Vec::new()); // region block accumulator
        let mut decl_vals = Vec::new();
        loop {
            let nt = self.ident_name()?;
            let id_ref = self.b.expr("jsir.identifier_ref", &[], &[AttrSpec::Str("name", o(nt).0, o(nt).1)]);
            let declr = if self.eat_punct(Pk::Assign) {
                let init = self.assignment()?;
                self.b.expr("jsir.variable_declarator", &[id_ref, init], &[])
            } else {
                self.b.expr("jsir.variable_declarator", &[id_ref], &[])
            };
            decl_vals.push(declr);
            if !self.eat_punct(Pk::Comma) {
                break;
            }
        }
        self.eat_punct(Pk::Semi);
        self.b.stmt("jsir.exprs_region_end", &decl_vals, &[]);
        let region_ops = self.b.stack.pop().unwrap();
        let region = self.b.region_from(&region_ops);
        let op = self.b.m.add_op("jsir.variable_declaration");
        self.b.m.set_attrs_spec(op, &[AttrSpec::Str("kind", o(kt).0, o(kt).1)]);
        self.b.m.set_results(op, &[]);
        self.b.m.set_regions(op, &[region]);
        self.b.push_op(op);
        Ok(())
    }

    fn ident_name(&mut self) -> Result<Token, String> {
        let t = self.peek();
        if t.kind != TokKind::Ident {
            return Err(format!("expected identifier at byte {}", t.start));
        }
        Ok(self.bump())
    }

    /// Assignment is the lowest-precedence, right-associative expression. The
    /// LHS of an assignment (a bare identifier in this subset) is emitted as an
    /// `identifier_ref` rather than a read `identifier`.
    fn assignment(&mut self) -> Result<ValueId, String> {
        let t = self.peek();
        if t.kind == TokKind::Ident && is_assign_op(self.peek2().punct) {
            // only when it's not a keyword-literal
            if !matches!(self.txt(t), "true" | "false" | "null") {
                let nt = self.bump(); // the lvalue identifier
                let ot = self.bump(); // = += -= ...
                let target =
                    self.b.expr("jsir.identifier_ref", &[], &[AttrSpec::Str("name", o(nt).0, o(nt).1)]);
                let value = self.assignment()?;
                return Ok(self.b.expr(
                    "jsir.assignment_expression",
                    &[target, value],
                    &[AttrSpec::Str("operator_", o(ot).0, o(ot).1)],
                ));
            }
        }
        self.binary(0)
    }

    fn binary(&mut self, min_bp: u8) -> Result<ValueId, String> {
        let mut left = self.unary()?;
        loop {
            let t = self.peek();
            let Some(bp) = binary_bp(t.punct) else { break };
            if bp < min_bp {
                break;
            }
            let ot = self.bump();
            let right = self.binary(bp + 1)?; // left-associative
            left = self.b.expr(
                "jsir.binary_expression",
                &[left, right],
                &[AttrSpec::Str("operator_", o(ot).0, o(ot).1)],
            );
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<ValueId, String> {
        let t = self.peek();
        match t.punct {
            Pk::Minus | Pk::Plus | Pk::Bang | Pk::Tilde => {
                self.bump();
                let arg = self.unary()?;
                return Ok(self.b.expr(
                    "jsir.unary_expression",
                    &[arg],
                    &[AttrSpec::Str("operator_", o(t).0, o(t).1), AttrSpec::Bool("prefix", true)],
                ));
            }
            Pk::PlusPlus | Pk::MinusMinus => {
                self.bump();
                let nt = self.ident_name()?; // prefix update target is an lvalue
                let target =
                    self.b.expr("jsir.identifier_ref", &[], &[AttrSpec::Str("name", o(nt).0, o(nt).1)]);
                return Ok(self.b.expr(
                    "jsir.update_expression",
                    &[target],
                    &[AttrSpec::Str("operator_", o(t).0, o(t).1), AttrSpec::Bool("prefix", true)],
                ));
            }
            _ => {}
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<ValueId, String> {
        let t = self.peek();
        match t.kind {
            TokKind::Number => {
                self.bump();
                let raw = self.txt(t);
                let (s, e) = o(t);
                if let Some(digits) = raw.strip_suffix('n') {
                    // bigint: value/raw_value = digits (raw minus the trailing `n`)
                    let _ = digits;
                    return Ok(self.b.expr(
                        "jsir.big_int_literal",
                        &[],
                        &[AttrSpec::BigExtra("extra", s, e, s, e - 1), AttrSpec::Str("value", s, e - 1)],
                    ));
                }
                let value: f64 = raw.parse().map_err(|_| format!("bad number {raw}"))?;
                Ok(self.b.expr(
                    "jsir.numeric_literal",
                    &[],
                    &[AttrSpec::NumExtra("extra", s, e, value), AttrSpec::F64("value", value)],
                ))
            }
            TokKind::Str => {
                self.bump();
                let (s, e) = o(t);
                // strip the quotes (no escapes in subset): inner = [s+1, e-1)
                Ok(self.b.expr(
                    "jsir.string_literal",
                    &[],
                    &[AttrSpec::StrExtra("extra", s, e, s + 1, e - 1), AttrSpec::Str("value", s + 1, e - 1)],
                ))
            }
            TokKind::Ident => {
                let name = self.txt(t);
                match name {
                    "true" | "false" => {
                        let v = name == "true";
                        self.bump();
                        Ok(self.b.expr("jsir.boolean_literal", &[], &[AttrSpec::Bool("value", v)]))
                    }
                    "null" => {
                        self.bump();
                        Ok(self.b.expr("jsir.null_literal", &[], &[]))
                    }
                    _ => {
                        self.bump();
                        let (s, e) = o(t);
                        // postfix update: the identifier is then an lvalue.
                        if matches!(self.peek().punct, Pk::PlusPlus | Pk::MinusMinus) {
                            let ot = self.bump();
                            let target = self
                                .b
                                .expr("jsir.identifier_ref", &[], &[AttrSpec::Str("name", s, e)]);
                            return Ok(self.b.expr(
                                "jsir.update_expression",
                                &[target],
                                &[AttrSpec::Str("operator_", o(ot).0, o(ot).1), AttrSpec::Bool("prefix", false)],
                            ));
                        }
                        Ok(self.b.expr("jsir.identifier", &[], &[AttrSpec::Str("name", s, e)]))
                    }
                }
            }
            _ => Err(format!("unexpected token {:?} at byte {}", t.kind, t.start)),
        }
    }
}

fn is_assign_op(pk: Pk) -> bool {
    matches!(
        pk,
        Pk::Assign
            | Pk::PlusEq
            | Pk::MinusEq
            | Pk::StarEq
            | Pk::SlashEq
            | Pk::PercentEq
            | Pk::ShlEq
            | Pk::ShrEq
            | Pk::UShrEq
            | Pk::AmpEq
            | Pk::PipeEq
            | Pk::CaretEq
            | Pk::StarStarEq
    )
}

/// Binary binding power (higher binds tighter). Logical/`??` are excluded (they
/// use the `jshir` region dialect, out of this subset).
fn binary_bp(pk: Pk) -> Option<u8> {
    Some(match pk {
        Pk::Pipe => 4,
        Pk::Caret => 5,
        Pk::Amp => 6,
        Pk::EqEq | Pk::Ne | Pk::EqEqEq | Pk::NeEq => 7,
        Pk::Lt | Pk::Le | Pk::Gt | Pk::Ge => 8,
        Pk::Shl | Pk::Shr | Pk::UShr => 9,
        Pk::Plus | Pk::Minus => 10,
        Pk::Star | Pk::Slash | Pk::Percent => 11,
        _ => return None,
    })
}

/// Parse a subset-JS program straight into a columnar [`Module`] — no AST and no
/// materialized token stream (the parser pulls tokens from the SIMD lexer).
pub fn parse_to_module(src: &str) -> Result<Module, String> {
    let bytes = src.as_bytes();
    let (start_masks, word_masks) = simd_lang::stage1::lex(bytes);
    parse_with_masks(src, &start_masks, &word_masks)
}

fn parse_with_masks<'a>(
    src: &'a str,
    start_masks: &'a [u64],
    word_masks: &'a [u64],
) -> Result<Module, String> {
    let mut b = Builder::new();
    b.m.reserve(src.len() / 3 + 8, src.len()); // estimate; no reallocation during build
    b.m.set_source(src); // source-span attr values index into this owned copy
    let mut p = Parser { src, lx: Lex::new(src, start_masks, word_masks), b };
    p.program()?;
    if let Some(e) = p.lx.err.take() {
        return Err(e);
    }

    // Wrap: jsir.file { jsir.program { <body> }, { <empty directives> } }.
    let body_ops = p.b.stack.pop().unwrap();
    let body_region = p.b.region_from(&body_ops);
    let dir_region = p.b.region_from(&[]); // empty directives block → prints `^bb0:`

    let program = p.b.m.add_op("jsir.program");
    // `source_type = "script"` is a constant, not a source substring — owned path.
    p.b.m.set_attrs(program, &[("source_type".to_string(), Attr::Str("script".to_string()))]);
    p.b.m.set_results(program, &[]);
    p.b.m.set_regions(program, &[body_region, dir_region]);

    let file_region = p.b.region_from(&[program]);
    let file = p.b.m.add_op("jsir.file");
    // `comments = []` (empty array) — built once, via the owned path.
    p.b.m.set_attrs(file, &[("comments".to_string(), Attr::Array(vec![]))]);
    p.b.m.set_results(file, &[]);
    p.b.m.set_regions(file, &[file_region]);
    p.b.m.set_root(file);

    Ok(p.b.m)
}

/// Convenience: parse + print the IR text.
pub fn parse_to_ir_text(src: &str) -> Result<String, String> {
    Ok(parse_to_module(src)?.print())
}
