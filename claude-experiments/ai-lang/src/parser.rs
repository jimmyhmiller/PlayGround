//! Recursive-descent parser: token stream → surface AST.
//!
//! Grammar implemented for v1:
//!
//! ```text
//! module      := def*
//! def         := "def" "local"? ident "(" params? ")" "->" type "=" expr
//! params      := param ("," param)* ","?
//! param       := ident ":" type
//! type        := ident                       -- only named types for v1
//! expr        := or_expr
//! or_expr     := and_expr ("||" and_expr)*
//! and_expr    := cmp_expr ("&&" cmp_expr)*
//! cmp_expr    := add_expr (cmp_op add_expr)? -- non-associative
//! add_expr    := mul_expr (("+"|"-") mul_expr)*
//! mul_expr    := unary    (("*"|"/"|"%") unary)*
//! unary       := ("-"|"!") unary | call
//! call        := atom ("(" args? ")")*
//! atom        := int | true | false | str | ident | "(" expr ")"
//! cmp_op      := "=="|"!="|"<"|"<="|">"|">="
//! ```
//!
//! Precedence (lowest → highest): `||`, `&&`, comparison, `+`/`-`,
//! `*`/`/`/`%`, unary `-`/`!`, function-call postfix, atom.
//! Comparison is non-associative — `a < b < c` is a parse error.
//! Calls are left-associative postfix: `f(x)(y)` is `(f(x))(y)`.

use crate::lexer::{LexError, Span, Token, TokenKind, lex};
use crate::surface::{
    BinOp, Module, SurfaceDef, SurfaceDefKind, SurfaceExpr, SurfaceMatchArm, SurfacePattern,
    SurfaceStmt, SurfaceType, UnaryOp,
};

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    Lex(LexError),
    Unexpected {
        expected: String,
        found: String,
        span: Span,
    },
    NonAssocComparison {
        span: Span,
    },
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseError::Lex(e) => write!(f, "lex error: {}", e),
            ParseError::Unexpected {
                expected, found, ..
            } => write!(f, "expected {}, found `{}`", expected, found),
            ParseError::NonAssocComparison { .. } => write!(
                f,
                "comparison operators are not associative; parenthesise: `(a < b) < c`"
            ),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<LexError> for ParseError {
    fn from(e: LexError) -> Self {
        ParseError::Lex(e)
    }
}

// =============================================================================
// Entry points
// =============================================================================

pub fn parse_module(src: &str) -> Result<Module, ParseError> {
    let tokens = lex(src)?;
    let mut p = Parser::new(&tokens);
    let m = p.parse_module()?;
    p.expect_eof()?;
    Ok(m)
}

pub fn parse_def(src: &str) -> Result<SurfaceDef, ParseError> {
    let tokens = lex(src)?;
    let mut p = Parser::new(&tokens);
    let d = p.parse_def()?;
    p.expect_eof()?;
    Ok(d)
}

pub fn parse_expr(src: &str) -> Result<SurfaceExpr, ParseError> {
    let tokens = lex(src)?;
    let mut p = Parser::new(&tokens);
    let e = p.parse_expr()?;
    p.expect_eof()?;
    Ok(e)
}

// =============================================================================
// Parser
// =============================================================================

struct Parser<'t> {
    tokens: &'t [Token],
    pos: usize,
    /// When `false`, `Ident { … }` does NOT parse as a struct literal —
    /// the `{` is left to the surrounding construct. Set false while
    /// parsing the scrutinee of `match`/`if`/`while`-style constructs
    /// where the trailing `{` opens a block, not a struct literal.
    allow_struct_lit: bool,
}

impl<'t> Parser<'t> {
    fn new(tokens: &'t [Token]) -> Self {
        Parser {
            tokens,
            pos: 0,
            allow_struct_lit: true,
        }
    }

    // ---- Cursor primitives ----

    fn peek(&self) -> &Token {
        // INVARIANT: lex always appends an Eof token, so this never panics
        // until we've stepped past it (which the parser doesn't do).
        &self.tokens[self.pos]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    /// Look ahead `n` tokens (0 = current). Saturates at the final Eof.
    fn peek_nth_kind(&self, n: usize) -> &TokenKind {
        let i = (self.pos + n).min(self.tokens.len() - 1);
        &self.tokens[i].kind
    }

    fn bump(&mut self) -> &Token {
        let t = &self.tokens[self.pos];
        if !matches!(t.kind, TokenKind::Eof) {
            self.pos += 1;
        }
        t
    }

    fn check(&self, k: &TokenKind) -> bool {
        core::mem::discriminant(self.peek_kind()) == core::mem::discriminant(k)
    }

    fn eat(&mut self, k: &TokenKind) -> bool {
        if self.check(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, k: &TokenKind, label: &str) -> Result<&Token, ParseError> {
        if self.check(k) {
            Ok(self.bump())
        } else {
            let found = self.peek();
            Err(ParseError::Unexpected {
                expected: label.to_owned(),
                found: found.kind.to_string(),
                span: found.span,
            })
        }
    }

    fn expect_ident(&mut self, label: &str) -> Result<(String, Span), ParseError> {
        let t = self.peek().clone();
        match t.kind {
            TokenKind::Ident(name) => {
                self.bump();
                Ok((name, t.span))
            }
            _ => Err(ParseError::Unexpected {
                expected: label.to_owned(),
                found: t.kind.to_string(),
                span: t.span,
            }),
        }
    }

    fn expect_eof(&mut self) -> Result<(), ParseError> {
        if matches!(self.peek_kind(), TokenKind::Eof) {
            Ok(())
        } else {
            let t = self.peek();
            Err(ParseError::Unexpected {
                expected: "end of input".to_owned(),
                found: t.kind.to_string(),
                span: t.span,
            })
        }
    }

    // ---- Module ----

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        let mut defs = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::Eof) {
            // `extern "C" lib "x" { fn ... }` expands to several extern
            // defs; everything else is exactly one def.
            if matches!(self.peek_kind(), TokenKind::Extern)
                && matches!(self.peek_nth_kind(1), TokenKind::Str(_))
            {
                self.parse_extern_c_block(&mut defs)?;
            } else {
                defs.push(self.parse_def()?);
            }
        }
        Ok(Module { defs })
    }

    fn parse_def(&mut self) -> Result<SurfaceDef, ParseError> {
        match self.peek_kind() {
            TokenKind::Def => self.parse_fn_def(),
            TokenKind::Struct => self.parse_struct_def(),
            TokenKind::Enum => self.parse_enum_def(),
            TokenKind::Extern => self.parse_extern_def(),
            TokenKind::State => self.parse_state_def(),
            _ => {
                let t = self.peek();
                Err(ParseError::Unexpected {
                    expected: "`def`, `struct`, `enum`, `extern`, or `state`".to_owned(),
                    found: t.kind.to_string(),
                    span: t.span,
                })
            }
        }
    }

    /// `state name: type = init` — a node-resident singleton binding.
    fn parse_state_def(&mut self) -> Result<SurfaceDef, ParseError> {
        let start = self.peek().span.start;
        self.expect(&TokenKind::State, "`state`")?;
        let (name, _) = self.expect_ident("state binding name")?;
        self.expect(&TokenKind::Colon, "`:`")?;
        let ty = self.parse_type()?;
        self.expect(&TokenKind::Eq, "`=`")?;
        let init = self.parse_expr()?;
        let end = init.span().end;
        Ok(SurfaceDef {
            name,
            span: Span { start, end },
            kind: SurfaceDefKind::State { ty, init },
        })
    }

    /// `extern fn name(params) -> ret` — no body. Declares a
    /// runtime-provided binding.
    fn parse_extern_def(&mut self) -> Result<SurfaceDef, ParseError> {
        let start = self.peek().span.start;
        self.expect(&TokenKind::Extern, "`extern`")?;
        self.expect(&TokenKind::Fn, "`fn`")?;
        let (name, _) = self.expect_ident("extern function name")?;
        self.expect(&TokenKind::LParen, "`(`")?;
        let params = self.parse_params()?;
        let close_end = self.expect(&TokenKind::RParen, "`)`")?.span.end;
        self.expect(&TokenKind::Arrow, "`->`")?;
        let ret = self.parse_type()?;
        let end = ret.span().end.max(close_end);
        Ok(SurfaceDef {
            name,
            span: Span { start, end },
            kind: SurfaceDefKind::Extern {
                params,
                ret,
                library: None,
                variadic: false,
            },
        })
    }

    /// `extern "C" lib "<libname>" { fn name(params) -> ret  ... }` — a
    /// block of real C function declarations resolved from a shared
    /// library at link time. Pushes one `Extern` def per `fn`, each
    /// tagged with `library`.
    fn parse_extern_c_block(
        &mut self,
        out: &mut Vec<SurfaceDef>,
    ) -> Result<(), ParseError> {
        self.expect(&TokenKind::Extern, "`extern`")?;
        // The ABI string — only "C" is supported.
        let abi_tok = self.bump().clone();
        match &abi_tok.kind {
            TokenKind::Str(s) if s == "C" => {}
            other => {
                return Err(ParseError::Unexpected {
                    expected: "ABI string \"C\"".to_owned(),
                    found: other.to_string(),
                    span: abi_tok.span,
                });
            }
        }
        // `lib "<name>"`
        let (lib_kw, lib_kw_span) = self.expect_ident("`lib`")?;
        if lib_kw != "lib" {
            return Err(ParseError::Unexpected {
                expected: "`lib`".to_owned(),
                found: lib_kw,
                span: lib_kw_span,
            });
        }
        let libname_tok = self.bump().clone();
        let library = match &libname_tok.kind {
            TokenKind::Str(s) => s.clone(),
            other => {
                return Err(ParseError::Unexpected {
                    expected: "library name string".to_owned(),
                    found: other.to_string(),
                    span: libname_tok.span,
                });
            }
        };
        self.expect(&TokenKind::LBrace, "`{`")?;
        while !matches!(self.peek_kind(), TokenKind::RBrace) {
            let start = self.peek().span.start;
            self.expect(&TokenKind::Fn, "`fn`")?;
            let (name, _) = self.expect_ident("C function name")?;
            self.expect(&TokenKind::LParen, "`(`")?;
            let (params, variadic) = self.parse_c_params()?;
            let close_end = self.expect(&TokenKind::RParen, "`)`")?.span.end;
            self.expect(&TokenKind::Arrow, "`->`")?;
            let ret = self.parse_type()?;
            let end = ret.span().end.max(close_end);
            out.push(SurfaceDef {
                name,
                span: Span { start, end },
                kind: SurfaceDefKind::Extern {
                    params,
                    ret,
                    library: Some(library.clone()),
                    variadic,
                },
            });
        }
        self.expect(&TokenKind::RBrace, "`}`")?;
        Ok(())
    }

    fn parse_enum_def(&mut self) -> Result<SurfaceDef, ParseError> {
        let start = self.peek().span.start;
        self.expect(&TokenKind::Enum, "`enum`")?;
        let (name, _) = self.expect_ident("enum name")?;
        let type_params = self.parse_type_params()?;
        self.expect(&TokenKind::LBrace, "`{`")?;
        let mut variants: Vec<(String, Option<SurfaceType>)> = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RBrace) {
            loop {
                let (vname, _) = self.expect_ident("variant name")?;
                let payload = if self.eat(&TokenKind::LParen) {
                    // v1 restriction: 0 or 1 payload type per variant.
                    let ty = self.parse_type()?;
                    if matches!(self.peek_kind(), TokenKind::Comma) {
                        let span = self.peek().span;
                        return Err(ParseError::Unexpected {
                            expected: "`)` (v1 variants have at most one payload type — wrap multiple values in a struct)".to_owned(),
                            found: ",".to_owned(),
                            span,
                        });
                    }
                    self.expect(&TokenKind::RParen, "`)`")?;
                    Some(ty)
                } else {
                    None
                };
                variants.push((vname, payload));
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::RBrace) {
                    break;
                }
            }
        }
        let close = self.expect(&TokenKind::RBrace, "`}`")?;
        Ok(SurfaceDef {
            name,
            span: Span {
                start,
                end: close.span.end,
            },
            kind: SurfaceDefKind::Enum {
                type_params,
                variants,
            },
        })
    }

    fn parse_fn_def(&mut self) -> Result<SurfaceDef, ParseError> {
        let start = self.peek().span.start;
        self.expect(&TokenKind::Def, "`def`")?;
        let is_local = self.eat(&TokenKind::Local);
        let (name, _) = self.expect_ident("function name")?;
        let type_params = self.parse_type_params()?;
        self.expect(&TokenKind::LParen, "`(`")?;
        let params = self.parse_params()?;
        self.expect(&TokenKind::RParen, "`)`")?;
        self.expect(&TokenKind::Arrow, "`->`")?;
        let ret = self.parse_type()?;
        self.expect(&TokenKind::Eq, "`=`")?;
        let body = self.parse_expr()?;
        let end = body.span().end;
        Ok(SurfaceDef {
            name,
            span: Span { start, end },
            kind: SurfaceDefKind::Fn {
                is_local,
                type_params,
                params,
                ret,
                body,
            },
        })
    }

    fn parse_struct_def(&mut self) -> Result<SurfaceDef, ParseError> {
        let start = self.peek().span.start;
        self.expect(&TokenKind::Struct, "`struct`")?;
        let (name, _) = self.expect_ident("struct name")?;
        let type_params = self.parse_type_params()?;
        self.expect(&TokenKind::LBrace, "`{`")?;
        let mut fields = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RBrace) {
            loop {
                let (fname, _) = self.expect_ident("field name")?;
                self.expect(&TokenKind::Colon, "`:`")?;
                let fty = self.parse_type()?;
                fields.push((fname, fty));
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::RBrace) {
                    break;
                }
            }
        }
        let close = self.expect(&TokenKind::RBrace, "`}`")?;
        let end = close.span.end;
        Ok(SurfaceDef {
            name,
            span: Span { start, end },
            kind: SurfaceDefKind::Struct {
                type_params,
                fields,
            },
        })
    }

    /// Parse an optional `<T1, T2, ..>` head — type parameter names.
    /// Returns an empty vec if no `<` is present.
    fn parse_type_params(&mut self) -> Result<Vec<String>, ParseError> {
        if !self.eat(&TokenKind::Lt) {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::Gt) {
            loop {
                let (name, _) = self.expect_ident("type parameter name")?;
                out.push(name);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::Gt) {
                    break;
                }
            }
        }
        self.expect(&TokenKind::Gt, "`>`")?;
        Ok(out)
    }

    fn parse_params(&mut self) -> Result<Vec<(String, SurfaceType)>, ParseError> {
        let mut params = Vec::new();
        if matches!(self.peek_kind(), TokenKind::RParen) {
            return Ok(params);
        }
        loop {
            let (name, _) = self.expect_ident("parameter name")?;
            self.expect(&TokenKind::Colon, "`:`")?;
            let ty = self.parse_type()?;
            params.push((name, ty));
            if !self.eat(&TokenKind::Comma) {
                break;
            }
            // Trailing comma allowed.
            if matches!(self.peek_kind(), TokenKind::RParen) {
                break;
            }
        }
        Ok(params)
    }

    /// Parse a C function's parameter list, which (unlike an ordinary
    /// param list) may end with `...` to mark the function variadic, e.g.
    /// `curl_easy_setopt(handle: Ptr, option: Int, ...)`. Returns the
    /// fixed params and whether `...` was present. The `...` is three
    /// `Dot` tokens; it must be the final entry.
    fn parse_c_params(
        &mut self,
    ) -> Result<(Vec<(String, SurfaceType)>, bool), ParseError> {
        let mut params = Vec::new();
        let mut variadic = false;
        if matches!(self.peek_kind(), TokenKind::RParen) {
            return Ok((params, variadic));
        }
        loop {
            // A leading `...` (variadic with no fixed params) or a `...`
            // after the last fixed param.
            if matches!(self.peek_kind(), TokenKind::Dot) {
                self.expect_ellipsis()?;
                variadic = true;
                break;
            }
            let (name, _) = self.expect_ident("parameter name")?;
            self.expect(&TokenKind::Colon, "`:`")?;
            let ty = self.parse_type()?;
            params.push((name, ty));
            if !self.eat(&TokenKind::Comma) {
                break;
            }
            if matches!(self.peek_kind(), TokenKind::RParen) {
                break;
            }
        }
        Ok((params, variadic))
    }

    /// Consume exactly `...` (three consecutive `Dot` tokens).
    fn expect_ellipsis(&mut self) -> Result<(), ParseError> {
        for _ in 0..3 {
            let tok = self.peek().clone();
            if !matches!(tok.kind, TokenKind::Dot) {
                return Err(ParseError::Unexpected {
                    expected: "`...` (variadic marker)".to_owned(),
                    found: tok.kind.to_string(),
                    span: tok.span,
                });
            }
            self.bump();
        }
        Ok(())
    }

    fn parse_type(&mut self) -> Result<SurfaceType, ParseError> {
        // `fn(...) -> ret` for function types; else an ident (optionally
        // followed by `<...>` for a generic instantiation).
        if matches!(self.peek_kind(), TokenKind::Fn) {
            let fn_tok = self.bump().clone();
            let start = fn_tok.span.start;
            self.expect(&TokenKind::LParen, "`(`")?;
            let mut params = Vec::new();
            if !matches!(self.peek_kind(), TokenKind::RParen) {
                loop {
                    params.push(self.parse_type()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                    if matches!(self.peek_kind(), TokenKind::RParen) {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RParen, "`)`")?;
            self.expect(&TokenKind::Arrow, "`->`")?;
            let ret = self.parse_type()?;
            let end = ret.span().end;
            Ok(SurfaceType::FnType {
                params,
                ret: Box::new(ret),
                span: Span { start, end },
            })
        } else {
            let (name, name_span) = self.expect_ident("type name")?;
            // Optional `<arg1, arg2, ...>` for generic instantiations.
            if matches!(self.peek_kind(), TokenKind::Lt) {
                self.bump();
                let mut args = Vec::new();
                if !matches!(self.peek_kind(), TokenKind::Gt) {
                    loop {
                        args.push(self.parse_type()?);
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                        if matches!(self.peek_kind(), TokenKind::Gt) {
                            break;
                        }
                    }
                }
                let close = self.expect(&TokenKind::Gt, "`>`")?;
                Ok(SurfaceType::Applied {
                    name,
                    name_span,
                    args,
                    span: Span {
                        start: name_span.start,
                        end: close.span.end,
                    },
                })
            } else {
                Ok(SurfaceType::Named { name, span: name_span })
            }
        }
    }

    // ---- Expressions (precedence climbing) ----

    fn parse_expr(&mut self) -> Result<SurfaceExpr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.parse_and()?;
        while self.eat(&TokenKind::PipePipe) {
            let right = self.parse_and()?;
            let span = Span {
                start: left.span().start,
                end: right.span().end,
            };
            left = SurfaceExpr::BinOp {
                op: BinOp::Or,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.parse_cmp()?;
        while self.eat(&TokenKind::AmpAmp) {
            let right = self.parse_cmp()?;
            let span = Span {
                start: left.span().start,
                end: right.span().end,
            };
            left = SurfaceExpr::BinOp {
                op: BinOp::And,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<SurfaceExpr, ParseError> {
        let left = self.parse_add()?;
        let op = match self.peek_kind() {
            TokenKind::EqEq => Some(BinOp::Eq),
            TokenKind::NotEq => Some(BinOp::NotEq),
            TokenKind::Lt => Some(BinOp::Lt),
            TokenKind::LtEq => Some(BinOp::LtEq),
            TokenKind::Gt => Some(BinOp::Gt),
            TokenKind::GtEq => Some(BinOp::GtEq),
            _ => None,
        };
        let Some(op) = op else {
            return Ok(left);
        };
        self.bump();
        let right = self.parse_add()?;
        // Non-associative: reject `a < b < c`.
        if matches!(
            self.peek_kind(),
            TokenKind::EqEq
                | TokenKind::NotEq
                | TokenKind::Lt
                | TokenKind::LtEq
                | TokenKind::Gt
                | TokenKind::GtEq
        ) {
            return Err(ParseError::NonAssocComparison {
                span: self.peek().span,
            });
        }
        let span = Span {
            start: left.span().start,
            end: right.span().end,
        };
        Ok(SurfaceExpr::BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span,
        })
    }

    fn parse_add(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.parse_mul()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => return Ok(left),
            };
            self.bump();
            let right = self.parse_mul()?;
            let span = Span {
                start: left.span().start,
                end: right.span().end,
            };
            left = SurfaceExpr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }
    }

    fn parse_mul(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Rem,
                _ => return Ok(left),
            };
            self.bump();
            let right = self.parse_unary()?;
            let span = Span {
                start: left.span().start,
                end: right.span().end,
            };
            left = SurfaceExpr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }
    }

    fn parse_unary(&mut self) -> Result<SurfaceExpr, ParseError> {
        match self.peek_kind() {
            TokenKind::Minus => {
                let start = self.bump().span.start;
                let operand = self.parse_unary()?;
                let end = operand.span().end;
                Ok(SurfaceExpr::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                    span: Span { start, end },
                })
            }
            TokenKind::Bang => {
                let start = self.bump().span.start;
                let operand = self.parse_unary()?;
                let end = operand.span().end;
                Ok(SurfaceExpr::UnaryOp {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                    span: Span { start, end },
                })
            }
            _ => self.parse_call(),
        }
    }

    fn parse_call(&mut self) -> Result<SurfaceExpr, ParseError> {
        let mut e = self.parse_atom()?;
        loop {
            match self.peek_kind() {
                // Turbofish: `callee::<T, ...>(args)`. The `::<...>` MUST be
                // immediately followed by a call — it names explicit type
                // arguments for that call.
                TokenKind::ColonColon => {
                    self.bump();
                    self.expect(&TokenKind::Lt, "`<` after `::`")?;
                    let mut type_args = Vec::new();
                    if !matches!(self.peek_kind(), TokenKind::Gt) {
                        loop {
                            type_args.push(self.parse_type()?);
                            if !self.eat(&TokenKind::Comma) {
                                break;
                            }
                            if matches!(self.peek_kind(), TokenKind::Gt) {
                                break;
                            }
                        }
                    }
                    self.expect(&TokenKind::Gt, "`>`")?;
                    self.expect(&TokenKind::LParen, "`(` (turbofish must be followed by a call)")?;
                    // Inside the `(...)`, struct literals are unambiguous (the
                    // `)` delimits them), so re-enable them even if we're in a
                    // struct-lit-suppressed context (an `if`/`match` head).
                    let prev_sl = self.allow_struct_lit;
                    self.allow_struct_lit = true;
                    let mut args = Vec::new();
                    if !matches!(self.peek_kind(), TokenKind::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if !self.eat(&TokenKind::Comma) {
                                break;
                            }
                            if matches!(self.peek_kind(), TokenKind::RParen) {
                                break;
                            }
                        }
                    }
                    self.allow_struct_lit = prev_sl;
                    let end_tok = self.expect(&TokenKind::RParen, "`)`")?;
                    let span = Span {
                        start: e.span().start,
                        end: end_tok.span.end,
                    };
                    e = SurfaceExpr::Call {
                        callee: Box::new(e),
                        args,
                        type_args,
                        span,
                    };
                }
                TokenKind::LParen => {
                    self.bump();
                    // Re-enable struct literals inside the argument list — the
                    // `)` disambiguates, even in an `if`/`match` head.
                    let prev_sl = self.allow_struct_lit;
                    self.allow_struct_lit = true;
                    let mut args = Vec::new();
                    if !matches!(self.peek_kind(), TokenKind::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if !self.eat(&TokenKind::Comma) {
                                break;
                            }
                            if matches!(self.peek_kind(), TokenKind::RParen) {
                                break;
                            }
                        }
                    }
                    self.allow_struct_lit = prev_sl;
                    let end_tok = self.expect(&TokenKind::RParen, "`)`")?;
                    let span = Span {
                        start: e.span().start,
                        end: end_tok.span.end,
                    };
                    e = SurfaceExpr::Call {
                        callee: Box::new(e),
                        args,
                        type_args: Vec::new(),
                        span,
                    };
                }
                TokenKind::Dot => {
                    self.bump();
                    let (field_name, field_span) = self.expect_ident("field name")?;
                    let span = Span {
                        start: e.span().start,
                        end: field_span.end,
                    };
                    e = SurfaceExpr::FieldAccess {
                        base: Box::new(e),
                        field_name,
                        field_span,
                        span,
                    };
                }
                TokenKind::Question => {
                    let q_end = self.bump().span.end;
                    let span = Span {
                        start: e.span().start,
                        end: q_end,
                    };
                    e = SurfaceExpr::Try {
                        expr: Box::new(e),
                        span,
                    };
                }
                _ => return Ok(e),
            }
        }
    }

    fn parse_atom(&mut self) -> Result<SurfaceExpr, ParseError> {
        let t = self.peek().clone();
        match t.kind {
            TokenKind::Int(value) => {
                self.bump();
                Ok(SurfaceExpr::IntLit {
                    value,
                    span: t.span,
                })
            }
            TokenKind::Float(value) => {
                self.bump();
                Ok(SurfaceExpr::FloatLit {
                    value,
                    span: t.span,
                })
            }
            TokenKind::True => {
                self.bump();
                Ok(SurfaceExpr::BoolLit {
                    value: true,
                    span: t.span,
                })
            }
            TokenKind::False => {
                self.bump();
                Ok(SurfaceExpr::BoolLit {
                    value: false,
                    span: t.span,
                })
            }
            TokenKind::Str(value) => {
                self.bump();
                Ok(SurfaceExpr::StringLit {
                    value,
                    span: t.span,
                })
            }
            TokenKind::Ident(name) => {
                self.bump();
                if self.allow_struct_lit && matches!(self.peek_kind(), TokenKind::LBrace) {
                    self.parse_struct_lit(name, t.span)
                } else {
                    Ok(SurfaceExpr::Var {
                        name,
                        span: t.span,
                    })
                }
            }
            TokenKind::LParen => {
                self.bump();
                // A parenthesized expression: struct literals are unambiguous
                // here (the `)` delimits), so re-enable them even inside an
                // `if`/`match` head.
                let prev_sl = self.allow_struct_lit;
                self.allow_struct_lit = true;
                let inner = self.parse_expr()?;
                self.allow_struct_lit = prev_sl;
                self.expect(&TokenKind::RParen, "`)`")?;
                Ok(inner)
            }
            TokenKind::LBrace => self.parse_block(),
            // `|x: T| body` — non-empty lambda parameter list.
            TokenKind::Pipe => self.parse_lambda(),
            // `|| body` — zero-arg lambda. `||` would only be the logical-OR
            // operator after a left expression, so in atom position it must
            // be an empty parameter list.
            TokenKind::PipePipe => self.parse_lambda_zero_arg(),
            TokenKind::Match => self.parse_match(),
            TokenKind::If => self.parse_if(),
            _ => Err(ParseError::Unexpected {
                expected: "expression".to_owned(),
                found: t.kind.to_string(),
                span: t.span,
            }),
        }
    }

    fn parse_lambda(&mut self) -> Result<SurfaceExpr, ParseError> {
        let open = self.expect(&TokenKind::Pipe, "`|`")?;
        let start = open.span.start;
        let mut params = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::Pipe) {
            loop {
                let (name, _) = self.expect_ident("closure parameter name")?;
                self.expect(&TokenKind::Colon, "`:` (closure parameters require explicit types in v1)")?;
                let ty = self.parse_type()?;
                params.push((name, ty));
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::Pipe) {
                    break;
                }
            }
        }
        self.expect(&TokenKind::Pipe, "`|`")?;
        let body = self.parse_expr()?;
        let end = body.span().end;
        Ok(SurfaceExpr::Lambda {
            params,
            body: Box::new(body),
            span: Span { start, end },
        })
    }

    fn parse_struct_lit(
        &mut self,
        type_name: String,
        type_name_span: Span,
    ) -> Result<SurfaceExpr, ParseError> {
        let start = type_name_span.start;
        self.expect(&TokenKind::LBrace, "`{`")?;
        let mut fields = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RBrace) {
            loop {
                let (fname, _) = self.expect_ident("field name")?;
                self.expect(&TokenKind::Colon, "`:`")?;
                let val = self.parse_expr()?;
                fields.push((fname, val));
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::RBrace) {
                    break;
                }
            }
        }
        let close = self.expect(&TokenKind::RBrace, "`}`")?;
        Ok(SurfaceExpr::StructLit {
            type_name,
            type_name_span,
            fields,
            span: Span {
                start,
                end: close.span.end,
            },
        })
    }

    /// `if cond { then } else { else }`. Both branches are blocks
    /// (so they get the implicit `{ tail_expr }` shape used elsewhere
    /// in the language). The condition's trailing `{` opens the then
    /// block, NOT a struct literal — suppress struct-lit recognition
    /// in the condition position the same way `match` does.
    fn parse_if(&mut self) -> Result<SurfaceExpr, ParseError> {
        let start = self.peek().span.start;
        self.expect(&TokenKind::If, "`if`")?;
        let prev = self.allow_struct_lit;
        self.allow_struct_lit = false;
        let cond = self.parse_expr();
        self.allow_struct_lit = prev;
        let cond = cond?;
        let then_branch = self.parse_block()?;
        self.expect(&TokenKind::Else, "`else`")?;
        let else_branch = self.parse_block()?;
        let end = else_branch.span().end;
        Ok(SurfaceExpr::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
            span: Span { start, end },
        })
    }

    /// `match expr { pat => expr, pat => expr, ... }`. A trailing comma
    /// after the last arm is allowed.
    fn parse_match(&mut self) -> Result<SurfaceExpr, ParseError> {
        let start = self.peek().span.start;
        self.expect(&TokenKind::Match, "`match`")?;
        // The scrutinee's trailing `{` is the match's opening brace,
        // not a struct literal. Suppress struct-literal recognition.
        let prev = self.allow_struct_lit;
        self.allow_struct_lit = false;
        let scrutinee = self.parse_expr();
        self.allow_struct_lit = prev;
        let scrutinee = scrutinee?;
        self.expect(&TokenKind::LBrace, "`{`")?;
        let mut arms: Vec<SurfaceMatchArm> = Vec::new();
        if !matches!(self.peek_kind(), TokenKind::RBrace) {
            loop {
                let arm_start = self.peek().span.start;
                let pattern = self.parse_pattern()?;
                self.expect(&TokenKind::FatArrow, "`=>`")?;
                let body = self.parse_expr()?;
                let arm_end = body.span().end;
                arms.push(SurfaceMatchArm {
                    pattern,
                    body,
                    span: Span {
                        start: arm_start,
                        end: arm_end,
                    },
                });
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
                if matches!(self.peek_kind(), TokenKind::RBrace) {
                    break;
                }
            }
        }
        let close = self.expect(&TokenKind::RBrace, "`}`")?;
        Ok(SurfaceExpr::Match {
            scrutinee: Box::new(scrutinee),
            arms,
            span: Span {
                start,
                end: close.span.end,
            },
        })
    }

    /// Patterns:
    ///   `_`              → wildcard
    ///   `ident`          → variable binding OR nullary constructor (resolver decides)
    ///   `Ident ( pat )`  → constructor with payload
    fn parse_pattern(&mut self) -> Result<SurfacePattern, ParseError> {
        let t = self.peek().clone();
        match t.kind {
            TokenKind::Underscore => {
                self.bump();
                Ok(SurfacePattern::Wildcard { span: t.span })
            }
            TokenKind::Ident(name) => {
                self.bump();
                if self.eat(&TokenKind::LParen) {
                    let inner = self.parse_pattern()?;
                    if matches!(self.peek_kind(), TokenKind::Comma) {
                        let span = self.peek().span;
                        return Err(ParseError::Unexpected {
                            expected: "`)` (v1 constructor patterns have at most one sub-pattern)".to_owned(),
                            found: ",".to_owned(),
                            span,
                        });
                    }
                    let close = self.expect(&TokenKind::RParen, "`)`")?;
                    Ok(SurfacePattern::Ctor {
                        name,
                        payload: Box::new(inner),
                        span: Span {
                            start: t.span.start,
                            end: close.span.end,
                        },
                    })
                } else {
                    Ok(SurfacePattern::Ident {
                        name,
                        span: t.span,
                    })
                }
            }
            _ => Err(ParseError::Unexpected {
                expected: "pattern".to_owned(),
                found: t.kind.to_string(),
                span: t.span,
            }),
        }
    }

    fn parse_lambda_zero_arg(&mut self) -> Result<SurfaceExpr, ParseError> {
        let open = self.expect(&TokenKind::PipePipe, "`||`")?;
        let start = open.span.start;
        let body = self.parse_expr()?;
        let end = body.span().end;
        Ok(SurfaceExpr::Lambda {
            params: Vec::new(),
            body: Box::new(body),
            span: Span { start, end },
        })
    }

    /// `{ stmt* tail_expr }` — a block expression.
    ///
    /// A `let x = e ;` statement extends scope for the rest of the block.
    /// The block evaluates to `tail_expr`. There is no implicit `()` /
    /// unit: a tail expression is required.
    fn parse_block(&mut self) -> Result<SurfaceExpr, ParseError> {
        let open = self.expect(&TokenKind::LBrace, "`{`")?;
        let start = open.span.start;
        let mut stmts = Vec::new();

        loop {
            if matches!(self.peek_kind(), TokenKind::RBrace) {
                return Err(ParseError::Unexpected {
                    expected: "tail expression before `}`".to_owned(),
                    found: "}".to_owned(),
                    span: self.peek().span,
                });
            }
            if matches!(self.peek_kind(), TokenKind::Let) {
                stmts.push(self.parse_let_stmt()?);
                continue;
            }
            if matches!(self.peek_kind(), TokenKind::Defer) {
                stmts.push(self.parse_defer_stmt()?);
                continue;
            }
            // Tail expression — the last thing in the block.
            let tail = self.parse_expr()?;
            let close = self.expect(&TokenKind::RBrace, "`}`")?;
            return Ok(SurfaceExpr::Block {
                stmts,
                tail: Box::new(tail),
                span: Span {
                    start,
                    end: close.span.end,
                },
            });
        }
    }

    fn parse_let_stmt(&mut self) -> Result<SurfaceStmt, ParseError> {
        let let_tok = self.expect(&TokenKind::Let, "`let`")?;
        let start = let_tok.span.start;
        let (name, _) = self.expect_ident("variable name in `let`")?;
        self.expect(&TokenKind::Eq, "`=`")?;
        let value = self.parse_expr()?;
        let semi = self.expect(&TokenKind::Semi, "`;`")?;
        Ok(SurfaceStmt::Let {
            name,
            value,
            span: Span {
                start,
                end: semi.span.end,
            },
        })
    }

    fn parse_defer_stmt(&mut self) -> Result<SurfaceStmt, ParseError> {
        let defer_tok = self.expect(&TokenKind::Defer, "`defer`")?;
        let start = defer_tok.span.start;
        let expr = self.parse_expr()?;
        let semi = self.expect(&TokenKind::Semi, "`;`")?;
        Ok(SurfaceStmt::Defer {
            expr,
            span: Span {
                start,
                end: semi.span.end,
            },
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn expr(s: &str) -> SurfaceExpr {
        parse_expr(s).unwrap()
    }

    #[test]
    fn int_literal() {
        match expr("42") {
            SurfaceExpr::IntLit { value: 42, .. } => {}
            other => panic!("expected IntLit(42), got {:?}", other),
        }
    }

    #[test]
    fn bool_literals() {
        match expr("true") {
            SurfaceExpr::BoolLit { value: true, .. } => {}
            other => panic!("expected BoolLit(true), got {:?}", other),
        }
        match expr("false") {
            SurfaceExpr::BoolLit { value: false, .. } => {}
            other => panic!("expected BoolLit(false), got {:?}", other),
        }
    }

    #[test]
    fn parses_def_returning_bool() {
        let d = parse_def("def b() -> Bool = true").unwrap();
        assert_eq!(d.name, "b");
        let (_, params, ret, body) = fn_kind(&d);
        assert!(params.is_empty());
        assert!(matches!(ret, SurfaceType::Named { name, .. } if name == "Bool"));
        assert!(matches!(body, SurfaceExpr::BoolLit { value: true, .. }));

        let d = parse_def("def b() -> Bool = false").unwrap();
        let (_, _, _, body) = fn_kind(&d);
        assert!(matches!(body, SurfaceExpr::BoolLit { value: false, .. }));
    }

    #[test]
    fn identifier() {
        match expr("foo") {
            SurfaceExpr::Var { name, .. } => assert_eq!(name, "foo"),
            other => panic!("expected Var, got {:?}", other),
        }
    }

    #[test]
    fn precedence_mul_binds_tighter_than_add() {
        // 1 + 2 * 3 == 1 + (2 * 3)
        match expr("1 + 2 * 3") {
            SurfaceExpr::BinOp {
                op: BinOp::Add,
                left,
                right,
                ..
            } => {
                assert!(matches!(*left, SurfaceExpr::IntLit { value: 1, .. }));
                assert!(matches!(
                    *right,
                    SurfaceExpr::BinOp {
                        op: BinOp::Mul,
                        ..
                    }
                ));
            }
            other => panic!("expected Add at root, got {:?}", other),
        }
    }

    #[test]
    fn add_is_left_associative() {
        // 1 - 2 - 3 == (1 - 2) - 3
        match expr("1 - 2 - 3") {
            SurfaceExpr::BinOp {
                op: BinOp::Sub,
                left,
                right,
                ..
            } => {
                assert!(matches!(
                    *left,
                    SurfaceExpr::BinOp {
                        op: BinOp::Sub,
                        ..
                    }
                ));
                assert!(matches!(*right, SurfaceExpr::IntLit { value: 3, .. }));
            }
            other => panic!("expected Sub at root, got {:?}", other),
        }
    }

    #[test]
    fn parens_override_precedence() {
        match expr("(1 + 2) * 3") {
            SurfaceExpr::BinOp {
                op: BinOp::Mul,
                left,
                ..
            } => {
                assert!(matches!(
                    *left,
                    SurfaceExpr::BinOp {
                        op: BinOp::Add,
                        ..
                    }
                ));
            }
            other => panic!("expected Mul at root, got {:?}", other),
        }
    }

    #[test]
    fn unary_minus() {
        match expr("-5") {
            SurfaceExpr::UnaryOp {
                op: UnaryOp::Neg, ..
            } => {}
            other => panic!("expected UnaryOp(Neg), got {:?}", other),
        }
    }

    #[test]
    fn function_call_no_args() {
        match expr("f()") {
            SurfaceExpr::Call { callee, args, .. } => {
                assert!(matches!(*callee, SurfaceExpr::Var { .. }));
                assert!(args.is_empty());
            }
            other => panic!("expected Call, got {:?}", other),
        }
    }

    #[test]
    fn function_call_with_args() {
        match expr("f(1, 2, x)") {
            SurfaceExpr::Call { args, .. } => assert_eq!(args.len(), 3),
            other => panic!("expected Call, got {:?}", other),
        }
    }

    #[test]
    fn nested_call() {
        // double(double(x))
        let e = expr("double(double(x))");
        let SurfaceExpr::Call { callee, args, .. } = e else {
            panic!("expected outer Call");
        };
        assert!(matches!(*callee, SurfaceExpr::Var { .. }));
        assert_eq!(args.len(), 1);
        assert!(matches!(&args[0], SurfaceExpr::Call { .. }));
    }

    #[test]
    fn comparison_non_associative() {
        let err = parse_expr("a < b < c").unwrap_err();
        assert!(matches!(err, ParseError::NonAssocComparison { .. }));
    }

    #[test]
    fn trailing_comma_in_args() {
        let e = expr("f(1, 2,)");
        let SurfaceExpr::Call { args, .. } = e else {
            panic!("expected Call");
        };
        assert_eq!(args.len(), 2);
    }

    // ---- Def parsing ----

    fn fn_kind(d: &SurfaceDef) -> (bool, &Vec<(String, SurfaceType)>, &SurfaceType, &SurfaceExpr) {
        match &d.kind {
            SurfaceDefKind::Fn {
                is_local,
                params,
                ret,
                body,
                ..
            } => (*is_local, params, ret, body),
            other => panic!("expected Fn def, got {:?}", other),
        }
    }

    #[test]
    fn parses_def_double() {
        let d = parse_def("def double(x: Int) -> Int = x * 2").unwrap();
        assert_eq!(d.name, "double");
        let (is_local, params, ret, body) = fn_kind(&d);
        assert!(!is_local);
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "x");
        assert!(matches!(&params[0].1, SurfaceType::Named { name, .. } if name == "Int"));
        assert!(matches!(ret, SurfaceType::Named { name, .. } if name == "Int"));
        assert!(matches!(
            body,
            SurfaceExpr::BinOp {
                op: BinOp::Mul, ..
            }
        ));
    }

    #[test]
    fn parses_def_local() {
        let d = parse_def("def local store(x: Int) -> Int = x").unwrap();
        assert_eq!(d.name, "store");
        let (is_local, _, _, _) = fn_kind(&d);
        assert!(is_local);
    }

    #[test]
    fn parses_def_no_params() {
        let d = parse_def("def main() -> Int = 0").unwrap();
        let (_, params, _, _) = fn_kind(&d);
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn parses_module_with_multiple_defs() {
        let src = "
            def double(x: Int) -> Int = x * 2
            def quadruple(x: Int) -> Int = double(double(x))
        ";
        let m = parse_module(src).unwrap();
        assert_eq!(m.defs.len(), 2);
        assert_eq!(m.defs[0].name, "double");
        assert_eq!(m.defs[1].name, "quadruple");
    }

    #[test]
    fn missing_arrow_error() {
        let err = parse_def("def f(x: Int) Int = x").unwrap_err();
        assert!(matches!(err, ParseError::Unexpected { ref expected, .. } if expected == "`->`"));
    }

    #[test]
    fn struct_literal_in_call_args_and_match_head() {
        // A struct literal is unambiguous inside a call's parens, so it must
        // parse as a non-last argument AND inside an `if`/`match` head (where
        // a bare `Ident { … }` would otherwise be a suppressed struct lit).
        let src = "
            struct P { x: Int, y: Int }
            def f(a: P, b: Int) -> Int = b
            def mid() -> Int = f(P { x: 1, y: 2 }, 7)
            def in_match() -> Int =
                match f(P { x: 1, y: 2 }, 7) { _ => 0 }
            def in_if() -> Int =
                if f(P { x: 1, y: 2 }, 7) == 7 { 1 } else { 0 }
        ";
        let m = parse_module(src).unwrap();
        assert_eq!(m.defs.len(), 5);
    }

    #[test]
    fn parses_state_binding() {
        let src = "
            state counter: Atom<Int> = atom(0)
            def handle(d: Int) -> Int = swap(counter, |n: Int| n + d)
        ";
        let m = parse_module(src).unwrap();
        assert_eq!(m.defs.len(), 2);
        assert_eq!(m.defs[0].name, "counter");
        match &m.defs[0].kind {
            SurfaceDefKind::State { ty, .. } => {
                assert!(matches!(ty, SurfaceType::Applied { name, .. } if name == "Atom"));
            }
            other => panic!("expected State, got {:?}", other),
        }
    }

    #[test]
    fn missing_def_keyword_error() {
        let err = parse_module("fn double(x: Int) -> Int = x * 2").unwrap_err();
        assert!(matches!(
            err,
            ParseError::Unexpected { ref expected, .. }
                if expected.contains("def") && expected.contains("struct") && expected.contains("enum")
        ));
    }

    #[test]
    fn lex_error_propagates_as_parse_error() {
        let err = parse_expr("@").unwrap_err();
        assert!(matches!(err, ParseError::Lex(_)));
    }

    // ---- Block / let ----

    #[test]
    fn parses_block_with_single_let() {
        let e = expr("{ let x = 1; x }");
        let SurfaceExpr::Block { stmts, tail, .. } = e else {
            panic!("expected Block");
        };
        assert_eq!(stmts.len(), 1);
        let SurfaceStmt::Let { name, .. } = &stmts[0] else {
            panic!("expected Let stmt");
        };
        assert_eq!(name, "x");
        assert!(matches!(*tail, SurfaceExpr::Var { .. }));
    }

    #[test]
    fn parses_nested_block() {
        // Inner block as the value of an outer let.
        let e = expr("{ let x = { let y = 1; y + 1 }; x * 2 }");
        let SurfaceExpr::Block { stmts, .. } = e else {
            panic!("expected outer Block");
        };
        let SurfaceStmt::Let { value, .. } = &stmts[0] else {
            panic!("expected Let stmt");
        };
        assert!(matches!(value, SurfaceExpr::Block { .. }));
    }

    #[test]
    fn block_without_tail_expression_errors() {
        // `{ let x = 1; }` has no tail expression — that's an error.
        let err = parse_expr("{ let x = 1; }").unwrap_err();
        assert!(matches!(
            err,
            ParseError::Unexpected { ref expected, .. } if expected.contains("tail expression")
        ));
    }

    #[test]
    fn let_requires_semicolon() {
        let err = parse_expr("{ let x = 1 x }").unwrap_err();
        assert!(matches!(err, ParseError::Unexpected { ref expected, .. } if expected == "`;`"));
    }

    // ---- Lambdas ----

    #[test]
    fn parses_single_param_lambda() {
        let e = expr("|x: Int| x + 1");
        let SurfaceExpr::Lambda { params, body, .. } = e else {
            panic!("expected Lambda");
        };
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].0, "x");
        assert!(matches!(params[0].1, SurfaceType::Named { ref name, .. } if name == "Int"));
        assert!(matches!(
            *body,
            SurfaceExpr::BinOp {
                op: BinOp::Add, ..
            }
        ));
    }

    #[test]
    fn parses_multi_param_lambda() {
        let e = expr("|a: Int, b: Int| a + b");
        let SurfaceExpr::Lambda { params, .. } = e else {
            panic!("expected Lambda");
        };
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "a");
        assert_eq!(params[1].0, "b");
    }

    #[test]
    fn parses_zero_arg_lambda() {
        let e = expr("|| 42");
        let SurfaceExpr::Lambda { params, body, .. } = e else {
            panic!("expected Lambda");
        };
        assert!(params.is_empty());
        assert!(matches!(*body, SurfaceExpr::IntLit { value: 42, .. }));
    }

    #[test]
    fn parses_lambda_in_let() {
        // Inside a block so the parser scope is normal.
        let e = expr("{ let f = |x: Int| x * 2; f }");
        let SurfaceExpr::Block { stmts, .. } = e else {
            panic!("expected Block");
        };
        let SurfaceStmt::Let { value, .. } = &stmts[0] else {
            panic!("expected Let stmt");
        };
        assert!(matches!(value, SurfaceExpr::Lambda { .. }));
    }

    #[test]
    fn lambda_without_param_types_errors() {
        let err = parse_expr("|x| x").unwrap_err();
        assert!(matches!(err, ParseError::Unexpected { ref expected, .. } if expected.contains("`:`")));
    }

    // ---- Function types ----

    #[test]
    fn parses_fn_type_in_return_position() {
        let d = parse_def("def make() -> fn(Int) -> Int = |x: Int| x").unwrap();
        let (_, _, ret, _) = fn_kind(&d);
        match ret {
            SurfaceType::FnType { params, ret, .. } => {
                assert_eq!(params.len(), 1);
                assert!(matches!(**ret, SurfaceType::Named { ref name, .. } if name == "Int"));
            }
            other => panic!("expected FnType, got {:?}", other),
        }
    }

    #[test]
    fn parses_zero_arg_fn_type() {
        let d = parse_def("def make() -> fn() -> Int = || 0").unwrap();
        let (_, _, ret, _) = fn_kind(&d);
        match ret {
            SurfaceType::FnType { params, .. } => assert!(params.is_empty()),
            other => panic!("expected FnType, got {:?}", other),
        }
    }

    #[test]
    fn parses_higher_order_fn_param() {
        let d = parse_def("def apply(f: fn(Int) -> Int, x: Int) -> Int = f(x)").unwrap();
        let (_, params, _, _) = fn_kind(&d);
        assert_eq!(params.len(), 2);
        assert!(matches!(params[0].1, SurfaceType::FnType { .. }));
    }

    // ---- Struct definitions ----

    #[test]
    fn parses_struct_def() {
        let d = parse_def("struct Point { x: Int, y: Int }").unwrap();
        assert_eq!(d.name, "Point");
        match &d.kind {
            SurfaceDefKind::Struct { fields, .. } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "x");
                assert_eq!(fields[1].0, "y");
            }
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn parses_empty_struct() {
        let d = parse_def("struct Unit { }").unwrap();
        match &d.kind {
            SurfaceDefKind::Struct { fields, .. } => assert!(fields.is_empty()),
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn parses_struct_with_trailing_comma() {
        let d = parse_def("struct P { x: Int, y: Int, }").unwrap();
        match &d.kind {
            SurfaceDefKind::Struct { fields, .. } => assert_eq!(fields.len(), 2),
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    // ---- Generics ----

    #[test]
    fn parses_generic_enum() {
        let d = parse_def("enum Option<T> { Some(T), None }").unwrap();
        match &d.kind {
            SurfaceDefKind::Enum {
                type_params,
                variants,
            } => {
                assert_eq!(type_params, &vec!["T".to_owned()]);
                assert_eq!(variants.len(), 2);
                match &variants[0].1 {
                    Some(SurfaceType::Named { name, .. }) => assert_eq!(name, "T"),
                    other => panic!("Some payload should be `T`, got {:?}", other),
                }
            }
            other => panic!("expected Enum, got {:?}", other),
        }
    }

    #[test]
    fn parses_generic_struct() {
        let d = parse_def("struct Pair<A, B> { left: A, right: B }").unwrap();
        match &d.kind {
            SurfaceDefKind::Struct {
                type_params,
                fields,
            } => {
                assert_eq!(type_params, &vec!["A".to_owned(), "B".to_owned()]);
                assert_eq!(fields.len(), 2);
            }
            other => panic!("expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn parses_generic_fn() {
        let d = parse_def("def id<T>(x: T) -> T = x").unwrap();
        match &d.kind {
            SurfaceDefKind::Fn {
                type_params,
                params,
                ret,
                ..
            } => {
                assert_eq!(type_params, &vec!["T".to_owned()]);
                assert!(matches!(&params[0].1, SurfaceType::Named { name, .. } if name == "T"));
                assert!(matches!(ret, SurfaceType::Named { name, .. } if name == "T"));
            }
            other => panic!("expected Fn, got {:?}", other),
        }
    }

    #[test]
    fn parses_applied_type() {
        let d = parse_def("def foo() -> Result<Int, Failure> = bar()").unwrap();
        let (_, _, ret, _) = fn_kind(&d);
        match ret {
            SurfaceType::Applied { name, args, .. } => {
                assert_eq!(name, "Result");
                assert_eq!(args.len(), 2);
                assert!(matches!(&args[0], SurfaceType::Named { name, .. } if name == "Int"));
            }
            other => panic!("expected Applied, got {:?}", other),
        }
    }

    #[test]
    fn parses_nested_applied_type() {
        let d = parse_def("def foo(x: Option<Pair<Int, Int>>) -> Int = 0").unwrap();
        let (_, params, _, _) = fn_kind(&d);
        match &params[0].1 {
            SurfaceType::Applied { name, args, .. } => {
                assert_eq!(name, "Option");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], SurfaceType::Applied { name, .. } if name == "Pair"));
            }
            other => panic!("expected Applied, got {:?}", other),
        }
    }
}
