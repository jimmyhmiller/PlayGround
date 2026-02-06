use crate::ast::*;
use crate::token::{Span, Token, TokenKind};

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    errors: Vec<ParseError>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            errors: Vec::new(),
        }
    }

    pub fn parse_module(mut self) -> Result<Module, Vec<ParseError>> {
        let mut path = None;
        if self.peek_kind_is(&TokenKind::Module) {
            let start = self.bump().span.start;
            let p = self.parse_path();
            self.expect(TokenKind::Semi, "expected ';' after module declaration");
            let end = self.last_span_end(start);
            path = Some(p);
            let _ = end;
        }

        let mut items = Vec::new();
        while !self.is_eof() {
            match self.peek_kind() {
                Some(TokenKind::Use) => items.push(Item::Use(self.parse_use_decl())),
                Some(TokenKind::Extern) => items.push(Item::ExternFn(self.parse_extern_fn_decl())),
                Some(TokenKind::Fn) => items.push(Item::Fn(self.parse_fn_decl())),
                Some(TokenKind::Struct) => items.push(Item::Struct(self.parse_struct_decl())),
                Some(TokenKind::Enum) => items.push(Item::Enum(self.parse_enum_decl())),
                Some(_) => {
                    let span = self.peek_span().unwrap_or(Span::new(0, 0));
                    self.errors.push(ParseError {
                        message: "unexpected token at item position".to_string(),
                        span,
                    });
                    self.bump();
                }
                None => break,
            }
        }

        if self.errors.is_empty() {
            Ok(Module { path, items })
        } else {
            Err(self.errors)
        }
    }

    fn parse_use_decl(&mut self) -> UseDecl {
        let start = self.expect(TokenKind::Use, "expected 'use'");
        let path = self.parse_path();
        let end = self.expect(TokenKind::Semi, "expected ';' after use");
        UseDecl {
            path,
            span: Span::new(start, end),
        }
    }

    fn parse_struct_decl(&mut self) -> StructDecl {
        let start = self.expect(TokenKind::Struct, "expected 'struct'");
        let name = self.expect_ident("expected struct name");
        self.expect(TokenKind::LBrace, "expected '{' in struct");
        let fields = if self.peek_kind_is(&TokenKind::RBrace) {
            Vec::new()
        } else {
            self.parse_field_list()
        };
        let end = self.expect(TokenKind::RBrace, "expected '}' after struct");
        StructDecl {
            name,
            fields,
            span: Span::new(start, end),
        }
    }

    fn parse_enum_decl(&mut self) -> EnumDecl {
        let start = self.expect(TokenKind::Enum, "expected 'enum'");
        let name = self.expect_ident("expected enum name");
        self.expect(TokenKind::LBrace, "expected '{' in enum");
        let mut variants = Vec::new();
        while !self.peek_kind_is(&TokenKind::RBrace) && !self.is_eof() {
            let v_name = self.expect_ident("expected variant name");
            let v_start = self.last_span_start();
            let kind = if self.peek_kind_is(&TokenKind::LParen) {
                self.bump();
                let mut payload = Vec::new();
                if !self.peek_kind_is(&TokenKind::RParen) {
                    payload = self.parse_type_list();
                }
                self.expect(TokenKind::RParen, "expected ')' in variant");
                EnumVariantKind::Tuple(payload)
            } else if self.peek_kind_is(&TokenKind::LBrace) {
                self.bump();
                let fields = if self.peek_kind_is(&TokenKind::RBrace) {
                    Vec::new()
                } else {
                    self.parse_field_list()
                };
                self.expect(TokenKind::RBrace, "expected '}' in variant");
                EnumVariantKind::Struct(fields)
            } else {
                EnumVariantKind::Unit
            };
            let v_end = self.last_span_end(v_start);
            variants.push(EnumVariant {
                name: v_name,
                kind,
                span: Span::new(v_start, v_end),
            });
            if self.peek_kind_is(&TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        let end = self.expect(TokenKind::RBrace, "expected '}' after enum");
        EnumDecl {
            name,
            variants,
            span: Span::new(start, end),
        }
    }

    fn parse_fn_decl(&mut self) -> FnDecl {
        let start = self.expect(TokenKind::Fn, "expected 'fn'");
        let name = self.expect_ident("expected function name");
        self.expect(TokenKind::LParen, "expected '(' after fn name");
        let params = if self.peek_kind_is(&TokenKind::RParen) {
            Vec::new()
        } else {
            self.parse_param_list()
        };
        self.expect(TokenKind::RParen, "expected ')' after params");
        self.expect(TokenKind::Arrow, "expected '->' return type");
        let ret_type = self.parse_type();
        let body = self.parse_block();
        let end = body.span.end;
        FnDecl {
            name,
            params,
            ret_type,
            body,
            span: Span::new(start, end),
        }
    }

    fn parse_extern_fn_decl(&mut self) -> ExternFnDecl {
        let start = self.expect(TokenKind::Extern, "expected 'extern'");
        self.expect(TokenKind::Fn, "expected 'fn' after extern");
        let name = self.expect_ident("expected extern fn name");
        self.expect(TokenKind::LParen, "expected '(' after extern fn name");
        let mut params = Vec::new();
        let mut varargs = false;
        if !self.peek_kind_is(&TokenKind::RParen) {
            loop {
                if self.peek_kind_is(&TokenKind::Ellipsis) {
                    self.bump();
                    varargs = true;
                    break;
                }
                let param_name = self.expect_ident("expected param name");
                let p_start = self.last_span_start();
                self.expect(TokenKind::Colon, "expected ':' in param");
                let ty = self.parse_type();
                let p_end = self.last_span_end(p_start);
                params.push(Param {
                    name: param_name,
                    ty,
                    span: Span::new(p_start, p_end),
                });
                if self.peek_kind_is(&TokenKind::Comma) {
                    self.bump();
                    continue;
                }
                if self.peek_kind_is(&TokenKind::Ellipsis) {
                    self.bump();
                    varargs = true;
                }
                break;
            }
        }
        self.expect(TokenKind::RParen, "expected ')' after extern params");
        self.expect(TokenKind::Arrow, "expected '->' return type");
        let ret_type = self.parse_type();
        let end = self.expect(TokenKind::Semi, "expected ';' after extern fn");
        ExternFnDecl {
            name,
            params,
            varargs,
            ret_type,
            span: Span::new(start, end),
        }
    }

    fn parse_param_list(&mut self) -> Vec<Param> {
        let mut params = Vec::new();
        loop {
            let name = self.expect_ident("expected param name");
            let start = self.last_span_start();
            self.expect(TokenKind::Colon, "expected ':' in param");
            let ty = self.parse_type();
            let end = self.last_span_end(start);
            params.push(Param {
                name,
                ty,
                span: Span::new(start, end),
            });
            if self.peek_kind_is(&TokenKind::Comma) {
                self.bump();
                if self.peek_kind_is(&TokenKind::RParen) {
                    break;
                }
                continue;
            }
            break;
        }
        params
    }

    fn parse_type(&mut self) -> Type {
        if self.peek_kind_is(&TokenKind::LParen) {
            let start = self.bump().span.start;
            let mut tys = Vec::new();
            if !self.peek_kind_is(&TokenKind::RParen) {
                tys = self.parse_type_list();
            }
            self.expect(TokenKind::RParen, "expected ')' in tuple type");
            let _ = start;
            return Type::Tuple(tys);
        }
        if self.peek_ident_is("RawPointer") {
            self.bump();
            self.expect(TokenKind::Lt, "expected '<' after RawPointer");
            let inner = self.parse_type();
            self.expect(TokenKind::Gt, "expected '>' after RawPointer");
            return Type::RawPointer(Box::new(inner));
        }
        let path = self.parse_path();
        Type::Path(path)
    }

    fn parse_type_list(&mut self) -> Vec<Type> {
        let mut types = Vec::new();
        loop {
            let ty = self.parse_type();
            types.push(ty);
            if self.peek_kind_is(&TokenKind::Comma) {
                self.bump();
                if self.peek_kind_is(&TokenKind::RParen) {
                    break;
                }
                continue;
            }
            break;
        }
        types
    }

    fn parse_block(&mut self) -> Block {
        let start = self.expect(TokenKind::LBrace, "expected '{' to start block");
        let mut stmts = Vec::new();
        let mut tail: Option<Box<Expr>> = None;
        while !self.peek_kind_is(&TokenKind::RBrace) && !self.is_eof() {
            if self.peek_kind_is(&TokenKind::Return) {
                let s_start = self.bump().span.start;
                if self.peek_kind_is(&TokenKind::Semi) {
                    let end = self.expect(TokenKind::Semi, "expected ';' after return");
                    stmts.push(Stmt::Return(None, Span::new(s_start, end)));
                    continue;
                }
                let expr = self.parse_expr();
                let end = self.expect(TokenKind::Semi, "expected ';' after return expr");
                stmts.push(Stmt::Return(Some(expr), Span::new(s_start, end)));
                continue;
            }

            let expr = self.parse_expr();
            if self.peek_kind_is(&TokenKind::Semi) {
                let end = self.expect(TokenKind::Semi, "expected ';'");
                let span = Span::new(expr_span_start(&expr), end);
                stmts.push(Stmt::Expr(expr, span));
            } else {
                tail = Some(Box::new(expr));
                break;
            }
        }
        let end = self.expect(TokenKind::RBrace, "expected '}' to end block");
        Block {
            stmts,
            tail,
            span: Span::new(start, end),
        }
    }

    fn parse_expr(&mut self) -> Expr {
        if self.peek_kind_is(&TokenKind::Let) {
            return self.parse_let_expr();
        }
        if self.peek_kind_is(&TokenKind::If) {
            return self.parse_if_expr();
        }
        if self.peek_kind_is(&TokenKind::While) {
            return self.parse_while_expr();
        }
        if self.peek_kind_is(&TokenKind::Match) {
            return self.parse_match_expr();
        }
        if self.peek_kind_is(&TokenKind::LBrace) {
            let block = self.parse_block();
            return Expr::Block(Box::new(block));
        }
        self.parse_assign_expr()
    }

    fn parse_let_expr(&mut self) -> Expr {
        let start = self.expect(TokenKind::Let, "expected 'let'");
        let mutable = self.peek_kind_is(&TokenKind::Mut);
        if mutable {
            self.bump();
        }
        let name = self.expect_ident("expected identifier after let");
        let mut ty = None;
        if self.peek_kind_is(&TokenKind::Colon) {
            self.bump();
            ty = Some(self.parse_type());
        }
        self.expect(TokenKind::Eq, "expected '=' in let binding");
        let value = self.parse_expr();
        let end = expr_span_end(&value);
        Expr::Let {
            name,
            mutable,
            ty,
            value: Box::new(value),
            span: Span::new(start, end),
        }
    }

    fn parse_if_expr(&mut self) -> Expr {
        let start = self.expect(TokenKind::If, "expected 'if'");
        let cond = self.parse_expr();
        let then_branch = Box::new(self.parse_block());
        let else_branch = if self.peek_kind_is(&TokenKind::Else) {
            self.bump();
            Some(Box::new(self.parse_block()))
        } else {
            None
        };
        let end = then_branch.span.end;
        Expr::If {
            cond: Box::new(cond),
            then_branch,
            else_branch,
            span: Span::new(start, end),
        }
    }

    fn parse_while_expr(&mut self) -> Expr {
        let start = self.expect(TokenKind::While, "expected 'while'");
        let cond = self.parse_expr();
        let body = Box::new(self.parse_block());
        let end = body.span.end;
        Expr::While {
            cond: Box::new(cond),
            body,
            span: Span::new(start, end),
        }
    }

    fn parse_match_expr(&mut self) -> Expr {
        let start = self.expect(TokenKind::Match, "expected 'match'");
        let scrutinee = self.parse_expr();
        self.expect(TokenKind::LBrace, "expected '{' after match scrutinee");
        let mut arms = Vec::new();
        while !self.peek_kind_is(&TokenKind::RBrace) && !self.is_eof() {
            let (pattern, p_span) = self.parse_pattern();
            self.expect(TokenKind::FatArrow, "expected '=>' in match arm");
            let body = self.parse_expr();
            let arm_span = Span::new(p_span.start, expr_span_end(&body));
            arms.push(MatchArm {
                pattern,
                body,
                span: arm_span,
            });
            if self.peek_kind_is(&TokenKind::Comma) {
                self.bump();
                if self.peek_kind_is(&TokenKind::RBrace) {
                    break;
                }
                continue;
            }
            break;
        }
        let end = self.expect(TokenKind::RBrace, "expected '}' after match arms");
        Expr::Match {
            scrutinee: Box::new(scrutinee),
            arms,
            span: Span::new(start, end),
        }
    }

    fn parse_assign_expr(&mut self) -> Expr {
        let left = self.parse_binary_expr(0);
        if self.peek_kind_is(&TokenKind::Eq) {
            let start = expr_span_start(&left);
            self.bump();
            let right = self.parse_expr();
            let end = expr_span_end(&right);
            return Expr::Assign {
                target: Box::new(left),
                value: Box::new(right),
                span: Span::new(start, end),
            };
        }
        left
    }

    fn parse_binary_expr(&mut self, min_prec: u8) -> Expr {
        let mut left = self.parse_unary_expr();
        loop {
            let (op, prec, right_assoc) = match self.peek_binary_op() {
                Some(info) => info,
                None => break,
            };
            if prec < min_prec {
                break;
            }
            self.bump();
            let next_min = if right_assoc { prec } else { prec + 1 };
            let right = self.parse_binary_expr(next_min);
            let span = Span::new(expr_span_start(&left), expr_span_end(&right));
            left = Expr::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            };
        }
        left
    }

    fn parse_unary_expr(&mut self) -> Expr {
        if self.peek_kind_is(&TokenKind::Bang) {
            let start = self.bump().span.start;
            let expr = self.parse_unary_expr();
            let end = expr_span_end(&expr);
            return Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(expr),
                span: Span::new(start, end),
            };
        }
        if self.peek_kind_is(&TokenKind::Minus) {
            let start = self.bump().span.start;
            let expr = self.parse_unary_expr();
            let end = expr_span_end(&expr);
            return Expr::Unary {
                op: UnaryOp::Neg,
                expr: Box::new(expr),
                span: Span::new(start, end),
            };
        }
        self.parse_call_expr()
    }

    fn parse_call_expr(&mut self) -> Expr {
        let mut expr = self.parse_primary_expr();
        loop {
            if self.peek_kind_is(&TokenKind::LParen) {
                let start = expr_span_start(&expr);
                self.bump();
                let mut args = Vec::new();
                if !self.peek_kind_is(&TokenKind::RParen) {
                    loop {
                        let arg = self.parse_expr();
                        args.push(arg);
                        if self.peek_kind_is(&TokenKind::Comma) {
                            self.bump();
                            if self.peek_kind_is(&TokenKind::RParen) {
                                break;
                            }
                            continue;
                        }
                        break;
                    }
                }
                let end = self.expect(TokenKind::RParen, "expected ')' after args");
                expr = Expr::Call {
                    callee: Box::new(expr),
                    args,
                    span: Span::new(start, end),
                };
                continue;
            }
            if self.peek_kind_is(&TokenKind::Dot) {
                let start = expr_span_start(&expr);
                self.bump();
                let name = self.expect_ident("expected field name after '.'");
                let end = self.last_span_end(start);
                expr = Expr::Field {
                    base: Box::new(expr),
                    name,
                    span: Span::new(start, end),
                };
                continue;
            }
            break;
        }
        expr
    }

    fn parse_primary_expr(&mut self) -> Expr {
        match self.peek_kind().cloned() {
            Some(TokenKind::Int(text)) => {
                let tok = self.bump();
                Expr::Literal(Literal::Int(text), tok.span)
            }
            Some(TokenKind::Float(text)) => {
                let tok = self.bump();
                Expr::Literal(Literal::Float(text), tok.span)
            }
            Some(TokenKind::Str(text)) => {
                let tok = self.bump();
                Expr::Literal(Literal::Str(text), tok.span)
            }
            Some(TokenKind::True) => {
                let tok = self.bump();
                Expr::Literal(Literal::Bool(true), tok.span)
            }
            Some(TokenKind::False) => {
                let tok = self.bump();
                Expr::Literal(Literal::Bool(false), tok.span)
            }
            Some(TokenKind::LParen) => {
                self.bump();
                let expr = self.parse_expr();
                self.expect(TokenKind::RParen, "expected ')' after expression");
                expr
            }
            Some(TokenKind::Ident(_)) => {
                let path = self.parse_path();
                let start = self.last_span_start();
                if self.peek_kind_is(&TokenKind::LBrace) && self.looks_like_struct_literal() {
                    self.bump();
                    let mut fields = Vec::new();
                    if !self.peek_kind_is(&TokenKind::RBrace) {
                        loop {
                            let name = self.expect_ident("expected field name in struct literal");
                            self.expect(TokenKind::Colon, "expected ':' in struct literal");
                            let value = self.parse_expr();
                            fields.push((name, value));
                            if self.peek_kind_is(&TokenKind::Comma) {
                                self.bump();
                                if self.peek_kind_is(&TokenKind::RBrace) {
                                    break;
                                }
                                continue;
                            }
                            break;
                        }
                    }
                    let end = self.expect(TokenKind::RBrace, "expected '}' after struct literal");
                    return Expr::StructLit {
                        path,
                        fields,
                        span: Span::new(start, end),
                    };
                }
                let end = self.last_span_end(start);
                Expr::Path(path, Span::new(start, end))
            }
            Some(TokenKind::LBrace) => {
                let block = self.parse_block();
                Expr::Block(Box::new(block))
            }
            _ => {
                let span = self.peek_span().unwrap_or(Span::new(0, 0));
                self.errors.push(ParseError {
                    message: "expected expression".to_string(),
                    span,
                });
                Expr::Literal(Literal::Bool(false), span)
            }
        }
    }

    fn parse_pattern(&mut self) -> (Pattern, Span) {
        match self.peek_kind().cloned() {
            Some(TokenKind::Ident(name)) => {
                let tok = self.bump();
                if name == "_" {
                    return (Pattern::Wildcard(tok.span), tok.span);
                }
                // path starts with already consumed ident
                let mut segments = vec![name];
                while self.peek_kind_is(&TokenKind::ColonColon) {
                    self.bump();
                    let seg = self.expect_ident("expected identifier after '::' in pattern");
                    segments.push(seg);
                }
                if self.peek_kind_is(&TokenKind::LBrace) {
                    self.bump();
                    let fields = self.parse_pattern_fields();
                    let end = self.expect(TokenKind::RBrace, "expected '}' in pattern");
                    let span = Span::new(tok.span.start, end);
                    (
                        Pattern::Struct {
                            path: segments,
                            fields,
                            span,
                        },
                        span,
                    )
                } else {
                    let span = Span::new(tok.span.start, self.last_span_end(tok.span.end));
                    (Pattern::Path(segments, span), span)
                }
            }
            _ => {
                let span = self.peek_span().unwrap_or(Span::new(0, 0));
                self.errors.push(ParseError {
                    message: "expected pattern".to_string(),
                    span,
                });
                (Pattern::Wildcard(span), span)
            }
        }
    }

    fn parse_pattern_fields(&mut self) -> Vec<PatternField> {
        let mut fields = Vec::new();
        while !self.peek_kind_is(&TokenKind::RBrace) && !self.is_eof() {
            let name = self.expect_ident("expected field name in pattern");
            let start = self.last_span_start();
            let mut binding = None;
            if self.peek_kind_is(&TokenKind::Colon) {
                self.bump();
                let bind = self.expect_ident("expected binding name in pattern");
                if bind != "_" {
                    binding = Some(bind);
                }
            } else {
                binding = Some(name.clone());
            }
            let end = self.last_span_end(start);
            fields.push(PatternField {
                name,
                binding,
                span: Span::new(start, end),
            });
            if self.peek_kind_is(&TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        fields
    }

    fn parse_field_list(&mut self) -> Vec<Field> {
        let mut fields = Vec::new();
        while !self.peek_kind_is(&TokenKind::RBrace) && !self.is_eof() {
            let field_name = self.expect_ident("expected field name");
            let field_start = self.last_span_start();
            self.expect(TokenKind::Colon, "expected ':' in field");
            let ty = self.parse_type();
            let field_end = self.last_span_end(field_start);
            fields.push(Field {
                name: field_name,
                ty,
                span: Span::new(field_start, field_end),
            });
            if self.peek_kind_is(&TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
        }
        fields
    }

    fn looks_like_struct_literal(&self) -> bool {
        if !self.peek_kind_is(&TokenKind::LBrace) {
            return false;
        }
        match self.peek_kind_n(1) {
            Some(TokenKind::RBrace) => true,
            Some(TokenKind::Ident(_)) => matches!(self.peek_kind_n(2), Some(TokenKind::Colon)),
            _ => false,
        }
    }

    fn peek_kind_n(&self, n: usize) -> Option<&TokenKind> {
        self.tokens.get(self.pos + n).map(|t| &t.kind)
    }

    fn parse_path(&mut self) -> Vec<String> {
        let mut segments = Vec::new();
        let first = self.expect_ident("expected identifier in path");
        segments.push(first);
        while self.peek_kind_is(&TokenKind::ColonColon) {
            self.bump();
            let seg = self.expect_ident("expected identifier after '::'");
            segments.push(seg);
        }
        segments
    }

    fn peek_binary_op(&self) -> Option<(BinaryOp, u8, bool)> {
        match self.peek_kind()? {
            TokenKind::Star => Some((BinaryOp::Mul, 5, false)),
            TokenKind::Slash => Some((BinaryOp::Div, 5, false)),
            TokenKind::Percent => Some((BinaryOp::Rem, 5, false)),
            TokenKind::Plus => Some((BinaryOp::Add, 4, false)),
            TokenKind::Minus => Some((BinaryOp::Sub, 4, false)),
            TokenKind::EqEq => Some((BinaryOp::Eq, 3, false)),
            TokenKind::NotEq => Some((BinaryOp::NotEq, 3, false)),
            TokenKind::Lt => Some((BinaryOp::Lt, 3, false)),
            TokenKind::LtEq => Some((BinaryOp::LtEq, 3, false)),
            TokenKind::Gt => Some((BinaryOp::Gt, 3, false)),
            TokenKind::GtEq => Some((BinaryOp::GtEq, 3, false)),
            TokenKind::AndAnd => Some((BinaryOp::AndAnd, 2, false)),
            TokenKind::OrOr => Some((BinaryOp::OrOr, 1, false)),
            _ => None,
        }
    }

    fn expect(&mut self, kind: TokenKind, msg: &str) -> usize {
        if self.peek_kind_is(&kind) {
            return self.bump().span.end;
        }
        let span = self.peek_span().unwrap_or(Span::new(0, 0));
        self.errors.push(ParseError {
            message: msg.to_string(),
            span,
        });
        span.end
    }

    fn expect_ident(&mut self, msg: &str) -> String {
        match self.peek_kind() {
            Some(TokenKind::Ident(name)) => {
                let name = name.clone();
                self.bump();
                name
            }
            _ => {
                let span = self.peek_span().unwrap_or(Span::new(0, 0));
                self.errors.push(ParseError {
                    message: msg.to_string(),
                    span,
                });
                "<error>".to_string()
            }
        }
    }

    fn peek_ident_is(&self, text: &str) -> bool {
        match self.peek_kind() {
            Some(TokenKind::Ident(name)) => name == text,
            _ => false,
        }
    }

    fn peek_kind(&self) -> Option<&TokenKind> {
        self.tokens.get(self.pos).map(|t| &t.kind)
    }

    fn peek_kind_is(&self, kind: &TokenKind) -> bool {
        self.peek_kind().map_or(false, |k| k == kind)
    }

    fn peek_span(&self) -> Option<Span> {
        self.tokens.get(self.pos).map(|t| t.span)
    }

    fn bump(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token {
            kind: TokenKind::Semi,
            span: Span::new(0, 0),
        });
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn last_span_start(&self) -> usize {
        self.tokens
            .get(self.pos.saturating_sub(1))
            .map(|t| t.span.start)
            .unwrap_or(0)
    }

    fn last_span_end(&self, fallback: usize) -> usize {
        self.tokens
            .get(self.pos.saturating_sub(1))
            .map(|t| t.span.end)
            .unwrap_or(fallback)
    }
}

fn expr_span_start(expr: &Expr) -> usize {
    match expr {
        Expr::Let { span, .. }
        | Expr::If { span, .. }
        | Expr::While { span, .. }
        | Expr::Match { span, .. }
        | Expr::Assign { span, .. }
        | Expr::Binary { span, .. }
        | Expr::Unary { span, .. }
        | Expr::Call { span, .. }
        | Expr::Field { span, .. }
        | Expr::StructLit { span, .. }
        | Expr::Literal(_, span)
        | Expr::Path(_, span) => span.start,
        Expr::Block(block) => block.span.start,
    }
}

fn expr_span_end(expr: &Expr) -> usize {
    match expr {
        Expr::Let { span, .. }
        | Expr::If { span, .. }
        | Expr::While { span, .. }
        | Expr::Match { span, .. }
        | Expr::Assign { span, .. }
        | Expr::Binary { span, .. }
        | Expr::Unary { span, .. }
        | Expr::Call { span, .. }
        | Expr::Field { span, .. }
        | Expr::StructLit { span, .. }
        | Expr::Literal(_, span)
        | Expr::Path(_, span) => span.end,
        Expr::Block(block) => block.span.end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse(src: &str) -> Module {
        let tokens = Lexer::new(src).lex_all().unwrap();
        Parser::new(tokens).parse_module().unwrap()
    }

    #[test]
    fn parse_extern_fn_varargs() {
        let src = r#"
            extern fn printf(fmt: RawPointer<I8>, ...) -> I32;
        "#;
        let module = parse(src);
        assert!(matches!(module.items[0], Item::ExternFn(_)));
    }

    #[test]
    fn parse_match_expr() {
        let src = r#"
            fn main() -> I64 {
                match x { _ => 1, }
            }
        "#;
        let module = parse(src);
        if let Item::Fn(f) = &module.items[0] {
            if let Some(tail) = &f.body.tail {
                match &**tail {
                    Expr::Match { arms, .. } => assert_eq!(arms.len(), 1),
                    _ => panic!("expected match expression"),
                }
            } else {
                panic!("expected tail expression");
            }
        } else {
            panic!("expected function item");
        }
    }

    #[test]
    fn parse_struct_literal_and_field() {
        let src = r#"
            struct User { id: I64 }
            fn main() -> I64 {
                let u = User { id: 1 };
                u.id
            }
        "#;
        let module = parse(src);
        assert_eq!(module.items.len(), 2);
    }

    #[test]
    fn parse_enum_struct_variant() {
        let src = r#"
            enum Opt { Some { value: I64 }, None {} }
        "#;
        let module = parse(src);
        if let Item::Enum(e) = &module.items[0] {
            assert_eq!(e.variants.len(), 2);
        } else {
            panic!("expected enum item");
        }
    }
}
