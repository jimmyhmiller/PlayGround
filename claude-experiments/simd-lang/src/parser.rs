use crate::ast::*;
use crate::lexer::{Lexer, Spanned, Token};

pub struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Spanned>) -> Self {
        Parser { tokens, pos: 0 }
    }

    // --- Helpers ---

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .map(|s| &s.token)
            .unwrap_or(&Token::Eof)
    }

    fn peek_at(&self, offset: usize) -> &Token {
        self.tokens
            .get(self.pos + offset)
            .map(|s| &s.token)
            .unwrap_or(&Token::Eof)
    }

    fn expect(&mut self, expected: &Token) {
        let tok = self.peek().clone();
        if &tok != expected {
            let span = &self.tokens[self.pos.min(self.tokens.len() - 1)];
            panic!(
                "expected {:?}, got {:?} at {}:{}",
                expected, tok, span.line, span.col
            );
        }
        self.pos += 1;
    }

    fn current_span(&self) -> Span {
        self.tokens.get(self.pos).map(|s| Span::new(s.line, s.col)).unwrap_or(Span::dummy())
    }

    fn expect_ident(&mut self) -> String {
        match self.peek().clone() {
            Token::Ident(name) => {
                self.pos += 1;
                name
            }
            other => {
                let span = &self.tokens[self.pos.min(self.tokens.len() - 1)];
                panic!(
                    "expected identifier, got {:?} at {}:{}",
                    other, span.line, span.col
                );
            }
        }
    }

    // --- Top-level ---

    pub fn parse_program(&mut self) -> Vec<Item> {
        let mut items = Vec::new();
        while *self.peek() != Token::Eof {
            match self.peek() {
                Token::Struct => items.push(Item::Struct(self.parse_struct_def())),
                Token::Fn => items.push(Item::Fn(self.parse_fn_def())),
                other => panic!("expected 'struct' or 'fn', got {:?}", other),
            }
        }
        items
    }

    fn parse_comptime_params(&mut self) -> Vec<ComptimeParam> {
        if *self.peek() != Token::LBracket {
            return vec![];
        }
        self.pos += 1; // consume [
        let mut params = Vec::new();
        while *self.peek() != Token::RBracket {
            let name = self.expect_ident();
            // Optional ": comptime" annotation
            if *self.peek() == Token::Colon {
                self.pos += 1;
                self.expect(&Token::Comptime);
            }
            params.push(ComptimeParam { name });
            if *self.peek() == Token::Comma {
                self.pos += 1;
            }
        }
        self.pos += 1; // consume ]
        params
    }

    fn parse_struct_def(&mut self) -> StructDef {
        self.expect(&Token::Struct);
        let name = self.expect_ident();
        let comptime_params = self.parse_comptime_params();
        self.expect(&Token::LBrace);

        let mut fields = Vec::new();
        while *self.peek() != Token::RBrace {
            let field_name = self.expect_ident();
            self.expect(&Token::Colon);
            let ty = self.parse_type();
            fields.push(FieldDef {
                name: field_name,
                ty,
            });
            if *self.peek() == Token::Comma {
                self.pos += 1;
            }
        }
        self.expect(&Token::RBrace);

        StructDef {
            name,
            comptime_params,
            fields,
        }
    }

    fn parse_fn_def(&mut self) -> FnDef {
        self.expect(&Token::Fn);
        let name = self.expect_ident();
        let comptime_params = self.parse_comptime_params();
        self.expect(&Token::LParen);

        let mut params = Vec::new();
        while *self.peek() != Token::RParen {
            let param_name = self.expect_ident();
            self.expect(&Token::Colon);
            let ty = self.parse_type();
            params.push(Param {
                name: param_name,
                ty,
            });
            if *self.peek() == Token::Comma {
                self.pos += 1;
            }
        }
        self.expect(&Token::RParen);

        let ret_ty = if *self.peek() == Token::Arrow {
            self.pos += 1;
            Some(self.parse_type())
        } else {
            None
        };

        self.expect(&Token::LBrace);
        let mut body = Vec::new();
        while *self.peek() != Token::RBrace {
            body.push(self.parse_stmt());
        }
        self.expect(&Token::RBrace);

        FnDef {
            name,
            comptime_params,
            params,
            ret_ty,
            body,
        }
    }

    fn parse_type(&mut self) -> Type {
        let name = self.expect_ident();
        let width = if *self.peek() == Token::LBracket {
            self.pos += 1;
            let w = self.parse_width();
            self.expect(&Token::RBracket);
            Some(w)
        } else {
            None
        };
        Type { name, width }
    }

    /// Parse a compile-time width expression.
    ///
    /// Grammar:
    ///   width_expr = width_term (('+' | '-') width_term)*
    ///   width_term = width_atom (('*' | '/') width_atom)*
    ///   width_atom = INT | '_' | IDENT | '(' width_expr ')'
    fn parse_width(&mut self) -> Width {
        self.parse_width_add()
    }

    fn parse_width_add(&mut self) -> Width {
        let mut left = self.parse_width_mul();
        loop {
            let op = match self.peek() {
                Token::Plus => ComptimeOp::Add,
                Token::Minus => ComptimeOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let right = self.parse_width_mul();
            left = Width::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        left
    }

    fn parse_width_mul(&mut self) -> Width {
        let mut left = self.parse_width_atom();
        loop {
            let op = match self.peek() {
                Token::Star => ComptimeOp::Mul,
                Token::Slash => ComptimeOp::Div,
                _ => break,
            };
            self.pos += 1;
            let right = self.parse_width_atom();
            left = Width::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        left
    }

    fn parse_width_atom(&mut self) -> Width {
        match self.peek().clone() {
            Token::IntLit(n) => {
                self.pos += 1;
                Width::Fixed(n as u64)
            }
            Token::Underscore => {
                self.pos += 1;
                Width::Native
            }
            Token::Ident(name) => {
                self.pos += 1;
                Width::Param(name)
            }
            Token::LParen => {
                self.pos += 1;
                let w = self.parse_width();
                self.expect(&Token::RParen);
                w
            }
            other => panic!("expected width expression (integer, _, identifier, or '('), got {:?}", other),
        }
    }

    // --- Statements ---

    fn parse_stmt(&mut self) -> Stmt {
        let span = self.current_span();

        if *self.peek() == Token::Return {
            self.pos += 1;
            return Stmt { kind: StmtKind::Return(self.parse_expr()), span };
        }

        // Stream: stream chunk: u8[64] over input carry (...) { ... }
        if *self.peek() == Token::Stream {
            return self.parse_stream();
        }

        // If/else: if cond { ... } else { ... }
        if *self.peek() == Token::If {
            return self.parse_if(span);
        }

        // While: while cond { ... }
        if *self.peek() == Token::While {
            return self.parse_while(span);
        }

        // Destructuring: (a, b) = expr
        if *self.peek() == Token::LParen {
            if let Some(stmt) = self.try_parse_destructure() {
                return stmt;
            }
        }

        // Typed assignment: ident: type = expr
        // Check for ident followed by colon where next token is a type name (ident) then [ or =
        if let Token::Ident(_) = self.peek() {
            if *self.peek_at(1) == Token::Colon {
                if let Token::Ident(_) = self.peek_at(2) {
                    // Could be typed assignment: name: type = expr
                    // Look ahead to confirm: type is ident optionally followed by [width], then =
                    let saved = self.pos;
                    let name = self.expect_ident();
                    self.pos += 1; // consume :
                    let ty = self.parse_type();
                    if *self.peek() == Token::Eq {
                        self.pos += 1;
                        let value = self.parse_expr();
                        return Stmt {
                            kind: StmtKind::Assign {
                                target: AssignTarget::Ident(name),
                                ty: Some(ty),
                                value,
                            },
                            span,
                        };
                    }
                    // Not a typed assignment, backtrack
                    self.pos = saved;
                }
            }
        }

        // Parse expression, check for assignment
        let expr = self.parse_expr();
        if *self.peek() == Token::Eq {
            self.pos += 1;
            let value = self.parse_expr();
            let target = match expr {
                Expr::Ident(name) => AssignTarget::Ident(name),
                Expr::Gather { base, index, mask } => AssignTarget::Scatter { base, index, mask },
                _ => panic!("invalid assignment target: {:?}", expr),
            };
            Stmt { kind: StmtKind::Assign { target, ty: None, value }, span }
        } else {
            Stmt { kind: StmtKind::Expr(expr), span }
        }
    }

    fn try_parse_destructure(&mut self) -> Option<Stmt> {
        let span = self.current_span();
        let saved = self.pos;

        self.pos += 1; // consume (

        let mut names = Vec::new();
        loop {
            match self.peek() {
                Token::Ident(_) => {
                    names.push(self.expect_ident());
                }
                _ => {
                    self.pos = saved;
                    return None;
                }
            }
            match self.peek() {
                Token::Comma => {
                    self.pos += 1;
                }
                Token::RParen => break,
                _ => {
                    self.pos = saved;
                    return None;
                }
            }
        }
        self.pos += 1; // consume )

        if *self.peek() != Token::Eq {
            self.pos = saved;
            return None;
        }
        self.pos += 1; // consume =

        let value = self.parse_expr();
        Some(Stmt {
            kind: StmtKind::Assign {
                target: AssignTarget::Destructure(names),
                ty: None,
                value,
            },
            span,
        })
    }

    // --- Stream ---

    /// Parse: stream chunk: u8[64] over buf carry (in_string: bool[1] = false) { ... carry x = expr }
    fn parse_stream(&mut self) -> Stmt {
        let span = self.current_span();
        self.expect(&Token::Stream);
        let chunk_name = self.expect_ident();
        self.expect(&Token::Colon);
        let chunk_ty = self.parse_type();
        self.expect(&Token::Over);
        let buffer = self.expect_ident();

        // Parse carry state: carry (name: type = init, ...)
        let mut carry = Vec::new();
        if *self.peek() == Token::Carry {
            self.pos += 1;
            self.expect(&Token::LParen);
            loop {
                if *self.peek() == Token::RParen {
                    break;
                }
                let name = self.expect_ident();
                self.expect(&Token::Colon);
                let ty = self.parse_type();
                self.expect(&Token::Eq);
                let init = self.parse_expr();
                carry.push(CarryDef { name, ty, init });
                if *self.peek() == Token::Comma {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            self.expect(&Token::RParen);
        }

        self.expect(&Token::LBrace);

        // Parse body statements and carry updates
        let mut body = Vec::new();
        let mut carry_updates = Vec::new();

        while *self.peek() != Token::RBrace {
            // Check for carry update: carry name = expr
            if *self.peek() == Token::Carry {
                self.pos += 1;
                let name = self.expect_ident();
                self.expect(&Token::Eq);
                let value = self.parse_expr();
                carry_updates.push((name, value));
            } else {
                body.push(self.parse_stmt());
            }
        }
        self.expect(&Token::RBrace);

        Stmt {
            kind: StmtKind::Stream {
                chunk_name,
                chunk_ty,
                buffer,
                carry,
                body,
                carry_updates,
            },
            span,
        }
    }

    /// Parse: if cond { ... } else { ... }
    fn parse_if(&mut self, span: Span) -> Stmt {
        self.expect(&Token::If);
        let cond = self.parse_expr();
        self.expect(&Token::LBrace);
        let mut then_body = Vec::new();
        while *self.peek() != Token::RBrace {
            then_body.push(self.parse_stmt());
        }
        self.expect(&Token::RBrace);

        let mut else_body = Vec::new();
        if *self.peek() == Token::Else {
            self.pos += 1;
            self.expect(&Token::LBrace);
            while *self.peek() != Token::RBrace {
                else_body.push(self.parse_stmt());
            }
            self.expect(&Token::RBrace);
        }

        Stmt {
            kind: StmtKind::If {
                cond,
                then_body,
                else_body,
            },
            span,
        }
    }

    /// Parse: while cond { ... }
    fn parse_while(&mut self, span: Span) -> Stmt {
        self.expect(&Token::While);
        let cond = self.parse_expr();
        self.expect(&Token::LBrace);
        let mut body = Vec::new();
        while *self.peek() != Token::RBrace {
            body.push(self.parse_stmt());
        }
        self.expect(&Token::RBrace);

        Stmt {
            kind: StmtKind::While { cond, body },
            span,
        }
    }

    // --- Expressions ---

    fn parse_expr(&mut self) -> Expr {
        if *self.peek() == Token::LBracket {
            // Could be:
            // [type: val, val, ...] — constant vector literal
            // [mask] body : fallback — masked expression
            //
            // Distinguish by: if [ is followed by Ident then :, it's a vec literal
            if let Token::Ident(_) = self.peek_at(1) {
                if *self.peek_at(2) == Token::Colon {
                    return self.parse_vec_lit();
                }
            }

            // Masked expression
            self.pos += 1;
            let mask = self.parse_or();
            self.expect(&Token::RBracket);
            let body = self.parse_or();
            let fallback = if *self.peek() == Token::Colon {
                self.pos += 1;
                Some(Box::new(self.parse_or()))
            } else {
                None
            };
            Expr::Masked {
                mask: Box::new(mask),
                body: Box::new(body),
                fallback,
            }
        } else {
            self.parse_or()
        }
    }

    /// Parse [type: val, val, ...] or [type: val; count] — constant vector literal
    fn parse_vec_lit(&mut self) -> Expr {
        self.expect(&Token::LBracket);
        let elem_type = self.expect_ident();
        self.expect(&Token::Colon);

        // Parse first value
        let neg = if *self.peek() == Token::Minus {
            self.pos += 1;
            true
        } else {
            false
        };
        let first = match self.peek().clone() {
            Token::IntLit(n) => {
                self.pos += 1;
                if neg { -n } else { n }
            }
            _ => panic!("expected integer in vector literal"),
        };

        // Check for repeat syntax: [type: val; count]
        if *self.peek() == Token::Semicolon {
            self.pos += 1;
            let count = match self.peek().clone() {
                Token::IntLit(n) => {
                    self.pos += 1;
                    n as usize
                }
                _ => panic!("expected integer count after ';' in vector repeat"),
            };
            self.expect(&Token::RBracket);
            return Expr::VecLit {
                elem_type,
                values: vec![first; count],
            };
        }

        let mut values = vec![first];
        while *self.peek() == Token::Comma {
            self.pos += 1;
            // Handle negative numbers
            let neg = if *self.peek() == Token::Minus {
                self.pos += 1;
                true
            } else {
                false
            };
            match self.peek().clone() {
                Token::IntLit(n) => {
                    self.pos += 1;
                    values.push(if neg { -n } else { n });
                }
                _ => panic!("expected integer in vector literal"),
            }
        }
        self.expect(&Token::RBracket);
        Expr::VecLit { elem_type, values }
    }

    fn parse_or(&mut self) -> Expr {
        let mut left = self.parse_xor();
        while *self.peek() == Token::Pipe {
            self.pos += 1;
            let right = self.parse_xor();
            left = Expr::BinOp {
                op: BinOp::Or,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        left
    }

    fn parse_xor(&mut self) -> Expr {
        let mut left = self.parse_and();
        while *self.peek() == Token::Caret {
            self.pos += 1;
            let right = self.parse_and();
            left = Expr::BinOp {
                op: BinOp::Xor,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        left
    }

    fn parse_and(&mut self) -> Expr {
        let mut left = self.parse_comparison();
        while *self.peek() == Token::Amp {
            self.pos += 1;
            let right = self.parse_comparison();
            left = Expr::BinOp {
                op: BinOp::And,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        left
    }

    fn parse_comparison(&mut self) -> Expr {
        let left = self.parse_shift();
        let op = match self.peek() {
            Token::Gt => BinOp::Gt,
            Token::Lt => BinOp::Lt,
            Token::GtEq => BinOp::GtEq,
            Token::LtEq => BinOp::LtEq,
            Token::EqEq => BinOp::EqEq,
            Token::BangEq => BinOp::NotEq,
            _ => return left,
        };
        self.pos += 1;
        let right = self.parse_shift();
        Expr::BinOp {
            op,
            lhs: Box::new(left),
            rhs: Box::new(right),
        }
    }

    fn parse_shift(&mut self) -> Expr {
        let mut left = self.parse_add();
        loop {
            let op = match self.peek() {
                Token::LtLt => BinOp::BitShl,
                Token::GtGt => BinOp::BitShr,
                _ => break,
            };
            self.pos += 1;
            let right = self.parse_add();
            left = Expr::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        left
    }

    fn parse_add(&mut self) -> Expr {
        let mut left = self.parse_mul();
        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let right = self.parse_mul();
            left = Expr::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        left
    }

    fn parse_mul(&mut self) -> Expr {
        let mut left = self.parse_unary();
        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                _ => break,
            };
            self.pos += 1;
            let right = self.parse_unary();
            left = Expr::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }
        left
    }

    fn parse_unary(&mut self) -> Expr {
        match self.peek() {
            Token::Tilde => {
                self.pos += 1;
                let operand = self.parse_unary();
                Expr::UnaryOp {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                }
            }
            Token::Minus => {
                self.pos += 1;
                let operand = self.parse_unary();
                Expr::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                }
            }
            Token::PlusSlash => {
                self.pos += 1;
                Expr::Reduction {
                    op: ReductionOp::Add,
                    operand: Box::new(self.parse_unary()),
                }
            }
            Token::StarSlash => {
                self.pos += 1;
                Expr::Reduction {
                    op: ReductionOp::Mul,
                    operand: Box::new(self.parse_unary()),
                }
            }
            Token::PipeSlash => {
                self.pos += 1;
                Expr::Reduction {
                    op: ReductionOp::Or,
                    operand: Box::new(self.parse_unary()),
                }
            }
            Token::AmpSlash => {
                self.pos += 1;
                Expr::Reduction {
                    op: ReductionOp::And,
                    operand: Box::new(self.parse_unary()),
                }
            }
            Token::MaxSlash => {
                self.pos += 1;
                Expr::Reduction {
                    op: ReductionOp::Max,
                    operand: Box::new(self.parse_unary()),
                }
            }
            Token::MinSlash => {
                self.pos += 1;
                Expr::Reduction {
                    op: ReductionOp::Min,
                    operand: Box::new(self.parse_unary()),
                }
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Expr {
        let mut expr = self.parse_primary();
        loop {
            match self.peek() {
                Token::Dot => {
                    self.pos += 1;
                    let field = self.expect_ident();
                    expr = Expr::Field {
                        base: Box::new(expr),
                        field,
                    };
                }
                Token::DotBracket => {
                    self.pos += 1;
                    let index = self.parse_expr();
                    let mask = if *self.peek() == Token::Comma {
                        self.pos += 1;
                        Some(Box::new(self.parse_expr()))
                    } else {
                        None
                    };
                    self.expect(&Token::RBracket);
                    expr = Expr::Gather {
                        base: Box::new(expr),
                        index: Box::new(index),
                        mask,
                    };
                }
                Token::LParen => {
                    self.pos += 1;
                    let args = self.parse_call_args();
                    self.expect(&Token::RParen);
                    expr = Expr::Call {
                        func: Box::new(expr),
                        args,
                    };
                }
                _ => break,
            }
        }
        expr
    }

    fn parse_call_args(&mut self) -> Vec<CallArg> {
        let mut args = Vec::new();
        while *self.peek() != Token::RParen {
            let arg = if let Token::Ident(_) = self.peek() {
                if *self.peek_at(1) == Token::Eq {
                    let name = self.expect_ident();
                    self.pos += 1; // consume =
                    let value = self.parse_expr();
                    CallArg {
                        name: Some(name),
                        value,
                    }
                } else {
                    CallArg {
                        name: None,
                        value: self.parse_expr(),
                    }
                }
            } else {
                CallArg {
                    name: None,
                    value: self.parse_expr(),
                }
            };
            args.push(arg);
            if *self.peek() == Token::Comma {
                self.pos += 1;
            }
        }
        args
    }

    fn parse_primary(&mut self) -> Expr {
        match self.peek().clone() {
            Token::IntLit(n) => {
                self.pos += 1;
                Expr::IntLit(n)
            }
            Token::FloatLit(f) => {
                self.pos += 1;
                Expr::FloatLit(f)
            }
            Token::True => {
                self.pos += 1;
                Expr::BoolLit(true)
            }
            Token::False => {
                self.pos += 1;
                Expr::BoolLit(false)
            }
            Token::CharLit(c) => {
                self.pos += 1;
                Expr::CharLit(c)
            }
            Token::LParen => {
                self.pos += 1;
                let expr = self.parse_expr();
                self.expect(&Token::RParen);
                expr
            }
            Token::Ident(name) => {
                // Check for special forms before consuming
                if name == "scan" && *self.peek_at(1) == Token::Dot {
                    return self.parse_scan();
                }
                if (name == "load" || name == "loadu" || name == "load_at") && *self.peek_at(1) == Token::LBracket {
                    return self.parse_load();
                }

                // Check for struct literal: Name[width] { ... }
                if *self.peek_at(1) == Token::LBracket {
                    if let Some(lit) = self.try_parse_struct_lit() {
                        return lit;
                    }
                }

                self.pos += 1;
                Expr::Ident(name)
            }
            other => {
                let span = &self.tokens[self.pos.min(self.tokens.len() - 1)];
                panic!(
                    "expected expression, got {:?} at {}:{}",
                    other, span.line, span.col
                );
            }
        }
    }

    fn parse_scan(&mut self) -> Expr {
        self.pos += 1; // consume "scan"
        self.pos += 1; // consume "."
        let op_name = self.expect_ident();
        let op = match op_name.as_str() {
            "add" => ScanOp::Add,
            "xor" => ScanOp::Xor,
            "max" => ScanOp::Max,
            "preceding_any" => ScanOp::PrecedingAny,
            _ => panic!("unknown scan operation: {}", op_name),
        };
        self.expect(&Token::LParen);
        let operand = self.parse_expr();
        let seed = if *self.peek() == Token::Comma {
            self.pos += 1;
            Some(Box::new(self.parse_expr()))
        } else {
            None
        };
        self.expect(&Token::RParen);
        Expr::Scan {
            op,
            operand: Box::new(operand),
            seed,
        }
    }

    fn parse_load(&mut self) -> Expr {
        let name = self.expect_ident();
        let aligned = name == "load" || name == "load_at";
        let has_offset = name == "load_at";
        self.expect(&Token::LBracket);
        let ty = self.parse_type();
        self.expect(&Token::RBracket);
        self.expect(&Token::LParen);
        let ptr = self.parse_expr();
        let offset = if has_offset {
            self.expect(&Token::Comma);
            Some(Box::new(self.parse_expr()))
        } else {
            None
        };
        self.expect(&Token::RParen);
        Expr::Load {
            aligned,
            ty,
            ptr: Box::new(ptr),
            offset,
        }
    }

    fn try_parse_struct_lit(&mut self) -> Option<Expr> {
        let saved = self.pos;

        let name = self.expect_ident();
        self.pos += 1; // consume [

        // Check that we start with something valid for a width expression
        match self.peek() {
            Token::IntLit(_) | Token::Underscore | Token::Ident(_) | Token::LParen => {}
            _ => {
                self.pos = saved;
                return None;
            }
        }

        let width = self.parse_width();

        if *self.peek() != Token::RBracket {
            self.pos = saved;
            return None;
        }
        self.pos += 1; // consume ]

        if *self.peek() != Token::LBrace {
            self.pos = saved;
            return None;
        }
        self.pos += 1; // consume {

        let mut fields = Vec::new();
        while *self.peek() != Token::RBrace {
            let field_name = self.expect_ident();
            self.expect(&Token::Colon);
            let value = self.parse_expr();
            fields.push((field_name, value));
            if *self.peek() == Token::Comma {
                self.pos += 1;
            }
        }
        self.pos += 1; // consume }

        Some(Expr::StructLit {
            name,
            width,
            fields,
        })
    }
}

// --- Convenience constructors for tests ---

pub fn parse(input: &str) -> Vec<Item> {
    let tokens = Lexer::new(input).tokenize();
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

pub fn parse_expr_str(input: &str) -> Expr {
    let tokens = Lexer::new(input).tokenize();
    let mut parser = Parser::new(tokens);
    parser.parse_expr()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helpers ---

    fn ident(s: &str) -> Expr {
        Expr::Ident(s.into())
    }

    fn int(n: i64) -> Expr {
        Expr::IntLit(n)
    }

    fn float(f: f64) -> Expr {
        Expr::FloatLit(f)
    }

    fn binop(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::BinOp {
            op,
            lhs: Box::new(l),
            rhs: Box::new(r),
        }
    }

    fn unary(op: UnaryOp, e: Expr) -> Expr {
        Expr::UnaryOp {
            op,
            operand: Box::new(e),
        }
    }

    fn field(base: Expr, name: &str) -> Expr {
        Expr::Field {
            base: Box::new(base),
            field: name.into(),
        }
    }

    fn call(func: Expr, args: Vec<CallArg>) -> Expr {
        Expr::Call {
            func: Box::new(func),
            args,
        }
    }

    fn pos_arg(e: Expr) -> CallArg {
        CallArg {
            name: None,
            value: e,
        }
    }

    fn named_arg(name: &str, e: Expr) -> CallArg {
        CallArg {
            name: Some(name.into()),
            value: e,
        }
    }

    fn ty(name: &str, width: Option<Width>) -> Type {
        Type {
            name: name.into(),
            width,
        }
    }

    fn cparam(name: &str) -> ComptimeParam {
        ComptimeParam { name: name.into() }
    }

    // ============================================================
    // Expression tests
    // ============================================================

    #[test]
    fn test_literal_int() {
        assert_eq!(parse_expr_str("42"), int(42));
    }

    #[test]
    fn test_literal_float() {
        assert_eq!(parse_expr_str("3.14"), float(3.14));
    }

    #[test]
    fn test_literal_bool() {
        assert_eq!(parse_expr_str("true"), Expr::BoolLit(true));
        assert_eq!(parse_expr_str("false"), Expr::BoolLit(false));
    }

    #[test]
    fn test_literal_char() {
        assert_eq!(parse_expr_str("'a'"), Expr::CharLit('a'));
    }

    #[test]
    fn test_ident() {
        assert_eq!(parse_expr_str("foo"), ident("foo"));
    }

    // --- Binary ops ---

    #[test]
    fn test_add() {
        assert_eq!(
            parse_expr_str("a + b"),
            binop(BinOp::Add, ident("a"), ident("b"))
        );
    }

    #[test]
    fn test_sub() {
        assert_eq!(
            parse_expr_str("a - b"),
            binop(BinOp::Sub, ident("a"), ident("b"))
        );
    }

    #[test]
    fn test_mul() {
        assert_eq!(
            parse_expr_str("a * b"),
            binop(BinOp::Mul, ident("a"), ident("b"))
        );
    }

    #[test]
    fn test_div() {
        assert_eq!(
            parse_expr_str("a / b"),
            binop(BinOp::Div, ident("a"), ident("b"))
        );
    }

    // --- Precedence ---

    #[test]
    fn test_mul_before_add() {
        assert_eq!(
            parse_expr_str("a + b * c"),
            binop(
                BinOp::Add,
                ident("a"),
                binop(BinOp::Mul, ident("b"), ident("c"))
            )
        );
    }

    #[test]
    fn test_parens_override_precedence() {
        assert_eq!(
            parse_expr_str("(a + b) * c"),
            binop(
                BinOp::Mul,
                binop(BinOp::Add, ident("a"), ident("b")),
                ident("c")
            )
        );
    }

    #[test]
    fn test_left_associative_add() {
        assert_eq!(
            parse_expr_str("a + b + c"),
            binop(
                BinOp::Add,
                binop(BinOp::Add, ident("a"), ident("b")),
                ident("c")
            )
        );
    }

    #[test]
    fn test_comparison_lower_than_add() {
        assert_eq!(
            parse_expr_str("a + b > c"),
            binop(
                BinOp::Gt,
                binop(BinOp::Add, ident("a"), ident("b")),
                ident("c")
            )
        );
    }

    #[test]
    fn test_and_lower_than_comparison() {
        assert_eq!(
            parse_expr_str("a > b & c > d"),
            binop(
                BinOp::And,
                binop(BinOp::Gt, ident("a"), ident("b")),
                binop(BinOp::Gt, ident("c"), ident("d"))
            )
        );
    }

    #[test]
    fn test_or_lower_than_and() {
        assert_eq!(
            parse_expr_str("a & b | c & d"),
            binop(
                BinOp::Or,
                binop(BinOp::And, ident("a"), ident("b")),
                binop(BinOp::And, ident("c"), ident("d"))
            )
        );
    }

    #[test]
    fn test_all_comparison_ops() {
        assert_eq!(parse_expr_str("a > b"), binop(BinOp::Gt, ident("a"), ident("b")));
        assert_eq!(parse_expr_str("a < b"), binop(BinOp::Lt, ident("a"), ident("b")));
        assert_eq!(parse_expr_str("a >= b"), binop(BinOp::GtEq, ident("a"), ident("b")));
        assert_eq!(parse_expr_str("a <= b"), binop(BinOp::LtEq, ident("a"), ident("b")));
        assert_eq!(parse_expr_str("a == b"), binop(BinOp::EqEq, ident("a"), ident("b")));
        assert_eq!(parse_expr_str("a != b"), binop(BinOp::NotEq, ident("a"), ident("b")));
    }

    // --- Unary ops ---

    #[test]
    fn test_unary_neg() {
        assert_eq!(parse_expr_str("-a"), unary(UnaryOp::Neg, ident("a")));
    }

    #[test]
    fn test_unary_not() {
        assert_eq!(parse_expr_str("~mask"), unary(UnaryOp::Not, ident("mask")));
    }

    #[test]
    fn test_unary_neg_float() {
        assert_eq!(parse_expr_str("-500.0"), unary(UnaryOp::Neg, float(500.0)));
    }

    #[test]
    fn test_double_neg() {
        // --a is a comment, must use parens
        assert_eq!(
            parse_expr_str("-(-a)"),
            unary(UnaryOp::Neg, unary(UnaryOp::Neg, ident("a")))
        );
    }

    // --- Field access ---

    #[test]
    fn test_field_access() {
        assert_eq!(parse_expr_str("p.pos_x"), field(ident("p"), "pos_x"));
    }

    #[test]
    fn test_chained_field_access() {
        assert_eq!(parse_expr_str("a.b.c"), field(field(ident("a"), "b"), "c"));
    }

    #[test]
    fn test_field_in_binop() {
        assert_eq!(
            parse_expr_str("p.vel_x * drag"),
            binop(BinOp::Mul, field(ident("p"), "vel_x"), ident("drag"))
        );
    }

    // --- Function calls ---

    #[test]
    fn test_simple_call() {
        assert_eq!(
            parse_expr_str("sqrt(x)"),
            call(ident("sqrt"), vec![pos_arg(ident("x"))])
        );
    }

    #[test]
    fn test_call_multiple_args() {
        assert_eq!(
            parse_expr_str("zip(a, b)"),
            call(ident("zip"), vec![pos_arg(ident("a")), pos_arg(ident("b"))])
        );
    }

    #[test]
    fn test_call_named_arg() {
        assert_eq!(
            parse_expr_str("broadcast(v, to=16)"),
            call(ident("broadcast"), vec![pos_arg(ident("v")), named_arg("to", int(16))])
        );
    }

    #[test]
    fn test_call_named_arg_lane() {
        assert_eq!(
            parse_expr_str("broadcast(v, lane=0)"),
            call(ident("broadcast"), vec![pos_arg(ident("v")), named_arg("lane", int(0))])
        );
    }

    #[test]
    fn test_call_no_args() {
        assert_eq!(parse_expr_str("foo()"), call(ident("foo"), vec![]));
    }

    // --- Masked expressions ---

    #[test]
    fn test_masked_without_fallback() {
        assert_eq!(
            parse_expr_str("[mask] a + b"),
            Expr::Masked {
                mask: Box::new(ident("mask")),
                body: Box::new(binop(BinOp::Add, ident("a"), ident("b"))),
                fallback: None,
            }
        );
    }

    #[test]
    fn test_masked_with_fallback() {
        assert_eq!(
            parse_expr_str("[mask] a + b : c"),
            Expr::Masked {
                mask: Box::new(ident("mask")),
                body: Box::new(binop(BinOp::Add, ident("a"), ident("b"))),
                fallback: Some(Box::new(ident("c"))),
            }
        );
    }

    #[test]
    fn test_masked_complex_body() {
        assert_eq!(
            parse_expr_str("[alive] p.vel + gravity * dt : p.vel"),
            Expr::Masked {
                mask: Box::new(ident("alive")),
                body: Box::new(binop(
                    BinOp::Add,
                    field(ident("p"), "vel"),
                    binop(BinOp::Mul, ident("gravity"), ident("dt"))
                )),
                fallback: Some(Box::new(field(ident("p"), "vel"))),
            }
        );
    }

    #[test]
    fn test_masked_comparison_as_mask() {
        assert_eq!(
            parse_expr_str("[a > 0.0] x : y"),
            Expr::Masked {
                mask: Box::new(binop(BinOp::Gt, ident("a"), float(0.0))),
                body: Box::new(ident("x")),
                fallback: Some(Box::new(ident("y"))),
            }
        );
    }

    // --- Reductions ---

    #[test]
    fn test_reduction_add() {
        assert_eq!(
            parse_expr_str("+/ v"),
            Expr::Reduction { op: ReductionOp::Add, operand: Box::new(ident("v")) }
        );
    }

    #[test]
    fn test_reduction_mul() {
        assert_eq!(
            parse_expr_str("*/ v"),
            Expr::Reduction { op: ReductionOp::Mul, operand: Box::new(ident("v")) }
        );
    }

    #[test]
    fn test_reduction_or() {
        assert_eq!(
            parse_expr_str("|/ mask"),
            Expr::Reduction { op: ReductionOp::Or, operand: Box::new(ident("mask")) }
        );
    }

    #[test]
    fn test_reduction_and() {
        assert_eq!(
            parse_expr_str("&/ mask"),
            Expr::Reduction { op: ReductionOp::And, operand: Box::new(ident("mask")) }
        );
    }

    #[test]
    fn test_reduction_max() {
        assert_eq!(
            parse_expr_str("max/ v"),
            Expr::Reduction { op: ReductionOp::Max, operand: Box::new(ident("v")) }
        );
    }

    #[test]
    fn test_reduction_min() {
        assert_eq!(
            parse_expr_str("min/ v"),
            Expr::Reduction { op: ReductionOp::Min, operand: Box::new(ident("v")) }
        );
    }

    #[test]
    fn test_dot_product() {
        assert_eq!(
            parse_expr_str("+/ (a * b)"),
            Expr::Reduction {
                op: ReductionOp::Add,
                operand: Box::new(binop(BinOp::Mul, ident("a"), ident("b"))),
            }
        );
    }

    // --- Scans ---

    #[test]
    fn test_scan_add() {
        assert_eq!(
            parse_expr_str("scan.add(v)"),
            Expr::Scan { op: ScanOp::Add, operand: Box::new(ident("v")), seed: None }
        );
    }

    #[test]
    fn test_scan_xor() {
        assert_eq!(
            parse_expr_str("scan.xor(v)"),
            Expr::Scan { op: ScanOp::Xor, operand: Box::new(ident("v")), seed: None }
        );
    }

    #[test]
    fn test_scan_max() {
        assert_eq!(
            parse_expr_str("scan.max(v)"),
            Expr::Scan { op: ScanOp::Max, operand: Box::new(ident("v")), seed: None }
        );
    }

    #[test]
    fn test_scan_preceding_any() {
        assert_eq!(
            parse_expr_str("scan.preceding_any(v)"),
            Expr::Scan { op: ScanOp::PrecedingAny, operand: Box::new(ident("v")), seed: None }
        );
    }

    #[test]
    fn test_not_scan() {
        assert_eq!(
            parse_expr_str("~scan.preceding_any(is_escape)"),
            unary(
                UnaryOp::Not,
                Expr::Scan { op: ScanOp::PrecedingAny, operand: Box::new(ident("is_escape")), seed: None }
            )
        );
    }

    // --- Gather ---

    #[test]
    fn test_gather() {
        assert_eq!(
            parse_expr_str("src.[indices]"),
            Expr::Gather {
                base: Box::new(ident("src")),
                index: Box::new(ident("indices")),
                mask: None,
            }
        );
    }

    #[test]
    fn test_gather_masked() {
        assert_eq!(
            parse_expr_str("src.[indices, mask]"),
            Expr::Gather {
                base: Box::new(ident("src")),
                index: Box::new(ident("indices")),
                mask: Some(Box::new(ident("mask"))),
            }
        );
    }

    // --- Load ---

    #[test]
    fn test_load_aligned() {
        assert_eq!(
            parse_expr_str("load[f32[8]](ptr)"),
            Expr::Load {
                aligned: true,
                ty: ty("f32", Some(Width::Fixed(8))),
                ptr: Box::new(ident("ptr")),
                offset: None,
            }
        );
    }

    #[test]
    fn test_load_unaligned() {
        assert_eq!(
            parse_expr_str("loadu[f32[8]](ptr)"),
            Expr::Load {
                aligned: false,
                ty: ty("f32", Some(Width::Fixed(8))),
                ptr: Box::new(ident("ptr")),
                offset: None,
            }
        );
    }

    #[test]
    fn test_load_with_param_width() {
        assert_eq!(
            parse_expr_str("load[f32[N]](ptr)"),
            Expr::Load {
                aligned: true,
                ty: ty("f32", Some(Width::Param("N".into()))),
                ptr: Box::new(ident("ptr")),
                offset: None,
            }
        );
    }

    // --- Struct literal ---

    #[test]
    fn test_struct_literal() {
        assert_eq!(
            parse_expr_str("Particle[1024] { pos_x: x, pos_y: y }"),
            Expr::StructLit {
                name: "Particle".into(),
                width: Width::Fixed(1024),
                fields: vec![
                    ("pos_x".into(), ident("x")),
                    ("pos_y".into(), ident("y")),
                ],
            }
        );
    }

    #[test]
    fn test_struct_literal_native_width() {
        assert_eq!(
            parse_expr_str("Vec[_] { x: a, y: b }"),
            Expr::StructLit {
                name: "Vec".into(),
                width: Width::Native,
                fields: vec![("x".into(), ident("a")), ("y".into(), ident("b"))],
            }
        );
    }

    #[test]
    fn test_struct_literal_param_width() {
        assert_eq!(
            parse_expr_str("Particle[N] { pos_x: x }"),
            Expr::StructLit {
                name: "Particle".into(),
                width: Width::Param("N".into()),
                fields: vec![("pos_x".into(), ident("x"))],
            }
        );
    }

    // --- Complex expressions from spec ---

    #[test]
    fn test_fma_pattern() {
        assert_eq!(
            parse_expr_str("a * b + c"),
            binop(BinOp::Add, binop(BinOp::Mul, ident("a"), ident("b")), ident("c"))
        );
    }

    #[test]
    fn test_complex_masked() {
        assert_eq!(
            parse_expr_str("[too_fast] (new_vel_x / speed) * 100.0 : new_vel_x"),
            Expr::Masked {
                mask: Box::new(ident("too_fast")),
                body: Box::new(binop(
                    BinOp::Mul,
                    binop(BinOp::Div, ident("new_vel_x"), ident("speed")),
                    float(100.0)
                )),
                fallback: Some(Box::new(ident("new_vel_x"))),
            }
        );
    }

    #[test]
    fn test_multi_and_chain() {
        assert_eq!(
            parse_expr_str("a > 0.0 & b < 1.0 & c > 0.0"),
            binop(
                BinOp::And,
                binop(
                    BinOp::And,
                    binop(BinOp::Gt, ident("a"), float(0.0)),
                    binop(BinOp::Lt, ident("b"), float(1.0))
                ),
                binop(BinOp::Gt, ident("c"), float(0.0))
            )
        );
    }

    #[test]
    fn test_multi_or_chain() {
        assert_eq!(
            parse_expr_str("a | b | c | d"),
            binop(
                BinOp::Or,
                binop(BinOp::Or, binop(BinOp::Or, ident("a"), ident("b")), ident("c")),
                ident("d")
            )
        );
    }

    #[test]
    fn test_json_classify_expression() {
        assert_eq!(
            parse_expr_str("chunk == '{' | chunk == '}'"),
            binop(
                BinOp::Or,
                binop(BinOp::EqEq, ident("chunk"), Expr::CharLit('{')),
                binop(BinOp::EqEq, ident("chunk"), Expr::CharLit('}')),
            )
        );
    }

    // ============================================================
    // Statement tests
    // ============================================================

    fn parse_stmts(input: &str) -> Vec<Stmt> {
        let wrapped = format!("fn __test__() {{ {} }}", input);
        let items = parse(&wrapped);
        match &items[0] {
            Item::Fn(f) => f.body.clone(),
            _ => panic!("expected function"),
        }
    }

    fn parse_one_stmt(input: &str) -> Stmt {
        let stmts = parse_stmts(input);
        assert_eq!(stmts.len(), 1, "expected one statement, got {}", stmts.len());
        stmts.into_iter().next().unwrap()
    }

    #[test]
    fn test_simple_assign() {
        assert_eq!(
            parse_one_stmt("x = 42"),
            Stmt { kind: StmtKind::Assign { target: AssignTarget::Ident("x".into()), ty: None, value: int(42) }, span: Span::dummy() }
        );
    }

    #[test]
    fn test_assign_expr() {
        assert_eq!(
            parse_one_stmt("alive = p.mass > 0.0"),
            Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Ident("alive".into()),
                    ty: None,
                    value: binop(BinOp::Gt, field(ident("p"), "mass"), float(0.0)),
                },
                span: Span::dummy(),
            }
        );
    }

    #[test]
    fn test_assign_masked() {
        assert_eq!(
            parse_one_stmt("new_vel = [alive] p.vel : p.vel"),
            Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Ident("new_vel".into()),
                    ty: None,
                    value: Expr::Masked {
                        mask: Box::new(ident("alive")),
                        body: Box::new(field(ident("p"), "vel")),
                        fallback: Some(Box::new(field(ident("p"), "vel"))),
                    },
                },
                span: Span::dummy(),
            }
        );
    }

    #[test]
    fn test_return_stmt() {
        assert_eq!(parse_one_stmt("return x"), Stmt { kind: StmtKind::Return(ident("x")), span: Span::dummy() });
    }

    #[test]
    fn test_return_struct_lit() {
        assert_eq!(
            parse_one_stmt("return Particle[1024] { pos_x: x }"),
            Stmt {
                kind: StmtKind::Return(Expr::StructLit {
                    name: "Particle".into(),
                    width: Width::Fixed(1024),
                    fields: vec![("pos_x".into(), ident("x"))],
                }),
                span: Span::dummy(),
            },
        );
    }

    #[test]
    fn test_expr_stmt() {
        assert_eq!(
            parse_one_stmt("store(ptr, v)"),
            Stmt { kind: StmtKind::Expr(call(ident("store"), vec![pos_arg(ident("ptr")), pos_arg(ident("v"))])), span: Span::dummy() }
        );
    }

    #[test]
    fn test_scatter_assign() {
        assert_eq!(
            parse_one_stmt("dst.[indices] = src"),
            Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Scatter {
                        base: Box::new(ident("dst")),
                        index: Box::new(ident("indices")),
                        mask: None,
                    },
                    ty: None,
                    value: ident("src"),
                },
                span: Span::dummy(),
            }
        );
    }

    #[test]
    fn test_masked_scatter_assign() {
        assert_eq!(
            parse_one_stmt("dst.[indices, mask] = src"),
            Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Scatter {
                        base: Box::new(ident("dst")),
                        index: Box::new(ident("indices")),
                        mask: Some(Box::new(ident("mask"))),
                    },
                    ty: None,
                    value: ident("src"),
                },
                span: Span::dummy(),
            }
        );
    }

    #[test]
    fn test_destructure_assign() {
        assert_eq!(
            parse_one_stmt("(evens, odds) = unzip(v)"),
            Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Destructure(vec!["evens".into(), "odds".into()]),
                    ty: None,
                    value: call(ident("unzip"), vec![pos_arg(ident("v"))]),
                },
                span: Span::dummy(),
            }
        );
    }

    // ============================================================
    // Struct definition tests
    // ============================================================

    #[test]
    fn test_struct_no_comptime_params() {
        let items = parse("struct Config { threshold: f32[8] }");
        assert_eq!(
            items,
            vec![Item::Struct(StructDef {
                name: "Config".into(),
                comptime_params: vec![],
                fields: vec![FieldDef {
                    name: "threshold".into(),
                    ty: ty("f32", Some(Width::Fixed(8))),
                }],
            })]
        );
    }

    #[test]
    fn test_struct_one_comptime_param() {
        let items = parse("struct Particle[N] { pos_x: f32[N], pos_y: f32[N] }");
        assert_eq!(
            items,
            vec![Item::Struct(StructDef {
                name: "Particle".into(),
                comptime_params: vec![cparam("N")],
                fields: vec![
                    FieldDef { name: "pos_x".into(), ty: ty("f32", Some(Width::Param("N".into()))) },
                    FieldDef { name: "pos_y".into(), ty: ty("f32", Some(Width::Param("N".into()))) },
                ],
            })]
        );
    }

    #[test]
    fn test_struct_comptime_with_annotation() {
        let items = parse("struct Particle[N: comptime] { pos_x: f32[N] }");
        assert_eq!(
            items,
            vec![Item::Struct(StructDef {
                name: "Particle".into(),
                comptime_params: vec![cparam("N")],
                fields: vec![
                    FieldDef { name: "pos_x".into(), ty: ty("f32", Some(Width::Param("N".into()))) },
                ],
            })]
        );
    }

    #[test]
    fn test_struct_multiple_comptime_params() {
        let items = parse("struct Grid[W, H] { data: f32[W] }");
        assert_eq!(
            items,
            vec![Item::Struct(StructDef {
                name: "Grid".into(),
                comptime_params: vec![cparam("W"), cparam("H")],
                fields: vec![
                    FieldDef { name: "data".into(), ty: ty("f32", Some(Width::Param("W".into()))) },
                ],
            })]
        );
    }

    // ============================================================
    // Function definition tests
    // ============================================================

    #[test]
    fn test_fn_def_no_comptime() {
        let items = parse("fn noop() {}");
        assert_eq!(
            items,
            vec![Item::Fn(FnDef {
                name: "noop".into(),
                comptime_params: vec![],
                params: vec![],
                ret_ty: None,
                body: vec![],
            })]
        );
    }

    #[test]
    fn test_fn_def_with_params_and_return() {
        let items = parse("fn add(a: f32[8], b: f32[8]) -> f32[8] { return a + b }");
        assert_eq!(
            items,
            vec![Item::Fn(FnDef {
                name: "add".into(),
                comptime_params: vec![],
                params: vec![
                    Param { name: "a".into(), ty: ty("f32", Some(Width::Fixed(8))) },
                    Param { name: "b".into(), ty: ty("f32", Some(Width::Fixed(8))) },
                ],
                ret_ty: Some(ty("f32", Some(Width::Fixed(8)))),
                body: vec![Stmt { kind: StmtKind::Return(binop(BinOp::Add, ident("a"), ident("b"))), span: Span::dummy() }],
            })]
        );
    }

    #[test]
    fn test_fn_native_width_params() {
        let items = parse("fn scale(v: f32[_], s: f32[_]) -> f32[_] { return v * s }");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(f.params[0].ty, ty("f32", Some(Width::Native)));
            assert_eq!(f.params[1].ty, ty("f32", Some(Width::Native)));
            assert_eq!(f.ret_ty, Some(ty("f32", Some(Width::Native))));
        } else {
            panic!("expected function");
        }
    }

    #[test]
    fn test_fn_comptime_param() {
        let items = parse("fn dot[N](a: f32[N], b: f32[N]) -> f32[1] { return +/ (a * b) }");
        assert_eq!(
            items,
            vec![Item::Fn(FnDef {
                name: "dot".into(),
                comptime_params: vec![cparam("N")],
                params: vec![
                    Param { name: "a".into(), ty: ty("f32", Some(Width::Param("N".into()))) },
                    Param { name: "b".into(), ty: ty("f32", Some(Width::Param("N".into()))) },
                ],
                ret_ty: Some(ty("f32", Some(Width::Fixed(1)))),
                body: vec![Stmt { kind: StmtKind::Return(Expr::Reduction {
                    op: ReductionOp::Add,
                    operand: Box::new(binop(BinOp::Mul, ident("a"), ident("b"))),
                }), span: Span::dummy() }],
            })]
        );
    }

    #[test]
    fn test_fn_comptime_with_annotation() {
        let items = parse("fn foo[N: comptime](v: f32[N]) -> f32[N] { return v }");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(f.comptime_params, vec![cparam("N")]);
        } else {
            panic!("expected function");
        }
    }

    #[test]
    fn test_fn_multiple_comptime_params() {
        let items = parse("fn cast[M, N](v: f32[M]) -> f32[N] { return widen(v) }");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(f.comptime_params, vec![cparam("M"), cparam("N")]);
            assert_eq!(f.params[0].ty, ty("f32", Some(Width::Param("M".into()))));
            assert_eq!(f.ret_ty, Some(ty("f32", Some(Width::Param("N".into())))));
        } else {
            panic!("expected function");
        }
    }

    // ============================================================
    // Full program tests (updated for comptime params)
    // ============================================================

    #[test]
    fn test_particle_update_kernel() {
        let input = r#"
            struct Particle[N] {
                pos_x: f32[N],
                pos_y: f32[N],
                vel_x: f32[N],
                vel_y: f32[N],
                mass:  f32[N],
            }

            fn update[N](p: Particle[N], dt: f32[N], gravity: f32[N]) -> Particle[N] {
                alive = p.mass > 0.0
                drag  = 1.0 - (0.01 * dt)

                new_vel_x = [alive] p.vel_x * drag : p.vel_x
                new_vel_y = [alive] (p.vel_y + gravity * dt) * drag : p.vel_y

                speed_sq  = new_vel_x * new_vel_x + new_vel_y * new_vel_y
                too_fast  = speed_sq > 10000.0
                speed     = sqrt(speed_sq)
                new_vel_x = [too_fast] (new_vel_x / speed) * 100.0 : new_vel_x
                new_vel_y = [too_fast] (new_vel_y / speed) * 100.0 : new_vel_y

                new_pos_x = p.pos_x + new_vel_x * dt
                new_pos_y = p.pos_y + new_vel_y * dt

                in_bounds = new_pos_x > -500.0 & new_pos_x < 500.0
                          & new_pos_y > -500.0 & new_pos_y < 500.0
                new_mass  = [in_bounds] p.mass : 0.0

                return Particle[N] {
                    pos_x: new_pos_x,
                    pos_y: new_pos_y,
                    vel_x: new_vel_x,
                    vel_y: new_vel_y,
                    mass:  new_mass,
                }
            }
        "#;
        let items = parse(input);
        assert_eq!(items.len(), 2);

        // Struct
        if let Item::Struct(s) = &items[0] {
            assert_eq!(s.name, "Particle");
            assert_eq!(s.comptime_params, vec![cparam("N")]);
            assert_eq!(s.fields.len(), 5);
            assert_eq!(s.fields[0].ty, ty("f32", Some(Width::Param("N".into()))));
        } else {
            panic!("expected struct");
        }

        // Function
        if let Item::Fn(f) = &items[1] {
            assert_eq!(f.name, "update");
            assert_eq!(f.comptime_params, vec![cparam("N")]);
            assert_eq!(f.params.len(), 3);
            assert_eq!(f.params[0].ty, ty("Particle", Some(Width::Param("N".into()))));
            assert_eq!(f.body.len(), 14);

            // Return has struct literal with Param width
            if let Stmt { kind: StmtKind::Return(Expr::StructLit { name, width, fields }), .. } = &f.body[13] {
                assert_eq!(name, "Particle");
                assert_eq!(width, &Width::Param("N".into()));
                assert_eq!(fields.len(), 5);
            } else {
                panic!("expected return struct literal");
            }
        } else {
            panic!("expected function");
        }
    }

    #[test]
    fn test_json_classify_kernel() {
        let input = r#"
            struct Masks[N] {
                is_quote:      bool[N],
                is_escape:     bool[N],
                is_structural: bool[N],
                is_whitespace: bool[N],
            }

            fn classify[N](chunk: u8[N]) -> Masks[N] {
                is_quote      = chunk == '"'
                is_escape     = chunk == '\\'
                is_whitespace = chunk == ' ' | chunk == '\t'
                              | chunk == '\n' | chunk == '\r'
                is_structural = chunk == '{' | chunk == '}'
                              | chunk == '[' | chunk == ']'
                              | chunk == ':' | chunk == ','

                return Masks[N] {
                    is_quote:      is_quote,
                    is_escape:     is_escape,
                    is_structural: is_structural,
                    is_whitespace: is_whitespace,
                }
            }

            fn string_mask[N](is_quote: bool[N], is_escape: bool[N]) -> bool[N] {
                real_quote = is_quote & ~scan.preceding_any(is_escape)
                return scan.xor(real_quote)
            }
        "#;
        let items = parse(input);
        assert_eq!(items.len(), 3);

        // Struct with comptime param
        if let Item::Struct(s) = &items[0] {
            assert_eq!(s.name, "Masks");
            assert_eq!(s.comptime_params, vec![cparam("N")]);
        } else {
            panic!("expected struct");
        }

        // classify function
        if let Item::Fn(f) = &items[1] {
            assert_eq!(f.name, "classify");
            assert_eq!(f.comptime_params, vec![cparam("N")]);
        } else {
            panic!("expected function");
        }

        // string_mask function
        if let Item::Fn(f) = &items[2] {
            assert_eq!(f.name, "string_mask");
            assert_eq!(f.comptime_params, vec![cparam("N")]);
            assert_eq!(f.body.len(), 2);
            assert!(matches!(&f.body[0], Stmt { kind: StmtKind::Assign { target: AssignTarget::Ident(n), .. }, .. } if n == "real_quote"));
            assert!(matches!(&f.body[1], Stmt { kind: StmtKind::Return(Expr::Scan { op: ScanOp::Xor, .. }), .. }));
        } else {
            panic!("expected function");
        }
    }

    // ============================================================
    // Edge cases
    // ============================================================

    #[test]
    fn test_nested_parens() {
        assert_eq!(parse_expr_str("((a))"), ident("a"));
    }

    #[test]
    fn test_unary_in_binop() {
        assert_eq!(
            parse_expr_str("-a + b"),
            binop(BinOp::Add, unary(UnaryOp::Neg, ident("a")), ident("b"))
        );
    }

    #[test]
    fn test_reduction_of_parenthesized() {
        assert_eq!(
            parse_expr_str("+/ (x)"),
            Expr::Reduction { op: ReductionOp::Add, operand: Box::new(ident("x")) }
        );
    }

    #[test]
    fn test_multiple_stmts_in_fn() {
        let items = parse("fn test() { x = 1 y = 2 return x + y }");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(f.body.len(), 3);
        }
    }

    #[test]
    #[should_panic(expected = "expected expression")]
    fn test_empty_parens_is_error() {
        parse_expr_str("()");
    }

    #[test]
    #[should_panic(expected = "unknown scan operation")]
    fn test_unknown_scan_op() {
        parse_expr_str("scan.unknown(v)");
    }

    #[test]
    fn test_call_on_field() {
        assert_eq!(
            parse_expr_str("foo.bar(x)"),
            call(field(ident("foo"), "bar"), vec![pos_arg(ident("x"))])
        );
    }

    #[test]
    fn test_store_with_mask() {
        assert_eq!(
            parse_one_stmt("store(ptr, v, mask)"),
            Stmt { kind: StmtKind::Expr(call(
                ident("store"),
                vec![pos_arg(ident("ptr")), pos_arg(ident("v")), pos_arg(ident("mask"))]
            )), span: Span::dummy() }
        );
    }

    #[test]
    fn test_speed_clamp_pattern() {
        assert_eq!(
            parse_one_stmt("speed_sq = new_vel_x * new_vel_x + new_vel_y * new_vel_y"),
            Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Ident("speed_sq".into()),
                    ty: None,
                    value: binop(
                        BinOp::Add,
                        binop(BinOp::Mul, ident("new_vel_x"), ident("new_vel_x")),
                        binop(BinOp::Mul, ident("new_vel_y"), ident("new_vel_y")),
                    ),
                },
                span: Span::dummy(),
            }
        );
    }

    #[test]
    fn test_struct_literal_with_expr_fields() {
        assert_eq!(
            parse_expr_str("V[8] { x: a + b, y: c * d }"),
            Expr::StructLit {
                name: "V".into(),
                width: Width::Fixed(8),
                fields: vec![
                    ("x".into(), binop(BinOp::Add, ident("a"), ident("b"))),
                    ("y".into(), binop(BinOp::Mul, ident("c"), ident("d"))),
                ],
            }
        );
    }

    #[test]
    fn test_ident_not_consumed_when_no_struct_lit() {
        assert_eq!(parse_expr_str("foo"), ident("foo"));
    }

    #[test]
    fn test_gather_in_expr() {
        assert_eq!(
            parse_expr_str("src.[idx] + 1"),
            binop(
                BinOp::Add,
                Expr::Gather {
                    base: Box::new(ident("src")),
                    index: Box::new(ident("idx")),
                    mask: None,
                },
                int(1),
            )
        );
    }

    // ============================================================
    // Width::Param in various positions
    // ============================================================

    #[test]
    fn test_type_with_param_width() {
        let items = parse("fn foo(v: f32[N]) {}");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(f.params[0].ty, ty("f32", Some(Width::Param("N".into()))));
        }
    }

    #[test]
    fn test_return_type_with_param_width() {
        let items = parse("fn foo() -> f32[N] { return x }");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(f.ret_ty, Some(ty("f32", Some(Width::Param("N".into())))));
        }
    }

    #[test]
    fn test_struct_fields_with_param_width() {
        let items = parse("struct V[N] { x: f32[N], y: i32[N] }");
        if let Item::Struct(s) = &items[0] {
            assert_eq!(s.comptime_params, vec![cparam("N")]);
            assert_eq!(s.fields[0].ty, ty("f32", Some(Width::Param("N".into()))));
            assert_eq!(s.fields[1].ty, ty("i32", Some(Width::Param("N".into()))));
        }
    }

    #[test]
    fn test_return_struct_lit_with_param() {
        let items = parse("fn foo[N]() -> V[N] { return V[N] { x: a } }");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(f.comptime_params, vec![cparam("N")]);
            if let Stmt { kind: StmtKind::Return(Expr::StructLit { name, width, .. }), .. } = &f.body[0] {
                assert_eq!(name, "V");
                assert_eq!(width, &Width::Param("N".into()));
            } else {
                panic!("expected return struct lit");
            }
        }
    }

    // ============================================================
    // Comptime width expressions
    // ============================================================

    fn width_mul(l: Width, r: Width) -> Width {
        Width::BinOp { op: ComptimeOp::Mul, lhs: Box::new(l), rhs: Box::new(r) }
    }

    fn width_add(l: Width, r: Width) -> Width {
        Width::BinOp { op: ComptimeOp::Add, lhs: Box::new(l), rhs: Box::new(r) }
    }

    fn width_sub(l: Width, r: Width) -> Width {
        Width::BinOp { op: ComptimeOp::Sub, lhs: Box::new(l), rhs: Box::new(r) }
    }

    fn width_div(l: Width, r: Width) -> Width {
        Width::BinOp { op: ComptimeOp::Div, lhs: Box::new(l), rhs: Box::new(r) }
    }

    #[test]
    fn test_width_expr_mul() {
        // f32[N * 2]
        let items = parse("fn foo(v: f32[N * 2]) {}");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.params[0].ty,
                ty("f32", Some(width_mul(Width::Param("N".into()), Width::Fixed(2))))
            );
        }
    }

    #[test]
    fn test_width_expr_add() {
        // f32[N + 4]
        let items = parse("fn foo(v: f32[N + 4]) {}");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.params[0].ty,
                ty("f32", Some(width_add(Width::Param("N".into()), Width::Fixed(4))))
            );
        }
    }

    #[test]
    fn test_width_expr_sub() {
        // f32[N - 1]
        let items = parse("fn foo(v: f32[N - 1]) {}");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.params[0].ty,
                ty("f32", Some(width_sub(Width::Param("N".into()), Width::Fixed(1))))
            );
        }
    }

    #[test]
    fn test_width_expr_div() {
        // f32[N / 2]
        let items = parse("fn foo(v: f32[N / 2]) {}");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.params[0].ty,
                ty("f32", Some(width_div(Width::Param("N".into()), Width::Fixed(2))))
            );
        }
    }

    #[test]
    fn test_width_expr_precedence() {
        // f32[N + M * 2] → N + (M * 2)
        let items = parse("fn foo(v: f32[N + M * 2]) {}");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.params[0].ty,
                ty("f32", Some(width_add(
                    Width::Param("N".into()),
                    width_mul(Width::Param("M".into()), Width::Fixed(2))
                )))
            );
        }
    }

    #[test]
    fn test_width_expr_parens() {
        // f32[(N + M) * 2]
        let items = parse("fn foo(v: f32[(N + M) * 2]) {}");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.params[0].ty,
                ty("f32", Some(width_mul(
                    width_add(Width::Param("N".into()), Width::Param("M".into())),
                    Width::Fixed(2)
                )))
            );
        }
    }

    #[test]
    fn test_width_expr_native_mul() {
        // f32[_ * 2]
        let items = parse("fn foo(v: f32[_ * 2]) {}");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.params[0].ty,
                ty("f32", Some(width_mul(Width::Native, Width::Fixed(2))))
            );
        }
    }

    #[test]
    fn test_width_expr_in_return_type() {
        let items = parse("fn interleave[N](a: f32[N], b: f32[N]) -> f32[N * 2] { return zip(a, b) }");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.ret_ty,
                Some(ty("f32", Some(width_mul(Width::Param("N".into()), Width::Fixed(2)))))
            );
        }
    }

    #[test]
    fn test_width_expr_in_struct_field() {
        let items = parse("struct Interleaved[N] { data: f32[N * 2] }");
        if let Item::Struct(s) = &items[0] {
            assert_eq!(
                s.fields[0].ty,
                ty("f32", Some(width_mul(Width::Param("N".into()), Width::Fixed(2))))
            );
        }
    }

    #[test]
    fn test_width_expr_in_struct_literal() {
        assert_eq!(
            parse_expr_str("V[N * 2] { x: a }"),
            Expr::StructLit {
                name: "V".into(),
                width: width_mul(Width::Param("N".into()), Width::Fixed(2)),
                fields: vec![("x".into(), ident("a"))],
            }
        );
    }

    #[test]
    fn test_width_expr_in_load() {
        assert_eq!(
            parse_expr_str("load[f32[N * 2]](ptr)"),
            Expr::Load {
                aligned: true,
                ty: ty("f32", Some(width_mul(Width::Param("N".into()), Width::Fixed(2)))),
                ptr: Box::new(ident("ptr")),
                offset: None,
            }
        );
    }

    #[test]
    fn test_width_expr_fixed_arithmetic() {
        // f32[8 * 2] — purely fixed, but still parsed as BinOp
        let items = parse("fn foo(v: f32[8 * 2]) {}");
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.params[0].ty,
                ty("f32", Some(width_mul(Width::Fixed(8), Width::Fixed(2))))
            );
        }
    }

    #[test]
    fn test_interleave_full_function() {
        let input = r#"
            fn interleave[N](a: f32[N], b: f32[N]) -> f32[N * 2] {
                return zip(a, b)
            }
        "#;
        let items = parse(input);
        if let Item::Fn(f) = &items[0] {
            assert_eq!(f.name, "interleave");
            assert_eq!(f.comptime_params, vec![cparam("N")]);
            assert_eq!(f.params[0].ty, ty("f32", Some(Width::Param("N".into()))));
            assert_eq!(f.params[1].ty, ty("f32", Some(Width::Param("N".into()))));
            assert_eq!(
                f.ret_ty,
                Some(ty("f32", Some(width_mul(Width::Param("N".into()), Width::Fixed(2)))))
            );
        }
    }

    #[test]
    fn test_half_width_function() {
        let input = "fn halve[N](v: f32[N]) -> f32[N / 2] { return narrow(v) }";
        let items = parse(input);
        if let Item::Fn(f) = &items[0] {
            assert_eq!(
                f.ret_ty,
                Some(ty("f32", Some(width_div(Width::Param("N".into()), Width::Fixed(2)))))
            );
        }
    }

    // ============================================================
    // Shift operators
    // ============================================================

    #[test]
    fn test_shift_right() {
        let expr = parse_expr_str("v >> 1");
        assert_eq!(
            expr,
            Expr::BinOp {
                op: BinOp::BitShr,
                lhs: Box::new(ident("v")),
                rhs: Box::new(Expr::IntLit(1)),
            }
        );
    }

    #[test]
    fn test_shift_left() {
        let expr = parse_expr_str("v << 2");
        assert_eq!(
            expr,
            Expr::BinOp {
                op: BinOp::BitShl,
                lhs: Box::new(ident("v")),
                rhs: Box::new(Expr::IntLit(2)),
            }
        );
    }

    #[test]
    fn test_shift_precedence_vs_comparison() {
        // a >> 1 == b should parse as (a >> 1) == b
        let expr = parse_expr_str("a >> 1 == b");
        match expr {
            Expr::BinOp { op: BinOp::EqEq, lhs, rhs } => {
                match *lhs {
                    Expr::BinOp { op: BinOp::BitShr, .. } => {}
                    other => panic!("expected Shr, got {:?}", other),
                }
                assert_eq!(*rhs, ident("b"));
            }
            other => panic!("expected EqEq, got {:?}", other),
        }
    }

    #[test]
    fn test_shift_precedence_vs_add() {
        // a + b >> 1 should parse as (a + b) >> 1
        let expr = parse_expr_str("a + b >> 1");
        match expr {
            Expr::BinOp { op: BinOp::BitShr, lhs, .. } => {
                match *lhs {
                    Expr::BinOp { op: BinOp::Add, .. } => {}
                    other => panic!("expected Add, got {:?}", other),
                }
            }
            other => panic!("expected Shr, got {:?}", other),
        }
    }

    // ============================================================
    // XOR operator
    // ============================================================

    #[test]
    fn test_xor_operator() {
        let expr = parse_expr_str("a ^ b");
        assert_eq!(
            expr,
            Expr::BinOp {
                op: BinOp::Xor,
                lhs: Box::new(ident("a")),
                rhs: Box::new(ident("b")),
            }
        );
    }

    #[test]
    fn test_xor_precedence_between_or_and_and() {
        // a | b ^ c & d should parse as a | (b ^ (c & d))
        let expr = parse_expr_str("a | b ^ c & d");
        match expr {
            Expr::BinOp { op: BinOp::Or, rhs, .. } => {
                match *rhs {
                    Expr::BinOp { op: BinOp::Xor, rhs: inner_rhs, .. } => {
                        match *inner_rhs {
                            Expr::BinOp { op: BinOp::And, .. } => {}
                            other => panic!("expected And, got {:?}", other),
                        }
                    }
                    other => panic!("expected Xor, got {:?}", other),
                }
            }
            other => panic!("expected Or, got {:?}", other),
        }
    }

    // ============================================================
    // Vector repeat syntax
    // ============================================================

    #[test]
    fn test_vec_repeat() {
        let expr = parse_expr_str("[u8: 0; 64]");
        assert_eq!(
            expr,
            Expr::VecLit {
                elem_type: "u8".into(),
                values: vec![0; 64],
            }
        );
    }

    #[test]
    fn test_vec_repeat_nonzero() {
        let expr = parse_expr_str("[i32: 42; 4]");
        assert_eq!(
            expr,
            Expr::VecLit {
                elem_type: "i32".into(),
                values: vec![42, 42, 42, 42],
            }
        );
    }

    // ============================================================
    // Typed assignments
    // ============================================================

    #[test]
    fn test_typed_assign() {
        assert_eq!(
            parse_one_stmt("x: u64[1] = ~0"),
            Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Ident("x".into()),
                    ty: Some(Type {
                        name: "u64".into(),
                        width: Some(Width::Fixed(1)),
                    }),
                    value: Expr::UnaryOp {
                        op: UnaryOp::Not,
                        operand: Box::new(Expr::IntLit(0)),
                    },
                },
                span: Span::dummy(),
            }
        );
    }

    #[test]
    fn test_typed_assign_zero() {
        assert_eq!(
            parse_one_stmt("z: u8[64] = 0"),
            Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Ident("z".into()),
                    ty: Some(Type {
                        name: "u8".into(),
                        width: Some(Width::Fixed(64)),
                    }),
                    value: Expr::IntLit(0),
                },
                span: Span::dummy(),
            }
        );
    }

    // ============================================================
    // Stream parsing
    // ============================================================

    #[test]
    fn test_stream_basic() {
        let input = r#"
            fn process(buf: u8[1]) {
                stream chunk: u8[8] over buf carry (in_str: bool[1] = false) {
                    is_quote = chunk == '"'
                    carry in_str = +/ is_quote
                }
            }
        "#;
        let items = parse(input);
        if let Item::Fn(f) = &items[0] {
            assert_eq!(f.body.len(), 1);
            match &f.body[0] {
                Stmt { kind: StmtKind::Stream {
                    chunk_name,
                    chunk_ty,
                    buffer,
                    carry,
                    body,
                    carry_updates,
                }, .. } => {
                    assert_eq!(chunk_name, "chunk");
                    assert_eq!(chunk_ty.name, "u8");
                    assert_eq!(buffer, "buf");
                    assert_eq!(carry.len(), 1);
                    assert_eq!(carry[0].name, "in_str");
                    assert_eq!(body.len(), 1);
                    assert_eq!(carry_updates.len(), 1);
                    assert_eq!(carry_updates[0].0, "in_str");
                }
                other => panic!("expected Stream, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_stream_no_carry() {
        let input = r#"
            fn process(buf: u8[1]) {
                stream chunk: u8[8] over buf {
                    result = chunk + 1
                }
            }
        "#;
        let items = parse(input);
        if let Item::Fn(f) = &items[0] {
            match &f.body[0] {
                Stmt { kind: StmtKind::Stream { carry, carry_updates, .. }, .. } => {
                    assert_eq!(carry.len(), 0);
                    assert_eq!(carry_updates.len(), 0);
                }
                other => panic!("expected Stream, got {:?}", other),
            }
        }
    }
}
