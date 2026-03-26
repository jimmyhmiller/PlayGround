use crate::ast::*;
use crate::lexer::Token;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    pub fn parse_program(&mut self) -> Vec<Item> {
        let mut items = Vec::new();
        while !self.at_eof() {
            items.push(self.parse_item());
        }
        items
    }

    fn parse_item(&mut self) -> Item {
        match self.peek() {
            Token::Fn => Item::FnDef(self.parse_fn_def()),
            Token::Let => Item::Let(self.parse_let()),
            _ => panic!("expected 'fn' or 'let', got {:?}", self.peek()),
        }
    }

    fn parse_fn_def(&mut self) -> FnDef {
        self.expect(Token::Fn);
        let name = self.expect_ident();
        self.expect(Token::LParen);
        let params = self.parse_comma_separated(|p| p.expect_ident(), Token::RParen);
        self.expect(Token::RParen);
        self.expect(Token::LBrace);
        let body = self.parse_body();
        self.expect(Token::RBrace);
        FnDef { name, params, body }
    }

    fn parse_body(&mut self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        while self.peek() != Token::RBrace {
            if self.peek() == Token::Let {
                stmts.push(Stmt::Let(self.parse_let()));
            } else {
                stmts.push(Stmt::Expr(self.parse_expr()));
            }
        }
        stmts
    }

    fn parse_let(&mut self) -> LetBinding {
        self.expect(Token::Let);
        let name = self.expect_ident();
        self.expect(Token::Eq);
        let value = self.parse_expr();
        LetBinding { name, value }
    }

    fn parse_expr(&mut self) -> Expr {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Expr {
        let mut lhs = self.parse_primary();

        loop {
            let Some((op, l_bp, r_bp)) = self.infix_bp() else {
                break;
            };
            if l_bp < min_bp {
                break;
            }
            self.advance();
            let rhs = self.parse_expr_bp(r_bp);
            lhs = Expr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }

        lhs
    }

    fn infix_bp(&self) -> Option<(BinOpKind, u8, u8)> {
        match self.peek() {
            Token::Plus  => Some((BinOpKind::Add, 1, 2)),
            Token::Minus => Some((BinOpKind::Sub, 1, 2)),
            Token::Star  => Some((BinOpKind::Mul, 3, 4)),
            _ => None,
        }
    }

    fn parse_primary(&mut self) -> Expr {
        match self.peek() {
            Token::Number(n) => {
                let n = n;
                self.advance();
                Expr::Number(n)
            }
            Token::Ident(name) => {
                let name = name;
                self.advance();
                if self.peek() == Token::LParen {
                    self.parse_call(name)
                } else {
                    Expr::Ident(name)
                }
            }
            Token::LBracket => self.parse_array(),
            Token::Minus => {
                self.advance();
                let expr = self.parse_primary();
                Expr::BinOp {
                    op: BinOpKind::Sub,
                    lhs: Box::new(Expr::Number(0.0)),
                    rhs: Box::new(expr),
                }
            }
            _ => panic!("expected expression, got {:?}", self.peek()),
        }
    }

    fn parse_call(&mut self, name: String) -> Expr {
        self.expect(Token::LParen);
        let args = self.parse_comma_separated(|p| p.parse_arg(), Token::RParen);
        self.expect(Token::RParen);
        Expr::Call { name, args }
    }

    fn parse_arg(&mut self) -> Arg {
        // Look ahead: if it's `ident:` then it's a named arg
        if let Token::Ident(name) = self.peek() {
            if self.peek_at(1) == Token::Colon {
                let name = name;
                self.advance(); // ident
                self.advance(); // colon
                let value = self.parse_expr();
                return Arg::Named { name, value };
            }
        }
        Arg::Positional(self.parse_expr())
    }

    fn parse_array(&mut self) -> Expr {
        self.expect(Token::LBracket);
        let elements = self.parse_comma_separated(|p| p.parse_expr(), Token::RBracket);
        self.expect(Token::RBracket);
        Expr::Array(elements)
    }

    // Helpers

    fn peek(&self) -> Token {
        self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof)
    }

    fn peek_at(&self, offset: usize) -> Token {
        self.tokens.get(self.pos + offset).cloned().unwrap_or(Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.peek();
        self.pos += 1;
        tok
    }

    fn at_eof(&self) -> bool {
        self.peek() == Token::Eof
    }

    fn expect(&mut self, expected: Token) {
        let tok = self.advance();
        if tok != expected {
            panic!("expected {expected:?}, got {tok:?}");
        }
    }

    fn expect_ident(&mut self) -> String {
        match self.advance() {
            Token::Ident(s) => s,
            tok => panic!("expected identifier, got {tok:?}"),
        }
    }

    fn parse_comma_separated<T>(
        &mut self,
        mut parse_one: impl FnMut(&mut Self) -> T,
        end: Token,
    ) -> Vec<T> {
        let mut items = Vec::new();
        if self.peek() == end {
            return items;
        }
        items.push(parse_one(self));
        while self.peek() == Token::Comma {
            self.advance();
            if self.peek() == end {
                break;
            }
            items.push(parse_one(self));
        }
        items
    }
}

pub fn parse(input: &str) -> Vec<Item> {
    let mut lexer = crate::lexer::Lexer::new(input);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_let() {
        let items = parse("let x = add(a, b)");
        assert_eq!(items, vec![
            Item::Let(LetBinding {
                name: "x".into(),
                value: Expr::Call {
                    name: "add".into(),
                    args: vec![
                        Arg::Positional(Expr::Ident("a".into())),
                        Arg::Positional(Expr::Ident("b".into())),
                    ],
                },
            }),
        ]);
    }

    #[test]
    fn test_parse_fn() {
        let items = parse("fn softmax(x) { let e = exp(x) e }");
        assert_eq!(items, vec![
            Item::FnDef(FnDef {
                name: "softmax".into(),
                params: vec!["x".into()],
                body: vec![
                    Stmt::Let(LetBinding {
                        name: "e".into(),
                        value: Expr::Call {
                            name: "exp".into(),
                            args: vec![Arg::Positional(Expr::Ident("x".into()))],
                        },
                    }),
                    Stmt::Expr(Expr::Ident("e".into())),
                ],
            }),
        ]);
    }

    #[test]
    fn test_parse_named_arg() {
        let items = parse("let s = sum(x, axis: 1)");
        assert_eq!(items, vec![
            Item::Let(LetBinding {
                name: "s".into(),
                value: Expr::Call {
                    name: "sum".into(),
                    args: vec![
                        Arg::Positional(Expr::Ident("x".into())),
                        Arg::Named { name: "axis".into(), value: Expr::Number(1.0) },
                    ],
                },
            }),
        ]);
    }

    #[test]
    fn test_parse_binop() {
        let items = parse("let y = a * b + c");
        assert_eq!(items, vec![
            Item::Let(LetBinding {
                name: "y".into(),
                value: Expr::BinOp {
                    op: BinOpKind::Add,
                    lhs: Box::new(Expr::BinOp {
                        op: BinOpKind::Mul,
                        lhs: Box::new(Expr::Ident("a".into())),
                        rhs: Box::new(Expr::Ident("b".into())),
                    }),
                    rhs: Box::new(Expr::Ident("c".into())),
                },
            }),
        ]);
    }

    #[test]
    fn test_precedence_mul_before_add() {
        // a + b * c should parse as a + (b * c)
        let items = parse("let y = a + b * c");
        assert_eq!(items, vec![
            Item::Let(LetBinding {
                name: "y".into(),
                value: Expr::BinOp {
                    op: BinOpKind::Add,
                    lhs: Box::new(Expr::Ident("a".into())),
                    rhs: Box::new(Expr::BinOp {
                        op: BinOpKind::Mul,
                        lhs: Box::new(Expr::Ident("b".into())),
                        rhs: Box::new(Expr::Ident("c".into())),
                    }),
                },
            }),
        ]);
    }

    #[test]
    fn test_left_associative() {
        // a - b - c should parse as (a - b) - c
        let items = parse("let y = a - b - c");
        assert_eq!(items, vec![
            Item::Let(LetBinding {
                name: "y".into(),
                value: Expr::BinOp {
                    op: BinOpKind::Sub,
                    lhs: Box::new(Expr::BinOp {
                        op: BinOpKind::Sub,
                        lhs: Box::new(Expr::Ident("a".into())),
                        rhs: Box::new(Expr::Ident("b".into())),
                    }),
                    rhs: Box::new(Expr::Ident("c".into())),
                },
            }),
        ]);
    }

    #[test]
    fn test_parse_array() {
        let items = parse("let x = load([10, 32])");
        assert_eq!(items, vec![
            Item::Let(LetBinding {
                name: "x".into(),
                value: Expr::Call {
                    name: "load".into(),
                    args: vec![
                        Arg::Positional(Expr::Array(vec![
                            Expr::Number(10.0),
                            Expr::Number(32.0),
                        ])),
                    ],
                },
            }),
        ]);
    }

    #[test]
    fn test_parse_full_softmax() {
        let input = r#"
            fn softmax(x) {
                let m = max(x, axis: 1)
                let e = exp(sub(x, m))
                let s = sum(e, axis: 1)
                mul(recip(s), e)
            }
        "#;
        let items = parse(input);
        assert_eq!(items.len(), 1);
        match &items[0] {
            Item::FnDef(f) => {
                assert_eq!(f.name, "softmax");
                assert_eq!(f.params, vec!["x"]);
                assert_eq!(f.body.len(), 4);
            }
            _ => panic!("expected FnDef"),
        }
    }

    #[test]
    fn test_parse_attention() {
        let input = r#"
            fn attention(q, k, v) {
                let scores = matmul(q, permute(k, [0, 2, 1]))
                let weights = softmax(scores)
                matmul(weights, v)
            }
        "#;
        let items = parse(input);
        assert_eq!(items.len(), 1);
        match &items[0] {
            Item::FnDef(f) => {
                assert_eq!(f.name, "attention");
                assert_eq!(f.params, vec!["q", "k", "v"]);
                assert_eq!(f.body.len(), 3);
            }
            _ => panic!("expected FnDef"),
        }
    }
}
