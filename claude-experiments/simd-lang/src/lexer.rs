#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Ident(String),
    IntLit(i64),
    FloatLit(f64),
    CharLit(char),

    // Keywords
    Struct,
    Fn,
    Return,
    Let,
    Comptime,
    True,
    False,
    Stream,
    Over,
    Carry,

    // Arithmetic operators
    Plus,
    Minus,
    Star,
    Slash,

    // Comparison operators
    Gt,
    Lt,
    GtEq,
    LtEq,
    EqEq,
    BangEq,

    // Logical/bitwise operators
    Amp,
    Pipe,
    Tilde,
    LtLt,  // <<
    GtGt,  // >>

    // Assignment
    Eq,

    // Reduction operators: +/ */ |/ &/ max/ min/
    PlusSlash,
    StarSlash,
    PipeSlash,
    AmpSlash,
    MaxSlash,
    MinSlash,

    // Delimiters
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Dot,
    Arrow,       // ->
    DotBracket,  // .[

    // Special
    Underscore, // _ in f32[_]

    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spanned {
    pub token: Token,
    pub line: usize,
    pub col: usize,
}

pub struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer {
            input: input.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.input.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.input.get(self.pos).copied()?;
        self.pos += 1;
        if ch == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while self.peek().is_some_and(|c| c.is_ascii_whitespace()) {
                self.advance();
            }
            // Skip line comments: --
            if self.peek() == Some(b'-') && self.peek2() == Some(b'-') {
                while self.peek().is_some_and(|c| c != b'\n') {
                    self.advance();
                }
                continue;
            }
            break;
        }
    }

    fn read_ident_or_keyword(&mut self) -> Token {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
            self.advance();
        }
        let text = std::str::from_utf8(&self.input[start..self.pos]).unwrap();

        // Check for reduction keywords: max/ min/
        if (text == "max" || text == "min") && self.peek() == Some(b'/') {
            self.advance(); // consume '/'
            return match text {
                "max" => Token::MaxSlash,
                "min" => Token::MinSlash,
                _ => unreachable!(),
            };
        }

        match text {
            "struct" => Token::Struct,
            "fn" => Token::Fn,
            "return" => Token::Return,
            "let" => Token::Let,
            "comptime" => Token::Comptime,
            "true" => Token::True,
            "false" => Token::False,
            "stream" => Token::Stream,
            "over" => Token::Over,
            "carry" => Token::Carry,
            "_" => Token::Underscore,
            _ => Token::Ident(text.to_string()),
        }
    }

    fn read_number(&mut self) -> Token {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.advance();
        }
        // Check for float: digit followed by '.' then digit (not '..' or '.[')
        if self.peek() == Some(b'.')
            && self
                .peek2()
                .is_some_and(|c| c.is_ascii_digit())
        {
            self.advance(); // consume '.'
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.advance();
            }
            let text = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
            Token::FloatLit(text.parse().unwrap())
        } else {
            let text = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
            Token::IntLit(text.parse().unwrap())
        }
    }

    fn read_char_lit(&mut self) -> Token {
        self.advance(); // consume opening '
        let ch = self.advance().expect("unexpected end of input in char literal");
        let ch = if ch == b'\\' {
            let esc = self.advance().expect("unexpected end of input in escape");
            match esc {
                b'n' => '\n',
                b'r' => '\r',
                b't' => '\t',
                b'\\' => '\\',
                b'\'' => '\'',
                _ => panic!("unknown escape sequence: \\{}", esc as char),
            }
        } else {
            ch as char
        };
        assert_eq!(
            self.advance(),
            Some(b'\''),
            "expected closing quote for char literal"
        );
        Token::CharLit(ch)
    }

    pub fn tokenize(&mut self) -> Vec<Spanned> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            let line = self.line;
            let col = self.col;
            let Some(ch) = self.peek() else {
                tokens.push(Spanned {
                    token: Token::Eof,
                    line,
                    col,
                });
                break;
            };

            let token = match ch {
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.read_ident_or_keyword(),

                b'0'..=b'9' => self.read_number(),

                b'\'' => self.read_char_lit(),

                b'+' => {
                    self.advance();
                    if self.peek() == Some(b'/') {
                        self.advance();
                        Token::PlusSlash
                    } else {
                        Token::Plus
                    }
                }
                b'-' => {
                    self.advance();
                    if self.peek() == Some(b'>') {
                        self.advance();
                        Token::Arrow
                    } else {
                        Token::Minus
                    }
                }
                b'*' => {
                    self.advance();
                    if self.peek() == Some(b'/') {
                        self.advance();
                        Token::StarSlash
                    } else {
                        Token::Star
                    }
                }
                b'/' => {
                    self.advance();
                    Token::Slash
                }
                b'>' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        Token::GtEq
                    } else if self.peek() == Some(b'>') {
                        self.advance();
                        Token::GtGt
                    } else {
                        Token::Gt
                    }
                }
                b'<' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        Token::LtEq
                    } else if self.peek() == Some(b'<') {
                        self.advance();
                        Token::LtLt
                    } else {
                        Token::Lt
                    }
                }
                b'=' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        Token::EqEq
                    } else {
                        Token::Eq
                    }
                }
                b'!' => {
                    self.advance();
                    assert_eq!(self.advance(), Some(b'='), "expected '=' after '!'");
                    Token::BangEq
                }
                b'&' => {
                    self.advance();
                    if self.peek() == Some(b'/') {
                        self.advance();
                        Token::AmpSlash
                    } else {
                        Token::Amp
                    }
                }
                b'|' => {
                    self.advance();
                    if self.peek() == Some(b'/') {
                        self.advance();
                        Token::PipeSlash
                    } else {
                        Token::Pipe
                    }
                }
                b'~' => {
                    self.advance();
                    Token::Tilde
                }
                b'(' => {
                    self.advance();
                    Token::LParen
                }
                b')' => {
                    self.advance();
                    Token::RParen
                }
                b'[' => {
                    self.advance();
                    Token::LBracket
                }
                b']' => {
                    self.advance();
                    Token::RBracket
                }
                b'{' => {
                    self.advance();
                    Token::LBrace
                }
                b'}' => {
                    self.advance();
                    Token::RBrace
                }
                b',' => {
                    self.advance();
                    Token::Comma
                }
                b':' => {
                    self.advance();
                    Token::Colon
                }
                b'.' => {
                    self.advance();
                    if self.peek() == Some(b'[') {
                        self.advance();
                        Token::DotBracket
                    } else {
                        Token::Dot
                    }
                }
                _ => panic!(
                    "unexpected character '{}' at {}:{}",
                    ch as char, line, col
                ),
            };

            tokens.push(Spanned { token, line, col });
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(input: &str) -> Vec<Token> {
        Lexer::new(input)
            .tokenize()
            .into_iter()
            .map(|s| s.token)
            .filter(|t| *t != Token::Eof)
            .collect()
    }

    // --- Basic tokens ---

    #[test]
    fn test_empty() {
        assert_eq!(lex(""), vec![]);
    }

    #[test]
    fn test_whitespace_only() {
        assert_eq!(lex("   \n\t  \n  "), vec![]);
    }

    #[test]
    fn test_identifier() {
        assert_eq!(lex("foo"), vec![Token::Ident("foo".into())]);
    }

    #[test]
    fn test_identifier_with_underscore() {
        assert_eq!(lex("pos_x"), vec![Token::Ident("pos_x".into())]);
    }

    #[test]
    fn test_identifier_with_digits() {
        assert_eq!(lex("v2"), vec![Token::Ident("v2".into())]);
    }

    #[test]
    fn test_underscore_alone() {
        assert_eq!(lex("_"), vec![Token::Underscore]);
    }

    #[test]
    fn test_underscore_prefix_is_ident() {
        assert_eq!(lex("_foo"), vec![Token::Ident("_foo".into())]);
    }

    // --- Keywords ---

    #[test]
    fn test_keywords() {
        assert_eq!(
            lex("struct fn return let comptime true false"),
            vec![
                Token::Struct,
                Token::Fn,
                Token::Return,
                Token::Let,
                Token::Comptime,
                Token::True,
                Token::False,
            ]
        );
    }

    // --- Number literals ---

    #[test]
    fn test_integer() {
        assert_eq!(lex("42"), vec![Token::IntLit(42)]);
    }

    #[test]
    fn test_zero() {
        assert_eq!(lex("0"), vec![Token::IntLit(0)]);
    }

    #[test]
    fn test_float() {
        assert_eq!(lex("3.14"), vec![Token::FloatLit(3.14)]);
    }

    #[test]
    fn test_float_leading_zero() {
        assert_eq!(lex("0.5"), vec![Token::FloatLit(0.5)]);
    }

    #[test]
    fn test_float_trailing_zeros() {
        assert_eq!(lex("1.00"), vec![Token::FloatLit(1.0)]);
    }

    #[test]
    fn test_integer_followed_by_dot_ident() {
        // 1.foo should be IntLit(1), Dot, Ident("foo")
        assert_eq!(
            lex("1.foo"),
            vec![Token::IntLit(1), Token::Dot, Token::Ident("foo".into())]
        );
    }

    // --- Char literals ---

    #[test]
    fn test_char_lit() {
        assert_eq!(lex("'a'"), vec![Token::CharLit('a')]);
    }

    #[test]
    fn test_char_lit_escape_n() {
        assert_eq!(lex("'\\n'"), vec![Token::CharLit('\n')]);
    }

    #[test]
    fn test_char_lit_backslash() {
        assert_eq!(lex("'\\\\'"), vec![Token::CharLit('\\')]);
    }

    // --- Operators ---

    #[test]
    fn test_arithmetic_ops() {
        assert_eq!(
            lex("+ - * /"),
            vec![Token::Plus, Token::Minus, Token::Star, Token::Slash]
        );
    }

    #[test]
    fn test_comparison_ops() {
        assert_eq!(
            lex("> < >= <= == !="),
            vec![
                Token::Gt,
                Token::Lt,
                Token::GtEq,
                Token::LtEq,
                Token::EqEq,
                Token::BangEq,
            ]
        );
    }

    #[test]
    fn test_logical_ops() {
        assert_eq!(lex("& | ~"), vec![Token::Amp, Token::Pipe, Token::Tilde]);
    }

    #[test]
    fn test_assignment() {
        assert_eq!(lex("="), vec![Token::Eq]);
    }

    // --- Reduction operators ---

    #[test]
    fn test_reduction_plus() {
        assert_eq!(lex("+/"), vec![Token::PlusSlash]);
    }

    #[test]
    fn test_reduction_star() {
        assert_eq!(lex("*/"), vec![Token::StarSlash]);
    }

    #[test]
    fn test_reduction_pipe() {
        assert_eq!(lex("|/"), vec![Token::PipeSlash]);
    }

    #[test]
    fn test_reduction_amp() {
        assert_eq!(lex("&/"), vec![Token::AmpSlash]);
    }

    #[test]
    fn test_reduction_max() {
        assert_eq!(lex("max/"), vec![Token::MaxSlash]);
    }

    #[test]
    fn test_reduction_min() {
        assert_eq!(lex("min/"), vec![Token::MinSlash]);
    }

    #[test]
    fn test_reduction_in_context() {
        // +/ v should be PlusSlash, Ident
        assert_eq!(
            lex("+/ v"),
            vec![Token::PlusSlash, Token::Ident("v".into())]
        );
    }

    // --- Delimiters ---

    #[test]
    fn test_delimiters() {
        assert_eq!(
            lex("( ) [ ] { } , : ."),
            vec![
                Token::LParen,
                Token::RParen,
                Token::LBracket,
                Token::RBracket,
                Token::LBrace,
                Token::RBrace,
                Token::Comma,
                Token::Colon,
                Token::Dot,
            ]
        );
    }

    #[test]
    fn test_arrow() {
        assert_eq!(lex("->"), vec![Token::Arrow]);
    }

    #[test]
    fn test_dot_bracket() {
        assert_eq!(lex(".["), vec![Token::DotBracket]);
    }

    // --- Comments ---

    #[test]
    fn test_line_comment() {
        assert_eq!(lex("-- this is a comment"), vec![]);
    }

    #[test]
    fn test_comment_before_code() {
        assert_eq!(
            lex("-- comment\nfoo"),
            vec![Token::Ident("foo".into())]
        );
    }

    #[test]
    fn test_inline_comment() {
        assert_eq!(
            lex("foo -- comment"),
            vec![Token::Ident("foo".into())]
        );
    }

    // --- Complex expressions ---

    #[test]
    fn test_type_annotation() {
        // f32[8]
        assert_eq!(
            lex("f32[8]"),
            vec![
                Token::Ident("f32".into()),
                Token::LBracket,
                Token::IntLit(8),
                Token::RBracket,
            ]
        );
    }

    #[test]
    fn test_native_width_type() {
        // f32[_]
        assert_eq!(
            lex("f32[_]"),
            vec![
                Token::Ident("f32".into()),
                Token::LBracket,
                Token::Underscore,
                Token::RBracket,
            ]
        );
    }

    #[test]
    fn test_masked_op() {
        // [mask] a + b : c
        assert_eq!(
            lex("[mask] a + b : c"),
            vec![
                Token::LBracket,
                Token::Ident("mask".into()),
                Token::RBracket,
                Token::Ident("a".into()),
                Token::Plus,
                Token::Ident("b".into()),
                Token::Colon,
                Token::Ident("c".into()),
            ]
        );
    }

    #[test]
    fn test_reduction_expr() {
        // +/ (a * b)
        assert_eq!(
            lex("+/ (a * b)"),
            vec![
                Token::PlusSlash,
                Token::LParen,
                Token::Ident("a".into()),
                Token::Star,
                Token::Ident("b".into()),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn test_struct_field_access() {
        assert_eq!(
            lex("p.pos_x"),
            vec![
                Token::Ident("p".into()),
                Token::Dot,
                Token::Ident("pos_x".into()),
            ]
        );
    }

    #[test]
    fn test_gather_syntax() {
        // src.[indices]
        assert_eq!(
            lex("src.[indices]"),
            vec![
                Token::Ident("src".into()),
                Token::DotBracket,
                Token::Ident("indices".into()),
                Token::RBracket,
            ]
        );
    }

    #[test]
    fn test_function_signature() {
        assert_eq!(
            lex("fn update(p: Particle[1024], dt: f32[_]) -> Particle[1024]"),
            vec![
                Token::Fn,
                Token::Ident("update".into()),
                Token::LParen,
                Token::Ident("p".into()),
                Token::Colon,
                Token::Ident("Particle".into()),
                Token::LBracket,
                Token::IntLit(1024),
                Token::RBracket,
                Token::Comma,
                Token::Ident("dt".into()),
                Token::Colon,
                Token::Ident("f32".into()),
                Token::LBracket,
                Token::Underscore,
                Token::RBracket,
                Token::RParen,
                Token::Arrow,
                Token::Ident("Particle".into()),
                Token::LBracket,
                Token::IntLit(1024),
                Token::RBracket,
            ]
        );
    }

    #[test]
    fn test_negative_number_is_minus_then_int() {
        // -500.0 should be Minus, FloatLit(500.0)
        assert_eq!(
            lex("-500.0"),
            vec![Token::Minus, Token::FloatLit(500.0)]
        );
    }

    // --- Span tracking ---

    #[test]
    fn test_span_line_col() {
        let tokens = Lexer::new("foo\nbar").tokenize();
        assert_eq!(tokens[0].line, 1);
        assert_eq!(tokens[0].col, 1);
        assert_eq!(tokens[1].line, 2);
        assert_eq!(tokens[1].col, 1);
    }

    #[test]
    fn test_span_col_tracking() {
        let tokens = Lexer::new("a + b").tokenize();
        assert_eq!(tokens[0].col, 1); // a
        assert_eq!(tokens[1].col, 3); // +
        assert_eq!(tokens[2].col, 5); // b
    }

    // --- Edge cases ---

    #[test]
    fn test_max_as_ident_without_slash() {
        // "max" alone (not followed by /) should be an identifier
        assert_eq!(lex("max"), vec![Token::Ident("max".into())]);
    }

    #[test]
    fn test_min_as_ident_without_slash() {
        assert_eq!(lex("min"), vec![Token::Ident("min".into())]);
    }

    #[test]
    fn test_consecutive_dots() {
        assert_eq!(lex(".."), vec![Token::Dot, Token::Dot]);
    }

    #[test]
    fn test_struct_literal() {
        assert_eq!(
            lex("Particle[1024] { pos_x: v }"),
            vec![
                Token::Ident("Particle".into()),
                Token::LBracket,
                Token::IntLit(1024),
                Token::RBracket,
                Token::LBrace,
                Token::Ident("pos_x".into()),
                Token::Colon,
                Token::Ident("v".into()),
                Token::RBrace,
            ]
        );
    }

    #[test]
    fn test_scan_dot_add() {
        assert_eq!(
            lex("scan.add(v)"),
            vec![
                Token::Ident("scan".into()),
                Token::Dot,
                Token::Ident("add".into()),
                Token::LParen,
                Token::Ident("v".into()),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn test_bool_type() {
        assert_eq!(
            lex("bool[32]"),
            vec![
                Token::Ident("bool".into()),
                Token::LBracket,
                Token::IntLit(32),
                Token::RBracket,
            ]
        );
    }

    // --- Shift operators ---

    #[test]
    fn test_shift_left() {
        assert_eq!(lex("<<"), vec![Token::LtLt]);
    }

    #[test]
    fn test_shift_right() {
        assert_eq!(lex(">>"), vec![Token::GtGt]);
    }

    #[test]
    fn test_shift_in_expr() {
        assert_eq!(
            lex("v >> 1"),
            vec![Token::Ident("v".into()), Token::GtGt, Token::IntLit(1)]
        );
    }

    #[test]
    fn test_shift_vs_comparison() {
        // >= should not be confused with >>
        assert_eq!(lex(">="), vec![Token::GtEq]);
        assert_eq!(lex(">>"), vec![Token::GtGt]);
    }

    // --- Stream keywords ---

    #[test]
    fn test_stream_keyword() {
        assert_eq!(lex("stream"), vec![Token::Stream]);
    }

    #[test]
    fn test_over_keyword() {
        assert_eq!(lex("over"), vec![Token::Over]);
    }

    #[test]
    fn test_carry_keyword() {
        assert_eq!(lex("carry"), vec![Token::Carry]);
    }

    #[test]
    fn test_stream_header() {
        assert_eq!(
            lex("stream chunk: u8[64] over input"),
            vec![
                Token::Stream,
                Token::Ident("chunk".into()),
                Token::Colon,
                Token::Ident("u8".into()),
                Token::LBracket,
                Token::IntLit(64),
                Token::RBracket,
                Token::Over,
                Token::Ident("input".into()),
            ]
        );
    }
}
