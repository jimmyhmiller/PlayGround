#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Literals
    Int(i64),
    Ident(String),

    // Keywords
    Fn,
    Let,
    If,
    Else,
    While,
    Reset,
    Shift,
    Resume,
    Clone,
    Return,
    True,
    False,
    Cont,   // cont type annotation
    Arrow,  // ->
    Colon,  // :

    // Punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Eq,       // =
    EqEq,     // ==
    BangEq,   // !=
    Lt,       // <
    LtEq,     // <=
    Gt,       // >
    GtEq,     // >=
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Pipe,     // |
    Amp,      // &
    Bang,     // !

    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub pos: usize,
}

pub fn lex(src: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Skip whitespace
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Line comments
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        let pos = i;

        // Numbers
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let n: i64 = src[start..i].parse().unwrap();
            tokens.push(Token { kind: TokenKind::Int(n), pos });
            continue;
        }

        // Identifiers and keywords
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &src[start..i];
            let kind = match word {
                "fn" => TokenKind::Fn,
                "let" => TokenKind::Let,
                "if" => TokenKind::If,
                "else" => TokenKind::Else,
                "while" => TokenKind::While,
                "reset" => TokenKind::Reset,
                "shift" => TokenKind::Shift,
                "resume" => TokenKind::Resume,
                "clone" => TokenKind::Clone,
                "return" => TokenKind::Return,
                "true" => TokenKind::True,
                "false" => TokenKind::False,
                "cont" => TokenKind::Cont,
                _ => TokenKind::Ident(word.to_string()),
            };
            tokens.push(Token { kind, pos });
            continue;
        }

        // Two-char tokens
        if i + 1 < bytes.len() {
            let two = &src[i..i + 2];
            let kind = match two {
                "==" => Some(TokenKind::EqEq),
                "!=" => Some(TokenKind::BangEq),
                "<=" => Some(TokenKind::LtEq),
                ">=" => Some(TokenKind::GtEq),
                "->" => Some(TokenKind::Arrow),
                _ => None,
            };
            if let Some(k) = kind {
                tokens.push(Token { kind: k, pos });
                i += 2;
                continue;
            }
        }

        // Single-char tokens
        let kind = match bytes[i] {
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b',' => TokenKind::Comma,
            b';' => TokenKind::Semicolon,
            b'=' => TokenKind::Eq,
            b'<' => TokenKind::Lt,
            b'>' => TokenKind::Gt,
            b'+' => TokenKind::Plus,
            b'-' => TokenKind::Minus,
            b'*' => TokenKind::Star,
            b'/' => TokenKind::Slash,
            b'%' => TokenKind::Percent,
            b'|' => TokenKind::Pipe,
            b'&' => TokenKind::Amp,
            b'!' => TokenKind::Bang,
            b':' => TokenKind::Colon,
            c => panic!("unexpected character '{}' at position {}", c as char, i),
        };
        tokens.push(Token { kind, pos });
        i += 1;
    }

    tokens.push(Token { kind: TokenKind::Eof, pos: i });
    tokens
}
