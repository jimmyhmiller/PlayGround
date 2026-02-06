#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Ident(String),
    Int(String),
    Float(String),
    Str(String),

    // Keywords
    Module,
    Use,
    Pub,
    Struct,
    Enum,
    Trait,
    Impl,
    Fn,
    Let,
    Mut,
    If,
    Else,
    While,
    Match,
    Return,
    Extern,
    Repr,
    True,
    False,

    // Symbols
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semi,
    Dot,
    Arrow,
    FatArrow,
    Eq,
    EqEq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,
    AndAnd,
    OrOr,
    ColonColon,
    Ellipsis,
}

impl TokenKind {
    pub fn is_keyword_like(&self) -> bool {
        matches!(
            self,
            TokenKind::Module
                | TokenKind::Use
                | TokenKind::Pub
                | TokenKind::Struct
                | TokenKind::Enum
                | TokenKind::Trait
                | TokenKind::Impl
                | TokenKind::Fn
                | TokenKind::Let
                | TokenKind::Mut
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::While
                | TokenKind::Match
                | TokenKind::Return
                | TokenKind::Extern
                | TokenKind::Repr
                | TokenKind::True
                | TokenKind::False
        )
    }
}
