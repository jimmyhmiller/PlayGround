use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct SrcLoc {
    pub start_line: usize,
    pub start_col: usize,
    pub start_pos: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub end_pos: usize,
}

impl SrcLoc {
    pub fn new(
        start_line: usize,
        start_col: usize,
        start_pos: usize,
        end_line: usize,
        end_col: usize,
        end_pos: usize,
    ) -> Self {
        SrcLoc {
            start_line,
            start_col,
            start_pos,
            end_line,
            end_col,
            end_pos,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Keywords
    And,
    As,
    Ascending,
    Ask,
    By,
    Cases,
    Check,
    Data,
    Descending,
    Do,
    DoesNotRaise,
    Else,
    ElseIf,
    End,
    Examples,
    TableExtend,
    TableExtract,
    False,
    For,
    From,
    Fun,
    Hiding,
    If,
    Import,
    Include,
    Is,
    IsEqualEqual,
    IsEqualTilde,
    IsNot,
    IsNotEqualEqual,
    IsNotEqualTilde,
    IsNotSpaceship,
    IsRoughly,
    IsNotRoughly,
    IsSpaceship,
    Because,
    Lam,
    Lazy,
    Let,
    Letrec,
    LoadTable,
    Method,
    Module,
    Newtype,
    Not,
    Of,
    Or,
    Provide,
    ProvideTypes,
    Raises,
    RaisesOtherThan,
    RaisesSatisfies,
    RaisesViolates,
    Reactor,
    Rec,
    Ref,
    Sanitize,
    Satisfies,
    TableSelect,
    Shadow,
    TableFilter,
    Spy,
    TableOrder,
    TableUpdate,
    True,
    Type,
    TypeLet,
    Using,
    Use,
    Var,
    Violates,
    When,

    // Symbols with colons
    Block,
    CheckColon,
    Doc,
    ElseColon,
    ExamplesColon,
    OtherwiseColon,
    ProvideColon,
    Row,
    Sharing,
    SourceColon,
    Table,
    ThenColon,
    Where,
    With,

    // Brackets and parentheses
    LBrack,
    BrackSpace,
    BrackNoSpace,
    RBrack,
    LBrace,
    RBrace,
    LParen,
    ParenSpace,
    ParenNoSpace,
    ParenAfterBrace,
    RParen,

    // Punctuation
    Semi,
    Backslash,
    DotDotDot,
    Dot,
    Bang,
    Percent,
    Comma,
    ThinArrow,
    ColonEquals,
    Colon,
    Bar,
    Equals,
    LAngle,
    Star,
    RAngle,

    // Operators
    Caret,
    Plus,
    Dash,
    Times,
    Slash,
    Spaceship,
    Leq,
    Geq,
    EqualEqual,
    EqualTilde,
    Neq,
    Lt,
    LtNoSpace,
    Gt,
    ThickArrow,
    ColonColon,

    // Literals
    Number,
    RoughNumber,      // Rough (approximate) numbers like ~0.8 or ~42
    RoughRational,
    Rational,
    String,
    Name,

    // Special tokens
    Comment,
    BlockComment,
    UnterminatedString,
    UnterminatedBlockComment,
    BadNumber,
    BadOper,

    // Internal
    Ws,
    Eof,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub token_type: TokenType,
    pub value: String,
    pub location: SrcLoc,
}

impl Token {
    pub fn new(token_type: TokenType, value: String, location: SrcLoc) -> Self {
        Token {
            token_type,
            value,
            location,
        }
    }
}

pub struct Tokenizer {
    input: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
    len: usize,
    paren_is_for_exp: bool,
    prior_whitespace: bool,
}

impl Tokenizer {
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let len = chars.len();
        Tokenizer {
            input: chars,
            pos: 0,
            line: 1,
            col: 0,
            len,
            paren_is_for_exp: true,
            prior_whitespace: false,
        }
    }

    fn current_char(&self) -> Option<char> {
        if self.pos < self.len {
            Some(self.input[self.pos])
        } else {
            None
        }
    }

    fn peek_char(&self, offset: usize) -> Option<char> {
        let peek_pos = self.pos + offset;
        if peek_pos < self.len {
            Some(self.input[peek_pos])
        } else {
            None
        }
    }

    fn advance(&mut self) -> Option<char> {
        if self.pos < self.len {
            let ch = self.input[self.pos];
            self.pos += 1;
            if ch == '\n' {
                self.line += 1;
                self.col = 0;
            } else {
                self.col += 1;
            }
            Some(ch)
        } else {
            None
        }
    }

    fn starts_with(&self, s: &str) -> bool {
        let chars: Vec<char> = s.chars().collect();
        if self.pos + chars.len() > self.len {
            return false;
        }
        for (i, ch) in chars.iter().enumerate() {
            if self.input[self.pos + i] != *ch {
                return false;
            }
        }
        true
    }

    fn is_whitespace(ch: char) -> bool {
        matches!(
            ch,
            ' ' | '\t' | '\n' | '\r' | '\x0C' | '\x0B' | '\u{00A0}' | '\u{1680}' | '\u{2000}'
                | '\u{2001}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}' | '\u{2006}'
                | '\u{2007}' | '\u{2008}' | '\u{2009}' | '\u{200A}' | '\u{2028}' | '\u{2029}'
                | '\u{202F}' | '\u{205F}' | '\u{3000}' | '\u{FEFF}'
        )
    }

    fn is_ident_start(ch: char) -> bool {
        ch.is_ascii_alphabetic() || ch == '_'
    }

    fn is_ident_continue(ch: char) -> bool {
        ch.is_ascii_alphanumeric() || ch == '_'
    }

    fn is_digit(ch: char) -> bool {
        ch.is_ascii_digit()
    }

    fn skip_whitespace(&mut self) -> Option<Token> {
        let start_line = self.line;
        let start_col = self.col;
        let start_pos = self.pos;

        while let Some(ch) = self.current_char() {
            if Self::is_whitespace(ch) {
                self.advance();
            } else {
                break;
            }
        }

        if self.pos > start_pos {
            self.paren_is_for_exp = true;
            self.prior_whitespace = true;
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            Some(Token::new(TokenType::Ws, String::new(), loc))
        } else {
            None
        }
    }

    fn tokenize_line_comment(&mut self) -> Option<Token> {
        if !self.starts_with("#") || self.starts_with("#|") {
            return None;
        }

        let start_line = self.line;
        let start_col = self.col;
        let start_pos = self.pos;

        self.advance(); // consume '#'

        while let Some(ch) = self.current_char() {
            if ch == '\n' || ch == '\r' {
                break;
            }
            self.advance();
        }

        let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
        Some(Token::new(TokenType::Comment, String::new(), loc))
    }

    fn tokenize_block_comment(&mut self) -> Option<Token> {
        if !self.starts_with("#|") {
            return None;
        }

        let start_line = self.line;
        let start_col = self.col;
        let start_pos = self.pos;

        self.advance(); // consume '#'
        self.advance(); // consume '|'

        let mut nesting_depth = 1;

        while nesting_depth > 0 && self.pos < self.len {
            if self.starts_with("#|") {
                nesting_depth += 1;
                self.advance();
                self.advance();
            } else if self.starts_with("|#") {
                nesting_depth -= 1;
                self.advance();
                self.advance();
            } else {
                self.advance();
            }
        }

        let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);

        if nesting_depth == 0 {
            Some(Token::new(TokenType::BlockComment, String::new(), loc))
        } else {
            let value: String = self.input[start_pos..self.pos].iter().collect();
            Some(Token::new(TokenType::UnterminatedBlockComment, value, loc))
        }
    }

    fn fix_escapes(&self, s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '\\' {
                if let Some(&next_ch) = chars.peek() {
                    chars.next(); // consume the escape character
                    match next_ch {
                        'n' => result.push('\n'),
                        'r' => result.push('\r'),
                        't' => result.push('\t'),
                        '"' => result.push('"'),
                        '\'' => result.push('\''),
                        '\\' => result.push('\\'),
                        'u' => {
                            // Unicode escape: \uXXXX
                            let mut hex = String::new();
                            for _ in 0..4 {
                                if let Some(&h) = chars.peek() {
                                    if h.is_ascii_hexdigit() {
                                        hex.push(h);
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                            }
                            if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                if let Some(unicode_char) = char::from_u32(code) {
                                    result.push(unicode_char);
                                }
                            }
                        }
                        'x' => {
                            // Hex escape: \xXX
                            let mut hex = String::new();
                            for _ in 0..2 {
                                if let Some(&h) = chars.peek() {
                                    if h.is_ascii_hexdigit() {
                                        hex.push(h);
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                            }
                            if let Ok(code) = u8::from_str_radix(&hex, 16) {
                                result.push(code as char);
                            }
                        }
                        c if c.is_ascii_digit() => {
                            // Octal escape: \XXX
                            let mut octal = String::new();
                            octal.push(c);
                            for _ in 0..2 {
                                if let Some(&o) = chars.peek() {
                                    if ('0'..='7').contains(&o) {
                                        octal.push(o);
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                            }
                            if let Ok(code) = u8::from_str_radix(&octal, 8) {
                                result.push(code as char);
                            }
                        }
                        '\n' | '\r' => {
                            // Line continuation - skip
                        }
                        _ => {
                            result.push('\\');
                            result.push(next_ch);
                        }
                    }
                } else {
                    result.push(ch);
                }
            } else {
                result.push(ch);
            }
        }

        result
    }

    fn tokenize_string(&mut self) -> Option<Token> {
        let quote = self.current_char()?;

        if quote == '`' && self.starts_with("```") {
            return self.tokenize_triple_backtick_string();
        }

        if quote != '"' && quote != '\'' {
            return None;
        }

        let start_line = self.line;
        let start_col = self.col;
        let start_pos = self.pos;

        self.advance(); // consume opening quote

        let mut terminated = false;

        while let Some(ch) = self.current_char() {
            if ch == '\\' {
                self.advance();
                if self.current_char().is_some() {
                    self.advance();
                }
            } else if ch == quote {
                self.advance(); // consume closing quote
                terminated = true;
                break;
            } else if ch == '\n' || ch == '\r' {
                // Unterminated string
                break;
            } else {
                self.advance();
            }
        }

        let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
        let raw_value: String = self.input[start_pos..self.pos].iter().collect();

        if terminated {
            // Strip the opening and closing quotes before processing escapes
            let without_quotes = if raw_value.len() >= 2 {
                &raw_value[1..raw_value.len() - 1]
            } else {
                ""
            };
            let value = self.fix_escapes(without_quotes);
            Some(Token::new(TokenType::String, value, loc))
        } else {
            Some(Token::new(TokenType::UnterminatedString, raw_value, loc))
        }
    }

    fn tokenize_triple_backtick_string(&mut self) -> Option<Token> {
        let start_line = self.line;
        let start_col = self.col;
        let start_pos = self.pos;

        // Consume opening ```
        self.advance();
        self.advance();
        self.advance();

        let mut terminated = false;

        while self.pos < self.len {
            if self.starts_with("```") {
                self.advance();
                self.advance();
                self.advance();
                terminated = true;
                break;
            } else if self.current_char() == Some('\\') {
                self.advance();
                if self.current_char().is_some() {
                    self.advance();
                }
            } else {
                self.advance();
            }
        }

        let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
        let raw_value: String = self.input[start_pos..self.pos].iter().collect();

        if terminated {
            // Strip the opening and closing triple backticks (```) before processing escapes
            let without_quotes = if raw_value.len() >= 6 {
                &raw_value[3..raw_value.len() - 3]
            } else {
                ""
            };
            // Trim leading/trailing whitespace (Pyret behavior for triple-backtick strings)
            let trimmed = without_quotes.trim();
            let value = self.fix_escapes(trimmed);
            Some(Token::new(TokenType::String, value, loc))
        } else {
            Some(Token::new(TokenType::UnterminatedString, raw_value, loc))
        }
    }

    fn tokenize_number(&mut self) -> Option<Token> {
        let start_line = self.line;
        let start_col = self.col;
        let start_pos = self.pos;

        let mut rough = false;
        if self.current_char() == Some('~') {
            rough = true;
            self.advance();
        }

        // Optional sign
        if matches!(self.current_char(), Some('+') | Some('-')) {
            self.advance();
        }

        // Check for bad number starting with '.'
        if self.current_char() == Some('.') {
            self.pos = start_pos;
            self.line = start_line;
            self.col = start_col;
            return None;
        }

        // Must have at least one digit
        if !matches!(self.current_char(), Some(ch) if Self::is_digit(ch)) {
            self.pos = start_pos;
            self.line = start_line;
            self.col = start_col;
            return None;
        }

        // Integer part or numerator
        while matches!(self.current_char(), Some(ch) if Self::is_digit(ch)) {
            self.advance();
        }

        // Check for rational (fraction)
        if self.current_char() == Some('/') {
            self.advance();
            if matches!(self.current_char(), Some(ch) if Self::is_digit(ch)) {
                while matches!(self.current_char(), Some(ch) if Self::is_digit(ch)) {
                    self.advance();
                }
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                let value: String = self.input[start_pos..self.pos].iter().collect();
                self.paren_is_for_exp = false;
                self.prior_whitespace = false;
                let token_type = if rough { TokenType::RoughRational } else { TokenType::Rational };
                return Some(Token::new(token_type, value, loc));
            } else {
                self.pos = start_pos;
                self.line = start_line;
                self.col = start_col;
                return None;
            }
        }

        // Check for decimal point
        if self.current_char() == Some('.') {
            let dot_pos = self.pos;
            self.advance();

            if matches!(self.current_char(), Some(ch) if Self::is_digit(ch)) {
                while matches!(self.current_char(), Some(ch) if Self::is_digit(ch)) {
                    self.advance();
                }
            } else {
                // No digits after dot - backtrack
                self.pos = dot_pos;
                self.col -= 1;
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                let value: String = self.input[start_pos..self.pos].iter().collect();
                self.paren_is_for_exp = false;
                self.prior_whitespace = false;
                return Some(Token::new(TokenType::Number, value, loc));
            }
        }

        // Check for exponent
        if matches!(self.current_char(), Some('e') | Some('E')) {
            let exp_pos = self.pos;
            self.advance();

            if matches!(self.current_char(), Some('+') | Some('-')) {
                self.advance();
            }

            if matches!(self.current_char(), Some(ch) if Self::is_digit(ch)) {
                while matches!(self.current_char(), Some(ch) if Self::is_digit(ch)) {
                    self.advance();
                }
            } else {
                // No valid exponent - backtrack
                self.pos = exp_pos;
                self.col = start_col + (exp_pos - start_pos);
            }
        }

        let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
        let value: String = self.input[start_pos..self.pos].iter().collect();
        self.paren_is_for_exp = false;
        self.prior_whitespace = false;
        let token_type = if rough { TokenType::RoughNumber } else { TokenType::Number };
        Some(Token::new(token_type, value, loc))
    }

    fn tokenize_name_or_keyword(&mut self) -> Option<Token> {
        if !matches!(self.current_char(), Some(ch) if Self::is_ident_start(ch)) {
            return None;
        }

        let start_line = self.line;
        let start_col = self.col;
        let start_pos = self.pos;

        // Check for special keywords with operators before scanning as identifier
        // These need to be checked first because '=' and '<' are not valid identifier chars
        if self.starts_with("is-not<=>") {
            for _ in 0..9 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::IsNotSpaceship, "is-not<=>".to_string(), loc));
        }
        if self.starts_with("is-not==") {
            for _ in 0..8 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::IsNotEqualEqual, "is-not==".to_string(), loc));
        }
        if self.starts_with("is-not=~") {
            for _ in 0..8 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::IsNotEqualTilde, "is-not=~".to_string(), loc));
        }
        if self.starts_with("is<=>") {
            for _ in 0..5 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::IsSpaceship, "is<=>".to_string(), loc));
        }
        if self.starts_with("is==") {
            for _ in 0..4 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::IsEqualEqual, "is==".to_string(), loc));
        }
        if self.starts_with("is=~") {
            for _ in 0..4 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::IsEqualTilde, "is=~".to_string(), loc));
        }

        // Check for special keyword-colon combinations before scanning as identifier
        if self.starts_with("block:") {
            for _ in 0..6 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Block, "block:".to_string(), loc));
        }
        if self.starts_with("check:") {
            for _ in 0..6 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::CheckColon, "check:".to_string(), loc));
        }
        if self.starts_with("doc:") {
            for _ in 0..4 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Doc, "doc:".to_string(), loc));
        }
        if self.starts_with("else:") {
            for _ in 0..5 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ElseColon, "else:".to_string(), loc));
        }
        if self.starts_with("examples:") {
            for _ in 0..9 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ExamplesColon, "examples:".to_string(), loc));
        }
        if self.starts_with("provide:") {
            for _ in 0..8 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ProvideColon, "provide:".to_string(), loc));
        }
        if self.starts_with("row:") {
            for _ in 0..4 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Row, "row:".to_string(), loc));
        }
        if self.starts_with("sharing:") {
            for _ in 0..8 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Sharing, "sharing:".to_string(), loc));
        }
        if self.starts_with("source:") {
            for _ in 0..7 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::SourceColon, "source:".to_string(), loc));
        }
        if self.starts_with("load-table:") {
            for _ in 0..11 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            self.paren_is_for_exp = true;
            return Some(Token::new(TokenType::LoadTable, "load-table:".to_string(), loc));
        }
        if self.starts_with("table:") {
            for _ in 0..6 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Table, "table:".to_string(), loc));
        }
        if self.starts_with("where:") {
            for _ in 0..6 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Where, "where:".to_string(), loc));
        }
        if self.starts_with("with:") {
            for _ in 0..5 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::With, "with:".to_string(), loc));
        }

        // First character
        self.advance();

        // Subsequent characters
        while self.pos < self.len {
            if let Some(ch) = self.current_char() {
                if Self::is_ident_continue(ch) {
                    self.advance();
                } else if ch == '-' {
                    // Check if hyphen is followed by valid identifier char
                    let mut lookahead = 1;
                    while self.peek_char(lookahead) == Some('-') {
                        lookahead += 1;
                    }
                    if let Some(next_ch) = self.peek_char(lookahead) {
                        if Self::is_ident_continue(next_ch) {
                            // Consume all hyphens and continue
                            for _ in 0..lookahead {
                                self.advance();
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let value: String = self.input[start_pos..self.pos].iter().collect();
        let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);

        // Check for keywords (checking multi-word keywords first)
        if value == "else" && self.pos < self.len {
            // Check for "else if"
            let save_pos = self.pos;
            let save_line = self.line;
            let save_col = self.col;

            // Skip whitespace
            while matches!(self.current_char(), Some(ch) if Self::is_whitespace(ch)) {
                self.advance();
            }

            if self.starts_with("if") {
                // Check that it's not part of a longer identifier
                if !matches!(self.peek_char(2), Some(ch) if Self::is_ident_continue(ch) || ch == '-') {
                    self.advance();
                    self.advance();
                    let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                    self.paren_is_for_exp = false;
                    self.prior_whitespace = false;
                    return Some(Token::new(TokenType::ElseIf, "else if".to_string(), loc));
                }
            }

            // Restore position
            self.pos = save_pos;
            self.line = save_line;
            self.col = save_col;
        }

        let token_type = match value.as_str() {
            "and" => { self.paren_is_for_exp = true; TokenType::And },
            "as" => TokenType::As,
            "ascending" => TokenType::Ascending,
            "ask" => { self.paren_is_for_exp = true; TokenType::Ask },
            "by" => TokenType::By,
            "cases" => { self.paren_is_for_exp = true; TokenType::Cases },
            "check" => TokenType::Check,
            "data" => TokenType::Data,
            "descending" => TokenType::Descending,
            "do" => TokenType::Do,
            "does-not-raise" => { self.paren_is_for_exp = true; TokenType::DoesNotRaise },
            "else" => TokenType::Else,
            "end" => { self.paren_is_for_exp = false; TokenType::End },
            "examples" => { self.paren_is_for_exp = true; TokenType::Examples },
            "extend" => TokenType::TableExtend,
            "extract" => TokenType::TableExtract,
            "false" => TokenType::False,
            "for" => TokenType::For,
            "from" => TokenType::From,
            "fun" => TokenType::Fun,
            "hiding" => TokenType::Hiding,
            "if" => TokenType::If,
            "import" => TokenType::Import,
            "include" => TokenType::Include,
            "is" => { self.paren_is_for_exp = true; TokenType::Is },
            "is==" => { self.paren_is_for_exp = true; TokenType::IsEqualEqual },
            "is=~" => { self.paren_is_for_exp = true; TokenType::IsEqualTilde },
            "is-not" => { self.paren_is_for_exp = true; TokenType::IsNot },
            "is-not==" => { self.paren_is_for_exp = true; TokenType::IsNotEqualEqual },
            "is-not=~" => { self.paren_is_for_exp = true; TokenType::IsNotEqualTilde },
            "is-not<=>" => { self.paren_is_for_exp = true; TokenType::IsNotSpaceship },
            "is-roughly" => { self.paren_is_for_exp = true; TokenType::IsRoughly },
            "is-not-roughly" => { self.paren_is_for_exp = true; TokenType::IsNotRoughly },
            "is<=>" => { self.paren_is_for_exp = true; TokenType::IsSpaceship },
            "because" => { self.paren_is_for_exp = true; TokenType::Because },
            "lam" => TokenType::Lam,
            "lazy" => TokenType::Lazy,
            "let" => TokenType::Let,
            "letrec" => TokenType::Letrec,
            // "load-table" => TokenType::LoadTable, // Removed - should only match "load-table:" with colon
            "method" => TokenType::Method,
            "module" => TokenType::Module,
            "newtype" => TokenType::Newtype,
            "of" => TokenType::Of,
            "or" => { self.paren_is_for_exp = true; TokenType::Or },
            "provide" => TokenType::Provide,
            "provide-types" => TokenType::ProvideTypes,
            "raises" => { self.paren_is_for_exp = true; TokenType::Raises },
            "raises-other-than" => { self.paren_is_for_exp = true; TokenType::RaisesOtherThan },
            "raises-satisfies" => { self.paren_is_for_exp = true; TokenType::RaisesSatisfies },
            "raises-violates" => { self.paren_is_for_exp = true; TokenType::RaisesViolates },
            "reactor" => TokenType::Reactor,
            "rec" => TokenType::Rec,
            "ref" => TokenType::Ref,
            "sanitize" => TokenType::Sanitize,
            "satisfies" => { self.paren_is_for_exp = true; TokenType::Satisfies },
            "select" => TokenType::TableSelect,
            "shadow" => TokenType::Shadow,
            "sieve" => TokenType::TableFilter,
            "spy" => TokenType::Spy,
            "order" => TokenType::TableOrder,
            "transform" => TokenType::TableUpdate,
            "true" => TokenType::True,
            "type" => TokenType::Type,
            "type-let" => TokenType::TypeLet,
            "using" => TokenType::Using,
            "use" => TokenType::Use,
            "var" => TokenType::Var,
            "violates" => { self.paren_is_for_exp = true; TokenType::Violates },
            "when" => { self.paren_is_for_exp = true; TokenType::When },
            _ => {
                // For Name tokens, set paren_is_for_exp to false so that f(x) gets ParenNoSpace
                self.paren_is_for_exp = false;
                TokenType::Name
            }
        };

        self.prior_whitespace = false;
        Some(Token::new(token_type, value, loc))
    }

    fn tokenize_symbol(&mut self) -> Option<Token> {
        let start_line = self.line;
        let start_col = self.col;
        let start_pos = self.pos;

        let ch = self.current_char()?;

        // Check multi-character symbols first
        if self.starts_with("block:") {
            for _ in 0..6 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Block, "block:".to_string(), loc));
        }
        if self.starts_with("check:") {
            for _ in 0..6 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::CheckColon, "check:".to_string(), loc));
        }
        if self.starts_with("doc:") {
            for _ in 0..4 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Doc, "doc:".to_string(), loc));
        }
        if self.starts_with("else:") {
            for _ in 0..5 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ElseColon, "else:".to_string(), loc));
        }
        if self.starts_with("examples:") {
            for _ in 0..9 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ExamplesColon, "examples:".to_string(), loc));
        }
        if self.starts_with("otherwise:") {
            for _ in 0..10 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::OtherwiseColon, "otherwise:".to_string(), loc));
        }
        if self.starts_with("provide:") {
            for _ in 0..8 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ProvideColon, "provide:".to_string(), loc));
        }
        if self.starts_with("row:") {
            for _ in 0..4 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Row, "row:".to_string(), loc));
        }
        if self.starts_with("sharing:") {
            for _ in 0..8 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Sharing, "sharing:".to_string(), loc));
        }
        if self.starts_with("source:") {
            for _ in 0..7 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::SourceColon, "source:".to_string(), loc));
        }
        if self.starts_with("load-table:") {
            for _ in 0..11 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            self.paren_is_for_exp = true;
            return Some(Token::new(TokenType::LoadTable, "load-table:".to_string(), loc));
        }
        if self.starts_with("table:") {
            for _ in 0..6 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Table, "table:".to_string(), loc));
        }
        if self.starts_with("then:") {
            for _ in 0..5 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ThenColon, "then:".to_string(), loc));
        }
        if self.starts_with("where:") {
            for _ in 0..6 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Where, "where:".to_string(), loc));
        }
        if self.starts_with("with:") {
            for _ in 0..5 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::With, "with:".to_string(), loc));
        }
        if self.starts_with("...") {
            for _ in 0..3 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::DotDotDot, "...".to_string(), loc));
        }
        if self.starts_with("<=>") {
            for _ in 0..3 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Spaceship, "<=>".to_string(), loc));
        }
        if self.starts_with("->") {
            for _ in 0..2 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ThinArrow, "->".to_string(), loc));
        }
        if self.starts_with(":=") {
            for _ in 0..2 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ColonEquals, ":=".to_string(), loc));
        }
        if self.starts_with("::") {
            for _ in 0..2 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ColonColon, "::".to_string(), loc));
        }
        if self.starts_with("<=") {
            for _ in 0..2 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Leq, "<=".to_string(), loc));
        }
        if self.starts_with(">=") {
            for _ in 0..2 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Geq, ">=".to_string(), loc));
        }
        if self.starts_with("==") {
            for _ in 0..2 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::EqualEqual, "==".to_string(), loc));
        }
        if self.starts_with("=~") {
            for _ in 0..2 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::EqualTilde, "=~".to_string(), loc));
        }
        if self.starts_with("<>") {
            for _ in 0..2 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::Neq, "<>".to_string(), loc));
        }
        if self.starts_with("=>") {
            for _ in 0..2 { self.advance(); }
            let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
            self.paren_is_for_exp = true;
            self.prior_whitespace = false;
            return Some(Token::new(TokenType::ThickArrow, "=>".to_string(), loc));
        }

        // Single character symbols
        match ch {
            '[' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                let token_type = if self.prior_whitespace || self.paren_is_for_exp {
                    TokenType::BrackSpace
                } else {
                    TokenType::BrackNoSpace
                };
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(token_type, "[".to_string(), loc))
            }
            ']' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.prior_whitespace = false;
                Some(Token::new(TokenType::RBrack, "]".to_string(), loc))
            }
            '{' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::LBrace, "{".to_string(), loc))
            }
            '}' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.prior_whitespace = false;
                Some(Token::new(TokenType::RBrace, "}".to_string(), loc))
            }
            '(' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                let token_type = if self.prior_whitespace || self.paren_is_for_exp {
                    TokenType::ParenSpace
                } else {
                    TokenType::ParenNoSpace
                };
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(token_type, "(".to_string(), loc))
            }
            ')' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                // After ), paren should be treated like after a name/number (can be followed by function call)
                self.paren_is_for_exp = false;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::RParen, ")".to_string(), loc))
            }
            ';' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Semi, ";".to_string(), loc))
            }
            '\\' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Backslash, "\\".to_string(), loc))
            }
            '.' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Dot, ".".to_string(), loc))
            }
            '!' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Bang, "!".to_string(), loc))
            }
            '%' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Percent, "%".to_string(), loc))
            }
            ',' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Comma, ",".to_string(), loc))
            }
            ':' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Colon, ":".to_string(), loc))
            }
            '|' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Bar, "|".to_string(), loc))
            }
            '=' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Equals, "=".to_string(), loc))
            }
            '<' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                let token_type = if self.prior_whitespace || self.paren_is_for_exp {
                    TokenType::Lt
                } else {
                    TokenType::LtNoSpace
                };
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(token_type, "<".to_string(), loc))
            }
            '>' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = false; // After >, allow ParenNoSpace for function application
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Gt, ">".to_string(), loc))
            }
            '*' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Times, "*".to_string(), loc))
            }
            '^' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Caret, "^".to_string(), loc))
            }
            '+' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Plus, "+".to_string(), loc))
            }
            '-' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Dash, "-".to_string(), loc))
            }
            '/' => {
                self.advance();
                let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
                self.paren_is_for_exp = true;
                self.prior_whitespace = false;
                Some(Token::new(TokenType::Slash, "/".to_string(), loc))
            }
            _ => None,
        }
    }

    pub fn next_token(&mut self) -> Option<Token> {
        // Skip whitespace
        if let Some(_ws_token) = self.skip_whitespace() {
            // Whitespace is typically ignored, but we track it for prior_whitespace
        }

        if self.pos >= self.len {
            return Some(Token::new(
                TokenType::Eof,
                String::new(),
                SrcLoc::new(self.line, self.col, self.pos, self.line, self.col, self.pos),
            ));
        }

        // Try block comment
        if let Some(token) = self.tokenize_block_comment() {
            return Some(token);
        }

        // Try line comment
        if let Some(token) = self.tokenize_line_comment() {
            return Some(token);
        }

        // Try string
        if let Some(token) = self.tokenize_string() {
            return Some(token);
        }

        // Try number
        if let Some(token) = self.tokenize_number() {
            return Some(token);
        }

        // Try name or keyword
        if let Some(token) = self.tokenize_name_or_keyword() {
            return Some(token);
        }

        // Try symbol
        if let Some(token) = self.tokenize_symbol() {
            return Some(token);
        }

        // Unknown character - advance and return error
        let start_line = self.line;
        let start_col = self.col;
        let start_pos = self.pos;
        let ch = self.advance()?;
        let loc = SrcLoc::new(start_line, start_col, start_pos, self.line, self.col, self.pos);
        Some(Token::new(
            TokenType::BadOper,
            ch.to_string(),
            loc,
        ))
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        while let Some(token) = self.next_token() {
            let is_eof = token.token_type == TokenType::Eof;
            // Skip whitespace and comment tokens
            if token.token_type != TokenType::Ws
                && token.token_type != TokenType::Comment
                && token.token_type != TokenType::BlockComment {
                tokens.push(token);
            }
            if is_eof {
                break;
            }
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_simple_keywords() {
        let mut tokenizer = Tokenizer::new("fun if else end");
        let tokens = tokenizer.tokenize();

        assert_eq!(tokens.len(), 5); // 4 keywords + EOF
        assert_eq!(tokens[0].token_type, TokenType::Fun);
        assert_eq!(tokens[1].token_type, TokenType::If);
        assert_eq!(tokens[2].token_type, TokenType::Else);
        assert_eq!(tokens[3].token_type, TokenType::End);
        assert_eq!(tokens[4].token_type, TokenType::Eof);
    }

    #[test]
    fn test_tokenize_numbers() {
        let mut tokenizer = Tokenizer::new("42 3.14 1/2 ~3.14");
        let tokens = tokenizer.tokenize();

        assert_eq!(tokens[0].token_type, TokenType::Number);
        assert_eq!(tokens[0].value, "42");
        assert_eq!(tokens[1].token_type, TokenType::Number);
        assert_eq!(tokens[1].value, "3.14");
        assert_eq!(tokens[2].token_type, TokenType::Rational);
        assert_eq!(tokens[2].value, "1/2");
        assert_eq!(tokens[3].token_type, TokenType::RoughNumber);
        assert_eq!(tokens[3].value, "~3.14");
    }

    #[test]
    fn test_tokenize_strings() {
        let mut tokenizer = Tokenizer::new(r#""hello" 'world'"#);
        let tokens = tokenizer.tokenize();

        assert_eq!(tokens[0].token_type, TokenType::String);
        assert!(tokens[0].value.contains("hello"));
        assert_eq!(tokens[1].token_type, TokenType::String);
        assert!(tokens[1].value.contains("world"));
    }

    #[test]
    fn test_tokenize_identifiers() {
        let mut tokenizer = Tokenizer::new("foo bar-baz my-long-name");
        let tokens = tokenizer.tokenize();

        assert_eq!(tokens[0].token_type, TokenType::Name);
        assert_eq!(tokens[0].value, "foo");
        assert_eq!(tokens[1].token_type, TokenType::Name);
        assert_eq!(tokens[1].value, "bar-baz");
        assert_eq!(tokens[2].token_type, TokenType::Name);
        assert_eq!(tokens[2].value, "my-long-name");
    }

    #[test]
    fn test_tokenize_symbols() {
        let mut tokenizer = Tokenizer::new("( ) [ ] { } , ;");
        let tokens = tokenizer.tokenize();

        assert!(matches!(tokens[0].token_type, TokenType::ParenSpace | TokenType::ParenNoSpace));
        assert_eq!(tokens[1].token_type, TokenType::RParen);
        assert!(matches!(tokens[2].token_type, TokenType::BrackSpace | TokenType::BrackNoSpace));
        assert_eq!(tokens[3].token_type, TokenType::RBrack);
        assert_eq!(tokens[4].token_type, TokenType::LBrace);
        assert_eq!(tokens[5].token_type, TokenType::RBrace);
        assert_eq!(tokens[6].token_type, TokenType::Comma);
        assert_eq!(tokens[7].token_type, TokenType::Semi);
    }

    #[test]
    fn test_tokenize_comments() {
        let mut tokenizer = Tokenizer::new("# line comment\nfun");
        let tokens = tokenizer.tokenize();

        // Comments are skipped
        assert_eq!(tokens[0].token_type, TokenType::Fun);
    }

    #[test]
    fn test_tokenize_block_comment() {
        let mut tokenizer = Tokenizer::new("#| block comment |# fun");
        let tokens = tokenizer.tokenize();

        // Block comments are skipped
        assert_eq!(tokens[0].token_type, TokenType::Fun);
    }

    #[test]
    fn test_simple_function() {
        let code = r#"
fun add(x, y):
  x + y
end
"#;
        let mut tokenizer = Tokenizer::new(code);
        let tokens = tokenizer.tokenize();

        assert!(tokens.iter().any(|t| t.token_type == TokenType::Fun));
        assert!(tokens.iter().any(|t| t.token_type == TokenType::Name && t.value == "add"));
        assert!(tokens.iter().any(|t| t.token_type == TokenType::End));
    }
}
