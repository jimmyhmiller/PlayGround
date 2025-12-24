use thiserror::Error;

use crate::namespace::NamespaceScope;
use crate::token::{Token, TokenType};
use crate::value::{Symbol, Value};

#[derive(Debug, Error)]
pub enum ReaderError {
    #[error("unexpected token {0:?} at line {1}, column {2}")]
    UnexpectedToken(TokenType, usize, usize),

    #[error("unmatched parenthesis at line {0}, column {1}")]
    UnmatchedParen(usize, usize),

    #[error("unmatched bracket at line {0}, column {1}")]
    UnmatchedBracket(usize, usize),

    #[error("unmatched brace at line {0}, column {1}")]
    UnmatchedBrace(usize, usize),

    #[error("map key must be a keyword at line {0}, column {1}")]
    MapKeyMustBeKeyword(usize, usize),

    #[error("map missing value at line {0}, column {1}")]
    MapMissingValue(usize, usize),

    #[error("invalid number '{0}' at line {1}, column {2}")]
    InvalidNumber(String, usize, usize),

    #[error("unknown dialect '{0}' at line {1}, column {2}. Did you forget to (require-dialect {0})?")]
    UnknownDialect(String, usize, usize),

    #[error("require path must be a string at line {0}, column {1}")]
    RequirePathMustBeString(usize, usize),

    #[error("require form missing :as alias at line {0}, column {1}")]
    RequireMissingAlias(usize, usize),
}

pub struct Reader<'a> {
    tokens: &'a [Token],
    current: usize,
    namespace_scope: NamespaceScope,
}

impl<'a> Reader<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            current: 0,
            namespace_scope: NamespaceScope::new(),
        }
    }

    pub fn read(&mut self) -> Result<Vec<Value>, ReaderError> {
        let mut values = Vec::new();

        while !self.is_at_end() {
            if let Some(val) = self.read_value()? {
                values.push(val);
            }
        }

        Ok(values)
    }

    fn read_value(&mut self) -> Result<Option<Value>, ReaderError> {
        let token = self.advance();

        match token.token_type {
            TokenType::LeftParen => self.read_list().map(Some),
            TokenType::LeftBracket => self.read_vector().map(Some),
            TokenType::LeftBrace => self.read_map().map(Some),
            TokenType::Symbol => self.read_symbol(&token).map(Some),
            TokenType::String => self.read_string(&token).map(Some),
            TokenType::Number => self.read_number(&token).map(Some),
            TokenType::Keyword => Ok(Some(Value::keyword(token.lexeme.clone()))),
            TokenType::BlockLabel => self.read_block_label(&token).map(Some),
            TokenType::Colon => {
                // Standalone ":" is used for type annotations and typed block args.
                Ok(Some(Value::symbol(":")))
            }
            TokenType::Backtick => {
                // Quasiquote: `form -> (quasiquote form)
                let form = self.read_value()?.ok_or_else(|| {
                    ReaderError::UnexpectedToken(TokenType::Eof, token.line, token.column)
                })?;
                Ok(Some(Value::List(vec![Value::symbol("quasiquote"), form])))
            }
            TokenType::Tilde => {
                // Unquote: ~form -> (unquote form)
                let form = self.read_value()?.ok_or_else(|| {
                    ReaderError::UnexpectedToken(TokenType::Eof, token.line, token.column)
                })?;
                Ok(Some(Value::List(vec![Value::symbol("unquote"), form])))
            }
            TokenType::TildeAt => {
                // Unquote-splice: ~@form -> (unquote-splice form)
                let form = self.read_value()?.ok_or_else(|| {
                    ReaderError::UnexpectedToken(TokenType::Eof, token.line, token.column)
                })?;
                Ok(Some(Value::List(vec![
                    Value::symbol("unquote-splice"),
                    form,
                ])))
            }
            TokenType::Eof => Ok(None),
            TokenType::RightParen | TokenType::RightBracket | TokenType::RightBrace => {
                Err(ReaderError::UnexpectedToken(
                    token.token_type,
                    token.line,
                    token.column,
                ))
            }
        }
    }

    fn read_list(&mut self) -> Result<Value, ReaderError> {
        let start_token = &self.tokens[self.current.saturating_sub(1)];
        let start_line = start_token.line;
        let start_column = start_token.column;

        let mut list = Value::list();

        while !self.check(TokenType::RightParen) && !self.is_at_end() {
            if let Some(val) = self.read_value()? {
                list.list_append(val);
            }
        }

        if !self.check(TokenType::RightParen) {
            return Err(ReaderError::UnmatchedParen(start_line, start_column));
        }
        self.advance(); // consume )

        // Check if this is a require-dialect, use-dialect, or require form
        if let Value::List(ref items) = list {
            if !items.is_empty() {
                if let Value::Symbol(ref sym) = items[0] {
                    if sym.name == "require-dialect" {
                        self.handle_require_dialect(&list);
                    } else if sym.name == "use-dialect" {
                        self.handle_use_dialect(&list);
                    } else if sym.name == "require" {
                        self.handle_require(&list);
                    }
                }
            }
        }

        Ok(list)
    }

    fn read_vector(&mut self) -> Result<Value, ReaderError> {
        let start_token = &self.tokens[self.current.saturating_sub(1)];
        let start_line = start_token.line;
        let start_column = start_token.column;

        let mut vector = Value::vector();

        while !self.check(TokenType::RightBracket) && !self.is_at_end() {
            if let Some(val) = self.read_value()? {
                vector.vector_append(val);
            }
        }

        if !self.check(TokenType::RightBracket) {
            return Err(ReaderError::UnmatchedBracket(start_line, start_column));
        }
        self.advance(); // consume ]

        Ok(vector)
    }

    fn read_map(&mut self) -> Result<Value, ReaderError> {
        let start_token = &self.tokens[self.current.saturating_sub(1)];
        let start_line = start_token.line;
        let start_column = start_token.column;

        let mut map = Value::map();

        while !self.check(TokenType::RightBrace) && !self.is_at_end() {
            // Read key (must be keyword)
            let key_line = self.peek().line;
            let key_column = self.peek().column;
            let key = match self.read_value()? {
                Some(val) => val,
                None => break,
            };

            if !matches!(key, Value::Keyword(_)) {
                return Err(ReaderError::MapKeyMustBeKeyword(key_line, key_column));
            }

            // Skip the leading : in keyword
            let key_str = match &key {
                Value::Keyword(k) => {
                    if k.starts_with(':') {
                        &k[1..]
                    } else {
                        k.as_str()
                    }
                }
                _ => unreachable!(),
            };

            // Read value
            if self.check(TokenType::RightBrace) {
                return Err(ReaderError::MapMissingValue(start_line, start_column));
            }
            let val = match self.read_value()? {
                Some(v) => v,
                None => {
                    return Err(ReaderError::MapMissingValue(start_line, start_column));
                }
            };

            map.map_put(key_str, val);
        }

        if !self.check(TokenType::RightBrace) {
            return Err(ReaderError::UnmatchedBrace(start_line, start_column));
        }
        self.advance(); // consume }

        Ok(map)
    }

    fn read_symbol(&mut self, token: &Token) -> Result<Value, ReaderError> {
        // Check for special boolean values
        if token.lexeme == "true" {
            return Ok(Value::boolean(true));
        }
        if token.lexeme == "false" {
            return Ok(Value::boolean(false));
        }
        if token.lexeme == "nil" {
            return Ok(Value::nil());
        }

        // Skip namespace resolution for MLIR dialect types (start with !)
        // These are passed through as-is to the MLIR type parser
        if token.lexeme.starts_with('!') {
            let symbol = Symbol::new(&token.lexeme);
            return Ok(Value::Symbol(symbol));
        }

        // Resolve namespace for the symbol
        let namespace = self.namespace_scope.resolve_symbol(&token.lexeme)
            .map_err(|unknown_dialect| {
                ReaderError::UnknownDialect(unknown_dialect, token.line, token.column)
            })?;
        let unqualified_name = self.namespace_scope.get_unqualified_name(&token.lexeme);

        let mut symbol = Symbol::with_namespace(unqualified_name, namespace);

        // Mark how this symbol was qualified
        if token.lexeme.contains('/') {
            symbol.uses_alias = true;
        } else if token.lexeme.contains('.') {
            symbol.uses_dot = true;
        }

        Ok(Value::Symbol(symbol))
    }

    fn read_string(&mut self, token: &Token) -> Result<Value, ReaderError> {
        // Remove quotes and handle escape sequences
        let content = &token.lexeme[1..token.lexeme.len() - 1];
        let mut result = String::new();

        let mut chars = content.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(&escaped) = chars.peek() {
                    chars.next();
                    let replacement = match escaped {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '\\' => '\\',
                        '"' => '"',
                        '0' => '\0',
                        other => other,
                    };
                    result.push(replacement);
                }
            } else {
                result.push(c);
            }
        }

        Ok(Value::string(result))
    }

    fn read_number(&mut self, token: &Token) -> Result<Value, ReaderError> {
        let num: f64 = token.lexeme.parse().map_err(|_| {
            ReaderError::InvalidNumber(token.lexeme.clone(), token.line, token.column)
        })?;
        Ok(Value::number(num))
    }

    fn read_block_label(&mut self, token: &Token) -> Result<Value, ReaderError> {
        // Block labels are represented as symbols with the ^ prefix
        Ok(Value::symbol(token.lexeme.clone()))
    }

    fn handle_require_dialect(&mut self, list: &Value) {
        // (require-dialect arith) or (require-dialect [arith :as a])
        if let Value::List(items) = list {
            for item in items.iter().skip(1) {
                match item {
                    Value::Symbol(sym) => {
                        // Simple require: (require-dialect arith)
                        self.namespace_scope.require_dialect(&sym.name, None);
                    }
                    Value::Vector(vec) => {
                        // Aliased require: (require-dialect [arith :as a])
                        if vec.len() >= 3 {
                            if let (Value::Symbol(name), Value::Keyword(kw), Value::Symbol(alias)) =
                                (&vec[0], &vec[1], &vec[2])
                            {
                                if kw == ":as" {
                                    self.namespace_scope
                                        .require_dialect(&name.name, Some(&alias.name));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn handle_use_dialect(&mut self, list: &Value) {
        // (use-dialect arith)
        if let Value::List(items) = list {
            for item in items.iter().skip(1) {
                if let Value::Symbol(sym) = item {
                    self.namespace_scope.use_dialect(&sym.name);
                }
            }
        }
    }

    fn handle_require(&mut self, list: &Value) {
        // (require ["./file.lisp" :as m])
        if let Value::List(items) = list {
            for item in items.iter().skip(1) {
                if let Value::Vector(vec) = item {
                    // Require must have format: ["path" :as alias]
                    if vec.len() >= 3 {
                        if let (Value::String(path), Value::Keyword(kw), Value::Symbol(alias)) =
                            (&vec[0], &vec[1], &vec[2])
                        {
                            if kw == ":as" {
                                self.namespace_scope.require_file(path, &alias.name);
                            }
                        }
                    }
                }
            }
        }
    }

    fn advance(&mut self) -> Token {
        if !self.is_at_end() {
            let token = self.tokens[self.current].clone();
            self.current += 1;
            token
        } else {
            self.tokens[self.tokens.len() - 1].clone() // Return EOF
        }
    }

    fn peek(&self) -> &Token {
        if self.is_at_end() {
            &self.tokens[self.tokens.len() - 1]
        } else {
            &self.tokens[self.current]
        }
    }

    fn check(&self, token_type: TokenType) -> bool {
        if self.is_at_end() {
            return false;
        }
        self.tokens[self.current].token_type == token_type
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len() || self.tokens[self.current].token_type == TokenType::Eof
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;
    use pretty_assertions::assert_eq;

    fn read_str(input: &str) -> Vec<Value> {
        let mut tokenizer = Tokenizer::new(input);
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        reader.read().unwrap()
    }

    #[test]
    fn test_basic_list() {
        let values = read_str("(1 2 3)");

        assert_eq!(values.len(), 1);
        assert!(values[0].is_list());
        assert_eq!(values[0].as_list().len(), 3);
    }

    #[test]
    fn test_nested_structures() {
        let values = read_str("([1 2] {:key 42})");

        assert_eq!(values.len(), 1);
        let list = &values[0];
        assert!(list.is_list());
        assert_eq!(list.as_list().len(), 2);
        assert!(list.as_list()[0].is_vector());
        assert!(list.as_list()[1].is_map());
    }

    #[test]
    fn test_symbols_with_namespaces() {
        let values = read_str("(require-dialect [arith :as a]) (a/addi 1 2)");

        assert_eq!(values.len(), 2);

        // Second value should be (a/addi 1 2)
        let list = &values[1];
        assert!(list.is_list());

        // First item should be symbol with namespace
        let sym = &list.as_list()[0];
        assert!(sym.is_symbol());
        let symbol = sym.as_symbol();
        assert_eq!(symbol.name, "addi");
        assert!(symbol.namespace.is_some());
        assert!(symbol.uses_alias);
    }

    #[test]
    fn test_strings_with_escapes() {
        let values = read_str("\"hello\\nworld\"");

        assert_eq!(values.len(), 1);
        assert_eq!(values[0].as_string(), "hello\nworld");
    }

    #[test]
    fn test_booleans_and_nil() {
        let values = read_str("true false nil");

        assert_eq!(values.len(), 3);
        assert_eq!(values[0], Value::Boolean(true));
        assert_eq!(values[1], Value::Boolean(false));
        assert_eq!(values[2], Value::Nil);
    }

    #[test]
    fn test_rejects_maps_with_non_keyword_keys() {
        let mut tokenizer = Tokenizer::new("{foo 1}");
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let result = reader.read();

        assert!(matches!(result, Err(ReaderError::MapKeyMustBeKeyword(_, _))));
    }

    #[test]
    fn test_reports_missing_map_values() {
        let mut tokenizer = Tokenizer::new("{:foo}");
        let tokens = tokenizer.tokenize().unwrap();
        let mut reader = Reader::new(&tokens);
        let result = reader.read();

        assert!(matches!(result, Err(ReaderError::MapMissingValue(_, _))));
    }

    #[test]
    fn test_dot_notation_namespaces() {
        let values = read_str("(require-dialect arith) (arith.addi 1 2)");

        assert_eq!(values.len(), 2);
        let list = &values[1];
        let sym = &list.as_list()[0];
        let symbol = sym.as_symbol();
        assert_eq!(symbol.name, "addi");
        assert!(symbol.namespace.is_some());
        assert_eq!(symbol.namespace.as_ref().unwrap().name, "arith");
        assert!(symbol.uses_dot);
    }

    #[test]
    fn test_block_labels() {
        let values = read_str("^bb1 ^loop");

        assert_eq!(values.len(), 2);
        assert!(values[0].is_symbol());
        assert_eq!(values[0].as_symbol().name, "^bb1");
        assert_eq!(values[1].as_symbol().name, "^loop");
    }

    #[test]
    fn test_keywords() {
        let values = read_str(":foo :bar-baz");

        assert_eq!(values.len(), 2);
        assert_eq!(values[0].as_keyword(), ":foo");
        assert_eq!(values[1].as_keyword(), ":bar-baz");
    }

    #[test]
    fn test_numbers() {
        let values = read_str("42 3.14 -10 1.5e-3");

        assert_eq!(values.len(), 4);
        assert_eq!(values[0].as_number(), 42.0);
        assert_eq!(values[1].as_number(), 3.14);
        assert_eq!(values[2].as_number(), -10.0);
        assert_eq!(values[3].as_number(), 0.0015);
    }
}
