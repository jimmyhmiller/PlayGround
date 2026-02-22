use std::collections::HashMap;
use crate::term::*;
use crate::pattern::*;

// ── Lexer ──

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Num(i64),
    Float(f64),
    Ident(String),
    Var(String),
    Scope(String),
    Keyword(String),
    KwFn,
    KwRule,
    KwIf,
    KwThen,
    KwElse,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    HashBrace,  // #{
    Comma,
    Colon,
    Str(String),
    Arrow,
    FatArrow,
    Eq,
    EqEq,
    BangEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Plus,
    PlusPlus,
    Minus,
    Star,
    Slash,
    Percent,
    Pipe,      // |
    Ellipsis,  // ...
    Eof,
}

pub struct Lexer {
    chars: Vec<char>,
    pos: usize,
}

impl Lexer {
    pub fn new(src: &str) -> Self {
        Lexer { chars: src.chars().collect(), pos: 0 }
    }

    fn peek(&self) -> char {
        self.chars.get(self.pos).copied().unwrap_or('\0')
    }

    fn advance(&mut self) -> char {
        let c = self.peek();
        self.pos += 1;
        c
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while self.peek().is_whitespace() { self.advance(); }
            if self.peek() == '#' {
                // #{ is a HashBrace token, not a comment
                if self.chars.get(self.pos + 1).copied() == Some('{') {
                    break;
                }
                while self.peek() != '\n' && self.peek() != '\0' { self.advance(); }
            } else {
                break;
            }
        }
    }

    fn read_ident(&mut self) -> String {
        let mut s = String::new();
        while self.peek().is_alphanumeric() || self.peek() == '_' {
            s.push(self.advance());
        }
        s
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();
        let c = self.peek();
        if c == '\0' { return Token::Eof; }

        if c.is_ascii_digit() {
            let mut n: i64 = 0;
            while self.peek().is_ascii_digit() {
                n = n * 10 + (self.advance() as i64 - '0' as i64);
            }
            // Check for float: digit followed by '.' followed by digit
            if self.peek() == '.' && self.chars.get(self.pos + 1).map_or(false, |c| c.is_ascii_digit()) {
                self.advance(); // consume '.'
                let mut frac: f64 = 0.0;
                let mut div: f64 = 1.0;
                while self.peek().is_ascii_digit() {
                    frac = frac * 10.0 + (self.advance() as u8 - b'0') as f64;
                    div *= 10.0;
                }
                return Token::Float(n as f64 + frac / div);
            }
            return Token::Num(n);
        }

        if c.is_alphabetic() || c == '_' {
            let s = self.read_ident();
            return match s.as_str() {
                "fn" => Token::KwFn,
                "rule" => Token::KwRule,
                "if" => Token::KwIf,
                "then" => Token::KwThen,
                "else" => Token::KwElse,
                _ => Token::Ident(s),
            };
        }

        if c == '"' {
            self.advance(); // opening quote
            let mut s = String::new();
            while self.peek() != '"' && self.peek() != '\0' {
                if self.peek() == '\\' {
                    self.advance();
                    match self.advance() {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        '\\' => s.push('\\'),
                        '"' => s.push('"'),
                        c => { s.push('\\'); s.push(c); }
                    }
                } else {
                    s.push(self.advance());
                }
            }
            self.advance(); // closing quote
            return Token::Str(s);
        }

        if c == '?' {
            self.advance();
            let name = self.read_ident();
            return Token::Var(name);
        }

        if c == '@' {
            self.advance();
            let name = self.read_ident();
            return Token::Scope(name);
        }

        if c == ':' && self.chars.get(self.pos + 1).map_or(false, |c| c.is_alphabetic()) {
            self.advance();
            let name = self.read_ident();
            return Token::Keyword(name);
        }

        self.advance();
        match c {
            '(' => Token::LParen,
            ')' => Token::RParen,
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            '#' => {
                if self.peek() == '{' { self.advance(); Token::HashBrace }
                else { panic!("Unexpected '#' — expected '#{{' for set literal") }
            }
            '|' => Token::Pipe,
            ',' => Token::Comma,
            ':' => Token::Colon,
            '+' => {
                if self.peek() == '+' { self.advance(); Token::PlusPlus }
                else { Token::Plus }
            }
            '*' => Token::Star,
            '/' => Token::Slash,
            '%' => Token::Percent,
            '<' => {
                if self.peek() == '=' { self.advance(); Token::LtEq }
                else { Token::Lt }
            }
            '>' => {
                if self.peek() == '=' { self.advance(); Token::GtEq }
                else { Token::Gt }
            }
            '-' => {
                if self.peek() == '>' { self.advance(); Token::Arrow }
                else { Token::Minus }
            }
            '!' => {
                if self.peek() == '=' { self.advance(); Token::BangEq }
                else { panic!("Unexpected character: '!'") }
            }
            '=' => {
                if self.peek() == '>' { self.advance(); Token::FatArrow }
                else if self.peek() == '=' { self.advance(); Token::EqEq }
                else { Token::Eq }
            }
            '.' => {
                if self.peek() == '.' {
                    self.advance();
                    if self.peek() == '.' {
                        self.advance();
                        Token::Ellipsis
                    } else {
                        panic!("Expected '...' (ellipsis)")
                    }
                } else {
                    panic!("Unexpected '.'")
                }
            }
            _ => panic!("Unexpected character: '{}'", c),
        }
    }

    pub fn tokenize(mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token();
            let done = tok == Token::Eof;
            tokens.push(tok);
            if done { break; }
        }
        tokens
    }
}

// ── Parser ──

pub struct Program {
    pub rules: Vec<Rule>,
    pub meta_rules: Vec<(Rule, String)>, // (rule, target_scope_name)
    pub emit_scopes: Vec<String>,        // scope names used via @scope in expressions
    pub expr: Pattern,
}

pub struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    store: &'a mut TermStore,
    clause_vars: HashMap<String, VarId>,
    next_var: u32,
    emit_scopes: Vec<String>,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: Vec<Token>, store: &'a mut TermStore) -> Self {
        Parser { tokens, pos: 0, store, clause_vars: HashMap::new(), next_var: 0, emit_scopes: Vec::new() }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: Token) {
        let tok = self.advance();
        if tok != expected {
            panic!("Expected {:?}, got {:?}", expected, tok);
        }
    }

    fn reset_vars(&mut self) {
        self.clause_vars.clear();
        self.next_var = 0;
    }

    fn get_var(&mut self, name: &str) -> VarId {
        if let Some(&id) = self.clause_vars.get(name) {
            return id;
        }
        let id = VarId(self.next_var);
        self.next_var += 1;
        assert!((self.next_var as usize) <= MAX_VARS, "Too many variables in clause (max {})", MAX_VARS);
        self.clause_vars.insert(name.to_string(), id);
        id
    }

    pub fn parse_program(&mut self) -> Program {
        let mut fn_clauses: HashMap<String, Vec<Clause>> = HashMap::new();
        let mut fn_order: Vec<String> = Vec::new();
        let mut meta_rules: Vec<(Rule, String)> = Vec::new();
        let mut main_rules: Vec<Rule> = Vec::new();
        let mut last_expr: Option<Pattern> = None;

        loop {
            match self.peek() {
                Token::KwFn => {
                    let (name, clause) = self.parse_fn_decl();
                    if !fn_clauses.contains_key(&name) {
                        fn_order.push(name.clone());
                    }
                    fn_clauses.entry(name).or_default().push(clause);
                }
                Token::KwRule => {
                    let (rule, is_meta, target_scope) = self.parse_rule_decl();
                    if is_meta {
                        meta_rules.push((rule, target_scope));
                    } else {
                        main_rules.push(rule);
                    }
                }
                Token::Eof => break,
                _ => {
                    self.reset_vars();
                    let expr = self.parse_expr();
                    last_expr = Some(expr);
                }
            }
        }

        let mut rules: Vec<Rule> = Vec::new();
        for name in fn_order {
            rules.push(Rule {
                name: name.clone(),
                clauses: fn_clauses.remove(&name).unwrap(),
            });
        }
        rules.extend(main_rules);

        let emit_scopes = std::mem::take(&mut self.emit_scopes);
        Program {
            rules,
            meta_rules,
            emit_scopes,
            expr: last_expr.expect("No expression to evaluate"),
        }
    }

    fn parse_fn_decl(&mut self) -> (String, Clause) {
        self.advance(); // 'fn'
        let name = match self.advance() {
            Token::Ident(s) => s,
            t => panic!("Expected function name, got {:?}", t),
        };

        self.reset_vars();

        self.expect(Token::LParen);
        let args = self.parse_expr_list_until(Token::RParen);
        self.expect(Token::RParen);

        // Parse optional where-guards: fn foo(?x, ?y) where ?x > 0, ?y > 0 = ...
        let guards = if matches!(self.peek(), Token::Ident(s) if s == "where") {
            self.advance(); // consume 'where'
            let mut guards = vec![self.parse_expr()];
            while *self.peek() == Token::Comma {
                self.advance();
                guards.push(self.parse_expr());
            }
            guards
        } else {
            vec![]
        };

        self.expect(Token::Eq);

        let name_sym = self.store.sym(&name);
        let lhs = Pattern::Call(Box::new(Pattern::Sym(name_sym)), args);
        let rhs = self.parse_expr();

        (name, Clause { lhs, rhs, guards })
    }

    fn parse_rule_decl(&mut self) -> (Rule, bool, String) {
        self.advance(); // 'rule'
        let name = match self.advance() {
            Token::Ident(s) => s,
            t => panic!("Expected rule name, got {:?}", t),
        };

        let mut is_meta = false;
        let mut target_scope = String::new();
        if *self.peek() == Token::Colon {
            self.advance();
            let from_scopes = self.parse_scope_list();
            self.expect(Token::Arrow);
            target_scope = match self.advance() {
                Token::Scope(s) => s,
                t => panic!("Expected scope after ->, got {:?}", t),
            };
            is_meta = from_scopes.iter().any(|s| s == "meta");
        }

        self.expect(Token::LBrace);
        let mut clauses = Vec::new();
        while *self.peek() != Token::RBrace {
            self.reset_vars();
            let lhs = self.parse_expr();
            self.expect(Token::FatArrow);
            let rhs = self.parse_expr();
            clauses.push(Clause { lhs, rhs, guards: vec![] });
        }
        self.expect(Token::RBrace);

        (Rule { name, clauses }, is_meta, target_scope)
    }

    /// Parse `rule name: @from -> @to { lhs => rhs ... }` in expression context.
    /// Outer variables (already in clause_vars) become Pattern::Var for substitution.
    /// Inner variables (new to the nested rule) become __pvar(N) template markers.
    fn parse_rule_decl_expr(&mut self) -> Pattern {
        self.advance(); // 'rule'
        let name = match self.advance() {
            Token::Ident(s) => s,
            t => panic!("Expected rule name, got {:?}", t),
        };

        self.expect(Token::Colon);
        let from_scopes = self.parse_scope_list();
        self.expect(Token::Arrow);
        let to_scope = match self.advance() {
            Token::Scope(s) => s,
            t => panic!("Expected scope after ->, got {:?}", t),
        };

        // Save outer var context
        let outer_vars = self.clause_vars.clone();
        let outer_next_var = self.next_var;

        let pvar_sym = self.store.sym("__pvar");
        let clause_sym = self.store.sym("__clause");
        let decl_sym = self.store.sym("__rule_decl");

        let name_sym = self.store.sym(&name);
        let from_str = if from_scopes.iter().any(|s| s == "meta") { "meta" } else { &from_scopes[0] };
        let from_sym = self.store.sym(from_str);
        let to_sym = self.store.sym(&to_scope);

        self.expect(Token::LBrace);
        let mut clause_args: Vec<Pattern> = Vec::new();

        while *self.peek() != Token::RBrace {
            // Reset to outer vars for each clause (inner vars are fresh per clause)
            self.clause_vars = outer_vars.clone();
            self.next_var = outer_next_var;

            let lhs = self.parse_expr();
            self.expect(Token::FatArrow);
            let rhs = self.parse_expr();

            // Convert inner vars (id >= outer_next_var) to __pvar(N) patterns
            let lhs = self.reify_inner_vars(&lhs, outer_next_var, pvar_sym);
            let rhs = self.reify_inner_vars(&rhs, outer_next_var, pvar_sym);

            clause_args.push(Pattern::Call(Box::new(Pattern::Sym(clause_sym)), vec![lhs, rhs]));
        }
        self.expect(Token::RBrace);

        // Restore outer vars
        self.clause_vars = outer_vars;
        self.next_var = outer_next_var;

        let mut args = vec![
            Pattern::Sym(name_sym),
            Pattern::Sym(from_sym),
            Pattern::Sym(to_sym),
        ];
        args.extend(clause_args);

        Pattern::Call(Box::new(Pattern::Sym(decl_sym)), args)
    }

    /// Replace inner pattern variables (id >= threshold) with __pvar(N) call patterns.
    /// Outer vars (id < threshold) stay as Pattern::Var for substitution by the outer rule.
    fn reify_inner_vars(&self, pat: &Pattern, outer_next_var: u32, pvar_sym: SymId) -> Pattern {
        match pat {
            Pattern::Var(v) if v.0 >= outer_next_var => {
                let inner_id = v.0 - outer_next_var;
                Pattern::Call(Box::new(Pattern::Sym(pvar_sym)), vec![Pattern::Num(inner_id as i64)])
            }
            Pattern::Wildcard | Pattern::WildSpread => pat.clone(),
            Pattern::Var(_) => pat.clone(),
            Pattern::Spread(v) if v.0 >= outer_next_var => {
                let inner_id = v.0 - outer_next_var;
                // Reify spread var as __pvar(N) wrapped in a Spread-like marker
                // For now, spreads in nested rules are uncommon; reify as Call pattern
                Pattern::Call(Box::new(Pattern::Sym(pvar_sym)), vec![Pattern::Num(inner_id as i64)])
            }
            Pattern::Spread(_) => pat.clone(),
            Pattern::Call(head, args) => {
                Pattern::Call(
                    Box::new(self.reify_inner_vars(head, outer_next_var, pvar_sym)),
                    args.iter().map(|a| self.reify_inner_vars(a, outer_next_var, pvar_sym)).collect(),
                )
            }
            Pattern::Map(entries) => {
                Pattern::Map(
                    entries.iter().map(|(k, v)| {
                        (self.reify_inner_vars(k, outer_next_var, pvar_sym),
                         self.reify_inner_vars(v, outer_next_var, pvar_sym))
                    }).collect(),
                )
            }
            Pattern::Set(elems) => {
                Pattern::Set(
                    elems.iter().map(|e| self.reify_inner_vars(e, outer_next_var, pvar_sym)).collect(),
                )
            }
            _ => pat.clone(),
        }
    }

    fn parse_scope_list(&mut self) -> Vec<String> {
        let mut scopes = Vec::new();
        match self.advance() {
            Token::Scope(s) => scopes.push(s),
            t => panic!("Expected scope, got {:?}", t),
        }
        while *self.peek() == Token::Comma {
            self.advance();
            match self.advance() {
                Token::Scope(s) => scopes.push(s),
                t => panic!("Expected scope, got {:?}", t),
            }
        }
        scopes
    }

    fn parse_expr(&mut self) -> Pattern {
        // if/then/else
        if *self.peek() == Token::KwIf {
            self.advance(); // 'if'
            let cond = self.parse_expr();
            self.expect(Token::KwThen);
            let then_branch = self.parse_expr();
            self.expect(Token::KwElse);
            let else_branch = self.parse_expr();
            let if_sym = self.store.sym("if");
            return Pattern::Call(Box::new(Pattern::Sym(if_sym)), vec![cond, then_branch, else_branch]);
        }
        self.parse_arrow()
    }

    fn parse_arrow(&mut self) -> Pattern {
        let left = self.parse_concat();
        if *self.peek() == Token::Arrow {
            self.advance();
            let right = self.parse_arrow(); // right-associative
            let s = self.store.sym("arrow");
            Pattern::Call(Box::new(Pattern::Sym(s)), vec![left, right])
        } else {
            left
        }
    }

    fn parse_concat(&mut self) -> Pattern {
        let mut left = self.parse_cmp();
        loop {
            if *self.peek() == Token::PlusPlus {
                self.advance();
                let right = self.parse_cmp();
                let s = self.store.sym("str_concat");
                left = Pattern::Call(Box::new(Pattern::Sym(s)), vec![left, right]);
            } else {
                break;
            }
        }
        left
    }

    fn parse_cmp(&mut self) -> Pattern {
        let left = self.parse_add();
        let tok = self.peek().clone();
        let op = match tok {
            Token::EqEq => Some("eq"),
            Token::BangEq => Some("neq"),
            Token::Lt => Some("lt"),
            Token::Gt => Some("gt"),
            Token::LtEq => Some("lte"),
            Token::GtEq => Some("gte"),
            _ => None,
        };
        if let Some(op_name) = op {
            self.advance();
            let right = self.parse_add();
            let s = self.store.sym(op_name);
            Pattern::Call(Box::new(Pattern::Sym(s)), vec![left, right])
        } else {
            left
        }
    }

    fn parse_add(&mut self) -> Pattern {
        let mut left = self.parse_mul();
        loop {
            let tok = self.peek().clone();
            match tok {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_mul();
                    let s = self.store.sym("add");
                    left = Pattern::Call(Box::new(Pattern::Sym(s)), vec![left, right]);
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_mul();
                    let s = self.store.sym("sub");
                    left = Pattern::Call(Box::new(Pattern::Sym(s)), vec![left, right]);
                }
                _ => break,
            }
        }
        left
    }

    fn parse_mul(&mut self) -> Pattern {
        let mut left = self.parse_call();
        loop {
            let tok = self.peek().clone();
            match tok {
                Token::Star => {
                    self.advance();
                    let right = self.parse_call();
                    let s = self.store.sym("mul");
                    left = Pattern::Call(Box::new(Pattern::Sym(s)), vec![left, right]);
                }
                Token::Slash => {
                    self.advance();
                    let right = self.parse_call();
                    let s = self.store.sym("div");
                    left = Pattern::Call(Box::new(Pattern::Sym(s)), vec![left, right]);
                }
                Token::Percent => {
                    self.advance();
                    let right = self.parse_call();
                    let s = self.store.sym("mod");
                    left = Pattern::Call(Box::new(Pattern::Sym(s)), vec![left, right]);
                }
                _ => break,
            }
        }
        left
    }

    fn parse_call(&mut self) -> Pattern {
        let atom = self.parse_atom();
        if *self.peek() == Token::LParen {
            self.advance();
            let args = self.parse_expr_list_until(Token::RParen);
            self.expect(Token::RParen);
            Pattern::Call(Box::new(atom), args)
        } else {
            atom
        }
    }

    fn parse_atom(&mut self) -> Pattern {
        let tok = self.tokens[self.pos].clone();
        match tok {
            Token::Num(n) => { self.pos += 1; Pattern::Num(n) }
            Token::Float(f) => { self.pos += 1; Pattern::Float(f.to_bits()) }
            Token::Gt => {
                // Prefix >: > x → dir_right(x) (unary direction marker)
                self.pos += 1;
                let inner = self.parse_atom();
                let s = self.store.sym("dir_right");
                Pattern::Call(Box::new(Pattern::Sym(s)), vec![inner])
            }
            Token::Lt => {
                // Prefix <: < x → dir_left(x) (unary direction marker)
                self.pos += 1;
                let inner = self.parse_atom();
                let s = self.store.sym("dir_left");
                Pattern::Call(Box::new(Pattern::Sym(s)), vec![inner])
            }
            Token::Ellipsis => {
                // Standalone ... → ellipsis symbol (for PuzzleScript cell patterns)
                self.pos += 1;
                let s = self.store.sym("ellipsis");
                Pattern::Sym(s)
            }
            Token::Minus => {
                // Unary minus: -3, -3.5, or -(expr) → sub(0, expr)
                self.pos += 1;
                match self.tokens[self.pos].clone() {
                    Token::Num(n) => { self.pos += 1; Pattern::Num(-n) }
                    Token::Float(f) => { self.pos += 1; Pattern::Float((-f).to_bits()) }
                    _ => {
                        let inner = self.parse_atom();
                        let s = self.store.sym("sub");
                        Pattern::Call(Box::new(Pattern::Sym(s)), vec![Pattern::Num(0), inner])
                    }
                }
            }
            Token::Ident(ref s) if s == "_" => {
                self.pos += 1;
                Pattern::Wildcard
            }
            Token::Ident(s) => {
                self.pos += 1;
                let sym = self.store.sym(&s);
                Pattern::Sym(sym)
            }
            Token::KwRule => {
                // rule name: @scope -> @scope { ... } — dynamic rule declaration
                if matches!(self.tokens.get(self.pos + 1), Some(Token::Ident(_))) {
                    return self.parse_rule_decl_expr();
                }
                // rule(lhs, rhs) — ground dynamic rule
                self.pos += 1;
                let sym = self.store.sym("rule");
                Pattern::Sym(sym)
            }
            Token::KwFn => {
                self.pos += 1;
                let sym = self.store.sym("fn");
                Pattern::Sym(sym)
            }
            Token::Str(s) => {
                self.pos += 1;
                let sym = self.store.sym(&s);
                Pattern::Sym(sym)
            }
            Token::Var(ref name) if name == "_" => {
                self.pos += 1;
                Pattern::Wildcard
            }
            Token::Var(name) => {
                self.pos += 1;
                let v = self.get_var(&name);
                Pattern::Var(v)
            }
            Token::Keyword(name) => {
                self.pos += 1;
                let kw = format!(":{}", name);
                let sym = self.store.sym(&kw);
                Pattern::Sym(sym)
            }
            Token::LParen => {
                self.pos += 1;
                let e = self.parse_expr();
                self.expect(Token::RParen);
                e
            }
            Token::LBrace => {
                self.pos += 1;
                // Disambiguate: if next is Keyword, }, or ?var, it's a map literal
                if matches!(self.peek(), Token::Keyword(_) | Token::RBrace | Token::Var(_)) {
                    return self.parse_map_literal();
                }
                // Otherwise block: seq of expressions
                let mut exprs = Vec::new();
                while *self.peek() != Token::RBrace {
                    exprs.push(self.parse_expr());
                }
                self.expect(Token::RBrace);
                assert!(!exprs.is_empty(), "Empty block");
                let seq_sym = self.store.sym("seq");
                let mut result = exprs.pop().unwrap();
                while let Some(expr) = exprs.pop() {
                    result = Pattern::Call(Box::new(Pattern::Sym(seq_sym)), vec![expr, result]);
                }
                result
            }
            Token::HashBrace => {
                // Set literal: #{elem, elem, ...} — produces Pattern::Set for subset matching
                self.pos += 1;
                if *self.peek() == Token::RBrace {
                    self.advance();
                    return Pattern::Set(vec![]);
                }
                let mut elems = vec![self.parse_expr()];
                // Check for spread
                if *self.peek() == Token::Ellipsis {
                    self.advance();
                    let last = elems.pop().unwrap();
                    match last {
                        Pattern::Var(v) => elems.push(Pattern::Spread(v)),
                        Pattern::Wildcard => elems.push(Pattern::WildSpread),
                        _ => panic!("Spread '...' can only follow a variable (?x...) or wildcard (_...)"),
                    }
                    self.expect(Token::RBrace);
                    return Pattern::Set(elems);
                }
                while *self.peek() == Token::Comma {
                    self.advance();
                    elems.push(self.parse_expr());
                    if *self.peek() == Token::Ellipsis {
                        self.advance();
                        let last = elems.pop().unwrap();
                        match last {
                            Pattern::Var(v) => elems.push(Pattern::Spread(v)),
                            Pattern::Wildcard => elems.push(Pattern::WildSpread),
                            _ => panic!("Spread '...' can only follow a variable (?x...) or wildcard (_...)"),
                        }
                        break;
                    }
                }
                self.expect(Token::RBrace);
                Pattern::Set(elems)
            }
            Token::LBracket => {
                // [elem, elem, ...] → vec(...)  (comma-separated, existing)
                // [elem | elem | ...] → cells(...) (pipe-separated, new)
                self.pos += 1;
                let vec_sym = self.store.sym("vec");
                if *self.peek() == Token::RBracket {
                    self.advance();
                    return Pattern::Call(Box::new(Pattern::Sym(vec_sym)), vec![]);
                }
                let first = self.parse_expr();
                // Check for spread on first element
                if *self.peek() == Token::Ellipsis {
                    self.advance();
                    let mut elems = Vec::new();
                    match first {
                        Pattern::Var(v) => elems.push(Pattern::Spread(v)),
                        Pattern::Wildcard => elems.push(Pattern::WildSpread),
                        _ => panic!("Spread '...' can only follow a variable (?x...) or wildcard (_...)"),
                    }
                    self.expect(Token::RBracket);
                    return Pattern::Call(Box::new(Pattern::Sym(vec_sym)), elems);
                }
                // Check for pipe → cells(...)
                if *self.peek() == Token::Pipe {
                    let cells_sym = self.store.sym("cells");
                    let mut elems = vec![first];
                    while *self.peek() == Token::Pipe {
                        self.advance();
                        elems.push(self.parse_expr());
                    }
                    self.expect(Token::RBracket);
                    return Pattern::Call(Box::new(Pattern::Sym(cells_sym)), elems);
                }
                // Comma-separated → vec(...)
                let mut elems = vec![first];
                while *self.peek() == Token::Comma {
                    self.advance();
                    elems.push(self.parse_expr());
                    // Check for spread on the element just parsed
                    if *self.peek() == Token::Ellipsis {
                        self.advance();
                        let last = elems.pop().unwrap();
                        match last {
                            Pattern::Var(v) => elems.push(Pattern::Spread(v)),
                            Pattern::Wildcard => elems.push(Pattern::WildSpread),
                            _ => panic!("Spread '...' can only follow a variable (?x...) or wildcard (_...)"),
                        }
                        break;
                    }
                }
                self.expect(Token::RBracket);
                Pattern::Call(Box::new(Pattern::Sym(vec_sym)), elems)
            }
            Token::KwIf => {
                // Handled in parse_expr, but can appear as atom if parenthesized
                self.pos += 1;
                let cond = self.parse_expr();
                self.expect(Token::KwThen);
                let then_branch = self.parse_expr();
                self.expect(Token::KwElse);
                let else_branch = self.parse_expr();
                let if_sym = self.store.sym("if");
                Pattern::Call(Box::new(Pattern::Sym(if_sym)), vec![cond, then_branch, else_branch])
            }
            Token::Scope(name) => {
                // @scope_name expr → emit(scope_name_sym, expr)
                self.pos += 1;
                if !self.emit_scopes.contains(&name) {
                    self.emit_scopes.push(name.clone());
                }
                let scope_sym = self.store.sym(&name);
                let expr = self.parse_expr();
                let emit_sym = self.store.sym("emit");
                Pattern::Call(Box::new(Pattern::Sym(emit_sym)), vec![Pattern::Sym(scope_sym), expr])
            }
            _ => panic!("Unexpected token in expression: {:?}", tok),
        }
    }

    fn parse_expr_list_until(&mut self, end: Token) -> Vec<Pattern> {
        if *self.peek() == end {
            return vec![];
        }
        let mut list = vec![self.parse_expr()];
        if *self.peek() == Token::Ellipsis {
            self.advance();
            let last = list.pop().unwrap();
            match last {
                Pattern::Var(v) => list.push(Pattern::Spread(v)),
                Pattern::Wildcard => list.push(Pattern::WildSpread),
                _ => panic!("Spread '...' can only follow a variable (?x...) or wildcard (_...)"),
            }
            return list;
        }
        while *self.peek() == Token::Comma {
            self.advance();
            list.push(self.parse_expr());
            if *self.peek() == Token::Ellipsis {
                self.advance();
                let last = list.pop().unwrap();
                match last {
                    Pattern::Var(v) => list.push(Pattern::Spread(v)),
                    Pattern::Wildcard => list.push(Pattern::WildSpread),
                    _ => panic!("Spread '...' can only follow a variable (?x...) or wildcard (_...)"),
                }
                break;
            }
        }
        list
    }

    /// Parse map literal body after the opening `{` has been consumed.
    /// Expects `:key expr` pairs until `}`. No commas between entries.
    fn parse_map_literal(&mut self) -> Pattern {
        let mut entries: Vec<(Pattern, Pattern)> = Vec::new();
        while *self.peek() != Token::RBrace {
            let key = self.parse_expr();
            let val = self.parse_expr();
            entries.push((key, val));
        }
        self.expect(Token::RBrace);
        Pattern::Map(entries)
    }
}
