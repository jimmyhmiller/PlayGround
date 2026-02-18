use std::collections::HashMap;
use std::fmt;
use std::io::{self, BufRead};
use std::time::Instant;

// ── Newtypes ──

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TermId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SymId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct VarId(u32);

// ── TermData (Copy — args live in a pool, no heap alloc per term) ──

#[derive(Clone, Copy, Debug)]
enum TermData {
    Num(i64),
    Sym(SymId),
    Call { head: TermId, args_start: u32, args_len: u16 },
}

// ── TermStore ──

const NUM_CACHE_SIZE: usize = 64;

struct TermStore {
    terms: Vec<TermData>,
    args_pool: Vec<TermId>,
    num_cache: [TermId; NUM_CACHE_SIZE],
    num_cache_valid: u64,
    num_dedup: HashMap<i64, TermId>,
    sym_term_dedup: HashMap<SymId, TermId>,
    call_dedup: HashMap<[u32; 6], TermId>,
    symbols: Vec<String>,
    sym_dedup: HashMap<String, SymId>,
}

impl TermStore {
    fn new() -> Self {
        TermStore {
            terms: Vec::with_capacity(1024),
            args_pool: Vec::with_capacity(2048),
            num_cache: [TermId(0); NUM_CACHE_SIZE],
            num_cache_valid: 0,
            num_dedup: HashMap::new(),
            sym_term_dedup: HashMap::new(),
            call_dedup: HashMap::new(),
            symbols: Vec::new(),
            sym_dedup: HashMap::new(),
        }
    }

    #[inline]
    fn get(&self, id: TermId) -> TermData {
        self.terms[id.0 as usize]
    }

    fn sym(&mut self, name: &str) -> SymId {
        if let Some(&id) = self.sym_dedup.get(name) {
            return id;
        }
        let id = SymId(self.symbols.len() as u32);
        self.symbols.push(name.to_string());
        self.sym_dedup.insert(name.to_string(), id);
        id
    }

    fn sym_name(&self, id: SymId) -> &str {
        &self.symbols[id.0 as usize]
    }

    #[inline]
    fn num(&mut self, n: i64) -> TermId {
        if n >= 0 && (n as usize) < NUM_CACHE_SIZE {
            let i = n as usize;
            if self.num_cache_valid & (1u64 << i) != 0 {
                return self.num_cache[i];
            }
            let id = TermId(self.terms.len() as u32);
            self.terms.push(TermData::Num(n));
            self.num_cache[i] = id;
            self.num_cache_valid |= 1u64 << i;
            id
        } else {
            if let Some(&id) = self.num_dedup.get(&n) {
                return id;
            }
            let id = TermId(self.terms.len() as u32);
            self.terms.push(TermData::Num(n));
            self.num_dedup.insert(n, id);
            id
        }
    }

    fn symbol(&mut self, name: &str) -> TermId {
        let s = self.sym(name);
        self.sym_term(s)
    }

    #[inline]
    fn sym_term(&mut self, s: SymId) -> TermId {
        if let Some(&id) = self.sym_term_dedup.get(&s) {
            return id;
        }
        let id = TermId(self.terms.len() as u32);
        self.terms.push(TermData::Sym(s));
        self.sym_term_dedup.insert(s, id);
        id
    }

    #[inline]
    fn call(&mut self, head: TermId, args: &[TermId]) -> TermId {
        if args.len() <= 4 {
            let mut key = [0u32; 6];
            key[0] = args.len() as u32;
            key[1] = head.0;
            for (i, a) in args.iter().enumerate() {
                key[i + 2] = a.0;
            }
            if let Some(&id) = self.call_dedup.get(&key) {
                return id;
            }
            let args_start = self.args_pool.len() as u32;
            self.args_pool.extend_from_slice(args);
            let id = TermId(self.terms.len() as u32);
            self.terms.push(TermData::Call { head, args_start, args_len: args.len() as u16 });
            self.call_dedup.insert(key, id);
            id
        } else {
            let args_start = self.args_pool.len() as u32;
            self.args_pool.extend_from_slice(args);
            let id = TermId(self.terms.len() as u32);
            self.terms.push(TermData::Call { head, args_start, args_len: args.len() as u16 });
            id
        }
    }

    fn display(&self, id: TermId) -> TermDisplay<'_> {
        TermDisplay { store: self, id }
    }
}

struct TermDisplay<'a> {
    store: &'a TermStore,
    id: TermId,
}

impl fmt::Display for TermDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.store.get(self.id) {
            TermData::Num(n) => write!(f, "{}", n),
            TermData::Sym(s) => write!(f, "{}", self.store.sym_name(s)),
            TermData::Call { head, args_start, args_len } => {
                // quote(x) displays transparently as x
                if args_len == 1 {
                    if let TermData::Sym(s) = self.store.get(head) {
                        if self.store.sym_name(s) == "quote" {
                            let inner = self.store.args_pool[args_start as usize];
                            return write!(f, "{}", self.store.display(inner));
                        }
                    }
                }
                write!(f, "{}(", self.store.display(head))?;
                for i in 0..args_len as usize {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    let arg = self.store.args_pool[args_start as usize + i];
                    write!(f, "{}", self.store.display(arg))?;
                }
                write!(f, ")")
            }
        }
    }
}

// ── Pattern, Env, Clause, Rule ──

#[derive(Clone, Debug)]
enum Pattern {
    Num(i64),
    Sym(SymId),
    Var(VarId),
    Call(Box<Pattern>, Vec<Pattern>),
}

const MAX_VARS: usize = 8;

#[derive(Clone, Copy)]
struct Env {
    bindings: [TermId; MAX_VARS],
    bound: u8,
}

impl Env {
    #[inline]
    fn new() -> Self {
        Env {
            bindings: [TermId(0); MAX_VARS],
            bound: 0,
        }
    }

    #[inline]
    fn get(&self, v: VarId) -> Option<TermId> {
        let i = v.0 as u8;
        if self.bound & (1 << i) != 0 {
            Some(self.bindings[i as usize])
        } else {
            None
        }
    }

    #[inline]
    fn set(&mut self, v: VarId, t: TermId) {
        let i = v.0 as usize;
        self.bindings[i] = t;
        self.bound |= 1 << (i as u8);
    }
}


struct Clause {
    lhs: Pattern,
    rhs: Pattern,
}

struct Rule {
    name: String,
    clauses: Vec<Clause>,
}

// ── Builtins ──

struct Builtins {
    add: SymId,
    sub: SymId,
    mul: SymId,
    println: SymId,
    readline: SymId,
    quote: SymId,
    seq: SymId,
}

// ── Meta symbols ──

struct MetaSyms {
    reduction: SymId,
    result_sym: SymId,
    rule_sym: SymId,
    fn_sym: SymId,
    builtin_sym: SymId,
}

// ── Pattern matching ──

#[inline]
fn unwrap_quote(store: &TermStore, term: TermId, quote_sym: SymId) -> TermId {
    if let TermData::Call { head, args_start, args_len: 1 } = store.get(term) {
        if let TermData::Sym(s) = store.get(head) {
            if s == quote_sym {
                return store.args_pool[args_start as usize];
            }
        }
    }
    term
}

fn match_pattern(store: &TermStore, pat: &Pattern, term: TermId, env: &mut Env, quote_sym: SymId) -> bool {
    match pat {
        Pattern::Num(n) => {
            let term = unwrap_quote(store, term, quote_sym);
            matches!(store.get(term), TermData::Num(m) if m == *n)
        }
        Pattern::Sym(s) => {
            let term = unwrap_quote(store, term, quote_sym);
            matches!(store.get(term), TermData::Sym(t) if t == *s)
        }
        Pattern::Var(v) => {
            // Don't unwrap quote — preserve protection for variables
            if let Some(bound) = env.get(*v) {
                bound == term
            } else {
                env.set(*v, term);
                true
            }
        }
        Pattern::Call(head_pat, arg_pats) => {
            let term = unwrap_quote(store, term, quote_sym);
            if let TermData::Call { head, args_start, args_len } = store.get(term) {
                if args_len as usize != arg_pats.len() {
                    return false;
                }
                if !match_pattern(store, head_pat, head, env, quote_sym) {
                    return false;
                }
                for (i, ap) in arg_pats.iter().enumerate() {
                    let arg = store.args_pool[args_start as usize + i];
                    if !match_pattern(store, ap, arg, env, quote_sym) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }
    }
}

// ── Substitution ──

fn substitute(store: &mut TermStore, pat: &Pattern, env: &Env) -> TermId {
    match pat {
        Pattern::Num(n) => store.num(*n),
        Pattern::Sym(s) => store.sym_term(*s),
        Pattern::Var(v) => env.get(*v).unwrap(),
        Pattern::Call(head, args) => {
            let h = substitute(store, head, env);
            let mut buf = [TermId(0); 16];
            let len = args.len();
            for (i, ap) in args.iter().enumerate() {
                buf[i] = substitute(store, ap, env);
            }
            store.call(h, &buf[..len])
        }
    }
}

// ── Lexer ──

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(i64),
    Ident(String),
    Var(String),
    Scope(String),
    Keyword(String),
    KwFn,
    KwRule,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Str(String),
    Arrow,
    FatArrow,
    Eq,
    Plus,
    Minus,
    Star,
    Eof,
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
}

impl Lexer {
    fn new(src: &str) -> Self {
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
            return Token::Num(n);
        }

        if c.is_alphabetic() || c == '_' {
            let s = self.read_ident();
            return match s.as_str() {
                "fn" => Token::KwFn,
                "rule" => Token::KwRule,
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
            ',' => Token::Comma,
            ':' => Token::Colon,
            '+' => Token::Plus,
            '*' => Token::Star,
            '-' => {
                if self.peek() == '>' { self.advance(); Token::Arrow }
                else { Token::Minus }
            }
            '=' => {
                if self.peek() == '>' { self.advance(); Token::FatArrow }
                else { Token::Eq }
            }
            _ => panic!("Unexpected character: '{}'", c),
        }
    }

    fn tokenize(mut self) -> Vec<Token> {
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

struct Program {
    rules: Vec<Rule>,
    meta_rules: Vec<Rule>,
    expr: Pattern,
}

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    store: &'a mut TermStore,
    clause_vars: HashMap<String, VarId>,
    next_var: u32,
}

impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token>, store: &'a mut TermStore) -> Self {
        Parser { tokens, pos: 0, store, clause_vars: HashMap::new(), next_var: 0 }
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

    fn parse_program(&mut self) -> Program {
        let mut fn_clauses: HashMap<String, Vec<Clause>> = HashMap::new();
        let mut fn_order: Vec<String> = Vec::new();
        let mut meta_rules: Vec<Rule> = Vec::new();
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
                    let (rule, is_meta) = self.parse_rule_decl();
                    if is_meta {
                        meta_rules.push(rule);
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

        Program {
            rules,
            meta_rules,
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
        self.expect(Token::Eq);

        let name_sym = self.store.sym(&name);
        let lhs = Pattern::Call(Box::new(Pattern::Sym(name_sym)), args);
        let rhs = self.parse_expr();

        (name, Clause { lhs, rhs })
    }

    fn parse_rule_decl(&mut self) -> (Rule, bool) {
        self.advance(); // 'rule'
        let name = match self.advance() {
            Token::Ident(s) => s,
            t => panic!("Expected rule name, got {:?}", t),
        };

        let mut is_meta = false;
        if *self.peek() == Token::Colon {
            self.advance();
            let from_scopes = self.parse_scope_list();
            self.expect(Token::Arrow);
            let _to_scope = match self.advance() {
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
            clauses.push(Clause { lhs, rhs });
        }
        self.expect(Token::RBrace);

        (Rule { name, clauses }, is_meta)
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
        self.parse_add()
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
            if *self.peek() == Token::Star {
                self.advance();
                let right = self.parse_call();
                let s = self.store.sym("mul");
                left = Pattern::Call(Box::new(Pattern::Sym(s)), vec![left, right]);
            } else {
                break;
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
            Token::Ident(s) => {
                self.pos += 1;
                let sym = self.store.sym(&s);
                Pattern::Sym(sym)
            }
            Token::KwRule => {
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
            _ => panic!("Unexpected token in expression: {:?}", tok),
        }
    }

    fn parse_expr_list_until(&mut self, end: Token) -> Vec<Pattern> {
        if *self.peek() == end {
            return vec![];
        }
        let mut list = vec![self.parse_expr()];
        while *self.peek() == Token::Comma {
            self.advance();
            list.push(self.parse_expr());
        }
        list
    }
}

fn pattern_to_term(store: &mut TermStore, pat: &Pattern) -> TermId {
    match pat {
        Pattern::Num(n) => store.num(*n),
        Pattern::Sym(s) => store.sym_term(*s),
        Pattern::Var(v) => panic!("Unbound variable ?{} in top-level expression", v.0),
        Pattern::Call(head, args) => {
            let h = pattern_to_term(store, head);
            let a: Vec<TermId> = args.iter().map(|a| pattern_to_term(store, a)).collect();
            store.call(h, &a)
        }
    }
}

// ── Evaluator ──

struct Evaluator<'a> {
    store: &'a mut TermStore,
    rules: &'a [Rule],
    builtins: &'a Builtins,
    meta_syms: MetaSyms,
    meta_rules: Vec<Rule>,
    dynamic_rules: HashMap<TermId, TermId>,
    step_count: usize,
    in_meta: bool,
}

impl Evaluator<'_> {
    fn eval(&mut self, term: TermId) -> TermId {
        let td = self.store.get(term);

        // 1. Innermost: eval head and args of Call terms
        let term = if let TermData::Call { head, args_start, args_len } = td {
            if let TermData::Sym(s) = self.store.get(head) {
                // quote(x) is a normal form — don't eval contents
                if s == self.builtins.quote {
                    return term;
                }
                // rule(lhs, rhs) — assert dynamic ground rule, don't eval lhs
                if s == self.meta_syms.rule_sym && args_len == 2 {
                    let lhs = self.store.args_pool[args_start as usize];
                    let rhs_raw = self.store.args_pool[args_start as usize + 1];
                    let rhs = self.eval(rhs_raw);
                    self.dynamic_rules.insert(lhs, rhs);
                    return self.store.num(0);
                }
            }
            let h = self.eval(head);
            let len = args_len as usize;
            let mut buf = [TermId(0); 16];
            let mut changed = h != head;
            for i in 0..len {
                let arg = self.store.args_pool[args_start as usize + i];
                buf[i] = self.eval(arg);
                changed |= buf[i] != arg;
            }
            if changed {
                self.store.call(h, &buf[..len])
            } else {
                term
            }
        } else {
            term
        };

        // 2. seq builtin — args already evaluated, just return last
        if let TermData::Call { head, args_start, args_len } = self.store.get(term) {
            if args_len == 2 {
                if let TermData::Sym(s) = self.store.get(head) {
                    if s == self.builtins.seq {
                        return self.store.args_pool[args_start as usize + 1];
                    }
                }
            }
        }

        // 3. Dynamic rules (asserted at runtime via meta rules)
        if let Some(&val) = self.dynamic_rules.get(&term) {
            return val;
        }

        // 4. Try user rules — first match wins
        let num_rules = self.rules.len();
        let quote_sym = self.builtins.quote;
        for rule_idx in 0..num_rules {
            let num_clauses = self.rules[rule_idx].clauses.len();
            for clause_idx in 0..num_clauses {
                let mut env = Env::new();
                if match_pattern(self.store, &self.rules[rule_idx].clauses[clause_idx].lhs, term, &mut env, quote_sym) {
                    let result = substitute(self.store, &self.rules[rule_idx].clauses[clause_idx].rhs, &env);
                    if !self.in_meta {
                        self.step_count += 1;
                        if !self.meta_rules.is_empty() {
                            let name = self.rules[rule_idx].name.clone();
                            let name_sym = self.store.sym(&name);
                            let name_term = self.store.sym_term(name_sym);
                            let clause_term = self.store.num(clause_idx as i64);
                            let fn_head = self.store.sym_term(self.meta_syms.fn_sym);
                            let kind = self.store.call(fn_head, &[name_term, clause_term]);
                            self.fire_meta(term, result, kind);
                        }
                    }
                    let final_val = self.eval(result);
                    if !self.in_meta && !self.meta_rules.is_empty() {
                        self.fire_result(term, final_val);
                    }
                    return final_val;
                }
            }
        }

        // 3. Try arithmetic builtins
        let td = self.store.get(term);
        if let TermData::Call { head, args_start, args_len: 2 } = td {
            if let TermData::Sym(s) = self.store.get(head) {
                let a = self.store.args_pool[args_start as usize];
                let b = self.store.args_pool[args_start as usize + 1];
                if let (TermData::Num(av), TermData::Num(bv)) = (self.store.get(a), self.store.get(b)) {
                    let (result, op_sym) = if s == self.builtins.add {
                        (Some(av + bv), self.builtins.add)
                    } else if s == self.builtins.sub {
                        (Some(av - bv), self.builtins.sub)
                    } else if s == self.builtins.mul {
                        (Some(av * bv), self.builtins.mul)
                    } else {
                        (None, SymId(0))
                    };
                    if let Some(r) = result {
                        let num = self.store.num(r);
                        if !self.in_meta {
                            self.step_count += 1;
                            if !self.meta_rules.is_empty() {
                                let op_term = self.store.sym_term(op_sym);
                                let builtin_head = self.store.sym_term(self.meta_syms.builtin_sym);
                                let kind = self.store.call(builtin_head, &[op_term]);
                                self.fire_meta(term, num, kind);
                            }
                        }
                        return num;
                    }
                }
            }
        }

        // 4. Try IO builtins
        if let TermData::Call { head, args_start, args_len } = self.store.get(term) {
            if let TermData::Sym(s) = self.store.get(head) {
                if s == self.builtins.println {
                    for i in 0..args_len as usize {
                        let arg = self.store.args_pool[args_start as usize + i];
                        print!("{}", self.store.display(arg));
                    }
                    println!();
                    return self.store.num(0);
                }
                if s == self.builtins.readline && args_len == 0 {
                    let mut line = String::new();
                    io::stdin().lock().read_line(&mut line).unwrap();
                    let trimmed = line.trim();
                    let sym = self.store.sym(trimmed);
                    return self.store.sym_term(sym);
                }
            }
        }

        // 5. Normal form
        term
    }

    fn fire_meta(&mut self, sub_old: TermId, sub_new: TermId, kind: TermId) {
        let reduction_head = self.store.sym_term(self.meta_syms.reduction);
        let step = self.store.num(self.step_count as i64);
        let quote_head = self.store.sym_term(self.builtins.quote);
        let quoted_old = self.store.call(quote_head, &[sub_old]);
        let quoted_new = self.store.call(quote_head, &[sub_new]);
        let event = self.store.call(reduction_head, &[step, quoted_old, quoted_new, kind]);

        let quote_sym = self.builtins.quote;
        self.in_meta = true;
        let num_meta = self.meta_rules.len();
        for i in 0..num_meta {
            let num_clauses = self.meta_rules[i].clauses.len();
            for j in 0..num_clauses {
                let mut env = Env::new();
                if match_pattern(self.store, &self.meta_rules[i].clauses[j].lhs, event, &mut env, quote_sym) {
                    let result = substitute(self.store, &self.meta_rules[i].clauses[j].rhs, &env);
                    self.eval(result);
                    break;
                }
            }
        }
        self.in_meta = false;
    }

    fn fire_result(&mut self, original: TermId, final_val: TermId) {
        let result_head = self.store.sym_term(self.meta_syms.result_sym);
        let step = self.store.num(self.step_count as i64);
        let event = self.store.call(result_head, &[step, original, final_val]);

        let quote_sym = self.builtins.quote;
        self.in_meta = true;
        let num_meta = self.meta_rules.len();
        for i in 0..num_meta {
            let num_clauses = self.meta_rules[i].clauses.len();
            for j in 0..num_clauses {
                let mut env = Env::new();
                if match_pattern(self.store, &self.meta_rules[i].clauses[j].lhs, event, &mut env, quote_sym) {
                    let result = substitute(self.store, &self.meta_rules[i].clauses[j].rhs, &env);
                    self.eval(result);
                    break;
                }
            }
        }
        self.in_meta = false;
    }
}

// ── Main ──

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rules4 <file>");
        std::process::exit(1);
    }

    let filename = &args[1];
    let src = std::fs::read_to_string(filename).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", filename, e);
        std::process::exit(1);
    });

    let mut store = TermStore::new();
    let tokens = Lexer::new(&src).tokenize();
    let program = Parser::new(tokens, &mut store).parse_program();

    let builtins = Builtins {
        add: store.sym("add"),
        sub: store.sym("sub"),
        mul: store.sym("mul"),
        println: store.sym("println"),
        readline: store.sym("readline"),
        quote: store.sym("quote"),
        seq: store.sym("seq"),
    };

    let meta_syms = MetaSyms {
        reduction: store.sym("reduction"),
        result_sym: store.sym("result"),
        rule_sym: store.sym("rule"),
        fn_sym: store.sym("fn"),
        builtin_sym: store.sym("builtin"),
    };

    let term = pattern_to_term(&mut store, &program.expr);

    let mut ev = Evaluator {
        store: &mut store,
        rules: &program.rules,
        builtins: &builtins,
        meta_syms,
        meta_rules: program.meta_rules,
        dynamic_rules: HashMap::new(),
        step_count: 0,
        in_meta: false,
    };

    let start = Instant::now();
    let result = ev.eval(term);
    let elapsed = start.elapsed();

    println!("{}", ev.store.display(result));
    eprintln!("{} reductions in {:.3}s", ev.step_count, elapsed.as_secs_f64());
}
