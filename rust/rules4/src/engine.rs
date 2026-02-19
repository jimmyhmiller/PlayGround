use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;
use crate::term::*;
use crate::pattern::*;
use crate::scope::*;

// ── Builtins ──

pub struct Builtins {
    pub add: SymId,
    pub sub: SymId,
    pub mul: SymId,
    pub div: SymId,
    pub mod_sym: SymId,
    pub quote: SymId,
    pub seq: SymId,
    pub true_sym: SymId,
    pub false_sym: SymId,
    pub if_sym: SymId,
    pub eq: SymId,
    pub neq: SymId,
    pub lt: SymId,
    pub gt: SymId,
    pub lte: SymId,
    pub gte: SymId,
    pub str_concat: SymId,
    pub str_len: SymId,
    pub cons: SymId,
    pub nil: SymId,
    pub retract: SymId,
    pub query_all: SymId,
    pub emit: SymId,
    pub abs_sym: SymId,
    pub floor_sym: SymId,
    pub random_sym: SymId,
}

// ── Meta symbols ──

pub struct MetaSyms {
    pub reduction: SymId,
    pub result_sym: SymId,
    pub rule_sym: SymId,
    pub fn_sym: SymId,
    pub builtin_sym: SymId,
    pub rule_decl_sym: SymId,
    pub clause_sym: SymId,
    pub pvar_sym: SymId,
}

// ── MetaRule ──

pub struct MetaRule {
    pub rule: Rule,
    pub target_scope: Option<usize>, // index into scope_handlers
}

fn rule_head_sym(rule: &Rule) -> Option<SymId> {
    if let Some(clause) = rule.clauses.first() {
        if let Pattern::Call(head, _) = &clause.lhs {
            if let Pattern::Sym(s) = **head {
                return Some(s);
            }
        }
    }
    None
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

// ── Engine ──

pub struct Engine {
    pub store: TermStore,
    rules: Vec<Rule>,
    rule_index: HashMap<SymId, Vec<usize>>,
    builtins: Builtins,
    meta_syms: MetaSyms,
    meta_rules: Vec<MetaRule>,
    dynamic_rules: Vec<(TermId, TermId)>,
    dynamic_rules_map: HashMap<TermId, TermId>,
    scope_handlers: Vec<Option<Box<dyn ScopeHandler>>>,
    scope_map: HashMap<String, usize>,
    active_scope: Option<usize>,
    pub step_count: usize,
    pub eval_calls: usize,
    eval_depth: usize,
    in_meta: bool,
    scope_pending: HashMap<String, Rc<RefCell<Vec<TermId>>>>,
    eval_cache: Vec<u32>,       // indexed by TermId.0, u32::MAX = uncached
    cache_generation: u64,      // bumped on every invalidation
    pub step_limit_exceeded: bool,
}

impl Engine {
    pub fn new(mut store: TermStore, rules: Vec<Rule>, parsed_meta: Vec<(Rule, String)>) -> Self {
        let builtins = Builtins {
            add: store.sym("add"),
            sub: store.sym("sub"),
            mul: store.sym("mul"),
            div: store.sym("div"),
            mod_sym: store.sym("mod"),
            quote: store.sym("quote"),
            seq: store.sym("seq"),
            true_sym: store.sym("true"),
            false_sym: store.sym("false"),
            if_sym: store.sym("if"),
            eq: store.sym("eq"),
            neq: store.sym("neq"),
            lt: store.sym("lt"),
            gt: store.sym("gt"),
            lte: store.sym("lte"),
            gte: store.sym("gte"),
            str_concat: store.sym("str_concat"),
            str_len: store.sym("str_len"),
            cons: store.sym("cons"),
            nil: store.sym("nil"),
            retract: store.sym("retract"),
            query_all: store.sym("query_all"),
            emit: store.sym("emit"),
            abs_sym: store.sym("abs"),
            floor_sym: store.sym("floor"),
            random_sym: store.sym("random"),
        };

        let meta_syms = MetaSyms {
            reduction: store.sym("reduction"),
            result_sym: store.sym("result"),
            rule_sym: store.sym("rule"),
            fn_sym: store.sym("fn"),
            builtin_sym: store.sym("builtin"),
            rule_decl_sym: store.sym("__rule_decl"),
            clause_sym: store.sym("__clause"),
            pvar_sym: store.sym("__pvar"),
        };

        #[allow(unused_mut)]
        let mut scope_handlers: Vec<Option<Box<dyn ScopeHandler>>> = Vec::new();
        #[allow(unused_mut)]
        let mut scope_map: HashMap<String, usize> = HashMap::new();

        // Register built-in @io scope (only on non-wasm)
        #[cfg(not(target_arch = "wasm32"))]
        {
            let println_sym = store.sym("println");
            let readline_sym = store.sym("readline");
            let io_idx = scope_handlers.len();
            scope_handlers.push(Some(Box::new(IoHandler { println_sym, readline_sym })));
            scope_map.insert("io".to_string(), io_idx);
        }

        let scope_pending: HashMap<String, Rc<RefCell<Vec<TermId>>>> = HashMap::new();

        // Build rule index: head symbol -> rule indices
        let mut rule_index: HashMap<SymId, Vec<usize>> = HashMap::new();
        for (i, rule) in rules.iter().enumerate() {
            if let Some(sym) = rule_head_sym(rule) {
                rule_index.entry(sym).or_default().push(i);
            }
        }

        // Resolve meta rules: map target_scope name -> handler index
        let meta_rules = parsed_meta.into_iter().map(|(rule, scope_name)| {
            MetaRule {
                rule,
                target_scope: scope_map.get(&scope_name).copied(),
            }
        }).collect();

        Engine {
            store, rules, rule_index, builtins, meta_syms, meta_rules,
            dynamic_rules: Vec::new(),
            dynamic_rules_map: HashMap::new(),
            scope_handlers, scope_map,
            active_scope: None,
            step_count: 0, eval_calls: 0, eval_depth: 0, in_meta: false,
            scope_pending, eval_cache: Vec::new(),
            cache_generation: 0, step_limit_exceeded: false,
        }
    }

    pub fn register_scope(&mut self, name: &str, handler: Box<dyn ScopeHandler>) {
        let idx = self.scope_handlers.len();
        self.scope_handlers.push(Some(handler));
        self.scope_map.insert(name.to_string(), idx);
    }

    /// Ensure a named scope exists, auto-creating a BrowserHandler if needed.
    /// Returns the scope index.
    fn ensure_scope(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.scope_map.get(name) {
            return idx;
        }
        let pending = self.scope_pending.entry(name.to_string())
            .or_insert_with(|| Rc::new(RefCell::new(Vec::new())))
            .clone();
        let idx = self.scope_handlers.len();
        self.scope_handlers.push(Some(Box::new(BrowserHandler::new(pending))));
        self.scope_map.insert(name.to_string(), idx);
        idx
    }

    const MAX_EVAL_CALLS: usize = 50_000_000;
    const MAX_EVAL_DEPTH: usize = 2_000;

    pub fn eval(&mut self, term: TermId) -> TermId {
        self.eval_calls += 1;
        self.eval_depth += 1;
        if self.eval_calls > Self::MAX_EVAL_CALLS || self.eval_depth > Self::MAX_EVAL_DEPTH {
            self.step_limit_exceeded = true;
            self.eval_depth -= 1;
            return term;
        }

        // Check cache — terms whose evaluation involved no side effects
        // are safe to reuse. The cache is invalidated whenever dynamic
        // rules change (rule()/retract()), and we only store results
        // when no invalidation happened during evaluation (generation check).
        let idx = term.0 as usize;
        if idx < self.eval_cache.len() {
            let cached = self.eval_cache[idx];
            if cached != u32::MAX {
                self.eval_depth -= 1;
                return TermId(cached);
            }
        }

        let gen_before = self.cache_generation;
        let result = self.eval_inner(term);

        // Only cache if no side effects occurred during this evaluation
        if self.cache_generation == gen_before {
            if idx >= self.eval_cache.len() {
                self.eval_cache.resize((idx + 256).max(self.eval_cache.len() * 2), u32::MAX);
            }
            self.eval_cache[idx] = result.0;
        }

        self.eval_depth -= 1;
        result
    }

    pub fn invalidate_cache(&mut self) {
        self.eval_cache.fill(u32::MAX);
        self.cache_generation = self.cache_generation.wrapping_add(1);
    }

    /// Reset eval counters for a new top-level evaluation.
    pub fn reset_eval_counters(&mut self) {
        self.eval_calls = 0;
        self.eval_depth = 0;
        self.step_limit_exceeded = false;
    }

    fn eval_inner(&mut self, term: TermId) -> TermId {
        let td = self.store.get(term);

        // 1. Innermost: eval head and args of Call terms (with special forms)
        let term = if let TermData::Call { head, args_start, args_len } = td {
            if let TermData::Sym(s) = self.store.get(head) {
                // quote(x) is a normal form — don't eval contents
                if s == self.builtins.quote {
                    return term;
                }
                // rule(lhs, rhs) — assert dynamic ground rule
                // Eval lhs args (to resolve dynamic vars like next_id → 1)
                // but don't fire rules on the lhs itself (to preserve keys like fib(5))
                if s == self.meta_syms.rule_sym && args_len == 2 {
                    let lhs_raw = self.store.args_pool[args_start as usize];
                    let lhs = self.eval_lhs(lhs_raw);
                    let rhs_raw = self.store.args_pool[args_start as usize + 1];
                    let rhs = self.eval(rhs_raw);
                    // Update existing or insert new
                    if self.dynamic_rules_map.contains_key(&lhs) {
                        self.dynamic_rules_map.insert(lhs, rhs);
                        if let Some(entry) = self.dynamic_rules.iter_mut().find(|(k, _)| *k == lhs) {
                            entry.1 = rhs;
                        }
                    } else {
                        self.dynamic_rules_map.insert(lhs, rhs);
                        self.dynamic_rules.push((lhs, rhs));
                    }
                    self.invalidate_cache();
                    return self.store.num(0);
                }
                // if(cond, then, else) — short-circuit
                if s == self.builtins.if_sym && args_len == 3 {
                    let cond_raw = self.store.args_pool[args_start as usize];
                    let cond = self.eval(cond_raw);
                    if let TermData::Sym(cs) = self.store.get(cond) {
                        if cs == self.builtins.true_sym {
                            let then_raw = self.store.args_pool[args_start as usize + 1];
                            return self.eval(then_raw);
                        }
                        if cs == self.builtins.false_sym {
                            let else_raw = self.store.args_pool[args_start as usize + 2];
                            return self.eval(else_raw);
                        }
                    }
                    // cond didn't reduce to true/false — return normal form
                    let then_raw = self.store.args_pool[args_start as usize + 1];
                    let else_raw = self.store.args_pool[args_start as usize + 2];
                    return self.store.call(head, &[cond, then_raw, else_raw]);
                }
                // retract(key) — remove a dynamic rule
                if s == self.builtins.retract && args_len == 1 {
                    let key_raw = self.store.args_pool[args_start as usize];
                    let key = self.eval_lhs(key_raw);
                    self.dynamic_rules_map.remove(&key);
                    self.dynamic_rules.retain(|&(k, _)| k != key);
                    self.invalidate_cache();
                    return self.store.num(0);
                }
                // query_all(tag) — scan dynamic rules, return cons-list
                if s == self.builtins.query_all && args_len == 1 {
                    let tag_raw = self.store.args_pool[args_start as usize];
                    let tag = self.eval(tag_raw);
                    // Get the SymId for matching
                    let tag_sym = if let TermData::Sym(ts) = self.store.get(tag) {
                        Some(ts)
                    } else {
                        None
                    };
                    let nil_sym = self.builtins.nil;
                    let nil_term = self.store.sym_term(nil_sym);
                    let cons_sym = self.builtins.cons;
                    let cons_head = self.store.sym_term(cons_sym);
                    // Collect matching values
                    let mut matches: Vec<TermId> = Vec::new();
                    for &(_, val) in &self.dynamic_rules {
                        if let Some(ts) = tag_sym {
                            if let TermData::Call { head: val_head, .. } = self.store.get(val) {
                                if let TermData::Sym(vs) = self.store.get(val_head) {
                                    if vs == ts {
                                        matches.push(val);
                                    }
                                }
                            }
                        }
                    }
                    // Build cons list (right fold)
                    let mut result = nil_term;
                    for val in matches.into_iter().rev() {
                        result = self.store.call(cons_head, &[val, result]);
                    }
                    return result;
                }
                // emit(scope_name, expr) — eval expr and send to named scope handler
                if s == self.builtins.emit && args_len == 2 {
                    let scope_name_raw = self.store.args_pool[args_start as usize];
                    let expr_raw = self.store.args_pool[args_start as usize + 1];
                    let val = self.eval(expr_raw);
                    // Get scope name
                    if let TermData::Sym(scope_sym) = self.store.get(scope_name_raw) {
                        let name = self.store.sym_name(scope_sym).to_string();
                        let scope_idx = self.ensure_scope(&name);
                        {
                            let mut handler = self.scope_handlers[scope_idx].take()
                                .expect("Scope handler already borrowed");
                            let result = handler.try_eval(&mut self.store, val);
                            self.scope_handlers[scope_idx] = Some(handler);
                            return result.unwrap_or(val);
                        }
                    }
                    return val;
                }
                // __rule_decl(name, from, to, __clause(lhs, rhs), ...)
                // Install a dynamic rule declaration — don't eval clause args
                if s == self.meta_syms.rule_decl_sym && args_len >= 3 {
                    return self.install_rule_decl(args_start, args_len);
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

        // 3. Dynamic rules (asserted at runtime) — O(1) HashMap lookup
        if let Some(&val) = self.dynamic_rules_map.get(&term) {
            return val;
        }

        // 4. Try user rules — indexed by head symbol
        let quote_sym = self.builtins.quote;
        if let TermData::Call { head, .. } = self.store.get(term) {
            if let TermData::Sym(head_sym) = self.store.get(head) {
                if let Some(indices) = self.rule_index.get(&head_sym) {
                    for &rule_idx in indices {
                        let num_clauses = self.rules[rule_idx].clauses.len();
                        for clause_idx in 0..num_clauses {
                            let mut env = Env::new();
                            if match_pattern(&self.store, &self.rules[rule_idx].clauses[clause_idx].lhs, term, &mut env, quote_sym) {
                                let result = substitute(&mut self.store, &self.rules[rule_idx].clauses[clause_idx].rhs, &env);
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
                }
            }
        }

        // 5. Try builtins (arithmetic, comparison, string)
        let td = self.store.get(term);
        if let TermData::Call { head, args_start, args_len } = td {
            if let TermData::Sym(s) = self.store.get(head) {
                // Binary builtins (arity 2)
                if args_len == 2 {
                    let a = self.store.args_pool[args_start as usize];
                    let b = self.store.args_pool[args_start as usize + 1];

                    // Extract numeric values (i64 or f64)
                    let a_num = match self.store.get(a) {
                        TermData::Num(n) => Some((n as f64, true)),
                        TermData::Float(bits) => Some((f64::from_bits(bits), false)),
                        _ => None,
                    };
                    let b_num = match self.store.get(b) {
                        TermData::Num(n) => Some((n as f64, true)),
                        TermData::Float(bits) => Some((f64::from_bits(bits), false)),
                        _ => None,
                    };

                    if let (Some((av, a_int)), Some((bv, b_int))) = (a_num, b_num) {
                        let both_int = a_int && b_int;

                        // Arithmetic
                        let arith_result = if s == self.builtins.add {
                            Some((av + bv, self.builtins.add))
                        } else if s == self.builtins.sub {
                            Some((av - bv, self.builtins.sub))
                        } else if s == self.builtins.mul {
                            Some((av * bv, self.builtins.mul))
                        } else if s == self.builtins.div {
                            Some((av / bv, self.builtins.div))
                        } else if s == self.builtins.mod_sym {
                            if both_int {
                                Some(((av as i64 % bv as i64) as f64, self.builtins.mod_sym))
                            } else {
                                Some((av.rem_euclid(bv), self.builtins.mod_sym))
                            }
                        } else {
                            None
                        };
                        if let Some((r, op_sym)) = arith_result {
                            let result_term = if both_int {
                                self.store.num(r as i64)
                            } else {
                                self.store.float(r)
                            };
                            if !self.in_meta {
                                self.step_count += 1;
                                if !self.meta_rules.is_empty() {
                                    let op_term = self.store.sym_term(op_sym);
                                    let builtin_head = self.store.sym_term(self.meta_syms.builtin_sym);
                                    let kind = self.store.call(builtin_head, &[op_term]);
                                    self.fire_meta(term, result_term, kind);
                                }
                            }
                            return result_term;
                        }

                        // Numeric comparisons
                        let cmp_result = if s == self.builtins.eq {
                            Some(av == bv)
                        } else if s == self.builtins.neq {
                            Some(av != bv)
                        } else if s == self.builtins.lt {
                            Some(av < bv)
                        } else if s == self.builtins.gt {
                            Some(av > bv)
                        } else if s == self.builtins.lte {
                            Some(av <= bv)
                        } else if s == self.builtins.gte {
                            Some(av >= bv)
                        } else {
                            None
                        };
                        if let Some(b) = cmp_result {
                            let sym = if b { self.builtins.true_sym } else { self.builtins.false_sym };
                            return self.store.sym_term(sym);
                        }
                    }

                    // eq/neq on any terms (structural equality via hash-consing)
                    if s == self.builtins.eq {
                        let sym = if a == b { self.builtins.true_sym } else { self.builtins.false_sym };
                        return self.store.sym_term(sym);
                    }
                    if s == self.builtins.neq {
                        let sym = if a != b { self.builtins.true_sym } else { self.builtins.false_sym };
                        return self.store.sym_term(sym);
                    }

                    // str_concat(a, b) — concatenate symbol names, auto-coerce nums/floats
                    if s == self.builtins.str_concat {
                        let sa = match self.store.get(a) {
                            TermData::Sym(s) => self.store.sym_name(s).to_string(),
                            TermData::Num(n) => n.to_string(),
                            TermData::Float(bits) => format!("{}", f64::from_bits(bits)),
                            _ => return term,
                        };
                        let sb = match self.store.get(b) {
                            TermData::Sym(s) => self.store.sym_name(s).to_string(),
                            TermData::Num(n) => n.to_string(),
                            TermData::Float(bits) => format!("{}", f64::from_bits(bits)),
                            _ => return term,
                        };
                        let combined = format!("{}{}", sa, sb);
                        let sym = self.store.sym(&combined);
                        return self.store.sym_term(sym);
                    }
                }

                // Unary builtins (arity 1)
                if args_len == 1 {
                    let a = self.store.args_pool[args_start as usize];

                    // str_len(a)
                    if s == self.builtins.str_len {
                        if let TermData::Sym(sym) = self.store.get(a) {
                            let len = self.store.sym_name(sym).len() as i64;
                            return self.store.num(len);
                        }
                    }

                    // abs(x)
                    if s == self.builtins.abs_sym {
                        match self.store.get(a) {
                            TermData::Num(n) => return self.store.num(n.abs()),
                            TermData::Float(bits) => return self.store.float(f64::from_bits(bits).abs()),
                            _ => {}
                        }
                    }

                    // floor(x) — Float → Num
                    if s == self.builtins.floor_sym {
                        match self.store.get(a) {
                            TermData::Num(_) => return a,
                            TermData::Float(bits) => return self.store.num(f64::from_bits(bits).floor() as i64),
                            _ => {}
                        }
                    }

                    // random(seed) — deterministic [0, 1) from seed term id
                    if s == self.builtins.random_sym {
                        let seed = a.0 as u64;
                        let hash = splitmix64(seed);
                        let f = (hash >> 11) as f64 / (1u64 << 53) as f64;
                        return self.store.float(f);
                    }
                }
            }
        }

        // 6. Try active scope handler
        if let Some(scope_idx) = self.active_scope {
            let mut handler = self.scope_handlers[scope_idx].take()
                .expect("Scope handler already borrowed");
            let result = handler.try_eval(&mut self.store, term);
            self.scope_handlers[scope_idx] = Some(handler);
            if let Some(val) = result {
                return val;
            }
        }

        // 7. Normal form
        term
    }

    /// Install a dynamic rule from a __rule_decl term.
    /// Args: name(sym), from_scope(sym), to_scope(sym), __clause(lhs, rhs), ...
    fn install_rule_decl(&mut self, args_start: u32, args_len: u16) -> TermId {
        let name_term = self.store.args_pool[args_start as usize];
        let from_term = self.store.args_pool[args_start as usize + 1];
        let to_term = self.store.args_pool[args_start as usize + 2];

        let name_str = if let TermData::Sym(s) = self.store.get(name_term) {
            self.store.sym_name(s).to_string()
        } else {
            "dynamic_rule".to_string()
        };

        let from_str = if let TermData::Sym(s) = self.store.get(from_term) {
            self.store.sym_name(s).to_string()
        } else {
            String::new()
        };

        let to_str = if let TermData::Sym(s) = self.store.get(to_term) {
            self.store.sym_name(s).to_string()
        } else {
            String::new()
        };

        let is_meta = from_str == "meta";

        // Extract clauses from __clause(lhs, rhs) terms
        let mut clauses = Vec::new();
        for i in 3..args_len as usize {
            let clause_term = self.store.args_pool[args_start as usize + i];
            if let TermData::Call { args_start: cl_start, args_len: 2, .. } = self.store.get(clause_term) {
                let lhs_term = self.store.args_pool[cl_start as usize];
                let rhs_term = self.store.args_pool[cl_start as usize + 1];
                let lhs_pat = self.term_to_pattern(lhs_term);
                let rhs_pat = self.term_to_pattern(rhs_term);
                clauses.push(Clause { lhs: lhs_pat, rhs: rhs_pat });
            }
        }

        let rule = Rule { name: name_str, clauses };

        if is_meta {
            let target_scope = Some(self.ensure_scope(&to_str));
            self.meta_rules.push(MetaRule { rule, target_scope });
        } else {
            let idx = self.rules.len();
            if let Some(sym) = rule_head_sym(&rule) {
                self.rule_index.entry(sym).or_default().push(idx);
            }
            self.rules.push(rule);
        }

        self.store.num(0)
    }

    /// Convert a TermId back to a Pattern, replacing __pvar(N) with Var(N).
    fn term_to_pattern(&self, term: TermId) -> Pattern {
        match self.store.get(term) {
            TermData::Num(n) => Pattern::Num(n),
            TermData::Float(bits) => Pattern::Float(bits),
            TermData::Sym(s) => Pattern::Sym(s),
            TermData::Call { head, args_start, args_len } => {
                // __pvar(N) → Pattern::Var(VarId(N))
                if let TermData::Sym(s) = self.store.get(head) {
                    if s == self.meta_syms.pvar_sym && args_len == 1 {
                        let arg = self.store.args_pool[args_start as usize];
                        if let TermData::Num(n) = self.store.get(arg) {
                            return Pattern::Var(VarId(n as u32));
                        }
                    }
                }
                let head_pat = self.term_to_pattern(head);
                let args_pat: Vec<Pattern> = (0..args_len as usize)
                    .map(|i| self.term_to_pattern(self.store.args_pool[args_start as usize + i]))
                    .collect();
                Pattern::Call(Box::new(head_pat), args_pat)
            }
        }
    }

    /// Evaluate the arguments of a Call term (to resolve dynamic vars like `next_id → 1`)
    /// but don't fire rules/builtins on the outer Call itself (to preserve keys like fib(5)).
    /// Non-Call terms (Sym, Num) are returned as-is — they're already ground keys.
    fn eval_lhs(&mut self, term: TermId) -> TermId {
        let td = self.store.get(term);
        if let TermData::Call { head, args_start, args_len } = td {
            let len = args_len as usize;
            let mut buf = [TermId(0); 16];
            let mut changed = false;
            for i in 0..len {
                let arg = self.store.args_pool[args_start as usize + i];
                buf[i] = self.eval(arg);
                changed |= buf[i] != arg;
            }
            if changed {
                self.store.call(head, &buf[..len])
            } else {
                term
            }
        } else {
            // Sym or Num — keep as-is (these are ground keys like `next_id`, `filter`)
            term
        }
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
            let num_clauses = self.meta_rules[i].rule.clauses.len();
            for j in 0..num_clauses {
                let mut env = Env::new();
                if match_pattern(&self.store, &self.meta_rules[i].rule.clauses[j].lhs, event, &mut env, quote_sym) {
                    let result = substitute(&mut self.store, &self.meta_rules[i].rule.clauses[j].rhs, &env);
                    // Eval the RHS normally (no scope), then send result to the target scope.
                    // Setting active_scope during eval would cause the scope handler to
                    // intercept every sub-expression, poisoning the eval cache.
                    let val = self.eval(result);
                    if let Some(scope_idx) = self.meta_rules[i].target_scope {
                        let mut handler = self.scope_handlers[scope_idx].take()
                            .expect("Scope handler already borrowed");
                        handler.try_eval(&mut self.store, val);
                        self.scope_handlers[scope_idx] = Some(handler);
                    }
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
            let num_clauses = self.meta_rules[i].rule.clauses.len();
            for j in 0..num_clauses {
                let mut env = Env::new();
                if match_pattern(&self.store, &self.meta_rules[i].rule.clauses[j].lhs, event, &mut env, quote_sym) {
                    let result = substitute(&mut self.store, &self.meta_rules[i].rule.clauses[j].rhs, &env);
                    // Eval the RHS normally (no scope), then send result to the target scope.
                    let val = self.eval(result);
                    if let Some(scope_idx) = self.meta_rules[i].target_scope {
                        let mut handler = self.scope_handlers[scope_idx].take()
                            .expect("Scope handler already borrowed");
                        handler.try_eval(&mut self.store, val);
                        self.scope_handlers[scope_idx] = Some(handler);
                    }
                    break;
                }
            }
        }
        self.in_meta = false;
    }

    // ── Public API ──

    pub fn load_program(&mut self, src: &str) -> TermId {
        use crate::parser::{Lexer, Parser};
        let tokens = Lexer::new(src).tokenize();
        let program = Parser::new(tokens, &mut self.store).parse_program();
        // Install rules and update index
        let base = self.rules.len();
        for (i, rule) in program.rules.iter().enumerate() {
            if let Some(sym) = rule_head_sym(rule) {
                self.rule_index.entry(sym).or_default().push(base + i);
            }
        }
        self.rules.extend(program.rules);
        // Install meta rules (auto-create target scopes)
        for (rule, scope_name) in program.meta_rules {
            let target_scope = Some(self.ensure_scope(&scope_name));
            self.meta_rules.push(MetaRule { rule, target_scope });
        }
        pattern_to_term(&mut self.store, &program.expr)
    }

    pub fn make_sym(&mut self, name: &str) -> TermId {
        self.store.symbol(name)
    }

    pub fn make_num(&mut self, n: i64) -> TermId {
        self.store.num(n)
    }

    pub fn make_float(&mut self, f: f64) -> TermId {
        self.store.float(f)
    }

    pub fn make_call(&mut self, head: TermId, args: &[TermId]) -> TermId {
        self.store.call(head, args)
    }

    pub fn term_tag(&self, id: TermId) -> u8 {
        match self.store.get(id) {
            TermData::Num(_) => 0,
            TermData::Sym(_) => 1,
            TermData::Call { .. } => 2,
            TermData::Float(_) => 3,
        }
    }

    pub fn term_num(&self, id: TermId) -> i64 {
        match self.store.get(id) {
            TermData::Num(n) => n,
            _ => panic!("term_num called on non-Num term"),
        }
    }

    pub fn term_float(&self, id: TermId) -> f64 {
        match self.store.get(id) {
            TermData::Float(bits) => f64::from_bits(bits),
            _ => panic!("term_float called on non-Float term"),
        }
    }

    pub fn term_sym_name(&self, id: TermId) -> &str {
        match self.store.get(id) {
            TermData::Sym(s) => self.store.sym_name(s),
            _ => panic!("term_sym_name called on non-Sym term"),
        }
    }

    pub fn term_call_head(&self, id: TermId) -> TermId {
        match self.store.get(id) {
            TermData::Call { head, .. } => head,
            _ => panic!("term_call_head called on non-Call term"),
        }
    }

    pub fn term_call_arity(&self, id: TermId) -> usize {
        match self.store.get(id) {
            TermData::Call { args_len, .. } => args_len as usize,
            _ => panic!("term_call_arity called on non-Call term"),
        }
    }

    pub fn term_call_arg(&self, id: TermId, idx: usize) -> TermId {
        match self.store.get(id) {
            TermData::Call { args_start, args_len, .. } => {
                assert!(idx < args_len as usize, "arg index out of bounds");
                self.store.args_pool[args_start as usize + idx]
            }
            _ => panic!("term_call_arg called on non-Call term"),
        }
    }

    pub fn display(&self, id: TermId) -> String {
        format!("{}", self.store.display(id))
    }

    // ── Generic scope access ──

    pub fn scope_pending_count(&self, name: &str) -> usize {
        self.scope_pending.get(name).map_or(0, |v| v.borrow().len())
    }

    pub fn scope_pending_get(&self, name: &str, idx: usize) -> TermId {
        self.scope_pending[name].borrow()[idx]
    }

    pub fn scope_pending_clear(&self, name: &str) {
        if let Some(v) = self.scope_pending.get(name) {
            v.borrow_mut().clear();
        }
    }

    pub fn scope_take_pending(&self, name: &str) -> Vec<TermId> {
        self.scope_pending.get(name)
            .map(|v| std::mem::take(&mut *v.borrow_mut()))
            .unwrap_or_default()
    }
}
