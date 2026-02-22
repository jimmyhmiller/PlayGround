use std::collections::HashMap;
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
    pub str_join_sym: SymId,
    pub list_reverse_sym: SymId,
    pub list_len_sym: SymId,
    // Collection types
    pub vec_sym: SymId,
    pub cells_sym: SymId,
    pub set_sym: SymId,
    pub map_sym: SymId,
    pub entry_sym: SymId,
    // Vec builtins
    pub vec_get: SymId,
    pub vec_len: SymId,
    pub vec_push: SymId,
    pub vec_concat: SymId,
    pub vec_slice: SymId,
    // Set builtins
    pub set_contains: SymId,
    pub set_insert: SymId,
    pub set_remove: SymId,
    pub set_union: SymId,
    pub set_intersect: SymId,
    pub set_len: SymId,
    // Map builtins
    pub map_get: SymId,
    pub map_set: SymId,
    pub map_remove: SymId,
    pub map_merge: SymId,
    pub map_keys: SymId,
    pub map_values: SymId,
    pub map_len: SymId,
    // Term construction / clause installation
    pub make_term_sym: SymId,
    pub defclause_sym: SymId,
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
    pub spread_sym: SymId,
    pub wild_spread_sym: SymId,
    pub wild_sym: SymId,
}

// ── MetaRule ──

pub struct MetaRule {
    pub rule: Rule,
    pub target_scope: Option<usize>, // index into scopes
}

fn entry_key_static(store: &TermStore, term: TermId, entry_sym: SymId) -> Option<TermId> {
    if let TermData::Call { head, args_start, args_len: 2 } = store.get(term) {
        if let TermData::Sym(s) = store.get(head) {
            if s == entry_sym {
                return Some(store.args_pool[args_start as usize]);
            }
        }
    }
    None
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
    pattern_ctx: PatternContext,
    // Scopes: index 0 is always the "main" scope
    scopes: Vec<Scope>,
    scope_map: HashMap<String, usize>,
    active_scope_idx: Option<usize>,
    main_scope: usize,
    // Eval state (global, not per-scope)
    pub step_count: usize,
    pub eval_calls: usize,
    eval_depth: usize,
    in_meta: bool,
    eval_cache: Vec<u64>,       // packed (generation:u32 << 32 | result:u32)
    cache_generation: u32,      // bumped on every invalidation
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
            str_join_sym: store.sym("str_join"),
            list_reverse_sym: store.sym("list_reverse"),
            list_len_sym: store.sym("list_len"),
            // Collection types
            vec_sym: store.sym("vec"),
            cells_sym: store.sym("cells"),
            set_sym: store.sym("set"),
            map_sym: store.sym("map"),
            entry_sym: store.sym("entry"),
            // Vec builtins
            vec_get: store.sym("vec_get"),
            vec_len: store.sym("vec_len"),
            vec_push: store.sym("vec_push"),
            vec_concat: store.sym("vec_concat"),
            vec_slice: store.sym("vec_slice"),
            // Set builtins
            set_contains: store.sym("set_contains"),
            set_insert: store.sym("set_insert"),
            set_remove: store.sym("set_remove"),
            set_union: store.sym("set_union"),
            set_intersect: store.sym("set_intersect"),
            set_len: store.sym("set_len"),
            // Map builtins
            map_get: store.sym("map_get"),
            map_set: store.sym("map_set"),
            map_remove: store.sym("map_remove"),
            map_merge: store.sym("map_merge"),
            map_keys: store.sym("map_keys"),
            map_values: store.sym("map_values"),
            map_len: store.sym("map_len"),
            // Term construction / clause installation
            make_term_sym: store.sym("make_term"),
            defclause_sym: store.sym("defclause"),
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
            spread_sym: store.sym("__spread"),
            wild_spread_sym: store.sym("__wild_spread"),
            wild_sym: store.sym("__wild"),
        };

        let pattern_ctx = PatternContext {
            quote_sym: builtins.quote,
            add_sym: builtins.add,
            sub_sym: builtins.sub,
            mul_sym: builtins.mul,
            div_sym: builtins.div,
            mod_sym: builtins.mod_sym,
        };

        // Create scopes vec with "main" at index 0
        let mut scopes: Vec<Scope> = Vec::new();
        let mut scope_map: HashMap<String, usize> = HashMap::new();

        let main_scope = Scope::new("main".to_string(), false);
        scopes.push(main_scope);
        scope_map.insert("main".to_string(), 0);

        // Register built-in @io scope (only on non-wasm)
        #[cfg(not(target_arch = "wasm32"))]
        {
            let println_sym = store.sym("println");
            let readline_sym = store.sym("readline");
            let io_idx = scopes.len();
            let mut io_scope = Scope::new("io".to_string(), true);
            io_scope.handler = Some(Box::new(IoHandler { println_sym, readline_sym }));
            scopes.push(io_scope);
            scope_map.insert("io".to_string(), io_idx);
        }

        // Build rule index: head symbol -> rule indices
        let mut rule_index: HashMap<SymId, Vec<usize>> = HashMap::new();
        for (i, rule) in rules.iter().enumerate() {
            if let Some(sym) = rule_head_sym(rule) {
                rule_index.entry(sym).or_default().push(i);
            }
        }

        // Resolve meta rules: map target_scope name -> scope index
        let meta_rules = parsed_meta.into_iter().map(|(rule, scope_name)| {
            let target_scope = if let Some(&idx) = scope_map.get(&scope_name) {
                Some(idx)
            } else {
                // Auto-create output scope for meta rule targets
                let idx = scopes.len();
                let mut scope = Scope::new(scope_name.clone(), true);
                let pending = scope.pending.clone();
                scope.handler = Some(Box::new(BrowserHandler::new(pending)));
                scopes.push(scope);
                scope_map.insert(scope_name, idx);
                Some(idx)
            };
            MetaRule { rule, target_scope }
        }).collect();

        Engine {
            store, rules, rule_index, builtins, meta_syms, meta_rules,
            pattern_ctx,
            scopes, scope_map,
            active_scope_idx: None,
            main_scope: 0,
            step_count: 0, eval_calls: 0, eval_depth: 0, in_meta: false,
            eval_cache: Vec::new(),
            cache_generation: 1, step_limit_exceeded: false,
        }
    }

    /// Get the index of the currently active scope (main if none set).
    #[inline]
    fn current_scope_idx(&self) -> usize {
        self.active_scope_idx.unwrap_or(self.main_scope)
    }

    pub fn register_scope(&mut self, name: &str, handler: Box<dyn ScopeHandler>) {
        if let Some(&idx) = self.scope_map.get(name) {
            self.scopes[idx].handler = Some(handler);
        } else {
            let idx = self.scopes.len();
            let mut scope = Scope::new(name.to_string(), true);
            scope.handler = Some(handler);
            self.scopes.push(scope);
            self.scope_map.insert(name.to_string(), idx);
        }
    }

    /// Ensure a named scope exists. `output_only` controls whether new scopes
    /// are buffer-only (true, gets a BrowserHandler) or full actors (false).
    /// Returns the scope index.
    fn ensure_scope(&mut self, name: &str, output_only: bool) -> usize {
        if let Some(&idx) = self.scope_map.get(name) {
            return idx;
        }
        let idx = self.scopes.len();
        let mut scope = Scope::new(name.to_string(), output_only);
        if output_only {
            let pending = scope.pending.clone();
            scope.handler = Some(Box::new(BrowserHandler::new(pending)));
        }
        self.scopes.push(scope);
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

        // Check cache — packed (generation:u32 << 32 | result:u32).
        // A hit requires the stored generation to match the current one.
        let idx = term.0 as usize;
        if idx < self.eval_cache.len() {
            let entry = self.eval_cache[idx];
            let stored_gen = (entry >> 32) as u32;
            if stored_gen == self.cache_generation {
                let cached = entry as u32;
                self.eval_depth -= 1;
                return TermId(cached);
            }
        }

        let gen_before = self.cache_generation;
        let result = self.eval_inner(term);

        // Only cache if no side effects occurred during this evaluation
        if self.cache_generation == gen_before {
            if idx >= self.eval_cache.len() {
                self.eval_cache.resize((idx + 256).max(self.eval_cache.len() * 2), 0);
            }
            self.eval_cache[idx] = ((self.cache_generation as u64) << 32) | (result.0 as u64);
        }

        self.eval_depth -= 1;
        result
    }

    pub fn invalidate_cache(&mut self) {
        // Skip generation 0 — it's the sentinel for "uninitialized" cache entries
        let next = self.cache_generation.wrapping_add(1);
        self.cache_generation = if next == 0 { 1 } else { next };
    }

    /// Reset eval counters for a new top-level evaluation.
    pub fn reset_eval_counters(&mut self) {
        self.eval_calls = 0;
        self.eval_depth = 0;
        self.step_count = 0;
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
                // rule(lhs, rhs) — assert dynamic ground rule in current scope
                if s == self.meta_syms.rule_sym && args_len == 2 {
                    let lhs_raw = self.store.args_pool[args_start as usize];
                    let lhs = self.eval_lhs(lhs_raw);
                    let rhs_raw = self.store.args_pool[args_start as usize + 1];
                    let rhs = self.eval(rhs_raw);
                    let sidx = self.current_scope_idx();
                    if self.scopes[sidx].dynamic_rules_map.contains_key(&lhs) {
                        self.scopes[sidx].dynamic_rules_map.insert(lhs, rhs);
                        if let Some(entry) = self.scopes[sidx].dynamic_rules.iter_mut().find(|(k, _)| *k == lhs) {
                            entry.1 = rhs;
                        }
                    } else {
                        self.scopes[sidx].dynamic_rules_map.insert(lhs, rhs);
                        self.scopes[sidx].dynamic_rules.push((lhs, rhs));
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
                // retract(key) — remove a dynamic rule from current scope
                if s == self.builtins.retract && args_len == 1 {
                    let key_raw = self.store.args_pool[args_start as usize];
                    let key = self.eval_lhs(key_raw);
                    let sidx = self.current_scope_idx();
                    self.scopes[sidx].dynamic_rules_map.remove(&key);
                    self.scopes[sidx].dynamic_rules.retain(|&(k, _)| k != key);
                    self.invalidate_cache();
                    return self.store.num(0);
                }
                // query_all(tag) — scan dynamic rules in current scope
                if s == self.builtins.query_all && args_len == 1 {
                    let tag_raw = self.store.args_pool[args_start as usize];
                    let tag = self.eval(tag_raw);
                    let tag_sym = if let TermData::Sym(ts) = self.store.get(tag) {
                        Some(ts)
                    } else {
                        None
                    };
                    let nil_sym = self.builtins.nil;
                    let nil_term = self.store.sym_term(nil_sym);
                    let cons_sym = self.builtins.cons;
                    let cons_head = self.store.sym_term(cons_sym);
                    let sidx = self.current_scope_idx();
                    let mut matches: Vec<TermId> = Vec::new();
                    for &(_, val) in &self.scopes[sidx].dynamic_rules {
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
                    let mut result = nil_term;
                    for val in matches.into_iter().rev() {
                        result = self.store.call(cons_head, &[val, result]);
                    }
                    return result;
                }
                // emit(scope_name, expr) — send expr to named scope
                if s == self.builtins.emit && args_len == 2 {
                    let scope_name_raw = self.store.args_pool[args_start as usize];
                    let expr_raw = self.store.args_pool[args_start as usize + 1];
                    if let TermData::Sym(scope_sym) = self.store.get(scope_name_raw) {
                        let name = self.store.sym_name(scope_sym).to_string();
                        // New scopes from emit are actor scopes (output_only = false)
                        let scope_idx = self.ensure_scope(&name, false);
                        if self.scopes[scope_idx].output_only {
                            // Output-only scope: eval first, then buffer via handler
                            let val = self.eval(expr_raw);
                            let mut handler = self.scopes[scope_idx].handler.take()
                                .expect("Scope handler already borrowed");
                            let result = handler.try_eval(&mut self.store, val);
                            self.scopes[scope_idx].handler = Some(handler);
                            return result.unwrap_or(val);
                        } else {
                            // Actor scope: push raw expression to work queue.
                            // tick() will evaluate it in the target scope's context.
                            self.scopes[scope_idx].work_queue.push_back(expr_raw);
                            return self.store.num(0);
                        }
                    }
                    let val = self.eval(expr_raw);
                    return val;
                }
                // __rule_decl(name, from, to, __clause(lhs, rhs), ...)
                if s == self.meta_syms.rule_decl_sym && args_len >= 3 {
                    return self.install_rule_decl(args_start, args_len);
                }
                // vec(...) — eval args using Vec (no 16-element limit)
                if s == self.builtins.vec_sym {
                    let len = args_len as usize;
                    let mut evaled: Vec<TermId> = Vec::with_capacity(len);
                    let mut changed = false;
                    for i in 0..len {
                        let arg = self.store.args_pool[args_start as usize + i];
                        let v = self.eval(arg);
                        changed |= v != arg;
                        evaled.push(v);
                    }
                    if changed {
                        return self.store.call(head, &evaled);
                    } else {
                        return term;
                    }
                }
                // set(...) — eval args, sort, dedup for canonical form
                if s == self.builtins.set_sym {
                    let len = args_len as usize;
                    let mut evaled: Vec<TermId> = Vec::with_capacity(len);
                    for i in 0..len {
                        let arg = self.store.args_pool[args_start as usize + i];
                        evaled.push(self.eval(arg));
                    }
                    evaled.sort_by(|a, b| self.store.term_cmp(*a, *b));
                    evaled.dedup();
                    return self.store.call(head, &evaled);
                }
                // map(entry(k,v), ...) — eval args, sort by key, dedup by key (last wins)
                if s == self.builtins.map_sym {
                    let len = args_len as usize;
                    let entry_sym = self.builtins.entry_sym;
                    let mut evaled: Vec<TermId> = Vec::with_capacity(len);
                    for i in 0..len {
                        let arg = self.store.args_pool[args_start as usize + i];
                        evaled.push(self.eval(arg));
                    }
                    // Extract keys for sorting
                    let keys: Vec<Option<TermId>> = evaled.iter().map(|t| entry_key_static(&self.store, *t, entry_sym)).collect();
                    // Sort entries by key using indices
                    let mut indices: Vec<usize> = (0..evaled.len()).collect();
                    indices.sort_by(|&ai, &bi| {
                        match (keys[ai], keys[bi]) {
                            (Some(ka), Some(kb)) => self.store.term_cmp(ka, kb),
                            _ => std::cmp::Ordering::Equal,
                        }
                    });
                    let sorted: Vec<TermId> = indices.iter().map(|&i| evaled[i]).collect();
                    // Dedup by key (keep last)
                    let mut deduped: Vec<TermId> = Vec::with_capacity(sorted.len());
                    for entry in sorted {
                        if let Some(last) = deduped.last() {
                            let lk = entry_key_static(&self.store, *last, entry_sym);
                            let ek = entry_key_static(&self.store, entry, entry_sym);
                            if let (Some(lk), Some(ek)) = (lk, ek) {
                                if lk == ek {
                                    *deduped.last_mut().unwrap() = entry;
                                    continue;
                                }
                            }
                        }
                        deduped.push(entry);
                    }
                    return self.store.call(head, &deduped);
                }
            }
            let h = self.eval(head);
            let len = args_len as usize;
            let mut evaled: Vec<TermId> = Vec::with_capacity(len);
            let mut changed = h != head;
            for i in 0..len {
                let arg = self.store.args_pool[args_start as usize + i];
                let v = self.eval(arg);
                changed |= v != arg;
                evaled.push(v);
            }
            if changed {
                self.store.call(h, &evaled)
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

        // 3. Dynamic rules (current scope) — O(1) HashMap lookup
        {
            let sidx = self.current_scope_idx();
            if let Some(&val) = self.scopes[sidx].dynamic_rules_map.get(&term) {
                return val;
            }
        }

        // 4. Try user rules — indexed by head symbol
        if let TermData::Call { head, .. } = self.store.get(term) {
            if let TermData::Sym(head_sym) = self.store.get(head) {
                if let Some(indices) = self.rule_index.get(&head_sym).cloned() {
                    for &rule_idx in &indices {
                        let num_clauses = self.rules[rule_idx].clauses.len();
                        for clause_idx in 0..num_clauses {
                            let mut env = Env::new();
                            if match_pattern(&mut self.store, &self.rules[rule_idx].clauses[clause_idx].lhs, term, &mut env, &self.pattern_ctx) {
                                // Evaluate where-guard conditions (all must be true)
                                let num_guards = self.rules[rule_idx].clauses[clause_idx].guards.len();
                                let mut guards_pass = true;
                                for guard_idx in 0..num_guards {
                                    let guard_term = substitute(&mut self.store, &self.rules[rule_idx].clauses[clause_idx].guards[guard_idx], &env);
                                    let guard_result = self.eval(guard_term);
                                    if let TermData::Sym(s) = self.store.get(guard_result) {
                                        if s != self.builtins.true_sym {
                                            guards_pass = false;
                                            break;
                                        }
                                    } else {
                                        guards_pass = false;
                                        break;
                                    }
                                }
                                if !guards_pass { continue; }
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

                    // str_join(list_or_vec) — concatenate values into one string
                    if s == self.builtins.str_join_sym {
                        // Try vec first
                        if let TermData::Call { head: vh, args_start: vas, args_len: vlen } = self.store.get(a) {
                            if let TermData::Sym(vhs) = self.store.get(vh) {
                                if vhs == self.builtins.vec_sym {
                                    let mut buf = String::new();
                                    for i in 0..vlen as usize {
                                        let elem = self.store.args_pool[vas as usize + i];
                                        match self.store.get(elem) {
                                            TermData::Sym(s) => buf.push_str(self.store.sym_name(s)),
                                            TermData::Num(n) => { use std::fmt::Write; let _ = write!(buf, "{}", n); }
                                            TermData::Float(bits) => { use std::fmt::Write; let _ = write!(buf, "{}", f64::from_bits(bits)); }
                                            _ => return term,
                                        }
                                    }
                                    return self.store.transient_sym_term(buf);
                                }
                            }
                        }
                        // Fall back to cons-list
                        let nil_sym = self.builtins.nil;
                        let cons_sym = self.builtins.cons;
                        let mut buf = String::new();
                        let mut cur = a;
                        loop {
                            match self.store.get(cur) {
                                TermData::Sym(sym) if sym == nil_sym => break,
                                TermData::Call { head, args_start, args_len } if args_len == 2 => {
                                    if let TermData::Sym(hs) = self.store.get(head) {
                                        if hs == cons_sym {
                                            let elem = self.store.args_pool[args_start as usize];
                                            match self.store.get(elem) {
                                                TermData::Sym(s) => buf.push_str(self.store.sym_name(s)),
                                                TermData::Num(n) => { use std::fmt::Write; let _ = write!(buf, "{}", n); }
                                                TermData::Float(bits) => { use std::fmt::Write; let _ = write!(buf, "{}", f64::from_bits(bits)); }
                                                _ => return term,
                                            }
                                            cur = self.store.args_pool[args_start as usize + 1];
                                            continue;
                                        }
                                    }
                                    return term;
                                }
                                _ => return term,
                            }
                        }
                        return self.store.transient_sym_term(buf);
                    }

                    // list_reverse(list_or_vec) — reverse natively
                    if s == self.builtins.list_reverse_sym {
                        // Try vec first
                        if let TermData::Call { head: vh, args_start: vas, args_len: vlen } = self.store.get(a) {
                            if let TermData::Sym(vhs) = self.store.get(vh) {
                                if vhs == self.builtins.vec_sym {
                                    let mut elems: Vec<TermId> = (0..vlen as usize)
                                        .map(|i| self.store.args_pool[vas as usize + i])
                                        .collect();
                                    elems.reverse();
                                    return self.store.call(vh, &elems);
                                }
                            }
                        }
                        // Fall back to cons-list
                        let nil_sym = self.builtins.nil;
                        let cons_sym = self.builtins.cons;
                        let cons_head = self.store.sym_term(cons_sym);
                        let mut elems: Vec<TermId> = Vec::new();
                        let mut cur = a;
                        loop {
                            match self.store.get(cur) {
                                TermData::Sym(sym) if sym == nil_sym => break,
                                TermData::Call { head, args_start, args_len } if args_len == 2 => {
                                    if let TermData::Sym(hs) = self.store.get(head) {
                                        if hs == cons_sym {
                                            elems.push(self.store.args_pool[args_start as usize]);
                                            cur = self.store.args_pool[args_start as usize + 1];
                                            continue;
                                        }
                                    }
                                    return term;
                                }
                                _ => return term,
                            }
                        }
                        let mut result = self.store.sym_term(nil_sym);
                        for elem in elems {
                            result = self.store.call(cons_head, &[elem, result]);
                        }
                        return result;
                    }

                    // list_len(list) — count elements of a cons-list natively
                    if s == self.builtins.list_len_sym {
                        let nil_sym = self.builtins.nil;
                        let cons_sym = self.builtins.cons;
                        let vec_sym = self.builtins.vec_sym;
                        let mut count: i64 = 0;
                        let mut cur = a;
                        // Try vec first
                        if let TermData::Call { head, args_len, .. } = self.store.get(cur) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == vec_sym {
                                    return self.store.num(args_len as i64);
                                }
                            }
                        }
                        loop {
                            match self.store.get(cur) {
                                TermData::Sym(sym) if sym == nil_sym => break,
                                TermData::Call { head, args_start, args_len } if args_len == 2 => {
                                    if let TermData::Sym(hs) = self.store.get(head) {
                                        if hs == cons_sym {
                                            count += 1;
                                            cur = self.store.args_pool[args_start as usize + 1];
                                            continue;
                                        }
                                    }
                                    return term;
                                }
                                _ => return term,
                            }
                        }
                        return self.store.num(count);
                    }

                    // vec_len(v) — length of a vec or cells
                    if s == self.builtins.vec_len {
                        if let TermData::Call { head, args_len, .. } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.vec_sym || hs == self.builtins.cells_sym {
                                    return self.store.num(args_len as i64);
                                }
                            }
                        }
                    }

                    // set_len(s) — cardinality of a set
                    if s == self.builtins.set_len {
                        if let TermData::Call { head, args_len, .. } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.set_sym {
                                    return self.store.num(args_len as i64);
                                }
                            }
                        }
                    }

                    // map_len(m) — number of entries in a map
                    if s == self.builtins.map_len {
                        if let TermData::Call { head, args_len, .. } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.map_sym {
                                    return self.store.num(args_len as i64);
                                }
                            }
                        }
                    }

                    // map_keys(m) — return vec of keys
                    if s == self.builtins.map_keys {
                        if let TermData::Call { head, args_start: mas, args_len } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.map_sym {
                                    let entry_sym = self.builtins.entry_sym;
                                    let vec_head = self.store.sym_term(self.builtins.vec_sym);
                                    let mut keys: Vec<TermId> = Vec::new();
                                    for i in 0..args_len as usize {
                                        let entry = self.store.args_pool[mas as usize + i];
                                        if let Some(k) = entry_key_static(&self.store, entry, entry_sym) {
                                            keys.push(k);
                                        }
                                    }
                                    return self.store.call(vec_head, &keys);
                                }
                            }
                        }
                    }

                    // map_values(m) — return vec of values
                    if s == self.builtins.map_values {
                        if let TermData::Call { head, args_start: mas, args_len } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.map_sym {
                                    let entry_sym = self.builtins.entry_sym;
                                    let vec_head = self.store.sym_term(self.builtins.vec_sym);
                                    let mut vals: Vec<TermId> = Vec::new();
                                    for i in 0..args_len as usize {
                                        let entry = self.store.args_pool[mas as usize + i];
                                        if let TermData::Call { head: eh, args_start: eas, args_len: 2 } = self.store.get(entry) {
                                            if let TermData::Sym(es) = self.store.get(eh) {
                                                if es == entry_sym {
                                                    vals.push(self.store.args_pool[eas as usize + 1]);
                                                }
                                            }
                                        }
                                    }
                                    return self.store.call(vec_head, &vals);
                                }
                            }
                        }
                    }

                }

                // Binary builtins for collections (arity 2)
                if args_len == 2 {
                    let a = self.store.args_pool[args_start as usize];
                    let b = self.store.args_pool[args_start as usize + 1];

                    // vec_get(v, i) — get element at index (works on vec and cells)
                    if s == self.builtins.vec_get {
                        if let TermData::Call { head, args_start: vas, args_len: vlen } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.vec_sym || hs == self.builtins.cells_sym {
                                    if let TermData::Num(idx) = self.store.get(b) {
                                        if idx >= 0 && (idx as usize) < vlen as usize {
                                            return self.store.args_pool[vas as usize + idx as usize];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // vec_push(v, elem) — append element
                    if s == self.builtins.vec_push {
                        if let TermData::Call { head, args_start: vas, args_len: vlen } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.vec_sym {
                                    let mut elems: Vec<TermId> = (0..vlen as usize)
                                        .map(|i| self.store.args_pool[vas as usize + i])
                                        .collect();
                                    elems.push(b);
                                    return self.store.call(head, &elems);
                                }
                            }
                        }
                    }

                    // vec_concat(v1, v2) — concatenate two vecs
                    if s == self.builtins.vec_concat {
                        if let TermData::Call { head: h1, args_start: as1, args_len: l1 } = self.store.get(a) {
                            if let TermData::Sym(hs1) = self.store.get(h1) {
                                if hs1 == self.builtins.vec_sym {
                                    if let TermData::Call { head: h2, args_start: as2, args_len: l2 } = self.store.get(b) {
                                        if let TermData::Sym(hs2) = self.store.get(h2) {
                                            if hs2 == self.builtins.vec_sym {
                                                let mut elems: Vec<TermId> = Vec::with_capacity(l1 as usize + l2 as usize);
                                                for i in 0..l1 as usize {
                                                    elems.push(self.store.args_pool[as1 as usize + i]);
                                                }
                                                for i in 0..l2 as usize {
                                                    elems.push(self.store.args_pool[as2 as usize + i]);
                                                }
                                                return self.store.call(h1, &elems);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // set_contains(s, e) — check if element is in set
                    if s == self.builtins.set_contains {
                        if let TermData::Call { head, args_start: sas, args_len: slen } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.set_sym {
                                    for i in 0..slen as usize {
                                        if self.store.args_pool[sas as usize + i] == b {
                                            return self.store.sym_term(self.builtins.true_sym);
                                        }
                                    }
                                    return self.store.sym_term(self.builtins.false_sym);
                                }
                            }
                        }
                    }

                    // set_insert(s, e) — add element to set
                    if s == self.builtins.set_insert {
                        if let TermData::Call { head, args_start: sas, args_len: slen } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.set_sym {
                                    let mut elems: Vec<TermId> = (0..slen as usize)
                                        .map(|i| self.store.args_pool[sas as usize + i])
                                        .collect();
                                    elems.push(b);
                                    elems.sort_by(|x, y| self.store.term_cmp(*x, *y));
                                    elems.dedup();
                                    return self.store.call(head, &elems);
                                }
                            }
                        }
                    }

                    // set_remove(s, e) — remove element from set
                    if s == self.builtins.set_remove {
                        if let TermData::Call { head, args_start: sas, args_len: slen } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.set_sym {
                                    let elems: Vec<TermId> = (0..slen as usize)
                                        .map(|i| self.store.args_pool[sas as usize + i])
                                        .filter(|e| *e != b)
                                        .collect();
                                    return self.store.call(head, &elems);
                                }
                            }
                        }
                    }

                    // set_union(s1, s2) — union of two sets
                    if s == self.builtins.set_union {
                        if let TermData::Call { head: h1, args_start: as1, args_len: l1 } = self.store.get(a) {
                            if let TermData::Sym(hs1) = self.store.get(h1) {
                                if hs1 == self.builtins.set_sym {
                                    if let TermData::Call { head: _h2, args_start: as2, args_len: l2 } = self.store.get(b) {
                                        let mut elems: Vec<TermId> = Vec::with_capacity(l1 as usize + l2 as usize);
                                        for i in 0..l1 as usize {
                                            elems.push(self.store.args_pool[as1 as usize + i]);
                                        }
                                        for i in 0..l2 as usize {
                                            elems.push(self.store.args_pool[as2 as usize + i]);
                                        }
                                        elems.sort_by(|x, y| self.store.term_cmp(*x, *y));
                                        elems.dedup();
                                        return self.store.call(h1, &elems);
                                    }
                                }
                            }
                        }
                    }

                    // set_intersect(s1, s2) — intersection of two sets
                    if s == self.builtins.set_intersect {
                        if let TermData::Call { head: h1, args_start: as1, args_len: l1 } = self.store.get(a) {
                            if let TermData::Sym(hs1) = self.store.get(h1) {
                                if hs1 == self.builtins.set_sym {
                                    if let TermData::Call { args_start: as2, args_len: l2, .. } = self.store.get(b) {
                                        let set2: Vec<TermId> = (0..l2 as usize)
                                            .map(|i| self.store.args_pool[as2 as usize + i])
                                            .collect();
                                        let elems: Vec<TermId> = (0..l1 as usize)
                                            .map(|i| self.store.args_pool[as1 as usize + i])
                                            .filter(|e| set2.contains(e))
                                            .collect();
                                        return self.store.call(h1, &elems);
                                    }
                                }
                            }
                        }
                    }

                    // map_get(m, k) — get value for key
                    if s == self.builtins.map_get {
                        if let TermData::Call { head, args_start: mas, args_len: mlen } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.map_sym {
                                    let entry_sym = self.builtins.entry_sym;
                                    for i in 0..mlen as usize {
                                        let entry = self.store.args_pool[mas as usize + i];
                                        if let Some(k) = entry_key_static(&self.store, entry, entry_sym) {
                                            if k == b {
                                                if let TermData::Call { args_start: eas, .. } = self.store.get(entry) {
                                                    return self.store.args_pool[eas as usize + 1];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // map_remove(m, k) — remove key from map
                    if s == self.builtins.map_remove {
                        if let TermData::Call { head, args_start: mas, args_len: mlen } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.map_sym {
                                    let entry_sym = self.builtins.entry_sym;
                                    let entries: Vec<TermId> = (0..mlen as usize)
                                        .map(|i| self.store.args_pool[mas as usize + i])
                                        .filter(|e| {
                                            entry_key_static(&self.store, *e, entry_sym) != Some(b)
                                        })
                                        .collect();
                                    return self.store.call(head, &entries);
                                }
                            }
                        }
                    }

                    // map_merge(m1, m2) — merge two maps (m2 values win on conflict)
                    if s == self.builtins.map_merge {
                        if let TermData::Call { head: h1, args_start: as1, args_len: l1 } = self.store.get(a) {
                            if let TermData::Sym(hs1) = self.store.get(h1) {
                                if hs1 == self.builtins.map_sym {
                                    if let TermData::Call { args_start: as2, args_len: l2, .. } = self.store.get(b) {
                                        let entry_sym = self.builtins.entry_sym;
                                        let mut all: Vec<TermId> = Vec::with_capacity(l1 as usize + l2 as usize);
                                        for i in 0..l1 as usize {
                                            all.push(self.store.args_pool[as1 as usize + i]);
                                        }
                                        for i in 0..l2 as usize {
                                            all.push(self.store.args_pool[as2 as usize + i]);
                                        }
                                        // Sort by key
                                        let keys: Vec<Option<TermId>> = all.iter()
                                            .map(|t| entry_key_static(&self.store, *t, entry_sym))
                                            .collect();
                                        let mut indices: Vec<usize> = (0..all.len()).collect();
                                        indices.sort_by(|&ai, &bi| {
                                            match (keys[ai], keys[bi]) {
                                                (Some(ka), Some(kb)) => self.store.term_cmp(ka, kb),
                                                _ => std::cmp::Ordering::Equal,
                                            }
                                        });
                                        let sorted: Vec<TermId> = indices.iter().map(|&i| all[i]).collect();
                                        // Dedup by key (keep last = m2 wins)
                                        let mut deduped: Vec<TermId> = Vec::with_capacity(sorted.len());
                                        for entry in sorted {
                                            if let Some(last) = deduped.last() {
                                                let lk = entry_key_static(&self.store, *last, entry_sym);
                                                let ek = entry_key_static(&self.store, entry, entry_sym);
                                                if let (Some(lk), Some(ek)) = (lk, ek) {
                                                    if lk == ek {
                                                        *deduped.last_mut().unwrap() = entry;
                                                        continue;
                                                    }
                                                }
                                            }
                                            deduped.push(entry);
                                        }
                                        return self.store.call(h1, &deduped);
                                    }
                                }
                            }
                        }
                    }
                }

                // make_term(head, args_vec) — construct call term without eval/sorting
                if args_len == 2 && s == self.builtins.make_term_sym {
                    let a = self.store.args_pool[args_start as usize];
                    let b = self.store.args_pool[args_start as usize + 1];
                    // head_term is first arg (a symbol term, already eval'd)
                    // args_vec is second arg (a vec(...) term, already eval'd)
                    if let TermData::Call { head: vh, args_start: vas, args_len: vlen } = self.store.get(b) {
                        if let TermData::Sym(vhs) = self.store.get(vh) {
                            if vhs == self.builtins.vec_sym {
                                let elements: Vec<TermId> = (0..vlen as usize)
                                    .map(|i| self.store.args_pool[vas as usize + i])
                                    .collect();
                                return self.store.call(a, &elements);
                            }
                        }
                    }
                }

                // Ternary builtins (arity 3)
                if args_len == 3 {
                    let a = self.store.args_pool[args_start as usize];
                    let b = self.store.args_pool[args_start as usize + 1];
                    let c = self.store.args_pool[args_start as usize + 2];

                    // defclause(name, args_vec, body) — install a new fn clause from terms
                    if s == self.builtins.defclause_sym {
                        // name must be a symbol
                        let name_sym = if let TermData::Sym(ns) = self.store.get(a) {
                            ns
                        } else {
                            return term;
                        };
                        let name_str = self.store.sym_name(name_sym).to_string();

                        // args_vec must be a vec(...)
                        let arg_pats: Vec<Pattern> = if let TermData::Call { head: vh, args_start: vas, args_len: vlen } = self.store.get(b) {
                            if let TermData::Sym(vhs) = self.store.get(vh) {
                                if vhs == self.builtins.vec_sym {
                                    (0..vlen as usize)
                                        .map(|i| self.term_to_pattern(self.store.args_pool[vas as usize + i]))
                                        .collect()
                                } else {
                                    return term;
                                }
                            } else {
                                return term;
                            }
                        } else {
                            return term;
                        };

                        let lhs = Pattern::Call(Box::new(Pattern::Sym(name_sym)), arg_pats);
                        let rhs = self.term_to_pattern(c);

                        let clause = Clause { lhs, rhs, guards: vec![] };
                        let rule = Rule { name: name_str, clauses: vec![clause] };

                        let idx = self.rules.len();
                        if let Some(sym) = rule_head_sym(&rule) {
                            self.rule_index.entry(sym).or_default().push(idx);
                        }
                        self.rules.push(rule);
                        self.invalidate_cache();
                        return self.store.num(0);
                    }

                    // vec_slice(v, start, end)
                    if s == self.builtins.vec_slice {
                        if let TermData::Call { head, args_start: vas, args_len: vlen } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.vec_sym {
                                    if let (TermData::Num(start), TermData::Num(end)) = (self.store.get(b), self.store.get(c)) {
                                        let start = start.max(0) as usize;
                                        let end = end.min(vlen as i64).max(start as i64) as usize;
                                        let elems: Vec<TermId> = (start..end)
                                            .map(|i| self.store.args_pool[vas as usize + i])
                                            .collect();
                                        return self.store.call(head, &elems);
                                    }
                                }
                            }
                        }
                    }

                    // map_set(m, k, v) — set/update key in map
                    if s == self.builtins.map_set {
                        if let TermData::Call { head, args_start: mas, args_len: mlen } = self.store.get(a) {
                            if let TermData::Sym(hs) = self.store.get(head) {
                                if hs == self.builtins.map_sym {
                                    let entry_sym = self.builtins.entry_sym;
                                    let entry_head = self.store.sym_term(entry_sym);
                                    let new_entry = self.store.call(entry_head, &[b, c]);
                                    let mut entries: Vec<TermId> = (0..mlen as usize)
                                        .map(|i| self.store.args_pool[mas as usize + i])
                                        .filter(|e| entry_key_static(&self.store, *e, entry_sym) != Some(b))
                                        .collect();
                                    entries.push(new_entry);
                                    // Re-sort by key
                                    let keys: Vec<Option<TermId>> = entries.iter()
                                        .map(|t| entry_key_static(&self.store, *t, entry_sym))
                                        .collect();
                                    let mut indices: Vec<usize> = (0..entries.len()).collect();
                                    indices.sort_by(|&ai, &bi| {
                                        match (keys[ai], keys[bi]) {
                                            (Some(ka), Some(kb)) => self.store.term_cmp(ka, kb),
                                            _ => std::cmp::Ordering::Equal,
                                        }
                                    });
                                    let sorted: Vec<TermId> = indices.iter().map(|&i| entries[i]).collect();
                                    return self.store.call(head, &sorted);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 6. Try active scope handler
        {
            let sidx = self.current_scope_idx();
            if self.scopes[sidx].handler.is_some() {
                let mut handler = self.scopes[sidx].handler.take().unwrap();
                let result = handler.try_eval(&mut self.store, term);
                self.scopes[sidx].handler = Some(handler);
                if let Some(val) = result {
                    return val;
                }
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
                clauses.push(Clause { lhs: lhs_pat, rhs: rhs_pat, guards: vec![] });
            }
        }

        let rule = Rule { name: name_str, clauses };

        if is_meta {
            let target_scope = Some(self.ensure_scope(&to_str, true));
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

    /// Convert a TermId back to a Pattern, replacing __pvar(N) with Var(N),
    /// __spread(N) with Spread(VarId(N)), __wild_spread with WildSpread, __wild with Wildcard.
    fn term_to_pattern(&self, term: TermId) -> Pattern {
        match self.store.get(term) {
            TermData::Num(n) => Pattern::Num(n),
            TermData::Float(bits) => Pattern::Float(bits),
            TermData::Sym(s) => {
                // __wild_spread → Pattern::WildSpread
                if s == self.meta_syms.wild_spread_sym {
                    return Pattern::WildSpread;
                }
                // __wild → Pattern::Wildcard
                if s == self.meta_syms.wild_sym {
                    return Pattern::Wildcard;
                }
                Pattern::Sym(s)
            }
            TermData::Call { head, args_start, args_len } => {
                // __pvar(N) → Pattern::Var(VarId(N))
                if let TermData::Sym(s) = self.store.get(head) {
                    if s == self.meta_syms.pvar_sym && args_len == 1 {
                        let arg = self.store.args_pool[args_start as usize];
                        if let TermData::Num(n) = self.store.get(arg) {
                            return Pattern::Var(VarId(n as u32));
                        }
                    }
                    // __spread(N) → Pattern::Spread(VarId(N))
                    if s == self.meta_syms.spread_sym && args_len == 1 {
                        let arg = self.store.args_pool[args_start as usize];
                        if let TermData::Num(n) = self.store.get(arg) {
                            return Pattern::Spread(VarId(n as u32));
                        }
                    }
                    // set(...) → Pattern::Set
                    if s == self.builtins.set_sym {
                        let elems: Vec<Pattern> = (0..args_len as usize)
                            .map(|i| self.term_to_pattern(self.store.args_pool[args_start as usize + i]))
                            .collect();
                        return Pattern::Set(elems);
                    }
                    // map(entry(k,v), ...) → Pattern::Map
                    if s == self.builtins.map_sym {
                        let entry_sym = self.builtins.entry_sym;
                        let mut entries: Vec<(Pattern, Pattern)> = Vec::new();
                        for i in 0..args_len as usize {
                            let entry_term = self.store.args_pool[args_start as usize + i];
                            if let TermData::Call { head: eh, args_start: eas, args_len: 2 } = self.store.get(entry_term) {
                                if let TermData::Sym(es) = self.store.get(eh) {
                                    if es == entry_sym {
                                        let k = self.term_to_pattern(self.store.args_pool[eas as usize]);
                                        let v = self.term_to_pattern(self.store.args_pool[eas as usize + 1]);
                                        entries.push((k, v));
                                        continue;
                                    }
                                }
                            }
                            // Not a valid entry — fall through to Call
                            let head_pat = self.term_to_pattern(head);
                            let args_pat: Vec<Pattern> = (0..args_len as usize)
                                .map(|i| self.term_to_pattern(self.store.args_pool[args_start as usize + i]))
                                .collect();
                            return Pattern::Call(Box::new(head_pat), args_pat);
                        }
                        return Pattern::Map(entries);
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
            let mut evaled: Vec<TermId> = Vec::with_capacity(len);
            let mut changed = false;
            for i in 0..len {
                let arg = self.store.args_pool[args_start as usize + i];
                let v = self.eval(arg);
                changed |= v != arg;
                evaled.push(v);
            }
            if changed {
                self.store.call(head, &evaled)
            } else {
                term
            }
        } else {
            // Sym or Num — keep as-is (these are ground keys like `next_id`, `filter`)
            term
        }
    }

    fn send_to_scope(&mut self, scope_idx: usize, val: TermId) {
        if self.scopes[scope_idx].output_only {
            if let Some(mut handler) = self.scopes[scope_idx].handler.take() {
                handler.try_eval(&mut self.store, val);
                self.scopes[scope_idx].handler = Some(handler);
            }
        } else {
            self.scopes[scope_idx].work_queue.push_back(val);
        }
    }

    fn fire_meta(&mut self, sub_old: TermId, sub_new: TermId, kind: TermId) {
        let reduction_head = self.store.sym_term(self.meta_syms.reduction);
        let step = self.store.num(self.step_count as i64);
        let quote_head = self.store.sym_term(self.builtins.quote);
        let quoted_old = self.store.call(quote_head, &[sub_old]);
        let quoted_new = self.store.call(quote_head, &[sub_new]);
        let event = self.store.call(reduction_head, &[step, quoted_old, quoted_new, kind]);

        self.in_meta = true;
        let num_meta = self.meta_rules.len();
        for i in 0..num_meta {
            let num_clauses = self.meta_rules[i].rule.clauses.len();
            for j in 0..num_clauses {
                let mut env = Env::new();
                if match_pattern(&mut self.store, &self.meta_rules[i].rule.clauses[j].lhs, event, &mut env, &self.pattern_ctx) {
                    let result = substitute(&mut self.store, &self.meta_rules[i].rule.clauses[j].rhs, &env);
                    let val = self.eval(result);
                    if let Some(scope_idx) = self.meta_rules[i].target_scope {
                        self.send_to_scope(scope_idx, val);
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

        self.in_meta = true;
        let num_meta = self.meta_rules.len();
        for i in 0..num_meta {
            let num_clauses = self.meta_rules[i].rule.clauses.len();
            for j in 0..num_clauses {
                let mut env = Env::new();
                if match_pattern(&mut self.store, &self.meta_rules[i].rule.clauses[j].lhs, event, &mut env, &self.pattern_ctx) {
                    let result = substitute(&mut self.store, &self.meta_rules[i].rule.clauses[j].rhs, &env);
                    let val = self.eval(result);
                    if let Some(scope_idx) = self.meta_rules[i].target_scope {
                        self.send_to_scope(scope_idx, val);
                    }
                    break;
                }
            }
        }
        self.in_meta = false;
    }

    // ── Tick: round-robin work queue processing ──

    /// Process up to `budget` work items from actor scopes in round-robin order.
    /// Returns the number of items processed.
    pub fn tick(&mut self, budget: usize) -> usize {
        let mut total_processed = 0;

        'outer: while total_processed < budget {
            let mut any_work = false;
            for scope_idx in 0..self.scopes.len() {
                // Skip main scope and output-only scopes
                if scope_idx == self.main_scope || self.scopes[scope_idx].output_only {
                    continue;
                }
                if let Some(term) = self.scopes[scope_idx].work_queue.pop_front() {
                    any_work = true;
                    self.active_scope_idx = Some(scope_idx);
                    self.invalidate_cache();
                    self.reset_eval_counters();
                    self.eval(term);
                    total_processed += 1;
                    if total_processed >= budget {
                        break 'outer;
                    }
                }
            }
            if !any_work {
                break;
            }
        }

        self.active_scope_idx = None;
        self.invalidate_cache();
        total_processed
    }

    /// Get the number of items in a scope's work queue.
    pub fn scope_queue_count(&self, name: &str) -> usize {
        if let Some(&idx) = self.scope_map.get(name) {
            self.scopes[idx].work_queue.len()
        } else {
            0
        }
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
        // Install meta rules (auto-create target scopes as output-only)
        for (rule, scope_name) in program.meta_rules {
            let target_scope = Some(self.ensure_scope(&scope_name, true));
            self.meta_rules.push(MetaRule { rule, target_scope });
        }
        // Create output-only scopes for @scope annotations in expressions
        for scope_name in program.emit_scopes {
            self.ensure_scope(&scope_name, true);
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
        if let Some(&idx) = self.scope_map.get(name) {
            self.scopes[idx].pending.borrow().len()
        } else {
            0
        }
    }

    pub fn scope_pending_get(&self, name: &str, idx: usize) -> TermId {
        let scope_idx = self.scope_map[name];
        self.scopes[scope_idx].pending.borrow()[idx]
    }

    pub fn scope_pending_clear(&self, name: &str) {
        if let Some(&idx) = self.scope_map.get(name) {
            self.scopes[idx].pending.borrow_mut().clear();
        }
    }

    pub fn scope_take_pending(&self, name: &str) -> Vec<TermId> {
        if let Some(&idx) = self.scope_map.get(name) {
            std::mem::take(&mut *self.scopes[idx].pending.borrow_mut())
        } else {
            Vec::new()
        }
    }

    pub fn term_count(&self) -> usize {
        self.store.term_count()
    }

    /// Mark-compact GC.  Collects roots from all scopes' dynamic rules,
    /// pending buffers, and work queues, compacts the term store, and remaps
    /// all internal TermId refs.
    pub fn gc(&mut self) {
        // Collect roots from ALL scopes
        let mut roots: Vec<TermId> = Vec::new();

        for scope in &self.scopes {
            for (&k, &v) in &scope.dynamic_rules_map {
                roots.push(k);
                roots.push(v);
            }
            for &tid in scope.pending.borrow().iter() {
                roots.push(tid);
            }
            for &tid in &scope.work_queue {
                roots.push(tid);
            }
        }

        // Run compacting GC on the term store
        let remap = self.store.gc(&roots);

        // Remap all scopes' TermIds
        for scope in &mut self.scopes {
            // Remap dynamic_rules Vec
            for pair in scope.dynamic_rules.iter_mut() {
                pair.0 = remap[pair.0 .0 as usize];
                pair.1 = remap[pair.1 .0 as usize];
            }

            // Rebuild dynamic_rules_map (keys change hash after remap)
            let old_entries: Vec<(TermId, TermId)> = scope.dynamic_rules_map.drain().collect();
            for (k, v) in old_entries {
                scope.dynamic_rules_map.insert(remap[k.0 as usize], remap[v.0 as usize]);
            }

            // Remap pending
            {
                let mut v = scope.pending.borrow_mut();
                for tid in v.iter_mut() {
                    *tid = remap[tid.0 as usize];
                }
            }

            // Remap work_queue
            for tid in scope.work_queue.iter_mut() {
                *tid = remap[tid.0 as usize];
            }
        }

        // Clear eval cache (cheaper than remapping packed entries)
        self.eval_cache.clear();
        self.invalidate_cache();
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::TermStore;

    fn new_engine() -> Engine {
        Engine::new(TermStore::new(), Vec::new(), Vec::new())
    }

    fn eval_program(src: &str) -> (Engine, TermId) {
        let mut engine = new_engine();
        let term = engine.load_program(src);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        let result = engine.eval(term);
        (engine, result)
    }

    fn eval_str(src: &str) -> String {
        let (engine, result) = eval_program(src);
        engine.display(result)
    }

    fn eval_num(src: &str) -> i64 {
        let (engine, result) = eval_program(src);
        engine.term_num(result)
    }

    fn eval_float(src: &str) -> f64 {
        let (engine, result) = eval_program(src);
        engine.term_float(result)
    }

    fn eval_sym(src: &str) -> String {
        let (engine, result) = eval_program(src);
        engine.term_sym_name(result).to_string()
    }

    // ── Arithmetic ──

    #[test]
    fn add() {
        assert_eq!(eval_num("2 + 3"), 5);
    }

    #[test]
    fn sub() {
        assert_eq!(eval_num("10 - 3"), 7);
    }

    #[test]
    fn mul() {
        assert_eq!(eval_num("4 * 5"), 20);
    }

    #[test]
    fn div() {
        assert_eq!(eval_num("10 / 3"), 3);
    }

    #[test]
    fn modulo() {
        assert_eq!(eval_num("10 % 3"), 1);
    }

    #[test]
    fn nested_arithmetic() {
        assert_eq!(eval_num("(2 + 3) * (4 - 1)"), 15);
    }

    #[test]
    fn negative_via_subtraction() {
        assert_eq!(eval_num("0 - 5"), -5);
    }

    // ── Floats ──

    #[test]
    fn float_add() {
        assert!((eval_float("1.5 + 2.5") - 4.0).abs() < 1e-10);
    }

    #[test]
    fn float_mul() {
        assert!((eval_float("2.0 * 3.5") - 7.0).abs() < 1e-10);
    }

    #[test]
    fn mixed_int_float_add() {
        assert!((eval_float("1 + 2.5") - 3.5).abs() < 1e-10);
    }

    #[test]
    fn floor_float() {
        assert_eq!(eval_num("floor(3.7)"), 3);
    }

    #[test]
    fn floor_int_passthrough() {
        assert_eq!(eval_num("floor(5)"), 5);
    }

    #[test]
    fn abs_negative() {
        assert_eq!(eval_num("abs(0 - 5)"), 5);
    }

    #[test]
    fn abs_positive() {
        assert_eq!(eval_num("abs(5)"), 5);
    }

    #[test]
    fn abs_float() {
        assert!((eval_float("abs(0.0 - 3.14)") - 3.14).abs() < 1e-10);
    }

    // ── Comparisons ──

    #[test]
    fn eq_true() {
        assert_eq!(eval_sym("3 == 3"), "true");
    }

    #[test]
    fn eq_false() {
        assert_eq!(eval_sym("3 == 4"), "false");
    }

    #[test]
    fn neq_true() {
        assert_eq!(eval_sym("3 != 4"), "true");
    }

    #[test]
    fn neq_false() {
        assert_eq!(eval_sym("3 != 3"), "false");
    }

    #[test]
    fn lt() {
        assert_eq!(eval_sym("2 < 3"), "true");
        assert_eq!(eval_sym("3 < 3"), "false");
        assert_eq!(eval_sym("4 < 3"), "false");
    }

    #[test]
    fn gt() {
        assert_eq!(eval_sym("4 > 3"), "true");
        assert_eq!(eval_sym("3 > 3"), "false");
        assert_eq!(eval_sym("2 > 3"), "false");
    }

    #[test]
    fn lte() {
        assert_eq!(eval_sym("2 <= 3"), "true");
        assert_eq!(eval_sym("3 <= 3"), "true");
        assert_eq!(eval_sym("4 <= 3"), "false");
    }

    #[test]
    fn gte() {
        assert_eq!(eval_sym("4 >= 3"), "true");
        assert_eq!(eval_sym("3 >= 3"), "true");
        assert_eq!(eval_sym("2 >= 3"), "false");
    }

    #[test]
    fn eq_syms() {
        assert_eq!(eval_sym("hello == hello"), "true");
        assert_eq!(eval_sym("hello == world"), "false");
    }

    // ── Booleans ──

    #[test]
    fn bool_not() {
        assert_eq!(eval_sym("fn not(true) = false\nfn not(false) = true\nnot(true)"), "false");
        assert_eq!(eval_sym("fn not(true) = false\nfn not(false) = true\nnot(false)"), "true");
    }

    // ── If/then/else ──

    #[test]
    fn if_true_branch() {
        assert_eq!(eval_num("if true then 1 else 2"), 1);
    }

    #[test]
    fn if_false_branch() {
        assert_eq!(eval_num("if false then 1 else 2"), 2);
    }

    #[test]
    fn if_with_comparison() {
        assert_eq!(eval_num("if 3 > 2 then 10 else 20"), 10);
    }

    #[test]
    fn user_defined_abs() {
        assert_eq!(eval_num("fn abs2(?n) = if ?n < 0 then 0 - ?n else ?n\nabs2(0 - 5)"), 5);
    }

    // ── Strings ──

    #[test]
    fn str_concat() {
        assert_eq!(eval_sym(r#""hello" ++ " world""#), "hello world");
    }

    #[test]
    fn str_concat_num_coercion() {
        assert_eq!(eval_sym(r#""count: " ++ 42"#), "count: 42");
    }

    #[test]
    fn str_len() {
        assert_eq!(eval_num(r#"str_len("hello")"#), 5);
    }

    // ── User rules ──

    #[test]
    fn factorial() {
        assert_eq!(eval_num("fn fact(0) = 1\nfn fact(?n) = ?n * fact(?n - 1)\nfact(10)"), 3628800);
    }

    #[test]
    fn fibonacci() {
        assert_eq!(eval_num(
            "fn fib(0) = 0\nfn fib(1) = 1\nfn fib(?n) = fib(?n - 1) + fib(?n - 2)\nfib(10)"
        ), 55);
    }

    #[test]
    fn higher_order_map() {
        // Use explicit cons-lists and 'fmap' to avoid clash with builtin 'map' type
        assert_eq!(eval_str(
            "fn fmap(?f, nil) = nil\n\
             fn fmap(?f, cons(?h, ?t)) = cons(?f(?h), fmap(?f, ?t))\n\
             fn double(?x) = ?x * 2\n\
             fmap(double, cons(1, cons(2, cons(3, nil))))"
        ), "cons(2, cons(4, cons(6, nil)))");
    }

    #[test]
    fn higher_order_filter() {
        assert_eq!(eval_str(
            "fn filter(?p, nil) = nil\n\
             fn filter(?p, cons(?h, ?t)) = if ?p(?h) then cons(?h, filter(?p, ?t)) else filter(?p, ?t)\n\
             fn is_even(?n) = ?n % 2 == 0\n\
             filter(is_even, cons(1, cons(2, cons(3, cons(4, cons(5, cons(6, nil)))))))"
        ), "cons(2, cons(4, cons(6, nil)))");
    }

    #[test]
    fn multi_clause_rule() {
        // fib(0)=0, fib(1)=1, fib(5)=5; 0+1+5=6
        assert_eq!(eval_num(
            "fn fib(0) = 0\nfn fib(1) = 1\nfn fib(?n) = fib(?n - 1) + fib(?n - 2)\n\
             fib(0) + fib(1) + fib(5)"
        ), 6);
    }

    // ── Lists ──

    #[test]
    fn list_literal() {
        assert_eq!(eval_str("[1, 2, 3]"), "[1, 2, 3]");
    }

    #[test]
    fn list_len_builtin() {
        assert_eq!(eval_num("list_len([1, 2, 3])"), 3);
    }

    #[test]
    fn list_len_empty() {
        assert_eq!(eval_num("list_len(nil)"), 0);
    }

    #[test]
    fn list_reverse() {
        assert_eq!(eval_str("list_reverse([1, 2, 3])"), "[3, 2, 1]");
    }

    #[test]
    fn list_reverse_empty() {
        assert_eq!(eval_str("list_reverse(nil)"), "nil");
    }

    #[test]
    fn user_defined_len() {
        assert_eq!(eval_num(
            "fn len(nil) = 0\nfn len(cons(?h, ?t)) = 1 + len(?t)\nlen(cons(1, cons(2, cons(3, nil))))"
        ), 3);
    }

    // ── Dynamic rules ──

    #[test]
    fn dynamic_rule_assert_and_lookup() {
        assert_eq!(eval_num("{ rule(x, 42) x }"), 42);
    }

    #[test]
    fn dynamic_rule_update() {
        assert_eq!(eval_num("{ rule(x, 1) rule(x, 2) x }"), 2);
    }

    #[test]
    fn dynamic_rule_retract() {
        // After retracting x, x should be a normal form (the symbol x)
        assert_eq!(eval_sym("{ rule(x, 42) retract(x) x }"), "x");
    }

    #[test]
    fn dynamic_rule_with_call_key() {
        assert_eq!(eval_num("{ rule(score(1), 100) score(1) }"), 100);
    }

    // ── query_all ──

    #[test]
    fn query_all_basic() {
        assert_eq!(eval_num(
            "fn count(nil) = 0\nfn count(cons(?h, ?t)) = 1 + count(?t)\n\
             { rule(todo(1), entry(\"a\", false))\n  \
               rule(todo(2), entry(\"b\", true))\n  \
               rule(todo(3), entry(\"c\", false))\n  \
               count(query_all(entry)) }"
        ), 3);
    }

    #[test]
    fn query_all_empty() {
        assert_eq!(eval_str("query_all(nothing)"), "nil");
    }

    #[test]
    fn query_all_after_retract() {
        assert_eq!(eval_num(
            "fn count(nil) = 0\nfn count(cons(?h, ?t)) = 1 + count(?t)\n\
             { rule(todo(1), entry(\"a\"))\n  \
               rule(todo(2), entry(\"b\"))\n  \
               retract(todo(1))\n  \
               count(query_all(entry)) }"
        ), 1);
    }

    // ── Seq ──

    #[test]
    fn seq_returns_last() {
        assert_eq!(eval_num("{ 1 2 3 }"), 3);
    }

    #[test]
    fn seq_side_effects() {
        assert_eq!(eval_num("{ rule(x, 10) rule(y, 20) x + y }"), 30);
    }

    // ── str_join ──

    #[test]
    fn str_join_list() {
        assert_eq!(eval_str(r#"str_join(["hello", " ", "world"])"#), "hello world");
    }

    #[test]
    fn str_join_with_nums() {
        assert_eq!(eval_str(r#"str_join(["x=", 42])"#), "x=42");
    }

    // ── random ──

    #[test]
    fn random_deterministic() {
        let (engine1, r1) = eval_program("random(42)");
        let (engine2, r2) = eval_program("random(42)");
        assert_eq!(engine1.term_float(r1), engine2.term_float(r2));
    }

    #[test]
    fn random_in_range() {
        let (engine, result) = eval_program("random(1)");
        let f = engine.term_float(result);
        assert!(f >= 0.0 && f < 1.0);
    }

    // ── Term construction from API ──

    #[test]
    fn api_make_terms() {
        let mut engine = new_engine();
        let n = engine.make_num(42);
        assert_eq!(engine.term_num(n), 42);

        let s = engine.make_sym("hello");
        assert_eq!(engine.term_sym_name(s), "hello");

        let f = engine.make_float(3.14);
        assert!((engine.term_float(f) - 3.14).abs() < 1e-10);
    }

    #[test]
    fn api_make_call_and_eval() {
        let mut engine = new_engine();
        engine.load_program("fn fact(0) = 1\nfn fact(?n) = ?n * fact(?n - 1)\n0");
        let fact = engine.make_sym("fact");
        let five = engine.make_num(5);
        let call = engine.make_call(fact, &[five]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        let result = engine.eval(call);
        assert_eq!(engine.term_num(result), 120);
    }

    #[test]
    fn api_term_tag() {
        let mut engine = new_engine();
        let n = engine.make_num(1);
        let s = engine.make_sym("x");
        let f = engine.make_float(1.0);
        let c = engine.make_call(s, &[n]);
        assert_eq!(engine.term_tag(n), 0);  // Num
        assert_eq!(engine.term_tag(s), 1);  // Sym
        assert_eq!(engine.term_tag(c), 2);  // Call
        assert_eq!(engine.term_tag(f), 3);  // Float
    }

    #[test]
    fn api_call_inspection() {
        let mut engine = new_engine();
        let head = engine.make_sym("f");
        let a = engine.make_num(1);
        let b = engine.make_num(2);
        let call = engine.make_call(head, &[a, b]);
        assert_eq!(engine.term_call_head(call), head);
        assert_eq!(engine.term_call_arity(call), 2);
        assert_eq!(engine.term_call_arg(call, 0), a);
        assert_eq!(engine.term_call_arg(call, 1), b);
    }

    // ── Display ──

    #[test]
    fn display_num() {
        let (engine, result) = eval_program("42");
        assert_eq!(engine.display(result), "42");
    }

    #[test]
    fn display_sym() {
        let (engine, result) = eval_program("hello");
        assert_eq!(engine.display(result), "hello");
    }

    #[test]
    fn display_call() {
        let (engine, result) = eval_program("f(1, 2)");
        assert_eq!(engine.display(result), "f(1, 2)");
    }

    #[test]
    fn display_float() {
        let (engine, result) = eval_program("3.0");
        assert_eq!(engine.display(result), "3.0");
    }

    // ── TodoMVC logic (integration) ──

    #[test]
    fn todomvc_logic() {
        assert_eq!(eval_num(
            "fn len(nil) = 0\n\
             fn len(cons(?h, ?t)) = 1 + len(?t)\n\
             fn filter_list(?p, nil) = nil\n\
             fn filter_list(?p, cons(?h, ?t)) = if ?p(?h) then cons(?h, filter_list(?p, ?t)) else filter_list(?p, ?t)\n\
             fn is_active(entry(?id, ?body, false)) = true\n\
             fn is_active(entry(?id, ?body, true)) = false\n\
             fn all_todos() = query_all(entry)\n\
             fn active_todos() = filter_list(is_active, all_todos())\n\
             fn items_left() = len(active_todos())\n\
             fn toggle_entry(?id) = toggle_with(?id, todo(?id))\n\
             fn toggle_with(?id, entry(?i, ?body, true)) = rule(todo(?id), entry(?i, ?body, false))\n\
             fn toggle_with(?id, entry(?i, ?body, false)) = rule(todo(?id), entry(?i, ?body, true))\n\
             {\n\
               rule(todo(1), entry(1, \"buy milk\", false))\n\
               rule(todo(2), entry(2, \"clean house\", true))\n\
               rule(todo(3), entry(3, \"write code\", false))\n\
               toggle_entry(2)\n\
               items_left()\n\
             }"
        ), 3);
    }

    // ── Floor with dynamic rules ──

    #[test]
    fn floor_dynamic() {
        assert_eq!(eval_num(
            "fn init() = rule(distance, 3.7)\n\
             fn score() = floor(distance)\n\
             { init() score() }"
        ), 3);
    }

    // ── Step limit ──

    #[test]
    fn step_limit_detected() {
        // Run on a thread with a larger stack to avoid OS stack overflow
        // before the engine's eval depth limit (2000) kicks in.
        let result = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let mut engine = new_engine();
                let term = engine.load_program("fn f(?x) = f(?x + 1)\nf(0)");
                engine.invalidate_cache();
                engine.reset_eval_counters();
                engine.eval(term);
                engine.step_limit_exceeded
            })
            .unwrap()
            .join()
            .unwrap();
        assert!(result);
    }

    // ── Scope: main scope is always present ──

    #[test]
    fn main_scope_exists() {
        let engine = new_engine();
        assert_eq!(engine.main_scope, 0);
        assert_eq!(engine.scopes[0].name, "main");
        assert!(!engine.scopes[0].output_only);
    }

    // ── Scope: dynamic rules are scope-local ──

    #[test]
    fn scope_dynamic_rules_isolation() {
        let mut engine = new_engine();
        engine.load_program("0");

        // Assert rule(x, 42) in main scope
        let rule_sym = engine.make_sym("rule");
        let x = engine.make_sym("x");
        let n42 = engine.make_num(42);
        let rule_call = engine.make_call(rule_sym, &[x, n42]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(rule_call);

        // Main scope should have x -> 42
        assert_eq!(engine.scopes[0].dynamic_rules_map.len(), 1);

        // Emit rule(y, 99) to actor scope "physics"
        let emit_sym = engine.make_sym("emit");
        let physics_sym = engine.make_sym("physics");
        let y = engine.make_sym("y");
        let n99 = engine.make_num(99);
        let rule_call2 = engine.make_call(rule_sym, &[y, n99]);
        let emit_call = engine.make_call(emit_sym, &[physics_sym, rule_call2]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(emit_call);

        // Physics scope should have 1 item in work queue
        assert_eq!(engine.scope_queue_count("physics"), 1);

        // Tick to process physics scope
        let processed = engine.tick(10);
        assert_eq!(processed, 1);

        // Physics scope should now have y -> 99
        let physics_idx = engine.scope_map["physics"];
        assert_eq!(engine.scopes[physics_idx].dynamic_rules_map.len(), 1);

        // Main scope should still only have x -> 42 (not y)
        assert_eq!(engine.scopes[0].dynamic_rules_map.len(), 1);
        assert!(engine.scopes[0].dynamic_rules_map.contains_key(&x));
        assert!(!engine.scopes[0].dynamic_rules_map.contains_key(&y));
    }

    // ── Scope: emit to output scope buffers in pending ──

    #[test]
    fn emit_output_scope_buffers() {
        let mut engine = new_engine();
        engine.load_program("0");

        // Meta rule targets create output scopes; manually create one
        let _idx = engine.ensure_scope("dom", true);

        let emit_sym = engine.make_sym("emit");
        let dom_sym = engine.make_sym("dom");
        let hello = engine.make_sym("hello");
        let emit_call = engine.make_call(emit_sym, &[dom_sym, hello]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(emit_call);

        assert_eq!(engine.scope_pending_count("dom"), 1);
        let pending = engine.scope_pending_get("dom", 0);
        assert_eq!(engine.term_sym_name(pending), "hello");

        engine.scope_pending_clear("dom");
        assert_eq!(engine.scope_pending_count("dom"), 0);
    }

    // ── Scope: emit to actor scope queues work ──

    #[test]
    fn emit_actor_scope_queues() {
        let mut engine = new_engine();
        engine.load_program("0");

        let emit_sym = engine.make_sym("emit");
        let scope_sym = engine.make_sym("worker");
        let val = engine.make_num(42);
        let emit_call = engine.make_call(emit_sym, &[scope_sym, val]);

        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(emit_call);

        assert_eq!(engine.scope_queue_count("worker"), 1);
        // Output-only = false for actor scopes
        let widx = engine.scope_map["worker"];
        assert!(!engine.scopes[widx].output_only);
    }

    // ── Tick: empty queues return 0 ──

    #[test]
    fn tick_empty() {
        let mut engine = new_engine();
        assert_eq!(engine.tick(100), 0);
    }

    // ── Tick: processes work items ──

    #[test]
    fn tick_processes_work() {
        let mut engine = new_engine();
        engine.load_program("0");

        // Emit three rule() calls to actor scope
        let emit_sym = engine.make_sym("emit");
        let scope_sym = engine.make_sym("actor");
        let rule_sym = engine.make_sym("rule");
        for i in 0..3 {
            let key = engine.make_sym(&format!("k{}", i));
            let val = engine.make_num(i as i64);
            let rule_call = engine.make_call(rule_sym, &[key, val]);
            let emit_call = engine.make_call(emit_sym, &[scope_sym, rule_call]);
            engine.invalidate_cache();
            engine.reset_eval_counters();
            engine.eval(emit_call);
        }

        assert_eq!(engine.scope_queue_count("actor"), 3);

        // Process with budget of 2
        let processed = engine.tick(2);
        assert_eq!(processed, 2);
        assert_eq!(engine.scope_queue_count("actor"), 1);

        // Process remaining
        let processed = engine.tick(10);
        assert_eq!(processed, 1);
        assert_eq!(engine.scope_queue_count("actor"), 0);

        // Actor scope should have 3 dynamic rules
        let aidx = engine.scope_map["actor"];
        assert_eq!(engine.scopes[aidx].dynamic_rules.len(), 3);
    }

    // ── Tick: round-robin across multiple scopes ──

    #[test]
    fn tick_round_robin() {
        let mut engine = new_engine();
        engine.load_program("0");

        let emit_sym = engine.make_sym("emit");
        let rule_sym = engine.make_sym("rule");

        // Emit 2 items to scope_a, 1 item to scope_b
        let scope_a = engine.make_sym("scope_a");
        let scope_b = engine.make_sym("scope_b");

        for i in 0..2 {
            let key = engine.make_sym(&format!("a{}", i));
            let val = engine.make_num(i as i64);
            let rule_call = engine.make_call(rule_sym, &[key, val]);
            let emit_call = engine.make_call(emit_sym, &[scope_a, rule_call]);
            engine.invalidate_cache();
            engine.reset_eval_counters();
            engine.eval(emit_call);
        }
        {
            let key = engine.make_sym("b0");
            let val = engine.make_num(99);
            let rule_call = engine.make_call(rule_sym, &[key, val]);
            let emit_call = engine.make_call(emit_sym, &[scope_b, rule_call]);
            engine.invalidate_cache();
            engine.reset_eval_counters();
            engine.eval(emit_call);
        }

        assert_eq!(engine.scope_queue_count("scope_a"), 2);
        assert_eq!(engine.scope_queue_count("scope_b"), 1);

        // Tick with large budget — processes all
        let processed = engine.tick(100);
        assert_eq!(processed, 3);
        assert_eq!(engine.scope_queue_count("scope_a"), 0);
        assert_eq!(engine.scope_queue_count("scope_b"), 0);
    }

    // ── Tick: skips main scope and output-only scopes ──

    #[test]
    fn tick_skips_main_and_output() {
        let mut engine = new_engine();
        engine.load_program("0");

        // Push directly into main scope's work queue (shouldn't be processed by tick)
        let one = engine.make_num(1);
        engine.scopes[0].work_queue.push_back(one);

        // Create an output scope and push into its queue
        let oidx = engine.ensure_scope("output", true);
        let two = engine.make_num(2);
        engine.scopes[oidx].work_queue.push_back(two);

        let processed = engine.tick(100);
        assert_eq!(processed, 0);
        // Queues should remain untouched
        assert_eq!(engine.scopes[0].work_queue.len(), 1);
        assert_eq!(engine.scopes[oidx].work_queue.len(), 1);
    }

    // ── GC: preserves all scope data ──

    #[test]
    fn gc_preserves_scope_data() {
        let mut engine = new_engine();
        engine.load_program("0");

        // Set up dynamic rule in main scope
        let rule_sym = engine.make_sym("rule");
        let x = engine.make_sym("x");
        let n42 = engine.make_num(42);
        let rule_call = engine.make_call(rule_sym, &[x, n42]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(rule_call);

        // Emit to actor scope
        let emit_sym = engine.make_sym("emit");
        let scope_sym = engine.make_sym("actor");
        let y = engine.make_sym("y");
        let n99 = engine.make_num(99);
        let rule_call2 = engine.make_call(rule_sym, &[y, n99]);
        let emit_call = engine.make_call(emit_sym, &[scope_sym, rule_call2]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(emit_call);

        // Emit to output scope
        engine.ensure_scope("dom", true);
        let hello = engine.make_sym("hello");
        let dom_sym2 = engine.make_sym("dom");
        let emit_dom = engine.make_call(emit_sym, &[dom_sym2, hello]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(emit_dom);

        let terms_before = engine.term_count();

        // Run GC
        engine.gc();

        // Main scope dynamic rules should survive
        assert_eq!(engine.scopes[0].dynamic_rules.len(), 1);
        let (k, v) = engine.scopes[0].dynamic_rules[0];
        assert_eq!(engine.term_sym_name(k), "x");
        assert_eq!(engine.term_num(v), 42);

        // Actor scope work queue should survive
        assert_eq!(engine.scope_queue_count("actor"), 1);

        // Output scope pending should survive
        assert_eq!(engine.scope_pending_count("dom"), 1);
        let pending = engine.scope_pending_get("dom", 0);
        assert_eq!(engine.term_sym_name(pending), "hello");

        // Term count should be reduced (GC compacted)
        assert!(engine.term_count() <= terms_before);
    }

    // ── GC: scope data valid after tick ──

    #[test]
    fn gc_after_tick() {
        let mut engine = new_engine();
        engine.load_program("0");

        let emit_sym = engine.make_sym("emit");
        let scope_sym = engine.make_sym("actor");
        let rule_sym = engine.make_sym("rule");
        let key = engine.make_sym("k");
        let val = engine.make_num(7);
        let rule_call = engine.make_call(rule_sym, &[key, val]);
        let emit_call = engine.make_call(emit_sym, &[scope_sym, rule_call]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(emit_call);

        engine.tick(10);

        // GC after tick
        engine.gc();

        // Actor scope's dynamic rule should still be valid
        let aidx = engine.scope_map["actor"];
        assert_eq!(engine.scopes[aidx].dynamic_rules.len(), 1);
        let (k, v) = engine.scopes[aidx].dynamic_rules[0];
        assert_eq!(engine.term_sym_name(k), "k");
        assert_eq!(engine.term_num(v), 7);
    }

    // ── Scope: query_all respects active scope ──

    #[test]
    fn query_all_scope_local() {
        let mut engine = new_engine();
        engine.load_program(
            "fn count(nil) = 0\nfn count(cons(?h, ?t)) = 1 + count(?t)\n0"
        );

        // Assert 2 rules in main scope
        let rule_sym = engine.make_sym("rule");
        for i in 0..2 {
            let key = engine.make_sym(&format!("item{}", i));
            let val_sym = engine.make_sym("entry");
            let num_i = engine.make_num(i as i64);
            let val = engine.make_call(val_sym, &[num_i]);
            let rule_call = engine.make_call(rule_sym, &[key, val]);
            engine.invalidate_cache();
            engine.reset_eval_counters();
            engine.eval(rule_call);
        }

        // query_all(entry) in main scope should find 2
        let qa = engine.make_sym("query_all");
        let entry = engine.make_sym("entry");
        let qa_call = engine.make_call(qa, &[entry]);
        let count_sym = engine.make_sym("count");
        let count_call = engine.make_call(count_sym, &[qa_call]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        let result = engine.eval(count_call);
        assert_eq!(engine.term_num(result), 2);
    }

    // ── Scope: scope_take_pending ──

    #[test]
    fn scope_take_pending() {
        let mut engine = new_engine();
        engine.load_program("0");
        let _idx = engine.ensure_scope("dom", true);

        let emit_sym = engine.make_sym("emit");
        let dom_sym = engine.make_sym("dom");
        for i in 0..3 {
            let val = engine.make_num(i);
            let emit_call = engine.make_call(emit_sym, &[dom_sym, val]);
            engine.invalidate_cache();
            engine.reset_eval_counters();
            engine.eval(emit_call);
        }

        assert_eq!(engine.scope_pending_count("dom"), 3);
        let taken = engine.scope_take_pending("dom");
        assert_eq!(taken.len(), 3);
        assert_eq!(engine.scope_pending_count("dom"), 0);
    }

    // ── Scope: nonexistent scope returns safe defaults ──

    #[test]
    fn scope_nonexistent_defaults() {
        let engine = new_engine();
        assert_eq!(engine.scope_pending_count("nope"), 0);
        assert_eq!(engine.scope_queue_count("nope"), 0);
        assert_eq!(engine.scope_take_pending("nope"), vec![]);
    }

    // ── Multiple load_program calls accumulate rules ──

    #[test]
    fn incremental_load_program() {
        let mut engine = new_engine();
        engine.load_program("fn double(?x) = ?x * 2\n0");
        let term = engine.load_program("double(5)");
        engine.invalidate_cache();
        engine.reset_eval_counters();
        let result = engine.eval(term);
        assert_eq!(engine.term_num(result), 10);
    }

    // ── Quote ──

    #[test]
    fn quote_prevents_eval() {
        // quote(add(1, 2)) should NOT reduce to 3
        let (engine, result) = eval_program("quote(1 + 2)");
        assert_eq!(engine.term_tag(result), 2); // Call
        assert_eq!(engine.display(result), "add(1, 2)");
    }

    // ── Wildcard pattern ──

    #[test]
    fn wildcard_pattern() {
        assert_eq!(eval_num("fn first(cons(?h, _)) = ?h\nfirst(cons(10, cons(20, cons(30, nil))))"), 10);
    }

    // ── Eval counters reset ──

    #[test]
    fn eval_counters_reset() {
        let mut engine = new_engine();
        let term = engine.load_program("1 + 1");
        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(term);
        assert!(engine.eval_calls > 0);
        engine.reset_eval_counters();
        assert_eq!(engine.eval_calls, 0);
        assert_eq!(engine.step_count, 0);
        assert!(!engine.step_limit_exceeded);
    }

    // ── Concurrency ──

    /// Helper: emit a term to a named actor scope
    fn emit(engine: &mut Engine, scope_name: &str, term: TermId) {
        let emit_sym = engine.make_sym("emit");
        let scope_sym = engine.make_sym(scope_name);
        let call = engine.make_call(emit_sym, &[scope_sym, term]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        engine.eval(call);
    }

    /// Helper: build rule(key, val) term without evaluating it
    fn make_rule(engine: &mut Engine, key: TermId, val: TermId) -> TermId {
        let rule_sym = engine.make_sym("rule");
        engine.make_call(rule_sym, &[key, val])
    }

    /// Helper: count results from query_all(tag) in the current scope context
    fn query_count(engine: &mut Engine, tag: &str) -> i64 {
        let count_sym = engine.make_sym("count");
        let qa_sym = engine.make_sym("query_all");
        let tag_sym = engine.make_sym(tag);
        let qa_call = engine.make_call(qa_sym, &[tag_sym]);
        let count_call = engine.make_call(count_sym, &[qa_call]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        let result = engine.eval(count_call);
        engine.term_num(result)
    }

    #[test]
    fn concurrent_two_scopes_independent_counters() {
        // Two actor scopes each maintain independent counters.
        // Incrementing one should not affect the other.
        let mut engine = new_engine();
        engine.load_program("0");

        let counter = engine.make_sym("counter");

        // Send counter=10 to scope "alpha", counter=20 to scope "beta"
        let n10 = engine.make_num(10);
        let r1 = make_rule(&mut engine, counter, n10);
        emit(&mut engine, "alpha", r1);

        let n20 = engine.make_num(20);
        let r2 = make_rule(&mut engine, counter, n20);
        emit(&mut engine, "beta", r2);

        // Both queued
        assert_eq!(engine.scope_queue_count("alpha"), 1);
        assert_eq!(engine.scope_queue_count("beta"), 1);

        // Tick processes both
        let processed = engine.tick(10);
        assert_eq!(processed, 2);

        // Verify each scope has its own counter value
        let alpha_idx = engine.scope_map["alpha"];
        let beta_idx = engine.scope_map["beta"];
        let alpha_val = engine.scopes[alpha_idx].dynamic_rules_map[&counter];
        let beta_val = engine.scopes[beta_idx].dynamic_rules_map[&counter];
        assert_eq!(engine.term_num(alpha_val), 10);
        assert_eq!(engine.term_num(beta_val), 20);

        // Main scope should have NO counter rule
        assert!(!engine.scopes[0].dynamic_rules_map.contains_key(&counter));
    }

    #[test]
    fn concurrent_scopes_use_global_user_rules() {
        // Actor scopes can use fn rules defined via load_program (they're global).
        // Each scope independently computes fact(5)=120 via the global fact rule.
        let mut engine = new_engine();
        engine.load_program(
            "fn fact(0) = 1\nfn fact(?n) = ?n * fact(?n - 1)\n0"
        );

        // Emit { rule(result, fact(5)) } to two scopes
        let result_sym = engine.make_sym("result");
        let fact_sym = engine.make_sym("fact");
        let five = engine.make_num(5);
        let fact_call = engine.make_call(fact_sym, &[five]);
        let r = make_rule(&mut engine, result_sym, fact_call);

        emit(&mut engine, "worker_a", r);
        emit(&mut engine, "worker_b", r);

        engine.tick(10);

        // Both scopes should have result=120
        let a_idx = engine.scope_map["worker_a"];
        let b_idx = engine.scope_map["worker_b"];
        let a_result = engine.scopes[a_idx].dynamic_rules_map[&result_sym];
        let b_result = engine.scopes[b_idx].dynamic_rules_map[&result_sym];
        assert_eq!(engine.term_num(a_result), 120);
        assert_eq!(engine.term_num(b_result), 120);

        // Main scope should not have this rule
        assert!(!engine.scopes[0].dynamic_rules_map.contains_key(&result_sym));
    }

    #[test]
    fn concurrent_interleaved_updates() {
        // Multiple rounds of tick() with interleaved emit — scopes accumulate
        // state independently across rounds.
        let mut engine = new_engine();
        engine.load_program("0");

        let x = engine.make_sym("x");
        let y = engine.make_sym("y");

        // Round 1: alpha gets x=1, beta gets x=100
        let n1 = engine.make_num(1);
        let r1 = make_rule(&mut engine, x, n1);
        emit(&mut engine, "alpha", r1);
        let n100 = engine.make_num(100);
        let r2 = make_rule(&mut engine, x, n100);
        emit(&mut engine, "beta", r2);
        assert_eq!(engine.tick(10), 2);

        // Round 2: alpha gets y=2, beta gets y=200
        let n2 = engine.make_num(2);
        let r3 = make_rule(&mut engine, y, n2);
        emit(&mut engine, "alpha", r3);
        let n200 = engine.make_num(200);
        let r4 = make_rule(&mut engine, y, n200);
        emit(&mut engine, "beta", r4);
        assert_eq!(engine.tick(10), 2);

        // Alpha has x=1, y=2
        let alpha_idx = engine.scope_map["alpha"];
        assert_eq!(engine.term_num(engine.scopes[alpha_idx].dynamic_rules_map[&x]), 1);
        assert_eq!(engine.term_num(engine.scopes[alpha_idx].dynamic_rules_map[&y]), 2);

        // Beta has x=100, y=200
        let beta_idx = engine.scope_map["beta"];
        assert_eq!(engine.term_num(engine.scopes[beta_idx].dynamic_rules_map[&x]), 100);
        assert_eq!(engine.term_num(engine.scopes[beta_idx].dynamic_rules_map[&y]), 200);
    }

    #[test]
    fn concurrent_scope_retract_isolation() {
        // Retracting a rule in one scope does not affect other scopes.
        let mut engine = new_engine();
        engine.load_program("0");

        let k = engine.make_sym("shared_key");

        // Both scopes get shared_key=42
        let n42 = engine.make_num(42);
        let r = make_rule(&mut engine, k, n42);
        emit(&mut engine, "left", r);
        emit(&mut engine, "right", r);
        engine.tick(10);

        // Retract shared_key in "left" only
        let retract_sym = engine.make_sym("retract");
        let retract_call = engine.make_call(retract_sym, &[k]);
        emit(&mut engine, "left", retract_call);
        engine.tick(10);

        // Left should have no rules, right should still have shared_key=42
        let left_idx = engine.scope_map["left"];
        let right_idx = engine.scope_map["right"];
        assert!(engine.scopes[left_idx].dynamic_rules_map.is_empty());
        assert_eq!(engine.term_num(engine.scopes[right_idx].dynamic_rules_map[&k]), 42);
    }

    #[test]
    fn concurrent_scope_query_all_isolation() {
        // query_all in one scope only sees that scope's dynamic rules.
        let mut engine = new_engine();
        engine.load_program(
            "fn count(nil) = 0\nfn count(cons(?h, ?t)) = 1 + count(?t)\n0"
        );

        let entry = engine.make_sym("entry");

        // Alpha gets 3 entries, beta gets 1 entry, main gets 2 entries
        for i in 0..3 {
            let k = engine.make_sym(&format!("a{}", i));
            let n = engine.make_num(i as i64);
            let v = engine.make_call(entry, &[n]);
            let r = make_rule(&mut engine, k, v);
            emit(&mut engine, "alpha", r);
        }
        {
            let k = engine.make_sym("b0");
            let n = engine.make_num(99);
            let v = engine.make_call(entry, &[n]);
            let r = make_rule(&mut engine, k, v);
            emit(&mut engine, "beta", r);
        }
        // Main scope gets 2 entries directly
        for i in 0..2 {
            let k = engine.make_sym(&format!("m{}", i));
            let n = engine.make_num(i as i64 + 50);
            let v = engine.make_call(entry, &[n]);
            let r = make_rule(&mut engine, k, v);
            engine.invalidate_cache();
            engine.reset_eval_counters();
            engine.eval(r);
        }

        engine.tick(10);

        // Main scope query_all should see 2
        assert_eq!(query_count(&mut engine, "entry"), 2);

        // We can't directly query in alpha/beta context from the API,
        // but we can verify the dynamic_rules counts
        let alpha_idx = engine.scope_map["alpha"];
        let beta_idx = engine.scope_map["beta"];
        assert_eq!(engine.scopes[alpha_idx].dynamic_rules.len(), 3);
        assert_eq!(engine.scopes[beta_idx].dynamic_rules.len(), 1);
        assert_eq!(engine.scopes[0].dynamic_rules.len(), 2);
    }

    #[test]
    fn concurrent_budget_fairness() {
        // With budget=1, tick processes one scope per call.
        // This verifies round-robin fairness.
        let mut engine = new_engine();
        engine.load_program("0");

        let x = engine.make_sym("x");

        // Queue 1 item in each of 3 scopes
        for (i, name) in ["s1", "s2", "s3"].iter().enumerate() {
            let n = engine.make_num(i as i64);
            let r = make_rule(&mut engine, x, n);
            emit(&mut engine, name, r);
        }

        // tick(1) should process exactly 1 item
        assert_eq!(engine.tick(1), 1);
        // 2 scopes still have work
        let total_remaining: usize = ["s1", "s2", "s3"].iter()
            .map(|n| engine.scope_queue_count(n))
            .sum();
        assert_eq!(total_remaining, 2);

        // tick(1) processes the next
        assert_eq!(engine.tick(1), 1);
        let total_remaining: usize = ["s1", "s2", "s3"].iter()
            .map(|n| engine.scope_queue_count(n))
            .sum();
        assert_eq!(total_remaining, 1);

        // tick(1) processes the last
        assert_eq!(engine.tick(1), 1);
        let total_remaining: usize = ["s1", "s2", "s3"].iter()
            .map(|n| engine.scope_queue_count(n))
            .sum();
        assert_eq!(total_remaining, 0);

        // tick with empty queues returns 0
        assert_eq!(engine.tick(1), 0);
    }

    #[test]
    fn concurrent_seq_in_actor_scope() {
        // A seq block { rule(a, 1) rule(b, 2) c } evaluated in an actor scope
        // should install both rules in that scope and return the last value.
        let mut engine = new_engine();
        engine.load_program("0");

        let a = engine.make_sym("a");
        let b = engine.make_sym("b");
        let n1 = engine.make_num(1);
        let n2 = engine.make_num(2);
        let rule_sym = engine.make_sym("rule");
        let seq_sym = engine.make_sym("seq");

        let r1 = engine.make_call(rule_sym, &[a, n1]);
        let r2 = engine.make_call(rule_sym, &[b, n2]);
        let seq_inner = engine.make_call(seq_sym, &[r1, r2]);

        emit(&mut engine, "worker", seq_inner);
        engine.tick(10);

        let widx = engine.scope_map["worker"];
        assert_eq!(engine.scopes[widx].dynamic_rules.len(), 2);
        assert_eq!(engine.term_num(engine.scopes[widx].dynamic_rules_map[&a]), 1);
        assert_eq!(engine.term_num(engine.scopes[widx].dynamic_rules_map[&b]), 2);

        // Main scope unaffected
        assert!(engine.scopes[0].dynamic_rules_map.is_empty());
    }

    #[test]
    fn concurrent_scope_survives_gc() {
        // Multiple scopes with work queues and dynamic rules survive GC.
        let mut engine = new_engine();
        engine.load_program("0");

        let counter = engine.make_sym("counter");

        // Set up 3 scopes with rules, then queue more work
        for (i, name) in ["a", "b", "c"].iter().enumerate() {
            let n = engine.make_num(i as i64);
            let r = make_rule(&mut engine, counter, n);
            emit(&mut engine, name, r);
        }
        engine.tick(10); // Install rules

        // Queue more work that hasn't been processed yet
        let n99 = engine.make_num(99);
        let pending_rule = make_rule(&mut engine, counter, n99);
        emit(&mut engine, "a", pending_rule);
        assert_eq!(engine.scope_queue_count("a"), 1);

        // GC
        engine.gc();

        // All 3 scopes should still have their counter rules
        for (i, name) in ["a", "b", "c"].iter().enumerate() {
            let sidx = engine.scope_map[*name];
            let val = engine.scopes[sidx].dynamic_rules_map[&counter];
            assert_eq!(engine.term_num(val), i as i64);
        }

        // Pending work in "a" should survive GC
        assert_eq!(engine.scope_queue_count("a"), 1);

        // Process the pending work — counter in "a" should update to 99
        engine.tick(10);
        let a_idx = engine.scope_map["a"];
        let a_val = engine.scopes[a_idx].dynamic_rules_map[&counter];
        assert_eq!(engine.term_num(a_val), 99);
    }

    #[test]
    fn concurrent_emit_from_actor_to_output() {
        // An actor scope processing a message that emits to an output scope.
        // The output scope should buffer the value.
        let mut engine = new_engine();
        engine.load_program(
            "fn do_work(?x) = { rule(done, true) @out ?x * 2 }\n0"
        );
        engine.ensure_scope("out", true);

        // Emit do_work(21) to actor scope
        let do_work = engine.make_sym("do_work");
        let n21 = engine.make_num(21);
        let work_call = engine.make_call(do_work, &[n21]);
        emit(&mut engine, "worker", work_call);

        engine.tick(10);

        // Worker scope should have done=true
        let widx = engine.scope_map["worker"];
        let done = engine.make_sym("done");
        let done_val = engine.scopes[widx].dynamic_rules_map[&done];
        assert_eq!(engine.term_sym_name(done_val), "true");

        // Output scope "out" should have buffered 42
        assert_eq!(engine.scope_pending_count("out"), 1);
        let pending = engine.scope_pending_get("out", 0);
        assert_eq!(engine.term_num(pending), 42);
    }

    // ── Vectors ──

    #[test]
    fn vec_empty() {
        assert_eq!(eval_str("[]"), "[]");
    }

    #[test]
    fn vec_display() {
        assert_eq!(eval_str("[1, 2, 3]"), "[1, 2, 3]");
    }

    #[test]
    fn vec_get() {
        assert_eq!(eval_num("vec_get([10, 20, 30], 0)"), 10);
        assert_eq!(eval_num("vec_get([10, 20, 30], 2)"), 30);
    }

    #[test]
    fn vec_len() {
        assert_eq!(eval_num("vec_len([1, 2, 3])"), 3);
        assert_eq!(eval_num("vec_len([])"), 0);
    }

    #[test]
    fn vec_push() {
        assert_eq!(eval_str("vec_push([1, 2], 3)"), "[1, 2, 3]");
    }

    #[test]
    fn vec_concat() {
        assert_eq!(eval_str("vec_concat([1, 2], [3, 4])"), "[1, 2, 3, 4]");
    }

    #[test]
    fn vec_slice() {
        assert_eq!(eval_str("vec_slice([10, 20, 30, 40], 1, 3)"), "[20, 30]");
    }

    #[test]
    fn vec_pattern_match() {
        assert_eq!(eval_num(
            "fn first(vec(?a, ?b, ?c)) = ?a\nfirst([10, 20, 30])"
        ), 10);
    }

    // ── Sets ──

    #[test]
    fn set_empty() {
        assert_eq!(eval_str("#{}"), "#{}");
    }

    #[test]
    fn set_display() {
        assert_eq!(eval_str("#{1, 2, 3}"), "#{1, 2, 3}");
    }

    #[test]
    fn set_canonical_order() {
        // Sets are sorted: 3,1,2 becomes 1,2,3
        assert_eq!(eval_str("#{3, 1, 2}"), "#{1, 2, 3}");
    }

    #[test]
    fn set_dedup() {
        assert_eq!(eval_str("#{1, 2, 2, 3}"), "#{1, 2, 3}");
    }

    #[test]
    fn set_contains() {
        assert_eq!(eval_sym("set_contains(#{1, 2, 3}, 2)"), "true");
        assert_eq!(eval_sym("set_contains(#{1, 2, 3}, 4)"), "false");
    }

    #[test]
    fn set_insert() {
        assert_eq!(eval_str("set_insert(#{1, 3}, 2)"), "#{1, 2, 3}");
    }

    #[test]
    fn set_insert_dedup() {
        assert_eq!(eval_str("set_insert(#{1, 2}, 2)"), "#{1, 2}");
    }

    #[test]
    fn set_remove() {
        assert_eq!(eval_str("set_remove(#{1, 2, 3}, 2)"), "#{1, 3}");
    }

    #[test]
    fn set_union() {
        assert_eq!(eval_str("set_union(#{1, 2}, #{2, 3})"), "#{1, 2, 3}");
    }

    #[test]
    fn set_intersect() {
        assert_eq!(eval_str("set_intersect(#{1, 2, 3}, #{2, 3, 4})"), "#{2, 3}");
    }

    #[test]
    fn set_len() {
        assert_eq!(eval_num("set_len(#{1, 2, 3})"), 3);
        assert_eq!(eval_num("set_len(#{})"), 0);
    }

    // ── Maps ──

    #[test]
    fn map_empty() {
        assert_eq!(eval_str("{}"), "{}");
    }

    #[test]
    fn map_display() {
        assert_eq!(eval_str("{:x 1 :y 2}"), "{:x 1 :y 2}");
    }

    #[test]
    fn map_canonical_order() {
        // Maps sorted by key: :y before :x alphabetically? No, :x < :y
        assert_eq!(eval_str("{:y 2 :x 1}"), "{:x 1 :y 2}");
    }

    #[test]
    fn map_dedup_keys() {
        // Last value wins for duplicate keys
        assert_eq!(eval_str("{:x 1 :x 2}"), "{:x 2}");
    }

    #[test]
    fn map_get() {
        assert_eq!(eval_num("map_get({:x 10 :y 20}, :x)"), 10);
        assert_eq!(eval_num("map_get({:x 10 :y 20}, :y)"), 20);
    }

    #[test]
    fn map_set() {
        assert_eq!(eval_str("map_set({:x 1}, :y, 2)"), "{:x 1 :y 2}");
    }

    #[test]
    fn map_set_overwrite() {
        assert_eq!(eval_str("map_set({:x 1 :y 2}, :x, 99)"), "{:x 99 :y 2}");
    }

    #[test]
    fn map_remove() {
        assert_eq!(eval_str("map_remove({:x 1 :y 2}, :x)"), "{:y 2}");
    }

    #[test]
    fn map_merge() {
        assert_eq!(eval_str("map_merge({:x 1 :y 2}, {:y 99 :z 3})"), "{:x 1 :y 99 :z 3}");
    }

    #[test]
    fn map_keys() {
        assert_eq!(eval_str("map_keys({:x 1 :y 2})"), "[:x, :y]");
    }

    #[test]
    fn map_values() {
        assert_eq!(eval_str("map_values({:x 10 :y 20})"), "[10, 20]");
    }

    #[test]
    fn map_len() {
        assert_eq!(eval_num("map_len({:x 1 :y 2})"), 2);
        assert_eq!(eval_num("map_len({})"), 0);
    }

    // ── Partial map matching ──

    #[test]
    fn map_partial_match() {
        assert_eq!(eval_num(
            "fn get_x({:x ?v}) = ?v\nget_x({:x 42 :y 99})"
        ), 42);
    }

    #[test]
    fn map_partial_match_multi_key() {
        assert_eq!(eval_num(
            "fn sum_xy({:x ?a :y ?b}) = ?a + ?b\nsum_xy({:x 10 :y 20 :z 30})"
        ), 30);
    }

    #[test]
    fn map_partial_match_greet() {
        assert_eq!(eval_sym(
            "fn greet({:name ?n}) = \"Hello, \" ++ ?n\ngreet({:name \"Alice\" :age 30})"
        ), "Hello, Alice");
    }

    #[test]
    fn map_in_vec() {
        assert_eq!(eval_str("[{:x 1}, {:y 2}]"), "[{:x 1}, {:y 2}]");
    }

    // ── Spread patterns ──

    #[test]
    fn vec_spread_first_rest() {
        // [?first, ?rest...] matches [1, 2, 3] → ?first=1, ?rest=[2, 3]
        assert_eq!(eval_num(
            "fn first([?first, ?rest...]) = ?first\nfirst([1, 2, 3])"
        ), 1);
    }

    #[test]
    fn vec_spread_rest_is_vec() {
        // The rest binding should be a vec
        assert_eq!(eval_str(
            "fn rest([?first, ?rest...]) = ?rest\nrest([1, 2, 3])"
        ), "[2, 3]");
    }

    #[test]
    fn vec_spread_rest_empty() {
        // When only one element, rest should be empty vec
        assert_eq!(eval_str(
            "fn rest([?first, ?rest...]) = ?rest\nrest([5])"
        ), "[]");
    }

    #[test]
    fn vec_exact_no_spread() {
        // [?only] matches [5] but NOT [5, 6] (exact without spread)
        assert_eq!(eval_num(
            "fn only([?x]) = ?x\nonly([5])"
        ), 5);
    }

    #[test]
    fn vec_wild_spread() {
        // [?a, ?b, _...] matches [1, 2, 3, 4] → ?a=1, ?b=2
        assert_eq!(eval_num(
            "fn second([?a, ?b, _...]) = ?b\nsecond([1, 2, 3, 4])"
        ), 2);
    }

    #[test]
    fn vec_spread_in_rhs() {
        // Spread in RHS: fn wrap(?items...) = [:wrapped, ?items...]
        assert_eq!(eval_str(
            "fn wrap(?items...) = [:wrapped, ?items...]\nwrap(1, 2, 3)"
        ), "[:wrapped, 1, 2, 3]");
    }

    // ── Set subset matching ──

    #[test]
    fn set_subset_match() {
        // #{1, 2} matches #{1, 2, 3} (subset)
        assert_eq!(eval_num(
            "fn has_one_two(#{1, 2}) = 1\nhas_one_two(#{1, 2, 3})"
        ), 1);
    }

    #[test]
    fn set_subset_match_var() {
        // #{1, ?x} matches #{1, 2} → ?x=2
        assert_eq!(eval_num(
            "fn get_other(#{1, ?x}) = ?x\nget_other(#{1, 2})"
        ), 2);
    }

    #[test]
    fn set_spread_rest() {
        // #{1, ?rest...} matches #{1, 2, 3} → ?rest=#{2, 3}
        assert_eq!(eval_str(
            "fn rest(#{1, ?rest...}) = ?rest\nrest(#{1, 2, 3})"
        ), "#{2, 3}");
    }

    // ── Map with variable keys ──

    #[test]
    fn map_var_key() {
        // {?k ?v} parses and matches {:x 1} → ?k=:x, ?v=1
        assert_eq!(eval_num(
            "fn val({?k ?v}) = ?v\nval({:x 42})"
        ), 42);
    }

    // ── PuzzleScript DSL tests ──

    #[test]
    fn ps_parse_cells() {
        assert_eq!(eval_str("[a | b | c]"), "cells(a, b, c)");
    }

    #[test]
    fn ps_parse_dir_right() {
        assert_eq!(eval_str("[> player | crate]"), "cells(dir_right(player), crate)");
    }

    #[test]
    fn ps_parse_dir_left() {
        assert_eq!(eval_str("[< player | crate]"), "cells(dir_left(player), crate)");
    }

    #[test]
    fn ps_parse_arrow() {
        assert_eq!(eval_str(
            "[> player | crate] -> [> player | > crate]"
        ), "arrow(cells(dir_right(player), crate), cells(dir_right(player), dir_right(crate)))");
    }

    #[test]
    fn ps_parse_dir_left_arrow() {
        assert_eq!(eval_str(
            "[< player | crate] -> [< player | < crate]"
        ), "arrow(cells(dir_left(player), crate), cells(dir_left(player), dir_left(crate)))");
    }

    #[test]
    fn ps_parse_ellipsis() {
        assert_eq!(eval_str("[eyeball | ... | player]"), "cells(eyeball, ellipsis, player)");
    }

    #[test]
    fn ps_parse_ellipsis_with_dirs() {
        assert_eq!(eval_str(
            "[eyeball | ... | player] -> [> eyeball | ... | player]"
        ), "arrow(cells(eyeball, ellipsis, player), cells(dir_right(eyeball), ellipsis, player))");
    }

    #[test]
    fn ps_parse_mixed() {
        assert_eq!(eval_str("[> a | b | < c]"), "cells(dir_right(a), b, dir_left(c))");
    }

    #[test]
    fn ps_cells_vec_len() {
        assert_eq!(eval_num("vec_len([a | b | c])"), 3);
    }

    #[test]
    fn ps_cells_vec_get() {
        assert_eq!(eval_str("vec_get([a | b | c], 1)"), "b");
    }

    #[test]
    fn ps_cells_vec_get_dir() {
        assert_eq!(eval_str("vec_get([> player | crate], 0)"), "dir_right(player)");
    }

    #[test]
    fn ps_cell_obj_pattern_match() {
        // Pattern matching can extract the inner object from dir_right/dir_left wrappers
        assert_eq!(eval_str(
            "fn cell_obj(dir_right(?x)) = ?x\n\
             fn cell_obj(dir_left(?x)) = ?x\n\
             fn cell_obj(?x) = ?x\n\
             cell_obj(dir_right(player))"
        ), "player");
    }

    #[test]
    fn ps_cell_obj_dir_left() {
        assert_eq!(eval_str(
            "fn cell_obj(dir_right(?x)) = ?x\n\
             fn cell_obj(dir_left(?x)) = ?x\n\
             fn cell_obj(?x) = ?x\n\
             cell_obj(dir_left(crate))"
        ), "crate");
    }

    #[test]
    fn ps_cell_obj_bare() {
        assert_eq!(eval_str(
            "fn cell_obj(dir_right(?x)) = ?x\n\
             fn cell_obj(dir_left(?x)) = ?x\n\
             fn cell_obj(?x) = ?x\n\
             cell_obj(floor)"
        ), "floor");
    }

    #[test]
    fn ps_cell_dir_pattern_match() {
        // cell_dir returns 1 for dir_right, -1 for dir_left, 0 for bare
        assert_eq!(eval_num(
            "fn cell_dir(dir_right(?x)) = 1\n\
             fn cell_dir(dir_left(?x)) = -1\n\
             fn cell_dir(?x) = 0\n\
             cell_dir(dir_right(player))"
        ), 1);
        assert_eq!(eval_num(
            "fn cell_dir(dir_right(?x)) = 1\n\
             fn cell_dir(dir_left(?x)) = -1\n\
             fn cell_dir(?x) = 0\n\
             cell_dir(dir_left(crate))"
        ), -1);
        assert_eq!(eval_num(
            "fn cell_dir(dir_right(?x)) = 1\n\
             fn cell_dir(dir_left(?x)) = -1\n\
             fn cell_dir(?x) = 0\n\
             cell_dir(wall)"
        ), 0);
    }

    #[test]
    fn ps_is_ellipsis() {
        assert_eq!(eval_str(
            "fn is_ellipsis(ellipsis) = true\n\
             fn is_ellipsis(?x) = false\n\
             is_ellipsis(ellipsis)"
        ), "true");
        assert_eq!(eval_str(
            "fn is_ellipsis(ellipsis) = true\n\
             fn is_ellipsis(?x) = false\n\
             is_ellipsis(player)"
        ), "false");
    }

    #[test]
    fn ps_rules_in_vec() {
        // game_rules() returns a vec of arrow terms
        assert_eq!(eval_num(
            "fn game_rules() = [\n\
               [> player | crate] -> [> player | > crate]\n\
             ]\n\
             vec_len(game_rules())"
        ), 1);
    }

    #[test]
    fn ps_extract_lhs_rhs() {
        // Pattern match on arrow to extract LHS and RHS cells
        assert_eq!(eval_str(
            "fn lhs(arrow(?l, ?r)) = ?l\n\
             fn rhs(arrow(?l, ?r)) = ?r\n\
             lhs([> player | crate] -> [> player | > crate])"
        ), "cells(dir_right(player), crate)");
    }

    // ── PuzzleScript generic rule application tests ──

    // Shared engine code for rule application tests
    const PS_ENGINE: &str = "\
fn cell_obj(dir_right(?x)) = ?x
fn cell_obj(dir_left(?x)) = ?x
fn cell_obj(?x) = ?x
fn cell_dir(dir_right(?x)) = 1
fn cell_dir(dir_left(?x)) = -1
fn cell_dir(?x) = 0
fn is_ellipsis(ellipsis) = true
fn is_ellipsis(?x) = false
fn is_passable(floor) = true
fn is_passable(?x) = false

fn first_dir(?cells, ?i) =
  if ?i == vec_len(?cells) then 0
  else if cell_dir(vec_get(?cells, ?i)) != 0 then cell_dir(vec_get(?cells, ?i))
  else first_dir(?cells, ?i + 1)

fn dir_ok(?fd, ?dx, ?dy, ?sdx, ?sdy) =
  if ?fd == 0 then true
  else if ?fd == 1 then if ?dx == ?sdx then if ?dy == ?sdy then true else false else false
  else if ?dx == (0 - ?sdx) then if ?dy == (0 - ?sdy) then true else false else false

fn scan_sdx(0) = 1
fn scan_sdx(1) = 0
fn scan_sdx(2) = 0 - 1
fn scan_sdx(3) = 0
fn scan_sdy(0) = 0
fn scan_sdy(1) = 1
fn scan_sdy(2) = 0
fn scan_sdy(3) = 0 - 1

fn count_non_ellipsis(?cells, ?i) =
  if ?i == vec_len(?cells) then 0
  else if is_ellipsis(vec_get(?cells, ?i)) then count_non_ellipsis(?cells, ?i + 1)
  else 1 + count_non_ellipsis(?cells, ?i + 1)

fn match_lhs(?cells, ?i, ?ax, ?ay, ?sdx, ?sdy, ?off) =
  if ?i == vec_len(?cells) then ?off
  else if is_ellipsis(vec_get(?cells, ?i)) then
    match_ellipsis(?cells, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off)
  else if cell_obj(vec_get(?cells, ?i)) == cell(?ax + ?off * ?sdx, ?ay + ?off * ?sdy) then
    match_lhs(?cells, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + 1)
  else false

fn match_ellipsis(?cells, ?next_i, ?ax, ?ay, ?sdx, ?sdy, ?off) =
  if ?next_i == vec_len(?cells) then ?off
  else if cell_obj(vec_get(?cells, ?next_i)) == cell(?ax + ?off * ?sdx, ?ay + ?off * ?sdy) then
    match_lhs(?cells, ?next_i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + 1)
  else if ?off > 20 then false
  else match_ellipsis(?cells, ?next_i, ?ax, ?ay, ?sdx, ?sdy, ?off + 1)

fn check_dest(?rhs, ?i, ?ax, ?ay, ?sdx, ?sdy, ?off, ?n, ?gap) =
  if ?i == vec_len(?rhs) then true
  else if is_ellipsis(vec_get(?rhs, ?i)) then
    check_dest(?rhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + ?gap, ?n, ?gap)
  else check_one_dest(vec_get(?rhs, ?i), ?i, ?ax, ?ay, ?sdx, ?sdy, ?off, ?rhs, ?n, ?gap)

fn check_one_dest(?cell, ?i, ?ax, ?ay, ?sdx, ?sdy, ?off, ?rhs, ?n, ?gap) =
  if cell_dir(?cell) == 0 then
    check_dest(?rhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + 1, ?n, ?gap)
  else if (?off + cell_dir(?cell)) >= 0 then
    if (?off + cell_dir(?cell)) < ?n then
      check_dest(?rhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + 1, ?n, ?gap)
    else
      if is_passable(cell(?ax + (?off + cell_dir(?cell)) * ?sdx, ?ay + (?off + cell_dir(?cell)) * ?sdy)) then
        check_dest(?rhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + 1, ?n, ?gap)
      else false
  else
    if is_passable(cell(?ax + (?off + cell_dir(?cell)) * ?sdx, ?ay + (?off + cell_dir(?cell)) * ?sdy)) then
      check_dest(?rhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + 1, ?n, ?gap)
    else false

fn clear_cells(?lhs, ?i, ?ax, ?ay, ?sdx, ?sdy, ?off, ?gap) =
  if ?i == vec_len(?lhs) then 0
  else if is_ellipsis(vec_get(?lhs, ?i)) then
    clear_cells(?lhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + ?gap, ?gap)
  else {
    rule(cell(?ax + ?off * ?sdx, ?ay + ?off * ?sdy), floor)
    clear_cells(?lhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + 1, ?gap)
  }

fn place_cells(?rhs, ?i, ?ax, ?ay, ?sdx, ?sdy, ?off, ?gap) =
  if ?i == vec_len(?rhs) then 0
  else if is_ellipsis(vec_get(?rhs, ?i)) then
    place_cells(?rhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + ?gap, ?gap)
  else {
    rule(cell(?ax + (?off + cell_dir(vec_get(?rhs, ?i))) * ?sdx, ?ay + (?off + cell_dir(vec_get(?rhs, ?i))) * ?sdy), cell_obj(vec_get(?rhs, ?i)))
    place_cells(?rhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + 1, ?gap)
  }

fn update_player_pos(?rhs, ?i, ?ax, ?ay, ?sdx, ?sdy, ?off, ?gap) =
  if ?i == vec_len(?rhs) then 0
  else if is_ellipsis(vec_get(?rhs, ?i)) then
    update_player_pos(?rhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + ?gap, ?gap)
  else if cell_obj(vec_get(?rhs, ?i)) == player then {
    rule(player_x, ?ax + (?off + cell_dir(vec_get(?rhs, ?i))) * ?sdx)
    rule(player_y, ?ay + (?off + cell_dir(vec_get(?rhs, ?i))) * ?sdy)
  }
  else update_player_pos(?rhs, ?i + 1, ?ax, ?ay, ?sdx, ?sdy, ?off + 1, ?gap)

fn apply_rule(arrow(?lhs, ?rhs), ?dx, ?dy) =
  try_scan(?lhs, ?rhs, ?dx, ?dy, first_dir(?lhs, 0), 0)

fn try_scan(?lhs, ?rhs, ?dx, ?dy, ?fd, 4) = false
fn try_scan(?lhs, ?rhs, ?dx, ?dy, ?fd, ?si) =
  if dir_ok(?fd, ?dx, ?dy, scan_sdx(?si), scan_sdy(?si)) then
    try_match(?lhs, ?rhs, ?dx, ?dy, ?fd, ?si, match_lhs(?lhs, 0, player_x, player_y, scan_sdx(?si), scan_sdy(?si), 0))
  else try_scan(?lhs, ?rhs, ?dx, ?dy, ?fd, ?si + 1)

fn try_match(?lhs, ?rhs, ?dx, ?dy, ?fd, ?si, false) =
  try_scan(?lhs, ?rhs, ?dx, ?dy, ?fd, ?si + 1)
fn try_match(?lhs, ?rhs, ?dx, ?dy, ?fd, ?si, ?n) =
  apply_with_gap(?lhs, ?rhs, scan_sdx(?si), scan_sdy(?si), ?n, ?n - count_non_ellipsis(?lhs, 0))

fn apply_with_gap(?lhs, ?rhs, ?sdx, ?sdy, ?n, ?gap) =
  if check_dest(?rhs, 0, player_x, player_y, ?sdx, ?sdy, 0, ?n, ?gap) == true then {
    clear_cells(?lhs, 0, player_x, player_y, ?sdx, ?sdy, 0, ?gap)
    place_cells(?rhs, 0, player_x, player_y, ?sdx, ?sdy, 0, ?gap)
    update_player_pos(?rhs, 0, player_x, player_y, ?sdx, ?sdy, 0, ?gap)
    true
  } else false

fn apply_rule(?other, ?dx, ?dy) = false
";

    #[test]
    fn ps_push_right() {
        // [> player | crate] -> [> player | > crate], move right
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), player) rule(cell(1, 0), crate) rule(cell(2, 0), floor)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([> player | crate] -> [> player | > crate], 1, 0)\n\
              cell(1, 0)\n\
            }");
        assert_eq!(eval_str(&code), "player");
    }

    #[test]
    fn ps_push_right_crate_moved() {
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), player) rule(cell(1, 0), crate) rule(cell(2, 0), floor)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([> player | crate] -> [> player | > crate], 1, 0)\n\
              cell(2, 0)\n\
            }");
        assert_eq!(eval_str(&code), "crate");
    }

    #[test]
    fn ps_push_right_old_pos_cleared() {
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), player) rule(cell(1, 0), crate) rule(cell(2, 0), floor)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([> player | crate] -> [> player | > crate], 1, 0)\n\
              cell(0, 0)\n\
            }");
        assert_eq!(eval_str(&code), "floor");
    }

    #[test]
    fn ps_push_left() {
        // Push left: player at (2,0), crate at (1,0), floor at (0,0)
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), floor) rule(cell(1, 0), crate) rule(cell(2, 0), player)\n\
              rule(player_x, 2) rule(player_y, 0)\n\
              apply_rule([> player | crate] -> [> player | > crate], -1, 0)\n\
              cell(1, 0)\n\
            }");
        assert_eq!(eval_str(&code), "player");
    }

    #[test]
    fn ps_push_down() {
        // Push down: player at (0,0), crate at (0,1), floor at (0,2)
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), player) rule(cell(0, 1), crate) rule(cell(0, 2), floor)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([> player | crate] -> [> player | > crate], 0, 1)\n\
              cell(0, 1)\n\
            }");
        assert_eq!(eval_str(&code), "player");
    }

    #[test]
    fn ps_push_blocked() {
        // Push into wall: player at (0,0), crate at (1,0), wall at (2,0)
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), player) rule(cell(1, 0), crate) rule(cell(2, 0), wall)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([> player | crate] -> [> player | > crate], 1, 0)\n\
            }");
        // Should return false (push blocked)
        assert_eq!(eval_str(&code), "false");
    }

    #[test]
    fn ps_chain_push_3_cells() {
        // [> player | crate | crate] -> [> player | > crate | > crate]
        // Player pushes two crates: p c c . -> . p c c
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), player) rule(cell(1, 0), crate) rule(cell(2, 0), crate) rule(cell(3, 0), floor)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([> player | crate | crate] -> [> player | > crate | > crate], 1, 0)\n\
              cell(0, 0) ++ \" \" ++ cell(1, 0) ++ \" \" ++ cell(2, 0) ++ \" \" ++ cell(3, 0)\n\
            }");
        assert_eq!(eval_str(&code), "floor player crate crate");
    }

    #[test]
    fn ps_dir_left_rule() {
        // [< player | crate] -> [< player | < crate]
        // Pull: player moves LEFT (dx=-1), crate at right follows.
        // < means "opposite of scan direction". With dx=-1, scan RIGHT matches.
        // Setup: floor at (-1,0), player at (0,0), crate at (1,0)
        // Result: player at (-1,0), crate at (0,0), floor at (1,0)
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(-1, 0), floor) rule(cell(0, 0), player) rule(cell(1, 0), crate)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([< player | crate] -> [< player | < crate], -1, 0)\n\
              cell(-1, 0) ++ \" \" ++ cell(0, 0) ++ \" \" ++ cell(1, 0)\n\
            }");
        assert_eq!(eval_str(&code), "player crate floor");
    }

    #[test]
    fn ps_dir_left_player_pos() {
        // After dir_left pull rule, player_x should be updated
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(-1, 0), floor) rule(cell(0, 0), player) rule(cell(1, 0), crate)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([< player | crate] -> [< player | < crate], -1, 0)\n\
              player_x\n\
            }");
        assert_eq!(eval_num(&code), -1);
    }

    #[test]
    fn ps_dir_left_not_when_pushing_right() {
        // Regression: [< player | crate] must NOT fire when moving right (dx=1).
        // The < constraint requires movement opposite to scan direction.
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(-1, 0), floor) rule(cell(0, 0), player) rule(cell(1, 0), crate)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([< player | crate] -> [< player | < crate], 1, 0)\n\
            }");
        assert_eq!(eval_str(&code), "false");
    }

    #[test]
    fn ps_mixed_dirs() {
        // [> player | < crate] -> [> player | < crate]
        // player moves forward, crate moves backward
        // Setup: player at (0,0), crate at (1,0), floor at (-1,0), floor at (2,0)
        // dx=1: player goes to 0+1=1, crate goes to 1+(-1)=0
        // They swap positions!
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), player) rule(cell(1, 0), crate) rule(cell(2, 0), floor)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([> player | < crate] -> [> player | < crate], 1, 0)\n\
              cell(0, 0) ++ \" \" ++ cell(1, 0)\n\
            }");
        assert_eq!(eval_str(&code), "crate player");
    }

    #[test]
    fn ps_stationary_objects() {
        // [> player | crate] -> [> player | crate]
        // player moves, crate stays (no dir marker on crate in RHS)
        // Setup: player at (0,0), crate at (1,0)
        // Result: crate stays at (1,0), player tries to go to (1,0) -> overlap!
        // Actually the player would land on the crate's cell.
        // This is how "destroy" works in PuzzleScript - objects can share cells.
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), player) rule(cell(1, 0), crate)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([> player | crate] -> [> player | crate], 1, 0)\n\
              cell(0, 0)\n\
            }");
        // Old position should be cleared
        assert_eq!(eval_str(&code), "floor");
    }

    #[test]
    fn ps_ellipsis_scan() {
        // [eyeball | ... | player] -> [> eyeball | ... | player]
        // Eyeball at (0,0), floor at (1,0), floor at (2,0), player at (3,0)
        // Match: eyeball at 0, ellipsis spans 1-2, player at 3
        // Result: eyeball moves to (1,0)
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), eyeball) rule(cell(1, 0), floor) rule(cell(2, 0), floor) rule(cell(3, 0), player)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([eyeball | ... | player] -> [> eyeball | ... | player], 1, 0)\n\
              cell(0, 0) ++ \" \" ++ cell(1, 0)\n\
            }");
        assert_eq!(eval_str(&code), "floor eyeball");
    }

    #[test]
    fn ps_ellipsis_gap_of_one() {
        // [eyeball | ... | player] with one cell gap
        // Eyeball at (0,0), floor at (1,0), player at (2,0)
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), eyeball) rule(cell(1, 0), floor) rule(cell(2, 0), player)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([eyeball | ... | player] -> [> eyeball | ... | player], 1, 0)\n\
              cell(0, 0) ++ \" \" ++ cell(1, 0) ++ \" \" ++ cell(2, 0)\n\
            }");
        // eyeball moves from (0,0) to (1,0), player stays at (2,0)
        assert_eq!(eval_str(&code), "floor eyeball player");
    }

    #[test]
    fn ps_ellipsis_no_match() {
        // [eyeball | ... | player] but no player in the scan direction
        let code = format!("{}\n{}", PS_ENGINE,
            "{\n\
              rule(cell(0, 0), eyeball) rule(cell(1, 0), floor) rule(cell(2, 0), wall)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              apply_rule([eyeball | ... | player] -> [> eyeball | ... | player], 1, 0)\n\
            }");
        assert_eq!(eval_str(&code), "false");
    }

    #[test]
    fn ps_multiple_rules() {
        // Try multiple rules via game_rules() vec
        let code = format!("{}\n{}", PS_ENGINE,
            "fn game_rules() = [\n\
               [> player | crate] -> [> player | > crate]\n\
             ]\n\
             fn try_rules(?rules, ?dx, ?dy, ?i) =\n\
               if ?i == vec_len(?rules) then false\n\
               else if apply_rule(vec_get(?rules, ?i), ?dx, ?dy) == true then true\n\
               else try_rules(?rules, ?dx, ?dy, ?i + 1)\n\
             {\n\
              rule(cell(0, 0), player) rule(cell(1, 0), crate) rule(cell(2, 0), floor)\n\
              rule(player_x, 0) rule(player_y, 0)\n\
              try_rules(game_rules(), 1, 0, 0)\n\
             }");
        assert_eq!(eval_str(&code), "true");
    }

    // ── make_term / defclause builtins ──

    #[test]
    fn make_term_basic() {
        // make_term(bar, [1, 2, 3]) should produce bar(1, 2, 3)
        assert_eq!(eval_str("make_term(bar, [1, 2, 3])"), "bar(1, 2, 3)");
    }

    #[test]
    fn make_term_set_no_sort() {
        // make_term(set, [3, 1, 2]) should NOT sort — raw call construction
        assert_eq!(eval_str("make_term(set, [3, 1, 2])"), "#{3, 1, 2}");
    }

    #[test]
    fn defclause_identity() {
        // defclause(foo, [__pvar(0)], __pvar(0)) installs fn foo(?v0) = ?v0
        // Then foo(42) should return 42
        assert_eq!(eval_str("{ defclause(foo, [__pvar(0)], __pvar(0)) \n foo(42) }"), "42");
    }

    #[test]
    fn defclause_constant() {
        // defclause(bar, [__pvar(0), __pvar(1)], __pvar(1)) installs fn bar(?v0, ?v1) = ?v1
        assert_eq!(eval_str("{ defclause(bar, [__pvar(0), __pvar(1)], __pvar(1)) \n bar(10, 20) }"), "20");
    }

    #[test]
    fn defclause_with_set_spread() {
        // Install step(right, #{at(?v0, ?v1, player), ?v2...}) = ?v2
        // This tests __spread in a set pattern constructed via make_term
        let code = "\
{ \
  defclause(step, \
    [right, make_term(set, [at(__pvar(0), __pvar(1), player), __spread(2)])], \
    __pvar(2)) \
  step(right, #{at(3, 4, player), at(0, 0, wall)}) \
}";
        // Should match and return the rest-set (just wall)
        assert_eq!(eval_str(code), "#{at(0, 0, wall)}");
    }

    #[test]
    fn defclause_push_rule() {
        // Full push rule: player + crate + floor → floor + player + crate
        let code = "\
{ \
  defclause(step, \
    [right, make_term(set, [at(__pvar(0), __pvar(1), player), at(add(__pvar(0), 1), __pvar(1), crate), at(add(__pvar(0), 2), __pvar(1), floor), __spread(2)])], \
    make_term(set, [at(__pvar(0), __pvar(1), floor), at(add(__pvar(0), 1), __pvar(1), player), at(add(__pvar(0), 2), __pvar(1), crate), __spread(2)])) \
  defclause(step, [__pvar(0), __pvar(1)], __pvar(1)) \
  step(right, #{at(0, 0, player), at(1, 0, crate), at(2, 0, floor), at(3, 0, wall)}) \
}";
        let result = eval_str(code);
        // Player pushed crate: player→1, crate→2, floor at 0
        assert!(result.contains("at(0, 0, floor)"));
        assert!(result.contains("at(1, 0, player)"));
        assert!(result.contains("at(2, 0, crate)"));
        assert!(result.contains("at(3, 0, wall)"));
    }

    #[test]
    fn defclause_with_wild() {
        // __wild should become Wildcard pattern
        assert_eq!(eval_str("{ defclause(f, [__wild], 99) \n f(anything) }"), "99");
    }

    #[test]
    fn compiler_rules_push_right() {
        // End-to-end: compiler rules expand ps() into step() clauses
        // Must use { } block to sequence compile() calls before step()
        let code = "\
fn compile(ps(?match, ?result)) = { \
  compile_dir(right, 1, 0, to_vec(?match), to_vec(?result)) \
  compile_dir(left, 0 - 1, 0, to_vec(?match), to_vec(?result)) \
  compile_dir(down, 0, 1, to_vec(?match), to_vec(?result)) \
  compile_dir(up, 0, 0 - 1, to_vec(?match), to_vec(?result)) \
}
fn to_vec(cells(?items...)) = ?items
fn to_vec(?v) = ?v
fn compile_dir(?dir, ?dx, ?dy, ?match, ?result) = \
  defclause(step, \
    [?dir, make_term(set, vec_push(build_ats(?match, 0, ?dx, ?dy), __spread(2)))], \
    make_term(set, vec_push(build_ats(?result, 0, ?dx, ?dy), __spread(2))))
fn build_ats(vec(?cell, ?rest...), ?i, ?dx, ?dy) = \
  vec_concat([mk_at(?cell, ?i, ?dx, ?dy)], build_ats(?rest, ?i + 1, ?dx, ?dy))
fn build_ats(vec(), ?i, ?dx, ?dy) = []
fn mk_at(m(?e), ?i, ?dx, ?dy) = mk_at(?e, ?i, ?dx, ?dy)
fn mk_at(?entity, ?i, ?dx, ?dy) = \
  at(mk_coord(0, ?i * ?dx), mk_coord(1, ?i * ?dy), ?entity)
fn mk_coord(?var_id, 0) = __pvar(?var_id)
fn mk_coord(?var_id, ?offset) where ?offset > 0 = add(__pvar(?var_id), ?offset)
fn mk_coord(?var_id, ?offset) = sub(__pvar(?var_id), 0 - ?offset)

{ \
  compile(ps( \
    [m(player) | crate | floor], \
    [floor | m(player) | m(crate)])) \
  compile(ps( \
    [m(player) | floor], \
    [floor | m(player)])) \
  defclause(step, [__pvar(0), __pvar(1)], __pvar(1)) \
  step(right, #{at(0, 0, player), at(1, 0, crate), at(2, 0, floor), at(3, 0, wall)}) \
}";
        let result = eval_str(code);
        assert!(result.contains("at(0, 0, floor)"), "player's old pos should be floor, got: {}", result);
        assert!(result.contains("at(1, 0, player)"), "player should move to 1, got: {}", result);
        assert!(result.contains("at(2, 0, crate)"), "crate should move to 2, got: {}", result);
        assert!(result.contains("at(3, 0, wall)"), "wall should remain, got: {}", result);
    }

    #[test]
    fn compiler_rules_walk_left() {
        // Walk left: player moves onto floor
        let code = "\
fn compile(ps(?match, ?result)) = { \
  compile_dir(right, 1, 0, to_vec(?match), to_vec(?result)) \
  compile_dir(left, 0 - 1, 0, to_vec(?match), to_vec(?result)) \
  compile_dir(down, 0, 1, to_vec(?match), to_vec(?result)) \
  compile_dir(up, 0, 0 - 1, to_vec(?match), to_vec(?result)) \
}
fn to_vec(cells(?items...)) = ?items
fn to_vec(?v) = ?v
fn compile_dir(?dir, ?dx, ?dy, ?match, ?result) = \
  defclause(step, \
    [?dir, make_term(set, vec_push(build_ats(?match, 0, ?dx, ?dy), __spread(2)))], \
    make_term(set, vec_push(build_ats(?result, 0, ?dx, ?dy), __spread(2))))
fn build_ats(vec(?cell, ?rest...), ?i, ?dx, ?dy) = \
  vec_concat([mk_at(?cell, ?i, ?dx, ?dy)], build_ats(?rest, ?i + 1, ?dx, ?dy))
fn build_ats(vec(), ?i, ?dx, ?dy) = []
fn mk_at(m(?e), ?i, ?dx, ?dy) = mk_at(?e, ?i, ?dx, ?dy)
fn mk_at(?entity, ?i, ?dx, ?dy) = \
  at(mk_coord(0, ?i * ?dx), mk_coord(1, ?i * ?dy), ?entity)
fn mk_coord(?var_id, 0) = __pvar(?var_id)
fn mk_coord(?var_id, ?offset) where ?offset > 0 = add(__pvar(?var_id), ?offset)
fn mk_coord(?var_id, ?offset) = sub(__pvar(?var_id), 0 - ?offset)

{ \
  compile(ps([m(player) | floor], [floor | m(player)])) \
  defclause(step, [__pvar(0), __pvar(1)], __pvar(1)) \
  step(left, #{at(2, 0, player), at(1, 0, floor), at(0, 0, wall)}) \
}";
        let result = eval_str(code);
        assert!(result.contains("at(2, 0, floor)"), "old pos should be floor, got: {}", result);
        assert!(result.contains("at(1, 0, player)"), "player should move left, got: {}", result);
        assert!(result.contains("at(0, 0, wall)"), "wall remains, got: {}", result);
    }

    #[test]
    fn compiler_full_browser_load() {
        // Simulate exactly what the browser loads: renderer + compiler + rules
        // fn declarations at top level, compile/defclause/init in { } block
        let renderer = r##"
fn grid_w(#{size(?w, ?h), _...}) = ?w
fn grid_h(#{size(?w, ?h), _...}) = ?h
fn cell_at(?x, ?y, #{at(?x, ?y, ?entity), _...}) = ?entity
fn cell_at(?x, ?y, ?world) = empty
fn is_target(?x, ?y, #{target(?x, ?y), _...}) = true
fn is_target(?x, ?y, ?world) = false
fn append(nil, ?ys) = ?ys
fn append(cons(?h, ?t), ?ys) = cons(?h, append(?t, ?ys))
fn floor_rect(?x, ?y) = element("rect",
  [attr("x", ?x * 64 + 1), attr("y", ?y * 64 + 1),
   attr("width", 62), attr("height", 62),
   attr("rx", 4), attr("fill", "#3a3a4a")], [])
fn tile(?x, ?y, wall, ?t) = cons(element("rect",
  [attr("x", ?x * 64 + 1), attr("y", ?y * 64 + 1),
   attr("width", 62), attr("height", 62),
   attr("rx", 4), attr("fill", "#5D4E37")], []), nil)
fn tile(?x, ?y, floor, ?t) = cons(floor_rect(?x, ?y), nil)
fn tile(?x, ?y, player, ?t) = cons(floor_rect(?x, ?y), nil)
fn tile(?x, ?y, crate, ?t) = cons(floor_rect(?x, ?y), nil)
fn tile(?x, ?y, empty, ?t) = nil
fn cell_elems(?x, ?y) = tile(?x, ?y, cell_at(?x, ?y, world), is_target(?x, ?y, world))
fn render_cells(?x, ?y, ?w, ?h) =
  if ?y == ?h then nil
  else if ?x == ?w then render_cells(0, ?y + 1, ?w, ?h)
  else append(cell_elems(?x, ?y), render_cells(?x + 1, ?y, ?w, ?h))
fn render() =
  @dom element("svg",
    [attr("viewBox", "0 0 " ++ (grid_w(world) * 64) ++ " " ++ (grid_h(world) * 64)),
     attr("width", grid_w(world) * 64),
     attr("height", grid_h(world) * 64)],
    render_cells(0, 0, grid_w(world), grid_h(world)))
"##;
        let compiler = r#"
fn compile(ps(?match, ?result)) = {
  compile_dir(right, 1, 0, to_vec(?match), to_vec(?result))
  compile_dir(left, 0 - 1, 0, to_vec(?match), to_vec(?result))
  compile_dir(down, 0, 1, to_vec(?match), to_vec(?result))
  compile_dir(up, 0, 0 - 1, to_vec(?match), to_vec(?result))
}
fn to_vec(cells(?items...)) = ?items
fn to_vec(?v) = ?v
fn compile_dir(?dir, ?dx, ?dy, ?match, ?result) =
  defclause(step,
    [?dir, make_term(set, vec_push(build_ats(?match, 0, ?dx, ?dy), __spread(2)))],
    make_term(set, vec_push(build_ats(?result, 0, ?dx, ?dy), __spread(2))))
fn build_ats(vec(?cell, ?rest...), ?i, ?dx, ?dy) =
  vec_concat([mk_at(?cell, ?i, ?dx, ?dy)], build_ats(?rest, ?i + 1, ?dx, ?dy))
fn build_ats(vec(), ?i, ?dx, ?dy) = []
fn mk_at(m(?e), ?i, ?dx, ?dy) = mk_at(?e, ?i, ?dx, ?dy)
fn mk_at(?entity, ?i, ?dx, ?dy) =
  at(mk_coord(0, ?i * ?dx), mk_coord(1, ?i * ?dy), ?entity)
fn mk_coord(?var_id, 0) = __pvar(?var_id)
fn mk_coord(?var_id, ?offset) where ?offset > 0 = add(__pvar(?var_id), ?offset)
fn mk_coord(?var_id, ?offset) = sub(__pvar(?var_id), 0 - ?offset)
"#;
        let rules = r#"
fn won(#{target(?x, ?y), at(?x, ?y, crate), ?rest...}) = won(?rest...)
fn won(#{target(?x, ?y), ?rest...}) = false
fn won(?world) = true
fn level(1) = #{
  at(0,0,wall), at(1,0,wall), at(2,0,wall), at(3,0,wall), at(4,0,wall), at(5,0,wall), at(6,0,wall),
  at(0,1,wall), at(1,1,floor), at(2,1,floor), at(3,1,wall), at(4,1,floor), at(5,1,floor), at(6,1,wall),
  at(0,2,wall), at(1,2,floor), at(2,2,floor), at(3,2,crate), at(4,2,floor), at(5,2,floor), at(6,2,wall),
  at(0,3,wall), at(1,3,floor), at(2,3,floor), at(3,3,player), at(4,3,floor), at(5,3,floor), at(6,3,wall),
  at(0,4,wall), at(1,4,wall), at(2,4,wall), at(3,4,wall), at(4,4,wall), at(5,4,wall), at(6,4,wall),
  target(2, 2), size(7, 5)
}
fn handle_event(keydown, right) = do_move(right)
fn handle_event(keydown, left) = do_move(left)
fn handle_event(keydown, up) = do_move(up)
fn handle_event(keydown, down) = do_move(down)
fn handle_event(?a, ?b) = 0
fn do_move(?dir) =
  if screen == won then 0
  else {
    rule(world, step(?dir, world))
    rule(moves, moves + 1)
    check_win()
  }
fn check_win() = if won(world) then rule(screen, won) else 0
fn init() = {
  rule(current_level, 1)
  rule(moves, 0)
  rule(screen, playing)
  rule(num_levels, 1)
  rule(world, level(1))
}

{
  compile(ps(
    [m(player) | crate | floor],
    [floor | m(player) | m(crate)]))
  compile(ps(
    [m(player) | floor],
    [floor | m(player)]))
  defclause(step, [__pvar(0), __pvar(1)], __pvar(1))
  init()
}
"#;
        let code = format!("{}\n{}\n{}", renderer, compiler, rules);
        let (mut engine, _result) = eval_program(&code);

        // Now simulate pressing right (like the browser does)
        let handle_event_sym = engine.store.sym("handle_event");
        let handle_event_head = engine.store.sym_term(handle_event_sym);
        let keydown_sym = engine.store.sym("keydown");
        let keydown_term = engine.store.sym_term(keydown_sym);
        let right_sym = engine.store.sym("right");
        let right_term = engine.store.sym_term(right_sym);
        let call = engine.store.call(handle_event_head, &[keydown_term, right_term]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        let _result = engine.eval(call);

        // Now simulate render()
        let render_sym = engine.store.sym("render");
        let render_head = engine.store.sym_term(render_sym);
        let render_call = engine.store.call(render_head, &[]);
        engine.invalidate_cache();
        engine.reset_eval_counters();
        let _render_result = engine.eval(render_call);
        // If we get here without panic, everything works
    }
}
