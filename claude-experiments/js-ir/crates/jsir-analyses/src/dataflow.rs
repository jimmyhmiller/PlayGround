//! A generic forward dataflow engine over the JSHIR.
//!
//! This is the reusable framework (mirroring upstream's
//! `ConditionalForwardPerVarDataflowAnalysis`): it owns all the structured
//! control-flow walking, the loop/branch fixpoints, conditional reachability,
//! `break`/`continue`/`switch`/`label` handling, and the per-SSA-value + per-
//! variable state plumbing. A concrete analysis only supplies a lattice
//! ([`Lattice`]) and per-op transfer functions ([`Analysis`]); constant
//! propagation is one instantiation, but any forward, per-variable analysis
//! (sign, nullness, type-of, taint, ...) plugs into the same engine.

use std::collections::HashMap;

use jsir_ir::{Attr, Op, Region, SymbolId, ValueId};

/// A lattice element for a single value/variable.
pub trait Lattice: Clone + PartialEq {
    /// Bottom: never assigned / unreachable.
    fn bottom() -> Self;
    /// Top: an unknown / over-defined value.
    fn top() -> Self;
    /// Join (least upper bound) in place; returns whether `self` changed.
    fn join(&mut self, other: &Self) -> bool;
    /// Render the element the way the analysis dump prints it.
    fn render(&self) -> String;
}

/// A concrete forward analysis: a lattice plus per-op transfer functions and a
/// few predicates the engine uses for branch pruning and short-circuiting.
pub trait Analysis {
    type V: Lattice;

    /// Transfer a single leaf (non-control-flow) op: read operands/variables and
    /// set this op's result value through `cx`. Control-flow ops (`if`, loops,
    /// `switch`, logical/conditional, `break`/`continue`, ...) are handled by the
    /// engine and never reach here.
    fn transfer(&self, op: &Op, cx: &mut Transfer<Self::V>);

    // Predicates over a condition's lattice value. Defaults leave every branch
    // reachable (correct for analyses that don't track truthiness).
    fn is_true(&self, _v: &Self::V) -> bool {
        false
    }
    fn is_false(&self, _v: &Self::V) -> bool {
        false
    }
    fn is_nullish(&self, _v: &Self::V) -> bool {
        false
    }
    fn is_nonnullish(&self, _v: &Self::V) -> bool {
        false
    }
}

/// Per-variable state at a program point: `name#scope` -> lattice value.
type State<V> = HashMap<String, V>;

/// The context handed to [`Analysis::transfer`] for one op.
pub struct Transfer<'a, V> {
    pub op: &'a Op,
    values: &'a HashMap<ValueId, V>,
    value_symbol: &'a HashMap<ValueId, String>,
    state: &'a mut State<V>,
    result: Option<V>,
}

impl<'a, V: Lattice> Transfer<'a, V> {
    /// The lattice value of the op's `i`-th operand (bottom if undefined).
    pub fn operand(&self, i: usize) -> V {
        self.op
            .operands
            .get(i)
            .and_then(|v| self.values.get(v))
            .cloned()
            .unwrap_or_else(V::bottom)
    }
    pub fn operand_count(&self) -> usize {
        self.op.operands.len()
    }
    /// The variable named by the op's `i`-th operand's defining value (used by
    /// declarators/assignments whose first operand is an l-value reference).
    pub fn operand_symbol(&self, i: usize) -> Option<String> {
        self.op.operands.get(i).and_then(|v| self.value_symbol.get(v)).cloned()
    }
    /// The variable this op itself names (an identifier's resolved symbol).
    pub fn symbol(&self) -> Option<String> {
        symbol_of(self.op)
    }
    /// The current value of a variable (top if unknown / unset).
    pub fn var(&self, key: &str) -> V {
        self.state.get(key).cloned().unwrap_or_else(V::top)
    }
    pub fn set_var(&mut self, key: String, v: V) {
        self.state.insert(key, v);
    }
    /// This op's named attribute.
    pub fn attr(&self, key: &str) -> Option<&'a Attr> {
        attr(self.op, key)
    }
    /// Set this op's result value.
    pub fn set_result(&mut self, v: V) {
        self.result = Some(v);
    }
}

/// Read an op's named attribute.
pub fn attr<'a>(op: &'a Op, key: &str) -> Option<&'a Attr> {
    op.attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn sym_key(s: &SymbolId) -> String {
    match s.def_scope_uid {
        Some(scope) => format!("{}#{}", s.name, scope),
        None => format!("{}#undeclared", s.name),
    }
}

/// The variable an identifier op names: its resolved `referencedSymbol`, a
/// declaration's first `definedSymbol`, else an undeclared global by name.
pub fn symbol_of(op: &Op) -> Option<String> {
    if let Some(t) = op.trivia.as_ref() {
        if let Some(rs) = &t.referenced_symbol {
            return Some(sym_key(rs));
        }
        if let Some(ds) = t.defined_symbols.as_ref().and_then(|d| d.first()) {
            return Some(sym_key(ds));
        }
    }
    match attr(op, "name") {
        Some(Attr::Str(name)) => Some(format!("{name}#undeclared")),
        _ => None,
    }
}

fn label_name(op: &Op) -> Option<String> {
    match attr(op, "label") {
        Some(Attr::Identifier(id)) => Some(id.name.clone()),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Normal,
    Break,
    Continue,
}

/// Merge a state into an optional accumulator (join, or initialize).
fn merge_into<V: Lattice>(acc: &mut Option<State<V>>, s: &State<V>) {
    match acc {
        Some(a) => {
            join_states(a, s);
        }
        None => *acc = Some(s.clone()),
    }
}

/// Join two per-variable states; returns whether `a` changed.
fn join_states<V: Lattice>(a: &mut State<V>, b: &State<V>) -> bool {
    let mut changed = false;
    for (k, bv) in b {
        match a.get_mut(k) {
            Some(av) => changed |= av.join(bv),
            None => {
                a.insert(k.clone(), bv.clone());
                changed = true;
            }
        }
    }
    changed
}

/// Max fixpoint iterations per loop before we give up (a non-convergence guard).
const ITER_CAP: u32 = 64;

/// The result of running an analysis: per-value lattice + ordered value facts.
pub struct Output<V> {
    pub values: HashMap<ValueId, V>,
    pub facts: Vec<String>,
    /// `true` iff every loop reached a fixpoint within [`ITER_CAP`] iterations
    /// (i.e. the result is an actual fixpoint, not a truncated approximation).
    pub converged: bool,
}

/// Run a forward analysis over a whole `jsir.file` op.
pub fn run<A: Analysis>(analysis: &A, file: &Op) -> Output<A::V> {
    let mut eng = Engine {
        analysis,
        values: HashMap::new(),
        value_symbol: HashMap::new(),
        flow: Flow::Normal,
        break_acc: None,
        continue_acc: None,
        break_label: None,
        pending_label: None,
        hit_cap: false,
    };
    let mut state: State<A::V> = HashMap::new();
    eng.seed_hoisted(file, false, &mut state);
    eng.process(file, &mut state, true);
    let mut facts = Vec::new();
    emit_facts(&eng.values, file, &mut facts);
    Output { values: eng.values, facts, converged: !eng.hit_cap }
}

/// Document-order pass emitting each value-producing op's fixpoint value.
fn emit_facts<V: Lattice>(values: &HashMap<ValueId, V>, op: &Op, facts: &mut Vec<String>) {
    for region in &op.regions {
        for block in &region.blocks {
            for inner in &block.ops {
                emit_facts(values, inner, facts);
            }
        }
    }
    if let Some(r) = op.results.first() {
        facts.push(values.get(r).cloned().unwrap_or_else(V::bottom).render());
    }
}

/// Verify the algebraic laws a [`Lattice`] must satisfy for the dataflow
/// framework to be sound and terminating. `samples` should be a representative
/// set of elements (e.g. every distinct value an analysis produced). Returns the
/// violated laws (empty = all hold); bottom and top are always included.
pub fn check_lattice_laws<V: Lattice>(samples: &[V]) -> Vec<String> {
    let join = |a: &V, b: &V| {
        let mut x = a.clone();
        x.join(b);
        x
    };
    let mut elems: Vec<V> = samples.to_vec();
    elems.push(V::bottom());
    elems.push(V::top());
    let mut deduped: Vec<V> = Vec::new();
    for e in elems {
        if !deduped.contains(&e) {
            deduped.push(e);
        }
    }
    let elems = deduped;

    let mut errs = Vec::new();
    let bottom = V::bottom();
    let top = V::top();
    for x in &elems {
        // Idempotency: x ⊔ x == x.
        if join(x, x) != *x {
            errs.push(format!("idempotency violated at {}", x.render()));
        }
        // Bottom is the identity: ⊥ ⊔ x == x.
        if join(&bottom, x) != *x {
            errs.push(format!("bottom-identity violated at {}", x.render()));
        }
        // Top is absorbing: ⊤ ⊔ x == ⊤.
        if join(&top, x) != top {
            errs.push(format!("top-absorbing violated at {}", x.render()));
        }
        // The `join` change flag must report exactly whether the value moved.
        {
            let mut a = x.clone();
            let changed = a.join(&top);
            if changed != (a != *x) {
                errs.push(format!("join change-flag wrong joining {} with top", x.render()));
            }
        }
        for y in &elems {
            // Commutativity: x ⊔ y == y ⊔ x.
            if join(x, y) != join(y, x) {
                errs.push(format!("commutativity violated at ({}, {})", x.render(), y.render()));
            }
            // Extensivity: x ⊑ x ⊔ y, i.e. (x⊔y) ⊔ x == x⊔y.
            let xy = join(x, y);
            if join(&xy, x) != xy {
                errs.push(format!("extensivity violated for {} vs {}", x.render(), y.render()));
            }
            for z in &elems {
                // Associativity.
                if join(&join(x, y), z) != join(x, &join(y, z)) {
                    errs.push(format!(
                        "associativity violated at ({}, {}, {})",
                        x.render(),
                        y.render(),
                        z.render()
                    ));
                }
            }
        }
    }
    errs.sort();
    errs.dedup();
    errs
}

struct Engine<'a, A: Analysis> {
    analysis: &'a A,
    values: HashMap<ValueId, A::V>,
    value_symbol: HashMap<ValueId, String>,
    flow: Flow,
    break_acc: Option<State<A::V>>,
    continue_acc: Option<State<A::V>>,
    break_label: Option<String>,
    pending_label: Option<String>,
    /// Set if any loop exhausted [`ITER_CAP`] without converging.
    hit_cap: bool,
}

impl<'a, A: Analysis> Engine<'a, A> {
    /// Seed every `var` binding's symbol to top (its hoisted `undefined`): a
    /// variable assigned only inside a loop/branch then joins to top afterward.
    fn seed_hoisted(&mut self, op: &Op, in_var: bool, state: &mut State<A::V>) {
        let is_var = op.name == "jsir.variable_declaration"
            && matches!(attr(op, "kind"), Some(Attr::Str(k)) if k == "var");
        if in_var && op.name == "jsir.identifier_ref" {
            if let Some(k) = symbol_of(op) {
                state.entry(k).or_insert_with(A::V::top);
            }
        }
        for region in &op.regions {
            for block in &region.blocks {
                for inner in &block.ops {
                    self.seed_hoisted(inner, in_var || is_var, state);
                }
            }
        }
    }

    fn process(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        match op.name.as_str() {
            "jshir.if_statement" => self.process_if(op, state, reachable),
            "jshir.while_statement" => self.process_while(op, state, reachable),
            "jshir.do_while_statement" => self.process_do_while(op, state, reachable),
            "jshir.for_statement" => self.process_for(op, state, reachable),
            "jshir.for_in_statement" | "jshir.for_of_statement" => {
                self.process_loop_body(op.regions.first(), state, reachable);
            }
            "jshir.try_statement" => self.process_try(op, state, reachable),
            "jshir.switch_statement" => self.process_switch(op, state, reachable),
            "jshir.labeled_statement" => {
                let saved = self.pending_label.take();
                self.pending_label = label_name(op);
                for region in &op.regions {
                    self.process_region(region, state, reachable);
                }
                self.pending_label = saved;
            }
            "jshir.break_statement" => {
                if reachable {
                    self.flow = Flow::Break;
                    self.break_label = label_name(op);
                    let s = state.clone();
                    merge_into(&mut self.break_acc, &s);
                }
            }
            "jshir.continue_statement" => {
                if reachable {
                    self.flow = Flow::Continue;
                    self.break_label = label_name(op);
                    let s = state.clone();
                    merge_into(&mut self.continue_acc, &s);
                }
            }
            "jshir.logical_expression" => self.process_logical(op, state, reachable),
            "jshir.conditional_expression" => self.process_conditional(op, state, reachable),
            _ if !op.regions.is_empty() && op.results.is_empty() => {
                // A sequential region container (program, block_statement, ...).
                for region in &op.regions {
                    self.process_region(region, state, reachable);
                }
            }
            _ => self.leaf(op, state, reachable),
        }
    }

    /// Apply the analysis to a leaf op, committing its result + symbol plumbing.
    fn leaf(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        // Remember which variable an identifier names (for l-value writes).
        if let (Some(r), Some(k)) = (op.results.first(), symbol_of(op)) {
            self.value_symbol.insert(*r, k);
        }
        if !reachable {
            // Dead code: every produced value is bottom; no state change.
            if let Some(r) = op.results.first() {
                self.values.insert(*r, A::V::bottom());
            }
            return;
        }
        let mut cx = Transfer {
            op,
            values: &self.values,
            value_symbol: &self.value_symbol,
            state,
            result: None,
        };
        self.analysis.transfer(op, &mut cx);
        // If the analysis didn't set a result, the op's value is over-defined.
        let v = cx.result.unwrap_or_else(A::V::top);
        if let Some(r) = op.results.first() {
            self.values.insert(*r, v);
        }
    }

    fn process_region(&mut self, region: &Region, state: &mut State<A::V>, reachable: bool) {
        for block in &region.blocks {
            for op in &block.ops {
                self.process(op, state, reachable);
            }
        }
    }

    fn process_if(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        let cond = self.operand(op, 0);
        let cons_reach = reachable && !self.analysis.is_false(&cond);
        let alt_reach = reachable && !self.analysis.is_true(&cond);
        let mut cons_state = state.clone();
        if let Some(r) = op.regions.first() {
            self.process_region(r, &mut cons_state, cons_reach);
        }
        let mut alt_state = state.clone();
        if let Some(r) = op.regions.get(1) {
            self.process_region(r, &mut alt_state, alt_reach);
        }
        match (cons_reach, alt_reach) {
            (true, true) => {
                let mut merged = cons_state;
                join_states(&mut merged, &alt_state);
                *state = merged;
            }
            (true, false) => *state = cons_state,
            (false, true) => *state = alt_state,
            (false, false) => {}
        }
    }

    fn process_while(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        let mut header = state.clone();
        let mut break_exit: Option<State<A::V>> = None;
        let my_label = self.pending_label.take();
        let mut test_always_true = false;
        let mut it = 0u32;
        for _ in 0..ITER_CAP {
            it += 1;
            let test_val = self.eval_test(op.regions.first(), &header, reachable);
            test_always_true = self.analysis.is_true(&test_val);
            let body_reachable = reachable && !self.analysis.is_false(&test_val);
            match self.run_body(op.regions.get(1), &header, body_reachable, &mut break_exit, my_label.as_deref()) {
                None => break,
                Some(lb) => {
                    if !join_states(&mut header, &lb) {
                        break;
                    }
                }
            }
        }
        if it >= ITER_CAP {
            self.hit_cap = true;
        }
        if test_always_true {
            if let Some(b) = break_exit {
                *state = b;
            }
        } else {
            if let Some(b) = break_exit {
                join_states(&mut header, &b);
            }
            *state = header;
        }
    }

    fn process_do_while(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        let mut header = state.clone();
        let mut break_exit: Option<State<A::V>> = None;
        let my_label = self.pending_label.take();
        let mut it = 0u32;
        for i in 0..ITER_CAP {
            it += 1;
            let lb = self.run_body(op.regions.first(), &header, reachable, &mut break_exit, my_label.as_deref());
            let Some(mut work) = lb else { break };
            if let Some(test) = op.regions.get(1) {
                self.process_region(test, &mut work, reachable);
            }
            let changed = join_states(&mut header, &work);
            if i > 0 && !changed {
                break;
            }
        }
        if it >= ITER_CAP {
            self.hit_cap = true;
        }
        if let Some(b) = break_exit {
            join_states(&mut header, &b);
        }
        *state = header;
    }

    fn process_for(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        if let Some(init) = op.regions.first() {
            self.process_region(init, state, reachable);
        }
        let mut header = state.clone();
        let mut break_exit: Option<State<A::V>> = None;
        let my_label = self.pending_label.take();
        let mut it = 0u32;
        for _ in 0..ITER_CAP {
            it += 1;
            let test_val = self.eval_test(op.regions.get(1), &header, reachable);
            let body_reachable = reachable && !self.analysis.is_false(&test_val);
            let lb = self.run_body(op.regions.get(3), &header, body_reachable, &mut break_exit, my_label.as_deref());
            let Some(mut work) = lb else { break };
            if let Some(update) = op.regions.get(2) {
                self.process_region(update, &mut work, body_reachable);
            }
            if !join_states(&mut header, &work) {
                break;
            }
        }
        if it >= ITER_CAP {
            self.hit_cap = true;
        }
        if let Some(b) = break_exit {
            join_states(&mut header, &b);
        }
        *state = header;
    }

    fn process_loop_body(&mut self, body: Option<&Region>, state: &mut State<A::V>, reachable: bool) {
        let mut header = state.clone();
        let mut break_exit: Option<State<A::V>> = None;
        let my_label = self.pending_label.take();
        let mut it = 0u32;
        for _ in 0..ITER_CAP {
            it += 1;
            match self.run_body(body, &header, reachable, &mut break_exit, my_label.as_deref()) {
                None => break,
                Some(lb) => {
                    if !join_states(&mut header, &lb) {
                        break;
                    }
                }
            }
        }
        if it >= ITER_CAP {
            self.hit_cap = true;
        }
        if let Some(b) = break_exit {
            join_states(&mut header, &b);
        }
        *state = header;
    }

    /// Run one loop-body iteration, capturing `break` exits and propagating any
    /// labeled break/continue not aimed at this loop.
    fn run_body(
        &mut self,
        body: Option<&Region>,
        header: &State<A::V>,
        reachable: bool,
        break_exit: &mut Option<State<A::V>>,
        my_label: Option<&str>,
    ) -> Option<State<A::V>> {
        let saved_break = self.break_acc.take();
        let saved_cont = self.continue_acc.take();
        let saved_flow = self.flow;
        self.flow = Flow::Normal;
        let mut work = header.clone();
        if let Some(b) = body {
            self.process_region(b, &mut work, reachable);
        }
        let f = self.flow;
        let target = self.break_label.clone();
        let bacc = self.break_acc.take();
        let cacc = self.continue_acc.take();
        self.break_acc = saved_break;
        self.continue_acc = saved_cont;
        let for_me = target.is_none() || target.as_deref() == my_label;
        match f {
            Flow::Break if for_me => {
                self.flow = saved_flow;
                self.break_label = None;
                if let Some(b) = &bacc {
                    merge_into(break_exit, b);
                }
                None
            }
            Flow::Break => {
                if let Some(b) = &bacc {
                    merge_into(&mut self.break_acc, b);
                }
                None
            }
            Flow::Continue if for_me => {
                self.flow = saved_flow;
                self.break_label = None;
                Some(cacc.unwrap_or(work))
            }
            Flow::Continue => {
                if let Some(c) = &cacc {
                    merge_into(&mut self.continue_acc, c);
                }
                None
            }
            Flow::Normal => {
                self.flow = saved_flow;
                Some(work)
            }
        }
    }

    fn process_try(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        // [block, catch, finally]. Upstream doesn't model exception flow, so the
        // catch is unreachable and the finally runs on the block's exit.
        let mut try_state = state.clone();
        if let Some(block) = op.regions.first() {
            self.process_region(block, &mut try_state, reachable);
        }
        if let Some(handler) = op.regions.get(1) {
            let mut dead = try_state.clone();
            self.process_region(handler, &mut dead, false);
        }
        if let Some(finalizer) = op.regions.get(2) {
            self.process_region(finalizer, &mut try_state, reachable);
        }
        *state = try_state;
    }

    fn process_switch(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        let disc = self.operand(op, 0);
        let cases: Vec<&Op> = op
            .regions
            .first()
            .into_iter()
            .flat_map(|r| r.blocks.iter())
            .flat_map(|b| b.ops.iter())
            .filter(|o| o.name == "jshir.switch_case")
            .collect();

        let mut test_vals: Vec<Option<A::V>> = Vec::new();
        for case in &cases {
            let test_region = case.regions.first();
            if let Some(t) = test_region {
                let mut s = state.clone();
                self.process_region(t, &mut s, reachable);
            }
            let has_test = test_region
                .map(|t| t.blocks.iter().any(|b| b.ops.iter().any(|o| o.name == "jsir.expr_region_end")))
                .unwrap_or(false);
            test_vals.push(if has_test { test_region.map(|t| self.expr_region_value(t)) } else { None });
        }

        let disc_unknown = disc == A::V::top() || disc == A::V::bottom();
        let mut started = disc_unknown;
        let matched_any = !disc_unknown && test_vals.iter().any(|t| matches!(t, Some(v) if *v == disc));
        let mut sw_state = state.clone();
        let mut break_exit: Option<State<A::V>> = None;
        let _ = self.pending_label.take();
        let mut exited = false;
        for (i, case) in cases.iter().enumerate() {
            let is_match = match &test_vals[i] {
                Some(v) => !disc_unknown && *v == disc,
                None => !disc_unknown && !matched_any,
            };
            if is_match {
                started = true;
            }
            let body_reachable = reachable && started && !exited;
            let saved_flow = self.flow;
            let saved_break = self.break_acc.take();
            self.flow = Flow::Normal;
            if let Some(body) = case.regions.get(1) {
                self.process_region(body, &mut sw_state, body_reachable);
            }
            if self.flow == Flow::Break {
                exited = true;
                if let Some(b) = self.break_acc.take() {
                    merge_into(&mut break_exit, &b);
                }
            }
            self.break_acc = saved_break;
            self.flow = saved_flow;
        }
        if !exited {
            merge_into(&mut break_exit, &sw_state);
        }
        if let Some(b) = break_exit {
            *state = b;
        }
    }

    /// `a && b` / `a || b` / `a ?? b`: the right operand lives in a region and is
    /// evaluated conditionally; the result short-circuits on the left.
    fn process_logical(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        let left = self.operand(op, 0);
        let operator = match attr(op, "operator_") {
            Some(Attr::Str(s)) => s.as_str(),
            _ => "",
        };
        let take_right = match operator {
            "&&" => !self.analysis.is_false(&left),
            "||" => !self.analysis.is_true(&left),
            "??" => !self.analysis.is_nonnullish(&left),
            _ => true,
        };
        let mut s = state.clone();
        if let Some(r) = op.regions.first() {
            self.process_region(r, &mut s, reachable && take_right);
        }
        if take_right {
            *state = s;
        }
        let right = op.regions.first().map(|r| self.expr_region_value(r)).unwrap_or_else(A::V::top);
        let val = if left == A::V::bottom() {
            A::V::bottom()
        } else {
            match operator {
                "&&" if self.analysis.is_false(&left) => left.clone(),
                "&&" if self.analysis.is_true(&left) => right,
                "||" if self.analysis.is_true(&left) => left.clone(),
                "||" if self.analysis.is_false(&left) => right,
                "??" if self.analysis.is_nullish(&left) => right,
                "??" if self.analysis.is_nonnullish(&left) => left.clone(),
                _ => {
                    let mut v = left.clone();
                    v.join(&right);
                    v
                }
            }
        };
        self.write_result(op, val);
    }

    fn process_conditional(&mut self, op: &Op, state: &mut State<A::V>, reachable: bool) {
        // Regions are [alternate, consequent].
        let test = self.operand(op, 0);
        let cons_reach = reachable && !self.analysis.is_false(&test);
        let alt_reach = reachable && !self.analysis.is_true(&test);
        let mut alt_state = state.clone();
        if let Some(r) = op.regions.first() {
            self.process_region(r, &mut alt_state, alt_reach);
        }
        let mut cons_state = state.clone();
        if let Some(r) = op.regions.get(1) {
            self.process_region(r, &mut cons_state, cons_reach);
        }
        let alt_val = op.regions.first().map(|r| self.expr_region_value(r)).unwrap_or_else(A::V::top);
        let cons_val = op.regions.get(1).map(|r| self.expr_region_value(r)).unwrap_or_else(A::V::top);
        let (val, new_state) = match (cons_reach, alt_reach) {
            (true, false) => (cons_val, cons_state),
            (false, true) => (alt_val, alt_state),
            (true, true) => {
                let mut merged = cons_state;
                join_states(&mut merged, &alt_state);
                let mut v = cons_val;
                v.join(&alt_val);
                (v, merged)
            }
            (false, false) => (A::V::bottom(), state.clone()),
        };
        *state = new_state;
        self.write_result(op, val);
    }

    /// Process a test ExprRegion against `header` and return its value.
    fn eval_test(&mut self, test: Option<&Region>, header: &State<A::V>, reachable: bool) -> A::V {
        match test {
            Some(t) => {
                let mut s = header.clone();
                self.process_region(t, &mut s, reachable);
                self.expr_region_value(t)
            }
            None => A::V::top(),
        }
    }

    /// The value an ExprRegion produces (the operand of its `expr_region_end`).
    fn expr_region_value(&self, region: &Region) -> A::V {
        for block in &region.blocks {
            for op in &block.ops {
                if op.name == "jsir.expr_region_end" {
                    return op
                        .operands
                        .first()
                        .and_then(|v| self.values.get(v))
                        .cloned()
                        .unwrap_or_else(A::V::top);
                }
            }
        }
        A::V::top()
    }

    fn operand(&self, op: &Op, i: usize) -> A::V {
        op.operands
            .get(i)
            .and_then(|v| self.values.get(v))
            .cloned()
            .unwrap_or_else(A::V::bottom)
    }

    fn write_result(&mut self, op: &Op, val: A::V) {
        if let Some(r) = op.results.first() {
            self.values.insert(*r, val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{check_lattice_laws, Lattice};

    /// A deliberately broken lattice: its join is non-commutative and
    /// non-idempotent, so the checker must flag it (proves the oracle isn't
    /// vacuous).
    #[derive(Clone, PartialEq)]
    enum Broken {
        Bottom,
        A,
        B,
        Top,
    }
    impl Lattice for Broken {
        fn bottom() -> Self {
            Broken::Bottom
        }
        fn top() -> Self {
            Broken::Top
        }
        fn join(&mut self, other: &Broken) -> bool {
            // Wrong on purpose: "first operand wins" is neither commutative nor
            // a real least-upper-bound.
            let before = self.clone();
            if matches!(self, Broken::Bottom) {
                *self = other.clone();
            }
            *self != before
        }
        fn render(&self) -> String {
            match self {
                Broken::Bottom => "_",
                Broken::A => "A",
                Broken::B => "B",
                Broken::Top => "T",
            }
            .to_string()
        }
    }

    #[test]
    fn law_checker_flags_a_broken_lattice() {
        let errs = check_lattice_laws(&[Broken::A, Broken::B]);
        assert!(!errs.is_empty(), "the checker should reject a non-lattice join");
        // It specifically catches that `A ⊔ B != B ⊔ A` and top isn't absorbing.
        assert!(
            errs.iter().any(|e| e.contains("commutativity") || e.contains("top-absorbing")),
            "expected a commutativity/absorbing violation, got {errs:?}"
        );
    }
}
