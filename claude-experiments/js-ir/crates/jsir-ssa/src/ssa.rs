//! SSA construction via Braun et al., *Simple and Efficient Construction of
//! Static Single Assignment Form* (CC 2013) — the same algorithm the React
//! Compiler's `EnterSSA` uses, and close in spirit to MLIR's `mem2reg`.
//!
//! Phi nodes are materialized as **block arguments** (MLIR's model): a surviving
//! phi becomes a parameter of its block, and every predecessor's terminator is
//! given the corresponding operand. `ReadVar`/`WriteVar` slots are removed.
//!
//! Trivial-phi removal (`try_remove_trivial_phi`) gives minimal SSA, so a
//! separate redundant-phi pass is unnecessary.

use std::collections::{HashMap, HashSet};

use crate::cfg::{BlockId, Cfg, Const, Op, Term, VarId, Value};

type PhiId = usize;

/// A value reference during construction: a concrete value or a (maybe later
/// removed) phi.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ref {
    Val(Value),
    Phi(PhiId),
}

struct Phi {
    block: BlockId,
    #[allow(dead_code)]
    var: VarId,
    param: Value,
    operands: Vec<(BlockId, Ref)>,
    users: Vec<PhiId>,
    /// `Some(r)` once removed as trivial: resolves to `r`.
    replaced_by: Option<Ref>,
}

struct Builder<'a> {
    cfg: &'a mut Cfg,
    preds: HashMap<BlockId, Vec<BlockId>>,
    current_def: HashMap<(VarId, BlockId), Ref>,
    phis: Vec<Phi>,
    sealed: HashSet<BlockId>,
    incomplete: HashMap<BlockId, Vec<(VarId, PhiId)>>,
    filled: HashSet<BlockId>,
    /// ReadVar result value -> the SSA ref it resolves to.
    read_subst: HashMap<Value, Ref>,
    undef: Option<Value>,
}

/// Construct SSA form in place: promote `ReadVar`/`WriteVar` variables to SSA
/// values + block arguments.
pub fn construct(cfg: &mut Cfg) {
    let preds = cfg.predecessors();
    let order = reverse_postorder(cfg);
    let mut b = Builder {
        cfg,
        preds,
        current_def: HashMap::new(),
        phis: Vec::new(),
        sealed: HashSet::new(),
        incomplete: HashMap::new(),
        filled: HashSet::new(),
        read_subst: HashMap::new(),
        undef: None,
    };

    for &block in &order {
        // A block can be sealed before filling if all preds are already filled.
        b.try_seal(block);
        b.fill(block);
        // Filling may complete some successor's predecessor set.
        let succs = b.cfg.block(block).term.successors();
        for s in succs {
            b.try_seal(s);
        }
    }
    // Seal anything still open (e.g. unreachable cycles).
    for &block in &order {
        b.try_seal(block);
    }

    b.materialize(&order);
}

impl<'a> Builder<'a> {
    fn all_preds_filled(&self, block: BlockId) -> bool {
        self.preds.get(&block).map(|ps| ps.iter().all(|p| self.filled.contains(p))).unwrap_or(true)
    }

    fn try_seal(&mut self, block: BlockId) {
        if self.sealed.contains(&block) || !self.all_preds_filled(block) {
            return;
        }
        let incompletes = self.incomplete.remove(&block).unwrap_or_default();
        self.sealed.insert(block);
        for (var, phi) in incompletes {
            self.add_phi_operands(var, phi);
        }
    }

    /// Process a block's `ReadVar`/`WriteVar` ops, recording defs and resolving
    /// reads. The ops themselves are removed later in `materialize`.
    fn fill(&mut self, block: BlockId) {
        // Snapshot the relevant ops together with each `ReadVar`'s result `Value`
        // up front. Capturing the result here (rather than re-reading the block
        // by index in the loop below) keeps `fill` immune to any mutation of the
        // block during processing — e.g. `undef_value()` inserting an undef
        // instruction at index 0 of the entry block, which would otherwise shift
        // every subsequent index and make a stale-index re-read read the wrong
        // instruction.
        let instrs: Vec<(Option<Value>, Op)> = self
            .cfg
            .block(block)
            .instrs
            .iter()
            .filter_map(|ins| match &ins.op {
                Op::ReadVar(_) | Op::WriteVar(_, _) => Some((ins.result, ins.op.clone())),
                _ => None,
            })
            .collect();
        for (result, op) in instrs {
            match op {
                Op::WriteVar(var, val) => {
                    // The stored value may itself be a `ReadVar` result that is
                    // being substituted away; record its resolved ref so reads of
                    // `var` see the real SSA value, not a deleted one.
                    let r = self.resolve_value(val);
                    self.write_variable(var, block, r);
                }
                Op::ReadVar(var) => {
                    let r = self.read_variable(var, block);
                    let result = result.expect("ReadVar has result");
                    self.read_subst.insert(result, r);
                }
                _ => unreachable!(),
            }
        }
        self.filled.insert(block);
    }

    fn write_variable(&mut self, var: VarId, block: BlockId, r: Ref) {
        self.current_def.insert((var, block), r);
    }

    /// Map a (possibly `ReadVar`-result) value to the SSA ref it stands for.
    fn resolve_value(&self, v: Value) -> Ref {
        self.read_subst.get(&v).copied().unwrap_or(Ref::Val(v))
    }

    fn read_variable(&mut self, var: VarId, block: BlockId) -> Ref {
        if let Some(r) = self.current_def.get(&(var, block)) {
            return *r;
        }
        self.read_variable_recursive(var, block)
    }

    fn read_variable_recursive(&mut self, var: VarId, block: BlockId) -> Ref {
        let r = if !self.sealed.contains(&block) {
            // Block not sealed: place an operandless phi as a placeholder.
            let phi = self.new_phi(var, block);
            self.incomplete.entry(block).or_default().push((var, phi));
            Ref::Phi(phi)
        } else {
            let preds = self.preds.get(&block).cloned().unwrap_or_default();
            if preds.is_empty() {
                // Entry / unreachable: undefined (matches reading an unset `var`).
                Ref::Val(self.undef_value())
            } else if preds.len() == 1 {
                self.read_variable(var, preds[0])
            } else {
                // Break potential cycles: define the phi before filling operands.
                let phi = self.new_phi(var, block);
                self.write_variable(var, block, Ref::Phi(phi));
                self.add_phi_operands(var, phi)
            }
        };
        self.write_variable(var, block, r);
        r
    }

    fn new_phi(&mut self, var: VarId, block: BlockId) -> PhiId {
        let param = self.cfg.fresh_value();
        let id = self.phis.len();
        self.phis.push(Phi { block, var, param, operands: Vec::new(), users: Vec::new(), replaced_by: None });
        id
    }

    fn add_phi_operands(&mut self, var: VarId, phi: PhiId) -> Ref {
        let block = self.phis[phi].block;
        let preds = self.preds.get(&block).cloned().unwrap_or_default();
        for pred in preds {
            let op = self.read_variable(var, pred);
            self.phis[phi].operands.push((pred, op));
            if let Ref::Phi(p) = op {
                if p != phi {
                    self.phis[p].users.push(phi);
                }
            }
        }
        self.try_remove_trivial_phi(phi)
    }

    fn try_remove_trivial_phi(&mut self, phi: PhiId) -> Ref {
        let mut same: Option<Ref> = None;
        for &(_, op) in &self.phis[phi].operands.clone() {
            let op = self.resolve(op);
            if op == Ref::Phi(phi) || Some(op) == same {
                continue;
            }
            if same.is_some() {
                return Ref::Phi(phi); // non-trivial: two distinct operands
            }
            same = Some(op);
        }
        // Unreachable phi (no operands) -> undefined.
        let replacement = same.unwrap_or_else(|| Ref::Val(self.undef_value()));
        self.phis[phi].replaced_by = Some(replacement);

        // Re-route users that became trivial.
        let users: Vec<PhiId> = self.phis[phi].users.clone();
        for u in users {
            if u != phi && self.phis[u].replaced_by.is_none() {
                self.try_remove_trivial_phi(u);
            }
        }
        replacement
    }

    /// Follow removed-phi chains to a stable ref.
    fn resolve(&self, mut r: Ref) -> Ref {
        loop {
            match r {
                Ref::Phi(p) => match self.phis[p].replaced_by {
                    Some(next) if next != r => r = next,
                    _ => return r,
                },
                Ref::Val(_) => return r,
            }
        }
    }

    /// The final concrete `Value` a ref resolves to.
    fn final_value(&self, r: Ref) -> Value {
        match self.resolve(r) {
            Ref::Val(v) => v,
            Ref::Phi(p) => self.phis[p].param,
        }
    }

    fn undef_value(&mut self) -> Value {
        if let Some(v) = self.undef {
            return v;
        }
        let entry = self.cfg.entry;
        let v = self.cfg.fresh_value();
        self.cfg.block_mut(entry).instrs.insert(0, crate::cfg::Instr {
            result: Some(v),
            op: Op::Const(Const::Undef),
            src: None, // synthetic: no JSIR statement of origin
        });
        self.undef = Some(v);
        v
    }

    /// Materialize: drop var ops, install surviving phis as block params, and
    /// give every predecessor terminator the matching operands.
    fn materialize(&mut self, order: &[BlockId]) {
        // Surviving phis grouped by block, in stable creation order.
        let mut phis_by_block: HashMap<BlockId, Vec<PhiId>> = HashMap::new();
        for (id, phi) in self.phis.iter().enumerate() {
            if phi.replaced_by.is_none() {
                phis_by_block.entry(phi.block).or_default().push(id);
            }
        }

        // Install block params.
        for (&block, ids) in &phis_by_block {
            let params: Vec<Value> = ids.iter().map(|&p| self.phis[p].param).collect();
            self.cfg.block_mut(block).params = params;
        }

        // Set terminator operands for each edge into a phi-bearing block.
        for &pred in order {
            let succs = self.cfg.block(pred).term.successors();
            // Build per-successor arg vectors.
            let mut succ_args: Vec<(BlockId, Vec<Value>)> = Vec::new();
            for s in &succs {
                let args = phis_by_block
                    .get(s)
                    .map(|ids| {
                        ids.iter()
                            .map(|&p| {
                                let opref = self.phis[p]
                                    .operands
                                    .iter()
                                    .find(|(pb, _)| *pb == pred)
                                    .map(|(_, r)| *r)
                                    .unwrap_or(Ref::Val(self.undef.unwrap_or(Value(0))));
                                self.final_value(opref)
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                succ_args.push((*s, args));
            }
            self.apply_term_args(pred, &succ_args);
        }

        // Drop ReadVar/WriteVar ops and rewrite remaining operands through subst.
        let read_subst = std::mem::take(&mut self.read_subst);
        let phis = &self.phis;
        let subst = |v: Value| -> Value {
            match read_subst.get(&v) {
                Some(r) => match resolve_ref(phis, *r) {
                    Ref::Val(v) => v,
                    Ref::Phi(p) => phis[p].param,
                },
                None => v,
            }
        };
        for block in &mut self.cfg.blocks {
            block.instrs.retain(|ins| !matches!(ins.op, Op::ReadVar(_) | Op::WriteVar(_, _)));
            for ins in &mut block.instrs {
                rewrite_operands(&mut ins.op, &subst);
            }
            rewrite_term(&mut block.term, &subst);
        }
    }

    fn apply_term_args(&mut self, block: BlockId, succ_args: &[(BlockId, Vec<Value>)]) {
        let arg_for = |b: BlockId| succ_args.iter().find(|(s, _)| *s == b).map(|(_, a)| a.clone()).unwrap_or_default();
        match &mut self.cfg.block_mut(block).term {
            Term::Br(t, args) => *args = arg_for(*t),
            Term::CondBr { then_block, then_args, else_block, else_args, .. } => {
                *then_args = arg_for(*then_block);
                *else_args = arg_for(*else_block);
            }
            Term::Ret(_) | Term::Unreachable => {}
        }
    }
}

fn resolve_ref(phis: &[Phi], mut r: Ref) -> Ref {
    loop {
        match r {
            Ref::Phi(p) => match phis[p].replaced_by {
                Some(next) if next != r => r = next,
                _ => return r,
            },
            Ref::Val(_) => return r,
        }
    }
}

fn rewrite_operands(op: &mut Op, subst: &impl Fn(Value) -> Value) {
    match op {
        Op::Bin(_, a, b) => {
            *a = subst(*a);
            *b = subst(*b);
        }
        Op::Un(_, a) => *a = subst(*a),
        Op::Call { callee, args } => {
            *callee = subst(*callee);
            for a in args {
                *a = subst(*a);
            }
        }
        Op::Member { obj, prop } => {
            *obj = subst(*obj);
            if let crate::cfg::MemberKey::Computed(c) = prop {
                *c = subst(*c);
            }
        }
        Op::StoreMember { obj, prop, value } => {
            *obj = subst(*obj);
            if let crate::cfg::MemberKey::Computed(c) = prop {
                *c = subst(*c);
            }
            *value = subst(*value);
        }
        Op::MakeArray(elems) => {
            for e in elems {
                *e = subst(*e);
            }
        }
        Op::MakeObject(props) => {
            for (k, v) in props {
                if let crate::cfg::PropKey::Computed(c) = k {
                    *c = subst(*c);
                }
                *v = subst(*v);
            }
        }
        Op::Const(_) | Op::Global(_) | Op::ReadVar(_) => {}
        Op::WriteVar(_, v) => *v = subst(*v),
    }
}

fn rewrite_term(term: &mut Term, subst: &impl Fn(Value) -> Value) {
    match term {
        Term::Br(_, args) => {
            for a in args {
                *a = subst(*a);
            }
        }
        Term::CondBr { cond, then_args, else_args, .. } => {
            *cond = subst(*cond);
            for a in then_args.iter_mut().chain(else_args.iter_mut()) {
                *a = subst(*a);
            }
        }
        Term::Ret(Some(v)) => *v = subst(*v),
        Term::Ret(None) | Term::Unreachable => {}
    }
}

/// Reverse postorder of reachable blocks from entry.
pub fn reverse_postorder(cfg: &Cfg) -> Vec<BlockId> {
    let mut visited = HashSet::new();
    let mut post = Vec::new();
    let mut stack = vec![(cfg.entry, false)];
    while let Some((b, processed)) = stack.pop() {
        if processed {
            post.push(b);
            continue;
        }
        if !visited.insert(b) {
            continue;
        }
        stack.push((b, true));
        for s in cfg.block(b).term.successors() {
            if !visited.contains(&s) {
                stack.push((s, false));
            }
        }
    }
    post.reverse();
    post
}
