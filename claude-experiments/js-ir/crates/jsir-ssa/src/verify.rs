//! A well-formedness verifier for SSA CFGs, mirroring MLIR's invariants:
//!  - every value has exactly one definition (param, block argument, or result);
//!  - every use is dominated by its definition (for block-argument operands, the
//!    operand must be available at the end of the passing predecessor);
//!  - each terminator passes exactly as many operands as the successor has block
//!    arguments.
//!
//! Returns a list of violations; empty means well-formed.

use std::collections::HashMap;

use crate::cfg::{BlockId, Cfg, MemberKey, Op, PropKey, Term, Value};

use crate::ssa::reverse_postorder;

/// Position of a use/def within the function: `(block, slot)` where slot `-1` is
/// a block argument/param (available at block entry), `0..n` are instruction
/// indices, and `usize::MAX` is the terminator (end of block).
#[derive(Clone, Copy)]
struct Pos {
    block: BlockId,
    slot: isize,
}

pub fn verify(cfg: &Cfg) -> Vec<String> {
    let mut errs = Vec::new();
    let order = reverse_postorder(cfg);
    let idom = dominators(cfg, &order);

    // 1. Single definition + def positions.
    let mut def: HashMap<Value, Pos> = HashMap::new();
    let mut define = |v: Value, pos: Pos, errs: &mut Vec<String>| {
        if def.insert(v, pos).is_some() {
            errs.push(format!("%{} defined more than once", v.0));
        }
    };
    for p in &cfg.params {
        define(*p, Pos { block: cfg.entry, slot: -1 }, &mut errs);
    }
    for b in &cfg.blocks {
        for p in &b.params {
            define(*p, Pos { block: b.id, slot: -1 }, &mut errs);
        }
        for (i, ins) in b.instrs.iter().enumerate() {
            if let Some(r) = ins.result {
                define(r, Pos { block: b.id, slot: i as isize }, &mut errs);
            }
        }
    }

    let dominates_pos = |d: &Pos, u: &Pos| -> bool {
        if d.block == u.block {
            d.slot < u.slot
        } else {
            strictly_or_equally_dominates(&idom, d.block, u.block)
        }
    };

    // 2. Uses: each operand must be defined and dominated.
    let check_use = |v: Value, at: Pos, ctx: &str, errs: &mut Vec<String>| match def.get(&v) {
        None => errs.push(format!("use of undefined %{} ({ctx})", v.0)),
        Some(d) => {
            if !dominates_pos(d, &at) {
                errs.push(format!("%{} used before def / not dominated ({ctx})", v.0));
            }
        }
    };

    for b in &cfg.blocks {
        for (i, ins) in b.instrs.iter().enumerate() {
            let at = Pos { block: b.id, slot: i as isize };
            for v in operand_values(&ins.op) {
                check_use(v, at, "instr", &mut errs);
            }
        }
        // Terminator operands are used at end of block (slot = +inf).
        let end = Pos { block: b.id, slot: isize::MAX };
        match &b.term {
            Term::Br(t, args) => {
                check_arity(cfg, *t, args.len(), b.id, &mut errs);
                for v in args {
                    check_use(*v, end, "br arg", &mut errs);
                }
            }
            Term::CondBr { cond, then_block, then_args, else_block, else_args } => {
                check_use(*cond, end, "cond", &mut errs);
                check_arity(cfg, *then_block, then_args.len(), b.id, &mut errs);
                check_arity(cfg, *else_block, else_args.len(), b.id, &mut errs);
                for v in then_args.iter().chain(else_args) {
                    check_use(*v, end, "condbr arg", &mut errs);
                }
            }
            Term::Ret(Some(v)) => check_use(*v, end, "ret", &mut errs),
            Term::Ret(None) | Term::Unreachable => {}
        }
    }

    errs
}

fn check_arity(cfg: &Cfg, succ: BlockId, n: usize, from: BlockId, errs: &mut Vec<String>) {
    let want = cfg.block(succ).params.len();
    if want != n {
        errs.push(format!("^bb{} -> ^bb{}: {n} args for {want} block params", from.0, succ.0));
    }
}

fn operand_values(op: &Op) -> Vec<Value> {
    match op {
        Op::Bin(_, a, b) => vec![*a, *b],
        Op::Un(_, a) => vec![*a],
        Op::Call { callee, args } => {
            let mut v = vec![*callee];
            v.extend(args.iter().copied());
            v
        }
        Op::Member { obj, prop } => {
            let mut v = vec![*obj];
            if let MemberKey::Computed(c) = prop {
                v.push(*c);
            }
            v
        }
        Op::StoreMember { obj, prop, value } => {
            let mut v = vec![*obj];
            if let MemberKey::Computed(c) = prop {
                v.push(*c);
            }
            v.push(*value);
            v
        }
        Op::MakeArray(e) => e.clone(),
        Op::MakeObject(p) => {
            let mut v = Vec::new();
            for (k, val) in p {
                if let PropKey::Computed(c) = k {
                    v.push(*c);
                }
                v.push(*val);
            }
            v
        }
        Op::WriteVar(_, v) => vec![*v],
        Op::Const(_) | Op::Global(_) | Op::ReadVar(_) => vec![],
    }
}

// --- dominators (Cooper-Harvey-Kennedy iterative) ---

fn dominators(cfg: &Cfg, order: &[BlockId]) -> HashMap<BlockId, BlockId> {
    let preds = cfg.predecessors();
    // RPO index of each block.
    let mut rpo_idx: HashMap<BlockId, usize> = HashMap::new();
    for (i, b) in order.iter().enumerate() {
        rpo_idx.insert(*b, i);
    }
    let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
    idom.insert(cfg.entry, cfg.entry);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().skip(1) {
            // skip entry
            let mut new_idom: Option<BlockId> = None;
            for &p in preds.get(&b).map(|v| v.as_slice()).unwrap_or(&[]) {
                if !idom.contains_key(&p) {
                    continue; // not yet processed
                }
                new_idom = Some(match new_idom {
                    None => p,
                    Some(n) => intersect(&idom, &rpo_idx, p, n),
                });
            }
            if let Some(ni) = new_idom {
                if idom.get(&b) != Some(&ni) {
                    idom.insert(b, ni);
                    changed = true;
                }
            }
        }
    }
    idom
}

fn intersect(
    idom: &HashMap<BlockId, BlockId>,
    rpo: &HashMap<BlockId, usize>,
    mut a: BlockId,
    mut b: BlockId,
) -> BlockId {
    while a != b {
        while rpo[&a] > rpo[&b] {
            a = idom[&a];
        }
        while rpo[&b] > rpo[&a] {
            b = idom[&b];
        }
    }
    a
}

/// Does `a` dominate `b` (a == b counts)?
fn strictly_or_equally_dominates(idom: &HashMap<BlockId, BlockId>, a: BlockId, b: BlockId) -> bool {
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        match idom.get(&x) {
            Some(&p) if p != x => x = p,
            _ => return false,
        }
    }
}
