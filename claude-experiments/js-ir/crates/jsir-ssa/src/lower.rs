//! Lower the reversible JSIR IR (`jshir`/`jsir`) into a pre-SSA CFG.
//!
//! This is the MLIR `scf`-to-`cf` analogue: structured control-flow ops
//! (`jshir.if_statement`, `jshir.while_statement`, …) become basic blocks with
//! branches, while expression value ops (already SSA at the value level in
//! `jsir`) map almost one-to-one to CFG instructions. JS variables stay as
//! named `ReadVar`/`WriteVar` slots here; [`crate::ssa`] promotes them.

use std::collections::HashMap;

use jsir_ir::{Attr, Op as IrOp, ValueId as IrValue};

use crate::cfg::{BinOp, BlockId, Cfg, Const, MemberKey, Op, PropKey, Term, UnOp, VarId, Value};

/// Lower the first function declaration in `file` (or the whole program body if
/// there is none) into a CFG. Returns the CFG and the parameter names.
pub fn lower_function(file: &IrOp) -> Result<Cfg, String> {
    let program = file
        .regions
        .first()
        .and_then(|r| r.blocks.first())
        .and_then(|b| b.ops.first())
        .ok_or("no program op")?;
    let stmts = program
        .regions
        .first()
        .and_then(|r| r.blocks.first())
        .map(|b| b.ops.as_slice())
        .unwrap_or(&[]);

    // Find the first function declaration, descending into `export function` /
    // `export default function` wrappers.
    let func = stmts.iter().find_map(find_function);

    let mut lc = Lower::new();
    let entry = lc.cfg.new_block();
    lc.cfg.entry = entry;
    let mut cur = entry;

    let body_ops: &[IrOp] = match func {
        Some(f) => {
            lc.cfg.fn_name = f.attrs.iter().find_map(|(k,v)| match v { jsir_ir::Attr::Identifier(i) if k=="id" => Some(i.name.clone()), _=>None });
            // region[0] = params, region[1] = body block_statement
            if let Some(params) = f.regions.first().and_then(|r| r.blocks.first()) {
                for p in &params.ops {
                    if p.name == "jsir.identifier_ref" {
                        let var = lc.var_of(p);
                        let v = lc.cfg.fresh_value();
                        lc.cfg.params.push(v);
                        lc.param_names.push(lc.cfg.var_names[var.0 as usize].clone());
                        // Seed the param's variable slot with its incoming value.
                        lc.cfg.push_effect(cur, Op::WriteVar(var, v));
                    }
                }
            }
            f.regions
                .get(1)
                .and_then(|r| r.blocks.first())
                .map(|b| b.ops.as_slice())
                .unwrap_or(&[])
        }
        None => stmts,
    };

    lc.lower_stmts(body_ops, &mut cur)?;
    // A fall-through with no explicit return yields undefined.
    if matches!(lc.cfg.block(cur).term, Term::Unreachable) {
        lc.cfg.block_mut(cur).term = Term::Ret(None);
    }

    let mut cfg = lc.cfg;
    cfg.var_names = lc.var_names_out;
    cfg.param_names = lc.param_names;
    Ok(cfg)
}

struct Lower {
    cfg: Cfg,
    /// jsir value id -> CFG value (lowered expression results).
    vmap: HashMap<IrValue, Value>,
    /// jsir value id -> variable slot, for `identifier_ref` l-values.
    lval: HashMap<IrValue, LvalTarget>,
    /// Variable interner: (name, def_scope) -> VarId.
    vars: HashMap<(String, Option<i64>), VarId>,
    var_names_out: Vec<String>,
    param_names: Vec<String>,
}

#[derive(Clone)]
#[allow(dead_code)] // Member l-values land in a later phase.
enum LvalTarget {
    Var(VarId),
    /// A member l-value `obj.p = ...` (not yet stored as SSA; kept for effects).
    Member { obj: Value, key: MemberKey },
}

impl Lower {
    fn new() -> Lower {
        Lower {
            cfg: Cfg::new(),
            vmap: HashMap::new(),
            lval: HashMap::new(),
            vars: HashMap::new(),
            var_names_out: Vec::new(),
            param_names: Vec::new(),
        }
    }

    /// Intern the variable an identifier(-ref) op refers to, by its resolved
    /// symbol `(name, def_scope)`; falls back to the bare name attr.
    fn var_of(&mut self, op: &IrOp) -> VarId {
        let key = match op.trivia.as_ref().and_then(|t| t.referenced_symbol.as_ref()) {
            Some(s) => (s.name.clone(), s.def_scope_uid),
            None => (str_attr(op, "name").unwrap_or_default(), op.trivia.as_ref().and_then(|t| t.scope_uid)),
        };
        if let Some(v) = self.vars.get(&key) {
            return *v;
        }
        let id = VarId(self.var_names_out.len() as u32);
        self.var_names_out.push(key.0.clone());
        self.cfg.var_names.push(key.0.clone());
        self.vars.insert(key, id);
        id
    }

    fn is_resolved(op: &IrOp) -> bool {
        op.trivia.as_ref().and_then(|t| t.referenced_symbol.as_ref()).is_some()
    }

    fn val(&self, ir: IrValue) -> Result<Value, String> {
        self.vmap.get(&ir).copied().ok_or_else(|| format!("unmapped value {ir:?}"))
    }

    /// Lower a sequence of statement ops into `cur` (which may change as control
    /// flow splits the block).
    fn lower_stmts(&mut self, ops: &[IrOp], cur: &mut BlockId) -> Result<(), String> {
        for op in ops {
            self.lower_op(op, cur)?;
            // Once a block is terminated (return/throw), the rest is unreachable
            // for the current block; stop emitting into it.
            if !matches!(self.cfg.block(*cur).term, Term::Unreachable) {
                break;
            }
        }
        Ok(())
    }

    fn lower_op(&mut self, op: &IrOp, cur: &mut BlockId) -> Result<(), String> {
        match op.name.as_str() {
            // --- expression value ops (already SSA) ---
            "jsir.numeric_literal" => {
                let n = f64_attr(op, "value").unwrap_or(0.0);
                self.def(op, *cur, Op::Const(Const::num(n)));
            }
            "jsir.string_literal" => {
                let s = str_attr(op, "value").unwrap_or_default();
                self.def(op, *cur, Op::Const(Const::Str(s)));
            }
            "jsir.boolean_literal" => {
                let b = bool_attr(op, "value").unwrap_or(false);
                self.def(op, *cur, Op::Const(Const::Bool(b)));
            }
            "jsir.null_literal" => {
                self.def(op, *cur, Op::Const(Const::Null));
            }
            "jsir.identifier" => {
                if Self::is_resolved(op) {
                    let var = self.var_of(op);
                    self.def(op, *cur, Op::ReadVar(var));
                } else {
                    let name = str_attr(op, "name").unwrap_or_default();
                    self.def(op, *cur, Op::Global(name));
                }
            }
            "jsir.identifier_ref" => {
                // An l-value target. Record it; do not emit a read.
                let var = self.var_of(op);
                if let Some(r) = op.results.first() {
                    self.lval.insert(*r, LvalTarget::Var(var));
                }
            }
            "jsir.binary_expression" => {
                let a = self.val(op.operands[0])?;
                let b = self.val(op.operands[1])?;
                let bop = bin_op(&str_attr(op, "operator_").unwrap_or_default())
                    .ok_or_else(|| format!("unsupported binop {:?}", str_attr(op, "operator_")))?;
                self.def(op, *cur, Op::Bin(bop, a, b));
            }
            "jsir.unary_expression" => {
                let a = self.val(op.operands[0])?;
                let uop = un_op(&str_attr(op, "operator_").unwrap_or_default())
                    .ok_or_else(|| format!("unsupported unop {:?}", str_attr(op, "operator_")))?;
                self.def(op, *cur, Op::Un(uop, a));
            }
            "jsir.assignment_expression" => {
                let opname = str_attr(op, "operator_").unwrap_or_default();
                let rhs = self.val(op.operands[1])?;
                let target = self.lval.get(&op.operands[0]).cloned()
                    .ok_or("assignment to unknown l-value")?;
                let stored = if opname == "=" {
                    rhs
                } else {
                    // compound: x op= y  ==>  x = (x op y)
                    let bop = bin_op(opname.trim_end_matches('='))
                        .ok_or_else(|| format!("unsupported compound assign {opname}"))?;
                    match &target {
                        LvalTarget::Var(var) => {
                            let cur_v = self.cfg.push(*cur, Op::ReadVar(*var));
                            self.cfg.push(*cur, Op::Bin(bop, cur_v, rhs))
                        }
                        LvalTarget::Member { .. } => return Err("compound member assign unsupported".into()),
                    }
                };
                match target {
                    LvalTarget::Var(var) => self.cfg.push_effect(*cur, Op::WriteVar(var, stored)),
                    LvalTarget::Member { obj, key } => {
                        self.cfg.push_effect(*cur, Op::StoreMember { obj, prop: key, value: stored })
                    }
                }
                // The assignment expression evaluates to the stored value.
                if let Some(r) = op.results.first() {
                    self.vmap.insert(*r, stored);
                }
            }
            "jsir.variable_declarator" => {
                let target = self.lval.get(&op.operands[0]).cloned();
                let init = if op.operands.len() > 1 {
                    self.val(op.operands[1])?
                } else {
                    self.cfg.push(*cur, Op::Const(Const::Undef))
                };
                if let Some(LvalTarget::Var(var)) = target {
                    self.cfg.push_effect(*cur, Op::WriteVar(var, init));
                }
            }
            "jsir.call_expression" => {
                let callee = self.val(op.operands[0])?;
                let args = op.operands[1..]
                    .iter()
                    .map(|v| self.val(*v))
                    .collect::<Result<Vec<_>, _>>()?;
                self.def(op, *cur, Op::Call { callee, args });
            }
            "jsir.member_expression" => {
                let obj = self.val(op.operands[0])?;
                let key = self.member_key(op)?;
                self.def(op, *cur, Op::Member { obj, prop: key });
            }
            "jsir.member_expression_ref" => {
                // A member l-value (`obj.p = ...`). Record the target.
                let obj = self.val(op.operands[0])?;
                let key = self.member_key(op)?;
                if let Some(r) = op.results.first() {
                    self.lval.insert(*r, LvalTarget::Member { obj, key });
                }
            }
            "jsir.array_expression" => {
                let elems = op.operands.iter().map(|v| self.val(*v)).collect::<Result<Vec<_>, _>>()?;
                self.def(op, *cur, Op::MakeArray(elems));
            }
            "jsir.object_expression" => {
                // The region holds the property value ops (post-order) followed
                // by `object_property` markers and a terminator. Lower the value
                // ops, then collect (key, value) pairs.
                let region: Vec<IrOp> = region_ops(op, 0).to_vec();
                let mut props = Vec::new();
                for p in &region {
                    match p.name.as_str() {
                        "jsir.object_property" => {
                            let key = self.obj_prop_key(p)?;
                            let value = self.val(p.operands[0])?;
                            props.push((key, value));
                        }
                        "jsir.exprs_region_end" | "jsir.expr_region_end" => {}
                        _ => self.lower_op(p, cur)?,
                    }
                }
                self.def(op, *cur, Op::MakeObject(props));
            }
            "jsir.parenthesized_expression" => {
                let inner = self.val(op.operands[0])?;
                if let Some(r) = op.results.first() {
                    self.vmap.insert(*r, inner);
                }
            }
            "jsir.expression_statement" => {} // side effects already emitted
            "jsir.expr_region_end" | "jsir.exprs_region_end" => {} // handled by region lowering
            "jsir.empty_statement" => {}

            // --- declarations / statements with nested op lists ---
            "jsir.variable_declaration" => {
                let inner = region_ops(op, 0);
                self.lower_stmts(inner, cur)?;
            }
            "jshir.block_statement" => {
                let inner = region_ops(op, 0);
                self.lower_stmts(inner, cur)?;
            }
            "jsir.return_statement" => {
                let v = match op.operands.first() {
                    Some(v) => Some(self.val(*v)?),
                    None => None,
                };
                self.cfg.block_mut(*cur).term = Term::Ret(v);
            }

            // --- structured control flow -> blocks ---
            "jshir.if_statement" => self.lower_if(op, cur)?,
            "jshir.while_statement" => self.lower_while(op, cur)?,
            "jshir.conditional_expression" => self.lower_conditional(op, cur)?,
            "jshir.logical_expression" => self.lower_logical(op, cur)?,

            other => return Err(format!("lower: unsupported op {other}")),
        }
        Ok(())
    }

    /// Define an instruction with a result and map the jsir result to it.
    fn def(&mut self, op: &IrOp, block: BlockId, cfg_op: Op) {
        let v = self.cfg.push(block, cfg_op);
        if let Some(r) = op.results.first() {
            self.vmap.insert(*r, v);
        }
    }

    fn lower_if(&mut self, op: &IrOp, cur: &mut BlockId) -> Result<(), String> {
        let cond = self.val(op.operands[0])?;
        let then_b = self.cfg.new_block();
        let else_b = self.cfg.new_block();
        let join_b = self.cfg.new_block();
        self.cfg.block_mut(*cur).term = Term::CondBr {
            cond,
            then_block: then_b,
            then_args: vec![],
            else_block: else_b,
            else_args: vec![],
        };
        // consequent
        let mut t = then_b;
        self.lower_stmts(region_ops(op, 0), &mut t)?;
        if matches!(self.cfg.block(t).term, Term::Unreachable) {
            self.cfg.block_mut(t).term = Term::Br(join_b, vec![]);
        }
        // alternate (region 1 may be empty)
        let mut e = else_b;
        self.lower_stmts(region_ops(op, 1), &mut e)?;
        if matches!(self.cfg.block(e).term, Term::Unreachable) {
            self.cfg.block_mut(e).term = Term::Br(join_b, vec![]);
        }
        *cur = join_b;
        Ok(())
    }

    /// `test ? a : b`: write a synthetic variable in each arm, read it after.
    /// SSA construction turns the synthetic var into a block-argument phi.
    fn lower_conditional(&mut self, op: &IrOp, cur: &mut BlockId) -> Result<(), String> {
        let cond = self.val(op.operands[0])?;
        let tmp = self.fresh_synth_var();
        let (then_b, else_b, join_b) = (self.cfg.new_block(), self.cfg.new_block(), self.cfg.new_block());
        self.cfg.block_mut(*cur).term = Term::CondBr {
            cond,
            then_block: then_b,
            then_args: vec![],
            else_block: else_b,
            else_args: vec![],
        };
        // NB: `jshir.conditional_expression` stores the **alternate** region
        // first (region 0), then the **consequent** (region 1).
        let mut t = then_b;
        let tv = self.lower_expr_region(op, 1, &mut t)?; // consequent
        self.cfg.push_effect(t, Op::WriteVar(tmp, tv));
        self.cfg.block_mut(t).term = Term::Br(join_b, vec![]);
        let mut e = else_b;
        let ev = self.lower_expr_region(op, 0, &mut e)?; // alternate
        self.cfg.push_effect(e, Op::WriteVar(tmp, ev));
        self.cfg.block_mut(e).term = Term::Br(join_b, vec![]);
        *cur = join_b;
        let r = self.cfg.push(join_b, Op::ReadVar(tmp));
        if let Some(res) = op.results.first() {
            self.vmap.insert(*res, r);
        }
        Ok(())
    }

    /// `a && b` / `a || b` / `a ?? b`: short-circuit via a synthetic variable.
    fn lower_logical(&mut self, op: &IrOp, cur: &mut BlockId) -> Result<(), String> {
        let left = self.val(op.operands[0])?;
        let oper = str_attr(op, "operator_").unwrap_or_default();
        let tmp = self.fresh_synth_var();
        self.cfg.push_effect(*cur, Op::WriteVar(tmp, left));
        // Condition under which we evaluate (and take) the right-hand side.
        let take_rhs = match oper.as_str() {
            "&&" => left,                                       // left truthy
            "||" => self.cfg.push(*cur, Op::Un(UnOp::Not, left)), // left falsy
            "??" => {
                let nullc = self.cfg.push(*cur, Op::Const(Const::Null));
                self.cfg.push(*cur, Op::Bin(BinOp::Eq, left, nullc)) // left == null/undefined
            }
            other => return Err(format!("unsupported logical op {other}")),
        };
        let (rhs_b, join_b) = (self.cfg.new_block(), self.cfg.new_block());
        self.cfg.block_mut(*cur).term = Term::CondBr {
            cond: take_rhs,
            then_block: rhs_b,
            then_args: vec![],
            else_block: join_b,
            else_args: vec![],
        };
        let mut r = rhs_b;
        let rv = self.lower_expr_region(op, 0, &mut r)?;
        self.cfg.push_effect(r, Op::WriteVar(tmp, rv));
        self.cfg.block_mut(r).term = Term::Br(join_b, vec![]);
        *cur = join_b;
        let res_v = self.cfg.push(join_b, Op::ReadVar(tmp));
        if let Some(res) = op.results.first() {
            self.vmap.insert(*res, res_v);
        }
        Ok(())
    }

    /// The member key of a `member_expression`(`_ref`): static if it carries a
    /// `literal_property` identifier attr, else the computed operand.
    /// An object-literal property's key (`{x: v}` / `{x}` / `{[k]: v}`).
    fn obj_prop_key(&self, p: &IrOp) -> Result<PropKey, String> {
        if let Some(name) = p.attrs.iter().find_map(|(k, v)| match v {
            Attr::Identifier(i) if k == "literal_key" => Some(i.name.clone()),
            Attr::Str(s) if k == "literal_key" => Some(s.clone()),
            _ => None,
        }) {
            Ok(PropKey::Ident(name))
        } else if p.operands.len() > 1 {
            Ok(PropKey::Computed(self.val(p.operands[1])?))
        } else {
            Err("object property: no key".into())
        }
    }

    fn member_key(&self, op: &IrOp) -> Result<MemberKey, String> {
        if let Some(name) = member_prop_name(op) {
            Ok(MemberKey::Static(name))
        } else {
            let prop = op.operands.get(1).ok_or("computed member without key")?;
            Ok(MemberKey::Computed(self.val(*prop)?))
        }
    }

    fn fresh_synth_var(&mut self) -> VarId {
        let id = VarId(self.var_names_out.len() as u32);
        let name = format!("$t{}", id.0);
        self.var_names_out.push(name.clone());
        self.cfg.var_names.push(name);
        id
    }

    fn lower_while(&mut self, op: &IrOp, cur: &mut BlockId) -> Result<(), String> {
        let header = self.cfg.new_block();
        let body = self.cfg.new_block();
        let exit = self.cfg.new_block();
        self.cfg.block_mut(*cur).term = Term::Br(header, vec![]);
        // header: evaluate the test (region 0 is an expr region)
        let mut h = header;
        let cond = self.lower_expr_region(op, 0, &mut h)?;
        self.cfg.block_mut(h).term = Term::CondBr {
            cond,
            then_block: body,
            then_args: vec![],
            else_block: exit,
            else_args: vec![],
        };
        // body
        let mut b = body;
        self.lower_stmts(region_ops(op, 1), &mut b)?;
        if matches!(self.cfg.block(b).term, Term::Unreachable) {
            self.cfg.block_mut(b).term = Term::Br(header, vec![]);
        }
        *cur = exit;
        Ok(())
    }

    /// Lower an expression region (ends in `expr_region_end`), returning its value.
    fn lower_expr_region(&mut self, op: &IrOp, idx: usize, cur: &mut BlockId) -> Result<Value, String> {
        let ops = region_ops(op, idx);
        self.lower_stmts(ops, cur)?;
        let end = ops
            .iter()
            .find(|o| o.name == "jsir.expr_region_end")
            .ok_or("expr region has no terminator")?;
        self.val(end.operands[0])
    }
}

/// The first `function_declaration` in `op`, descending one level into export
/// wrappers (`export function` / `export default function`).
fn find_function(op: &IrOp) -> Option<&IrOp> {
    if op.name == "jsir.function_declaration" {
        return Some(op);
    }
    if op.name == "jsir.export_named_declaration" || op.name == "jsir.export_default_declaration" {
        return region_ops(op, 0).iter().find(|o| o.name == "jsir.function_declaration");
    }
    None
}

fn region_ops(op: &IrOp, idx: usize) -> &[IrOp] {
    op.regions
        .get(idx)
        .and_then(|r| r.blocks.first())
        .map(|b| b.ops.as_slice())
        .unwrap_or(&[])
}

fn str_attr(op: &IrOp, key: &str) -> Option<String> {
    op.attrs.iter().find_map(|(k, v)| match v {
        Attr::Str(s) if k == key => Some(s.clone()),
        _ => None,
    })
}
fn f64_attr(op: &IrOp, key: &str) -> Option<f64> {
    op.attrs.iter().find_map(|(k, v)| match v {
        Attr::F64(f) if k == key => Some(*f),
        _ => None,
    })
}
fn bool_attr(op: &IrOp, key: &str) -> Option<bool> {
    op.attrs.iter().find_map(|(k, v)| match v {
        Attr::Bool(b) if k == key => Some(*b),
        _ => None,
    })
}
fn member_prop_name(op: &IrOp) -> Option<String> {
    op.attrs.iter().find_map(|(k, v)| match v {
        Attr::Identifier(i) if k == "literal_property" => Some(i.name.clone()),
        Attr::Str(s) if k == "literal_property" => Some(s.clone()),
        _ => None,
    })
}

fn bin_op(s: &str) -> Option<BinOp> {
    Some(match s {
        "+" => BinOp::Add, "-" => BinOp::Sub, "*" => BinOp::Mul, "/" => BinOp::Div,
        "%" => BinOp::Mod, "**" => BinOp::Pow,
        "==" => BinOp::Eq, "!=" => BinOp::Ne, "===" => BinOp::StrictEq, "!==" => BinOp::StrictNe,
        "<" => BinOp::Lt, "<=" => BinOp::Le, ">" => BinOp::Gt, ">=" => BinOp::Ge,
        "&" => BinOp::BitAnd, "|" => BinOp::BitOr, "^" => BinOp::BitXor,
        "<<" => BinOp::Shl, ">>" => BinOp::Shr, ">>>" => BinOp::UShr,
        _ => return None,
    })
}
fn un_op(s: &str) -> Option<UnOp> {
    Some(match s {
        "-" => UnOp::Neg, "+" => UnOp::Pos, "!" => UnOp::Not, "~" => UnOp::BitNot,
        "typeof" => UnOp::TypeOf, "void" => UnOp::Void,
        _ => return None,
    })
}

// Silence unused warnings for fields used by later phases.
#[allow(dead_code)]
fn _uses(_: &PropKey) {}
