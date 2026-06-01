//! Lower the reversible JSIR IR (`jshir`/`jsir`) into a pre-SSA CFG.
//!
//! This is the MLIR `scf`-to-`cf` analogue: structured control-flow ops
//! (`jshir.if_statement`, `jshir.while_statement`, …) become basic blocks with
//! branches, while expression value ops (already SSA at the value level in
//! `jsir`) map almost one-to-one to CFG instructions. JS variables stay as
//! named `ReadVar`/`WriteVar` slots here; [`crate::ssa`] promotes them.

use std::collections::HashMap;

use jsir_ir::{Attr, Op as IrOp, ValueId as IrValue};

use crate::cfg::{BinOp, BlockId, BlockKind, Cfg, Const, MemberKey, Op, PropKey, SrcRef, Term, UnOp, VarId, Value};

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
    /// The `node_id` of the statement-root JSIR op currently being lowered.
    /// Threaded onto each emitted [`crate::cfg::Instr`] via `Cfg::cur_src` so an
    /// IR-rewrite memoizer can map analysis `Value`s back to the JSIR statement
    /// that produced them. `None` when no statement context is established.
    cur_stmt_node_id: Option<u32>,
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
enum LvalTarget {
    Var(VarId),
    /// A member l-value `obj.p = ...` (not yet stored as SSA; kept for effects).
    Member { obj: Value, key: MemberKey },
    /// An object destructuring pattern `{a, b: c} = init`: each entry binds the
    /// member `init[key]` to a variable. (Flat identifier targets only.)
    ObjectPattern(Vec<(MemberKey, VarId)>),
    /// An array destructuring pattern `[x, , y] = init`: each slot binds
    /// `init[index]` to a variable (`None` for an elision hole).
    ArrayPattern(Vec<Option<VarId>>),
}

impl Lower {
    fn new() -> Lower {
        Lower {
            cfg: Cfg::new(),
            cur_stmt_node_id: None,
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
    ///
    /// A statements region is a flat post-order list: a run of 1-result operand
    /// ops followed by their 0-result statement-root op. Every instruction
    /// emitted while lowering that run (operands *and* root) is stamped with the
    /// statement root's `node_id` so analysis results can be mapped back to the
    /// originating JSIR statement.
    fn lower_stmts(&mut self, ops: &[IrOp], cur: &mut BlockId) -> Result<(), String> {
        let owners = stmt_owner_node_ids(ops);
        let saved = self.cur_stmt_node_id;
        for (op, owner) in ops.iter().zip(owners.iter()) {
            // Establish statement provenance for this op (operands inherit the
            // following statement root's id; the root keeps its own).
            if let Some(id) = owner {
                self.cur_stmt_node_id = Some(*id);
            }
            self.cfg.cur_src = self.cur_stmt_node_id.map(|stmt_node_id| SrcRef { stmt_node_id });
            self.lower_op(op, cur)?;
            // Once a block is terminated (return/throw), the rest is unreachable
            // for the current block; stop emitting into it.
            if !matches!(self.cfg.block(*cur).term, Term::Unreachable) {
                break;
            }
        }
        self.cur_stmt_node_id = saved;
        self.cfg.cur_src = saved.map(|stmt_node_id| SrcRef { stmt_node_id });
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
            "jsir.new_expression" => {
                // `new Callee(args)` — an allocation. For the analysis it behaves
                // like a (non-pure) call: the result is a fresh reference that may
                // capture/mutate its arguments. The reversible IR keeps the `new`.
                let callee = self.val(op.operands[0])?;
                let args = op.operands[1..]
                    .iter()
                    .map(|v| self.val(*v))
                    .collect::<Result<Vec<_>, _>>()?;
                self.def(op, *cur, Op::Call { callee, args });
            }
            "jsir.sequence_expression" => {
                // `(a, b, c)` — operands' side effects are already emitted in
                // order; the expression's value is the last operand.
                let last = *op.operands.last().ok_or("empty sequence expression")?;
                let v = self.val(last)?;
                if let Some(r) = op.results.first() {
                    self.vmap.insert(*r, v);
                }
            }
            "jsir.update_expression" => {
                // `x++` / `--x` / `obj.p++`. Desugar to read, ±1, write-back; the
                // expression's value is the new value (prefix) or old (postfix).
                let opname = str_attr(op, "operator_").unwrap_or_default();
                let prefix = bool_attr(op, "prefix").unwrap_or(false);
                let bop = match opname.as_str() {
                    "++" => BinOp::Add,
                    "--" => BinOp::Sub,
                    _ => return Err(format!("unsupported update operator {opname:?}")),
                };
                let target = self
                    .lval
                    .get(&op.operands[0])
                    .cloned()
                    .ok_or("update of unknown l-value")?;
                let one = self.cfg.push(*cur, Op::Const(Const::num(1.0)));
                let (old, newv) = match &target {
                    LvalTarget::Var(var) => {
                        let cur_v = self.cfg.push(*cur, Op::ReadVar(*var));
                        let nv = self.cfg.push(*cur, Op::Bin(bop, cur_v, one));
                        self.cfg.push_effect(*cur, Op::WriteVar(*var, nv));
                        (cur_v, nv)
                    }
                    LvalTarget::Member { obj, key } => {
                        let cur_v = self.cfg.push(*cur, Op::Member { obj: *obj, prop: key.clone() });
                        let nv = self.cfg.push(*cur, Op::Bin(bop, cur_v, one));
                        self.cfg.push_effect(
                            *cur,
                            Op::StoreMember { obj: *obj, prop: key.clone(), value: nv },
                        );
                        (cur_v, nv)
                    }
                    _ => return Err("update expression on a destructuring pattern".into()),
                };
                if let Some(r) = op.results.first() {
                    self.vmap.insert(*r, if prefix { newv } else { old });
                }
            }
            "jsir.template_element_value" => {
                // The cooked text of one template quasi (a string primitive).
                let s = str_attr(op, "cooked").unwrap_or_default();
                self.def(op, *cur, Op::Const(Const::Str(s)));
            }
            "jsir.template_element" => {
                // Wraps a `template_element_value`; pass its value through.
                let v = self.val(op.operands[0])?;
                if let Some(r) = op.results.first() {
                    self.vmap.insert(*r, v);
                }
            }
            "jsir.template_literal" => {
                // `` `q0${e0}q1${e1}…` `` — operands are the quasis then the
                // expressions (`operandSegmentSizes = [num_quasis, num_exprs]`).
                // Desugar to string concatenation q0 + e0 + q1 + e1 + …; the
                // result is a primitive that depends on the embedded expressions.
                let n_quasi = op
                    .attrs
                    .iter()
                    .find_map(|(k, v)| match v {
                        Attr::I32Array(a) if k == "operandSegmentSizes" => a.first().copied(),
                        _ => None,
                    })
                    .unwrap_or(0) as usize;
                let quasis: Vec<Value> = op.operands[..n_quasi]
                    .iter()
                    .map(|v| self.val(*v))
                    .collect::<Result<_, _>>()?;
                let exprs: Vec<Value> = op.operands[n_quasi..]
                    .iter()
                    .map(|v| self.val(*v))
                    .collect::<Result<_, _>>()?;
                let mut acc = *quasis.first().ok_or("template literal without quasis")?;
                for (i, e) in exprs.iter().enumerate() {
                    acc = self.cfg.push(*cur, Op::Bin(BinOp::Add, acc, *e));
                    if let Some(q) = quasis.get(i + 1) {
                        acc = self.cfg.push(*cur, Op::Bin(BinOp::Add, acc, *q));
                    }
                }
                if let Some(r) = op.results.first() {
                    self.vmap.insert(*r, acc);
                }
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
                        // `{a} op= x` / `[a] op= x` are not valid JS.
                        _ => return Err("compound assignment to a destructuring pattern".into()),
                    }
                };
                self.bind_target(&target, stored, *cur);
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
                if let Some(t) = target {
                    self.bind_target(&t, init, *cur);
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
            "jsir.arrow_function_expression" | "jsir.function_expression" => {
                // A closure is an allocation that captures its free variables. We
                // do NOT lower the body into the CFG; for the memoization analysis
                // it suffices to model the closure as an allocation whose operands
                // are reads of the outer variables it references (its reactive
                // captures become the scope's dependencies, exactly as React
                // memoizes `useCallback`-style closures). The reversible JSIR keeps
                // the real function, so the IR-rewrite codegen reprints it verbatim
                // — no variable renaming, so this is sound (unlike the retired
                // string-codegen closure path). Represented as `MakeArray` of the
                // captures so every allocation/aliasing site treats it uniformly.
                let caps = self.collect_captures(op, *cur);
                self.def(op, *cur, Op::MakeArray(caps));
            }
            "jsir.member_expression" | "jsir.optional_member_expression" => {
                // `a.b` and `a?.b` read the same way for the analysis; the `?.`
                // short-circuit does not change the memoization structure.
                let obj = self.val(op.operands[0])?;
                let key = self.member_key(op)?;
                self.def(op, *cur, Op::Member { obj, prop: key });
            }
            "jsir.tagged_template_expression" => {
                // `` tag`…` `` is a call `tag(strings, ...exprs)`; model it as a
                // call of the tag with the template value (which carries the
                // embedded expressions as its dependencies).
                let callee = self.val(op.operands[0])?;
                let args = op.operands[1..]
                    .iter()
                    .map(|v| self.val(*v))
                    .collect::<Result<Vec<_>, _>>()?;
                self.def(op, *cur, Op::Call { callee, args });
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
                        "jsir.spread_element" => {
                            // `{...x}` captures `x` into the new object. We record
                            // it as a captured value (synthetic key) so the object
                            // aliases and depends on `x`; the reversible IR keeps
                            // the real spread for codegen, so the key is never
                            // emitted.
                            let value = self.val(p.operands[0])?;
                            props.push((PropKey::Ident("...".into()), value));
                        }
                        "jsir.object_method" => {
                            // `{ m() {…} }` — a method is a closure-valued property.
                            // Capture its free variables (like an arrow) and record
                            // it as the property's value.
                            let caps = self.collect_captures(p, *cur);
                            let closure = self.cfg.push(*cur, Op::MakeArray(caps));
                            let key = self.obj_prop_key(p)?;
                            props.push((key, closure));
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
            "jsir.spread_element" => {
                // `...x` in an array/call (object spread is handled inside
                // `object_expression`). For the analysis, a spread captures its
                // source into the container — modelling the spread's result *as*
                // its inner value makes the container alias and depend on the
                // source (`[...a]` invalidates when `a` does), which is what the
                // memoization analysis needs. The reversible IR keeps the real
                // spread, so emitted code is unaffected.
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

            // --- destructuring l-value patterns ---
            "jsir.object_pattern_ref" => {
                // The pattern's region holds `identifier_ref` (binding targets)
                // and `object_property_ref` (key + target) ops. Lower the nested
                // identifier-refs so their `LvalTarget::Var` entries exist, then
                // pair each property key with its target var.
                let region: Vec<IrOp> = region_ops(op, 0).to_vec();
                let mut binds: Vec<(MemberKey, VarId)> = Vec::new();
                for p in &region {
                    match p.name.as_str() {
                        "jsir.identifier_ref" => self.lower_op(p, cur)?,
                        "jsir.object_property_ref" => {
                            let key = match self.obj_prop_key(p)? {
                                PropKey::Ident(n) => MemberKey::Static(n),
                                PropKey::Computed(c) => MemberKey::Computed(c),
                            };
                            match self.lval.get(&p.operands[0]).cloned() {
                                Some(LvalTarget::Var(var)) => binds.push((key, var)),
                                _ => return Err(
                                    "object pattern: only flat identifier bindings supported \
                                     (nested patterns / defaults / rest not yet lowered)"
                                        .into(),
                                ),
                            }
                        }
                        "jsir.exprs_region_end" | "jsir.expr_region_end" => {}
                        other => {
                            return Err(format!("object pattern: unsupported element {other}"))
                        }
                    }
                }
                if let Some(r) = op.results.first() {
                    self.lval.insert(*r, LvalTarget::ObjectPattern(binds));
                }
            }
            "jsir.array_pattern_ref" => {
                // Operands are the element l-value results in order (lowered as
                // preceding sibling ops). A missing operand slot is an elision.
                let mut slots: Vec<Option<VarId>> = Vec::new();
                for o in &op.operands {
                    match self.lval.get(o).cloned() {
                        Some(LvalTarget::Var(var)) => slots.push(Some(var)),
                        _ => return Err(
                            "array pattern: only flat identifier bindings supported \
                             (nested patterns / defaults / rest not yet lowered)"
                                .into(),
                        ),
                    }
                }
                if let Some(r) = op.results.first() {
                    self.lval.insert(*r, LvalTarget::ArrayPattern(slots));
                }
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

    /// Store `init` into an l-value target, expanding destructuring patterns
    /// into the member reads + variable writes React's lowering also produces
    /// (`const {a} = props` => `a = props.a`).
    fn bind_target(&mut self, target: &LvalTarget, init: Value, block: BlockId) {
        match target {
            LvalTarget::Var(var) => {
                self.cfg.push_effect(block, Op::WriteVar(*var, init));
            }
            LvalTarget::Member { obj, key } => {
                self.cfg.push_effect(
                    block,
                    Op::StoreMember { obj: *obj, prop: key.clone(), value: init },
                );
            }
            LvalTarget::ObjectPattern(binds) => {
                for (key, var) in binds {
                    let m = self.cfg.push(block, Op::Member { obj: init, prop: key.clone() });
                    self.cfg.push_effect(block, Op::WriteVar(*var, m));
                }
            }
            LvalTarget::ArrayPattern(slots) => {
                for (i, slot) in slots.iter().enumerate() {
                    if let Some(var) = slot {
                        let idx = self.cfg.push(block, Op::Const(Const::num(i as f64)));
                        let m = self
                            .cfg
                            .push(block, Op::Member { obj: init, prop: MemberKey::Computed(idx) });
                        self.cfg.push_effect(block, Op::WriteVar(*var, m));
                    }
                }
            }
        }
    }

    /// Collect a closure's free-variable captures: emit a `ReadVar` for each
    /// distinct outer variable its body references, returning those values. An
    /// identifier is a capture iff its resolved symbol `(name, def_scope)` is one
    /// we have already interned (i.e. declared in an enclosing function we are
    /// lowering); the closure's own params/locals live in a different scope and so
    /// are not matched. We over-approximate at property granularity (`props.a`
    /// captures `props`), which preserves the *number* of dependencies React uses.
    fn collect_captures(&mut self, op: &IrOp, cur: BlockId) -> Vec<Value> {
        // Flatten every op in the closure body.
        let mut ops: Vec<&IrOp> = Vec::new();
        let mut stack: Vec<&IrOp> = Vec::new();
        for r in &op.regions {
            for b in &r.blocks {
                for o in &b.ops {
                    stack.push(o);
                }
            }
        }
        while let Some(o) = stack.pop() {
            ops.push(o);
            for r in &o.regions {
                for b in &r.blocks {
                    for c in &b.ops {
                        stack.push(c);
                    }
                }
            }
        }
        // Map each identifier value that is the base of a static member access to
        // the property name, so we capture at property-path granularity (React's
        // dependency granularity: `() => p.a + p.b` depends on `p.a` and `p.b`,
        // two deps — not on `p`). First-level only (`p.a.b` captures `p.a`), which
        // matches React's dependency *count* in the common cases.
        let mut member_prop: HashMap<IrValue, String> = HashMap::new();
        for o in &ops {
            if o.name == "jsir.member_expression" {
                if let (Some(&base), Some(prop)) = (o.operands.first(), member_prop_name(o)) {
                    member_prop.entry(base).or_insert(prop);
                }
            }
        }
        // For each identifier resolving to an already-interned OUTER variable,
        // capture the (var, optional property) path it is read through. Deduped.
        let mut keys: Vec<(VarId, Option<String>)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for o in &ops {
            if o.name != "jsir.identifier" {
                continue;
            }
            let Some(sym) = o.trivia.as_ref().and_then(|t| t.referenced_symbol.as_ref()) else {
                continue;
            };
            let vkey = (sym.name.clone(), sym.def_scope_uid);
            let Some(&var) = self.vars.get(&vkey) else { continue };
            let prop = o.results.first().and_then(|r| member_prop.get(r)).cloned();
            if seen.insert((var, prop.clone())) {
                keys.push((var, prop));
            }
        }
        // Synthesize the capturing reads in the outer block.
        let mut caps = Vec::new();
        for (var, prop) in keys {
            let base = self.cfg.push(cur, Op::ReadVar(var));
            let v = match prop {
                Some(p) => self.cfg.push(cur, Op::Member { obj: base, prop: MemberKey::Static(p) }),
                None => base,
            };
            caps.push(v);
        }
        caps
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
        let head = *cur;
        let then_b = self.cfg.new_block();
        let else_b = self.cfg.new_block();
        let join_b = self.cfg.new_block();
        self.cfg.record_join(head, join_b, BlockKind::Block)?;
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
        let head = *cur;
        let tmp = self.fresh_synth_var();
        let (then_b, else_b, join_b) = (self.cfg.new_block(), self.cfg.new_block(), self.cfg.new_block());
        self.cfg.record_join(head, join_b, BlockKind::Block)?;
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
        let head = *cur;
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
        self.cfg.record_join(head, join_b, BlockKind::Block)?;
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
        // The loop header is the test block whose terminator is the CondBr; its
        // join is the loop-exit block. `lower_expr_region` cannot split a block
        // (an expr region has no statement-level control flow), so `h == header`.
        if h != header {
            return Err(format!(
                "lower: while header split across blocks ({header:?} -> {h:?}); cannot record loop join"
            ));
        }
        self.cfg.record_join(header, exit, BlockKind::Loop)?;
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

/// For each op in a flat statements region, the `node_id` of the statement root
/// that owns it. A statements region is a post-order list: a run of 1-result
/// operand ops followed by their 0-result statement root (the same grouping
/// `hir2ast`'s `stmts` uses: 0-result ops are statement roots). Every operand op
/// in a run inherits the `node_id` of the statement root that closes the run.
/// Trailing operand ops with no following statement root (and roots without a
/// `node_id`) map to `None`.
fn stmt_owner_node_ids(ops: &[IrOp]) -> Vec<Option<u32>> {
    let mut owners: Vec<Option<u32>> = vec![None; ops.len()];
    let mut run_start = 0usize;
    for (i, op) in ops.iter().enumerate() {
        if op.results.is_empty() {
            // Statement root: it and every operand op since the last root share
            // its node_id.
            for slot in owners.iter_mut().take(i + 1).skip(run_start) {
                *slot = op.node_id;
            }
            run_start = i + 1;
        }
    }
    owners
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
