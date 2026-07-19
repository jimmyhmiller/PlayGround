//! `jsir_to_hir` — a converter from a JSLIR function-body CFG to a real
//! `react_compiler_hir::HirFunction`, so the **actual** React Compiler passes can
//! run on JSIR-derived code (strategy (B) in `docs/HIR_COMPARISON.md` §9).
//!
//! This is feasible because JSLIR is already a flat, per-block SSA-value op stream
//! (each `jsir.*` op produces a result `ValueId`; operands are earlier results —
//! see `jsir-convert`'s `lower_expr`), which lines up almost 1:1 with React's flat
//! `HirFunction.instructions` arena (`Instruction { lvalue: Place, value:
//! InstructionValue }`). So the converter is mostly a per-op mapping, not a
//! scalarization engine.
//!
//! Scope (bounded and **sound** — anything unhandled returns `Err`, never a wrong
//! HIR): straight-line + `if`-diamond control flow; the common value ops
//! (literals, identifier loads, binary/unary, calls, `store_local`). It produces a
//! structurally well-formed `HirFunction` in **non-SSA** form (named locals share
//! one `IdentifierId`), which is exactly the shape `react_compiler_ssa::enter_ssa`
//! expects as input. Loops, switch, try, destructuring, member/computed access,
//! objects/arrays/JSX, etc. are out of scope for this first cut and reported as
//! errors with the offending op name.

use std::collections::HashMap;

use jsir_ir::{Attr, Op, Region, ValueId};
use react_compiler_hir::environment::Environment;
use react_compiler_hir::{
    BasicBlock, BlockId as RBlockId, BlockKind, Effect, EvaluationOrder, FloatValue, GotoVariant,
    HirFunction, IdentifierId, IdentifierName, Instruction, InstructionId, InstructionKind,
    InstructionValue, LValue, NonLocalBinding, Place, PlaceOrSpread, PrimitiveValue,
    ReactFunctionType, ReturnVariant, Terminal, HIR,
};

use crate::dialect;

type R<T> = Result<T, String>;

/// Convert a JSLIR function-body CFG (`region`) into a `HirFunction`. `env` owns
/// the identifier/type arenas and id counters the result references; pass a fresh
/// `Environment::new()` (or reuse one across functions).
pub fn convert_function(region: &Region, env: &mut Environment) -> R<HirFunction> {
    if !dialect::region_is_cfg(region) {
        return Err("region is not a lowered JSLIR CFG".into());
    }
    let mut cx = Cx {
        env,
        value_ids: HashMap::new(),
        vars: HashMap::new(),
        ref_targets: HashMap::new(),
        instructions: Vec::new(),
    };

    let mut blocks = indexmap::IndexMap::new();
    for block in &region.blocks {
        let rblock = cx.convert_block(block)?;
        blocks.insert(RBlockId(block.id.0), rblock);
    }
    let entry = RBlockId(region.blocks.first().ok_or("empty region")?.id.0);

    // Populate predecessors from the JSLIR successor edges (which are the real CFG
    // edges — `if` arms/`br` targets — and match React's `each_terminal_successor`).
    // Passes like `enter_ssa` consult `block.preds`.
    for block in &region.blocks {
        if let Some(term) = block.ops.last() {
            for succ in &term.successors {
                if let Some(rb) = blocks.get_mut(&RBlockId(succ.block.0)) {
                    rb.preds.insert(RBlockId(block.id.0));
                }
            }
        }
    }

    // `returns` is a synthetic place for the function's return value (React always
    // has one). A fresh temp suffices for a structural conversion.
    let returns = cx.fresh_place();

    Ok(HirFunction {
        loc: None,
        id: None,
        name_hint: None,
        fn_type: ReactFunctionType::Other,
        params: Vec::new(),
        return_type_annotation: None,
        returns,
        context: Vec::new(),
        body: HIR { entry, blocks },
        instructions: cx.instructions,
        generator: false,
        is_async: false,
        directives: Vec::new(),
        aliasing_effects: None,
    })
}

struct Cx<'a> {
    env: &'a mut Environment,
    /// JSLIR result `ValueId` → the React temp `IdentifierId` it became.
    value_ids: HashMap<u32, IdentifierId>,
    /// Named local variable → its shared (non-SSA) `IdentifierId`.
    vars: HashMap<String, IdentifierId>,
    /// `jsir.identifier_ref` result `ValueId` → the variable name it targets.
    /// An l-value reference emits no React instruction; the consuming
    /// `assignment_expression` reads this to build the `StoreLocal` target.
    ref_targets: HashMap<u32, String>,
    /// The flat instruction arena being built (indexed by `InstructionId`).
    instructions: Vec<Instruction>,
}

impl<'a> Cx<'a> {
    /// A fresh anonymous temp place (e.g. a function `returns` slot).
    fn fresh_place(&mut self) -> Place {
        place_of(self.env.next_identifier_id())
    }

    /// The temp place a JSLIR result `ValueId` maps to (stable per value).
    fn value_place(&mut self, v: ValueId) -> Place {
        let id = *self.value_ids.entry(v.0).or_insert_with(|| self.env.next_identifier_id());
        place_of(id)
    }

    /// The shared place for a named local (non-SSA: one identifier per name).
    fn var_place(&mut self, name: &str) -> Place {
        let id = match self.vars.get(name) {
            Some(id) => *id,
            None => {
                let id = self.env.next_identifier_id();
                self.env.identifiers[id.0 as usize].name = Some(IdentifierName::Named(name.into()));
                self.vars.insert(name.into(), id);
                id
            }
        };
        place_of(id)
    }

    /// Append an instruction whose lvalue is the temp for `result`, returning its
    /// `InstructionId` (its index in the arena).
    fn emit(&mut self, result: Option<ValueId>, value: InstructionValue) -> InstructionId {
        let lvalue = match result {
            Some(v) => self.value_place(v),
            None => self.fresh_place(),
        };
        let id = InstructionId(self.instructions.len() as u32);
        self.instructions.push(Instruction {
            id: EvaluationOrder(id.0),
            lvalue,
            value,
            loc: None,
            effects: None,
        });
        id
    }

    fn convert_block(&mut self, block: &jsir_ir::Block) -> R<BasicBlock> {
        let mut instr_ids = Vec::new();
        // The terminator is the block's last op (if it is a JSLIR terminator);
        // everything before it is a value/statement instruction.
        let (body, terminator) = split_terminator(block);
        for op in body {
            if let Some(id) = self.convert_op(op)? {
                instr_ids.push(id);
            }
        }
        let terminal = match terminator {
            Some(t) => self.convert_terminal(t)?,
            None => Terminal::Unsupported { id: EvaluationOrder(0), loc: None },
        };
        Ok(BasicBlock {
            kind: BlockKind::Block,
            id: RBlockId(block.id.0),
            instructions: instr_ids,
            terminal,
            preds: Default::default(),
            phis: Vec::new(),
        })
    }

    /// Convert one value/statement op. Returns `None` for ops that carry no
    /// instruction (e.g. the `expr_region_end` marker).
    fn convert_op(&mut self, op: &Op) -> R<Option<InstructionId>> {
        let result = op.results.first().copied();
        let value = match op.name.as_str() {
            // Statement wrappers with no value of their own: the inner expression
            // op has already emitted its instruction in the flat stream.
            "jsir.expr_region_end" | "jsir.expression_statement" => return Ok(None),

            "jsir.numeric_literal" => InstructionValue::Primitive {
                value: PrimitiveValue::Number(FloatValue::new(attr_f64(op, "value").unwrap_or(0.0))),
                loc: None,
            },
            "jsir.string_literal" => InstructionValue::Primitive {
                value: PrimitiveValue::String(attr_str(op, "value").unwrap_or_default().to_string()),
                loc: None,
            },
            "jsir.boolean_literal" => InstructionValue::Primitive {
                value: PrimitiveValue::Boolean(attr_bool(op, "value")),
                loc: None,
            },
            "jsir.null_literal" => {
                InstructionValue::Primitive { value: PrimitiveValue::Null, loc: None }
            }

            "jsir.identifier" => self.convert_identifier(op),

            // An l-value reference (`x` in `x = …`): emits no instruction; record
            // the target name for the consuming assignment.
            "jsir.identifier_ref" => {
                if let (Some(result), Some(name)) = (result, ref_name(op)) {
                    self.ref_targets.insert(result.0, name);
                }
                return Ok(None);
            }

            "jsir.assignment_expression" => {
                let op_str = attr_str(op, "operator_").unwrap_or("=");
                if op_str != "=" {
                    return Err(format!("compound assignment `{op_str}` not supported"));
                }
                let (target_v, value_v) = two_operands(op)?;
                let name = self
                    .ref_targets
                    .get(&target_v.0)
                    .cloned()
                    .ok_or("assignment target is not a simple identifier")?;
                let value = self.value_place(value_v);
                let place = self.var_place(&name);
                InstructionValue::StoreLocal {
                    lvalue: LValue { place, kind: InstructionKind::Reassign },
                    value,
                    type_annotation: None,
                    loc: None,
                }
            }

            "jsir.binary_expression" => {
                let (l, r) = two_operands(op)?;
                let left = self.value_place(l);
                let right = self.value_place(r);
                InstructionValue::BinaryExpression {
                    operator: binary_operator(attr_str(op, "operator_").unwrap_or(""))?,
                    left,
                    right,
                    loc: None,
                }
            }
            "jsir.unary_expression" => {
                let a = one_operand(op)?;
                let v = self.value_place(a);
                InstructionValue::UnaryExpression {
                    operator: unary_operator(attr_str(op, "operator_").unwrap_or(""))?,
                    value: v,
                    loc: None,
                }
            }
            "jsir.call_expression" => {
                let mut it = op.operands.iter().copied();
                let callee_v = it.next().ok_or("call_expression: no callee")?;
                let callee = self.value_place(callee_v);
                let args: Vec<PlaceOrSpread> =
                    it.map(|v| PlaceOrSpread::Place(self.value_place(v))).collect();
                InstructionValue::CallExpression { callee, args, loc: None }
            }

            n if dialect::is_store_local(op) => {
                let (name, kind) = dialect::store_local_parts(op)
                    .ok_or_else(|| format!("malformed store_local: {n}"))?;
                let stored = op
                    .operands
                    .first()
                    .copied()
                    .ok_or("store_local without init not supported")?;
                let value = self.value_place(stored);
                let place = self.var_place(name);
                InstructionValue::StoreLocal {
                    lvalue: LValue { place, kind: instruction_kind(kind) },
                    value,
                    type_annotation: None,
                    loc: None,
                }
            }

            other => return Err(format!("unsupported value op: {other}")),
        };
        Ok(Some(self.emit(result, value)))
    }

    fn convert_identifier(&mut self, op: &Op) -> InstructionValue {
        // Resolved local (has a binding scope) → LoadLocal; otherwise a free name
        // → LoadGlobal. Matches `ssa.rs`'s symbol-name keying.
        let sym = op.trivia.as_ref().and_then(|t| t.referenced_symbol.as_ref());
        match sym {
            Some(s) if s.def_scope_uid.is_some() => {
                let place = self.var_place(&s.name);
                InstructionValue::LoadLocal { place, loc: None }
            }
            _ => {
                let name = sym
                    .map(|s| s.name.clone())
                    .or_else(|| attr_str(op, "name").map(str::to_string))
                    .unwrap_or_default();
                InstructionValue::LoadGlobal {
                    binding: NonLocalBinding::Global { name },
                    loc: None,
                }
            }
        }
    }

    fn convert_terminal(&mut self, op: &Op) -> R<Terminal> {
        if op.name == dialect::RETURN {
            let value = match op.operands.first().copied() {
                Some(v) => self.value_place(v),
                // bare `return;` — a synthetic undefined place.
                None => {
                    let p = self.fresh_place();
                    self.emit(None, InstructionValue::Primitive {
                        value: PrimitiveValue::Undefined,
                        loc: None,
                    });
                    p
                }
            };
            return Ok(Terminal::Return {
                value,
                return_variant: ReturnVariant::Explicit,
                id: EvaluationOrder(0),
                loc: None,
                effects: None,
            });
        }
        if op.name == dialect::BR {
            let target = op.successors.first().ok_or("br without target")?.block;
            let variant = match dialect::loop_jump(op) {
                Some("continue") => GotoVariant::Continue,
                _ => GotoVariant::Break,
            };
            return Ok(Terminal::Goto { block: RBlockId(target.0), variant, id: EvaluationOrder(0), loc: None });
        }
        if op.name == dialect::COND_BR && dialect::is_if_header(op) {
            let test = self.value_place(op.operands.first().copied().ok_or("cond_br without test")?);
            let consequent = op.successors.first().ok_or("if: no then")?.block;
            let alternate = op.successors.get(1).ok_or("if: no else")?.block;
            let merge = dialect::merge_of(op).ok_or("if: no merge block")?;
            return Ok(Terminal::If {
                test,
                consequent: RBlockId(consequent.0),
                alternate: RBlockId(alternate.0),
                fallthrough: RBlockId(merge.0),
                id: EvaluationOrder(0),
                loc: None,
            });
        }
        Err(format!("unsupported terminator: {}", op.name))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The variable name an `identifier`/`identifier_ref` op refers to (resolved
/// symbol first, falling back to the `name` attr).
fn ref_name(op: &Op) -> Option<String> {
    op.trivia
        .as_ref()
        .and_then(|t| t.referenced_symbol.as_ref())
        .map(|s| s.name.clone())
        .or_else(|| attr_str(op, "name").map(str::to_string))
}

fn place_of(identifier: IdentifierId) -> Place {
    Place { identifier, effect: Effect::Unknown, reactive: false, loc: None }
}

/// Split a block into (non-terminator ops, terminator op?).
fn split_terminator(block: &jsir_ir::Block) -> (&[Op], Option<&Op>) {
    match block.ops.last() {
        Some(last) if dialect::is_terminator(&last.name) => {
            (&block.ops[..block.ops.len() - 1], Some(last))
        }
        _ => (&block.ops[..], None),
    }
}

fn one_operand(op: &Op) -> R<ValueId> {
    op.operands.first().copied().ok_or_else(|| format!("{}: expected 1 operand", op.name))
}
fn two_operands(op: &Op) -> R<(ValueId, ValueId)> {
    match (op.operands.first().copied(), op.operands.get(1).copied()) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err(format!("{}: expected 2 operands", op.name)),
    }
}

fn attr_str<'a>(op: &'a Op, key: &str) -> Option<&'a str> {
    op.attrs.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Attr::Str(s) => Some(s.as_str()),
        _ => None,
    })
}
fn attr_f64(op: &Op, key: &str) -> Option<f64> {
    op.attrs.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Attr::F64(n) => Some(*n),
        Attr::I64(n) => Some(*n as f64),
        _ => None,
    })
}
fn attr_bool(op: &Op, key: &str) -> bool {
    op.attrs.iter().find(|(k, _)| k == key).map_or(false, |(_, v)| matches!(v, Attr::Bool(true)))
}

fn instruction_kind(decl_kind: &str) -> InstructionKind {
    match decl_kind {
        "const" => InstructionKind::Const,
        _ => InstructionKind::Let, // var / let
    }
}

fn binary_operator(s: &str) -> R<react_compiler_hir::BinaryOperator> {
    use react_compiler_hir::BinaryOperator::*;
    Ok(match s {
        "+" => Add,
        "-" => Subtract,
        "*" => Multiply,
        "/" => Divide,
        "%" => Modulo,
        "**" => Exponent,
        "==" => Equal,
        "!=" => NotEqual,
        "===" => StrictEqual,
        "!==" => StrictNotEqual,
        "<" => LessThan,
        "<=" => LessEqual,
        ">" => GreaterThan,
        ">=" => GreaterEqual,
        "<<" => ShiftLeft,
        ">>" => ShiftRight,
        ">>>" => UnsignedShiftRight,
        "|" => BitwiseOr,
        "^" => BitwiseXor,
        "&" => BitwiseAnd,
        "in" => In,
        "instanceof" => InstanceOf,
        other => return Err(format!("unsupported binary operator: {other}")),
    })
}

fn unary_operator(s: &str) -> R<react_compiler_hir::UnaryOperator> {
    use react_compiler_hir::UnaryOperator::*;
    Ok(match s {
        "-" => Minus,
        "+" => Plus,
        "!" => Not,
        "~" => BitwiseNot,
        "typeof" => TypeOf,
        "void" => Void,
        other => return Err(format!("unsupported unary operator: {other}")),
    })
}
