//! The mutation/aliasing **effect inference pass** — `compute_signature` (per
//! CFG op) + the abstract-interpretation worklist driver. Faithful port of the
//! driver in `react_compiler_inference::infer_mutation_aliasing_effects`.
//!
//! Produces an [`EffectResults`]: the resolved [`AliasingEffect`]s for every
//! instruction (consumed by the ranges pass) plus the error effects
//! (`MutateFrozen`/`MutateGlobal`/`Impure`) the validation passes turn into
//! bail decisions.

use std::collections::HashMap;

use crate::cfg::{BlockId, Cfg, MemberKey, Op, Term, Value};
use crate::effects::{
    apply_effect, compute_effects_for_legacy_signature, hashset_of, AbstractValue, AliasingEffect,
    ApplyArg, Context, EffectErrorKind, InferenceState, MutationReason, ValueId,
};
use crate::infer_types::Types;
use crate::types::{
    FunctionSignature, ShapeRegistry, Type, ValueKind, ValueReason,
};

/// Effects recorded for one instruction.
#[derive(Debug, Clone)]
pub struct InstrEffects {
    pub block: BlockId,
    pub index: usize,
    pub result: Option<Value>,
    pub effects: Vec<AliasingEffect>,
}

/// The output of effect inference.
#[derive(Debug, Clone, Default)]
pub struct EffectResults {
    /// Per-instruction effects, in reverse-postorder program order.
    pub instrs: Vec<InstrEffects>,
    /// Terminal effects per block (e.g. the return-value freeze).
    pub terminal: Vec<(BlockId, Vec<AliasingEffect>)>,
}

impl EffectResults {
    /// All recorded effects (instruction + terminal), in order.
    pub fn all(&self) -> impl Iterator<Item = &AliasingEffect> {
        self.instrs
            .iter()
            .flat_map(|i| i.effects.iter())
            .chain(self.terminal.iter().flat_map(|(_, e)| e.iter()))
    }

    /// The disqualifying error effects (mutating a frozen/global value, impure
    /// calls). A non-empty result means React would bail / error on this
    /// function rather than memoize it.
    pub fn errors(&self) -> Vec<(Value, EffectErrorKind)> {
        self.all()
            .filter_map(|e| match e {
                AliasingEffect::Error { place, kind } => Some((*place, kind.clone())),
                _ => None,
            })
            .collect()
    }
}

/// Run effect inference over `cfg`. `is_component` selects component param
/// typing (props frozen, ref mutable). Faithful to
/// `infer_mutation_aliasing_effects`.
pub fn infer(cfg: &Cfg, types: &Types, shapes: &ShapeRegistry, is_component: bool) -> EffectResults {
    let mut ctx = Context::new();
    let def_op = build_def_op(cfg);
    let phi_operands = collect_phi_operands(cfg);

    // Initial state + param kinds. Top-level functions are not function
    // expressions, so params are frozen (reason ReactiveFunctionArgument).
    let mut initial = InferenceState::empty(false);
    let frozen_param = AbstractValue::new(ValueKind::Frozen, ValueReason::ReactiveFunctionArgument);
    if is_component {
        if let Some(p) = cfg.params.first() {
            infer_param(&mut initial, *p, frozen_param.clone());
        }
        if let Some(p) = cfg.params.get(1) {
            // The ref param is mutable.
            infer_param(&mut initial, *p, AbstractValue::new(ValueKind::Mutable, ValueReason::Other));
        }
    } else {
        for p in &cfg.params {
            infer_param(&mut initial, *p, frozen_param.clone());
        }
    }

    // Worklist over blocks (faithful to upstream queue/merge fixpoint).
    let mut queued: HashMap<BlockId, InferenceState> = HashMap::new();
    let mut states_by_block: HashMap<BlockId, InferenceState> = HashMap::new();
    queued.insert(cfg.entry, initial);

    let mut effects_by_instr: HashMap<(BlockId, usize), InstrEffects> = HashMap::new();
    let mut terminal_effects: HashMap<BlockId, Vec<AliasingEffect>> = HashMap::new();

    let block_order: Vec<BlockId> = cfg.blocks.iter().map(|b| b.id).collect();
    let mut iterations = 0;
    while !queued.is_empty() {
        iterations += 1;
        if iterations > 100 {
            break; // upstream: invariant error (effect not cached). We stop.
        }
        for &bid in &block_order {
            let incoming = match queued.remove(&bid) {
                Some(s) => s,
                None => continue,
            };
            states_by_block.insert(bid, incoming.clone());
            let mut state = incoming;

            infer_block(
                &mut ctx, &mut state, cfg, bid, types, shapes, &def_op, &phi_operands,
                &mut effects_by_instr, &mut terminal_effects,
            );

            for succ in cfg.block(bid).term.successors() {
                queue(&mut queued, &states_by_block, succ, state.clone());
            }
        }
    }

    // Flatten effects into RPO program order.
    let order = crate::ssa::reverse_postorder(cfg);
    let mut instrs = Vec::new();
    for &bid in &order {
        let block = cfg.block(bid);
        for idx in 0..block.instrs.len() {
            if let Some(ie) = effects_by_instr.remove(&(bid, idx)) {
                instrs.push(ie);
            }
        }
    }
    let mut terminal: Vec<(BlockId, Vec<AliasingEffect>)> = Vec::new();
    for &bid in &order {
        if let Some(e) = terminal_effects.remove(&bid) {
            terminal.push((bid, e));
        }
    }

    EffectResults { instrs, terminal }
}

/// Convenience: build the default shapes/globals, run type inference, and infer
/// effects. Used by the compile path's validation bail.
pub fn analyze_default(cfg: &Cfg, is_component: bool) -> EffectResults {
    let mut shapes = crate::types::build_builtin_shapes();
    let globals = crate::types::build_default_globals(&mut shapes);
    let types = crate::infer_types::infer(cfg, &shapes, &globals, is_component, Default::default());
    infer(cfg, &types, &shapes, is_component)
}

fn infer_param(state: &mut InferenceState, place: Value, kind: AbstractValue) {
    let vid = ValueId::new();
    state.initialize(vid, kind);
    state.define(place, vid);
}

/// Merge `state` into the queue for `block_id`. Faithful to upstream `queue`.
fn queue(
    queued: &mut HashMap<BlockId, InferenceState>,
    states_by_block: &HashMap<BlockId, InferenceState>,
    block_id: BlockId,
    state: InferenceState,
) {
    if let Some(queued_state) = queued.get(&block_id) {
        let merged = queued_state.merge(&state).unwrap_or_else(|| queued_state.clone());
        queued.insert(block_id, merged);
    } else if let Some(prev) = states_by_block.get(&block_id) {
        if let Some(next) = prev.merge(&state) {
            queued.insert(block_id, next);
        }
    } else {
        queued.insert(block_id, state);
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_block(
    ctx: &mut Context,
    state: &mut InferenceState,
    cfg: &Cfg,
    bid: BlockId,
    types: &Types,
    shapes: &ShapeRegistry,
    def_op: &HashMap<Value, Op>,
    phi_operands: &HashMap<(BlockId, usize), Vec<Value>>,
    effects_by_instr: &mut HashMap<(BlockId, usize), InstrEffects>,
    terminal_effects: &mut HashMap<BlockId, Vec<AliasingEffect>>,
) {
    let block = cfg.block(bid);

    // Phis.
    for (i, param) in block.params.iter().enumerate() {
        if let Some(ops) = phi_operands.get(&(bid, i)) {
            state.infer_phi(*param, ops);
        }
    }

    // Instructions.
    for (idx, ins) in block.instrs.iter().enumerate() {
        let sig_effects = compute_signature(ins, types, shapes, def_op);
        let mut initialized = std::collections::HashSet::new();
        let mut out = Vec::new();
        for e in sig_effects {
            apply_effect(ctx, state, types, e, &mut initialized, &mut out);
        }
        // Fallback: ensure the result is defined (upstream invariant).
        if let Some(r) = ins.result {
            if !state.is_defined(r) {
                let vid = ValueId::stable_for(r);
                state.initialize(vid, AbstractValue::mutable_other());
                state.define(r, vid);
            }
        }
        effects_by_instr.insert((bid, idx), InstrEffects { block: bid, index: idx, result: ins.result, effects: out });
    }

    // Terminal: the returned value is frozen (JsxCaptured) for non-function-exprs.
    if let Term::Ret(Some(v)) = &block.term {
        if !state.is_function_expression {
            let mut initialized = std::collections::HashSet::new();
            let mut out = Vec::new();
            apply_effect(ctx, state, types, AliasingEffect::Freeze { value: *v, reason: ValueReason::JsxCaptured }, &mut initialized, &mut out);
            terminal_effects.insert(bid, out);
        }
    }
}

// =============================================================================
// compute_signature — map a CFG op to its aliasing effects.
// Faithful port of `compute_signature_for_instruction` for our op set.
// =============================================================================

fn compute_signature(
    ins: &crate::cfg::Instr,
    types: &Types,
    shapes: &ShapeRegistry,
    def_op: &HashMap<Value, Op>,
) -> Vec<AliasingEffect> {
    let result = ins.result;
    // For ops without a result (e.g. a bare WriteVar — shouldn't occur post-SSA)
    // there is nothing to bind; emit no effects.
    let lvalue = match result {
        Some(r) => r,
        None => match &ins.op {
            // StoreMember can have no result; we still emit its mutation below.
            Op::StoreMember { .. } => Value(u32::MAX), // sentinel; Create skipped below
            _ => return Vec::new(),
        },
    };
    let mut effects: Vec<AliasingEffect> = Vec::new();

    match &ins.op {
        Op::Const(_) | Op::Bin(_, _, _) | Op::Un(_, _) => {
            effects.push(AliasingEffect::Create { into: lvalue, value: ValueKind::Primitive, reason: ValueReason::Other });
        }

        Op::Global(_) => {
            effects.push(AliasingEffect::Create { into: lvalue, value: ValueKind::Global, reason: ValueReason::Global });
        }

        Op::MakeArray(elems) => {
            effects.push(AliasingEffect::Create { into: lvalue, value: ValueKind::Mutable, reason: ValueReason::Other });
            for e in elems {
                effects.push(AliasingEffect::Capture { from: *e, into: lvalue });
            }
        }

        Op::MakeObject(props) => {
            effects.push(AliasingEffect::Create { into: lvalue, value: ValueKind::Mutable, reason: ValueReason::Other });
            for (_k, v) in props {
                effects.push(AliasingEffect::Capture { from: *v, into: lvalue });
            }
        }

        Op::Member { obj, .. } => {
            // PropertyLoad/ComputedLoad: primitive result -> Create Primitive,
            // else CreateFrom(object).
            if matches!(types.get(lvalue), Type::Primitive) {
                effects.push(AliasingEffect::Create { into: lvalue, value: ValueKind::Primitive, reason: ValueReason::Other });
            } else {
                effects.push(AliasingEffect::CreateFrom { from: *obj, into: lvalue });
            }
        }

        Op::StoreMember { obj, prop, value } => {
            // Mutate the object; AssignCurrentProperty when storing `.current`
            // on an object whose type is still unresolved (ref-write heuristic).
            let reason = match prop {
                MemberKey::Static(name) if name == "current" && type_is_unresolved(types.get(*obj)) => {
                    Some(MutationReason::AssignCurrentProperty)
                }
                _ => None,
            };
            effects.push(AliasingEffect::Mutate { value: *obj, reason });
            effects.push(AliasingEffect::Capture { from: *value, into: *obj });
            if let Some(r) = result {
                effects.push(AliasingEffect::Create { into: r, value: ValueKind::Primitive, reason: ValueReason::Other });
            }
        }

        Op::Call { callee, args } => {
            // Reconstruct method calls: if the callee is a member load, the
            // receiver is its base object and the call doesn't mutate the
            // function (upstream MethodCall). Otherwise it's a plain call.
            let (receiver, mutates_function) = match def_op.get(callee) {
                Some(Op::Member { obj, .. }) => (*obj, false),
                _ => (*callee, true),
            };
            let apply_args: Vec<ApplyArg> = args.iter().map(|a| ApplyArg::Place(*a)).collect();
            let sig = signature_of(types, shapes, *callee);
            match sig {
                Some(sig) => {
                    // Signature path -> legacy effect computation (hooks freeze
                    // args, etc.). `lvalue` is the call result.
                    effects.extend(compute_effects_for_legacy_signature(&sig, lvalue, receiver, &apply_args));
                }
                None => {
                    effects.push(AliasingEffect::Apply { receiver, function: *callee, mutates_function, args: apply_args, into: lvalue });
                }
            }
        }

        Op::ReadVar(_) | Op::WriteVar(_, _) => {
            // Pre-SSA forms — eliminated by SSA. No effects.
        }
    }

    effects
}

/// Whether a type is still an unresolved variable / polymorphic (used for the
/// ref-write `.current` heuristic, mirroring upstream's `Type::TypeVar` check).
fn type_is_unresolved(ty: &Type) -> bool {
    matches!(ty, Type::TypeVar { .. } | Type::Poly)
}

/// Look up a callee's [`FunctionSignature`] from its resolved type's shape.
/// Faithful to upstream `get_function_call_signature`.
fn signature_of(types: &Types, shapes: &ShapeRegistry, callee: Value) -> Option<FunctionSignature> {
    match types.get(callee) {
        Type::Function { shape_id: Some(id), .. } => {
            shapes.get(id).and_then(|s| s.function_type.clone())
        }
        _ => None,
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn build_def_op(cfg: &Cfg) -> HashMap<Value, Op> {
    let mut m = HashMap::new();
    for b in &cfg.blocks {
        for ins in &b.instrs {
            if let Some(r) = ins.result {
                m.insert(r, ins.op.clone());
            }
        }
    }
    m
}

fn collect_phi_operands(cfg: &Cfg) -> HashMap<(BlockId, usize), Vec<Value>> {
    let mut out: HashMap<(BlockId, usize), Vec<Value>> = HashMap::new();
    let mut push = |target: BlockId, args: &[Value]| {
        for (i, a) in args.iter().enumerate() {
            out.entry((target, i)).or_default().push(*a);
        }
    };
    for b in &cfg.blocks {
        match &b.term {
            Term::Br(t, a) => push(*t, a),
            Term::CondBr { then_block, then_args, else_block, else_args, .. } => {
                push(*then_block, then_args);
                push(*else_block, else_args);
            }
            Term::Ret(_) | Term::Unreachable => {}
        }
    }
    out
}

// Silence an unused import in some build configs.
#[allow(unused_imports)]
use crate::effects::hashset_of as _hashset_of;
#[allow(dead_code)]
fn _hashset_helper() -> std::collections::HashSet<ValueReason> {
    hashset_of(ValueReason::Other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{build_builtin_shapes, build_default_globals};

    fn setup(src: &str, is_component: bool) -> (Cfg, EffectResults) {
        let mut shapes = build_builtin_shapes();
        let globals = build_default_globals(&mut shapes);
        let cfg = crate::compile_ssa(src).expect("compile");
        let types = crate::infer_types::infer(&cfg, &shapes, &globals, is_component, Default::default());
        let res = infer(&cfg, &types, &shapes, is_component);
        (cfg, res)
    }

    #[test]
    fn mutating_a_global_is_an_error() {
        // Reassigning / mutating a global object during render is disqualifying.
        let (_, res) = setup("function f() { const x = globalThing; x.y = 1; return x; }", false);
        // globalThing is Global; x aliases it; x.y = 1 mutates a global.
        assert!(
            res.errors().iter().any(|(_, k)| *k == EffectErrorKind::MutateGlobal),
            "expected a MutateGlobal error, got {:?}", res.errors()
        );
    }

    #[test]
    fn mutating_frozen_props_is_an_error() {
        // props is frozen; mutating props.x during render is disqualifying.
        let (_, res) = setup("function Foo(props) { props.x = 1; return props; }", true);
        assert!(
            res.errors().iter().any(|(_, k)| *k == EffectErrorKind::MutateFrozen),
            "expected a MutateFrozen error, got {:?}", res.errors()
        );
    }

    #[test]
    fn local_object_mutation_is_fine() {
        // Mutating a locally-created object is allowed (no error).
        let (_, res) = setup("function f() { const x = {}; x.y = 1; return x; }", false);
        assert!(res.errors().is_empty(), "local mutation should be fine, got {:?}", res.errors());
    }
}
