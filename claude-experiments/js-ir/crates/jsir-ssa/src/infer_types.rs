//! Type inference pass — a faithful port of upstream
//! `react_compiler_typeinference::infer_types`.
//!
//! Generates type equations from the CFG, unifies them, and resolves a concrete
//! [`Type`] for every SSA [`Value`]. Upstream keys types off `Identifier`s (many
//! SSA values can share one identifier, merged by phis); our CFG is already SSA,
//! so we key a type slot off each `Value` directly and treat block parameters as
//! phis (their operands are the values branch-passed by each predecessor).
//!
//! Source of truth: `infer_types.rs` in `react_compiler_typeinference`. The
//! `generate`/`Unifier`/`apply` structure and the unification rules (phi-union,
//! occurs-check, property resolution via the shapes registry, ref-like-name
//! handling) mirror it op-for-op where our op set overlaps.

use std::collections::HashMap;

use crate::cfg::{BinOp, Block, BlockId, Cfg, MemberKey, Op, Term, Value};
use crate::types::{
    is_hook_name, GlobalRegistry, ObjectShape, PropertyLiteral, PropertyNameKind, ShapeRegistry,
    Type, TypeId, BUILT_IN_ARRAY_ID, BUILT_IN_MIXED_READONLY_ID, BUILT_IN_OBJECT_ID,
    BUILT_IN_PROPS_ID, BUILT_IN_REF_VALUE_ID, BUILT_IN_SET_STATE_ID, BUILT_IN_USE_REF_ID,
};

/// Inference result: a resolved [`Type`] per [`Value`].
pub struct Types {
    /// Type arena. `types[slot.0]` holds the resolved type for that slot.
    types: Vec<Type>,
    /// Each value's type slot.
    slot: HashMap<Value, TypeId>,
}

impl Types {
    /// The resolved type of `v` (defaults to `Poly` for values never constrained).
    pub fn get(&self, v: Value) -> &Type {
        match self.slot.get(&v) {
            Some(id) => &self.types[id.0 as usize],
            None => &Type::Poly,
        }
    }

    /// Whether `v` resolved to the given builtin object shape.
    pub fn is_object_shape(&self, v: Value, shape_id: &str) -> bool {
        matches!(self.get(v), Type::Object { shape_id: Some(s) } if s == shape_id)
    }

    /// Whether `v` is a ref value (the thing `someRef.current` resolves to).
    pub fn is_ref_value(&self, v: Value) -> bool {
        self.is_object_shape(v, BUILT_IN_REF_VALUE_ID)
    }

    /// Whether `v` is a `useRef()` container object.
    pub fn is_use_ref(&self, v: Value) -> bool {
        self.is_object_shape(v, BUILT_IN_USE_REF_ID)
    }
}

/// Config flags mirroring the upstream environment flags that influence type
/// inference. Defaults match upstream `EnvironmentConfig::default()` (off).
#[derive(Debug, Clone, Copy, Default)]
pub struct InferConfig {
    pub enable_treat_ref_like_identifiers_as_refs: bool,
    pub enable_treat_set_identifiers_as_state_setters: bool,
}

/// Run type inference over `cfg`. `is_component` selects whether the first two
/// params are typed as props / ref (upstream keys this off `func.fn_type`).
pub fn infer(
    cfg: &Cfg,
    shapes: &ShapeRegistry,
    globals: &GlobalRegistry,
    is_component: bool,
    config: InferConfig,
) -> Types {
    let mut ctx = Ctx::new(cfg, config);
    let mut unifier = Unifier::new(config);
    ctx.generate(cfg, shapes, globals, is_component, &mut unifier);
    ctx.apply(&unifier);
    Types { types: ctx.types, slot: ctx.slot }
}

// =============================================================================
// Context: type slots + name tracking
// =============================================================================

struct Ctx {
    types: Vec<Type>,
    slot: HashMap<Value, TypeId>,
    /// Source names for values, used for ref-like-name detection. Seeded from
    /// params and globals (upstream propagates more through LoadLocal copies,
    /// which our SSA form has already eliminated).
    names: HashMap<Value, String>,
}

impl Ctx {
    fn new(cfg: &Cfg, _config: InferConfig) -> Ctx {
        let mut ctx = Ctx { types: Vec::new(), slot: HashMap::new(), names: HashMap::new() };
        // Seed param names.
        for (p, name) in cfg.params.iter().zip(&cfg.param_names) {
            ctx.names.insert(*p, name.clone());
        }
        ctx
    }

    /// `v`'s type as a `TypeVar` referencing its slot, allocating a slot the
    /// first time. Mirrors upstream `get_type`.
    fn get_type(&mut self, v: Value) -> Type {
        if let Some(id) = self.slot.get(&v) {
            return Type::TypeVar { id: *id };
        }
        let id = TypeId(self.types.len() as u32);
        self.types.push(Type::TypeVar { id });
        self.slot.insert(v, id);
        Type::TypeVar { id }
    }

    /// Allocate a fresh anonymous TypeVar (for call/new return types). Mirrors
    /// upstream `make_type`.
    fn make_type(&mut self) -> Type {
        let id = TypeId(self.types.len() as u32);
        self.types.push(Type::TypeVar { id });
        Type::TypeVar { id }
    }

    fn name_of(&self, v: Value) -> String {
        self.names.get(&v).cloned().unwrap_or_default()
    }
}

// =============================================================================
// Generate equations
// =============================================================================

impl Ctx {
    fn generate(
        &mut self,
        cfg: &Cfg,
        shapes: &ShapeRegistry,
        globals: &GlobalRegistry,
        is_component: bool,
        unifier: &mut Unifier,
    ) {
        // Component params: first = props, second = ref.
        if is_component {
            if let Some(p) = cfg.params.first() {
                let ty = self.get_type(*p);
                unifier.unify(ty, Type::Object { shape_id: Some(BUILT_IN_PROPS_ID.to_string()) }, shapes, &mut self.types);
            }
            if let Some(p) = cfg.params.get(1) {
                let ty = self.get_type(*p);
                unifier.unify(ty, Type::Object { shape_id: Some(BUILT_IN_USE_REF_ID.to_string()) }, shapes, &mut self.types);
            }
        }

        // Phi operands: for each block param, gather the value passed for that
        // param index by every predecessor edge.
        let phi_operands = collect_phi_operands(cfg);

        let order = crate::ssa::reverse_postorder(cfg);
        let mut return_types: Vec<Type> = Vec::new();

        for &bid in &order {
            let block = cfg.block(bid);

            // Phis (block parameters).
            for (i, param) in block.params.iter().enumerate() {
                let left = self.get_type(*param);
                let operands: Vec<Type> = phi_operands
                    .get(&(bid, i))
                    .map(|vs| vs.iter().map(|v| self.get_type(*v)).collect())
                    .unwrap_or_default();
                if !operands.is_empty() {
                    unifier.unify(left, Type::Phi { operands }, shapes, &mut self.types);
                }
            }

            for ins in &block.instrs {
                self.generate_instr(ins, shapes, globals, unifier);
            }

            if let Term::Ret(Some(v)) = &block.term {
                return_types.push(self.get_type(*v));
            }
        }

        // Unify return types into a single synthetic returns var (kept for
        // faithfulness / future nested-function support).
        if return_types.len() > 1 {
            let returns = self.make_type();
            unifier.unify(returns, Type::Phi { operands: return_types }, shapes, &mut self.types);
        }
    }

    fn generate_instr(
        &mut self,
        ins: &crate::cfg::Instr,
        shapes: &ShapeRegistry,
        globals: &GlobalRegistry,
        unifier: &mut Unifier,
    ) {
        let result = ins.result;
        // `left` is only meaningful when the instruction has a result.
        match &ins.op {
            Op::Const(_) => {
                if let Some(r) = result {
                    let left = self.get_type(r);
                    unifier.unify(left, Type::Primitive, shapes, &mut self.types);
                }
            }

            Op::Bin(op, a, b) => {
                if is_primitive_binary_op(*op) {
                    let at = self.get_type(*a);
                    unifier.unify(at, Type::Primitive, shapes, &mut self.types);
                    let bt = self.get_type(*b);
                    unifier.unify(bt, Type::Primitive, shapes, &mut self.types);
                }
                if let Some(r) = result {
                    let left = self.get_type(r);
                    unifier.unify(left, Type::Primitive, shapes, &mut self.types);
                }
            }

            Op::Un(_, _) => {
                if let Some(r) = result {
                    let left = self.get_type(r);
                    unifier.unify(left, Type::Primitive, shapes, &mut self.types);
                }
            }

            Op::Global(name) => {
                if let Some(r) = result {
                    self.names.insert(r, name.clone());
                    let left = self.get_type(r);
                    if let Some(global_ty) = globals.get(name).cloned() {
                        unifier.unify(left, global_ty, shapes, &mut self.types);
                    } else if is_hook_name(name) {
                        // No custom_hook_type configured; upstream would bind the
                        // custom hook type here. Leave unconstrained for now.
                    }
                }
            }

            Op::Call { callee, .. } => {
                if let Some(r) = result {
                    let return_type = self.make_type();
                    let mut shape_id = None;
                    if unifier.config.enable_treat_set_identifiers_as_state_setters {
                        let name = self.name_of(*callee);
                        if name.starts_with("set") {
                            shape_id = Some(BUILT_IN_SET_STATE_ID.to_string());
                        }
                    }
                    let callee_type = self.get_type(*callee);
                    unifier.unify(
                        callee_type,
                        Type::Function { shape_id, return_type: Box::new(return_type.clone()), is_constructor: false },
                        shapes,
                        &mut self.types,
                    );
                    let left = self.get_type(r);
                    unifier.unify(left, return_type, shapes, &mut self.types);
                }
            }

            Op::Member { obj, prop } => {
                if let Some(r) = result {
                    let object_type = self.get_type(*obj);
                    let object_name = self.name_of(*obj);
                    let property_name = match prop {
                        MemberKey::Static(name) => {
                            PropertyNameKind::Literal { value: PropertyLiteral::String(name.clone()) }
                        }
                        MemberKey::Computed(c) => {
                            let prop_type = self.get_type(*c);
                            PropertyNameKind::Computed { value: Box::new(prop_type) }
                        }
                    };
                    let left = self.get_type(r);
                    unifier.unify(
                        left,
                        Type::Property { object_type: Box::new(object_type), object_name, property_name },
                        shapes,
                        &mut self.types,
                    );
                }
            }

            Op::StoreMember { obj, prop, .. } => {
                // Mirrors upstream PropertyStore: constrain a dummy via the
                // property type (only for static keys; ComputedStore has no
                // type equation upstream).
                if let MemberKey::Static(name) = prop {
                    let dummy = self.make_type();
                    let object_type = self.get_type(*obj);
                    let object_name = self.name_of(*obj);
                    unifier.unify(
                        dummy,
                        Type::Property {
                            object_type: Box::new(object_type),
                            object_name,
                            property_name: PropertyNameKind::Literal { value: PropertyLiteral::String(name.clone()) },
                        },
                        shapes,
                        &mut self.types,
                    );
                }
            }

            Op::MakeObject(props) => {
                // Computed keys are primitive.
                for (k, _) in props {
                    if let crate::cfg::PropKey::Computed(c) = k {
                        let kt = self.get_type(*c);
                        unifier.unify(kt, Type::Primitive, shapes, &mut self.types);
                    }
                }
                if let Some(r) = result {
                    let left = self.get_type(r);
                    unifier.unify(left, Type::Object { shape_id: Some(BUILT_IN_OBJECT_ID.to_string()) }, shapes, &mut self.types);
                }
            }

            Op::MakeArray(_) => {
                if let Some(r) = result {
                    let left = self.get_type(r);
                    unifier.unify(left, Type::Object { shape_id: Some(BUILT_IN_ARRAY_ID.to_string()) }, shapes, &mut self.types);
                }
            }

            // Pre-SSA forms — eliminated by SSA construction. No equations.
            Op::ReadVar(_) | Op::WriteVar(_, _) => {}
        }
    }
}

/// `(block, param_index) -> values passed for that phi by predecessors`.
fn collect_phi_operands(cfg: &Cfg) -> HashMap<(BlockId, usize), Vec<Value>> {
    let mut out: HashMap<(BlockId, usize), Vec<Value>> = HashMap::new();
    for b in &cfg.blocks {
        match &b.term {
            Term::Br(target, args) => push_edge(&mut out, *target, args),
            Term::CondBr { then_block, then_args, else_block, else_args, .. } => {
                push_edge(&mut out, *then_block, then_args);
                push_edge(&mut out, *else_block, else_args);
            }
            Term::Ret(_) | Term::Unreachable => {}
        }
    }
    out
}

fn push_edge(out: &mut HashMap<(BlockId, usize), Vec<Value>>, target: BlockId, args: &[Value]) {
    for (i, a) in args.iter().enumerate() {
        out.entry((target, i)).or_default().push(*a);
    }
}

// =============================================================================
// Apply resolved types
// =============================================================================

impl Ctx {
    fn apply(&mut self, unifier: &Unifier) {
        // Resolve every allocated slot to its final type.
        let ids: Vec<TypeId> = self.slot.values().copied().collect();
        for id in ids {
            let current = self.types[id.0 as usize].clone();
            let resolved = unifier.get(&current);
            self.types[id.0 as usize] = resolved;
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Faithful to upstream `is_primitive_binary_op`: arithmetic / bitwise / shift /
/// relational ops force primitive operands. Notably **excludes equality** ops.
fn is_primitive_binary_op(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Mod
            | BinOp::Pow
            | BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
            | BinOp::Shl
            | BinOp::Shr
            | BinOp::UShr
            | BinOp::Lt
            | BinOp::Le
            | BinOp::Gt
            | BinOp::Ge
    )
}

fn type_equals(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::TypeVar { id: ia }, Type::TypeVar { id: ib }) => ia == ib,
        (Type::Primitive, Type::Primitive) => true,
        (Type::Poly, Type::Poly) => true,
        (Type::ObjectMethod, Type::ObjectMethod) => true,
        (Type::Object { shape_id: sa }, Type::Object { shape_id: sb }) => sa == sb,
        // Function equality compares only return types (faithful to upstream
        // `funcTypeEquals`, which ignores shape_id / is_constructor).
        (Type::Function { return_type: ra, .. }, Type::Function { return_type: rb, .. }) => {
            type_equals(ra, rb)
        }
        _ => false,
    }
}

/// `someRef.current` ref-like detection. Faithful to upstream `is_ref_like_name`:
/// property is `current`, object name is `ref` or matches `/^[A-Za-z$_]...Ref$/`.
fn is_ref_like_name(object_name: &str, property_name: &PropertyNameKind) -> bool {
    let is_current = matches!(
        property_name,
        PropertyNameKind::Literal { value: PropertyLiteral::String(s) } if s == "current"
    );
    if !is_current {
        return false;
    }
    object_name == "ref"
        || (object_name.len() > 3
            && object_name.ends_with("Ref")
            && object_name
                .as_bytes()
                .first()
                .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'$' || *c == b'_'))
}

/// Resolve a property type against the shapes registry. Faithful to upstream
/// `resolve_property_type` (no custom-hook fallback configured yet).
fn resolve_property_type(
    shapes: &ShapeRegistry,
    resolved_object: &Type,
    property_name: &PropertyNameKind,
) -> Option<Type> {
    let shape_id = match resolved_object {
        Type::Object { shape_id } | Type::Function { shape_id, .. } => shape_id.as_deref()?,
        _ => return None,
    };
    let shape: &ObjectShape = shapes.get(shape_id)?;
    match property_name {
        PropertyNameKind::Literal { value } => match value {
            PropertyLiteral::String(s) => shape
                .properties
                .get(s.as_str())
                .or_else(|| shape.properties.get("*"))
                .cloned(),
            PropertyLiteral::Number(_) => shape.properties.get("*").cloned(),
        },
        PropertyNameKind::Computed { .. } => shape.properties.get("*").cloned(),
    }
}

/// Union of a MixedReadonly with another type. Faithful to `try_union_types`.
fn try_union_types(ty1: &Type, ty2: &Type) -> Option<Type> {
    let is_mixed = |t: &Type| matches!(t, Type::Object { shape_id } if shape_id.as_deref() == Some(BUILT_IN_MIXED_READONLY_ID));
    let (readonly_type, other_type) = if is_mixed(ty1) {
        (ty1, ty2)
    } else if is_mixed(ty2) {
        (ty2, ty1)
    } else {
        return None;
    };
    if matches!(other_type, Type::Primitive) {
        return Some(readonly_type.clone());
    }
    if matches!(other_type, Type::Object { shape_id } if shape_id.as_deref() == Some(BUILT_IN_ARRAY_ID)) {
        return Some(other_type.clone());
    }
    None
}

// =============================================================================
// Unifier (faithful port)
// =============================================================================

struct Unifier {
    substitutions: HashMap<TypeId, Type>,
    config: InferConfig,
}

impl Unifier {
    fn new(config: InferConfig) -> Unifier {
        Unifier { substitutions: HashMap::new(), config }
    }

    fn unify(&mut self, t_a: Type, t_b: Type, shapes: &ShapeRegistry, types: &mut Vec<Type>) {
        // Note: upstream propagates a Result for invariant violations (cycles,
        // empty phi). Those are internal-compiler-error paths; here we treat
        // them as no-ops (leave the var unbound) rather than abort the whole
        // compile, since type inference is advisory for our validation passes.
        self.unify_impl(t_a, t_b, shapes, types);
    }

    fn unify_impl(&mut self, t_a: Type, t_b: Type, shapes: &ShapeRegistry, types: &mut Vec<Type>) {
        // Property on the RHS: ref-like-name handling, then shape resolution.
        if let Type::Property { object_type, object_name, property_name } = &t_b {
            if self.config.enable_treat_ref_like_identifiers_as_refs
                && is_ref_like_name(object_name, property_name)
            {
                self.unify_impl(
                    (**object_type).clone(),
                    Type::Object { shape_id: Some(BUILT_IN_USE_REF_ID.to_string()) },
                    shapes,
                    types,
                );
                self.unify_impl(
                    t_a,
                    Type::Object { shape_id: Some(BUILT_IN_REF_VALUE_ID.to_string()) },
                    shapes,
                    types,
                );
                return;
            }
            let resolved_object = self.get(object_type);
            if let Some(property_type) = resolve_property_type(shapes, &resolved_object, property_name) {
                self.unify_impl(t_a, property_type, shapes, types);
            }
            return;
        }

        if type_equals(&t_a, &t_b) {
            return;
        }

        if let Type::TypeVar { .. } = &t_a {
            self.bind_variable_to(t_a, t_b, shapes, types);
            return;
        }
        if let Type::TypeVar { .. } = &t_b {
            self.bind_variable_to(t_b, t_a, shapes, types);
            return;
        }

        if let (
            Type::Function { return_type: ret_a, is_constructor: con_a, .. },
            Type::Function { return_type: ret_b, is_constructor: con_b, .. },
        ) = (&t_a, &t_b)
        {
            if con_a == con_b {
                self.unify_impl((**ret_a).clone(), (**ret_b).clone(), shapes, types);
            }
        }
    }

    fn bind_variable_to(&mut self, v: Type, ty: Type, shapes: &ShapeRegistry, types: &mut Vec<Type>) {
        let v_id = match &v {
            Type::TypeVar { id } => *id,
            _ => return,
        };

        if let Type::Poly = &ty {
            return; // ignore PolyType
        }

        if let Some(existing) = self.substitutions.get(&v_id).cloned() {
            self.unify_impl(existing, ty, shapes, types);
            return;
        }

        if let Type::TypeVar { id: ty_id } = &ty {
            if let Some(existing) = self.substitutions.get(ty_id).cloned() {
                self.unify_impl(v, existing, shapes, types);
                return;
            }
        }

        if let Type::Phi { operands } = &ty {
            if operands.is_empty() {
                return; // upstream: invariant error; we no-op
            }
            let mut candidate_type: Option<Type> = None;
            for operand in operands {
                let resolved = self.get(operand);
                match &candidate_type {
                    None => candidate_type = Some(resolved),
                    Some(candidate) => {
                        if !type_equals(&resolved, candidate) {
                            match try_union_types(&resolved, candidate) {
                                Some(union) => candidate_type = Some(union),
                                None => {
                                    candidate_type = None;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if let Some(candidate) = candidate_type {
                self.unify_impl(v, candidate, shapes, types);
                return;
            }
        }

        if self.occurs_check(&v, &ty) {
            if let Some(resolved) = self.try_resolve_type(&v, &ty) {
                self.substitutions.insert(v_id, resolved);
            }
            // else: upstream errors "cycle detected"; we drop the binding.
            return;
        }

        self.substitutions.insert(v_id, ty);
    }

    fn try_resolve_type(&mut self, v: &Type, ty: &Type) -> Option<Type> {
        match ty {
            Type::Phi { operands } => {
                let mut new_operands = Vec::new();
                for operand in operands {
                    if let (Type::TypeVar { id }, Type::TypeVar { id: v_id }) = (operand, v) {
                        if id == v_id {
                            continue; // skip self-reference
                        }
                    }
                    let resolved = self.try_resolve_type(v, operand)?;
                    new_operands.push(resolved);
                }
                Some(Type::Phi { operands: new_operands })
            }
            Type::TypeVar { id } => {
                let substitution = self.get(ty);
                if !type_equals(&substitution, ty) {
                    let resolved = self.try_resolve_type(v, &substitution)?;
                    self.substitutions.insert(*id, resolved.clone());
                    Some(resolved)
                } else {
                    Some(ty.clone())
                }
            }
            Type::Property { object_type, object_name, property_name } => {
                let resolved_obj = self.get(object_type);
                let object_type = self.try_resolve_type(v, &resolved_obj)?;
                Some(Type::Property {
                    object_type: Box::new(object_type),
                    object_name: object_name.clone(),
                    property_name: property_name.clone(),
                })
            }
            Type::Function { shape_id, return_type, is_constructor } => {
                let resolved_ret = self.get(return_type);
                let return_type = self.try_resolve_type(v, &resolved_ret)?;
                Some(Type::Function {
                    shape_id: shape_id.clone(),
                    return_type: Box::new(return_type),
                    is_constructor: *is_constructor,
                })
            }
            Type::ObjectMethod | Type::Object { .. } | Type::Primitive | Type::Poly => Some(ty.clone()),
        }
    }

    fn occurs_check(&self, v: &Type, ty: &Type) -> bool {
        if type_equals(v, ty) {
            return true;
        }
        if let Type::TypeVar { id } = ty {
            if let Some(sub) = self.substitutions.get(id) {
                return self.occurs_check(v, sub);
            }
        }
        if let Type::Phi { operands } = ty {
            return operands.iter().any(|o| self.occurs_check(v, o));
        }
        if let Type::Function { return_type, .. } = ty {
            return self.occurs_check(v, return_type);
        }
        false
    }

    /// Resolve a type to its substituted form (following var chains, mapping
    /// through phi/function structure). Faithful to upstream `get`.
    fn get(&self, ty: &Type) -> Type {
        if let Type::TypeVar { id } = ty {
            if let Some(sub) = self.substitutions.get(id) {
                return self.get(sub);
            }
        }
        if let Type::Phi { operands } = ty {
            return Type::Phi { operands: operands.iter().map(|o| self.get(o)).collect() };
        }
        if let Type::Function { is_constructor, shape_id, return_type } = ty {
            return Type::Function {
                is_constructor: *is_constructor,
                shape_id: shape_id.clone(),
                return_type: Box::new(self.get(return_type)),
            };
        }
        ty.clone()
    }
}

// Quiet unused warnings for fields read only via pattern matches in tests/future
// passes.
#[allow(dead_code)]
fn _assert_block_used(_: &Block) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{build_builtin_shapes, build_default_globals};

    fn setup() -> (ShapeRegistry, GlobalRegistry) {
        let mut shapes = build_builtin_shapes();
        let globals = build_default_globals(&mut shapes);
        (shapes, globals)
    }

    fn infer_src(src: &str, is_component: bool, config: InferConfig) -> (Cfg, Types) {
        let (shapes, globals) = setup();
        let cfg = crate::compile_ssa(src).expect("compile");
        let types = infer(&cfg, &shapes, &globals, is_component, config);
        (cfg, types)
    }

    /// Find the result value of the first `Member(obj, .name)` static load.
    fn first_member(cfg: &Cfg, name: &str) -> Value {
        for b in &cfg.blocks {
            for ins in &b.instrs {
                if let Op::Member { prop: MemberKey::Static(s), .. } = &ins.op {
                    if s == name {
                        return ins.result.expect("member has result");
                    }
                }
            }
        }
        panic!("no member .{name}");
    }

    #[test]
    fn const_is_primitive() {
        let (cfg, types) = infer_src("function f() { const x = 1; return x; }", false, InferConfig::default());
        // The constant value resolves to Primitive.
        let mut found = false;
        for b in &cfg.blocks {
            for ins in &b.instrs {
                if matches!(ins.op, Op::Const(_)) {
                    assert!(matches!(types.get(ins.result.unwrap()), Type::Primitive));
                    found = true;
                }
            }
        }
        assert!(found, "expected a const instruction");
    }

    #[test]
    fn array_literal_has_array_shape() {
        let (cfg, types) = infer_src("function f() { const a = [1, 2]; return a; }", false, InferConfig::default());
        let mut found = false;
        for b in &cfg.blocks {
            for ins in &b.instrs {
                if matches!(ins.op, Op::MakeArray(_)) {
                    assert!(types.is_object_shape(ins.result.unwrap(), BUILT_IN_ARRAY_ID));
                    found = true;
                }
            }
        }
        assert!(found, "expected an array literal");
    }

    #[test]
    fn use_ref_dot_current_is_ref_value() {
        // useRef() -> UseRef object; .current -> RefValue.
        let (cfg, types) = infer_src(
            "function f() { const r = useRef(null); const c = r.current; return c; }",
            false,
            InferConfig::default(),
        );
        let current = first_member(&cfg, "current");
        assert!(types.is_ref_value(current), "r.current should be a ref value, got {:?}", types.get(current));
    }

    #[test]
    fn ref_value_member_stays_ref_value() {
        // r.current.foo should remain a RefValue (self-referencing `*`).
        let (cfg, types) = infer_src(
            "function f() { const r = useRef(null); const x = r.current.foo; return x; }",
            false,
            InferConfig::default(),
        );
        let foo = first_member(&cfg, "foo");
        assert!(types.is_ref_value(foo), "r.current.foo should stay a ref value, got {:?}", types.get(foo));
    }

    #[test]
    fn ref_like_name_requires_config() {
        let src = "function f(fooRef) { const c = fooRef.current; return c; }";
        // Without the flag, fooRef.current does not resolve to a ref value.
        let (cfg, types) = infer_src(src, false, InferConfig::default());
        let current = first_member(&cfg, "current");
        assert!(!types.is_ref_value(current));
        // With the flag, it does.
        let (cfg2, types2) = infer_src(
            src,
            false,
            InferConfig { enable_treat_ref_like_identifiers_as_refs: true, ..Default::default() },
        );
        let current2 = first_member(&cfg2, "current");
        assert!(types2.is_ref_value(current2), "got {:?}", types2.get(current2));
    }

    #[test]
    fn component_props_member_resolves() {
        // Component first param is props; props.ref -> UseRef object.
        let (cfg, types) = infer_src(
            "function Foo(props) { const r = props.ref; return r; }",
            true,
            InferConfig::default(),
        );
        let r = first_member(&cfg, "ref");
        assert!(types.is_use_ref(r), "props.ref should be a useRef object, got {:?}", types.get(r));
    }
}
