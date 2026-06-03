//! Mutation/aliasing **effect system** — abstract-interpretation state + effect
//! vocabulary. Faithful port of upstream
//! `react_compiler_inference::infer_mutation_aliasing_effects` (the data model +
//! lattice). The per-instruction effect *generation* (`compute_signature`) and
//! effect *application* (`apply_effect`) + the worklist driver and the
//! ranges pass build on this in later modules / follow-up commits.
//!
//! Source of truth: `infer_mutation_aliasing_effects.rs` in
//! `react_compiler_inference`. Upstream keys abstract values off `IdentifierId`
//! (post-SSA identifiers); our CFG is already SSA, so we key off [`Value`]. A
//! `Value` can still point to multiple allocation sites once a phi merges
//! branches, exactly like upstream's `variables: Id -> Set<ValueId>`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::cfg::Value;
use crate::infer_types::Types;
use crate::types::{ValueKind, ValueReason};

// =============================================================================
// ValueId — allocation-site identity (replaces upstream's object identity)
// =============================================================================

/// Unique allocation-site id. Faithful to upstream `ValueId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

static NEXT_VALUE_ID: AtomicU32 = AtomicU32::new(1);

impl ValueId {
    pub fn new() -> Self {
        ValueId(NEXT_VALUE_ID.fetch_add(1, Ordering::Relaxed))
    }
    /// Deterministic id derived from a `Value`, for the "uninitialized
    /// identifier" fallback (upstream uses `id | 0x8000_0000`).
    pub fn stable_for(v: Value) -> Self {
        ValueId(v.0 | 0x8000_0000)
    }
}

// =============================================================================
// AbstractValue
// =============================================================================

/// The abstract value at an allocation site: its kind + the reasons that
/// contributed. Faithful to upstream `AbstractValue`.
#[derive(Debug, Clone)]
pub struct AbstractValue {
    pub kind: ValueKind,
    pub reason: HashSet<ValueReason>,
}

impl AbstractValue {
    pub fn new(kind: ValueKind, reason: ValueReason) -> Self {
        AbstractValue { kind, reason: hashset_of(reason) }
    }
    /// The default for an unknown/uninitialized value: mutable, reason Other.
    pub fn mutable_other() -> Self {
        AbstractValue::new(ValueKind::Mutable, ValueReason::Other)
    }
}

pub fn hashset_of(r: ValueReason) -> HashSet<ValueReason> {
    let mut s = HashSet::new();
    s.insert(r);
    s
}

fn is_superset(a: &HashSet<ValueReason>, b: &HashSet<ValueReason>) -> bool {
    b.iter().all(|x| a.contains(x))
}

/// Join two abstract values. Faithful to upstream `merge_abstract_values`.
pub fn merge_abstract_values(a: &AbstractValue, b: &AbstractValue) -> AbstractValue {
    let kind = merge_value_kinds(a.kind, b.kind);
    if kind == a.kind && kind == b.kind && is_superset(&a.reason, &b.reason) {
        return a.clone();
    }
    let mut reason = a.reason.clone();
    for r in &b.reason {
        reason.insert(*r);
    }
    AbstractValue { kind, reason }
}

/// Join two value kinds. Faithful to upstream `merge_value_kinds`.
pub fn merge_value_kinds(a: ValueKind, b: ValueKind) -> ValueKind {
    use ValueKind::*;
    if a == b {
        return a;
    }
    if a == MaybeFrozen || b == MaybeFrozen {
        return MaybeFrozen;
    }
    if a == Mutable || b == Mutable {
        if a == Frozen || b == Frozen {
            return MaybeFrozen;
        } else if a == Context || b == Context {
            return Context;
        } else {
            return Mutable;
        }
    }
    if a == Context || b == Context {
        if a == Frozen || b == Frozen {
            return MaybeFrozen;
        } else {
            return Context;
        }
    }
    if a == Frozen || b == Frozen {
        return Frozen;
    }
    if a == Global || b == Global {
        return Global;
    }
    // Remaining: both Primitive (handled by a == b) — fall back to Mutable to
    // match upstream's final return.
    Mutable
}

// =============================================================================
// Effect vocabulary (AliasingEffect)
// =============================================================================

/// Why a value can't be modified (carried by error effects). A lightweight
/// stand-in for upstream's embedded `CompilerDiagnostic`; the validation passes
/// consume `kind` + the offending `value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectErrorKind {
    MutateFrozen,
    MutateGlobal,
    Impure,
}

/// Reason annotation on a `Mutate` effect (drives the "rename to Ref" hint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationReason {
    AssignCurrentProperty,
}

/// Describes the aliasing/mutation/data-flow effect of an instruction or
/// terminal. Faithful port of upstream `AliasingEffect` (over [`Value`]).
#[derive(Debug, Clone)]
pub enum AliasingEffect {
    /// Freeze `value` and its direct aliases.
    Freeze { value: Value, reason: ValueReason },
    /// Mutate `value` and direct aliases.
    Mutate { value: Value, reason: Option<MutationReason> },
    /// Mutate `value` only if it is mutable.
    MutateConditionally { value: Value },
    /// Mutate `value` and transitive captures.
    MutateTransitive { value: Value },
    /// Mutate `value` and transitive captures, conditionally.
    MutateTransitiveConditionally { value: Value },
    /// Information flow `from` -> `into` (capture, not aliasing).
    Capture { from: Value, into: Value },
    /// Direct aliasing: mutation of `into` implies mutation of `from`.
    Alias { from: Value, into: Value },
    /// Potential aliasing.
    MaybeAlias { from: Value, into: Value },
    /// Direct assignment `into = from`.
    Assign { from: Value, into: Value },
    /// Create a value of `value` kind at `into`.
    Create { into: Value, value: ValueKind, reason: ValueReason },
    /// Create a new value at `into` with the same kind as `from`.
    CreateFrom { from: Value, into: Value },
    /// Immutable data flow (escape analysis only).
    ImmutableCapture { from: Value, into: Value },
    /// Function-call application.
    Apply {
        receiver: Value,
        function: Value,
        mutates_function: bool,
        args: Vec<ApplyArg>,
        into: Value,
    },
    /// Mutation of a frozen/global value, or impure call (validation errors).
    Error { place: Value, kind: EffectErrorKind },
    /// Value accessed during render (ref-in-render analysis).
    Render { place: Value },
}

/// An argument position in an `Apply` effect.
#[derive(Debug, Clone)]
pub enum ApplyArg {
    Place(Value),
    Spread(Value),
    Hole,
}

// =============================================================================
// Mutation classification
// =============================================================================

#[derive(Debug, Clone, Copy)]
pub enum MutateVariant {
    Mutate,
    MutateConditionally,
    MutateTransitive,
    MutateTransitiveConditionally,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationResult {
    None,
    Mutate,
    MutateFrozen,
    MutateGlobal,
    MutateRef,
}

// =============================================================================
// InferenceState — the abstract state
// =============================================================================

/// Abstract state tracked during effect inference. Faithful to upstream
/// `InferenceState`.
#[derive(Debug, Clone)]
pub struct InferenceState {
    pub is_function_expression: bool,
    /// Abstract value per allocation site.
    values: HashMap<ValueId, AbstractValue>,
    /// The allocation sites each SSA value may point to.
    variables: HashMap<Value, HashSet<ValueId>>,
}

impl InferenceState {
    pub fn empty(is_function_expression: bool) -> Self {
        InferenceState {
            is_function_expression,
            values: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    /// The merged abstract value pointed to by `place`. Faithful to upstream
    /// `kind` (the uninitialized-access tracking is dropped; we return the
    /// mutable/other default like upstream does after recording the error).
    pub fn kind(&self, place: Value) -> AbstractValue {
        let values = match self.variables.get(&place) {
            Some(v) => v,
            None => return AbstractValue::mutable_other(),
        };
        let mut merged: Option<AbstractValue> = None;
        for value_id in values {
            let k = match self.values.get(value_id) {
                Some(k) => k,
                None => continue,
            };
            merged = Some(match merged {
                Some(prev) => merge_abstract_values(&prev, k),
                None => k.clone(),
            });
        }
        merged.unwrap_or_else(AbstractValue::mutable_other)
    }

    pub fn initialize(&mut self, value_id: ValueId, kind: AbstractValue) {
        self.values.insert(value_id, kind);
    }

    pub fn define(&mut self, place: Value, value_id: ValueId) {
        let mut set = HashSet::new();
        set.insert(value_id);
        self.variables.insert(place, set);
    }

    /// `into = from` (copy the pointed-to allocation set). Faithful to `assign`.
    pub fn assign(&mut self, into: Value, from: Value) {
        let values = match self.variables.get(&from) {
            Some(v) => v.clone(),
            None => {
                let vid = ValueId::stable_for(from);
                let mut set = HashSet::new();
                set.insert(vid);
                self.values.entry(vid).or_insert_with(AbstractValue::mutable_other);
                set
            }
        };
        self.variables.insert(into, values);
    }

    /// Add `value`'s allocation set into `place`'s. Faithful to `append_alias`.
    pub fn append_alias(&mut self, place: Value, value: Value) {
        let new_values = match self.variables.get(&value) {
            Some(v) => v.clone(),
            None => return,
        };
        let prev_values = match self.variables.get(&place) {
            Some(v) => v.clone(),
            None => return,
        };
        let merged: HashSet<ValueId> = prev_values.union(&new_values).copied().collect();
        self.variables.insert(place, merged);
    }

    pub fn is_defined(&self, place: Value) -> bool {
        self.variables.contains_key(&place)
    }

    pub fn values_for(&self, place: Value) -> Vec<ValueId> {
        self.variables.get(&place).map(|s| s.iter().copied().collect()).unwrap_or_default()
    }

    /// Freeze `place` (and its allocations) if mutable/context/maybe-frozen.
    /// Returns whether anything was frozen. Faithful to `freeze`.
    pub fn freeze(&mut self, place: Value, reason: ValueReason) -> bool {
        if !self.variables.contains_key(&place) {
            return false;
        }
        match self.kind(place).kind {
            ValueKind::Context | ValueKind::Mutable | ValueKind::MaybeFrozen => {
                for vid in self.values_for(place) {
                    self.freeze_value(vid, reason);
                }
                true
            }
            ValueKind::Frozen | ValueKind::Global | ValueKind::Primitive => false,
        }
    }

    pub fn freeze_value(&mut self, value_id: ValueId, reason: ValueReason) {
        self.values.insert(value_id, AbstractValue::new(ValueKind::Frozen, reason));
    }

    /// Classify a mutation of `place`. Faithful to `mutate`. `is_ref` consults
    /// the type table (ref/ref-value types are special-cased to `MutateRef`).
    pub fn mutate(&self, variant: MutateVariant, place: Value, types: &Types) -> MutationResult {
        if types.is_ref_or_ref_value(place) {
            return MutationResult::MutateRef;
        }
        let kind = self.kind(place).kind;
        match variant {
            MutateVariant::MutateConditionally | MutateVariant::MutateTransitiveConditionally => {
                match kind {
                    ValueKind::Mutable | ValueKind::Context => MutationResult::Mutate,
                    _ => MutationResult::None,
                }
            }
            MutateVariant::Mutate | MutateVariant::MutateTransitive => match kind {
                ValueKind::Mutable | ValueKind::Context => MutationResult::Mutate,
                ValueKind::Primitive => MutationResult::None,
                ValueKind::Frozen | ValueKind::MaybeFrozen => MutationResult::MutateFrozen,
                ValueKind::Global => MutationResult::MutateGlobal,
            },
        }
    }

    /// A phi's value set is the union of its operands' sets. Faithful to
    /// `infer_phi`. Operands not yet defined (back-edges) are handled by merge.
    pub fn infer_phi(&mut self, phi: Value, operands: &[Value]) {
        let mut values: HashSet<ValueId> = HashSet::new();
        for op in operands {
            if let Some(op_values) = self.variables.get(op) {
                for v in op_values {
                    values.insert(*v);
                }
            }
        }
        if !values.is_empty() {
            self.variables.insert(phi, values);
        }
    }

    /// Join `other` into `self`, returning the merged state if anything changed
    /// (else `None`, signalling a fixpoint). Faithful to `merge`.
    pub fn merge(&self, other: &InferenceState) -> Option<InferenceState> {
        let mut next_values: Option<HashMap<ValueId, AbstractValue>> = None;
        let mut next_variables: Option<HashMap<Value, HashSet<ValueId>>> = None;

        for (id, this_value) in &self.values {
            if let Some(other_value) = other.values.get(id) {
                let merged = merge_abstract_values(this_value, other_value);
                if merged.kind != this_value.kind || !is_superset(&this_value.reason, &merged.reason) {
                    next_values.get_or_insert_with(|| self.values.clone()).insert(*id, merged);
                }
            }
        }
        for (id, other_value) in &other.values {
            if !self.values.contains_key(id) {
                next_values.get_or_insert_with(|| self.values.clone()).insert(*id, other_value.clone());
            }
        }

        for (id, this_values) in &self.variables {
            if let Some(other_values) = other.variables.get(id) {
                if other_values.iter().any(|ov| !this_values.contains(ov)) {
                    let merged: HashSet<ValueId> = this_values.union(other_values).copied().collect();
                    next_variables.get_or_insert_with(|| self.variables.clone()).insert(*id, merged);
                }
            }
        }
        for (id, other_values) in &other.variables {
            if !self.variables.contains_key(id) {
                next_variables.get_or_insert_with(|| self.variables.clone()).insert(*id, other_values.clone());
            }
        }

        if next_variables.is_none() && next_values.is_none() {
            None
        } else {
            Some(InferenceState {
                is_function_expression: self.is_function_expression,
                values: next_values.unwrap_or_else(|| self.values.clone()),
                variables: next_variables.unwrap_or_else(|| self.variables.clone()),
            })
        }
    }
}

// =============================================================================
// Context — value-id caching across fixpoint iterations
// =============================================================================

/// Per-function inference context. Faithful subset of upstream `Context`:
/// keeps a stable `ValueId` per effect across worklist iterations so the
/// abstract state converges. (The function-signature / function-value caches
/// are omitted because our CFG has no `FunctionExpression` op — closures are
/// lowered to `MakeArray` captures.)
#[derive(Default)]
pub struct Context {
    effect_value_id_cache: HashMap<String, ValueId>,
}

impl Context {
    pub fn new() -> Self {
        Context::default()
    }

    /// Stable `ValueId` for an effect, ensuring fixpoint convergence. Faithful
    /// to upstream `get_or_create_value_id`.
    fn value_id_for(&mut self, key: String) -> ValueId {
        *self.effect_value_id_cache.entry(key).or_insert_with(ValueId::new)
    }
}

/// Pick the primary reason from a set. Faithful to upstream `primary_reason`
/// (deterministic: lowest-discriminant by a fixed order, defaulting to Other).
fn primary_reason(reasons: &HashSet<ValueReason>) -> ValueReason {
    // Upstream returns the first by insertion; we need determinism without
    // insertion order, so fix a priority order matching the enum declaration.
    const ORDER: [ValueReason; 12] = [
        ValueReason::KnownReturnSignature,
        ValueReason::State,
        ValueReason::ReducerState,
        ValueReason::Context,
        ValueReason::Effect,
        ValueReason::HookCaptured,
        ValueReason::HookReturn,
        ValueReason::Global,
        ValueReason::JsxCaptured,
        ValueReason::StoreLocal,
        ValueReason::ReactiveFunctionArgument,
        ValueReason::Other,
    ];
    for r in ORDER {
        if reasons.contains(&r) {
            return r;
        }
    }
    ValueReason::Other
}

// =============================================================================
// apply_effect — apply one effect to the abstract state, recording the
// resolved effect(s). Faithful port of upstream `apply_effect` for our op
// subset (no function-value Apply path; no aliasing-config signatures).
// =============================================================================

fn key_create(into: Value, kind: ValueKind, reason: ValueReason) -> String {
    format!("Create:{}:{:?}:{:?}", into.0, kind, reason)
}
fn key_create_from(from: Value, into: Value) -> String {
    format!("CreateFrom:{}:{}", from.0, into.0)
}

pub fn apply_effect(
    ctx: &mut Context,
    state: &mut InferenceState,
    types: &Types,
    effect: AliasingEffect,
    initialized: &mut HashSet<Value>,
    out: &mut Vec<AliasingEffect>,
) {
    match &effect {
        AliasingEffect::Freeze { value, reason } => {
            if state.freeze(*value, *reason) {
                out.push(effect.clone());
                // (Transitive freeze through function-expression captures is a
                // no-op here: our CFG has no function values to recurse into.)
            }
        }

        AliasingEffect::Create { into, value: kind, reason } => {
            initialized.insert(*into);
            let vid = ctx.value_id_for(key_create(*into, *kind, *reason));
            state.initialize(vid, AbstractValue::new(*kind, *reason));
            state.define(*into, vid);
            out.push(effect.clone());
        }

        AliasingEffect::ImmutableCapture { from, .. } => {
            match state.kind(*from).kind {
                ValueKind::Global | ValueKind::Primitive => {} // copy types: no data flow
                _ => out.push(effect.clone()),
            }
        }

        AliasingEffect::CreateFrom { from, into } => {
            initialized.insert(*into);
            let from_value = state.kind(*from);
            let vid = ctx.value_id_for(key_create_from(*from, *into));
            state.initialize(vid, from_value.clone());
            state.define(*into, vid);
            match from_value.kind {
                ValueKind::Primitive | ValueKind::Global => {
                    out.push(AliasingEffect::Create {
                        into: *into,
                        value: from_value.kind,
                        reason: primary_reason(&from_value.reason),
                    });
                }
                ValueKind::Frozen => {
                    out.push(AliasingEffect::Create {
                        into: *into,
                        value: from_value.kind,
                        reason: primary_reason(&from_value.reason),
                    });
                    apply_effect(ctx, state, types, AliasingEffect::ImmutableCapture { from: *from, into: *into }, initialized, out);
                }
                _ => out.push(effect.clone()),
            }
        }

        AliasingEffect::MaybeAlias { from, into }
        | AliasingEffect::Alias { from, into }
        | AliasingEffect::Capture { from, into } => {
            let is_maybe_alias = matches!(effect, AliasingEffect::MaybeAlias { .. });
            let into_kind = state.kind(*into).kind;
            let destination_type = match into_kind {
                ValueKind::Context => Some("context"),
                ValueKind::Mutable | ValueKind::MaybeFrozen => Some("mutable"),
                _ => None,
            };
            let from_kind = state.kind(*from).kind;
            let source_type = match from_kind {
                ValueKind::Context => Some("context"),
                ValueKind::Global | ValueKind::Primitive => None,
                ValueKind::MaybeFrozen | ValueKind::Frozen => Some("frozen"),
                ValueKind::Mutable => Some("mutable"),
            };

            if source_type == Some("frozen") {
                apply_effect(ctx, state, types, AliasingEffect::ImmutableCapture { from: *from, into: *into }, initialized, out);
            } else if (source_type == Some("mutable") && destination_type == Some("mutable")) || is_maybe_alias {
                out.push(effect.clone());
            } else if (source_type == Some("context") && destination_type.is_some())
                || (source_type == Some("mutable") && destination_type == Some("context"))
            {
                apply_effect(ctx, state, types, AliasingEffect::MaybeAlias { from: *from, into: *into }, initialized, out);
            }
        }

        AliasingEffect::Assign { from, into } => {
            initialized.insert(*into);
            let from_value = state.kind(*from);
            match from_value.kind {
                ValueKind::Frozen => {
                    apply_effect(ctx, state, types, AliasingEffect::ImmutableCapture { from: *from, into: *into }, initialized, out);
                    let vid = ctx.value_id_for(format!("Assign_frozen:{}:{}", from.0, into.0));
                    state.initialize(vid, from_value.clone());
                    state.define(*into, vid);
                }
                ValueKind::Global | ValueKind::Primitive => {
                    let vid = ctx.value_id_for(format!("Assign_copy:{}:{}", from.0, into.0));
                    state.initialize(vid, from_value.clone());
                    state.define(*into, vid);
                }
                _ => {
                    state.assign(*into, *from);
                    out.push(effect.clone());
                }
            }
        }

        AliasingEffect::Apply { receiver, function, mutates_function, args, into } => {
            // Our CFG has no function-value callees, so the only inputs are the
            // signature (legacy) path and the default no-signature path. The
            // signature is selected by the caller (compute_signature) and passed
            // as a separate effect list; here Apply with no signature falls to
            // the default behavior. (Signature application happens in
            // compute_signature, which emits legacy effects directly.)
            let _ = (receiver, function);
            apply_effect(ctx, state, types, AliasingEffect::Create {
                into: *into, value: ValueKind::Mutable, reason: ValueReason::Other,
            }, initialized, out);
            let operands: Vec<Value> = std::iter::once(*receiver)
                .chain(std::iter::once(*function))
                .chain(args.iter().filter_map(|a| match a {
                    ApplyArg::Place(p) | ApplyArg::Spread(p) => Some(*p),
                    ApplyArg::Hole => None,
                }))
                .collect();
            for operand in &operands {
                if *operand == *function && !mutates_function {
                    // don't mutate callee for non-mutating calls
                } else {
                    apply_effect(ctx, state, types, AliasingEffect::MutateTransitiveConditionally { value: *operand }, initialized, out);
                }
                apply_effect(ctx, state, types, AliasingEffect::MaybeAlias { from: *operand, into: *into }, initialized, out);
                for other in &operands {
                    if other == operand {
                        continue;
                    }
                    apply_effect(ctx, state, types, AliasingEffect::Capture { from: *operand, into: *other }, initialized, out);
                }
            }
        }

        AliasingEffect::Mutate { value, .. }
        | AliasingEffect::MutateConditionally { value }
        | AliasingEffect::MutateTransitive { value }
        | AliasingEffect::MutateTransitiveConditionally { value } => {
            let variant = match &effect {
                AliasingEffect::Mutate { .. } => MutateVariant::Mutate,
                AliasingEffect::MutateConditionally { .. } => MutateVariant::MutateConditionally,
                AliasingEffect::MutateTransitive { .. } => MutateVariant::MutateTransitive,
                AliasingEffect::MutateTransitiveConditionally { .. } => MutateVariant::MutateTransitiveConditionally,
                _ => unreachable!(),
            };
            let result = state.mutate(variant, *value, types);
            if result == MutationResult::Mutate {
                out.push(effect.clone());
            } else if result == MutationResult::MutateRef {
                // no-op
            } else if result != MutationResult::None
                && matches!(variant, MutateVariant::Mutate | MutateVariant::MutateTransitive)
            {
                let kind = if result == MutationResult::MutateFrozen {
                    EffectErrorKind::MutateFrozen
                } else {
                    EffectErrorKind::MutateGlobal
                };
                out.push(AliasingEffect::Error { place: *value, kind });
            }
        }

        AliasingEffect::Error { .. } | AliasingEffect::Render { .. } => {
            out.push(effect.clone());
        }
    }
}

// =============================================================================
// compute_effects_for_legacy_signature — map a function signature's per-arg
// effects to aliasing effects. Faithful port (the path hooks/setState/JSX use).
// =============================================================================

/// Convert a builtin/legacy [`FunctionSignature`] call into aliasing effects.
/// Faithful to upstream `compute_effects_for_legacy_signature` (minus the
/// `mutable_only_if_operands_are_mutable` fast path and spread-todo errors,
/// noted inline).
pub fn compute_effects_for_legacy_signature(
    signature: &crate::types::FunctionSignature,
    lvalue: Value,
    receiver: Value,
    args: &[ApplyArg],
) -> Vec<AliasingEffect> {
    use crate::types::Effect as E;
    let return_reason = signature.return_value_reason.unwrap_or(ValueReason::Other);
    let mut effects: Vec<AliasingEffect> = Vec::new();

    effects.push(AliasingEffect::Create {
        into: lvalue,
        value: signature.return_value_kind,
        reason: return_reason,
    });
    if signature.impure {
        effects.push(AliasingEffect::Error { place: receiver, kind: EffectErrorKind::Impure });
    }

    let mut stores: Vec<Value> = Vec::new();
    let mut captures: Vec<Value> = Vec::new();

    let mut visit = |place: Value, effect: E, effects: &mut Vec<AliasingEffect>| match effect {
        E::Store => {
            effects.push(AliasingEffect::Mutate { value: place, reason: None });
            stores.push(place);
        }
        E::Capture => captures.push(place),
        E::ConditionallyMutate => {
            effects.push(AliasingEffect::MutateTransitiveConditionally { value: place });
        }
        E::ConditionallyMutateIterator => {
            // Iterator-mutation modelling is deferred; capture into lvalue (the
            // observable aliasing) so deps stay sound.
            effects.push(AliasingEffect::Capture { from: place, into: lvalue });
        }
        E::Freeze => effects.push(AliasingEffect::Freeze { value: place, reason: return_reason }),
        E::Mutate => effects.push(AliasingEffect::MutateTransitive { value: place }),
        E::Read => effects.push(AliasingEffect::ImmutableCapture { from: place, into: lvalue }),
        E::Unknown => {}
    };

    if signature.callee_effect != E::Capture {
        effects.push(AliasingEffect::Alias { from: receiver, into: lvalue });
    }
    visit(receiver, signature.callee_effect, &mut effects);

    for (i, arg) in args.iter().enumerate() {
        let (place, is_spread) = match arg {
            ApplyArg::Hole => continue,
            ApplyArg::Place(p) => (*p, false),
            ApplyArg::Spread(p) => (*p, true),
        };
        let sig_effect = if !is_spread && i < signature.positional_params.len() {
            signature.positional_params[i]
        } else {
            signature.rest_param.unwrap_or(E::ConditionallyMutate)
        };
        // Spread of a non-mutating effect degrades to iterator mutation upstream;
        // for our subset we apply the effect directly (spread is rare in targets).
        visit(place, sig_effect, &mut effects);
    }

    if !captures.is_empty() {
        if stores.is_empty() {
            for capture in &captures {
                effects.push(AliasingEffect::Alias { from: *capture, into: lvalue });
            }
        } else {
            for capture in &captures {
                for store in &stores {
                    effects.push(AliasingEffect::Capture { from: *capture, into: *store });
                }
            }
        }
    }

    effects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_kind_lattice() {
        use ValueKind::*;
        // identity
        assert_eq!(merge_value_kinds(Frozen, Frozen), Frozen);
        // mutable + frozen = maybe-frozen
        assert_eq!(merge_value_kinds(Mutable, Frozen), MaybeFrozen);
        assert_eq!(merge_value_kinds(Frozen, Mutable), MaybeFrozen);
        // mutable + context = context
        assert_eq!(merge_value_kinds(Mutable, Context), Context);
        // context + frozen = maybe-frozen
        assert_eq!(merge_value_kinds(Context, Frozen), MaybeFrozen);
        // anything + maybe-frozen = maybe-frozen
        assert_eq!(merge_value_kinds(Primitive, MaybeFrozen), MaybeFrozen);
        // frozen + global = frozen
        assert_eq!(merge_value_kinds(Frozen, Global), Frozen);
        // global + primitive = global
        assert_eq!(merge_value_kinds(Global, Primitive), Global);
    }

    #[test]
    fn freeze_transitions() {
        let mut st = InferenceState::empty(false);
        let v = Value(0);
        let vid = ValueId::new();
        st.initialize(vid, AbstractValue::new(ValueKind::Mutable, ValueReason::Other));
        st.define(v, vid);
        assert_eq!(st.kind(v).kind, ValueKind::Mutable);
        // Freezing a mutable value works and flips it to Frozen.
        assert!(st.freeze(v, ValueReason::JsxCaptured));
        assert_eq!(st.kind(v).kind, ValueKind::Frozen);
        // Freezing an already-frozen value is a no-op.
        assert!(!st.freeze(v, ValueReason::JsxCaptured));
    }

    #[test]
    fn assign_shares_allocations_then_alias_unions() {
        let mut st = InferenceState::empty(false);
        let (a, b, c) = (Value(0), Value(1), Value(2));
        let va = ValueId::new();
        let vc = ValueId::new();
        st.initialize(va, AbstractValue::new(ValueKind::Mutable, ValueReason::Other));
        st.initialize(vc, AbstractValue::new(ValueKind::Frozen, ValueReason::Other));
        st.define(a, va);
        st.define(c, vc);
        // b = a  -> b points where a points
        st.assign(b, a);
        assert_eq!(st.values_for(b), vec![va]);
        // append c's allocations into b -> b now {va, vc}, kind = Mutable+Frozen = MaybeFrozen
        st.append_alias(b, c);
        assert_eq!(st.kind(b).kind, ValueKind::MaybeFrozen);
    }

    fn define_mutable(st: &mut InferenceState, v: Value) {
        let vid = ValueId::new();
        st.initialize(vid, AbstractValue::new(ValueKind::Mutable, ValueReason::Other));
        st.define(v, vid);
    }

    #[test]
    fn legacy_hook_signature_freezes_args() {
        use crate::types::{Effect, FunctionSignature, HookKind};
        // A useState-like hook: rest_param Freeze, returns Frozen.
        let sig = FunctionSignature {
            rest_param: Some(Effect::Freeze),
            return_value_kind: ValueKind::Frozen,
            return_value_reason: Some(ValueReason::State),
            hook_kind: Some(HookKind::UseState),
            ..Default::default()
        };
        let (callee, arg, lvalue) = (Value(0), Value(1), Value(2));
        let effects = compute_effects_for_legacy_signature(&sig, lvalue, callee, &[ApplyArg::Place(arg)]);

        let mut st = InferenceState::empty(false);
        define_mutable(&mut st, callee);
        define_mutable(&mut st, arg);
        let types = Types::empty();
        let mut ctx = Context::new();
        let mut initialized = HashSet::new();
        let mut out = Vec::new();
        for e in effects {
            apply_effect(&mut ctx, &mut st, &types, e, &mut initialized, &mut out);
        }
        // The hook's argument is now frozen.
        assert_eq!(st.kind(arg).kind, ValueKind::Frozen, "hook arg should be frozen");
        // The lvalue (hook result) is frozen too.
        assert_eq!(st.kind(lvalue).kind, ValueKind::Frozen, "hook result should be frozen");
    }

    #[test]
    fn store_member_mutate_then_frozen_error() {
        // Mutating a frozen value records a MutateFrozen error effect.
        let mut st = InferenceState::empty(false);
        let v = Value(0);
        let vid = ValueId::new();
        st.initialize(vid, AbstractValue::new(ValueKind::Frozen, ValueReason::ReactiveFunctionArgument));
        st.define(v, vid);
        let types = Types::empty();
        let mut ctx = Context::new();
        let mut initialized = HashSet::new();
        let mut out = Vec::new();
        apply_effect(&mut ctx, &mut st, &types, AliasingEffect::Mutate { value: v, reason: None }, &mut initialized, &mut out);
        assert!(
            out.iter().any(|e| matches!(e, AliasingEffect::Error { kind: EffectErrorKind::MutateFrozen, .. })),
            "expected a MutateFrozen error, got {out:?}"
        );
    }

    #[test]
    fn merge_detects_fixpoint() {
        let mut a = InferenceState::empty(false);
        let v = Value(0);
        let vid = ValueId::new();
        a.initialize(vid, AbstractValue::new(ValueKind::Mutable, ValueReason::Other));
        a.define(v, vid);
        // merging with itself is a fixpoint (no change)
        assert!(a.merge(&a).is_none());
        // merging in a frozen view of the same value changes the kind
        let mut b = a.clone();
        b.freeze_value(vid, ValueReason::Other);
        let merged = a.merge(&b).expect("kind changed -> Some");
        assert_eq!(merged.kind(v).kind, ValueKind::MaybeFrozen);
    }
}
