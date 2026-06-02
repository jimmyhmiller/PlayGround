//! Type system for React-Compiler parity.
//!
//! A faithful port of the upstream Rust React Compiler's type lattice
//! (`react_compiler_hir::Type`), object-shape registry
//! (`react_compiler_hir::object_shape`), and the builtin shapes / global
//! declarations from `react_compiler_hir::globals`. Source of truth:
//! `~/Documents/Code/open-source/react-rust-pr36173/compiler/crates/react_compiler_hir`.
//!
//! Scope note: the upstream `globals.rs` is ~2500 lines covering every builtin
//! (`Math`, `Object`, full `Array`/`Map`/`Set` method tables with aliasing
//! signatures, etc.). This port grows that registry fixture-by-fixture, exactly
//! as upstream did. We currently cover the props / array / object / ref / state /
//! reducer / jsx / function / mixed-readonly path that the ref-access and
//! mutation validations need. The `FunctionSignature` carries the full upstream
//! field set (effects, value kinds, hook kind) so the effect-system port
//! (`InferMutationAliasingEffects`) consumes it unchanged; `InferTypes` itself
//! only reads `shape.properties` and `function_type.return_type`.

use std::collections::HashMap;

// =============================================================================
// Effect / ValueKind / ValueReason (from type_config.rs + lib.rs)
// =============================================================================

/// Mirrors upstream `Effect`. The mutation kind a function applies to an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    Unknown,
    Freeze,
    Read,
    Capture,
    ConditionallyMutateIterator,
    ConditionallyMutate,
    Mutate,
    Store,
}

impl Effect {
    /// Capture, Store, ConditionallyMutate(Iterator), Mutate are mutable.
    pub fn is_mutable(&self) -> bool {
        matches!(
            self,
            Effect::Capture
                | Effect::Store
                | Effect::ConditionallyMutate
                | Effect::ConditionallyMutateIterator
                | Effect::Mutate
        )
    }
}

/// Mirrors upstream `ValueKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Mutable,
    Frozen,
    Primitive,
    MaybeFrozen,
    Global,
    Context,
}

/// Mirrors upstream `ValueReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueReason {
    KnownReturnSignature,
    State,
    ReducerState,
    Context,
    Effect,
    HookCaptured,
    HookReturn,
    Global,
    JsxCaptured,
    StoreLocal,
    ReactiveFunctionArgument,
    Other,
}

// =============================================================================
// Type lattice (from lib.rs `Type`)
// =============================================================================

/// A type-variable slot id, indexing the `Vec<Type>` arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(pub u32);

/// A literal property name (string or numeric index).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyLiteral {
    String(String),
    Number(u32),
}

/// The name of a property access, either a literal or a computed expression
/// (whose type is carried along).
#[derive(Debug, Clone)]
pub enum PropertyNameKind {
    Literal { value: PropertyLiteral },
    Computed { value: Box<Type> },
}

/// The type lattice. Faithful port of upstream `Type`.
#[derive(Debug, Clone)]
pub enum Type {
    Primitive,
    Function {
        shape_id: Option<String>,
        return_type: Box<Type>,
        is_constructor: bool,
    },
    Object {
        shape_id: Option<String>,
    },
    TypeVar {
        id: TypeId,
    },
    Poly,
    Phi {
        operands: Vec<Type>,
    },
    Property {
        object_type: Box<Type>,
        object_name: String,
        property_name: PropertyNameKind,
    },
    ObjectMethod,
}

// =============================================================================
// Shape id constants (from object_shape.rs)
// =============================================================================

pub const BUILT_IN_PROPS_ID: &str = "BuiltInProps";
pub const BUILT_IN_ARRAY_ID: &str = "BuiltInArray";
pub const BUILT_IN_FUNCTION_ID: &str = "BuiltInFunction";
pub const BUILT_IN_JSX_ID: &str = "BuiltInJsx";
pub const BUILT_IN_OBJECT_ID: &str = "BuiltInObject";
pub const BUILT_IN_USE_STATE_ID: &str = "BuiltInUseState";
pub const BUILT_IN_SET_STATE_ID: &str = "BuiltInSetState";
pub const BUILT_IN_USE_REF_ID: &str = "BuiltInUseRefId";
pub const BUILT_IN_REF_VALUE_ID: &str = "BuiltInRefValue";
pub const BUILT_IN_MIXED_READONLY_ID: &str = "BuiltInMixedReadonly";
pub const BUILT_IN_USE_REDUCER_ID: &str = "BuiltInUseReducer";
pub const BUILT_IN_DISPATCH_ID: &str = "BuiltInDispatch";
pub const BUILT_IN_USE_CONTEXT_HOOK_ID: &str = "BuiltInUseContextHook";

// =============================================================================
// Hook kinds + function signatures (from object_shape.rs)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookKind {
    UseContext,
    UseState,
    UseReducer,
    UseRef,
    UseEffect,
    UseLayoutEffect,
    UseInsertionEffect,
    UseMemo,
    UseCallback,
    UseImperativeHandle,
    Custom,
}

/// Call signature of a function/hook. Faithful field set from upstream
/// `FunctionSignature`; the effect-system port consumes the effect fields.
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub positional_params: Vec<Effect>,
    pub rest_param: Option<Effect>,
    pub return_type: Type,
    pub return_value_kind: ValueKind,
    pub return_value_reason: Option<ValueReason>,
    pub callee_effect: Effect,
    pub hook_kind: Option<HookKind>,
    pub no_alias: bool,
    pub mutable_only_if_operands_are_mutable: bool,
    pub impure: bool,
}

impl Default for FunctionSignature {
    fn default() -> Self {
        FunctionSignature {
            positional_params: Vec::new(),
            rest_param: None,
            return_type: Type::Poly,
            return_value_kind: ValueKind::Mutable,
            return_value_reason: None,
            callee_effect: Effect::Read,
            hook_kind: None,
            no_alias: false,
            mutable_only_if_operands_are_mutable: false,
            impure: false,
        }
    }
}

/// Shape of an object or function type (its property table + optional call sig).
#[derive(Debug, Clone)]
pub struct ObjectShape {
    pub properties: HashMap<String, Type>,
    pub function_type: Option<FunctionSignature>,
}

/// Registry mapping shape ids to their shapes.
#[derive(Debug, Clone, Default)]
pub struct ShapeRegistry {
    entries: HashMap<String, ObjectShape>,
}

impl ShapeRegistry {
    pub fn new() -> Self {
        ShapeRegistry { entries: HashMap::new() }
    }
    pub fn get(&self, key: &str) -> Option<&ObjectShape> {
        self.entries.get(key)
    }
    pub fn insert(&mut self, key: String, value: ObjectShape) {
        self.entries.insert(key, value);
    }
}

// =============================================================================
// Builders (from object_shape.rs add_function / add_hook / add_object)
// =============================================================================

fn add_shape(
    registry: &mut ShapeRegistry,
    id: &str,
    properties: Vec<(String, Type)>,
    function_type: Option<FunctionSignature>,
) {
    registry.insert(
        id.to_string(),
        ObjectShape { properties: properties.into_iter().collect(), function_type },
    );
}

static ANON_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn next_anon_id() -> String {
    let id = ANON_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("<generated_{}>", id)
}

fn add_function(
    registry: &mut ShapeRegistry,
    properties: Vec<(String, Type)>,
    sig: FunctionSignature,
    id: Option<&str>,
    is_constructor: bool,
) -> Type {
    let shape_id = id.map(|s| s.to_string()).unwrap_or_else(next_anon_id);
    let return_type = sig.return_type.clone();
    add_shape(registry, &shape_id, properties, Some(sig));
    Type::Function {
        shape_id: Some(shape_id),
        return_type: Box::new(return_type),
        is_constructor,
    }
}

fn add_hook(registry: &mut ShapeRegistry, sig: FunctionSignature, id: Option<&str>) -> Type {
    debug_assert!(sig.hook_kind.is_some(), "add_hook requires a hook_kind");
    add_function(registry, Vec::new(), sig, id, false)
}

fn add_object(registry: &mut ShapeRegistry, id: Option<&str>, properties: Vec<(String, Type)>) -> Type {
    let shape_id = id.map(|s| s.to_string()).unwrap_or_else(next_anon_id);
    add_shape(registry, &shape_id, properties, None);
    Type::Object { shape_id: Some(shape_id) }
}

fn obj(id: &str) -> Type {
    Type::Object { shape_id: Some(id.to_string()) }
}

/// Pure function returning `return_type`, reading its args.
fn simple_function(
    shapes: &mut ShapeRegistry,
    positional_params: Vec<Effect>,
    rest_param: Option<Effect>,
    return_type: Type,
    return_value_kind: ValueKind,
) -> Type {
    add_function(
        shapes,
        Vec::new(),
        FunctionSignature {
            positional_params,
            rest_param,
            return_type,
            return_value_kind,
            ..Default::default()
        },
        None,
        false,
    )
}

fn pure_primitive_fn(shapes: &mut ShapeRegistry) -> Type {
    simple_function(shapes, Vec::new(), Some(Effect::Read), Type::Primitive, ValueKind::Primitive)
}

// =============================================================================
// Builtin shapes (subset of build_builtin_shapes)
// =============================================================================

/// Build the builtin shape registry. Faithful to `build_builtin_shapes`, but
/// covering the currently-needed builtins (grows fixture-by-fixture).
pub fn build_builtin_shapes() -> ShapeRegistry {
    let mut shapes = ShapeRegistry::new();

    // BuiltInProps: { ref: UseRefType }
    add_object(&mut shapes, Some(BUILT_IN_PROPS_ID), vec![("ref".to_string(), obj(BUILT_IN_USE_REF_ID))]);

    build_array_shape(&mut shapes);
    build_object_shape(&mut shapes);
    build_ref_shapes(&mut shapes);
    build_state_shapes(&mut shapes);

    // BuiltInFunction / BuiltInJsx / BuiltInMixedReadonly are leaf object shapes.
    add_object(&mut shapes, Some(BUILT_IN_FUNCTION_ID), vec![]);
    add_object(&mut shapes, Some(BUILT_IN_JSX_ID), vec![]);
    add_object(&mut shapes, Some(BUILT_IN_MIXED_READONLY_ID), vec![("*".to_string(), obj(BUILT_IN_MIXED_READONLY_ID))]);

    shapes
}

fn build_array_shape(shapes: &mut ShapeRegistry) {
    // Methods whose property types matter for inference. Returns are typed; full
    // aliasing effects live with the effect-system port.
    let index_of = pure_primitive_fn(shapes);
    let includes = pure_primitive_fn(shapes);
    let join = pure_primitive_fn(shapes);
    let push = simple_function(shapes, Vec::new(), Some(Effect::Capture), Type::Primitive, ValueKind::Primitive);
    let map = simple_function(shapes, Vec::new(), Some(Effect::ConditionallyMutate), obj(BUILT_IN_ARRAY_ID), ValueKind::Mutable);
    let filter = simple_function(shapes, Vec::new(), Some(Effect::ConditionallyMutate), obj(BUILT_IN_ARRAY_ID), ValueKind::Mutable);
    let slice = simple_function(shapes, vec![Effect::Read], Some(Effect::Read), obj(BUILT_IN_ARRAY_ID), ValueKind::Mutable);
    let concat = simple_function(shapes, Vec::new(), Some(Effect::Capture), obj(BUILT_IN_ARRAY_ID), ValueKind::Mutable);

    add_object(
        shapes,
        Some(BUILT_IN_ARRAY_ID),
        vec![
            ("indexOf".to_string(), index_of),
            ("includes".to_string(), includes),
            ("join".to_string(), join),
            ("push".to_string(), push),
            ("map".to_string(), map),
            ("filter".to_string(), filter),
            ("slice".to_string(), slice),
            ("concat".to_string(), concat),
            ("length".to_string(), Type::Primitive),
        ],
    );
}

fn build_object_shape(shapes: &mut ShapeRegistry) {
    // BuiltInObject: a plain object literal. No fixed properties; property loads
    // fall through (no `*`), matching upstream BuiltInObject.
    add_object(shapes, Some(BUILT_IN_OBJECT_ID), vec![]);
}

fn build_ref_shapes(shapes: &mut ShapeRegistry) {
    // BuiltInUseRefId: { current: RefValue }
    add_object(shapes, Some(BUILT_IN_USE_REF_ID), vec![("current".to_string(), obj(BUILT_IN_REF_VALUE_ID))]);
    // BuiltInRefValue: { *: RefValue } (self-referencing — any access stays a ref value)
    add_object(shapes, Some(BUILT_IN_REF_VALUE_ID), vec![("*".to_string(), obj(BUILT_IN_REF_VALUE_ID))]);
}

fn build_state_shapes(shapes: &mut ShapeRegistry) {
    // BuiltInSetState: function freezing its argument.
    let set_state = add_function(
        shapes,
        Vec::new(),
        FunctionSignature {
            rest_param: Some(Effect::Freeze),
            return_type: Type::Primitive,
            return_value_kind: ValueKind::Primitive,
            ..Default::default()
        },
        Some(BUILT_IN_SET_STATE_ID),
        false,
    );
    // BuiltInUseState: { 0: Poly, 1: setState }
    add_object(
        shapes,
        Some(BUILT_IN_USE_STATE_ID),
        vec![("0".to_string(), Type::Poly), ("1".to_string(), set_state)],
    );

    // BuiltInDispatch + BuiltInUseReducer.
    let dispatch = add_function(
        shapes,
        Vec::new(),
        FunctionSignature {
            rest_param: Some(Effect::Freeze),
            return_type: Type::Primitive,
            return_value_kind: ValueKind::Primitive,
            ..Default::default()
        },
        Some(BUILT_IN_DISPATCH_ID),
        false,
    );
    add_object(
        shapes,
        Some(BUILT_IN_USE_REDUCER_ID),
        vec![("0".to_string(), Type::Poly), ("1".to_string(), dispatch)],
    );
}

// =============================================================================
// Globals (subset of build_default_globals + build_react_apis)
// =============================================================================

/// Maps global identifier names to their types.
#[derive(Debug, Clone, Default)]
pub struct GlobalRegistry {
    entries: HashMap<String, Type>,
}

impl GlobalRegistry {
    pub fn new() -> Self {
        GlobalRegistry { entries: HashMap::new() }
    }
    pub fn get(&self, key: &str) -> Option<&Type> {
        self.entries.get(key)
    }
    pub fn insert(&mut self, key: String, value: Type) {
        self.entries.insert(key, value);
    }
}

/// Build the default globals (React API hooks + React namespace). Faithful to
/// `build_default_globals`/`build_react_apis` for the hooks currently modeled.
pub fn build_default_globals(shapes: &mut ShapeRegistry) -> GlobalRegistry {
    let mut globals = GlobalRegistry::new();
    let react_apis = build_react_apis(shapes);

    // React namespace object carries the same API types as properties.
    let react_obj = add_object(shapes, Some("React"), react_apis.clone());
    globals.insert("React".to_string(), react_obj);

    for (name, ty) in react_apis {
        globals.insert(name, ty);
    }

    globals
}

fn build_react_apis(shapes: &mut ShapeRegistry) -> Vec<(String, Type)> {
    let mut apis: Vec<(String, Type)> = Vec::new();

    let use_context = add_hook(
        shapes,
        FunctionSignature {
            rest_param: Some(Effect::Read),
            return_type: Type::Poly,
            return_value_kind: ValueKind::Frozen,
            return_value_reason: Some(ValueReason::Context),
            hook_kind: Some(HookKind::UseContext),
            ..Default::default()
        },
        Some(BUILT_IN_USE_CONTEXT_HOOK_ID),
    );
    apis.push(("useContext".to_string(), use_context));

    let use_state = add_hook(
        shapes,
        FunctionSignature {
            rest_param: Some(Effect::Freeze),
            return_type: obj(BUILT_IN_USE_STATE_ID),
            return_value_kind: ValueKind::Frozen,
            return_value_reason: Some(ValueReason::State),
            hook_kind: Some(HookKind::UseState),
            ..Default::default()
        },
        None,
    );
    apis.push(("useState".to_string(), use_state));

    let use_reducer = add_hook(
        shapes,
        FunctionSignature {
            rest_param: Some(Effect::Freeze),
            return_type: obj(BUILT_IN_USE_REDUCER_ID),
            return_value_kind: ValueKind::Frozen,
            return_value_reason: Some(ValueReason::ReducerState),
            hook_kind: Some(HookKind::UseReducer),
            ..Default::default()
        },
        None,
    );
    apis.push(("useReducer".to_string(), use_reducer));

    let use_ref = add_hook(
        shapes,
        FunctionSignature {
            rest_param: Some(Effect::Capture),
            return_type: obj(BUILT_IN_USE_REF_ID),
            return_value_kind: ValueKind::Mutable,
            hook_kind: Some(HookKind::UseRef),
            ..Default::default()
        },
        None,
    );
    apis.push(("useRef".to_string(), use_ref));

    let use_memo = add_hook(
        shapes,
        FunctionSignature {
            rest_param: Some(Effect::Freeze),
            return_type: Type::Poly,
            return_value_kind: ValueKind::Frozen,
            hook_kind: Some(HookKind::UseMemo),
            ..Default::default()
        },
        None,
    );
    apis.push(("useMemo".to_string(), use_memo));

    let use_callback = add_hook(
        shapes,
        FunctionSignature {
            rest_param: Some(Effect::Freeze),
            return_type: Type::Poly,
            return_value_kind: ValueKind::Frozen,
            hook_kind: Some(HookKind::UseCallback),
            ..Default::default()
        },
        None,
    );
    apis.push(("useCallback".to_string(), use_callback));

    let use_effect = add_hook(
        shapes,
        FunctionSignature {
            rest_param: Some(Effect::Freeze),
            return_type: Type::Primitive,
            return_value_kind: ValueKind::Frozen,
            hook_kind: Some(HookKind::UseEffect),
            ..Default::default()
        },
        None,
    );
    apis.push(("useEffect".to_string(), use_effect));

    apis
}

// =============================================================================
// Name predicates (from environment.rs)
// =============================================================================

/// `use[A-Z0-9]...` — at least 4 chars. Faithful to upstream `is_hook_name`.
pub fn is_hook_name(name: &str) -> bool {
    if name.len() < 4 {
        return false;
    }
    if !name.starts_with("use") {
        return false;
    }
    let fourth = name.as_bytes()[3];
    fourth.is_ascii_uppercase() || fourth.is_ascii_digit()
}

/// Component (capitalized) or hook name. Faithful to `is_react_like_name`.
pub fn is_react_like_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    if name.as_bytes()[0].is_ascii_uppercase() {
        return true;
    }
    is_hook_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_name_predicate() {
        assert!(is_hook_name("useState"));
        assert!(is_hook_name("use123"));
        assert!(!is_hook_name("use"));
        assert!(!is_hook_name("usefoo")); // lowercase fourth char
        assert!(!is_hook_name("foo"));
    }

    #[test]
    fn builtin_shapes_resolve_ref_chain() {
        let shapes = build_builtin_shapes();
        // useRef shape has `current` -> RefValue
        let use_ref = shapes.get(BUILT_IN_USE_REF_ID).unwrap();
        match use_ref.properties.get("current") {
            Some(Type::Object { shape_id }) => assert_eq!(shape_id.as_deref(), Some(BUILT_IN_REF_VALUE_ID)),
            other => panic!("expected current -> RefValue, got {other:?}"),
        }
        // RefValue is self-referencing via `*`
        let ref_value = shapes.get(BUILT_IN_REF_VALUE_ID).unwrap();
        assert!(ref_value.properties.contains_key("*"));
    }

    #[test]
    fn globals_define_use_ref_returning_ref_shape() {
        let mut shapes = build_builtin_shapes();
        let globals = build_default_globals(&mut shapes);
        match globals.get("useRef") {
            Some(Type::Function { return_type, .. }) => match return_type.as_ref() {
                Type::Object { shape_id } => assert_eq!(shape_id.as_deref(), Some(BUILT_IN_USE_REF_ID)),
                other => panic!("useRef should return UseRef object, got {other:?}"),
            },
            other => panic!("useRef should be a function global, got {other:?}"),
        }
    }
}
