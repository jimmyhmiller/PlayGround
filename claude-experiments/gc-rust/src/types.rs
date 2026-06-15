//! Semantic types and the type checker.
//!
//! gc-rust uses straightforward bidirectional checking with light unification
//! (enough for `let` inference, literal defaulting, and generic instantiation).
//! Generics are checked once in their generic form here; the *monomorphizer*
//! (`src/mono.rs`) later specializes each instantiation. The checker records,
//! at every call site and constructor, the concrete type arguments so the
//! monomorphizer can do its job without re-running inference.

use crate::ast::*;
use crate::lexer::{NumSuffix, Span};
use std::collections::HashMap;
use std::rc::Rc;

/// A semantic type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    /// A primitive scalar / unit / never.
    Prim(Prim),
    /// A user struct or enum applied to type args: `Vec<i64>`, `Point`,
    /// `Option<T>`. `name` is the fully-qualified def name.
    Named { name: String, args: Vec<Ty> },
    /// A generic parameter (de-name'd to its declared name in scope).
    Var(String),
    /// `fn(A, B) -> R` (also the type of closures, structurally).
    Fn { params: Vec<Ty>, ret: Box<Ty> },
    /// Tuple. Empty = unit (but unit is `Prim::Unit`).
    Tuple(Vec<Ty>),
    /// A fixed array `[T; N]`.
    Array(Box<Ty>, u64),
    /// An inference hole (only during checking).
    Infer(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prim {
    I8, I16, I32, I64, U8, U16, U32, U64, F32, F64,
    Bool, Char, Str, Unit,
}

impl Ty {
    pub fn unit() -> Ty { Ty::Prim(Prim::Unit) }
    pub fn bool() -> Ty { Ty::Prim(Prim::Bool) }
    pub fn i64() -> Ty { Ty::Prim(Prim::I64) }

    pub fn is_unit(&self) -> bool { matches!(self, Ty::Prim(Prim::Unit)) }
}

#[derive(Debug)]
pub struct TypeError {
    pub msg: String,
    pub span: Span,
}

type TResult<T> = Result<T, TypeError>;

/// Definitions the checker needs, indexed by fully-qualified name.
pub struct TyCtx {
    pub structs: HashMap<String, Rc<StructDef>>,
    pub enums: HashMap<String, Rc<EnumDef>>,
    pub fns: HashMap<String, Rc<FnDef>>,
    pub traits: HashMap<String, Rc<TraitDef>>,
    /// Trait impls, grouped for method lookup.
    pub impls: Vec<Rc<ImplBlock>>,
    /// Variant name → (enum fq-name, index, payload types as written).
    pub variants: HashMap<String, (String, u32)>,
    /// Method index: (receiver base type name, method name) → the impl method.
    /// Used to resolve `recv.method(..)` to a concrete function. Inherent and
    /// trait impls are both indexed here (trait method resolution is by name).
    pub methods: HashMap<(String, String), MethodEntry>,
}

#[derive(Clone)]
pub struct MethodEntry {
    /// The impl's `Self` type as written (e.g. `Option<T>`), for binding `Self`.
    pub self_ty: Type,
    /// The impl block's own generic parameter names (`impl<T> ... for Foo<T>`).
    pub impl_generics: Vec<String>,
    pub method: Rc<FnDef>,
}

impl TyCtx {
    pub fn from_globals(g: &crate::resolve::GlobalTable) -> Self {
        let mut variants = HashMap::new();
        for (name, e) in &g.enums {
            for (i, v) in e.variants.iter().enumerate() {
                variants.insert(format!("{}::{}", name, v.name), (name.clone(), i as u32));
                // also bare variant name for unqualified `Some`/`None`/`Ok`/`Err`
                variants
                    .entry(v.name.clone())
                    .or_insert((name.clone(), i as u32));
            }
        }
        // Build the method index from every impl block.
        let mut methods = HashMap::new();
        for imp in &g.impls {
            let base = type_base_name(&imp.self_ty);
            let impl_generics: Vec<String> =
                imp.generics.params.iter().map(|p| p.name.clone()).collect();
            for m in &imp.items {
                methods.insert(
                    (base.clone(), m.name.clone()),
                    MethodEntry {
                        self_ty: imp.self_ty.clone(),
                        impl_generics: impl_generics.clone(),
                        method: Rc::new(m.clone()),
                    },
                );
            }
        }
        TyCtx {
            structs: g.structs.iter().map(|(k, v)| (k.clone(), Rc::new(v.clone()))).collect(),
            enums: g.enums.iter().map(|(k, v)| (k.clone(), Rc::new(v.clone()))).collect(),
            fns: g.fns.iter().map(|(k, v)| (k.clone(), Rc::new(v.clone()))).collect(),
            traits: g.traits.iter().map(|(k, v)| (k.clone(), Rc::new(v.clone()))).collect(),
            impls: g.impls.iter().map(|i| Rc::new(i.clone())).collect(),
            variants,
            methods,
        }
    }

    /// Resolve a fully-qualified or unqualified type/enum/struct name to its
    /// canonical fq-name.
    pub fn canon(&self, name: &str) -> Option<String> {
        if self.structs.contains_key(name) || self.enums.contains_key(name) {
            return Some(name.to_string());
        }
        // last-segment match
        let last = name.rsplit("::").next().unwrap();
        for k in self.structs.keys().chain(self.enums.keys()) {
            if k.rsplit("::").next().unwrap() == last {
                return Some(k.clone());
            }
        }
        None
    }
}

/// The base (head) type name of a surface type: `Vec<i64>` → `"Vec"`,
/// `i64` → `"i64"`. Used to key the method index by receiver type.
pub fn type_base_name(t: &Type) -> String {
    match &t.kind {
        TypeKind::Path(p, _) => p.last().to_string(),
        TypeKind::SelfType => "Self".to_string(),
        TypeKind::Tuple(_) => "(tuple)".to_string(),
        TypeKind::Array(..) => "(array)".to_string(),
        TypeKind::Fn(..) => "(fn)".to_string(),
    }
}

/// Convert a surface `Type` (with generic params in scope) to a semantic `Ty`.
pub fn lower_type(t: &Type, generics: &[String], ctx: &TyCtx) -> TResult<Ty> {
    match &t.kind {
        TypeKind::Path(path, args) => {
            let name = path.last();
            if let Some(p) = prim_of(name) {
                return Ok(Ty::Prim(p));
            }
            if generics.iter().any(|g| g == name) {
                return Ok(Ty::Var(name.to_string()));
            }
            let targs: Vec<Ty> = args.iter().map(|a| lower_type(a, generics, ctx)).collect::<Result<_, _>>()?;
            // builtin generic containers stay as Named with their short name.
            if matches!(name, "Vec" | "Array" | "Option" | "Result") {
                return Ok(Ty::Named { name: name.to_string(), args: targs });
            }
            let canon = ctx.canon(name).unwrap_or_else(|| name.to_string());
            Ok(Ty::Named { name: canon, args: targs })
        }
        TypeKind::Tuple(tys) => {
            if tys.is_empty() {
                Ok(Ty::Prim(Prim::Unit))
            } else {
                Ok(Ty::Tuple(tys.iter().map(|t| lower_type(t, generics, ctx)).collect::<Result<_, _>>()?))
            }
        }
        TypeKind::Array(elem, count) => {
            let e = lower_type(elem, generics, ctx)?;
            let n = const_usize(count)?;
            Ok(Ty::Array(Box::new(e), n))
        }
        TypeKind::Fn(params, ret) => {
            let ps = params.iter().map(|p| lower_type(p, generics, ctx)).collect::<Result<_, _>>()?;
            let r = match ret {
                Some(r) => lower_type(r, generics, ctx)?,
                None => Ty::Prim(Prim::Unit),
            };
            Ok(Ty::Fn { params: ps, ret: Box::new(r) })
        }
        TypeKind::SelfType => Ok(Ty::Var("Self".to_string())),
    }
}

fn prim_of(name: &str) -> Option<Prim> {
    Some(match name {
        "i8" => Prim::I8, "i16" => Prim::I16, "i32" => Prim::I32, "i64" => Prim::I64,
        "u8" => Prim::U8, "u16" => Prim::U16, "u32" => Prim::U32, "u64" => Prim::U64,
        "f32" => Prim::F32, "f64" => Prim::F64,
        "bool" => Prim::Bool, "char" => Prim::Char, "String" => Prim::Str,
        _ => return None,
    })
}

fn const_usize(e: &Expr) -> TResult<u64> {
    match &*e.kind {
        ExprKind::Int(n, _) => Ok(*n),
        _ => Err(TypeError { msg: "array length must be an integer literal".into(), span: e.span }),
    }
}

/// The default scalar type for an unsuffixed/suffixed numeric literal.
pub fn suffix_prim(s: NumSuffix, is_float: bool) -> Prim {
    match s {
        NumSuffix::I8 => Prim::I8, NumSuffix::I16 => Prim::I16,
        NumSuffix::I32 => Prim::I32, NumSuffix::I64 => Prim::I64,
        NumSuffix::U8 => Prim::U8, NumSuffix::U16 => Prim::U16,
        NumSuffix::U32 => Prim::U32, NumSuffix::U64 => Prim::U64,
        NumSuffix::F32 => Prim::F32, NumSuffix::F64 => Prim::F64,
        NumSuffix::None => if is_float { Prim::F64 } else { Prim::I64 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;

    fn ctx(src: &str) -> TyCtx {
        let m = parse_module(&lex(src).unwrap()).unwrap();
        let r = resolve_module(m).unwrap();
        TyCtx::from_globals(&r.globals)
    }

    #[test]
    fn prims_and_named() {
        let c = ctx("struct Point { x: i64 } enum E { A, B }");
        let g: Vec<String> = vec![];
        let span = Span::new(0, 0);
        let ti = |k| Type { kind: k, span };
        assert_eq!(
            lower_type(&ti(TypeKind::Path(Path::single("i64".into(), span), vec![])), &g, &c).unwrap(),
            Ty::Prim(Prim::I64)
        );
        let pt = lower_type(&ti(TypeKind::Path(Path::single("Point".into(), span), vec![])), &g, &c).unwrap();
        assert!(matches!(pt, Ty::Named { name, .. } if name == "Point"));
    }

    #[test]
    fn generic_var() {
        let c = ctx("fn id<T>(x: T) -> T { x }");
        let g = vec!["T".to_string()];
        let span = Span::new(0, 0);
        let t = Type { kind: TypeKind::Path(Path::single("T".into(), span), vec![]), span };
        assert_eq!(lower_type(&t, &g, &c).unwrap(), Ty::Var("T".into()));
    }

    #[test]
    fn variants_indexed() {
        let c = ctx("enum Option<T> { None, Some(T) }");
        assert_eq!(c.variants.get("Option::Some").unwrap().1, 1);
        assert_eq!(c.variants.get("None").unwrap().1, 0);
    }
}
