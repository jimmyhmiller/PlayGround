//! Rust interop: value conversions, function registration, host types.
//!
//! ```ignore
//! let mut vm = Funct::new();
//! vm.register1("double", |x: i64| x * 2);
//! vm.register_type::<Player>("Player")
//!     .field("hp", |p| p.hp)
//!     .method1("damage", |p, n: i64| { p.hp -= n; });
//! vm.eval("player |> damage(3)")?;
//! ```
//!
//! Every registered function is a global `NativeFn` value, so it works as
//! `f(x)`, `x.f()` and `x |> f` with no extra registration.

use crate::value::shared::{HostBound, NativeBound, Sh};
use crate::value::{Value, VariantPayload};
use crate::vm::{Fault, Funct, FunctError, HostFn, NativeEntry};
use std::collections::BTreeMap;
use std::marker::PhantomData;

// ---------- conversions ----------

pub trait ToValue {
    fn to_value(self) -> Value;
}

pub trait FromValue: Sized {
    fn from_value(v: Value) -> Result<Self, Fault>;
}

fn conv_err<T>(expected: &str, got: &Value) -> Result<T, Fault> {
    Err(Fault::new(format!("expected {}, got {}", expected, got.type_name())))
}

impl ToValue for Value {
    fn to_value(self) -> Value {
        self
    }
}
impl FromValue for Value {
    fn from_value(v: Value) -> Result<Self, Fault> {
        Ok(v)
    }
}

impl ToValue for () {
    fn to_value(self) -> Value {
        Value::Unit
    }
}
impl FromValue for () {
    fn from_value(v: Value) -> Result<Self, Fault> {
        match v {
            Value::Unit => Ok(()),
            other => conv_err("Unit", &other),
        }
    }
}

impl ToValue for bool {
    fn to_value(self) -> Value {
        Value::Bool(self)
    }
}
impl FromValue for bool {
    fn from_value(v: Value) -> Result<Self, Fault> {
        match v {
            Value::Bool(b) => Ok(b),
            other => conv_err("Bool", &other),
        }
    }
}

macro_rules! int_conv {
    ($($t:ty),*) => {$(
        impl ToValue for $t {
            fn to_value(self) -> Value { Value::Int(self as i64) }
        }
        impl FromValue for $t {
            fn from_value(v: Value) -> Result<Self, Fault> {
                match v {
                    Value::Int(i) => <$t>::try_from(i)
                        .map_err(|_| Fault::new(format!("integer {} out of range for {}", i, stringify!($t)))),
                    other => conv_err("Int", &other),
                }
            }
        }
    )*};
}
int_conv!(i64, i32, i16, u8, u16, u32, usize);

impl ToValue for f64 {
    fn to_value(self) -> Value {
        Value::Float(self)
    }
}
impl FromValue for f64 {
    fn from_value(v: Value) -> Result<Self, Fault> {
        match v {
            Value::Float(f) => Ok(f),
            Value::Int(i) => Ok(i as f64),
            other => conv_err("Float", &other),
        }
    }
}

impl ToValue for f32 {
    fn to_value(self) -> Value {
        Value::Float(self as f64)
    }
}

impl ToValue for String {
    fn to_value(self) -> Value {
        Value::str(self)
    }
}
impl ToValue for &str {
    fn to_value(self) -> Value {
        Value::str(self)
    }
}
impl FromValue for String {
    fn from_value(v: Value) -> Result<Self, Fault> {
        match v {
            Value::Str(s) => Ok(s.to_string()),
            other => conv_err("Str", &other),
        }
    }
}

impl<T: ToValue> ToValue for Vec<T> {
    fn to_value(self) -> Value {
        Value::list_v(self.into_iter().map(|x| x.to_value()).collect())
    }
}
impl<T: FromValue> FromValue for Vec<T> {
    fn from_value(v: Value) -> Result<Self, Fault> {
        match v {
            Value::List(items) => items.iter().cloned().map(T::from_value).collect(),
            other => conv_err("List", &other),
        }
    }
}

impl<T: ToValue> ToValue for BTreeMap<String, T> {
    fn to_value(self) -> Value {
        Value::Record(self.into_iter().map(|(k, v)| (k, v.to_value())).collect())
    }
}
impl<T: FromValue> FromValue for BTreeMap<String, T> {
    fn from_value(v: Value) -> Result<Self, Fault> {
        match v {
            Value::Record(r) => r
                .iter()
                .map(|(k, v)| Ok((k.clone(), T::from_value(v.clone())?)))
                .collect(),
            other => conv_err("Record", &other),
        }
    }
}

impl<T: ToValue> ToValue for Option<T> {
    fn to_value(self) -> Value {
        match self {
            Some(x) => Value::some(x.to_value()),
            None => Value::none(),
        }
    }
}
impl<T: FromValue> FromValue for Option<T> {
    fn from_value(v: Value) -> Result<Self, Fault> {
        match &v {
            Value::Variant(var) => match (var.tag.as_str(), &var.payload) {
                ("Some", VariantPayload::Positional(p)) if p.len() == 1 => {
                    Ok(Some(T::from_value(p[0].clone())?))
                }
                ("None", _) => Ok(None),
                _ => conv_err("Some/None", &v),
            },
            _ => conv_err("Some/None", &v),
        }
    }
}

/// Rust `Result` ⇒ script `Ok(..)`/`Err(..)` and back (spec §9).
impl<T: ToValue, E: ToValue> ToValue for Result<T, E> {
    fn to_value(self) -> Value {
        match self {
            Ok(x) => Value::ok(x.to_value()),
            Err(e) => Value::err(e.to_value()),
        }
    }
}
impl<T: FromValue, E: FromValue> FromValue for Result<T, E> {
    fn from_value(v: Value) -> Result<Self, Fault> {
        match &v {
            Value::Variant(var) => match (var.tag.as_str(), &var.payload) {
                ("Ok", VariantPayload::Positional(p)) if p.len() == 1 => {
                    Ok(Ok(T::from_value(p[0].clone())?))
                }
                ("Err", VariantPayload::Positional(p)) if p.len() == 1 => {
                    Ok(Err(E::from_value(p[0].clone())?))
                }
                _ => conv_err("Ok/Err", &v),
            },
            _ => conv_err("Ok/Err", &v),
        }
    }
}

macro_rules! tuple_conv {
    ($(($($T:ident . $idx:tt),+ ; $n:expr)),+) => {$(
        impl<$($T: ToValue),+> ToValue for ($($T,)+) {
            fn to_value(self) -> Value {
                Value::tuple(vec![$(self.$idx.to_value()),+])
            }
        }
        impl<$($T: FromValue),+> FromValue for ($($T,)+) {
            fn from_value(v: Value) -> Result<Self, Fault> {
                match v {
                    Value::Tuple(items) if items.len() == $n => {
                        let mut it = items.iter().cloned();
                        Ok(($($T::from_value(it.next().unwrap())?,)+))
                    }
                    other => conv_err(concat!("Tuple of ", $n), &other),
                }
            }
        }
    )+};
}
tuple_conv!(
    (A.0, B.1; 2),
    (A.0, B.1, C.2; 3),
    (A.0, B.1, C.2, D.3; 4),
    (A.0, B.1, C.2, D.3, E.4; 5),
    (A.0, B.1, C.2, D.3, E.4, F.5; 6)
);

// ---------- function registration ----------

macro_rules! register_fns {
    ($(($method:ident, $($A:ident : $i:tt),*; $n:expr)),+) => {$(
        /// Register a Rust fn of this arity as a global script function.
        pub fn $method<$($A: FromValue + 'static,)* R: ToValue + 'static>(
            &mut self,
            name: &str,
            f: impl Fn($($A),*) -> R + HostBound,
        ) {
            self.register_raw(name, move |_vm, args: Vec<Value>| {
                if args.len() != $n {
                    return Err(Fault::new(format!(
                        "{} expects {} argument(s), got {}",
                        stringify!($method), $n, args.len()
                    )));
                }
                #[allow(unused_mut, unused_variables)]
                let mut it = args.into_iter();
                Ok(f($($A::from_value(it.next().unwrap())
                        .map_err(|e| Fault::new(format!("argument {}: {}", $i + 1, e.msg)))?),*)
                    .to_value())
            });
        }
    )+};
}

impl Funct {
    /// Register a raw native: full access to the engine (can call script
    /// closures via `vm.call_value`) and to untyped `Value`s.
    pub fn register_raw(&mut self, name: &str, f: impl HostFn) {
        let f: crate::vm::NativeImpl = Sh::new(f);
        let id = match self.native_ids.get(name) {
            Some(&id) => {
                self.natives[id as usize] = NativeEntry { name: name.to_string(), f };
                id
            }
            None => {
                let id = self.natives.len() as u32;
                self.natives.push(NativeEntry { name: name.to_string(), f });
                self.native_ids.insert(name.to_string(), id);
                id
            }
        };
        let g = self.ctx.ensure_global(name);
        self.ctx.shared.insert(g); // natives are visible inside modules
        self.sync_globals();
        self.globals[g as usize] = Some(Value::NativeFn(id));
    }

    register_fns!(
        (register0, ; 0),
        (register1, A: 0; 1),
        (register2, A: 0, B: 1; 2),
        (register3, A: 0, B: 1, C: 2; 3),
        (register4, A: 0, B: 1, C: 2, D: 3; 4),
        (register5, A: 0, B: 1, C: 2, D: 3, E: 4; 5),
        (register6, A: 0, B: 1, C: 2, D: 3, E: 4, F: 5; 6)
    );

    /// Register a host module: a global record whose fields are the module's
    /// functions, usable directly as `math.lerp(a, b, t)` and importable with
    /// `import { lerp } from "math"` / `import "math" as m`.
    pub fn register_module(&mut self, name: &str, fns: Vec<(&str, Value)>) {
        let mut map = BTreeMap::new();
        for (fname, v) in fns {
            map.insert(fname.to_string(), v);
        }
        let g = self.ctx.ensure_global(name);
        self.ctx.shared.insert(g); // host modules are visible inside modules
        self.sync_globals();
        self.globals[g as usize] = Some(Value::record(map));
        self.register_host_module(name, g);
    }

    /// Look up a registered native fn as a Value (for register_module).
    pub fn native_fn(&self, name: &str) -> Option<Value> {
        self.native_ids.get(name).map(|&id| Value::NativeFn(id))
    }

    /// Call a global function and convert the result.
    pub fn call_typed<R: FromValue>(&mut self, name: &str, args: Vec<Value>) -> Result<R, FunctError> {
        let v = self.call(name, args)?;
        R::from_value(v).map_err(FunctError::Fault)
    }

    /// Register a host type: field getters + UFCS methods (`T: Send`).
    pub fn register_type<T: NativeBound>(&mut self, name: &str) -> TypeBuilder<'_, T> {
        TypeBuilder { vm: self, type_name: name.to_string(), _p: PhantomData }
    }
}

/// Convert `Vec<impl ToValue>`-ish heterogeneous args: `vals![1, "x", 2.0]`.
#[macro_export]
macro_rules! vals {
    ($($e:expr),* $(,)?) => {
        vec![$($crate::interop::ToValue::to_value($e)),*]
    };
}

// ---------- host types ----------

pub struct TypeBuilder<'a, T: NativeBound> {
    vm: &'a mut Funct,
    type_name: String,
    _p: PhantomData<T>,
}

fn with_native<T: 'static, R>(
    type_name: &str,
    v: &Value,
    f: impl FnOnce(&mut T) -> R,
) -> Result<R, Fault> {
    match v {
        Value::Native(n) => n.with_mut(f).ok_or_else(|| {
            Fault::new(format!("expected native {}, got native {}", type_name, n.type_name))
        }),
        other => Err(Fault::new(format!(
            "expected native {}, got {}",
            type_name,
            other.type_name()
        ))),
    }
}

macro_rules! type_methods {
    ($(($method:ident, $($A:ident : $i:tt),*; $n:expr)),+) => {$(
        /// Register a method `fn(&mut T, ...) -> R` as a UFCS function:
        /// callable as `obj.name(args)`, `obj |> name(args)`, `name(obj, args)`.
        #[allow(non_snake_case)]
        pub fn $method<$($A: FromValue + 'static,)* R: ToValue + 'static>(
            self,
            name: &str,
            f: impl Fn(&mut T, $($A),*) -> R + HostBound,
        ) -> Self {
            let tname = self.type_name.clone();
            self.vm.register_raw(name, move |_vm, args: Vec<Value>| {
                if args.len() != $n + 1 {
                    return Err(Fault::new(format!(
                        "method expects {} argument(s) plus receiver, got {}",
                        $n, args.len().saturating_sub(1)
                    )));
                }
                #[allow(unused_mut, unused_variables)]
                let mut it = args.into_iter();
                let recv = it.next().unwrap();
                $(let $A = $A::from_value(it.next().unwrap())
                    .map_err(|e| Fault::new(format!("argument {}: {}", $i + 1, e.msg)))?;)*
                with_native::<T, R>(&tname, &recv, |t| f(t, $($A),*)).map(|r| r.to_value())
            });
            self
        }
    )+};
}

macro_rules! type_ctors {
    ($(($method:ident, $($A:ident : $i:tt),*; $n:expr)),+) => {$(
        /// Register a constructor returning the host type as a Native value.
        #[allow(non_snake_case)]
        pub fn $method<$($A: FromValue + 'static),*>(
            self,
            name: &str,
            f: impl Fn($($A),*) -> T + HostBound,
        ) -> Self {
            let tname = self.type_name.clone();
            self.vm.register_raw(name, move |_vm, args: Vec<Value>| {
                if args.len() != $n {
                    return Err(Fault::new(format!(
                        "constructor expects {} argument(s), got {}",
                        $n, args.len()
                    )));
                }
                #[allow(unused_mut, unused_variables)]
                let mut it = args.into_iter();
                $(let $A = $A::from_value(it.next().unwrap())
                    .map_err(|e| Fault::new(format!("argument {}: {}", $i + 1, e.msg)))?;)*
                Ok(Value::native(&tname, f($($A),*)))
            });
            self
        }
    )+};
}

#[allow(non_snake_case)]
impl<'a, T: NativeBound> TypeBuilder<'a, T> {
    /// Register a field getter: `obj.name` in script.
    pub fn field<R: ToValue + 'static>(self, name: &str, get: impl Fn(&T) -> R + HostBound) -> Self {
        let key = (self.type_name.clone(), name.to_string());
        let tname = self.type_name.clone();
        let getter: crate::vm::Getter =
            Sh::new(move |_vm: &mut Funct, v: &Value| with_native::<T, R>(&tname, v, |t| get(t)).map(|r| r.to_value()));
        self.vm.getters.insert(key, getter);
        self
    }

    type_methods!(
        (method0, ; 0),
        (method1, A: 0; 1),
        (method2, A: 0, B: 1; 2),
        (method3, A: 0, B: 1, C: 2; 3),
        (method4, A: 0, B: 1, C: 2, D: 3; 4),
        (method5, A: 0, B: 1, C: 2, D: 3, E: 4; 5)
    );

    type_ctors!(
        (ctor0, ; 0),
        (ctor1, A: 0; 1),
        (ctor2, A: 0, B: 1; 2),
        (ctor3, A: 0, B: 1, C: 2; 3),
        (ctor4, A: 0, B: 1, C: 2, D: 3; 4),
        (ctor5, A: 0, B: 1, C: 2, D: 3, E: 4; 5)
    );
}
