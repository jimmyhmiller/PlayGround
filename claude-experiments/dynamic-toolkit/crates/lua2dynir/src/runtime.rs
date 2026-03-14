/// Lua runtime functions operating on NanBox-encoded u64 values.
///
/// Tag scheme:
///   - Unboxed float: normal f64 (not a tagged NaN)
///   - Tag 0: GC pointer (tables, strings, closures) — 48-bit payload
///   - Tag 1: nil (payload 0)
///   - Tag 2: boolean (payload 0=false, 1=true)
///   - Tag 3: string constant index / built-in function ID

use std::collections::HashMap;

use dynvalue::{NanBox, TagScheme};

// NanBox tag constants
const TAG_PTR: u32 = 0;
const TAG_NIL: u32 = 1;
const TAG_BOOL: u32 = 2;
const TAG_INTERN: u32 = 3;

pub fn make_nil() -> u64 {
    NanBox::encode_tagged(TAG_NIL, 0)
}

pub fn make_bool(b: bool) -> u64 {
    NanBox::encode_tagged(TAG_BOOL, b as u64)
}

pub fn make_number(f: f64) -> u64 {
    NanBox::encode_float(f)
}

fn is_nil(v: u64) -> bool {
    NanBox::has_tag(v, TAG_NIL)
}

fn is_bool(v: u64) -> bool {
    NanBox::has_tag(v, TAG_BOOL)
}

fn is_number(v: u64) -> bool {
    NanBox::is_float(v)
}

fn is_intern(v: u64) -> bool {
    NanBox::has_tag(v, TAG_INTERN)
}

fn as_number(v: u64) -> f64 {
    f64::from_bits(v)
}

fn as_bool_payload(v: u64) -> bool {
    NanBox::extract_payload(v) != 0
}

fn is_truthy(v: u64) -> bool {
    if is_nil(v) { return false; }
    if is_bool(v) { return as_bool_payload(v); }
    true
}

const CLOSURE_MARKER: u64 = 0xC105_C105_C105_C105;

/// Closure layout (raw bytes, inline upvalues for direct IR struct access):
///   offset 0:  marker     (u64) = CLOSURE_MARKER
///   offset 8:  func_id    (u64) — index into function table
///   offset 16: num_upvals (u64)
///   offset 24: upval[0]   (u64) — NanBox-encoded
///   offset 32: upval[1]   (u64)
///   ...
pub fn make_closure(func_id: usize, upvalues: &[u64]) -> u64 {
    let total_size = 24 + upvalues.len() * 8;
    let layout = std::alloc::Layout::from_size_align(total_size, 8).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) as *mut u64 };
    unsafe {
        *ptr = CLOSURE_MARKER;
        *ptr.add(1) = func_id as u64;
        *ptr.add(2) = upvalues.len() as u64;
        for (i, &v) in upvalues.iter().enumerate() {
            *ptr.add(3 + i) = v;
        }
    }
    NanBox::encode_tagged(TAG_PTR, ptr as u64 & 0x0000_FFFF_FFFF_FFFF)
}

pub fn is_closure(v: u64) -> bool {
    if !NanBox::has_tag(v, TAG_PTR) { return false; }
    let payload = NanBox::extract_payload(v);
    if payload == 0 { return false; }
    let ptr = payload as *const u64;
    unsafe { *ptr == CLOSURE_MARKER }
}

pub fn closure_func_id(v: u64) -> Option<usize> {
    if !is_closure(v) { return None; }
    let payload = NanBox::extract_payload(v);
    let ptr = payload as *const u64;
    Some(unsafe { *ptr.add(1) } as usize)
}

pub fn closure_get_upvalue(v: u64, idx: usize) -> u64 {
    let payload = NanBox::extract_payload(v);
    let ptr = payload as *const u64;
    unsafe { *ptr.add(3 + idx) }
}

fn as_lua_string(v: u64) -> Option<&'static str> {
    if !NanBox::has_tag(v, TAG_PTR) { return None; }
    let payload = NanBox::extract_payload(v);
    if payload == 0 { return None; }
    let ptr = payload as *const u64;
    let first_word = unsafe { *ptr };
    if first_word == CLOSURE_MARKER { return None; }
    if first_word == STRING_MARKER {
        let str_ptr = payload as *const StringRepr;
        Some(unsafe { (*str_ptr).data.as_str() })
    } else {
        None
    }
}

const STRING_MARKER: u64 = 0x5754_5754_5754_5754;

pub fn make_string(s: String) -> u64 {
    let layout = std::alloc::Layout::new::<StringRepr>();
    let ptr = unsafe { std::alloc::alloc(layout) as *mut StringRepr };
    unsafe {
        (*ptr).marker = STRING_MARKER;
        // Use ptr::write to avoid dropping uninitialized memory
        std::ptr::addr_of_mut!((*ptr).data).write(s);
    }
    NanBox::encode_tagged(TAG_PTR, ptr as u64 & 0x0000_FFFF_FFFF_FFFF)
}

#[repr(C)]
struct StringRepr {
    marker: u64,
    data: String,
}

/// The Lua runtime state — a bag of functions, globals, and constants.
/// No interpreter logic — that stays in the test harness using DynIR's Interpreter.
pub struct LuaRuntime {
    globals: HashMap<String, u64>,
    pub constants: Vec<String>,
    /// Register file snapshot for passing args to calls
    pub register_file: Vec<u64>,
    /// Captured output from print
    pub output: String,
}

impl LuaRuntime {
    pub fn new(constants: &[crate::bytecode::Constant]) -> Self {
        let mut string_constants = Vec::new();
        for c in constants {
            match c {
                crate::bytecode::Constant::String(s) => string_constants.push(s.clone()),
                _ => string_constants.push(String::new()),
            }
        }

        let mut rt = LuaRuntime {
            globals: HashMap::new(),
            constants: string_constants,
            register_file: Vec::new(),
            output: String::new(),
        };

        rt.init_stdlib();
        rt
    }

    fn init_stdlib(&mut self) {
        self.globals.insert("print".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0000));
        self.globals.insert("type".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0001));
        self.globals.insert("tostring".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0002));
        self.globals.insert("tonumber".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0003));
        self.globals.insert("pairs".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0004));
        self.globals.insert("ipairs".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0005));
        self.globals.insert("assert".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0006));
        self.globals.insert("error".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0007));
        self.globals.insert("pcall".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0008));
        self.globals.insert("select".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0009));
        self.globals.insert("unpack".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000A));
        self.globals.insert("rawget".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000B));
        self.globals.insert("rawset".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000C));
        self.globals.insert("setmetatable".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000D));
        self.globals.insert("getmetatable".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000E));

        self.globals.insert("math".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0003_0000_0000));
        self.globals.insert("string".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0003_0000_0001));
        self.globals.insert("table".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0003_0000_0002));
        self.globals.insert("io".to_string(), NanBox::encode_tagged(TAG_INTERN, 0x0003_0000_0003));
    }

    pub fn resolve_string(&self, v: u64) -> Option<String> {
        if is_intern(v) {
            let payload = NanBox::extract_payload(v);
            let high = (payload >> 32) as u32;
            if high == 0 {
                let idx = payload as usize;
                if idx < self.constants.len() {
                    return Some(self.constants[idx].clone());
                }
            }
        }
        if let Some(s) = as_lua_string(v) {
            return Some(s.to_string());
        }
        None
    }

    pub fn value_to_string(&self, v: u64) -> String {
        if is_nil(v) {
            "nil".to_string()
        } else if is_bool(v) {
            if as_bool_payload(v) { "true".to_string() } else { "false".to_string() }
        } else if is_number(v) {
            let n = as_number(v);
            if n == n.floor() && n.abs() < 1e15 && !n.is_infinite() {
                format!("{}", n as i64)
            } else {
                format!("{}", n)
            }
        } else if let Some(s) = self.resolve_string(v) {
            s
        } else if is_closure(v) {
            "function".to_string()
        } else if is_intern(v) {
            let payload = NanBox::extract_payload(v);
            format!("function: 0x{:x}", payload)
        } else {
            format!("userdata: 0x{:016x}", v)
        }
    }

    // ── Arithmetic ─────────────────────────────────────────────

    pub fn lua_add(&self, a: u64, b: u64) -> u64 {
        if is_number(a) && is_number(b) {
            make_number(as_number(a) + as_number(b))
        } else {
            panic!("attempt to perform arithmetic on non-number values");
        }
    }

    pub fn lua_sub(&self, a: u64, b: u64) -> u64 {
        if is_number(a) && is_number(b) {
            make_number(as_number(a) - as_number(b))
        } else {
            panic!("attempt to perform arithmetic on non-number values");
        }
    }

    pub fn lua_mul(&self, a: u64, b: u64) -> u64 {
        if is_number(a) && is_number(b) {
            make_number(as_number(a) * as_number(b))
        } else {
            panic!("attempt to perform arithmetic on non-number values");
        }
    }

    pub fn lua_div(&self, a: u64, b: u64) -> u64 {
        if is_number(a) && is_number(b) {
            make_number(as_number(a) / as_number(b))
        } else {
            panic!("attempt to perform arithmetic on non-number values");
        }
    }

    pub fn lua_mod(&self, a: u64, b: u64) -> u64 {
        if is_number(a) && is_number(b) {
            let na = as_number(a);
            let nb = as_number(b);
            make_number(na - (na / nb).floor() * nb)
        } else {
            panic!("attempt to perform arithmetic on non-number values");
        }
    }

    pub fn lua_pow(&self, a: u64, b: u64) -> u64 {
        if is_number(a) && is_number(b) {
            make_number(as_number(a).powf(as_number(b)))
        } else {
            panic!("attempt to perform arithmetic on non-number values");
        }
    }

    pub fn lua_unm(&self, a: u64) -> u64 {
        if is_number(a) {
            make_number(-as_number(a))
        } else {
            panic!("attempt to perform arithmetic on a non-number value");
        }
    }

    pub fn lua_not(&self, a: u64) -> u64 {
        make_bool(!is_truthy(a))
    }

    pub fn lua_len(&self, a: u64) -> u64 {
        if let Some(s) = self.resolve_string(a) {
            make_number(s.len() as f64)
        } else {
            make_number(0.0)
        }
    }

    // ── Comparison ─────────────────────────────────────────────

    pub fn lua_eq(&self, a: u64, b: u64) -> u64 {
        let result = if is_nil(a) && is_nil(b) {
            true
        } else if is_nil(a) || is_nil(b) {
            false
        } else if is_bool(a) && is_bool(b) {
            as_bool_payload(a) == as_bool_payload(b)
        } else if is_number(a) && is_number(b) {
            as_number(a) == as_number(b)
        } else {
            a == b
        };
        make_bool(result)
    }

    pub fn lua_lt(&self, a: u64, b: u64) -> u64 {
        if is_number(a) && is_number(b) {
            make_bool(as_number(a) < as_number(b))
        } else {
            panic!("attempt to compare non-number values");
        }
    }

    pub fn lua_le(&self, a: u64, b: u64) -> u64 {
        if is_number(a) && is_number(b) {
            make_bool(as_number(a) <= as_number(b))
        } else {
            panic!("attempt to compare non-number values");
        }
    }

    // ── String ─────────────────────────────────────────────────

    pub fn lua_concat(&self, a: u64, b: u64) -> u64 {
        let sa = self.value_to_string(a);
        let sb = self.value_to_string(b);
        make_string(format!("{}{}", sa, sb))
    }

    // ── Globals ────────────────────────────────────────────────

    pub fn lua_getglobal(&self, name: u64) -> u64 {
        if let Some(name_str) = self.resolve_string(name) {
            self.globals.get(&name_str).copied().unwrap_or_else(make_nil)
        } else {
            make_nil()
        }
    }

    pub fn lua_setglobal(&mut self, name: u64, val: u64) {
        if let Some(name_str) = self.resolve_string(name) {
            self.globals.insert(name_str, val);
        }
    }

    // ── Tables ─────────────────────────────────────────────────

    pub fn lua_newtable(&mut self) -> u64 {
        let table = Box::new(LuaTable::new());
        let ptr = Box::into_raw(table) as u64;
        NanBox::encode_tagged(TAG_PTR, ptr & 0x0000_FFFF_FFFF_FFFF)
    }

    pub fn lua_gettable(&self, table: u64, key: u64) -> u64 {
        if NanBox::has_tag(table, TAG_PTR) && !is_closure(table) {
            let ptr = NanBox::extract_payload(table) as *const LuaTable;
            if !ptr.is_null() {
                let t = unsafe { &*ptr };
                return t.get(key);
            }
        }
        if is_intern(table) {
            let payload = NanBox::extract_payload(table);
            let high = (payload >> 32) as u32;
            if high == 0x0003 {
                if let Some(field_name) = self.resolve_string(key) {
                    let lib_id = payload as u32;
                    return self.get_lib_field(lib_id, &field_name);
                }
            }
        }
        make_nil()
    }

    pub fn lua_settable(&self, table: u64, key: u64, val: u64) {
        if NanBox::has_tag(table, TAG_PTR) && !is_closure(table) {
            let ptr = NanBox::extract_payload(table) as *mut LuaTable;
            if !ptr.is_null() {
                let t = unsafe { &mut *ptr };
                t.set(key, val);
            }
        }
    }

    pub fn lua_setlist_from_regfile(&self, table: u64, base: usize, offset: usize, count: usize) {
        if NanBox::has_tag(table, TAG_PTR) && !is_closure(table) {
            let ptr = NanBox::extract_payload(table) as *mut LuaTable;
            if !ptr.is_null() {
                let t = unsafe { &mut *ptr };
                let actual_count = if count == 0 {
                    self.register_file.len().saturating_sub(base)
                } else {
                    count
                };
                for i in 0..actual_count {
                    if base + i < self.register_file.len() {
                        let key = make_number((offset + i + 1) as f64);
                        t.set(key, self.register_file[base + i]);
                    }
                }
            }
        }
    }

    fn get_lib_field(&self, lib_id: u32, field: &str) -> u64 {
        match lib_id {
            0 => match field {
                "sqrt" => NanBox::encode_tagged(TAG_INTERN, 0x0004_0000_0000),
                "abs" => NanBox::encode_tagged(TAG_INTERN, 0x0004_0000_0001),
                "floor" => NanBox::encode_tagged(TAG_INTERN, 0x0004_0000_0002),
                "ceil" => NanBox::encode_tagged(TAG_INTERN, 0x0004_0000_0003),
                "sin" => NanBox::encode_tagged(TAG_INTERN, 0x0004_0000_0004),
                "cos" => NanBox::encode_tagged(TAG_INTERN, 0x0004_0000_0005),
                "max" => NanBox::encode_tagged(TAG_INTERN, 0x0004_0000_0006),
                "min" => NanBox::encode_tagged(TAG_INTERN, 0x0004_0000_0007),
                "pi" => make_number(std::f64::consts::PI),
                "huge" => make_number(f64::INFINITY),
                _ => make_nil(),
            },
            _ => make_nil(),
        }
    }

    // ── Calls ──────────────────────────────────────────────────

    /// Call a Lua function value. For built-in functions, dispatch here.
    /// For closures, the caller (test harness) handles it via call_handler.
    /// Call a built-in function. Closures are dispatched externally by the test harness.
    pub fn lua_call(&mut self, func: u64, base_reg: usize, nargs: usize) -> u64 {
        if is_intern(func) {
            let payload = NanBox::extract_payload(func);
            let high = (payload >> 32) as u32;
            let low = payload as u32;

            if high == 0x0002 {
                return self.call_builtin(low, base_reg, nargs);
            } else if high == 0x0004 {
                return self.call_math(low, base_reg, nargs);
            }
        }
        panic!("attempt to call a {} value (closures must be dispatched externally)",
            if is_nil(func) { "nil" } else { "non-function" });
    }

    fn call_builtin(&mut self, id: u32, base_reg: usize, nargs: usize) -> u64 {
        match id {
            0 => {
                let mut parts = Vec::new();
                for i in 0..nargs {
                    parts.push(self.value_to_string(self.register_file[base_reg + i]));
                }
                let line = parts.join("\t");
                self.output.push_str(&line);
                self.output.push('\n');
                make_nil()
            }
            6 => {
                if nargs < 1 { panic!("assertion failed!"); }
                let v = self.register_file[base_reg];
                if !is_truthy(v) { panic!("assertion failed!"); }
                v
            }
            7 => panic!("error raised"),
            3 => {
                if nargs < 1 { return make_nil(); }
                let v = self.register_file[base_reg];
                if is_number(v) { v } else { make_nil() }
            }
            _ => make_nil(),
        }
    }

    fn call_math(&self, id: u32, base_reg: usize, nargs: usize) -> u64 {
        if nargs < 1 { return make_nil(); }
        let a = self.register_file[base_reg];
        if !is_number(a) { panic!("bad argument to math function"); }
        let na = as_number(a);

        match id {
            0 => make_number(na.sqrt()),
            1 => make_number(na.abs()),
            2 => make_number(na.floor()),
            3 => make_number(na.ceil()),
            4 => make_number(na.sin()),
            5 => make_number(na.cos()),
            6 => {
                if nargs < 2 { return a; }
                let b = self.register_file[base_reg + 1];
                if is_number(b) { make_number(na.max(as_number(b))) } else { a }
            }
            7 => {
                if nargs < 2 { return a; }
                let b = self.register_file[base_reg + 1];
                if is_number(b) { make_number(na.min(as_number(b))) } else { a }
            }
            _ => make_nil(),
        }
    }

    // ── For loops ──────────────────────────────────────────────

    pub fn lua_forprep(&self, init: u64, _limit: u64, step: u64) -> u64 {
        if is_number(init) && is_number(step) {
            make_number(as_number(init) - as_number(step))
        } else {
            panic!("'for' initial value must be a number");
        }
    }

    pub fn lua_forloop(&self, index: u64, limit: u64, step: u64) -> u64 {
        if is_number(index) && is_number(limit) && is_number(step) {
            let idx = as_number(index) + as_number(step);
            let lim = as_number(limit);
            let stp = as_number(step);
            let continue_loop = if stp > 0.0 { idx <= lim } else { idx >= lim };
            if continue_loop { make_number(idx) } else { make_nil() }
        } else {
            panic!("'for' values must be numbers");
        }
    }

    pub fn lua_is_nil(&self, v: u64) -> u64 {
        make_bool(is_nil(v))
    }
}

/// Simple Lua table using a HashMap<u64, u64>.
pub struct LuaTable {
    hash: HashMap<u64, u64>,
    array: Vec<u64>,
}

impl LuaTable {
    fn new() -> Self {
        LuaTable {
            hash: HashMap::new(),
            array: Vec::new(),
        }
    }

    fn get(&self, key: u64) -> u64 {
        if is_number(key) {
            let n = as_number(key);
            if n == n.floor() && n >= 1.0 {
                let idx = n as usize;
                if idx <= self.array.len() && idx > 0 {
                    return self.array[idx - 1];
                }
            }
        }
        self.hash.get(&key).copied().unwrap_or_else(make_nil)
    }

    fn set(&mut self, key: u64, val: u64) {
        if is_number(key) {
            let n = as_number(key);
            if n == n.floor() && n >= 1.0 {
                let idx = n as usize;
                if idx <= self.array.len() + 1 {
                    while self.array.len() < idx {
                        self.array.push(make_nil());
                    }
                    self.array[idx - 1] = val;
                    return;
                }
            }
        }
        if is_nil(val) {
            self.hash.remove(&key);
        } else {
            self.hash.insert(key, val);
        }
    }
}
