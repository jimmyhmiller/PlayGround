/// Lua runtime functions operating on NanBox-encoded u64 values.
///
/// Tag scheme:
///   - Unboxed float: normal f64 (not a tagged NaN)
///   - Tag 0: GC pointer (tables, strings, closures) — 48-bit payload
///   - Tag 1: nil (payload 0)
///   - Tag 2: boolean (payload 0=false, 1=true)
///   - Tag 3: string constant index / built-in function ID
use std::collections::HashMap;

use dynalloc::Heap;
use dynobj::{Compact, ObjHeader, TypeInfo};
use dynobj::{
    raw_data_mut, read_raw_bytes, read_type_id, read_value_field, read_varlen_bytes,
    read_varlen_value, write_value_field, write_varlen_value,
};
use dynruntime::{ScopedJitRoot, ScopedJitRoots};
use dynvalue::{NanBox, TagScheme, Value};

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
    if is_nil(v) {
        return false;
    }
    if is_bool(v) {
        return as_bool_payload(v);
    }
    true
}

// ── GC TypeInfo statics ──────────────────────────────────────

// Numeric type ids used to discriminate objects at runtime.
const TYPE_ID_CLOSURE: u16 = 0;
const TYPE_ID_STRING: u16 = 1;
const TYPE_ID_TABLE: u16 = 2;
const TYPE_ID_ARRAY: u16 = 3;

/// Closure: Compact header + func_id (raw u64) + varlen upvalues (GC-traced)
static CLOSURE_TYPE: TypeInfo = TypeInfo::for_header(Compact::SIZE)
    .with_type_id(TYPE_ID_CLOSURE)
    .with_raw_bytes(8)
    .with_varlen_values(0);

/// String: Compact header + varlen bytes (string content)
static STRING_TYPE: TypeInfo = TypeInfo::for_header(Compact::SIZE)
    .with_type_id(TYPE_ID_STRING)
    .with_varlen_bytes(0);

/// Table header: Compact header + 2 value fields (hash ptr, array ptr) + 32 raw bytes (metadata)
static TABLE_TYPE: TypeInfo = TypeInfo::for_header(Compact::SIZE)
    .with_type_id(TYPE_ID_TABLE)
    .with_fields(2)
    .with_raw_bytes(32);

/// Array of GC-traced values (used for hash entries and array parts of tables)
static ARRAY_TYPE: TypeInfo = TypeInfo::for_header(Compact::SIZE)
    .with_type_id(TYPE_ID_ARRAY)
    .with_varlen_values(0);

/// Returns the TypeInfo table for the Lua runtime. Ordered by type_id so
/// the GC can look up each TypeInfo by the id stored in an object's header.
pub fn type_table() -> Vec<TypeInfo> {
    vec![CLOSURE_TYPE, STRING_TYPE, TABLE_TYPE, ARRAY_TYPE]
}

// ── Object type discrimination ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjType {
    Closure,
    String,
    Table,
    Unknown,
}

fn obj_type_from_ptr(ptr: *const u8) -> ObjType {
    let tid = unsafe { read_type_id(ptr, Compact::TYPE_ID_OFFSET) };
    match tid {
        TYPE_ID_CLOSURE => ObjType::Closure,
        TYPE_ID_STRING => ObjType::String,
        TYPE_ID_TABLE => ObjType::Table,
        _ => ObjType::Unknown,
    }
}

fn encode_gc_ptr(ptr: *const u8) -> u64 {
    NanBox::encode_tagged(TAG_PTR, ptr as u64 & 0x0000_FFFF_FFFF_FFFF)
}

// ── Closures ──────────────────────────────────────────────────

/// Allocate a closure on the GC heap.
///
/// Layout: [Compact header: 8B] [func_id: 8B raw] [varlen_count: 8B] [upval0: 8B] ...
pub fn make_closure(heap: &Heap, func_id: usize, upvalues: &[u64]) -> u64 {
    let ptr = heap.alloc_obj::<Compact>(&CLOSURE_TYPE, upvalues.len());
    assert!(
        !ptr.is_null(),
        "GC heap exhausted during closure allocation"
    );
    unsafe {
        let raw = raw_data_mut(ptr, &CLOSURE_TYPE);
        raw[0..8].copy_from_slice(&(func_id as u64).to_ne_bytes());
        for (i, &v) in upvalues.iter().enumerate() {
            write_varlen_value::<NanBox>(ptr, &CLOSURE_TYPE, i, Value::from_bits(v));
        }
    }
    encode_gc_ptr(ptr)
}

pub(crate) fn make_closure_from_roots(
    heap: &Heap,
    func_id: usize,
    roots: &ScopedJitRoots,
    upvalues: &[ScopedJitRoot],
) -> u64 {
    let ptr = heap.alloc_obj::<Compact>(&CLOSURE_TYPE, upvalues.len());
    assert!(
        !ptr.is_null(),
        "GC heap exhausted during closure allocation"
    );
    unsafe {
        let raw = raw_data_mut(ptr, &CLOSURE_TYPE);
        raw[0..8].copy_from_slice(&(func_id as u64).to_ne_bytes());
        for (i, root) in upvalues.iter().copied().enumerate() {
            write_varlen_value::<NanBox>(ptr, &CLOSURE_TYPE, i, Value::from_bits(roots.get(root)));
        }
    }
    encode_gc_ptr(ptr)
}

pub fn is_closure(v: u64) -> bool {
    if !NanBox::has_tag(v, TAG_PTR) {
        return false;
    }
    let payload = NanBox::extract_payload(v);
    if payload == 0 {
        return false;
    }
    obj_type_from_ptr(payload as *const u8) == ObjType::Closure
}

pub fn closure_func_id(v: u64) -> Option<usize> {
    if !is_closure(v) {
        return None;
    }
    let payload = NanBox::extract_payload(v);
    let ptr = payload as *const u8;
    let raw = unsafe { read_raw_bytes(ptr, &CLOSURE_TYPE) };
    Some(u64::from_ne_bytes(raw[0..8].try_into().unwrap()) as usize)
}

pub fn closure_get_upvalue(v: u64, idx: usize) -> u64 {
    let payload = NanBox::extract_payload(v);
    let ptr = payload as *const u8;
    unsafe { read_varlen_value::<NanBox>(ptr, &CLOSURE_TYPE, idx).to_bits() }
}

// ── Strings ───────────────────────────────────────────────────

/// Allocate a string on the GC heap.
///
/// Layout: [Compact header: 8B] [varlen_count/len: 8B] [bytes...] [padding]
fn make_string_on_heap(heap: &Heap, s: &str) -> u64 {
    let bytes = s.as_bytes();
    let ptr = heap.alloc_obj::<Compact>(&STRING_TYPE, bytes.len());
    assert!(!ptr.is_null(), "GC heap exhausted during string allocation");
    unsafe {
        let base = STRING_TYPE.varlen_count_offset() + 8;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(base), bytes.len());
    }
    encode_gc_ptr(ptr)
}

fn is_gc_string(v: u64) -> bool {
    if !NanBox::has_tag(v, TAG_PTR) {
        return false;
    }
    let payload = NanBox::extract_payload(v);
    if payload == 0 {
        return false;
    }
    obj_type_from_ptr(payload as *const u8) == ObjType::String
}

fn as_lua_string(v: u64) -> Option<&'static str> {
    if !is_gc_string(v) {
        return None;
    }
    let payload = NanBox::extract_payload(v);
    let ptr = payload as *const u8;
    let bytes = unsafe { read_varlen_bytes(ptr, &STRING_TYPE) };
    // Safety: we only create strings from valid UTF-8 (&str)
    Some(unsafe { std::str::from_utf8_unchecked(bytes) })
}

fn is_table(v: u64) -> bool {
    if !NanBox::has_tag(v, TAG_PTR) {
        return false;
    }
    let payload = NanBox::extract_payload(v);
    if payload == 0 {
        return false;
    }
    obj_type_from_ptr(payload as *const u8) == ObjType::Table
}

// ── Table helpers ─────────────────────────────────────────────

fn table_read_meta(ptr: *const u8) -> (usize, usize, usize, usize) {
    let raw = unsafe { read_raw_bytes(ptr, &TABLE_TYPE) };
    let hash_count = u64::from_ne_bytes(raw[0..8].try_into().unwrap()) as usize;
    let hash_cap = u64::from_ne_bytes(raw[8..16].try_into().unwrap()) as usize;
    let array_len = u64::from_ne_bytes(raw[16..24].try_into().unwrap()) as usize;
    let array_cap = u64::from_ne_bytes(raw[24..32].try_into().unwrap()) as usize;
    (hash_count, hash_cap, array_len, array_cap)
}

fn table_write_meta(
    ptr: *mut u8,
    hash_count: usize,
    hash_cap: usize,
    array_len: usize,
    array_cap: usize,
) {
    let raw = unsafe { raw_data_mut(ptr, &TABLE_TYPE) };
    raw[0..8].copy_from_slice(&(hash_count as u64).to_ne_bytes());
    raw[8..16].copy_from_slice(&(hash_cap as u64).to_ne_bytes());
    raw[16..24].copy_from_slice(&(array_len as u64).to_ne_bytes());
    raw[24..32].copy_from_slice(&(array_cap as u64).to_ne_bytes());
}

fn table_get_hash_ptr(ptr: *const u8) -> Option<*mut u8> {
    let val = unsafe { read_value_field::<NanBox>(ptr, &TABLE_TYPE, 0) };
    let bits = val.to_bits();
    if is_nil(bits) {
        return None;
    }
    Some(NanBox::extract_payload(bits) as *mut u8)
}

fn table_get_array_ptr(ptr: *const u8) -> Option<*mut u8> {
    let val = unsafe { read_value_field::<NanBox>(ptr, &TABLE_TYPE, 1) };
    let bits = val.to_bits();
    if is_nil(bits) {
        return None;
    }
    Some(NanBox::extract_payload(bits) as *mut u8)
}

fn table_ptr_from_boxed(table: u64) -> Option<*mut u8> {
    if !is_table(table) {
        return None;
    }
    Some(NanBox::extract_payload(table) as *mut u8)
}

fn hash_u64(key: u64) -> usize {
    key.wrapping_mul(0x517cc1b727220a95) as usize
}

/// Allocate a GC array of values, initialized to nil.
fn alloc_nil_values(heap: &Heap, count: usize) -> *mut u8 {
    let ptr = heap.alloc_obj::<Compact>(&ARRAY_TYPE, count);
    assert!(!ptr.is_null(), "GC heap exhausted during array allocation");
    let nil_val = Value::<NanBox>::from_bits(make_nil());
    for i in 0..count {
        unsafe {
            write_varlen_value::<NanBox>(ptr, &ARRAY_TYPE, i, nil_val);
        }
    }
    ptr
}

fn table_get(table_ptr: *const u8, key: u64) -> u64 {
    let (_, hash_cap, array_len, _) = table_read_meta(table_ptr);

    // Check array part for positive integer keys
    if is_number(key) {
        let n = as_number(key);
        if n == n.floor() && n >= 1.0 {
            let idx = n as usize;
            if idx <= array_len && idx > 0 {
                if let Some(arr) = table_get_array_ptr(table_ptr) {
                    return unsafe {
                        read_varlen_value::<NanBox>(arr, &ARRAY_TYPE, idx - 1).to_bits()
                    };
                }
            }
        }
    }

    // Check hash part
    if hash_cap == 0 {
        return make_nil();
    }
    let hash_arr = match table_get_hash_ptr(table_ptr) {
        Some(p) => p,
        None => return make_nil(),
    };

    let nil = make_nil();
    let mask = hash_cap - 1;
    let mut slot = hash_u64(key) & mask;

    for _ in 0..hash_cap {
        let k = unsafe { read_varlen_value::<NanBox>(hash_arr, &ARRAY_TYPE, slot * 2).to_bits() };
        if k == nil {
            return make_nil();
        } // empty slot — key not found
        if k == key {
            return unsafe {
                read_varlen_value::<NanBox>(hash_arr, &ARRAY_TYPE, slot * 2 + 1).to_bits()
            };
        }
        slot = (slot + 1) & mask;
    }

    make_nil()
}

fn table_set_rooted(
    heap: &Heap,
    roots: &ScopedJitRoots,
    table_root: ScopedJitRoot,
    key_root: ScopedJitRoot,
    val_root: ScopedJitRoot,
) {
    let key = roots.get(key_root);
    if is_number(key) {
        let n = as_number(key);
        if n == n.floor() && n >= 1.0 {
            let idx = n as usize;
            let table_ptr = table_ptr_from_boxed(roots.get(table_root))
                .expect("rooted table value must stay a table");
            let (hc, hcap, array_len, array_cap) = table_read_meta(table_ptr);
            if idx <= array_len + 1 {
                table_array_set_rooted(
                    heap,
                    roots,
                    table_root,
                    val_root,
                    idx,
                    hc,
                    hcap,
                    array_len,
                    array_cap,
                );
                return;
            }
        }
    }

    table_hash_set_rooted(heap, roots, table_root, key_root, val_root);
}

fn table_array_set_rooted(
    heap: &Heap,
    roots: &ScopedJitRoots,
    table_root: ScopedJitRoot,
    val_root: ScopedJitRoot,
    idx: usize,
    hash_count: usize,
    hash_cap: usize,
    mut array_len: usize,
    mut array_cap: usize,
) {
    if idx > array_cap {
        let new_cap = (array_cap * 2).max(8).max(idx);
        let table_ptr = table_ptr_from_boxed(roots.get(table_root))
            .expect("rooted table value must stay a table");
        let new_arr = alloc_nil_values(heap, new_cap);

        if let Some(old_arr) = table_get_array_ptr(table_ptr) {
            for i in 0..array_len {
                let v = unsafe { read_varlen_value::<NanBox>(old_arr, &ARRAY_TYPE, i) };
                unsafe {
                    write_varlen_value::<NanBox>(new_arr, &ARRAY_TYPE, i, v);
                }
            }
        }

        let table_ptr = table_ptr_from_boxed(roots.get(table_root))
            .expect("rooted table value must stay a table");
        unsafe {
            write_value_field::<NanBox>(
                table_ptr,
                &TABLE_TYPE,
                1,
                Value::from_bits(encode_gc_ptr(new_arr)),
            );
        }
        array_cap = new_cap;
    }

    if idx > array_len {
        array_len = idx;
    }
    let table_ptr = table_ptr_from_boxed(roots.get(table_root))
        .expect("rooted table value must stay a table");
    table_write_meta(table_ptr, hash_count, hash_cap, array_len, array_cap);

    let arr = table_get_array_ptr(table_ptr).unwrap();
    unsafe {
        write_varlen_value::<NanBox>(
            arr,
            &ARRAY_TYPE,
            idx - 1,
            Value::from_bits(roots.get(val_root)),
        );
    }
}

fn table_hash_set_rooted(
    heap: &Heap,
    roots: &ScopedJitRoots,
    table_root: ScopedJitRoot,
    key_root: ScopedJitRoot,
    val_root: ScopedJitRoot,
) {
    let table_ptr = table_ptr_from_boxed(roots.get(table_root))
        .expect("rooted table value must stay a table");
    let (mut hash_count, mut hash_cap, array_len, array_cap) = table_read_meta(table_ptr);
    let nil = make_nil();

    if hash_cap > 0 {
        if let Some(hash_arr) = table_get_hash_ptr(table_ptr) {
            let mask = hash_cap - 1;
            let key = roots.get(key_root);
            let mut slot = hash_u64(key) & mask;
            for _ in 0..hash_cap {
                let k = unsafe {
                    read_varlen_value::<NanBox>(hash_arr, &ARRAY_TYPE, slot * 2).to_bits()
                };
                if k == nil {
                    break;
                }
                if k == key {
                    unsafe {
                        write_varlen_value::<NanBox>(
                            hash_arr,
                            &ARRAY_TYPE,
                            slot * 2 + 1,
                            Value::from_bits(roots.get(val_root)),
                        );
                    }
                    return;
                }
                slot = (slot + 1) & mask;
            }
        }
    }

    if is_nil(roots.get(val_root)) {
        return;
    }

    if hash_cap == 0 || hash_count >= hash_cap * 3 / 4 {
        let new_cap = if hash_cap == 0 { 4 } else { hash_cap * 2 };
        let table_ptr = table_ptr_from_boxed(roots.get(table_root))
            .expect("rooted table value must stay a table");
        let new_arr = alloc_nil_values(heap, new_cap * 2);

        if let Some(old_arr) = table_get_hash_ptr(table_ptr) {
            let new_mask = new_cap - 1;
            for i in 0..hash_cap {
                let k =
                    unsafe { read_varlen_value::<NanBox>(old_arr, &ARRAY_TYPE, i * 2).to_bits() };
                if k != nil {
                    let v = unsafe {
                        read_varlen_value::<NanBox>(old_arr, &ARRAY_TYPE, i * 2 + 1).to_bits()
                    };
                    if !is_nil(v) {
                        let mut s = hash_u64(k) & new_mask;
                        loop {
                            let sk = unsafe {
                                read_varlen_value::<NanBox>(new_arr, &ARRAY_TYPE, s * 2).to_bits()
                            };
                            if sk == nil {
                                unsafe {
                                    write_varlen_value::<NanBox>(
                                        new_arr,
                                        &ARRAY_TYPE,
                                        s * 2,
                                        Value::from_bits(k),
                                    );
                                    write_varlen_value::<NanBox>(
                                        new_arr,
                                        &ARRAY_TYPE,
                                        s * 2 + 1,
                                        Value::from_bits(v),
                                    );
                                }
                                break;
                            }
                            s = (s + 1) & new_mask;
                        }
                    }
                }
            }
        }

        let mut new_count = 0;
        for i in 0..new_cap {
            let k = unsafe { read_varlen_value::<NanBox>(new_arr, &ARRAY_TYPE, i * 2).to_bits() };
            if k != nil {
                new_count += 1;
            }
        }

        let table_ptr = table_ptr_from_boxed(roots.get(table_root))
            .expect("rooted table value must stay a table");
        unsafe {
            write_value_field::<NanBox>(
                table_ptr,
                &TABLE_TYPE,
                0,
                Value::from_bits(encode_gc_ptr(new_arr)),
            );
        }
        hash_count = new_count;
        hash_cap = new_cap;
        table_write_meta(table_ptr, hash_count, hash_cap, array_len, array_cap);
    }

    let table_ptr = table_ptr_from_boxed(roots.get(table_root))
        .expect("rooted table value must stay a table");
    let hash_arr = table_get_hash_ptr(table_ptr).unwrap();
    let key = roots.get(key_root);
    let mask = hash_cap - 1;
    let mut slot = hash_u64(key) & mask;
    loop {
        let k = unsafe { read_varlen_value::<NanBox>(hash_arr, &ARRAY_TYPE, slot * 2).to_bits() };
        if k == nil {
            unsafe {
                write_varlen_value::<NanBox>(
                    hash_arr,
                    &ARRAY_TYPE,
                    slot * 2,
                    Value::from_bits(key),
                );
                write_varlen_value::<NanBox>(
                    hash_arr,
                    &ARRAY_TYPE,
                    slot * 2 + 1,
                    Value::from_bits(roots.get(val_root)),
                );
            }
            hash_count += 1;
            table_write_meta(table_ptr, hash_count, hash_cap, array_len, array_cap);
            return;
        }
        slot = (slot + 1) & mask;
    }
}

// ── LuaRuntime ────────────────────────────────────────────────

/// The Lua runtime state — a bag of functions, globals, and constants used by the JIT externs.
pub struct LuaRuntime {
    heap: *const Heap,
    globals: HashMap<String, u64>,
    pub constants: Vec<String>,
    /// Register file snapshot for passing args to calls
    pub register_file: Vec<u64>,
    /// Captured output from print
    pub output: String,
}

// Safety: LuaRuntime is only used single-threaded. The raw pointer to Heap
// is valid for the lifetime of the runtime.
unsafe impl Send for LuaRuntime {}

impl LuaRuntime {
    pub fn new(heap: &Heap, constants: &[crate::bytecode::Constant]) -> Self {
        let mut string_constants = Vec::new();
        for c in constants {
            match c {
                crate::bytecode::Constant::String(s) => string_constants.push(s.clone()),
                _ => string_constants.push(String::new()),
            }
        }

        let mut rt = LuaRuntime {
            heap: heap as *const Heap,
            globals: HashMap::new(),
            constants: string_constants,
            register_file: Vec::new(),
            output: String::new(),
        };

        rt.init_stdlib();
        rt
    }

    pub fn heap_ref(&self) -> &Heap {
        unsafe { &*self.heap }
    }

    fn init_stdlib(&mut self) {
        self.globals.insert(
            "print".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0000),
        );
        self.globals.insert(
            "type".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0001),
        );
        self.globals.insert(
            "tostring".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0002),
        );
        self.globals.insert(
            "tonumber".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0003),
        );
        self.globals.insert(
            "pairs".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0004),
        );
        self.globals.insert(
            "ipairs".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0005),
        );
        self.globals.insert(
            "assert".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0006),
        );
        self.globals.insert(
            "error".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0007),
        );
        self.globals.insert(
            "pcall".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0008),
        );
        self.globals.insert(
            "select".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_0009),
        );
        self.globals.insert(
            "unpack".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000A),
        );
        self.globals.insert(
            "rawget".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000B),
        );
        self.globals.insert(
            "rawset".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000C),
        );
        self.globals.insert(
            "setmetatable".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000D),
        );
        self.globals.insert(
            "getmetatable".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0002_0000_000E),
        );

        self.globals.insert(
            "math".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0003_0000_0000),
        );
        self.globals.insert(
            "string".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0003_0000_0001),
        );
        self.globals.insert(
            "table".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0003_0000_0002),
        );
        self.globals.insert(
            "io".to_string(),
            NanBox::encode_tagged(TAG_INTERN, 0x0003_0000_0003),
        );
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
            if as_bool_payload(v) {
                "true".to_string()
            } else {
                "false".to_string()
            }
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
        } else if is_table(v) {
            "table".to_string()
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
        } else if is_table(a) {
            let ptr = NanBox::extract_payload(a) as *const u8;
            let (_, _, array_len, _) = table_read_meta(ptr);
            make_number(array_len as f64)
        } else {
            make_number(0.0)
        }
    }

    // ── Comparison ─────────────────────────────────────────────

    pub fn lua_eq(&self, a: u64, b: u64) -> u64 {
        let result = if a == b {
            true // same bits = always equal
        } else if is_nil(a) || is_nil(b) {
            false
        } else if is_bool(a) && is_bool(b) {
            as_bool_payload(a) == as_bool_payload(b)
        } else if is_number(a) && is_number(b) {
            as_number(a) == as_number(b)
        } else {
            // String comparison: resolve both sides and compare
            match (self.resolve_string(a), self.resolve_string(b)) {
                (Some(sa), Some(sb)) => sa == sb,
                _ => false,
            }
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
        let combined = format!("{}{}", sa, sb);
        make_string_on_heap(self.heap_ref(), &combined)
    }

    // ── Globals ────────────────────────────────────────────────

    pub fn lua_getglobal(&self, name: u64) -> u64 {
        if let Some(name_str) = self.resolve_string(name) {
            self.globals
                .get(&name_str)
                .copied()
                .unwrap_or_else(make_nil)
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
        let heap = self.heap_ref();
        let ptr = heap.alloc_obj::<Compact>(&TABLE_TYPE, 0);
        assert!(!ptr.is_null(), "GC heap exhausted during table allocation");
        unsafe {
            write_value_field::<NanBox>(ptr, &TABLE_TYPE, 0, Value::from_bits(make_nil()));
            write_value_field::<NanBox>(ptr, &TABLE_TYPE, 1, Value::from_bits(make_nil()));
            let raw = raw_data_mut(ptr, &TABLE_TYPE);
            raw.fill(0);
        }
        encode_gc_ptr(ptr)
    }

    pub fn lua_gettable(&self, table: u64, key: u64) -> u64 {
        if NanBox::has_tag(table, TAG_PTR) {
            let payload = NanBox::extract_payload(table);
            if payload != 0 {
                let ptr = payload as *const u8;
                if obj_type_from_ptr(ptr) == ObjType::Table {
                    return table_get(ptr, key);
                }
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

    pub fn lua_settable(&mut self, table: u64, key: u64, val: u64) {
        if !is_table(table) {
            return;
        }
        let heap = self.heap_ref();
        let mut roots = ScopedJitRoots::new();
        let table_root = roots.push(table);
        let key_root = roots.push(key);
        let val_root = roots.push(val);
        roots.with_active(|| {
            table_set_rooted(heap, &roots, table_root, key_root, val_root);
        });
    }

    pub fn lua_setlist_from_regfile(&mut self, table: u64, base: usize, offset: usize, count: usize) {
        if !is_table(table) {
            return;
        }
        let actual_count = if count == 0 {
            self.register_file.len().saturating_sub(base)
        } else {
            count
        };
        let heap = self.heap_ref();
        for i in 0..actual_count {
            if base + i < self.register_file.len() {
                let val = self.register_file[base + i];
                let key = make_number((offset + i + 1) as f64);
                let mut roots = ScopedJitRoots::new();
                let table_root = roots.push(table);
                let key_root = roots.push(key);
                let val_root = roots.push(val);
                roots.with_active(|| {
                    table_set_rooted(heap, &roots, table_root, key_root, val_root);
                });
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
        panic!(
            "attempt to call a {} value (closures must be dispatched externally)",
            if is_nil(func) { "nil" } else { "non-function" }
        );
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
            1 => {
                // type(val) → string name
                if nargs < 1 { return make_nil(); }
                let v = self.register_file[base_reg];
                let type_name = if is_nil(v) {
                    "nil"
                } else if is_bool(v) {
                    "boolean"
                } else if is_number(v) {
                    "number"
                } else if is_intern(v) {
                    let payload = NanBox::extract_payload(v);
                    let high = (payload >> 32) as u32;
                    if high == 0 {
                        "string" // constant string
                    } else {
                        "function" // builtin function
                    }
                } else if is_closure(v) {
                    "function"
                } else if is_table(v) {
                    "table"
                } else if NanBox::has_tag(v, TAG_PTR) {
                    let payload = NanBox::extract_payload(v);
                    if payload != 0 && obj_type_from_ptr(payload as *const u8) == ObjType::String {
                        "string"
                    } else {
                        "userdata"
                    }
                } else {
                    "userdata"
                };
                make_string_on_heap(self.heap_ref(), type_name)
            }
            2 => {
                // tostring(val)
                if nargs < 1 { return make_nil(); }
                let v = self.register_file[base_reg];
                let s = self.value_to_string(v);
                make_string_on_heap(self.heap_ref(), &s)
            }
            6 => {
                if nargs < 1 {
                    panic!("assertion failed!");
                }
                let v = self.register_file[base_reg];
                if !is_truthy(v) {
                    panic!("assertion failed!");
                }
                v
            }
            7 => panic!("error raised"),
            3 => {
                if nargs < 1 {
                    return make_nil();
                }
                let v = self.register_file[base_reg];
                if is_number(v) { v } else { make_nil() }
            }
            _ => make_nil(),
        }
    }

    fn call_math(&self, id: u32, base_reg: usize, nargs: usize) -> u64 {
        if nargs < 1 {
            return make_nil();
        }
        let a = self.register_file[base_reg];
        if !is_number(a) {
            panic!("bad argument to math function");
        }
        let na = as_number(a);

        match id {
            0 => make_number(na.sqrt()),
            1 => make_number(na.abs()),
            2 => make_number(na.floor()),
            3 => make_number(na.ceil()),
            4 => make_number(na.sin()),
            5 => make_number(na.cos()),
            6 => {
                if nargs < 2 {
                    return a;
                }
                let b = self.register_file[base_reg + 1];
                if is_number(b) {
                    make_number(na.max(as_number(b)))
                } else {
                    a
                }
            }
            7 => {
                if nargs < 2 {
                    return a;
                }
                let b = self.register_file[base_reg + 1];
                if is_number(b) {
                    make_number(na.min(as_number(b)))
                } else {
                    a
                }
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
            if continue_loop {
                make_number(idx)
            } else {
                make_nil()
            }
        } else {
            panic!("'for' values must be numbers");
        }
    }

    pub fn lua_is_nil(&self, v: u64) -> u64 {
        make_bool(is_nil(v))
    }

    /// Scan all GC roots in this runtime for the collector.
    ///
    /// # Safety
    /// Must only be called during a GC safepoint when no other code
    /// is accessing the runtime. Uses a raw pointer to allow the GC
    /// to update forwarded pointers in-place.
    pub unsafe fn scan_gc_roots(rt_ptr: *mut LuaRuntime, visitor: &mut dyn FnMut(*mut u64)) {
        unsafe {
            let rt = &mut *rt_ptr;
            for val in rt.globals.values_mut() {
                visitor(val as *mut u64);
            }
            for val in rt.register_file.iter_mut() {
                visitor(val as *mut u64);
            }
        }
    }
}
