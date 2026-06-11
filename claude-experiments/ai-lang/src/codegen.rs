//! LLVM codegen for the canonical AST.
//!
//! ## Scope
//!
//! - Top-level `def fn(Int, …, Int) -> (Int | fn(Int…) -> Int)` definitions.
//! - Expressions: `IntLit`, `LocalVar`, `Let`, `Lambda`,
//!   `Call(BuiltinRef("core/i64.*"), …)`, `Call(TopRef(h), …)`,
//!   `Call(closure_expr, …)` (indirect call through a closure value).
//! - Every JIT'd function takes `*mut Thread` as its first parameter and
//!   maintains an intrusive shadow-stack frame (see `runtime`).
//!
//! ## v1 closure restrictions (each errors cleanly)
//!
//! - **No nested lambdas.** A `Lambda` inside another `Lambda`'s body is
//!   rejected during codegen.
//!
//! What's supported: pointer captures (inferred via `TypeInfo`), non-Int
//! lambda params (passed via the uniform closure ABI), indirect calls
//! through closures held in `let` bindings or returned by other defs,
//! and first-class function references (eta-expanded to adapter closures).
//!
//! ## Symbol naming
//!
//! - `def_<hex(hash)>` — entry point for each `Def`.
//! - `lambda_<hex(hash)>` — lifted entry point for each unique `Lambda`
//!   expression. Deduped by content hash.
//! - `__frame_origin_def_<hex(hash)>` / `__frame_origin_lambda_<hex(hash)>` —
//!   per-function FrameOrigin globals (private constants).
//! - `__closure_ti_<hex(hash)>` — `TypeInfo` global per unique lambda
//!   shape.

use crate::ast::{Def, Expr, MatchArm, Pattern, Type};
use crate::codec::encode_expr;
use crate::gc::{ObjHeader, TypeInfo, VarLenKind};
use crate::hash::Hash;
use crate::resolve::{ResolvedDef, ResolvedModule};
use crate::runtime::{
    Runtime, Thread, ai_array_get, ai_array_get_i64, ai_atom_load, ai_atom_new, ai_atom_swap_local, ai_array_len, ai_array_new, ai_array_new_prim, ai_array_set, ai_array_set_i64, ai_bytes_copy, ai_str_copy,
    ai_bytes_get, ai_bytes_new, ai_bytes_set, ai_bytes_slice, ai_gc_alloc_closure, ai_gc_box_int,
    ai_gc_force_collect, ai_gc_lookup_code, ai_gc_pollcheck_slow, ai_gc_unbox_int, ai_abort, ai_state_get, ai_thread_join, ai_thread_spawn, ai_thread_spawn_shared,
    ai_state_present, ai_state_set, ai_value_eq, ai_value_hash, ai_str_concat, ai_str_eq,
    ai_str_len, ai_str_new, closure_offsets, thread_offsets,
};

use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::{Linkage, Module};
use inkwell::types::{BasicMetadataTypeEnum, IntType, PointerType, StructType};
use inkwell::values::{
    AnyValue, BasicMetadataValueEnum, BasicValue, BasicValueEnum, FloatValue, FunctionValue,
    GlobalValue, IntValue, PointerValue,
};
use inkwell::{AddressSpace, IntPredicate};

use std::collections::{BTreeSet, HashMap, HashSet};

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodegenError {
    Unsupported { what: String },
    UnknownTopRef { hash: Hash },
    UnknownBuiltin { name: String, arity: usize },
    JitInit(String),
    FunctionNotFound { symbol: String },
    TypeMismatch { what: String },
}

impl core::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CodegenError::Unsupported { what } => {
                write!(f, "codegen does not yet support: {}", what)
            }
            CodegenError::UnknownTopRef { hash } => write!(f, "TopRef to unknown hash {}", hash),
            CodegenError::UnknownBuiltin { name, arity } => {
                write!(f, "unknown builtin `{}` (arity {})", name, arity)
            }
            CodegenError::JitInit(msg) => write!(f, "JIT init failed: {}", msg),
            CodegenError::FunctionNotFound { symbol } => {
                write!(f, "function `{}` not found in compiled module", symbol)
            }
            CodegenError::TypeMismatch { what } => write!(f, "type mismatch: {}", what),
        }
    }
}

impl std::error::Error for CodegenError {}

// =============================================================================
// Public entry points
// =============================================================================

pub fn init_native_target() -> Result<(), CodegenError> {
    // Register the standard I/O externs (print_int, print_string,
    // println, read_line, int_to_string, string_to_int, ...) in the
    // global FFI registry. Idempotent via an internal `Once`. Tests
    // and runners call init_native_target() before constructing a JIT;
    // this ensures the stdlib's `extern fn` declarations resolve.
    crate::io_externs::register_io_externs();
    // Everything else that used to be a host shim now runs through the
    // generic C FFI or pure ai-lang: HTTP drives real libcurl, crypto
    // calls OpenSSL's libcrypto, OS (env/clock/fs) calls libc, and JSON
    // is a recursive-descent parser written in ai-lang. The only host
    // externs left are the legitimate runtime I/O ones above.
    inkwell::targets::Target::initialize_native(&inkwell::targets::InitializationConfig::default())
        .map_err(CodegenError::JitInit)
}

/// Run the standard `default<O2>` new-pass-manager pipeline over a module.
fn run_o2_passes(module: &Module<'_>) -> Result<(), CodegenError> {
    use inkwell::targets::{CodeModel, RelocMode, Target, TargetMachine};
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple)
        .map_err(|e| CodegenError::JitInit(format!("target from triple: {}", e)))?;
    let tm = target
        .create_target_machine(
            &triple,
            TargetMachine::get_host_cpu_name().to_str().unwrap_or(""),
            TargetMachine::get_host_cpu_features().to_str().unwrap_or(""),
            OptimizationLevel::Default,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .ok_or_else(|| CodegenError::JitInit("create_target_machine failed".into()))?;
    let opts = inkwell::passes::PassBuilderOptions::create();
    module
        .run_passes("default<O2>", &tm, opts)
        .map_err(|e| CodegenError::JitInit(format!("run_passes O2: {}", e)))
}

pub fn def_symbol(hash: &Hash) -> String {
    format!("def_{}", hash.to_hex())
}

pub fn lambda_symbol(hash: &Hash) -> String {
    format!("lambda_{}", hash.to_hex())
}

/// Symbol of a node `state` binding's installer function. Run once per
/// node (idempotently) to evaluate the initializer and populate the
/// state table.
pub fn state_init_symbol(hash: &Hash) -> String {
    format!("state_init_{}", hash.to_hex())
}

pub fn frame_origin_symbol(prefix: &str, hash: &Hash) -> String {
    format!("__frame_origin_{}_{}", prefix, hash.to_hex())
}

// =============================================================================
// Bitcode cache — skip compilation when the transitive closure hasn't changed
// =============================================================================

/// Bump on ANY codegen change that alters emitted IR or the runtime ABI
/// it compiles against (Thread layout, runtime fn signatures, builtin
/// lowering). Salted into [`module_hash`] so stale cached bitcode from an
/// older compiler can never be loaded against a newer runtime.
const CODEGEN_CACHE_VERSION: u64 = 11;

/// Deterministic hash of a set of def hashes (order-independent),
/// salted with [`CODEGEN_CACHE_VERSION`].
pub fn module_hash(hashes: &[Hash]) -> Hash {
    let mut sorted: Vec<&Hash> = hashes.iter().collect();
    sorted.sort_by_key(|h| h.to_hex());
    let mut hasher = blake3::Hasher::new();
    hasher.update(&CODEGEN_CACHE_VERSION.to_be_bytes());
    for h in &sorted {
        hasher.update(&h.0);
    }
    let digest = hasher.finalize();
    Hash(digest.into())
}

fn bc_cache_path(root: &std::path::Path, mh: &Hash) -> std::path::PathBuf {
    root.join("obj").join(format!("{}.bc", mh.to_hex()))
}

fn shapes_cache_path(root: &std::path::Path, mh: &Hash) -> std::path::PathBuf {
    root.join("obj").join(format!("{}.shapes", mh.to_hex()))
}

/// Write compiled bitcode + shape metadata to the obj cache.
pub fn write_bitcode_cache(
    cm: &CompiledModule,
    root: &std::path::Path,
    hashes: &[Hash],
) -> Result<(), String> {
    let mh = module_hash(hashes);
    let obj_dir = root.join("obj");
    std::fs::create_dir_all(&obj_dir).map_err(|e| format!("create obj dir: {}", e))?;

    let bc_path = bc_cache_path(root, &mh);
    cm.module.write_bitcode_to_path(&bc_path);

    let shapes = shape_metadata_to_json(cm);
    let shapes_path = shapes_cache_path(root, &mh);
    std::fs::write(&shapes_path, shapes.to_string())
        .map_err(|e| format!("write shapes: {}", e))?;

    Ok(())
}

/// Try to load a compiled module from the bitcode cache. Returns `None` on miss.
pub fn load_bitcode_cache<'ctx>(
    ctx: &'ctx Context,
    root: &std::path::Path,
    hashes: &[Hash],
) -> Option<CompiledModule<'ctx>> {
    let mh = module_hash(hashes);
    let bc_path = bc_cache_path(root, &mh);
    let shapes_path = shapes_cache_path(root, &mh);

    if !bc_path.exists() || !shapes_path.exists() {
        return None;
    }

    let bc = inkwell::memory_buffer::MemoryBuffer::create_from_file(&bc_path).ok()?;
    let module = ctx.create_module_from_ir(bc).ok()?;

    let shapes_text = std::fs::read_to_string(&shapes_path).ok()?;
    let shapes_json = crate::jsonl::parse(&shapes_text).ok()?;
    let (
        closure_type_infos,
        shape_registry,
        shape_meta,
        shape_by_type_id,
        state_hashes,
        stateful_hashes,
    ) = parse_shape_metadata(&shapes_json)?;

    let mut functions: HashMap<Hash, FunctionValue<'ctx>> = HashMap::new();
    for h in hashes {
        let sym = def_symbol(h);
        if let Some(fv) = module.get_function(&sym) {
            functions.insert(*h, fv);
        }
    }

    let mut lifted_lambdas: HashMap<Hash, FunctionValue<'ctx>> = HashMap::new();
    let mut f = module.get_first_function();
    while let Some(fv) = f {
        let name = fv.get_name().to_string_lossy().to_string();
        if let Some(hex) = name.strip_prefix("lambda_") {
            if let Some(h) = crate::codebase::parse_hex_hash(hex) {
                lifted_lambdas.insert(h, fv);
            }
        }
        f = fv.get_next_function();
    }

    Some(CompiledModule {
        context: ctx,
        module,
        functions,
        lifted_lambdas,
        closure_type_infos,
        shape_registry,
        shape_meta,
        shape_by_type_id,
        state_hashes,
        stateful_hashes,
    })
}


fn shape_metadata_to_json(cm: &CompiledModule) -> crate::jsonl::Json {
    use crate::gc::VarLenKind;
    use crate::jsonl::Json;

    let tis: Vec<Json> = cm
        .closure_type_infos
        .iter()
        .map(|ti| {
            let varlen_str = match ti.varlen {
                VarLenKind::None => "none",
                VarLenKind::Values => "values",
                VarLenKind::Bytes => "bytes",
            };
            Json::obj([
                ("type_id", Json::Int(ti.type_id as i64)),
                ("header_size", Json::Int(ti.header_size as i64)),
                ("value_field_count", Json::Int(ti.value_field_count as i64)),
                ("raw_byte_count", Json::Int(ti.raw_byte_count as i64)),
                ("varlen", Json::Str(varlen_str.to_string())),
                ("align_log2", Json::Int(ti.align_log2 as i64)),
            ])
        })
        .collect();

    let mut sr_obj = std::collections::BTreeMap::new();
    for (h, id) in &cm.shape_registry {
        sr_obj.insert(h.to_hex(), Json::Int(*id as i64));
    }

    let mut sm_obj = std::collections::BTreeMap::new();
    for (h, meta) in &cm.shape_meta {
        sm_obj.insert(h.to_hex(), shape_meta_to_json(meta));
    }

    let sbti: Vec<Json> = cm
        .shape_by_type_id
        .iter()
        .map(|opt| match opt {
            Some(h) => Json::Str(h.to_hex()),
            None => Json::Null,
        })
        .collect();

    let state_hs: Vec<Json> = cm.state_hashes.iter().map(|h| Json::Str(h.to_hex())).collect();
    let sf_hs: Vec<Json> = cm
        .stateful_hashes
        .iter()
        .map(|h| Json::Str(h.to_hex()))
        .collect();

    Json::obj([
        ("type_infos", Json::Array(tis)),
        ("shape_registry", Json::Object(sr_obj)),
        ("shape_meta", Json::Object(sm_obj)),
        ("shape_by_type_id", Json::Array(sbti)),
        ("state_hashes", Json::Array(state_hs)),
        ("stateful_hashes", Json::Array(sf_hs)),
    ])
}

fn shape_meta_to_json(meta: &ShapeMeta) -> crate::jsonl::Json {
    use crate::jsonl::Json;
    match meta {
        ShapeMeta::Closure {
            code_hash, captures,
        } => {
            let caps: Vec<Json> = captures
                .iter()
                .map(|c| {
                    Json::obj([
                        ("offset", Json::Int(c.offset as i64)),
                        ("is_pointer", Json::Bool(c.is_pointer)),
                    ])
                })
                .collect();
            Json::obj([
                ("kind", Json::Str("closure".to_string())),
                ("code_hash", Json::Str(code_hash.to_hex())),
                ("captures", Json::Array(caps)),
            ])
        }
        ShapeMeta::Struct {
            struct_ref, fields,
        } => {
            let fs: Vec<Json> = fields
                .iter()
                .map(|f| {
                    Json::obj([
                        ("offset", Json::Int(f.offset as i64)),
                        ("is_pointer", Json::Bool(f.is_pointer)),
                    ])
                })
                .collect();
            Json::obj([
                ("kind", Json::Str("struct".to_string())),
                ("struct_ref", Json::Str(struct_ref.to_hex())),
                ("fields", Json::Array(fs)),
            ])
        }
        ShapeMeta::EnumVariant {
            enum_ref,
            variant_index,
            tag_offset,
            payload,
        } => {
            let p = match payload {
                Some(f) => Json::obj([
                    ("offset", Json::Int(f.offset as i64)),
                    ("is_pointer", Json::Bool(f.is_pointer)),
                ]),
                None => Json::Null,
            };
            Json::obj([
                ("kind", Json::Str("enum_variant".to_string())),
                ("enum_ref", Json::Str(enum_ref.to_hex())),
                ("variant_index", Json::Int(*variant_index as i64)),
                ("tag_offset", Json::Int(*tag_offset as i64)),
                ("payload", p),
            ])
        }
    }
}

fn parse_shape_metadata(
    json: &crate::jsonl::Json,
) -> Option<(
    Vec<crate::gc::TypeInfo>,
    HashMap<Hash, u16>,
    HashMap<Hash, ShapeMeta>,
    Vec<Option<Hash>>,
    Vec<Hash>,
    Vec<Hash>,
)> {
    use crate::gc::{TypeInfo, VarLenKind};
    use crate::jsonl::Json;

    let parse_hash = |s: &str| crate::codebase::parse_hex_hash(s);

    // TypeInfos
    let tis_arr = json.get("type_infos")?.as_array()?;
    let mut type_infos = Vec::new();
    for ti in tis_arr {
        let varlen_str = ti.get("varlen")?.as_str()?;
        let varlen = match varlen_str {
            "none" => VarLenKind::None,
            "values" => VarLenKind::Values,
            "bytes" => VarLenKind::Bytes,
            _ => return None,
        };
        type_infos.push(TypeInfo {
            type_id: ti.get("type_id")?.as_i64()? as u16,
            header_size: ti.get("header_size")?.as_i64()? as u16,
            value_field_count: ti.get("value_field_count")?.as_i64()? as u16,
            raw_byte_count: ti.get("raw_byte_count")?.as_i64()? as u16,
            varlen,
            align_log2: ti.get("align_log2")?.as_i64()? as u8,
        });
    }

    // Shape registry
    let sr_obj = match json.get("shape_registry")? {
        Json::Object(m) => m,
        _ => return None,
    };
    let mut shape_registry = HashMap::new();
    for (k, v) in sr_obj {
        let h = parse_hash(k)?;
        shape_registry.insert(h, v.as_i64()? as u16);
    }

    // Shape meta
    let sm_obj = match json.get("shape_meta")? {
        Json::Object(m) => m,
        _ => return None,
    };
    let mut shape_meta = HashMap::new();
    for (k, v) in sm_obj {
        let h = parse_hash(k)?;
        shape_meta.insert(h, parse_shape_meta_entry(v)?);
    }

    // Shape by type ID
    let sbti_arr = json.get("shape_by_type_id")?.as_array()?;
    let shape_by_type_id: Vec<Option<Hash>> = sbti_arr
        .iter()
        .map(|j| j.as_str().and_then(|s| parse_hash(s)))
        .collect();

    let state_hashes: Vec<Hash> = json
        .get("state_hashes")?
        .as_array()?
        .iter()
        .filter_map(|j| j.as_str().and_then(|s| parse_hash(s)))
        .collect();

    let stateful_hashes: Vec<Hash> = json
        .get("stateful_hashes")?
        .as_array()?
        .iter()
        .filter_map(|j| j.as_str().and_then(|s| parse_hash(s)))
        .collect();

    Some((
        type_infos,
        shape_registry,
        shape_meta,
        shape_by_type_id,
        state_hashes,
        stateful_hashes,
    ))
}

fn parse_shape_meta_entry(json: &crate::jsonl::Json) -> Option<ShapeMeta> {
    use crate::jsonl::Json;
    let parse_hash = |s: &str| crate::codebase::parse_hex_hash(s);

    let kind = json.get("kind")?.as_str()?;
    match kind {
        "closure" => {
            let caps_arr = json.get("captures")?.as_array()?;
            let mut captures = Vec::new();
            for c in caps_arr {
                captures.push(CaptureMeta {
                    offset: c.get("offset")?.as_i64()? as u32,
                    is_pointer: c.get("is_pointer")?.as_bool()?,
                });
            }
            Some(ShapeMeta::Closure {
                code_hash: parse_hash(json.get("code_hash")?.as_str()?)?,
                captures,
            })
        }
        "struct" => {
            let fields_arr = json.get("fields")?.as_array()?;
            let mut fields = Vec::new();
            for f in fields_arr {
                fields.push(FieldMeta {
                    offset: f.get("offset")?.as_i64()? as u32,
                    is_pointer: f.get("is_pointer")?.as_bool()?,
                });
            }
            Some(ShapeMeta::Struct {
                struct_ref: parse_hash(json.get("struct_ref")?.as_str()?)?,
                fields,
            })
        }
        "enum_variant" => {
            let payload = match json.get("payload")? {
                Json::Null => None,
                p => Some(FieldMeta {
                    offset: p.get("offset")?.as_i64()? as u32,
                    is_pointer: p.get("is_pointer")?.as_bool()?,
                }),
            };
            Some(ShapeMeta::EnumVariant {
                enum_ref: parse_hash(json.get("enum_ref")?.as_str()?)?,
                variant_index: json.get("variant_index")?.as_i64()? as u32,
                tag_offset: json.get("tag_offset")?.as_i64()? as u32,
                payload,
            })
        }
        _ => None,
    }
}
/// LLVM symbol name for a user-defined `extern fn`. Same as the
/// canonical `ext/<name>` builtin name to keep things readable.
fn user_extern_symbol(name: &str) -> String {
    format!("ext/{}", name)
}

/// Collect the bare names (without the `ext/` prefix) of every extern
/// referenced by `e`. Used so the C FFI only resolves library symbols
/// (via dlopen/dlsym) that the module actually calls — declaring
/// `curl_*` in the prelude shouldn't force every program to load
/// libcurl.
fn collect_ext_refs(e: &Expr, out: &mut std::collections::HashSet<String>) {
    match e {
        Expr::BuiltinRef(name) => {
            if let Some(ext) = name.strip_prefix("ext/") {
                out.insert(ext.to_owned());
            }
        }
        Expr::Lambda { body, .. } => collect_ext_refs(body, out),
        Expr::Call(callee, args) => {
            collect_ext_refs(callee, out);
            for a in args {
                collect_ext_refs(a, out);
            }
        }
        Expr::Let { value, body } => {
            collect_ext_refs(value, out);
            collect_ext_refs(body, out);
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                collect_ext_refs(f, out);
            }
        }
        Expr::Field { base, .. } => collect_ext_refs(base, out),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                collect_ext_refs(p, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_ext_refs(scrutinee, out);
            for arm in arms {
                collect_ext_refs(&arm.body, out);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_ext_refs(cond, out);
            collect_ext_refs(then_branch, out);
            collect_ext_refs(else_branch, out);
        }
        Expr::Try { expr, .. } => collect_ext_refs(expr, out),
        Expr::Defer { cleanup, body } => {
            collect_ext_refs(cleanup, out);
            collect_ext_refs(body, out);
        }
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_) => {}
    }
}

/// Resolve + register every C-FFI extern the module references, so the
/// JIT wiring step can map `ext/<name>` to a real address. Codegen does
/// this as part of `declare_user_externs`; a bitcode-cache hit skips
/// codegen entirely, so the runner must call this separately before JIT
/// init or every dlsym-backed extern would map to null and the first
/// call into one would jump to address 0.
pub fn register_referenced_c_externs(rm: &ResolvedModule) -> Result<(), String> {
    let mut referenced = std::collections::HashSet::new();
    for rd in &rm.defs {
        match &rd.def {
            Def::Fn { body, .. } => collect_ext_refs(body, &mut referenced),
            Def::State { init, .. } => collect_ext_refs(init, &mut referenced),
            _ => {}
        }
    }
    for (name, sig) in &rm.externs {
        if sig.library.is_none() || !referenced.contains(name.as_str()) {
            continue;
        }
        let lib = sig.library.as_deref().unwrap_or("");
        match crate::cffi::resolve_symbol(lib, name) {
            Some(addr) => unsafe {
                crate::ffi::register_extern(name, sig.params.clone(), sig.ret.clone(), addr);
            },
            None => {
                return Err(format!(
                    "could not resolve C symbol `{}` from library \"{}\" \
                     (dlopen/dlsym failed)",
                    name, lib
                ));
            }
        }
    }
    Ok(())
}

/// Walk every `ai_user_ext__*` LLVM function in `module` and link it
/// to its registered Rust function pointer (via the global FFI
/// registry). Externs declared but not registered are left unmapped
/// — the first call into them will crash, which is the expected
/// error mode for "you forgot to register before invoking."
fn wire_user_externs_into<'ctx>(engine: &ExecutionEngine<'ctx>, module: &Module<'ctx>) {
    const PREFIX: &str = "ext/";
    let mut f = module.get_first_function();
    while let Some(fv) = f {
        let name = fv.get_name().to_string_lossy().to_string();
        if let Some(ext_name) = name.strip_prefix(PREFIX) {
            if let Some(entry) = crate::ffi::lookup_extern(ext_name) {
                engine.add_global_mapping(&fv, entry.fn_ptr);
            }
        }
        f = fv.get_next_function();
    }
}

pub fn closure_ti_symbol(hash: &Hash) -> String {
    format!("__closure_ti_{}", hash.to_hex())
}

// =============================================================================
// Runtime values
// =============================================================================

/// A value produced by `compile_expr`.
///
/// Today every value is either an `Int` (i64) or a `Closure` (heap
/// pointer to a Closure object). A typechecker would let us replace
/// these branches with explicit, checked conversions.
#[derive(Copy, Clone, Debug)]
enum Value<'ctx> {
    Int(IntValue<'ctx>),
    Closure(PointerValue<'ctx>),
}

impl<'ctx> Value<'ctx> {
    fn as_int(&self) -> Result<IntValue<'ctx>, CodegenError> {
        match self {
            Value::Int(v) => Ok(*v),
            Value::Closure(_) => Err(CodegenError::TypeMismatch {
                what: "expected Int, got Closure".to_owned(),
            }),
        }
    }

    fn as_closure(&self) -> Result<PointerValue<'ctx>, CodegenError> {
        match self {
            Value::Closure(p) => Ok(*p),
            Value::Int(_) => Err(CodegenError::TypeMismatch {
                what: "expected Closure, got Int".to_owned(),
            }),
        }
    }

    fn into_basic(self) -> BasicValueEnum<'ctx> {
        match self {
            Value::Int(v) => v.into(),
            Value::Closure(p) => p.into(),
        }
    }
}

// =============================================================================
// Module compilation
// =============================================================================

/// Result of compiling a `ResolvedModule` into LLVM IR.
///
/// Pair with a [`Runtime`] (created from `closure_type_infos`) and a
/// [`Jit`] to actually execute code.
pub struct CompiledModule<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    /// Def hash → LLVM function value, for direct caller-side lookup.
    pub functions: HashMap<Hash, FunctionValue<'ctx>>,
    /// Unique lambda hashes that were lifted, with their LLVM functions.
    pub lifted_lambdas: HashMap<Hash, FunctionValue<'ctx>>,
    /// `TypeInfo`s for each registered closure shape, in `type_id` order.
    /// Pass these to `Runtime::new` so the heap and the JIT'd code agree
    /// on the closure layouts.
    pub closure_type_infos: Vec<TypeInfo>,

    /// Maps a content hash to the index of its `TypeInfo` in
    /// `closure_type_infos`. Used by the wire deserializer to find the
    /// right `TypeInfo` when reconstructing an incoming closure /
    /// struct / enum-variant value.
    ///
    /// Keys:
    /// - Lambda code hash → closure shape's type_id
    /// - Struct def hash → struct's type_id
    /// - Variant hash (`derive_variant_hash(enum_hash, variant_name)`)
    ///   → variant's type_id
    pub shape_registry: HashMap<Hash, u16>,

    /// Maps the same keys as `shape_registry` to layout info that the
    /// wire encoder/decoder uses to walk a heap instance.
    pub shape_meta: HashMap<Hash, ShapeMeta>,

    /// type_id → shape hash. Encoder reads `type_id` from the heap
    /// header and looks up the shape here.
    pub shape_by_type_id: Vec<Option<Hash>>,

    /// Content hashes of node `state` bindings in this module, in
    /// dependency order. Their installer functions (`state_init_<hash>`)
    /// must run once at node startup to populate the state table.
    pub state_hashes: Vec<Hash>,

    /// Def/lambda hashes whose thunks transitively touch a node `state`
    /// cell and so must bypass the `at()` result cache. Copied into
    /// `Runtime.stateful_hashes` at JIT install.
    pub stateful_hashes: Vec<Hash>,
}

/// Layout metadata for one heap shape, used by the wire encoder/decoder.
#[derive(Clone, Debug)]
pub enum ShapeMeta {
    Closure {
        code_hash: Hash,
        /// Per-capture layout in **source order** (the order the
        /// lambda's free variables appear in `LambdaSpec::captures`).
        /// Pointer captures live in `value_field` slots before the
        /// `code_hash` header so the GC traces them; Int captures
        /// live in raw bytes after the `n_captures` field. Each
        /// `CaptureMeta::offset` is the absolute byte offset within
        /// the heap object.
        captures: Vec<CaptureMeta>,
    },
    Struct {
        struct_ref: Hash,
        fields: Vec<FieldMeta>,
    },
    EnumVariant {
        enum_ref: Hash,
        variant_index: u32,
        tag_offset: u32,
        payload: Option<FieldMeta>,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct CaptureMeta {
    /// Absolute byte offset of this capture's slot from the start
    /// of the heap object.
    pub offset: u32,
    /// `true` if the slot holds a heap pointer (GC-traced
    /// `value_field`); `false` if it holds a raw i64 (after the
    /// closure header in the raw-bytes section).
    pub is_pointer: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct FieldMeta {
    pub offset: u32,
    pub is_pointer: bool,
}

impl<'ctx> CompiledModule<'ctx> {
    pub fn build(context: &'ctx Context, rm: &ResolvedModule) -> Result<Self, CodegenError> {
        Self::build_with_externals(context, rm, &HashSet::new(), &HashSet::new(), 0)
    }

    /// Build a module that compiles only the defs/lambdas NOT listed in
    /// `external_defs` / `external_lambdas`. Items in those sets are
    /// emitted as `declare`-only externals so the new module can call
    /// them; the JIT engine resolves them to the previously-installed
    /// addresses at `add_module` time.
    ///
    /// `type_id_base` is the starting `type_id` to use for new shapes
    /// (closures, structs, enums) — when installing incrementally, this
    /// must equal the runtime's current `type_table.len()` so freshly
    /// registered shapes get type_ids that don't collide with existing
    /// ones.
    ///
    /// All `external_defs` MUST appear in `rm.defs` (so we know their
    /// declared signature). All `external_lambdas` MUST appear in
    /// `rm`'s reachable lambda set.
    pub fn build_with_externals(
        context: &'ctx Context,
        rm: &ResolvedModule,
        external_defs: &HashSet<Hash>,
        external_lambdas: &HashSet<Hash>,
        type_id_base: u16,
    ) -> Result<Self, CodegenError> {
        Self::build_full(
            context,
            rm,
            external_defs,
            external_lambdas,
            &[],
            type_id_base,
        )
    }

    /// The most general build entry point. `extra_lambdas` is a list of
    /// `(hash, Expr::Lambda{...})` pairs to pre-seed into the codegen's
    /// lambda registry — useful for the incremental code-fetch path,
    /// where the server receives top-level `Expr::Lambda` items that
    /// aren't reachable from any def body in the install batch (the
    /// def body that constructs them lives on the client side and is
    /// never installed remotely).
    pub fn build_full(
        context: &'ctx Context,
        rm: &ResolvedModule,
        external_defs: &HashSet<Hash>,
        external_lambdas: &HashSet<Hash>,
        extra_lambdas: &[(Hash, Expr)],
        type_id_base: u16,
    ) -> Result<Self, CodegenError> {
        let mut cg = Codegen::new(context);
        cg.next_type_id = type_id_base;

        // ---- Pre-seed extra lambdas ----
        for (h, e) in extra_lambdas {
            match e {
                Expr::Lambda { params, body } => {
                    // Pointer params are fine; see scan_lambdas.
                    check_no_nested_lambdas(body)?;
                    let arity = params.len() as u32;
                    let captures = collect_captures(body, arity);
                    let n_caps = captures.len();
                    cg.lambdas.insert(
                        *h,
                        LambdaSpec {
                            params: params.clone(),
                            body: body.clone(),
                            captures,
                            capture_is_pointer: vec![false; n_caps],
                        },
                    );
                }
                other => {
                    return Err(CodegenError::Unsupported {
                        what: format!(
                            "extra_lambdas entry isn't a Lambda expression: {:?}",
                            other
                        ),
                    });
                }
            }
        }

        // ---- Pre-scan: collect every unique Lambda expression ----
        for rd in &rm.defs {
            match &rd.def {
                Def::Fn { body, .. } => cg.scan_lambdas(body)?,
                Def::State { init, .. } => cg.scan_lambdas(init)?,
                _ => {}
            }
        }

        // ---- Declare extern runtime functions ----
        cg.declare_runtime_externs();

        // ---- Declare user-defined `extern fn`s from the module ----
        // These come from surface `extern fn` decls. We emit one LLVM
        // extern declaration per declared name (`ext/<name>`); the
        // JIT-init phase wires each to its registered Rust fn pointer.
        // C-FFI externs additionally resolve their real library symbol
        // now, but only if the module actually references them.
        let mut referenced_externs = std::collections::HashSet::new();
        for rd in &rm.defs {
            match &rd.def {
                Def::Fn { body, .. } => collect_ext_refs(body, &mut referenced_externs),
                Def::State { init, .. } => collect_ext_refs(init, &mut referenced_externs),
                _ => {}
            }
        }
        cg.declare_user_externs(&rm.externs, &referenced_externs)?;

        // ---- Pass 1: declare every def's prototype ----
        //
        // Defs are declared BEFORE lambdas because the lambda
        // capture-pointer-inference walker (`infer_capture_pointer_flags`)
        // consults `def_signatures`, `struct_field_types`, and
        // `enum_variant_types`, all populated here.
        for rd in &rm.defs {
            cg.declare_def(rd)?;
        }

        // ---- Eta-expand first-class function references ----
        // A named top-level function used as a value (not as a direct
        // call callee) needs an adapter closure. Register those adapter
        // lambdas now — AFTER `declare_def` (we need `def_signatures`)
        // and BEFORE the lambda passes below pick up `cg.lambdas`.
        for rd in &rm.defs {
            match &rd.def {
                Def::Fn { body, .. } => cg.scan_fn_value_refs(body)?,
                Def::State { init, .. } => cg.scan_fn_value_refs(init)?,
                _ => {}
            }
        }

        // ---- Declare lifted lambda prototypes + register their TypeInfos ----
        // We iterate in a stable order so external/non-external classification
        // and type_id assignment are deterministic.
        let mut lambda_hashes: Vec<Hash> = cg.lambdas.keys().copied().collect();
        lambda_hashes.sort();
        // First, infer per-capture pointer-ness for every lambda — this
        // requires `def_signatures` etc. to be populated. We do it
        // before any `declare_lifted_lambda` call so the shape
        // registration sees the inferred flags.
        for h in &lambda_hashes {
            let flags = cg.infer_capture_pointer_flags(h);
            if let Some(spec) = cg.lambdas.get_mut(h) {
                spec.capture_is_pointer = flags;
            }
        }
        for h in &lambda_hashes {
            cg.declare_lifted_lambda(h)?;
        }

        // ---- Pass 2: compile bodies — skipping external defs/lambdas ----
        for rd in &rm.defs {
            if external_defs.contains(&rd.hash) {
                continue;
            }
            cg.compile_def(rd)?;
        }
        for h in &lambda_hashes {
            if external_lambdas.contains(h) {
                continue;
            }
            cg.compile_lifted_lambda(h)?;
        }

        // ---- Pin metadata globals via @llvm.compiler.used ----
        cg.emit_compiler_used();

        if let Err(msg) = cg.module.verify() {
            return Err(CodegenError::JitInit(format!(
                "LLVM module verification failed:\n{}",
                msg.to_string()
            )));
        }

        // ---- Optimize ----
        // Run the standard O2 pipeline over the module (mem2reg, inlining,
        // GVN, LICM, ...). The engine's OptimizationLevel only drives
        // instruction selection; without this the IR goes to ISel raw —
        // every alloca stays in memory and nothing inlines. GC safety
        // under optimization is carried by volatiles (root slots, the
        // epilogue pop, the safepoint state load) and by frame allocas
        // escaping into the thread's frame chain. Runs BEFORE the
        // bitcode cache is written, so cached modules are optimized too;
        // @llvm.compiler.used (emitted above) pins the metadata globals
        // through GlobalDCE.
        run_o2_passes(&cg.module)?;

        let closure_type_infos = cg.closure_type_infos.clone();
        let shape_registry = cg.shape_registry.clone();
        let shape_meta = cg.shape_meta.clone();
        let shape_by_type_id = cg.shape_by_type_id.clone();
        // State installers to run once at node startup, in dependency order
        // (rm.defs is dependency-first, so a state init referencing another
        // state installs after it).
        let state_hashes: Vec<Hash> = rm
            .defs
            .iter()
            .filter(|rd| matches!(rd.def, Def::State { .. }))
            .map(|rd| rd.hash)
            .collect();
        // Thunks that are not safe to memoize across an `at` call are
        // cache-unsafe. This is the effect-based superset of "touches node
        // state": it also rejects thunks performing IO / Net / Atom / FFI
        // effects (caching those would skip the real effect on a repeat
        // call). Local mutation of owned values and panic stay cacheable.
        // Includes standalone (code-fetch) lambdas so a shipped entry
        // closure that transitively does an effect is itself marked.
        // (Stored under the historical `stateful_hashes` name; the runtime
        // consults it as "bypass the result cache".)
        let stateful_hashes: Vec<Hash> =
            crate::knowledge::non_cacheable_hashes(&rm.defs, extra_lambdas)
                .into_iter()
                .collect();

        Ok(CompiledModule {
            context,
            module: cg.module,
            functions: cg.functions,
            lifted_lambdas: cg.lifted_lambdas,
            closure_type_infos,
            shape_registry,
            shape_meta,
            shape_by_type_id,
            state_hashes,
            stateful_hashes,
        })
    }

    pub fn ir(&self) -> String {
        self.module.print_to_string().to_string()
    }
}

// =============================================================================
// Codegen state
// =============================================================================

struct Codegen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    i64_ty: IntType<'ctx>,
    ptr_ty: PointerType<'ctx>,
    frame_origin_ty: StructType<'ctx>,
    /// Highest root-slot index + 1 actually written while compiling the
    /// CURRENT function body (every slot write funnels through
    /// `write_root_slot`). Lets `finalize_frame_zeroing` shrink the
    /// prologue memset and the origin's scanned-slot count from the
    /// conservative pre-scan reservation (often 10-100x too big) down
    /// to what the body really uses.
    cur_frame_max_slot: u32,
    /// (frame alloca, dead marker instruction) for the current function:
    /// the marker pins where the right-sized memset is inserted once the
    /// body is compiled and the real slot count is known.
    cur_frame_zero_point: Option<(PointerValue<'ctx>, inkwell::values::InstructionValue<'ctx>)>,

    functions: HashMap<Hash, FunctionValue<'ctx>>,
    /// Node `state` installer functions, keyed by the state's content hash.
    /// Run once per node to populate the state table.
    state_installers: HashMap<Hash, FunctionValue<'ctx>>,
    lifted_lambdas: HashMap<Hash, FunctionValue<'ctx>>,
    frame_origins: HashMap<String, GlobalValue<'ctx>>,
    used_globals: Vec<PointerValue<'ctx>>,

    /// Unique lambdas discovered during pre-scan, keyed by content hash.
    lambdas: HashMap<Hash, LambdaSpec>,

    /// TypeInfo registered for each lambda shape, in type_id order.
    /// (This is misnamed — it now holds ALL heap-object TypeInfos, both
    /// closures AND structs. Renaming is a future cleanup.)
    closure_type_infos: Vec<TypeInfo>,
    /// LLVM global holding a constant TypeInfo per lambda, used as the
    /// argument to `ai_gc_alloc_closure`.
    closure_type_info_globals: HashMap<Hash, GlobalValue<'ctx>>,

    /// Per-struct codegen info: physical layout + TypeInfo global.
    structs: HashMap<Hash, StructInfo<'ctx>>,

    /// Per-enum codegen info: per-variant TypeInfo globals + payload
    /// layout. Indexed by enum's content hash.
    enums: HashMap<Hash, EnumInfo<'ctx>>,

    /// Mirror of `CompiledModule::shape_registry`, built as we register
    /// each shape's TypeInfo.
    shape_registry: HashMap<Hash, u16>,

    /// Mirror of `CompiledModule::shape_meta`.
    shape_meta: HashMap<Hash, ShapeMeta>,

    /// Mirror of `CompiledModule::shape_by_type_id`.
    shape_by_type_id: Vec<Option<Hash>>,

    /// Per-def or per-lambda metadata captured during pass 1.
    def_info: HashMap<Hash, DefInfo<'ctx>>,
    lambda_info: HashMap<Hash, DefInfo<'ctx>>,

    /// Extern declarations for runtime fns.
    extern_alloc_closure: Option<FunctionValue<'ctx>>,
    extern_lookup_code: Option<FunctionValue<'ctx>>,
    extern_net_at: Option<FunctionValue<'ctx>>,
    extern_net_at_async: Option<FunctionValue<'ctx>>,
    /// Located-state (distributed atom) runtime fns. Each returns a
    /// heap `Result<T, Failure>` pointer, like `ai_net_at`.
    extern_box_int: Option<FunctionValue<'ctx>>,
    extern_unbox_int: Option<FunctionValue<'ctx>>,
    extern_force_collect: Option<FunctionValue<'ctx>>,
    /// `ai_gc_pollcheck_slow(thread, origin)` — the safepoint slow path,
    /// branched to from the inline poll emitted at every `loop_body`
    /// entry (function entry + self-tail-call backedge). Parks the
    /// thread when another thread has requested a STW collection.
    extern_pollcheck: Option<FunctionValue<'ctx>>,
    extern_state_get: Option<FunctionValue<'ctx>>,
    extern_state_present: Option<FunctionValue<'ctx>>,
    extern_state_set: Option<FunctionValue<'ctx>>,
    extern_value_hash: Option<FunctionValue<'ctx>>,
    extern_value_eq: Option<FunctionValue<'ctx>>,
    extern_abort: Option<FunctionValue<'ctx>>,
    extern_str_new: Option<FunctionValue<'ctx>>,
    extern_str_len: Option<FunctionValue<'ctx>>,
    extern_str_eq: Option<FunctionValue<'ctx>>,
    extern_str_concat: Option<FunctionValue<'ctx>>,
    extern_bytes_new: Option<FunctionValue<'ctx>>,
    extern_bytes_get: Option<FunctionValue<'ctx>>,
    extern_bytes_set: Option<FunctionValue<'ctx>>,
    extern_bytes_slice: Option<FunctionValue<'ctx>>,
    extern_bytes_copy: Option<FunctionValue<'ctx>>,
    extern_str_copy: Option<FunctionValue<'ctx>>,
    extern_array_new: Option<FunctionValue<'ctx>>,
    extern_array_new_prim: Option<FunctionValue<'ctx>>,
    extern_array_len: Option<FunctionValue<'ctx>>,
    extern_array_get: Option<FunctionValue<'ctx>>,
    extern_array_set: Option<FunctionValue<'ctx>>,
    extern_array_get_i64: Option<FunctionValue<'ctx>>,
    extern_array_set_i64: Option<FunctionValue<'ctx>>,
    extern_atom_swap_local: Option<FunctionValue<'ctx>>,
    extern_atom_new: Option<FunctionValue<'ctx>>,
    extern_atom_load: Option<FunctionValue<'ctx>>,
    extern_thread_spawn: Option<FunctionValue<'ctx>>,
    extern_thread_spawn_shared: Option<FunctionValue<'ctx>>,
    extern_thread_join: Option<FunctionValue<'ctx>>,
    /// Value-boundary fns exposing the wire codec + closure invocation to
    /// ai-lang, so a node loop can be written in the language.
    extern_wire_encode: Option<FunctionValue<'ctx>>,
    extern_wire_decode_int: Option<FunctionValue<'ctx>>,
    extern_wire_decode_ptr: Option<FunctionValue<'ctx>>,
    extern_wire_decode_checked: Option<FunctionValue<'ctx>>,
    extern_wire_invoke: Option<FunctionValue<'ctx>>,

    /// In-flight tail-call context: only populated while compiling a
    /// single fn body. Calls inside the body that target this fn's
    /// hash AND sit in tail position get rewritten as branches back
    /// to `loop_body`, after storing the new arg values into the
    /// per-param slots. `None` outside `compile_def`.
    tail_ctx: Option<TailCtx<'ctx>>,

    /// Per-def declared param and return types — looked up by
    /// `infer_type` when crossing generic call boundaries to decide
    /// where to box/unbox.
    def_signatures: HashMap<Hash, FnSigSimple>,
    /// Declared type of each node `state` binding, keyed by content hash.
    /// `infer_type` consults this so `StateRef`-based boxing decisions
    /// (e.g. `deref`/`swap` over `Atom<Int>`) are correct.
    state_types: HashMap<Hash, Type>,
    /// Per-struct field types in declaration order, for `Field`
    /// projection's inferred type.
    struct_field_types: HashMap<Hash, Vec<Type>>,
    /// Per-enum variant payload types — `None` for nullary variants.
    enum_variant_types: HashMap<Hash, Vec<Option<Type>>>,

    /// Next type_id to assign to a newly-registered shape. Defaults to
    /// 0 (single-shot compile) but the incremental path sets it to
    /// the runtime's current type_table length so new shapes don't
    /// collide with previously-installed ones.
    next_type_id: u16,

    /// User `extern fn` signatures by surface name, captured at
    /// `declare_user_externs`. Lets the call-site codegen tell a host
    /// extern (leading `Thread*`, registry-resolved) from a C-FFI extern
    /// (plain C ABI, dlsym-resolved) by looking up `ExternSig.library`.
    user_externs: HashMap<String, crate::resolve::ExternSig>,

    /// Stack of pending `defer` cleanups for the function/lambda body
    /// currently being compiled. Each entry captures the cleanup
    /// expression plus the environment snapshot (slot addresses + de
    /// Bruijn depth) at the `defer` site, so a `?` early-return can emit
    /// the cleanups (LIFO) before returning. Pushed on entering a `Defer`
    /// body, popped on exit. Always empty between top-level defs.
    deferred: Vec<DeferEntry<'ctx>>,
}

/// A registered-but-not-yet-run `defer` cleanup. See `Codegen.deferred`.
#[derive(Clone)]
struct DeferEntry<'ctx> {
    cleanup: Expr,
    env: Env<'ctx>,
    next_root_slot: u32,
}

#[derive(Clone)]
struct LambdaSpec {
    /// The lambda's declared parameter types (all `Int` in v1).
    params: Vec<Type>,
    /// The lambda's canonical body.
    body: Box<Expr>,
    /// Outer de Bruijn indices captured, in ascending stable order.
    /// E.g. if the body uses `LocalVar(arity)` and `LocalVar(arity + 2)`,
    /// captures = [0, 2].
    captures: Vec<u32>,
    /// Per-capture pointer-ness, populated *after* `declare_def`
    /// runs (when struct/enum/fn signatures are available for the
    /// body walker). Same length as `captures`. `true` means the
    /// capture slot holds a heap pointer (GC-traced).
    capture_is_pointer: Vec<bool>,
}

#[derive(Copy, Clone)]
struct DefInfo<'ctx> {
    /// Whether this function pushes a GC frame at all. False when the
    /// body holds NO pointer roots and no `defer` (e.g. all-scalar
    /// recursion like fib): no alloca, no memset, no chain link/unlink —
    /// the GC walker simply never sees such frames, and there was
    /// nothing in them to trace. This is the Go lesson: pointer-free
    /// frames pay zero GC bookkeeping. Provisionally `true` at declare
    /// time; the compile pass recomputes it with full signature info.
    has_frame: bool,
    /// `{ ptr parent, ptr origin, [0 x ptr] roots }` — a PLACEHOLDER
    /// layout shared by every function, used only for the fixed-offset
    /// GEPs (parent 0, origin 8, roots base 16). There is NO
    /// conservative slot reservation anywhere: the prologue allocas
    /// this placeholder, the body compiles against it, and
    /// `finalize_frame_zeroing` replaces it with an alloca of EXACTLY
    /// the high-water slot count the body wrote (and sizes the memset
    /// and the origin's scanned-slot count to match).
    frame_ty: StructType<'ctx>,
}

#[derive(Clone)]
struct StructInfo<'ctx> {
    /// LLVM global holding the TypeInfo for this struct shape.
    ti_global: GlobalValue<'ctx>,
    /// Physical byte offset (from start of heap object, i.e. including
    /// the GC header) of each field, indexed by declaration position.
    /// Pointer fields are packed first (at header_size + i*8), value
    /// fields follow (raw bytes section).
    field_offsets: Vec<u32>,
    /// Whether each field (in declaration order) is pointer-typed.
    field_is_pointer: Vec<bool>,
}

#[derive(Clone)]
struct EnumInfo<'ctx> {
    /// One entry per variant, in declaration order.
    variants: Vec<VariantInfo<'ctx>>,
}

#[derive(Clone, Copy)]
struct VariantInfo<'ctx> {
    /// LLVM global holding this variant's TypeInfo (used at alloc time).
    ti_global: GlobalValue<'ctx>,
    /// Byte offset (from object start, post-header) of the tag word.
    /// Varies by variant: pointer-payload variants have the pointer
    /// first (in value_fields), so the tag follows.
    tag_offset: u32,
    /// Byte offset of the payload value (None for nullary variants).
    payload_offset: Option<u32>,
    /// Whether this variant's payload is pointer-typed (else Int).
    /// Meaningless if `payload_offset` is None.
    payload_is_pointer: bool,
}

impl<'ctx> Codegen<'ctx> {
    fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("ai_lang");
        let builder = context.create_builder();
        let i64_ty = context.i64_type();
        let ptr_ty = context.ptr_type(AddressSpace::default());
        let i32_ty = context.i32_type();
        let frame_origin_ty =
            context.struct_type(&[i32_ty.into(), i32_ty.into(), ptr_ty.into()], false);
        Codegen {
            context,
            module,
            builder,
            i64_ty,
            ptr_ty,
            frame_origin_ty,
            cur_frame_max_slot: 0,
            cur_frame_zero_point: None,
            functions: HashMap::new(),
            state_installers: HashMap::new(),
            lifted_lambdas: HashMap::new(),
            frame_origins: HashMap::new(),
            used_globals: Vec::new(),
            lambdas: HashMap::new(),
            closure_type_infos: Vec::new(),
            closure_type_info_globals: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            shape_registry: HashMap::new(),
            shape_meta: HashMap::new(),
            shape_by_type_id: Vec::new(),
            def_info: HashMap::new(),
            lambda_info: HashMap::new(),
            extern_alloc_closure: None,
            extern_lookup_code: None,
            extern_net_at: None,
            extern_net_at_async: None,
            extern_box_int: None,
            extern_unbox_int: None,
            extern_force_collect: None,
            extern_pollcheck: None,
            extern_state_get: None,
            extern_state_present: None,
            extern_state_set: None,
            extern_value_hash: None,
            extern_value_eq: None,
            extern_abort: None,
            extern_str_new: None,
            extern_str_len: None,
            extern_str_eq: None,
            extern_str_concat: None,
            extern_bytes_new: None,
            extern_bytes_get: None,
            extern_bytes_set: None,
            extern_bytes_slice: None,
            extern_bytes_copy: None,
            extern_str_copy: None,
            extern_array_new: None,
            extern_array_new_prim: None,
            extern_array_len: None,
            extern_array_get: None,
            extern_array_set: None,
            extern_array_get_i64: None,
            extern_array_set_i64: None,
            extern_atom_swap_local: None,
            extern_atom_new: None,
            extern_atom_load: None,
            extern_thread_spawn: None,
            extern_thread_spawn_shared: None,
            extern_thread_join: None,
            extern_wire_encode: None,
            extern_wire_decode_int: None,
            extern_wire_decode_ptr: None,
            extern_wire_decode_checked: None,
            extern_wire_invoke: None,
            tail_ctx: None,
            def_signatures: HashMap::new(),
            state_types: HashMap::new(),
            struct_field_types: HashMap::new(),
            enum_variant_types: HashMap::new(),
            next_type_id: 0,
            user_externs: HashMap::new(),
            deferred: Vec::new(),
        }
    }

    /// Allocate the next `type_id` for a fresh shape. Bumps the counter
    /// AND pushes the TypeInfo into `closure_type_infos` at the slot
    /// matching the new id (extending with placeholder slots if needed
    /// so indices line up with the runtime's contiguous type_table).
    fn next_type_id(&mut self) -> u16 {
        let id = self.next_type_id;
        self.next_type_id = self
            .next_type_id
            .checked_add(1)
            .expect("type_id overflow (more than 65 535 shapes registered)");
        id
    }

    // -------------------------------------------------------------------------
    // Lambda pre-scan
    // -------------------------------------------------------------------------

    /// Walk an expression, registering every nested `Lambda` we find.
    /// Rejects nested lambdas (Lambda inside Lambda body) per the v1
    /// restrictions. Lambda params may be any type (pointer-typed params
    /// are passed via the uniform closure ABI).
    fn scan_lambdas(&mut self, e: &Expr) -> Result<(), CodegenError> {
        match e {
            Expr::Lambda { params, body } => {
                // Pointer-typed lambda params are fine — the
                // uniform closure ABI passes them as ptr and
                // `compile_lifted_lambda` puts them in a root slot.
                // Reject nested lambdas inside the body.
                check_no_nested_lambdas(body)?;

                // Compute the lambda's content hash.
                let lambda_expr = Expr::Lambda {
                    params: params.clone(),
                    body: body.clone(),
                };
                let bytes = encode_expr(&lambda_expr);
                let hash = Hash::of_bytes(&bytes);

                if !self.lambdas.contains_key(&hash) {
                    let arity = params.len() as u32;
                    let captures = collect_captures(body, arity);
                    let n_caps = captures.len();
                    self.lambdas.insert(
                        hash,
                        LambdaSpec {
                            params: params.clone(),
                            body: body.clone(),
                            captures,
                            capture_is_pointer: vec![false; n_caps],
                        },
                    );
                }
                // Also walk the lambda body to register lambdas it
                // contains (in expression position other than the
                // outermost — there shouldn't be any thanks to the
                // nested-lambda check, but a Let value etc. could
                // contain something. Actually no: check_no_nested_lambdas
                // covers that. We still walk to be defensive.)
                self.scan_lambdas(body)?;
            }
            Expr::Call(callee, args) => {
                self.scan_lambdas(callee)?;
                for a in args {
                    self.scan_lambdas(a)?;
                }
            }
            Expr::Let { value, body } => {
                self.scan_lambdas(value)?;
                self.scan_lambdas(body)?;
            }
            Expr::StructNew { fields, .. } => {
                for f in fields {
                    self.scan_lambdas(f)?;
                }
            }
            Expr::Field { base, .. } => {
                self.scan_lambdas(base)?;
            }
            Expr::EnumNew { payload, .. } => {
                if let Some(p) = payload {
                    self.scan_lambdas(p)?;
                }
            }
            Expr::Match { scrutinee, arms } => {
                self.scan_lambdas(scrutinee)?;
                for arm in arms {
                    self.scan_lambdas(&arm.body)?;
                }
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.scan_lambdas(cond)?;
                self.scan_lambdas(then_branch)?;
                self.scan_lambdas(else_branch)?;
            }
            Expr::Try { expr, .. } => {
                self.scan_lambdas(expr)?;
            }
            Expr::Defer { cleanup, body } => {
                self.scan_lambdas(cleanup)?;
                self.scan_lambdas(body)?;
            }
            Expr::IntLit(_)
            | Expr::FloatLit(_)
            | Expr::BoolLit(_)
            | Expr::StringLit(_)
            | Expr::LocalVar(_)
            | Expr::TopRef(_)
            | Expr::SelfRef(_)
            | Expr::StateRef(_)
            | Expr::StateSelfRef(_)
            | Expr::BuiltinRef(_) => {}
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // First-class function references (eta-expansion)
    // -------------------------------------------------------------------------
    //
    // A bare reference to a function used as a VALUE — a named top-level
    // `def` (`Expr::TopRef`, e.g. `swap(counter, inc)`) or a core builtin
    // (`Expr::BuiltinRef`, e.g. `list_map(strs, string_len)`) — has no
    // closure object to call indirectly. We lower it by eta-expanding the
    // reference into an adapter closure: a lifted lambda with the uniform
    // closure ABI whose body just forwards its params to a direct call of
    // the referenced function/builtin. The adapter has no captures (it
    // references only its own params + the global ref), so it reuses the
    // entire existing lambda machinery unchanged.
    //
    // This is purely a codegen lowering detail: it does not touch the
    // stored AST or content hashes. `def_signatures` must already be
    // populated (i.e. run this AFTER `declare_def`).

    /// Build the `(params, body)` of the adapter lambda for a value-position
    /// reference `callee` (a `TopRef` or `BuiltinRef`). `None` if we can't
    /// form an adapter:
    ///   - `TopRef` to an unknown hash (not a function def),
    ///   - `BuiltinRef` with no signature (call-site-special, e.g.
    ///     `core/net.at`).
    ///
    /// Generic builtins (a `TypeVar` in the signature, e.g.
    /// `core/array.get`) ARE supported: the adapter carries the generic
    /// `FnType` and composes through the uniform closure ABI exactly like a
    /// generic named function used as a value — everything is a boxed
    /// pointer in the generic context, and the concrete (un)boxing happens
    /// at the instantiation boundary (the direct call that pins the
    /// TypeVars; see the `TopRef` arm of `compile_call`).
    fn value_ref_adapter_parts(&self, callee: &Expr) -> Option<(Vec<Type>, Expr)> {
        let params: Vec<Type> = match callee {
            Expr::TopRef(h) => self.def_signatures.get(h)?.params.clone(),
            Expr::BuiltinRef(name) => crate::typecheck::builtin_signature(name)?.0,
            _ => return None,
        };
        let arity = params.len() as u32;
        // Forward params in declared order. The lifted env lays out
        // params as LocalVar(arity-1-j) for declared param `j` (param 0
        // is the outermost binder, hence the highest de Bruijn index).
        let args: Vec<Expr> = (0..arity)
            .map(|j| Expr::LocalVar(arity - 1 - j))
            .collect();
        let body = Expr::Call(Box::new(callee.clone()), args);
        Some((params, body))
    }

    /// Error describing why a value-position reference can't be eta-expanded.
    fn value_ref_unsupported(callee: &Expr) -> CodegenError {
        let what = match callee {
            Expr::TopRef(h) => format!(
                "first-class reference to unknown function {} \
                 (no signature available)",
                h
            ),
            Expr::BuiltinRef(name) => format!(
                "first-class reference to builtin `{}` — it is either \
                 call-site-special (e.g. `at`) or generic, and cannot be \
                 used as a bare value; wrap it in a lambda instead, e.g. \
                 `|x| {}(x)`",
                name, name
            ),
            Expr::SelfRef(_) => "first-class self-reference — the resolver \
                 rewrites self-references to top-level references before \
                 codegen, so this should be unreachable"
                .to_owned(),
            other => format!("first-class reference to non-reference expr {:?}", other),
        };
        CodegenError::Unsupported { what }
    }

    /// Register the adapter lambda for a value-position reference, if not
    /// already present, so the normal lambda passes (pointer-flag
    /// inference, declare, compile) emit it.
    ///
    /// A reference we can't adapt (generic/special builtin, unknown fn) is
    /// silently skipped here rather than rejected: it may live in an
    /// external def that is never compiled. If it IS compiled, `compile_expr`
    /// hits `value_ref_adapter_parts` → `None` and emits a precise error at
    /// the real use site.
    fn register_value_ref_adapter(&mut self, callee: &Expr) -> Result<(), CodegenError> {
        let Some((params, body)) = self.value_ref_adapter_parts(callee) else {
            return Ok(());
        };
        let lambda_expr = Expr::Lambda {
            params: params.clone(),
            body: Box::new(body.clone()),
        };
        let hash = Hash::of_bytes(&encode_expr(&lambda_expr));
        self.lambdas.entry(hash).or_insert(LambdaSpec {
            params,
            body: Box::new(body),
            captures: Vec::new(),
            capture_is_pointer: Vec::new(),
        });
        Ok(())
    }

    /// Walk an expression and register an adapter lambda for every
    /// function/builtin reference that appears in VALUE position (anywhere
    /// except as the immediate callee of a `Call`, which is a direct call).
    fn scan_fn_value_refs(&mut self, e: &Expr) -> Result<(), CodegenError> {
        match e {
            Expr::TopRef(_) | Expr::BuiltinRef(_) => self.register_value_ref_adapter(e)?,
            Expr::Call(callee, args) => {
                // A bare ref callee is a DIRECT call, not a value — skip it.
                // Anything else in callee position (an indirect call through
                // a computed closure) is scanned normally.
                if !matches!(
                    callee.as_ref(),
                    Expr::TopRef(_) | Expr::SelfRef(_) | Expr::BuiltinRef(_)
                ) {
                    self.scan_fn_value_refs(callee)?;
                }
                for a in args {
                    self.scan_fn_value_refs(a)?;
                }
            }
            Expr::Lambda { body, .. } => self.scan_fn_value_refs(body)?,
            Expr::Let { value, body } => {
                self.scan_fn_value_refs(value)?;
                self.scan_fn_value_refs(body)?;
            }
            Expr::StructNew { fields, .. } => {
                for f in fields {
                    self.scan_fn_value_refs(f)?;
                }
            }
            Expr::Field { base, .. } => self.scan_fn_value_refs(base)?,
            Expr::EnumNew { payload, .. } => {
                if let Some(p) = payload {
                    self.scan_fn_value_refs(p)?;
                }
            }
            Expr::Match { scrutinee, arms } => {
                self.scan_fn_value_refs(scrutinee)?;
                for arm in arms {
                    self.scan_fn_value_refs(&arm.body)?;
                }
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.scan_fn_value_refs(cond)?;
                self.scan_fn_value_refs(then_branch)?;
                self.scan_fn_value_refs(else_branch)?;
            }
            Expr::Try { expr, .. } => self.scan_fn_value_refs(expr)?,
            Expr::Defer { cleanup, body } => {
                self.scan_fn_value_refs(cleanup)?;
                self.scan_fn_value_refs(body)?;
            }
            Expr::IntLit(_)
            | Expr::FloatLit(_)
            | Expr::BoolLit(_)
            | Expr::StringLit(_)
            | Expr::LocalVar(_)
            // SelfRef is rewritten to TopRef by the resolver before
            // codegen; if one ever reaches here it's a value-position
            // self-reference, errored on by compile_expr. A StateRef is a
            // value (the live cell), not a fn reference, so it needs no
            // adapter.
            | Expr::SelfRef(_)
            | Expr::StateRef(_)
            | Expr::StateSelfRef(_) => {}
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Runtime externs
    // -------------------------------------------------------------------------

    fn declare_runtime_externs(&mut self) {
        // ai_gc_alloc_closure(thread, type_info) -> ptr
        let alloc_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let alloc =
            self.module
                .add_function("ai_gc_alloc_closure", alloc_ty, Some(Linkage::External));
        self.extern_alloc_closure = Some(alloc);

        // ai_gc_lookup_code(thread, hash_ptr) -> ptr
        let lookup_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let lookup =
            self.module
                .add_function("ai_gc_lookup_code", lookup_ty, Some(Linkage::External));
        self.extern_lookup_code = Some(lookup);

        // ai_net_at(thread, node_ptr, closure_ptr) -> *u8
        // Language-level `at(node, thunk)` lowers to this call. The
        // returned pointer is the heap-allocated `Result` enum value
        // the runtime constructed (Ok(Int) on success, Err(Failure)
        // on a network error).
        let net_at_ty = self.ptr_ty.fn_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.ptr_ty.into(),
            ],
            false,
        );
        let net_at =
            self.module
                .add_function("ai_net_at", net_at_ty, Some(Linkage::External));
        self.extern_net_at = Some(net_at);

        // ai_net_at_async(thread, node_ptr, closure_ptr) -> *u8 — ships
        // the thunk from a background thread and returns a ThreadHandle
        // (BoxedInt registry id) immediately; `join` awaits the
        // Result<T, Failure>.
        let net_at_async_ty = self.ptr_ty.fn_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.ptr_ty.into(),
            ],
            false,
        );
        let net_at_async = self.module.add_function(
            "ai_net_at_async",
            net_at_async_ty,
            Some(Linkage::External),
        );
        self.extern_net_at_async = Some(net_at_async);

        // ai_wire_encode(thread, value_ptr) -> *u8 (Bytes)
        let wire_encode_ty =
            self.ptr_ty.fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        self.extern_wire_encode = Some(self.module.add_function(
            "ai_wire_encode",
            wire_encode_ty,
            Some(Linkage::External),
        ));
        // ai_wire_decode_int(thread, bytes_ptr) -> i64
        let wire_decode_int_ty =
            self.i64_ty.fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        self.extern_wire_decode_int = Some(self.module.add_function(
            "ai_wire_decode_int",
            wire_decode_int_ty,
            Some(Linkage::External),
        ));
        // ai_wire_decode_ptr(thread, bytes_ptr) -> *u8 (decoded heap value)
        let wire_decode_ptr_ty =
            self.ptr_ty.fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        self.extern_wire_decode_ptr = Some(self.module.add_function(
            "ai_wire_decode_ptr",
            wire_decode_ptr_ty,
            Some(Linkage::External),
        ));
        // ai_wire_decode_checked(thread, bytes, h0, h1, h2, h3) -> *u8 (Result)
        let wire_decode_checked_ty = self.ptr_ty.fn_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.i64_ty.into(),
                self.i64_ty.into(),
                self.i64_ty.into(),
                self.i64_ty.into(),
            ],
            false,
        );
        self.extern_wire_decode_checked = Some(self.module.add_function(
            "ai_wire_decode_checked",
            wire_decode_checked_ty,
            Some(Linkage::External),
        ));
        // ai_wire_invoke(thread, closure_bytes_ptr) -> *u8 (Bytes)
        let wire_invoke_ty =
            self.ptr_ty.fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        self.extern_wire_invoke = Some(self.module.add_function(
            "ai_wire_invoke",
            wire_invoke_ty,
            Some(Linkage::External),
        ));

        // ai_gc_box_int(thread, i64) -> *u8 — boxes an Int into a
        // BoxedInt heap object so generic-typed slots can store it.
        let box_int_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.i64_ty.into()], false);
        let box_int =
            self.module
                .add_function("ai_gc_box_int", box_int_ty, Some(Linkage::External));
        self.extern_box_int = Some(box_int);

        // ai_gc_unbox_int(*u8) -> i64 — extracts the Int from a
        // BoxedInt heap object.
        let unbox_int_ty = self.i64_ty.fn_type(&[self.ptr_ty.into()], false);
        let unbox_int =
            self.module
                .add_function("ai_gc_unbox_int", unbox_int_ty, Some(Linkage::External));
        self.extern_unbox_int = Some(unbox_int);

        // ai_gc_force_collect(thread) -> i64 — language-visible hook
        // that triggers a stop-the-world collection. Used by the
        // `gc_collect()` builtin to verify root-tracking in tests.
        let force_collect_ty = self.i64_ty.fn_type(&[self.ptr_ty.into()], false);
        let force_collect = self.module.add_function(
            "ai_gc_force_collect",
            force_collect_ty,
            Some(Linkage::External),
        );
        self.extern_force_collect = Some(force_collect);

        // ai_gc_pollcheck_slow(thread, origin) -> void — safepoint slow
        // path. The inline poll (load thread.state; branch-if-nonzero)
        // calls this when a STW collection has been requested by another
        // thread. Single-threaded, state stays 0 and this is never hit.
        let pollcheck_ty = self
            .context
            .void_type()
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let pollcheck = self.module.add_function(
            "ai_gc_pollcheck_slow",
            pollcheck_ty,
            Some(Linkage::External),
        );
        self.extern_pollcheck = Some(pollcheck);

        // Node `state` primitives. Each takes a pointer to 32 content-hash
        // bytes (a private module constant emitted at the reference / install
        // site).
        // ai_state_get(thread, hash_ptr) -> ptr (the live cell)
        let state_get_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let state_get =
            self.module
                .add_function("ai_state_get", state_get_ty, Some(Linkage::External));
        self.extern_state_get = Some(state_get);
        // ai_state_present(thread, hash_ptr) -> i64
        let state_present_ty = self
            .i64_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let state_present = self.module.add_function(
            "ai_state_present",
            state_present_ty,
            Some(Linkage::External),
        );
        self.extern_state_present = Some(state_present);
        // ai_state_set(thread, hash_ptr, val) -> i64
        let state_set_ty = self.i64_ty.fn_type(
            &[self.ptr_ty.into(), self.ptr_ty.into(), self.ptr_ty.into()],
            false,
        );
        let state_set =
            self.module
                .add_function("ai_state_set", state_set_ty, Some(Linkage::External));
        self.extern_state_set = Some(state_set);

        // ai_value_hash(thread, v_ptr) -> i64 — structural hash of any value.
        let value_hash_ty = self
            .i64_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let value_hash =
            self.module
                .add_function("ai_value_hash", value_hash_ty, Some(Linkage::External));
        self.extern_value_hash = Some(value_hash);
        // ai_value_eq(thread, a_ptr, b_ptr) -> i64 — structural equality.
        let value_eq_ty = self.i64_ty.fn_type(
            &[self.ptr_ty.into(), self.ptr_ty.into(), self.ptr_ty.into()],
            false,
        );
        let value_eq =
            self.module
                .add_function("ai_value_eq", value_eq_ty, Some(Linkage::External));
        self.extern_value_eq = Some(value_eq);

        // ai_abort(thread, msg_ptr) -> ! — hard-aborts the process with
        // the message. The single fate of a contract violation; every
        // call site terminates its block with `unreachable`.
        let abort_ty = self
            .context
            .void_type()
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let abort_fn = self
            .module
            .add_function("ai_abort", abort_ty, Some(Linkage::External));
        self.extern_abort = Some(abort_fn);

        // String runtime fns. Layout: see runtime::ai_str_*.
        let str_new_ty = self.ptr_ty.fn_type(
            &[self.ptr_ty.into(), self.ptr_ty.into(), self.i64_ty.into()],
            false,
        );
        let str_new = self.module.add_function(
            "ai_str_new",
            str_new_ty,
            Some(Linkage::External),
        );
        self.extern_str_new = Some(str_new);

        let str_len_ty = self.i64_ty.fn_type(&[self.ptr_ty.into()], false);
        let str_len = self.module.add_function(
            "ai_str_len",
            str_len_ty,
            Some(Linkage::External),
        );
        self.extern_str_len = Some(str_len);

        let str_eq_ty = self
            .i64_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let str_eq = self.module.add_function(
            "ai_str_eq",
            str_eq_ty,
            Some(Linkage::External),
        );
        self.extern_str_eq = Some(str_eq);

        let str_concat_ty = self.ptr_ty.fn_type(
            &[self.ptr_ty.into(), self.ptr_ty.into(), self.ptr_ty.into()],
            false,
        );
        let str_concat = self.module.add_function(
            "ai_str_concat",
            str_concat_ty,
            Some(Linkage::External),
        );
        self.extern_str_concat = Some(str_concat);

        // Bytes runtime fns. Layout shares String's heap shape; see
        // runtime::ai_bytes_*.
        // ai_bytes_new(thread, len) -> *u8
        let bytes_new_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.i64_ty.into()], false);
        let bytes_new = self.module.add_function(
            "ai_bytes_new",
            bytes_new_ty,
            Some(Linkage::External),
        );
        self.extern_bytes_new = Some(bytes_new);

        // ai_bytes_get(thread, bytes, i) -> i64
        let bytes_get_ty = self.i64_ty.fn_type(
            &[self.ptr_ty.into(), self.ptr_ty.into(), self.i64_ty.into()],
            false,
        );
        let bytes_get = self.module.add_function(
            "ai_bytes_get",
            bytes_get_ty,
            Some(Linkage::External),
        );
        self.extern_bytes_get = Some(bytes_get);

        // ai_bytes_set(thread, bytes, i, v) -> i64
        let bytes_set_ty = self.i64_ty.fn_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.i64_ty.into(),
                self.i64_ty.into(),
            ],
            false,
        );
        let bytes_set = self.module.add_function(
            "ai_bytes_set",
            bytes_set_ty,
            Some(Linkage::External),
        );
        self.extern_bytes_set = Some(bytes_set);

        // ai_bytes_slice(thread, bytes, start, len) -> *u8
        let bytes_slice_ty = self.ptr_ty.fn_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.i64_ty.into(),
                self.i64_ty.into(),
            ],
            false,
        );
        let bytes_slice = self.module.add_function(
            "ai_bytes_slice",
            bytes_slice_ty,
            Some(Linkage::External),
        );
        self.extern_bytes_slice = Some(bytes_slice);

        // ai_bytes_copy(thread, src) -> *u8 — backs bytes_from_string
        // (copies a String's bytes into a fresh, mutable Bytes).
        let bytes_copy_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let bytes_copy = self.module.add_function(
            "ai_bytes_copy",
            bytes_copy_ty,
            Some(Linkage::External),
        );
        self.extern_bytes_copy = Some(bytes_copy);

        // ai_str_copy(thread, src) -> *u8 — backs string_from_bytes
        // (copies a Bytes's bytes into a fresh, immutable String).
        let str_copy_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let str_copy = self.module.add_function(
            "ai_str_copy",
            str_copy_ty,
            Some(Linkage::External),
        );
        self.extern_str_copy = Some(str_copy);

        // Array runtime fns. Heap shape = Runtime.array_ti (varlen
        // Values / pointer slots). See runtime::ai_array_*.
        // ai_array_new(thread, n) -> *u8
        let array_new_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.i64_ty.into()], false);
        let array_new = self.module.add_function(
            "ai_array_new",
            array_new_ty,
            Some(Linkage::External),
        );
        self.extern_array_new = Some(array_new);

        // ai_array_new_prim(thread, n) -> *u8 (unboxed scalar slots)
        let array_new_prim_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.i64_ty.into()], false);
        let array_new_prim = self.module.add_function(
            "ai_array_new_prim",
            array_new_prim_ty,
            Some(Linkage::External),
        );
        self.extern_array_new_prim = Some(array_new_prim);

        // ai_array_len(thread, array) -> i64
        let array_len_ty = self
            .i64_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let array_len = self.module.add_function(
            "ai_array_len",
            array_len_ty,
            Some(Linkage::External),
        );
        self.extern_array_len = Some(array_len);

        // ai_array_get_i64(thread, array, i) -> i64 (scalar fast path)
        let array_get_i64_ty = self.i64_ty.fn_type(
            &[self.ptr_ty.into(), self.ptr_ty.into(), self.i64_ty.into()],
            false,
        );
        let array_get_i64 = self.module.add_function(
            "ai_array_get_i64",
            array_get_i64_ty,
            Some(Linkage::External),
        );
        self.extern_array_get_i64 = Some(array_get_i64);

        // ai_array_set_i64(thread, array, i, v) -> i64 (scalar fast path)
        let array_set_i64_ty = self.i64_ty.fn_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.i64_ty.into(),
                self.i64_ty.into(),
            ],
            false,
        );
        let array_set_i64 = self.module.add_function(
            "ai_array_set_i64",
            array_set_i64_ty,
            Some(Linkage::External),
        );
        self.extern_array_set_i64 = Some(array_set_i64);

        // ai_array_get(thread, array, i) -> *u8
        let array_get_ty = self.ptr_ty.fn_type(
            &[self.ptr_ty.into(), self.ptr_ty.into(), self.i64_ty.into()],
            false,
        );
        let array_get = self.module.add_function(
            "ai_array_get",
            array_get_ty,
            Some(Linkage::External),
        );
        self.extern_array_get = Some(array_get);

        // ai_array_set(thread, array, i, ptr) -> i64
        let array_set_ty = self.i64_ty.fn_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.i64_ty.into(),
                self.ptr_ty.into(),
            ],
            false,
        );
        let array_set = self.module.add_function(
            "ai_array_set",
            array_set_ty,
            Some(Linkage::External),
        );
        self.extern_array_set = Some(array_set);

        // ai_atom_new(thread, init_ptr) -> *u8
        // Allocate a fresh dedicated `Atom` cell holding `init`.
        let atom_new_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let atom_new =
            self.module
                .add_function("ai_atom_new", atom_new_ty, Some(Linkage::External));
        self.extern_atom_new = Some(atom_new);

        // ai_atom_load(atom) -> *u8 — acquiring read of the cell's value.
        let atom_load_ty = self.ptr_ty.fn_type(&[self.ptr_ty.into()], false);
        let atom_load =
            self.module
                .add_function("ai_atom_load", atom_load_ty, Some(Linkage::External));
        self.extern_atom_load = Some(atom_load);

        // ai_atom_swap(thread, atom, updater_closure) -> *u8
        // The lock-free atom primitive: load slot, run closure, CAS,
        // retry. Returns the installed value pointer.
        let atom_swap_ty = self.ptr_ty.fn_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.ptr_ty.into(),
            ],
            false,
        );
        let atom_swap =
            self.module
                .add_function("ai_atom_swap_local", atom_swap_ty, Some(Linkage::External));
        self.extern_atom_swap_local = Some(atom_swap);

        // ai_thread_spawn(thread, closure) -> *u8 (ThreadHandle, a BoxedInt id)
        let thread_spawn_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let thread_spawn = self.module.add_function(
            "ai_thread_spawn",
            thread_spawn_ty,
            Some(Linkage::External),
        );
        self.extern_thread_spawn = Some(thread_spawn);

        // ai_thread_spawn_shared(thread, closure) -> *u8 (zero-copy opt-out)
        let thread_spawn_shared_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let thread_spawn_shared = self.module.add_function(
            "ai_thread_spawn_shared",
            thread_spawn_shared_ty,
            Some(Linkage::External),
        );
        self.extern_thread_spawn_shared = Some(thread_spawn_shared);

        // ai_thread_join(thread, handle) -> *u8 (the spawned thunk's result)
        let thread_join_ty = self
            .ptr_ty
            .fn_type(&[self.ptr_ty.into(), self.ptr_ty.into()], false);
        let thread_join = self.module.add_function(
            "ai_thread_join",
            thread_join_ty,
            Some(Linkage::External),
        );
        self.extern_thread_join = Some(thread_join);
    }

    // -------------------------------------------------------------------------
    // User-defined `extern fn` declarations
    // -------------------------------------------------------------------------

    /// Declare an LLVM extern (`ext/<name>`) for each user-defined
    /// `extern fn` in the module. The JIT-init phase wires each one
    /// to its registered Rust function pointer.
    ///
    /// ABI: first param is always the thread pointer. For user params:
    /// `Int` → i64; `String` → ptr (the heap-resident String pointer).
    /// Same mapping for the return type.
    fn declare_user_externs(
        &mut self,
        externs: &HashMap<String, crate::resolve::ExternSig>,
        referenced: &std::collections::HashSet<String>,
    ) -> Result<(), CodegenError> {
        // Keep the signatures around so the call-site codegen can tell a
        // host extern from a C-FFI one.
        self.user_externs = externs.clone();
        for (name, sig) in externs {
            let symbol = user_extern_symbol(name);
            if sig.library.is_some() {
                // C-FFI extern: a real C symbol. Plain C ABI — no leading
                // `Thread*`. Every arg/return is an i64-width C scalar
                // (`Int` or `Ptr`); other types aren't marshalable here.
                for p in &sig.params {
                    if !is_c_scalar_type(p) {
                        return Err(CodegenError::Unsupported {
                            what: format!(
                                "C extern `{}` parameter type {:?} (supported: Int, Ptr)",
                                name, p
                            ),
                        });
                    }
                }
                if !is_c_scalar_type(&sig.ret) {
                    return Err(CodegenError::Unsupported {
                        what: format!(
                            "C extern `{}` return type {:?} (supported: Int, Ptr)",
                            name, sig.ret
                        ),
                    });
                }
                let param_tys: Vec<BasicMetadataTypeEnum> =
                    sig.params.iter().map(|_| self.i64_ty.into()).collect();
                // A variadic C function (e.g. `curl_easy_setopt`) is
                // declared with `is_var_args = true` and only the fixed
                // params named; the arm64 / x86-64 variadic ABI then
                // places trailing args correctly (stack vs registers).
                let fn_ty = self.i64_ty.fn_type(&param_tys, sig.variadic);
                self.module
                    .add_function(&symbol, fn_ty, Some(Linkage::External));
                // Only resolve the real library symbol if the module
                // actually calls it. This keeps unused prelude
                // declarations (e.g. the whole libcurl surface) from
                // forcing every program to dlopen that library.
                if !referenced.contains(name.as_str()) {
                    continue;
                }
                // Resolve the real C symbol now (via dlopen/dlsym) and
                // register its address in the global FFI registry under
                // this name, so the existing `wire_user_externs` step
                // maps `ext/<name>` to it — same as a host extern, just a
                // different source of the address.
                let lib = sig.library.as_deref().unwrap_or("");
                match crate::cffi::resolve_symbol(lib, name) {
                    Some(addr) => unsafe {
                        crate::ffi::register_extern(
                            name,
                            sig.params.clone(),
                            sig.ret.clone(),
                            addr,
                        );
                    },
                    None => {
                        return Err(CodegenError::Unsupported {
                            what: format!(
                                "could not resolve C symbol `{}` from library \"{}\" \
                                 (dlopen/dlsym failed)",
                                name, lib
                            ),
                        });
                    }
                }
                continue;
            }
            // Host (Rust) extern: leading `Thread*`, Int/String args.
            for p in &sig.params {
                if !is_extern_supported_type(p) {
                    return Err(CodegenError::Unsupported {
                        what: format!(
                            "extern `{}` parameter type {:?} (supported: Int, String)",
                            name, p
                        ),
                    });
                }
            }
            if !is_extern_supported_type(&sig.ret) {
                return Err(CodegenError::Unsupported {
                    what: format!(
                        "extern `{}` return type {:?} (supported: Int, String)",
                        name, sig.ret
                    ),
                });
            }
            let mut param_tys: Vec<BasicMetadataTypeEnum> =
                Vec::with_capacity(sig.params.len() + 1);
            param_tys.push(self.ptr_ty.into());
            for p in &sig.params {
                param_tys.push(self.extern_llvm_param_ty(p));
            }
            let fn_ty = if is_int_type(&sig.ret) {
                self.i64_ty.fn_type(&param_tys, false)
            } else {
                // String/Ptr return → ptr.
                self.ptr_ty.fn_type(&param_tys, false)
            };
            self.module
                .add_function(&symbol, fn_ty, Some(Linkage::External));
        }
        Ok(())
    }

    fn extern_llvm_param_ty(&self, t: &Type) -> BasicMetadataTypeEnum<'ctx> {
        if is_int_type(t) {
            self.i64_ty.into()
        } else {
            // String or any other pointer-typed extern arg.
            self.ptr_ty.into()
        }
    }

    // -------------------------------------------------------------------------
    // Lifted lambda declaration
    // -------------------------------------------------------------------------

    /// Walk a lambda's body to infer, for each capture, whether the
    /// capture slot needs to hold a heap pointer (`true`) or a raw
    /// i64 (`false`). The classification is based on how the body
    /// uses `LocalVar(arity + outer_idx)`:
    ///
    /// - `base` of `Expr::Field` → pointer
    /// - `scrutinee` of `Expr::Match` → pointer (enums are heap)
    /// - Arg `j` of a `Call` whose callee has a known signature with
    ///   pointer-typed param `j` → pointer
    /// - Field `j` of a `StructNew` whose declared field type is
    ///   pointer → pointer
    /// - `EnumNew` payload whose variant's declared payload is
    ///   pointer → pointer
    ///
    /// Any other context (arithmetic, raw stores, etc.) defaults
    /// to Int. If a single capture is used both ways the pointer
    /// classification wins (we can always box an Int into a
    /// BoxedInt on the store side).
    fn infer_capture_pointer_flags(&self, h: &Hash) -> Vec<bool> {
        let spec = match self.lambdas.get(h) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let arity = spec.params.len() as u32;
        let mut flags = vec![false; spec.captures.len()];
        self.walk_capture_uses(&spec.body, arity, &spec.captures, &mut flags);

        // Pass-through detection: if a capture flows directly to the
        // lambda's return value without any int-implying use, the
        // capture must be pointer-shaped because the lambda's return
        // travels through the uniform-ABI pointer slot. Without this,
        // `|| x` in a generic context (e.g. `value_pure<T>`) would
        // be classified Int and break at construction when T is a
        // heap value.
        if let Some(idx) = passthrough_capture_index(&spec.body, arity) {
            if let Some(pos) = spec.captures.iter().position(|&o| o == idx) {
                flags[pos] = true;
            }
        }
        flags
    }

    fn walk_capture_uses(
        &self,
        e: &Expr,
        arity: u32,
        captures: &[u32],
        flags: &mut [bool],
    ) {
        let mark = |idx: u32, ar: u32, flags: &mut [bool]| {
            if idx < ar {
                return;
            }
            let outer = idx - ar;
            if let Some(pos) = captures.iter().position(|&o| o == outer) {
                flags[pos] = true;
            }
        };
        let arg_is_capture_localvar = |e: &Expr, ar: u32| -> Option<u32> {
            if let Expr::LocalVar(i) = e {
                if *i >= ar {
                    return Some(*i);
                }
            }
            None
        };
        match e {
            Expr::LocalVar(_) | Expr::IntLit(_) | Expr::FloatLit(_) | Expr::BoolLit(_)
            | Expr::StringLit(_) | Expr::TopRef(_)
            | Expr::BuiltinRef(_) | Expr::SelfRef(_)
            | Expr::StateRef(_) | Expr::StateSelfRef(_) => {}
            Expr::Field { base, .. } => {
                if let Some(idx) = arg_is_capture_localvar(base, arity) {
                    mark(idx, arity, flags);
                }
                self.walk_capture_uses(base, arity, captures, flags);
            }
            Expr::Match { scrutinee, arms } => {
                if let Some(idx) = arg_is_capture_localvar(scrutinee, arity) {
                    mark(idx, arity, flags);
                }
                self.walk_capture_uses(scrutinee, arity, captures, flags);
                for arm in arms {
                    let bindings = count_pattern_vars(&arm.pattern);
                    self.walk_capture_uses(&arm.body, arity + bindings, captures, flags);
                }
            }
            Expr::Call(callee, args) => {
                // A callee that's a captured LocalVar must be a
                // closure (only callable values are functions, and
                // functions cross the closure boundary as heap
                // pointers). Mark it pointer.
                if let Some(idx) = arg_is_capture_localvar(callee, arity) {
                    mark(idx, arity, flags);
                }
                // If the callee is a TopRef / BuiltinRef whose signature
                // we know, use it to classify each arg's expected type.
                let param_tys: Option<Vec<Type>> = match callee.as_ref() {
                    Expr::TopRef(h) => self
                        .def_signatures
                        .get(h)
                        .map(|s| s.params.clone()),
                    Expr::BuiltinRef(name) => {
                        if crate::resolve::parse_at_builtin_name(name).is_some() {
                            // at(Node, fn() -> T): both args are
                            // pointer-typed at the call boundary.
                            Some(vec![
                                Type::TypeRef(Hash([0; 32])),
                                Type::FnType {
                                    params: vec![],
                                    ret: Box::new(Type::Builtin("Int".to_owned())),
                                },
                            ])
                        } else {
                            // Any other known builtin: use its authoritative
                            // signature so a captured pointer passed to it
                            // (e.g. a `Bytes` to `bytes_set`, an `Array` to
                            // `array_get`) is classified pointer-shaped.
                            crate::typecheck::builtin_signature(name).map(|(p, _)| p)
                        }
                    }
                    _ => None,
                };
                for (j, arg) in args.iter().enumerate() {
                    if let Some(idx) = arg_is_capture_localvar(arg, arity) {
                        if let Some(tys) = &param_tys {
                            if let Some(p) = tys.get(j) {
                                if is_pointer_type(p) {
                                    mark(idx, arity, flags);
                                }
                            }
                        }
                    }
                    self.walk_capture_uses(arg, arity, captures, flags);
                }
                self.walk_capture_uses(callee, arity, captures, flags);
            }
            Expr::StructNew { struct_ref, fields } => {
                let field_tys = self.struct_field_types.get(struct_ref).cloned();
                for (j, fexpr) in fields.iter().enumerate() {
                    if let Some(idx) = arg_is_capture_localvar(fexpr, arity) {
                        if let Some(tys) = &field_tys {
                            if let Some(t) = tys.get(j) {
                                if is_pointer_type(t) {
                                    mark(idx, arity, flags);
                                }
                            }
                        }
                    }
                    self.walk_capture_uses(fexpr, arity, captures, flags);
                }
            }
            Expr::EnumNew {
                enum_ref,
                variant_index,
                payload,
                ..
            } => {
                if let Some(pexpr) = payload.as_deref() {
                    let payload_ty = self
                        .enum_variant_types
                        .get(enum_ref)
                        .and_then(|vs| vs.get(*variant_index as usize).cloned())
                        .flatten();
                    if let Some(idx) = arg_is_capture_localvar(pexpr, arity) {
                        if let Some(t) = payload_ty.as_ref() {
                            if is_pointer_type(t) {
                                mark(idx, arity, flags);
                            }
                        }
                    }
                    self.walk_capture_uses(pexpr, arity, captures, flags);
                }
            }
            Expr::Let { value, body } => {
                self.walk_capture_uses(value, arity, captures, flags);
                self.walk_capture_uses(body, arity + 1, captures, flags);
            }
            Expr::If { cond, then_branch, else_branch } => {
                self.walk_capture_uses(cond, arity, captures, flags);
                self.walk_capture_uses(then_branch, arity, captures, flags);
                self.walk_capture_uses(else_branch, arity, captures, flags);
            }
            Expr::Try { expr, .. } => {
                // The operand is a Result (a heap pointer); a captured
                // LocalVar used directly as the operand must be pointer.
                if let Some(idx) = arg_is_capture_localvar(expr, arity) {
                    mark(idx, arity, flags);
                }
                self.walk_capture_uses(expr, arity, captures, flags);
            }
            Expr::Defer { cleanup, body } => {
                // `defer` adds no binder; both sub-expressions see the
                // same environment as the Defer node.
                self.walk_capture_uses(cleanup, arity, captures, flags);
                self.walk_capture_uses(body, arity, captures, flags);
            }
            Expr::Lambda { .. } => {
                // No nested lambdas per `check_no_nested_lambdas`.
            }
        }
    }

    fn declare_lifted_lambda(&mut self, h: &Hash) -> Result<(), CodegenError> {
        let spec = self.lambdas[h].clone();

        // Always-boxed ABI: closures take and return pointers (uniform
        // representation across generic boundaries). Bodies unbox Int
        // params at entry; Int returns get boxed at exit. Indirect call
        // sites box/unbox symmetrically based on the closure's
        // declared FnType.
        //
        // Signature: (ptr thread, ptr closure, ptr args...) -> ptr
        let mut param_tys: Vec<BasicMetadataTypeEnum> = Vec::with_capacity(spec.params.len() + 2);
        param_tys.push(self.ptr_ty.into());
        param_tys.push(self.ptr_ty.into());
        for _ in &spec.params {
            param_tys.push(self.ptr_ty.into());
        }
        let fn_ty = self.ptr_ty.fn_type(&param_tys, false);
        let symbol = lambda_symbol(h);
        let fv = self.module.add_function(&symbol, fn_ty, None);

        let frame_ty = self.placeholder_frame_ty();

        let name_global = self.emit_name_string(&symbol, h, "lambda");
        let origin_init = self.frame_origin_ty.const_named_struct(&[
            self.context.i32_type().const_zero().into(),
            self.context.i32_type().const_zero().into(),
            name_global.as_pointer_value().into(),
        ]);
        let origin_sym = frame_origin_symbol("lambda", h);
        let origin_global =
            self.module
                .add_global(self.frame_origin_ty, Some(AddressSpace::default()), &origin_sym);
        origin_global.set_linkage(Linkage::Private);
        origin_global.set_constant(true);
        origin_global.set_initializer(&origin_init);

        // Register the closure shape's TypeInfo + per-capture layout.
        //
        // Capture slot assignment (source order):
        //   - pointer captures fill `value_field` slots starting at
        //     `header_size`, packed in source order among pointers
        //   - int captures fill raw-byte slots starting at
        //     `header_size + ptr_count*8 + NON_POINTER_CAPTURES`,
        //     packed in source order among ints
        //
        // `closure_offsets::NON_POINTER_CAPTURES = 40` is the size of
        // the header prefix (code_hash 32 + n_captures 4 + pad 4)
        // that sits between value_field slots and int slots.
        let type_id = self.next_type_id();
        let header_size = crate::gc::Full::SIZE as u32;
        let ptr_count: u16 = spec
            .capture_is_pointer
            .iter()
            .filter(|p| **p)
            .count() as u16;
        let int_count: u16 = (spec.capture_is_pointer.len() as u16) - ptr_count;
        let raw_bytes: u16 = closure_offsets::NON_POINTER_CAPTURES as u16 + 8 * int_count;
        let ti = TypeInfo::for_header(crate::gc::Full::SIZE as usize)
            .with_type_id(type_id)
            .with_fields(ptr_count)
            .with_raw_bytes(raw_bytes);
        self.closure_type_infos.push(ti);
        self.shape_registry.insert(*h, type_id);
        let int_base =
            header_size + (ptr_count as u32) * 8 + closure_offsets::NON_POINTER_CAPTURES as u32;
        let mut captures_meta: Vec<CaptureMeta> = Vec::with_capacity(spec.capture_is_pointer.len());
        let mut next_ptr: u32 = 0;
        let mut next_int: u32 = 0;
        for &is_ptr in &spec.capture_is_pointer {
            if is_ptr {
                let offset = header_size + next_ptr * 8;
                captures_meta.push(CaptureMeta { offset, is_pointer: true });
                next_ptr += 1;
            } else {
                let offset = int_base + next_int * 8;
                captures_meta.push(CaptureMeta { offset, is_pointer: false });
                next_int += 1;
            }
        }
        self.shape_meta.insert(
            *h,
            ShapeMeta::Closure {
                code_hash: *h,
                captures: captures_meta,
            },
        );
        self.register_type_id_shape(type_id, *h);

        // Emit a private constant LLVM global mirroring this TypeInfo so
        // we can pass its address to `ai_gc_alloc_closure` from JIT'd
        // code.
        let ti_global = self.emit_type_info_global(h, &ti);

        self.lifted_lambdas.insert(*h, fv);
        self.frame_origins.insert(origin_sym, origin_global);
        self.used_globals.push(origin_global.as_pointer_value());
        self.closure_type_info_globals.insert(*h, ti_global);
        self.lambda_info.insert(
            *h,
            DefInfo {
                frame_ty,
                has_frame: true,
            },
        );
        Ok(())
    }

    /// Mirror a Rust `TypeInfo` value as an LLVM constant global so we
    /// can pass its address to runtime calls. Layout *must* match the
    /// `TypeInfo` struct in `gc/type_info.rs`.
    fn emit_type_info_global(&mut self, lambda_hash: &Hash, ti: &TypeInfo) -> GlobalValue<'ctx> {
        let i16_ty = self.context.i16_type();
        let i8_ty = self.context.i8_type();
        // TypeInfo layout (from gc/type_info.rs):
        //   u16 type_id, u16 header_size, u16 value_field_count,
        //   u16 raw_byte_count, u8 varlen (None=0, Values=1, Bytes=2),
        //   u8 align_log2.
        let ti_struct_ty = self.context.struct_type(
            &[
                i16_ty.into(),
                i16_ty.into(),
                i16_ty.into(),
                i16_ty.into(),
                i8_ty.into(),
                i8_ty.into(),
            ],
            false,
        );
        let varlen_byte = match ti.varlen {
            VarLenKind::None => 0u64,
            VarLenKind::Values => 1u64,
            VarLenKind::Bytes => 2u64,
        };
        let init = ti_struct_ty.const_named_struct(&[
            i16_ty.const_int(ti.type_id as u64, false).into(),
            i16_ty.const_int(ti.header_size as u64, false).into(),
            i16_ty.const_int(ti.value_field_count as u64, false).into(),
            i16_ty.const_int(ti.raw_byte_count as u64, false).into(),
            i8_ty.const_int(varlen_byte, false).into(),
            i8_ty.const_int(ti.align_log2 as u64, false).into(),
        ]);
        let sym = closure_ti_symbol(lambda_hash);
        let g = self
            .module
            .add_global(ti_struct_ty, Some(AddressSpace::default()), &sym);
        g.set_linkage(Linkage::Private);
        g.set_constant(true);
        g.set_initializer(&init);
        self.used_globals.push(g.as_pointer_value());
        g
    }

    fn emit_name_string(
        &mut self,
        full_symbol: &str,
        hash: &Hash,
        kind: &str,
    ) -> GlobalValue<'ctx> {
        let mut bytes = full_symbol.as_bytes().to_vec();
        bytes.push(0);
        let str_const = self.context.const_string(&bytes, false);
        let str_ty = str_const.get_type();
        let sym = format!("__name_{}_{}", kind, hash.to_hex());
        let g = self
            .module
            .add_global(str_ty, Some(AddressSpace::default()), &sym);
        g.set_linkage(Linkage::Private);
        g.set_constant(true);
        g.set_initializer(&str_const);
        self.used_globals.push(g.as_pointer_value());
        g
    }

    // -------------------------------------------------------------------------
    // Struct declaration
    // -------------------------------------------------------------------------

    /// Register a struct shape: build its TypeInfo (pointer fields first,
    /// then value fields), compute physical field offsets, emit the
    /// TypeInfo LLVM global, and store in `self.structs`.
    fn declare_struct(
        &mut self,
        hash: &Hash,
        name: &str,
        fields: &[(String, Type)],
    ) -> Result<(), CodegenError> {
        // Validate field types early. Only Int, FnType, TypeRef supported
        // in v1. Strings/Bytes/etc. error cleanly.
        for (fname, fty) in fields {
            require_supported_type(fty, name, &format!("field `{}`", fname))?;
        }

        // Classify each field. Build the declaration → physical offset
        // mapping. Pointer fields come first (offsets header + 0, +8, …)
        // then value fields (after pointers, each 8 bytes).
        let n = fields.len();
        let mut field_is_pointer = Vec::with_capacity(n);
        let mut field_offsets = vec![0u32; n];
        let header_size = crate::gc::Full::SIZE as u32;
        let mut ptr_count: u16 = 0;
        let mut val_count: u16 = 0;

        // First pass: classify.
        for (_, fty) in fields {
            field_is_pointer.push(is_pointer_type(fty));
        }

        // Second pass: assign offsets — pointer fields first.
        for (i, &is_ptr) in field_is_pointer.iter().enumerate() {
            if is_ptr {
                field_offsets[i] = header_size + (ptr_count as u32) * 8;
                ptr_count += 1;
            }
        }
        let value_base = header_size + (ptr_count as u32) * 8;
        for (i, &is_ptr) in field_is_pointer.iter().enumerate() {
            if !is_ptr {
                field_offsets[i] = value_base + (val_count as u32) * 8;
                val_count += 1;
            }
        }

        // TypeInfo: value_field_count = ptr_count (these get scanned by
        // the GC), raw_byte_count = 8 * val_count.
        let type_id = self.next_type_id();
        let ti = TypeInfo::for_header(crate::gc::Full::SIZE as usize)
            .with_type_id(type_id)
            .with_fields(ptr_count)
            .with_raw_bytes(val_count * 8);
        self.closure_type_infos.push(ti);
        self.shape_registry.insert(*hash, type_id);
        let field_metas: Vec<FieldMeta> = field_offsets
            .iter()
            .zip(field_is_pointer.iter())
            .map(|(&offset, &is_pointer)| FieldMeta { offset, is_pointer })
            .collect();
        self.shape_meta.insert(
            *hash,
            ShapeMeta::Struct {
                struct_ref: *hash,
                fields: field_metas,
            },
        );
        self.register_type_id_shape(type_id, *hash);
        let ti_global = self.emit_type_info_global(hash, &ti);

        self.structs.insert(
            *hash,
            StructInfo {
                ti_global,
                field_offsets,
                field_is_pointer,
            },
        );
        Ok(())
    }

    /// Maintain the type_id → hash reverse map by index. Grows the
    /// vec with `None` placeholders as needed.
    fn register_type_id_shape(&mut self, type_id: u16, hash: Hash) {
        while self.shape_by_type_id.len() <= type_id as usize {
            self.shape_by_type_id.push(None);
        }
        self.shape_by_type_id[type_id as usize] = Some(hash);
    }

    // -------------------------------------------------------------------------
    // Enum declaration
    // -------------------------------------------------------------------------

    /// Register an enum shape: one [`TypeInfo`] per (enum, variant)
    /// pair, plus per-variant codegen info (payload kind & physical
    /// offsets within the heap object).
    ///
    /// Heap layout for an enum value of variant V:
    ///
    /// ```text
    /// offset 0          : ObjHeader      (16 bytes; type_id = per-variant ID)
    /// offset 16         : tag (u32)      (= variant index within the enum)
    /// offset 20         : _pad (u32)
    /// offset 24         : payload[0]     ← pointer-typed payload (in value_fields)
    ///                                       OR Int payload (in raw bytes)
    /// ```
    ///
    /// For v1 each variant has 0 or 1 payload, so the offset table is
    /// trivial: pointer payload at 24, Int payload at 24 too (just a
    /// different category in TypeInfo). The GC uses the type_id stored
    /// in the header to find the right `TypeInfo` and scan the right
    /// fields.
    fn declare_enum(
        &mut self,
        hash: &Hash,
        name: &str,
        variants: &[(String, Option<Type>)],
    ) -> Result<(), CodegenError> {
        for (vname, payload_ty) in variants {
            if let Some(t) = payload_ty {
                require_supported_type(t, name, &format!("variant `{}` payload", vname))?;
            }
        }

        // For all variants of an enum to share the same tag offset, we
        // pad every variant to `max_value_fields` pointer slots. That
        // way the tag (in raw_bytes) sits at the same byte offset for
        // every variant. Without this, a match couldn't read the tag
        // before knowing the variant.
        //
        // v1 enums have 0 or 1 payload per variant, and a payload is
        // either Int (raw) or pointer (value_field). So max_value_fields
        // is 1 if ANY variant has a pointer payload, else 0.
        let mut max_value_fields: u16 = 0;
        for (_, payload_ty) in variants {
            if let Some(t) = payload_ty {
                if is_pointer_type(t) {
                    max_value_fields = max_value_fields.max(1);
                }
            }
        }

        let header_size = crate::gc::Full::SIZE as u32;
        // Tag offset is fixed per-enum: right after the (padded)
        // value_fields. Tag is a u32 followed by 4 bytes of pad to
        // 8-byte alignment.
        let tag_offset = header_size + (max_value_fields as u32) * 8;

        let mut info = EnumInfo {
            variants: Vec::with_capacity(variants.len()),
        };
        for (vname, payload_ty) in variants {
            // Per-variant raw_byte_count:
            //   8 bytes for tag + pad, plus 8 more if this variant has
            //   an Int payload (which lives in raw_bytes, after the
            //   tag/pad). Pointer payloads go in value_field[0] and
            //   need no raw bytes.
            let (payload_offset, payload_is_pointer, raw_bytes): (Option<u32>, bool, u16) =
                match payload_ty {
                    None => (None, false, 8),
                    Some(t) if is_pointer_type(t) => {
                        // Pointer payload sits in value_field[0] —
                        // offset = header_size.
                        (Some(header_size), true, 8)
                    }
                    Some(_) => {
                        // Int payload sits in raw_bytes AFTER tag+pad.
                        // offset = tag_offset + 8.
                        (Some(tag_offset + 8), false, 16)
                    }
                };

            let type_id = self.next_type_id();
            let ti = TypeInfo::for_header(crate::gc::Full::SIZE as usize)
                .with_type_id(type_id)
                .with_fields(max_value_fields)
                .with_raw_bytes(raw_bytes);
            self.closure_type_infos.push(ti);

            let variant_hash = derive_variant_hash(hash, vname);
            self.shape_registry.insert(variant_hash, type_id);
            self.shape_meta.insert(
                variant_hash,
                ShapeMeta::EnumVariant {
                    enum_ref: *hash,
                    variant_index: info.variants.len() as u32,
                    tag_offset,
                    payload: payload_offset.map(|off| FieldMeta {
                        offset: off,
                        is_pointer: payload_is_pointer,
                    }),
                },
            );
            self.register_type_id_shape(type_id, variant_hash);
            let ti_global = self.emit_type_info_global(&variant_hash, &ti);

            info.variants.push(VariantInfo {
                ti_global,
                tag_offset,
                payload_offset,
                payload_is_pointer,
            });
        }
        self.enums.insert(*hash, info);
        let _ = name;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Def declaration / compilation
    // -------------------------------------------------------------------------

    fn declare_def(&mut self, rd: &ResolvedDef) -> Result<(), CodegenError> {
        match &rd.def {
            Def::Struct {
                type_params: _,
                fields,
            } => {
                // Generic structs use the same layout regardless of
                // instantiation (uniform pointer representation), so
                // one TypeInfo per def suffices.
                self.struct_field_types.insert(
                    rd.hash,
                    fields.iter().map(|(_, t)| t.clone()).collect(),
                );
                return self.declare_struct(&rd.hash, &rd.name, fields);
            }
            Def::Enum {
                type_params: _,
                variants,
            } => {
                self.enum_variant_types.insert(
                    rd.hash,
                    variants.iter().map(|(_, p)| p.clone()).collect(),
                );
                return self.declare_enum(&rd.hash, &rd.name, variants);
            }
            Def::State { ty, init } => {
                // Record the declared type so `infer_type(StateRef)` can
                // drive box/unbox at `deref`/`swap` sites.
                self.state_types.insert(rd.hash, ty.clone());
                // Declare the installer: `state_init_<hash>(thread) -> i64`.
                // Its body (emitted in `compile_def`) idempotently runs the
                // initializer once and stores the result in the node table.
                // It carries a GC frame because the initializer allocates.
                let symbol = state_init_symbol(&rd.hash);
                let fn_ty = self.i64_ty.fn_type(&[self.ptr_ty.into()], false);
                let fv = self.module.add_function(&symbol, fn_ty, None);

                let frame_ty = self.placeholder_frame_ty();
                let name_global = self.emit_name_string(&symbol, &rd.hash, "state");
                let origin_init = self.frame_origin_ty.const_named_struct(&[
                    self.context.i32_type().const_zero().into(),
                    self.context.i32_type().const_zero().into(),
                    name_global.as_pointer_value().into(),
                ]);
                let origin_sym = frame_origin_symbol("state", &rd.hash);
                let origin_global = self.module.add_global(
                    self.frame_origin_ty,
                    Some(AddressSpace::default()),
                    &origin_sym,
                );
                origin_global.set_linkage(Linkage::Private);
                origin_global.set_constant(true);
                origin_global.set_initializer(&origin_init);

                self.state_installers.insert(rd.hash, fv);
                self.frame_origins.insert(origin_sym, origin_global);
                self.used_globals.push(origin_global.as_pointer_value());
                self.def_info.insert(
                    rd.hash,
                    DefInfo {
                        frame_ty,
                        has_frame: true,
                    },
                );
                return Ok(());
            }
            Def::Fn { .. } => {}
        }
        let Def::Fn {
            params,
            ret,
            body,
            type_params: _,
            is_local: _,
        } = &rd.def
        else {
            unreachable!("matched above");
        };
        require_supported_type(ret, &rd.name, "return")?;
        for p in params {
            require_supported_type(p, &rd.name, "parameter")?;
        }
        // Record the declared signature for later type queries.
        self.def_signatures.insert(
            rd.hash,
            FnSigSimple {
                params: params.clone(),
                ret: ret.clone(),
            },
        );

        // All params and ret are pointer-sized in our ABI:
        // Int → i64, FnType / TypeRef → ptr.
        let mut param_tys: Vec<BasicMetadataTypeEnum> = Vec::with_capacity(params.len() + 1);
        param_tys.push(self.ptr_ty.into());
        for p in params {
            param_tys.push(if is_pointer_type(p) {
                self.ptr_ty.into()
            } else {
                self.i64_ty.into()
            });
        }
        let fn_ty = if is_pointer_type(ret) {
            self.ptr_ty.fn_type(&param_tys, false)
        } else {
            self.i64_ty.fn_type(&param_tys, false)
        };
        let symbol = def_symbol(&rd.hash);
        let fv = self.module.add_function(&symbol, fn_ty, None);

        let frame_ty = self.placeholder_frame_ty();
        let name_global = self.emit_name_string(&symbol, &rd.hash, "def");
        let origin_init = self.frame_origin_ty.const_named_struct(&[
            self.context.i32_type().const_zero().into(),
            self.context.i32_type().const_zero().into(),
            name_global.as_pointer_value().into(),
        ]);
        let origin_sym = frame_origin_symbol("def", &rd.hash);
        let origin_global =
            self.module
                .add_global(self.frame_origin_ty, Some(AddressSpace::default()), &origin_sym);
        origin_global.set_linkage(Linkage::Private);
        origin_global.set_constant(true);
        origin_global.set_initializer(&origin_init);

        self.functions.insert(rd.hash, fv);
        self.frame_origins.insert(origin_sym, origin_global);
        self.used_globals.push(origin_global.as_pointer_value());
        self.def_info.insert(
            rd.hash,
            DefInfo {
                frame_ty,
                has_frame: true,
            },
        );
        Ok(())
    }

    fn compile_def(&mut self, rd: &ResolvedDef) -> Result<(), CodegenError> {
        // Node `state` binding: emit its installer body.
        if let Def::State { init, .. } = &rd.def {
            return self.compile_state_installer(rd, init);
        }
        // Struct defs have no body to compile — only TypeInfo registration
        // (already done in declare_def). Skip.
        let Def::Fn { body, params, ret, .. } = &rd.def else {
            return Ok(());
        };
        // Defer stack is per-body; start clean (pushes/pops balance within
        // a body, but never let one body's state leak into the next).
        self.deferred.clear();
        let fv = self.functions[&rd.hash];
        let mut info = self.def_info[&rd.hash];
        // Frame decision (precise, no pre-scan anywhere): a provably
        // scalar-only body (no pointers — e.g. fib, a mandelbrot kernel)
        // pushes NO frame. Everything else gets the placeholder frame,
        // sized to its exact slot high-water mark by
        // `finalize_frame_zeroing` once the body is compiled.
        info.has_frame = body_mentions_defer(body)
            || !params.iter().all(is_scalar_type)
            || !self.scalar_only_body(body);
        let origin_sym = frame_origin_symbol("def", &rd.hash);

        let entry = self.context.append_basic_block(fv, "entry");
        self.builder.position_at_end(entry);

        let thread_param = fv.get_nth_param(0).unwrap().into_pointer_value();

        // Prologue: alloca + link frame.
        let frame_alloca = if info.has_frame {
            self.emit_prologue(thread_param, info, &origin_sym)?
        } else {
            // Pointer-free, defer-free body: no GC frame at all. The
            // null is never read — nothing roots, the epilogue is
            // skipped, and the safepoint poll uses the origin global.
            self.ptr_ty.const_null()
        };

        // Per-param storage for tail-call updates. Int params get
        // their own i64 alloca in `entry`. Pointer params reuse the
        // root-slot alloca'd in the frame header.
        let mut next_root_slot = 0u32;
        let mut param_slots: Vec<TailParamSlot<'ctx>> = Vec::with_capacity(params.len());
        for (i, ty) in params.iter().enumerate() {
            let pv = fv
                .get_nth_param(1 + i as u32)
                .expect("declared with this many params");
            if is_pointer_type(ty) {
                let ptr = pv.into_pointer_value();
                let slot =
                    self.write_root_slot(frame_alloca, info, next_root_slot, ptr)?;
                next_root_slot += 1;
                param_slots.push(TailParamSlot::Ptr(slot));
            } else {
                let slot = self
                    .builder
                    .build_alloca(self.i64_ty, &format!("param{}_slot", i))
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_alloca param slot: {}", e),
                    ))?;
                self.builder
                    .build_store(slot, pv.into_int_value())
                    .map_err(|e| CodegenError::JitInit(
                        format!("store initial int param: {}", e),
                    ))?;
                param_slots.push(TailParamSlot::Int(slot));
            }
        }

        // Create `loop_body` and branch entry → loop_body. Body code
        // is emitted at `loop_body`, so a tail-call jump back here
        // re-enters with the current slot values.
        let loop_body_bb = self.context.append_basic_block(fv, "loop_body");
        self.builder
            .build_unconditional_branch(loop_body_bb)
            .map_err(|e| CodegenError::JitInit(
                format!("br entry→loop_body: {}", e),
            ))?;
        self.builder.position_at_end(loop_body_bb);

        // GC safepoint poll. `loop_body` is both the function-body entry
        // (reached once per call) and the self-tail-call backedge target
        // (reached once per loop iteration), so a single poll here covers
        // both: every call boundary hits the callee's entry poll, and
        // every self-tail loop hits this on each turn. Emitted before the
        // param reloads so the branch-back re-runs the poll then reloads.
        let origin_ptr = self.frame_origins[&origin_sym].as_pointer_value();
        self.emit_safepoint_poll(thread_param, origin_ptr)?;

        // Build env at `loop_body`. Int params are loaded fresh
        // each iteration (the load instruction re-executes on each
        // tail-call branch back). Pointer params already have
        // root-slot reads via `EnvSlot::Closure`.
        let mut env = Env::new();
        for (i, ty) in params.iter().enumerate() {
            match param_slots[i] {
                TailParamSlot::Int(slot) => {
                    let loaded = self
                        .builder
                        .build_load(self.i64_ty, slot, &format!("p{}_load", i))
                        .map_err(|e| CodegenError::JitInit(
                            format!("load int param slot: {}", e),
                        ))?
                        .into_int_value();
                    env.push(EnvSlot::Int(loaded), ty.clone());
                }
                TailParamSlot::Ptr(slot) => {
                    env.push(EnvSlot::Closure(slot), ty.clone());
                }
            }
        }

        // Install the tail-call context for this fn's body compilation.
        self.tail_ctx = Some(TailCtx {
            self_hash: rd.hash,
            loop_body: loop_body_bb,
            param_slots,
        });

        // Body — emitted at loop_body. The body is in tail position
        // for the enclosing fn.
        let result = self.compile_expr(
            body,
            &mut env,
            CompileCtx {
                thread_param,
                frame_alloca,
                info,
                next_root_slot,
                is_tail: true,
            },
        );

        // Clear the tail-call context BEFORE handling the result —
        // the next fn's compile_def will install its own.
        self.tail_ctx = None;
        let result = result?;

        // If the body already terminated its block (e.g. the whole body
        // is a diverging `panic`), there is nothing to return — emitting
        // an epilogue + return here would add instructions after the
        // `unreachable` terminator. The process aborts before any caller
        // resumes, so skipping the shadow-stack pop is correct.
        if self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_terminator())
            .is_some()
        {
            return Ok(());
        }

        // Epilogue + return
        self.emit_epilogue(thread_param, frame_alloca, info)?;
        let basic = if is_pointer_type(ret) {
            match result {
                Value::Closure(p) => Some(p.as_basic_value_enum()),
                Value::Int(_) => {
                    return Err(CodegenError::TypeMismatch {
                        what: format!(
                            "def `{}` declared to return a pointer-typed value but body returned Int",
                            rd.name
                        ),
                    });
                }
            }
        } else {
            match result {
                Value::Int(v) => Some(v.as_basic_value_enum()),
                Value::Closure(_) => {
                    return Err(CodegenError::TypeMismatch {
                        what: format!(
                            "def `{}` declared to return Int but body returned a pointer-typed value",
                            rd.name
                        ),
                    });
                }
            }
        };
        self.builder
            .build_return(basic.as_ref().map(|b| b as &dyn inkwell::values::BasicValue))
            .map_err(|e| CodegenError::JitInit(format!("build_return: {}", e)))?;
        self.finalize_frame_zeroing(&origin_sym)?;
        Ok(())
    }

    /// Emit the body of a node `state` installer:
    /// `if !ai_state_present(hash) { ai_state_set(hash, <init>) }; ret 0`.
    /// Idempotent by hash (the runtime no-ops a re-install), so the
    /// initializer runs at most once per node. The init allocates, so this
    /// fn carries a GC frame (prologue/epilogue) like any def body.
    fn compile_state_installer(
        &mut self,
        rd: &ResolvedDef,
        init: &Expr,
    ) -> Result<(), CodegenError> {
        self.deferred.clear();
        let fv = self.state_installers[&rd.hash];
        let info = self.def_info[&rd.hash];
        let origin_sym = frame_origin_symbol("state", &rd.hash);

        let entry = self.context.append_basic_block(fv, "entry");
        self.builder.position_at_end(entry);
        let thread_param = fv.get_nth_param(0).unwrap().into_pointer_value();
        let frame_alloca = if info.has_frame {
            self.emit_prologue(thread_param, info, &origin_sym)?
        } else {
            // Pointer-free, defer-free body: no GC frame at all. The
            // null is never read — nothing roots, the epilogue is
            // skipped, and the safepoint poll uses the origin global.
            self.ptr_ty.const_null()
        };

        let hash_ptr = self.emit_hash_constant(&rd.hash).as_pointer_value();
        let present_fn = self.extern_state_present.expect("ai_state_present declared");
        let present = self
            .builder
            .build_call(present_fn, &[thread_param.into(), hash_ptr.into()], "state_present")
            .map_err(|e| CodegenError::JitInit(format!("build_call ai_state_present: {}", e)))?
            .as_any_value_enum()
            .into_int_value();
        let is_absent = self
            .builder
            .build_int_compare(
                inkwell::IntPredicate::EQ,
                present,
                self.i64_ty.const_zero(),
                "state_absent",
            )
            .map_err(|e| CodegenError::JitInit(format!("icmp state_absent: {}", e)))?;
        let do_init = self.context.append_basic_block(fv, "do_init");
        let done = self.context.append_basic_block(fv, "done");
        self.builder
            .build_conditional_branch(is_absent, do_init, done)
            .map_err(|e| CodegenError::JitInit(format!("br state install: {}", e)))?;

        // do_init: evaluate the initializer (rooted by the frame) and store.
        self.builder.position_at_end(do_init);
        let mut env = Env::new();
        let v = self.compile_expr(
            init,
            &mut env,
            CompileCtx {
                thread_param,
                frame_alloca,
                info,
                next_root_slot: 0,
                is_tail: false,
            },
        )?;
        // Box an Int initializer so the cell holds a uniform pointer.
        let v_ptr = match v {
            Value::Closure(p) => p,
            Value::Int(iv) => {
                let box_fn = self.extern_box_int.expect("ai_gc_box_int declared");
                self.builder
                    .build_call(box_fn, &[thread_param.into(), iv.into()], "state_init_boxed")
                    .map_err(|e| CodegenError::JitInit(format!(
                        "build_call ai_gc_box_int (state init): {}", e
                    )))?
                    .as_any_value_enum()
                    .into_pointer_value()
            }
        };
        let set_fn = self.extern_state_set.expect("ai_state_set declared");
        self.builder
            .build_call(
                set_fn,
                &[thread_param.into(), hash_ptr.into(), v_ptr.into()],
                "state_set",
            )
            .map_err(|e| CodegenError::JitInit(format!("build_call ai_state_set: {}", e)))?;
        // The init block may have been split (e.g. by a match); branch from
        // wherever the builder now sits.
        if self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_terminator())
            .is_none()
        {
            self.builder
                .build_unconditional_branch(done)
                .map_err(|e| CodegenError::JitInit(format!("br do_init→done: {}", e)))?;
        }

        self.builder.position_at_end(done);
        self.emit_epilogue(thread_param, frame_alloca, info)?;
        self.builder
            .build_return(Some(&self.i64_ty.const_zero()))
            .map_err(|e| CodegenError::JitInit(format!("build_return state installer: {}", e)))?;
        self.finalize_frame_zeroing(&origin_sym)?;
        Ok(())
    }

    fn compile_lifted_lambda(&mut self, h: &Hash) -> Result<(), CodegenError> {
        self.deferred.clear();
        let spec = self.lambdas[h].clone();
        let fv = self.lifted_lambdas[h];
        let mut info = self.lambda_info[h];
        // Frame decision, same precise rule as defs: any pointer param,
        // pointer capture, defer, or non-scalar body keeps the frame.
        info.has_frame = body_mentions_defer(&spec.body)
            || !spec.params.iter().all(is_scalar_type)
            || spec.capture_is_pointer.iter().any(|&p| p)
            || !self.scalar_only_body(&spec.body);
        let origin_sym = frame_origin_symbol("lambda", h);

        let entry = self.context.append_basic_block(fv, "entry");
        self.builder.position_at_end(entry);

        let thread_param = fv.get_nth_param(0).unwrap().into_pointer_value();
        let closure_param = fv.get_nth_param(1).unwrap().into_pointer_value();

        let frame_alloca = if info.has_frame {
            self.emit_prologue(thread_param, info, &origin_sym)?
        } else {
            // Pointer-free, defer-free body: no GC frame at all. The
            // null is never read — nothing roots, the epilogue is
            // skipped, and the safepoint poll uses the origin global.
            self.ptr_ty.const_null()
        };

        // Build env: params first (innermost, de Bruijn 0..arity-1),
        // then captures (de Bruijn arity..).
        //
        // Stored order in env (oldest first): captures, then params.
        // Since lookup_local walks from the tail (innermost = last
        // pushed = highest index in env-vec terms but lowest de Bruijn),
        // pushing captures FIRST means they get higher de Bruijn indices,
        // which is what we want (captures are outside-the-arity).
        let mut env = Env::new();

        // Captures: load each from closure.captures[i] (all Int in v1).
        // Offset from closure base: closure_offsets::NON_POINTER_CAPTURES
        //   = 40, plus the GC header (16 for Full) = 56 from object start.
        // Captures are pushed in ASCENDING outer-idx order; in the
        // lambda's environment, capture[0] has the HIGHEST de Bruijn idx
        // (because higher outer-idx means farther outside).
        //
        // env scheme: env[len-1-i] is read for LocalVar(i). So we push
        // in the order: captures[highest_outer_idx] FIRST (oldest),
        // ..., captures[0] LAST among captures, then params.
        //
        // Actually simpler: push captures highest-outer-idx first, then
        // lowest-outer-idx last among captures, then params. So:
        //   env (oldest → newest):
        //     captures.rev()  (= highest outer idx → 0)
        //     params (param0 → param_{arity-1})
        // Then LocalVar(0) reads params[arity-1] (innermost), and
        // LocalVar(arity + k) reads captures[k] (the k-th outer idx in
        // ascending order).
        //
        // Hmm let's double-check.  Suppose arity = 1, captures = [0, 2]
        // (outer indices). After pushing in our scheme:
        //   env = [cap_2, cap_0, param_0]
        //   length = 3.
        //   LocalVar(0) → env[2] = param_0  ✓
        //   LocalVar(1) → env[1] = cap_0    ← outer idx 0  ✓
        //   LocalVar(2) → env[0] = cap_2    ← outer idx 2  ✓
        //
        // Great. So we push captures in REVERSE order (highest first),
        // then params.

        // Load each capture from its assigned slot. Pointer captures
        // live in `value_field` slots; Int captures live in raw
        // bytes after the closure header. The exact offset comes
        // from the ShapeMeta::Closure registered at lambda-declare
        // time. Pointer captures need a shadow-stack root slot so
        // the GC traces them; Int captures don't.
        //
        // Push order (per the env scheme above): captures in REVERSE
        // (highest outer-idx first), then params last. Within each
        // capture entry we read from the heap and push appropriately.
        let cap_layouts: Vec<CaptureMeta> = match self.shape_meta.get(h) {
            Some(ShapeMeta::Closure { captures, .. }) => captures.clone(),
            _ => Vec::new(),
        };
        // Count how many pointer captures the lambda has; each takes
        // one shadow-stack root slot. We reserve slots [0..ptr_caps)
        // for captures and start param-slot assignment after them.
        let ptr_caps: u32 =
            spec.capture_is_pointer.iter().filter(|p| **p).count() as u32;
        // Build a stable mapping: source-order capture index → root
        // slot index (only used for pointer captures). We assign
        // root slots in source order so the first pointer capture
        // gets slot 0, etc.
        let mut cap_root_slot: Vec<u32> = Vec::with_capacity(spec.captures.len());
        {
            let mut next = 0u32;
            for &is_ptr in &spec.capture_is_pointer {
                if is_ptr {
                    cap_root_slot.push(next);
                    next += 1;
                } else {
                    cap_root_slot.push(u32::MAX);
                }
            }
        }

        for (idx, _outer_idx) in spec.captures.iter().enumerate().rev() {
            let cap = cap_layouts
                .get(idx)
                .copied()
                .unwrap_or(CaptureMeta { offset: 0, is_pointer: false });
            let offset = self.i64_ty.const_int(cap.offset as u64, false);
            let slot_addr = unsafe {
                self.builder
                    .build_in_bounds_gep(self.context.i8_type(), closure_param, &[offset], "cap_slot")
                    .map_err(|e| CodegenError::JitInit(format!("gep capture: {}", e)))?
            };
            if cap.is_pointer {
                // Pointer capture: load as ptr, write into a
                // shadow-stack root slot so it's traced.
                let load = self
                    .builder
                    .build_load(self.ptr_ty, slot_addr, "cap_ptr")
                    .map_err(|e| CodegenError::JitInit(format!("load ptr capture: {}", e)))?
                    .into_pointer_value();
                let root_idx = cap_root_slot[idx];
                let slot = self.write_root_slot(frame_alloca, info, root_idx, load)?;
                // Type tracking: we infer the capture's effective
                // type from how the body uses it. For now we don't
                // have a precise type on hand — TypeVar(0) is a
                // safe placeholder that lets pointer-shaped uses
                // (Field, Match, Apply'd things) succeed; specific
                // sites (compile_field / compile_match) refine via
                // their own inference when needed.
                env.push(EnvSlot::Closure(slot), Type::TypeVar(0));
            } else {
                let load = self
                    .builder
                    .build_load(self.i64_ty, slot_addr, "cap_val")
                    .map_err(|e| CodegenError::JitInit(format!("load capture: {}", e)))?;
                env.push(EnvSlot::Int(load.into_int_value()), Type::Builtin("Int".to_owned()));
            }
        }
        // Params arrive as ptrs (uniform closure ABI). For each
        // declared-Int param, unbox the BoxedInt back to i64.
        // Pointer params land in a root slot so GC tracks them.
        // Captures already consumed `ptr_caps` root slots above, so
        // params start at slot index `ptr_caps`.
        let mut next_root_slot = ptr_caps;
        for (i, p_ty) in spec.params.iter().enumerate() {
            let pv = fv
                .get_nth_param((2 + i) as u32)
                .expect("declared with this many params")
                .into_pointer_value();
            if is_int_type(p_ty) {
                let unbox_fn = self
                    .extern_unbox_int
                    .expect("ai_gc_unbox_int declared");
                let call = self
                    .builder
                    .build_call(
                        unbox_fn,
                        &[pv.into()],
                        &format!("unbox_lambda_param_{}", i),
                    )
                    .map_err(|e| CodegenError::JitInit(format!(
                        "build_call ai_gc_unbox_int (lambda param {}): {}", i, e
                    )))?;
                let iv = call.as_any_value_enum().into_int_value();
                env.push(EnvSlot::Int(iv), p_ty.clone());
            } else {
                let slot = self.write_root_slot(frame_alloca, info, next_root_slot, pv)?;
                next_root_slot += 1;
                env.push(EnvSlot::Closure(slot), p_ty.clone());
            }
        }

        // Rewrite the body's de Bruijn indices to address the lifted
        // env (captures + params), not the original outer scope.
        let rewritten = rewrite_body_for_lifted(
            &spec.body,
            spec.params.len() as u32,
            &spec.captures,
        );

        let _ = fv;
        let result = self.compile_expr(
            &rewritten,
            &mut env,
            CompileCtx {
                thread_param,
                frame_alloca,
                info,
                next_root_slot,
                // Lambda bodies are tail position for the lambda itself.
                is_tail: true,
            },
        )?;

        // A diverging lambda body (e.g. one that `panic`s) already
        // terminated its block; skip the epilogue + return.
        if self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_terminator())
            .is_some()
        {
            return Ok(());
        }

        self.emit_epilogue(thread_param, frame_alloca, info)?;
        // Uniform closure ABI: return a pointer. If the body produced
        // an Int, box it into a BoxedInt heap object. Pointer values
        // pass through.
        let ret_ptr = match result {
            Value::Int(iv) => {
                let box_fn = self.extern_box_int.expect("ai_gc_box_int declared");
                let call = self
                    .builder
                    .build_call(
                        box_fn,
                        &[thread_param.into(), iv.into()],
                        "box_lambda_ret",
                    )
                    .map_err(|e| CodegenError::JitInit(format!(
                        "build_call ai_gc_box_int (lambda ret): {}", e
                    )))?;
                call.as_any_value_enum().into_pointer_value()
            }
            Value::Closure(p) => p,
        };
        self.builder
            .build_return(Some(&ret_ptr as &dyn inkwell::values::BasicValue))
            .map_err(|e| CodegenError::JitInit(format!("build_return: {}", e)))?;
        self.finalize_frame_zeroing(&origin_sym)?;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Prologue / epilogue / root-slot access
    // -------------------------------------------------------------------------

    fn emit_prologue(
        &mut self,
        thread_param: PointerValue<'ctx>,
        info: DefInfo<'ctx>,
        origin_sym: &str,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        let frame = self
            .builder
            .build_alloca(info.frame_ty, "gc_frame")
            .map_err(|e| CodegenError::JitInit(format!("build_alloca frame: {}", e)))?;

        // The slots-region memset is emitted HERE, but only after the
        // body is compiled and the true high-water slot count is known —
        // the conservative pre-scan reservation is often 10-100x larger
        // than what the body touches, and zeroing it per call dominated
        // hot functions. A dead marker instruction pins the position;
        // `finalize_frame_zeroing` replaces it with the right-sized
        // memset (and shrinks the origin's scanned-slot count to match).
        let marker = self
            .builder
            .build_ptr_to_int(frame, self.i64_ty, "frame_zero_marker")
            .map_err(|e| CodegenError::JitInit(format!("frame zero marker: {}", e)))?;
        self.cur_frame_max_slot = 0;
        self.cur_frame_zero_point =
            Some((frame, marker.as_instruction_value().expect("marker inst")));

        let top_frame_slot =
            self.thread_field(thread_param, thread_offsets::TOP_FRAME, "top_frame_ptr")?;
        let old_top = self
            .builder
            .build_load(self.ptr_ty, top_frame_slot, "old_top_frame")
            .map_err(|e| CodegenError::JitInit(format!("load top_frame: {}", e)))?;

        let parent_slot = self.frame_field(frame, info.frame_ty, 0, "parent_slot")?;
        self.builder
            .build_store(parent_slot, old_top)
            .map_err(|e| CodegenError::JitInit(format!("store frame.parent: {}", e)))?;

        // Escape: link our frame in.
        self.builder
            .build_store(top_frame_slot, frame)
            .map_err(|e| CodegenError::JitInit(format!("store thread.top_frame: {}", e)))?;

        let origin_global = self.frame_origins[origin_sym];
        let origin_slot = self.frame_field(frame, info.frame_ty, 1, "origin_slot")?;
        self.builder
            .build_store(origin_slot, origin_global.as_pointer_value())
            .map_err(|e| CodegenError::JitInit(format!("store frame.origin: {}", e)))?;

        Ok(frame)
    }

    /// After a function body is fully compiled, materialize its EXACT
    /// frame — there is no conservative reservation anywhere:
    ///
    /// 1. Replace the placeholder `{ptr, ptr, [0 x ptr]}` alloca with
    ///    one carrying exactly `used` slots (opaque-pointer GEPs make
    ///    replace-all-uses safe — every slot access was byte/element
    ///    arithmetic off the frame pointer).
    /// 2. Insert the slots memset (only `used * 8` bytes) at the marker
    ///    planted by `emit_prologue`.
    /// 3. Shrink the frame origin's `num_roots` to `used` so the GC
    ///    walker scans exactly the slots the body can write.
    fn finalize_frame_zeroing(&mut self, origin_sym: &str) -> Result<(), CodegenError> {
        let Some((frame, marker)) = self.cur_frame_zero_point.take() else {
            return Ok(());
        };
        let used = self.cur_frame_max_slot;
        self.cur_frame_max_slot = 0;
        let saved_block = self.builder.get_insert_block();

        // 1. Exact-size alloca, swapped in for the placeholder.
        let old_inst = frame
            .as_instruction_value()
            .expect("frame placeholder is an alloca instruction");
        self.builder.position_before(&old_inst);
        let exact_ty = self.context.struct_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.ptr_ty.array_type(used).into(),
            ],
            false,
        );
        let new_frame = self
            .builder
            .build_alloca(exact_ty, "gc_frame")
            .map_err(|e| CodegenError::JitInit(format!("build_alloca exact frame: {}", e)))?;
        let new_inst = new_frame
            .as_instruction_value()
            .expect("alloca is an instruction");
        old_inst.replace_all_uses_with(&new_inst);
        old_inst.erase_from_basic_block();

        // 2. Right-sized zeroing of the slots region.
        if used > 0 {
            self.builder.position_before(&marker);
            let i8_ty = self.context.i8_type();
            let slots_base = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        i8_ty,
                        new_frame,
                        &[self.i64_ty.const_int(16, false)],
                        "frame_slots_base",
                    )
                    .map_err(|e| CodegenError::JitInit(format!("gep slots base: {}", e)))?
            };
            self.builder
                .build_memset(
                    slots_base,
                    8,
                    i8_ty.const_zero(),
                    self.i64_ty.const_int(used as u64 * 8, false),
                )
                .map_err(|e| CodegenError::JitInit(format!("build_memset frame: {}", e)))?;
        }
        marker
            .erase_from_basic_block();
        if let Some(bb) = saved_block {
            self.builder.position_at_end(bb);
        }
        // Shrink the origin's scanned-slot count to match. The third
        // field (the name pointer) is re-read from the existing global's
        // initializer via the frame_origins table's paired name global —
        // we rebuild the whole constant.
        let origin_global = self.frame_origins[origin_sym];
        let old_init = origin_global
            .get_initializer()
            .expect("origin global has initializer")
            .into_struct_value();
        let name_ptr = old_init
            .get_field_at_index(2)
            .expect("origin name field");
        let new_init = self.frame_origin_ty.const_named_struct(&[
            self.context.i32_type().const_int(used as u64, false).into(),
            self.context.i32_type().const_zero().into(),
            name_ptr,
        ]);
        origin_global.set_initializer(&new_init);
        Ok(())
    }

    /// Emit an inline GC safepoint poll at the current insert point.
    ///
    /// Loads `thread.state` (the first byte of the `Thread` struct) and,
    /// if non-zero, branches to `ai_gc_pollcheck_slow(thread, origin)`
    /// which parks this thread until the requesting collector finishes.
    /// `origin` is read from `frame.origin` (frame field 1) so the slow
    /// path can expose this thread's live JIT frame to the GC root scan.
    ///
    /// Single-threaded, `state` is always 0 so the branch is never taken
    /// — the cost is one relaxed byte load + a predictable branch per
    /// `loop_body` entry. Becomes load-bearing once concurrent mutator
    /// threads exist: a thread spinning in a self-tail-call loop with no
    /// allocations would otherwise never reach a safepoint, deadlocking
    /// any STW collection another thread requests.
    ///
    /// On return the builder is positioned at the continuation block, so
    /// callers continue emitting straight-line code.
    fn emit_safepoint_poll(
        &self,
        thread_param: PointerValue<'ctx>,
        origin: PointerValue<'ctx>,
    ) -> Result<(), CodegenError> {
        let i8_ty = self.context.i8_type();
        let cur_fn = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or_else(|| CodegenError::JitInit(
                "emit_safepoint_poll: no current function".into(),
            ))?;

        let state_slot =
            self.thread_field(thread_param, thread_offsets::STATE, "sp_state_ptr")?;
        let state_load = self
            .builder
            .build_load(i8_ty, state_slot, "sp_state")
            .map_err(|e| CodegenError::JitInit(format!("load thread.state: {}", e)))?;
        // The STW coordinator flips this byte from ANOTHER thread; the
        // load must be volatile or the optimizer hoists it out of loops
        // and the thread never sees the stop request.
        state_load
            .as_instruction_value()
            .expect("load is an instruction")
            .set_volatile(true)
            .map_err(|e| CodegenError::JitInit(format!("set_volatile sp_state: {}", e)))?;
        let state = state_load.into_int_value();
        let requested = self
            .builder
            .build_int_compare(
                inkwell::IntPredicate::NE,
                state,
                i8_ty.const_zero(),
                "sp_requested",
            )
            .map_err(|e| CodegenError::JitInit(format!("icmp state: {}", e)))?;

        let slow_bb = self.context.append_basic_block(cur_fn, "sp_slow");
        let cont_bb = self.context.append_basic_block(cur_fn, "sp_cont");
        self.builder
            .build_conditional_branch(requested, slow_bb, cont_bb)
            .map_err(|e| CodegenError::JitInit(format!("br safepoint: {}", e)))?;

        // Slow path: park via the runtime handler, then rejoin.
        self.builder.position_at_end(slow_bb);
        let pollcheck = self.extern_pollcheck.expect("ai_gc_pollcheck_slow declared");
        self.builder
            .build_call(
                pollcheck,
                &[thread_param.into(), origin.into()],
                "",
            )
            .map_err(|e| CodegenError::JitInit(format!("call pollcheck_slow: {}", e)))?;
        self.builder
            .build_unconditional_branch(cont_bb)
            .map_err(|e| CodegenError::JitInit(format!("br sp_slow→cont: {}", e)))?;

        // Continue at the join block.
        self.builder.position_at_end(cont_bb);
        Ok(())
    }

    fn emit_epilogue(
        &mut self,
        thread_param: PointerValue<'ctx>,
        frame: PointerValue<'ctx>,
        info: DefInfo<'ctx>,
    ) -> Result<(), CodegenError> {
        if !info.has_frame {
            return Ok(());
        }
        let parent = self
            .builder
            .build_load(self.ptr_ty, frame, "parent_for_pop")
            .map_err(|e| CodegenError::JitInit(format!("load frame.parent: {}", e)))?;
        let top_frame_slot =
            self.thread_field(thread_param, thread_offsets::TOP_FRAME, "top_frame_for_pop")?;
        let store = self
            .builder
            .build_store(top_frame_slot, parent)
            .map_err(|e| CodegenError::JitInit(format!("store pop top_frame: {}", e)))?;
        store
            .set_volatile(true)
            .map_err(|e| CodegenError::JitInit(format!("set_volatile epilogue: {}", e)))?;
        Ok(())
    }

    /// The shared placeholder frame layout: `{ ptr parent, ptr origin,
    /// [0 x ptr] roots }`. Real slot counts are never pre-reserved; the
    /// prologue allocas this, and `finalize_frame_zeroing` swaps in an
    /// alloca of exactly the slots the body wrote.
    fn placeholder_frame_ty(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.ptr_ty.array_type(0).into(),
            ],
            false,
        )
    }

    fn thread_field(
        &self,
        thread: PointerValue<'ctx>,
        byte_offset: usize,
        name: &str,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        let i8_ty = self.context.i8_type();
        let offset = self.i64_ty.const_int(byte_offset as u64, false);
        unsafe {
            self.builder
                .build_in_bounds_gep(i8_ty, thread, &[offset], name)
                .map_err(|e| CodegenError::JitInit(format!("gep thread.{:#x}: {}", byte_offset, e)))
        }
    }

    fn frame_field(
        &self,
        base: PointerValue<'ctx>,
        struct_ty: StructType<'ctx>,
        idx: u32,
        name: &str,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        self.builder
            .build_struct_gep(struct_ty, base, idx, name)
            .map_err(|e| CodegenError::JitInit(format!("struct_gep idx {}: {}", idx, e)))
    }

    /// Get the pointer to root slot `idx` within `frame`. Returns the
    /// address of the slot — caller stores/loads through this pointer.
    fn root_slot_ptr(
        &self,
        frame: PointerValue<'ctx>,
        info: DefInfo<'ctx>,
        slot_idx: u32,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        // GEP into the roots array (field index 2) at element slot_idx.
        let roots = self.frame_field(frame, info.frame_ty, 2, "roots")?;
        let elem_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(
                    self.ptr_ty,
                    roots,
                    &[self.i64_ty.const_int(slot_idx as u64, false)],
                    "root_slot",
                )
                .map_err(|e| CodegenError::JitInit(format!("gep root slot: {}", e)))?
        };
        Ok(elem_ptr)
    }

    /// Write a pointer into root slot `slot_idx` and return the slot's
    /// address (for later volatile loads).
    fn write_root_slot(
        &mut self,
        frame: PointerValue<'ctx>,
        info: DefInfo<'ctx>,
        slot_idx: u32,
        value: PointerValue<'ctx>,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        self.cur_frame_max_slot = self.cur_frame_max_slot.max(slot_idx + 1);
        let slot = self.root_slot_ptr(frame, info, slot_idx)?;
        let store = self
            .builder
            .build_store(slot, value)
            .map_err(|e| CodegenError::JitInit(format!("store root slot: {}", e)))?;
        store
            .set_volatile(true)
            .map_err(|e| CodegenError::JitInit(format!("set_volatile root store: {}", e)))?;
        Ok(slot)
    }

    /// Volatile-load a pointer from a previously-recorded root slot.
    fn read_root_slot(
        &self,
        slot: PointerValue<'ctx>,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        let load = self
            .builder
            .build_load(self.ptr_ty, slot, "root_load")
            .map_err(|e| CodegenError::JitInit(format!("load root slot: {}", e)))?;
        // load is a BasicValueEnum — set_volatile lives on the underlying
        // InstructionValue.
        load.as_instruction_value()
            .expect("load returns an instruction")
            .set_volatile(true)
            .map_err(|e| CodegenError::JitInit(format!("set_volatile root load: {}", e)))?;
        Ok(load.into_pointer_value())
    }

    /// Compile a sequence of operand expressions for a multi-operand
    /// construct (call args, struct/enum payloads, multi-arg builtins),
    /// rooting every pointer-typed result across the evaluation of LATER
    /// operands.
    ///
    /// THE BUG THIS FIXES: operands are evaluated left-to-right, but an
    /// earlier operand's heap pointer would otherwise live only in an SSA
    /// register while a later operand is compiled. If that later operand
    /// allocates and triggers a relocating GC, the earlier pointer goes
    /// stale (points into from-space) and the consuming call dereferences
    /// garbage. Normally GC fires rarely so the window is almost never hit;
    /// `AI_LANG_GC_STRESS=1` (collect-on-every-alloc) hits it deterministically.
    ///
    /// Fix: as soon as operand `i` is computed (and boxed, if
    /// `box_int_to_ptr[i]` and it came back as an `Int`), spill any pointer
    /// result to a dedicated frame root slot. After the whole list is
    /// computed, reload each spilled pointer from its (now-relocated) slot.
    /// `Int` results pass through unrooted — they're plain i64, not traced.
    ///
    /// Slot budget: this uses `operands.len()` root slots starting at
    /// `ctx.next_root_slot`, and compiles each operand with `next_root_slot`
    /// advanced PAST them so operand-internal `let`s/sub-calls never collide
    /// with the spill slots. `count_gc_locals` mirrors this by reserving
    /// `args.len() + 1` slots per `Call` and `fields.len()` per `StructNew`.
    /// Whether compiling `e` can NEVER allocate (and therefore never
    /// trigger a collection): literals, locals, field loads, and trees
    /// of scalar arithmetic builtins over such. Calls to user defs are
    /// conservatively allocating (their bodies may allocate even when
    /// their signatures are scalar). Used to skip the volatile
    /// spill+reload of an earlier pointer operand when every LATER
    /// operand is inert — the dominant cost of hot array code.
    fn expr_cannot_trigger_gc(e: &Expr) -> bool {
        match e {
            Expr::IntLit(_)
            | Expr::FloatLit(_)
            | Expr::BoolLit(_)
            | Expr::LocalVar(_) => true,
            Expr::Field { base, .. } => Self::expr_cannot_trigger_gc(base),
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                Self::expr_cannot_trigger_gc(cond)
                    && Self::expr_cannot_trigger_gc(then_branch)
                    && Self::expr_cannot_trigger_gc(else_branch)
            }
            Expr::Call(callee, args) => {
                matches!(
                    callee.as_ref(),
                    Expr::BuiltinRef(n)
                        if n.starts_with("core/i64.") || n.starts_with("core/f64.")
                ) && args.iter().all(Self::expr_cannot_trigger_gc)
            }
            _ => false,
        }
    }

    fn compile_operands_rooted(
        &mut self,
        operands: &[Expr],
        box_int_to_ptr: &[bool],
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Vec<Value<'ctx>>, CodegenError> {
        let base = ctx.next_root_slot;
        let k = operands.len() as u32;
        // Operands compile ABOVE the reserved spill region [base, base+k).
        let operand_ctx = CompileCtx {
            is_tail: false,
            next_root_slot: base + k,
            ..ctx
        };
        let mut slots: Vec<Option<PointerValue<'ctx>>> = Vec::with_capacity(operands.len());
        let mut vals: Vec<Value<'ctx>> = Vec::with_capacity(operands.len());
        for (i, op) in operands.iter().enumerate() {
            let mut v = self.compile_expr(op, env, operand_ctx)?;
            // Box an Int flowing into a pointer-typed (e.g. TypeVar) slot
            // BEFORE spilling — the box itself allocates, so its result must
            // be rooted just like any other operand pointer.
            if box_int_to_ptr.get(i).copied().unwrap_or(false) {
                if let Value::Int(iv) = v {
                    let box_fn = self.extern_box_int.expect("ai_gc_box_int declared");
                    let call = self
                        .builder
                        .build_call(box_fn, &[ctx.thread_param.into(), iv.into()], "box_int_arg")
                        .map_err(|e| {
                            CodegenError::JitInit(format!("build_call ai_gc_box_int: {}", e))
                        })?;
                    v = Value::Closure(call.as_any_value_enum().into_pointer_value());
                }
            }
            match v {
                Value::Closure(p) => {
                    // Root only if something AFTER this operand can
                    // allocate: either a later operand's own compilation,
                    // or a later box-to-pointer (the box allocates).
                    let later_can_gc = operands
                        .iter()
                        .enumerate()
                        .skip(i + 1)
                        .any(|(j, later)| {
                            !Self::expr_cannot_trigger_gc(later)
                                || box_int_to_ptr.get(j).copied().unwrap_or(false)
                        });
                    if later_can_gc {
                        let slot = self.write_root_slot(
                            ctx.frame_alloca,
                            ctx.info,
                            base + i as u32,
                            p,
                        )?;
                        slots.push(Some(slot));
                    } else {
                        slots.push(None);
                    }
                    vals.push(v);
                }
                Value::Int(_) => {
                    slots.push(None);
                    vals.push(v);
                }
            }
        }
        // Reload each spilled pointer from its (possibly relocated) slot.
        for (i, slot) in slots.into_iter().enumerate() {
            if let Some(slot) = slot {
                vals[i] = Value::Closure(self.read_root_slot(slot)?);
            }
        }
        Ok(vals)
    }

    // -------------------------------------------------------------------------
    // Expressions
    // -------------------------------------------------------------------------

    /// Coerce a compiled value to a uniform heap pointer: box an `Int` into
    /// a `BoxedInt`, pass a pointer through unchanged. Used where a runtime
    /// primitive needs a heap object regardless of static type (e.g. the
    /// structural `value_hash`/`value_eq`).
    fn value_as_ptr(
        &self,
        v: Value<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        match v {
            Value::Closure(p) => Ok(p),
            Value::Int(iv) => {
                let box_fn = self.extern_box_int.expect("ai_gc_box_int declared");
                let boxed = self
                    .builder
                    .build_call(box_fn, &[ctx.thread_param.into(), iv.into()], "as_ptr_boxed")
                    .map_err(|e| CodegenError::JitInit(format!(
                        "build_call ai_gc_box_int (value_as_ptr): {}", e
                    )))?;
                Ok(boxed.as_any_value_enum().into_pointer_value())
            }
        }
    }

    fn compile_expr(
        &mut self,
        e: &Expr,
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        match e {
            Expr::IntLit(n) => Ok(Value::Int(self.i64_ty.const_int(*n as u64, true))),
            // Float is carried as the i64 bit-pattern of the f64 (see the
            // `core/f64.*` builtins). A literal is the const f64's bits.
            Expr::FloatLit(x) => Ok(Value::Int(
                self.i64_ty.const_int(x.to_bits(), false),
            )),

            // A node `state` reference: resolve the live cell on the
            // executing node via `ai_state_get(thread, &hash)`. The cell
            // is always a heap pointer; a bare-`Int` state is stored
            // boxed (the installer boxes it), so the load side unboxes
            // symmetrically and yields an Int value.
            Expr::StateRef(h) => {
                let hp = self.emit_hash_constant(h).as_pointer_value();
                let f = self.extern_state_get.expect("ai_state_get declared");
                let call = self
                    .builder
                    .build_call(f, &[ctx.thread_param.into(), hp.into()], "state_get")
                    .map_err(|e| CodegenError::JitInit(format!(
                        "build_call ai_state_get: {}", e
                    )))?;
                let cell = call.as_any_value_enum().into_pointer_value();
                let is_int_state = matches!(
                    self.state_types.get(h),
                    Some(Type::Builtin(n)) if n == "Int"
                );
                if is_int_state {
                    let unbox = self
                        .extern_unbox_int
                        .expect("ai_gc_unbox_int declared");
                    let v = self
                        .builder
                        .build_call(unbox, &[cell.into()], "state_unbox")
                        .map_err(|e| CodegenError::JitInit(format!(
                            "build_call ai_gc_unbox_int: {}", e
                        )))?;
                    Ok(Value::Int(v.as_any_value_enum().into_int_value()))
                } else {
                    Ok(Value::Closure(cell))
                }
            }
            // StateSelfRef only appears in hashing bytes; the resolver
            // rewrites it to StateRef before storing, so reaching codegen
            // is a bug, not a user error.
            Expr::StateSelfRef(_) => Err(CodegenError::Unsupported {
                what: "StateSelfRef reached codegen (resolver should have \
                       rewritten it to StateRef)".to_owned(),
            }),

            Expr::LocalVar(idx) => {
                let n = env.len();
                let i = (n as i64) - 1 - (*idx as i64);
                if i < 0 || (i as usize) >= n {
                    return Err(CodegenError::Unsupported {
                        what: format!("LocalVar({}) out of range (env depth {})", idx, n),
                    });
                }
                Ok(env.get(i as usize).read(self)?)
            }

            Expr::Call(callee, args) => self.compile_call(callee, args, env, ctx),

            // A named top-level function or concrete builtin used as a
            // value: eta-expand it into its adapter closure (registered
            // during the pre-scan). SelfRef / generic-or-special builtins
            // fall through to a clear error.
            Expr::TopRef(_) | Expr::BuiltinRef(_) => {
                match self.value_ref_adapter_parts(e) {
                    Some((params, body)) => {
                        self.compile_lambda_construction(&params, &body, env, ctx)
                    }
                    None => Err(Self::value_ref_unsupported(e)),
                }
            }
            Expr::SelfRef(_) => Err(Self::value_ref_unsupported(e)),

            // Bool is represented as i64 0/1 throughout the runtime (the
            // comparison builtins already return 0/1 widened to i64), so a
            // `BoolLit` lowers to the same constant form as an `IntLit`:
            // `false` -> 0, `true` -> 1.
            Expr::BoolLit(b) => Ok(Value::Int(self.i64_ty.const_int(*b as u64, false))),

            Expr::StringLit(s) => {
                // Emit a private constant byte array global holding
                // the literal data, then call ai_str_new(thread,
                // bytes, len) to allocate the heap-resident String
                // and copy the bytes in.
                let bytes = s.as_bytes();
                let arr_const = self.context.const_string(bytes, false);
                let g = self.module.add_global(
                    arr_const.get_type(),
                    Some(AddressSpace::default()),
                    "string_lit",
                );
                g.set_linkage(Linkage::Private);
                g.set_constant(true);
                g.set_initializer(&arr_const);
                self.used_globals.push(g.as_pointer_value());

                let str_new = self
                    .extern_str_new
                    .expect("ai_str_new extern declared");
                let len_v = self.i64_ty.const_int(bytes.len() as u64, false);
                let call = self
                    .builder
                    .build_call(
                        str_new,
                        &[
                            ctx.thread_param.into(),
                            g.as_pointer_value().into(),
                            len_v.into(),
                        ],
                        "string_lit_alloc",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_str_new: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }

            Expr::Lambda { params, body } => {
                self.compile_lambda_construction(params, body, env, ctx)
            }

            Expr::StructNew { struct_ref, fields } => {
                self.compile_struct_new(struct_ref, fields, env, ctx)
            }

            Expr::Field { base, struct_ref, index } => {
                self.compile_field(base, struct_ref, *index, env, ctx)
            }

            Expr::EnumNew {
                enum_ref,
                variant_index,
                payload,
            } => self.compile_enum_new(enum_ref, *variant_index, payload.as_deref(), env, ctx),

            Expr::Match { scrutinee, arms } => self.compile_match(scrutinee, arms, env, ctx),

            Expr::Let { value, body } => {
                // The Let value is NOT in tail position (something
                // continues after it — the body). The body inherits
                // the enclosing tail context.
                let value_ctx = CompileCtx {
                    is_tail: false,
                    ..ctx
                };
                let v = self.compile_expr(value, env, value_ctx)?;
                let value_ty = self.infer_type(value, env);
                let slot = match v {
                    Value::Closure(p) => {
                        let slot_idx = ctx.next_root_slot;
                        let slot = self.write_root_slot(ctx.frame_alloca, ctx.info, slot_idx, p)?;
                        env.push(EnvSlot::Closure(slot), value_ty);
                        true
                    }
                    Value::Int(iv) => {
                        env.push(EnvSlot::Int(iv), value_ty);
                        false
                    }
                };
                let result_ctx = if slot {
                    CompileCtx {
                        next_root_slot: ctx.next_root_slot + 1,
                        ..ctx
                    }
                } else {
                    ctx
                };
                let result = self.compile_expr(body, env, result_ctx)?;
                env.pop();
                Ok(result)
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => self.compile_if(cond, then_branch, else_branch, env, ctx),
            Expr::Try {
                expr,
                enum_ref,
                ok_index,
                err_index,
            } => self.compile_try(expr, enum_ref, *ok_index, *err_index, env, ctx),
            Expr::Defer { cleanup, body } => self.compile_defer(cleanup, body, env, ctx),
        }
    }

    /// Compile `defer cleanup; body`. The cleanup runs on every way out:
    /// normal completion of `body` (emitted here) and any `?` inside
    /// `body` that early-returns (emitted by `compile_try`, which walks
    /// `self.deferred`). Cleanups run in LIFO order.
    ///
    /// The body is compiled NOT in tail position — work (the cleanup)
    /// happens after it, so its final call cannot be a tail call. A
    /// pointer-valued body result is spilled to the frame's `defer_scratch`
    /// root slot across the cleanup, so that a cleanup which itself
    /// allocates (and triggers a collection) cannot leave the result a
    /// stale, pre-move pointer.
    fn compile_defer(
        &mut self,
        cleanup: &Expr,
        body: &Expr,
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // Register the cleanup so a `?` inside `body` runs it. The env
        // snapshot pins the slot addresses + de Bruijn depth at this site.
        self.deferred.push(DeferEntry {
            cleanup: cleanup.clone(),
            env: env.clone(),
            next_root_slot: ctx.next_root_slot,
        });
        let body_ctx = CompileCtx {
            is_tail: false,
            ..ctx
        };
        let result = self.compile_expr(body, env, body_ctx)?;
        // Unregister before emitting the normal-path cleanup (it is not a
        // pending cleanup for itself).
        self.deferred.pop();

        // If `body` diverged (its block is already terminated, e.g. an
        // unconditional `panic` or a `?` on every path), there is no
        // normal fall-through to clean up; the early-return paths already
        // ran the cleanup.
        let terminated = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_terminator())
            .is_some();
        if terminated {
            return Ok(result);
        }

        // Spill a pointer result across the cleanup, then run the cleanup,
        // then reload (so a collection triggered by the cleanup updates
        // the result through the root slot). Int results need no spill.
        // The scratch slot is the next free index; the cleanup compiles
        // with its free region bumped ABOVE it so its own spills can
        // never clobber the held result.
        let spill_slot = match result {
            Value::Closure(p) => Some(self.write_root_slot(
                ctx.frame_alloca,
                ctx.info,
                ctx.next_root_slot,
                p,
            )?),
            Value::Int(_) => None,
        };
        let cctx = CompileCtx {
            is_tail: false,
            next_root_slot: ctx.next_root_slot + 1,
            ..ctx
        };
        let _ = self.compile_expr(cleanup, env, cctx)?;
        match spill_slot {
            Some(slot) => Ok(Value::Closure(self.read_root_slot(slot)?)),
            None => Ok(result),
        }
    }

    /// Compile `expr?` (the try operator). `expr` evaluates to a heap
    /// `Result<T, E>` pointer. We read the discriminant: on `Err` we
    /// early-return the whole `Result` value from the enclosing function
    /// (its return type is a `Result<_, E>`, a pointer, so the value
    /// passes through unchanged); on `Ok` we extract and yield the `T`
    /// payload, unboxing it from a `BoxedInt` when `T` is a scalar.
    /// Emit the `?` error path: run every pending `defer` cleanup in LIFO
    /// order (spilling the Err Result pointer to the next free root slot
    /// so a cleanup that allocates — and collects — cannot leave it a
    /// stale, pre-move pointer; cleanups compile with their free region
    /// bumped ABOVE the scratch), then pop the frame and early-return the
    /// Result. Terminates the current block.
    fn emit_try_err_return(
        &mut self,
        result_ptr: PointerValue<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<(), CodegenError> {
        let err_result_ptr = if self.deferred.is_empty() {
            result_ptr
        } else {
            let scratch = self.write_root_slot(
                ctx.frame_alloca,
                ctx.info,
                ctx.next_root_slot,
                result_ptr,
            )?;
            let pending: Vec<DeferEntry<'ctx>> = self.deferred.clone();
            for entry in pending.iter().rev() {
                let mut cenv = entry.env.clone();
                let cctx = CompileCtx {
                    is_tail: false,
                    next_root_slot: ctx.next_root_slot + 1,
                    ..ctx
                };
                let _ = self.compile_expr(&entry.cleanup, &mut cenv, cctx)?;
            }
            self.read_root_slot(scratch)?
        };
        self.emit_epilogue(ctx.thread_param, ctx.frame_alloca, ctx.info)?;
        self.builder
            .build_return(Some(&err_result_ptr as &dyn inkwell::values::BasicValue))
            .map_err(|e| CodegenError::JitInit(format!("build_return try err: {}", e)))?;
        Ok(())
    }

    fn compile_try(
        &mut self,
        operand: &Expr,
        enum_ref: &Hash,
        ok_index: u32,
        err_index: u32,
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // The operand's `Result<T, E>` instantiation drives Ok-payload
        // box/unbox, just like a match scrutinee.
        let instantiation: Vec<Type> = match self.infer_type(operand, env) {
            Type::Apply(_, args) => args,
            _ => Vec::new(),
        };

        // Compile the operand — not in tail position.
        let op_ctx = CompileCtx {
            is_tail: false,
            ..ctx
        };
        let rv = self.compile_expr(operand, env, op_ctx)?;
        let result_ptr = rv.as_closure().map_err(|_| CodegenError::TypeMismatch {
            what: "`?` operand must be a Result (heap pointer)".to_owned(),
        })?;

        // A SYNTACTICALLY-Err operand (`Err(e)?` — emitted by the fused
        // checked-accessor expansion's out-of-bounds arms) early-returns
        // unconditionally: no tag check, and crucially no dead Ok path
        // (whose payload extraction would fabricate a value of the wrong
        // kind). Leaves the block TERMINATED; enclosing if/match merges
        // detect that and skip the phi, exactly like a diverging `abort`.
        if matches!(
            operand,
            Expr::EnumNew { enum_ref: er, variant_index, .. }
                if er == enum_ref && *variant_index == err_index
        ) {
            self.emit_try_err_return(result_ptr, ctx)?;
            return Ok(Value::Int(self.i64_ty.const_zero()));
        }

        let einfo = self
            .enums
            .get(enum_ref)
            .cloned()
            .ok_or(CodegenError::UnknownTopRef { hash: *enum_ref })?;

        // Read the discriminant tag (i32), shared offset across variants.
        let tag_offset = einfo.variants[0].tag_offset;
        let tag_off_const = self.i64_ty.const_int(tag_offset as u64, false);
        let tag_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(
                    self.context.i8_type(),
                    result_ptr,
                    &[tag_off_const],
                    "try_tag_ptr",
                )
                .map_err(|e| CodegenError::JitInit(format!("gep try tag: {}", e)))?
        };
        let tag = self
            .builder
            .build_load(self.context.i32_type(), tag_ptr, "try_tag")
            .map_err(|e| CodegenError::JitInit(format!("load try tag: {}", e)))?
            .into_int_value();

        let cur_fn = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .expect("try compile must be inside a fn");
        let err_bb = self.context.append_basic_block(cur_fn, "try_err");
        let ok_bb = self.context.append_basic_block(cur_fn, "try_ok");

        // tag == err_index → err_bb (early return), else ok_bb.
        let err_tag = self
            .context
            .i32_type()
            .const_int(err_index as u64, false);
        let is_err = self
            .builder
            .build_int_compare(IntPredicate::EQ, tag, err_tag, "try_is_err")
            .map_err(|e| CodegenError::JitInit(format!("cmp try tag: {}", e)))?;
        self.builder
            .build_conditional_branch(is_err, err_bb, ok_bb)
            .map_err(|e| CodegenError::JitInit(format!("br try: {}", e)))?;

        // ---- err_bb: run pending `defer` cleanups, then early-return ----
        self.builder.position_at_end(err_bb);
        self.emit_try_err_return(result_ptr, ctx)?;

        // ---- ok_bb: extract the Ok payload ----
        self.builder.position_at_end(ok_bb);
        let ok_v = einfo.variants[ok_index as usize];
        let payload_off = ok_v.payload_offset.ok_or(CodegenError::TypeMismatch {
            what: "`?` requires Result::Ok to carry a payload".to_owned(),
        })?;
        let off_const = self.i64_ty.const_int(payload_off as u64, false);
        let payload_slot = unsafe {
            self.builder
                .build_in_bounds_gep(
                    self.context.i8_type(),
                    result_ptr,
                    &[off_const],
                    "try_ok_payload_ptr",
                )
                .map_err(|e| CodegenError::JitInit(format!("gep try ok payload: {}", e)))?
        };
        // Whether the Ok payload slot holds a BoxedInt (declared TypeVar
        // instantiated to a scalar) we must unbox — mirrors match.
        let declared_payload_ty = self
            .enum_variant_types
            .get(enum_ref)
            .and_then(|vs| vs.get(ok_index as usize).cloned().flatten());
        let needs_unbox = match &declared_payload_ty {
            Some(Type::TypeVar(_))
                if matches!(
                    substitute_type(declared_payload_ty.as_ref().unwrap(), &instantiation),
                    Type::Builtin(ref n) if is_boxed_scalar(n)
                ) =>
            {
                true
            }
            _ => false,
        };
        if ok_v.payload_is_pointer && !needs_unbox {
            let loaded = self
                .builder
                .build_load(self.ptr_ty, payload_slot, "try_ok_ptr")
                .map_err(|e| CodegenError::JitInit(format!("load try ok ptr: {}", e)))?
                .into_pointer_value();
            Ok(Value::Closure(loaded))
        } else if ok_v.payload_is_pointer && needs_unbox {
            let boxed_ptr = self
                .builder
                .build_load(self.ptr_ty, payload_slot, "try_ok_boxed_ptr")
                .map_err(|e| CodegenError::JitInit(format!("load try ok boxed ptr: {}", e)))?
                .into_pointer_value();
            let unbox_fn = self.extern_unbox_int.expect("ai_gc_unbox_int declared");
            let call = self
                .builder
                .build_call(unbox_fn, &[boxed_ptr.into()], "try_ok_unboxed")
                .map_err(|e| {
                    CodegenError::JitInit(format!("build_call ai_gc_unbox_int (try): {}", e))
                })?;
            Ok(Value::Int(call.as_any_value_enum().into_int_value()))
        } else {
            let loaded = self
                .builder
                .build_load(self.i64_ty, payload_slot, "try_ok_int")
                .map_err(|e| CodegenError::JitInit(format!("load try ok int: {}", e)))?
                .into_int_value();
            Ok(Value::Int(loaded))
        }
    }

    /// Emit a self-tail-call as a branch back to `loop_body`. Computes
    /// each new argument expression, writes it into the corresponding
    /// param slot, then branches. Positions the builder at an
    /// unreachable post-branch block so subsequent emitted code (e.g.
    /// an enclosing if's `br merge_bb`) lands somewhere LLVM accepts
    /// and gets dead-code-eliminated.
    ///
    /// Returns a placeholder `Value` of the right LLVM kind so the
    /// caller's type-shape expectations are satisfied. The placeholder
    /// is unreachable at runtime.
    fn emit_self_tail_call(
        &mut self,
        args: &[Expr],
        tail: &TailCtx<'ctx>,
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // Evaluate every arg first (to fresh SSA values), THEN write them
        // all into the param slots. Evaluating-before-storing handles the
        // case where an arg references a param that's about to be
        // overwritten. `compile_operands_rooted` additionally spills each
        // pointer arg to a TEMPORARY root slot across the evaluation of
        // later args — without it, an earlier pointer arg (e.g. `cell.tail`)
        // goes stale when a later arg (e.g. `List::Cons(...)`) allocates and
        // triggers a relocating GC. It also boxes Int args bound for a
        // pointer-typed param slot (uniform ABI).
        let box_flags: Vec<bool> = tail
            .param_slots
            .iter()
            .take(args.len())
            .map(|s| matches!(s, TailParamSlot::Ptr(_)))
            .collect();
        let arg_vals = self.compile_operands_rooted(args, &box_flags, env, ctx)?;
        for (i, v) in arg_vals.into_iter().enumerate() {
            match tail.param_slots[i] {
                TailParamSlot::Int(slot) => {
                    let iv = v.as_int().map_err(|_| CodegenError::TypeMismatch {
                        what: format!(
                            "self-tail-call arg {} must be Int (param is Int-typed)",
                            i
                        ),
                    })?;
                    self.builder.build_store(slot, iv).map_err(|e| {
                        CodegenError::JitInit(format!("store tail arg {}: {}", i, e))
                    })?;
                }
                TailParamSlot::Ptr(slot) => {
                    let pv =
                        v.as_closure().map_err(|_| CodegenError::TypeMismatch {
                            what: format!(
                                "self-tail-call arg {} must be a pointer-typed value",
                                i
                            ),
                        })?;
                    let store = self.builder.build_store(slot, pv).map_err(|e| {
                        CodegenError::JitInit(format!(
                            "store tail ptr arg {}: {}",
                            i, e
                        ))
                    })?;
                    // Match the volatile convention used by the
                    // root-slot read path so the GC always sees the
                    // most recent value.
                    store.set_volatile(true).map_err(|e| {
                        CodegenError::JitInit(format!(
                            "set_volatile tail ptr store: {}",
                            e
                        ))
                    })?;
                }
            }
        }
        // Branch back to the body's loop entry.
        self.builder
            .build_unconditional_branch(tail.loop_body)
            .map_err(|e| CodegenError::JitInit(format!("br loop_body: {}", e)))?;

        // Position at a fresh unreachable block so any continuation
        // code the caller emits has somewhere to go. LLVM's DCE will
        // remove it.
        let cur_fn = tail
            .loop_body
            .get_parent()
            .expect("loop_body must have a parent fn");
        let dead_bb = self.context.append_basic_block(cur_fn, "after_tail_call");
        self.builder.position_at_end(dead_bb);

        // Return a placeholder of the same shape as the fn's return.
        // The fn's LLVM return type tells us whether to use an Int or
        // a Closure placeholder.
        let ret_ty = cur_fn.get_type().get_return_type();
        if matches!(
            ret_ty,
            Some(inkwell::types::BasicTypeEnum::PointerType(_))
        ) {
            Ok(Value::Closure(self.ptr_ty.const_null()))
        } else {
            Ok(Value::Int(self.i64_ty.const_zero()))
        }
    }

    /// `if cond { then } else { else }` — emit branch + phi.
    ///
    /// `cond` is Int (0 = false, non-zero = true). Both branches
    /// must produce the same LLVM type — we copy the typechecker's
    /// agreement contract: the first branch's result type wins, and
    /// the second is asserted to match at the phi.
    fn compile_if(
        &mut self,
        cond: &Expr,
        then_branch: &Expr,
        else_branch: &Expr,
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        let cur_fn = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .expect("if compile must be inside a fn");

        // The condition is never in tail position.
        let cond_ctx = CompileCtx {
            is_tail: false,
            ..ctx
        };
        let cond_v = self.compile_expr(cond, env, cond_ctx)?;
        let cond_int = cond_v.as_int().map_err(|_| CodegenError::TypeMismatch {
            what: "if condition must be Int (0 = false, non-zero = true)".to_owned(),
        })?;
        // Compare cond != 0 to get an i1.
        let zero = self.i64_ty.const_zero();
        let cond_bool = self
            .builder
            .build_int_compare(IntPredicate::NE, cond_int, zero, "if_cond")
            .map_err(|e| CodegenError::JitInit(format!("build_int_compare if: {}", e)))?;

        let then_bb = self.context.append_basic_block(cur_fn, "if_then");
        let else_bb = self.context.append_basic_block(cur_fn, "if_else");
        let merge_bb = self.context.append_basic_block(cur_fn, "if_merge");

        self.builder
            .build_conditional_branch(cond_bool, then_bb, else_bb)
            .map_err(|e| CodegenError::JitInit(format!("build_conditional_branch: {}", e)))?;

        // ---- then ----
        // A branch whose body already terminated its block (e.g. it
        // ended in a `panic(...)` lowering to `unreachable`) contributes
        // no value or branch to the merge phi.
        self.builder.position_at_end(then_bb);
        let then_val = self.compile_expr(then_branch, env, ctx)?;
        let then_end = self
            .builder
            .get_insert_block()
            .expect("then branch must have a current block");
        let then_incoming = if then_end.get_terminator().is_none() {
            let b = then_val.into_basic();
            self.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| CodegenError::JitInit(format!("br then→merge: {}", e)))?;
            Some((b, then_end))
        } else {
            None
        };

        // ---- else ----
        self.builder.position_at_end(else_bb);
        let else_val = self.compile_expr(else_branch, env, ctx)?;
        let else_end = self
            .builder
            .get_insert_block()
            .expect("else branch must have a current block");
        let else_incoming = if else_end.get_terminator().is_none() {
            let b = else_val.into_basic();
            self.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| CodegenError::JitInit(format!("br else→merge: {}", e)))?;
            Some((b, else_end))
        } else {
            None
        };

        // ---- merge: phi over the surviving branch(es) ----
        self.builder.position_at_end(merge_bb);
        let incoming: Vec<(BasicValueEnum<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> =
            [then_incoming, else_incoming].into_iter().flatten().collect();
        if incoming.is_empty() {
            // Both branches diverge; merge_bb is unreachable. Return a
            // dummy value — the block is dead and LLVM eliminates it.
            return Ok(Value::Int(self.i64_ty.const_zero()));
        }
        let phi_ty = incoming[0].0.get_type();
        for (v, _) in &incoming {
            if v.get_type() != phi_ty {
                return Err(CodegenError::TypeMismatch {
                    what: format!(
                        "if branches produce different LLVM types: {:?} vs {:?}",
                        phi_ty,
                        v.get_type()
                    ),
                });
            }
        }
        let phi = self
            .builder
            .build_phi(phi_ty, "if_result")
            .map_err(|e| CodegenError::JitInit(format!("build_phi if: {}", e)))?;
        for (v, b) in &incoming {
            phi.add_incoming(&[(v, *b)]);
        }
        let result = phi.as_basic_value();
        match result {
            BasicValueEnum::IntValue(iv) => Ok(Value::Int(iv)),
            BasicValueEnum::PointerValue(pv) => Ok(Value::Closure(pv)),
            other => Err(CodegenError::TypeMismatch {
                what: format!("unexpected if-result type: {:?}", other),
            }),
        }
    }

    fn compile_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // Self-tail-call optimization: at a tail-position Call whose
        // callee is the current fn itself, branch back to loop_body
        // instead of emitting a real call. Direct recursion runs as
        // a loop, no native-stack growth. We do this BEFORE clearing
        // `is_tail` for the args (because args are themselves not
        // tail).
        if ctx.is_tail {
            if let Expr::TopRef(h) = callee {
                if let Some(tail) = self.tail_ctx.clone() {
                    if tail.self_hash == *h
                        && tail.param_slots.len() == args.len()
                    {
                        return self.emit_self_tail_call(args, &tail, env, ctx);
                    }
                }
            }
        }
        // From here on, args are evaluated with is_tail=false. This
        // covers args of arithmetic builtins, indirect calls,
        // externs, etc. The callee's compile_expr also doesn't need
        // tail context.
        let ctx = CompileCtx {
            is_tail: false,
            ..ctx
        };
        match callee {
            Expr::BuiltinRef(name)
                if name == "core/net.at"
                    || crate::resolve::parse_at_builtin_name(name).is_some() =>
            {
                // at(node, thunk) / at_async(node, thunk) — both args are
                // heap pointers. Lower to the matching extern; `at`
                // returns a heap Result enum pointer, `at_async` a
                // ThreadHandle (BoxedInt) that `join` awaits.
                if args.len() != 2 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let is_async =
                    name.starts_with(crate::resolve::AT_ASYNC_BUILTIN_PREFIX);
                // Root the node across the thunk's compilation: building
                // the closure ALLOCATES, and a collection there would
                // leave a bare-SSA node pointer stale (ai_net_at would
                // read a forwarding word out of from-space as the port).
                let vals = self.compile_operands_rooted(args, &[false, false], env, ctx)?;
                let node_ptr = vals[0].as_closure().map_err(|_| {
                    CodegenError::TypeMismatch {
                        what: "at: first arg must be a Node (struct pointer)".to_owned(),
                    }
                })?;
                let thunk_ptr = vals[1].as_closure().map_err(|_| {
                    CodegenError::TypeMismatch {
                        what: "at: second arg must be a closure".to_owned(),
                    }
                })?;
                let net_at = if is_async {
                    self.extern_net_at_async.expect("ai_net_at_async declared")
                } else {
                    self.extern_net_at.expect("ai_net_at declared")
                };
                let call = self
                    .builder
                    .build_call(
                        net_at,
                        &[
                            ctx.thread_param.into(),
                            node_ptr.into(),
                            thunk_ptr.into(),
                        ],
                        if is_async { "at_async_handle" } else { "at_result" },
                    )
                    .map_err(|e| CodegenError::JitInit(format!("build_call ai_net_at: {}", e)))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/wire.encode" => {
                // wire_encode(x) -> Bytes. x is any value; an Int is boxed
                // first so the codec always sees a heap pointer.
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let arg_v = self.compile_expr(&args[0], env, ctx)?;
                let val_ptr = match arg_v {
                    Value::Closure(p) => p,
                    Value::Int(iv) => {
                        let box_fn = self.extern_box_int.expect("ai_gc_box_int declared");
                        self.builder
                            .build_call(
                                box_fn,
                                &[ctx.thread_param.into(), iv.into()],
                                "box_for_encode",
                            )
                            .map_err(|e| CodegenError::JitInit(format!(
                                "build_call ai_gc_box_int (wire.encode): {}", e
                            )))?
                            .as_any_value_enum()
                            .into_pointer_value()
                    }
                };
                let f = self.extern_wire_encode.expect("ai_wire_encode declared");
                let call = self
                    .builder
                    .build_call(f, &[ctx.thread_param.into(), val_ptr.into()], "wire_encode")
                    .map_err(|e| CodegenError::JitInit(format!("build_call ai_wire_encode: {}", e)))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/wire.decode_int" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let bytes = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let f = self.extern_wire_decode_int.expect("ai_wire_decode_int declared");
                let call = self
                    .builder
                    .build_call(f, &[ctx.thread_param.into(), bytes.into()], "wire_decode_int")
                    .map_err(|e| CodegenError::JitInit(format!("build_call ai_wire_decode_int: {}", e)))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/wire.decode_fn1" => {
                // Decode a shipped closure into a callable value (a heap
                // pointer under the uniform ABI). The resolver typed the
                // result as `fn(Int)->Int` so it can be passed to `swap`.
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let bytes = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let f = self.extern_wire_decode_ptr.expect("ai_wire_decode_ptr declared");
                let call = self
                    .builder
                    .build_call(f, &[ctx.thread_param.into(), bytes.into()], "wire_decode_fn1")
                    .map_err(|e| CodegenError::JitInit(format!("build_call ai_wire_decode_ptr: {}", e)))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/wire.invoke" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let bytes = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let f = self.extern_wire_invoke.expect("ai_wire_invoke declared");
                let call = self
                    .builder
                    .build_call(f, &[ctx.thread_param.into(), bytes.into()], "wire_invoke")
                    .map_err(|e| CodegenError::JitInit(format!("build_call ai_wire_invoke: {}", e)))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name.starts_with("core/wire.decode#") => {
                // decode::<T>(bytes) -> Result<T, Int>. The 32-byte expected
                // identity hash for T is baked into the name as 64 hex chars;
                // pass it to the runtime as four little-endian i64 words.
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                // Name: core/wire.decode#<expected>#<result_hash>#<okint>.
                // The runtime check only needs the first hash.
                let hex = name
                    .strip_prefix("core/wire.decode#")
                    .and_then(|rest| rest.split('#').next())
                    .unwrap_or("");
                if hex.len() != 64 {
                    return Err(CodegenError::JitInit(format!(
                        "core/wire.decode: malformed hash in builtin name: {}",
                        name
                    )));
                }
                let mut hb = [0u8; 32];
                for i in 0..32 {
                    hb[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).map_err(|e| {
                        CodegenError::JitInit(format!("core/wire.decode: bad hash hex: {}", e))
                    })?;
                }
                let word = |lo: usize| -> u64 {
                    let mut w = [0u8; 8];
                    w.copy_from_slice(&hb[lo..lo + 8]);
                    u64::from_le_bytes(w)
                };
                let bytes = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let f = self
                    .extern_wire_decode_checked
                    .expect("ai_wire_decode_checked declared");
                let h0 = self.i64_ty.const_int(word(0), false);
                let h1 = self.i64_ty.const_int(word(8), false);
                let h2 = self.i64_ty.const_int(word(16), false);
                let h3 = self.i64_ty.const_int(word(24), false);
                let call = self
                    .builder
                    .build_call(
                        f,
                        &[
                            ctx.thread_param.into(),
                            bytes.into(),
                            h0.into(),
                            h1.into(),
                            h2.into(),
                            h3.into(),
                        ],
                        "wire_decode_checked",
                    )
                    .map_err(|e| CodegenError::JitInit(format!("build_call ai_wire_decode_checked: {}", e)))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/string.len" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let s = self.compile_expr(&args[0], env, ctx)?;
                let sp = s.as_closure().map_err(|_| CodegenError::TypeMismatch {
                    what: "core/string.len: arg must be a String".to_owned(),
                })?;
                let fv = self.extern_str_len.expect("ai_str_len declared");
                let call = self
                    .builder
                    .build_call(fv, &[sp.into()], "str_len_result")
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_str_len: {}", e),
                    ))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/string.eq" => {
                if args.len() != 2 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let vals = self.compile_operands_rooted(args, &[false, false], env, ctx)?;
                let ap = vals[0].as_closure().map_err(|_| CodegenError::TypeMismatch {
                    what: "core/string.eq: args must be Strings".to_owned(),
                })?;
                let bp = vals[1].as_closure().map_err(|_| CodegenError::TypeMismatch {
                    what: "core/string.eq: args must be Strings".to_owned(),
                })?;
                let fv = self.extern_str_eq.expect("ai_str_eq declared");
                let call = self
                    .builder
                    .build_call(fv, &[ap.into(), bp.into()], "str_eq_result")
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_str_eq: {}", e),
                    ))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/string.concat" => {
                if args.len() != 2 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let vals = self.compile_operands_rooted(args, &[false, false], env, ctx)?;
                let ap = vals[0].as_closure().map_err(|_| CodegenError::TypeMismatch {
                    what: "core/string.concat: args must be Strings".to_owned(),
                })?;
                let bp = vals[1].as_closure().map_err(|_| CodegenError::TypeMismatch {
                    what: "core/string.concat: args must be Strings".to_owned(),
                })?;
                let fv = self.extern_str_concat.expect("ai_str_concat declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), ap.into(), bp.into()],
                        "str_concat_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_str_concat: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/bytes.new" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let len = self.compile_expr(&args[0], env, ctx)?.as_int()?;
                let fv = self.extern_bytes_new.expect("ai_bytes_new declared");
                let call = self
                    .builder
                    .build_call(fv, &[ctx.thread_param.into(), len.into()], "bytes_new_result")
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_bytes_new: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/bytes.len" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                // Bytes shares String's layout; reuse ai_str_len.
                let b = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let fv = self.extern_str_len.expect("ai_str_len declared");
                let call = self
                    .builder
                    .build_call(fv, &[b.into()], "bytes_len_result")
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_str_len (bytes.len): {}", e),
                    ))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/bytes.get" => {
                if args.len() != 2 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let vals = self.compile_operands_rooted(args, &[false, false], env, ctx)?;
                let b = vals[0].as_closure()?;
                let i = vals[1].as_int()?;
                let fv = self.extern_bytes_get.expect("ai_bytes_get declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), b.into(), i.into()],
                        "bytes_get_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_bytes_get: {}", e),
                    ))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/bytes.set" => {
                if args.len() != 3 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let vals = self.compile_operands_rooted(args, &[false, false, false], env, ctx)?;
                let b = vals[0].as_closure()?;
                let i = vals[1].as_int()?;
                let v = vals[2].as_int()?;
                let fv = self.extern_bytes_set.expect("ai_bytes_set declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), b.into(), i.into(), v.into()],
                        "bytes_set_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_bytes_set: {}", e),
                    ))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/bytes.slice" => {
                if args.len() != 3 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let vals = self.compile_operands_rooted(args, &[false, false, false], env, ctx)?;
                let b = vals[0].as_closure()?;
                let start = vals[1].as_int()?;
                let len = vals[2].as_int()?;
                let fv = self.extern_bytes_slice.expect("ai_bytes_slice declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), b.into(), start.into(), len.into()],
                        "bytes_slice_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_bytes_slice: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/bytes.concat" => {
                if args.len() != 2 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                // Bytes shares String's layout; reuse ai_str_concat.
                let vals = self.compile_operands_rooted(args, &[false, false], env, ctx)?;
                let a = vals[0].as_closure()?;
                let b = vals[1].as_closure()?;
                let fv = self.extern_str_concat.expect("ai_str_concat declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), a.into(), b.into()],
                        "bytes_concat_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_str_concat (bytes.concat): {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name)
                if name == "core/bytes.from_string" || name == "core/string.from_bytes" =>
            {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                // String and Bytes are distinct shapes now, so the
                // conversion is a real cross-shape copy: bytes_from_string
                // produces a mutable Bytes, string_from_bytes an immutable
                // String.
                let src = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let fv = if name == "core/string.from_bytes" {
                    self.extern_str_copy.expect("ai_str_copy declared")
                } else {
                    self.extern_bytes_copy.expect("ai_bytes_copy declared")
                };
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), src.into()],
                        "from_copy_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call from-conversion copy: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/array.new" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let n = self.compile_expr(&args[0], env, ctx)?.as_int()?;
                let fv = self.extern_array_new.expect("ai_array_new declared");
                let call = self
                    .builder
                    .build_call(fv, &[ctx.thread_param.into(), n.into()], "array_new_result")
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_array_new: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/array.new_prim" => {
                // Unboxed array: the resolver emits this when the creation
                // site's contextual element type is a scalar (Int/Float/
                // Bool). Slots hold raw bits; `get`/`set` go through the
                // scalar fast path with no per-element boxing.
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let n = self.compile_expr(&args[0], env, ctx)?.as_int()?;
                let fv = self
                    .extern_array_new_prim
                    .expect("ai_array_new_prim declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), n.into()],
                        "array_new_prim_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_array_new_prim: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/array.len" => {
                // Fully inline — len needs no slow path: null → 0, the
                // count word holds slots (boxed) or bytes (prim, >>3).
                // All loads are invariant (shape and length never change
                // after allocation), so repeated len checks CSE away.
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let a = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let v = self.emit_array_len_inline(a, ctx)?;
                Ok(Value::Int(v))
            }
            Expr::BuiltinRef(name) if name == "core/array.get" => {
                if args.len() != 2 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                // Element type: if the array's instantiation pins T to a
                // scalar (Int/Float/Bool), use the scalar fast path —
                // `ai_array_get_i64` loads the raw bits directly from an
                // unboxed PrimArray (no allocation) and unboxes from a
                // boxed Array.
                let elem_is_int = array_element_is_int(&self.infer_type(&args[0], env));
                let vals = self.compile_operands_rooted(args, &[false, false], env, ctx)?;
                let a = vals[0].as_closure()?;
                let i = vals[1].as_int()?;
                if elem_is_int {
                    let v = self.emit_array_scalar_fastpath(a, i, None, ctx)?;
                    return Ok(Value::Int(v));
                }
                let fv = self.extern_array_get.expect("ai_array_get declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), a.into(), i.into()],
                        "array_get_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_array_get: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/array.set" => {
                if args.len() != 3 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                // Scalar element (Int/Float/Bool) that compiled to a raw
                // i64: use the scalar fast path — `ai_array_set_i64`
                // stores the bits directly into an unboxed PrimArray (no
                // allocation) and boxes only for a boxed Array. Pointer
                // values (and scalars that already arrived boxed) go
                // through the uniform path, which the runtime keeps
                // correct for both representations.
                let elem_is_int = array_element_is_int(&self.infer_type(&args[0], env));
                let vals = self.compile_operands_rooted(
                    args,
                    &[false, false, !elem_is_int],
                    env,
                    ctx,
                )?;
                let a = vals[0].as_closure()?;
                let i = vals[1].as_int()?;
                if elem_is_int {
                    if let Value::Int(v) = vals[2] {
                        let fv = self
                            .extern_array_set_i64
                            .expect("ai_array_set_i64 declared");
                        let call = self
                            .builder
                            .build_call(
                                fv,
                                &[ctx.thread_param.into(), a.into(), i.into(), v.into()],
                                "array_set_i64_result",
                            )
                            .map_err(|e| CodegenError::JitInit(
                                format!("build_call ai_array_set_i64: {}", e),
                            ))?;
                            return Ok(Value::Int(call.as_any_value_enum().into_int_value()));
                    }
                }
                // Reaching here, the value is a pointer: either the element
                // type is non-scalar (the helper boxed any raw Int via
                // box_flags), or a scalar-typed value arrived already boxed
                // (e.g. out of generic code) — the runtime's uniform `set`
                // handles both array representations for that case.
                let v_ptr = vals[2].as_closure()?;
                let fv = self.extern_array_set.expect("ai_array_set declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), a.into(), i.into(), v_ptr.into()],
                        "array_set_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_array_set: {}", e),
                    ))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/atom.new" => {
                // atom_new(init) -> Atom<T>. Allocate a dedicated `Atom`
                // cell (NOT a 1-slot Array) holding `init`. Box an Int
                // init so the slot holds a uniform pointer.
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let v = self.compile_expr(&args[0], env, ctx)?;
                let init_ptr = match v {
                    Value::Int(iv) => {
                        let box_fn = self.extern_box_int.expect("ai_gc_box_int declared");
                        let boxed = self
                            .builder
                            .build_call(
                                box_fn,
                                &[ctx.thread_param.into(), iv.into()],
                                "atom_new_boxed",
                            )
                            .map_err(|e| CodegenError::JitInit(
                                format!("build_call ai_gc_box_int (atom.new): {}", e),
                            ))?;
                        boxed.as_any_value_enum().into_pointer_value()
                    }
                    Value::Closure(p) => p,
                };
                let fv = self.extern_atom_new.expect("ai_atom_new declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), init_ptr.into()],
                        "atom_new_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_atom_new: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/atom.load" => {
                // atom_load(a) -> T. Acquiring read of the cell's value.
                // Unbox if the atom's instantiation pins T to Int.
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let elem_is_int = atom_element_is_int(&self.infer_type(&args[0], env));
                let a = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let fv = self.extern_atom_load.expect("ai_atom_load declared");
                let call = self
                    .builder
                    .build_call(fv, &[a.into()], "atom_load_result")
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_atom_load: {}", e),
                    ))?;
                let res_ptr = call.as_any_value_enum().into_pointer_value();
                if elem_is_int {
                    let unbox_fn = self.extern_unbox_int.expect("ai_gc_unbox_int declared");
                    let unboxed = self
                        .builder
                        .build_call(unbox_fn, &[res_ptr.into()], "atom_load_unboxed")
                        .map_err(|e| CodegenError::JitInit(format!(
                            "build_call ai_gc_unbox_int (atom.load): {}", e
                        )))?;
                    Ok(Value::Int(unboxed.as_any_value_enum().into_int_value()))
                } else {
                    Ok(Value::Closure(res_ptr))
                }
            }
            Expr::BuiltinRef(name) if name == "core/atom.swap" => {
                // The ONE atom primitive: lock-free swap. Args:
                //   (atom_cell, updater_closure)
                // The runtime fn runs the full CAS retry loop — load the
                // slot pointer, invoke the closure to produce a new
                // object, compare-exchange the real pointers, retry on
                // failure. Works for ANY value type because it operates
                // on raw object identity (the closure does its own
                // box/unbox via the uniform ABI). Returns the installed
                // value pointer.
                if args.len() != 2 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let elem_is_int = atom_element_is_int(&self.infer_type(&args[0], env));
                let vals = self.compile_operands_rooted(args, &[false, false], env, ctx)?;
                let a = vals[0].as_closure()?;
                let f = vals[1].as_closure()?;
                let fv = self.extern_atom_swap_local.expect("ai_atom_swap declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), a.into(), f.into()],
                        "atom_swap_result",
                    )
                    .map_err(|e| CodegenError::JitInit(format!("build_call ai_atom_swap: {}", e)))?;
                let res_ptr = call.as_any_value_enum().into_pointer_value();
                if elem_is_int {
                    let unbox_fn = self.extern_unbox_int.expect("ai_gc_unbox_int declared");
                    let unboxed = self
                        .builder
                        .build_call(unbox_fn, &[res_ptr.into()], "atom_swap_unboxed")
                        .map_err(|e| CodegenError::JitInit(format!(
                            "build_call ai_gc_unbox_int (atom.swap): {}", e
                        )))?;
                    Ok(Value::Int(unboxed.as_any_value_enum().into_int_value()))
                } else {
                    Ok(Value::Closure(res_ptr))
                }
            }
            Expr::BuiltinRef(name)
                if name == "core/thread.spawn" || name == "core/thread.spawn_shared" =>
            {
                // spawn(thunk) -> ThreadHandle<T>. The thunk is a heap
                // closure pointer; pass it straight to the runtime, which
                // starts an OS thread and returns a handle (BoxedInt id).
                // `spawn` isolates (runtime deep-copies the closure);
                // `spawn_shared` runs it zero-copy (shares the heap).
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let thunk = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let fv = if name == "core/thread.spawn_shared" {
                    self.extern_thread_spawn_shared
                        .expect("ai_thread_spawn_shared declared")
                } else {
                    self.extern_thread_spawn.expect("ai_thread_spawn declared")
                };
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), thunk.into()],
                        "thread_spawn_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_thread_spawn: {}", e),
                    ))?;
                Ok(Value::Closure(call.as_any_value_enum().into_pointer_value()))
            }
            Expr::BuiltinRef(name) if name == "core/thread.join" => {
                // join(handle) -> T. Block until the spawned thunk finishes
                // and return its result. Unbox if the handle's instantiation
                // pins T to Int (the thunk boxed its Int return).
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let elem_is_int =
                    thread_handle_element_is_int(&self.infer_type(&args[0], env));
                let h = self.compile_expr(&args[0], env, ctx)?.as_closure()?;
                let fv = self.extern_thread_join.expect("ai_thread_join declared");
                let call = self
                    .builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), h.into()],
                        "thread_join_result",
                    )
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_thread_join: {}", e),
                    ))?;
                // A worker panic is re-raised on this (joining) thread by
                // ai_thread_join; propagate it before unboxing the dummy.
                let res_ptr = call.as_any_value_enum().into_pointer_value();
                if elem_is_int {
                    let unbox_fn = self.extern_unbox_int.expect("ai_gc_unbox_int declared");
                    let unboxed = self
                        .builder
                        .build_call(unbox_fn, &[res_ptr.into()], "thread_join_unboxed")
                        .map_err(|e| CodegenError::JitInit(format!(
                            "build_call ai_gc_unbox_int (thread.join): {}", e
                        )))?;
                    Ok(Value::Int(unboxed.as_any_value_enum().into_int_value()))
                } else {
                    Ok(Value::Closure(res_ptr))
                }
            }
            // Structural hash / equality of ANY value. Args are boxed to a
            // uniform pointer (an Int key is boxed; a pointer key passes
            // through) so the runtime walker always gets a heap object.
            Expr::BuiltinRef(name) if name == "core/hash.value" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let v = self.compile_expr(&args[0], env, ctx)?;
                let p = self.value_as_ptr(v, ctx)?;
                let f = self.extern_value_hash.expect("ai_value_hash declared");
                let call = self
                    .builder
                    .build_call(f, &[ctx.thread_param.into(), p.into()], "value_hash")
                    .map_err(|e| CodegenError::JitInit(format!(
                        "build_call ai_value_hash: {}", e
                    )))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/value.eq" => {
                if args.len() != 2 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                // Both operands coerce to a uniform heap pointer (Int → box),
                // and must be rooted across each other's evaluation.
                let vals = self.compile_operands_rooted(args, &[true, true], env, ctx)?;
                let ap = vals[0].as_closure()?;
                let bp = vals[1].as_closure()?;
                let f = self.extern_value_eq.expect("ai_value_eq declared");
                let call = self
                    .builder
                    .build_call(f, &[ctx.thread_param.into(), ap.into(), bp.into()], "value_eq")
                    .map_err(|e| CodegenError::JitInit(format!(
                        "build_call ai_value_eq: {}", e
                    )))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/gc.collect" => {
                // No language-level args — but the runtime fn needs
                // `thread` so it can publish parked_jit_fp + reach the
                // heap.
                let fc = self.extern_force_collect.expect("ai_gc_force_collect declared");
                let call = self
                    .builder
                    .build_call(fc, &[ctx.thread_param.into()], "gc_collect_result")
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_gc_force_collect: {}", e),
                    ))?;
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name == "core/abort" => {
                // `abort(msg)` — a reached BUG, not an error value:
                // compile the message (a heap String), call ai_abort
                // (which never returns), and terminate this block with
                // `unreachable`. Errors a program models are `Result`
                // values; this is for impossible states only. The value
                // returned to the compiler is never used; callers that
                // branch out of this block (e.g. a match arm) detect the
                // existing terminator and skip the merge branch/phi.
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.clone(),
                        arity: args.len(),
                    });
                }
                let msg = self.compile_expr(&args[0], env, ctx)?;
                let mp = msg.as_closure().map_err(|_| CodegenError::TypeMismatch {
                    what: "core/abort: arg must be a String".to_owned(),
                })?;
                let abort_fn = self.extern_abort.expect("ai_abort declared");
                self.builder
                    .build_call(abort_fn, &[ctx.thread_param.into(), mp.into()], "")
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_call ai_abort: {}", e),
                    ))?;
                self.builder
                    .build_unreachable()
                    .map_err(|e| CodegenError::JitInit(
                        format!("build_unreachable (abort): {}", e),
                    ))?;
                Ok(Value::Int(self.i64_ty.const_zero()))
            }
            Expr::BuiltinRef(name)
                if name.starts_with("ext/")
                    && self
                        .user_externs
                        .get(&name["ext/".len()..])
                        .map(|s| s.library.is_some())
                        .unwrap_or(false) =>
            {
                // C-FFI extern call. The LLVM extern was declared with
                // the plain C ABI (no leading `Thread*`); every arg and
                // the return is an i64-width C scalar (Int or Ptr =
                // address). Compile each arg to an i64 and call directly.
                let ext_name = &name["ext/".len()..];
                let symbol = user_extern_symbol(ext_name);
                let fv = self
                    .module
                    .get_function(&symbol)
                    .ok_or_else(|| CodegenError::UnknownBuiltin {
                        name: format!(
                            "C extern `{}` was called but not declared in module",
                            ext_name
                        ),
                        arity: args.len(),
                    })?;
                let mut call_args: Vec<BasicMetadataValueEnum<'ctx>> =
                    Vec::with_capacity(args.len());
                for (i, a) in args.iter().enumerate() {
                    let v = self.compile_expr(a, env, ctx)?;
                    let iv = v.as_int().map_err(|_| CodegenError::TypeMismatch {
                        what: format!(
                            "C extern `{}` arg {} must be an Int or Ptr (i64)",
                            ext_name, i
                        ),
                    })?;
                    call_args.push(iv.into());
                }
                let call = self
                    .builder
                    .build_call(fv, &call_args, "cffi_call")
                    .map_err(|e| CodegenError::JitInit(format!(
                        "build_call C extern {}: {}",
                        ext_name, e
                    )))?;
                // The return is always declared i64 (Int or Ptr address);
                // void C fns leave a junk value we simply ignore.
                Ok(Value::Int(call.as_any_value_enum().into_int_value()))
            }
            Expr::BuiltinRef(name) if name.starts_with("ext/") => {
                // User-defined extern call. The LLVM extern was
                // declared at module-build time (see
                // `declare_user_externs`) with per-arg/ret types
                // matching the registered signature (Int → i64,
                // String → ptr). The JIT-init mapping wired its
                // address. Dispatch each arg based on the LLVM
                // param type we see here.
                let ext_name = &name["ext/".len()..];
                let symbol = user_extern_symbol(ext_name);
                let fv = self
                    .module
                    .get_function(&symbol)
                    .ok_or_else(|| CodegenError::UnknownBuiltin {
                        name: format!(
                            "extern `{}` was called but not declared in module",
                            ext_name
                        ),
                        arity: args.len(),
                    })?;
                let fn_ty = fv.get_type();
                let llvm_param_tys = fn_ty.get_param_types();
                // llvm_param_tys[0] is the thread ptr; user args start at [1].
                // Root each pointer arg (e.g. String) across later args.
                let no_box = vec![false; args.len()];
                let arg_vals = self.compile_operands_rooted(args, &no_box, env, ctx)?;
                let mut call_args: Vec<BasicMetadataValueEnum<'ctx>> =
                    Vec::with_capacity(args.len() + 1);
                call_args.push(ctx.thread_param.into());
                for (i, v) in arg_vals.into_iter().enumerate() {
                    let expected = llvm_param_tys.get(i + 1).ok_or_else(|| {
                        CodegenError::UnknownBuiltin {
                            name: format!("extern `{}` arity mismatch", ext_name),
                            arity: args.len(),
                        }
                    })?;
                    let arg_meta: BasicMetadataValueEnum<'ctx> = match expected {
                        inkwell::types::BasicMetadataTypeEnum::IntType(_) => v
                            .as_int()
                            .map_err(|_| CodegenError::TypeMismatch {
                                what: format!(
                                    "extern `{}` arg {} expects Int",
                                    ext_name, i
                                ),
                            })?
                            .into(),
                        inkwell::types::BasicMetadataTypeEnum::PointerType(_) => v
                            .as_closure()
                            .map_err(|_| CodegenError::TypeMismatch {
                                what: format!(
                                    "extern `{}` arg {} expects pointer (e.g. String)",
                                    ext_name, i
                                ),
                            })?
                            .into(),
                        other => {
                            return Err(CodegenError::Unsupported {
                                what: format!(
                                    "extern `{}` arg {} unexpected LLVM type {:?}",
                                    ext_name, i, other
                                ),
                            });
                        }
                    };
                    call_args.push(arg_meta);
                }
                let call = self
                    .builder
                    .build_call(fv, &call_args, "ext_call")
                    .map_err(|e| CodegenError::JitInit(format!(
                        "build_call ext/{}: {}",
                        ext_name, e
                    )))?;
                let ret_ty = fn_ty.get_return_type();
                if matches!(
                    ret_ty,
                    Some(inkwell::types::BasicTypeEnum::PointerType(_))
                ) {
                    Ok(Value::Closure(
                        call.as_any_value_enum().into_pointer_value(),
                    ))
                } else {
                    Ok(Value::Int(call.as_any_value_enum().into_int_value()))
                }
            }
            Expr::BuiltinRef(name) => {
                let arg_vs = args
                    .iter()
                    .map(|a| self.compile_expr(a, env, ctx))
                    .collect::<Result<Vec<_>, _>>()?;
                let int_args: Result<Vec<IntValue<'ctx>>, _> =
                    arg_vs.iter().map(|v| v.as_int()).collect();
                let result = self.compile_builtin(name, &int_args?)?;
                Ok(Value::Int(result))
            }
            Expr::TopRef(h) => {
                let target = *self
                    .functions
                    .get(h)
                    .ok_or(CodegenError::UnknownTopRef { hash: *h })?;

                // Look up the callee's declared signature so we can
                // tell which parameters are generic (TypeVar). Box
                // any `Int` argument flowing into a TypeVar slot, and
                // unbox the return value if the declared return is a
                // TypeVar but the caller's instantiation makes it Int.
                let sig = self.def_signatures.get(h).cloned();
                let (param_decls, ret_decl) = match &sig {
                    Some(s) => (s.params.clone(), s.ret.clone()),
                    None => (Vec::new(), Type::Builtin("Int".to_owned())),
                };

                // Unify declared params against actual arg types to
                // recover the instantiation (if generic).
                let n_vars =
                    max_typevar_in_types(&param_decls, &ret_decl);
                let mut subst: Vec<Option<Type>> = vec![None; n_vars as usize];
                if n_vars > 0 {
                    for (decl, arg_expr) in param_decls.iter().zip(args.iter()) {
                        let actual = self.infer_type(arg_expr, env);
                        let _ = unify_for_codegen(decl, &actual, &mut subst);
                    }
                }
                let concrete_subst: Vec<Type> = subst
                    .into_iter()
                    .enumerate()
                    .map(|(i, o)| o.unwrap_or(Type::TypeVar(i as u32)))
                    .collect();

                // Box an Int arg flowing into a generic (TypeVar) param.
                let box_flags: Vec<bool> = args
                    .iter()
                    .enumerate()
                    .map(|(i, _)| matches!(param_decls.get(i), Some(Type::TypeVar(_))))
                    .collect();
                // Root each pointer arg across the evaluation of later args
                // (which may allocate and relocate it).
                let arg_vals = self.compile_operands_rooted(args, &box_flags, env, ctx)?;
                let mut call_args: Vec<BasicMetadataValueEnum> = Vec::with_capacity(args.len() + 1);
                call_args.push(ctx.thread_param.into());
                for v in arg_vals {
                    call_args.push(v.into_basic().into());
                }
                let call = self
                    .builder
                    .build_call(target, &call_args, "calltmp")
                    .map_err(|e| CodegenError::JitInit(format!("build_call: {}", e)))?;
                // Errors are values: if the callee raised a panic, its
                // result is a dummy — propagate the panic up before
                // touching (e.g. unboxing) the return value.
                let any = call.as_any_value_enum();
                // Inspect declared return type of the callee to decide.
                let ret_ty = target.get_type().get_return_type();
                let return_value = if matches!(
                    ret_ty,
                    Some(inkwell::types::BasicTypeEnum::PointerType(_))
                ) {
                    Value::Closure(any.into_pointer_value())
                } else {
                    Value::Int(any.into_int_value())
                };
                // If the declared return is a TypeVar that the call's
                // instantiation pins to Int, unbox the returned
                // BoxedInt pointer to recover the raw i64.
                if matches!(ret_decl, Type::TypeVar(_)) {
                    let inst_ret = substitute_type(&ret_decl, &concrete_subst);
                    if matches!(&inst_ret, Type::Builtin(n) if is_boxed_scalar(n)) {
                        if let Value::Closure(ptr) = return_value {
                            let unbox_fn = self
                                .extern_unbox_int
                                .expect("ai_gc_unbox_int declared");
                            let call = self
                                .builder
                                .build_call(
                                    unbox_fn,
                                    &[ptr.into()],
                                    "unbox_int_ret",
                                )
                                .map_err(|e| CodegenError::JitInit(format!(
                                    "build_call ai_gc_unbox_int: {}",
                                    e
                                )))?;
                            return Ok(Value::Int(
                                call.as_any_value_enum().into_int_value(),
                            ));
                        }
                    }
                }
                Ok(return_value)
            }
            Expr::SelfRef(_) => Err(CodegenError::Unsupported {
                what: "SelfRef (mutually recursive component) call".to_owned(),
            }),
            other => {
                // Indirect call: callee should evaluate to a closure pointer.
                // Capture the callee's declared type BEFORE compiling so we
                // know which (un)boxing the boxed closure ABI needs.
                let callee_ty = self.infer_type(other, env);
                let cv = self.compile_expr(other, env, ctx)?;
                let closure_ptr = cv.as_closure()?;
                self.compile_indirect_call(closure_ptr, &callee_ty, args, env, ctx)
            }
        }
    }

    fn compile_indirect_call(
        &mut self,
        closure_ptr: PointerValue<'ctx>,
        callee_ty: &Type,
        args: &[Expr],
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // Uniform closure ABI: every lifted lambda has LLVM signature
        // (ptr_thread, ptr_closure, ptr × N) -> ptr. Box Int args before
        // passing; unbox Int return based on the closure's declared
        // FnType. Pointer-typed args/rets (Apply, TypeVar, FnType,
        // String, TypeRef) pass through.
        let (decl_param_tys, decl_ret_ty) = match callee_ty {
            Type::FnType { params, ret } => (params.clone(), (**ret).clone()),
            _ => {
                // Unknown callee type; assume Int args + Int return
                // (legacy v1 behavior — matches what the existing
                // lambdas-take-only-Int tests still exercise).
                (
                    std::iter::repeat_n(Type::Builtin("Int".to_owned()), args.len()).collect(),
                    Type::Builtin("Int".to_owned()),
                )
            }
        };

        // If the closure is generic (a `TypeVar` in its declared type),
        // recover this call's instantiation by unifying the declared
        // params against the actual arg types — exactly as the direct
        // `TopRef` call path does. This lets us unbox a `TypeVar` return
        // that THIS call site pins to a boxed scalar (e.g. `array_get`
        // passed as a value, then called on an `Array<Int>`), which the
        // closure's own generic declared return can't tell us.
        let n_vars = max_typevar_in_types(&decl_param_tys, &decl_ret_ty);
        let mut subst: Vec<Option<Type>> = vec![None; n_vars as usize];
        if n_vars > 0 {
            for (decl, arg_expr) in decl_param_tys.iter().zip(args.iter()) {
                let actual = self.infer_type(arg_expr, env);
                let _ = unify_for_codegen(decl, &actual, &mut subst);
            }
        }
        let concrete_subst: Vec<Type> = subst
            .into_iter()
            .enumerate()
            .map(|(i, o)| o.unwrap_or(Type::TypeVar(i as u32)))
            .collect();

        // The closure pointer is held across the argument evaluation, which
        // allocates (boxing + arg sub-calls) — root it in slot `base` so a
        // collection there relocates it; reload before the call. Args are
        // rooted across each other by `compile_operands_rooted`. The uniform
        // closure ABI boxes EVERY Int arg (both `Int`-declared and
        // generic/TypeVar params), so all box-flags are true.
        let base = ctx.next_root_slot;
        let closure_slot = self.write_root_slot(ctx.frame_alloca, ctx.info, base, closure_ptr)?;
        let arg_ctx = CompileCtx {
            next_root_slot: base + 1,
            ..ctx
        };
        let box_flags: Vec<bool> = vec![true; args.len()];
        let arg_vals = self.compile_operands_rooted(args, &box_flags, env, arg_ctx)?;
        // Reload the (possibly relocated) closure pointer.
        let closure_ptr = self.read_root_slot(closure_slot)?;

        // code_ptr = ai_gc_lookup_code(thread, closure_ptr)
        //
        // Pass the closure pointer; `ai_gc_lookup_code` reads the
        // type_id from the closure's GC header (fixed offset) and
        // resolves the code_hash through the code-table's type_id
        // map. The code_hash itself sits at a variable offset that
        // depends on the closure's pointer-capture count, which we
        // can't compute statically here. (No allocation happens between
        // the reload and the indirect call, so `closure_ptr` stays valid.)
        let lookup = self.extern_lookup_code.expect("lookup declared");
        let call = self
            .builder
            .build_call(
                lookup,
                &[ctx.thread_param.into(), closure_ptr.into()],
                "code_ptr",
            )
            .map_err(|e| CodegenError::JitInit(format!("build_call lookup: {}", e)))?;
        let code_ptr = call.as_any_value_enum().into_pointer_value();

        // Build call args: thread, closure, then each arg. Every arg is
        // already a heap pointer (Ints were boxed by the helper).
        let mut call_args: Vec<BasicMetadataValueEnum> = Vec::with_capacity(args.len() + 2);
        call_args.push(ctx.thread_param.into());
        call_args.push(closure_ptr.into());
        for v in arg_vals {
            let arg_ptr = v.as_closure().map_err(|_| CodegenError::TypeMismatch {
                what: "indirect call arg should be a heap pointer after boxing".to_owned(),
            })?;
            call_args.push(arg_ptr.into());
        }

        // Indirect call type: (ptr, ptr, ptr × N) -> ptr
        let mut param_tys: Vec<BasicMetadataTypeEnum> = Vec::with_capacity(args.len() + 2);
        param_tys.push(self.ptr_ty.into());
        param_tys.push(self.ptr_ty.into());
        for _ in args {
            param_tys.push(self.ptr_ty.into());
        }
        let fn_ty = self.ptr_ty.fn_type(&param_tys, false);

        let icall = self
            .builder
            .build_indirect_call(fn_ty, code_ptr, &call_args, "indirect_call")
            .map_err(|e| CodegenError::JitInit(format!("build_indirect_call: {}", e)))?;
        // Propagate a callee panic before unboxing the dummy result.
        let ret_ptr = icall.as_any_value_enum().into_pointer_value();

        // Unbox return if the closure's declared ret is Int, OR if it's a
        // TypeVar that this call's instantiation pins to a boxed scalar
        // (Int/Float). The latter handles a generic function/builtin value
        // (`fn(Array<T>,Int)->T`) invoked at a site that makes T concrete.
        let ret_is_unboxable = is_int_type(&decl_ret_ty)
            || (matches!(decl_ret_ty, Type::TypeVar(_))
                && matches!(
                    substitute_type(&decl_ret_ty, &concrete_subst),
                    Type::Builtin(ref n) if is_boxed_scalar(n)
                ));
        if ret_is_unboxable {
            let unbox_fn = self.extern_unbox_int.expect("ai_gc_unbox_int declared");
            let call = self
                .builder
                .build_call(
                    unbox_fn,
                    &[ret_ptr.into()],
                    "unbox_indirect_ret",
                )
                .map_err(|e| CodegenError::JitInit(format!(
                    "build_call ai_gc_unbox_int (indirect ret): {}", e
                )))?;
            Ok(Value::Int(call.as_any_value_enum().into_int_value()))
        } else {
            Ok(Value::Closure(ret_ptr))
        }
    }

    fn compile_lambda_construction(
        &mut self,
        params: &[Type],
        body: &Expr,
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // Recompute hash of this lambda — must match the one registered
        // during pre-scan.
        let lambda_expr = Expr::Lambda {
            params: params.to_vec(),
            body: Box::new(body.clone()),
        };
        let bytes = encode_expr(&lambda_expr);
        let lambda_hash = Hash::of_bytes(&bytes);

        let spec = self
            .lambdas
            .get(&lambda_hash)
            .cloned()
            .ok_or_else(|| CodegenError::JitInit(format!(
                "lambda {} not pre-scanned (bug)",
                lambda_hash
            )))?;
        let ti_global = self.closure_type_info_globals[&lambda_hash];

        // ai_gc_alloc_closure(thread, &type_info)
        let alloc = self.extern_alloc_closure.expect("alloc declared");
        let call = self
            .builder
            .build_call(
                alloc,
                &[
                    ctx.thread_param.into(),
                    ti_global.as_pointer_value().into(),
                ],
                "closure",
            )
            .map_err(|e| CodegenError::JitInit(format!("build_call alloc_closure: {}", e)))?;
        let closure_ptr = call.as_any_value_enum().into_pointer_value();

        // Pointer captures live in `value_field` slots BEFORE the
        // closure-header (code_hash, n_captures, pad). Their count
        // shifts the absolute offsets of everything that follows.
        // Pull the per-capture metadata so we know each slot's
        // exact offset.
        let cap_layouts: Vec<CaptureMeta> = match self.shape_meta.get(&lambda_hash) {
            Some(ShapeMeta::Closure { captures, .. }) => captures.clone(),
            _ => Vec::new(),
        };
        let ptr_count: u64 = cap_layouts.iter().filter(|c| c.is_pointer).count() as u64;
        let header_base: u64 = crate::gc::Full::SIZE as u64 + ptr_count * 8;

        // Store code_hash starting at `header_base`.
        let hash_const = self.emit_hash_constant(&lambda_hash);
        let dest_hash = unsafe {
            self.builder
                .build_in_bounds_gep(
                    self.context.i8_type(),
                    closure_ptr,
                    &[self.i64_ty.const_int(header_base, false)],
                    "dst_hash",
                )
                .map_err(|e| CodegenError::JitInit(format!("gep dst_hash: {}", e)))?
        };
        self.builder
            .build_memcpy(
                dest_hash,
                1,
                hash_const.as_pointer_value(),
                1,
                self.i64_ty.const_int(32, false),
            )
            .map_err(|e| CodegenError::JitInit(format!("build_memcpy hash: {}", e)))?;

        // Store n_captures (u32) at `header_base + 32`.
        let n_caps_off = self
            .i64_ty
            .const_int(header_base + closure_offsets::N_CAPTURES as u64, false);
        let n_caps_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(
                    self.context.i8_type(),
                    closure_ptr,
                    &[n_caps_off],
                    "n_caps_ptr",
                )
                .map_err(|e| CodegenError::JitInit(format!("gep n_caps: {}", e)))?
        };
        let n_caps = self
            .context
            .i32_type()
            .const_int(spec.captures.len() as u64, false);
        self.builder
            .build_store(n_caps_ptr, n_caps)
            .map_err(|e| CodegenError::JitInit(format!("store n_caps: {}", e)))?;

        // Store each capture at its CaptureMeta.offset. Pointer
        // captures land in value_field slots (GC-traced); Int
        // captures land in raw-byte slots after the closure header.
        for (i, outer_idx) in spec.captures.iter().enumerate() {
            // Resolve outer_idx in the CURRENT env (outer_idx counts
            // out from the innermost binder OUTSIDE the lambda).
            let env_pos = (env.len() as i64) - 1 - (*outer_idx as i64);
            if env_pos < 0 || (env_pos as usize) >= env.len() {
                return Err(CodegenError::JitInit(format!(
                    "lambda capture outer_idx {} out of range (env depth {})",
                    outer_idx,
                    env.len()
                )));
            }
            let val = env.get(env_pos as usize).read(self)?;
            let cap = cap_layouts
                .get(i)
                .copied()
                .unwrap_or(CaptureMeta { offset: 0, is_pointer: false });
            let off = self.i64_ty.const_int(cap.offset as u64, false);
            let slot = unsafe {
                self.builder
                    .build_in_bounds_gep(self.context.i8_type(), closure_ptr, &[off], "cap_slot")
                    .map_err(|e| CodegenError::JitInit(format!("gep cap_slot: {}", e)))?
            };
            if cap.is_pointer {
                // Store an 8-byte heap pointer.
                let p = match val {
                    Value::Closure(p) => p,
                    Value::Int(iv) => {
                        // Source value is Int but the slot was
                        // classified as pointer-only — box the i64
                        // into a BoxedInt so the slot still holds a
                        // real heap pointer (the body will unbox).
                        let box_fn = self.extern_box_int.expect("ai_gc_box_int declared");
                        let call = self
                            .builder
                            .build_call(
                                box_fn,
                                &[ctx.thread_param.into(), iv.into()],
                                "box_cap_int",
                            )
                            .map_err(|e| {
                                CodegenError::JitInit(format!(
                                    "build_call ai_gc_box_int (cap): {}", e
                                ))
                            })?;
                        call.as_any_value_enum().into_pointer_value()
                    }
                };
                self.builder
                    .build_store(slot, p)
                    .map_err(|e| CodegenError::JitInit(format!("store ptr cap: {}", e)))?;
            } else {
                // Store a raw i64.
                let int_val = val.as_int().map_err(|_| CodegenError::TypeMismatch {
                    what: "capture classified as Int but env holds a pointer".to_owned(),
                })?;
                self.builder
                    .build_store(slot, int_val)
                    .map_err(|e| CodegenError::JitInit(format!("store int cap: {}", e)))?;
            }
        }

        Ok(Value::Closure(closure_ptr))
    }

    fn compile_struct_new(
        &mut self,
        struct_ref: &Hash,
        fields: &[Expr],
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // Field expressions are NOT in tail position.
        let ctx = CompileCtx {
            is_tail: false,
            ..ctx
        };
        let info = self
            .structs
            .get(struct_ref)
            .cloned()
            .ok_or(CodegenError::UnknownTopRef { hash: *struct_ref })?;

        if fields.len() != info.field_offsets.len() {
            return Err(CodegenError::TypeMismatch {
                what: format!(
                    "struct {} expects {} fields, got {}",
                    struct_ref,
                    info.field_offsets.len(),
                    fields.len()
                ),
            });
        }

        // ai_gc_alloc_closure is misnamed but actually generic over heap
        // objects. Reuse for struct allocation.
        let alloc = self.extern_alloc_closure.expect("alloc declared");
        let call = self
            .builder
            .build_call(
                alloc,
                &[
                    ctx.thread_param.into(),
                    info.ti_global.as_pointer_value().into(),
                ],
                "struct_alloc",
            )
            .map_err(|e| CodegenError::JitInit(format!("build_call alloc struct: {}", e)))?;
        let obj_ptr = call.as_any_value_enum().into_pointer_value();

        // The object is allocated FIRST, but evaluating the field values
        // (and boxing Int fields) allocates too — a collection there would
        // relocate `obj_ptr`. Root it in slot `base` so it's relocated,
        // then reload before storing. Field values are rooted across each
        // other by `compile_operands_rooted`, which also boxes Int values
        // bound for pointer-typed (TypeVar/Apply) field slots. (Field reads
        // and match payload extraction do the symmetric unbox.)
        let base = ctx.next_root_slot;
        let obj_slot = self.write_root_slot(ctx.frame_alloca, ctx.info, base, obj_ptr)?;
        let field_ctx = CompileCtx {
            next_root_slot: base + 1,
            ..ctx
        };
        let vals = self.compile_operands_rooted(fields, &info.field_is_pointer, env, field_ctx)?;
        // Reload the (possibly relocated) object; no allocation happens
        // between here and the stores, so it stays valid.
        let obj_ptr = self.read_root_slot(obj_slot)?;

        for (i, v) in vals.iter().enumerate() {
            let offset = self.i64_ty.const_int(info.field_offsets[i] as u64, false);
            let slot = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        self.context.i8_type(),
                        obj_ptr,
                        &[offset],
                        &format!("field_{}", i),
                    )
                    .map_err(|e| CodegenError::JitInit(format!("gep field {}: {}", i, e)))?
            };
            let basic = v.into_basic();
            self.builder
                .build_store(slot, basic)
                .map_err(|e| CodegenError::JitInit(format!("store field {}: {}", i, e)))?;
        }

        Ok(Value::Closure(obj_ptr))
    }

    fn compile_field(
        &mut self,
        base: &Expr,
        struct_ref: &Hash,
        index: u32,
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // Base expression is NOT in tail position.
        let ctx = CompileCtx {
            is_tail: false,
            ..ctx
        };
        // Capture the base's inferred type BEFORE compiling so we can
        // pull a `Type::Apply(...)` instantiation off it (same trick
        // as compile_match for variant payload binding). For a
        // ListCell<Int>'s `head` field declared as TypeVar(0), the
        // instantiation gives us [Int], which means the pointer-typed
        // slot actually holds a BoxedInt and we should unbox after
        // loading.
        let base_ty = self.infer_type(base, env);
        let instantiation: Vec<Type> = match &base_ty {
            Type::Apply(_, args) => args.clone(),
            _ => Vec::new(),
        };

        let base_val = self.compile_expr(base, env, ctx)?;
        let base_ptr = base_val.as_closure().map_err(|_| CodegenError::TypeMismatch {
            what: "field access on a non-struct value".to_owned(),
        })?;
        let info = self
            .structs
            .get(struct_ref)
            .cloned()
            .ok_or(CodegenError::UnknownTopRef { hash: *struct_ref })?;

        if (index as usize) >= info.field_offsets.len() {
            return Err(CodegenError::TypeMismatch {
                what: format!(
                    "field index {} out of range (struct has {} fields)",
                    index,
                    info.field_offsets.len()
                ),
            });
        }
        let offset = info.field_offsets[index as usize];
        let is_ptr = info.field_is_pointer[index as usize];

        // If the field's declared type is a TypeVar and the base's
        // instantiation pins it to Int, the slot is a BoxedInt that we
        // must unbox after loading.
        let declared_field_ty = self
            .struct_field_types
            .get(struct_ref)
            .and_then(|fs| fs.get(index as usize).cloned());
        let needs_unbox = match &declared_field_ty {
            Some(t @ Type::TypeVar(_)) if !instantiation.is_empty() => {
                matches!(substitute_type(t, &instantiation),
                    Type::Builtin(ref n) if is_boxed_scalar(n))
            }
            _ => false,
        };

        let slot = unsafe {
            self.builder
                .build_in_bounds_gep(
                    self.context.i8_type(),
                    base_ptr,
                    &[self.i64_ty.const_int(offset as u64, false)],
                    "field_addr",
                )
                .map_err(|e| CodegenError::JitInit(format!("gep field: {}", e)))?
        };

        // Struct fields are IMMUTABLE in the language (there is no field
        // assignment — mutation lives in Array/Bytes/Atom/state), so every
        // field load is invariant: LLVM may CSE/hoist repeated reads of
        // `s.x` across stores it cannot otherwise disambiguate.
        if is_ptr && needs_unbox {
            let boxed_load = self
                .builder
                .build_load(self.ptr_ty, slot, "field_boxed_int_ptr")
                .map_err(|e| CodegenError::JitInit(format!("load boxed int field ptr: {}", e)))?;
            self.mark_invariant_load(boxed_load);
            let boxed_ptr = boxed_load.into_pointer_value();
            let unbox_fn = self
                .extern_unbox_int
                .expect("ai_gc_unbox_int declared");
            let call = self
                .builder
                .build_call(
                    unbox_fn,
                    &[boxed_ptr.into()],
                    "field_unboxed_int",
                )
                .map_err(|e| CodegenError::JitInit(format!("build_call ai_gc_unbox_int (field): {}", e)))?;
            Ok(Value::Int(call.as_any_value_enum().into_int_value()))
        } else if is_ptr {
            let load = self
                .builder
                .build_load(self.ptr_ty, slot, "field_ptr")
                .map_err(|e| CodegenError::JitInit(format!("load field ptr: {}", e)))?;
            self.mark_invariant_load(load);
            Ok(Value::Closure(load.into_pointer_value()))
        } else {
            let load = self
                .builder
                .build_load(self.i64_ty, slot, "field_int")
                .map_err(|e| CodegenError::JitInit(format!("load field int: {}", e)))?;
            self.mark_invariant_load(load);
            Ok(Value::Int(load.into_int_value()))
        }
    }

    fn compile_enum_new(
        &mut self,
        enum_ref: &Hash,
        variant_index: u32,
        payload: Option<&Expr>,
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // Payload sub-expression is NOT in tail position.
        let ctx = CompileCtx {
            is_tail: false,
            ..ctx
        };
        let info = self
            .enums
            .get(enum_ref)
            .cloned()
            .ok_or(CodegenError::UnknownTopRef { hash: *enum_ref })?;
        let v = (variant_index as usize)
            .checked_sub(0)
            .and_then(|i| info.variants.get(i).copied())
            .ok_or_else(|| CodegenError::TypeMismatch {
                what: format!(
                    "enum {} has no variant index {}",
                    enum_ref, variant_index
                ),
            })?;

        // Evaluate the payload BEFORE the alloc, and BOX it (if the variant
        // declares a pointer-typed payload but we have an Int) BEFORE the
        // alloc too — boxing and the variant alloc both allocate, so a
        // collection at the alloc would relocate a payload pointer held only
        // in an SSA register. Spill the final payload pointer to slot `base`
        // so it's relocated, then reload it after the alloc (no allocation
        // happens between the reload and the store).
        let base = ctx.next_root_slot;
        let payload_ctx = CompileCtx {
            next_root_slot: base + 1,
            ..ctx
        };
        // (Some(slot), None) for a spilled pointer payload; (None, Some(v))
        // for a non-pointer (Int) payload passed through; None for no payload.
        let payload_spill: Option<(Option<PointerValue<'ctx>>, Option<Value<'ctx>>)> =
            match payload {
                None => None,
                Some(e) => {
                    let mut pv = self.compile_expr(e, env, payload_ctx)?;
                    if v.payload_is_pointer {
                        if let Value::Int(iv) = pv {
                            let box_fn = self.extern_box_int.expect("ai_gc_box_int declared");
                            let call = self
                                .builder
                                .build_call(
                                    box_fn,
                                    &[ctx.thread_param.into(), iv.into()],
                                    "box_int_for_variant_payload",
                                )
                                .map_err(|e| {
                                    CodegenError::JitInit(format!(
                                        "build_call ai_gc_box_int (enum payload): {}",
                                        e
                                    ))
                                })?;
                            pv = Value::Closure(call.as_any_value_enum().into_pointer_value());
                        }
                    }
                    match pv {
                        Value::Closure(p) => {
                            let slot = self.write_root_slot(ctx.frame_alloca, ctx.info, base, p)?;
                            Some((Some(slot), None))
                        }
                        Value::Int(_) => Some((None, Some(pv))),
                    }
                }
            };

        // Allocate the variant's heap object.
        let alloc = self.extern_alloc_closure.expect("alloc declared");
        let call = self
            .builder
            .build_call(
                alloc,
                &[
                    ctx.thread_param.into(),
                    v.ti_global.as_pointer_value().into(),
                ],
                "enum_alloc",
            )
            .map_err(|e| CodegenError::JitInit(format!("build_call alloc enum: {}", e)))?;
        let obj_ptr = call.as_any_value_enum().into_pointer_value();

        // Store the tag.
        let tag_off = self.i64_ty.const_int(v.tag_offset as u64, false);
        let tag_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(self.context.i8_type(), obj_ptr, &[tag_off], "tag_ptr")
                .map_err(|e| CodegenError::JitInit(format!("gep tag: {}", e)))?
        };
        let tag_val = self.context.i32_type().const_int(variant_index as u64, false);
        self.builder
            .build_store(tag_ptr, tag_val)
            .map_err(|e| CodegenError::JitInit(format!("store tag: {}", e)))?;

        // Store the payload, if any. Reload a spilled pointer payload from
        // its root slot (the alloc above may have relocated it).
        if let (Some((slot, nonptr)), Some(off)) = (payload_spill, v.payload_offset) {
            let off_const = self.i64_ty.const_int(off as u64, false);
            let payload_slot = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        self.context.i8_type(),
                        obj_ptr,
                        &[off_const],
                        "payload_ptr",
                    )
                    .map_err(|e| CodegenError::JitInit(format!("gep payload: {}", e)))?
            };
            let to_store = match (slot, nonptr) {
                (Some(slot), _) => Value::Closure(self.read_root_slot(slot)?),
                (None, Some(v)) => v,
                (None, None) => unreachable!("payload_spill has neither pointer slot nor value"),
            };
            let basic = to_store.into_basic();
            self.builder
                .build_store(payload_slot, basic)
                .map_err(|e| CodegenError::JitInit(format!("store payload: {}", e)))?;
        }

        Ok(Value::Closure(obj_ptr))
    }

    /// Tag a load with `!invariant.load` — the loaded location is
    /// immutable for the lifetime of the object (array headers, count
    /// words, the runtime's TypeInfo). Lets LLVM CSE/hoist the shape +
    /// bounds checks across repeated accesses to the same array.
    fn mark_invariant_load(&self, inst: inkwell::values::BasicValueEnum<'ctx>) {
        if let Some(iv) = inst.as_instruction_value() {
            let kind = self.context.get_kind_id("invariant.load");
            let node = self.context.metadata_node(&[]);
            let _ = iv.set_metadata(node, kind);
        }
    }

    /// Inline `array_len`: branch-free except for the null guard. Reads
    /// the header type_id and the count word (both `!invariant.load` —
    /// immutable for the object's lifetime) and selects slots vs
    /// bytes>>3 by shape. No runtime call.
    fn emit_array_len_inline(
        &mut self,
        a: PointerValue<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<IntValue<'ctx>, CodegenError> {
        let j = |e: String| CodegenError::JitInit(e);
        let i8_ty = self.context.i8_type();
        let i16_ty = self.context.i16_type();
        let header_size = <crate::gc::Full as crate::gc::ObjHeader>::SIZE as u64;
        let tid_off = <crate::gc::Full as crate::gc::ObjHeader>::TYPE_ID_OFFSET as u64;
        let ti_tid_off = core::mem::offset_of!(crate::gc::TypeInfo, type_id) as u64;

        let cur_fn = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .expect("array len inside a fn");
        let read_bb = self.context.append_basic_block(cur_fn, "len_read");
        let merge_bb = self.context.append_basic_block(cur_fn, "len_merge");
        let entry_end = self.builder.get_insert_block().unwrap();

        let nonnull = self
            .builder
            .build_is_not_null(a, "len_nonnull")
            .map_err(|e| j(format!("len nonnull: {}", e)))?;
        self.builder
            .build_conditional_branch(nonnull, read_bb, merge_bb)
            .map_err(|e| j(format!("br len nonnull: {}", e)))?;

        self.builder.position_at_end(read_bb);
        let tid_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(
                    i8_ty,
                    a,
                    &[self.i64_ty.const_int(tid_off, false)],
                    "len_tid_ptr",
                )
                .map_err(|e| j(format!("gep len tid: {}", e)))?
        };
        let tid_load = self
            .builder
            .build_load(i16_ty, tid_ptr, "len_tid")
            .map_err(|e| j(format!("load len tid: {}", e)))?;
        self.mark_invariant_load(tid_load);
        let tid = tid_load.into_int_value();
        let prim_ti_slot = self.thread_field(
            ctx.thread_param,
            thread_offsets::PRIM_ARRAY_TI,
            "len_prim_ti_slot",
        )?;
        let prim_ti_load = self
            .builder
            .build_load(self.ptr_ty, prim_ti_slot, "len_prim_ti")
            .map_err(|e| j(format!("load len prim_ti: {}", e)))?;
        self.mark_invariant_load(prim_ti_load);
        let prim_ti = prim_ti_load.into_pointer_value();
        let prim_tid_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(
                    i8_ty,
                    prim_ti,
                    &[self.i64_ty.const_int(ti_tid_off, false)],
                    "len_prim_tid_ptr",
                )
                .map_err(|e| j(format!("gep len prim tid: {}", e)))?
        };
        let prim_tid_load = self
            .builder
            .build_load(i16_ty, prim_tid_ptr, "len_prim_tid")
            .map_err(|e| j(format!("load len prim tid: {}", e)))?;
        self.mark_invariant_load(prim_tid_load);
        let prim_tid = prim_tid_load.into_int_value();
        let is_prim = self
            .builder
            .build_int_compare(IntPredicate::EQ, tid, prim_tid, "len_is_prim")
            .map_err(|e| j(format!("icmp len prim: {}", e)))?;
        let count_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(
                    i8_ty,
                    a,
                    &[self.i64_ty.const_int(header_size, false)],
                    "len_count_ptr",
                )
                .map_err(|e| j(format!("gep len count: {}", e)))?
        };
        let count_load = self
            .builder
            .build_load(self.i64_ty, count_ptr, "len_count")
            .map_err(|e| j(format!("load len count: {}", e)))?;
        self.mark_invariant_load(count_load);
        let count = count_load.into_int_value();
        let shifted = self
            .builder
            .build_right_shift(count, self.i64_ty.const_int(3, false), false, "len_shr")
            .map_err(|e| j(format!("lshr len: {}", e)))?;
        let picked = self
            .builder
            .build_select(is_prim, shifted, count, "len_val")
            .map_err(|e| j(format!("select len: {}", e)))?
            .into_int_value();
        self.builder
            .build_unconditional_branch(merge_bb)
            .map_err(|e| j(format!("br len→merge: {}", e)))?;
        let read_end = self.builder.get_insert_block().unwrap();

        self.builder.position_at_end(merge_bb);
        let phi = self
            .builder
            .build_phi(self.i64_ty, "len_result")
            .map_err(|e| j(format!("phi len: {}", e)))?;
        let zero = self.i64_ty.const_zero();
        phi.add_incoming(&[(&picked, read_end), (&zero, entry_end)]);
        Ok(phi.as_basic_value().into_int_value())
    }

    /// Emit the INLINE unboxed-array fast path for a scalar-element
    /// `array_get`/`array_set`: null-check, shape check against the
    /// PrimArray type_id (two dependent loads off the thread, both
    /// cache-hot), unsigned bounds check, then a raw 8-byte load/store.
    /// Everything else — boxed arrays, out-of-bounds (which aborts with
    /// a message), null — falls to the out-of-line runtime call, which
    /// re-derives all of it. Turns the ~5ns call per hot array access
    /// into a couple of predictable branches + one memory op.
    ///
    /// `store` = None → get (returns the loaded value); Some(v) → set
    /// (returns the runtime's 0). The slow call's result is merged via a
    /// phi either way.
    fn emit_array_scalar_fastpath(
        &mut self,
        a: PointerValue<'ctx>,
        i: IntValue<'ctx>,
        store: Option<IntValue<'ctx>>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<IntValue<'ctx>, CodegenError> {
        let j = |e: String| CodegenError::JitInit(e);
        let i8_ty = self.context.i8_type();
        let i16_ty = self.context.i16_type();
        let header_size = <crate::gc::Full as crate::gc::ObjHeader>::SIZE as u64;
        let tid_off = <crate::gc::Full as crate::gc::ObjHeader>::TYPE_ID_OFFSET as u64;
        let ti_tid_off =
            core::mem::offset_of!(crate::gc::TypeInfo, type_id) as u64;

        let cur_fn = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .expect("array fastpath inside a fn");
        let tid_bb = self.context.append_basic_block(cur_fn, "arr_tid");
        let bounds_bb = self.context.append_basic_block(cur_fn, "arr_bounds");
        let fast_bb = self.context.append_basic_block(cur_fn, "arr_fast");
        let slow_bb = self.context.append_basic_block(cur_fn, "arr_slow");
        let merge_bb = self.context.append_basic_block(cur_fn, "arr_merge");

        // null → slow (the runtime aborts with the proper message).
        let nonnull = self
            .builder
            .build_is_not_null(a, "arr_nonnull")
            .map_err(|e| j(format!("arr nonnull: {}", e)))?;
        self.builder
            .build_conditional_branch(nonnull, tid_bb, slow_bb)
            .map_err(|e| j(format!("br arr nonnull: {}", e)))?;

        // Shape check: header type_id == (*thread.prim_array_ti).type_id.
        self.builder.position_at_end(tid_bb);
        let tid_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(
                    i8_ty,
                    a,
                    &[self.i64_ty.const_int(tid_off, false)],
                    "arr_tid_ptr",
                )
                .map_err(|e| j(format!("gep arr tid: {}", e)))?
        };
        let tid_load = self
            .builder
            .build_load(i16_ty, tid_ptr, "arr_tid")
            .map_err(|e| j(format!("load arr tid: {}", e)))?;
        self.mark_invariant_load(tid_load);
        let tid = tid_load.into_int_value();
        let prim_ti_slot = self.thread_field(
            ctx.thread_param,
            thread_offsets::PRIM_ARRAY_TI,
            "prim_ti_slot",
        )?;
        let prim_ti_load = self
            .builder
            .build_load(self.ptr_ty, prim_ti_slot, "prim_ti")
            .map_err(|e| j(format!("load prim_ti: {}", e)))?;
        self.mark_invariant_load(prim_ti_load);
        let prim_ti = prim_ti_load.into_pointer_value();
        let prim_tid_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(
                    i8_ty,
                    prim_ti,
                    &[self.i64_ty.const_int(ti_tid_off, false)],
                    "prim_tid_ptr",
                )
                .map_err(|e| j(format!("gep prim tid: {}", e)))?
        };
        let prim_tid_load = self
            .builder
            .build_load(i16_ty, prim_tid_ptr, "prim_tid")
            .map_err(|e| j(format!("load prim tid: {}", e)))?;
        self.mark_invariant_load(prim_tid_load);
        let prim_tid = prim_tid_load.into_int_value();
        let is_prim = self
            .builder
            .build_int_compare(IntPredicate::EQ, tid, prim_tid, "arr_is_prim")
            .map_err(|e| j(format!("icmp arr prim: {}", e)))?;
        self.builder
            .build_conditional_branch(is_prim, bounds_bb, slow_bb)
            .map_err(|e| j(format!("br arr prim: {}", e)))?;

        // Bounds: unsigned i < (count_bytes >> 3). Negative i wraps to a
        // huge unsigned and fails into the slow path's abort.
        self.builder.position_at_end(bounds_bb);
        let count_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(
                    i8_ty,
                    a,
                    &[self.i64_ty.const_int(header_size, false)],
                    "arr_count_ptr",
                )
                .map_err(|e| j(format!("gep arr count: {}", e)))?
        };
        let count_load = self
            .builder
            .build_load(self.i64_ty, count_ptr, "arr_count_bytes")
            .map_err(|e| j(format!("load arr count: {}", e)))?;
        self.mark_invariant_load(count_load);
        let count = count_load.into_int_value();
        let len = self
            .builder
            .build_right_shift(
                count,
                self.i64_ty.const_int(3, false),
                false,
                "arr_len",
            )
            .map_err(|e| j(format!("lshr arr len: {}", e)))?;
        let in_bounds = self
            .builder
            .build_int_compare(IntPredicate::ULT, i, len, "arr_in_bounds")
            .map_err(|e| j(format!("icmp arr bounds: {}", e)))?;
        self.builder
            .build_conditional_branch(in_bounds, fast_bb, slow_bb)
            .map_err(|e| j(format!("br arr bounds: {}", e)))?;

        // Fast: raw 8-byte slot at header + 8 (count word) + i*8.
        self.builder.position_at_end(fast_bb);
        let scaled = self
            .builder
            .build_int_mul(i, self.i64_ty.const_int(8, false), "arr_i8")
            .map_err(|e| j(format!("mul arr idx: {}", e)))?;
        let base_off = self
            .builder
            .build_int_add(
                scaled,
                self.i64_ty.const_int(header_size + 8, false),
                "arr_off",
            )
            .map_err(|e| j(format!("add arr off: {}", e)))?;
        let slot = unsafe {
            self.builder
                .build_in_bounds_gep(i8_ty, a, &[base_off], "arr_slot")
                .map_err(|e| j(format!("gep arr slot: {}", e)))?
        };
        let fast_val = match store {
            None => self
                .builder
                .build_load(self.i64_ty, slot, "arr_fast_val")
                .map_err(|e| j(format!("load arr slot: {}", e)))?
                .into_int_value(),
            Some(v) => {
                self.builder
                    .build_store(slot, v)
                    .map_err(|e| j(format!("store arr slot: {}", e)))?;
                self.i64_ty.const_zero()
            }
        };
        self.builder
            .build_unconditional_branch(merge_bb)
            .map_err(|e| j(format!("br arr fast→merge: {}", e)))?;
        let fast_end = self.builder.get_insert_block().unwrap();

        // Slow: the runtime fn (boxed arrays; aborts on bounds/null).
        self.builder.position_at_end(slow_bb);
        let slow_val = match store {
            None => {
                let fv = self
                    .extern_array_get_i64
                    .expect("ai_array_get_i64 declared");
                self.builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), a.into(), i.into()],
                        "arr_slow_get",
                    )
                    .map_err(|e| j(format!("call ai_array_get_i64: {}", e)))?
                    .as_any_value_enum()
                    .into_int_value()
            }
            Some(v) => {
                let fv = self
                    .extern_array_set_i64
                    .expect("ai_array_set_i64 declared");
                self.builder
                    .build_call(
                        fv,
                        &[ctx.thread_param.into(), a.into(), i.into(), v.into()],
                        "arr_slow_set",
                    )
                    .map_err(|e| j(format!("call ai_array_set_i64: {}", e)))?
                    .as_any_value_enum()
                    .into_int_value()
            }
        };
        self.builder
            .build_unconditional_branch(merge_bb)
            .map_err(|e| j(format!("br arr slow→merge: {}", e)))?;
        let slow_end = self.builder.get_insert_block().unwrap();

        self.builder.position_at_end(merge_bb);
        let phi = self
            .builder
            .build_phi(self.i64_ty, "arr_result")
            .map_err(|e| j(format!("phi arr: {}", e)))?;
        phi.add_incoming(&[(&fast_val, fast_end), (&slow_val, slow_end)]);
        Ok(phi.as_basic_value().into_int_value())
    }

    /// PRECISE sufficient condition for "this body can never hold a GC
    /// pointer": every subexpression is statically scalar (Int/Float/
    /// Bool), every call goes to a def whose signature is all-scalar or
    /// to a scalar arithmetic builtin, and nothing allocates. When true,
    /// compilation provably never spills to a root slot, so the GC frame
    /// is dead weight and the prologue skips it entirely (the Go lesson:
    /// pointer-free frames pay no GC bookkeeping). Conservative `false`
    /// everywhere else — that only costs the frame, never correctness.
    fn scalar_only_body(&self, e: &Expr) -> bool {
        match e {
            Expr::IntLit(_) | Expr::FloatLit(_) | Expr::BoolLit(_) => true,
            // A local is scalar iff its binder was — lets and params are
            // checked at their introduction sites, so a bare LocalVar
            // here is fine.
            Expr::LocalVar(_) => true,
            Expr::Call(callee, args) => {
                let callee_ok = match callee.as_ref() {
                    Expr::TopRef(h) => match self.def_signatures.get(h) {
                        Some(sig) => {
                            sig.params.iter().all(is_scalar_type)
                                && is_scalar_type(&sig.ret)
                        }
                        None => false,
                    },
                    Expr::BuiltinRef(name) => {
                        name.starts_with("core/i64.") || name.starts_with("core/f64.")
                    }
                    _ => false,
                };
                callee_ok && args.iter().all(|a| self.scalar_only_body(a))
            }
            Expr::Let { value, body } => {
                self.scalar_only_body(value) && self.scalar_only_body(body)
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.scalar_only_body(cond)
                    && self.scalar_only_body(then_branch)
                    && self.scalar_only_body(else_branch)
            }
            // Everything else may allocate, bind, or carry a pointer.
            _ => false,
        }
    }

    /// Best-effort recovery of the typechecker's view of an expression's
    /// type, so codegen can decide where to box/unbox at generic
    /// boundaries. Conservative: falls back to `Type::Builtin("Int")`
    /// for any case we don't model yet (which is harmless — we only
    /// consult this for boxing decisions, never for correctness of
    /// non-generic code).
    fn infer_type(&self, expr: &Expr, env: &Env<'ctx>) -> Type {
        match expr {
            Expr::IntLit(_) => Type::Builtin("Int".to_owned()),
            Expr::FloatLit(_) => Type::Builtin("Float".to_owned()),
            Expr::BoolLit(_) => Type::Builtin("Bool".to_owned()),
            Expr::StringLit(_) => Type::Builtin("String".to_owned()),
            Expr::LocalVar(i) => {
                let n = env.len();
                let idx = *i as usize;
                if idx >= n {
                    return Type::Builtin("Int".to_owned());
                }
                env.type_at(n - 1 - idx).clone()
            }
            Expr::StateRef(h) => self
                .state_types
                .get(h)
                .cloned()
                .unwrap_or_else(|| Type::Builtin("Int".to_owned())),
            Expr::TopRef(h) => {
                if let Some(spec) = self.def_signatures.get(h) {
                    Type::FnType {
                        params: spec.params.clone(),
                        ret: Box::new(spec.ret.clone()),
                    }
                } else {
                    Type::TypeRef(*h)
                }
            }
            Expr::BuiltinRef(name) => {
                if let Some((r, fopt)) = crate::resolve::parse_at_builtin_name(name) {
                    // Represent at() polymorphically as
                    //   `fn(Node, fn() -> T) -> Result<T, Failure>`
                    // — TypeVar(0) is T. Call's infer-type unifier
                    // recovers T from the thunk's actual return type
                    // and substitutes it through to the Result's
                    // first type arg, so e.g. `at(node, || pair)`
                    // gives `Apply(Result, [Pair, Failure])`.
                    // at_async wraps the Result in a ThreadHandle.
                    let t_var = Type::TypeVar(0);
                    let mut ret = match fopt {
                        Some(f) => Type::Apply(
                            Box::new(Type::TypeRef(r)),
                            vec![t_var.clone(), Type::TypeRef(f)],
                        ),
                        None => Type::TypeRef(r),
                    };
                    if name.starts_with(crate::resolve::AT_ASYNC_BUILTIN_PREFIX) {
                        ret = Type::Apply(
                            Box::new(Type::Builtin("ThreadHandle".to_owned())),
                            vec![ret],
                        );
                    }
                    Type::FnType {
                        params: vec![
                            Type::TypeRef(Hash([0; 32])),
                            Type::FnType {
                                params: vec![],
                                ret: Box::new(t_var),
                            },
                        ],
                        ret: Box::new(ret),
                    }
                } else if let Some(rest) = name.strip_prefix("core/wire.decode#") {
                    // decode : fn(Bytes) -> Result<okty, Int>. Rebuilt from
                    // the baked `#<expected>#<result_hash>#<okint>` so the
                    // match below knows to unbox an Int Ok payload and the
                    // Int Err payload (the error code).
                    // rest = <expected>#<result_hash>#<okint>#<decode_error_hash>
                    let parts: Vec<&str> = rest.split('#').collect();
                    let hex32 = |s: &str| -> Option<Hash> {
                        if s.len() != 64 {
                            return None;
                        }
                        let mut hb = [0u8; 32];
                        for i in 0..32 {
                            hb[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
                        }
                        Some(Hash(hb))
                    };
                    let parsed = if parts.len() >= 4 {
                        match (hex32(parts[1]), hex32(parts[3])) {
                            (Some(result_h), Some(decode_err_h)) => {
                                let okint = parts[2] == "1";
                                let ok_ty = if okint {
                                    Type::Builtin("Int".to_owned())
                                } else {
                                    Type::Builtin("Ptr".to_owned())
                                };
                                Some(Type::FnType {
                                    params: vec![Type::Builtin("Bytes".to_owned())],
                                    ret: Box::new(Type::Apply(
                                        Box::new(Type::TypeRef(result_h)),
                                        vec![ok_ty, Type::TypeRef(decode_err_h)],
                                    )),
                                })
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    parsed.unwrap_or(Type::Builtin("Int".to_owned()))
                } else if let Some((params, ret)) =
                    crate::typecheck::builtin_signature(name)
                {
                    // Every other known builtin: consult the authoritative
                    // signature table (shared with the typechecker), the
                    // single source of truth. Covers the generic container
                    // builtins (`array.get : fn(Array<T>, Int) -> T`,
                    // `atom`/`thread` ops) AND scalar ones, so a call's
                    // result type is inferred precisely — e.g.
                    // `string_concat(..) -> String`, letting a thunk that
                    // returns it infer `fn() -> String` so `spawn`/`join`
                    // recover `T = String`. Uses the `TypeVar(0)`
                    // convention, so the Call unifier below recovers
                    // generics from the actual argument types.
                    Type::FnType {
                        params,
                        ret: Box::new(ret),
                    }
                } else {
                    // Truly unknown builtins fall back to Int.
                    Type::Builtin("Int".to_owned())
                }
            }
            Expr::Call(callee, args) => {
                let callee_ty = self.infer_type(callee, env);
                let (param_tys, ret_ty) = match &callee_ty {
                    Type::FnType { params, ret } => (params.clone(), (**ret).clone()),
                    _ => return Type::Builtin("Int".to_owned()),
                };
                // If the callee has type-vars, unify against arg types
                // to recover the instantiation.
                let n_vars = max_typevar_in_types(&param_tys, &ret_ty);
                if n_vars > 0 && args.len() == param_tys.len() {
                    let mut subst: Vec<Option<Type>> = vec![None; n_vars as usize];
                    for (declared, arg) in param_tys.iter().zip(args.iter()) {
                        let actual = self.infer_type(arg, env);
                        let _ = unify_for_codegen(declared, &actual, &mut subst);
                    }
                    let concrete: Vec<Type> = subst
                        .into_iter()
                        .enumerate()
                        .map(|(i, o)| o.unwrap_or(Type::TypeVar(i as u32)))
                        .collect();
                    substitute_type(&ret_ty, &concrete)
                } else {
                    ret_ty
                }
            }
            Expr::Lambda { params, body } => {
                // Infer the body's return type for the FnType wrapper.
                // Most callers (let-bound lambdas, etc.) only care about
                // pointer-ness, but `at(node, thunk)` reads through the
                // FnType to recover the thunk's T for the Result<T, E>
                // instantiation — so a hardcoded Int placeholder would
                // strand non-Int returns.
                //
                // Push the lambda's params into a scratch env before
                // inferring the body, so a body whose return type depends
                // on a param (e.g. `|a: Array<Int>, i| array_get_trusted(a, i)`,
                // whose element type comes from `a`) infers correctly.
                // The slot values are never read by `infer_type` (it only
                // consults the recorded TYPES), so undef placeholders are
                // safe.
                let mut body_env = env.clone();
                for p in params {
                    let slot = if is_int_type(p) {
                        EnvSlot::Int(self.i64_ty.get_undef())
                    } else {
                        EnvSlot::Closure(self.ptr_ty.const_null())
                    };
                    body_env.push(slot, p.clone());
                }
                let ret_ty = self.infer_type(body, &body_env);
                Type::FnType {
                    params: params.clone(),
                    ret: Box::new(ret_ty),
                }
            }
            Expr::Let { value, body: _ } => {
                // Body inference would require pushing into env; that's
                // overkill for current callers. Return the value's type
                // as a conservative approximation.
                self.infer_type(value, env)
            }
            Expr::StructNew { struct_ref, .. } => Type::TypeRef(*struct_ref),
            Expr::EnumNew {
                enum_ref,
                variant_index,
                payload,
            } => {
                // Recover the (partial) generic instantiation from the
                // payload, so a locally-constructed `Result::Ok(5)` infers
                // `Apply(Result, [Int, TypeVar(1)])` and a later match
                // unboxes its scalar payload correctly.
                let payload_tys = self.enum_variant_types.get(enum_ref);
                let n_vars = payload_tys
                    .map(|vs| {
                        let all: Vec<Type> =
                            vs.iter().flatten().cloned().collect();
                        max_typevar_in_types(&all, &Type::Builtin("Int".to_owned()))
                    })
                    .unwrap_or(0);
                if n_vars == 0 {
                    Type::TypeRef(*enum_ref)
                } else {
                    let mut subst: Vec<Option<Type>> = vec![None; n_vars as usize];
                    if let (Some(decl), Some(p)) = (
                        payload_tys
                            .and_then(|vs| vs.get(*variant_index as usize))
                            .cloned()
                            .flatten(),
                        payload.as_ref(),
                    ) {
                        let actual = self.infer_type(p, env);
                        let _ = unify_for_codegen(&decl, &actual, &mut subst);
                    }
                    let args: Vec<Type> = subst
                        .into_iter()
                        .enumerate()
                        .map(|(i, o)| o.unwrap_or(Type::TypeVar(i as u32)))
                        .collect();
                    Type::Apply(Box::new(Type::TypeRef(*enum_ref)), args)
                }
            }
            Expr::Field {
                struct_ref, index, ..
            } => {
                if let Some(info) = self.struct_field_types.get(struct_ref) {
                    info.get(*index as usize)
                        .cloned()
                        .unwrap_or(Type::Builtin("Int".to_owned()))
                } else {
                    Type::Builtin("Int".to_owned())
                }
            }
            Expr::Match { arms, .. } => {
                // First-arm body's type approximates the match type.
                if let Some(arm) = arms.first() {
                    self.infer_type(&arm.body, env)
                } else {
                    Type::Builtin("Int".to_owned())
                }
            }
            Expr::If { then_branch, .. } => self.infer_type(then_branch, env),
            Expr::Try {
                expr,
                enum_ref,
                ok_index,
                ..
            } => {
                // `expr?` evaluates to the Ok payload type `T`, with the
                // operand's `Result<T, E>` instantiation substituted.
                let declared = self
                    .enum_variant_types
                    .get(enum_ref)
                    .and_then(|vs| vs.get(*ok_index as usize).cloned().flatten())
                    .unwrap_or(Type::Builtin("Int".to_owned()));
                let instantiation: Vec<Type> = match self.infer_type(expr, env) {
                    Type::Apply(_, args) => args,
                    _ => Vec::new(),
                };
                if instantiation.is_empty() {
                    declared
                } else {
                    substitute_type(&declared, &instantiation)
                }
            }
            // `defer cleanup; body` has the body's type (its value is the
            // body's value; cleanup runs for effect only).
            Expr::Defer { body, .. } => self.infer_type(body, env),
            Expr::SelfRef(_) | Expr::StateSelfRef(_) => Type::Builtin("Int".to_owned()),
            // `StateRef` is handled near the top of this match.
        }
    }

    /// Emit a hard panic with a static message: allocate a heap String
    /// for `msg`, call `ai_panic` (which raises the pending panic), then
    /// terminate the current block with the panic early-return. The
    /// caller must not emit further instructions into this block
    /// afterwards.
    fn emit_panic(
        &mut self,
        msg: &str,
        ctx: CompileCtx<'ctx>,
    ) -> Result<(), CodegenError> {
        let thread_param = ctx.thread_param;
        let bytes = msg.as_bytes();
        let arr_const = self.context.const_string(bytes, false);
        let g = self.module.add_global(
            arr_const.get_type(),
            Some(AddressSpace::default()),
            "panic_msg",
        );
        g.set_linkage(Linkage::Private);
        g.set_constant(true);
        g.set_initializer(&arr_const);
        self.used_globals.push(g.as_pointer_value());

        let str_new = self.extern_str_new.expect("ai_str_new declared");
        let len_v = self.i64_ty.const_int(bytes.len() as u64, false);
        let msg_str = self
            .builder
            .build_call(
                str_new,
                &[
                    thread_param.into(),
                    g.as_pointer_value().into(),
                    len_v.into(),
                ],
                "panic_msg_str",
            )
            .map_err(|e| {
                CodegenError::JitInit(format!("build_call ai_str_new (panic): {}", e))
            })?
            .as_any_value_enum()
            .into_pointer_value();

        let abort_fn = self.extern_abort.expect("ai_abort declared");
        self.builder
            .build_call(abort_fn, &[thread_param.into(), msg_str.into()], "")
            .map_err(|e| CodegenError::JitInit(format!("build_call ai_abort: {}", e)))?;
        self.builder
            .build_unreachable()
            .map_err(|e| CodegenError::JitInit(format!("build_unreachable (abort): {}", e)))?;
        Ok(())
    }

    fn compile_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
    ) -> Result<Value<'ctx>, CodegenError> {
        // Capture the scrutinee's inferred type BEFORE compiling so
        // we can extract any `Type::Apply(...)` instantiation args
        // and substitute them into generic variant payloads. Pattern
        // binders need this to know whether to unbox an extracted
        // payload from a `BoxedInt` to a raw `i64`.
        let scrut_ty = self.infer_type(scrutinee, env);

        // Compile the scrutinee. We expect an enum value (heap pointer)
        // for variant patterns. Var/Wildcard patterns work on Int too,
        // but for v1 we always go through the enum dispatch path —
        // catch-all on a non-enum is supportable but we don't yet need
        // it.
        // Scrutinee is not in tail position; arm bodies inherit ctx.
        let scrut_ctx = CompileCtx {
            is_tail: false,
            ..ctx
        };
        let scrut = self.compile_expr(scrutinee, env, scrut_ctx)?;
        let scrut_ptr = scrut.as_closure().map_err(|_| CodegenError::TypeMismatch {
            what: "match scrutinee must be an enum (heap pointer) in v1".to_owned(),
        })?;

        // Dispatch is a BACKTRACKING CHAIN: each arm gets a "test" block
        // that checks its (possibly nested) pattern against the scrutinee
        // and, on any tag mismatch, falls through to the next arm's test.
        // This handles nested patterns and multiple arms sharing an outer
        // variant (e.g. `Ok(VInt(n))` and `Ok(VStr(_))`), which a single
        // flat switch on the outer tag cannot express.
        let entry_block = self.builder.get_insert_block().unwrap();
        let parent = entry_block.get_parent().unwrap();
        let merge_bb = self.context.append_basic_block(parent, "match_end");
        let fail_bb = self.context.append_basic_block(parent, "match_fail");

        let test_blocks: Vec<inkwell::basic_block::BasicBlock<'ctx>> = (0..arms.len())
            .map(|i| {
                self.context
                    .append_basic_block(parent, &format!("match_test_{}", i))
            })
            .collect();

        // Falling off the end of the chain is a non-exhaustive match — a
        // clear hard error (only reachable if no arm matched).
        self.builder.position_at_end(fail_bb);
        self.emit_panic(
            "non-exhaustive match: value did not match any arm",
            ctx,
        )?;

        // Enter the chain at the first arm's test.
        self.builder.position_at_end(entry_block);
        let first = test_blocks.first().copied().unwrap_or(fail_bb);
        self.builder
            .build_unconditional_branch(first)
            .map_err(|e| CodegenError::JitInit(format!("br match chain: {}", e)))?;

        // Emit each arm: test its pattern, then (on full match) bind its
        // single variable and run the body. Collect (value, end_block) for
        // the result phi.
        let mut incoming: Vec<(BasicValueEnum<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> =
            Vec::with_capacity(arms.len());
        let mut result_ty: Option<inkwell::types::BasicTypeEnum<'ctx>> = None;

        for (i, arm) in arms.iter().enumerate() {
            self.builder.position_at_end(test_blocks[i]);
            let arm_fail = test_blocks.get(i + 1).copied().unwrap_or(fail_bb);
            // Emits the (possibly nested) tag tests; on mismatch branches to
            // `arm_fail`, on full match leaves the builder positioned in the
            // matched block with `pushed` bindings on `env`.
            let pushed = self.emit_pattern_match(
                scrut_ptr,
                &scrut_ty,
                &arm.pattern,
                arm_fail,
                env,
                ctx,
                0,
            )?;

            let arm_ctx = if pushed > 0 {
                CompileCtx {
                    next_root_slot: ctx.next_root_slot + pushed as u32,
                    ..ctx
                }
            } else {
                ctx
            };

            let body_val = self.compile_expr(&arm.body, env, arm_ctx)?;
            for _ in 0..pushed {
                env.pop();
            }
            let end_block = self.builder.get_insert_block().unwrap();
            // If the arm body already terminated its block (e.g. it ended
            // in a `panic(...)` lowering to `unreachable`), it contributes
            // no value or branch to the merge phi.
            if end_block.get_terminator().is_none() {
                let basic = body_val.into_basic();
                if result_ty.is_none() {
                    result_ty = Some(basic.get_type());
                }
                incoming.push((basic, end_block));
                self.builder
                    .build_unconditional_branch(merge_bb)
                    .map_err(|e| CodegenError::JitInit(format!("br merge: {}", e)))?;
            }
        }

        // Build the phi in the merge block.
        self.builder.position_at_end(merge_bb);
        let phi_ty = match result_ty {
            Some(t) => t,
            None => {
                // Every arm terminated its own block (e.g. all arms
                // `panic`). merge_bb has no incoming values, so there is
                // no phi to build — the block is unreachable. Return a
                // dummy value; LLVM eliminates the dead block.
                return Ok(Value::Int(self.i64_ty.const_zero()));
            }
        };
        let phi = self
            .builder
            .build_phi(phi_ty, "match_result")
            .map_err(|e| CodegenError::JitInit(format!("build_phi: {}", e)))?;
        for (val, blk) in &incoming {
            phi.add_incoming(&[(val, *blk)]);
        }
        let any = phi.as_basic_value();
        match any {
            BasicValueEnum::IntValue(iv) => Ok(Value::Int(iv)),
            BasicValueEnum::PointerValue(pv) => Ok(Value::Closure(pv)),
            other => Err(CodegenError::TypeMismatch {
                what: format!("unexpected match-result type: {:?}", other),
            }),
        }
    }

    /// Emit a refutable test of `pattern` against `value_ptr` (a heap
    /// pointer to an enum value, or — for a top-level `Var` — the scrutinee).
    /// On any tag mismatch the builder branches to `fail_bb`. On full match it
    /// writes the pattern's single binding (if any), pushes it onto `env`, and
    /// leaves the builder positioned in the matched block. Returns the number
    /// of bindings pushed (0 or 1 — patterns are linear chains).
    ///
    /// `value_ty` is the static type of `value_ptr` (the scrutinee type at the
    /// top level, the substituted payload type when recursing); it supplies the
    /// inner enum's generic instantiation and the unbox decision. `pushed` is
    /// the number of bindings already placed in this arm, for slot indexing.
    #[allow(clippy::too_many_arguments)]
    fn emit_pattern_match(
        &mut self,
        value_ptr: inkwell::values::PointerValue<'ctx>,
        value_ty: &Type,
        pattern: &Pattern,
        fail_bb: inkwell::basic_block::BasicBlock<'ctx>,
        env: &mut Env<'ctx>,
        ctx: CompileCtx<'ctx>,
        pushed: usize,
    ) -> Result<usize, CodegenError> {
        match pattern {
            // Top-level catch-all: matches unconditionally, binds nothing.
            Pattern::Wildcard => Ok(0),
            // Top-level catch-all binding: bind the whole value pointer.
            Pattern::Var => {
                let slot_idx = ctx.next_root_slot + pushed as u32;
                let slot =
                    self.write_root_slot(ctx.frame_alloca, ctx.info, slot_idx, value_ptr)?;
                env.push(EnvSlot::Closure(slot), value_ty.clone());
                Ok(1)
            }
            Pattern::Enum {
                enum_ref,
                variant_index,
                payload,
            } => {
                let einfo = self
                    .enums
                    .get(enum_ref)
                    .cloned()
                    .ok_or(CodegenError::UnknownTopRef { hash: *enum_ref })?;
                let instantiation: Vec<Type> = match value_ty {
                    Type::Apply(_, args) => args.clone(),
                    _ => Vec::new(),
                };

                // Load the tag and branch on tag == variant_index.
                let tag_offset = einfo.variants[0].tag_offset;
                let tag_off_const = self.i64_ty.const_int(tag_offset as u64, false);
                let tag_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            self.context.i8_type(),
                            value_ptr,
                            &[tag_off_const],
                            "tag_ptr",
                        )
                        .map_err(|e| CodegenError::JitInit(format!("gep tag for match: {}", e)))?
                };
                let tag = self
                    .builder
                    .build_load(self.context.i32_type(), tag_ptr, "tag")
                    .map_err(|e| CodegenError::JitInit(format!("load tag: {}", e)))?
                    .into_int_value();
                let want = self
                    .context
                    .i32_type()
                    .const_int(*variant_index as u64, false);
                let cond = self
                    .builder
                    .build_int_compare(inkwell::IntPredicate::EQ, tag, want, "tag_eq")
                    .map_err(|e| CodegenError::JitInit(format!("cmp tag: {}", e)))?;
                let parent = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();
                let matched_bb = self.context.append_basic_block(parent, "match_tag_ok");
                self.builder
                    .build_conditional_branch(cond, matched_bb, fail_bb)
                    .map_err(|e| CodegenError::JitInit(format!("br tag test: {}", e)))?;
                self.builder.position_at_end(matched_bb);

                let sub = match payload {
                    Some(sub) => sub,
                    None => return Ok(0), // nullary variant: nothing to bind
                };

                let v = einfo.variants[*variant_index as usize];
                let payload_off =
                    v.payload_offset.ok_or(CodegenError::TypeMismatch {
                        what: format!(
                            "match arm pattern has payload but variant {} is nullary",
                            variant_index
                        ),
                    })?;
                let off_const = self.i64_ty.const_int(payload_off as u64, false);
                let payload_slot = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            self.context.i8_type(),
                            value_ptr,
                            &[off_const],
                            "match_payload_ptr",
                        )
                        .map_err(|e| {
                            CodegenError::JitInit(format!("gep match payload: {}", e))
                        })?
                };
                let declared_payload_ty = self
                    .enum_variant_types
                    .get(enum_ref)
                    .and_then(|vs| vs.get(*variant_index as usize).cloned().flatten());
                let inst_payload_ty = declared_payload_ty
                    .as_ref()
                    .map(|t| substitute_type(t, &instantiation))
                    .unwrap_or(Type::Builtin("Int".to_owned()));
                let needs_unbox = matches!(&declared_payload_ty, Some(Type::TypeVar(_)))
                    && matches!(&inst_payload_ty, Type::Builtin(n) if is_boxed_scalar(n));

                match sub.as_ref() {
                    Pattern::Wildcard => Ok(0),
                    Pattern::Var => {
                        if v.payload_is_pointer && !needs_unbox {
                            let loaded = self
                                .builder
                                .build_load(self.ptr_ty, payload_slot, "match_payload_ptr_val")
                                .map_err(|e| {
                                    CodegenError::JitInit(format!("load match payload ptr: {}", e))
                                })?
                                .into_pointer_value();
                            let slot_idx = ctx.next_root_slot + pushed as u32;
                            let slot = self.write_root_slot(
                                ctx.frame_alloca,
                                ctx.info,
                                slot_idx,
                                loaded,
                            )?;
                            env.push(EnvSlot::Closure(slot), inst_payload_ty);
                        } else if v.payload_is_pointer && needs_unbox {
                            let boxed_ptr = self
                                .builder
                                .build_load(self.ptr_ty, payload_slot, "match_boxed_int_ptr")
                                .map_err(|e| {
                                    CodegenError::JitInit(format!("load boxed int ptr: {}", e))
                                })?
                                .into_pointer_value();
                            let unbox_fn =
                                self.extern_unbox_int.expect("ai_gc_unbox_int declared");
                            let call = self
                                .builder
                                .build_call(unbox_fn, &[boxed_ptr.into()], "match_unboxed_int")
                                .map_err(|e| {
                                    CodegenError::JitInit(format!(
                                        "build_call ai_gc_unbox_int: {}",
                                        e
                                    ))
                                })?;
                            let iv = call.as_any_value_enum().into_int_value();
                            env.push(EnvSlot::Int(iv), inst_payload_ty);
                        } else {
                            let loaded = self
                                .builder
                                .build_load(self.i64_ty, payload_slot, "match_payload_int")
                                .map_err(|e| {
                                    CodegenError::JitInit(format!("load match payload int: {}", e))
                                })?
                                .into_int_value();
                            env.push(EnvSlot::Int(loaded), inst_payload_ty);
                        }
                        Ok(1)
                    }
                    // Nested enum pattern: the payload is a heap pointer to the
                    // inner enum value. Load it and recurse — the inner tag
                    // test shares the same `fail_bb`, so any mismatch at any
                    // depth backtracks to the next arm.
                    Pattern::Enum { .. } => {
                        let inner_ptr = self
                            .builder
                            .build_load(self.ptr_ty, payload_slot, "match_inner_ptr")
                            .map_err(|e| {
                                CodegenError::JitInit(format!("load nested payload ptr: {}", e))
                            })?
                            .into_pointer_value();
                        self.emit_pattern_match(
                            inner_ptr,
                            &inst_payload_ty,
                            sub,
                            fail_bb,
                            env,
                            ctx,
                            pushed,
                        )
                    }
                }
            }
        }
    }

    fn emit_hash_constant(&mut self, h: &Hash) -> GlobalValue<'ctx> {
        let sym = format!("__hash_{}", h.to_hex());
        if let Some(existing) = self.module.get_global(&sym) {
            return existing;
        }
        let arr_const = self.context.const_string(h.as_bytes(), false);
        let g = self
            .module
            .add_global(arr_const.get_type(), Some(AddressSpace::default()), &sym);
        g.set_linkage(Linkage::Private);
        g.set_constant(true);
        g.set_initializer(&arr_const);
        // Don't pin in @llvm.compiler.used — referenced by an actual
        // instruction (memcpy) so LLVM won't elide it.
        g
    }

    fn compile_builtin(
        &mut self,
        name: &str,
        args: &[IntValue<'ctx>],
    ) -> Result<IntValue<'ctx>, CodegenError> {
        let builder = &self.builder;
        let i64_ty = self.i64_ty;
        let bin = |op: &'static str,
                   build: fn(
            &Builder<'ctx>,
            IntValue<'ctx>,
            IntValue<'ctx>,
            &str,
        )
            -> Result<IntValue<'ctx>, String>|
         -> Result<IntValue<'ctx>, CodegenError> {
            if args.len() != 2 {
                return Err(CodegenError::UnknownBuiltin {
                    name: name.to_owned(),
                    arity: args.len(),
                });
            }
            build(builder, args[0], args[1], op)
                .map_err(|e| CodegenError::JitInit(format!("build {}: {}", op, e)))
        };
        let cmp = |pred: IntPredicate| -> Result<IntValue<'ctx>, CodegenError> {
            if args.len() != 2 {
                return Err(CodegenError::UnknownBuiltin {
                    name: name.to_owned(),
                    arity: args.len(),
                });
            }
            let cmp_bit = builder
                .build_int_compare(pred, args[0], args[1], "cmptmp")
                .map_err(|e| CodegenError::JitInit(format!("build_int_compare: {}", e)))?;
            builder
                .build_int_z_extend(cmp_bit, i64_ty, "cmp_i64")
                .map_err(|e| CodegenError::JitInit(format!("build_int_z_extend: {}", e)))
        };
        // Float ops. A Float value is carried as the i64 bit-pattern of
        // its f64; we bitcast in, operate, and bitcast the result back to
        // i64. (This reuses all the Int machinery for storage/ABI; only
        // arithmetic/comparison/conversion know it's really a double.)
        let f64_ty = self.context.f64_type();
        let as_f64 = |v: IntValue<'ctx>, n: &str| -> Result<FloatValue<'ctx>, CodegenError> {
            builder
                .build_bit_cast(v, f64_ty, n)
                .map(|b| b.into_float_value())
                .map_err(|e| CodegenError::JitInit(format!("bitcast i64->f64: {}", e)))
        };
        let f64_bits = |v: FloatValue<'ctx>, n: &str| -> Result<IntValue<'ctx>, CodegenError> {
            builder
                .build_bit_cast(v, i64_ty, n)
                .map(|b| b.into_int_value())
                .map_err(|e| CodegenError::JitInit(format!("bitcast f64->i64: {}", e)))
        };
        let fbin = |op: &'static str,
                    build: fn(
            &Builder<'ctx>,
            FloatValue<'ctx>,
            FloatValue<'ctx>,
            &str,
        ) -> Result<FloatValue<'ctx>, String>|
         -> Result<IntValue<'ctx>, CodegenError> {
            if args.len() != 2 {
                return Err(CodegenError::UnknownBuiltin {
                    name: name.to_owned(),
                    arity: args.len(),
                });
            }
            let l = as_f64(args[0], "fl")?;
            let r = as_f64(args[1], "fr")?;
            let res = build(builder, l, r, op)
                .map_err(|e| CodegenError::JitInit(format!("build f64.{}: {}", op, e)))?;
            f64_bits(res, "fres_bits")
        };
        let fcmp = |pred: inkwell::FloatPredicate| -> Result<IntValue<'ctx>, CodegenError> {
            if args.len() != 2 {
                return Err(CodegenError::UnknownBuiltin {
                    name: name.to_owned(),
                    arity: args.len(),
                });
            }
            let l = as_f64(args[0], "fl")?;
            let r = as_f64(args[1], "fr")?;
            let bit = builder
                .build_float_compare(pred, l, r, "fcmptmp")
                .map_err(|e| CodegenError::JitInit(format!("build_float_compare: {}", e)))?;
            builder
                .build_int_z_extend(bit, i64_ty, "fcmp_i64")
                .map_err(|e| CodegenError::JitInit(format!("build_int_z_extend (f): {}", e)))
        };
        Ok(match name {
            "core/i64.add" => bin("add", |b, l, r, n| {
                b.build_int_add(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/i64.sub" => bin("sub", |b, l, r, n| {
                b.build_int_sub(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/i64.mul" => bin("mul", |b, l, r, n| {
                b.build_int_mul(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/i64.div" => bin("sdiv", |b, l, r, n| {
                b.build_int_signed_div(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/i64.rem" => bin("srem", |b, l, r, n| {
                b.build_int_signed_rem(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/i64.eq" => cmp(IntPredicate::EQ)?,
            "core/i64.ne" => cmp(IntPredicate::NE)?,
            "core/i64.lt" => cmp(IntPredicate::SLT)?,
            "core/i64.le" => cmp(IntPredicate::SLE)?,
            "core/i64.gt" => cmp(IntPredicate::SGT)?,
            "core/i64.ge" => cmp(IntPredicate::SGE)?,
            "core/i64.neg" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.to_owned(),
                        arity: args.len(),
                    });
                }
                self.builder
                    .build_int_neg(args[0], "neg")
                    .map_err(|e| CodegenError::JitInit(format!("build_int_neg: {}", e)))?
            }
            // Bitwise ops on i64 (used by base64 / CRC32 / zip and any bit
            // manipulation). `shr` is a LOGICAL (zero-filling) shift.
            "core/i64.and" => bin("and", |b, l, r, n| {
                b.build_and(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/i64.or" => bin("or", |b, l, r, n| {
                b.build_or(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/i64.xor" => bin("xor", |b, l, r, n| {
                b.build_xor(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/i64.shl" => bin("shl", |b, l, r, n| {
                b.build_left_shift(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/i64.shr" => bin("shr", |b, l, r, n| {
                b.build_right_shift(l, r, false, n).map_err(|e| e.to_string())
            })?,
            "core/f64.add" => fbin("add", |b, l, r, n| {
                b.build_float_add(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/f64.sub" => fbin("sub", |b, l, r, n| {
                b.build_float_sub(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/f64.mul" => fbin("mul", |b, l, r, n| {
                b.build_float_mul(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/f64.div" => fbin("div", |b, l, r, n| {
                b.build_float_div(l, r, n).map_err(|e| e.to_string())
            })?,
            "core/f64.rem" => fbin("rem", |b, l, r, n| {
                b.build_float_rem(l, r, n).map_err(|e| e.to_string())
            })?,
            // Ordered comparisons (false if either operand is NaN).
            "core/f64.eq" => fcmp(inkwell::FloatPredicate::OEQ)?,
            "core/f64.ne" => fcmp(inkwell::FloatPredicate::ONE)?,
            "core/f64.lt" => fcmp(inkwell::FloatPredicate::OLT)?,
            "core/f64.le" => fcmp(inkwell::FloatPredicate::OLE)?,
            "core/f64.gt" => fcmp(inkwell::FloatPredicate::OGT)?,
            "core/f64.ge" => fcmp(inkwell::FloatPredicate::OGE)?,
            "core/f64.of_int" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.to_owned(),
                        arity: args.len(),
                    });
                }
                let f = self
                    .builder
                    .build_signed_int_to_float(args[0], f64_ty, "of_int")
                    .map_err(|e| CodegenError::JitInit(format!("build_signed_int_to_float: {}", e)))?;
                f64_bits(f, "of_int_bits")?
            }
            "core/f64.to_int" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.to_owned(),
                        arity: args.len(),
                    });
                }
                let f = as_f64(args[0], "to_int_f")?;
                self.builder
                    .build_float_to_signed_int(f, i64_ty, "to_int")
                    .map_err(|e| CodegenError::JitInit(format!("build_float_to_signed_int: {}", e)))?
            }
            "core/f64.sqrt" => {
                if args.len() != 1 {
                    return Err(CodegenError::UnknownBuiltin {
                        name: name.to_owned(),
                        arity: args.len(),
                    });
                }
                let f = as_f64(args[0], "sqrt_f")?;
                let intrinsic = inkwell::intrinsics::Intrinsic::find("llvm.sqrt.f64")
                    .ok_or_else(|| CodegenError::JitInit("llvm.sqrt.f64 intrinsic not found".to_owned()))?;
                let decl = intrinsic
                    .get_declaration(&self.module, &[f64_ty.into()])
                    .ok_or_else(|| CodegenError::JitInit("llvm.sqrt.f64 declaration failed".to_owned()))?;
                let res = self
                    .builder
                    .build_call(decl, &[f.into()], "sqrt")
                    .map_err(|e| CodegenError::JitInit(format!("build_call llvm.sqrt.f64: {}", e)))?
                    .as_any_value_enum()
                    .into_float_value();
                f64_bits(res, "sqrt_bits")?
            }

            // ---- raw pointer / memory intrinsics (the C FFI's hands) ----
            // A `Ptr` is an i64 address. These are the irreducible
            // primitives every FFI needs: read/write memory and form/test
            // null. Everything else (cstr, malloc, libcurl) is ai-lang +
            // dlsym'd C functions on top of these.
            "core/ptr.null" => {
                self.arity_check(name, args, 0)?;
                i64_ty.const_zero()
            }
            "core/ptr.is_null" => {
                self.arity_check(name, args, 1)?;
                let bit = builder
                    .build_int_compare(IntPredicate::EQ, args[0], i64_ty.const_zero(), "is_null")
                    .map_err(|e| CodegenError::JitInit(format!("ptr.is_null cmp: {}", e)))?;
                builder
                    .build_int_z_extend(bit, i64_ty, "is_null_i64")
                    .map_err(|e| CodegenError::JitInit(format!("ptr.is_null zext: {}", e)))?
            }
            "core/ptr.add" => {
                self.arity_check(name, args, 2)?;
                builder
                    .build_int_add(args[0], args[1], "ptr_add")
                    .map_err(|e| CodegenError::JitInit(format!("ptr.add: {}", e)))?
            }
            "core/ptr.read_u8" => {
                self.arity_check(name, args, 2)?;
                let p = self.addr_at(args[0], args[1], "ptr_read_u8")?;
                let loaded = builder
                    .build_load(self.context.i8_type(), p, "u8")
                    .map_err(|e| CodegenError::JitInit(format!("ptr.read_u8 load: {}", e)))?
                    .into_int_value();
                builder
                    .build_int_z_extend(loaded, i64_ty, "u8_i64")
                    .map_err(|e| CodegenError::JitInit(format!("ptr.read_u8 zext: {}", e)))?
            }
            "core/ptr.write_u8" => {
                self.arity_check(name, args, 3)?;
                let p = self.addr_at(args[0], args[1], "ptr_write_u8")?;
                let byte = builder
                    .build_int_truncate(args[2], self.context.i8_type(), "u8_trunc")
                    .map_err(|e| CodegenError::JitInit(format!("ptr.write_u8 trunc: {}", e)))?;
                builder
                    .build_store(p, byte)
                    .map_err(|e| CodegenError::JitInit(format!("ptr.write_u8 store: {}", e)))?;
                i64_ty.const_zero()
            }
            "core/ptr.read_i64" | "core/ptr.read_ptr" => {
                self.arity_check(name, args, 2)?;
                let p = self.addr_at(args[0], args[1], "ptr_read_i64")?;
                builder
                    .build_load(i64_ty, p, "i64v")
                    .map_err(|e| CodegenError::JitInit(format!("ptr.read_i64 load: {}", e)))?
                    .into_int_value()
            }
            "core/ptr.write_i64" | "core/ptr.write_ptr" => {
                self.arity_check(name, args, 3)?;
                let p = self.addr_at(args[0], args[1], "ptr_write_i64")?;
                builder
                    .build_store(p, args[2])
                    .map_err(|e| CodegenError::JitInit(format!("ptr.write_i64 store: {}", e)))?;
                i64_ty.const_zero()
            }
            // Reinterpret casts between Ptr and Int. Both are i64 at
            // runtime, so the value passes through unchanged; only the
            // type classification differs. This is the explicit boundary
            // that lets an address travel as plain data (RemotePtr).
            "core/ptr.to_int" | "core/ptr.from_int" => {
                self.arity_check(name, args, 1)?;
                args[0]
            }

            _ => {
                return Err(CodegenError::UnknownBuiltin {
                    name: name.to_owned(),
                    arity: args.len(),
                });
            }
        })
    }

    /// Validate a builtin's argument count, producing a clear error.
    fn arity_check(
        &self,
        name: &str,
        args: &[IntValue<'ctx>],
        expected: usize,
    ) -> Result<(), CodegenError> {
        if args.len() != expected {
            return Err(CodegenError::UnknownBuiltin {
                name: name.to_owned(),
                arity: args.len(),
            });
        }
        Ok(())
    }

    /// Form a pointer to `base + offset` (both i64) for a load/store.
    fn addr_at(
        &self,
        base: IntValue<'ctx>,
        offset: IntValue<'ctx>,
        label: &str,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        let addr = self
            .builder
            .build_int_add(base, offset, label)
            .map_err(|e| CodegenError::JitInit(format!("{} add: {}", label, e)))?;
        self.builder
            .build_int_to_ptr(addr, self.ptr_ty, label)
            .map_err(|e| CodegenError::JitInit(format!("{} inttoptr: {}", label, e)))
    }

    fn emit_compiler_used(&mut self) {
        if self.used_globals.is_empty() {
            return;
        }
        let arr_ty = self.ptr_ty.array_type(self.used_globals.len() as u32);
        let elems: Vec<_> = self.used_globals.iter().copied().collect();
        let init = self.ptr_ty.const_array(&elems);
        let g = self.module.add_global(arr_ty, None, "llvm.compiler.used");
        g.set_initializer(&init);
        g.set_linkage(Linkage::Appending);
    }
}

// =============================================================================
// Per-function env
// =============================================================================

#[derive(Clone)]
struct Env<'ctx> {
    slots: Vec<EnvSlot<'ctx>>,
    /// Parallel to `slots`, the canonical-AST type of each binding.
    /// Used by codegen to decide where to box/unbox when crossing
    /// generic boundaries (TypeVar slots are pointers; the body may
    /// use them at a concrete type the typechecker substituted in).
    types: Vec<Type>,
}

impl<'ctx> Env<'ctx> {
    fn new() -> Self {
        Env {
            slots: Vec::new(),
            types: Vec::new(),
        }
    }

    fn push(&mut self, s: EnvSlot<'ctx>, ty: Type) {
        self.slots.push(s);
        self.types.push(ty);
    }

    fn pop(&mut self) -> Option<EnvSlot<'ctx>> {
        self.types.pop();
        self.slots.pop()
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    fn get(&self, i: usize) -> &EnvSlot<'ctx> {
        &self.slots[i]
    }

    fn type_at(&self, i: usize) -> &Type {
        &self.types[i]
    }
}

/// Per-binding storage. `Int` is a plain SSA value; `Closure` is an
/// alloca slot (a pointer-to-pointer) that lives in the function's
/// shadow-stack frame so the GC can update it.
#[derive(Copy, Clone)]
enum EnvSlot<'ctx> {
    Int(IntValue<'ctx>),
    Closure(PointerValue<'ctx>), // address of the slot
}

impl<'ctx> EnvSlot<'ctx> {
    fn read(&self, cg: &Codegen<'ctx>) -> Result<Value<'ctx>, CodegenError> {
        match self {
            EnvSlot::Int(v) => Ok(Value::Int(*v)),
            EnvSlot::Closure(slot) => {
                let p = cg.read_root_slot(*slot)?;
                Ok(Value::Closure(p))
            }
        }
    }
}

#[derive(Copy, Clone)]
struct CompileCtx<'ctx> {
    thread_param: PointerValue<'ctx>,
    frame_alloca: PointerValue<'ctx>,
    info: DefInfo<'ctx>,
    next_root_slot: u32,
    /// True iff the current expression sits in tail position relative
    /// to the enclosing fn. Propagated through if-branches, match-arm
    /// bodies, and let-bodies. Cleared for args, conditions,
    /// scrutinees, and arbitrary intermediate subexpressions. Used by
    /// `compile_call` to detect self-tail-call sites and emit them as
    /// branches back to the loop-body block instead of real calls.
    is_tail: bool,
}

// =============================================================================
// Helpers
// =============================================================================

/// Per-fn tail-call optimization context. Established at the start
/// of `compile_def`, consumed by `compile_call` when it detects a
/// tail-position call back to the same fn. Cleared at fn end.
#[derive(Clone)]
struct TailCtx<'ctx> {
    /// Content hash of the fn currently being compiled. A `TopRef(h)`
    /// callee matches if `h == self_hash`.
    self_hash: Hash,
    /// Block to branch to for tail-call jumps. Loaded params are
    /// re-fetched at this block's start each iteration.
    loop_body: inkwell::basic_block::BasicBlock<'ctx>,
    /// One slot per param. For Int params: an `i64` alloca holding
    /// the current value. For pointer-typed params: the root slot
    /// (shadow-stack alloca) the prologue allocated. Tail-calls write
    /// the new arg value into the right slot before branching back.
    param_slots: Vec<TailParamSlot<'ctx>>,
}

#[derive(Clone, Copy)]
enum TailParamSlot<'ctx> {
    Int(PointerValue<'ctx>),
    /// The root slot (already pointer-shaped). For pointer params we
    /// also need to know whether it's a real heap pointer vs. a
    /// String-typed value etc.; we just reuse the slot.
    Ptr(PointerValue<'ctx>),
}

/// A minimal mirror of a fn def's declared types — captured at
/// declare time so codegen can introspect signatures (especially
/// for boxing/unboxing across generic call boundaries).
#[derive(Clone, Debug)]
struct FnSigSimple {
    params: Vec<Type>,
    ret: Type,
}

/// `max_typevar_in_types(params, ret) = 1 + max index of any TypeVar
/// in params ∪ {ret}` (or 0 if none). Mirrors `max_type_var` in
/// typecheck.rs but lives here to avoid a cross-module dep.
fn max_typevar_in_types(params: &[Type], ret: &Type) -> u32 {
    let mut m = max_tv_in(ret);
    for p in params {
        m = m.max(max_tv_in(p));
    }
    m
}

fn max_tv_in(ty: &Type) -> u32 {
    match ty {
        Type::TypeVar(i) => i + 1,
        Type::Builtin(_) | Type::TypeRef(_) | Type::SelfRef(_) => 0,
        Type::FnType { params, ret } => max_typevar_in_types(params, ret),
        Type::Apply(head, args) => {
            let mut m = max_tv_in(head);
            for a in args {
                m = m.max(max_tv_in(a));
            }
            m
        }
    }
}

/// Substitute (positional) type-var entries throughout `ty`.
fn substitute_type(ty: &Type, subst: &[Type]) -> Type {
    match ty {
        Type::TypeVar(i) => subst
            .get(*i as usize)
            .cloned()
            .unwrap_or_else(|| Type::TypeVar(*i)),
        Type::Builtin(_) | Type::TypeRef(_) | Type::SelfRef(_) => ty.clone(),
        Type::FnType { params, ret } => Type::FnType {
            params: params.iter().map(|p| substitute_type(p, subst)).collect(),
            ret: Box::new(substitute_type(ret, subst)),
        },
        Type::Apply(head, args) => Type::Apply(
            Box::new(substitute_type(head, subst)),
            args.iter().map(|a| substitute_type(a, subst)).collect(),
        ),
    }
}

/// Best-effort unification: fills in `subst` by matching `declared`
/// (which may contain TypeVars) against the concrete `actual`.
/// Returns `Err(())` if a non-TypeVar mismatch is encountered — the
/// caller can fall through to a default (codegen treats this as
/// "no instantiation", same as a monomorphic call).
fn unify_for_codegen(
    declared: &Type,
    actual: &Type,
    subst: &mut Vec<Option<Type>>,
) -> Result<(), ()> {
    match (declared, actual) {
        (Type::TypeVar(i), other) => {
            let idx = *i as usize;
            if idx >= subst.len() {
                subst.resize(idx + 1, None);
            }
            // Prefer concrete bindings over TypeVar placeholders. A
            // generic call inside a generic fn (e.g. `opt_unwrap_or`
            // applied to a `Apply(Option, [TypeVar(0)])` carried up
            // from an unresolved inner call) initially stores
            // `subst[0] = Some(TypeVar(0))`. Later args that pin the
            // same slot to `Int` need to override that placeholder.
            match &subst[idx] {
                None => subst[idx] = Some(other.clone()),
                Some(Type::TypeVar(_)) if !matches!(other, Type::TypeVar(_)) => {
                    subst[idx] = Some(other.clone());
                }
                _ => {}
            }
            Ok(())
        }
        (Type::Builtin(a), Type::Builtin(b)) if a == b => Ok(()),
        (Type::TypeRef(a), Type::TypeRef(b)) if a == b => Ok(()),
        (
            Type::FnType { params: pa, ret: ra },
            Type::FnType { params: pb, ret: rb },
        ) if pa.len() == pb.len() => {
            for (a, b) in pa.iter().zip(pb.iter()) {
                unify_for_codegen(a, b, subst)?;
            }
            unify_for_codegen(ra, rb, subst)
        }
        (Type::Apply(ha, aa), Type::Apply(hb, ab)) if aa.len() == ab.len() => {
            unify_for_codegen(ha, hb, subst)?;
            for (a, b) in aa.iter().zip(ab.iter()) {
                unify_for_codegen(a, b, subst)?;
            }
            Ok(())
        }
        _ => Err(()),
    }
}

fn require_supported_type(t: &Type, def_name: &str, role: &str) -> Result<(), CodegenError> {
    match t {
        Type::Builtin(n) if n == "Int" => Ok(()),
        // Bool is i64 0/1 — same ABI carrier as Int (see `is_int_type`).
        Type::Builtin(n) if n == "Bool" => Ok(()),
        Type::Builtin(n) if n == "Float" => Ok(()),
        Type::Builtin(n) if n == "String" => Ok(()),
        Type::Builtin(n) if n == "Bytes" => Ok(()),
        // Raw C pointer: an i64-represented address, non-GC. Used by the
        // C FFI and the pointer/memory intrinsics.
        Type::Builtin(n) if n == "Ptr" => Ok(()),
        Type::FnType { .. } => Ok(()),
        Type::TypeRef(_) => Ok(()),
        // Generic-typed values: TypeVar (declared param) and Apply
        // (instantiated generic). Both are represented as boxed
        // pointers under the uniform-representation scheme.
        Type::TypeVar(_) => Ok(()),
        Type::Apply(_, _) => Ok(()),
        other => Err(CodegenError::Unsupported {
            what: format!("unsupported {} type in `{}`: {:?}", role, def_name, other),
        }),
    }
}

fn is_int_type(t: &Type) -> bool {
    // Bool shares the i64 representation (0/1), so it rides the same
    // scalar-int ABI: i64 params, i64 return, no heap pointer.
    matches!(t, Type::Builtin(n) if n == "Int" || n == "Bool")
}

/// Scalars that live boxed (as a `BoxedInt`) when stored in a generic
/// TypeVar slot: `Int` and `Float`. Both are 8-byte values carried in
/// the i64 representation, so the same box/unbox path serves both
/// (`Float` is the f64 bit-pattern). Used to decide where generic code
/// must box on the way in and unbox on the way out.
fn is_boxed_scalar(n: &str) -> bool {
    n == "Int" || n == "Float"
}


/// Whether an `Atom<T>` type's value `T` is `Int` — i.e. the cell holds a
/// `BoxedInt` that `atom.new`/`atom.load`/`atom.swap` must box/unbox.
fn atom_element_is_int(atom_ty: &Type) -> bool {
    match atom_ty {
        Type::Apply(head, args) => {
            matches!(head.as_ref(), Type::Builtin(n) if n == "Atom")
                && args
                    .first()
                    .map(|t| matches!(t, Type::Builtin(n) if is_boxed_scalar(n)))
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// Whether a `ThreadHandle<T>`'s `T` is `Int` — i.e. `join` must unbox the
/// `BoxedInt` the spawned thunk produced for its Int return.
fn thread_handle_element_is_int(handle_ty: &Type) -> bool {
    match handle_ty {
        Type::Apply(head, args) => {
            matches!(head.as_ref(), Type::Builtin(n) if n == "ThreadHandle")
                && args
                    .first()
                    .map(|t| matches!(t, Type::Builtin(n) if is_boxed_scalar(n)))
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// Whether an `Array<T>` type's element `T` is `Int` — i.e. its slots
/// hold `BoxedInt`s that `array.get`/`array.set` must unbox/box. Any
/// other (or unpinned/unknown) element type uses the raw pointer slot.
fn array_element_is_int(array_ty: &Type) -> bool {
    match array_ty {
        Type::Apply(head, args) => {
            matches!(head.as_ref(), Type::Builtin(n) if n == "Array")
                && args
                    .first()
                    .map(|t| matches!(t, Type::Builtin(n) if is_boxed_scalar(n)))
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// Types we let cross the FFI boundary today. Int and String are
/// supported via the Layer-1/Layer-2 ABI (i64 and heap-pointer
/// respectively). Other types need their own marshaling story.
fn is_extern_supported_type(t: &Type) -> bool {
    matches!(t, Type::Builtin(n) if n == "Int" || n == "String" || n == "Ptr")
}

/// Whether a type is carried as a raw i64 at the C-FFI boundary: `Int`
/// and `Ptr` (a pointer is an integer-width address). Used to build the
/// LLVM signature for a `extern "C"` declaration.
fn is_c_scalar_type(t: &Type) -> bool {
    matches!(t, Type::Builtin(n) if n == "Int" || n == "Ptr")
}

/// Whether a value of this type is represented as a pointer at runtime.
/// Pointer-typed values participate in GC root scanning and live in
/// frame root slots when held as locals.
///
/// `String` and `Bytes` are pointer-typed (both use the heap varlen-bytes
/// shape Runtime registers). The other current `Type::Builtin` values
/// (Int, Bool, Float) are i64 / unsupported.
///
/// `TypeVar` and `Apply` are pointer-typed because under our uniform
/// representation, every generic-typed slot holds a heap pointer (a
/// boxed primitive or a real heap object).
/// A statically scalar type: carried as a raw i64 (Int / Bool) or the
/// i64 bit-pattern of an f64 (Float). The complement of pointer-typed
/// for CONCRETE builtins only — TypeVar/Apply are pointers under the
/// uniform representation.
fn is_scalar_type(t: &Type) -> bool {
    matches!(t, Type::Builtin(n) if n == "Int" || n == "Float" || n == "Bool")
}

fn is_pointer_type(t: &Type) -> bool {
    match t {
        Type::Builtin(n) if n == "String" || n == "Bytes" => true,
        Type::FnType { .. } | Type::TypeRef(_) | Type::TypeVar(_) | Type::Apply(_, _) => true,
        _ => false,
    }
}

/// Pre-scan: count GC-typed locals introduced by heap-allocating
/// `let` values or by match-arm payload bindings.
/// Whether the body contains a `defer` anywhere — if so the function
/// keeps its GC frame even with zero pointer roots, because `defer`
/// compilation uses the frame's scratch slot.
fn body_mentions_defer(e: &Expr) -> bool {
    match e {
        Expr::Defer { .. } => true,
        Expr::Lambda { body, .. } => body_mentions_defer(body),
        Expr::Call(callee, args) => {
            body_mentions_defer(callee) || args.iter().any(body_mentions_defer)
        }
        Expr::Let { value, body } => {
            body_mentions_defer(value) || body_mentions_defer(body)
        }
        Expr::StructNew { fields, .. } => fields.iter().any(body_mentions_defer),
        Expr::Field { base, .. } => body_mentions_defer(base),
        Expr::EnumNew { payload, .. } => {
            payload.as_ref().is_some_and(|p| body_mentions_defer(p))
        }
        Expr::Match { scrutinee, arms } => {
            body_mentions_defer(scrutinee)
                || arms.iter().any(|a| body_mentions_defer(&a.body))
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            body_mentions_defer(cond)
                || body_mentions_defer(then_branch)
                || body_mentions_defer(else_branch)
        }
        Expr::Try { expr, .. } => body_mentions_defer(expr),
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => false,
    }
}


fn count_pattern_vars(p: &crate::ast::Pattern) -> u32 {
    use crate::ast::Pattern;
    match p {
        Pattern::Wildcard => 0,
        Pattern::Var => 1,
        Pattern::Enum { payload, .. } => payload.as_deref().map(count_pattern_vars).unwrap_or(0),
    }
}

/// If the lambda's body returns a captured outer var **directly**
/// (as the whole expression value), return that var's outer-de-Bruijn
/// index. Otherwise None. Recognizes the body shape `LocalVar(outer)`,
/// `Let { body: LocalVar(outer), ... }` chains, and `Match` arms that
/// all return the same capture.
///
/// Used by capture-type inference: a pass-through capture must be
/// pointer-shaped because the lambda's return slot follows the
/// uniform pointer ABI.
fn passthrough_capture_index(body: &Expr, arity: u32) -> Option<u32> {
    fn capture_of(e: &Expr, arity: u32) -> Option<u32> {
        if let Expr::LocalVar(i) = e {
            if *i >= arity {
                return Some(*i - arity);
            }
        }
        None
    }
    match body {
        Expr::LocalVar(i) if *i >= arity => Some(*i - arity),
        Expr::Let { body, .. } => passthrough_capture_index(body, arity + 1),
        Expr::Match { arms, .. } => {
            // All arms must produce the same passthrough capture.
            let mut shared: Option<u32> = None;
            for arm in arms {
                let bindings = count_pattern_vars(&arm.pattern);
                let cap = capture_of(&arm.body, arity + bindings)?;
                match shared {
                    None => shared = Some(cap),
                    Some(s) if s == cap => {}
                    _ => return None,
                }
            }
            shared
        }
        _ => None,
    }
}

/// Compute the capture set of a lambda body: outer de Bruijn indices
/// referenced by any `LocalVar(i)` with `i >= arity_so_far`.
///
/// Returns indices in ascending order.
fn collect_captures(body: &Expr, arity: u32) -> Vec<u32> {
    let mut out: BTreeSet<u32> = BTreeSet::new();
    walk_captures(body, arity, &mut out);
    out.into_iter().collect()
}

fn walk_captures(e: &Expr, arity_so_far: u32, out: &mut BTreeSet<u32>) {
    match e {
        Expr::LocalVar(idx) => {
            if *idx >= arity_so_far {
                out.insert(*idx - arity_so_far);
            }
        }
        Expr::Call(callee, args) => {
            walk_captures(callee, arity_so_far, out);
            for a in args {
                walk_captures(a, arity_so_far, out);
            }
        }
        Expr::Let { value, body } => {
            walk_captures(value, arity_so_far, out);
            walk_captures(body, arity_so_far + 1, out);
        }
        Expr::Lambda { params, body } => {
            walk_captures(body, arity_so_far + params.len() as u32, out);
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                walk_captures(f, arity_so_far, out);
            }
        }
        Expr::Field { base, .. } => walk_captures(base, arity_so_far, out),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                walk_captures(p, arity_so_far, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            walk_captures(scrutinee, arity_so_far, out);
            for arm in arms {
                // Each Var binding in the pattern introduces ONE binder
                // for the arm body. Wildcard introduces zero. Nested
                // patterns aren't supported in v1 so the count is at
                // most the number of Vars in the pattern.
                let pat_binds = count_pattern_vars(&arm.pattern);
                walk_captures(&arm.body, arity_so_far + pat_binds, out);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            walk_captures(cond, arity_so_far, out);
            walk_captures(then_branch, arity_so_far, out);
            walk_captures(else_branch, arity_so_far, out);
        }
        Expr::Try { expr, .. } => walk_captures(expr, arity_so_far, out),
        Expr::Defer { cleanup, body } => {
            walk_captures(cleanup, arity_so_far, out);
            walk_captures(body, arity_so_far, out);
        }
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => {}
    }
}

/// Rewrite a lambda body's de Bruijn indices to address the *lifted*
/// environment (captures + params), not the original outer scope.
///
/// In the lifted body, the env is laid out as:
/// ```text
/// env[0..N-1]   = captures (pushed reverse of `captures` order)
/// env[N..N+A-1] = params
/// ```
/// where N = captures.len() and A = arity. The de Bruijn indices for
/// the lifted env are:
///   - LocalVar(0..A-1)        → params (innermost)
///   - LocalVar(A..A+N-1)      → captures, where LocalVar(A + pos)
///                                refers to `captures[pos]`.
///
/// Original body indices: LocalVar(0..A-1) are params, LocalVar(A..)
/// reference outer scope. For each such reference, find `outer_idx`
/// in the captures list and rewrite to `LocalVar(A + pos)`.
///
/// Nested binders (Let) shift indices by 1.
fn rewrite_body_for_lifted(body: &Expr, arity: u32, captures: &[u32]) -> Expr {
    let cap_pos: HashMap<u32, u32> = captures
        .iter()
        .enumerate()
        .map(|(i, &outer)| (outer, i as u32))
        .collect();
    rewrite_expr(body, arity, 0, &cap_pos)
}

fn rewrite_expr(e: &Expr, arity: u32, depth: u32, cap_pos: &HashMap<u32, u32>) -> Expr {
    match e {
        Expr::LocalVar(idx) => {
            if *idx >= arity + depth {
                let outer_idx = *idx - arity - depth;
                let pos = *cap_pos
                    .get(&outer_idx)
                    .expect("capture set should include every outer reference in the body");
                Expr::LocalVar(arity + pos + depth)
            } else {
                e.clone()
            }
        }
        Expr::Call(callee, args) => Expr::Call(
            Box::new(rewrite_expr(callee, arity, depth, cap_pos)),
            args.iter()
                .map(|a| rewrite_expr(a, arity, depth, cap_pos))
                .collect(),
        ),
        Expr::Let { value, body } => Expr::Let {
            value: Box::new(rewrite_expr(value, arity, depth, cap_pos)),
            body: Box::new(rewrite_expr(body, arity, depth + 1, cap_pos)),
        },
        Expr::Lambda { .. } => {
            panic!("rewrite_expr: nested lambda should have been rejected");
        }
        Expr::StructNew { struct_ref, fields } => Expr::StructNew {
            struct_ref: *struct_ref,
            fields: fields
                .iter()
                .map(|f| rewrite_expr(f, arity, depth, cap_pos))
                .collect(),
        },
        Expr::Field {
            base,
            struct_ref,
            index,
        } => Expr::Field {
            base: Box::new(rewrite_expr(base, arity, depth, cap_pos)),
            struct_ref: *struct_ref,
            index: *index,
        },
        Expr::EnumNew {
            enum_ref,
            variant_index,
            payload,
        } => Expr::EnumNew {
            enum_ref: *enum_ref,
            variant_index: *variant_index,
            payload: payload
                .as_ref()
                .map(|p| Box::new(rewrite_expr(p, arity, depth, cap_pos))),
        },
        Expr::Match { scrutinee, arms } => Expr::Match {
            scrutinee: Box::new(rewrite_expr(scrutinee, arity, depth, cap_pos)),
            arms: arms
                .iter()
                .map(|arm| {
                    let pat_binds = count_pattern_vars(&arm.pattern);
                    MatchArm {
                        pattern: arm.pattern.clone(),
                        body: rewrite_expr(&arm.body, arity, depth + pat_binds, cap_pos),
                    }
                })
                .collect(),
        },
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => Expr::If {
            cond: Box::new(rewrite_expr(cond, arity, depth, cap_pos)),
            then_branch: Box::new(rewrite_expr(then_branch, arity, depth, cap_pos)),
            else_branch: Box::new(rewrite_expr(else_branch, arity, depth, cap_pos)),
        },
        Expr::Try {
            expr,
            enum_ref,
            ok_index,
            err_index,
        } => Expr::Try {
            expr: Box::new(rewrite_expr(expr, arity, depth, cap_pos)),
            enum_ref: *enum_ref,
            ok_index: *ok_index,
            err_index: *err_index,
        },
        Expr::Defer { cleanup, body } => Expr::Defer {
            cleanup: Box::new(rewrite_expr(cleanup, arity, depth, cap_pos)),
            body: Box::new(rewrite_expr(body, arity, depth, cap_pos)),
        },
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => e.clone(),
    }
}

/// Stable hash for a per-variant TypeInfo global. Combines the enum's
/// hash with the variant's name so each variant's `__closure_ti_<hex>`
/// symbol is unique.
pub fn derive_variant_hash(enum_hash: &Hash, variant_name: &str) -> Hash {
    let mut bytes = Vec::with_capacity(32 + variant_name.len() + 8);
    bytes.extend_from_slice(enum_hash.as_bytes());
    bytes.extend_from_slice(b"#variant:");
    bytes.extend_from_slice(variant_name.as_bytes());
    Hash::of_bytes(&bytes)
}

fn check_no_nested_lambdas(body: &Expr) -> Result<(), CodegenError> {
    match body {
        Expr::Lambda { .. } => Err(CodegenError::Unsupported {
            what: "nested lambdas (Lambda inside Lambda body) — v1 restriction".to_owned(),
        }),
        Expr::Call(callee, args) => {
            check_no_nested_lambdas(callee)?;
            for a in args {
                check_no_nested_lambdas(a)?;
            }
            Ok(())
        }
        Expr::Let { value, body } => {
            check_no_nested_lambdas(value)?;
            check_no_nested_lambdas(body)?;
            Ok(())
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                check_no_nested_lambdas(f)?;
            }
            Ok(())
        }
        Expr::Field { base, .. } => check_no_nested_lambdas(base),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                check_no_nested_lambdas(p)?;
            }
            Ok(())
        }
        Expr::Match { scrutinee, arms } => {
            check_no_nested_lambdas(scrutinee)?;
            for arm in arms {
                check_no_nested_lambdas(&arm.body)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

// =============================================================================
// JIT
// =============================================================================

pub struct Jit<'ctx> {
    pub engine: ExecutionEngine<'ctx>,
    pub functions: HashMap<Hash, FunctionValue<'ctx>>,
}

impl<'ctx> Jit<'ctx> {
    /// Build a JIT engine from a compiled module and wire up runtime
    /// hooks: register the Rust extern functions and populate the
    /// per-runtime code table with all def + lifted-lambda addresses.
    pub fn new(cm: CompiledModule<'ctx>, runtime: &Runtime) -> Result<Self, CodegenError> {
        // Strip the optimizer-pin global before handing off to the JIT.
        if let Some(used) = cm.module.get_global("llvm.compiler.used") {
            unsafe { used.delete() };
        }

        let engine = cm
            .module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| CodegenError::JitInit(e.to_string()))?;

        // Register the runtime extern fns by address.
        if let Some(alloc_fn) = cm.module.get_function("ai_gc_alloc_closure") {
            engine.add_global_mapping(&alloc_fn, ai_gc_alloc_closure as usize);
        }
        if let Some(lookup_fn) = cm.module.get_function("ai_gc_lookup_code") {
            engine.add_global_mapping(&lookup_fn, ai_gc_lookup_code as usize);
        }
        if let Some(net_at_fn) = cm.module.get_function("ai_net_at") {
            engine.add_global_mapping(&net_at_fn, crate::net::ai_net_at as usize);
        }
        if let Some(f) = cm.module.get_function("ai_net_at_async") {
            engine.add_global_mapping(&f, crate::net::ai_net_at_async as usize);
        }
        if let Some(f) = cm.module.get_function("ai_wire_encode") {
            engine.add_global_mapping(&f, crate::net::ai_wire_encode as usize);
        }
        if let Some(f) = cm.module.get_function("ai_wire_decode_int") {
            engine.add_global_mapping(&f, crate::net::ai_wire_decode_int as usize);
        }
        if let Some(f) = cm.module.get_function("ai_wire_decode_ptr") {
            engine.add_global_mapping(&f, crate::net::ai_wire_decode_ptr as usize);
        }
        if let Some(f) = cm.module.get_function("ai_wire_decode_checked") {
            engine.add_global_mapping(&f, crate::net::ai_wire_decode_checked as usize);
        }
        if let Some(f) = cm.module.get_function("ai_wire_invoke") {
            engine.add_global_mapping(&f, crate::net::ai_wire_invoke as usize);
        }
        if let Some(box_fn) = cm.module.get_function("ai_gc_box_int") {
            engine.add_global_mapping(&box_fn, ai_gc_box_int as usize);
        }
        if let Some(unbox_fn) = cm.module.get_function("ai_gc_unbox_int") {
            engine.add_global_mapping(&unbox_fn, ai_gc_unbox_int as usize);
        }
        if let Some(fc_fn) = cm.module.get_function("ai_gc_force_collect") {
            engine.add_global_mapping(&fc_fn, ai_gc_force_collect as usize);
        }
        if let Some(f) = cm.module.get_function("ai_gc_pollcheck_slow") {
            engine.add_global_mapping(&f, ai_gc_pollcheck_slow as usize);
        }
        if let Some(f) = cm.module.get_function("ai_state_get") {
            engine.add_global_mapping(&f, ai_state_get as usize);
        }
        if let Some(f) = cm.module.get_function("ai_state_present") {
            engine.add_global_mapping(&f, ai_state_present as usize);
        }
        if let Some(f) = cm.module.get_function("ai_state_set") {
            engine.add_global_mapping(&f, ai_state_set as usize);
        }
        if let Some(f) = cm.module.get_function("ai_value_hash") {
            engine.add_global_mapping(&f, ai_value_hash as usize);
        }
        if let Some(f) = cm.module.get_function("ai_value_eq") {
            engine.add_global_mapping(&f, ai_value_eq as usize);
        }
        if let Some(f) = cm.module.get_function("ai_abort") {
            engine.add_global_mapping(&f, ai_abort as usize);
        }
        if let Some(f) = cm.module.get_function("ai_str_new") {
            engine.add_global_mapping(&f, ai_str_new as usize);
        }
        if let Some(f) = cm.module.get_function("ai_str_len") {
            engine.add_global_mapping(&f, ai_str_len as usize);
        }
        if let Some(f) = cm.module.get_function("ai_str_eq") {
            engine.add_global_mapping(&f, ai_str_eq as usize);
        }
        if let Some(f) = cm.module.get_function("ai_str_concat") {
            engine.add_global_mapping(&f, ai_str_concat as usize);
        }
        if let Some(f) = cm.module.get_function("ai_bytes_new") {
            engine.add_global_mapping(&f, ai_bytes_new as usize);
        }
        if let Some(f) = cm.module.get_function("ai_bytes_get") {
            engine.add_global_mapping(&f, ai_bytes_get as usize);
        }
        if let Some(f) = cm.module.get_function("ai_bytes_set") {
            engine.add_global_mapping(&f, ai_bytes_set as usize);
        }
        if let Some(f) = cm.module.get_function("ai_bytes_slice") {
            engine.add_global_mapping(&f, ai_bytes_slice as usize);
        }
        if let Some(f) = cm.module.get_function("ai_bytes_copy") {
            engine.add_global_mapping(&f, ai_bytes_copy as usize);
        }
        if let Some(f) = cm.module.get_function("ai_str_copy") {
            engine.add_global_mapping(&f, ai_str_copy as usize);
        }
        if let Some(f) = cm.module.get_function("ai_array_new") {
            engine.add_global_mapping(&f, ai_array_new as usize);
        }
        if let Some(f) = cm.module.get_function("ai_array_new_prim") {
            engine.add_global_mapping(&f, ai_array_new_prim as usize);
        }
        if let Some(f) = cm.module.get_function("ai_array_len") {
            engine.add_global_mapping(&f, ai_array_len as usize);
        }
        if let Some(f) = cm.module.get_function("ai_array_get") {
            engine.add_global_mapping(&f, ai_array_get as usize);
        }
        if let Some(f) = cm.module.get_function("ai_array_set") {
            engine.add_global_mapping(&f, ai_array_set as usize);
        }
        if let Some(f) = cm.module.get_function("ai_array_get_i64") {
            engine.add_global_mapping(&f, ai_array_get_i64 as usize);
        }
        if let Some(f) = cm.module.get_function("ai_array_set_i64") {
            engine.add_global_mapping(&f, ai_array_set_i64 as usize);
        }
        if let Some(f) = cm.module.get_function("ai_atom_swap_local") {
            engine.add_global_mapping(&f, ai_atom_swap_local as usize);
        }
        if let Some(f) = cm.module.get_function("ai_thread_spawn") {
            engine.add_global_mapping(&f, ai_thread_spawn as usize);
        }
        if let Some(f) = cm.module.get_function("ai_thread_spawn_shared") {
            engine.add_global_mapping(&f, ai_thread_spawn_shared as usize);
        }
        if let Some(f) = cm.module.get_function("ai_thread_join") {
            engine.add_global_mapping(&f, ai_thread_join as usize);
        }
        if let Some(f) = cm.module.get_function("ai_atom_new") {
            engine.add_global_mapping(&f, ai_atom_new as usize);
        }
        if let Some(f) = cm.module.get_function("ai_atom_load") {
            engine.add_global_mapping(&f, ai_atom_load as usize);
        }

        // User-defined `extern fn`s declared via the FFI registry.
        wire_user_externs_into(&engine, &cm.module);

        // Populate the runtime's code table: every JIT'd def + lifted
        // lambda is reachable by content hash.
        for (h, _fv) in cm.functions.iter() {
            let sym = def_symbol(h);
            // Safety: the symbol was just JIT'd; we trust the engine.
            let addr = engine
                .get_function_address(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym.clone() })?;
            runtime.code_table.insert(*h, addr as *const u8);
        }
        for (h, _fv) in cm.lifted_lambdas.iter() {
            let sym = lambda_symbol(h);
            let addr = engine
                .get_function_address(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym.clone() })?;
            runtime.code_table.insert(*h, addr as *const u8);
            // Mirror lambda → type_id mapping into the code table
            // so indirect calls can resolve `closure header type_id`
            // to the lambda's code hash.
            if let Some(&type_id) = cm.shape_registry.get(h) {
                runtime.code_table.register_type_id(type_id, *h);
            }
        }

        // ---- Mark cache-unsafe (stateful) thunks ----
        for h in &cm.stateful_hashes {
            runtime.mark_stateful(*h);
        }

        // ---- Run node `state` installers once, in dependency order ----
        // Externs are mapped and the code table is populated, so each
        // initializer's calls resolve. Installation is idempotent by hash.
        for h in &cm.state_hashes {
            let sym = state_init_symbol(h);
            let installer: JitFunction<'ctx, unsafe extern "C" fn(*mut Thread) -> i64> = unsafe {
                engine
                    .get_function(&sym)
                    .map_err(|_| CodegenError::FunctionNotFound { symbol: sym.clone() })?
            };
            // No panic channel: a contract violation in the initializer
            // aborts the process before this returns.
            unsafe { installer.call(runtime.thread_ptr()) };
        }

        Ok(Jit {
            engine,
            functions: cm.functions,
        })
    }

    pub unsafe fn get_fn1(
        &self,
        hash: &Hash,
    ) -> Result<JitFunction<'ctx, unsafe extern "C" fn(*mut Thread, i64) -> i64>, CodegenError>
    {
        let sym = def_symbol(hash);
        unsafe {
            self.engine
                .get_function(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym })
        }
    }

    pub unsafe fn get_fn2(
        &self,
        hash: &Hash,
    ) -> Result<
        JitFunction<'ctx, unsafe extern "C" fn(*mut Thread, i64, i64) -> i64>,
        CodegenError,
    > {
        let sym = def_symbol(hash);
        unsafe {
            self.engine
                .get_function(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym })
        }
    }

    pub unsafe fn get_fn0(
        &self,
        hash: &Hash,
    ) -> Result<JitFunction<'ctx, unsafe extern "C" fn(*mut Thread) -> i64>, CodegenError> {
        let sym = def_symbol(hash);
        unsafe {
            self.engine
                .get_function(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym })
        }
    }

    /// Look up a def whose declared return type is `fn(...) -> ...` — it
    /// returns a closure pointer (`*mut u8`).
    pub unsafe fn get_fn0_returning_closure(
        &self,
        hash: &Hash,
    ) -> Result<JitFunction<'ctx, unsafe extern "C" fn(*mut Thread) -> *mut u8>, CodegenError>
    {
        let sym = def_symbol(hash);
        unsafe {
            self.engine
                .get_function(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym })
        }
    }

    pub unsafe fn get_fn1_returning_closure(
        &self,
        hash: &Hash,
    ) -> Result<
        JitFunction<'ctx, unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>,
        CodegenError,
    > {
        let sym = def_symbol(hash);
        unsafe {
            self.engine
                .get_function(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym })
        }
    }
}

// =============================================================================
// IncrementalJit — install fetched code into a live JIT
// =============================================================================

/// A JIT that supports adding new code over time. Used by the code-fetch
/// handshake: the server boots with an empty knowledge set, then grows
/// it as clients ship over closures.
///
/// Unlike [`Jit`], which compiles one `CompiledModule` and is done,
/// `IncrementalJit` keeps the `Context` alive across multiple
/// `install` calls. Each call:
///
/// 1. Decodes the supplied canonical bytes into `Def`s and `Expr::Lambda`s.
/// 2. Builds a new LLVM module containing only the NEW items as bodies;
///    everything previously installed is declared `extern` so the JIT
///    can cross-link to the existing functions.
/// 3. Adds the module to the engine.
/// 4. Reads back each new function's address and registers it in the
///    `Runtime`'s `CodeTable`.
/// 5. Boxes new TypeInfos for stable addresses, pushes them into the
///    `Runtime`'s `type_infos` (the `*const TypeInfo` that
///    `ai_gc_alloc_closure` receives must remain valid forever), and
///    grows the `Heap`'s `type_table` via `dynamic_add_type`.
/// 6. Merges new `shape_registry` / `shape_meta` / `shape_by_type_id`
///    entries.
pub struct IncrementalJit<'ctx> {
    pub context: &'ctx Context,
    pub engine: ExecutionEngine<'ctx>,

    /// All defs installed so far, in install order. Used as the
    /// `external_defs` set when building a new install batch.
    installed_defs: Vec<ResolvedDef>,
    installed_defs_set: HashSet<Hash>,

    /// All extra-lambda hashes installed so far (those that came in as
    /// standalone `ItemKind::Lambda` items). Their `Expr::Lambda` bodies
    /// are also remembered so subsequent install batches can re-declare
    /// them as externals.
    installed_extra_lambdas: HashMap<Hash, Expr>,
    installed_lambdas_set: HashSet<Hash>,

    /// Extern requirements accumulated across installs (shipped as
    /// `ItemKind::Extern`). Used as the union module's externs map so
    /// `declare_user_externs` declares + resolves each referenced symbol
    /// (or fails clearly when the library/symbol isn't on this node).
    installed_externs: HashMap<String, crate::resolve::ExternSig>,

    /// Names provided per-def by the caller. Optional — purely for
    /// human-readable runtime errors. Mirrors `ResolvedDef.name` for
    /// installed defs.
    next_type_id: u16,
}

impl<'ctx> IncrementalJit<'ctx> {
    /// Build an IncrementalJit from an initial `CompiledModule`. The
    /// `CompiledModule` MUST be the source of the engine for this JIT
    /// (it is consumed). For a truly-empty start, build an empty
    /// `ResolvedModule { defs: vec![] }` and pass through
    /// `CompiledModule::build`.
    pub fn new(
        cm: CompiledModule<'ctx>,
        runtime: &Runtime,
    ) -> Result<Self, CodegenError> {
        let context = cm.context;

        // Strip the optimizer-pin global before creating the engine
        // (LLVM's JIT objects don't like @llvm.compiler.used).
        if let Some(used) = cm.module.get_global("llvm.compiler.used") {
            unsafe { used.delete() };
        }

        let engine = cm
            .module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| CodegenError::JitInit(e.to_string()))?;

        Self::wire_runtime_externs(&engine, &cm.module);

        // Populate code_table for every def + lambda in the initial module.
        for (h, _fv) in cm.functions.iter() {
            let sym = def_symbol(h);
            let addr = engine
                .get_function_address(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym.clone() })?;
            runtime.code_table.insert(*h, addr as *const u8);
        }
        for (h, _fv) in cm.lifted_lambdas.iter() {
            let sym = lambda_symbol(h);
            let addr = engine
                .get_function_address(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym.clone() })?;
            runtime.code_table.insert(*h, addr as *const u8);
            if let Some(&type_id) = cm.shape_registry.get(h) {
                runtime.code_table.register_type_id(type_id, *h);
            }
        }

        // Track type_id watermark = sum of all shapes registered in the
        // initial module PLUS reserved slots for the runtime-managed
        // shapes (BoxedInt + String + Array + Atom + Bytes + PrimArray)
        // at the trailing end of the heap's type-table. Keep this in sync
        // with `Runtime::new_with_metadata`.
        const RUNTIME_RESERVED_SHAPES: u16 = 6;
        let next_type_id =
            cm.closure_type_infos.len() as u16 + RUNTIME_RESERVED_SHAPES;

        Ok(IncrementalJit {
            context,
            engine,
            installed_defs: Vec::new(),
            installed_defs_set: HashSet::new(),
            installed_extra_lambdas: HashMap::new(),
            installed_lambdas_set: HashSet::new(),
            installed_externs: HashMap::new(),
            next_type_id,
        })
    }

    /// Every def installed so far, in install order. The deploy layer
    /// uses this to rebuild the union module for node-side typechecking.
    pub fn installed_defs(&self) -> &[ResolvedDef] {
        &self.installed_defs
    }

    /// Whether a def hash has already been installed.
    pub fn is_installed_def(&self, h: &Hash) -> bool {
        self.installed_defs_set.contains(h)
    }

    /// Extern requirements accumulated across installs.
    pub fn installed_externs(&self) -> &HashMap<String, crate::resolve::ExternSig> {
        &self.installed_externs
    }

    fn wire_runtime_externs(engine: &ExecutionEngine<'ctx>, module: &Module<'ctx>) {
        if let Some(alloc_fn) = module.get_function("ai_gc_alloc_closure") {
            engine.add_global_mapping(&alloc_fn, ai_gc_alloc_closure as usize);
        }
        if let Some(lookup_fn) = module.get_function("ai_gc_lookup_code") {
            engine.add_global_mapping(&lookup_fn, ai_gc_lookup_code as usize);
        }
        if let Some(net_at_fn) = module.get_function("ai_net_at") {
            engine.add_global_mapping(&net_at_fn, crate::net::ai_net_at as usize);
        }
        if let Some(f) = module.get_function("ai_net_at_async") {
            engine.add_global_mapping(&f, crate::net::ai_net_at_async as usize);
        }
        if let Some(f) = module.get_function("ai_wire_encode") {
            engine.add_global_mapping(&f, crate::net::ai_wire_encode as usize);
        }
        if let Some(f) = module.get_function("ai_wire_decode_int") {
            engine.add_global_mapping(&f, crate::net::ai_wire_decode_int as usize);
        }
        if let Some(f) = module.get_function("ai_wire_decode_ptr") {
            engine.add_global_mapping(&f, crate::net::ai_wire_decode_ptr as usize);
        }
        if let Some(f) = module.get_function("ai_wire_decode_checked") {
            engine.add_global_mapping(&f, crate::net::ai_wire_decode_checked as usize);
        }
        if let Some(f) = module.get_function("ai_wire_invoke") {
            engine.add_global_mapping(&f, crate::net::ai_wire_invoke as usize);
        }
        if let Some(box_fn) = module.get_function("ai_gc_box_int") {
            engine.add_global_mapping(&box_fn, ai_gc_box_int as usize);
        }
        if let Some(unbox_fn) = module.get_function("ai_gc_unbox_int") {
            engine.add_global_mapping(&unbox_fn, ai_gc_unbox_int as usize);
        }
        if let Some(fc_fn) = module.get_function("ai_gc_force_collect") {
            engine.add_global_mapping(&fc_fn, ai_gc_force_collect as usize);
        }
        if let Some(f) = module.get_function("ai_gc_pollcheck_slow") {
            engine.add_global_mapping(&f, ai_gc_pollcheck_slow as usize);
        }
        if let Some(f) = module.get_function("ai_state_get") {
            engine.add_global_mapping(&f, ai_state_get as usize);
        }
        if let Some(f) = module.get_function("ai_state_present") {
            engine.add_global_mapping(&f, ai_state_present as usize);
        }
        if let Some(f) = module.get_function("ai_state_set") {
            engine.add_global_mapping(&f, ai_state_set as usize);
        }
        if let Some(f) = module.get_function("ai_value_hash") {
            engine.add_global_mapping(&f, ai_value_hash as usize);
        }
        if let Some(f) = module.get_function("ai_value_eq") {
            engine.add_global_mapping(&f, ai_value_eq as usize);
        }
        if let Some(f) = module.get_function("ai_abort") {
            engine.add_global_mapping(&f, ai_abort as usize);
        }
        if let Some(f) = module.get_function("ai_str_new") {
            engine.add_global_mapping(&f, ai_str_new as usize);
        }
        if let Some(f) = module.get_function("ai_str_len") {
            engine.add_global_mapping(&f, ai_str_len as usize);
        }
        if let Some(f) = module.get_function("ai_str_eq") {
            engine.add_global_mapping(&f, ai_str_eq as usize);
        }
        if let Some(f) = module.get_function("ai_str_concat") {
            engine.add_global_mapping(&f, ai_str_concat as usize);
        }
        // Bytes + Array runtime ops (must mirror the non-incremental Jit
        // wiring — on Linux the ORC engine resolves these eagerly at
        // install, so a missing one aborts with "Symbol not found").
        if let Some(f) = module.get_function("ai_bytes_new") {
            engine.add_global_mapping(&f, ai_bytes_new as usize);
        }
        if let Some(f) = module.get_function("ai_bytes_get") {
            engine.add_global_mapping(&f, ai_bytes_get as usize);
        }
        if let Some(f) = module.get_function("ai_bytes_set") {
            engine.add_global_mapping(&f, ai_bytes_set as usize);
        }
        if let Some(f) = module.get_function("ai_bytes_slice") {
            engine.add_global_mapping(&f, ai_bytes_slice as usize);
        }
        if let Some(f) = module.get_function("ai_bytes_copy") {
            engine.add_global_mapping(&f, ai_bytes_copy as usize);
        }
        if let Some(f) = module.get_function("ai_str_copy") {
            engine.add_global_mapping(&f, ai_str_copy as usize);
        }
        if let Some(f) = module.get_function("ai_array_new") {
            engine.add_global_mapping(&f, ai_array_new as usize);
        }
        if let Some(f) = module.get_function("ai_array_new_prim") {
            engine.add_global_mapping(&f, ai_array_new_prim as usize);
        }
        if let Some(f) = module.get_function("ai_array_len") {
            engine.add_global_mapping(&f, ai_array_len as usize);
        }
        if let Some(f) = module.get_function("ai_array_get") {
            engine.add_global_mapping(&f, ai_array_get as usize);
        }
        if let Some(f) = module.get_function("ai_array_set") {
            engine.add_global_mapping(&f, ai_array_set as usize);
        }
        if let Some(f) = module.get_function("ai_array_get_i64") {
            engine.add_global_mapping(&f, ai_array_get_i64 as usize);
        }
        if let Some(f) = module.get_function("ai_array_set_i64") {
            engine.add_global_mapping(&f, ai_array_set_i64 as usize);
        }
        if let Some(f) = module.get_function("ai_atom_swap_local") {
            engine.add_global_mapping(&f, ai_atom_swap_local as usize);
        }
        if let Some(f) = module.get_function("ai_thread_spawn") {
            engine.add_global_mapping(&f, ai_thread_spawn as usize);
        }
        if let Some(f) = module.get_function("ai_thread_spawn_shared") {
            engine.add_global_mapping(&f, ai_thread_spawn_shared as usize);
        }
        if let Some(f) = module.get_function("ai_thread_join") {
            engine.add_global_mapping(&f, ai_thread_join as usize);
        }
        if let Some(f) = module.get_function("ai_atom_new") {
            engine.add_global_mapping(&f, ai_atom_new as usize);
        }
        if let Some(f) = module.get_function("ai_atom_load") {
            engine.add_global_mapping(&f, ai_atom_load as usize);
        }

        // ---- User-defined externs: walk every `ext/<name>` LLVM
        // function in the module and link it to its registered Rust
        // fn pointer. Module declarations the registry doesn't know
        // about are left unmapped; the first call into them will
        // hard-crash, which is the behavior we want — the test /
        // host needs to register before invoking.
        Self::wire_user_externs(engine, module);
    }

    fn wire_user_externs(engine: &ExecutionEngine<'ctx>, module: &Module<'ctx>) {
        wire_user_externs_into(engine, module);
    }

    /// Install a batch of fetched code items. Each item is
    /// `(ItemKind, hash, canonical_bytes)`:
    ///
    /// - `ItemKind::Def`: `bytes` decode to a `Def` (Fn/Struct/Enum). The
    ///   def is added to the live JIT under symbol `def_<hex>`.
    /// - `ItemKind::Lambda`: `bytes` decode to an `Expr::Lambda { ... }`.
    ///   The lambda is lifted, JIT'd under symbol `lambda_<hex>`, and
    ///   gets a `ShapeMeta::Closure` entry.
    ///
    /// Items are processed in the given order — callers should send
    /// dependencies before dependents (`KnowledgeBase::collect_transitive_deps`
    /// produces this order).
    ///
    /// Each batch is idempotent for items already installed (they're
    /// skipped silently). Re-installing a hash with different bytes is
    /// a programmer error and panics: content-addressing means two
    /// canonical encodings under the same hash must be byte-identical.
    pub fn install(
        &mut self,
        runtime: &mut Runtime,
        items: Vec<(crate::net::ItemKind, Hash, Vec<u8>)>,
    ) -> Result<(), CodegenError> {
        self.install_with(runtime, items, &HashSet::new())
    }

    /// Like [`install`](Self::install), but suppresses the `state`
    /// installer thunk for hashes in `skip_state_init`. The deploy path
    /// uses this for state cells it provides itself: a carried-over cell
    /// (aliased to the previous version's live cell) and a migrated cell
    /// (computed from the old value by a typechecked migration) must not
    /// have their initializers run — the initializer would create a
    /// fresh cell and orphan the data the deploy is preserving.
    pub fn install_with(
        &mut self,
        runtime: &mut Runtime,
        items: Vec<(crate::net::ItemKind, Hash, Vec<u8>)>,
        skip_state_init: &HashSet<Hash>,
    ) -> Result<(), CodegenError> {
        use crate::codec::{decode_def, decode_expr};
        use crate::net::ItemKind;

        // 1. Verify and decode. Reject duplicates that disagree with
        //    the installed bytes.
        let mut new_defs: Vec<ResolvedDef> = Vec::new();
        let mut new_lambdas: Vec<(Hash, Expr)> = Vec::new();

        for (kind, hash, bytes) in items {
            // We trust the caller-provided hash. Re-hashing the bytes
            // is unsafe for recursive-type SCC members whose stored
            // (TopRef-substituted) form doesn't round-trip the
            // resolver's canonical (SelfRef) hash. The protocol is
            // hash-key-by-fiat: both ends share content-addressing
            // by convention. End-to-end verification (a re-canonicalise
            // pass) is future work.
            match kind {
                ItemKind::Def => {
                    if self.installed_defs_set.contains(&hash) {
                        continue;
                    }
                    let def = decode_def(&bytes).map_err(|e| {
                        CodegenError::JitInit(format!("decode_def: {}", e))
                    })?;
                    let name = format!("installed_{}", &hash.to_hex()[..8]);
                    new_defs.push(ResolvedDef {
                        name,
                        hash,
                        def,
                    });
                }
                ItemKind::Lambda => {
                    if self.installed_lambdas_set.contains(&hash) {
                        continue;
                    }
                    let e = decode_expr(&bytes).map_err(|err| {
                        CodegenError::JitInit(format!("decode_expr (lambda): {}", err))
                    })?;
                    if !matches!(e, Expr::Lambda { .. }) {
                        return Err(CodegenError::JitInit(format!(
                            "install: ItemKind::Lambda payload didn't decode to Expr::Lambda (hash {})",
                            hash
                        )));
                    }
                    new_lambdas.push((hash, e));
                }
                ItemKind::Extern => {
                    // A C/host extern requirement travelling with the
                    // code. Decode and remember it; `declare_user_externs`
                    // (during the union build below) resolves the real
                    // symbol — or fails clearly if it isn't available here.
                    let (name, params, ret, library, variadic) =
                        crate::codec::decode_extern(&bytes).map_err(|e| {
                            CodegenError::JitInit(format!("decode_extern: {}", e))
                        })?;
                    self.installed_externs.insert(
                        name,
                        crate::resolve::ExternSig {
                            params,
                            ret,
                            library,
                            variadic,
                        },
                    );
                }
            }
        }

        if new_defs.is_empty() && new_lambdas.is_empty() {
            return Ok(());
        }

        // 2. Build a union ResolvedModule containing both old + new defs.
        //    External defs include all prior installs + initial-module defs
        //    (the latter we don't track here, but Jit::new already ran and
        //    JIT'd them — so they're addressable as `def_<hex>` symbols
        //    already, but our codegen needs ResolvedDef entries to learn
        //    the signatures). For the v1 demo, the initial module is
        //    empty, so the union is just installed + new.
        let mut union = ResolvedModule {
            defs: self.installed_defs.clone(),
            at_binding: None,
            // Carry every extern requirement received so far, so a def
            // installed in an earlier batch can still resolve its extern
            // when the union is recompiled. `declare_user_externs` only
            // resolves the symbols actually referenced by the union.
            externs: self.installed_externs.clone(),
        };
        union.defs.extend(new_defs.iter().cloned());

        let external_defs: HashSet<Hash> = self.installed_defs_set.clone();

        // 3. Extra-lambda set: previously installed + new.
        let mut extra_lambdas: Vec<(Hash, Expr)> = self
            .installed_extra_lambdas
            .iter()
            .map(|(h, e)| (*h, e.clone()))
            .collect();
        extra_lambdas.extend(new_lambdas.iter().cloned());

        // External lambdas = the previously installed extra-lambdas
        // PLUS any lambdas reachable from previously installed def
        // bodies (we don't have a separate tracker for those, but
        // re-scanning them isn't needed: scan_lambdas re-discovers
        // them in the union, and we mark them external via the
        // installed_defs hashes — but actually external_lambdas is
        // a separate set keyed by lambda hash).
        //
        // We approximate: the prior-installed extra-lambdas are
        // external. Lambdas inside previously installed def bodies
        // are re-discovered by scan_lambdas, but we also need them
        // declared as external since their JIT code is already in
        // the engine. Gather them now.
        let mut external_lambdas: HashSet<Hash> = self.installed_lambdas_set.clone();
        // Walk every previously installed def's body to find embedded
        // lambdas; those are already JIT'd.
        for rd in &self.installed_defs {
            match &rd.def {
                Def::Fn { body, .. } => collect_all_lambda_hashes(body, &mut external_lambdas),
                Def::State { init, .. } => collect_all_lambda_hashes(init, &mut external_lambdas),
                _ => {}
            }
        }

        // 4. Compile the delta.
        let type_id_base = self.next_type_id;
        let cm = CompiledModule::build_full(
            self.context,
            &union,
            &external_defs,
            &external_lambdas,
            &extra_lambdas,
            type_id_base,
        )?;

        // Strip optimizer-pin before handing off.
        if let Some(used) = cm.module.get_global("llvm.compiler.used") {
            unsafe { used.delete() };
        }

        // Pause every OTHER mutator thread (and block concurrent GC) for
        // the engine + table mutations below. `engine.add_module`, the
        // code-table writes, `type_infos`/`shape_*` growth, and
        // `dynamic_add_type` (which mutates the GC type table) are not safe
        // to run while another thread executes JIT'd code or the GC walks
        // the type table. The guard auto-resumes everyone on drop.
        //
        // State installers (step 6.5) and any other allocating / JIT-
        // executing work are deferred until AFTER the guard drops: running
        // them here would re-enter `gc_lock` (held by the pause) on GC and
        // deadlock. Held via a cloned Arc so it doesn't borrow `runtime`.
        let pause_heap = runtime.heap.clone();
        let world_pause = pause_heap.pause_world();

        // 5. Add the module to the engine.
        self.engine
            .add_module(&cm.module)
            .map_err(|()| CodegenError::JitInit(
                "engine.add_module failed".to_owned(),
            ))?;

        // The new module needs its own runtime-extern mappings — but
        // add_global_mapping is per-module: we only need to map the
        // EXTERN-declared functions in the NEW module. Map them now.
        Self::wire_runtime_externs(&self.engine, &cm.module);

        // 6. Register new function addresses in code_table.
        for rd in &new_defs {
            if matches!(
                rd.def,
                Def::Struct { .. } | Def::Enum { .. } | Def::State { .. }
            ) {
                // Structs/enums have no fn symbol; states use a separate
                // `state_init_<hex>` installer symbol, run in step 6.5.
                continue;
            }
            let sym = def_symbol(&rd.hash);
            let addr = self
                .engine
                .get_function_address(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym.clone() })?;
            runtime.code_table.insert(rd.hash, addr as *const u8);
        }

        // 6.4. Mark cache-unsafe (stateful) thunks. Recomputed over the
        //      union so transitive `TopRef` chains across batches resolve;
        //      marking is idempotent.
        for h in &cm.stateful_hashes {
            runtime.mark_stateful(*h);
        }

        // (Step 6.5 — node `state` installers — runs AFTER the world
        //  pause is dropped; see below. They allocate / execute JIT code.)
        for (h, _) in &new_lambdas {
            let sym = lambda_symbol(h);
            let addr = self
                .engine
                .get_function_address(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym.clone() })?;
            runtime.code_table.insert(*h, addr as *const u8);
            if let Some(&type_id) = cm.shape_registry.get(h) {
                runtime.code_table.register_type_id(type_id, *h);
            }
        }

        // 6.6. Register every lambda the union build lifted out of the new
        //      def bodies. Shipped (ItemKind::Lambda) items were handled
        //      above, but a lambda EMBEDDED in a shipped def's body is
        //      compiled here for the first time and needs its hash -> addr
        //      entry too, or the first indirect call through a closure it
        //      produces (`ai_gc_lookup_code`) panics with an unregistered
        //      hash. Previously-installed lambdas resolve to their original
        //      address (they're declared external), so re-inserting is a
        //      no-op.
        for (h, _fv) in cm.lifted_lambdas.iter() {
            let sym = lambda_symbol(h);
            if let Ok(addr) = self.engine.get_function_address(&sym) {
                runtime.code_table.insert(*h, addr as *const u8);
                if let Some(&type_id) = cm.shape_registry.get(h) {
                    runtime.code_table.register_type_id(type_id, *h);
                }
            }
        }

        // 7. Push new TypeInfos into runtime + heap.
        //    cm.closure_type_infos lists the new shapes in this batch,
        //    in the order their type_ids were assigned. Type_id of
        //    shape at index `i` is `type_id_base + i`.
        for (i, ti) in cm.closure_type_infos.iter().enumerate() {
            let expected_id = type_id_base + i as u16;
            debug_assert_eq!(
                ti.type_id, expected_id,
                "TypeInfo type_id should match its slot offset"
            );
            // Push into runtime's stable-pointer table.
            runtime.type_infos.push(Box::new(*ti));
            // Grow Heap's type_table.
            let new_id = unsafe { runtime.heap.dynamic_add_type(*ti) };
            debug_assert_eq!(new_id, expected_id);
        }

        // 8. Merge shape_registry / shape_meta / shape_by_type_id.
        for (h, type_id) in &cm.shape_registry {
            runtime.shape_registry.insert(*h, *type_id);
        }
        for (h, meta) in &cm.shape_meta {
            runtime.shape_meta.insert(*h, meta.clone());
        }
        // shape_by_type_id from this batch contains entries indexed by
        // absolute type_id (with None placeholders for the prior range).
        // Extend the runtime's shape_by_type_id to match, AND mirror
        // each new entry into the code table's type_id_to_hash so
        // indirect calls (`ai_gc_lookup_code`) can resolve closures
        // by type_id alone. Without this, a closure shipped to the
        // worker has a type_id the worker's code table doesn't know
        // about, and the next indirect call panics.
        for (i, slot) in cm.shape_by_type_id.iter().enumerate() {
            if let Some(h) = slot {
                while runtime.shape_by_type_id.len() <= i {
                    runtime.shape_by_type_id.push(None);
                }
                runtime.shape_by_type_id[i] = Some(*h);
                runtime.code_table.register_type_id(i as u16, *h);
            }
        }

        // End of the no-allocation critical section: resume all paused
        // mutator threads and release `gc_lock` before any JIT/allocating
        // work below.
        drop(world_pause);

        // 6.5 (deferred). Run node `state` installers for freshly-installed
        //      states, in batch order (dependency-first). Idempotent by
        //      hash: a state the node already has is a no-op, so a shipped
        //      handler never clobbers the node's existing cell. This is what
        //      makes `at(node, || handler(msg))` mutate the node's OWN
        //      state. Deferred to here because it executes JIT'd code and
        //      allocates — both forbidden under the world pause, and both
        //      now able to see every type/shape/code registered above.
        for rd in &new_defs {
            if !matches!(rd.def, Def::State { .. }) {
                continue;
            }
            if skip_state_init.contains(&rd.hash) {
                // The deploy that requested this install provides the
                // cell itself (carryover alias or migration result).
                continue;
            }
            let sym = state_init_symbol(&rd.hash);
            let addr = self
                .engine
                .get_function_address(&sym)
                .map_err(|_| CodegenError::FunctionNotFound { symbol: sym.clone() })?;
            let installer: unsafe extern "C" fn(*mut Thread) -> i64 =
                unsafe { core::mem::transmute(addr) };
            // No panic channel: a contract violation in the initializer
            // aborts the process before this returns.
            unsafe { installer(runtime.thread_ptr()) };
        }

        // 9. Update IncrementalJit state.
        self.next_type_id = self
            .next_type_id
            .checked_add(cm.closure_type_infos.len() as u16)
            .expect("type_id watermark overflow");
        for rd in new_defs {
            self.installed_defs_set.insert(rd.hash);
            self.installed_defs.push(rd);
        }
        for (h, e) in new_lambdas {
            self.installed_lambdas_set.insert(h);
            self.installed_extra_lambdas.insert(h, e);
        }

        Ok(())
    }
}

/// Walk an expression, recording the content hash of every `Expr::Lambda`
/// node encountered. Used by `IncrementalJit::install` to mark previously
/// JIT'd lifted lambdas as externally-defined.
fn collect_all_lambda_hashes(e: &Expr, out: &mut HashSet<Hash>) {
    match e {
        Expr::Lambda { params, body } => {
            let lambda_expr = Expr::Lambda {
                params: params.clone(),
                body: body.clone(),
            };
            let bytes = encode_expr(&lambda_expr);
            out.insert(Hash::of_bytes(&bytes));
            collect_all_lambda_hashes(body, out);
        }
        Expr::Call(callee, args) => {
            collect_all_lambda_hashes(callee, out);
            for a in args {
                collect_all_lambda_hashes(a, out);
            }
        }
        Expr::Let { value, body } => {
            collect_all_lambda_hashes(value, out);
            collect_all_lambda_hashes(body, out);
        }
        Expr::StructNew { fields, .. } => {
            for f in fields {
                collect_all_lambda_hashes(f, out);
            }
        }
        Expr::Field { base, .. } => collect_all_lambda_hashes(base, out),
        Expr::EnumNew { payload, .. } => {
            if let Some(p) = payload {
                collect_all_lambda_hashes(p, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_all_lambda_hashes(scrutinee, out);
            for arm in arms {
                collect_all_lambda_hashes(&arm.body, out);
            }
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_all_lambda_hashes(cond, out);
            collect_all_lambda_hashes(then_branch, out);
            collect_all_lambda_hashes(else_branch, out);
        }
        Expr::Try { expr, .. } => collect_all_lambda_hashes(expr, out),
        Expr::Defer { cleanup, body } => {
            collect_all_lambda_hashes(cleanup, out);
            collect_all_lambda_hashes(body, out);
        }
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::LocalVar(_)
        | Expr::TopRef(_)
        | Expr::SelfRef(_)
        | Expr::StateRef(_)
        | Expr::StateSelfRef(_)
        | Expr::BuiltinRef(_) => {}
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn init() {
        INIT.call_once(|| {
            init_native_target().expect("init native target");
        });
    }

    /// Restore the extern baseline after a test that registered its own
    /// externs. The I/O externs live in a process-global write-once table
    /// (always present, never cleared), so "baseline" just means dropping
    /// this thread's dynamic overrides. No cross-test race is possible:
    /// clearing only touches the calling thread's thread-local registry.
    fn reset_externs_to_baseline() {
        crate::ffi::clear_externs();
        crate::io_externs::register_io_externs();
    }

    fn build_for<'ctx>(
        ctx: &'ctx Context,
        src: &str,
    ) -> (Runtime, Jit<'ctx>, HashMap<String, Hash>) {
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let names: HashMap<String, Hash> = r
            .defs
            .iter()
            .map(|d| (d.name.clone(), d.hash))
            .collect();
        let cm = CompiledModule::build(ctx, &r).unwrap();
        let rt = Runtime::new_with_registry(cm.closure_type_infos.clone(), cm.shape_registry.clone());
        let jit = Jit::new(cm, &rt).unwrap();
        (rt, jit, names)
    }

    // ---- Int-only tests (regression: existing behaviour after refactor) ----

    #[test]
    fn jit_double_of_21_is_42() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(&ctx, "def double(x: Int) -> Int = x * 2");
        unsafe {
            let f = jit.get_fn1(&names["double"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 21), 42);
            assert_eq!(f.call(rt.thread_ptr(), -3), -6);
        }
    }

    #[test]
    fn jit_quadruple_calls_double_by_hash() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            def double(x: Int) -> Int = x * 2
            def quadruple(x: Int) -> Int = double(double(x))
        ",
        );
        unsafe {
            let q = jit.get_fn1(&names["quadruple"]).unwrap();
            assert_eq!(q.call(rt.thread_ptr(), 3), 12);
        }
    }

    /// The GC safepoint poll emitted at every `loop_body` entry must
    /// actually fire when `Thread.state` is non-zero, trampoline into
    /// `ai_gc_pollcheck_slow`, and resume the loop without corrupting it.
    /// Before this task the poll was never emitted and the slow path was
    /// dead code; this is the deterministic regression test that the poll
    /// is wired end-to-end. (The genuine multi-thread STW coordination —
    /// one thread parking a sibling spinning in this loop — lands with the
    /// per-thread `Thread` task; here we drive the same code path single-
    /// threaded by pre-arming the safepoint release.)
    #[test]
    fn safepoint_poll_fires_and_resumes_the_loop() {
        init();
        let ctx = Context::create();
        // Self-tail-recursive countdown: each turn re-enters `loop_body`
        // (the backedge), so the safepoint poll runs every iteration.
        let (rt, jit, names) = build_for(
            &ctx,
            "def countdown(n: Int) -> Int = if n == 0 { 99 } else { countdown(n - 1) }",
        );
        unsafe {
            let f = jit.get_fn1(&names["countdown"]).unwrap();
            let thread = rt.thread_ptr();

            // Pre-arm the release so the slow path's `enter_safepoint()`
            // returns immediately (no coordinator thread in this unit
            // test), then request a safepoint. The next inline poll sees
            // `state != 0` and trampolines into `ai_gc_pollcheck_slow`.
            rt.dyna_thread.resume();
            (*thread).state = 1;

            // Wired correctly: the loop runs to completion (99) and the
            // slow path clears the state byte. A missing poll would leave
            // state == 1; a slow path that clobbered the frame/loop would
            // produce a wrong result or crash.
            let result = f.call(thread, 5);
            assert_eq!(result, 99, "countdown result after safepoint poll");
            assert_eq!((*thread).state, 0, "slow path must clear the state byte");
        }
    }

    /// End-to-end proof of tasks #1 + #2: a worker OS thread running an
    /// allocation-free self-tail loop on its OWN `ThreadContext` can be
    /// stopped for a stop-the-world collection requested by another
    /// thread. The coordinator raises the worker's JIT poll flag; the
    /// inline poll parks the worker; the collector scans and resumes it.
    ///
    /// This is deterministic, not timing-dependent: if the poll were not
    /// wired, `mutator_triggered_gc` would spin forever in
    /// `is_safely_at_safepoint` (the worker never observing the request)
    /// and the test would hang. Completion IS the proof; the `99` result
    /// confirms the STW pause didn't corrupt the worker's loop.
    #[test]
    fn stw_stops_a_worker_spinning_in_a_jit_loop() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def countdown(n: Int) -> Int = if n == 0 { 99 } else { countdown(n - 1) }",
        );
        // Raw, Send-able address of the JIT'd function. The `jit` (and
        // its execution engine that owns the code) stays alive in scope.
        let fn_addr: usize = jit
            .engine
            .get_function_address(&def_symbol(&names["countdown"]))
            .expect("countdown address");

        // RuntimeHandle is Send+Sync so the worker can borrow the runtime
        // (its heap, code table, shapes) across the thread boundary. The
        // runtime owns everything the JIT'd code dereferences, so it must
        // outlive the worker — guaranteed here by `scope`.
        let handle = crate::net::RuntimeHandle(rt);
        // `&RuntimeHandle` is Send (RuntimeHandle is unsafe-Sync), whereas
        // `&Runtime` is not — so the worker captures the handle reference
        // as a whole rather than disjointly borrowing the inner Runtime.
        let hp: &crate::net::RuntimeHandle = &handle;

        use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
        let started = AtomicBool::new(false);
        let result = AtomicI64::new(0);

        std::thread::scope(|scope| {
            scope.spawn(|| {
                let hp = hp; // capture the &RuntimeHandle by copy (Send)
                let rt = &hp.0;
                // The worker's OWN execution context: its own shadow-stack
                // head + its own registered GC ThreadState.
                let wctx = rt.new_thread_context();
                let f: unsafe extern "C" fn(*mut crate::runtime::Thread, i64) -> i64 =
                    unsafe { std::mem::transmute(fn_addr) };
                started.store(true, Ordering::Release);
                // ~10^8 allocation-free iterations: long enough that the
                // coordinator's STW request lands while we're still
                // spinning. Each iteration runs the safepoint poll.
                let r = unsafe { f(wctx.thread_ptr(), 100_000_000) };
                result.store(r, Ordering::Release);
            });

            // Once the worker is in its loop, demand a stop-the-world
            // collection from this (the home) thread. Blocks until the
            // worker parks at a safepoint poll, then resumes it.
            while !started.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            unsafe {
                hp.0.heap
                    .mutator_triggered_gc::<crate::gc::IdentityPtrPolicy>(&hp.0.dyna_thread);
            }
        });

        assert_eq!(
            result.load(Ordering::Acquire),
            99,
            "worker loop result after concurrent STW collection"
        );
    }

    /// Task #6: a registered thread parked in a blocking syscall (modeled
    /// here by an explicit `enter_blocked` region, exactly what
    /// `net::blocking_accept` and the server frame reads now do) must NOT
    /// stall a stop-the-world collection. The collector scans a
    /// STATE_BLOCKED thread's roots in place instead of busy-waiting for
    /// it to poll — which it never would while parked in `accept`/`read`.
    ///
    /// Deterministic: if blocked threads were not scanned-in-place,
    /// `mutator_triggered_gc` would spin forever in `is_safely_at_safepoint`
    /// and the test would hang. Completion is the proof.
    #[test]
    fn stw_does_not_hang_on_a_blocked_registered_thread() {
        init();
        let ctx = Context::create();
        let (rt, _jit, _names) = build_for(&ctx, "def id(x: Int) -> Int = x");
        let handle = crate::net::RuntimeHandle(rt);
        let hp: &crate::net::RuntimeHandle = &handle;

        use std::sync::atomic::{AtomicBool, Ordering};
        let blocked = AtomicBool::new(false);
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        // `Receiver` is !Sync so it must be moved into the worker; pre-bind
        // a Copy reference to the shared flag so `move` doesn't take it.
        let blocked_ref = &blocked;

        std::thread::scope(|scope| {
            scope.spawn(move || {
                let hp = hp;
                let rx = rx;
                let wctx = hp.0.new_thread_context();
                // Register + enter a blocked region (as a server thread
                // parked in accept would), announce it, then wait for
                // release while STATE_BLOCKED.
                wctx.dyna_thread().enter_blocked();
                blocked_ref.store(true, Ordering::Release);
                rx.recv().unwrap();
                wctx.dyna_thread().exit_blocked(&hp.0.heap);
            });

            while !blocked.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            // Worker is registered AND blocked: this must complete.
            unsafe {
                hp.0.heap
                    .mutator_triggered_gc::<crate::gc::IdentityPtrPolicy>(&hp.0.dyna_thread);
            }
            tx.send(()).unwrap();
        });
    }

    #[test]
    fn jit_let_binding() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def f(x: Int) -> Int = { let y = x * 2; y + 1 }",
        );
        unsafe {
            let f = jit.get_fn1(&names["f"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 3), 7);
        }
    }

    // ---- Closures ----

    #[test]
    fn jit_closure_no_captures_invoked_locally() {
        init();
        let ctx = Context::create();
        // |x: Int| x + 1 — no captures.
        let (rt, jit, names) = build_for(
            &ctx,
            "def run(x: Int) -> Int = {
                let f = |y: Int| y + 1;
                f(x)
            }",
        );
        unsafe {
            let r = jit.get_fn1(&names["run"]).unwrap();
            assert_eq!(r.call(rt.thread_ptr(), 41), 42);
        }
    }

    #[test]
    fn jit_closure_one_capture_invoked_locally() {
        init();
        let ctx = Context::create();
        // Closure captures `n` from the surrounding def.
        let (rt, jit, names) = build_for(
            &ctx,
            "def add_n(n: Int, x: Int) -> Int = {
                let f = |y: Int| y + n;
                f(x)
            }",
        );
        unsafe {
            let r = jit.get_fn2(&names["add_n"]).unwrap();
            assert_eq!(r.call(rt.thread_ptr(), 10, 5), 15);
            assert_eq!(r.call(rt.thread_ptr(), 100, 7), 107);
        }
    }

    #[test]
    fn jit_closure_multiple_captures() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def compute(a: Int, b: Int, x: Int) -> Int = {
                let f = |y: Int| (y + a) * b;
                f(x)
            }",
        );
        unsafe {
            let r = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64, i64) -> i64>(
                    &def_symbol(&names["compute"]),
                )
                .unwrap();
            // f(x) = (x + a) * b
            assert_eq!(r.call(rt.thread_ptr(), 1, 2, 3), 8); // (3+1)*2
            assert_eq!(r.call(rt.thread_ptr(), 10, 5, 4), 70); // (4+10)*5
        }
    }

    #[test]
    fn jit_def_returns_closure() {
        init();
        let ctx = Context::create();
        // The classic make_adder: def returns a closure capturing n.
        let src = "
            def make_adder(n: Int) -> fn(Int) -> Int = |x: Int| x + n
            def run(n: Int, x: Int) -> Int = {
                let f = make_adder(n);
                f(x)
            }
        ";
        let (rt, jit, names) = build_for(&ctx, src);
        unsafe {
            let run = jit.get_fn2(&names["run"]).unwrap();
            assert_eq!(run.call(rt.thread_ptr(), 10, 5), 15);
            assert_eq!(run.call(rt.thread_ptr(), 100, 42), 142);
        }
    }

    #[test]
    fn nested_lambda_errors_clearly() {
        init();
        // `|x: Int| |y: Int| x + y` — nested. v1 rejects this.
        let src = "def f() -> fn(Int) -> fn(Int) -> Int = |x: Int| |y: Int| x + y";
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let ctx = Context::create();
        match CompiledModule::build(&ctx, &r) {
            Err(CodegenError::Unsupported { ref what }) if what.contains("nested") => {}
            other => panic!("expected nested-lambda error, got {:?}", other.as_ref().err()),
        }
    }

    #[test]
    fn jit_zero_arg_function() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(&ctx, "def answer() -> Int = 42");
        unsafe {
            let f = jit.get_fn0(&names["answer"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 42);
        }
    }

    #[test]
    fn jit_let_shadows_param() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def f(x: Int) -> Int = {
                let x = x * 10;
                x + 1
            }",
        );
        unsafe {
            let f = jit.get_fn1(&names["f"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 4), 41);
        }
    }

    #[test]
    fn ir_contains_shadow_stack_machinery() {
        init();
        // A pointer-typed param forces a GC frame (it must be rooted).
        let m =
            parse_module("def f(s: String) -> Int = string_len(s)").unwrap();
        let r = resolve_module(&m).unwrap();
        let ctx = Context::create();
        let cm = CompiledModule::build(&ctx, &r).unwrap();
        let ir = cm.ir();
        assert!(
            ir.contains("__frame_origin_def_"),
            "IR missing FrameOrigin global"
        );
        assert!(ir.contains("gc_frame"), "IR missing gc_frame alloca");
        assert!(
            ir.contains("llvm.compiler.used"),
            "IR missing llvm.compiler.used pin"
        );
    }

    #[test]
    fn scalar_only_def_skips_gc_frame() {
        init();
        // An all-scalar body provably never holds a heap pointer, so
        // its prologue must push NO frame at all (the fib fast path).
        let m = parse_module(
            "def f(x: Int) -> Int = if x < 2 { x } else { f(x - 1) + f(x - 2) }",
        )
        .unwrap();
        let r = resolve_module(&m).unwrap();
        let ctx = Context::create();
        let cm = CompiledModule::build(&ctx, &r).unwrap();
        let ir = cm.ir();
        assert!(
            !ir.contains("gc_frame"),
            "scalar-only def should not alloca a GC frame:\n{}",
            ir
        );
    }

    #[test]
    fn jit_many_closure_allocations_survive_gc_pressure() {
        // Stress test: allocate many closures back-to-back to force at
        // least one GC cycle. The shadow stack must keep alive any
        // closures whose pointers are held in let bindings (root slots).
        //
        // This program builds a closure capturing n, calls it once,
        // builds another closure capturing the result, calls THAT, etc.
        // Each iteration is recursion through TopRefs, allocating one
        // closure per call. With a small heap (32 MiB) and ~1M calls,
        // we'd OOM without GC. With GC, only one closure is live at a
        // time (the current call's f), so we never run out.
        init();
        let ctx = Context::create();
        let src = "
            def step(n: Int, x: Int) -> Int = {
                let f = |y: Int| y + n;
                f(x)
            }
            def run(iters: Int, x: Int) -> Int = {
                let v = step(1, x);
                if_zero_or_recurse(iters, x, v)
            }
            def if_zero_or_recurse(iters: Int, x: Int, v: Int) -> Int = run(iters - 1, v)
        ";
        // Without `if` in the language we can't write a real loop; the
        // test below issues 1000 calls from the host side instead. Each
        // call to `step` allocates a closure on the heap.
        let m = parse_module(
            "def step(n: Int, x: Int) -> Int = {
                let f = |y: Int| y + n;
                f(x)
            }",
        )
        .unwrap();
        let _ = src;
        let r = resolve_module(&m).unwrap();
        let cm = CompiledModule::build(&ctx, &r).unwrap();
        let rt = Runtime::new_with_registry(cm.closure_type_infos.clone(), cm.shape_registry.clone());
        let names: HashMap<String, Hash> =
            r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
        let jit = Jit::new(cm, &rt).unwrap();

        // Force GC on every alloc so even tiny allocations trigger a
        // collection. Exercises the root-walk for every test iteration.
        rt.heap.set_gc_every_alloc(true);

        unsafe {
            let step = jit.get_fn2(&names["step"]).unwrap();
            let mut acc = 0i64;
            for _ in 0..1000 {
                acc = step.call(rt.thread_ptr(), 1, acc);
            }
            assert_eq!(acc, 1000);
        }
    }

    // ---- Structs ----

    #[test]
    fn jit_struct_construction_and_field_access() {
        init();
        let ctx = Context::create();
        let src = "
            struct Point { x: Int, y: Int }
            def get_x(p: Point) -> Int = p.x
            def make(a: Int, b: Int) -> Int = {
                let p = Point { x: a, y: b };
                get_x(p) + p.y
            }
        ";
        let (rt, jit, names) = build_for(&ctx, src);
        unsafe {
            let f = jit.get_fn2(&names["make"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 10, 5), 15); // 10 + 5
            assert_eq!(f.call(rt.thread_ptr(), 100, 23), 123);
        }
    }

    #[test]
    fn jit_struct_with_three_fields() {
        init();
        let ctx = Context::create();
        let src = "
            struct V3 { x: Int, y: Int, z: Int }
            def sum(v: V3) -> Int = v.x + v.y + v.z
            def build(a: Int, b: Int, c: Int) -> Int = {
                let v = V3 { x: a, y: b, z: c };
                sum(v)
            }
        ";
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let cm = CompiledModule::build(&ctx, &r).unwrap();
        let rt = Runtime::new_with_registry(cm.closure_type_infos.clone(), cm.shape_registry.clone());
        let h = r.get("build").unwrap().hash;
        let jit = Jit::new(cm, &rt).unwrap();
        unsafe {
            let f = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64, i64) -> i64>(
                    &def_symbol(&h),
                )
                .unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 1, 2, 3), 6);
            assert_eq!(f.call(rt.thread_ptr(), 10, 20, 30), 60);
        }
    }

    #[test]
    fn jit_struct_fields_in_any_order_at_literal_site() {
        // The literal can list fields in a different order from the
        // declaration; the resolver reorders to canonical positions.
        init();
        let ctx = Context::create();
        let src = "
            struct P { x: Int, y: Int }
            def run() -> Int = {
                let p = P { y: 100, x: 7 };
                p.x
            }
        ";
        let (rt, jit, names) = build_for(&ctx, src);
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 7);
        }
    }

    #[test]
    fn struct_literal_missing_field_errors() {
        let src = "
            struct P { x: Int, y: Int }
            def f() -> Int = {
                let p = P { x: 1 };
                p.x
            }
        ";
        let m = parse_module(src).unwrap();
        let err = resolve_module(&m).unwrap_err();
        assert!(
            matches!(&err, crate::resolve::ResolveError::MissingField { field, .. } if field == "y"),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn struct_literal_unknown_field_errors() {
        let src = "
            struct P { x: Int }
            def f() -> Int = {
                let p = P { x: 1, z: 2 };
                p.x
            }
        ";
        let m = parse_module(src).unwrap();
        let err = resolve_module(&m).unwrap_err();
        assert!(
            matches!(&err, crate::resolve::ResolveError::UnknownField { field, .. } if field == "z"),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn field_access_on_non_struct_errors() {
        let src = "def f(x: Int) -> Int = x.foo";
        let m = parse_module(src).unwrap();
        let err = resolve_module(&m).unwrap_err();
        assert!(
            matches!(err, crate::resolve::ResolveError::FieldOnNonStruct { .. }),
            "got: {:?}",
            err
        );
    }

    #[test]
    fn struct_with_pointer_field_traces_correctly_under_gc_pressure() {
        // Struct holding a closure pointer. Force GC on every alloc;
        // the struct must keep the closure alive across GCs (the GC
        // walks the struct's value-fields, which is where the closure
        // pointer sits).
        init();
        let ctx = Context::create();
        let src = "
            struct Box { val: fn(Int) -> Int, k: Int }
            def make(n: Int) -> Box = {
                let f = |x: Int| x + n;
                Box { val: f, k: n * 10 }
            }
            def run(n: Int, x: Int) -> Int = {
                let b = make(n);
                b.val(x) + b.k
            }
        ";
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let cm = CompiledModule::build(&ctx, &r).unwrap();
        let rt = Runtime::new_with_registry(cm.closure_type_infos.clone(), cm.shape_registry.clone());
        let h = r.get("run").unwrap().hash;
        let jit = Jit::new(cm, &rt).unwrap();

        rt.heap.set_gc_every_alloc(true);

        unsafe {
            let f = jit.get_fn2(&h).unwrap();
            // run(n=5, x=2): closure f(x) = x + 5 = 7; b.k = 50; total = 57.
            assert_eq!(f.call(rt.thread_ptr(), 5, 2), 57);
        }
    }

    // ---- Enums + match ----

    #[test]
    fn jit_enum_with_int_payload() {
        init();
        let ctx = Context::create();
        let src = "
            enum IntOpt { Some(Int), None }
            def get_or(o: IntOpt, default: Int) -> Int = match o {
                IntOpt::Some(x) => x,
                IntOpt::None => default,
            }
            def use_some(v: Int, default: Int) -> Int = {
                let o = IntOpt::Some(v);
                get_or(o, default)
            }
            def use_none(default: Int) -> Int = {
                let o = IntOpt::None;
                get_or(o, default)
            }
        ";
        let (rt, jit, names) = build_for(&ctx, src);
        unsafe {
            let some = jit.get_fn2(&names["use_some"]).unwrap();
            assert_eq!(some.call(rt.thread_ptr(), 42, 0), 42);
            let none = jit.get_fn1(&names["use_none"]).unwrap();
            assert_eq!(none.call(rt.thread_ptr(), 99), 99);
        }
    }

    #[test]
    fn jit_enum_with_pointer_payload() {
        init();
        let ctx = Context::create();
        // Enum payload is a struct (heap pointer). Match extracts the
        // payload and uses it.
        let src = "
            struct Point { x: Int, y: Int }
            enum Located { Here(Point), Nowhere }
            def x_or(l: Located, default: Int) -> Int = match l {
                Located::Here(p) => p.x,
                Located::Nowhere => default,
            }
            def make_here(a: Int, b: Int, default: Int) -> Int = {
                let p = Point { x: a, y: b };
                let l = Located::Here(p);
                x_or(l, default)
            }
        ";
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let cm = CompiledModule::build(&ctx, &r).unwrap();
        let rt = Runtime::new_with_registry(cm.closure_type_infos.clone(), cm.shape_registry.clone());
        let h = r.get("make_here").unwrap().hash;
        let jit = Jit::new(cm, &rt).unwrap();
        unsafe {
            let f = jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64, i64) -> i64>(
                    &def_symbol(&h),
                )
                .unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 7, 9, -1), 7);
        }
    }

    #[test]
    fn jit_enum_with_three_variants() {
        init();
        let ctx = Context::create();
        let src = "
            enum Color { Red, Green, Blue }
            def code(c: Color) -> Int = match c {
                Color::Red => 1,
                Color::Green => 2,
                Color::Blue => 3,
            }
            def red() -> Int = code(Color::Red)
            def green() -> Int = code(Color::Green)
            def blue() -> Int = code(Color::Blue)
        ";
        let (rt, jit, names) = build_for(&ctx, src);
        unsafe {
            assert_eq!(jit.get_fn0(&names["red"]).unwrap().call(rt.thread_ptr()), 1);
            assert_eq!(jit.get_fn0(&names["green"]).unwrap().call(rt.thread_ptr()), 2);
            assert_eq!(jit.get_fn0(&names["blue"]).unwrap().call(rt.thread_ptr()), 3);
        }
    }

    #[test]
    fn jit_enum_wildcard_arm() {
        init();
        let ctx = Context::create();
        let src = "
            enum Choice { A(Int), B(Int), C(Int) }
            def is_b(c: Choice) -> Int = match c {
                Choice::B(_) => 1,
                _ => 0,
            }
            def test_a() -> Int = is_b(Choice::A(10))
            def test_b() -> Int = is_b(Choice::B(20))
            def test_c() -> Int = is_b(Choice::C(30))
        ";
        let (rt, jit, names) = build_for(&ctx, src);
        unsafe {
            assert_eq!(jit.get_fn0(&names["test_a"]).unwrap().call(rt.thread_ptr()), 0);
            assert_eq!(jit.get_fn0(&names["test_b"]).unwrap().call(rt.thread_ptr()), 1);
            assert_eq!(jit.get_fn0(&names["test_c"]).unwrap().call(rt.thread_ptr()), 0);
        }
    }

    #[test]
    fn jit_enum_survives_gc_pressure() {
        // Allocate an enum holding a pointer payload (struct), under
        // gc_every_alloc. The enum object's value_field[0] holds the
        // payload pointer and must be traced by the GC.
        init();
        let ctx = Context::create();
        let src = "
            struct P { v: Int }
            enum Maybe { Some(P), None }
            def make_some(v: Int) -> Maybe = Maybe::Some(P { v: v })
            def unwrap_or(m: Maybe, default: Int) -> Int = match m {
                Maybe::Some(p) => p.v,
                Maybe::None => default,
            }
            def run(v: Int) -> Int = {
                let m = make_some(v);
                unwrap_or(m, -1)
            }
        ";
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let cm = CompiledModule::build(&ctx, &r).unwrap();
        let rt = Runtime::new_with_registry(cm.closure_type_infos.clone(), cm.shape_registry.clone());
        let h = r.get("run").unwrap().hash;
        let jit = Jit::new(cm, &rt).unwrap();
        rt.heap.set_gc_every_alloc(true);
        unsafe {
            let f = jit.get_fn1(&h).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 42), 42);
            assert_eq!(f.call(rt.thread_ptr(), 7), 7);
        }
    }

    #[test]
    fn jit_nested_enum_patterns() {
        // Nested enum patterns: match through two enum layers in one arm, and
        // let multiple arms share an outer variant (`A(I)` vs `A(J)`). A flat
        // switch on the outer tag can't express this; the backtracking matcher
        // can. Run under GC pressure so the heap-pointer payloads are traced.
        init();
        let ctx = Context::create();
        let src = "
            enum Inner { I(Int), J }
            enum Outer { A(Inner), B(Inner), C }
            def classify(o: Outer) -> Int = match o {
                Outer::A(Inner::I(n)) => n,
                Outer::A(Inner::J) => 0 - 7,
                Outer::B(Inner::I(n)) => n + 100,
                Outer::B(Inner::J) => 0 - 8,
                Outer::C => 0 - 9,
            }
            def run_ai(n: Int) -> Int = classify(Outer::A(Inner::I(n)))
            def run_aj() -> Int = classify(Outer::A(Inner::J))
            def run_bi(n: Int) -> Int = classify(Outer::B(Inner::I(n)))
            def run_c() -> Int = classify(Outer::C)
        ";
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let names: HashMap<String, Hash> =
            r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
        let cm = CompiledModule::build(&ctx, &r).unwrap();
        let rt = Runtime::new_with_registry(cm.closure_type_infos.clone(), cm.shape_registry.clone());
        let jit = Jit::new(cm, &rt).unwrap();
        rt.heap.set_gc_every_alloc(true);
        unsafe {
            let ai = jit.get_fn1(&names["run_ai"]).unwrap();
            let aj = jit.get_fn0(&names["run_aj"]).unwrap();
            let bi = jit.get_fn1(&names["run_bi"]).unwrap();
            let c = jit.get_fn0(&names["run_c"]).unwrap();
            assert_eq!(ai.call(rt.thread_ptr(), 5), 5, "A(I(5)) -> 5");
            assert_eq!(aj.call(rt.thread_ptr()), -7, "A(J) -> -7 (shares outer A)");
            assert_eq!(bi.call(rt.thread_ptr(), 3), 103, "B(I(3)) -> 103");
            assert_eq!(c.call(rt.thread_ptr()), -9, "C -> -9");
        }
    }

    #[test]
    fn lower_case_ident_pattern_is_a_binding_not_a_variant() {
        // `notAVariant` doesn't match any registered variant, so the
        // resolver treats it as a catch-all Var binding. The module
        // resolves cleanly.
        let src = "
            enum E { A }
            def f(e: E) -> Int = match e {
                E::A => 1,
                catch_all => 2,
            }
        ";
        let m = parse_module(src).unwrap();
        resolve_module(&m).expect("should resolve as A=>1, catch_all=>2");
    }

    #[test]
    fn jit_bool_literal_lowers_to_zero_one() {
        // Bool is represented as i64 0/1 at runtime, so `BoolLit` now lowers
        // (mirroring `IntLit`): `true` -> 1, `false` -> 0. There is no surface
        // `true`/`false` syntax yet, so inject the literal into the AST and
        // JIT it directly. The fn is declared `-> Int` (the ABI carrier; Bool
        // and Int share the i64 representation).
        init();
        for (lit, expect) in [(true, 1i64), (false, 0i64)] {
            let ctx = Context::create();
            let mut r = resolve_module(&parse_module("def t() -> Int = 1").unwrap()).unwrap();
            if let Def::Fn { body, .. } = &mut r.defs[0].def {
                *body = Expr::BoolLit(lit);
            }
            let names: HashMap<String, Hash> =
                r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
            let cm = CompiledModule::build(&ctx, &r)
                .expect("BoolLit should lower, not error");
            let rt = Runtime::new_with_registry(
                cm.closure_type_infos.clone(),
                cm.shape_registry.clone(),
            );
            let jit = Jit::new(cm, &rt).expect("jit");
            unsafe {
                let f = jit.get_fn0(&names["t"]).unwrap();
                assert_eq!(f.call(rt.thread_ptr()), expect, "BoolLit({lit}) -> {expect}");
            }
        }
    }

    // ---- Generics ----

    /// Generic identity fn called with an Int. The arg is boxed into a
    /// `BoxedInt` at the call site, the generic fn body passes the
    /// pointer through, and the return is unboxed back to i64.
    #[test]
    fn jit_generic_identity_on_int() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            def id(x: Int) -> Int = x
            def run(x: Int) -> Int = id(x)
            ",
        );
        unsafe {
            let f = jit.get_fn1(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 42), 42);
        }
    }

    #[test]
    fn jit_generic_identity_explicit_typevar() {
        init();
        let ctx = Context::create();
        // The `<T>` declares a type parameter; `id` becomes the
        // single uniform-representation identity. Calling with an Int
        // requires box → call → unbox.
        let (rt, jit, names) = build_for(
            &ctx,
            "
            def id<T>(x: T) -> T = x
            def run(x: Int) -> Int = id(x)
            ",
        );
        unsafe {
            let f = jit.get_fn1(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 99), 99);
            assert_eq!(f.call(rt.thread_ptr(), -5), -5);
        }
    }

    /// **GC root-tracking proof at the JIT layer.**
    ///
    /// `make_adder(n)` returns a closure capturing `n`. We bind it
    /// in `run`'s let frame (which spills the pointer to a shadow-
    /// stack root slot), then call `gc_collect()` which forces a
    /// stop-the-world GC. After the collect, the closure pointer in
    /// the root slot must have been rewritten to point at the
    /// to-space copy. The subsequent `f(x)` indirect-call dereferences
    /// the (now-relocated) closure; if root tracking is broken the
    /// JIT would read stale from-space bits and either crash, return
    /// garbage, or invoke the wrong code.
    ///
    /// Pre-GC heap usage gives us a low-water mark; we sanity-check
    /// that the collection actually happened by asserting collection
    /// count went up.
    #[test]
    fn jit_closure_survives_forced_gc() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            def make_adder(n: Int) -> fn(Int) -> Int = |x: Int| x + n
            def run(n: Int, x: Int) -> Int = {
                let f = make_adder(n);
                let _trigger = gc_collect();
                f(x)
            }
            ",
        );
        let before = rt.heap.collections();
        unsafe {
            let r = jit.get_fn2(&names["run"]).unwrap();
            assert_eq!(r.call(rt.thread_ptr(), 10, 32), 42);
            assert_eq!(r.call(rt.thread_ptr(), 100, 5), 105);
        }
        let after = rt.heap.collections();
        assert!(
            after > before,
            "gc_collect() should have actually run a collection \
             (before={}, after={})",
            before,
            after,
        );
    }

    /// Stress: allocate, GC, allocate more (garbage), GC, then call
    /// the long-lived closure. The captured `n` and the closure's
    /// code-hash must both survive multiple relocations. Asserts the
    /// collection counter went up by AT LEAST 2.
    #[test]
    fn jit_closure_survives_multiple_gcs_with_intervening_alloc() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            def make_adder(n: Int) -> fn(Int) -> Int = |x: Int| x + n
            def make_multiplier(k: Int) -> fn(Int) -> Int = |x: Int| x * k
            def stress(n: Int, x: Int) -> Int = {
                let f = make_adder(n);
                let _g1 = make_multiplier(1);
                let _g2 = gc_collect();
                let _g3 = make_multiplier(2);
                let _g4 = gc_collect();
                let _g5 = make_multiplier(3);
                let _g6 = gc_collect();
                f(x)
            }
            ",
        );
        let before = rt.heap.collections();
        unsafe {
            let r = jit.get_fn2(&names["stress"]).unwrap();
            assert_eq!(r.call(rt.thread_ptr(), 1000, 42), 1042);
        }
        let after = rt.heap.collections();
        assert!(
            after >= before + 3,
            "expected 3 collections, got {} (before={}, after={})",
            after - before,
            before,
            after,
        );
    }

    /// Generic match: define a non-generic `Result` wrapper around an
    /// `Int` so the match-arm Int handling exercises the existing
    /// monomorphic path (sanity-check) before we lean on at()'s box.
    #[test]
    fn jit_match_on_user_enum_int_payload() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            enum Tag { Some(Int), None }
            def make() -> Tag = Tag::Some(7)
            def run() -> Int = match make() {
                Tag::Some(n) => n + 1,
                Tag::None => 0,
            }
            ",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 8);
        }
    }

    // ---- FFI / `extern fn` ----

    /// Register a Layer-1 extern (Int args + Int return), call it from
    /// JIT'd ai-lang code, verify the Rust function actually ran.
    ///
    /// The Rust impl receives `*mut Thread, i64` and returns `i64`.
    /// ai-lang sees `extern fn double_in_rust(n: Int) -> Int`.
    #[test]
    fn ffi_extern_double_called_from_lang() {
        init();
        // Rust impl matching the Layer-1 ABI.
        #[unsafe(no_mangle)]
        unsafe extern "C" fn double_in_rust(_thread: *mut Thread, n: i64) -> i64 {
            n * 2
        }

        // Register it. (Process-global; clear at end for hygiene.)
        unsafe {
            crate::ffi::register_extern(
                "double_in_rust",
                vec![Type::Builtin("Int".to_owned())],
                Type::Builtin("Int".to_owned()),
                double_in_rust as usize,
            );
        }

        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            extern fn double_in_rust(n: Int) -> Int
            def run(x: Int) -> Int = double_in_rust(x) + 1
            ",
        );
        unsafe {
            let f = jit.get_fn1(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 21), 43);
            assert_eq!(f.call(rt.thread_ptr(), -5), -9);
        }

        reset_externs_to_baseline();
    }

    /// Two-arg extern + composition with another fn. Exercises arg
    /// ordering and shows the extern is callable from inside any
    /// def body (not just the entry point).
    #[test]
    fn ffi_extern_two_arg_composed() {
        init();
        #[unsafe(no_mangle)]
        unsafe extern "C" fn add_in_rust(_thread: *mut Thread, a: i64, b: i64) -> i64 {
            a + b
        }
        unsafe {
            crate::ffi::register_extern(
                "add_in_rust",
                vec![
                    Type::Builtin("Int".to_owned()),
                    Type::Builtin("Int".to_owned()),
                ],
                Type::Builtin("Int".to_owned()),
                add_in_rust as usize,
            );
        }

        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            extern fn add_in_rust(a: Int, b: Int) -> Int
            def triple_sum(a: Int, b: Int) -> Int = add_in_rust(a, add_in_rust(b, b))
            def run(a: Int, b: Int) -> Int = triple_sum(a, b)
            ",
        );
        unsafe {
            let f = jit.get_fn2(&names["run"]).unwrap();
            // triple_sum(2, 3) = 2 + (3 + 3) = 8
            assert_eq!(f.call(rt.thread_ptr(), 2, 3), 8);
        }
        reset_externs_to_baseline();
    }

    /// Zero-arg extern. The thread pointer is always passed, so the
    /// Rust signature is `unsafe extern "C" fn(*mut Thread) -> i64`.
    #[test]
    fn ffi_extern_zero_arg() {
        init();
        #[unsafe(no_mangle)]
        unsafe extern "C" fn answer(_thread: *mut Thread) -> i64 {
            42
        }
        unsafe {
            crate::ffi::register_extern(
                "answer",
                vec![],
                Type::Builtin("Int".to_owned()),
                answer as usize,
            );
        }
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            extern fn answer() -> Int
            def run() -> Int = answer()
            ",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 42);
        }
        reset_externs_to_baseline();
    }

    // ---- Layer 2 FFI: String marshaling ----

    /// Rust extern takes a String, returns Int (parse it).
    #[test]
    fn ffi_extern_string_in_int_out() {
        init();
        #[unsafe(no_mangle)]
        unsafe extern "C" fn parse_int(_thread: *mut Thread, s: *const u8) -> i64 {
            let owned = unsafe { crate::ffi::heap_str_to_owned(s) };
            owned.trim().parse::<i64>().unwrap_or(-1)
        }
        unsafe {
            crate::ffi::register_extern(
                "parse_int",
                vec![Type::Builtin("String".to_owned())],
                Type::Builtin("Int".to_owned()),
                parse_int as usize,
            );
        }
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "extern fn parse_int(s: String) -> Int
             def run() -> Int = parse_int(\"42\")",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 42);
        }
        reset_externs_to_baseline();
    }

    /// Rust extern takes Int, returns a new heap String. The returned
    /// pointer is used by ai-lang `string_len` to confirm correctness.
    #[test]
    fn ffi_extern_int_in_string_out() {
        init();
        #[unsafe(no_mangle)]
        unsafe extern "C" fn int_to_str(thread: *mut Thread, n: i64) -> *mut u8 {
            let s = format!("{}", n);
            unsafe { crate::ffi::owned_str_to_heap(thread, &s) }
        }
        unsafe {
            crate::ffi::register_extern(
                "int_to_str",
                vec![Type::Builtin("Int".to_owned())],
                Type::Builtin("String".to_owned()),
                int_to_str as usize,
            );
        }
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "extern fn int_to_str(n: Int) -> String
             def run() -> Int = string_len(int_to_str(12345))",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 5); // "12345".len()
        }
        reset_externs_to_baseline();
    }

    /// String round-trip: take a String, return a derived String.
    /// Tests both directions of marshaling in a single op.
    #[test]
    fn ffi_extern_string_roundtrip() {
        init();
        #[unsafe(no_mangle)]
        unsafe extern "C" fn to_upper(thread: *mut Thread, s: *const u8) -> *mut u8 {
            let owned = unsafe { crate::ffi::heap_str_to_owned(s) };
            let upper = owned.to_uppercase();
            unsafe { crate::ffi::owned_str_to_heap(thread, &upper) }
        }
        unsafe {
            crate::ffi::register_extern(
                "to_upper",
                vec![Type::Builtin("String".to_owned())],
                Type::Builtin("String".to_owned()),
                to_upper as usize,
            );
        }
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "extern fn to_upper(s: String) -> String
             def run() -> Int =
                 string_eq(to_upper(\"hello\"), \"HELLO\")",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
        }
        reset_externs_to_baseline();
    }

    /// The actual `print_string` we promised: Rust writes to stdout.
    /// We test against a side channel (a captured atomic) instead of
    /// stdout for deterministic verification.
    #[test]
    fn ffi_print_string_via_side_channel() {
        use std::sync::Mutex;
        use std::sync::OnceLock;
        static CAPTURED: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        fn captured() -> &'static Mutex<Vec<String>> {
            CAPTURED.get_or_init(|| Mutex::new(Vec::new()))
        }
        init();
        #[unsafe(no_mangle)]
        unsafe extern "C" fn print_string(_thread: *mut Thread, s: *const u8) -> i64 {
            let owned = unsafe { crate::ffi::heap_str_to_owned(s) };
            captured().lock().unwrap().push(owned);
            0
        }
        captured().lock().unwrap().clear();
        unsafe {
            crate::ffi::register_extern(
                "print_string",
                vec![Type::Builtin("String".to_owned())],
                Type::Builtin("Int".to_owned()),
                print_string as usize,
            );
        }
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "extern fn print_string(s: String) -> Int
             def run() -> Int = {
                 let _a = print_string(\"hello, \");
                 let _b = print_string(\"world\");
                 0
             }",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
        let log = captured().lock().unwrap().clone();
        assert_eq!(log, vec!["hello, ".to_owned(), "world".to_owned()]);
        reset_externs_to_baseline();
    }

    // ---- if / else ----

    #[test]
    fn jit_if_branch_taken() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def abs(x: Int) -> Int =
                if x < 0 { 0 - x } else { x }",
        );
        unsafe {
            let f = jit.get_fn1(&names["abs"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), -7), 7);
            assert_eq!(f.call(rt.thread_ptr(), 12), 12);
            assert_eq!(f.call(rt.thread_ptr(), 0), 0);
        }
    }

    #[test]
    fn jit_if_returns_int_either_way() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def sign(x: Int) -> Int =
                if x == 0 { 0 } else {
                    if x < 0 { 0 - 1 } else { 1 }
                }",
        );
        unsafe {
            let f = jit.get_fn1(&names["sign"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 0);
            assert_eq!(f.call(rt.thread_ptr(), 42), 1);
            assert_eq!(f.call(rt.thread_ptr(), -42), -1);
        }
    }

    #[test]
    fn jit_if_inside_let_and_call() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def double(x: Int) -> Int = x * 2
             def f(x: Int) -> Int = {
                 let y = if x > 10 { double(x) } else { x };
                 y + 1
             }",
        );
        unsafe {
            let f = jit.get_fn1(&names["f"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 5), 6);     // x=5 → y=5 → 6
            assert_eq!(f.call(rt.thread_ptr(), 20), 41);   // x=20 → y=40 → 41
        }
    }

    // ---- Recursion ----

    /// Self-recursive factorial.
    #[test]
    fn jit_recursive_factorial() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def fact(n: Int) -> Int =
                if n == 0 { 1 } else { n * fact(n - 1) }",
        );
        unsafe {
            let f = jit.get_fn1(&names["fact"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 1);
            assert_eq!(f.call(rt.thread_ptr(), 1), 1);
            assert_eq!(f.call(rt.thread_ptr(), 5), 120);
            assert_eq!(f.call(rt.thread_ptr(), 10), 3628800);
        }
    }

    /// Self-tail-call optimization: deep direct self-recursion in
    /// tail position becomes a loop and doesn't grow the native
    /// stack. 1M iterations would overflow without TCO.
    #[test]
    fn jit_deep_self_tail_call_does_not_overflow() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def loop_(n: Int) -> Int =
                if n <= 0 { 0 } else { loop_(n - 1) }",
        );
        unsafe {
            let f = jit.get_fn1(&names["loop_"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 1_000_000), 0);
        }
    }

    /// Tail-recursive factorial using an accumulator. With TCO this
    /// runs to depth N without stack growth.
    #[test]
    fn jit_tail_recursive_fact_accumulator() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def fact_acc(n: Int, acc: Int) -> Int =
                if n <= 0 { acc } else { fact_acc(n - 1, n * acc) }",
        );
        unsafe {
            let f = jit.get_fn2(&names["fact_acc"]).unwrap();
            // Small N: correctness.
            assert_eq!(f.call(rt.thread_ptr(), 0, 1), 1);
            assert_eq!(f.call(rt.thread_ptr(), 5, 1), 120);
            assert_eq!(f.call(rt.thread_ptr(), 10, 1), 3628800);
            // Deep N: doesn't overflow. (i64 will wrap on multiplies
            // but the recursion still terminates.)
            let _ = f.call(rt.thread_ptr(), 100_000, 1);
        }
    }

    /// Tail call inside a `let`-body still recognized: the body of a
    /// let in tail position inherits tail context.
    #[test]
    fn jit_tail_call_through_let_body() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def go(n: Int) -> Int = {
                let m = n - 1;
                if m < 0 { 0 } else { go(m) }
             }",
        );
        unsafe {
            let f = jit.get_fn1(&names["go"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 1_000_000), 0);
        }
    }

    /// Non-tail recursive call (recursive result consumed by `+`)
    /// still runs through the normal call path. Small N works,
    /// deep N would still overflow — that's expected.
    #[test]
    fn jit_non_tail_recursion_still_works_at_shallow_depth() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def fact(n: Int) -> Int =
                if n == 0 { 1 } else { n * fact(n - 1) }",
        );
        unsafe {
            let f = jit.get_fn1(&names["fact"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 1);
            assert_eq!(f.call(rt.thread_ptr(), 5), 120);
            assert_eq!(f.call(rt.thread_ptr(), 10), 3628800);
        }
    }

    /// Self-recursive sum-to-n.
    #[test]
    fn jit_recursive_sum_to_n() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def sum(n: Int) -> Int =
                if n <= 0 { 0 } else { n + sum(n - 1) }",
        );
        unsafe {
            let f = jit.get_fn1(&names["sum"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 0);
            assert_eq!(f.call(rt.thread_ptr(), 1), 1);
            assert_eq!(f.call(rt.thread_ptr(), 10), 55);
            assert_eq!(f.call(rt.thread_ptr(), 100), 5050);
        }
    }

    /// Mutually-recursive even/odd. Tests SCC discovery.
    #[test]
    fn jit_mutual_recursion_even_odd() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def is_even(n: Int) -> Int =
                 if n == 0 { 1 } else { is_odd(n - 1) }
             def is_odd(n: Int) -> Int =
                 if n == 0 { 0 } else { is_even(n - 1) }",
        );
        unsafe {
            let f = jit.get_fn1(&names["is_even"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 1);
            assert_eq!(f.call(rt.thread_ptr(), 1), 0);
            assert_eq!(f.call(rt.thread_ptr(), 2), 1);
            assert_eq!(f.call(rt.thread_ptr(), 7), 0);
            assert_eq!(f.call(rt.thread_ptr(), 100), 1);
            let f = jit.get_fn1(&names["is_odd"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 1), 1);
            assert_eq!(f.call(rt.thread_ptr(), 2), 0);
            assert_eq!(f.call(rt.thread_ptr(), 99), 1);
        }
    }

    /// Forward reference — `f` calls `g` defined later. Singleton
    /// SCCs handled in topological order (g first, then f).
    #[test]
    fn jit_forward_reference() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def f(x: Int) -> Int = g(x) + 1
             def g(x: Int) -> Int = x * 2",
        );
        unsafe {
            let f = jit.get_fn1(&names["f"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 5), 11);
        }
    }

    /// Self-recursive fib — the textbook benchmark.
    #[test]
    fn jit_recursive_fib() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def fib(n: Int) -> Int =
                if n < 2 { n } else { fib(n - 1) + fib(n - 2) }",
        );
        unsafe {
            let f = jit.get_fn1(&names["fib"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 0), 0);
            assert_eq!(f.call(rt.thread_ptr(), 1), 1);
            assert_eq!(f.call(rt.thread_ptr(), 10), 55);
            assert_eq!(f.call(rt.thread_ptr(), 20), 6765);
        }
    }

    // ---- Strings ----

    /// Confirm a string literal allocates a heap String and that its
    /// length matches what we wrote.
    #[test]
    fn jit_string_lit_len() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def hello_len() -> Int = string_len(\"hello\")",
        );
        unsafe {
            let f = jit.get_fn0(&names["hello_len"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 5);
        }
    }

    /// Equality of two equal literals + inequality of two different ones.
    #[test]
    fn jit_string_eq() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def eq_same() -> Int = string_eq(\"foo\", \"foo\")
             def eq_diff() -> Int = string_eq(\"foo\", \"bar\")
             def eq_lens() -> Int = string_eq(\"foo\", \"foobar\")",
        );
        unsafe {
            let f = jit.get_fn0(&names["eq_same"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
            let f = jit.get_fn0(&names["eq_diff"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
            let f = jit.get_fn0(&names["eq_lens"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
    }

    /// Concat allocates a new String. Verify both the length and an
    /// equality check against a single literal that should match.
    #[test]
    fn jit_string_concat_then_eq() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def concat_len() -> Int = string_len(
                 string_concat(\"hello, \", \"world\")
             )
             def concat_eq_full() -> Int = string_eq(
                 string_concat(\"hello, \", \"world\"),
                 \"hello, world\"
             )",
        );
        unsafe {
            let f = jit.get_fn0(&names["concat_len"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 12);
            let f = jit.get_fn0(&names["concat_eq_full"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 1);
        }
    }

    /// Strings are GC-managed: rebind a let-held String across a
    /// forced collection and verify the bytes survive relocation.
    #[test]
    fn jit_string_survives_gc() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def stress() -> Int = {
                 let s = string_concat(\"head_\", \"tail\");
                 let trigger = gc_collect();
                 string_len(s) + trigger
             }",
        );
        unsafe {
            let f = jit.get_fn0(&names["stress"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 9);
        }
    }

    #[test]
    fn typecheck_rejects_mismatched_if_branches() {
        // Then branch returns a heap pointer (closure), else branch
        // returns Int → typechecker should flag.
        init();
        let m = parse_module(
            "def f(b: Int) -> Int =
                if b { || 1 } else { 0 }",
        )
        .unwrap();
        let r = resolve_module(&m).unwrap();
        let mut cache = crate::typecheck::TypeCache::new();
        let err = crate::typecheck::typecheck_module(&r, &mut cache).expect_err("should err");
        match err {
            crate::typecheck::TypeError::MatchArmsDisagree { .. } => {}
            other => panic!("expected MatchArmsDisagree, got {:?}", other),
        }
    }

    // ---- Bytes ----

    #[test]
    fn jit_bytes_new_len() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(&ctx, "def run() -> Int = bytes_len(bytes_new(16))");
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 16);
        }
    }

    #[test]
    fn jit_bytes_new_is_zero_filled() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(&ctx, "def run() -> Int = bytes_get_trusted(bytes_new(8), 5)");
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 0);
        }
    }

    #[test]
    fn jit_bytes_set_get() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def run() -> Int = {
                let b = bytes_new(4);
                let _x = bytes_set_trusted(b, 2, 99);
                bytes_get_trusted(b, 2)
            }",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 99);
        }
    }

    #[test]
    fn jit_bytes_set_truncates_to_low_byte() {
        // 511 == 0x1FF; low byte is 0xFF == 255.
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def run() -> Int = {
                let b = bytes_new(1);
                let _x = bytes_set_trusted(b, 0, 511);
                bytes_get_trusted(b, 0)
            }",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 255);
        }
    }

    #[test]
    fn jit_bytes_slice() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def run() -> Int = {
                let b = bytes_new(5);
                let _x = bytes_set_trusted(b, 0, 10);
                let _x = bytes_set_trusted(b, 1, 20);
                let _x = bytes_set_trusted(b, 2, 30);
                let _x = bytes_set_trusted(b, 3, 40);
                let _x = bytes_set_trusted(b, 4, 50);
                let s = bytes_slice(b, 1, 3);
                bytes_get_trusted(s, 0) + bytes_len(s) * 100
            }",
        );
        // slice = [20, 30, 40] → 20 + 3*100 = 320
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 320);
        }
    }

    #[test]
    fn jit_bytes_concat() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def run() -> Int = {
                let a = bytes_new(2);
                let _x = bytes_set_trusted(a, 0, 1);
                let _x = bytes_set_trusted(a, 1, 2);
                let b = bytes_new(3);
                let _x = bytes_set_trusted(b, 0, 3);
                let c = bytes_concat(a, b);
                bytes_len(c) * 100 + bytes_get_trusted(c, 2)
            }",
        );
        // len 5, c[2] = b[0] = 3 → 503
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 503);
        }
    }

    #[test]
    fn jit_bytes_from_string_reads_ascii() {
        // Uses a string literal (not an extern) so it doesn't touch the
        // global FFI registry — avoids racing clear_externs with other
        // extern tests under parallel `cargo test`.
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def run() -> Int = {
                let b = bytes_from_string(\"ABC\");
                bytes_get_trusted(b, 0) * 10000 + bytes_get_trusted(b, 1) * 100 + bytes_get_trusted(b, 2)
             }",
        );
        // 'A'=65, 'B'=66, 'C'=67 → 656667
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 656667);
        }
    }

    #[test]
    fn jit_string_from_bytes_then_len() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def run() -> Int = string_len(string_from_bytes(bytes_new(7)))",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 7);
        }
    }

    #[test]
    fn jit_bytes_survive_gc() {
        // `keep` is a live root through a collection triggered by lots of
        // garbage allocation; its bytes must survive relocation intact.
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def alloc_garbage(n: Int) -> Int =
                if n <= 0 { 0 } else {
                    let junk = bytes_new(64);
                    let _x = bytes_set_trusted(junk, 0, n);
                    alloc_garbage(n - 1)
                }
             def run() -> Int = {
                let keep = bytes_new(8);
                let _x = bytes_set_trusted(keep, 3, 123);
                let _x = alloc_garbage(2000);
                let _x = gc_collect();
                bytes_get_trusted(keep, 3)
             }",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 123);
        }
    }

    // ---- Array ----

    #[test]
    fn jit_array_len() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(&ctx, "def run() -> Int = array_len(array_new(7))");
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 7);
        }
    }

    #[test]
    fn jit_array_int_set_get() {
        // `Array<Int>`: elements are boxed on set, unboxed on get. T is
        // pinned via make4's `-> Array<Int>` return annotation.
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def make4() -> Array<Int> = array_new(4)
             def run() -> Int = {
                let a = make4();
                let _x = array_set_trusted(a, 0, 10);
                let _y = array_set_trusted(a, 1, 32);
                array_get_trusted(a, 0) + array_get_trusted(a, 1)
             }",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 42);
        }
    }

    #[test]
    fn jit_array_of_structs_field_access() {
        // Pointer elements: store a struct, read it back, project a field.
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "struct P { x: Int, y: Int }
             def make() -> Array<P> = array_new(2)
             def run() -> Int = {
                let a = make();
                let _s = array_set_trusted(a, 0, P { x: 7, y: 9 });
                let p = array_get_trusted(a, 0);
                p.x + p.y
             }",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 16);
        }
    }

    #[test]
    fn jit_array_int_element_survives_gc() {
        // The array AND its boxed-Int element must survive relocation.
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def make4() -> Array<Int> = array_new(4)
             def garbage(n: Int) -> Int =
                if n <= 0 { 0 } else { let j = array_new(16); garbage(n - 1) }
             def run() -> Int = {
                let a = make4();
                let _x = array_set_trusted(a, 0, 111);
                let _g = garbage(2000);
                let _c = gc_collect();
                array_get_trusted(a, 0)
             }",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 111);
        }
    }

    #[test]
    fn jit_array_struct_element_survives_gc() {
        // A pointer element (struct) must be relocated correctly: after
        // GC, reading the slot yields the moved struct, whose field is
        // still intact.
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "struct Box { v: Int }
             def make() -> Array<Box> = array_new(4)
             def garbage(n: Int) -> Int =
                if n <= 0 { 0 } else { let j = array_new(16); garbage(n - 1) }
             def run() -> Int = {
                let a = make();
                let _s = array_set_trusted(a, 1, Box { v: 99 });
                let _g = garbage(2000);
                let _c = gc_collect();
                let b = array_get_trusted(a, 1);
                b.v
             }",
        );
        unsafe {
            let f = jit.get_fn0(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr()), 99);
        }
    }

    // ---- Float ----
    // A Float value is carried as the i64 bit-pattern of its f64. A
    // Float-returning fn therefore returns that bit pattern through
    // get_fn0 (i64), which we read back with f64::from_bits.

    #[test]
    fn jit_float_arithmetic() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def add() -> Float = 1.5 + 2.25
             def mul() -> Float = 2.5 * 4.0
             def div() -> Float = 10.0 / 4.0
             def sub() -> Float = 1.0e3 - 0.5",
        );
        unsafe {
            let bits = |n: &str| -> f64 {
                f64::from_bits(jit.get_fn0(&names[n]).unwrap().call(rt.thread_ptr()) as u64)
            };
            assert_eq!(bits("add"), 3.75);
            assert_eq!(bits("mul"), 10.0);
            assert_eq!(bits("div"), 2.5);
            assert_eq!(bits("sub"), 999.5);
        }
    }

    #[test]
    fn jit_float_comparisons() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "def lt() -> Int = if 1.5 < 2.5 { 1 } else { 0 }
             def gt() -> Int = if 3.0 > 5.0 { 1 } else { 0 }
             def eq() -> Int = if 2.5 == 2.5 { 1 } else { 0 }",
        );
        unsafe {
            assert_eq!(jit.get_fn0(&names["lt"]).unwrap().call(rt.thread_ptr()), 1);
            assert_eq!(jit.get_fn0(&names["gt"]).unwrap().call(rt.thread_ptr()), 0);
            assert_eq!(jit.get_fn0(&names["eq"]).unwrap().call(rt.thread_ptr()), 1);
        }
    }

    #[test]
    fn jit_float_int_conversions() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            // half(n) = floor(n / 2) computed in float.
            "def half(n: Int) -> Int = float_to_int(int_to_float(n) / 2.0)
             def trunc() -> Int = float_to_int(3.99)",
        );
        unsafe {
            let f = jit.get_fn1(&names["half"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), 7), 3);
            assert_eq!(f.call(rt.thread_ptr(), 10), 5);
            assert_eq!(jit.get_fn0(&names["trunc"]).unwrap().call(rt.thread_ptr()), 3);
        }
    }

    #[test]
    fn jit_float_param_passthrough() {
        init();
        let ctx = Context::create();
        // Float param + return: the bit pattern must survive the ABI.
        let (rt, jit, names) = build_for(
            &ctx,
            "def scale(x: Float, k: Float) -> Float = x * k
             def run() -> Float = scale(1.25, 8.0)",
        );
        unsafe {
            let r = jit.get_fn0(&names["run"]).unwrap().call(rt.thread_ptr());
            assert_eq!(f64::from_bits(r as u64), 10.0);
        }
    }

    #[test]
    fn jit_float_concrete_struct_field() {
        init();
        let ctx = Context::create();
        // Float fields live in the non-pointer (raw) section as 8-byte
        // values, loaded as i64 bits then bitcast for arithmetic.
        let (rt, jit, names) = build_for(
            &ctx,
            "struct Vec2 { x: Float, y: Float }
             def run() -> Int = {
                let v = Vec2 { x: 1.5, y: 2.5 };
                float_to_int((v.x + v.y) * 10.0)
             }",
        );
        unsafe {
            // (1.5 + 2.5) * 10 = 40
            assert_eq!(jit.get_fn0(&names["run"]).unwrap().call(rt.thread_ptr()), 40);
        }
    }

    // ---- Generic C FFI: real symbols via dlopen/dlsym ----

    /// Call libc `abs` directly. The simplest possible C-FFI: one
    /// scalar in, one scalar out, no memory. Proves dlsym resolution
    /// and the plain-C-ABI (no Thread*) call path work end to end.
    #[test]
    fn cffi_libc_abs() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            extern \"C\" lib \"c\" {
                fn abs(n: Int) -> Int
            }
            def run(n: Int) -> Int = abs(n)
            ",
        );
        unsafe {
            let f = jit.get_fn1(&names["run"]).unwrap();
            assert_eq!(f.call(rt.thread_ptr(), -7), 7);
            assert_eq!(f.call(rt.thread_ptr(), 42), 42);
        }
    }

    /// Full chain: `malloc` a buffer, write a NUL-terminated C string
    /// into it with the `ptr_write_u8` memory intrinsic, hand the raw
    /// pointer to libc `strlen`, then `free` it. Exercises Ptr values,
    /// memory intrinsics, and three distinct C symbols cooperating.
    #[test]
    fn cffi_libc_malloc_strlen_free() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            extern \"C\" lib \"c\" {
                fn malloc(size: Int) -> Ptr
                fn free(p: Ptr) -> Int
                fn strlen(s: Ptr) -> Int
            }
            def run() -> Int = {
                let p = malloc(8);
                let w0 = ptr_write_u8(p, 0, 104);
                let w1 = ptr_write_u8(p, 1, 105);
                let w2 = ptr_write_u8(p, 2, 33);
                let w3 = ptr_write_u8(p, 3, 0);
                let n = strlen(p);
                let fr = free(p);
                fr - fr + n + w0 + w1 + w2 + w3
            }
            ",
        );
        unsafe {
            // "hi!" → length 3.
            assert_eq!(jit.get_fn0(&names["run"]).unwrap().call(rt.thread_ptr()), 3);
        }
    }

    /// Round-trip bytes through libc `memcpy` using the i64 memory
    /// intrinsics, then read the copied value back. Confirms
    /// ptr_read_i64 / ptr_write_i64 and a Ptr-returning C fn.
    #[test]
    fn cffi_libc_memcpy_i64() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            extern \"C\" lib \"c\" {
                fn malloc(size: Int) -> Ptr
                fn free(p: Ptr) -> Int
                fn memcpy(dst: Ptr, src: Ptr, n: Int) -> Ptr
            }
            def run() -> Int = {
                let src = malloc(8);
                let dst = malloc(8);
                let w = ptr_write_i64(src, 0, 123456789);
                let cp = memcpy(dst, src, 8);
                let v = ptr_read_i64(dst, 0);
                let f1 = free(src);
                let f2 = free(dst);
                v + w
            }
            ",
        );
        unsafe {
            assert_eq!(
                jit.get_fn0(&names["run"]).unwrap().call(rt.thread_ptr()),
                123456789
            );
        }
    }

    /// Variadic C FFI through libc `snprintf` — a classic variadic
    /// function. Proves the `...` declaration drives a real varargs
    /// call (correct on the arm64 / x86-64 variadic ABI, where a
    /// non-variadic prototype would misplace the trailing args). We
    /// format an integer into a buffer and read the resulting bytes
    /// back to confirm they landed correctly.
    #[test]
    fn cffi_variadic_snprintf() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            extern \"C\" lib \"c\" {
                fn malloc(size: Int) -> Ptr
                fn free(p: Ptr) -> Int
                fn snprintf(buf: Ptr, size: Int, fmt: Ptr, ...) -> Int
            }
            def run() -> Int = {
                let buf = malloc(16);
                let fmt = malloc(8);
                let f0 = ptr_write_u8(fmt, 0, 37);
                let f1 = ptr_write_u8(fmt, 1, 100);
                let f2 = ptr_write_u8(fmt, 2, 0);
                let n = snprintf(buf, 16, fmt, 4097);
                let b0 = ptr_read_u8(buf, 0);
                let b1 = ptr_read_u8(buf, 1);
                let b2 = ptr_read_u8(buf, 2);
                let b3 = ptr_read_u8(buf, 3);
                let fr1 = free(buf);
                let fr2 = free(fmt);
                n + b0 + b1 + b2 + b3
            }
            ",
        );
        unsafe {
            // fmt = \"%d\"; snprintf(buf, 16, \"%d\", 4097) writes \"4097\".
            // returns 4 (chars written). bytes: '4'=52 '0'=48 '9'=57 '7'=55.
            // total = 4 + 52 + 48 + 57 + 55 = 216.
            assert_eq!(
                jit.get_fn0(&names["run"]).unwrap().call(rt.thread_ptr()),
                216
            );
        }
    }

    /// An unresolvable C symbol fails the build with a CLEAR error that
    /// names the symbol and library — not a silent miscompile or a later
    /// segfault. This is the same `declare_user_externs` path the remote
    /// install uses, so shipping FFI code to a node that lacks the library
    /// fails this way too ("this C code is required to run").
    #[test]
    fn cffi_unresolvable_symbol_fails_clearly() {
        init();
        let src = "
            extern \"C\" lib \"definitely_not_a_real_library_xyz\" {
                fn nope_fn(n: Int) -> Int
            }
            def run(x: Int) -> Int = nope_fn(x)
            ";
        let ctx = Context::create();
        let m = parse_module(src).unwrap();
        let r = resolve_module(&m).unwrap();
        let err = CompiledModule::build(&ctx, &r)
            .err()
            .expect("build must fail when the C symbol can't be resolved");
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("nope_fn") && msg.contains("definitely_not_a_real_library_xyz"),
            "error should name the unresolved symbol + library, got: {}",
            msg
        );
    }

    /// Round-trip a C string through the FFI: build a buffer with the
    /// `cstr`-style pattern inline (malloc + ptr_write_u8), hand it to
    /// libc `strlen`, and confirm the length. The full `cstr` /
    /// `cstr_to_string` helpers (and libcurl HTTP) are exercised against
    /// the real stdlib in `stdlib.rs` tests.
    #[test]
    fn cffi_cstr_pattern_strlen() {
        init();
        let ctx = Context::create();
        let (rt, jit, names) = build_for(
            &ctx,
            "
            extern \"C\" lib \"c\" {
                fn malloc(size: Int) -> Ptr
                fn free(p: Ptr) -> Int
                fn strlen(s: Ptr) -> Int
            }
            def run() -> Int = {
                let p = malloc(6);
                let a = ptr_write_u8(p, 0, 119);
                let b = ptr_write_u8(p, 1, 111);
                let c = ptr_write_u8(p, 2, 114);
                let d = ptr_write_u8(p, 3, 100);
                let e = ptr_write_u8(p, 4, 0);
                let n = strlen(p);
                let fr = free(p);
                n + a + b + c + d + e
            }
            ",
        );
        unsafe {
            // \"word\" → strlen 4; the writes all return 0.
            assert_eq!(jit.get_fn0(&names["run"]).unwrap().call(rt.thread_ptr()), 4);
        }
    }
}
