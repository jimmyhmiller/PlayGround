//! LLVM codegen (via inkwell): [`CoreProgram`] → an executable LLVM module.
//!
//! v0 slice: scalar functions (the fib class). Every compiled function takes a
//! `Thread*` as its first parameter — matching the GC ABI in `docs/gc.md` — even
//! when it doesn't allocate, so the calling convention is uniform once heap
//! allocation + GC frames are layered on. No frames are emitted yet for
//! allocation-free functions (they have no roots to track).

use crate::ast::{BinOp, UnOp};
use crate::core::*;

use inkwell::OptimizationLevel;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::debug_info::{
    AsDIScope, DICompileUnit, DIFile, DIFlags, DIFlagsConstants, DIScope, DISubprogram, DIType,
    DWARFEmissionKind, DWARFSourceLanguage, DebugInfoBuilder,
};
use inkwell::module::{FlagBehavior, Module};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{AsValueRef, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, AtomicOrdering, FloatPredicate, IntPredicate};
use std::collections::HashMap;

/// DWARF line-table emission state (debugger P2). Present only for AOT builds
/// (`gcr build`) — JIT-code DWARF needs the LLDB JIT-registration interface
/// (deferred). One [`DIFile`] per [`crate::lexer::SourceId`] (keyed by the
/// SourceMap index) so a span resolves to the RIGHT source file in the line
/// table — the multi-source analog of the source-id alloc-site fix.
struct DebugCx<'ctx> {
    di: DebugInfoBuilder<'ctx>,
    cu: DICompileUnit<'ctx>,
    /// `DIFile` per `SourceId` (index = source id; same order as `prog.sources`).
    files: Vec<DIFile<'ctx>>,
    /// `true` for **full** DWARF (debugger P3): emit local/parameter variable
    /// DIEs so `frame variable` shows source names + values. `false` =
    /// line-tables-only (P2): stepping + breakpoints only. Full debug also
    /// implies the AOT path skips optimization (so allocas/locals survive).
    full: bool,
    /// Native DWARF composite type per heap layout (`type_id` → struct DIE), so
    /// a `Ref` local renders as `*p = (Point) { x = 3, y = 4 }` natively in lldb
    /// — no Python pretty-printer. `None` for enum/opaque/varlen layouts (those
    /// fall back to an address). Empty unless `full`. Members sit at their
    /// absolute, header-included offsets (see `gc::reflect`), so dereferencing
    /// the pointer reads the right bytes.
    layout_types: Vec<Option<DIType<'ctx>>>,
}

/// How much DWARF a build emits. `None` for JIT / `emit llvm` (no debug info);
/// `LineTables` is the default `gcr build` (P2 — stepping + breakpoints, kept
/// even under O2); `Full` is `gcr build --debug` (P3 — local-variable
/// inspection, requires unoptimized codegen).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugLevel {
    None,
    LineTables,
    Full,
}

impl DebugLevel {
    fn emits_dwarf(self) -> bool {
        !matches!(self, DebugLevel::None)
    }
    fn is_full(self) -> bool {
        matches!(self, DebugLevel::Full)
    }
}

#[derive(Debug)]
pub struct CodegenError(pub String);

/// A compiled, JIT-ready program.
pub struct Compiled<'ctx> {
    pub module: Module<'ctx>,
    pub entry_name: String,
    /// Allocation-site table (Target-1b), indexed by `site_id`. Installed into
    /// the heap via `Heap::set_alloc_sites` (JIT) or baked into the reflection
    /// blob (AOT) so a profile can name each site's `(function, type)`.
    pub alloc_sites: Vec<crate::gc::AllocSite>,
}

/// Codegen without DWARF (JIT / `emit llvm`).
pub fn codegen<'ctx>(
    ctx: &'ctx Context,
    prog: &CoreProgram,
) -> Result<Compiled<'ctx>, CodegenError> {
    codegen_with_debug(ctx, prog, DebugLevel::None)
}

/// Codegen, optionally emitting DWARF (debugger P2/P3). DWARF should be emitted
/// ONLY for AOT builds — it lands in the emitted object where `lldb`/
/// `llvm-dwarfdump` read it; the JIT can't use it without the LLDB
/// JIT-registration interface (deferred). [`DebugLevel::Full`] also emits
/// local-variable DIEs (P3) and must be paired with unoptimized codegen.
pub fn codegen_with_debug<'ctx>(
    ctx: &'ctx Context,
    prog: &CoreProgram,
    level: DebugLevel,
) -> Result<Compiled<'ctx>, CodegenError> {
    let module = ctx.create_module("gcr");
    let builder = ctx.create_builder();
    let debug = if level.emits_dwarf() {
        build_debug_cx(ctx, &module, prog, level.is_full())
    } else {
        None
    };
    let mut cg = Codegen {
        ctx,
        module,
        builder,
        prog,
        funcs: HashMap::new(),
        trampolines: HashMap::new(),
        alloc_sites: Vec::new(),
        alloc_site_ids: HashMap::new(),
        debug,
    };
    cg.declare_runtime_externs();

    // Declare every function first (so calls can reference forward decls).
    for (i, f) in prog.funcs.iter().enumerate() {
        let fv = cg.declare_fn(i as FuncId, f);
        cg.funcs.insert(i as FuncId, fv);
    }
    // Define bodies.
    for (i, f) in prog.funcs.iter().enumerate() {
        cg.define_fn(i as FuncId, f)?;
    }

    // Finalize DWARF before verify/optimize (resolves forward metadata refs).
    if let Some(d) = &cg.debug {
        d.di.finalize();
    }

    let entry = prog.entry.ok_or_else(|| CodegenError("no entry".into()))?;
    let entry_name = prog.funcs[entry as usize].name.clone();

    if std::env::var("GCR_DUMP_IR").is_ok() {
        eprintln!("{}", cg.module.print_to_string().to_string());
    }
    cg.module
        .verify()
        .map_err(|e| CodegenError(format!("LLVM module verify failed: {}", e.to_string())))?;
    Ok(Compiled { module: cg.module, entry_name, alloc_sites: cg.alloc_sites })
}

/// First node with a real span in a block (shallow: statement values + the tail
/// expr). Used to locate a function for its `DISubprogram`.
fn first_block_span(b: &CoreBlock) -> Option<SpanId> {
    for s in &b.stmts {
        let e = match s {
            CoreStmt::Let(_, e) => e,
            CoreStmt::Expr(e) => e,
        };
        if e.span != NO_SPAN {
            return Some(e.span);
        }
    }
    b.tail.as_ref().filter(|e| e.span != NO_SPAN).map(|e| e.span)
}

/// Split a source path into `(filename, directory)` for DWARF `DIFile`.
fn split_path(path: &str) -> (String, String) {
    match path.rfind('/') {
        Some(i) => (path[i + 1..].to_string(), path[..i].to_string()),
        None => (path.to_string(), ".".to_string()),
    }
}

/// A scalar reflection kind → its DWARF `(name, DW_ATE encoding, bit width)`.
/// Mirrors [`Codegen::di_type_for_repr`]'s scalar arm but keyed on the runtime
/// reflection `ScalarKind` (what field metadata carries). `None` for the FFI raw
/// pointer (no pointee type yet).
fn scalar_kind_di(s: crate::gc::ScalarKind) -> Option<(&'static str, u32, u64)> {
    use crate::gc::ScalarKind::*;
    Some(match s {
        I8 => ("i8", 5, 8),
        I16 => ("i16", 5, 16),
        I32 => ("i32", 5, 32),
        I64 => ("i64", 5, 64),
        U8 => ("u8", 7, 8),
        U16 => ("u16", 7, 16),
        U32 => ("u32", 7, 32),
        U64 => ("u64", 7, 64),
        F32 => ("f32", 4, 32),
        F64 => ("f64", 4, 64),
        Bool => ("bool", 2, 8),
        Char => ("char", 7, 32),
        Ptr => return None,
    })
}

/// Builds native DWARF composite types from the reflection metadata (debugger
/// P3), so lldb renders heap structs/enums by field name with no Python printer.
/// One memoized struct DIE per heap `type_id`; member offsets are absolute
/// (header included), so dereferencing the `Ref` pointer reads the right bytes.
/// An enum becomes a struct with one `tag` member typed as a DWARF enumeration
/// (tag value → variant name) — the active variant shows, the payload doesn't
/// (needs variant parts, absent from the C API). Recursion (a `List` whose tail
/// points back at itself) is broken by an in-progress guard: a still-being-built
/// layout reached through a pointer degrades to a raw address. Opaque and varlen
/// (Array/String) layouts are `None` (rendered as an address). Value-typed
/// fields are skipped (no per-value offset metadata here yet) — other members
/// keep their explicit offsets, so the struct still renders minus that field.
struct DiTypeBuilder<'a, 'ctx> {
    di: &'a DebugInfoBuilder<'ctx>,
    scope: DIScope<'ctx>,
    file: DIFile<'ctx>,
    prog: &'a CoreProgram,
    layout: Vec<Option<DIType<'ctx>>>,
    layout_inprog: Vec<bool>,
}

impl<'a, 'ctx> DiTypeBuilder<'a, 'ctx> {
    fn new(
        di: &'a DebugInfoBuilder<'ctx>,
        scope: DIScope<'ctx>,
        file: DIFile<'ctx>,
        prog: &'a CoreProgram,
    ) -> Self {
        let nl = prog.layouts.len();
        DiTypeBuilder {
            di,
            scope,
            file,
            prog,
            layout: vec![None; nl],
            layout_inprog: vec![false; nl],
        }
    }

    /// Run the builder over every layout, returning the `type_id`→struct cache.
    fn build_all(mut self) -> Vec<Option<DIType<'ctx>>> {
        for tid in 0..self.prog.layouts.len() {
            self.layout_type(tid as u16);
        }
        self.layout
    }

    fn scalar(&self, s: crate::gc::ScalarKind) -> Option<DIType<'ctx>> {
        let (name, enc, bits) = scalar_kind_di(s)?;
        Some(self.di.create_basic_type(name, bits, enc, DIFlags::ZERO).ok()?.as_type())
    }

    /// A `DW_ATE_address` basic type — an opaque pointer shown as a hex address
    /// (used when a field/local points at a type we don't model yet).
    fn opaque_addr(&self) -> Option<DIType<'ctx>> {
        Some(self.di.create_basic_type("ptr", 64, 1, DIFlags::ZERO).ok()?.as_type())
    }

    /// The DWARF type for a reference to heap layout `tid`: a pointer to its
    /// struct DIE, or an opaque address if that layout isn't modeled / is still
    /// being built (cycle).
    fn ref_ptr(&mut self, tid: u16) -> Option<DIType<'ctx>> {
        match self.layout_type(tid) {
            Some(p) => Some(
                self.di
                    .create_pointer_type("", p, 64, 64, AddressSpace::default())
                    .as_type(),
            ),
            None => self.opaque_addr(),
        }
    }

    fn field_ty(&mut self, ft: crate::gc::FieldTy) -> Option<DIType<'ctx>> {
        use crate::gc::FieldTy;
        match ft {
            FieldTy::Scalar(s) => self.scalar(s),
            FieldTy::Ref(tid) => self.ref_ptr(tid),
            // Value-typed fields are not modeled yet (no per-value offsets here);
            // skip the member — siblings keep their explicit offsets.
            FieldTy::Value(_) => None,
        }
    }

    /// Memoized DWARF struct DIE for heap layout `tid`. `None` for enum / opaque
    /// / varlen layouts and during in-progress recursion.
    fn layout_type(&mut self, tid: u16) -> Option<DIType<'ctx>> {
        let i = tid as usize;
        if i >= self.prog.layouts.len() {
            return None;
        }
        if let Some(t) = self.layout[i] {
            return Some(t);
        }
        if self.layout_inprog[i] {
            return None; // cycle: caller (a pointer) falls back to an address
        }
        let layout = &self.prog.layouts[i];
        // Varlen (Array/String) → opaque (address). Extract owned kind data so we
        // can take `&mut self` for the recursive member builds below (`layout`
        // borrows `prog`, not `self`, but cloning keeps it simple).
        if !matches!(layout.varlen, VarLen::None) {
            return None;
        }
        let total_bytes = 16u64 + layout.ptr_fields as u64 * 8 + layout.raw_bytes as u64;
        let name = layout.meta.name.clone();
        enum Kind {
            Struct(Vec<crate::gc::FieldMeta>),
            Enum(u16, Vec<crate::gc::VariantMeta>),
        }
        let kind = match &layout.meta.kind {
            crate::gc::TypeKind::Struct { fields } => Kind::Struct(fields.clone()),
            crate::gc::TypeKind::Enum { tag_offset, variants } => {
                Kind::Enum(*tag_offset, variants.clone())
            }
            crate::gc::TypeKind::Opaque => return None,
        };
        self.layout_inprog[i] = true;
        let members: Vec<DIType<'ctx>> = match kind {
            Kind::Struct(fields) => fields
                .iter()
                .filter_map(|f| {
                    let fty = self.field_ty(f.ty)?;
                    let size_bits = di_field_size_bits(self.prog, f.ty);
                    Some(
                        self.di
                            .create_member_type(
                                self.scope,
                                &f.name,
                                self.file,
                                0,
                                size_bits,
                                0,
                                f.offset as u64 * 8,
                                DIFlags::ZERO,
                                fty,
                            )
                            .as_type(),
                    )
                })
                .collect(),
            // A heap enum: model the u32 discriminant at `tag_offset` as a DWARF
            // enumeration type (tag value → variant name), so `frame variable e`
            // shows `{ tag = Node }` — the active variant. Per-variant PAYLOAD
            // isn't rendered (needs DWARF variant parts, absent from the C API,
            // or a reflection synthetic provider — see DEBUGGER_P3_PLAN.md).
            Kind::Enum(tag_offset, variants) => {
                let enumerators: Vec<_> = variants
                    .iter()
                    .map(|v| self.di.create_enumerator(&v.name, v.tag as i64, true))
                    .collect();
                match self.di.create_basic_type("u32", 32, 7, DIFlags::ZERO) {
                    Ok(u32t) => {
                        let enum_ty = self
                            .di
                            .create_enumeration_type(
                                self.scope,
                                &format!("{}::tag", name),
                                self.file,
                                0,
                                32,
                                0,
                                &enumerators,
                                u32t.as_type(),
                            )
                            .as_type();
                        vec![self
                            .di
                            .create_member_type(
                                self.scope,
                                "tag",
                                self.file,
                                0,
                                32,
                                0,
                                tag_offset as u64 * 8,
                                DIFlags::ZERO,
                                enum_ty,
                            )
                            .as_type()]
                    }
                    Err(_) => vec![],
                }
            }
        };
        let st = self
            .di
            .create_struct_type(
                self.scope,
                &name,
                self.file,
                0,
                total_bytes * 8,
                0,
                DIFlags::ZERO,
                None,
                &members,
                0,
                None,
                "",
            )
            .as_type();
        self.layout_inprog[i] = false;
        self.layout[i] = Some(st);
        Some(st)
    }
}

/// Size in bits of a field's value (the storage the member occupies).
fn di_field_size_bits(prog: &CoreProgram, ft: crate::gc::FieldTy) -> u64 {
    use crate::gc::FieldTy;
    match ft {
        FieldTy::Scalar(s) => scalar_kind_di(s).map(|(_, _, b)| b).unwrap_or(64),
        FieldTy::Ref(_) => 64,
        FieldTy::Value(vid) => prog
            .values
            .get(vid as usize)
            .map(|v| v.size as u64 * 8)
            .unwrap_or(0),
    }
}

/// Build the DWARF emitter + a `DIFile` per source. Returns `None` if there is no
/// source map (e.g. a program built directly in a test) — nothing to attribute.
fn build_debug_cx<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    prog: &CoreProgram,
    full: bool,
) -> Option<DebugCx<'ctx>> {
    if prog.sources.is_empty() {
        return None;
    }
    let emission = if full {
        DWARFEmissionKind::Full
    } else {
        DWARFEmissionKind::LineTablesOnly
    };
    let (filename, directory) = split_path(&prog.sources[0].path);
    let (di, cu) = module.create_debug_info_builder(
        /* allow_unresolved */ true,
        DWARFSourceLanguage::C,
        &filename,
        &directory,
        /* producer */ "gcr",
        /* is_optimized */ false,
        /* flags */ "",
        /* runtime_ver */ 0,
        /* split_name */ "",
        emission,
        /* dwo_id */ 0,
        /* split_debug_inlining */ false,
        /* debug_info_for_profiling */ false,
        /* sysroot */ "",
        /* sdk */ "",
    );
    // The backend only emits DWARF if the module declares the debug-info version.
    let i32t = ctx.i32_type();
    module.add_basic_value_flag(
        "Debug Info Version",
        FlagBehavior::Warning,
        i32t.const_int(inkwell::debug_info::debug_metadata_version() as u64, false),
    );
    if cfg!(target_os = "macos") {
        module.add_basic_value_flag(
            "Dwarf Version",
            FlagBehavior::Warning,
            i32t.const_int(4, false),
        );
    }
    // One DIFile per source (index = SourceId), so a span resolves to the right
    // file in the line table.
    let files: Vec<DIFile<'ctx>> = prog
        .sources
        .iter()
        .map(|s| {
            let (f, d) = split_path(&s.path);
            di.create_file(&f, &d)
        })
        .collect();
    // Full debug: pre-build a native DWARF struct DIE per heap layout so `Ref`
    // locals render by field name in lldb. Built before `di` is moved into the
    // DebugCx (the builder only borrows it; `DIType` is `'ctx`-scoped, not tied
    // to the borrow).
    let layout_types = if full {
        DiTypeBuilder::new(&di, cu.as_debug_info_scope(), files[0], prog).build_all()
    } else {
        Vec::new()
    };
    Some(DebugCx { di, cu, files, full, layout_types })
}

/// Emit the LLVM IR text for a whole program — the `gcr emit llvm` tap that
/// completes the source→silicon chain. With `optimize`, runs the same O2
/// pipeline the JIT/AOT paths use, so you see the IR that actually executes;
/// without it, the naive pre-optimization IR (every local a stack slot), which
/// maps more directly onto the Core IR.
pub fn emit_llvm_ir(prog: &CoreProgram, optimize: bool) -> Result<String, CodegenError> {
    let ctx = Context::create();
    let compiled = codegen(&ctx, prog)?;
    if optimize {
        optimize_module(&compiled.module);
    }
    Ok(compiled.module.print_to_string().to_string())
}

struct Codegen<'ctx, 'p> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    prog: &'p CoreProgram,
    funcs: HashMap<FuncId, FunctionValue<'ctx>>,
    /// Cache of synthesized FFI callback trampolines, keyed by the gc-rust
    /// FuncId they wrap (one trampoline per referenced function).
    trampolines: HashMap<FuncId, FunctionValue<'ctx>>,
    /// Allocation-site table (Target-1b), indexed by the `site_id` constant
    /// passed to `ai_gc_alloc_*`. One entry per unique `(function, type_id)`
    /// pair — the honest v1 granularity (Core IR has no source span, so two
    /// allocations of the same type in the same function share a site).
    alloc_sites: Vec<crate::gc::AllocSite>,
    /// Dedup index: `(function symbol, type_id) -> site_id`, so a repeated
    /// `(function, type)` pair reuses its id instead of minting a distinct id
    /// we couldn't label distinctly.
    alloc_site_ids: HashMap<(String, u16, String), u32>,
    /// DWARF line-table state — `Some` only when codegen is asked to emit debug
    /// info (AOT builds). `None` for JIT/`emit llvm` (no DWARF, zero overhead).
    debug: Option<DebugCx<'ctx>>,
}

impl<'ctx, 'p> Codegen<'ctx, 'p> {
    fn llvm_ty(&self, repr: &Repr) -> Option<BasicTypeEnum<'ctx>> {
        match repr {
            Repr::Unit => None,
            Repr::Scalar(s) => Some(self.scalar_ty(*s)),
            // Heap refs are opaque pointers.
            Repr::Ref(_) => Some(self.ctx.ptr_type(AddressSpace::default()).as_basic_type_enum()),
            // Value aggregates are inline LLVM structs (passed by value).
            Repr::Value(vid) => Some(self.value_struct_ty(*vid).as_basic_type_enum()),
        }
    }

    /// The LLVM aggregate type for an inline value layout. Value structs store
    /// their fields in declaration order; value enums lower as `{ i32 tag,
    /// [N x i8] payload }` (handled by their byte size).
    fn value_struct_ty(&self, vid: ValueId) -> inkwell::types::StructType<'ctx> {
        let v = &self.prog.values[vid as usize];
        if let Some(variants) = &v.variants {
            let max_ptrs = crate::core::value_enum_max_ptrs(variants) as u32;
            if max_ptrs == 0 {
                // Compact value enum: `{ i32 tag, [payload bytes] }`.
                let payload_bytes = v.size.saturating_sub(4) as u32;
                return self.ctx.struct_type(
                    &[
                        self.ctx.i32_type().as_basic_type_enum(),
                        self.ctx.i8_type().array_type(payload_bytes).as_basic_type_enum(),
                    ],
                    false,
                );
            }
            // Pointers-first value enum: `{ [max_ptrs x ptr], i32 tag, [raw] }`.
            // Ref payloads share the leading slots across variants (fixed offsets,
            // GC-traceable); scalars/POD-values go in the raw byte region.
            let ptr = self.ctx.ptr_type(AddressSpace::default());
            let raw_bytes = v.size.saturating_sub(max_ptrs * 8 + 4) as u32;
            return self.ctx.struct_type(
                &[
                    ptr.array_type(max_ptrs).as_basic_type_enum(),
                    self.ctx.i32_type().as_basic_type_enum(),
                    self.ctx.i8_type().array_type(raw_bytes).as_basic_type_enum(),
                ],
                false,
            );
        }
        let fields: Vec<BasicTypeEnum> = v.fields.iter()
            .filter_map(|r| self.llvm_ty(r))
            .collect();
        self.ctx.struct_type(&fields, false)
    }

    /// If value layout `vid` is a homogeneous floating-point aggregate (AAPCS64
    /// HFA: 1–4 members, all the same float type, recursing into nested value
    /// structs), returns `(is_f64, count)`. Such an aggregate is passed/returned
    /// in consecutive SIMD registers as `[count x float|double]`. `None` for
    /// anything else (mixed/int fields, value enums, > 4 elements).
    fn value_hfa(&self, vid: ValueId) -> Option<(bool, u32)> {
        fn walk(
            values: &[crate::core::ValueLayout],
            vid: u32,
            kind: &mut Option<bool>,
            count: &mut u32,
        ) -> bool {
            let vl = &values[vid as usize];
            if vl.variants.is_some() {
                return false; // a value enum is never an HFA
            }
            for f in &vl.fields {
                match f {
                    Repr::Scalar(ScalarRepr::F32) => {
                        if *kind == Some(true) { return false; }
                        *kind = Some(false);
                        *count += 1;
                    }
                    Repr::Scalar(ScalarRepr::F64) => {
                        if *kind == Some(false) { return false; }
                        *kind = Some(true);
                        *count += 1;
                    }
                    Repr::Value(v2) => {
                        if !walk(values, *v2, kind, count) { return false; }
                    }
                    _ => return false,
                }
                if *count > 4 { return false; }
            }
            true
        }
        let mut kind = None;
        let mut count = 0;
        if walk(&self.prog.values, vid, &mut kind, &mut count) && count >= 1 && count <= 4 {
            Some((kind.unwrap(), count))
        } else {
            None
        }
    }

    /// AAPCS64 classification of a blittable value struct crossing the FFI **by
    /// value** (non-`mut` extern param, or a return). Returns the LLVM type the
    /// struct is COERCED to for the call/return, or `None` when it must go
    /// indirect (> 16 bytes → a pointer to a caller-allocated copy). `is_return`
    /// picks the tighter integer coercion the C ABI uses for small returns
    /// (a 4-byte struct returns as `i32`, but an argument occupies a full `i64`
    /// register slot). Matches `clang --target=arm64-apple-macos`.
    fn abi_coerce(&self, vid: ValueId, is_return: bool) -> Option<BasicTypeEnum<'ctx>> {
        if let Some((is_f64, n)) = self.value_hfa(vid) {
            let elem = if is_f64 { self.ctx.f64_type() } else { self.ctx.f32_type() };
            return Some(elem.array_type(n).as_basic_type_enum());
        }
        let size = self.prog.values[vid as usize].size;
        let i64t = self.ctx.i64_type();
        if size <= 8 {
            if is_return && size <= 4 {
                Some(self.ctx.i32_type().as_basic_type_enum())
            } else {
                Some(i64t.as_basic_type_enum())
            }
        } else if size <= 16 {
            Some(i64t.array_type(2).as_basic_type_enum())
        } else {
            None // indirect (by pointer)
        }
    }

    fn scalar_ty(&self, s: ScalarRepr) -> BasicTypeEnum<'ctx> {
        match s {
            ScalarRepr::I8 | ScalarRepr::U8 => self.ctx.i8_type().as_basic_type_enum(),
            ScalarRepr::I16 | ScalarRepr::U16 => self.ctx.i16_type().as_basic_type_enum(),
            ScalarRepr::I32 | ScalarRepr::U32 | ScalarRepr::Char => {
                self.ctx.i32_type().as_basic_type_enum()
            }
            ScalarRepr::I64 | ScalarRepr::U64 => self.ctx.i64_type().as_basic_type_enum(),
            ScalarRepr::F32 => self.ctx.f32_type().as_basic_type_enum(),
            ScalarRepr::F64 => self.ctx.f64_type().as_basic_type_enum(),
            ScalarRepr::Bool => self.ctx.bool_type().as_basic_type_enum(),
            ScalarRepr::Ptr => self.ctx.ptr_type(AddressSpace::default()).as_basic_type_enum(),
        }
    }

    fn declare_fn(&self, _id: FuncId, f: &CoreFn) -> FunctionValue<'ctx> {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        // Foreign `extern "C"` function: a plain C declaration with exactly its
        // written signature — NO leading Thread*, no env pointer, name unmangled
        // (the C symbol). External linkage so the JIT/host or AOT linker resolves
        // it. See `docs/ffi.md`.
        if f.is_extern {
            let mut params: Vec<BasicMetadataTypeEnum> = Vec::new();
            for (i, p) in f.params.iter().enumerate() {
                match p {
                    // A `#[repr(C)]` value struct crosses either BY VALUE (the
                    // common case — coerced per the C ABI into register-shaped
                    // types) or BY POINTER (a `mut` out-param, or > 16 bytes which
                    // the ABI passes indirectly). See `docs/ffi.md`.
                    Repr::Value(vid) => {
                        let by_ref = f.extern_by_ref.get(i).copied().unwrap_or(false);
                        match if by_ref { None } else { self.abi_coerce(*vid, false) } {
                            Some(coerced) => params.push(coerced.into()),
                            None => params.push(ptr.into()),
                        }
                    }
                    _ => {
                        if let Some(t) = self.llvm_ty(p) {
                            params.push(t.into());
                        }
                    }
                }
            }
            // A value-struct return is coerced too (≤ 16 bytes — `lower` rejects
            // larger). Scalars/unit use their natural LLVM type.
            let fn_ty = match &f.ret {
                Repr::Value(vid) => match self.abi_coerce(*vid, true) {
                    Some(coerced) => coerced.fn_type(&params, false),
                    None => self.ctx.void_type().fn_type(&params, false),
                },
                _ => match self.llvm_ty(&f.ret) {
                    Some(rt) => rt.fn_type(&params, false),
                    None => self.ctx.void_type().fn_type(&params, false),
                },
            };
            return self.module.add_function(
                &f.name,
                fn_ty,
                Some(inkwell::module::Linkage::External),
            );
        }
        // First param is the Thread*. A lifted closure fn takes the env pointer
        // as its second param (before the value params).
        let mut params: Vec<BasicMetadataTypeEnum> = vec![ptr.into()];
        if !f.closure_captures.is_empty() || f.name.starts_with("__closure_") {
            params.push(ptr.into());
        }
        for p in &f.params {
            if let Some(t) = self.llvm_ty(p) {
                params.push(t.into());
            }
        }
        let fn_ty = match self.llvm_ty(&f.ret) {
            Some(rt) => rt.fn_type(&params, false),
            None => self.ctx.void_type().fn_type(&params, false),
        };
        self.module.add_function(&f.name, fn_ty, None)
    }

    fn is_closure_fn(f: &CoreFn) -> bool {
        f.name.starts_with("__closure_")
    }

    /// Get (or synthesize) a C-ABI trampoline for gc-rust function `fid`, used as
    /// an FFI callback. The trampoline has the callback's C signature (the wrapped
    /// function's value params, no leading `Thread*`), recovers the ambient
    /// thread, re-enters managed state, calls the real function, restores native
    /// state, and returns. See `docs/ffi.md`.
    fn callback_trampoline(&mut self, fid: FuncId) -> FunctionValue<'ctx> {
        if let Some(tr) = self.trampolines.get(&fid) {
            return *tr;
        }
        let target = self.funcs[&fid];
        let f = &self.prog.funcs[fid as usize];
        let ptr = self.ctx.ptr_type(AddressSpace::default());

        // C signature: the value params (skip Unit), no Thread*.
        let mut c_params: Vec<BasicMetadataTypeEnum> = Vec::new();
        for p in &f.params {
            if let Some(t) = self.llvm_ty(p) {
                c_params.push(t.into());
            }
        }
        let fn_ty = match self.llvm_ty(&f.ret) {
            Some(rt) => rt.fn_type(&c_params, false),
            None => self.ctx.void_type().fn_type(&c_params, false),
        };
        let name = format!("__cb_{}", f.name);
        let tramp = self.module.add_function(&name, fn_ty, Some(inkwell::module::Linkage::Internal));
        self.trampolines.insert(fid, tramp);

        // Build the body. (Save/restore the builder position — we may be mid-call.)
        let saved = self.builder.get_insert_block();
        let entry = self.ctx.append_basic_block(tramp, "entry");
        self.builder.position_at_end(entry);

        // thread = ai_current_thread()
        let cur = self.module.get_function("ai_current_thread").unwrap();
        let thread = call_result(self.builder.build_call(cur, &[], "thr").unwrap()).into_pointer_value();
        // ai_ffi_reenter(thread) — we're back in managed code.
        let reenter = self.module.get_function("ai_ffi_reenter").unwrap();
        self.builder.build_call(reenter, &[thread.into()], "").unwrap();

        // Call the real gc-rust function: (thread, c_params...).
        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = vec![thread.into()];
        for (i, _) in c_params.iter().enumerate() {
            call_args.push(tramp.get_nth_param(i as u32).unwrap().into());
        }
        let cs = self.builder.build_call(target, &call_args, "cbret").unwrap();

        // ai_ffi_exit(thread) — return to foreign code, re-publish frame.
        let exit = self.module.get_function("ai_ffi_exit").unwrap();
        self.builder.build_call(exit, &[thread.into()], "").unwrap();

        match cs.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => { self.builder.build_return(Some(&v)).unwrap(); }
            inkwell::values::ValueKind::Instruction(_) => { self.builder.build_return(None).unwrap(); }
        }

        if let Some(bb) = saved { self.builder.position_at_end(bb); }
        tramp
    }

    /// The function's representative source location — the first spanned node in
    /// its body — as `(SourceId, line, col)`. Defaults to `(0, 1, 1)` (user
    /// source, line 1) when the function has no spanned node, so it still gets a
    /// DIFile + scope. Shallow scan (post-ANF the construction/statement nodes
    /// are at statement position) — enough to locate the function.
    fn fn_debug_loc(&self, f: &CoreFn) -> (crate::lexer::SourceId, u32, u32) {
        // The function's own definition span (robust: names its source even for a
        // monomorphized prelude/`mod` function). Fall back to the first spanned
        // body node, then to (user, line 1).
        if let Some((sid, line, col)) = self.prog.span_source_loc(f.span) {
            return (sid, line as u32, col as u32);
        }
        if let Some(sp) = first_block_span(&f.body) {
            if let Some((sid, line, col)) = self.prog.span_source_loc(sp) {
                return (sid, line as u32, col as u32);
            }
        }
        (0, 1, 1)
    }

    /// Create `f`'s DWARF `DISubprogram` (debugger P2) and attach it to `func`.
    /// `None` when not emitting debug info. The subprogram's file is the source
    /// the function body comes from (so its line table + any backtrace frame name
    /// the right file — prelude/`mod`/user).
    fn create_subprogram(&self, f: &CoreFn, func: FunctionValue<'ctx>) -> Option<DISubprogram<'ctx>> {
        let d = self.debug.as_ref()?;
        let (sid, line, _col) = self.fn_debug_loc(f);
        let file = d.files.get(sid as usize).copied().unwrap_or(d.files[0]);
        // Full debug: give the subprogram a real signature (return + param types)
        // so backtraces show typed prototypes. Line-tables-only keeps the empty
        // `(void)` type (no type DIEs are built there anyway).
        let sub_ty = if d.full {
            let ret_ty = self.di_signature_type(&f.ret);
            let param_tys: Vec<DIType<'ctx>> =
                f.params.iter().filter_map(|p| self.di_signature_type(p)).collect();
            d.di.create_subroutine_type(file, ret_ty, &param_tys, DIFlags::ZERO)
        } else {
            d.di.create_subroutine_type(file, None, &[], DIFlags::ZERO)
        };
        let sp = d.di.create_function(
            d.cu.as_debug_info_scope(),
            &f.name,
            Some(&f.name), // linkage name = the mangled symbol
            file,
            line,
            sub_ty,
            /* is_local_to_unit */ true,
            /* is_definition */ true,
            /* scope_line */ line,
            DIFlags::ZERO,
            /* is_optimized */ false,
        );
        func.set_subprogram(sp);
        Some(sp)
    }

    /// The DWARF `DIType` for a local (debugger P3), so a variable DIE carries a
    /// real type and `frame variable` decodes the bytes. Scalars → a basic type;
    /// `Ref` → a pointer to that layout's native struct DIE (lldb renders the
    /// fields), or a raw address for unmodeled (enum/opaque/varlen) layouts.
    /// `None` for `Value`/`Unit` and the FFI raw pointer, and when not emitting
    /// debug info.
    fn di_type_for_repr(&self, r: &Repr) -> Option<DIType<'ctx>> {
        let d = self.debug.as_ref()?;
        // `Ref` locals are handled by the caller (it needs the GC-frame slot +
        // deref expression to render the struct in place); this returns scalars.
        let s = match r {
            Repr::Scalar(s) => *s,
            _ => return None,
        };
        use crate::core::ScalarRepr::*;
        // DWARF `DW_ATE_*` encodings: boolean=2, float=4, signed=5, unsigned=7.
        let (name, encoding, bits): (&str, u32, u64) = match s {
            I8 => ("i8", 5, 8),
            I16 => ("i16", 5, 16),
            I32 => ("i32", 5, 32),
            I64 => ("i64", 5, 64),
            U8 => ("u8", 7, 8),
            U16 => ("u16", 7, 16),
            U32 => ("u32", 7, 32),
            U64 => ("u64", 7, 64),
            F32 => ("f32", 4, 32),
            F64 => ("f64", 4, 64),
            // `bool` lowers to `i1` but occupies a byte in its alloca; describe it
            // as an 8-bit boolean so lldb reads the byte.
            Bool => ("bool", 2, 8),
            // `char` is a 32-bit Unicode scalar.
            Char => ("char", 7, 32),
            // Raw FFI pointer — skip for now (no pointee type yet).
            Ptr => return None,
        };
        Some(d.di.create_basic_type(name, bits, encoding, DIFlags::ZERO).ok()?.as_type())
    }

    /// The DWARF type for a value in a function SIGNATURE (return/param), used to
    /// give the `DISubprogram` a real prototype. Unlike a local, a `Ref` here is a
    /// plain pointer to the struct (the passed value is the object pointer);
    /// `Value` is an opaque address; `Unit` is `None` (omitted / void).
    fn di_signature_type(&self, r: &Repr) -> Option<DIType<'ctx>> {
        let d = self.debug.as_ref()?;
        let opaque = |d: &DebugCx<'ctx>| {
            d.di.create_basic_type("ptr", 64, 1, DIFlags::ZERO).ok().map(|t| t.as_type())
        };
        match r {
            Repr::Unit => None,
            Repr::Scalar(_) => self.di_type_for_repr(r),
            Repr::Ref(lid) => match d.layout_types.get(*lid as usize).copied().flatten() {
                Some(st) => Some(
                    d.di.create_pointer_type("", st, 64, 64, AddressSpace::default()).as_type(),
                ),
                None => opaque(d),
            },
            Repr::Value(_) => opaque(d),
        }
    }

    fn define_fn(&mut self, id: FuncId, f: &CoreFn) -> Result<(), CodegenError> {
        // Foreign `extern "C"` functions have no body — they're resolved at
        // link/JIT time, not defined here.
        if f.is_extern {
            return Ok(());
        }
        let func = self.funcs[&id];
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        // DWARF (P2): create this function's DISubprogram + set its default debug
        // location BEFORE the prologue, so prologue/unspanned instructions carry
        // THIS function's scope (not the previous function's leftover location —
        // which would fail verify with "wrong subprogram"). Always reset the
        // builder's location per function.
        let subprogram = self.create_subprogram(f, func);
        if let (Some(d), Some(sp)) = (&self.debug, subprogram) {
            let (_, line, col) = self.fn_debug_loc(f);
            let loc = d.di.create_debug_location(self.ctx, line, col, sp.as_debug_info_scope(), None);
            self.builder.set_current_debug_location(loc);
        } else {
            self.builder.unset_current_debug_location();
        }

        let ptr = self.ctx.ptr_type(AddressSpace::default());

        // Partition locals: Ref-typed locals become GC frame *direct* root slots;
        // all others get plain allocas. Count the refs to size the frame.
        let num_roots = f.locals.iter().filter(|r| matches!(r, Repr::Ref(_))).count();
        // Map: local id -> root index (only for Ref locals).
        let mut root_index: Vec<Option<u32>> = vec![None; f.locals.len()];
        {
            let mut ri = 0u32;
            for (i, r) in f.locals.iter().enumerate() {
                if matches!(r, Repr::Ref(_)) {
                    root_index[i] = Some(ri);
                    ri += 1;
                }
            }
        }

        // Value-with-ref locals: a flattened `#[value]` aggregate (in a plain
        // alloca) holding GC refs. Those refs live in untraced stack memory, so a
        // collection while the local is live would dangle them. We register each
        // interior ref's *address* as an INDIRECT frame root: the GC dereferences
        // it and relocates the pointer in place inside the alloca. `(local_idx,
        // value-relative ref offsets)`.
        let value_indirect: Vec<(usize, Vec<u16>)> = f
            .locals
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r {
                Repr::Value(vid) if value_has_ref(&self.prog.values, *vid) => {
                    let mut offs = Vec::new();
                    value_interior_offsets(&self.prog.values, *vid, 0, &mut offs);
                    if offs.is_empty() { None } else { Some((i, offs)) }
                }
                _ => None,
            })
            .collect();
        let num_indirect: usize = value_indirect.iter().map(|(_, o)| o.len()).sum();

        // Emit the GC frame `{ parent, origin, [num_roots x ptr], [num_indirect x
        // ptr] }` when there are any roots (direct or indirect). A zero-length
        // trailing array adds no bytes, so frames without indirect roots are
        // byte-identical to before.
        let i32t = self.ctx.i32_type();
        let frame = if num_roots > 0 || num_indirect > 0 {
            let roots_arr = ptr.array_type(num_roots as u32);
            let ind_arr = ptr.array_type(num_indirect as u32);
            let frame_ty = self
                .ctx
                .struct_type(&[ptr.into(), ptr.into(), roots_arr.into(), ind_arr.into()], false);
            let frame = self.builder.build_alloca(frame_ty, "gcframe").unwrap();
            let origin = self.frame_origin(&f.name, num_roots as u32, num_indirect as u32);
            let origin_field = self.builder.build_struct_gep(frame_ty, frame, 1, "origin.f").unwrap();
            self.builder.build_store(origin_field, origin).unwrap();
            // link: parent = thread.top_frame; thread.top_frame = &frame
            let tf_ptr = self.thread_field_ptr(func, crate::runtime::thread_offsets::TOP_FRAME);
            let prev = self.builder.build_load(ptr, tf_ptr, "prevtop").unwrap();
            let parent_field = self.builder.build_struct_gep(frame_ty, frame, 0, "parent.f").unwrap();
            self.builder.build_store(parent_field, prev).unwrap();
            self.builder.build_store(tf_ptr, frame).unwrap();
            // Zero all direct root slots.
            let roots_field = self.builder.build_struct_gep(frame_ty, frame, 2, "roots.f").unwrap();
            for k in 0..num_roots {
                let slot = unsafe {
                    self.builder.build_in_bounds_gep(
                        roots_arr, roots_field,
                        &[i32t.const_zero(), i32t.const_int(k as u64, false)],
                        "rinit",
                    ).unwrap()
                };
                self.builder.build_store(slot, ptr.const_null()).unwrap();
            }
            // Zero all indirect slots (filled with alloca-interior addresses below).
            let ind_field = self.builder.build_struct_gep(frame_ty, frame, 3, "ind.f").unwrap();
            for k in 0..num_indirect {
                let slot = unsafe {
                    self.builder.build_in_bounds_gep(
                        ind_arr, ind_field,
                        &[i32t.const_zero(), i32t.const_int(k as u64, false)],
                        "iinit",
                    ).unwrap()
                };
                self.builder.build_store(slot, ptr.const_null()).unwrap();
            }
            Some((frame, frame_ty, roots_arr, roots_field, ind_arr, ind_field, tf_ptr))
        } else {
            None
        };

        // Build slots. Ref locals point at their frame root slot; others get
        // plain allocas.
        let mut slots: Vec<Option<PointerValue<'ctx>>> = Vec::with_capacity(f.locals.len());
        for (i, lr) in f.locals.iter().enumerate() {
            if let Some(ri) = root_index[i] {
                let (_, _, roots_arr, roots_field, ..) = frame.unwrap();
                let slot = unsafe {
                    self.builder.build_in_bounds_gep(
                        roots_arr, roots_field,
                        &[i32t.const_zero(), i32t.const_int(ri as u64, false)],
                        &format!("root{}", ri),
                    ).unwrap()
                };
                slots.push(Some(slot));
            } else {
                match self.llvm_ty(lr) {
                    Some(t) => slots.push(Some(self.builder.build_alloca(t, &format!("l{}", i)).unwrap())),
                    None => slots.push(None),
                }
            }
        }

        // Wire indirect roots: zero each value-with-ref local's alloca (so its
        // interior refs start null and GC-safe before assignment), then store the
        // address of each interior ref into an indirect slot. The alloca is stable
        // on the stack, so these addresses are set once and the GC updates the
        // refs they point at in place.
        if let Some((_, _, _, _, ind_arr, ind_field, _)) = frame {
            let mut j = 0u32;
            for (local_idx, offs) in &value_indirect {
                let alloca = slots[*local_idx].expect("value-with-ref local has an alloca");
                if let Some(vty) = self.llvm_ty(&f.locals[*local_idx]) {
                    self.builder.build_store(alloca, vty.const_zero()).unwrap();
                }
                for &off in offs {
                    let addr = self.obj_addr(alloca, off as u64);
                    let slot = unsafe {
                        self.builder.build_in_bounds_gep(
                            ind_arr, ind_field,
                            &[i32t.const_zero(), i32t.const_int(j as u64, false)],
                            "islot",
                        ).unwrap()
                    };
                    self.builder.build_store(slot, addr).unwrap();
                    j += 1;
                }
            }
        }

        // Store incoming params (LLVM param 0 is Thread*; a closure fn has the
        // env pointer as LLVM param 1). For a closure fn, the value params map
        // to locals AFTER the capture locals.
        let is_closure = Self::is_closure_fn(f);
        let ncaptures = f.closure_captures.len();
        let nparams = f.params.len();
        let mut llvm_idx = if is_closure { 2u32 } else { 1u32 };
        // Capture locals (the first `ncaptures` locals) are initialized from the
        // env pointer (LLVM param 1), reading each capture's recorded location.
        if is_closure {
            let env = func.get_nth_param(1).unwrap().into_pointer_value();
            for cap in &f.closure_captures {
                let lty = self.llvm_ty(&f.locals[cap.local as usize]);
                if let (Some(slot), Some(lty)) = (slots[cap.local as usize], lty) {
                    let addr = self.obj_addr(env, cap.offset);
                    let v = self.builder.build_load(lty, addr, "cap").unwrap();
                    self.builder.build_store(slot, v).unwrap();
                }
            }
        }
        let param_local_base = if is_closure { ncaptures } else { 0 };
        for i in 0..nparams {
            let local = param_local_base + i;
            if self.llvm_ty(&f.locals[local]).is_some() {
                if let Some(slot) = slots[local] {
                    let arg = func.get_nth_param(llvm_idx).unwrap();
                    self.builder.build_store(slot, arg).unwrap();
                }
                llvm_idx += 1;
            }
        }

        // DWARF local-variable DIEs (debugger P3). For each named local with a
        // DWARF type, emit a `dbg.declare` describing where it lives, so lldb's
        // `frame variable` shows the source name + value. Only in full-debug
        // builds — the AOT path leaves those unoptimized, so the allocas survive
        // (an optimized build would coalesce them away and the declares too).
        // Params get a parameter DIE (lldb lists them as arguments); other named
        // locals get an auto-variable DIE.
        //
        // Location: a plain alloca local points at the alloca (empty expr). A
        // `Ref` local lives in a GC frame ROOT slot — a `gep` into the `gcframe`
        // alloca, which has NO simple DWARF location — so we instead point at the
        // gcframe alloca and append `DW_OP_plus_uconst <slot-offset>`, giving a
        // real `fbreg+offset` location the collector also keeps current (it
        // updates the slot in place on relocation — the §5 moving-GC property).
        if let (Some(d), Some(sp)) = (&self.debug, subprogram) {
            if d.full {
                let (sid, fline, _) = self.fn_debug_loc(f);
                let file = d.files.get(sid as usize).copied().unwrap_or(d.files[0]);
                let scope = sp.as_debug_info_scope();
                let dloc = d.di.create_debug_location(self.ctx, fline, 0, scope, None);
                for (i, repr) in f.locals.iter().enumerate() {
                    let name = match f.local_names.get(i).and_then(|n| n.as_deref()) {
                        Some(n) => n,
                        None => continue,
                    };
                    if slots[i].is_none() {
                        continue;
                    }
                    // Compute (DWARF type, storage, location expression):
                    // `gcframe` = `{ ptr parent, ptr origin, [roots], [ind] }`, so
                    // root `ri` is at byte offset 16 + ri*8.
                    const DW_OP_PLUS_UCONST: i64 = 0x23;
                    const DW_OP_DEREF: i64 = 0x06;
                    let (dty, storage, expr) = if let Repr::Ref(lid) = repr {
                        let frame_ptr = frame.expect("Ref local implies a gcframe").0;
                        let ri = root_index[i].expect("Ref local implies a root slot");
                        let off = 16i64 + ri as i64 * 8;
                        match d.layout_types.get(*lid as usize).copied().flatten() {
                            // Modeled struct: the variable IS the struct living at
                            // `*(frame+off)` (the slot holds the object pointer the
                            // collector keeps current) — `+off` then `deref`. lldb
                            // renders `p = (Point) { x = 3, y = 4 }` directly.
                            Some(st) => (
                                st,
                                frame_ptr,
                                d.di.create_expression(vec![DW_OP_PLUS_UCONST, off, DW_OP_DEREF]),
                            ),
                            // Unmodeled (enum/opaque/varlen): show the slot's
                            // pointer value as a raw address (no deref).
                            None => {
                                let addr = match d.di.create_basic_type("ptr", 64, 1, DIFlags::ZERO) {
                                    Ok(t) => t.as_type(),
                                    Err(_) => continue,
                                };
                                (addr, frame_ptr, d.di.create_expression(vec![DW_OP_PLUS_UCONST, off]))
                            }
                        }
                    } else {
                        match self.di_type_for_repr(repr) {
                            Some(t) => (t, slots[i].unwrap(), d.di.create_expression(vec![])),
                            None => continue,
                        }
                    };
                    let is_param = i >= param_local_base && i < param_local_base + nparams;
                    let var = if is_param {
                        let arg_no = (i - param_local_base + 1) as u32;
                        d.di.create_parameter_variable(
                            scope, name, arg_no, file, fline, dty,
                            /* always_preserve */ true, DIFlags::ZERO,
                        )
                    } else {
                        d.di.create_auto_variable(
                            scope, name, file, fline, dty,
                            /* always_preserve */ true, DIFlags::ZERO, /* align */ 0,
                        )
                    };
                    // Emit the `dbg.declare` record directly via llvm-sys,
                    // bypassing inkwell's `insert_declare_at_end`: under LLVM 21
                    // that wrapper calls the record API but then casts the
                    // returned `DbgRecord` to a value and asserts it is an
                    // instruction — UB that panics nondeterministically. We only
                    // need the side effect (the record attached to `entry`), so
                    // we call the C function and discard its return.
                    unsafe {
                        inkwell::llvm_sys::debuginfo::LLVMDIBuilderInsertDeclareRecordAtEnd(
                            d.di.as_mut_ptr(),
                            storage.as_value_ref(),
                            var.as_mut_ptr(),
                            expr.as_mut_ptr(),
                            dloc.as_mut_ptr(),
                            entry.as_mut_ptr(),
                        );
                    }
                }
            }
        }

        let unlink = frame.map(|(frame_ptr, frame_ty, _, _, _, _, tf_ptr)| {
            (frame_ptr, frame_ty, tf_ptr)
        });

        let mut fcx = FnCtx {
            func,
            slots,
            local_reprs: f.locals.clone(),
            thread: func.get_nth_param(0).unwrap().into_pointer_value(),
            loops: Vec::new(),
            loop_headers: Vec::new(),
            unlink,
            pending_copy_outs: Vec::new(),
            subprogram,
        };
        let val = self.gen_block(&mut fcx, &f.body)?;

        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            self.emit_unlink(&fcx);
            match val {
                Some(v) => { self.builder.build_return(Some(&v)).unwrap(); }
                // No tail value. If the function returns non-unit, the body must
                // have diverged (a `loop`/`return` tail, e.g. `swap`'s retry
                // loop) and this fall-through is unreachable — emitting `ret void`
                // against a non-void signature is invalid. Emit `unreachable`.
                None if self.llvm_ty(&f.ret).is_some() => { self.builder.build_unreachable().unwrap(); }
                None => { self.builder.build_return(None).unwrap(); }
            }
        }
        Ok(())
    }

    /// Restore `thread.top_frame = frame.parent` (no-op if this fn has no frame).
    fn emit_unlink(&self, fcx: &FnCtx<'ctx>) {
        if let Some((frame, frame_ty, tf_ptr)) = fcx.unlink {
            let parent_field = self.builder.build_struct_gep(frame_ty, frame, 0, "parent.r").unwrap();
            let ptr = self.ctx.ptr_type(AddressSpace::default());
            let parent = self.builder.build_load(ptr, parent_field, "parent.v").unwrap();
            self.builder.build_store(tf_ptr, parent).unwrap();
        }
    }

    /// Pointer to a field of the Thread struct at `offset` (via ptr arithmetic).
    fn thread_field_ptr(&self, func: FunctionValue<'ctx>, offset: usize) -> PointerValue<'ctx> {
        let thread = func.get_nth_param(0).unwrap().into_pointer_value();
        let i64t = self.ctx.i64_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let base = self.builder.build_ptr_to_int(thread, i64t, "t.i").unwrap();
        let addr = self.builder.build_int_add(base, i64t.const_int(offset as u64, false), "t.fa").unwrap();
        self.builder.build_int_to_ptr(addr, ptr, "t.fp").unwrap()
    }

    /// A private constant `FrameOrigin { num_roots, pad, name }` global.
    fn frame_origin(&self, fn_name: &str, num_roots: u32, num_indirect: u32) -> PointerValue<'ctx> {
        let i32t = self.ctx.i32_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        // FrameOrigin (runtime, #[repr(C)]): { u32 num_roots, u32 num_indirect, ptr name }.
        let origin_ty = self.ctx.struct_type(&[i32t.into(), i32t.into(), ptr.into()], false);
        let g = self.module.add_global(origin_ty, None, &format!("__origin_{}", fn_name));
        g.set_constant(true);
        g.set_initializer(&origin_ty.const_named_struct(&[
            i32t.const_int(num_roots as u64, false).into(),
            i32t.const_int(num_indirect as u64, false).into(),
            ptr.const_null().into(),
        ]));
        g.as_pointer_value()
    }

    fn gen_block(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        b: &CoreBlock,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        for s in &b.stmts {
            match s {
                CoreStmt::Let(local, e) => {
                    let v = self.gen_expr(fcx, e)?;
                    if let (Some(slot), Some(v)) = (fcx.slots[*local as usize], v) {
                        self.builder.build_store(slot, v).unwrap();
                    }
                }
                CoreStmt::Expr(e) => {
                    self.gen_expr(fcx, e)?;
                }
            }
            if self.builder.get_insert_block().unwrap().get_terminator().is_some() {
                return Ok(None);
            }
        }
        match &b.tail {
            Some(e) => self.gen_expr(fcx, e),
            None => Ok(None),
        }
    }

    fn gen_expr(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        e: &CoreExpr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        // DWARF (P2): stamp this node's instructions with its source location.
        // One function = one source, so the scope is the function's DISubprogram
        // and the location's file stays consistent. Nodes without a span keep the
        // last location (the function default at minimum) — no fabrication.
        if let (Some(d), Some(sp)) = (&self.debug, fcx.subprogram) {
            if let Some((_, line, col)) = self.prog.span_source_loc(e.span) {
                let loc =
                    d.di.create_debug_location(self.ctx, line as u32, col as u32, sp.as_debug_info_scope(), None);
                self.builder.set_current_debug_location(loc);
            }
        }
        match &*e.kind {
            CoreExprKind::ConstInt(n, sr) => {
                let t = self.scalar_ty(*sr).into_int_type();
                Ok(Some(t.const_int(*n, sr.is_signed()).into()))
            }
            CoreExprKind::ConstFloat(f, sr) => {
                let t = self.scalar_ty(*sr).into_float_type();
                Ok(Some(t.const_float(*f).into()))
            }
            CoreExprKind::ConstBool(b) => {
                Ok(Some(self.ctx.bool_type().const_int(*b as u64, false).into()))
            }
            CoreExprKind::ConstChar(c) => {
                Ok(Some(self.ctx.i32_type().const_int(*c as u64, false).into()))
            }
            CoreExprKind::ConstStr(s) => self.gen_const_str(fcx, s, &e.repr, e.span),
            CoreExprKind::ConstZero(repr) => {
                // Zero/null of the repr: scalars → 0, refs → null pointer.
                match self.llvm_ty(repr) {
                    Some(ty) => Ok(Some(ty.const_zero())),
                    None => Ok(None),
                }
            }
            CoreExprKind::Unit => Ok(None),
            CoreExprKind::Local(id) => {
                let slot = fcx.slots[*id as usize];
                match slot {
                    Some(slot) => {
                        let ty = self.llvm_ty(&fcx.local_reprs[*id as usize]).unwrap();
                        let v = self.builder.build_load(ty, slot, "ld").unwrap();
                        Ok(Some(v))
                    }
                    None => Ok(None),
                }
            }
            CoreExprKind::Bin(op, l, r) => self.gen_bin(fcx, *op, l, r),
            CoreExprKind::Un(op, inner) => self.gen_un(fcx, *op, inner),
            CoreExprKind::FloatIntrinsic(intr, inner) => self.gen_float_intrinsic(fcx, *intr, inner),
            CoreExprKind::Print(inner) => {
                let v = self.gen_expr(fcx, inner)?.unwrap();
                let is_float = matches!(&inner.repr, Repr::Scalar(s) if s.is_float());
                let fname = if is_float { "ai_print_float" } else { "ai_print_int" };
                // Widen integer prints to i64.
                let arg: BasicValueEnum = if is_float {
                    if matches!(&inner.repr, Repr::Scalar(ScalarRepr::F32)) {
                        self.builder.build_float_ext(v.into_float_value(), self.ctx.f64_type(), "pw").unwrap().into()
                    } else { v }
                } else {
                    let iv = v.into_int_value();
                    let signed = matches!(&inner.repr, Repr::Scalar(s) if s.is_signed());
                    if iv.get_type().get_bit_width() < 64 {
                        if signed {
                            self.builder.build_int_s_extend(iv, self.ctx.i64_type(), "pw").unwrap().into()
                        } else {
                            self.builder.build_int_z_extend(iv, self.ctx.i64_type(), "pw").unwrap().into()
                        }
                    } else { iv.into() }
                };
                let f = self.module.get_function(fname).unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into(), arg.into()], "print").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::PrintStr(s) => {
                let sv = self.gen_expr(fcx, s)?.unwrap().into_pointer_value();
                let f = self.module.get_function("ai_print_str").unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into(), sv.into()], "prints").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::PrintStrRaw(s) => {
                let sv = self.gen_expr(fcx, s)?.unwrap().into_pointer_value();
                let f = self.module.get_function("ai_print_str_raw").unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into(), sv.into()], "printsr").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::StrLen(s) => {
                let sv = self.gen_expr(fcx, s)?.unwrap().into_pointer_value();
                let f = self.module.get_function("ai_str_len").unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into(), sv.into()], "strlen").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::StrEq(a, b) => {
                let av = self.gen_expr(fcx, a)?.unwrap().into_pointer_value();
                let bv = self.gen_expr(fcx, b)?.unwrap().into_pointer_value();
                let f = self.module.get_function("ai_str_eq").unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into(), av.into(), bv.into()], "streq").unwrap());
                // ai_str_eq returns i64 (0/1); narrow to i1 for the bool repr.
                let iv = r.into_int_value();
                let b1 = self.builder.build_int_truncate(iv, self.ctx.bool_type(), "streqb").unwrap();
                Ok(Some(b1.into()))
            }
            CoreExprKind::StrConcat { layout, a, b } => {
                // `a`/`b` are live GC roots that the allocating call may move, so
                // they must already be spilled to frame slots by the root
                // discipline (same as any other runtime call that allocates).
                let av = self.gen_expr(fcx, a)?.unwrap().into_pointer_value();
                let bv = self.gen_expr(fcx, b)?.unwrap().into_pointer_value();
                let i32t = self.ctx.i32_type();
                let f = self.module.get_function("ai_str_concat").unwrap();
                let r = call_result(self.builder.build_call(
                    f,
                    &[fcx.thread.into(), i32t.const_int(*layout as u64, false).into(), av.into(), bv.into()],
                    "strcat",
                ).unwrap());
                Ok(Some(r))
            }
            CoreExprKind::StrGet(s, i) => {
                let sv = self.gen_expr(fcx, s)?.unwrap().into_pointer_value();
                let iv = self.gen_expr(fcx, i)?.unwrap().into_int_value();
                let f = self.module.get_function("ai_str_get").unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into(), sv.into(), iv.into()], "strget").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::StrSubstring { layout, s, start, end } => {
                let sv = self.gen_expr(fcx, s)?.unwrap().into_pointer_value();
                let st = self.gen_expr(fcx, start)?.unwrap().into_int_value();
                let en = self.gen_expr(fcx, end)?.unwrap().into_int_value();
                let i32t = self.ctx.i32_type();
                let f = self.module.get_function("ai_str_substring").unwrap();
                let r = call_result(self.builder.build_call(
                    f,
                    &[fcx.thread.into(), i32t.const_int(*layout as u64, false).into(), sv.into(), st.into(), en.into()],
                    "strsub",
                ).unwrap());
                Ok(Some(r))
            }
            CoreExprKind::ReadFile { layout, path } => {
                let pv = self.gen_expr(fcx, path)?.unwrap().into_pointer_value();
                let i32t = self.ctx.i32_type();
                let f = self.module.get_function("ai_read_file").unwrap();
                let r = call_result(self.builder.build_call(
                    f,
                    &[fcx.thread.into(), i32t.const_int(*layout as u64, false).into(), pv.into()],
                    "readfile",
                ).unwrap());
                Ok(Some(r))
            }
            CoreExprKind::StrToFloat(s) => {
                let sv = self.gen_expr(fcx, s)?.unwrap().into_pointer_value();
                let f = self.module.get_function("ai_str_to_float").unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into(), sv.into()], "strtof").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::FloatBits(f) => {
                let fv = self.gen_expr(fcx, f)?.unwrap().into_float_value();
                let r = self.builder.build_bit_cast(fv, self.ctx.i64_type(), "fbits").unwrap();
                Ok(Some(r))
            }
            CoreExprKind::StrHash(s) => {
                let sv = self.gen_expr(fcx, s)?.unwrap().into_pointer_value();
                let f = self.module.get_function("ai_str_hash").unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into(), sv.into()], "strhash").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::TypeIdOf(obj) => {
                // The type_id is a u16 in the object header at offset 8
                // ([gc_word: u64 @0][type_id: u16 @8]); zero-extend to i64.
                let ov = self.gen_expr(fcx, obj)?.unwrap().into_pointer_value();
                let i16t = self.ctx.i16_type();
                let i64t = self.ctx.i64_type();
                let addr = self.obj_addr(ov, 8);
                let tid = self.builder.build_load(i16t, addr, "tid").unwrap().into_int_value();
                let r = self.builder.build_int_z_extend(tid, i64t, "tid64").unwrap();
                Ok(Some(r.into()))
            }
            CoreExprKind::TypeNameOf { layout, obj } => {
                let ov = self.gen_expr(fcx, obj)?.unwrap().into_pointer_value();
                let i32t = self.ctx.i32_type();
                let f = self.module.get_function("ai_type_name").unwrap();
                let r = call_result(self.builder.build_call(
                    f,
                    &[fcx.thread.into(), i32t.const_int(*layout as u64, false).into(), ov.into()],
                    "typename",
                ).unwrap());
                Ok(Some(r))
            }
            CoreExprKind::AsCBytes { src, elem, is_string, copy_out } => {
                // Copy a String/scalar-Array's heap bytes into a dynamically-sized
                // STACK buffer and yield its pointer. The stack never moves, so
                // the pointer is stable for the enclosing extern call with no
                // pinning. A `copy_out` (mut array) buffer is written back into
                // the heap object after the call (queued for `gen_call` to drain).
                // See `docs/ffi.md`.
                let sv = self.gen_expr(fcx, src)?.unwrap().into_pointer_value();
                let i8t = self.ctx.i8_type();
                let i64t = self.ctx.i64_type();
                let stride = (elem.bits().max(8) / 8) as u64;

                if *is_string {
                    // byte_len = ai_str_len(s); buffer = byte_len + 1 (NUL).
                    let lenf = self.module.get_function("ai_str_len").unwrap();
                    let len = call_result(
                        self.builder.build_call(lenf, &[fcx.thread.into(), sv.into()], "ascb.len").unwrap()
                    ).into_int_value();
                    let cap = self.builder.build_int_add(len, i64t.const_int(1, false), "ascb.cap").unwrap();
                    let buf = self.builder.build_array_alloca(i8t, cap, "ascb.buf").unwrap();
                    let cpf = self.module.get_function("ai_str_copy_to_buf").unwrap();
                    self.builder.build_call(cpf, &[fcx.thread.into(), sv.into(), buf.into()], "").unwrap();
                    return Ok(Some(buf.into()));
                }

                // Scalar array: byte_len from the varlen count word at HEADER.
                let lid = match &src.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("as_c_bytes on non-array".into())) };
                let is_values = matches!(self.prog.layouts[lid as usize].varlen, crate::core::VarLen::Values);
                let count_addr = self.obj_addr(sv, Self::HEADER);
                let count = self.builder.build_load(i64t, count_addr, "ascb.cnt").unwrap().into_int_value();
                let byte_len = if is_values {
                    self.builder.build_int_mul(count, i64t.const_int(stride, false), "ascb.bytes").unwrap()
                } else {
                    count // Bytes array: count word already holds the byte length.
                };
                let buf = self.builder.build_array_alloca(i8t, byte_len, "ascb.buf").unwrap();
                let cin = self.module.get_function("ai_buf_copy_in").unwrap();
                self.builder.build_call(cin, &[fcx.thread.into(), sv.into(), buf.into(), byte_len.into()], "").unwrap();
                if *copy_out {
                    // Queue a write-back; gen_call replays it after the extern call.
                    fcx.pending_copy_outs.push((sv, buf, byte_len));
                }
                Ok(Some(buf.into()))
            }
            CoreExprKind::StrFromNum { layout, is_float, v } => {
                let vv = self.gen_expr(fcx, v)?.unwrap();
                let i32t = self.ctx.i32_type();
                let fname = if *is_float { "ai_str_from_float" } else { "ai_str_from_int" };
                let f = self.module.get_function(fname).unwrap();
                let r = call_result(self.builder.build_call(
                    f,
                    &[fcx.thread.into(), i32t.const_int(*layout as u64, false).into(), vv.into()],
                    "strfromnum",
                ).unwrap());
                Ok(Some(r))
            }
            CoreExprKind::StrFromChar { layout, cp } => {
                let cpv = self.gen_expr(fcx, cp)?.unwrap();
                let i32t = self.ctx.i32_type();
                let f = self.module.get_function("ai_char_to_str").unwrap();
                let r = call_result(self.builder.build_call(
                    f,
                    &[fcx.thread.into(), i32t.const_int(*layout as u64, false).into(), cpv.into()],
                    "strfromchar",
                ).unwrap());
                Ok(Some(r))
            }
            CoreExprKind::Cast { value, from, to } => self.gen_cast(fcx, value, from, to),
            CoreExprKind::Call(fid, args) => self.gen_call(fcx, *fid, args),
            CoreExprKind::PtrReadI64(p) => {
                let pv = self.gen_expr(fcx, p)?.unwrap().into_pointer_value();
                let v = self.builder.build_load(self.ctx.i64_type(), pv, "ptrd").unwrap();
                Ok(Some(v))
            }
            CoreExprKind::CallbackPtr(fid) => {
                // The address of the synthesized C-ABI trampoline for `fid`.
                let tramp = self.callback_trampoline(*fid);
                Ok(Some(tramp.as_global_value().as_pointer_value().into()))
            }
            CoreExprKind::ThreadSpawn(closure) => {
                // Evaluate the closure env, read its code pointer (first word of
                // the raw section, like gen_call_closure), and hand both to the
                // runtime. ai_thread_spawn registers a child mutator + runs it.
                let env = self.gen_expr(fcx, closure)?.unwrap().into_pointer_value();
                // Recover the code pointer at runtime from the env's header
                // type_id — the static repr here may be the placeholder closure
                // layout (ptr_fields=0) which gives the WRONG offset for a closure
                // that captured GC pointers. See `ai_closure_code_ptr`.
                let codef = self.module.get_function("ai_closure_code_ptr").unwrap();
                let code = call_result(self.builder.build_call(
                    codef, &[fcx.thread.into(), env.into()], "code",
                ).unwrap()).into_pointer_value();
                let f = self.module.get_function("ai_thread_spawn").unwrap();
                let r = call_result(self.builder.build_call(
                    f, &[fcx.thread.into(), env.into(), code.into()], "spawn",
                ).unwrap());
                Ok(Some(r))
            }
            CoreExprKind::ThreadJoin(handle) => {
                let h = self.gen_expr(fcx, handle)?.unwrap().into_pointer_value();
                let f = self.module.get_function("ai_thread_join").unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into(), h.into()], "join").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::AtomLoad { atom, elem } => {
                // Atomic load of the atom's value field (first pointer slot).
                let a = self.gen_expr(fcx, atom)?.unwrap().into_pointer_value();
                let slot = self.obj_addr(a, Self::HEADER);
                let lty = self.llvm_ty(elem).ok_or_else(|| CodegenError("atom of unit".into()))?;
                let load = self.builder.build_load(lty, slot, "atom.load").unwrap();
                let inst = load.as_instruction_value().unwrap();
                // Atomics require natural alignment (8 for our pointer/i64-wide
                // value fields); the default field alignment (4) is UB for an
                // atomic load and faults at runtime.
                inst.set_alignment(8).ok();
                inst.set_atomic_ordering(AtomicOrdering::SequentiallyConsistent).ok();
                Ok(Some(load))
            }
            CoreExprKind::AtomCas { atom, old, new } => {
                // Atomic compare-and-swap the value field; barrier on success.
                let a = self.gen_expr(fcx, atom)?.unwrap().into_pointer_value();
                let oldv = self.gen_expr(fcx, old)?.unwrap();
                let newv = self.gen_expr(fcx, new)?.unwrap();
                let slot = self.obj_addr(a, Self::HEADER);
                let cas = self.builder.build_cmpxchg(
                    slot, oldv, newv,
                    AtomicOrdering::SequentiallyConsistent,
                    AtomicOrdering::SequentiallyConsistent,
                ).unwrap();
                // cmpxchg requires natural alignment of the value field.
                cas.as_instruction_value().unwrap().set_alignment(8).ok();
                // cmpxchg yields { value, i1 success }; element 1 is the success bool.
                let ok = self.builder.build_extract_value(cas, 1, "cas.ok").unwrap().into_int_value();
                // Generational write barrier: a successful install may create an
                // old(atom)→young(new) edge. (No-ops cheaply if non-generational.)
                self.emit_write_barrier(fcx, a, newv);
                Ok(Some(ok.into()))
            }
            CoreExprKind::ChanSend { buf, ctrl, value } => {
                let b = self.gen_expr(fcx, buf)?.unwrap().into_pointer_value();
                let c = self.gen_expr(fcx, ctrl)?.unwrap().into_pointer_value();
                let v = self.gen_expr(fcx, value)?.unwrap().into_pointer_value();
                let f = self.module.get_function("ai_chan_send").unwrap();
                let r = call_result(self.builder.build_call(
                    f, &[fcx.thread.into(), b.into(), c.into(), v.into()], "chsend").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::ChanRecv { buf, ctrl, elem: _ } => {
                let b = self.gen_expr(fcx, buf)?.unwrap().into_pointer_value();
                let c = self.gen_expr(fcx, ctrl)?.unwrap().into_pointer_value();
                let f = self.module.get_function("ai_chan_recv").unwrap();
                let r = call_result(self.builder.build_call(
                    f, &[fcx.thread.into(), b.into(), c.into()], "chrecv").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::RuntimeCall { func, args, ret } => {
                let f = self.module.get_function(func)
                    .ok_or_else(|| CodegenError(format!("unknown runtime extern `{}`", func)))?;
                let mut cargs: Vec<inkwell::values::BasicMetadataValueEnum> = vec![fcx.thread.into()];
                for a in args {
                    if let Some(v) = self.gen_expr(fcx, a)? {
                        cargs.push(v.into());
                    }
                }
                let cs = self.builder.build_call(f, &cargs, "rtcall").unwrap();
                match (cs.try_as_basic_value(), self.llvm_ty(ret)) {
                    (inkwell::values::ValueKind::Basic(v), _) => Ok(Some(v)),
                    (inkwell::values::ValueKind::Instruction(_), _) => Ok(None),
                }
            }
            CoreExprKind::ThreadSleep(ms) => {
                let m = self.gen_expr(fcx, ms)?.unwrap().into_int_value();
                let f = self.module.get_function("ai_thread_sleep").unwrap();
                self.builder.build_call(f, &[fcx.thread.into(), m.into()], "").unwrap();
                Ok(Some(self.ctx.i64_type().const_zero().into()))
            }
            CoreExprKind::ThreadYield => {
                let f = self.module.get_function("ai_thread_yield").unwrap();
                self.builder.build_call(f, &[fcx.thread.into()], "").unwrap();
                Ok(Some(self.ctx.i64_type().const_zero().into()))
            }
            CoreExprKind::ThreadCurrentId => {
                let f = self.module.get_function("ai_thread_current_id").unwrap();
                let r = call_result(self.builder.build_call(f, &[fcx.thread.into()], "tid").unwrap());
                Ok(Some(r))
            }
            CoreExprKind::If(cond, then_b, else_b) => self.gen_if(fcx, cond, then_b, else_b, &e.repr),
            CoreExprKind::Block(b) => self.gen_block(fcx, b),
            CoreExprKind::Loop(body) => self.gen_loop(fcx, body),
            CoreExprKind::Break(v) => {
                let bv = match v {
                    Some(e) => self.gen_expr(fcx, e)?,
                    None => None,
                };
                let (cont_bb, phi_in) = *fcx.loops.last()
                    .ok_or_else(|| CodegenError("break outside loop".into()))?;
                if let (Some(slot), Some(bv)) = (phi_in, bv) {
                    self.builder.build_store(slot, bv).unwrap();
                }
                self.builder.build_unconditional_branch(cont_bb).unwrap();
                Ok(None)
            }
            CoreExprKind::Continue => {
                let header_bb = fcx.loop_headers.last()
                    .ok_or_else(|| CodegenError("continue outside loop".into()))?;
                self.builder.build_unconditional_branch(*header_bb).unwrap();
                Ok(None)
            }
            CoreExprKind::Assign { local, value } => {
                let v = self.gen_expr(fcx, value)?;
                if let (Some(slot), Some(v)) = (fcx.slots[*local as usize], v) {
                    self.builder.build_store(slot, v).unwrap();
                }
                Ok(None)
            }
            CoreExprKind::Return(v) => {
                match v {
                    Some(e) => {
                        let rv = self.gen_expr(fcx, e)?;
                        self.emit_unlink(fcx);
                        match rv {
                            Some(rv) => { self.builder.build_return(Some(&rv)).unwrap(); }
                            None => { self.builder.build_return(None).unwrap(); }
                        }
                    }
                    None => { self.emit_unlink(fcx); self.builder.build_return(None).unwrap(); }
                }
                Ok(None)
            }
            CoreExprKind::MakeValue { value, fields } => self.gen_make_value(fcx, *value, fields),
            CoreExprKind::MakeValueVariant { value, tag, fields } => {
                self.gen_make_value_variant(fcx, *value, *tag, fields)
            }
            CoreExprKind::ValueMatch { scrutinee, arms } => self.gen_value_match(fcx, scrutinee, arms, &e.repr),
            CoreExprKind::New { layout, fields } => self.gen_alloc(fcx, *layout, None, fields, e.span),
            CoreExprKind::MakeVariant { layout, tag, fields } => {
                self.gen_alloc(fcx, *layout, Some(*tag), fields, e.span)
            }
            CoreExprKind::Field { base, loc } => self.gen_field(fcx, base, loc),
            CoreExprKind::SetField { base, loc, value } => {
                let obj = self.gen_expr(fcx, base)?.unwrap().into_pointer_value();
                let v = self.gen_expr(fcx, value)?.unwrap();
                let off = match loc {
                    FieldLoc::Ptr { idx } => Self::HEADER + (*idx as u64) * 8,
                    FieldLoc::Raw { offset, .. } | FieldLoc::ValueAt { offset, .. } => {
                        let lid = match &base.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("setfield on non-ref".into())) };
                        let lay = &self.prog.layouts[lid as usize];
                        Self::HEADER + (lay.ptr_fields as u64) * 8 + *offset as u64
                    }
                    FieldLoc::ValueField { .. } => return Err(CodegenError("value-struct field is immutable".into())),
                };
                let addr = self.obj_addr(obj, off);
                self.builder.build_store(addr, v).unwrap();
                // Generational write barrier. A direct pointer store (FieldLoc::Ptr)
                // may create an old→young edge. So may a flattened value-with-
                // references store (FieldLoc::ValueAt): each reference embedded in
                // the value is its own potential old→young edge. The old code
                // barriered only Ptr stores, so mutating a value-with-ref field in a
                // TENURED object left the card unmarked — a minor GC would not
                // rescan it and could reclaim a still-referenced young object. Mark
                // for each interior reference of the stored value.
                match loc {
                    FieldLoc::Ptr { .. } => self.emit_write_barrier(fcx, obj, v),
                    FieldLoc::ValueAt { value, .. } => {
                        // `lay.interior_ptrs` are absolute object offsets of refs
                        // embedded in flattened value fields; those within this
                        // field's byte range [off, off+size) are exactly the refs
                        // of the value we just stored. Re-load each and barrier it
                        // (the barrier no-ops unless obj is tenured and the ref is
                        // young), so the card is marked iff a real old→young edge
                        // was created.
                        let lid = match &base.repr {
                            Repr::Ref(l) => *l,
                            _ => return Err(CodegenError("setfield on non-ref".into())),
                        };
                        let lay = &self.prog.layouts[lid as usize];
                        let vsize = self.prog.values[*value as usize].size as u64;
                        let (lo, hi) = (off, off + vsize);
                        let interior: Vec<u64> = lay
                            .interior_ptrs
                            .iter()
                            .map(|&p| p as u64)
                            .filter(|&p| p >= lo && p < hi)
                            .collect();
                        let ptrty = self.ctx.ptr_type(AddressSpace::default());
                        for p in interior {
                            let pa = self.obj_addr(obj, p);
                            let refv = self.builder.build_load(ptrty, pa, "iref").unwrap();
                            self.emit_write_barrier(fcx, obj, refv);
                        }
                    }
                    _ => {}
                }
                Ok(Some(self.ctx.i64_type().const_zero().into()))
            }
            CoreExprKind::Match { scrutinee, arms } => self.gen_match(fcx, scrutinee, arms, &e.repr),
            CoreExprKind::EnumTag(scrut) => {
                let obj = self.gen_expr(fcx, scrut)?.unwrap().into_pointer_value();
                let lid = match &scrut.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("EnumTag on non-ref enum".into())) };
                let ptr_fields = self.prog.layouts[lid as usize].ptr_fields as u64;
                let i32t = self.ctx.i32_type();
                let tag_off = Self::HEADER + ptr_fields * 8;
                let addr = self.obj_addr(obj, tag_off);
                let tag = self.builder.build_load(i32t, addr, "etag").unwrap();
                Ok(Some(tag))
            }
            CoreExprKind::EnumPayload { scrutinee, field, repr, payload_reprs } => {
                let obj = self.gen_expr(fcx, scrutinee)?.unwrap().into_pointer_value();
                let lid = match &scrutinee.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("EnumPayload on non-ref enum".into())) };
                let ptr_fields = self.prog.layouts[lid as usize].ptr_fields as u64;
                let tag_off = Self::HEADER + ptr_fields * 8;
                let val = self.load_enum_payload(obj, tag_off, *field as usize, repr, payload_reprs);
                Ok(Some(val))
            }
            CoreExprKind::ArrayNew { layout, len, elem } => self.gen_array_new(fcx, *layout, len, elem, e.span),
            CoreExprKind::ArrayLen(arr) => self.gen_array_len(fcx, arr),
            CoreExprKind::ArrayGet { array, index, elem } => self.gen_array_get(fcx, array, index, elem, &e.repr),
            CoreExprKind::ArrayGetUnchecked { array, index, elem } => self.gen_array_get_unchecked(fcx, array, index, elem),
            CoreExprKind::ArrayGetChecked { array, index, elem } => self.gen_array_get_checked(fcx, array, index, elem),
            CoreExprKind::ArraySet { array, index, value, elem } => self.gen_array_set(fcx, array, index, value, elem),
            CoreExprKind::MakeClosure { code, env, captures } => self.gen_make_closure(fcx, *code, *env, captures, e.span),
            CoreExprKind::CallClosure { callee, args } => self.gen_call_closure(fcx, callee, args, &e.repr),
            other => Err(CodegenError(format!("codegen unsupported in v0 slice: {:?}", core_disc(other)))),
        }
    }

    /// Return the allocation-site id (Target-1b) for `(function, type_id)`,
    /// assigning a fresh id and recording the site on first sight. Dedups by
    /// pair so the v1 function+type granularity stays honest: every emitted id
    /// maps to exactly one labelable `(function, type)`, never a distinct id we
    /// couldn't tell apart at profile time.
    fn alloc_site_id(&mut self, function: &str, type_id: u16, span: crate::core::SpanId) -> u32 {
        // Resolve the construction node's span to `file:line:col` if source is
        // available (debugger P1 span-threading); empty otherwise. The location
        // is part of the dedup key, so two allocations of the same type in the
        // same function at DIFFERENT lines become DISTINCT sites — the precise
        // upgrade over the old (function, type)-only key.
        let location = self.prog.span_label(span).unwrap_or_default();
        let key = (function.to_string(), type_id, location.clone());
        if let Some(&id) = self.alloc_site_ids.get(&key) {
            return id;
        }
        let id = self.alloc_sites.len() as u32;
        self.alloc_sites.push(crate::gc::AllocSite {
            function: function.to_string(),
            type_id,
            location,
        });
        self.alloc_site_ids.insert(key, id);
        id
    }

    /// The current function's compiled symbol name (the site label).
    fn current_fn_name(fcx: &FnCtx<'ctx>) -> String {
        fcx.func
            .get_name()
            .to_str()
            .unwrap_or("<unknown-fn>")
            .to_string()
    }

    fn declare_runtime_externs(&self) {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let i32t = self.ctx.i32_type();
        let i64t = self.ctx.i64_type();
        // ptr ai_gc_alloc_fixed(ptr thread, i32 type_id, i32 site_id)
        self.module.add_function(
            "ai_gc_alloc_fixed",
            ptr.fn_type(&[ptr.into(), i32t.into(), i32t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_gc_alloc_varlen(ptr thread, i32 type_id, i64 n, i32 site_id)
        self.module.add_function(
            "ai_gc_alloc_varlen",
            ptr.fn_type(&[ptr.into(), i32t.into(), i64t.into(), i32t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // void ai_gc_write_barrier(ptr thread, ptr obj, ptr new_val)
        self.module.add_function(
            "ai_gc_write_barrier",
            self.ctx.void_type().fn_type(&[ptr.into(), ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // void ai_gc_pollcheck_slow(ptr thread)
        self.module.add_function(
            "ai_gc_pollcheck_slow",
            self.ctx.void_type().fn_type(&[ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // void ai_bounds_fail(ptr thread, i64 index, i64 len) -- aborts, never
        // returns. Marked noreturn + cold so the inlined array bounds check keeps
        // the in-bounds path straight-line and the trap path out of line.
        {
            let f = self.module.add_function(
                "ai_bounds_fail",
                self.ctx.void_type().fn_type(&[ptr.into(), i64t.into(), i64t.into()], false),
                Some(inkwell::module::Linkage::External),
            );
            use inkwell::attributes::AttributeLoc;
            for name in ["noreturn", "cold"] {
                let kind = inkwell::attributes::Attribute::get_named_enum_kind_id(name);
                f.add_attribute(AttributeLoc::Function, self.ctx.create_enum_attribute(kind, 0));
            }
        }
        // void ai_ffi_enter(ptr thread) / void ai_ffi_leave(ptr thread)
        // The managed↔native transition wrapped around every extern "C" call.
        self.module.add_function(
            "ai_ffi_enter",
            self.ctx.void_type().fn_type(&[ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        self.module.add_function(
            "ai_ffi_leave",
            self.ctx.void_type().fn_type(&[ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // Callback support: ptr ai_current_thread(void); reenter/exit transitions.
        self.module.add_function(
            "ai_current_thread",
            ptr.fn_type(&[], false),
            Some(inkwell::module::Linkage::External),
        );
        self.module.add_function(
            "ai_ffi_reenter",
            self.ctx.void_type().fn_type(&[ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        self.module.add_function(
            "ai_ffi_exit",
            self.ctx.void_type().fn_type(&[ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_print_int(ptr thread, i64 v)
        self.module.add_function(
            "ai_print_int",
            i64t.fn_type(&[ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_print_float(ptr thread, f64 v)
        self.module.add_function(
            "ai_print_float",
            i64t.fn_type(&[ptr.into(), self.ctx.f64_type().into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // String primitives (see crates/gcrust-rt/src/runtime.rs).
        // i64 ai_print_str(ptr thread, ptr s)
        self.module.add_function(
            "ai_print_str",
            i64t.fn_type(&[ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_print_str_raw(ptr thread, ptr s) — no trailing newline
        self.module.add_function(
            "ai_print_str_raw",
            i64t.fn_type(&[ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_str_len(ptr thread, ptr s)
        self.module.add_function(
            "ai_str_len",
            i64t.fn_type(&[ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_str_eq(ptr thread, ptr a, ptr b)
        self.module.add_function(
            "ai_str_eq",
            i64t.fn_type(&[ptr.into(), ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_str_concat(ptr thread, i32 type_id, ptr a, ptr b)
        self.module.add_function(
            "ai_str_concat",
            ptr.fn_type(&[ptr.into(), i32t.into(), ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_str_get(ptr thread, ptr s, i64 i)
        self.module.add_function(
            "ai_str_get",
            i64t.fn_type(&[ptr.into(), ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_str_substring(ptr thread, i32 type_id, ptr s, i64 start, i64 end)
        self.module.add_function(
            "ai_str_substring",
            ptr.fn_type(&[ptr.into(), i32t.into(), ptr.into(), i64t.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_read_file(ptr thread, i32 type_id, ptr path)
        self.module.add_function(
            "ai_read_file",
            ptr.fn_type(&[ptr.into(), i32t.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_str_from_int(ptr thread, i32 type_id, i64 v)
        self.module.add_function(
            "ai_str_from_int",
            ptr.fn_type(&[ptr.into(), i32t.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_char_to_str(ptr thread, i32 type_id, i64 cp)
        self.module.add_function(
            "ai_char_to_str",
            ptr.fn_type(&[ptr.into(), i32t.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_type_name(ptr thread, i32 str_type_id, ptr obj)
        self.module.add_function(
            "ai_type_name",
            ptr.fn_type(&[ptr.into(), i32t.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // void ai_heap_snapshot(ptr thread) — heap-explorer P2 on-demand snapshot.
        self.module.add_function(
            "ai_heap_snapshot",
            self.ctx.void_type().fn_type(&[ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // In-language field reflection (all thread-first; RuntimeCall-dispatched).
        // i64 ai_reflect_field_count(ptr thread, ptr obj)
        self.module.add_function(
            "ai_reflect_field_count",
            i64t.fn_type(&[ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_reflect_field_kind(ptr thread, ptr obj, i64 i)
        self.module.add_function(
            "ai_reflect_field_kind",
            i64t.fn_type(&[ptr.into(), ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_reflect_field_i64(ptr thread, ptr obj, i64 i)
        self.module.add_function(
            "ai_reflect_field_i64",
            i64t.fn_type(&[ptr.into(), ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_reflect_field_name(ptr thread, i64 str_type_id, ptr obj, i64 i)
        self.module.add_function(
            "ai_reflect_field_name",
            ptr.fn_type(&[ptr.into(), i64t.into(), ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_str_from_float(ptr thread, i32 type_id, f64 v)
        self.module.add_function(
            "ai_str_from_float",
            ptr.fn_type(&[ptr.into(), i32t.into(), self.ctx.f64_type().into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // f64 ai_str_to_float(ptr thread, ptr s)
        self.module.add_function(
            "ai_str_to_float",
            self.ctx.f64_type().fn_type(&[ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_str_hash(ptr thread, ptr s)
        self.module.add_function(
            "ai_str_hash",
            i64t.fn_type(&[ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_str_copy_to_buf(ptr thread, ptr s, ptr dst)
        self.module.add_function(
            "ai_str_copy_to_buf",
            i64t.fn_type(&[ptr.into(), ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // void ai_buf_copy_in(ptr thread, ptr obj, ptr dst, i64 byte_len)
        self.module.add_function(
            "ai_buf_copy_in",
            self.ctx.void_type().fn_type(&[ptr.into(), ptr.into(), ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // void ai_buf_copy_out(ptr thread, ptr obj, ptr src, i64 byte_len)
        self.module.add_function(
            "ai_buf_copy_out",
            self.ctx.void_type().fn_type(&[ptr.into(), ptr.into(), ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // Threads. ptr ai_thread_spawn(ptr parent, ptr env, ptr code)
        self.module.add_function(
            "ai_thread_spawn",
            ptr.fn_type(&[ptr.into(), ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_thread_join(ptr thread, ptr handle)
        self.module.add_function(
            "ai_thread_join",
            i64t.fn_type(&[ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // void ai_thread_sleep(ptr thread, i64 ms) / void ai_thread_yield(ptr thread)
        self.module.add_function(
            "ai_thread_sleep",
            self.ctx.void_type().fn_type(&[ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        self.module.add_function(
            "ai_thread_yield",
            self.ctx.void_type().fn_type(&[ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_thread_current_id(ptr thread)
        self.module.add_function(
            "ai_thread_current_id",
            i64t.fn_type(&[ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_closure_code_ptr(ptr thread, ptr env) — recover a closure's code
        // pointer from its env header type_id (correct for any capture count).
        self.module.add_function(
            "ai_closure_code_ptr",
            ptr.fn_type(&[ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // Channels.
        self.module.add_function("ai_chan_new",
            ptr.fn_type(&[ptr.into(), i64t.into()], false), Some(inkwell::module::Linkage::External));
        self.module.add_function("ai_chan_send",
            i64t.fn_type(&[ptr.into(), ptr.into(), ptr.into(), ptr.into()], false), Some(inkwell::module::Linkage::External));
        self.module.add_function("ai_chan_recv",
            ptr.fn_type(&[ptr.into(), ptr.into(), ptr.into()], false), Some(inkwell::module::Linkage::External));
        self.module.add_function("ai_chan_sender_clone",
            self.ctx.void_type().fn_type(&[ptr.into(), ptr.into()], false), Some(inkwell::module::Linkage::External));
        self.module.add_function("ai_chan_sender_drop",
            self.ctx.void_type().fn_type(&[ptr.into(), ptr.into()], false), Some(inkwell::module::Linkage::External));
        // AtomicI64. ptr ai_atomic_i64_new(ptr thread, i64 v)
        self.module.add_function(
            "ai_atomic_i64_new",
            ptr.fn_type(&[ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_atomic_i64_load(ptr thread, ptr a)
        self.module.add_function(
            "ai_atomic_i64_load",
            i64t.fn_type(&[ptr.into(), ptr.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_atomic_i64_store(ptr thread, ptr a, i64 v)
        self.module.add_function(
            "ai_atomic_i64_store",
            i64t.fn_type(&[ptr.into(), ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_atomic_i64_fetch_add(ptr thread, ptr a, i64 delta)
        self.module.add_function(
            "ai_atomic_i64_fetch_add",
            i64t.fn_type(&[ptr.into(), ptr.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // i64 ai_atomic_i64_compare_and_set(ptr thread, ptr a, i64 expected, i64 new)
        self.module.add_function(
            "ai_atomic_i64_compare_and_set",
            i64t.fn_type(&[ptr.into(), ptr.into(), i64t.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
    }

    /// The byte offset of a heap object's data start (the `Full` header size).
    const HEADER: u64 = 16;

    /// Whether a value of this repr can be moved by the GC — a heap reference, or
    /// a flattened `#[value]` aggregate holding one. Such a value cached in a
    /// register before a safepoint goes stale and must be reloaded after.
    fn repr_relocates(&self, r: &Repr) -> bool {
        match r {
            Repr::Ref(_) => true,
            Repr::Value(v) => value_has_ref(&self.prog.values, *v),
            _ => false,
        }
    }

    /// Allocate a heap object of `layout`, store `tag` (for enums) + `fields`,
    /// and return the pointer. The fields are evaluated and stored; pointer
    /// fields go in the leading pointer slots, raw fields at their byte offset.
    fn gen_alloc(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        layout: LayoutId,
        tag: Option<u32>,
        fields: &[CoreExpr],
        span: crate::core::SpanId,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let i32t = self.ctx.i32_type();
        let i64t = self.ctx.i64_type();

        // Evaluate field values FIRST (they may allocate; their own roots are
        // Evaluate field values FIRST. Then allocate, then store. The allocation
        // is a SAFEPOINT: a GC there relocates every rooted local, so any GC value
        // we cached in a register before the alloc is now STALE (it points at the
        // pre-relocation address). ANF guarantees every GC field is an atomic
        // local (or null), so after the alloc we RELOAD those fields from their
        // (now-relocated) slots — a side-effect-free load — before storing them.
        // Scalar/POD-value fields don't relocate, so their cached values stand.
        let mut vals = Vec::with_capacity(fields.len());
        for fe in fields {
            vals.push((self.gen_expr(fcx, fe)?, fe.repr.clone()));
        }

        let site = self.alloc_site_id(&Self::current_fn_name(fcx), layout as u16, span);
        let alloc = self.module.get_function("ai_gc_alloc_fixed").unwrap();
        let obj = call_result(
            self.builder.build_call(
                alloc,
                &[
                    fcx.thread.into(),
                    i32t.const_int(layout as u64, false).into(),
                    i32t.const_int(site as u64, false).into(),
                ],
                "obj",
            ).unwrap(),
        ).into_pointer_value();

        // Reload GC-valued fields from their slots post-allocation (see above).
        for (fe, slot) in fields.iter().zip(vals.iter_mut()) {
            if self.repr_relocates(&fe.repr) {
                slot.0 = self.gen_expr(fcx, fe)?;
            }
        }

        let lay = &self.prog.layouts[layout as usize];
        // Tag (for enums) at raw offset 0.
        if let Some(t) = tag {
            let raw_base = Self::HEADER + (lay.ptr_fields as u64) * 8;
            let addr = self.obj_addr(obj, raw_base);
            self.builder.build_store(addr, i32t.const_int(t as u64, false)).unwrap();
        }

        // Store fields. For an enum, payload fields go after the tag; we place
        // pointer payloads in the pointer slots and raw payloads after the tag
        // word. For a struct, use the layout's field_map.
        let mut ptr_slot = 0u64;
        let mut raw_cursor = Self::HEADER + (lay.ptr_fields as u64) * 8 + if tag.is_some() { 8 } else { 0 };
        for (v, repr) in &vals {
            match repr {
                Repr::Ref(_) => {
                    let off = Self::HEADER + ptr_slot * 8;
                    ptr_slot += 1;
                    let addr = self.obj_addr(obj, off);
                    if let Some(v) = v { self.builder.build_store(addr, *v).unwrap(); }
                }
                Repr::Scalar(s) => {
                    let sz = (s.bits().max(8) / 8) as u64;
                    raw_cursor = align_up64(raw_cursor, sz);
                    let addr = self.obj_addr(obj, raw_cursor);
                    raw_cursor += sz;
                    if let Some(v) = v { self.builder.build_store(addr, *v).unwrap(); }
                }
                Repr::Value(vid) => {
                    // A flattened value aggregate: store the whole LLVM struct at
                    // its inline byte offset. Offset progression mirrors
                    // `build_ref_layout` (align to 8, advance by the value's
                    // size), so loads via `FieldLoc::ValueAt` find it. Value types
                    // reaching here are POD (no embedded GC refs — `build_ref_layout`
                    // rejects ref-containing ones), so no write barrier is needed.
                    let sz = self.prog.values[*vid as usize].size as u64;
                    raw_cursor = align_up64(raw_cursor, 8);
                    let addr = self.obj_addr(obj, raw_cursor);
                    raw_cursor += sz;
                    if let Some(v) = v {
                        self.builder.build_store(addr, *v).unwrap();
                    }
                }
                Repr::Unit => {
                    // Zero-size: no storage.
                }
            }
        }
        let _ = i64t;
        Ok(Some(obj.into()))
    }

    /// Build an inline value-struct aggregate from its field values.
    fn gen_make_value(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        value: ValueId,
        fields: &[CoreExpr],
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let sty = self.value_struct_ty(value);
        let mut agg = sty.get_undef();
        let mut idx = 0u32;
        for fe in fields {
            let v = self.gen_expr(fcx, fe)?;
            if let Some(v) = v {
                agg = self.builder
                    .build_insert_value(agg, v, idx, "vf")
                    .unwrap()
                    .into_struct_value();
                idx += 1;
            }
        }
        Ok(Some(agg.into()))
    }

    /// Build an inline value-enum variant `{ i32 tag, [N x i8] payload }`. The
    /// payload fields are stored into the byte region via an alloca (typed GEPs
    /// + a final load); mem2reg keeps this in registers after O2.
    fn gen_make_value_variant(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        value: ValueId,
        tag: u32,
        fields: &[CoreExpr],
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        // Evaluate payload field values first, then build the aggregate.
        let mut vals = Vec::with_capacity(fields.len());
        for f in fields {
            if let Some(v) = self.gen_expr(fcx, f)? {
                vals.push((v, f.repr.clone()));
            }
        }
        Ok(Some(self.build_value_variant(value, tag, &vals)))
    }

    /// Build a value-enum variant aggregate from already-computed payload values
    /// (the guts of `gen_make_value_variant`, factored out so codegen-internal
    /// constructions — e.g. wrapping an array element in `Option::Some` — can
    /// reuse the exact layout logic without re-evaluating field expressions).
    fn build_value_variant(
        &self,
        value: ValueId,
        tag: u32,
        vals: &[(BasicValueEnum<'ctx>, Repr)],
    ) -> BasicValueEnum<'ctx> {
        let sty = self.value_struct_ty(value);
        let i32t = self.ctx.i32_type();
        let max_ptrs = crate::core::value_enum_max_ptrs(
            self.prog.values[value as usize].variants.as_ref().unwrap(),
        ) as u32;
        let slot = self.builder.build_alloca(sty, "ve").unwrap();
        self.builder.build_store(slot, sty.const_zero()).unwrap();

        if max_ptrs == 0 {
            // Compact `{ i32 tag, [payload bytes] }`: tag field 0, payload field 1.
            let tag_addr = self.builder.build_struct_gep(sty, slot, 0, "ve.tag").unwrap();
            self.builder.build_store(tag_addr, i32t.const_int(tag as u64, false)).unwrap();
            if sty.count_fields() > 1 {
                let payload_addr = self.builder.build_struct_gep(sty, slot, 1, "ve.pl").unwrap();
                let mut off = 0u64;
                for (v, repr) in vals {
                    let (sz, _) = Self::repr_size_align(repr);
                    off = align_up64(off, sz);
                    let field_addr = self.payload_field_addr(payload_addr, off);
                    self.builder.build_store(field_addr, *v).unwrap();
                    off += sz;
                }
            }
        } else {
            // Pointers-first `{ [max_ptrs x ptr], i32 tag, [raw] }`: ref payloads
            // into the shared leading slots (field 0), tag (field 1), scalars/
            // POD-values into the raw region (field 2). Mirrors the reference-enum
            // heap layout so embedded refs sit at fixed, GC-traceable offsets.
            let ptr = self.ctx.ptr_type(AddressSpace::default());
            let ptr_arr_ty = ptr.array_type(max_ptrs);
            let ptr_arr = self.builder.build_struct_gep(sty, slot, 0, "ve.ptrs").unwrap();
            let tag_addr = self.builder.build_struct_gep(sty, slot, 1, "ve.tag").unwrap();
            self.builder.build_store(tag_addr, i32t.const_int(tag as u64, false)).unwrap();
            let raw_addr = self.builder.build_struct_gep(sty, slot, 2, "ve.raw").unwrap();
            let mut ptr_slot = 0u64;
            let mut raw_off = 0u64;
            for (v, repr) in vals {
                match repr {
                    Repr::Ref(_) => {
                        let elem = unsafe {
                            self.builder.build_in_bounds_gep(
                                ptr_arr_ty, ptr_arr,
                                &[i32t.const_zero(), i32t.const_int(ptr_slot, false)],
                                "ve.pslot",
                            ).unwrap()
                        };
                        ptr_slot += 1;
                        self.builder.build_store(elem, *v).unwrap();
                    }
                    _ => {
                        let (sz, _) = Self::repr_size_align(repr);
                        raw_off = align_up64(raw_off, sz);
                        let field_addr = self.payload_field_addr(raw_addr, raw_off);
                        raw_off += sz;
                        self.builder.build_store(field_addr, *v).unwrap();
                    }
                }
            }
        }
        self.builder.build_load(sty, slot, "ve.val").unwrap()
    }

    /// Address of a payload field at `byte_off` within the value-enum's byte
    /// region (an `[N x i8]` base pointer).
    fn payload_field_addr(&self, payload_base: PointerValue<'ctx>, byte_off: u64) -> PointerValue<'ctx> {
        let i64t = self.ctx.i64_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let base = self.builder.build_ptr_to_int(payload_base, i64t, "pl.i").unwrap();
        let addr = self.builder.build_int_add(base, i64t.const_int(byte_off, false), "pl.a").unwrap();
        self.builder.build_int_to_ptr(addr, ptr, "pl.p").unwrap()
    }

    /// Byte size + alignment of a repr (for laying out value-enum payloads).
    fn repr_size_align(repr: &Repr) -> (u64, u64) {
        match repr {
            Repr::Unit => (0, 1),
            Repr::Scalar(s) => { let b = (s.bits().max(8) / 8) as u64; (b, b) }
            Repr::Ref(_) => (8, 8),
            Repr::Value(_) => (8, 8), // nested value aggregates: conservative
        }
    }

    fn obj_addr(&self, obj: PointerValue<'ctx>, byte_off: u64) -> PointerValue<'ctx> {
        let i64t = self.ctx.i64_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let base = self.builder.build_ptr_to_int(obj, i64t, "o.i").unwrap();
        let addr = self.builder.build_int_add(base, i64t.const_int(byte_off, false), "o.fa").unwrap();
        self.builder.build_int_to_ptr(addr, ptr, "o.fp").unwrap()
    }

    /// Emit a generational write barrier: `ai_gc_write_barrier(thread, obj, new)`
    /// after storing pointer `new` into a field/element of heap object `obj`.
    /// Called unconditionally on every pointer store; the runtime fast-paths the
    /// non-generational / non-old→young cases. Keeps the minor GC's old→young
    /// root set correct — a missed barrier would be a use-after-free.
    fn emit_write_barrier(&self, fcx: &FnCtx<'ctx>, obj: PointerValue<'ctx>, new_val: BasicValueEnum<'ctx>) {
        if !new_val.is_pointer_value() {
            return; // only pointer stores can create an old→young edge
        }
        let f = self.module.get_function("ai_gc_write_barrier").unwrap();
        self.builder
            .build_call(f, &[fcx.thread.into(), obj.into(), new_val.into()], "")
            .unwrap();
    }

    /// The byte stride of an array element repr, and whether it's a traced
    /// (pointer/Values) array.
    fn elem_stride(elem: &Repr) -> (u64, bool) {
        match elem {
            Repr::Ref(_) => (8, true),
            Repr::Scalar(s) => ((s.bits().max(8) / 8) as u64, false),
            _ => (8, false),
        }
    }

    /// Sign-extend an array index to i64 (indices are i64-typed for addressing
    /// and the bounds check).
    fn idx_to_i64(&self, index: IntValue<'ctx>) -> IntValue<'ctx> {
        let i64t = self.ctx.i64_type();
        if index.get_type().get_bit_width() < 64 {
            self.builder.build_int_s_extend(index, i64t, "a.ix").unwrap()
        } else {
            index
        }
    }

    /// Element address: HEADER + 8 (count word) + index*stride, as an `inbounds`
    /// GEP off the array pointer. Using a GEP (rather than ptrtoint/inttoptr
    /// arithmetic) preserves pointer provenance, so LLVM's alias analysis and
    /// strength reduction can optimize loops of array accesses. `idx64` must
    /// already be i64 (see `idx_to_i64`).
    fn array_elem_addr(&self, arr: PointerValue<'ctx>, idx64: IntValue<'ctx>, stride: u64) -> PointerValue<'ctx> {
        let i8t = self.ctx.i8_type();
        let i64t = self.ctx.i64_type();
        let off = self.builder.build_int_mul(idx64, i64t.const_int(stride, false), "a.off").unwrap();
        let off = self.builder.build_int_add(off, i64t.const_int(Self::HEADER + 8, false), "a.h").unwrap();
        unsafe { self.builder.build_in_bounds_gep(i8t, arr, &[off], "a.ep").unwrap() }
    }

    /// Load an array's logical element count for a bounds check. The count word
    /// (at `HEADER`) holds the element count for Values (traced) arrays and the
    /// byte length for Bytes (scalar) arrays; the latter is divided by the
    /// element stride. `arr_repr` is the array object's repr (`Repr::Ref(lid)`).
    fn array_logical_len(&self, obj: PointerValue<'ctx>, arr_repr: &Repr) -> Result<IntValue<'ctx>, CodegenError> {
        let i64t = self.ctx.i64_type();
        let count_addr = self.obj_addr(obj, Self::HEADER);
        let count_load = self.builder.build_load(i64t, count_addr, "cnt").unwrap();
        // The array length is immutable after allocation, so the count word never
        // changes. Mark the load `!invariant.load` so LLVM may coalesce repeated
        // bounds-check length loads and not treat element stores as clobbering it.
        if let Some(inst) = count_load.as_instruction_value() {
            let kind = self.ctx.get_kind_id("invariant.load");
            let node = self.ctx.metadata_node(&[]);
            inst.set_metadata(node, kind).ok();
        }
        let count = count_load.into_int_value();
        let lid = match arr_repr {
            Repr::Ref(l) => *l,
            _ => return Err(CodegenError("array bounds check on non-array".into())),
        };
        let lay = &self.prog.layouts[lid as usize];
        let len = if matches!(lay.varlen, crate::core::VarLen::Values) {
            count
        } else {
            let stride = lay.elem_stride.max(1) as u64;
            self.builder.build_int_unsigned_div(count, i64t.const_int(stride, false), "alen").unwrap()
        };
        Ok(len)
    }

    /// Emit an inlined bounds check: if `idx64 >=u len` (unsigned, so negative
    /// indices wrap large and also trap) call `ai_bounds_fail` (noreturn) on a
    /// cold out-of-line path; otherwise fall through to the in-bounds block.
    fn emit_bounds_check(&self, fcx: &FnCtx<'ctx>, idx64: IntValue<'ctx>, len: IntValue<'ctx>) {
        let oob = self.builder.build_int_compare(IntPredicate::UGE, idx64, len, "oob").unwrap();
        let fail_bb = self.ctx.append_basic_block(fcx.func, "bounds.fail");
        let ok_bb = self.ctx.append_basic_block(fcx.func, "bounds.ok");
        self.builder.build_conditional_branch(oob, fail_bb, ok_bb).unwrap();
        self.builder.position_at_end(fail_bb);
        let f = self.module.get_function("ai_bounds_fail").unwrap();
        self.builder.build_call(f, &[fcx.thread.into(), idx64.into(), len.into()], "").unwrap();
        self.builder.build_unreachable().unwrap();
        self.builder.position_at_end(ok_bb);
    }

    /// Allocate a `String` varlen object and copy the literal's UTF-8 bytes in.
    fn gen_const_str(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        s: &str,
        repr: &Repr,
        span: crate::core::SpanId,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let i32t = self.ctx.i32_type();
        let i64t = self.ctx.i64_type();
        let lid = match repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("string repr".into())) };
        let bytes = s.as_bytes();
        let n = bytes.len() as u64;
        let site = self.alloc_site_id(&Self::current_fn_name(fcx), lid as u16, span);
        let alloc = self.module.get_function("ai_gc_alloc_varlen").unwrap();
        let obj = call_result(self.builder.build_call(
            alloc,
            &[
                fcx.thread.into(),
                i32t.const_int(lid as u64, false).into(),
                i64t.const_int(n, false).into(),
                i32t.const_int(site as u64, false).into(),
            ],
            "str",
        ).unwrap()).into_pointer_value();
        // Store bytes at HEADER + 8 (after the count word). Byte-by-byte; small
        // literals so this is fine, and it's GC-safe (no allocation between).
        let i8t = self.ctx.i8_type();
        for (i, b) in bytes.iter().enumerate() {
            let addr = self.obj_addr(obj, Self::HEADER + 8 + i as u64);
            self.builder.build_store(addr, i8t.const_int(*b as u64, false)).unwrap();
        }
        Ok(Some(obj.into()))
    }

    fn gen_array_new(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        layout: LayoutId,
        len: &CoreExpr,
        elem: &Repr,
        span: crate::core::SpanId,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let i32t = self.ctx.i32_type();
        let i64t = self.ctx.i64_type();
        let (stride, traced) = Self::elem_stride(elem);
        let n = self.gen_expr(fcx, len)?.unwrap().into_int_value();
        let n64 = if n.get_type().get_bit_width() < 64 {
            self.builder.build_int_s_extend(n, i64t, "n64").unwrap()
        } else { n };
        // varlen_len: for Values arrays it's element count; for Bytes arrays
        // it's the byte length (n * stride).
        let varlen_len = if traced { n64 } else {
            self.builder.build_int_mul(n64, i64t.const_int(stride, false), "blen").unwrap()
        };
        let site = self.alloc_site_id(&Self::current_fn_name(fcx), layout as u16, span);
        let alloc = self.module.get_function("ai_gc_alloc_varlen").unwrap();
        let obj = call_result(self.builder.build_call(
            alloc,
            &[
                fcx.thread.into(),
                i32t.const_int(layout as u64, false).into(),
                varlen_len.into(),
                i32t.const_int(site as u64, false).into(),
            ],
            "arr",
        ).unwrap()).into_pointer_value();
        Ok(Some(obj.into()))
    }

    fn gen_array_len(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        arr: &CoreExpr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let i64t = self.ctx.i64_type();
        let obj = self.gen_expr(fcx, arr)?.unwrap().into_pointer_value();
        // Count word at HEADER. For Bytes arrays it's byte-length; divide by the
        // element stride (from the array's layout via its element repr).
        let count_addr = self.obj_addr(obj, Self::HEADER);
        let count = self.builder.build_load(i64t, count_addr, "cnt").unwrap().into_int_value();
        let lid = match &arr.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("array_len on non-array".into())) };
        let lay = &self.prog.layouts[lid as usize];
        let logical = if matches!(lay.varlen, crate::core::VarLen::Values) {
            // Values arrays store the element count directly.
            count
        } else {
            // Bytes arrays store the byte length; divide by the element stride.
            let stride = lay.elem_stride.max(1) as u64;
            self.builder.build_int_unsigned_div(count, i64t.const_int(stride, false), "alen").unwrap()
        };
        Ok(Some(logical.into()))
    }

    /// `array_get(a, i)` yields `Option<T>`: `Some(a[i])` when in bounds, `None`
    /// when out of bounds (no abort — the absence of a value is signalled in the
    /// type). `result_repr` is the `Option<T>` value-enum repr assigned in
    /// lowering. The in-bounds load is unchecked (it only runs after the bounds
    /// compare), and when a caller immediately unwraps the result LLVM's SROA
    /// collapses the Option away, leaving just the compare + load.
    fn gen_array_get(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        array: &CoreExpr,
        index: &CoreExpr,
        elem: &Repr,
        result_repr: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        // Option<T> from the prelude `enum Option<T> { None, Some(T) }`:
        // declaration order fixes None = tag 0, Some = tag 1.
        const OPTION_NONE_TAG: u32 = 0;
        const OPTION_SOME_TAG: u32 = 1;
        let option_vid = match result_repr {
            Repr::Value(v) => *v,
            _ => return Err(CodegenError("array_get result is not an Option value enum".into())),
        };
        let obj = self.gen_expr(fcx, array)?.unwrap().into_pointer_value();
        let idx = self.gen_expr(fcx, index)?.unwrap().into_int_value();
        let idx64 = self.idx_to_i64(idx);
        let len = self.array_logical_len(obj, &array.repr)?;
        let sty = self.value_struct_ty(option_vid);

        let oob = self.builder.build_int_compare(IntPredicate::UGE, idx64, len, "oob").unwrap();
        let some_bb = self.ctx.append_basic_block(fcx.func, "aget.some");
        let none_bb = self.ctx.append_basic_block(fcx.func, "aget.none");
        let merge_bb = self.ctx.append_basic_block(fcx.func, "aget.merge");
        self.builder.build_conditional_branch(oob, none_bb, some_bb).unwrap();

        // In bounds: load the element and wrap it in Some.
        self.builder.position_at_end(some_bb);
        let (stride, _) = Self::elem_stride(elem);
        let addr = self.array_elem_addr(obj, idx64, stride);
        let lty = self.llvm_ty(elem).ok_or_else(|| CodegenError("array of unit".into()))?;
        let v = self.builder.build_load(lty, addr, "aget").unwrap();
        let some = self.build_value_variant(option_vid, OPTION_SOME_TAG, &[(v, elem.clone())]);
        let some_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(merge_bb).unwrap();

        // Out of bounds: None.
        self.builder.position_at_end(none_bb);
        let none = self.build_value_variant(option_vid, OPTION_NONE_TAG, &[]);
        let none_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(merge_bb).unwrap();

        // Merge the two Option values.
        self.builder.position_at_end(merge_bb);
        let phi = self.builder.build_phi(sty, "aget.opt").unwrap();
        phi.add_incoming(&[(&some, some_end), (&none, none_end)]);
        Ok(Some(phi.as_basic_value()))
    }

    /// `array_get_unchecked(a, i)` — a raw element load with no bounds check,
    /// yielding `T` directly (the unsafe escape hatch behind `array_get`).
    fn gen_array_get_unchecked(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        array: &CoreExpr,
        index: &CoreExpr,
        elem: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let obj = self.gen_expr(fcx, array)?.unwrap().into_pointer_value();
        let idx = self.gen_expr(fcx, index)?.unwrap().into_int_value();
        let idx64 = self.idx_to_i64(idx);
        let (stride, _) = Self::elem_stride(elem);
        let addr = self.array_elem_addr(obj, idx64, stride);
        let lty = self.llvm_ty(elem).ok_or_else(|| CodegenError("array of unit".into()))?;
        let v = self.builder.build_load(lty, addr, "aget.unchecked").unwrap();
        Ok(Some(v))
    }

    /// `a[i]` index operator: a bounds-checked element load that yields `T`
    /// directly and aborts (clear error) on out-of-bounds — Rust-like indexing.
    fn gen_array_get_checked(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        array: &CoreExpr,
        index: &CoreExpr,
        elem: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let obj = self.gen_expr(fcx, array)?.unwrap().into_pointer_value();
        let idx = self.gen_expr(fcx, index)?.unwrap().into_int_value();
        let idx64 = self.idx_to_i64(idx);
        let len = self.array_logical_len(obj, &array.repr)?;
        self.emit_bounds_check(fcx, idx64, len);
        let (stride, _) = Self::elem_stride(elem);
        let addr = self.array_elem_addr(obj, idx64, stride);
        let lty = self.llvm_ty(elem).ok_or_else(|| CodegenError("array of unit".into()))?;
        let v = self.builder.build_load(lty, addr, "aget.checked").unwrap();
        Ok(Some(v))
    }

    fn gen_array_set(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        array: &CoreExpr,
        index: &CoreExpr,
        value: &CoreExpr,
        elem: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let obj = self.gen_expr(fcx, array)?.unwrap().into_pointer_value();
        let idx = self.gen_expr(fcx, index)?.unwrap().into_int_value();
        let val = self.gen_expr(fcx, value)?.unwrap();
        let idx64 = self.idx_to_i64(idx);
        let len = self.array_logical_len(obj, &array.repr)?;
        self.emit_bounds_check(fcx, idx64, len);
        let (stride, _) = Self::elem_stride(elem);
        let addr = self.array_elem_addr(obj, idx64, stride);
        self.builder.build_store(addr, val).unwrap();
        // Generational write barrier: a long-lived (tenured) array may receive a
        // young pointer element. `emit_write_barrier` no-ops for scalar elements.
        self.emit_write_barrier(fcx, obj, val);
        Ok(Some(self.ctx.i64_type().const_zero().into()))
    }

    /// Build a closure: allocate the env, store the code pointer (the lifted
    /// function's address) at the start of the raw section, then the captures.
    fn gen_make_closure(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        code: FuncId,
        env: LayoutId,
        captures: &[CoreExpr],
        span: crate::core::SpanId,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let i32t = self.ctx.i32_type();
        // Evaluate captures first. The env allocation below is a SAFEPOINT, so any
        // GC-valued capture cached in a register before it goes stale on a
        // collection (relocated slot, stale register) — exactly the `gen_alloc`
        // hazard. ANF makes every GC capture an atomic local, so we reload them
        // from their relocated slots after the alloc (a side-effect-free load).
        let mut vals = Vec::with_capacity(captures.len());
        for c in captures {
            vals.push((self.gen_expr(fcx, c)?, c.repr.clone()));
        }
        let site = self.alloc_site_id(&Self::current_fn_name(fcx), env as u16, span);
        let alloc = self.module.get_function("ai_gc_alloc_fixed").unwrap();
        let obj = call_result(
            self.builder.build_call(
                alloc,
                &[
                    fcx.thread.into(),
                    i32t.const_int(env as u64, false).into(),
                    i32t.const_int(site as u64, false).into(),
                ],
                "clo",
            ).unwrap(),
        ).into_pointer_value();
        for (c, slot) in captures.iter().zip(vals.iter_mut()) {
            if self.repr_relocates(&c.repr) {
                slot.0 = self.gen_expr(fcx, c)?;
            }
        }

        let lay = &self.prog.layouts[env as usize];
        let ptr_fields = lay.ptr_fields as u64;
        let raw_base = Self::HEADER + ptr_fields * 8;
        // Store the code pointer at raw offset 0.
        let code_fn = self.funcs[&code];
        let code_ptr = code_fn.as_global_value().as_pointer_value();
        let addr = self.obj_addr(obj, raw_base);
        self.builder.build_store(addr, code_ptr).unwrap();

        // Store captures: pointer captures in pointer slots, scalars after the
        // code pointer (mirrors lower_lifted's offsets).
        let mut ptr_slot = 0u64;
        let mut raw_off = 8u64;
        for (v, repr) in &vals {
            match repr {
                Repr::Ref(_) => {
                    let off = Self::HEADER + ptr_slot * 8;
                    ptr_slot += 1;
                    let a = self.obj_addr(obj, off);
                    if let Some(v) = v { self.builder.build_store(a, *v).unwrap(); }
                }
                Repr::Scalar(s) => {
                    let sz = (s.bits().max(8) / 8) as u64;
                    raw_off = align_up64(raw_off, sz);
                    let a = self.obj_addr(obj, raw_base + raw_off);
                    raw_off += sz;
                    if let Some(v) = v { self.builder.build_store(a, *v).unwrap(); }
                }
                // Inline value-aggregate capture: store its bytes in the raw
                // section at an 8-aligned offset (mirrors lower_lifted).
                Repr::Value(vid) => {
                    let sz = self.prog.values[*vid as usize].size as u64;
                    raw_off = align_up64(raw_off, 8);
                    let a = self.obj_addr(obj, raw_base + raw_off);
                    raw_off += sz;
                    if let Some(v) = v { self.builder.build_store(a, *v).unwrap(); }
                }
                Repr::Unit => {}
            }
        }
        Ok(Some(obj.into()))
    }

    /// Call a closure value: load the code pointer from the env's raw section
    /// and indirect-call `(thread, env, args...)`.
    fn gen_call_closure(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        callee: &CoreExpr,
        args: &[CoreExpr],
        ret: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let env = self.gen_expr(fcx, callee)?.unwrap().into_pointer_value();
        // Recover the code pointer at runtime from the env header's type_id. The
        // static `callee.repr` is often the placeholder closure layout
        // (ptr_fields=0) — wrong for a closure that captured GC pointers — so we
        // must NOT compute the offset from it. See `ai_closure_code_ptr`.
        let codef = self.module.get_function("ai_closure_code_ptr").unwrap();
        let code_ptr = call_result(self.builder.build_call(
            codef, &[fcx.thread.into(), env.into()], "code",
        ).unwrap()).into_pointer_value();

        // Build the call signature: (thread, env, args...) -> ret.
        let mut arg_vals: Vec<inkwell::values::BasicMetadataValueEnum> =
            vec![fcx.thread.into(), env.into()];
        let mut arg_tys: Vec<BasicMetadataTypeEnum> = vec![ptr.into(), ptr.into()];
        for a in args {
            if let Some(v) = self.gen_expr(fcx, a)? {
                arg_vals.push(v.into());
                arg_tys.push(self.llvm_ty(&a.repr).unwrap().into());
            }
        }
        let fn_ty = match self.llvm_ty(ret) {
            Some(rt) => rt.fn_type(&arg_tys, false),
            None => self.ctx.void_type().fn_type(&arg_tys, false),
        };
        let cs = self.builder.build_indirect_call(fn_ty, code_ptr, &arg_vals, "cclo").unwrap();
        Ok(match cs.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => Some(v),
            inkwell::values::ValueKind::Instruction(_) => None,
        })
    }

    fn gen_field(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        base: &CoreExpr,
        loc: &FieldLoc,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        // A value-struct field is an extractvalue on the inline aggregate.
        if let FieldLoc::ValueField { index } = loc {
            let agg = self.gen_expr(fcx, base)?.unwrap().into_struct_value();
            let v = self.builder.build_extract_value(agg, *index, "vfld").unwrap();
            return Ok(Some(v));
        }
        let obj = self.gen_expr(fcx, base)?.unwrap().into_pointer_value();
        let (off, lty): (u64, BasicTypeEnum) = match loc {
            FieldLoc::Ptr { idx } => (
                Self::HEADER + (*idx as u64) * 8,
                self.ctx.ptr_type(AddressSpace::default()).as_basic_type_enum(),
            ),
            FieldLoc::Raw { offset, repr } => {
                let lid = match &base.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("field on non-ref".into())) };
                let lay = &self.prog.layouts[lid as usize];
                let raw_base = Self::HEADER + (lay.ptr_fields as u64) * 8;
                (raw_base + *offset as u64, self.scalar_ty(*repr))
            }
            FieldLoc::ValueAt { offset, value } => {
                // A flattened value aggregate: load the whole LLVM struct from
                // its inline byte offset in the raw region.
                let lid = match &base.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("field on non-ref".into())) };
                let lay = &self.prog.layouts[lid as usize];
                let raw_base = Self::HEADER + (lay.ptr_fields as u64) * 8;
                (raw_base + *offset as u64, self.value_struct_ty(*value).as_basic_type_enum())
            }
            FieldLoc::ValueField { .. } => unreachable!(),
        };
        let addr = self.obj_addr(obj, off);
        let v = self.builder.build_load(lty, addr, "fld").unwrap();
        Ok(Some(v))
    }

    fn gen_value_match(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        scrutinee: &CoreExpr,
        arms: &[CoreArm],
        repr: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let scrut_ty = self.llvm_ty(&scrutinee.repr).unwrap().into_struct_type();
        let vid = match &scrutinee.repr { Repr::Value(v) => *v, _ => unreachable!("value match on non-value") };
        let max_ptrs = crate::core::value_enum_max_ptrs(
            self.prog.values[vid as usize].variants.as_ref().unwrap(),
        ) as u32;
        let agg = self.gen_expr(fcx, scrutinee)?.unwrap().into_struct_value();
        // Spill the aggregate so we can address its payload bytes for binds.
        let scrut_slot = self.builder.build_alloca(scrut_ty, "vm.scrut").unwrap();
        self.builder.build_store(scrut_slot, agg).unwrap();
        // Compact layout: tag is field 0, payload bytes field 1. Pointers-first:
        // tag is field 1, the ptr-slot array field 0, the raw bytes field 2.
        let (tag_idx, ptr_arr, payload_ptr) = if max_ptrs == 0 {
            let pl = if scrut_ty.count_fields() > 1 {
                Some(self.builder.build_struct_gep(scrut_ty, scrut_slot, 1, "vm.pl").unwrap())
            } else { None };
            (0u32, None, pl)
        } else {
            let arr = self.builder.build_struct_gep(scrut_ty, scrut_slot, 0, "vm.ptrs").unwrap();
            let raw = self.builder.build_struct_gep(scrut_ty, scrut_slot, 2, "vm.raw").unwrap();
            (1u32, Some(arr), Some(raw))
        };
        let tag = self.builder.build_extract_value(agg, tag_idx, "vetag").unwrap().into_int_value();
        let func = fcx.func;
        let cont_bb = self.ctx.append_basic_block(func, "vm.cont");
        let result_slot = self.llvm_ty(repr).map(|t| self.builder.build_alloca(t, "vm.res").unwrap());

        let mut default_bb = None;
        let mut cases = Vec::new();
        let mut arm_blocks = Vec::new();
        let i32t = self.ctx.i32_type();
        for (i, arm) in arms.iter().enumerate() {
            let bb = self.ctx.append_basic_block(func, &format!("varm{}", i));
            arm_blocks.push((arm, bb));
            if arm.tag == u32::MAX { default_bb = Some(bb); }
            else { cases.push((i32t.const_int(arm.tag as u64, false), bb)); }
        }
        let unreachable_bb = self.ctx.append_basic_block(func, "vm.unreach");
        self.builder.build_switch(tag, default_bb.unwrap_or(unreachable_bb), &cases).unwrap();
        self.builder.position_at_end(unreachable_bb);
        self.builder.build_unreachable().unwrap();

        let ptr_arr_ty = self.ctx.ptr_type(AddressSpace::default()).array_type(max_ptrs);
        for (arm, bb) in arm_blocks {
            self.builder.position_at_end(bb);
            // Bind payload fields. Compact: all binds come from the byte region at
            // running offsets. Pointers-first: `Ref` binds come from the leading
            // ptr-slot array, the rest from the raw byte region — the exact
            // partition `gen_make_value_variant` used to store them.
            let mut raw_off = 0u64;
            let mut ptr_slot = 0u64;
            for &local in &arm.binds {
                let lrepr = fcx.local_reprs[local as usize].clone();
                let Some(lty) = self.llvm_ty(&lrepr) else { continue };
                let faddr = if max_ptrs > 0 && matches!(lrepr, Repr::Ref(_)) {
                    let elem = unsafe {
                        self.builder.build_in_bounds_gep(
                            ptr_arr_ty, ptr_arr.unwrap(),
                            &[i32t.const_zero(), i32t.const_int(ptr_slot, false)],
                            "vm.pslot",
                        ).unwrap()
                    };
                    ptr_slot += 1;
                    elem
                } else {
                    let (sz, _) = Self::repr_size_align(&lrepr);
                    raw_off = align_up64(raw_off, sz);
                    let a = self.payload_field_addr(payload_ptr.unwrap(), raw_off);
                    raw_off += sz;
                    a
                };
                let v = self.builder.build_load(lty, faddr, "vm.bind").unwrap();
                if let Some(slot) = fcx.slots[local as usize] {
                    self.builder.build_store(slot, v).unwrap();
                }
            }
            let v = self.gen_expr(fcx, &arm.body)?;
            if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                if let (Some(slot), Some(v)) = (result_slot, v) {
                    self.builder.build_store(slot, v).unwrap();
                }
                self.builder.build_unconditional_branch(cont_bb).unwrap();
            }
        }
        self.builder.position_at_end(cont_bb);
        match result_slot {
            Some(slot) => Ok(Some(self.builder.build_load(self.llvm_ty(repr).unwrap(), slot, "vm.v").unwrap())),
            None => Ok(None),
        }
    }

    fn gen_match(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        scrutinee: &CoreExpr,
        arms: &[CoreArm],
        repr: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let obj = self.gen_expr(fcx, scrutinee)?.unwrap().into_pointer_value();
        let lid = match &scrutinee.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("match on non-ref enum".into())) };
        let lay = &self.prog.layouts[lid as usize];
        let ptr_fields = lay.ptr_fields as u64;
        let i32t = self.ctx.i32_type();
        // Load the tag (raw u32 at start of the raw section).
        let tag_off = Self::HEADER + ptr_fields * 8;
        let tag_addr = self.obj_addr(obj, tag_off);
        let tag = self.builder.build_load(i32t, tag_addr, "tag").unwrap().into_int_value();

        let func = fcx.func;
        let cont_bb = self.ctx.append_basic_block(func, "match.cont");
        // Result slot (if non-unit).
        let result_slot = self.llvm_ty(repr).map(|t| {
            // alloca in entry-ish position; here is fine for v0.
            self.builder.build_alloca(t, "match.res").unwrap()
        });

        // Build arm blocks. A wildcard (tag u32::MAX) becomes the default.
        let mut default_bb = None;
        let mut cases = Vec::new();
        let mut arm_blocks = Vec::new();
        for (i, arm) in arms.iter().enumerate() {
            let bb = self.ctx.append_basic_block(func, &format!("arm{}", i));
            arm_blocks.push((arm, bb));
            if arm.tag == u32::MAX {
                default_bb = Some(bb);
            } else {
                cases.push((i32t.const_int(arm.tag as u64, false), bb));
            }
        }
        let unreachable_bb = self.ctx.append_basic_block(func, "match.unreach");
        let default = default_bb.unwrap_or(unreachable_bb);
        self.builder.build_switch(tag, default, &cases).unwrap();

        // Fill the unreachable default (exhaustive matches without wildcard).
        self.builder.position_at_end(unreachable_bb);
        self.builder.build_unreachable().unwrap();

        for (arm, bb) in arm_blocks {
            self.builder.position_at_end(bb);
            // Bind payload fields into their (already-allocated) local slots.
            // Payload layout: pointer payloads in pointer slots [0..], raw
            // payloads after the tag word.
            let mut ptr_slot = 0u64;
            let mut raw_cursor = tag_off + 8;
            for &local in &arm.binds {
                let lrepr = fcx.local_reprs[local as usize].clone();
                match &lrepr {
                    Repr::Ref(_) => {
                        let off = Self::HEADER + ptr_slot * 8;
                        ptr_slot += 1;
                        let addr = self.obj_addr(obj, off);
                        let v = self.builder.build_load(self.ctx.ptr_type(AddressSpace::default()), addr, "pl").unwrap();
                        if let Some(slot) = fcx.slots[local as usize] {
                            self.builder.build_store(slot, v).unwrap();
                        }
                    }
                    Repr::Scalar(s) => {
                        let sz = (s.bits().max(8) / 8) as u64;
                        raw_cursor = align_up64(raw_cursor, sz);
                        let addr = self.obj_addr(obj, raw_cursor);
                        raw_cursor += sz;
                        let v = self.builder.build_load(self.scalar_ty(*s), addr, "pl").unwrap();
                        if let Some(slot) = fcx.slots[local as usize] {
                            self.builder.build_store(slot, v).unwrap();
                        }
                    }
                    _ => {}
                }
            }
            let v = self.gen_expr(fcx, &arm.body)?;
            if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                if let (Some(slot), Some(v)) = (result_slot, v) {
                    self.builder.build_store(slot, v).unwrap();
                }
                self.builder.build_unconditional_branch(cont_bb).unwrap();
            }
        }

        self.builder.position_at_end(cont_bb);
        match result_slot {
            Some(slot) => {
                let ty = self.llvm_ty(repr).unwrap();
                Ok(Some(self.builder.build_load(ty, slot, "match.v").unwrap()))
            }
            None => Ok(None),
        }
    }

    /// Load payload field `field` from a reference enum object, replicating the
    /// physical slot layout the tag-switch binding path uses: pointer payloads
    /// occupy the leading pointer slots (`HEADER + k*8`), raw payloads are packed
    /// (aligned) after the tag word at `tag_off + 8`. `payload_reprs` is the full
    /// ordered payload list so we can find `field`'s slot.
    fn load_enum_payload(
        &self,
        obj: PointerValue<'ctx>,
        tag_off: u64,
        field: usize,
        repr: &Repr,
        payload_reprs: &[Repr],
    ) -> BasicValueEnum<'ctx> {
        let mut ptr_slot = 0u64;
        let mut raw_cursor = tag_off + 8;
        for (i, r) in payload_reprs.iter().enumerate() {
            match r {
                Repr::Ref(_) => {
                    let off = Self::HEADER + ptr_slot * 8;
                    ptr_slot += 1;
                    if i == field {
                        let addr = self.obj_addr(obj, off);
                        return self.builder.build_load(self.ctx.ptr_type(AddressSpace::default()), addr, "pl").unwrap();
                    }
                }
                Repr::Scalar(s) => {
                    let sz = (s.bits().max(8) / 8) as u64;
                    raw_cursor = align_up64(raw_cursor, sz);
                    let off = raw_cursor;
                    raw_cursor += sz;
                    if i == field {
                        let addr = self.obj_addr(obj, off);
                        return self.builder.build_load(self.scalar_ty(*s), addr, "pl").unwrap();
                    }
                }
                _ => {
                    // Value payloads aren't bound through this path in v0.
                    if i == field {
                        // Fallback: load as a pointer (won't happen for scalar/ref).
                        let addr = self.obj_addr(obj, Self::HEADER + ptr_slot * 8);
                        return self.builder.build_load(self.ctx.ptr_type(AddressSpace::default()), addr, "pl").unwrap();
                    }
                }
            }
        }
        // Unreachable if `field` is valid.
        let _ = repr;
        self.ctx.i64_type().const_zero().into()
    }

    fn gen_bin(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        op: BinOp,
        l: &CoreExpr,
        r: &CoreExpr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        // Short-circuit && and ||.
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.gen_logical(fcx, op, l, r);
        }
        let scalar = match &l.repr {
            Repr::Scalar(s) => *s,
            _ => return Err(CodegenError("binary op on non-scalar".into())),
        };
        let lv = self.gen_expr(fcx, l)?.unwrap();
        let rv = self.gen_expr(fcx, r)?.unwrap();
        let b = &self.builder;

        if scalar.is_float() {
            let lf = lv.into_float_value();
            let rf = rv.into_float_value();
            let v: BasicValueEnum = match op {
                BinOp::Add => b.build_float_add(lf, rf, "fadd").unwrap().into(),
                BinOp::Sub => b.build_float_sub(lf, rf, "fsub").unwrap().into(),
                BinOp::Mul => b.build_float_mul(lf, rf, "fmul").unwrap().into(),
                BinOp::Div => b.build_float_div(lf, rf, "fdiv").unwrap().into(),
                BinOp::Rem => b.build_float_rem(lf, rf, "frem").unwrap().into(),
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    let pred = float_pred(op);
                    b.build_float_compare(pred, lf, rf, "fcmp").unwrap().into()
                }
                _ => return Err(CodegenError("invalid float op".into())),
            };
            return Ok(Some(v));
        }

        let li = lv.into_int_value();
        let ri = rv.into_int_value();
        let signed = scalar.is_signed();
        // Integer overflow WRAPS (two's complement), like Java/Go/OCaml — see
        // docs/overflow.md. The plain `build_int_*` builders emit LLVM `add`/
        // `sub`/`mul` with NO `nsw`/`nuw` flags, which is defined wrapping. Do
        // NOT switch to the `*_nsw`/`*_nuw` variants: those make overflow UB and
        // would silently break this language guarantee.
        let v: BasicValueEnum = match op {
            BinOp::Add => b.build_int_add(li, ri, "add").unwrap().into(),
            BinOp::Sub => b.build_int_sub(li, ri, "sub").unwrap().into(),
            BinOp::Mul => b.build_int_mul(li, ri, "mul").unwrap().into(),
            BinOp::Div => if signed {
                b.build_int_signed_div(li, ri, "sdiv").unwrap().into()
            } else {
                b.build_int_unsigned_div(li, ri, "udiv").unwrap().into()
            },
            BinOp::Rem => if signed {
                b.build_int_signed_rem(li, ri, "srem").unwrap().into()
            } else {
                b.build_int_unsigned_rem(li, ri, "urem").unwrap().into()
            },
            BinOp::BitAnd => b.build_and(li, ri, "and").unwrap().into(),
            BinOp::BitOr => b.build_or(li, ri, "or").unwrap().into(),
            BinOp::BitXor => b.build_xor(li, ri, "xor").unwrap().into(),
            BinOp::Shl => b.build_left_shift(li, ri, "shl").unwrap().into(),
            BinOp::Shr => b.build_right_shift(li, ri, signed, "shr").unwrap().into(),
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                let pred = int_pred(op, signed);
                b.build_int_compare(pred, li, ri, "icmp").unwrap().into()
            }
            BinOp::And | BinOp::Or => unreachable!(),
        };
        Ok(Some(v))
    }

    fn gen_logical(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        op: BinOp,
        l: &CoreExpr,
        r: &CoreExpr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        // a && b  =>  if a { b } else { false }
        // a || b  =>  if a { true } else { b }
        let lv = self.gen_expr(fcx, l)?.unwrap().into_int_value();
        let func = fcx.func;
        let rhs_bb = self.ctx.append_basic_block(func, "logic.rhs");
        let cont_bb = self.ctx.append_basic_block(func, "logic.cont");
        let entry_bb = self.builder.get_insert_block().unwrap();

        match op {
            BinOp::And => self.builder.build_conditional_branch(lv, rhs_bb, cont_bb).unwrap(),
            BinOp::Or => self.builder.build_conditional_branch(lv, cont_bb, rhs_bb).unwrap(),
            _ => unreachable!(),
        };

        self.builder.position_at_end(rhs_bb);
        let rv = self.gen_expr(fcx, r)?.unwrap().into_int_value();
        let rhs_end = self.builder.get_insert_block().unwrap();
        self.builder.build_unconditional_branch(cont_bb).unwrap();

        self.builder.position_at_end(cont_bb);
        let phi = self.builder.build_phi(self.ctx.bool_type(), "logic").unwrap();
        let short = self.ctx.bool_type().const_int((op == BinOp::Or) as u64, false);
        phi.add_incoming(&[(&short, entry_bb), (&rv, rhs_end)]);
        Ok(Some(phi.as_basic_value()))
    }

    fn gen_un(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        op: UnOp,
        inner: &CoreExpr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let v = self.gen_expr(fcx, inner)?.unwrap();
        let res: BasicValueEnum = match op {
            UnOp::Neg => match &inner.repr {
                Repr::Scalar(s) if s.is_float() => {
                    self.builder.build_float_neg(v.into_float_value(), "fneg").unwrap().into()
                }
                _ => self.builder.build_int_neg(v.into_int_value(), "neg").unwrap().into(),
            },
            UnOp::Not => {
                // bool not, or bitwise not on ints.
                self.builder.build_not(v.into_int_value(), "not").unwrap().into()
            }
        };
        Ok(Some(res))
    }

    fn gen_float_intrinsic(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        intr: crate::core::FloatIntrinsic,
        inner: &CoreExpr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        use crate::core::FloatIntrinsic::*;
        let v = self.gen_expr(fcx, inner)?.unwrap().into_float_value();
        let bits = match &inner.repr { Repr::Scalar(s) => s.bits(), _ => 64 };
        let suffix = if bits == 32 { "f32" } else { "f64" };
        let fty = if bits == 32 { self.ctx.f32_type() } else { self.ctx.f64_type() };
        let intr_name = match intr {
            Sqrt => "llvm.sqrt",
            Abs => "llvm.fabs",
            Floor => "llvm.floor",
            Ceil => "llvm.ceil",
        };
        let full = format!("{}.{}", intr_name, suffix);
        let f = self.module.get_function(&full).unwrap_or_else(|| {
            let fnty = fty.fn_type(&[fty.into()], false);
            self.module.add_function(&full, fnty, None)
        });
        let r = call_result(self.builder.build_call(f, &[v.into()], "fi").unwrap());
        Ok(Some(r))
    }

    fn gen_cast(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        value: &CoreExpr,
        from: &Repr,
        to: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let v = self.gen_expr(fcx, value)?.unwrap();
        let (fs, ts) = match (from, to) {
            (Repr::Scalar(a), Repr::Scalar(b)) => (*a, *b),
            _ => return Err(CodegenError("non-scalar cast in v0 slice".into())),
        };
        let b = &self.builder;
        let res: BasicValueEnum = match (fs.is_float(), ts.is_float()) {
            (false, false) => {
                let iv = v.into_int_value();
                let dst = self.scalar_ty(ts).into_int_type();
                if ts.bits() < fs.bits() {
                    b.build_int_truncate(iv, dst, "trunc").unwrap().into()
                } else if ts.bits() > fs.bits() {
                    if fs.is_signed() {
                        b.build_int_s_extend(iv, dst, "sext").unwrap().into()
                    } else {
                        b.build_int_z_extend(iv, dst, "zext").unwrap().into()
                    }
                } else {
                    iv.into()
                }
            }
            (true, true) => {
                let fv = v.into_float_value();
                let dst = self.scalar_ty(ts).into_float_type();
                if ts.bits() < fs.bits() {
                    b.build_float_trunc(fv, dst, "fptrunc").unwrap().into()
                } else if ts.bits() > fs.bits() {
                    b.build_float_ext(fv, dst, "fpext").unwrap().into()
                } else {
                    fv.into()
                }
            }
            (false, true) => {
                let iv = v.into_int_value();
                let dst = self.scalar_ty(ts).into_float_type();
                if fs.is_signed() {
                    b.build_signed_int_to_float(iv, dst, "sitofp").unwrap().into()
                } else {
                    b.build_unsigned_int_to_float(iv, dst, "uitofp").unwrap().into()
                }
            }
            (true, false) => {
                let fv = v.into_float_value();
                let dst = self.scalar_ty(ts).into_int_type();
                if ts.is_signed() {
                    b.build_float_to_signed_int(fv, dst, "fptosi").unwrap().into()
                } else {
                    b.build_float_to_unsigned_int(fv, dst, "fptoui").unwrap().into()
                }
            }
        };
        Ok(Some(res))
    }

    fn gen_call(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        fid: FuncId,
        args: &[CoreExpr],
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let callee = self.funcs[&fid];
        // Foreign `extern "C"` callees take no leading Thread* — they're plain C.
        let is_extern = self.prog.funcs[fid as usize].is_extern;
        let mut cargs: Vec<inkwell::values::BasicMetadataValueEnum> =
            if is_extern { vec![] } else { vec![fcx.thread.into()] };
        let param_reprs = self.prog.funcs[fid as usize].params.clone();
        let extern_by_ref = self.prog.funcs[fid as usize].extern_by_ref.clone();
        // Any FFI copy-out buffers queued while evaluating THIS call's arguments
        // are drained after the call; snapshot the queue depth to isolate them.
        let copy_out_base = fcx.pending_copy_outs.len();

        // GC-SAFETY: a `Ref` argument backed by a `Local` must be loaded from its
        // root slot AFTER every sibling argument that might allocate — otherwise a
        // collection triggered while evaluating a later argument (e.g. the closure
        // in `atom.swap(|x| ..)`) relocates the object, leaving the
        // already-materialized pointer dangling. So we DEFER such loads to a
        // second pass, right before the call. Their slots are always current (the
        // collector updates them), so reloading is always safe.
        // `cargs` is built positionally; `deferred` holds (cargs-index, local-id).
        let mut deferred: Vec<(usize, u32)> = Vec::new();
        for (i, a) in args.iter().enumerate() {
            // For an extern callee, a value-struct argument crosses either BY
            // VALUE (coerced into register-shaped types per the C ABI) or BY
            // POINTER (a `mut` out-param, or > 16 bytes passed indirectly). The
            // struct lives on the native stack — never a GC heap object — so its
            // address is stable for the call and needs no pinning. See docs/ffi.md.
            if is_extern {
                if let Some(Repr::Value(vid)) = param_reprs.get(i) {
                    let by_ref = extern_by_ref.get(i).copied().unwrap_or(false);
                    let coerce = if by_ref { None } else { self.abi_coerce(*vid, false) };
                    match coerce {
                        // By value (≤ 16B): materialize the struct, reinterpret it
                        // into the coercion type through a stack slot (the upper
                        // bytes of a partly-filled register slot are don't-care, as
                        // in clang's lowering), and pass the coerced value.
                        Some(coerce_ty) => {
                            if let Some(v) = self.gen_expr(fcx, a)? {
                                let slot =
                                    self.builder.build_alloca(coerce_ty, "ffi.coerce").unwrap();
                                self.builder.build_store(slot, v).unwrap();
                                let loaded =
                                    self.builder.build_load(coerce_ty, slot, "ffi.arg").unwrap();
                                cargs.push(loaded.into());
                            }
                            continue;
                        }
                        // By pointer: a value-struct LOCAL passes its own alloca
                        // (so a `mut` out-param's writes land back in the caller's
                        // variable); a temporary spills to a fresh alloca.
                        None => {
                            if let CoreExprKind::Local(id) = a.kind.as_ref() {
                                if let Some(slot) = fcx.slots[*id as usize] {
                                    cargs.push(slot.into());
                                    continue;
                                }
                            }
                            if let Some(v) = self.gen_expr(fcx, a)? {
                                let slot =
                                    self.builder.build_alloca(v.get_type(), "ffi.arg").unwrap();
                                self.builder.build_store(slot, v).unwrap();
                                cargs.push(slot.into());
                            }
                            continue;
                        }
                    }
                }
            }
            // Defer a `Ref` argument that is a plain `Local` (reload from its slot
            // after all args are evaluated — see GC-SAFETY note above).
            if let CoreExprKind::Local(id) = a.kind.as_ref() {
                if matches!(a.repr, Repr::Ref(_)) && fcx.slots[*id as usize].is_some() {
                    deferred.push((cargs.len(), *id));
                    cargs.push(self.ctx.ptr_type(AddressSpace::default()).const_null().into());
                    continue;
                }
            }
            if let Some(v) = self.gen_expr(fcx, a)? {
                cargs.push(v.into());
            }
        }
        // Second pass: reload deferred `Ref`-`Local` args from their (now-current,
        // post-any-allocation) root slots.
        for (ci, id) in deferred {
            let slot = fcx.slots[id as usize].unwrap();
            let lty = self.llvm_ty(&fcx.local_reprs[id as usize]).unwrap();
            let v = self.builder.build_load(lty, slot, "arg.reload").unwrap();
            cargs[ci] = v.into();
        }
        // Wrap a foreign call in the managed↔native transition: publish our
        // frame chain so a concurrent GC can find this thread's roots while we
        // are in native code, then clear it on return. Safe by default — the
        // call's correctness does not depend on the C function not allocating.
        if is_extern {
            let enter = self.module.get_function("ai_ffi_enter").unwrap();
            self.builder.build_call(enter, &[fcx.thread.into()], "").unwrap();
        }
        let cs = self.builder.build_call(callee, &cargs, "call").unwrap();
        if is_extern {
            let leave = self.module.get_function("ai_ffi_leave").unwrap();
            self.builder.build_call(leave, &[fcx.thread.into()], "").unwrap();
        }
        // Copy-out: write each `mut` array's stack buffer back into its heap
        // object now that C has filled it (e.g. read(fd, buf, n)).
        if fcx.pending_copy_outs.len() > copy_out_base {
            let cout = self.module.get_function("ai_buf_copy_out").unwrap();
            let pending: Vec<_> = fcx.pending_copy_outs.drain(copy_out_base..).collect();
            for (obj, buf, byte_len) in pending {
                self.builder.build_call(
                    cout,
                    &[fcx.thread.into(), obj.into(), buf.into(), byte_len.into()],
                    "",
                ).unwrap();
            }
        }
        let ret_val = match cs.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => Some(v),
            inkwell::values::ValueKind::Instruction(_) => None,
        };
        // Decode a by-value value-struct return: the call yielded the ABI
        // coercion type (an `iN` / `[N x i64]` / `[N x float]`); reinterpret it
        // back into the value struct through a stack slot.
        if is_extern {
            if let Repr::Value(vid) = self.prog.funcs[fid as usize].ret {
                if let (Some(coerce_ty), Some(coerced)) =
                    (self.abi_coerce(vid, true), ret_val)
                {
                    let struct_ty = self.value_struct_ty(vid);
                    let slot = self.builder.build_alloca(coerce_ty, "ffi.ret").unwrap();
                    self.builder.build_store(slot, coerced).unwrap();
                    let sval =
                        self.builder.build_load(struct_ty, slot, "ffi.retval").unwrap();
                    return Ok(Some(sval));
                }
            }
        }
        Ok(ret_val)
    }

    /// Emit a GC safepoint poll: `if (thread.state != 0) ai_gc_pollcheck_slow(thread)`.
    /// The load is volatile so the optimizer can't hoist it out of the loop.
    fn emit_safepoint_poll(&self, fcx: &FnCtx<'ctx>) {
        let i8t = self.ctx.i8_type();
        let state_ptr = self.thread_field_ptr(fcx.func, crate::runtime::thread_offsets::STATE);
        let load = self.builder.build_load(i8t, state_ptr, "gcstate").unwrap();
        load.as_instruction_value().unwrap().set_volatile(true).ok();
        let is_set = self.builder.build_int_compare(
            IntPredicate::NE, load.into_int_value(), i8t.const_zero(), "gcpoll",
        ).unwrap();
        let slow_bb = self.ctx.append_basic_block(fcx.func, "gc.slow");
        let cont_bb = self.ctx.append_basic_block(fcx.func, "gc.cont");
        self.builder.build_conditional_branch(is_set, slow_bb, cont_bb).unwrap();
        self.builder.position_at_end(slow_bb);
        let poll = self.module.get_function("ai_gc_pollcheck_slow").unwrap();
        self.builder.build_call(poll, &[fcx.thread.into()], "").unwrap();
        self.builder.build_unconditional_branch(cont_bb).unwrap();
        self.builder.position_at_end(cont_bb);
    }

    fn gen_loop(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        body: &CoreBlock,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let func = fcx.func;
        let header_bb = self.ctx.append_basic_block(func, "loop.header");
        let cont_bb = self.ctx.append_basic_block(func, "loop.cont");
        self.builder.build_unconditional_branch(header_bb).unwrap();

        // v0 loops yield unit, so no break-value slot is needed.
        fcx.loops.push((cont_bb, None));
        fcx.loop_headers.push(header_bb);

        self.builder.position_at_end(header_bb);
        // GC safepoint poll at the loop header: load thread.state (volatile);
        // if non-zero, trap into ai_gc_pollcheck_slow so the mutator parks.
        self.emit_safepoint_poll(fcx);
        self.gen_block(fcx, body)?;
        // Back-edge: if the body fell through, loop again.
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            self.builder.build_unconditional_branch(header_bb).unwrap();
        }

        fcx.loops.pop();
        fcx.loop_headers.pop();
        self.builder.position_at_end(cont_bb);
        Ok(None)
    }

    fn gen_if(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        cond: &CoreExpr,
        then_b: &CoreBlock,
        else_b: &CoreBlock,
        repr: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let cv = self.gen_expr(fcx, cond)?.unwrap().into_int_value();
        let func = fcx.func;
        let then_bb = self.ctx.append_basic_block(func, "then");
        let else_bb = self.ctx.append_basic_block(func, "else");
        let cont_bb = self.ctx.append_basic_block(func, "ifcont");
        self.builder.build_conditional_branch(cv, then_bb, else_bb).unwrap();

        self.builder.position_at_end(then_bb);
        let tv = self.gen_block(fcx, then_b)?;
        let then_end = self.builder.get_insert_block().unwrap();
        let then_open = then_end.get_terminator().is_none();
        if then_open {
            self.builder.build_unconditional_branch(cont_bb).unwrap();
        }

        self.builder.position_at_end(else_bb);
        let ev = self.gen_block(fcx, else_b)?;
        let else_end = self.builder.get_insert_block().unwrap();
        let else_open = else_end.get_terminator().is_none();
        if else_open {
            self.builder.build_unconditional_branch(cont_bb).unwrap();
        }

        self.builder.position_at_end(cont_bb);
        match (self.llvm_ty(repr), then_open || else_open) {
            (Some(ty), true) => {
                let phi = self.builder.build_phi(ty, "ifval").unwrap();
                if then_open {
                    if let Some(tv) = tv {
                        phi.add_incoming(&[(&tv, then_end)]);
                    }
                }
                if else_open {
                    if let Some(ev) = ev {
                        phi.add_incoming(&[(&ev, else_end)]);
                    }
                }
                Ok(Some(phi.as_basic_value()))
            }
            _ => Ok(None),
        }
    }
}

struct FnCtx<'ctx> {
    func: FunctionValue<'ctx>,
    slots: Vec<Option<PointerValue<'ctx>>>,
    local_reprs: Vec<Repr>,
    thread: PointerValue<'ctx>,
    /// Per active loop: (continuation block, optional break-value slot).
    loops: Vec<(BasicBlock<'ctx>, Option<PointerValue<'ctx>>)>,
    /// Per active loop: the header block (`continue` target).
    loop_headers: Vec<BasicBlock<'ctx>>,
    /// `(frame ptr, frame type, &thread.top_frame)` for the GC frame epilogue.
    /// `None` when this function allocated no GC frame (no Ref locals).
    unlink: Option<(PointerValue<'ctx>, inkwell::types::StructType<'ctx>, PointerValue<'ctx>)>,
    /// FFI copy-out buffers awaiting write-back: `(heap object, stack buffer,
    /// byte length)`. Filled by an `AsCBytes { copy_out: true }` argument and
    /// drained by `gen_call` AFTER the extern call returns. See `docs/ffi.md`.
    pending_copy_outs: Vec<(PointerValue<'ctx>, PointerValue<'ctx>, IntValue<'ctx>)>,
    /// This function's DWARF `DISubprogram` scope (debugger P2) — `Some` only when
    /// emitting debug info. Debug locations for the function's nodes use it as
    /// their scope (one function = one source, so scope/file stay consistent).
    subprogram: Option<DISubprogram<'ctx>>,
}

fn align_up64(n: u64, a: u64) -> u64 {
    if a == 0 { n } else { (n + a - 1) & !(a - 1) }
}

/// Extract a call's basic-value result (this inkwell fork returns a `ValueKind`).
fn call_result<'ctx>(cs: inkwell::values::CallSiteValue<'ctx>) -> BasicValueEnum<'ctx> {
    match cs.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        inkwell::values::ValueKind::Instruction(_) => panic!("call produced no value"),
    }
}

fn int_pred(op: BinOp, signed: bool) -> IntPredicate {
    use BinOp::*;
    match op {
        Eq => IntPredicate::EQ,
        Ne => IntPredicate::NE,
        Lt => if signed { IntPredicate::SLT } else { IntPredicate::ULT },
        Le => if signed { IntPredicate::SLE } else { IntPredicate::ULE },
        Gt => if signed { IntPredicate::SGT } else { IntPredicate::UGT },
        Ge => if signed { IntPredicate::SGE } else { IntPredicate::UGE },
        _ => unreachable!(),
    }
}

fn float_pred(op: BinOp) -> FloatPredicate {
    use BinOp::*;
    match op {
        Eq => FloatPredicate::OEQ,
        Ne => FloatPredicate::ONE,
        Lt => FloatPredicate::OLT,
        Le => FloatPredicate::OLE,
        Gt => FloatPredicate::OGT,
        Ge => FloatPredicate::OGE,
        _ => unreachable!(),
    }
}

fn core_disc(e: &CoreExprKind) -> &'static str {
    match e {
        CoreExprKind::CallClosure { .. } => "call-closure",
        CoreExprKind::MakeClosure { .. } => "make-closure",
        CoreExprKind::New { .. } => "new",
        CoreExprKind::MakeValue { .. } => "make-value",
        CoreExprKind::MakeVariant { .. } => "make-variant",
        CoreExprKind::Field { .. } => "field",
        CoreExprKind::Match { .. } => "match",
        CoreExprKind::Loop(_) => "loop",
        CoreExprKind::Break(_) => "break",
        CoreExprKind::Continue => "continue",
        CoreExprKind::Assign { .. } => "assign",
        _ => "expr",
    }
}

/// Run the standard LLVM optimization pipeline (`default<O2>`) over the module
/// in place: mem2reg, inlining, instcombine, GVN, loop opts, etc. This is where
/// monomorphized gc-rust code gets its speed.
fn optimize_module(module: &Module) {
    use inkwell::OptimizationLevel;
    use inkwell::passes::PassBuilderOptions;
    use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};

    Target::initialize_native(&InitializationConfig::default()).ok();
    let triple = TargetMachine::get_default_triple();
    let Ok(target) = Target::from_triple(&triple) else { return };
    let Some(machine) = target.create_target_machine(
        &triple,
        &TargetMachine::get_host_cpu_name().to_string(),
        &TargetMachine::get_host_cpu_features().to_string(),
        OptimizationLevel::Aggressive,
        RelocMode::Default,
        CodeModel::Default,
    ) else { return };
    let opts = PassBuilderOptions::create();
    // If the pipeline fails to parse/run, leave the module unoptimized (still
    // correct, just slower) rather than aborting.
    let _ = module.run_passes("default<O2>", &machine, opts);
}

/// Convert the program's core [`Layout`]s into `gc::TypeInfo`s, one per
/// `LayoutId` (index = `type_id`). This is the bridge to the collector: pointer
/// fields first (traced), then raw bytes, then any varlen tail.
pub fn layouts_to_type_infos(prog: &CoreProgram) -> Vec<crate::gc::TypeInfo> {
    use crate::gc::{Full, ObjHeader, TypeInfo};
    prog.layouts
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let mut ti = TypeInfo::for_header(Full::SIZE)
                .with_type_id(i as u16)
                .with_fields(l.ptr_fields)
                .with_raw_bytes(l.raw_bytes);
            ti = match l.varlen {
                crate::core::VarLen::None => ti,
                crate::core::VarLen::Values => ti.with_varlen_values(l.ptr_fields),
                crate::core::VarLen::Bytes => ti.with_varlen_bytes(l.ptr_fields),
            };
            // Interior pointers (GC refs inside flattened value fields). Leaked
            // to `&'static` — the type table lives for the whole program, so this
            // is a bounded one-time leak, not a per-object cost. Empty stays `&[]`.
            if !l.interior_ptrs.is_empty() {
                let leaked: &'static [u16] = Box::leak(l.interior_ptrs.clone().into_boxed_slice());
                ti = ti.with_interior_ptrs(leaked);
            }
            ti
        })
        .collect()
}

/// Size and alignment (bytes) of a value-aggregate field's repr — mirrors
/// `layout::value_size_align`'s per-field rule so interior offsets match the LLVM
/// struct layout codegen actually uses.
fn value_field_size_align(values: &[crate::core::ValueLayout], f: &crate::core::Repr) -> (u32, u32) {
    use crate::core::Repr;
    match f {
        Repr::Unit => (0, 1),
        Repr::Scalar(s) => {
            let b = (s.bits().max(8) / 8).max(1);
            (b, b)
        }
        Repr::Ref(_) => (8, 8),
        Repr::Value(sub) => (values[*sub as usize].size, values[*sub as usize].align),
    }
}

/// Whether value type `vid` (transitively) holds a GC reference.
fn value_has_ref(values: &[crate::core::ValueLayout], vid: u32) -> bool {
    use crate::core::Repr;
    let vl = &values[vid as usize];
    let mut all: Vec<&Repr> = Vec::new();
    match &vl.variants {
        Some(variants) => variants.iter().for_each(|v| all.extend(v.fields.iter())),
        None => all.extend(vl.fields.iter()),
    }
    all.iter().any(|f| match f {
        Repr::Ref(_) => true,
        Repr::Value(s) => value_has_ref(values, *s),
        _ => false,
    })
}

/// Byte offsets (value-relative, shifted by `base`) of GC refs inside a flattened
/// value aggregate `vid`, recursing into nested value structs. Mirrors
/// `layout::value_ref_offsets`. A value **enum** reserves `max_ptrs` leading slots
/// shared across variants (pointers-first layout), so its embedded refs are at
/// fixed offsets `base, base+8, …` regardless of the active variant.
fn value_interior_offsets(values: &[crate::core::ValueLayout], vid: u32, base: u16, out: &mut Vec<u16>) {
    use crate::core::Repr;
    let vl = &values[vid as usize];
    if let Some(variants) = &vl.variants {
        let max_ptrs = crate::core::value_enum_max_ptrs(variants);
        for k in 0..max_ptrs {
            out.push(base + k * 8);
        }
        return;
    }
    let mut off = 0u32;
    for f in &vl.fields {
        let (sz, align) = value_field_size_align(values, f);
        off = (off + align - 1) & !(align - 1);
        match f {
            Repr::Ref(_) => out.push(base + off as u16),
            Repr::Value(sub) => value_interior_offsets(values, *sub, base + off as u16, out),
            _ => {}
        }
        off += sz;
    }
}

/// Collect the program's per-layout reflection metadata into a runtime
/// [`gc::TypeMeta`] table, one entry per `LayoutId` (index = `type_id`). The
/// metadata is built during layout lowering (see `src/layout.rs`) and travels
/// inside each `Layout`; here we just clone it out and stamp the `type_id` from
/// its table position. Parallel to [`layouts_to_type_infos`].
pub fn layouts_to_type_meta(prog: &CoreProgram) -> Vec<crate::gc::TypeMeta> {
    prog.layouts
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let mut m = l.meta.clone();
            m.type_id = i as u16;
            m
        })
        .collect()
}

/// Build the runtime [`gc::ValueMeta`] table — reflection metadata for inline
/// `#[value]` aggregates, indexed by `ValueId`. Value-struct fields get
/// value-relative byte offsets (mirroring `value_size_align`'s placement) so the
/// heap-dump renderer can recurse into flattened value fields. Value enums are
/// emitted as `Opaque` for now (their tag/union offsets aren't decoded yet).
pub fn layouts_to_value_meta(prog: &CoreProgram) -> Vec<crate::gc::ValueMeta> {
    use crate::core::Repr;
    use crate::gc::{FieldMeta, FieldTy, TypeKind, ValueMeta};
    let align_up = |off: u32, a: u32| -> u32 { (off + a - 1) & !(a - 1) };
    prog.values
        .iter()
        .enumerate()
        .map(|(vid, vl)| {
            let kind = if vl.variants.is_some() {
                TypeKind::Opaque
            } else {
                let mut off = 0u32;
                let mut fields = Vec::new();
                for (i, r) in vl.fields.iter().enumerate() {
                    let name = vl.field_names.get(i).cloned().unwrap_or_else(|| i.to_string());
                    let (sz, align, fty) = match r {
                        Repr::Unit => continue,
                        Repr::Scalar(s) => {
                            let b = (s.bits().max(8) / 8).max(1);
                            (b, b, FieldTy::Scalar(crate::layout::scalar_kind(*s)))
                        }
                        Repr::Ref(lid) => (8, 8, FieldTy::Ref(*lid as u16)),
                        Repr::Value(sub) => {
                            let v = &prog.values[*sub as usize];
                            (v.size, v.align, FieldTy::Value(*sub as u16))
                        }
                    };
                    off = align_up(off, align);
                    fields.push(FieldMeta { name, offset: off as u16, ty: fty });
                    off += sz;
                }
                TypeKind::Struct { fields }
            };
            ValueMeta { value_id: vid as u16, name: vl.name.clone(), kind }
        })
        .collect()
}

/// JIT-compile and run a 0-arg `-> i64` entry with a real GC runtime, returning
/// its result. Sets up a [`RuntimeContext`] over the program's layouts, wires
/// the `ai_gc_*` externs, and passes a live `Thread*`. `stress` forces a
/// collection on every allocation (exercises the GC + precise roots).
/// How the JIT driver configures the GC heap. Lets tests force frequent
/// collections (a tiny semi-space that fills during construction) to exercise
/// precise rooting and relocation without the (multithread-unsafe) every-alloc
/// stress mode.
#[derive(Clone, Copy)]
pub enum GcRunMode {
    /// Production default: generational nursery over tenured.
    Generational,
    /// `--gc-stress`: semi-space + collect on every allocation (single-thread aid).
    Stress,
    /// Semi-space of the given byte size; collects when it fills. A small size
    /// triggers real collections during a program's own construction.
    SemiSpace(usize),
}

pub fn jit_run_i64_gc(prog: &CoreProgram, stress: bool) -> Result<i64, CodegenError> {
    jit_run_i64_mode(prog, if stress { GcRunMode::Stress } else { GcRunMode::Generational })
}

/// JIT-run with an explicit GC heap mode (see [`GcRunMode`]).
pub fn jit_run_i64_mode(prog: &CoreProgram, mode: GcRunMode) -> Result<i64, CodegenError> {
    use crate::runtime::{self, RuntimeContext};
    let ctx = Context::create();
    let compiled = codegen(&ctx, prog)?;
    // Optimize the module (mem2reg + inlining + the standard O2 pipeline) so the
    // monomorphized, GC-framed code is actually fast. Without this the JIT runs
    // naive IR (every local a stack slot).
    optimize_module(&compiled.module);
    let ee = compiled
        .module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|e| CodegenError(e.to_string()))?;

    // Map runtime externs to their Rust implementations.
    for (name, addr) in [
        ("ai_gc_alloc_fixed", runtime::ai_gc_alloc_fixed as *const () as usize),
        ("ai_gc_alloc_varlen", runtime::ai_gc_alloc_varlen as *const () as usize),
        ("ai_gc_pollcheck_slow", runtime::ai_gc_pollcheck_slow as *const () as usize),
        ("ai_ffi_enter", runtime::ai_ffi_enter as *const () as usize),
        ("ai_ffi_leave", runtime::ai_ffi_leave as *const () as usize),
        ("ai_current_thread", runtime::ai_current_thread as *const () as usize),
        ("ai_ffi_reenter", runtime::ai_ffi_reenter as *const () as usize),
        ("ai_ffi_exit", runtime::ai_ffi_exit as *const () as usize),
        ("ai_print_int", runtime::ai_print_int as *const () as usize),
        ("ai_print_float", runtime::ai_print_float as *const () as usize),
        ("ai_print_str", runtime::ai_print_str as *const () as usize),
        ("ai_print_str_raw", runtime::ai_print_str_raw as *const () as usize),
        ("ai_str_len", runtime::ai_str_len as *const () as usize),
        ("ai_str_eq", runtime::ai_str_eq as *const () as usize),
        ("ai_str_concat", runtime::ai_str_concat as *const () as usize),
        ("ai_str_get", runtime::ai_str_get as *const () as usize),
        ("ai_str_substring", runtime::ai_str_substring as *const () as usize),
        ("ai_str_from_int", runtime::ai_str_from_int as *const () as usize),
        ("ai_str_from_float", runtime::ai_str_from_float as *const () as usize),
        ("ai_char_to_str", runtime::ai_char_to_str as *const () as usize),
        ("ai_type_name", runtime::ai_type_name as *const () as usize),
        ("ai_heap_snapshot", runtime::ai_heap_snapshot as *const () as usize),
        ("ai_reflect_field_count", runtime::ai_reflect_field_count as *const () as usize),
        ("ai_reflect_field_kind", runtime::ai_reflect_field_kind as *const () as usize),
        ("ai_reflect_field_i64", runtime::ai_reflect_field_i64 as *const () as usize),
        ("ai_reflect_field_name", runtime::ai_reflect_field_name as *const () as usize),
        ("ai_str_to_float", runtime::ai_str_to_float as *const () as usize),
        ("ai_str_hash", runtime::ai_str_hash as *const () as usize),
        ("ai_str_copy_to_buf", runtime::ai_str_copy_to_buf as *const () as usize),
        ("ai_buf_copy_in", runtime::ai_buf_copy_in as *const () as usize),
        ("ai_buf_copy_out", runtime::ai_buf_copy_out as *const () as usize),
        ("ai_thread_spawn", runtime::ai_thread_spawn as *const () as usize),
        ("ai_thread_join", runtime::ai_thread_join as *const () as usize),
        ("ai_thread_sleep", runtime::ai_thread_sleep as *const () as usize),
        ("ai_thread_yield", runtime::ai_thread_yield as *const () as usize),
        ("ai_thread_current_id", runtime::ai_thread_current_id as *const () as usize),
        ("ai_closure_code_ptr", runtime::ai_closure_code_ptr as *const () as usize),
        ("ai_chan_new", runtime::ai_chan_new as *const () as usize),
        ("ai_chan_send", runtime::ai_chan_send as *const () as usize),
        ("ai_chan_recv", runtime::ai_chan_recv as *const () as usize),
        ("ai_chan_sender_clone", runtime::ai_chan_sender_clone as *const () as usize),
        ("ai_chan_sender_drop", runtime::ai_chan_sender_drop as *const () as usize),
        ("ai_atomic_i64_new", runtime::ai_atomic_i64_new as *const () as usize),
        ("ai_atomic_i64_load", runtime::ai_atomic_i64_load as *const () as usize),
        ("ai_atomic_i64_store", runtime::ai_atomic_i64_store as *const () as usize),
        ("ai_atomic_i64_fetch_add", runtime::ai_atomic_i64_fetch_add as *const () as usize),
        ("ai_atomic_i64_compare_and_set", runtime::ai_atomic_i64_compare_and_set as *const () as usize),
        ("ai_gc_write_barrier", runtime::ai_gc_write_barrier as *const () as usize),
        ("ai_bounds_fail", runtime::ai_bounds_fail as *const () as usize),
    ] {
        if let Some(f) = compiled.module.get_function(name) {
            ee.add_global_mapping(&f, addr);
        }
    }

    let tis = layouts_to_type_infos(prog);
    // Normal runs use the GENERATIONAL heap: a small nursery (collected cheaply
    // and often by minor GCs) over a large tenured generation.
    //
    // `--gc-stress` instead uses the single-generation SEMI-SPACE heap with
    // collect-on-every-alloc. That mode's invariant checks (e.g. "every root
    // points to from-space after a flip") are semi-space-specific and don't hold
    // for a nursery, so stress validates relocation correctness on the
    // semi-space collector, the collector it was designed to torture-test.
    let mut rt = match mode {
        GcRunMode::Stress => {
            let rt = RuntimeContext::new(8 << 20, tis);
            rt.heap().set_gc_every_alloc(true);
            rt
        }
        GcRunMode::SemiSpace(size) => RuntimeContext::new(size, tis),
        GcRunMode::Generational => {
            // Shared with the AOT runtime so JIT and AOT agree (was 16 MB here vs
            // 1 MB in gcr_runtime_main — same program, different GC by run mode).
            let (nursery, tenured) = runtime::configured_heap_sizes();
            RuntimeContext::new_generational(nursery, tenured, tis)
        }
    };
    // Install reflection metadata so heap-exploration tooling and in-language
    // reflection can recover type/field names (the GC type table is nameless).
    rt.heap().set_type_meta(layouts_to_type_meta(prog));
    rt.heap().set_value_meta(layouts_to_value_meta(prog));
    // Install the allocation-site table (Target-1b) so GCR_ALLOC_PROFILE can
    // name each site's (function, type).
    rt.heap().set_alloc_sites(compiled.alloc_sites.clone());

    let addr = ee
        .get_function_address(&compiled.entry_name)
        .map_err(|_| CodegenError("entry not found".into()))?;
    type EntryFn = unsafe extern "C" fn(*mut runtime::Thread) -> i64;
    let f: EntryFn = unsafe { std::mem::transmute(addr) };
    let thread = rt.thread_ptr();
    // Publish the current thread so FFI callback trampolines can recover it.
    runtime::set_current_thread(thread);
    let result = unsafe { f(thread) };
    // Opt-in heap dump once the program returns (heap quiescent).
    // `GCR_HEAP_DUMP=json` emits a structured snapshot; other values emit text.
    if let Some(mode) = std::env::var_os("GCR_HEAP_DUMP") {
        if mode == "json" {
            eprint!("{}", unsafe { crate::gc::dump_heap_json(rt.heap()) });
        } else {
            eprint!("{}", unsafe { crate::gc::dump_heap_text(rt.heap()) });
        }
    }
    // Opt-in GC stats (set GCR_GC_STATS=1) — useful to confirm the generational
    // split is working: minor collections should dominate on young-heavy loads.
    if std::env::var("GCR_GC_STATS").is_ok() {
        eprintln!(
            "gc-rust: {} minor + {} major collections",
            rt.heap().minor_collections(),
            rt.heap().collections(),
        );
        // Pause-time + reclaim/promote summary (per kind, p50/p99/max).
        eprint!("{}", rt.heap().gc_stats_summary());
    }
    // Opt-in GC log: GCR_GC_LOG=<path> writes one JSON object per collection.
    if let Some(path) = std::env::var_os("GCR_GC_LOG") {
        let _ = std::fs::write(&path, rt.heap().gc_log_jsonl());
    }
    // Opt-in allocation-site profile (Target-1b): GCR_ALLOC_PROFILE=1 prints the
    // per-site count+bytes table at program end (heap quiescent).
    if std::env::var_os("GCR_ALLOC_PROFILE").is_some() {
        eprint!("{}", rt.heap().alloc_site_profile_report());
    }
    Ok(result)
}

/// JIT-compile and run a 0-arg `-> i64` entry. Uses the GC runtime so heap
/// programs work; allocation-free programs still run (null-frame fast path).
pub fn jit_run_i64(prog: &CoreProgram) -> Result<i64, CodegenError> {
    jit_run_i64_gc(prog, false)
}

// =============================================================================
// AOT (ahead-of-time) compilation: emit a native object + link an executable
// =============================================================================

/// Encode the program's layouts as `AotLayout` source records (the same data
/// the JIT path derives `TypeInfo`s from). `varlen`: 0=None, 1=Values, 2=Bytes.
/// The runtime's `gcr_runtime_main` rebuilds the `TypeInfo` table from these.
fn layouts_to_aot_records(prog: &CoreProgram) -> Vec<(u16, u16, u8)> {
    prog.layouts
        .iter()
        .map(|l| {
            let varlen = match l.varlen {
                crate::core::VarLen::None => 0u8,
                crate::core::VarLen::Values => 1u8,
                crate::core::VarLen::Bytes => 2u8,
            };
            (l.ptr_fields, l.raw_bytes, varlen)
        })
        .collect()
}

/// Build a `TargetMachine` configured for the host (mirrors `optimize_module`'s
/// setup) for object emission.
fn host_target_machine() -> Result<inkwell::targets::TargetMachine, CodegenError> {
    host_target_machine_opt(OptimizationLevel::Aggressive)
}

/// Build the host [`TargetMachine`] at a given backend optimization level. The
/// level matters for debug builds: even with the IR-level `optimize_module`
/// skipped, an `Aggressive` backend promotes allocas into registers and the
/// `dbg.declare` frame-slot locations go stale. Full-debug (P3) passes `None`
/// so locals stay in their stack slots and `frame variable` reads them.
fn host_target_machine_opt(
    opt: OptimizationLevel,
) -> Result<inkwell::targets::TargetMachine, CodegenError> {
    use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| CodegenError(format!("target init failed: {}", e)))?;
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple)
        .map_err(|e| CodegenError(format!("no target for {}: {}", triple, e.to_string())))?;
    // PIC reloc + Default code model so the object links cleanly into a normal
    // (PIE) executable produced by `cc`.
    target
        .create_target_machine(
            &triple,
            &TargetMachine::get_host_cpu_name().to_string(),
            &TargetMachine::get_host_cpu_features().to_string(),
            opt,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| CodegenError("could not create target machine".into()))
}

/// Compile `prog` to a native object file at `obj_path`.
///
/// The compiled program entry is renamed to `gcrust_entry` and a native
/// `int main()` is emitted that calls `gcr_runtime_main(&TYPE_TABLE, N,
/// gcrust_entry)` (which sets up the GC `RuntimeContext` and runs the program)
/// and returns the program's `i64` result truncated to the process exit code
/// (`i32`, so the shell sees it modulo 256). The per-program layout table is
/// emitted as an `AotLayout[]` global constant so the runtime can rebuild the
/// GC `TypeInfo` table at startup.
pub fn codegen_aot_object(
    prog: &CoreProgram,
    obj_path: &std::path::Path,
) -> Result<(), CodegenError> {
    codegen_aot_object_level(prog, obj_path, DebugLevel::LineTables)
}

/// Like [`codegen_aot_object`] but with an explicit [`DebugLevel`].
/// [`DebugLevel::Full`] (debugger P3) emits local-variable DIEs **and skips
/// optimization** — locals/allocas must survive for `frame variable` to read
/// them. The default ([`DebugLevel::LineTables`], P2) keeps the O2 pipeline;
/// LLVM preserves line debug-locations through it, so stepping still works.
pub fn codegen_aot_object_level(
    prog: &CoreProgram,
    obj_path: &std::path::Path,
    level: DebugLevel,
) -> Result<(), CodegenError> {
    use inkwell::targets::FileType;

    let ctx = Context::create();
    // AOT: emit DWARF (debugger P2/P3) — it lands in the object where
    // lldb / llvm-dwarfdump read it.
    let compiled = codegen_with_debug(&ctx, prog, level)?;
    let module = &compiled.module;

    // ---- Rename the program entry to a stable symbol ----------------------
    // It is currently named `prog.funcs[entry].name` (e.g. "main"); rename it so
    // the native `main` we emit below doesn't collide and the runtime can call a
    // known symbol.
    let entry_fn = module
        .get_function(&compiled.entry_name)
        .ok_or_else(|| CodegenError(format!("entry `{}` not found in module", compiled.entry_name)))?;
    entry_fn.as_global_value().set_name("gcrust_entry");

    let i16t = ctx.i16_type();
    let i8t = ctx.i8_type();
    let i32t = ctx.i32_type();
    let i64t = ctx.i64_type();
    let ptr = ctx.ptr_type(AddressSpace::default());

    // ---- Emit the layout table as an `AotLayout[]` constant ---------------
    // AotLayout (runtime, #[repr(C)]): { u16 ptr_fields, u16 raw_bytes,
    // u8 varlen, [3 x u8] pad } — size 8, align 2. The LLVM struct mirrors it.
    let aot_ty = ctx.struct_type(
        &[
            i16t.into(),
            i16t.into(),
            i8t.into(),
            i8t.array_type(3).into(),
        ],
        false,
    );
    let records = layouts_to_aot_records(prog);
    let pad0 = i8t.const_array(&[i8t.const_zero(), i8t.const_zero(), i8t.const_zero()]);
    let elems: Vec<_> = records
        .iter()
        .map(|(pf, rb, vl)| {
            aot_ty.const_named_struct(&[
                i16t.const_int(*pf as u64, false).into(),
                i16t.const_int(*rb as u64, false).into(),
                i8t.const_int(*vl as u64, false).into(),
                pad0.into(),
            ])
        })
        .collect();
    let table_arr = aot_ty.const_array(&elems);
    let table_global = module.add_global(
        aot_ty.array_type(records.len() as u32),
        None,
        "gcrust_type_table",
    );
    table_global.set_constant(true);
    table_global.set_initializer(&table_arr);

    // ---- Emit the reflection metadata blob as an i8[] constant ------------
    // Encoded by `gc::reflect::encode`; decoded by `gcr_runtime_main` at startup
    // into the heap's TypeMeta table (type/field names + field types). Nameless
    // programs (no reflection) still get a valid empty-table blob.
    let interior: Vec<Vec<u16>> = prog.layouts.iter().map(|l| l.interior_ptrs.clone()).collect();
    let meta_bytes = crate::gc::reflect::encode(
        &layouts_to_type_meta(prog),
        &layouts_to_value_meta(prog),
        &interior,
        &compiled.alloc_sites,
    );
    let meta_vals: Vec<_> = meta_bytes
        .iter()
        .map(|b| i8t.const_int(*b as u64, false))
        .collect();
    let meta_arr = i8t.const_array(&meta_vals);
    let meta_global = module.add_global(
        i8t.array_type(meta_bytes.len() as u32),
        None,
        "gcrust_type_meta",
    );
    meta_global.set_constant(true);
    meta_global.set_initializer(&meta_arr);

    // ---- Declare gcr_runtime_main -----------------------------------------
    // i64 gcr_runtime_main(ptr layouts, i64 ti_count, ptr meta, i64 meta_len,
    //                      ptr entry)
    let runtime_main_ty = i64t.fn_type(
        &[ptr.into(), i64t.into(), ptr.into(), i64t.into(), ptr.into()],
        false,
    );
    let runtime_main = module.add_function(
        "gcr_runtime_main",
        runtime_main_ty,
        Some(inkwell::module::Linkage::External),
    );

    // ---- Emit native `int main()` -----------------------------------------
    let main_ty = i32t.fn_type(&[], false);
    let main_fn = module.add_function("main", main_ty, None);
    let builder = ctx.create_builder();
    let bb = ctx.append_basic_block(main_fn, "entry");
    builder.position_at_end(bb);
    let entry_ptr = entry_fn.as_global_value().as_pointer_value();
    let call = builder
        .build_call(
            runtime_main,
            &[
                table_global.as_pointer_value().into(),
                i64t.const_int(records.len() as u64, false).into(),
                meta_global.as_pointer_value().into(),
                i64t.const_int(meta_bytes.len() as u64, false).into(),
                entry_ptr.into(),
            ],
            "rc",
        )
        .unwrap();
    let result = call_result(call).into_int_value();
    // Truncate the i64 program result to the process exit code (i32).
    let code = builder.build_int_truncate(result, i32t, "code").unwrap();
    builder.build_return(Some(&code)).unwrap();

    module
        .verify()
        .map_err(|e| CodegenError(format!("AOT module verify failed: {}", e.to_string())))?;

    // ---- Optimize then emit the object ------------------------------------
    // Full-debug builds (P3) skip optimization so locals/allocas survive for
    // `frame variable`. Line-table builds (P2) optimize as usual — debug
    // locations survive the O2 pipeline, so stepping is unaffected.
    // Full-debug builds (P3) keep the backend at `None` too: an optimizing
    // backend promotes the (still-present) allocas into registers, leaving the
    // `dbg.declare` frame-slot locations stale.
    let machine = if level.is_full() {
        host_target_machine_opt(OptimizationLevel::None)?
    } else {
        optimize_module(module);
        host_target_machine()?
    };
    machine
        .write_to_file(module, FileType::Object, obj_path)
        .map_err(|e| CodegenError(format!("object emission failed: {}", e.to_string())))?;
    Ok(())
}

/// AOT-compile `prog` into a standalone native executable at `out_path`.
///
/// Emits a native object (via [`codegen_aot_object`]) into a temp file, then
/// links it against the gc-rust runtime staticlib (`libgcrust_rt.a`, built by
/// cargo as a `staticlib` crate type) using the system `cc`. The runtime
/// provides the `ai_gc_*` / `ai_print_*` externs and `gcr_runtime_main`.
pub fn build_executable(
    prog: &CoreProgram,
    out_path: &std::path::Path,
    extra_link_args: &[String],
) -> Result<(), CodegenError> {
    build_executable_level(prog, out_path, extra_link_args, DebugLevel::LineTables)
}

/// Like [`build_executable`] but with an explicit [`DebugLevel`].
/// [`DebugLevel::Full`] is `gcr build --debug` (debugger P3: unoptimized +
/// local-variable DIEs so `lldb`'s `frame variable` shows source names/values).
pub fn build_executable_level(
    prog: &CoreProgram,
    out_path: &std::path::Path,
    extra_link_args: &[String],
    level: DebugLevel,
) -> Result<(), CodegenError> {
    // Emit the object next to the output so cleanup is easy and paths are stable.
    let obj_path = out_path.with_extension("o");
    codegen_aot_object_level(prog, &obj_path, level)?;

    let staticlib = locate_runtime_staticlib()?;

    // Link: object + runtime staticlib + system libs. The runtime is Rust, so it
    // needs libpthread/libdl/libm/libc + (on glibc) libgcc_s for unwinding. `cc`
    // pulls libc; we add the rest explicitly. `extra_link_args` (from `gcr build
    // --link-arg …`) is appended for FFI programs — notably the self-hosting
    // compiler, which links libLLVM (`-L<llvm>/lib -lLLVM -Wl,-rpath,…`).
    let status = std::process::Command::new("cc")
        .arg("-o")
        .arg(out_path)
        .arg(&obj_path)
        .arg(&staticlib)
        .args(["-lpthread", "-ldl", "-lm"])
        .args(extra_link_args)
        .status()
        .map_err(|e| CodegenError(format!("failed to invoke linker (cc): {}", e)))?;
    if !status.success() {
        return Err(CodegenError(format!(
            "linker (cc) failed with status {}",
            status
        )));
    }

    // On macOS, DWARF stays in the `.o` (the linked executable only holds a debug
    // MAP referencing it). Run `dsymutil` BEFORE we delete the `.o` to collect a
    // `.dSYM` bundle next to the executable, so `lldb` resolves gc-rust source
    // lines (debugger P2). Best-effort: if dsymutil is absent the binary still
    // runs, just without source-level debug. On ELF, DWARF is in the executable
    // already — no dSYM step.
    #[cfg(target_os = "macos")]
    {
        let ran = std::process::Command::new("dsymutil").arg(out_path).status();
        if ran.is_err() {
            let _ = std::process::Command::new("xcrun")
                .args(["dsymutil"])
                .arg(out_path)
                .status();
        }
    }

    // Remove the intermediate object on success.
    let _ = std::fs::remove_file(&obj_path);
    Ok(())
}

/// Locate the gc-rust runtime staticlib for AOT linking.
///
/// The path is baked in at build time by `build.rs` via the `GCRUST_RT_STATICLIB`
/// env var: the build script copies the `gcrust-rt` staticlib (built in the SAME
/// cargo profile as this `gcr`) into `OUT_DIR/libgcrust_rt_aot.a` and records its
/// path. This guarantees a profile-matched runtime (a debug `gcr` links the debug
/// runtime, a release `gcr` the release one) with no path-sniffing or runtime
/// `cargo` calls. `$GCRUST_RUNTIME_LIB` still overrides it for special cases.
fn locate_runtime_staticlib() -> Result<std::path::PathBuf, CodegenError> {
    use std::path::PathBuf;
    if let Ok(p) = std::env::var("GCRUST_RUNTIME_LIB") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Ok(p);
        }
        return Err(CodegenError(format!(
            "GCRUST_RUNTIME_LIB={} does not exist",
            p.display()
        )));
    }
    // Baked in by build.rs — always matches this binary's build profile.
    let baked = PathBuf::from(env!("GCRUST_RT_STATICLIB"));
    // Auto-refresh stale staticlibs. Cargo rebuilds the gcrust-rt *rlib* on every
    // `cargo build`/`run`/`test` (it's a dependency of `gcr`), but NOT the
    // *staticlib* AOT links — only `-p gcrust-rt` does. So after a runtime change
    // the `.a` goes stale and AOT binaries crash from an ABI mismatch. Here, in a
    // dev tree, we detect that (the always-fresh rlib being newer than the `.a`)
    // and rebuild the staticlib once, so `gcr build` is always correct without the
    // caller remembering `cargo build -p gcrust-rt`.
    refresh_staticlib_if_stale(&baked);
    if baked.exists() {
        return Ok(baked);
    }
    Err(CodegenError(format!(
        "gc-rust runtime staticlib not found at {} (set by build.rs). \
         Rebuild gcr (`cargo build [--release]`), or set $GCRUST_RUNTIME_LIB.",
        baked.display()
    )))
}

/// Rebuild the gcrust-rt staticlib via `cargo` if it is missing or older than the
/// runtime's own sources. No-op outside a dev tree — for an installed `gcr` the
/// source tree is absent, so the baked staticlib is used as-is.
///
/// Staleness is measured against the *sources* (newest `.rs`/`Cargo.toml` under
/// `crates/gcrust-rt`), not against the sibling top-level rlib: that rlib, like
/// the staticlib, is only refreshed by `-p gcrust-rt`/`--workspace`, so the two
/// move in lockstep and never reveal staleness. (The genuinely-fresh rlib a
/// normal `cargo build` produces lives hashed under `target/<p>/deps/`.) Source
/// mtime directly answers the real question: "was the `.a` built after the last
/// runtime edit?"
fn refresh_staticlib_if_stale(staticlib: &std::path::Path) {
    use std::fs;
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let rt_dir = manifest.join("crates").join("gcrust-rt");
    // Outside a dev checkout (installed gcr) there is nothing to rebuild from.
    if !rt_dir.exists() {
        return;
    }
    let mtime = |p: &std::path::Path| fs::metadata(p).and_then(|m| m.modified()).ok();
    let a_time = mtime(staticlib);
    let newest_src = newest_mtime_under(&rt_dir.join("src"))
        .into_iter()
        .chain(mtime(&rt_dir.join("Cargo.toml")))
        .max();
    let stale = match (a_time, newest_src) {
        (None, _) => true,                 // missing/cleaned → build it
        (Some(a), Some(src)) => src > a,    // a runtime source is newer than the .a
        (Some(_), None) => false,           // no sources found (shouldn't happen)
    };
    if !stale {
        return;
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    eprintln!("gcr: gcrust-rt staticlib is stale — rebuilding (cargo build -p gcrust-rt)…");
    let mut cmd = std::process::Command::new(cargo);
    cmd.args(["build", "-p", "gcrust-rt"]).current_dir(manifest);
    if !cfg!(debug_assertions) {
        cmd.arg("--release");
    }
    match cmd.status() {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("gcr: warning: staticlib rebuild exited with {s} (linking the existing .a)"),
        Err(e) => eprintln!("gcr: warning: could not run cargo to refresh staticlib: {e} (linking the existing .a)"),
    }
}

/// Newest modification time of any file in `dir` (recursive), or `None` if the
/// directory is empty/absent.
fn newest_mtime_under(dir: &std::path::Path) -> Option<std::time::SystemTime> {
    let mut newest: Option<std::time::SystemTime> = None;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(ft) if ft.is_dir() => stack.push(path),
                Ok(_) => {
                    if let Ok(m) = entry.metadata().and_then(|m| m.modified()) {
                        newest = Some(newest.map_or(m, |n| n.max(m)));
                    }
                }
                Err(_) => {}
            }
        }
    }
    newest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::lower::lower_program;
    use crate::parser::parse_module;
    use crate::resolve::resolve_module;

    fn run(src: &str) -> i64 {
        let m = parse_module(&lex(src).unwrap()).unwrap();
        let r = resolve_module(m).unwrap();
        let prog = lower_program(&r.globals).unwrap();
        jit_run_i64(&prog).unwrap()
    }

    #[test]
    fn arithmetic() {
        assert_eq!(run("fn main() -> i64 { 2 + 3 * 4 }"), 14);
        assert_eq!(run("fn main() -> i64 { (10 - 2) / 4 }"), 2);
    }

    #[test]
    fn if_expr() {
        assert_eq!(run("fn main() -> i64 { let x = 7; if x < 10 { 1 } else { 0 } }"), 1);
    }

    #[test]
    fn fib_runs() {
        assert_eq!(run(include_str!("../examples/fib.gcr")), 2178309);
    }

    #[test]
    fn unsigned_div() {
        // 200/3 unsigned = 66
        assert_eq!(run("fn main() -> i64 { let x = 200u32 / 3u32; x as i64 }"), 66);
    }

    #[test]
    fn recursion_and_calls() {
        assert_eq!(
            run("fn sq(n: i64) -> i64 { n * n } fn main() -> i64 { sq(9) + sq(3) }"),
            90
        );
    }

    #[test]
    fn while_loop_sum() {
        // sum 1..=100 iteratively = 5050
        assert_eq!(
            run("fn main() -> i64 { \
                   let mut i = 1; let mut acc = 0; \
                   while i <= 100 { acc = acc + i; i = i + 1; } \
                   acc }"),
            5050
        );
    }

    #[test]
    fn loop_with_break() {
        assert_eq!(
            run("fn main() -> i64 { \
                   let mut i = 0; \
                   loop { if i >= 42 { break; } i = i + 1; } \
                   i }"),
            42
        );
    }

    #[test]
    fn compound_assign() {
        assert_eq!(
            run("fn main() -> i64 { let mut x = 10; x += 5; x *= 2; x }"),
            30
        );
    }

    // ---- floats ------------------------------------------------------------

    fn run_f64(src: &str) -> i64 {
        // Wrap: a program returning i64 that does float work internally.
        run(src)
    }

    #[test]
    fn float_arithmetic() {
        // (1.5 * 4.0 + 1.0) as i64 = 7
        assert_eq!(run_f64("fn main() -> i64 { let x = 1.5 * 4.0 + 1.0; x as i64 }"), 7);
    }

    #[test]
    fn float_compare_and_branch() {
        let src = "fn main() -> i64 { \
                     let a = 3.14; \
                     if a > 3.0 { 1 } else { 0 } \
                   }";
        assert_eq!(run_f64(src), 1);
    }

    #[test]
    fn float_loop_accumulate() {
        // Sum 1.0..=10.0 as f64, return as i64 = 55.
        let src = "fn main() -> i64 { \
                     let mut i = 0; \
                     let mut acc = 0.0; \
                     while i < 10 { acc = acc + (i as f64) + 1.0; i = i + 1; } \
                     acc as i64 \
                   }";
        assert_eq!(run_f64(src), 55);
    }

    #[test]
    fn float_f32() {
        assert_eq!(run("fn main() -> i64 { let x = 2.5f32 * 2.0f32; x as i64 }"), 5);
    }

    #[test]
    fn block_stmt_then_paren() {
        // Regression: `while c {} (expr)` must be two statements, not a call.
        assert_eq!(
            run("fn main() -> i64 { let mut i = 0; while i < 10 { i = i + 1; } (i * 2) as i64 }"),
            20
        );
        // `if {} else {}` as a statement followed by `(...)`.
        assert_eq!(
            run("fn main() -> i64 { let mut x = 0; if true { x = 5; } else { x = 1; } (x + 1) }"),
            6
        );
    }

    #[test]
    fn binary_trees_allocation_heavy() {
        // The GC-throughput benchmark shape: build + check many heap trees.
        // depth 12 = 8191 nodes/tree; 8 trees = 65528. Exercises alloc + trace
        // + collection (the live tree is rooted across recursive allocations).
        let src = "enum Tree { Leaf, Node(Tree, Tree) } \
                   fn make(d: i64) -> Tree { if d == 0 { Tree::Leaf } else { Tree::Node(make(d-1), make(d-1)) } } \
                   fn check(t: Tree) -> i64 { match t { Tree::Leaf => 1, Tree::Node(l, r) => 1 + check(l) + check(r) } } \
                   fn main() -> i64 { \
                       let mut total = 0; let mut i = 0; \
                       while i < 8 { total = total + check(make(12)); i = i + 1; } \
                       total \
                   }";
        assert_eq!(run_gc(src, false), 65528);
    }

    #[test]
    fn nbody_kernel_runs() {
        // A scaled n-body-like float kernel with sqrt; deterministic checksum.
        let src = "fn pair(ax: f64, bx: f64, dt: f64) -> f64 { \
                       let dx = ax - bx; let d2 = dx * dx + 1.0; \
                       dt / (d2 * sqrt(d2)) \
                   } \
                   fn main() -> i64 { \
                       let mut i = 0; let mut acc = 0.0; let mut px = 1.0; let mut vx = 0.0; \
                       while i < 1000 { \
                           let dv = pair(px, 0.0, 0.01); \
                           vx = vx + dv; px = px + vx * 0.01; \
                           acc = acc + sqrt(px * px + 1.0); \
                           i = i + 1; \
                       } \
                       (acc * 100.0) as i64 \
                   }";
        // Just assert it runs and is positive + deterministic.
        let v = run(src);
        assert!(v > 0, "nbody kernel checksum should be positive, got {}", v);
    }

    #[test]
    fn float_sqrt() {
        // sqrt(144.0) = 12
        assert_eq!(run("fn main() -> i64 { let x = sqrt(144.0); x as i64 }"), 12);
    }

    #[test]
    fn float_sqrt_in_expr() {
        // sqrt(3*3 + 4*4) = sqrt(25) = 5
        let src = "fn main() -> i64 { \
                     let a = 3.0; let b = 4.0; \
                     let d = sqrt(a * a + b * b); \
                     d as i64 \
                   }";
        assert_eq!(run(src), 5);
    }

    // ---- heap types + GC ---------------------------------------------------

    fn run_gc(src: &str, stress: bool) -> i64 {
        let m = parse_module(&lex(src).unwrap()).unwrap();
        let r = resolve_module(m).unwrap();
        let prog = lower_program(&r.globals).unwrap();
        jit_run_i64_gc(&prog, stress).unwrap()
    }

    #[test]
    fn struct_alloc_and_field() {
        let src = "struct Point { x: i64, y: i64 } \
                   fn main() -> i64 { let p = Point { x: 3, y: 4 }; p.x + p.y }";
        assert_eq!(run_gc(src, false), 7);
    }

    #[test]
    fn struct_survives_gc_stress() {
        // Allocate, then more allocations (each triggers a collection under
        // stress) — the rooted struct must survive relocation and still read 7.
        let src = "struct Point { x: i64, y: i64 } \
                   fn mk(a: i64, b: i64) -> Point { Point { x: a, y: b } } \
                   fn main() -> i64 { \
                       let p = mk(3, 4); \
                       let _q = mk(100, 200); \
                       let _r = mk(5, 6); \
                       p.x + p.y \
                   }";
        assert_eq!(run_gc(src, true), 7);
    }

    #[test]
    fn enum_match() {
        let src = "enum Shape { Circle(i64), Square(i64) } \
                   fn area(s: Shape) -> i64 { \
                       match s { Shape::Circle(r) => r * r * 3, Shape::Square(w) => w * w } \
                   } \
                   fn main() -> i64 { area(Shape::Circle(2)) + area(Shape::Square(3)) }";
        // 2*2*3 + 3*3 = 12 + 9 = 21
        assert_eq!(run_gc(src, false), 21);
    }

    #[test]
    fn enum_match_under_stress() {
        let src = "enum Shape { Circle(i64), Square(i64) } \
                   fn area(s: Shape) -> i64 { \
                       match s { Shape::Circle(r) => r * r * 3, Shape::Square(w) => w * w } \
                   } \
                   fn main() -> i64 { area(Shape::Square(7)) }";
        assert_eq!(run_gc(src, true), 49);
    }

    // ---- match completion (scalars, literals, guards) ---------------------

    #[test]
    fn match_on_integer_literals() {
        let src = "fn classify(n: i64) -> i64 { \
                     match n { 0 => 100, 1 => 200, 2 => 300, _ => 999 } \
                   } \
                   fn main() -> i64 { \
                     classify(0) + classify(1) + classify(2) + classify(9) \
                   }";
        // 100 + 200 + 300 + 999 = 1599
        assert_eq!(run_gc(src, false), 1599);
    }

    #[test]
    fn match_with_guards() {
        let src = "fn sign(n: i64) -> i64 { \
                     match n { x if x < 0 => 0 - 1, 0 => 0, _ => 1 } \
                   } \
                   fn main() -> i64 { \
                     (sign(0 - 5) + 10) * 100 + (sign(0) + 10) * 10 + (sign(42) + 10) \
                   }";
        // sign: -1, 0, 1 -> (9)*100 + (10)*10 + (11) = 900+100+11 = 1011
        assert_eq!(run_gc(src, false), 1011);
    }

    #[test]
    fn match_on_string_literals() {
        let src = "fn code(s: String) -> i64 { \
                     match s { \"red\" => 1, \"green\" => 2, \"blue\" => 3, _ => 0 } \
                   } \
                   fn main() -> i64 { code(\"green\") * 100 + code(\"blue\") * 10 + code(\"x\") }";
        // 2*100 + 3*10 + 0 = 230
        assert_eq!(run_gc(src, false), 230);
    }

    #[test]
    fn match_enum_with_guard() {
        let src = "enum Opt { None, Some(i64) } \
                   fn describe(o: Opt) -> i64 { \
                     match o { \
                       Opt::Some(x) if x > 10 => 1000 + x, \
                       Opt::Some(x) => x, \
                       Opt::None => 0 - 1, \
                     } \
                   } \
                   fn main() -> i64 { describe(Opt::Some(42)) + describe(Opt::Some(5)) + describe(Opt::None) }";
        // 1042 + 5 + (-1) = 1046
        assert_eq!(run_gc(src, true), 1046);
    }

    #[test]
    fn deferred_init_let() {
        let src = "fn main() -> i64 { \
                     let mut x: i64; \
                     if true { x = 7; } else { x = 0; } \
                     let mut y: i64; \
                     y = x * 6; \
                     y \
                   }";
        assert_eq!(run_gc(src, false), 42);
    }

    // ---- monomorphized generics -------------------------------------------

    #[test]
    fn generic_identity() {
        let src = "fn id<T>(x: T) -> T { x } \
                   fn main() -> i64 { id(41) + 1 }";
        assert_eq!(run("fn id<T>(x: T) -> T { x } fn main() -> i64 { id(41) + 1 }"), 42);
        let _ = src;
    }

    #[test]
    fn generic_used_at_two_types() {
        // `first<T>` instantiated at i64 and at u32 → two specialized funcs.
        let src = "fn dup<T>(x: T) -> T { x } \
                   fn main() -> i64 { \
                       let a = dup(10); \
                       let b = dup(5u32); \
                       a + (b as i64) \
                   }";
        assert_eq!(run(src), 15);
    }

    #[test]
    fn generic_arithmetic_specializes() {
        // A generic that does work; signedness must follow the instantiation.
        let src = "fn twice<T>(x: T) -> T { x } \
                   fn main() -> i64 { \
                       let x = twice(7); \
                       let y = twice(200u32) / twice(3u32); \
                       x + (y as i64) \
                   }";
        // 7 + (200/3 unsigned = 66) = 73
        assert_eq!(run(src), 73);
    }

    #[test]
    fn string_literal_in_result() {
        // A String literal as a Result Err payload; match returns an i64 sentinel.
        let src = format!(
            "{PRELUDE} \
             fn check(b: i64) -> Result<i64, String> {{ \
                 if b == 0 {{ Result::Err(\"zero\") }} else {{ Result::Ok(b) }} \
             }} \
             fn get(r: Result<i64, String>) -> i64 {{ \
                 match r {{ Result::Ok(v) => v, Result::Err(_e) => -1 }} \
             }} \
             fn main() -> i64 {{ get(check(5)) + get(check(0)) }}"
        );
        // 5 + (-1) = 4
        assert_eq!(run_gc(&src, true), 4);
    }

    #[test]
    fn tuples() {
        assert_eq!(run("fn main() -> i64 { let t = (10, 32); t.0 + t.1 }"), 42);
        assert_eq!(
            run("fn pair() -> (i64, i64) { (3, 4) } fn main() -> i64 { let p = pair(); p.0 * p.1 }"),
            12
        );
        // mixed-type tuple
        assert_eq!(
            run("fn main() -> i64 { let t = (5, 2.5); t.0 + (t.1 as i64) }"),
            7
        );
    }

    // ---- ergonomics: for / index / field-assign ---------------------------

    #[test]
    fn for_range_loop() {
        assert_eq!(run("fn main() -> i64 { let mut s = 0; for i in 0..100 { s = s + i; } s }"), 4950);
        // inclusive range
        assert_eq!(run("fn main() -> i64 { let mut s = 0; for i in 1..=10 { s = s + i; } s }"), 55);
    }

    #[test]
    fn index_get_set() {
        let src = "fn main() -> i64 { \
                     let mut a: Array<i64> = array_new(4); \
                     a[0] = 10; a[1] = 20; a[3] = 12; \
                     a[0] + a[1] + a[3] \
                   }";
        assert_eq!(run_gc(src, false), 42);
    }

    #[test]
    fn index_in_for_loop() {
        let src = "fn main() -> i64 { \
                     let mut a: Array<i64> = array_new(50); \
                     for i in 0..50 { a[i] = i * 2; } \
                     let mut sum = 0; \
                     for j in 0..50 { sum = sum + a[j]; } \
                     sum \
                   }";
        // 2 * (0+..+49) = 2 * 1225 = 2450
        assert_eq!(run_gc(src, false), 2450);
    }

    #[test]
    fn struct_field_assignment() {
        let src = "struct Counter { n: i64 } \
                   fn main() -> i64 { \
                       let mut c = Counter { n: 0 }; \
                       c.n = 10; \
                       c.n = c.n + 32; \
                       c.n \
                   }";
        assert_eq!(run_gc(src, true), 42);
    }

    #[test]
    fn compound_index_assign() {
        let src = "fn main() -> i64 { \
                     let mut a: Array<i64> = array_new(2); \
                     a[0] = 10; a[0] += 5; a[0] *= 2; \
                     a[0] \
                   }";
        assert_eq!(run_gc(src, false), 30);
    }

    // ---- value types (inline, flat) ---------------------------------------

    #[test]
    fn value_enum_with_payloads() {
        let src = "#[value] enum Shape { Circle(i64), Rect(i64, i64), Empty } \
                   fn area(s: Shape) -> i64 { \
                       match s { \
                           Shape::Circle(r) => r * r * 3, \
                           Shape::Rect(w, h) => w * h, \
                           Shape::Empty => 0, \
                       } \
                   } \
                   fn main() -> i64 { \
                       area(Shape::Circle(2)) + area(Shape::Rect(4, 5)) + area(Shape::Empty) \
                   }";
        // 12 + 20 + 0 = 32
        assert_eq!(run_gc(src, false), 32);
    }

    #[test]
    fn value_option_with_payload() {
        let src = "#[value] enum Opt { None, Some(i64) } \
                   fn unwrap(o: Opt, d: i64) -> i64 { match o { Opt::Some(x) => x, Opt::None => d } } \
                   fn main() -> i64 { unwrap(Opt::Some(42), 0) + unwrap(Opt::None, 8) }";
        assert_eq!(run_gc(src, false), 50);
    }

    #[test]
    fn match_exhaustiveness_enforced() {
        // `f` IS reachable from main, so its non-exhaustive match is checked.
        let src = "enum E { A, B, C } \
                   fn f(e: E) -> i64 { match e { E::A => 1, E::B => 2 } } \
                   fn main() -> i64 { f(E::A) }";
        let m = parse_module(&lex(src).unwrap()).unwrap();
        let r = resolve_module(m).unwrap();
        assert!(lower_program(&r.globals).is_err(), "non-exhaustive match should be rejected");
        // A wildcard makes it exhaustive.
        let ok = "enum E { A, B, C } \
                  fn f(e: E) -> i64 { match e { E::A => 1, _ => 0 } } \
                  fn main() -> i64 { f(E::B) }";
        let m2 = parse_module(&lex(ok).unwrap()).unwrap();
        let r2 = resolve_module(m2).unwrap();
        assert!(lower_program(&r2.globals).is_ok());
    }

    #[test]
    fn value_enum_construct_and_match() {
        let src = "#[value] enum Color { Red, Green, Blue } \
                   fn code(c: Color) -> i64 { \
                       match c { Color::Red => 1, Color::Green => 2, Color::Blue => 3 } \
                   } \
                   fn main() -> i64 { \
                       code(Color::Red) + code(Color::Green) * 10 + code(Color::Blue) * 100 \
                   }";
        assert_eq!(run_gc(src, false), 321);
    }

    #[test]
    fn value_enum_no_heap_under_stress() {
        // C-style value enums are inline (no heap), so they survive GC stress.
        let src = "#[value] enum Dir { N, S, E, W } \
                   fn dx(d: Dir) -> i64 { match d { Dir::E => 1, Dir::W => 0 - 1, Dir::N => 0, Dir::S => 0 } } \
                   fn main() -> i64 { dx(Dir::E) + dx(Dir::W) + dx(Dir::E) }";
        assert_eq!(run_gc(src, true), 1);
    }

    #[test]
    fn value_struct_field_access() {
        let src = "#[value] struct Vec3 { x: i64, y: i64, z: i64 } \
                   fn main() -> i64 { \
                       let v = Vec3 { x: 3, y: 4, z: 5 }; \
                       v.x + v.y + v.z \
                   }";
        assert_eq!(run_gc(src, false), 12);
    }

    #[test]
    fn value_struct_passed_by_value() {
        let src = "#[value] struct Vec3 { x: f64, y: f64, z: f64 } \
                   fn dot(a: Vec3, b: Vec3) -> f64 { a.x * b.x + a.y * b.y + a.z * b.z } \
                   fn main() -> i64 { \
                       let v = Vec3 { x: 1.0, y: 2.0, z: 3.0 }; \
                       dot(v, v) as i64 \
                   }";
        // 1+4+9 = 14
        assert_eq!(run_gc(src, false), 14);
    }

    #[test]
    fn value_struct_in_let_chain() {
        let src = "#[value] struct P { a: i64, b: i64 } \
                   fn mk(x: i64) -> P { P { a: x, b: x * 2 } } \
                   fn main() -> i64 { \
                       let p = mk(10); \
                       let q = mk(5); \
                       p.a + p.b + q.a + q.b \
                   }";
        // (10+20) + (5+10) = 45
        assert_eq!(run_gc(src, false), 45);
    }

    #[test]
    fn value_struct_no_heap_alloc_under_stress() {
        // Value structs are inline — even under GC-every-alloc stress, building
        // and reading them triggers no heap allocation for the struct itself.
        let src = "#[value] struct Pt { x: i64, y: i64 } \
                   fn sum(p: Pt) -> i64 { p.x + p.y } \
                   fn main() -> i64 { \
                       let a = Pt { x: 40, y: 2 }; \
                       sum(a) \
                   }";
        assert_eq!(run_gc(src, true), 42);
    }

    // ---- arrays -----------------------------------------------------------

    #[test]
    fn array_int_set_get() {
        let src = "fn main() -> i64 { \
                     let mut a: Array<i64> = array_new(5); \
                     array_set(a, 0, 10); \
                     array_set(a, 1, 20); \
                     array_set(a, 4, 12); \
                     array_get_unchecked(a, 0) + array_get_unchecked(a, 1) + array_get_unchecked(a, 4) \
                   }";
        assert_eq!(run_gc(src, false), 42);
    }

    #[test]
    fn array_len_works() {
        let src = "fn main() -> i64 { let a: Array<i64> = array_new(7); array_len(a) }";
        assert_eq!(run_gc(src, false), 7);
    }

    #[test]
    fn array_sum_loop() {
        let src = "fn main() -> i64 { \
                     let mut a: Array<i64> = array_new(100); \
                     let mut i = 0; \
                     while i < 100 { array_set(a, i, i); i = i + 1; } \
                     let mut sum = 0; \
                     let mut j = 0; \
                     while j < 100 { sum = sum + array_get_unchecked(a, j); j = j + 1; } \
                     sum \
                   }";
        // sum 0..99 = 4950
        assert_eq!(run_gc(src, false), 4950);
    }

    #[test]
    fn array_float() {
        let src = "fn main() -> i64 { \
                     let mut a: Array<f64> = array_new(3); \
                     array_set(a, 0, 1.5); \
                     array_set(a, 1, 2.5); \
                     array_set(a, 2, 4.0); \
                     (array_get_unchecked(a, 0) + array_get_unchecked(a, 1) + array_get_unchecked(a, 2)) as i64 \
                   }";
        assert_eq!(run_gc(src, false), 8);
    }

    #[test]
    fn array_survives_gc_stress() {
        // Array of i64 filled, then more allocations under stress; the array is
        // rooted and its scalar contents must survive (untraced Bytes tail).
        let src = "fn mk() -> Array<i64> { array_new(10) } \
                   fn main() -> i64 { \
                       let mut a: Array<i64> = array_new(3); \
                       array_set(a, 0, 100); array_set(a, 1, 20); array_set(a, 2, 3); \
                       let _x = mk(); let _y = mk(); \
                       array_get_unchecked(a, 0) + array_get_unchecked(a, 1) + array_get_unchecked(a, 2) \
                   }";
        assert_eq!(run_gc(src, true), 123);
    }

    // ---- strings ----------------------------------------------------------

    #[test]
    fn str_len_and_eq() {
        let src = "fn main() -> i64 { \
                     let s = \"hello\"; \
                     let n = str_len(s); \
                     let e = if str_eq(s, \"hello\") { 1 } else { 0 }; \
                     let d = if str_eq(s, \"nope\") { 1 } else { 0 }; \
                     n * 100 + e * 10 + d \
                   }";
        // len 5, eq 1, neq 0 -> 510
        assert_eq!(run_gc(src, false), 510);
    }

    #[test]
    fn str_concat_basic() {
        let src = "fn main() -> i64 { \
                     let t = str_concat(\"foo\", \"barbaz\"); \
                     str_len(t) \
                   }";
        assert_eq!(run_gc(src, false), 9);
    }

    #[test]
    fn str_concat_survives_gc_stress() {
        // Repeatedly concatenate under stress (a collection per allocation); the
        // accumulated String is a rooted local and must survive every relocation.
        let src = "fn main() -> i64 { \
                     let mut acc = \"\"; \
                     let mut i = 0; \
                     while i < 50 { acc = str_concat(acc, \"ab\"); i = i + 1; } \
                     str_len(acc) \
                   }";
        assert_eq!(run_gc(src, true), 100);
    }

    #[test]
    fn str_get_and_substring() {
        let src = "fn main() -> i64 { \
                     let s = \"hello world\"; \
                     let c = str_get(s, 0); \
                     let sub = str_substring(s, 6, 11); \
                     c * 1000 + str_len(sub) \
                   }";
        // 'h' = 104, substring \"world\" len 5 -> 104005
        assert_eq!(run_gc(src, false), 104005);
    }

    #[test]
    fn to_string_int_and_concat() {
        let src = "fn main() -> i64 { \
                     let s = str_concat(\"n=\", to_string(42)); \
                     str_len(s) \
                   }";
        // \"n=42\" -> length 4
        assert_eq!(run_gc(src, false), 4);
    }

    #[test]
    fn str_to_float_intrinsic() {
        // \"2.5\" parses to 2.5; *4 = 10.0 -> 10 as i64.
        let src = "fn main() -> i64 { (str_to_float(\"2.5\") * 4.0) as i64 }";
        assert_eq!(run_gc(src, false), 10);
    }

    #[test]
    fn substring_survives_gc_stress() {
        // Take a substring then allocate heavily; the substring is a fresh rooted
        // String and must survive collection with its bytes intact.
        let src = "fn main() -> i64 { \
                     let sub = str_substring(\"abcdefgh\", 2, 6); \
                     let mut i = 0; \
                     while i < 30 { let mut j: Array<i64> = array_new(40); array_set(j, 0, i); i = i + 1; } \
                     str_len(sub) * 100 + str_get(sub, 0) \
                   }";
        // sub = \"cdef\", len 4, first byte 'c' = 99 -> 499
        assert_eq!(run_gc(src, true), 499);
    }

    #[test]
    fn print_returns_zero_and_runs() {
        // print_int returns 0; the program drives output as a side effect.
        assert_eq!(run("fn main() -> i64 { print_int(7); print_int(42); 0 }"), 0);
        assert_eq!(run("fn main() -> i64 { print_float(2.5); 0 }"), 0);
    }

    // ---- closures ---------------------------------------------------------

    #[test]
    fn closure_no_capture() {
        let src = "fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) } \
                   fn main() -> i64 { let inc = |n: i64| n + 1; apply(inc, 41) }";
        assert_eq!(run_gc(src, false), 42);
    }

    #[test]
    fn closure_captures_scalar() {
        let src = "fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) } \
                   fn main() -> i64 { \
                       let k = 10; \
                       let add_k = |n: i64| n + k; \
                       apply(add_k, 32) \
                   }";
        assert_eq!(run_gc(src, false), 42);
    }

    #[test]
    fn closure_called_directly() {
        let src = "fn main() -> i64 { let f = |a: i64, b: i64| a * b; f(6, 7) }";
        assert_eq!(run_gc(src, false), 42);
    }

    #[test]
    fn closure_captures_under_gc_stress() {
        let src = "fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) } \
                   fn main() -> i64 { \
                       let base = 100; \
                       let g = |n: i64| n + base; \
                       let _junk = apply(g, 1); \
                       let _junk2 = apply(g, 2); \
                       apply(g, 23) \
                   }";
        assert_eq!(run_gc(src, true), 123);
    }

    // ---- methods + traits -------------------------------------------------

    #[test]
    fn inherent_method() {
        let src = "struct Point { x: i64, y: i64 } \
                   impl Point { \
                       fn sum(self) -> i64 { self.x + self.y } \
                   } \
                   fn main() -> i64 { let p = Point { x: 3, y: 4 }; p.sum() }";
        assert_eq!(run_gc(src, false), 7);
    }

    #[test]
    fn method_with_args() {
        let src = "struct Counter { n: i64 } \
                   impl Counter { \
                       fn add(self, k: i64) -> i64 { self.n + k } \
                   } \
                   fn main() -> i64 { let c = Counter { n: 10 }; c.add(32) }";
        assert_eq!(run_gc(src, false), 42);
    }

    #[test]
    fn trait_method_static_dispatch() {
        let src = "trait Area { fn area(self) -> i64; } \
                   struct Square { side: i64 } \
                   impl Area for Square { fn area(self) -> i64 { self.side * self.side } } \
                   fn main() -> i64 { let s = Square { side: 6 }; s.area() }";
        assert_eq!(run_gc(src, false), 36);
    }

    #[test]
    fn method_under_gc_stress() {
        let src = "struct Vec2 { x: i64, y: i64 } \
                   impl Vec2 { fn dot(self, o: Vec2) -> i64 { self.x * o.x + self.y * o.y } } \
                   fn main() -> i64 { \
                       let a = Vec2 { x: 2, y: 3 }; \
                       let b = Vec2 { x: 4, y: 5 }; \
                       a.dot(b) \
                   }";
        // 2*4 + 3*5 = 23
        assert_eq!(run_gc(src, true), 23);
    }

    // ---- generic heap types (Option/Result) -------------------------------

    const PRELUDE: &str = "#[value] enum Option<T> { None, Some(T) } \
                           enum Result<T, E> { Ok(T), Err(E) } ";

    #[test]
    fn generic_option_construct_and_match() {
        let src = format!(
            "{PRELUDE} \
             fn unwrap_or(o: Option<i64>, d: i64) -> i64 {{ \
                 match o {{ Option::Some(x) => x, Option::None => d }} \
             }} \
             fn main() -> i64 {{ unwrap_or(Option::Some(42), 0) + unwrap_or(Option::None, 7) }}"
        );
        assert_eq!(run_gc(&src, false), 49);
    }

    #[test]
    fn generic_result_and_match() {
        let src = format!(
            "{PRELUDE} \
             fn safe_div(a: i64, b: i64) -> Result<i64, i64> {{ \
                 if b == 0 {{ Result::Err(0 - 1) }} else {{ Result::Ok(a / b) }} \
             }} \
             fn get(r: Result<i64, i64>) -> i64 {{ \
                 match r {{ Result::Ok(v) => v, Result::Err(e) => e }} \
             }} \
             fn main() -> i64 {{ get(safe_div(20, 4)) + get(safe_div(1, 0)) }}"
        );
        // 5 + (-1) = 4
        assert_eq!(run_gc(&src, false), 4);
    }

    #[test]
    fn generic_option_under_gc_stress() {
        let src = format!(
            "{PRELUDE} \
             fn unwrap_or(o: Option<i64>, d: i64) -> i64 {{ \
                 match o {{ Option::Some(x) => x, Option::None => d }} \
             }} \
             fn main() -> i64 {{ \
                 let a = Option::Some(100); \
                 let _b = Option::Some(200); \
                 let _c = Option::Some(300); \
                 unwrap_or(a, 0) \
             }}"
        );
        assert_eq!(run_gc(&src, true), 100);
    }

    #[test]
    fn try_operator() {
        let src = format!(
            "{PRELUDE} \
             fn checked(a: i64, b: i64) -> Result<i64, i64> {{ \
                 if b == 0 {{ Result::Err(0 - 99) }} else {{ Result::Ok(a / b) }} \
             }} \
             fn compute(x: i64, d: i64) -> Result<i64, i64> {{ \
                 let q = checked(x, d)?; \
                 let r = checked(q + 6, 2)?; \
                 Result::Ok(r + 1) \
             }} \
             fn get(r: Result<i64, i64>) -> i64 {{ \
                 match r {{ Result::Ok(v) => v, Result::Err(e) => e }} \
             }} \
             fn main() -> i64 {{ get(compute(20, 2)) + get(compute(1, 0)) }}"
        );
        // compute(20,2): q=10, r=(10+6)/2=8, Ok(9) -> 9
        // compute(1,0): checked(1,0)=Err(-99) -> ? returns Err(-99) -> get= -99
        // 9 + (-99) = -90
        assert_eq!(run_gc(&src, false), -90);
    }

    #[test]
    fn try_operator_under_stress() {
        let src = format!(
            "{PRELUDE} \
             fn checked(a: i64, b: i64) -> Result<i64, i64> {{ \
                 if b == 0 {{ Result::Err(0 - 1) }} else {{ Result::Ok(a / b) }} \
             }} \
             fn chain(x: i64) -> Result<i64, i64> {{ \
                 let a = checked(x, 2)?; \
                 let b = checked(a, 1)?; \
                 Result::Ok(b) \
             }} \
             fn get(r: Result<i64, i64>) -> i64 {{ match r {{ Result::Ok(v) => v, Result::Err(e) => e }} }} \
             fn main() -> i64 {{ get(chain(40)) }}"
        );
        assert_eq!(run_gc(&src, true), 20);
    }

    #[test]
    fn generic_struct_pair() {
        let src = "struct Pair<A, B> { first: A, second: B } \
                   fn main() -> i64 { \
                       let p = Pair { first: 10, second: 32 }; \
                       p.first + p.second \
                   }";
        assert_eq!(run_gc(src, true), 42);
    }

    #[test]
    fn nested_ref_struct_survives_gc() {
        // Wrap holds a *reference* field (inner: Pair) plus a raw tag. Under
        // stress, the inner Pair must be traced through Wrap's pointer slot and
        // both must relocate correctly.
        let src = "struct Pair { a: i64, b: i64 } \
                   struct Wrap { inner: Pair, tag: i64 } \
                   fn main() -> i64 { \
                       let p = Pair { a: 5, b: 6 }; \
                       let w = Wrap { inner: p, tag: 9 }; \
                       w.inner.a + w.inner.b + w.tag \
                   }";
        assert_eq!(run_gc(src, true), 20);
    }
}
