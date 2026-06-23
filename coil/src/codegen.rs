//! LLVM lowering via inkwell.
//!
//! Conventions (M1/M2): `:native` emits one function with a built-in LLVM
//! calling convention; `:shim` emits a `ccc` `__impl` + a `naked` trampoline.
//!
//! Allocation/types (M3+): values carry their Coil `Type` through codegen as a
//! `(BasicValueEnum, Type)` pair, so `load`/`store!`/`index` use the right width
//! and pointee, integer widths (i8/i16/i32/i64) and `cast` work, and pointers
//! carry a region + pointee type (e.g. C `char**`).

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::debug_info::{
    AsDIScope, DIFile, DISubprogram, DISubroutineType, DWARFEmissionKind, DWARFSourceLanguage,
    DebugInfoBuilder, DIFlagsConstants,
};
use inkwell::module::{FlagBehavior, Module};
use inkwell::targets::{
    CodeModel, InitializationConfig, RelocMode, Target, TargetData, TargetMachine,
};
use inkwell::types::{
    AnyType, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType, StructType,
};
use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue};
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::{AddressSpace, InlineAsmDialect, IntPredicate, OptimizationLevel};

use crate::abi::{self, AbiCtx, Class, StructAbi};
use crate::ast::*;
use crate::convention::Lowering;

/// The architecture codegen is targeting. This selects the C-ABI argument
/// register sequence and the assembly dialect/prologue used by `:shim`
/// trampolines and register-constrained shim call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetArch {
    X86_64,
    AArch64,
}

impl TargetArch {
    /// Derive the arch from an LLVM triple's arch component (the part before
    /// the first `-`). Unsupported arches are a hard error rather than a silent
    /// fallback: shim trampolines emit raw asm and would miscompile otherwise.
    fn from_triple(triple: &str) -> Result<TargetArch, String> {
        let arch = triple.split('-').next().unwrap_or("");
        match arch {
            "x86_64" | "amd64" => Ok(TargetArch::X86_64),
            "aarch64" | "arm64" | "arm64e" => Ok(TargetArch::AArch64),
            other => Err(format!(
                "codegen: unsupported target architecture '{other}' (triple '{triple}'); \
                 shim calling conventions are only implemented for x86_64 and aarch64"
            )),
        }
    }

    /// The C-ABI integer/pointer argument registers, in order. x86-64 SysV uses
    /// six; AArch64 AAPCS64 uses eight (x0-x7).
    fn c_arg_regs(self) -> &'static [&'static str] {
        match self {
            TargetArch::X86_64 => &["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
            TargetArch::AArch64 => &["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
        }
    }

    /// The C-ABI integer return register.
    fn c_ret_reg(self) -> &'static str {
        match self {
            TargetArch::X86_64 => "rax",
            TargetArch::AArch64 => "x0",
        }
    }

    /// Whether a register name is a valid general-purpose register on this arch.
    /// Used to reject a convention naming registers that don't exist on the
    /// target (e.g. an x86 `defcc` compiled for aarch64) with a clear error
    /// instead of emitting asm the assembler will choke on.
    fn is_valid_gpr(self, reg: &str) -> bool {
        match self {
            TargetArch::X86_64 => matches!(
                reg,
                "rax" | "rbx" | "rcx" | "rdx" | "rsi" | "rdi" | "rbp" | "rsp"
                    | "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15"
            ),
            // x0..x30 plus the special names. (We don't model the 32-bit w-regs
            // here; shim conventions pass 64-bit values.)
            TargetArch::AArch64 => {
                matches!(reg, "sp" | "lr" | "fp" | "xzr")
                    || (reg.starts_with('x')
                        && reg[1..].parse::<u8>().map(|n| n <= 30).unwrap_or(false))
            }
        }
    }

    fn arch_name(self) -> &'static str {
        match self {
            TargetArch::X86_64 => "x86_64",
            TargetArch::AArch64 => "aarch64",
        }
    }

    fn to_abi(self) -> abi::Arch {
        match self {
            TargetArch::X86_64 => abi::Arch::X86_64,
            TargetArch::AArch64 => abi::Arch::AArch64,
        }
    }
}

/// A value plus its Coil type.
type Tv<'ctx> = (BasicValueEnum<'ctx>, Type);

struct ShimInfo<'ctx> {
    impl_fn: FunctionValue<'ctx>,
    param_regs: Vec<String>,
    ret_reg: String,
    clobber: Vec<String>,
}

struct StructInfo<'ctx> {
    fields: Vec<(String, Type)>,
    ty: StructType<'ctx>,
    layout: Layout,
}

/// How one original parameter is realized at the LLVM/ABI level.
enum ArgAbi<'ctx> {
    /// An ordinary scalar/pointer/sum value, passed unchanged.
    Scalar,
    /// A by-value struct passed directly in registers, coerced to these slot
    /// types (one LLVM parameter each).
    Direct(StructAbi<'ctx>),
    /// A by-value struct passed indirectly: a pointer to a copy, `byval(T)` on
    /// x86-64, a plain pointer on AArch64.
    Indirect(StructAbi<'ctx>),
}

/// The C-ABI realization of a callable's signature: how the return value and each
/// argument are lowered, plus the resulting LLVM function type. Computed once per
/// callable and reused at its declaration, body, and every call site so all three
/// agree on the wire format.
struct CSig<'ctx> {
    /// If the return is a large struct, its `sret` classification (a hidden
    /// pointer becomes the first LLVM parameter).
    sret: Option<StructAbi<'ctx>>,
    /// The coerced LLVM return type for a small by-value struct return (`None`
    /// for a scalar return or an `sret` return, which is LLVM-void).
    ret_direct: Option<StructAbi<'ctx>>,
    /// Per original parameter, in source order.
    args: Vec<ArgAbi<'ctx>>,
    fn_ty: FunctionType<'ctx>,
}

/// A sum type's runtime shape: `{ i32 tag, [words x i64] payload }`, plus a
/// per-variant struct type used to read/write the variant's fields out of the
/// payload.
struct SumInfo<'ctx> {
    variants: Vec<(String, Vec<(String, Type)>)>,
    ty: StructType<'ctx>,
    variant_structs: Vec<StructType<'ctx>>,
}

/// An LLVM integer type of arbitrary bit width. inkwell ≥0.7 takes a `NonZeroU32`
/// and returns a `Result` (rejecting a zero width); Coil widths are validated
/// nonzero at parse, so neither failure mode is reachable here.
pub(crate) fn int_width(ctx: &Context, bits: u32) -> inkwell::types::IntType<'_> {
    ctx.custom_width_int_type(std::num::NonZeroU32::new(bits).expect("int width is nonzero"))
        .expect("valid int width")
}

/// What the caller supplies to turn on DWARF emission: the source text (to map a
/// byte span to a line/column) and the file identity for the `DIFile`.
pub struct DebugInput<'a> {
    pub source: &'a str,
    pub file_name: String,
    pub directory: String,
}

/// Live DWARF-emission state held on `Cg` while a module is built. Function
/// granularity for now (a CU + a `DISubprogram` per function with its source
/// line); per-statement line tables and local-variable info are a later slice.
struct DebugCtx<'ctx> {
    builder: DebugInfoBuilder<'ctx>,
    file: DIFile<'ctx>,
    /// Reused subroutine type — line info needs a type, but typed parameters are
    /// a later increment, so every function shares one opaque `() -> ?` signature.
    subroutine_ty: DISubroutineType<'ctx>,
    /// Byte offset of the start of each source line, for offset → (line, col).
    line_starts: Vec<u32>,
}

impl<'ctx> DebugCtx<'ctx> {
    /// 1-based (line, column) for a byte offset; `(0, 0)` for a dummy/out-of-range
    /// span (a synthesized or included-file node — no line, never a wrong one).
    fn line_col(&self, off: u32) -> (u32, u32) {
        if off == u32::MAX {
            return (0, 0);
        }
        // Last line whose start is <= off.
        let line = self.line_starts.partition_point(|&s| s <= off);
        let line_start = self.line_starts.get(line.saturating_sub(1)).copied().unwrap_or(0);
        (line as u32, off - line_start + 1)
    }
}

struct Cg<'ctx> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    /// DWARF emission state, when building with debug info (`-g`).
    di: Option<DebugCtx<'ctx>>,
    /// The DISubprogram of the function whose body is currently being emitted —
    /// the scope for per-statement debug locations set in `emit_expr`. `None`
    /// outside a `-g` body or for a dummy-span function.
    cur_sp: std::cell::RefCell<Option<DISubprogram<'ctx>>>,
    funcs: HashMap<String, FunctionValue<'ctx>>,
    shims: HashMap<String, ShimInfo<'ctx>>,
    structs: HashMap<String, StructInfo<'ctx>>,
    sums: HashMap<String, SumInfo<'ctx>>,
    /// Coil return type of every callable (function/extern), for typing calls.
    rets: HashMap<String, Type>,
    /// Full signature of every callable, for `fnptr-of` (cc, params, ret).
    callables: HashMap<String, (String, Vec<Type>, Type)>,
    /// C-ABI realization of every native-C callable that passes/returns a struct
    /// by value (so declaration, body, and call sites agree on the wire format).
    csigs: HashMap<String, CSig<'ctx>>,
    /// Native LLVM calling-convention id for each convention name.
    conv_ids: HashMap<String, u32>,
    /// Target layout, for sizeof/alignof/offsetof and static asserts.
    target_data: TargetData,
    /// Target architecture, selecting the ABI register sequence and the asm
    /// dialect/prologue for `:shim` trampolines and call sites.
    arch: TargetArch,
    /// Whether the object format is Mach-O (Apple), where global symbols carry a
    /// leading underscore that inline-asm symbol references must spell out.
    mach_o: bool,
    globals: Cell<u32>,
    /// Monotonic counter for naming `(llvm-ir ...)` inlined helper functions.
    llvm_seq: Cell<u32>,
    /// Stack of enclosing loops (innermost last) during body emission: their
    /// continue/break target blocks and the values seen at break sites (for the
    /// after-block phi). Interior mutability because `emit_expr` takes `&self`.
    loops: RefCell<Vec<LoopCtxCg<'ctx>>>,
}

/// Codegen bookkeeping for one enclosing loop.
struct LoopCtxCg<'ctx> {
    label: Option<String>,
    /// Loop header — the `continue` target and the back-edge destination.
    body_bb: BasicBlock<'ctx>,
    /// Block after the loop — the `break` target; the loop's value materializes
    /// here as a phi over `breaks`.
    after_bb: BasicBlock<'ctx>,
    /// (value, predecessor-block) contributed by each `break` that fires.
    breaks: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)>,
    /// Coil type of the break value (from the first break), for the phi/result.
    break_ty: Option<Type>,
}

pub fn compile<'ctx>(ctx: &'ctx Context, program: &Program) -> Result<Module<'ctx>, String> {
    compile_for(ctx, program, target_triple())
}

/// The triple codegen and object emission target: the explicit `COIL_TARGET`
/// override (e.g. `x86_64-apple-macosx11.0.0` to cross-produce the SysV ABI from
/// an arm64 host), else the host triple. Shared by `codegen` and `lib` so the IR
/// and the emitted object agree on the ABI.
pub fn target_triple() -> inkwell::targets::TargetTriple {
    match std::env::var("COIL_TARGET") {
        Ok(t) if !t.is_empty() => inkwell::targets::TargetTriple::create(&t),
        _ => TargetMachine::get_default_triple(),
    }
}

/// Like `compile`, but for an explicitly chosen target triple (cross-targeting).
pub fn compile_for<'ctx>(
    ctx: &'ctx Context,
    program: &Program,
    triple: inkwell::targets::TargetTriple,
) -> Result<Module<'ctx>, String> {
    compile_for_dbg(ctx, program, triple, None)
}

/// `compile_for`, optionally emitting DWARF debug info (`Some(DebugInput)` ⇒ a
/// compile unit + per-function line info, so `lldb`/`gdb` can map functions to
/// source and show file:line in backtraces).
pub fn compile_for_dbg<'ctx>(
    ctx: &'ctx Context,
    program: &Program,
    triple: inkwell::targets::TargetTriple,
    dbg: Option<DebugInput>,
) -> Result<Module<'ctx>, String> {
    // Initialize all targets so an arbitrary (possibly non-host) triple resolves.
    Target::initialize_all(&InitializationConfig::default());
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let tm = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::Default,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or("codegen: could not create target machine")?;
    let target_data = tm.get_target_data();
    let triple_str = triple.as_str().to_string_lossy().into_owned();
    let arch = TargetArch::from_triple(&triple_str)?;
    // Mach-O (Apple) prefixes global symbols with `_`; inline asm referencing a
    // symbol by name must include it. ELF (Linux) does not.
    let mach_o = triple_str.contains("apple") || triple_str.contains("darwin") || triple_str.contains("macosx");
    let module = ctx.create_module("coil");
    module.set_triple(&triple);
    module.set_data_layout(&target_data.get_data_layout());

    // DWARF setup (when building with `-g`): a compile unit + the module flags the
    // verifier and the optimizer require for debug info to survive `-O3`.
    let di = dbg.map(|d| {
        // LLVM requires these module flags; inkwell's builder doesn't add them.
        module.add_basic_value_flag(
            "Debug Info Version",
            FlagBehavior::Warning,
            ctx.i32_type().const_int(3, false), // LLVMDebugMetadataVersion
        );
        module.add_basic_value_flag(
            "Dwarf Version",
            FlagBehavior::Warning,
            ctx.i32_type().const_int(4, false),
        );
        let (dib, cu) = module.create_debug_info_builder(
            true, // allow unresolved (finalized later)
            DWARFSourceLanguage::C, // Coil isn't C, but C is the honest closest
            &d.file_name,
            &d.directory,
            "coil",
            // NOT optimized: a `-g` build runs an almost-empty pipeline, so lldb
            // can trust the DWARF (prologue_end, variable locations) instead of
            // heuristically skipping the prologue — which otherwise lands a
            // breakpoint before the parameter stores and shows stale values.
            false,
            "",   // flags
            0,    // runtime version
            "",   // split name
            DWARFEmissionKind::Full,
            0,     // DWO id
            false, // split debug inlining
            false, // debug info for profiling
            "",    // sysroot
            "",    // SDK
        );
        let file = cu.get_file();
        let subroutine_ty =
            dib.create_subroutine_type(file, None, &[], inkwell::debug_info::DIFlags::ZERO);
        // Precompute line starts (offset 0, then each byte after a '\n').
        let mut line_starts = vec![0u32];
        line_starts.extend(
            d.source
                .bytes()
                .enumerate()
                .filter(|&(_, b)| b == b'\n')
                .map(|(i, _)| i as u32 + 1),
        );
        DebugCtx { builder: dib, file, subroutine_ty, line_starts }
    });

    let mut cg = Cg {
        ctx,
        module,
        builder: ctx.create_builder(),
        di,
        cur_sp: std::cell::RefCell::new(None),
        funcs: HashMap::new(),
        shims: HashMap::new(),
        structs: HashMap::new(),
        sums: HashMap::new(),
        rets: HashMap::new(),
        callables: HashMap::new(),
        csigs: HashMap::new(),
        conv_ids: HashMap::new(),
        target_data,
        arch,
        mach_o,
        globals: Cell::new(0),
        llvm_seq: Cell::new(0),
        loops: RefCell::new(Vec::new()),
    };

    for (name, conv) in &program.conventions {
        if let Some(id) = conv.native_id() {
            cg.conv_ids.insert(name.clone(), id);
        }
    }

    // 0. build aggregate types. Two-phase (opaque names first, then bodies) so
    //    definition order doesn't matter (monomorphization emits in any order).
    // 0a. opaque struct names.
    for sd in &program.structs {
        let ty = ctx.opaque_struct_type(&sd.name);
        cg.structs.insert(
            sd.name.clone(),
            StructInfo {
                fields: sd.fields.clone(),
                ty,
                layout: sd.layout.clone(),
            },
        );
    }
    // 0b. sum types `{ i32 tag, [words x i64] payload }` (size from a conservative
    //     layout of the Coil types — no LLVM layout needed, just an upper bound).
    let struct_map: HashMap<&str, &StructDef> =
        program.structs.iter().map(|s| (s.name.as_str(), s)).collect();
    let sum_map: HashMap<&str, &SumDef> =
        program.sums.iter().map(|s| (s.name.as_str(), s)).collect();
    for sd in &program.sums {
        let words = sum_words(sd, &struct_map, &sum_map);
        let payload = ctx.i64_type().array_type(words);
        let ty = ctx.opaque_struct_type(&sd.name);
        ty.set_body(&[ctx.i32_type().into(), payload.into()], false);
        cg.sums.insert(
            sd.name.clone(),
            SumInfo {
                variants: sd.variants.iter().map(|v| (v.name.clone(), v.fields.clone())).collect(),
                ty,
                variant_structs: vec![], // filled in 0d
            },
        );
    }
    // 0c. struct bodies (may reference sums, which are now complete).
    for sd in &program.structs {
        let ty = cg.structs[&sd.name].ty;
        match &sd.layout {
            // explicit layout -> a flat byte blob; field access is byte-offset GEP.
            Layout::Explicit(e) => {
                let computed = sd
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, (_, fty))| e.offsets[i] + cg.target_data.get_abi_size(&cg.basic_ty(fty)))
                    .max()
                    .unwrap_or(0);
                let size = e.size.map_or(computed, |s| s.max(computed));
                ty.set_body(&[ctx.i8_type().array_type(size as u32).into()], true);
            }
            // bit structs are an integer; give the placeholder type a body anyway.
            Layout::Bits(b) => {
                ty.set_body(&[int_width(ctx, b.backing).into()], false);
            }
            other => {
                let field_types: Vec<BasicTypeEnum> =
                    sd.fields.iter().map(|(_, t)| cg.basic_ty(t)).collect();
                ty.set_body(&field_types, matches!(other, Layout::Packed));
            }
        }
    }
    // 0d. per-variant field structs (used to read/write payloads).
    for sd in &program.sums {
        let vss: Vec<StructType> = sd
            .variants
            .iter()
            .map(|v| {
                let fs: Vec<BasicTypeEnum> = v.fields.iter().map(|(_, t)| cg.basic_ty(t)).collect();
                ctx.struct_type(&fs, false)
            })
            .collect();
        cg.sums.get_mut(&sd.name).unwrap().variant_structs = vss;
    }

    // 0e. static asserts — evaluate now that every layout is known.
    for a in &program.asserts {
        if cg.const_eval(&a.cond)? == 0 {
            return Err(format!("static assertion failed: {}", a.msg));
        }
    }

    // 1a. declare externs (foreign symbols resolved at link time).
    for e in &program.externs {
        let conv = program
            .conventions
            .get(&e.cc)
            .ok_or_else(|| format!("codegen: unknown convention '{}'", e.cc))?;
        let cc_id = conv
            .native_id()
            .ok_or_else(|| format!("codegen: extern '{}' needs a native convention", e.name))?;
        let native_c = matches!(&conv.lowering, Lowering::Native(crate::convention::NativeCc::C));
        cg.check_c_abi_types(&format!("extern '{}'", e.name), &e.params, &e.ret)?;
        let fv = if cg.needs_c_abi(&e.params, &e.ret) {
            // A by-value struct in the signature: lower via the C struct ABI
            // (coerced register slots, `byval`/`sret`) instead of the naive types.
            let sig = cg.c_signature(&e.name, &e.params, &e.ret, e.variadic, native_c)?;
            let fv = cg.module.add_function(&e.name, sig.fn_ty, None);
            fv.set_call_conventions(cc_id);
            for (loc, attr) in cg.csig_attrs(&sig) {
                fv.add_attribute(loc, attr);
            }
            cg.csigs.insert(e.name.clone(), sig);
            fv
        } else {
            let p: Vec<BasicMetadataTypeEnum> =
                e.params.iter().map(|t| cg.basic_ty(t).into()).collect();
            let fn_ty = cg.fn_type_with_ret(&e.ret, &p, e.variadic);
            let fv = cg.module.add_function(&e.name, fn_ty, None);
            fv.set_call_conventions(cc_id);
            fv
        };
        cg.funcs.insert(e.name.clone(), fv);
        cg.rets.insert(e.name.clone(), e.ret.clone());
        cg.callables
            .insert(e.name.clone(), (e.cc.clone(), e.params.clone(), e.ret.clone()));
    }

    // 1b. declare all functions (so mutual recursion resolves).
    for f in &program.funcs {
        let conv = program
            .conventions
            .get(&f.cc)
            .ok_or_else(|| format!("codegen: unknown convention '{}'", f.cc))?;
        let fn_ty = cg.fn_type(&f.params, &f.ret);
        cg.rets.insert(f.name.clone(), f.ret.clone());
        cg.callables.insert(
            f.name.clone(),
            (
                f.cc.clone(),
                f.params.iter().map(|p| p.ty.clone()).collect(),
                f.ret.clone(),
            ),
        );

        let f_params: Vec<Type> = f.params.iter().map(|p| p.ty.clone()).collect();
        match &conv.lowering {
            Lowering::Native(cc) => {
                // A function that returns (or, in principle, takes) a struct by
                // value across the C ABI is lowered the same way as an extern, so
                // a C caller — or a Coil call site — agrees on the wire format.
                if cg.needs_c_abi(&f_params, &f.ret) {
                    let native_c = *cc == crate::convention::NativeCc::C;
                    let sig = cg.c_signature(&f.name, &f_params, &f.ret, false, native_c)?;
                    let fv = cg.module.add_function(&f.name, sig.fn_ty, None);
                    fv.set_call_conventions(cc.id());
                    for (loc, attr) in cg.csig_attrs(&sig) {
                        fv.add_attribute(loc, attr);
                    }
                    cg.csigs.insert(f.name.clone(), sig);
                    cg.funcs.insert(f.name.clone(), fv);
                } else {
                    let fv = cg.module.add_function(&f.name, fn_ty, None);
                    fv.set_call_conventions(cc.id());
                    cg.funcs.insert(f.name.clone(), fv);
                }
            }
            Lowering::Shim => {
                let ret_reg = conv
                    .ret
                    .clone()
                    .ok_or_else(|| format!("shim convention '{}' needs :ret", conv.name))?;
                let impl_fn = cg.module.add_function(&format!("{}__impl", f.name), fn_ty, None);
                let void_ty = ctx.void_type().fn_type(&[], false);
                let tramp = cg.module.add_function(&f.name, void_ty, None);
                for kind in ["naked", "noinline"] {
                    let attr = ctx.create_enum_attribute(Attribute::get_named_enum_kind_id(kind), 0);
                    tramp.add_attribute(AttributeLoc::Function, attr);
                }
                cg.funcs.insert(f.name.clone(), tramp);
                cg.shims.insert(
                    f.name.clone(),
                    ShimInfo {
                        impl_fn,
                        param_regs: conv.params.clone(),
                        ret_reg,
                        clobber: conv.clobber.clone(),
                    },
                );
            }
        }
    }

    // 2. emit bodies (+ trampolines).
    for f in &program.funcs {
        if let Some(shim) = cg.shims.get(&f.name) {
            cg.emit_func(f, shim.impl_fn)?;
            let tramp = cg.funcs[&f.name];
            cg.emit_trampoline(tramp, shim, f.params.len())?;
        } else {
            let fv = cg.funcs[&f.name];
            cg.emit_func(f, fv)?;
        }
    }

    // Finalize debug info (resolve forward refs) before verifying / optimizing.
    if let Some(di) = &cg.di {
        di.builder.finalize();
    }

    cg.module
        .verify()
        .map_err(|e| format!("LLVM module verification failed:\n{}", e.to_string()))?;
    Ok(cg.module)
}

impl<'ctx> Cg<'ctx> {
    fn basic_ty(&self, t: &Type) -> BasicTypeEnum<'ctx> {
        match t {
            // A Never value is never materialized (the expression diverges first);
            // this placeholder only labels slots/phis on dead, unreachable paths.
            Type::Never => self.ctx.i64_type().into(),
            Type::Int(bits, _) => int_width(self.ctx, *bits).into(),
            Type::Float(32) => self.ctx.f32_type().into(),
            Type::Float(_) => self.ctx.f64_type().into(),
            Type::Bool => self.ctx.bool_type().into(),
            Type::Ptr(..) => self.ctx.ptr_type(AddressSpace::default()).into(),
            Type::Struct(name) => {
                if let Some(s) = self.structs.get(name) {
                    // a :layout bits struct is represented by its backing integer.
                    if let Layout::Bits(b) = &s.layout {
                        return int_width(self.ctx, b.backing).into();
                    }
                    s.ty.into()
                } else if let Some(s) = self.sums.get(name) {
                    s.ty.into()
                } else {
                    panic!("unknown nominal type '{name}'")
                }
            }
            Type::Array(elem, n) => self.basic_ty(elem).array_type(*n).into(),
            // A slice is a fat pointer `{ data: ptr, len: i64 }`. The element
            // type only matters at the source level (element pointer arithmetic
            // lives in the library); the machine shape is always {ptr, i64}.
            Type::Slice(_) => self
                .ctx
                .struct_type(
                    &[
                        self.ctx.ptr_type(AddressSpace::default()).into(),
                        self.ctx.i64_type().into(),
                    ],
                    false,
                )
                .into(),
            Type::Vec(elem, n) => match self.basic_ty(elem) {
                BasicTypeEnum::IntType(t) => t.vec_type(*n).into(),
                BasicTypeEnum::FloatType(t) => t.vec_type(*n).into(),
                other => panic!("vec element must be a scalar int/float, got {other:?}"),
            },
            Type::Fn(..) => self.ctx.ptr_type(AddressSpace::default()).into(),
            Type::Ref(..) => self.ctx.ptr_type(AddressSpace::default()).into(),
            // `void` has no value representation — it appears only as a return type
            // (handled in `fn_type_with_ret`/`c_signature`/the return/call paths). The
            // CHECKER is the primary gate: it must reject void in every value/param/
            // field/type-arg position before codegen. This is the DEFENSIVE BACKSTOP:
            // two reviews each found a checker hole that let void slip through here
            // (variadic→silent-i64-0; type-arg→panic), so the invariant is fragile.
            // A clear, loud diagnostic (never a silent-wrong placeholder) means any
            // FUTURE void-path that slips the checker is immediately legible here.
            Type::Void => panic!(
                "codegen: void reached `basic_ty` as a value type — this is a CHECKER \
                 HOLE: void is return-position-only and must be rejected before \
                 codegen (every value/param/field/type-argument position). Please \
                 report the source that produced it."
            ),
            Type::App(..) => unreachable!("generic type survived monomorphization"),
        }
    }

    fn fn_type(&self, params: &[Param], ret: &Type) -> FunctionType<'ctx> {
        let types: Vec<Type> = params.iter().map(|p| p.ty.clone()).collect();
        self.fn_type_types(&types, ret)
    }

    fn fn_type_types(&self, params: &[Type], ret: &Type) -> FunctionType<'ctx> {
        let p: Vec<BasicMetadataTypeEnum> = params.iter().map(|t| self.basic_ty(t).into()).collect();
        self.fn_type_with_ret(ret, &p, false)
    }

    /// Build a function type, lowering a `void` return to the LLVM `void` type
    /// (which `basic_ty` deliberately can't produce — void isn't a value type).
    fn fn_type_with_ret(
        &self,
        ret: &Type,
        params: &[BasicMetadataTypeEnum<'ctx>],
        variadic: bool,
    ) -> FunctionType<'ctx> {
        if matches!(ret, Type::Void) {
            self.ctx.void_type().fn_type(params, variadic)
        } else {
            self.basic_ty(ret).fn_type(params, variadic)
        }
    }

    /// If `t` is a real (C-layout) struct passed by value across the C boundary —
    /// i.e. a `defstruct` (not a sum, not a `:layout bits` struct, which is just
    /// an integer) — return its field list and LLVM struct type, so it can be
    /// ABI-classified. Bits structs and sums return `None` (they're already a
    /// scalar / handled separately).
    fn abi_struct(&self, t: &Type) -> Option<(Vec<(String, Type)>, StructType<'ctx>)> {
        if let Type::Struct(name) = t {
            if let Some(info) = self.structs.get(name) {
                if !matches!(info.layout, Layout::Bits(_)) {
                    return Some((info.fields.clone(), info.ty));
                }
            }
        }
        None
    }

    /// Classify a struct for the current target's C ABI.
    fn classify_struct(
        &self,
        fields: &[(String, Type)],
        sty: StructType<'ctx>,
    ) -> Result<StructAbi<'ctx>, String> {
        let field_llvm = |t: &Type| self.basic_ty(t);
        let resolve = |name: &str| -> Option<Vec<(String, Type)>> {
            self.structs.get(name).map(|i| i.fields.clone())
        };
        let abi_ctx = AbiCtx {
            field_llvm: &field_llvm,
            resolve_struct: &resolve,
        };
        abi::classify(self.arch.to_abi(), self.ctx, &self.target_data, fields, sty, &abi_ctx)
    }

    /// Build the C-ABI realization of a signature: the LLVM function type plus how
    /// the return value and each parameter are marshalled. Only `c`/native
    /// conventions use the C struct ABI; any other convention with a by-value
    /// struct parameter/return is a hard error (its lowering doesn't define one).
    fn c_signature(
        &self,
        name: &str,
        params: &[Type],
        ret: &Type,
        variadic: bool,
        native_c: bool,
    ) -> Result<CSig<'ctx>, String> {
        let mut llvm_params: Vec<BasicMetadataTypeEnum<'ctx>> = Vec::new();
        let mut args: Vec<ArgAbi<'ctx>> = Vec::with_capacity(params.len());

        // Return classification (a hidden `sret` pointer is the first parameter).
        // `None` LLVM return type means void (an `sret` return).
        let mut sret: Option<StructAbi<'ctx>> = None;
        let mut ret_direct: Option<StructAbi<'ctx>> = None;
        let mut llvm_ret: Option<BasicTypeEnum<'ctx>> = None;
        if let Some((fields, sty)) = self.abi_struct(ret) {
            if !native_c {
                return Err(format!(
                    "codegen: '{name}' returns struct by value but its convention is not the C \
                     convention; only the C ABI defines by-value struct passing"
                ));
            }
            let sa = self.classify_struct(&fields, sty)?;
            if sa.ret.is_indirect() {
                // sret: void return, hidden pointer first arg.
                llvm_params.push(self.ctx.ptr_type(AddressSpace::default()).into());
                sret = Some(sa);
            } else {
                let rt = sa
                    .direct_return_type(self.ctx)
                    .ok_or_else(|| format!("codegen: '{name}': empty direct return classification"))?;
                llvm_ret = Some(rt);
                ret_direct = Some(sa);
            }
        } else if matches!(ret, Type::Void) {
            // A `(-> void)` function that ALSO needs the C ABI (e.g. a by-value
            // struct parameter): leave `llvm_ret = None` → an LLVM void return,
            // exactly like the sret case. (`basic_ty` can't lower void.)
            llvm_ret = None;
        } else {
            llvm_ret = Some(self.basic_ty(ret));
        }

        for (i, pty) in params.iter().enumerate() {
            if let Some((fields, sty)) = self.abi_struct(pty) {
                if !native_c {
                    return Err(format!(
                        "codegen: '{name}' parameter {} is a by-value struct but its convention is \
                         not the C convention; only the C ABI defines by-value struct passing",
                        i + 1
                    ));
                }
                let sa = self.classify_struct(&fields, sty)?;
                match &sa.arg {
                    Class::Direct(slots) => {
                        for s in slots {
                            llvm_params.push((*s).into());
                        }
                        args.push(ArgAbi::Direct(sa));
                    }
                    Class::Indirect { .. } => {
                        llvm_params.push(self.ctx.ptr_type(AddressSpace::default()).into());
                        args.push(ArgAbi::Indirect(sa));
                    }
                }
            } else {
                llvm_params.push(self.basic_ty(pty).into());
                args.push(ArgAbi::Scalar);
            }
        }

        let fn_ty = match llvm_ret {
            Some(rt) => rt.fn_type(&llvm_params, variadic),
            None => self.ctx.void_type().fn_type(&llvm_params, variadic),
        };

        Ok(CSig {
            sret,
            ret_direct,
            args,
            fn_ty,
        })
    }

    /// True if the signature needs ABI struct marshalling (any by-value struct in
    /// a parameter or the return). Scalar-only signatures keep the simple path.
    fn needs_c_abi(&self, params: &[Type], ret: &Type) -> bool {
        self.abi_struct(ret).is_some() || params.iter().any(|t| self.abi_struct(t).is_some())
    }

    /// Reject a by-value slice or SIMD vector crossing the C boundary — as a
    /// direct parameter/return, or nested inside a by-value struct field. These
    /// are Coil view/SIMD types with no defined C representation; without this a
    /// slice would silently cross a C extern as a raw `{ptr,len}`. Runs BEFORE
    /// (and independent of) the per-target struct classification, so it holds on
    /// EVERY target — not just the field-walking x86 SysV path (AArch64
    /// classifies a struct by size and never reaches the field-level check).
    fn check_c_abi_types(&self, who: &str, params: &[Type], ret: &Type) -> Result<(), String> {
        self.reject_view_at_c_abi(ret, who)?;
        for p in params {
            self.reject_view_at_c_abi(p, who)?;
        }
        Ok(())
    }

    fn reject_view_at_c_abi(&self, t: &Type, who: &str) -> Result<(), String> {
        match t {
            Type::Slice(_) => Err(format!(
                "{who}: a slice cannot cross the C ABI by value (it is a Coil view type); \
                 for FFI pass a c\"…\"/(ptr i8) and a length"
            )),
            Type::Vec(..) => Err(format!(
                "{who}: a vec cannot cross the C ABI by value; pass a pointer to it"
            )),
            // A by-value struct (not a `:bits` integer): a slice/vec FIELD is
            // forbidden too — recurse, since size-based classification (AArch64)
            // would otherwise wave it through.
            Type::Struct(name) => {
                if let Some(info) = self.structs.get(name) {
                    if !matches!(info.layout, Layout::Bits(_)) {
                        for (_, ft) in &info.fields {
                            self.reject_view_at_c_abi(ft, who)?;
                        }
                    }
                }
                Ok(())
            }
            Type::Array(e, _) => self.reject_view_at_c_abi(e, who),
            // Pointers (incl. `(ptr (slice T))`) cross fine — only BY-VALUE
            // slices/vecs are rejected.
            _ => Ok(()),
        }
    }

    /// The (location, attribute) pairs a `CSig` requires: `sret(T) align N` on a
    /// hidden return pointer, `byval(T) align N` on x86-64 indirect arguments. The
    /// same list applies to both the function declaration and every call site.
    fn csig_attrs(&self, sig: &CSig<'ctx>) -> Vec<(AttributeLoc, Attribute)> {
        let mut out = Vec::new();
        let mut idx = 0u32;
        if let Some(sa) = &sig.sret {
            let loc = AttributeLoc::Param(idx);
            out.push((loc, self.type_attr("sret", sa.llvm_ty)));
            out.push((loc, self.align_attr(sa.align)));
            idx += 1;
        }
        for a in &sig.args {
            match a {
                ArgAbi::Direct(sa) => {
                    if let Class::Direct(slots) = &sa.arg {
                        idx += slots.len() as u32; // one slot per coerced eightbyte
                    }
                }
                ArgAbi::Indirect(sa) => {
                    if self.arch == TargetArch::X86_64 {
                        let loc = AttributeLoc::Param(idx);
                        out.push((loc, self.type_attr("byval", sa.llvm_ty)));
                        out.push((loc, self.align_attr(sa.align)));
                    }
                    idx += 1;
                }
                ArgAbi::Scalar => idx += 1,
            }
        }
        out
    }

    /// A type attribute such as `byval(T)` or `sret(T)`.
    fn type_attr(&self, kind: &str, ty: StructType<'ctx>) -> Attribute {
        let kind_id = Attribute::get_named_enum_kind_id(kind);
        self.ctx.create_type_attribute(kind_id, ty.as_any_type_enum())
    }

    fn align_attr(&self, align: u32) -> Attribute {
        let kind_id = Attribute::get_named_enum_kind_id("align");
        self.ctx.create_enum_attribute(kind_id, align as u64)
    }

    /// Allocate a scratch stack slot for a struct (for spilling a by-value
    /// argument to memory or receiving an `sret`/coerced return). Aligned to at
    /// least 8 so that whole-eightbyte (i64/double) coerced loads/stores into it
    /// are naturally aligned, and never below the struct's own ABI alignment.
    fn struct_slot(&self, sa: &StructAbi<'ctx>, name: &str) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        let p = self.builder.build_alloca(sa.llvm_ty, name).map_err(le)?;
        if let Some(instr) = p.as_instruction() {
            let _ = instr.set_alignment(sa.align.max(8));
        }
        Ok(p)
    }

    /// Get a pointer to a by-value struct argument's bytes. The checker passes a
    /// place (its value is already a pointer to the struct); a bare struct *value*
    /// is spilled to a fresh stack slot first.
    fn struct_arg_ptr(
        &self,
        tv: &Tv<'ctx>,
        sa: &StructAbi<'ctx>,
    ) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        let (v, _) = tv;
        if v.is_pointer_value() {
            return Ok(v.into_pointer_value());
        }
        // A struct value in SSA: materialize it in memory so we can read the
        // coerced register slots / pass a pointer.
        let slot = self.struct_slot(sa, "abi.arg")?;
        self.builder.build_store(slot, *v).map_err(le)?;
        Ok(slot)
    }

    /// Lower a call to a callable whose signature passes/returns a struct by value
    /// (per its precomputed `CSig`). Marshals each argument into the coerced
    /// register slots / `byval` pointers the C ABI requires, and reconstructs a
    /// struct value from the coerced return / `sret` slot.
    fn emit_c_call(
        &self,
        callee: FunctionValue<'ctx>,
        sig: &CSig<'ctx>,
        argtv: &[Tv<'ctx>],
        ret_ty: Type,
    ) -> Result<Tv<'ctx>, String> {
        let mut meta: Vec<BasicMetadataValueEnum<'ctx>> = Vec::new();

        // sret: allocate the result struct and pass its pointer as the hidden
        // first argument.
        let sret_slot = if let Some(sa) = &sig.sret {
            let slot = self.struct_slot(sa, "abi.sret")?;
            meta.push(slot.into());
            Some((slot, sa))
        } else {
            None
        };

        for (a, tv) in sig.args.iter().zip(argtv) {
            match a {
                ArgAbi::Scalar => meta.push(tv.0.into()),
                ArgAbi::Indirect(sa) => {
                    // Pass a pointer to a fresh copy of the struct (the callee may
                    // mutate its `byval` copy; the C ABI says caller-allocated).
                    let src = self.struct_arg_ptr(tv, sa)?;
                    let copy = self.struct_slot(sa, "abi.byval")?;
                    self.copy_struct(copy, src, sa)?;
                    meta.push(copy.into());
                }
                ArgAbi::Direct(sa) => {
                    let src = self.struct_arg_ptr(tv, sa)?;
                    let slots = match &sa.arg {
                        Class::Direct(s) => s,
                        Class::Indirect { .. } => unreachable!("Direct arg has Direct class"),
                    };
                    // Load each coerced slot from its eightbyte offset.
                    for (i, slot_ty) in slots.iter().enumerate() {
                        let off = (i as u64) * 8;
                        let gep = if off == 0 {
                            src
                        } else {
                            unsafe {
                                self.builder
                                    .build_gep(self.ctx.i8_type(), src, &[self.ctx.i64_type().const_int(off, false)], "abi.eb")
                                    .map_err(le)?
                            }
                        };
                        let v = self.builder.build_load(*slot_ty, gep, "abi.slot").map_err(le)?;
                        meta.push(v.into());
                    }
                }
            }
        }

        // Variadic extra arguments (past the fixed `sig.args`) are passed through
        // as-is. A by-value *struct* in the variadic tail would need extra ABI
        // work we don't do yet, so reject it loudly rather than miscompile.
        for tv in argtv.iter().skip(sig.args.len()) {
            if self.abi_struct(&tv.1).is_some() && !tv.0.is_pointer_value() {
                return Err(
                    "codegen: passing a struct by value in a variadic argument is not supported"
                        .to_string(),
                );
            }
            meta.push(tv.0.into());
        }

        let cs = self.builder.build_call(callee, &meta, "ccall").map_err(le)?;
        cs.set_call_convention(callee.get_call_conventions());
        for (loc, attr) in self.csig_attrs(sig) {
            cs.add_attribute(loc, attr);
        }

        // Reconstruct the returned struct value.
        if let Some((slot, sa)) = sret_slot {
            let v = self.builder.build_load(sa.llvm_ty, slot, "abi.ret").map_err(le)?;
            return Ok((v, ret_ty));
        }
        if let Some(sa) = &sig.ret_direct {
            // The call returned the coerced type; store it into a struct slot and
            // reload the struct value.
            let coerced = cs
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| "codegen: C-ABI direct return produced void".to_string())?;
            let slot = self.struct_slot(sa, "abi.ret")?;
            self.builder.build_store(slot, coerced).map_err(le)?;
            let v = self.builder.build_load(sa.llvm_ty, slot, "abi.ret.val").map_err(le)?;
            return Ok((v, ret_ty));
        }
        // A `(-> void)` function that needed the C ABI (e.g. it only *takes* a
        // struct by value): no value to read; the checker forbids using the result.
        if matches!(ret_ty, Type::Void) {
            return Ok((self.ctx.i64_type().const_zero().into(), Type::Void));
        }
        // A scalar return (e.g. a function that only *takes* a struct by value).
        let v = cs
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| "codegen: C-ABI call returned void".to_string())?;
        Ok((v, ret_ty))
    }

    /// `memcpy` `size` bytes from `src` to `dst` (both struct pointers).
    fn copy_struct(
        &self,
        dst: inkwell::values::PointerValue<'ctx>,
        src: inkwell::values::PointerValue<'ctx>,
        sa: &StructAbi<'ctx>,
    ) -> Result<(), String> {
        self.builder
            .build_memcpy(dst, sa.align, src, sa.align, self.ctx.i64_type().const_int(sa.size, false))
            .map_err(|e| le(format!("{e:?}")))?;
        Ok(())
    }

    fn emit_func(&self, f: &Func, function: FunctionValue<'ctx>) -> Result<(), String> {
        let entry = self.ctx.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // DWARF: attach a DISubprogram (function-granularity line info) and set the
        // builder's debug location so every instruction in the body carries `!dbg`
        // — required once a function has a subprogram. A function from an
        // included/imported file (dummy span) gets none, never a wrong line.
        if let Some(di) = &self.di {
            let (line, col) = di.line_col(f.span.lo);
            if line != 0 {
                let sp = di.builder.create_function(
                    di.file.as_debug_info_scope(),
                    &f.name,
                    Some(&f.name), // linkage name = the symbol
                    di.file,
                    line,
                    di.subroutine_ty,
                    false, // not local to unit
                    true,  // is a definition
                    line,  // scope line
                    inkwell::debug_info::DIFlags::ZERO,
                    false, // not optimized (the -g pipeline is ~-O0; see the CU)
                );
                function.set_subprogram(sp);
                // Keep user functions out of the inliner so their breakpoints
                // reliably resolve and backtraces stay legible; `alwaysinline`
                // `(llvm-ir …)` helpers are driven by a different pass and still
                // inline, so zero-overhead intrinsics are unaffected.
                let noinline =
                    self.ctx.create_enum_attribute(Attribute::get_named_enum_kind_id("noinline"), 0);
                function.add_attribute(AttributeLoc::Function, noinline);
                let loc = di.builder.create_debug_location(
                    self.ctx,
                    line,
                    col,
                    sp.as_debug_info_scope(),
                    None,
                );
                self.builder.set_current_debug_location(loc);
                // Make the subprogram the active scope so `emit_expr` can set a
                // per-statement location for each body expression (line tables).
                *self.cur_sp.borrow_mut() = Some(sp);
            }
        }

        // A C-ABI'd function (one returning a struct by value, or — in principle —
        // taking one) has its signature reshaped: a leading `sret` pointer and/or
        // struct parameters split into coerced register slots. Map the user's
        // parameters onto the right LLVM parameter indices.
        let sig = self.csigs.get(&f.name);
        let base = sig.and_then(|s| s.sret.as_ref()).map_or(0u32, |_| 1);

        let mut scope: HashMap<String, Tv<'ctx>> = HashMap::new();
        let mut idx = base;
        for (pos, p) in f.params.iter().enumerate() {
            // Struct-by-value parameters are reconstructed from their coerced
            // slots into a stack copy whose pointer the body uses (the checker
            // erases struct params to pointers, so this matches `(ptr S)`).
            let abi = sig.and_then(|s| s.args.get(pos));
            match abi {
                Some(ArgAbi::Direct(sa)) => {
                    let slot = self.struct_slot(sa, &format!("{}.arg", p.name))?;
                    if let Class::Direct(slots) = &sa.arg {
                        for (k, slot_ty) in slots.iter().enumerate() {
                            let v = function
                                .get_nth_param(idx)
                                .ok_or("codegen: missing coerced struct param")?;
                            let off = (k as u64) * 8;
                            let gep = if off == 0 {
                                slot
                            } else {
                                unsafe {
                                    self.builder
                                        .build_gep(self.ctx.i8_type(), slot, &[self.ctx.i64_type().const_int(off, false)], "abi.eb")
                                        .map_err(le)?
                                }
                            };
                            let _ = slot_ty;
                            self.builder.build_store(gep, v).map_err(le)?;
                            idx += 1;
                        }
                    }
                    // A by-value struct param reaches this path ONLY for a
                    // generic function instantiated with a struct type-arg (a
                    // non-generic struct param is by-reference per the reference
                    // model, so its `p.ty` is already a pointer and it takes the
                    // scalar arm below). The generic body was elaborated treating
                    // the param as a VALUE of type `p.ty`, so bind the loaded
                    // struct value — not the reconstruction pointer (binding the
                    // pointer silently corrupts every use of the param).
                    let val = self.builder.build_load(sa.llvm_ty, slot, &p.name).map_err(le)?;
                    scope.insert(p.name.clone(), (val, p.ty.clone()));
                }
                Some(ArgAbi::Indirect(sa)) => {
                    // Same as Direct: the `byval` pointer is the caller's copy;
                    // the generic body wants the value, so load it.
                    let ptr = function
                        .get_nth_param(idx)
                        .ok_or("codegen: missing param")?
                        .into_pointer_value();
                    let val = self.builder.build_load(sa.llvm_ty, ptr, &p.name).map_err(le)?;
                    scope.insert(p.name.clone(), (val, p.ty.clone()));
                    idx += 1;
                }
                _ => {
                    let v = function.get_nth_param(idx).ok_or("codegen: missing param")?;
                    v.set_name(&p.name);
                    scope.insert(p.name.clone(), (v, p.ty.clone()));
                    idx += 1;
                }
            }
        }

        // Parameter debug info (-g): describe each scalar/pointer arg so the
        // debugger can show them. Done after binding (values are in `scope`) and
        // before the body, so the dbg.declares sit in the entry block.
        self.emit_param_debug(f, entry, &scope);

        let mut last: Tv = (self.ctx.i64_type().const_zero().into(), Type::Int(64, true));
        for e in &f.body {
            // A diverging statement (an unbroken loop, or a `break`/`continue`
            // in a tail position) terminates the block; the rest is dead.
            if self.block_terminated() {
                break;
            }
            last = self.emit_expr(e, &scope)?;
        }

        // If the body diverged, the block already has a terminator — no `ret`.
        if !self.block_terminated() {
            self.emit_c_return(function, sig, &f.ret, last)?;
        }
        // Clear the debug location so the next function (which may have none) does
        // not inherit a scope from this one (a verifier error).
        if self.di.is_some() {
            self.builder.unset_current_debug_location();
            *self.cur_sp.borrow_mut() = None;
        }
        Ok(())
    }

    /// Emit the function's return, applying the C-ABI return lowering when the
    /// function returns a struct by value: store the struct value through the
    /// hidden `sret` pointer (large), or coerce it to the register return type
    /// (small). A scalar return is returned directly.
    fn emit_c_return(
        &self,
        function: FunctionValue<'ctx>,
        sig: Option<&CSig<'ctx>>,
        ret: &Type,
        last: Tv<'ctx>,
    ) -> Result<(), String> {
        // A `(-> void)` function: the body ran for effect; discard its last value
        // and return void. (Checked on the COIL return type — an `sret` struct
        // return is ALSO an LLVM void function, but must NOT take this path.)
        if matches!(ret, Type::Void) {
            self.builder.build_return(None).map_err(le)?;
            return Ok(());
        }
        let sig = match sig {
            Some(s) if s.sret.is_some() || s.ret_direct.is_some() => s,
            _ => {
                self.builder.build_return(Some(&last.0)).map_err(le)?;
                return Ok(());
            }
        };
        // Get a pointer to the struct value being returned.
        let sa = sig.sret.as_ref().or(sig.ret_direct.as_ref()).unwrap();
        let src = if last.0.is_pointer_value() {
            last.0.into_pointer_value()
        } else {
            let slot = self.struct_slot(sa, "ret.spill")?;
            self.builder.build_store(slot, last.0).map_err(le)?;
            slot
        };
        if let Some(sa) = &sig.sret {
            // Copy the struct into the caller-provided sret slot, return void.
            let dst = function
                .get_nth_param(0)
                .ok_or("codegen: missing sret param")?
                .into_pointer_value();
            self.copy_struct(dst, src, sa)?;
            self.builder.build_return(None).map_err(le)?;
            return Ok(());
        }
        // Direct: load the coerced return type from the struct memory.
        let sa = sig.ret_direct.as_ref().unwrap();
        let rt = sa
            .direct_return_type(self.ctx)
            .ok_or("codegen: empty direct return type")?;
        let v = self.builder.build_load(rt, src, "abi.ret.coerce").map_err(le)?;
        self.builder.build_return(Some(&v)).map_err(le)?;
        Ok(())
    }

    fn emit_trampoline(
        &self,
        tramp: FunctionValue<'ctx>,
        shim: &ShimInfo<'ctx>,
        n: usize,
    ) -> Result<(), String> {
        self.check_shim_regs(shim, n)?;
        let entry = self.ctx.append_basic_block(tramp, "entry");
        self.builder.position_at_end(entry);

        let impl_name = shim.impl_fn.get_name().to_str().map_err(|_| "bad impl name")?;
        let impl_name = self.asm_sym(impl_name);
        let impl_name = impl_name.as_str();
        let (asm, dialect) = match self.arch {
            TargetArch::X86_64 => (self.x86_trampoline(shim, n, impl_name), InlineAsmDialect::ATT),
            // On AArch64 the ATT/Intel distinction is x86-only; we use ATT so
            // LLVM does not prepend an x86 `.intel_syntax` directive (which the
            // AArch64 assembler rejects) and writes the asm verbatim.
            TargetArch::AArch64 => {
                (self.aarch64_trampoline(shim, n, impl_name), InlineAsmDialect::ATT)
            }
        };

        let void_ty = self.ctx.void_type().fn_type(&[], false);
        let asm_ptr = self.ctx.create_inline_asm(
            void_ty,
            asm,
            "~{memory}".to_string(),
            true,
            false,
            Some(dialect),
            false,
        );
        self.builder
            .build_indirect_call(void_ty, asm_ptr, &[], "")
            .map_err(le)?;
        self.builder.build_unreachable().map_err(le)?;
        Ok(())
    }

    /// x86-64 trampoline (AT&T): move the convention's arg registers into the
    /// SysV arg registers, realign the stack to 16 across the call (the naked
    /// entry has rsp%16==8 because of the return address), call the `ccc` impl,
    /// then move rax into the convention's return register.
    fn x86_trampoline(&self, shim: &ShimInfo<'ctx>, n: usize, impl_name: &str) -> String {
        let c_args = self.arch.c_arg_regs();
        let mut asm = String::new();
        for i in 0..n {
            asm += &format!("movq %{}, %{}\n", shim.param_regs[i], c_args[i]);
        }
        asm += "subq $$8, %rsp\n";
        asm += &format!("call {}\n", impl_name);
        asm += "addq $$8, %rsp\n";
        if shim.ret_reg != self.arch.c_ret_reg() {
            asm += &format!("movq %rax, %{}\n", shim.ret_reg);
        }
        asm += "ret\n";
        asm
    }

    /// AArch64 trampoline (AAPCS64): the naked entry holds the return address in
    /// LR (x30) and SP is 16-aligned. Save LR (and keep SP 16-aligned) on the
    /// stack, marshal the convention's arg registers into x0-x7, `bl` the `ccc`
    /// impl, move x0 into the convention's return register, restore LR, return.
    fn aarch64_trampoline(&self, shim: &ShimInfo<'ctx>, n: usize, impl_name: &str) -> String {
        let c_args = self.arch.c_arg_regs();
        let mut asm = String::new();
        // Prologue: push LR. `str lr, [sp, #-16]!` keeps SP 16-aligned (we only
        // need 8 bytes for LR, but AAPCS64 requires 16-byte SP alignment).
        asm += "str x30, [sp, #-16]!\n";
        for i in 0..n {
            // `mov dst, src`; a no-op self-move is harmless.
            asm += &format!("mov {}, {}\n", c_args[i], shim.param_regs[i]);
        }
        asm += &format!("bl {}\n", impl_name);
        if shim.ret_reg != self.arch.c_ret_reg() {
            asm += &format!("mov {}, x0\n", shim.ret_reg);
        }
        // Epilogue: pop LR, return.
        asm += "ldr x30, [sp], #16\n";
        asm += "ret\n";
        asm
    }

    /// The name to use when referencing a global symbol from inline asm. On
    /// Mach-O (Apple) the linker symbol carries a leading `_`; on ELF it does not.
    fn asm_sym(&self, name: &str) -> String {
        if self.mach_o {
            format!("_{name}")
        } else {
            name.to_string()
        }
    }

    /// Reject a shim convention whose registers don't exist on the target arch
    /// (e.g. an x86 `defcc` reaching codegen on aarch64). Silently ignoring the
    /// names would emit asm the assembler rejects or, worse, miscompile.
    fn check_shim_regs(&self, shim: &ShimInfo<'ctx>, n: usize) -> Result<(), String> {
        let check = |reg: &str, what: &str| -> Result<(), String> {
            if self.arch.is_valid_gpr(reg) {
                Ok(())
            } else {
                Err(format!(
                    "codegen: shim convention names {what} register '{reg}', which is not a \
                     general-purpose register on the target architecture ({}). Declare the \
                     convention per-arch (see examples/per-arch.coil).",
                    self.arch.arch_name()
                ))
            }
        };
        for reg in &shim.param_regs[..n] {
            check(reg, "parameter")?;
        }
        check(&shim.ret_reg, "return")?;
        for reg in &shim.clobber {
            check(reg, "clobber")?;
        }
        Ok(())
    }

    fn emit_shim_call(
        &self,
        name: &str,
        shim: &ShimInfo<'ctx>,
        args: &[BasicValueEnum<'ctx>],
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let n = args.len();
        self.check_shim_regs(shim, n)?;
        let mut cons = format!("={{{}}}", shim.ret_reg);
        for r in &shim.param_regs[..n] {
            cons += &format!(",{{{}}}", r);
        }
        let mut used: HashSet<&str> = shim.param_regs[..n].iter().map(String::as_str).collect();
        used.insert(shim.ret_reg.as_str());
        for c in &shim.clobber {
            if !used.contains(c.as_str()) {
                cons += &format!(",~{{{}}}", c);
            }
        }
        // The call clobbers the link register on AArch64 (`bl` writes x30/lr).
        // Spell it `lr` (not `x30`): LLVM's clobber parser treats `~{lr}` as the
        // link register and forces the caller to save/restore it (making a leaf
        // caller non-leaf); `~{x30}` does not trigger that frame setup, so the
        // caller's own return address would be lost.
        if self.arch == TargetArch::AArch64 && !used.contains("lr") && !used.contains("x30") {
            cons += ",~{lr}";
        }
        cons += ",~{memory}";

        // The convention's trampoline is reached with an ordinary direct call;
        // the instruction mnemonic and dialect differ per arch.
        let sym = self.asm_sym(name);
        let (body, dialect) = match self.arch {
            TargetArch::X86_64 => (format!("call {}", sym), InlineAsmDialect::ATT),
            // ATT (not Intel) on AArch64: the dialect is x86-only and Intel would
            // prepend a `.intel_syntax` directive the AArch64 assembler rejects.
            TargetArch::AArch64 => (format!("bl {}", sym), InlineAsmDialect::ATT),
        };

        let arg_types: Vec<BasicMetadataTypeEnum> = args.iter().map(|v| v.get_type().into()).collect();
        let fn_ty = self.ctx.i64_type().fn_type(&arg_types, false);
        let asm_ptr = self.ctx.create_inline_asm(
            fn_ty,
            body,
            cons,
            true,
            true,
            Some(dialect),
            false,
        );
        let argvals: Vec<_> = args.iter().map(|v| (*v).into()).collect();
        let cs = self
            .builder
            .build_indirect_call(fn_ty, asm_ptr, &argvals, "shimcall")
            .map_err(le)?;
        cs.try_as_basic_value()
            .basic()
            .ok_or_else(|| "codegen: shim call returned void".to_string())
    }

    /// Lower a `(llvm-ir RESULT [ops…] "BODY")` form. Build an `alwaysinline`
    /// helper function from the raw IR (filling `$ret`/`$tN`/`$N` placeholders
    /// and hoisting `declare` lines to module scope), parse + link it into the
    /// module, and emit a call. The `-O3` pipeline inlines the helper away, so
    /// it is zero-overhead; the LLVM verifier validates the body.
    fn emit_llvm_ir(
        &self,
        result: &Type,
        args: &[Tv<'ctx>],
        body: &str,
    ) -> Result<Tv<'ctx>, String> {
        let ret_str = self.basic_ty(result).print_to_string().to_string();
        let arg_strs: Vec<String> = args
            .iter()
            .map(|(_, t)| self.basic_ty(t).print_to_string().to_string())
            .collect();

        // Substitute placeholders. Highest index first so `$t1` doesn't eat into
        // `$t10`; `$tN` (types) before `$N` (operand SSA names) before `$ret`.
        let mut text = body.to_string();
        for i in (0..args.len()).rev() {
            text = text.replace(&format!("$t{i}"), &arg_strs[i]);
            text = text.replace(&format!("${i}"), &format!("%{i}"));
        }
        text = text.replace("$ret", &ret_str);

        // Hoist top-level `declare` lines (e.g. intrinsics) out of the body.
        let mut decls = String::new();
        let mut fnbody = String::new();
        for line in text.lines() {
            if line.trim_start().starts_with("declare") {
                decls.push_str(line);
                decls.push('\n');
            } else {
                fnbody.push_str(line);
                fnbody.push('\n');
            }
        }

        let seq = self.llvm_seq.get();
        self.llvm_seq.set(seq + 1);
        let name = format!("__coil_llvm_ir_{seq}");
        let params: Vec<String> =
            arg_strs.iter().enumerate().map(|(i, t)| format!("{t} %{i}")).collect();
        let module_text = format!(
            "{decls}define {ret_str} @{name}({}) alwaysinline {{\n{fnbody}}}\n",
            params.join(", ")
        );

        // inkwell ≥0.9 *asserts* the input slice ends in a NUL (both the copy and
        // non-copy `create_from_memory_range*` decrement len past it), so hand it
        // an explicitly NUL-terminated copy of the IR text.
        let mut ir_bytes = module_text.clone().into_bytes();
        ir_bytes.push(0);
        let buf = MemoryBuffer::create_from_memory_range_copy(&ir_bytes, "coil_llvm_ir");
        let helper = self.ctx.create_module_from_ir(buf).map_err(|e| {
            format!("llvm-ir: could not parse IR: {e}\n--- generated module ---\n{module_text}")
        })?;
        helper.set_triple(&self.module.get_triple());
        let dl = self.module.get_data_layout();
        helper.set_data_layout(&dl);
        self.module
            .link_in_module(helper)
            .map_err(|e| format!("llvm-ir: link failed: {e}"))?;

        let callee = self
            .module
            .get_function(&name)
            .ok_or("llvm-ir: helper function vanished after link")?;
        // Emitted with external linkage so it survives `link_in_module` (an
        // internal helper with no uses yet would be dropped at link time). Now
        // flip it to internal so, once its `alwaysinline` body is inlined at the
        // call below, `-O3`/globaldce deletes the helper entirely — zero cost.
        callee.set_linkage(inkwell::module::Linkage::Internal);
        let meta: Vec<BasicMetadataValueEnum> = args.iter().map(|(v, _)| (*v).into()).collect();
        let cs = self.builder.build_call(callee, &meta, "llvmir").map_err(le)?;
        let v = cs
            .try_as_basic_value()
            .basic()
            .ok_or("llvm-ir: helper returned void (the body must `ret` a value)")?;
        Ok((v, result.clone()))
    }

    /// Set the builder's current debug location from a source span, scoped to the
    /// function being emitted. A no-op without `-g` or for a dummy/synthesized
    /// span (which keeps the enclosing statement's location — never a wrong one).
    fn set_dbg_loc(&self, span: crate::span::Span) {
        if let Some(di) = &self.di {
            if let Some(sp) = *self.cur_sp.borrow() {
                let (line, col) = di.line_col(span.lo);
                if line != 0 {
                    let loc = di
                        .builder
                        .create_debug_location(self.ctx, line, col, sp.as_debug_info_scope(), None);
                    self.builder.set_current_debug_location(loc);
                }
            }
        }
    }

    /// A DWARF type for a Coil type, for local-variable debug info. Scalars and
    /// pointers map to DWARF basic/pointer types; aggregates (struct/sum/slice/
    /// array/fnptr) return `None` for now — those locals are simply not described
    /// (no wrong type, just no `frame variable` entry).
    fn di_type(&self, di: &DebugCtx<'ctx>, t: &Type) -> Option<inkwell::debug_info::DIType<'ctx>> {
        use inkwell::debug_info::DIFlags;
        // DW_ATE encodings: boolean=0x02, float=0x04, signed=0x05, unsigned=0x07.
        let basic = |name: String, bits: u64, enc: u32| {
            di.builder.create_basic_type(&name, bits, enc, DIFlags::ZERO).ok().map(|b| b.as_type())
        };
        match t {
            Type::Int(b, signed) => basic(format!("i{b}"), *b as u64, if *signed { 0x05 } else { 0x07 }),
            Type::Float(32) => basic("f32".into(), 32, 0x04),
            Type::Float(_) => basic("f64".into(), 64, 0x04),
            Type::Bool => basic("bool".into(), 8, 0x02),
            Type::Ptr(_) => {
                let byte = basic("i8".into(), 8, 0x05)?;
                Some(
                    di.builder
                        .create_pointer_type("ptr", byte, 64, 0, AddressSpace::default())
                        .as_type(),
                )
            }
            _ => None,
        }
    }

    /// Describe a function's scalar parameters for the debugger: spill each into a
    /// `-g`-only alloca (the minimal `-g` pipeline keeps it in memory) and attach
    /// a `DILocalVariable` via `llvm.dbg.declare`, so `frame variable` / `p x`
    /// show the arguments. Aggregates (no `di_type`) are skipped.
    fn emit_param_debug(&self, f: &Func, entry: BasicBlock<'ctx>, scope: &HashMap<String, Tv<'ctx>>) {
        let (di, sp) = match (&self.di, *self.cur_sp.borrow()) {
            (Some(di), Some(sp)) => (di, sp),
            _ => return,
        };
        let (line, _) = di.line_col(f.span.lo);
        if line == 0 {
            return;
        }
        let loc = di.builder.create_debug_location(self.ctx, line, 1, sp.as_debug_info_scope(), None);
        // The spill stores are prologue, not a steppable statement: emit them with
        // NO line so LLVM puts `prologue_end` on the first *body* instruction. A
        // function breakpoint then lands after the stores (params already set),
        // instead of before them (showing stale stack bytes).
        self.builder.unset_current_debug_location();
        for (i, p) in f.params.iter().enumerate() {
            let Some((val, ty)) = scope.get(&p.name) else { continue };
            let Some(dty) = self.di_type(di, ty) else { continue };
            let Ok(slot) = self.builder.build_alloca(val.get_type(), &format!("{}.dbg", p.name)) else {
                continue;
            };
            let _ = self.builder.build_store(slot, *val);
            let var = di.builder.create_parameter_variable(
                sp.as_debug_info_scope(),
                &p.name,
                i as u32 + 1,
                di.file,
                line,
                dty,
                true,
                inkwell::debug_info::DIFlags::ZERO,
            );
            self.dbg_declare_at_end(di, slot, var, loc, entry);
        }
    }

    /// Allocate a slot in the *entry* block (regardless of the current insert
    /// point), so a debug spill for a `let` inside a loop doesn't grow the stack
    /// each iteration. The builder is restored to where it was.
    fn entry_alloca(
        &self,
        ty: BasicTypeEnum<'ctx>,
        name: &str,
    ) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        let cur = self.builder.get_insert_block().ok_or("codegen: no insert block")?;
        let func = cur.get_parent().ok_or("codegen: block has no parent function")?;
        let entry = func.get_first_basic_block().ok_or("codegen: function has no entry block")?;
        // The alloca is prologue, not a statement: emit it with NO line so it does
        // not pollute the line table (a body-line alloca at the front of `entry`
        // would mis-place line breakpoints). Save/restore the caller's location.
        let saved = self.builder.get_current_debug_location();
        match entry.get_first_instruction() {
            Some(first) => self.builder.position_before(&first),
            None => self.builder.position_at_end(entry),
        }
        self.builder.unset_current_debug_location();
        let slot = self.builder.build_alloca(ty, name).map_err(le)?;
        self.builder.position_at_end(cur);
        match saved {
            Some(loc) => self.builder.set_current_debug_location(loc),
            None => self.builder.unset_current_debug_location(),
        }
        Ok(slot)
    }

    /// Insert an `llvm.dbg.declare` for `var` at `slot`, at the end of `block`.
    /// Calls the LLVM-C API directly: inkwell 0.9's `insert_declare_at_end` wraps
    /// the LLVM-19+ *DbgRecord* return as an `InstructionValue` and trips a
    /// `debug_assert!(is_instruction())` (intermittently, a hard panic in dev/test
    /// builds). We don't use the return, so bypass the wrapper.
    fn dbg_declare_at_end(
        &self,
        di: &DebugCtx<'ctx>,
        slot: inkwell::values::PointerValue<'ctx>,
        var: inkwell::debug_info::DILocalVariable<'ctx>,
        loc: inkwell::debug_info::DILocation<'ctx>,
        block: BasicBlock<'ctx>,
    ) {
        use inkwell::values::AsValueRef;
        let empty_expr = di.builder.create_expression(vec![]);
        unsafe {
            inkwell::llvm_sys::debuginfo::LLVMDIBuilderInsertDeclareRecordAtEnd(
                di.builder.as_mut_ptr(),
                slot.as_value_ref(),
                var.as_mut_ptr(),
                empty_expr.as_mut_ptr(),
                loc.as_mut_ptr(),
                block.as_mut_ptr(),
            );
        }
    }

    /// Describe a `let` binding for the debugger (`-g`): spill its value to an
    /// entry-block debug slot and attach a `DILocalVariable` (auto variable), so
    /// `frame variable` shows it. Scalars/pointers only (aggregates skipped).
    fn declare_let_local(&self, name: &str, val: BasicValueEnum<'ctx>, ty: &Type, span: crate::span::Span) {
        let (di, sp) = match (&self.di, *self.cur_sp.borrow()) {
            (Some(di), Some(sp)) => (di, sp),
            _ => return,
        };
        let Some(dty) = self.di_type(di, ty) else { return };
        let (line, _) = di.line_col(span.lo);
        if line == 0 {
            return;
        }
        let Ok(slot) = self.entry_alloca(val.get_type(), &format!("{name}.dbg")) else { return };
        let _ = self.builder.build_store(slot, val);
        let var = di.builder.create_auto_variable(
            sp.as_debug_info_scope(),
            name,
            di.file,
            line,
            dty,
            true,
            inkwell::debug_info::DIFlags::ZERO,
            0,
        );
        let loc = di.builder.create_debug_location(self.ctx, line, 1, sp.as_debug_info_scope(), None);
        if let Some(block) = self.builder.get_insert_block() {
            self.dbg_declare_at_end(di, slot, var, loc, block);
        }
    }

    fn emit_expr(&self, e: &Expr, scope: &HashMap<String, Tv<'ctx>>) -> Result<Tv<'ctx>, String> {
        // Per-statement line info: map the instructions this expression emits to
        // its source line (drives line-by-line stepping in lldb/gdb).
        self.set_dbg_loc(e.span);
        let i64t = self.ctx.i64_type();
        match &e.kind {
            ExprKind::Int(n) => Ok((i64t.const_int(*n as u64, true).into(), Type::Int(64, true))),
            ExprKind::Float(x) => Ok((self.ctx.f64_type().const_float(*x).into(), Type::Float(64))),
            ExprKind::Bool(b) => Ok((self.ctx.bool_type().const_int(*b as u64, false).into(), Type::Bool)),
            ExprKind::LlvmIr { result, args, body } => {
                let argtv: Vec<Tv<'ctx>> = args
                    .iter()
                    .map(|a| self.emit_expr(a, scope))
                    .collect::<Result<_, _>>()?;
                self.emit_llvm_ir(result, &argtv, body)
            }
            // The zero value of a type (used to initialize a fresh local).
            ExprKind::Zeroed(t) => Ok((self.basic_ty(t).const_zero(), t.clone())),
            // A borrow is the underlying place's pointer (the checker normally
            // erases these, but a place used directly lowers to its pointer).
            ExprKind::Borrow { place, .. } => self.emit_expr(place, scope),
            // Spill an rvalue to a fresh stack slot and yield a pointer to it —
            // the same `alloca` + `store` the match scrutinee uses — so a
            // temporary can be passed to a by-immutable-reference parameter.
            ExprKind::SpillRef(inner) => {
                let (v, t) = self.emit_expr(inner, scope)?;
                let slot = self.builder.build_alloca(self.basic_ty(&t), "spill").map_err(le)?;
                self.builder.build_store(slot, v).map_err(le)?;
                Ok((slot.into(), Type::Ptr(Box::new(t))))
            }
            ExprKind::Str(s) => {
                // "…" is a (slice u8) VIEW: a private [N x i8] global (no NUL —
                // the length carries the extent) plus a {ptr, len} fat-pointer
                // constant. No allocation, no copy.
                let bytes = self.ctx.const_string(s.as_bytes(), false);
                let n = self.globals.get();
                self.globals.set(n + 1);
                let g = self.module.add_global(bytes.get_type(), None, &format!("str.{n}"));
                g.set_initializer(&bytes);
                g.set_constant(true);
                g.set_linkage(inkwell::module::Linkage::Private);
                let len = self.ctx.i64_type().const_int(s.len() as u64, false);
                let slice = self
                    .ctx
                    .const_struct(&[g.as_pointer_value().into(), len.into()], false);
                Ok((slice.into(), Type::Slice(Box::new(Type::Int(8, false)))))
            }
            ExprKind::CStr(s) => {
                // c"…" is a (ptr i8) to a private, NUL-terminated [N+1 x i8]
                // global — the distinct FFI spelling.
                let bytes = self.ctx.const_string(s.as_bytes(), true);
                let n = self.globals.get();
                self.globals.set(n + 1);
                let g = self.module.add_global(bytes.get_type(), None, &format!("cstr.{n}"));
                g.set_initializer(&bytes);
                g.set_constant(true);
                g.set_linkage(inkwell::module::Linkage::Private);
                Ok((g.as_pointer_value().into(), Type::Ptr(Box::new(Type::Int(8, true)))))
            }
            ExprKind::Var(name) => scope
                .get(name)
                .cloned()
                .ok_or_else(|| format!("codegen: unbound '{name}'")),
            ExprKind::Bin { op, lhs, rhs } => {
                let (lv, lt) = self.emit_expr(lhs, scope)?;
                let (rv, _) = self.emit_expr(rhs, scope)?;
                if matches!(lt, Type::Float(_)) {
                    let l = lv.into_float_value();
                    let r = rv.into_float_value();
                    let v = match op {
                        BinOp::Add => self.builder.build_float_add(l, r, "fadd"),
                        BinOp::Sub => self.builder.build_float_sub(l, r, "fsub"),
                        BinOp::Mul => self.builder.build_float_mul(l, r, "fmul"),
                        BinOp::Div => self.builder.build_float_div(l, r, "fdiv"),
                        BinOp::Rem => self.builder.build_float_rem(l, r, "frem"),
                        _ => return Err("codegen: bitwise op on a float".to_string()),
                    };
                    return Ok((v.map_err(le)?.into(), lt));
                }
                let l = lv.into_int_value();
                let r = rv.into_int_value();
                // The checker unifies both operands to one type, so the left
                // operand's signedness IS the operation's signedness (div/rem/shr).
                let signed = matches!(lt, Type::Int(_, true));
                let v = match op {
                    BinOp::Add => self.builder.build_int_add(l, r, "add"),
                    BinOp::Sub => self.builder.build_int_sub(l, r, "sub"),
                    BinOp::Mul => self.builder.build_int_mul(l, r, "mul"),
                    BinOp::Div if signed => self.builder.build_int_signed_div(l, r, "div"),
                    BinOp::Div => self.builder.build_int_unsigned_div(l, r, "div"),
                    BinOp::Rem if signed => self.builder.build_int_signed_rem(l, r, "rem"),
                    BinOp::Rem => self.builder.build_int_unsigned_rem(l, r, "rem"),
                    // udiv/urem: always unsigned, whatever the operand types.
                    BinOp::UDiv => self.builder.build_int_unsigned_div(l, r, "udiv"),
                    BinOp::URem => self.builder.build_int_unsigned_rem(l, r, "urem"),
                    BinOp::And => self.builder.build_and(l, r, "and"),
                    BinOp::Or => self.builder.build_or(l, r, "or"),
                    BinOp::Xor => self.builder.build_xor(l, r, "xor"),
                    BinOp::Shl => self.builder.build_left_shift(l, r, "shl"),
                    // arithmetic shift for signed, logical for unsigned.
                    BinOp::Shr => self.builder.build_right_shift(l, r, signed, "shr"),
                };
                Ok((v.map_err(le)?.into(), lt))
            }
            ExprKind::Not(x) => {
                let (xv, xt) = self.emit_expr(x, scope)?;
                let v = self.builder.build_not(xv.into_int_value(), "not").map_err(le)?;
                Ok((v.into(), xt))
            }
            ExprKind::Cmp { op, lhs, rhs } => {
                let (lv, lt) = self.emit_expr(lhs, scope)?;
                let (rv, _) = self.emit_expr(rhs, scope)?;
                if matches!(lt, Type::Float(_)) {
                    use inkwell::FloatPredicate as FP;
                    // ordered comparisons (false if either operand is NaN).
                    let pred = match op {
                        CmpOp::Lt => FP::OLT,
                        CmpOp::Le => FP::OLE,
                        CmpOp::Gt => FP::OGT,
                        CmpOp::Ge => FP::OGE,
                        CmpOp::Eq => FP::OEQ,
                        CmpOp::Ne => FP::ONE,
                    };
                    let b = self
                        .builder
                        .build_float_compare(pred, lv.into_float_value(), rv.into_float_value(), "fcmp")
                        .map_err(le)?;
                    return Ok((b.into(), Type::Bool));
                }
                // Operands share a type (the checker unifies them), so the left
                // operand's signedness selects signed vs unsigned comparison.
                let signed = matches!(lt, Type::Int(_, true));
                let pred = match op {
                    CmpOp::Lt if signed => IntPredicate::SLT,
                    CmpOp::Lt => IntPredicate::ULT,
                    CmpOp::Le if signed => IntPredicate::SLE,
                    CmpOp::Le => IntPredicate::ULE,
                    CmpOp::Gt if signed => IntPredicate::SGT,
                    CmpOp::Gt => IntPredicate::UGT,
                    CmpOp::Ge if signed => IntPredicate::SGE,
                    CmpOp::Ge => IntPredicate::UGE,
                    CmpOp::Eq => IntPredicate::EQ,
                    CmpOp::Ne => IntPredicate::NE,
                };
                let b = self
                    .builder
                    .build_int_compare(pred, lv.into_int_value(), rv.into_int_value(), "cmp")
                    .map_err(le)?;
                Ok((b.into(), Type::Bool))
            }
            ExprKind::Do(es) => {
                let mut last: Tv = (i64t.const_zero().into(), Type::Int(64, true));
                for e in es {
                    if self.block_terminated() {
                        break;
                    }
                    last = self.emit_expr(e, scope)?;
                }
                Ok(last)
            }
            ExprKind::Let { binds, body } => {
                let mut child = scope.clone();
                for (name, _mutable, val) in binds {
                    // The checker erases mutable `let` places to an alloca plus a
                    // store, so by codegen every binding is an ordinary value.
                    let v = self.emit_expr(val, &child)?;
                    // Debug info (-g): make the local visible to `frame variable`.
                    self.declare_let_local(name, v.0, &v.1, val.span);
                    child.insert(name.clone(), v);
                }
                let mut last: Tv = (i64t.const_zero().into(), Type::Int(64, true));
                for e in body {
                    if self.block_terminated() {
                        break;
                    }
                    last = self.emit_expr(e, &child)?;
                }
                Ok(last)
            }
            ExprKind::If { cond, then, els } => self.emit_if(cond, then, els, scope),
            ExprKind::Loop { label, body } => self.emit_loop(label.as_deref(), body, scope),
            ExprKind::Break { label, value } => {
                self.emit_break(label.as_deref(), value.as_deref(), scope)
            }
            ExprKind::Continue { label } => self.emit_continue(label.as_deref()),
            ExprKind::Call { func, args, .. } => {
                let argtv: Vec<Tv<'ctx>> = args
                    .iter()
                    .map(|a| self.emit_expr(a, scope))
                    .collect::<Result<_, _>>()?;
                let ret_ty = self
                    .rets
                    .get(func)
                    .cloned()
                    .ok_or_else(|| format!("codegen: unknown callable '{func}'"))?;
                if let Some(shim) = self.shims.get(func) {
                    let raw: Vec<_> = argtv.iter().map(|(v, _)| *v).collect();
                    let v = self.emit_shim_call(func, shim, &raw)?;
                    return Ok((v, ret_ty));
                }
                let callee = *self
                    .funcs
                    .get(func)
                    .ok_or_else(|| format!("codegen: call to undefined '{func}'"))?;
                // A by-value-struct callable goes through the C ABI marshalling.
                if self.csigs.contains_key(func) {
                    let sig = &self.csigs[func];
                    return self.emit_c_call(callee, sig, &argtv, ret_ty);
                }
                let meta: Vec<_> = argtv.iter().map(|(v, _)| (*v).into()).collect();
                let cs = self.builder.build_call(callee, &meta, "call").map_err(le)?;
                cs.set_call_convention(callee.get_call_conventions());
                // A `(-> void)` call yields no value; the checker forbids using its
                // result, so a placeholder labels the (statement-position) slot.
                if matches!(ret_ty, Type::Void) {
                    return Ok((self.ctx.i64_type().const_zero().into(), Type::Void));
                }
                let v = cs
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| "codegen: call returned void".to_string())?;
                Ok((v, ret_ty))
            }
            ExprKind::Alloc { storage, ty } => self.emit_alloc(*storage, ty),
            ExprKind::Field { ptr, field } => {
                let (pv, pt) = self.emit_expr(ptr, scope)?;
                let sname = match pt {
                    Type::Ptr(pointee) => match *pointee {
                        Type::Struct(s) => s,
                        other => return Err(format!("codegen: field on (ptr {other:?})")),
                    },
                    other => return Err(format!("codegen: field on non-pointer {other:?}")),
                };
                let info = self
                    .structs
                    .get(&sname)
                    .ok_or_else(|| format!("codegen: unknown struct '{sname}'"))?;
                let idx = info
                    .fields
                    .iter()
                    .position(|(n, _)| n == field)
                    .ok_or_else(|| format!("codegen: struct '{sname}' has no field '{field}'"))?;
                let fty = info.fields[idx].1.clone();
                let gep = match &info.layout {
                    // explicit layout: a byte blob -> GEP to the declared offset.
                    Layout::Explicit(e) => {
                        let off = self.ctx.i64_type().const_int(e.offsets[idx], false);
                        unsafe {
                            self.builder
                                .build_gep(self.ctx.i8_type(), pv.into_pointer_value(), &[off], "field")
                                .map_err(le)?
                        }
                    }
                    _ => self
                        .builder
                        .build_struct_gep(info.ty, pv.into_pointer_value(), idx as u32, "field")
                        .map_err(le)?,
                };
                Ok((gep.into(), Type::Ptr(Box::new(fty))))
            }
            ExprKind::BitGet { ptr, field } => {
                let (pv, off, width, backing) = self.bit_access(ptr, field, scope)?;
                let backing_ty = int_width(self.ctx, backing);
                let loaded = self
                    .builder
                    .build_load(backing_ty, pv, "bits")
                    .map_err(le)?
                    .into_int_value();
                let shifted = if off > 0 {
                    self.builder
                        .build_right_shift(loaded, backing_ty.const_int(off as u64, false), false, "shr")
                        .map_err(le)?
                } else {
                    loaded
                };
                let field_ty = int_width(self.ctx, width);
                let res = if width < backing {
                    self.builder.build_int_truncate(shifted, field_ty, "bget").map_err(le)?
                } else {
                    shifted
                };
                Ok((res.into(), Type::Int(width, false)))
            }
            ExprKind::BitSet { ptr, field, val } => {
                let (pv, off, width, backing) = self.bit_access(ptr, field, scope)?;
                let (vv, _) = self.emit_expr(val, scope)?;
                let backing_ty = int_width(self.ctx, backing);
                let loaded = self
                    .builder
                    .build_load(backing_ty, pv, "bits")
                    .map_err(le)?
                    .into_int_value();
                // clear the field's bits, then OR in the (shifted) new value.
                let field_mask = (((1u128 << width) - 1) << off) as u64;
                let clear = backing_ty.const_int(!field_mask, false);
                let cleared = self.builder.build_and(loaded, clear, "clear").map_err(le)?;
                let v = vv.into_int_value();
                let vext = if width < backing {
                    self.builder.build_int_z_extend(v, backing_ty, "vext").map_err(le)?
                } else {
                    v
                };
                let vshift = if off > 0 {
                    self.builder
                        .build_left_shift(vext, backing_ty.const_int(off as u64, false), "vshl")
                        .map_err(le)?
                } else {
                    vext
                };
                let newval = self.builder.build_or(cleared, vshift, "set").map_err(le)?;
                self.builder.build_store(pv, newval).map_err(le)?;
                Ok((vv, Type::Int(width, false)))
            }
            ExprKind::Load(p) => {
                let (pv, pt) = self.emit_expr(p, scope)?;
                let pointee = match pt {
                    Type::Ptr(pointee) => *pointee,
                    other => return Err(format!("codegen: load of non-pointer {other:?}")),
                };
                let v = self
                    .builder
                    .build_load(self.basic_ty(&pointee), pv.into_pointer_value(), "load")
                    .map_err(le)?;
                Ok((v, pointee))
            }
            ExprKind::Store { ptr, val } => {
                let (pv, _) = self.emit_expr(ptr, scope)?;
                let (vv, vt) = self.emit_expr(val, scope)?;
                self.builder
                    .build_store(pv.into_pointer_value(), vv)
                    .map_err(le)?;
                Ok((vv, vt))
            }
            ExprKind::Index { ptr, idx } => {
                let (pv, pt) = self.emit_expr(ptr, scope)?;
                let (iv, _) = self.emit_expr(idx, scope)?;
                let pointee = match pt {
                    Type::Ptr(pointee) => *pointee,
                    other => return Err(format!("codegen: index of non-pointer {other:?}")),
                };
                let ptr_val = pv.into_pointer_value();
                let i = iv.into_int_value();
                match &pointee {
                    // pointer to an array: GEP [0, i] yields a pointer to elem i.
                    Type::Array(elem, _) => {
                        let zero = self.ctx.i64_type().const_zero();
                        let gep = unsafe {
                            self.builder
                                .build_gep(self.basic_ty(&pointee), ptr_val, &[zero, i], "idx")
                                .map_err(le)?
                        };
                        Ok((gep.into(), Type::Ptr(elem.clone())))
                    }
                    // pointer to a scalar/struct: GEP [i] is pointer arithmetic.
                    _ => {
                        let gep = unsafe {
                            self.builder
                                .build_gep(self.basic_ty(&pointee), ptr_val, &[i], "idx")
                                .map_err(le)?
                        };
                        Ok((gep.into(), Type::Ptr(Box::new(pointee))))
                    }
                }
            }
            ExprKind::Cast { ty, expr } => {
                let (v, vt) = self.emit_expr(expr, scope)?;
                match ty {
                    // pointer -> integer (address of): ptrtoint.
                    Type::Int(to, _) if matches!(vt, Type::Ptr(..)) => {
                        let target = int_width(self.ctx, *to);
                        let out = self
                            .builder
                            .build_ptr_to_int(v.into_pointer_value(), target, "ptrtoint")
                            .map_err(le)?;
                        Ok((out.into(), ty.clone()))
                    }
                    // float -> integer: fptosi / fptoui.
                    Type::Int(to, signed) if matches!(vt, Type::Float(_)) => {
                        let target = int_width(self.ctx, *to);
                        let fv = v.into_float_value();
                        let out = if *signed {
                            self.builder.build_float_to_signed_int(fv, target, "fptosi").map_err(le)?
                        } else {
                            self.builder.build_float_to_unsigned_int(fv, target, "fptoui").map_err(le)?
                        };
                        Ok((out.into(), ty.clone()))
                    }
                    Type::Int(to, _) => {
                        let iv = v.into_int_value();
                        let from = iv.get_type().get_bit_width();
                        let src_signed = matches!(vt, Type::Int(_, true));
                        let target = int_width(self.ctx, *to);
                        let out = if *to > from {
                            // widen: sign-extend a signed source, zero-extend unsigned.
                            if src_signed {
                                self.builder.build_int_s_extend(iv, target, "sext").map_err(le)?
                            } else {
                                self.builder.build_int_z_extend(iv, target, "zext").map_err(le)?
                            }
                        } else if *to < from {
                            self.builder.build_int_truncate(iv, target, "trunc").map_err(le)?
                        } else {
                            iv
                        };
                        Ok((out.into(), ty.clone()))
                    }
                    // integer -> pointer: inttoptr.
                    Type::Ptr(..) if matches!(vt, Type::Int(..)) => {
                        let pt = self.ctx.ptr_type(AddressSpace::default());
                        let out = self
                            .builder
                            .build_int_to_ptr(v.into_int_value(), pt, "inttoptr")
                            .map_err(le)?;
                        Ok((out.into(), ty.clone()))
                    }
                    // opaque pointers: a ptr->ptr reinterpret leaves it untouched.
                    Type::Ptr(..) => Ok((v, ty.clone())),
                    // -> float: from an integer (sitofp/uitofp) or another float
                    // (fpext/fptrunc).
                    Type::Float(to) => {
                        let ft = self.basic_ty(ty).into_float_type();
                        let out = match vt {
                            Type::Int(_, src_signed) => {
                                let iv = v.into_int_value();
                                if src_signed {
                                    self.builder.build_signed_int_to_float(iv, ft, "sitofp").map_err(le)?
                                } else {
                                    self.builder.build_unsigned_int_to_float(iv, ft, "uitofp").map_err(le)?
                                }
                            }
                            Type::Float(from) => {
                                let fv = v.into_float_value();
                                if *to > from {
                                    self.builder.build_float_ext(fv, ft, "fpext").map_err(le)?
                                } else if *to < from {
                                    self.builder.build_float_trunc(fv, ft, "fptrunc").map_err(le)?
                                } else {
                                    fv
                                }
                            }
                            other => return Err(format!("codegen: cannot cast {other:?} to float")),
                        };
                        Ok((out.into(), ty.clone()))
                    }
                    other => Err(format!("codegen: cannot cast to {other:?}")),
                }
            }
            ExprKind::SizeOf(ty) => {
                let n = self.target_data.get_abi_size(&self.basic_ty(ty));
                Ok((i64t.const_int(n, false).into(), Type::Int(64, true)))
            }
            ExprKind::AlignOf(ty) => {
                let n = self.target_data.get_abi_alignment(&self.basic_ty(ty)) as u64;
                Ok((i64t.const_int(n, false).into(), Type::Int(64, true)))
            }
            ExprKind::OffsetOf(ty, field) => {
                let n = self.offset_of(ty, field)?;
                Ok((i64t.const_int(n, false).into(), Type::Int(64, true)))
            }
            ExprKind::Free(p) => {
                let (pv, _) = self.emit_expr(p, scope)?;
                self.builder.build_free(pv.into_pointer_value()).map_err(le)?;
                Ok((i64t.const_zero().into(), Type::Int(64, true)))
            }
            ExprKind::Construct { sum, variant, args } => self.emit_construct(sum, variant, args, scope),
            ExprKind::Match { scrut, arms } => self.emit_match(scrut, arms, scope),
            ExprKind::FnPtrOf(name) => {
                let fv = *self
                    .funcs
                    .get(name)
                    .ok_or_else(|| format!("codegen: fnptr-of unknown '{name}'"))?;
                let (cc, params, ret) = self
                    .callables
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("codegen: no signature for '{name}'"))?;
                let ptr = fv.as_global_value().as_pointer_value();
                Ok((ptr.into(), Type::Fn(cc, params, Box::new(ret))))
            }
            ExprKind::CallPtr { fp, args } => {
                let (fpv, fpt) = self.emit_expr(fp, scope)?;
                let (cc, params, ret) = match fpt {
                    Type::Fn(cc, params, ret) => (cc, params, *ret),
                    other => return Err(format!("codegen: call-ptr on non-fnptr {other:?}")),
                };
                let fn_ty = self.fn_type_types(&params, &ret);
                let argtv: Vec<Tv<'ctx>> = args
                    .iter()
                    .map(|a| self.emit_expr(a, scope))
                    .collect::<Result<_, _>>()?;
                let meta: Vec<_> = argtv.iter().map(|(v, _)| (*v).into()).collect();
                let cs = self
                    .builder
                    .build_indirect_call(fn_ty, fpv.into_pointer_value(), &meta, "callptr")
                    .map_err(le)?;
                if let Some(id) = self.conv_ids.get(&cc) {
                    cs.set_call_convention(*id);
                }
                let v = cs
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| "codegen: call-ptr returned void".to_string())?;
                Ok((v, ret))
            }
        }
    }

    fn emit_construct(
        &self,
        sum: &str,
        variant: &str,
        args: &[Expr],
        scope: &HashMap<String, Tv<'ctx>>,
    ) -> Result<Tv<'ctx>, String> {
        let info = self.sums.get(sum).ok_or_else(|| format!("codegen: unknown sum '{sum}'"))?;
        let vidx = info
            .variants
            .iter()
            .position(|(n, _)| n == variant)
            .ok_or_else(|| format!("codegen: sum '{sum}' has no variant '{variant}'"))?;
        let sum_ty = info.ty;
        let var_struct = info.variant_structs[vidx];

        let tmp = self.builder.build_alloca(sum_ty, "sum.tmp").map_err(le)?;
        let tagptr = self.builder.build_struct_gep(sum_ty, tmp, 0, "tag").map_err(le)?;
        self.builder
            .build_store(tagptr, self.ctx.i32_type().const_int(vidx as u64, false))
            .map_err(le)?;
        let payload = self.builder.build_struct_gep(sum_ty, tmp, 1, "payload").map_err(le)?;
        for (i, a) in args.iter().enumerate() {
            let (v, _) = self.emit_expr(a, scope)?;
            let fptr = self
                .builder
                .build_struct_gep(var_struct, payload, i as u32, "vf")
                .map_err(le)?;
            self.builder.build_store(fptr, v).map_err(le)?;
        }
        let val = self.builder.build_load(sum_ty, tmp, "sum").map_err(le)?;
        Ok((val, Type::Struct(sum.to_string())))
    }

    fn emit_match(
        &self,
        scrut: &Expr,
        arms: &[Arm],
        scope: &HashMap<String, Tv<'ctx>>,
    ) -> Result<Tv<'ctx>, String> {
        let (sumval, st) = self.emit_expr(scrut, scope)?;
        let sumname = match st {
            Type::Struct(s) => s,
            other => return Err(format!("codegen: match on non-sum {other:?}")),
        };
        let info = self
            .sums
            .get(&sumname)
            .ok_or_else(|| format!("codegen: match on non-sum '{sumname}'"))?;
        let sum_ty = info.ty;

        // spill the scrutinee so we can GEP into it
        let tmp = self.builder.build_alloca(sum_ty, "match.tmp").map_err(le)?;
        self.builder.build_store(tmp, sumval).map_err(le)?;
        let tagptr = self.builder.build_struct_gep(sum_ty, tmp, 0, "tag").map_err(le)?;
        let tag = self
            .builder
            .build_load(self.ctx.i32_type(), tagptr, "tag")
            .map_err(le)?
            .into_int_value();
        let payload = self.builder.build_struct_gep(sum_ty, tmp, 1, "payload").map_err(le)?;

        let function = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or("codegen: no current function")?;
        let arm_blocks: Vec<BasicBlock> =
            arms.iter().map(|_| self.ctx.append_basic_block(function, "arm")).collect();
        let default = self.ctx.append_basic_block(function, "match.default");
        let merge = self.ctx.append_basic_block(function, "match.end");

        let cases: Vec<(inkwell::values::IntValue, BasicBlock)> = arms
            .iter()
            .enumerate()
            .map(|(k, arm)| {
                let vidx = info.variants.iter().position(|(n, _)| n == &arm.variant).unwrap();
                (self.ctx.i32_type().const_int(vidx as u64, false), arm_blocks[k])
            })
            .collect();
        self.builder.build_switch(tag, default, &cases).map_err(le)?;
        self.builder.position_at_end(default);
        self.builder.build_unreachable().map_err(le)?;

        let mut incoming: Vec<(BasicValueEnum<'ctx>, BasicBlock)> = Vec::new();
        let mut result_ty = Type::Int(64, true);
        for (k, arm) in arms.iter().enumerate() {
            self.builder.position_at_end(arm_blocks[k]);
            let vidx = info.variants.iter().position(|(n, _)| n == &arm.variant).unwrap();
            let var_struct = info.variant_structs[vidx];
            let mut child = scope.clone();
            for (i, b) in arm.binds.iter().enumerate() {
                let fptr = self
                    .builder
                    .build_struct_gep(var_struct, payload, i as u32, "vf")
                    .map_err(le)?;
                let fty = info.variants[vidx].1[i].1.clone();
                let fval = self.builder.build_load(self.basic_ty(&fty), fptr, b).map_err(le)?;
                child.insert(b.clone(), (fval, fty));
            }
            let (bval, bty) = self.emit_expr(&arm.body, &child)?;
            // An arm may diverge (end in break/continue/return-from): only a
            // falling-through arm branches to the merge and feeds the result phi.
            // Take the phi's type from a non-diverging arm — a diverging arm's
            // type is Never (placeholder i64), which would mistype the phi.
            let end = self.builder.get_insert_block().unwrap();
            if end.get_terminator().is_none() {
                result_ty = bty;
                self.builder.build_unconditional_branch(merge).map_err(le)?;
                incoming.push((bval, end));
            }
        }

        self.builder.position_at_end(merge);
        if incoming.is_empty() {
            // Every arm diverged: the merge is unreachable.
            self.builder.build_unreachable().map_err(le)?;
            return Ok((self.ctx.i64_type().const_zero().into(), result_ty));
        }
        let phi = self.builder.build_phi(self.basic_ty(&result_ty), "match.val").map_err(le)?;
        let inc: Vec<(&dyn BasicValue, BasicBlock)> =
            incoming.iter().map(|(v, b)| (v as &dyn BasicValue, *b)).collect();
        phi.add_incoming(&inc);
        Ok((phi.as_basic_value(), result_ty))
    }

    fn emit_alloc(&self, storage: Storage, ty: &Type) -> Result<Tv<'ctx>, String> {
        let bt = self.basic_ty(ty);
        // a struct with `:layout (align N)` (or explicit `:align`) forces its
        // allocation alignment.
        let force_align = match ty {
            Type::Struct(name) => match self.structs.get(name).map(|s| &s.layout) {
                Some(Layout::Aligned(n)) => Some(*n),
                Some(Layout::Explicit(e)) if e.align > 0 => Some(e.align),
                _ => None,
            },
            _ => None,
        };
        let ptr = match storage {
            Storage::Stack => {
                let p = self.builder.build_alloca(bt, "stack.slot").map_err(le)?;
                if let (Some(a), Some(instr)) = (force_align, p.as_instruction()) {
                    let _ = instr.set_alignment(a);
                }
                p
            }
            Storage::Heap => self.builder.build_malloc(bt, "heap.box").map_err(le)?,
            Storage::Static => {
                let n = self.globals.get();
                self.globals.set(n + 1);
                let g = self.module.add_global(bt, None, &format!("g.{n}"));
                g.set_initializer(&bt.const_zero());
                if let Some(a) = force_align {
                    g.set_alignment(a);
                }
                g.as_pointer_value()
            }
        };
        Ok((ptr.into(), Type::Ptr(Box::new(ty.clone()))))
    }

    /// Resolve a bitfield access to (struct pointer, bit offset, field width,
    /// backing width).
    fn bit_access(
        &self,
        ptr: &Expr,
        field: &str,
        scope: &HashMap<String, Tv<'ctx>>,
    ) -> Result<(inkwell::values::PointerValue<'ctx>, u32, u32, u32), String> {
        let (pv, pt) = self.emit_expr(ptr, scope)?;
        let sname = match pt {
            Type::Ptr(p) => match *p {
                Type::Struct(s) => s,
                other => return Err(format!("codegen: bit access on (ptr {other:?})")),
            },
            other => return Err(format!("codegen: bit access on {other:?}")),
        };
        let info = self
            .structs
            .get(&sname)
            .ok_or_else(|| format!("codegen: unknown struct '{sname}'"))?;
        let b = match &info.layout {
            Layout::Bits(b) => b,
            _ => return Err(format!("codegen: '{sname}' is not a bit struct")),
        };
        let idx = info
            .fields
            .iter()
            .position(|(n, _)| n == field)
            .ok_or_else(|| format!("codegen: bit struct '{sname}' has no field '{field}'"))?;
        let width = match info.fields[idx].1 {
            Type::Int(w, _) => w,
            _ => return Err("codegen: bitfield is not an integer".to_string()),
        };
        Ok((pv.into_pointer_value(), b.offsets[idx], width, b.backing))
    }

    fn offset_of(&self, ty: &Type, field: &str) -> Result<u64, String> {
        match ty {
            Type::Struct(name) => {
                let info = self
                    .structs
                    .get(name)
                    .ok_or_else(|| format!("codegen: offsetof on non-struct '{name}'"))?;
                let idx = info
                    .fields
                    .iter()
                    .position(|(n, _)| n == field)
                    .ok_or_else(|| format!("codegen: struct '{name}' has no field '{field}'"))?;
                match &info.layout {
                    Layout::Explicit(e) => Ok(e.offsets[idx]),
                    _ => self
                        .target_data
                        .offset_of_element(&info.ty, idx as u32)
                        .ok_or_else(|| "codegen: could not compute field offset".to_string()),
                }
            }
            other => Err(format!("codegen: offsetof needs a struct, got {other:?}")),
        }
    }

    /// Evaluate a compile-time constant expression (for `static-assert`).
    fn const_eval(&self, e: &Expr) -> Result<i64, String> {
        Ok(self.const_eval_t(e)?.0)
    }

    /// Constant-fold to `(value, is_unsigned)`. `is_unsigned` tracks operand
    /// signedness — set by a `(cast uN …)` — so division, remainder, shift, and
    /// comparison fold with the SAME signedness the runtime codegen uses (which
    /// dispatches by operand type). A signed-only fold silently diverges from the
    /// runtime for unsigned operands (e.g. `(idiv (cast u64 -1) …)` in a
    /// static-assert) — a const/runtime mismatch this avoids.
    fn const_eval_t(&self, e: &Expr) -> Result<(i64, bool), String> {
        Ok(match &e.kind {
            ExprKind::Int(n) => (*n, false),
            ExprKind::Bool(b) => (*b as i64, false),
            ExprKind::If { cond, then, els } => {
                // covers `and`/`or`/`not`, which desugar to `if`.
                if self.const_eval_t(cond)?.0 != 0 {
                    self.const_eval_t(then)?
                } else {
                    self.const_eval_t(els)?
                }
            }
            ExprKind::SizeOf(t) => (self.target_data.get_abi_size(&self.basic_ty(t)) as i64, false),
            ExprKind::AlignOf(t) => (self.target_data.get_abi_alignment(&self.basic_ty(t)) as i64, false),
            ExprKind::OffsetOf(t, f) => (self.offset_of(t, f)? as i64, false),
            // A cast to a `uN` type makes the value unsigned for later div/cmp/shr.
            ExprKind::Cast { ty, expr } => {
                let v = self.const_eval_t(expr)?.0;
                (v, matches!(ty, Type::Int(_, false)))
            }
            ExprKind::Bin { op, lhs, rhs } => {
                let (l, lu) = self.const_eval_t(lhs)?;
                let (r, ru) = self.const_eval_t(rhs)?;
                let uns = lu || ru; // operands share a type, so either flag implies both
                let zero = "static-assert: divide/mod by zero".to_string();
                let v = match op {
                    BinOp::Add => l.wrapping_add(r),
                    BinOp::Sub => l.wrapping_sub(r),
                    BinOp::Mul => l.wrapping_mul(r),
                    BinOp::And => l & r,
                    BinOp::Or => l | r,
                    BinOp::Xor => l ^ r,
                    BinOp::Shl => l.wrapping_shl(r as u32),
                    // unsigned >> is logical, signed >> is arithmetic.
                    BinOp::Shr if uns => ((l as u64).wrapping_shr(r as u32)) as i64,
                    BinOp::Shr => l.wrapping_shr(r as u32),
                    BinOp::Div if r == 0 => return Err(zero),
                    BinOp::Div if uns => ((l as u64) / (r as u64)) as i64,
                    BinOp::Div => l.wrapping_div(r),
                    BinOp::Rem if r == 0 => return Err(zero),
                    BinOp::Rem if uns => ((l as u64) % (r as u64)) as i64,
                    BinOp::Rem => l.wrapping_rem(r),
                    BinOp::UDiv if r == 0 => return Err(zero),
                    BinOp::UDiv => ((l as u64) / (r as u64)) as i64,
                    BinOp::URem if r == 0 => return Err(zero),
                    BinOp::URem => ((l as u64) % (r as u64)) as i64,
                };
                (v, uns)
            }
            ExprKind::Cmp { op, lhs, rhs } => {
                let (l, lu) = self.const_eval_t(lhs)?;
                let (r, ru) = self.const_eval_t(rhs)?;
                let b = if lu || ru {
                    let (l, r) = (l as u64, r as u64);
                    match op {
                        CmpOp::Eq => l == r,
                        CmpOp::Ne => l != r,
                        CmpOp::Lt => l < r,
                        CmpOp::Le => l <= r,
                        CmpOp::Gt => l > r,
                        CmpOp::Ge => l >= r,
                    }
                } else {
                    match op {
                        CmpOp::Eq => l == r,
                        CmpOp::Ne => l != r,
                        CmpOp::Lt => l < r,
                        CmpOp::Le => l <= r,
                        CmpOp::Gt => l > r,
                        CmpOp::Ge => l >= r,
                    }
                };
                (i64::from(b), false)
            }
            _ => {
                return Err(
                    "static-assert: only int literals, sizeof/alignof/offsetof, casts, arithmetic \
                     and comparisons are allowed"
                        .to_string(),
                )
            }
        })
    }

    /// True if the block the builder currently sits on already ends with a
    /// terminator — i.e. control diverged here (a `break`/`continue`, an
    /// unbroken loop, or an `if`/`match` whose every arm diverged). Used to skip
    /// dead trailing code and to avoid emitting a second terminator.
    fn block_terminated(&self) -> bool {
        self.builder
            .get_insert_block()
            .and_then(|b| b.get_terminator())
            .is_some()
    }

    /// The structured loop: a header block (the `continue` target and back-edge
    /// destination), then an after block (the `break` target) where the loop's
    /// value materializes as a phi over the break sites. A loop with no break is
    /// divergent — the after block is `unreachable`.
    fn emit_loop(
        &self,
        label: Option<&str>,
        body: &[Expr],
        scope: &HashMap<String, Tv<'ctx>>,
    ) -> Result<Tv<'ctx>, String> {
        let function = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or("codegen: no current function")?;
        let body_bb = self.ctx.append_basic_block(function, "loop.body");
        let after_bb = self.ctx.append_basic_block(function, "loop.after");
        self.builder.build_unconditional_branch(body_bb).map_err(le)?;

        self.builder.position_at_end(body_bb);
        self.loops.borrow_mut().push(LoopCtxCg {
            label: label.map(str::to_string),
            body_bb,
            after_bb,
            breaks: Vec::new(),
            break_ty: None,
        });
        for e in body {
            if self.block_terminated() {
                break;
            }
            if let Err(err) = self.emit_expr(e, scope) {
                self.loops.borrow_mut().pop();
                return Err(err);
            }
        }
        // Fall-through at the body's end is the back-edge to the header.
        if !self.block_terminated() {
            self.builder.build_unconditional_branch(body_bb).map_err(le)?;
        }
        let ctx = self.loops.borrow_mut().pop().expect("loop ctx present");

        self.builder.position_at_end(after_bb);
        if ctx.breaks.is_empty() {
            // No `break` reaches here: the loop diverges; after is unreachable.
            // The value is conventional and never consumed (callers check
            // `block_terminated`).
            self.builder.build_unreachable().map_err(le)?;
            Ok((self.ctx.i64_type().const_zero().into(), Type::Int(64, true)))
        } else {
            let ty = ctx.break_ty.clone().unwrap_or(Type::Int(64, true));
            let phi = self.builder.build_phi(self.basic_ty(&ty), "loop.val").map_err(le)?;
            for (v, bb) in &ctx.breaks {
                phi.add_incoming(&[(v as &dyn BasicValue, *bb)]);
            }
            Ok((phi.as_basic_value(), ty))
        }
    }

    fn emit_break(
        &self,
        label: Option<&str>,
        value: Option<&Expr>,
        scope: &HashMap<String, Tv<'ctx>>,
    ) -> Result<Tv<'ctx>, String> {
        // Evaluate the value (default i64 0) BEFORE borrowing the loop stack.
        let (v, vty) = match value {
            Some(e) => self.emit_expr(e, scope)?,
            None => (self.ctx.i64_type().const_zero().into(), Type::Int(64, true)),
        };
        // If the value expression itself diverged (e.g. it contained an inner
        // `break`), control already left this block — our branch would be a
        // second terminator. Nothing to contribute.
        if self.block_terminated() {
            return Ok((self.ctx.i64_type().const_zero().into(), Type::Int(64, true)));
        }
        let pred = self.builder.get_insert_block().ok_or("codegen: no current block")?;
        let after = {
            let mut loops = self.loops.borrow_mut();
            let idx = loop_index(&loops, label, "break")?;
            loops[idx].breaks.push((v, pred));
            if loops[idx].break_ty.is_none() {
                loops[idx].break_ty = Some(vty);
            }
            loops[idx].after_bb
        };
        self.builder.build_unconditional_branch(after).map_err(le)?;
        Ok((self.ctx.i64_type().const_zero().into(), Type::Int(64, true)))
    }

    fn emit_continue(&self, label: Option<&str>) -> Result<Tv<'ctx>, String> {
        let body = {
            let loops = self.loops.borrow();
            let idx = loop_index(&loops, label, "continue")?;
            loops[idx].body_bb
        };
        self.builder.build_unconditional_branch(body).map_err(le)?;
        Ok((self.ctx.i64_type().const_zero().into(), Type::Int(64, true)))
    }

    fn emit_if(
        &self,
        cond: &Expr,
        then: &Expr,
        els: &Expr,
        scope: &HashMap<String, Tv<'ctx>>,
    ) -> Result<Tv<'ctx>, String> {
        let function = self
            .builder
            .get_insert_block()
            .and_then(|b| b.get_parent())
            .ok_or("codegen: no current function")?;

        let (cv, _) = self.emit_expr(cond, scope)?;
        let c = cv.into_int_value();
        let cmp = self
            .builder
            .build_int_compare(IntPredicate::NE, c, c.get_type().const_zero(), "ifc")
            .map_err(le)?;

        let then_bb = self.ctx.append_basic_block(function, "then");
        let else_bb = self.ctx.append_basic_block(function, "else");

        self.builder
            .build_conditional_branch(cmp, then_bb, else_bb)
            .map_err(le)?;

        // Each arm may diverge (e.g. end in a `break`/`continue`). The merge block
        // is created lazily so the one-arm-diverges cases don't leave an empty
        // passthrough block: when only one arm falls through, execution simply
        // continues on that arm's end block.
        self.builder.position_at_end(then_bb);
        let (tv, tt) = self.emit_expr(then, scope)?;
        let then_end = self.builder.get_insert_block().unwrap();
        let then_div = then_end.get_terminator().is_some();

        self.builder.position_at_end(else_bb);
        let (ev, et) = self.emit_expr(els, scope)?;
        let else_end = self.builder.get_insert_block().unwrap();
        let else_div = else_end.get_terminator().is_some();

        match (then_div, else_div) {
            // Both reach: a merge block with a phi reconciling the (unified) types.
            (false, false) => {
                let merge_bb = self.ctx.append_basic_block(function, "ifcont");
                self.builder.position_at_end(then_end);
                self.builder.build_unconditional_branch(merge_bb).map_err(le)?;
                self.builder.position_at_end(else_end);
                self.builder.build_unconditional_branch(merge_bb).map_err(le)?;
                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(self.basic_ty(&tt), "ifval").map_err(le)?;
                phi.add_incoming(&[
                    (&tv as &dyn BasicValue, then_end),
                    (&ev as &dyn BasicValue, else_end),
                ]);
                Ok((phi.as_basic_value(), tt))
            }
            // Exactly one arm falls through: continue on it; its value dominates
            // the continuation, so no merge block and no phi are needed. (Position
            // the builder back onto the live arm — emission left it on the other.)
            (false, true) => {
                self.builder.position_at_end(then_end);
                Ok((tv, tt))
            }
            (true, false) => {
                self.builder.position_at_end(else_end);
                Ok((ev, et))
            }
            // Both arms diverged: the continuation is unreachable. Land on a fresh
            // block with `unreachable` so any (dead) trailing code has an insert
            // point; callers check `block_terminated` and skip it.
            (true, true) => {
                let dead_bb = self.ctx.append_basic_block(function, "ifcont.dead");
                self.builder.position_at_end(dead_bb);
                self.builder.build_unreachable().map_err(le)?;
                Ok((self.ctx.i64_type().const_zero().into(), tt))
            }
        }
    }
}

/// Resolve a `break`/`continue` target on the loop stack: the loop named
/// `label`, or the innermost loop when unlabeled. A hard error if none matches
/// (the checker rejects this first; this guards codegen invariants).
fn loop_index(loops: &[LoopCtxCg], label: Option<&str>, what: &str) -> Result<usize, String> {
    match label {
        Some(l) => loops
            .iter()
            .rposition(|c| c.label.as_deref() == Some(l))
            .ok_or_else(|| format!("codegen: {what} to unknown loop label ':{l}'")),
        None => loops
            .len()
            .checked_sub(1)
            .ok_or_else(|| format!("codegen: {what} outside of a loop")),
    }
}

fn le<E: std::fmt::Display>(e: E) -> String {
    format!("llvm: {e}")
}

fn align8(x: u64) -> u64 {
    x.div_ceil(8) * 8
}

/// Number of i64 words the payload union needs: the largest variant's size
/// (conservative — every field rounded up to 8 bytes; an upper bound is fine
/// since we read/write through the real per-variant struct type).
fn sum_words(sd: &SumDef, structs: &HashMap<&str, &StructDef>, sums: &HashMap<&str, &SumDef>) -> u32 {
    let max_bytes = sd
        .variants
        .iter()
        .map(|v| {
            v.fields
                .iter()
                .map(|(_, t)| align8(type_bytes(t, structs, sums)))
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0);
    (max_bytes / 8) as u32
}

fn type_bytes(t: &Type, structs: &HashMap<&str, &StructDef>, sums: &HashMap<&str, &SumDef>) -> u64 {
    match t {
        Type::Never => unreachable!("Never type has no size"),
        Type::Void => unreachable!("void has no size (return type only)"),
        Type::Int(bits, _) => (*bits as u64).div_ceil(8),
        Type::Float(bits) => (*bits as u64) / 8,
        Type::Bool => 1,
        Type::Ptr(_) | Type::Fn(..) | Type::Ref(..) => 8,
        Type::Array(e, n) => align8(type_bytes(e, structs, sums)) * (*n as u64),
        // A slice is a fat pointer: a ptr (8) + an i64 length (8).
        Type::Slice(_) => 16,
        // A vector is tightly packed (no per-lane padding): lanes * element bytes.
        Type::Vec(e, n) => type_bytes(e, structs, sums) * (*n as u64),
        Type::Struct(name) => {
            if let Some(s) = structs.get(name.as_str()) {
                s.fields.iter().map(|(_, t)| align8(type_bytes(t, structs, sums))).sum()
            } else if let Some(sm) = sums.get(name.as_str()) {
                8 + (sum_words(sm, structs, sums) as u64) * 8
            } else {
                8
            }
        }
        Type::App(..) => 8,
    }
}
