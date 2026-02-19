use crate::ast::*;
use inkwell::builder::Builder;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicTypeEnum, StructType};
use inkwell::values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PointerValue};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

#[derive(Debug)]
pub struct CodegenError {
    pub message: String,
}

pub struct Codegen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    thread_struct_ty: StructType<'ctx>,
    thread_ty: inkwell::types::PointerType<'ctx>,
    mode: CodegenMode,
    functions: HashMap<String, FunctionValue<'ctx>>,
    externs: HashMap<String, FunctionValue<'ctx>>,
    gc_pollcheck: FunctionValue<'ctx>,
    gc_allocate_obj: FunctionValue<'ctx>,
    gc_write_barrier: FunctionValue<'ctx>,
    structs: HashMap<String, StructLayout<'ctx>>,
    enums: HashMap<String, EnumLayout<'ctx>>,
    tuples: HashMap<String, StructLayout<'ctx>>,
    str_lit_id: usize,
    next_type_id: u16,
    compiler_used_globals: Vec<PointerValue<'ctx>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenMode {
    Aot,
}

#[derive(Clone)]
struct StructLayout<'ctx> {
    field_count: u64,
    ptr_field_count: u64,
    type_id: u16,
    fields: Vec<FieldLayout<'ctx>>,
}

#[derive(Clone)]
struct FieldLayout<'ctx> {
    name: String,
    ty: Type,
    llvm_ty: BasicTypeEnum<'ctx>,
    index: u64,
    is_gc_ref: bool,
}

#[derive(Clone)]
struct EnumLayout<'ctx> {
    field_count: u64,
    ptr_field_count: u64,
    type_id: u16,
    variants: HashMap<String, VariantLayout<'ctx>>,
}

#[derive(Clone)]
struct VariantLayout<'ctx> {
    tag: u32,
    fields: Vec<VariantFieldLayout<'ctx>>,
}

#[derive(Clone)]
struct VariantFieldLayout<'ctx> {
    name: Option<String>,
    ty: Type,
    llvm_ty: BasicTypeEnum<'ctx>,
    is_gc_ref: bool,
    field_index: u64,
}

pub fn compile_to_object(modules: &[crate::ast::Module], output: &std::path::Path) -> Result<(), CodegenError> {
    use inkwell::targets::{InitializationConfig, RelocMode, Target, TargetMachine, CodeModel, FileType};

    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| CodegenError { message: format!("target init failed: {e}") })?;
    let context = Context::create();
    let mut gen = Codegen::new(&context, "lang_module", CodegenMode::Aot);
    gen.declare_structs(modules)?;
    gen.declare_enums(modules)?;
    gen.declare_tuples(modules)?;
    gen.declare_functions(modules)?;
    gen.define_functions(modules)?;
    gen.emit_aot_wrapper_main()?;
    gen.emit_compiler_used();
    if std::env::var("LANGC_DUMP_IR").is_ok() {
        eprintln!("{}", gen.module.print_to_string().to_string());
    }

    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple)
        .map_err(|e| CodegenError { message: format!("target lookup failed: {e}") })?;
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();
    let cpu_str = cpu.to_str().unwrap_or("generic");
    let feat_str = features.to_str().unwrap_or("");
    let target_machine = target
        .create_target_machine(
            &triple,
            cpu_str,
            feat_str,
            OptimizationLevel::None,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or_else(|| CodegenError { message: "failed to create target machine".to_string() })?;

    gen.module.set_triple(&triple);
    let data_layout = target_machine.get_target_data().get_data_layout();
    gen.module.set_data_layout(&data_layout);

    target_machine
        .write_to_file(&gen.module, FileType::Object, output)
        .map_err(|e| CodegenError { message: format!("write object failed: {e}") })?;
    Ok(())
}

pub fn compile_to_executable(
    modules: &[crate::ast::Module],
    output: &std::path::Path,
    _link_libs: &[String],
) -> Result<(), CodegenError> {
    use std::process::Command;

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let runtime_dir = manifest_dir.join("runtime");

    // Step 1: Compile to object file
    let obj_path = std::env::temp_dir().join(format!("langc_obj_{}.o", std::process::id()));
    compile_to_object(modules, &obj_path)?;

    // Step 2: Compile runtime.c
    let runtime_o = runtime_dir.join("runtime.o");
    let status = Command::new("cc")
        .arg("-c")
        .arg(runtime_dir.join("runtime.c"))
        .arg("-O2")
        .arg("-o")
        .arg(&runtime_o)
        .status()
        .map_err(|e| CodegenError { message: format!("failed to invoke cc: {e}") })?;
    if !status.success() {
        return Err(CodegenError { message: "failed to compile runtime.c".to_string() });
    }

    // Step 3: Compile gc_bridge.c
    let gc_bridge_o = runtime_dir.join("gc_bridge.o");
    let gc_include = manifest_dir.join("../../claude-experiments/gc-library/include");
    let status = Command::new("cc")
        .arg("-c")
        .arg(runtime_dir.join("gc_bridge.c"))
        .arg("-I").arg(&gc_include)
        .arg("-O2")
        .arg("-o")
        .arg(&gc_bridge_o)
        .status()
        .map_err(|e| CodegenError { message: format!("failed to invoke cc: {e}") })?;
    if !status.success() {
        return Err(CodegenError { message: "failed to compile gc_bridge.c".to_string() });
    }

    // Step 4: Compile llvm_shims.c
    let llvm_config = crate::find_llvm_config();
    let shims_c = runtime_dir.join("llvm_shims.c");
    let shims_o = runtime_dir.join("llvm_shims.o");
    let cflags = Command::new(&llvm_config)
        .arg("--cflags")
        .output()
        .map_err(|e| CodegenError { message: format!("failed to run llvm-config: {e}") })?;
    let mut cc_cmd = Command::new("cc");
    let cflags_str = String::from_utf8_lossy(&cflags.stdout);
    for flag in cflags_str.split_whitespace() {
        if !flag.is_empty() {
            cc_cmd.arg(flag);
        }
    }
    cc_cmd.arg("-c").arg(&shims_c).arg("-o").arg(&shims_o);
    let status = cc_cmd.status()
        .map_err(|e| CodegenError { message: format!("failed to invoke cc: {e}") })?;
    if !status.success() {
        return Err(CodegenError { message: "failed to compile llvm_shims.c".to_string() });
    }

    // Step 5: Find libgc_library.a
    let gc_lib = manifest_dir.join("../../claude-experiments/gc-library/target/release/libgc_library.a");

    // Step 6: Link everything
    let mut link_cmd = Command::new("cc");
    link_cmd
        .arg(&obj_path)
        .arg(&runtime_o)
        .arg(&gc_bridge_o)
        .arg(&shims_o)
        .arg(&gc_lib)
        .arg("-O2")
        .arg("-o")
        .arg(output);

    // Add LLVM link flags
    if let Ok(llvm_output) = Command::new(&llvm_config).args(["--ldflags", "--libs", "--system-libs"]).output() {
        if llvm_output.status.success() {
            let flags = String::from_utf8_lossy(&llvm_output.stdout);
            for flag in flags.split_whitespace() {
                if !flag.is_empty() {
                    link_cmd.arg(flag);
                }
            }
        }
    }
    link_cmd.arg("-lc++").arg("-lz").arg("-lzstd");

    let status = link_cmd.status()
        .map_err(|e| CodegenError { message: format!("failed to invoke linker: {e}") })?;
    if !status.success() {
        return Err(CodegenError { message: "linking failed".to_string() });
    }

    // Clean up temp object file
    let _ = std::fs::remove_file(&obj_path);

    Ok(())
}

impl<'ctx> Codegen<'ctx> {
    fn declare_structs(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            let module_path = module.path.clone().unwrap_or_default();
            for item in &module.items {
                if let Item::Struct(s) = item {
                    let full_name = full_item_name(&module_path, &s.name);
                    let layout = self.compute_struct_layout(&full_name, s)?;
                    self.structs.insert(full_name, layout);
                }
            }
        }
        Ok(())
    }

    fn declare_enums(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            let module_path = module.path.clone().unwrap_or_default();
            for item in &module.items {
                if let Item::Enum(e) = item {
                    let full_name = full_item_name(&module_path, &e.name);
                    let layout = self.compute_enum_layout(&full_name, e)?;
                    self.enums.insert(full_name, layout);
                }
            }
        }
        Ok(())
    }

    fn declare_tuples(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            for item in &module.items {
                match item {
                    Item::Struct(s) => {
                        for field in &s.fields {
                            self.collect_tuple_types(&field.ty)?;
                        }
                    }
                    Item::Enum(e) => {
                        for variant in &e.variants {
                            match &variant.kind {
                                EnumVariantKind::Unit => {}
                                EnumVariantKind::Tuple(types) => {
                                    for ty in types {
                                        self.collect_tuple_types(ty)?;
                                    }
                                }
                                EnumVariantKind::Struct(fields) => {
                                    for field in fields {
                                        self.collect_tuple_types(&field.ty)?;
                                    }
                                }
                            }
                        }
                    }
                    Item::Fn(f) => {
                        for param in &f.params {
                            self.collect_tuple_types(&param.ty)?;
                        }
                        self.collect_tuple_types(&f.ret_type)?;
                        self.collect_tuple_types_in_block(&f.body)?;
                    }
                    Item::ExternFn(f) => {
                        for param in &f.params {
                            self.collect_tuple_types(&param.ty)?;
                        }
                        self.collect_tuple_types(&f.ret_type)?;
                    }
                    Item::Use(_) | Item::Link(_) => {}
                }
            }
        }
        Ok(())
    }

    fn compute_struct_layout(&mut self, _full_name: &str, s: &StructDecl) -> Result<StructLayout<'ctx>, CodegenError> {
        let mut ptr_fields = Vec::new();
        let mut raw_fields = Vec::new();

        for field in &s.fields {
            let is_gc_ref = self.is_gc_ref_type(&field.ty);
            let llvm_ty = self.field_llvm_type(&field.ty)?;
            let entry = FieldLayout {
                name: field.name.clone(),
                ty: field.ty.clone(),
                llvm_ty,
                index: 0,
                is_gc_ref,
            };
            if is_gc_ref {
                ptr_fields.push(entry);
            } else {
                raw_fields.push(entry);
            }
        }

        let ptr_field_count = ptr_fields.len() as u64;
        let mut fields = Vec::new();
        let mut index = 0u64;

        for mut field in ptr_fields {
            field.index = index;
            index += 1;
            fields.push(field);
        }

        for mut field in raw_fields {
            field.index = index;
            index += 1;
            fields.push(field);
        }

        let field_count = index;
        let type_id = self.next_type_id;
        self.next_type_id += 1;

        Ok(StructLayout {
            field_count,
            ptr_field_count,
            type_id,
            fields,
        })
    }

    fn compute_enum_layout(&mut self, _full_name: &str, e: &EnumDecl) -> Result<EnumLayout<'ctx>, CodegenError> {
        let mut variants = HashMap::new();
        let mut max_ptrs = 0usize;
        let mut max_raw_fields = 0usize; // raw fields excluding tag

        // First pass: find max ptr count and max raw field count across all variants
        for variant in &e.variants {
            let field_specs: Vec<(Option<String>, Type)> = match &variant.kind {
                EnumVariantKind::Unit => Vec::new(),
                EnumVariantKind::Tuple(types) => types.iter().map(|t| (None, t.clone())).collect(),
                EnumVariantKind::Struct(fields) => fields.iter().map(|f| (Some(f.name.clone()), f.ty.clone())).collect(),
            };
            let mut ptr_count = 0usize;
            let mut raw_count = 0usize;
            for (_name, ty) in &field_specs {
                if self.is_gc_ref_type(ty) {
                    ptr_count += 1;
                } else {
                    raw_count += 1;
                }
            }
            if ptr_count > max_ptrs {
                max_ptrs = ptr_count;
            }
            if raw_count > max_raw_fields {
                max_raw_fields = raw_count;
            }
        }

        // Layout: [ptr fields 0..max_ptrs-1] [tag at max_ptrs] [raw fields max_ptrs+1..]
        // total_fields = max_ptrs + 1 (tag) + max_raw_fields
        let tag_index = max_ptrs as u64;
        let field_count = max_ptrs as u64 + 1 + max_raw_fields as u64;

        // Second pass: assign field indices per variant
        for (tag, variant) in e.variants.iter().enumerate() {
            let field_specs: Vec<(Option<String>, Type)> = match &variant.kind {
                EnumVariantKind::Unit => Vec::new(),
                EnumVariantKind::Tuple(types) => types.iter().map(|t| (None, t.clone())).collect(),
                EnumVariantKind::Struct(fields) => fields.iter().map(|f| (Some(f.name.clone()), f.ty.clone())).collect(),
            };
            let mut ptr_idx = 0u64;
            let mut raw_idx = tag_index + 1; // first raw field after tag
            let mut fields = Vec::new();
            for (name, ty) in field_specs {
                let is_gc_ref = self.is_gc_ref_type(&ty);
                let llvm_ty = self.field_llvm_type(&ty)?;
                if is_gc_ref {
                    fields.push(VariantFieldLayout {
                        name,
                        ty: ty.clone(),
                        llvm_ty,
                        is_gc_ref: true,
                        field_index: ptr_idx,
                    });
                    ptr_idx += 1;
                } else {
                    fields.push(VariantFieldLayout {
                        name,
                        ty: ty.clone(),
                        llvm_ty,
                        is_gc_ref: false,
                        field_index: raw_idx,
                    });
                    raw_idx += 1;
                }
            }
            variants.insert(
                variant.name.clone(),
                VariantLayout {
                    tag: tag as u32,
                    fields,
                },
            );
        }

        let type_id = self.next_type_id;
        self.next_type_id += 1;

        Ok(EnumLayout {
            field_count,
            ptr_field_count: max_ptrs as u64,
            type_id,
            variants,
        })
    }

    fn collect_tuple_types(&mut self, ty: &Type) -> Result<(), CodegenError> {
        match ty {
            Type::Tuple(items) => {
                self.ensure_tuple_layout(items)?;
                for item in items {
                    self.collect_tuple_types(item)?;
                }
            }
            Type::RawPointer(inner) => self.collect_tuple_types(inner)?,
            Type::Path(_, _) => {}
        }
        Ok(())
    }

    fn collect_tuple_types_in_block(&mut self, block: &Block) -> Result<(), CodegenError> {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => self.collect_tuple_types_in_expr(expr)?,
                Stmt::Return(expr, _) => {
                    if let Some(expr) = expr {
                        self.collect_tuple_types_in_expr(expr)?;
                    }
                }
            }
        }
        if let Some(tail) = &block.tail {
            self.collect_tuple_types_in_expr(tail)?;
        }
        Ok(())
    }

    fn collect_tuple_types_in_expr(&mut self, expr: &Expr) -> Result<(), CodegenError> {
        match expr {
            Expr::Let { ty, value, .. } => {
                if let Some(ty) = ty {
                    self.collect_tuple_types(ty)?;
                }
                self.collect_tuple_types_in_expr(value)?;
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                self.collect_tuple_types_in_expr(cond)?;
                self.collect_tuple_types_in_block(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.collect_tuple_types_in_block(else_branch)?;
                }
            }
            Expr::While { cond, body, .. } => {
                self.collect_tuple_types_in_expr(cond)?;
                self.collect_tuple_types_in_block(body)?;
            }
            Expr::Match { scrutinee, arms, .. } => {
                self.collect_tuple_types_in_expr(scrutinee)?;
                for arm in arms {
                    self.collect_tuple_types_in_expr(&arm.body)?;
                }
            }
            Expr::Assign { target, value, .. } => {
                self.collect_tuple_types_in_expr(target)?;
                self.collect_tuple_types_in_expr(value)?;
            }
            Expr::Binary { left, right, .. } => {
                self.collect_tuple_types_in_expr(left)?;
                self.collect_tuple_types_in_expr(right)?;
            }
            Expr::Unary { expr, .. } => self.collect_tuple_types_in_expr(expr)?,
            Expr::Call { callee, args, .. } => {
                self.collect_tuple_types_in_expr(callee)?;
                for arg in args {
                    self.collect_tuple_types_in_expr(arg)?;
                }
            }
            Expr::Field { base, .. } => self.collect_tuple_types_in_expr(base)?,
            Expr::StructLit { fields, .. } => {
                for (_, expr) in fields {
                    self.collect_tuple_types_in_expr(expr)?;
                }
            }
            Expr::Tuple { items, .. } => {
                for item in items {
                    self.collect_tuple_types_in_expr(item)?;
                }
            }
            Expr::Block(block) => {
                self.collect_tuple_types_in_block(block)?;
            }
            Expr::Literal(_, _) | Expr::Path(_, _) => {}
            Expr::Break { .. } | Expr::Continue { .. } => {}
        }
        Ok(())
    }

    fn ensure_tuple_layout(&mut self, items: &[Type]) -> Result<String, CodegenError> {
        let key = tuple_key(items);
        if !self.tuples.contains_key(&key) {
            let layout = self.compute_tuple_layout(&key, items)?;
            self.tuples.insert(key.clone(), layout);
        }
        Ok(key)
    }

    fn compute_tuple_layout(&mut self, _name: &str, items: &[Type]) -> Result<StructLayout<'ctx>, CodegenError> {
        let mut ptr_fields = Vec::new();
        let mut raw_fields = Vec::new();

        for (i, ty) in items.iter().enumerate() {
            let is_gc_ref = self.is_gc_ref_type(ty);
            let llvm_ty = self.field_llvm_type(ty)?;
            let entry = FieldLayout {
                name: i.to_string(),
                ty: ty.clone(),
                llvm_ty,
                index: 0,
                is_gc_ref,
            };
            if is_gc_ref {
                ptr_fields.push(entry);
            } else {
                raw_fields.push(entry);
            }
        }

        let ptr_field_count = ptr_fields.len() as u64;
        let mut fields = Vec::new();
        let mut index = 0u64;

        for mut field in ptr_fields {
            field.index = index;
            index += 1;
            fields.push(field);
        }

        for mut field in raw_fields {
            field.index = index;
            index += 1;
            fields.push(field);
        }

        let field_count = index;
        let type_id = self.next_type_id;
        self.next_type_id += 1;

        Ok(StructLayout {
            field_count,
            ptr_field_count,
            type_id,
            fields,
        })
    }

    fn new(context: &'ctx Context, name: &str, mode: CodegenMode) -> Self {
        let module = context.create_module(name);
        let builder = context.create_builder();
        let ptr_ty = context.ptr_type(inkwell::AddressSpace::default());
        let thread_struct_ty = context.struct_type(
            &[
                ptr_ty.into(),               // top_frame
                context.i32_type().into(),   // state
                context.i32_type().into(),   // padding
            ],
            false,
        );
        let thread_ty = thread_struct_ty.ptr_type(inkwell::AddressSpace::default());
        let gc_pollcheck = Self::declare_gc_pollcheck(&module, thread_ty, context);
        let gc_allocate_obj = Self::declare_gc_alloc(&module, context);
        let gc_write_barrier = Self::declare_gc_write_barrier(&module, thread_ty, context);
        Self {
            context,
            module,
            builder,
            thread_struct_ty,
            thread_ty,
            mode,
            functions: HashMap::new(),
            externs: HashMap::new(),
            gc_pollcheck,
            gc_allocate_obj,
            gc_write_barrier,
            structs: HashMap::new(),
            enums: HashMap::new(),
            tuples: HashMap::new(),
            str_lit_id: 0,
            next_type_id: 0,
            compiler_used_globals: Vec::new(),
        }
    }

    fn llvm_fn_name(&self, full_name: &str) -> String {
        if self.mode == CodegenMode::Aot && full_name == "main" {
            "__lang_main".to_string()
        } else {
            mangle_name(full_name)
        }
    }

    fn declare_gc_pollcheck(
        module: &Module<'ctx>,
        thread_ty: inkwell::types::PointerType<'ctx>,
        context: &'ctx Context,
    ) -> FunctionValue<'ctx> {
        let void_ty = context.void_type();
        let ptr_ty = context.ptr_type(inkwell::AddressSpace::default());
        let fn_ty = void_ty.fn_type(&[thread_ty.into(), ptr_ty.into()], false);
        module.add_function("gc_pollcheck_slow", fn_ty, None)
    }

    fn declare_gc_alloc(
        module: &Module<'ctx>,
        context: &'ctx Context,
    ) -> FunctionValue<'ctx> {
        let ptr_ty = context.ptr_type(inkwell::AddressSpace::default());
        let i64_ty = context.i64_type();
        let fn_ty = ptr_ty.fn_type(&[i64_ty.into(), i64_ty.into(), i64_ty.into()], false);
        module.add_function("gc_alloc", fn_ty, None)
    }

    fn declare_gc_write_barrier(
        module: &Module<'ctx>,
        thread_ty: inkwell::types::PointerType<'ctx>,
        context: &'ctx Context,
    ) -> FunctionValue<'ctx> {
        let ptr_ty = context.ptr_type(inkwell::AddressSpace::default());
        let fn_ty = context.void_type().fn_type(&[thread_ty.into(), ptr_ty.into(), ptr_ty.into()], false);
        module.add_function("gc_write_barrier", fn_ty, None)
    }

    fn declare_functions(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            let module_path = module.path.clone().unwrap_or_default();
            for item in &module.items {
                match item {
                    Item::Fn(f) => {
                        let full_name = full_item_name(&module_path, &f.name);
                        let fn_val = self.declare_fn(&full_name, f)?;
                        self.functions.insert(full_name, fn_val);
                    }
                    Item::ExternFn(f) => {
                        let full_name = full_item_name(&module_path, &f.name);
                        let fn_val = self.declare_extern_fn(&full_name, f)?;
                        self.externs.insert(full_name, fn_val);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn declare_fn(&self, full_name: &str, f: &FnDecl) -> Result<FunctionValue<'ctx>, CodegenError> {
        let mut param_tys = Vec::new();
        param_tys.push(self.thread_ty.into());
        for param in &f.params {
            param_tys.push(self.llvm_type(&param.ty)?.into());
        }
        let ret_ty = self.llvm_type(&f.ret_type)?;
        let fn_ty = match ret_ty {
            BasicTypeEnum::IntType(t) => t.fn_type(&param_tys, false),
            BasicTypeEnum::PointerType(t) => t.fn_type(&param_tys, false),
            _ => return Err(CodegenError { message: "unsupported return type".to_string() }),
        };
        let llvm_name = self.llvm_fn_name(full_name);
        Ok(self.module.add_function(&llvm_name, fn_ty, None))
    }

    fn declare_extern_fn(&self, full_name: &str, f: &ExternFnDecl) -> Result<FunctionValue<'ctx>, CodegenError> {
        let symbol = extern_symbol_name(full_name);
        // Reuse existing LLVM function if already declared by another module
        if let Some(existing) = self.module.get_function(symbol) {
            return Ok(existing);
        }
        let mut param_tys = Vec::new();
        for param in &f.params {
            param_tys.push(self.llvm_type(&param.ty)?.into());
        }
        let ret_ty = self.llvm_type(&f.ret_type)?;
        let fn_ty = match ret_ty {
            BasicTypeEnum::IntType(t) => t.fn_type(&param_tys, f.varargs),
            BasicTypeEnum::PointerType(t) => t.fn_type(&param_tys, f.varargs),
            _ => return Err(CodegenError { message: "unsupported extern return type".to_string() }),
        };
        Ok(self.module.add_function(symbol, fn_ty, None))
    }

    fn define_functions(&mut self, modules: &[crate::ast::Module]) -> Result<(), CodegenError> {
        for module in modules {
            let module_path = module.path.clone().unwrap_or_default();
            for item in &module.items {
                if let Item::Fn(f) = item {
                    let full_name = full_item_name(&module_path, &f.name);
                    self.define_fn(&full_name, f)?;
                }
            }
        }
        Ok(())
    }

    fn define_fn(&mut self, full_name: &str, f: &FnDecl) -> Result<(), CodegenError> {
        let function = *self
            .functions
            .get(full_name)
            .ok_or_else(|| CodegenError { message: "missing function".to_string() })?;
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        let mut locals: HashMap<String, Local<'ctx>> = HashMap::new();
        let thread = function.get_nth_param(0).unwrap().into_pointer_value();

        let num_roots = self.count_roots_in_fn(f);
        let frame_ty = self.frame_type(num_roots);
        let (frame_ptr, frame_origin, root_base) = self.emit_prologue(thread, full_name, num_roots, frame_ty)?;

        let mut param_root_index = 0usize;
        for (i, param) in f.params.iter().enumerate() {
            let local = self.create_local_storage_in_ctx(
                function,
                &param.name,
                &param.ty,
                root_base,
                param_root_index,
            )?;
            let arg = function.get_nth_param((i + 1) as u32).unwrap();
            self.store_local(&local, arg)?;
            locals.insert(param.name.clone(), local);
            if self.is_gc_ref_type(&param.ty) {
                param_root_index += 1;
            }
        }

        let mut ctx = FnCtx {
            function,
            locals,
            scopes: Vec::new(),
            thread,
            frame_ptr,
            frame_origin,
            frame_ty,
            root_base,
            next_root: self.count_param_roots(&f.params),
            loop_stack: Vec::new(),
        };

        let value = self.codegen_block_value(&f.body, &mut ctx, Some(&f.ret_type))
            .map_err(|e| CodegenError { message: format!("{} (in fn {})", e.message, full_name) })?;
        let terminated = self
            .builder
            .get_insert_block()
            .and_then(|bb| bb.get_terminator())
            .is_some();
        if !terminated {
            let ret_val: BasicValueEnum = match value {
                Some(v) => v,
                None => match self.llvm_type(&f.ret_type)? {
                    BasicTypeEnum::IntType(_) => self.context.i64_type().const_int(0, false).into(),
                    _ => return Err(CodegenError { message: "unsupported return type".to_string() }),
                },
            };
            self.emit_epilogue(&ctx)?;
            self.map_builder(self.builder.build_return(Some(&ret_val)), "return")?;
        }
        Ok(())
    }

    fn emit_aot_wrapper_main(&mut self) -> Result<(), CodegenError> {
        if self.mode != CodegenMode::Aot {
            return Ok(());
        }
        let lang_main = *self
            .functions
            .get("main")
            .ok_or_else(|| CodegenError { message: "missing main function".to_string() })?;

        let i32_ty = self.context.i32_type();
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let fn_ty = i32_ty.fn_type(&[i32_ty.into(), ptr_ty.into()], false);
        let wrapper = self.module.add_function("main", fn_ty, None);
        let entry = self.context.append_basic_block(wrapper, "entry");
        self.builder.position_at_end(entry);

        let argc = wrapper.get_nth_param(0).unwrap();
        let argv = wrapper.get_nth_param(1).unwrap();
        let init_ty = self.context.void_type().fn_type(&[i32_ty.into(), ptr_ty.into()], false);
        let init_fn = self
            .module
            .get_function("rt_init_args")
            .unwrap_or_else(|| self.module.add_function("rt_init_args", init_ty, None));
        self.map_builder(
            self.builder.build_call(init_fn, &[argc.into(), argv.into()], "init_args"),
            "call",
        )?;

        // Call gc_init() before any allocations
        let gc_init_ty = self.context.void_type().fn_type(&[], false);
        let gc_init_fn = self
            .module
            .get_function("gc_init")
            .unwrap_or_else(|| self.module.add_function("gc_init", gc_init_ty, None));
        self.map_builder(
            self.builder.build_call(gc_init_fn, &[], "gc_init"),
            "call",
        )?;

        let thread_alloca = self.map_builder(self.builder.build_alloca(self.thread_struct_ty, "thread"), "alloca")?;
        let top_ptr = self.map_builder(
            self.builder
                .build_struct_gep(self.thread_struct_ty, thread_alloca, 0, "thread.top"),
            "gep",
        )?;
        let state_ptr = self.map_builder(
            self.builder
                .build_struct_gep(self.thread_struct_ty, thread_alloca, 1, "thread.state"),
            "gep",
        )?;
        let pad_ptr = self.map_builder(
            self.builder
                .build_struct_gep(self.thread_struct_ty, thread_alloca, 2, "thread.pad"),
            "gep",
        )?;
        self.map_builder(self.builder.build_store(top_ptr, ptr_ty.const_null()), "store")?;
        self.map_builder(self.builder.build_store(state_ptr, i32_ty.const_zero()), "store")?;
        self.map_builder(self.builder.build_store(pad_ptr, i32_ty.const_zero()), "store")?;

        let thread_ptr = self.map_builder(self.builder.build_bit_cast(thread_alloca, ptr_ty, "thread_ptr"), "bitcast")?;

        // Call gc_set_thread(thread_ptr)
        let gc_set_thread_ty = self.context.void_type().fn_type(&[ptr_ty.into()], false);
        let gc_set_thread_fn = self
            .module
            .get_function("gc_set_thread")
            .unwrap_or_else(|| self.module.add_function("gc_set_thread", gc_set_thread_ty, None));
        self.map_builder(
            self.builder.build_call(gc_set_thread_fn, &[thread_ptr.into()], "set_thread"),
            "call",
        )?;

        let call = self.map_builder(self.builder.build_call(lang_main, &[thread_ptr.into()], "lang_main"), "call")?;
        let ret_val = call
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CodegenError { message: "main returned void".to_string() })?;
        let ret_i32 = self.map_builder(
            self.builder
                .build_int_truncate(ret_val.into_int_value(), i32_ty, "ret"),
            "trunc",
        )?;
        self.map_builder(self.builder.build_return(Some(&ret_i32)), "return")?;
        Ok(())
    }

    fn codegen_block(&mut self, block: &Block, ctx: &mut FnCtx<'ctx>, expected: Option<&Type>) -> Result<bool, CodegenError> {
        self.push_scope(ctx);
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => {
                    let _ = self.codegen_expr(expr, ctx, None)?;
                }
                Stmt::Return(expr, _) => {
                    let value = if let Some(expr) = expr {
                        self.codegen_expr(expr, ctx, expected)?
                    } else {
                        None
                    };
                    if let Some(v) = value {
                        self.emit_epilogue(ctx)?;
                        self.map_builder(self.builder.build_return(Some(&v)), "return")?;
                    } else {
                        let zero = self.context.i64_type().const_int(0, false);
                        self.emit_epilogue(ctx)?;
                        self.map_builder(self.builder.build_return(Some(&zero)), "return")?;
                    }
                    self.pop_scope(ctx);
                    return Ok(true);
                }
            }
        }
        if let Some(tail) = &block.tail {
            let _ = self.codegen_expr(tail, ctx, expected)?;
        }
        self.pop_scope(ctx);
        Ok(false)
    }

    fn codegen_expr(
        &mut self,
        expr: &Expr,
        ctx: &mut FnCtx<'ctx>,
        expected: Option<&Type>,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        match expr {
            Expr::Let { name, ty, value, .. } => {
                let val = self.codegen_expr(value, ctx, ty.as_ref().or(expected))?;
                let ty = ty.as_ref().or(expected).ok_or_else(|| CodegenError { message: "let requires type".to_string() })?;
                let local = self.create_local_storage_in_ctx(ctx.function, name, ty, ctx.root_base, ctx.next_root)?;
                if let Some(v) = val {
                    self.store_local(&local, v)?;
                }
                ctx.locals.insert(name.clone(), local);
                if let Some(scope) = ctx.scopes.last_mut() {
                    scope.push(name.clone());
                }
                if self.is_gc_ref_type(ty) {
                    ctx.next_root += 1;
                }
                Ok(None)
            }
            Expr::Assign { target, value, .. } => {
                let v = self.codegen_expr(value, ctx, None)?;
                match &**target {
                    Expr::Path(path, _) => {
                        let name = path.last().ok_or_else(|| CodegenError { message: "invalid assignment".to_string() })?;
                        let local = ctx.locals.get(name).ok_or_else(|| CodegenError { message: format!("unknown local '{}' in assign", name) })?;
                        if let Some(v) = v {
                            self.store_local(local, v)?;
                        }
                        Ok(None)
                    }
                    Expr::Field { base, name, .. } => {
                        let (obj_ptr, _layout, field) = self.resolve_field(base, name, ctx)?;
                        let field_ptr = self.field_ptr(obj_ptr, &field)?;
                        if let Some(v) = v {
                            self.store_gc_field(field_ptr, &field, v)?;
                            if self.is_gc_ref_type(&field.ty) {
                                let args = &[ctx.thread.into(), obj_ptr.into(), v.into()];
                                let _ = self.map_builder(self.builder.build_call(self.gc_write_barrier, args, "wb"), "call")?;
                            }
                        }
                        Ok(None)
                    }
                    _ => Err(CodegenError { message: "unsupported assignment target".to_string() }),
                }
            }
            Expr::Literal(lit, _) => match lit {
                Literal::Int(text) => Ok(Some(self.const_i64(text)?)),
                Literal::Char(value) => Ok(Some(self.context.i64_type().const_int(*value as u64, false).into())),
                Literal::Float(text) => {
                    let is_f32 = matches!(
                        expected,
                        Some(Type::Path(path, _)) if path.last().map(|s| s.as_str()) == Some("F32")
                    );
                    if is_f32 {
                        let value = text.parse::<f32>().map_err(|_| CodegenError { message: "invalid float literal".to_string() })?;
                        Ok(Some(self.context.f32_type().const_float(value as f64).into()))
                    } else {
                        let value = text.parse::<f64>().map_err(|_| CodegenError { message: "invalid float literal".to_string() })?;
                        Ok(Some(self.context.f64_type().const_float(value).into()))
                    }
                }
                Literal::Bool(b) => Ok(Some(self.context.bool_type().const_int(*b as u64, false).into())),
                Literal::Unit => Ok(Some(self.context.i64_type().const_zero().into())),
                Literal::Str(text) => {
                    let bytes = text.as_bytes();
                    let const_str = self.context.const_string(bytes, true);
                    let name = format!("__str_lit_{}", self.str_lit_id);
                    self.str_lit_id += 1;
                    let global = self
                        .module
                        .add_global(const_str.get_type(), None, &name);
                    global.set_initializer(&const_str);
                    let zero = self.context.i32_type().const_zero();
                    let ptr = unsafe {
                        global
                            .as_pointer_value()
                            .const_gep(const_str.get_type(), &[zero, zero])
                    };
                    Ok(Some(ptr.into()))
                }
                _ => Err(CodegenError { message: "unsupported literal".to_string() }),
            },
            Expr::Path(path, _) => {
                let name = path.last().ok_or_else(|| CodegenError { message: "empty path".to_string() })?;
                let local = ctx.locals.get(name).ok_or_else(|| CodegenError { message: format!("unknown local '{}' in path expr", name) })?;
                let loaded = self.load_local(local, name)?;
                Ok(Some(loaded))
            }
            Expr::Field { base, name, .. } => {
                let (obj_ptr, _layout, field) = self.resolve_field(base, name, ctx)?;
                let field_ptr = self.field_ptr(obj_ptr, &field)?;
                let loaded = self.load_gc_field(field_ptr, &field)?;
                Ok(Some(loaded))
            }
            Expr::Binary { op, left, right, .. } => {
                let l = self.codegen_expr(left, ctx, None)?.ok_or_else(|| CodegenError { message: "missing lhs".to_string() })?;
                let r = self.codegen_expr(right, ctx, None)?.ok_or_else(|| CodegenError { message: "missing rhs".to_string() })?;
                Ok(Some(self.codegen_binary(op, l, r)?))
            }
            Expr::Unary { op, expr, .. } => {
                let v = self.codegen_expr(expr, ctx, None)?.ok_or_else(|| CodegenError { message: "missing operand".to_string() })?;
                match op {
                    UnaryOp::Neg => {
                        if v.is_float_value() {
                            let value = self.map_builder(self.builder.build_float_neg(v.into_float_value(), "fneg"), "fneg")?;
                            Ok(Some(value.into()))
                        } else {
                            let zero = self.context.i64_type().const_int(0, false);
                            let value = self.map_builder(self.builder.build_int_sub(zero, v.into_int_value(), "neg"), "neg")?;
                            Ok(Some(value.into()))
                        }
                    }
                    UnaryOp::Not => {
                        let one = self.context.bool_type().const_int(1, false);
                        let value = self.map_builder(self.builder.build_xor(v.into_int_value(), one, "not"), "not")?;
                        Ok(Some(value.into()))
                    }
                }
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                let cond_val = self.codegen_expr(cond, ctx, None)?.ok_or_else(|| CodegenError { message: "missing cond".to_string() })?;
                let func = ctx.function;
                let then_bb = self.context.append_basic_block(func, "then");
                let else_bb = self.context.append_basic_block(func, "else");
                let cont_bb = self.context.append_basic_block(func, "cont");
                self.map_builder(
                    self.builder
                        .build_conditional_branch(cond_val.into_int_value(), then_bb, else_bb),
                    "condbr",
                )?;

                self.builder.position_at_end(then_bb);
                let then_val = self.codegen_block_value(then_branch, ctx, expected)?;
                let then_val_bb = self.builder.get_insert_block().unwrap();
                if !then_val_bb.get_terminator().is_some() {
                    self.map_builder(self.builder.build_unconditional_branch(cont_bb), "br")?;
                }

                self.builder.position_at_end(else_bb);
                let else_val = if let Some(else_branch) = else_branch {
                    self.codegen_block_value(else_branch, ctx, expected)?
                } else {
                    None
                };
                let else_val_bb = self.builder.get_insert_block().unwrap();
                if !else_val_bb.get_terminator().is_some() {
                    self.map_builder(self.builder.build_unconditional_branch(cont_bb), "br")?;
                }

                self.builder.position_at_end(cont_bb);
                if let (Some(tv), Some(ev)) = (then_val, else_val) {
                    let phi = self.map_builder(self.builder.build_phi(tv.get_type(), "iftmp"), "phi")?;
                    phi.add_incoming(&[(&tv, then_val_bb), (&ev, else_val_bb)]);
                    Ok(Some(phi.as_basic_value()))
                } else {
                    Ok(None)
                }
            }
            Expr::While { cond, body, .. } => {
                let func = ctx.function;
                let cond_bb = self.context.append_basic_block(func, "while.cond");
                let body_bb = self.context.append_basic_block(func, "while.body");
                let cont_bb = self.context.append_basic_block(func, "while.cont");
                self.map_builder(self.builder.build_unconditional_branch(cond_bb), "br")?;

                self.builder.position_at_end(cond_bb);
                let cond_val = self.codegen_expr(cond, ctx, None)?.ok_or_else(|| CodegenError { message: "missing cond".to_string() })?;
                self.map_builder(
                    self.builder
                        .build_conditional_branch(cond_val.into_int_value(), body_bb, cont_bb),
                    "condbr",
                )?;

                self.builder.position_at_end(body_bb);
                ctx.loop_stack.push((cond_bb, cont_bb));
                let _ = self.codegen_block(body, ctx, None)?;
                ctx.loop_stack.pop();
                if !self.builder.get_insert_block().unwrap().get_terminator().is_some() {
                    self.emit_pollcheck(ctx)?;
                    self.map_builder(self.builder.build_unconditional_branch(cond_bb), "br")?;
                }

                self.builder.position_at_end(cont_bb);
                Ok(None)
            }
            Expr::Match { scrutinee, arms, .. } => {
                let mut enum_name = None;
                for arm in arms {
                    match &arm.pattern {
                        Pattern::Path(path, _) | Pattern::Struct { path, .. } => {
                            if path.len() >= 2 {
                                let (enum_path, _) = enum_path_and_variant(path);
                                enum_name = Some(enum_path);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let enum_name = enum_name.ok_or_else(|| CodegenError { message: "match missing enum pattern".to_string() })?;
                let enum_layout = self
                    .enums
                    .get(&enum_name)
                    .cloned()
                    .ok_or_else(|| CodegenError { message: "unknown enum layout".to_string() })?;
                let obj_val = self.codegen_expr(scrutinee, ctx, None)?.ok_or_else(|| CodegenError { message: "missing match scrutinee".to_string() })?;
                let obj_ptr = obj_val.into_pointer_value();
                let tag_val = self.load_enum_tag(obj_ptr, &enum_layout)?;
                let func = ctx.function;
                let cont_bb = self.context.append_basic_block(func, "match.cont");
                let mut default_bb = cont_bb;
                let switch_bb = self.builder.get_insert_block().unwrap();
                let mut cases: Vec<(inkwell::values::IntValue<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();
                let mut arm_blocks = Vec::new();

                for arm in arms {
                    let arm_bb = self.context.append_basic_block(func, "match.arm");
                    match &arm.pattern {
                        Pattern::Wildcard(_) => {
                            default_bb = arm_bb;
                        }
                        Pattern::Path(path, _) => {
                            let variant = path.last().ok_or_else(|| CodegenError { message: "invalid pattern".to_string() })?;
                            let vlayout = enum_layout
                                .variants
                                .get(variant)
                                .ok_or_else(|| CodegenError { message: "unknown variant".to_string() })?;
                            let tag = self.context.i32_type().const_int(vlayout.tag as u64, false);
                            cases.push((tag, arm_bb));
                        }
                        Pattern::Struct { path, .. } => {
                            let variant = path.last().ok_or_else(|| CodegenError { message: "invalid pattern".to_string() })?;
                            let vlayout = enum_layout
                                .variants
                                .get(variant)
                                .ok_or_else(|| CodegenError { message: "unknown variant".to_string() })?;
                            let tag = self.context.i32_type().const_int(vlayout.tag as u64, false);
                            cases.push((tag, arm_bb));
                        }
                    }
                    arm_blocks.push((arm_bb, arm));
                }

                self.map_builder(
                    self.builder.build_switch(tag_val, default_bb, &cases),
                    "switch",
                )?;

                let mut incoming = Vec::new();
                for (bb, arm) in arm_blocks {
                    self.builder.position_at_end(bb);
                    self.push_scope(ctx);
                    if let Pattern::Struct { path, fields, .. } = &arm.pattern {
                        let variant_name = path.last().ok_or_else(|| CodegenError { message: "invalid pattern".to_string() })?;
                        let variant = enum_layout
                            .variants
                            .get(variant_name)
                            .ok_or_else(|| CodegenError { message: "unknown variant".to_string() })?;
                        for pat_field in fields {
                            let binding = match &pat_field.binding {
                                Some(b) => b,
                                None => continue,
                            };
                            let field = variant
                                .fields
                                .iter()
                                .find(|f| f.name.as_deref() == Some(pat_field.name.as_str()))
                                .ok_or_else(|| CodegenError { message: "unknown field".to_string() })?;
                            let fld_ptr = self.enum_field_ptr(obj_ptr, field.field_index)?;
                            let loaded = if field.is_gc_ref {
                                let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                                self.map_builder(
                                    self.builder.build_load(ptr_ty, fld_ptr, "enum_field"),
                                    "load",
                                )?
                            } else {
                                let load_ty = self.context.i64_type();
                                let raw = self.map_builder(self.builder.build_load(load_ty, fld_ptr, "enum_field"), "load")?;
                                if self.is_bool_type(&field.ty) {
                                    let zero = self.context.i64_type().const_zero();
                                    let cmp = self.map_builder(
                                        self.builder
                                            .build_int_compare(inkwell::IntPredicate::NE, raw.into_int_value(), zero, "bool"),
                                        "icmp",
                                    )?;
                                    cmp.into()
                                } else {
                                    raw
                                }
                            };
                            let local = self.create_local_storage_in_ctx(ctx.function, binding, &field.ty, ctx.root_base, ctx.next_root)?;
                            self.store_local(&local, loaded)?;
                            if let Some(scope) = ctx.scopes.last_mut() {
                                scope.push(binding.clone());
                            }
                            ctx.locals.insert(binding.clone(), local);
                            if self.is_gc_ref_type(&field.ty) {
                                ctx.next_root += 1;
                            }
                        }
                    }
                    let value = self.codegen_expr(&arm.body, ctx, expected)?;
                    let value_bb = self.builder.get_insert_block().unwrap();
                    let terminated = value_bb.get_terminator().is_some();
                    if !terminated {
                        self.map_builder(self.builder.build_unconditional_branch(cont_bb), "br")?;
                    }
                    if let Some(v) = value {
                        incoming.push((v, value_bb));
                    }
                    self.pop_scope(ctx);
                }

                self.builder.position_at_end(cont_bb);
                if default_bb == cont_bb && !incoming.is_empty() {
                    let zero = self.zero_value(incoming[0].0.get_type())?;
                    incoming.push((zero, switch_bb));
                }
                if !incoming.is_empty() {
                    let ty = incoming[0].0.get_type();
                    let phi = self.map_builder(self.builder.build_phi(ty, "matchtmp"), "phi")?;
                    let mut pairs: Vec<(&dyn BasicValue<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> = Vec::new();
                    for (v, bb) in &incoming {
                        pairs.push((v as &dyn BasicValue, *bb));
                    }
                    phi.add_incoming(&pairs);
                    Ok(Some(phi.as_basic_value()))
                } else {
                    Ok(None)
                }
            }
            Expr::Call { callee, args, .. } => {
                if let Expr::Path(path, _) = &**callee {
                    if path.len() >= 2 {
                        let (enum_name, variant_name) = enum_path_and_variant(path);
                        if let Some(layout) = self.enums.get(&enum_name).cloned() {
                            let variant = layout
                                .variants
                                .get(&variant_name)
                                .cloned()
                                .ok_or_else(|| CodegenError { message: "unknown enum variant".to_string() })?;
                            if variant.fields.iter().any(|f| f.name.is_some()) {
                                return Err(CodegenError { message: "struct variant requires field syntax".to_string() });
                            }
                            let i64_ty = self.context.i64_type();
                            let total_fields = i64_ty.const_int(layout.field_count, false);
                            let ptr_fields = i64_ty.const_int(layout.ptr_field_count, false);
                            let type_id_val = i64_ty.const_int(layout.type_id as u64, false);
                            let args_alloc = &[total_fields.into(), ptr_fields.into(), type_id_val.into()];
                            let call = self.map_builder(self.builder.build_call(self.gc_allocate_obj, args_alloc, "alloc_enum"), "call")?;
                            let obj_ptr = call
                                .try_as_basic_value()
                                .basic()
                                .ok_or_else(|| CodegenError { message: "alloc returned void".to_string() })?
                                .into_pointer_value();

                            // Write tag as i64 at tag field index
                            let tag_index = layout.ptr_field_count;
                            let tag_ptr = self.gc_field_ptr(obj_ptr, tag_index)?;
                            let tag_val = i64_ty.const_int(variant.tag as u64, false);
                            self.map_builder(self.builder.build_store(tag_ptr, tag_val), "store")?;

                            if args.len() != variant.fields.len() {
                                return Err(CodegenError { message: "arg count mismatch".to_string() });
                            }
                            for (i, arg) in args.iter().enumerate() {
                                let arg_val = self.codegen_expr(arg, ctx, None)?.ok_or_else(|| CodegenError { message: "missing arg".to_string() })?;
                                let arg_layout = variant.fields.get(i).ok_or_else(|| CodegenError { message: "arg count mismatch".to_string() })?;
                                let fld_ptr = self.enum_field_ptr(obj_ptr, arg_layout.field_index)?;
                                let store_val = if self.is_bool_type(&arg_layout.ty) {
                                    let zext = self.map_builder(
                                        self.builder.build_int_z_extend(arg_val.into_int_value(), i64_ty, "boolz"),
                                        "zext",
                                    )?;
                                    zext.into()
                                } else {
                                    arg_val
                                };
                                self.map_builder(self.builder.build_store(fld_ptr, store_val), "store")?;
                            }
                            self.emit_pollcheck(ctx)?;
                            return Ok(Some(obj_ptr.into()));
                        }
                    }
                }

                let (func, pass_thread) = match &**callee {
                    Expr::Path(path, _) => {
                        let key = path_to_string(path);
                        if let Some(f) = self.functions.get(&key) {
                            (*f, true)
                        } else if let Some(f) = self.externs.get(&key) {
                            (*f, false)
                        } else {
                            return Err(CodegenError { message: "unknown function".to_string() });
                        }
                    }
                    _ => return Err(CodegenError { message: "callee must be path".to_string() }),
                };
                let mut arg_vals: Vec<BasicMetadataValueEnum> = Vec::new();
                if pass_thread {
                    arg_vals.push(ctx.thread.into());
                }
                for arg in args {
                    let v = self.codegen_expr(arg, ctx, None)?.ok_or_else(|| CodegenError { message: "missing arg".to_string() })?;
                    arg_vals.push(v.into());
                }
                let call = self.map_builder(self.builder.build_call(func, &arg_vals, "call"), "call")?;
                self.emit_pollcheck(ctx)?;
                Ok(call.try_as_basic_value().basic())
            }
            Expr::StructLit { path, fields, .. } => {
                if path.len() >= 2 {
                    let (enum_name, variant_name) = enum_path_and_variant(path);
                    if let Some(layout) = self.enums.get(&enum_name).cloned() {
                        let variant = layout
                            .variants
                            .get(&variant_name)
                            .cloned()
                            .ok_or_else(|| CodegenError { message: "unknown enum variant".to_string() })?;
                        if variant.fields.iter().all(|f| f.name.is_some()) {
                            let i64_ty = self.context.i64_type();
                            let total_fields = i64_ty.const_int(layout.field_count, false);
                            let ptr_fields_val = i64_ty.const_int(layout.ptr_field_count, false);
                            let type_id_val = i64_ty.const_int(layout.type_id as u64, false);
                            let args_alloc = &[total_fields.into(), ptr_fields_val.into(), type_id_val.into()];
                            let call = self.map_builder(self.builder.build_call(self.gc_allocate_obj, args_alloc, "alloc_enum"), "call")?;
                            let obj_ptr = call
                                .try_as_basic_value()
                                .basic()
                                .ok_or_else(|| CodegenError { message: "alloc returned void".to_string() })?
                                .into_pointer_value();

                            // Write tag as i64 at tag field index
                            let tag_index = layout.ptr_field_count;
                            let tag_ptr = self.gc_field_ptr(obj_ptr, tag_index)?;
                            let tag_val = i64_ty.const_int(variant.tag as u64, false);
                            self.map_builder(self.builder.build_store(tag_ptr, tag_val), "store")?;

                            for (field_name, field_expr) in fields {
                                let field = variant
                                    .fields
                                    .iter()
                                    .find(|f| f.name.as_deref() == Some(field_name.as_str()))
                                    .cloned()
                                    .ok_or_else(|| CodegenError { message: "unknown field".to_string() })?;
                                let value = self.codegen_expr(field_expr, ctx, Some(&field.ty))?.ok_or_else(|| CodegenError { message: "missing field value".to_string() })?;
                                let fld_ptr = self.enum_field_ptr(obj_ptr, field.field_index)?;
                                let store_val = if self.is_bool_type(&field.ty) {
                                    let zext = self.map_builder(
                                        self.builder.build_int_z_extend(value.into_int_value(), i64_ty, "boolz"),
                                        "zext",
                                    )?;
                                    zext.into()
                                } else {
                                    value
                                };
                                self.map_builder(self.builder.build_store(fld_ptr, store_val), "store")?;
                            }
                            self.emit_pollcheck(ctx)?;
                            return Ok(Some(obj_ptr.into()));
                        }
                    }
                }

                let name = path_to_string(path);
                let layout = self
                    .structs
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| CodegenError { message: "unknown struct".to_string() })?;
                let i64_ty = self.context.i64_type();
                let total_fields = i64_ty.const_int(layout.field_count, false);
                let ptr_fields_val = i64_ty.const_int(layout.ptr_field_count, false);
                let type_id_val = i64_ty.const_int(layout.type_id as u64, false);
                let alloc_args = &[total_fields.into(), ptr_fields_val.into(), type_id_val.into()];
                let call = self.map_builder(self.builder.build_call(self.gc_allocate_obj, alloc_args, "alloc"), "call")?;
                let obj_ptr = call
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError { message: "alloc returned void".to_string() })?
                    .into_pointer_value();
                for (field_name, field_expr) in fields {
                    let field = layout
                        .fields
                        .iter()
                        .find(|f| f.name == *field_name)
                        .cloned()
                        .ok_or_else(|| CodegenError { message: "unknown field".to_string() })?;
                    let value = self.codegen_expr(field_expr, ctx, Some(&field.ty))?.ok_or_else(|| CodegenError { message: "missing field value".to_string() })?;
                    let field_ptr = self.field_ptr(obj_ptr, &field)?;
                    self.store_gc_field(field_ptr, &field, value)?;
                }
                self.emit_pollcheck(ctx)?;
                Ok(Some(obj_ptr.into()))
            }
            Expr::Tuple { items, .. } => {
                let expected = expected.ok_or_else(|| CodegenError { message: "tuple requires type context".to_string() })?;
                let elem_types = match expected {
                    Type::Tuple(types) => types,
                    _ => return Err(CodegenError { message: "tuple requires tuple type".to_string() }),
                };
                if elem_types.len() != items.len() {
                    return Err(CodegenError { message: "tuple arity mismatch".to_string() });
                }
                let key = self.ensure_tuple_layout(elem_types)?;
                let layout = self
                    .tuples
                    .get(&key)
                    .ok_or_else(|| CodegenError { message: "unknown tuple layout".to_string() })?
                    .clone();
                let i64_ty = self.context.i64_type();
                let total_fields = i64_ty.const_int(layout.field_count, false);
                let ptr_fields_val = i64_ty.const_int(layout.ptr_field_count, false);
                let type_id_val = i64_ty.const_int(layout.type_id as u64, false);
                let alloc_args = &[total_fields.into(), ptr_fields_val.into(), type_id_val.into()];
                let call = self.map_builder(self.builder.build_call(self.gc_allocate_obj, alloc_args, "alloc_tuple"), "call")?;
                let obj_ptr = call
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CodegenError { message: "alloc returned void".to_string() })?
                    .into_pointer_value();
                for (i, item) in items.iter().enumerate() {
                    let field_name = i.to_string();
                    let field = layout
                        .fields
                        .iter()
                        .find(|f| f.name == field_name)
                        .cloned()
                        .ok_or_else(|| CodegenError { message: "unknown tuple field".to_string() })?;
                    let value = self.codegen_expr(item, ctx, Some(&field.ty))?.ok_or_else(|| CodegenError { message: "missing tuple value".to_string() })?;
                    let field_ptr = self.field_ptr(obj_ptr, &field)?;
                    self.store_gc_field(field_ptr, &field, value)?;
                    if field.is_gc_ref {
                        let args = &[ctx.thread.into(), obj_ptr.into(), value.into()];
                        let _ = self.map_builder(self.builder.build_call(self.gc_write_barrier, args, "wb"), "call")?;
                    }
                }
                self.emit_pollcheck(ctx)?;
                Ok(Some(obj_ptr.into()))
            }
            Expr::Block(block) => self.codegen_block_value(block, ctx, expected),
            Expr::Break { .. } => {
                let (_, break_bb) = ctx.loop_stack.last().ok_or_else(|| CodegenError { message: "break outside of loop".to_string() })?;
                let break_bb = *break_bb;
                self.map_builder(self.builder.build_unconditional_branch(break_bb), "br")?;
                Ok(None)
            }
            Expr::Continue { .. } => {
                let (continue_bb, _) = ctx.loop_stack.last().ok_or_else(|| CodegenError { message: "continue outside of loop".to_string() })?;
                let continue_bb = *continue_bb;
                self.emit_pollcheck(ctx)?;
                self.map_builder(self.builder.build_unconditional_branch(continue_bb), "br")?;
                Ok(None)
            }
            _ => Err(CodegenError { message: "unsupported expression".to_string() }),
        }
    }

    fn codegen_block_value(
        &mut self,
        block: &Block,
        ctx: &mut FnCtx<'ctx>,
        expected: Option<&Type>,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        self.push_scope(ctx);
        let mut last = None;
        let mut terminated = false;
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => {
                    last = self.codegen_expr(expr, ctx, expected)?;
                }
                Stmt::Return(expr, _) => {
                    let val = if let Some(expr) = expr {
                        self.codegen_expr(expr, ctx, expected)?
                    } else {
                        None
                    };
                    if let Some(v) = val {
                        self.emit_epilogue(ctx)?;
                        self.map_builder(self.builder.build_return(Some(&v)), "return")?;
                    } else {
                        let zero = self.context.i64_type().const_int(0, false);
                        self.emit_epilogue(ctx)?;
                        self.map_builder(self.builder.build_return(Some(&zero)), "return")?;
                    }
                    terminated = true;
                    break;
                }
            }
        }
        if !terminated {
            if let Some(tail) = &block.tail {
                last = self.codegen_expr(tail, ctx, expected)?;
            }
        }
        self.pop_scope(ctx);
        Ok(last)
    }

    fn codegen_binary(
        &self,
        op: &BinaryOp,
        left: BasicValueEnum<'ctx>,
        right: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        if left.is_float_value() || right.is_float_value() {
            let l = left.into_float_value();
            let r = right.into_float_value();
            let res = match op {
                BinaryOp::Add => self.map_builder(self.builder.build_float_add(l, r, "fadd"), "fadd")?.into(),
                BinaryOp::Sub => self.map_builder(self.builder.build_float_sub(l, r, "fsub"), "fsub")?.into(),
                BinaryOp::Mul => self.map_builder(self.builder.build_float_mul(l, r, "fmul"), "fmul")?.into(),
                BinaryOp::Div => self.map_builder(self.builder.build_float_div(l, r, "fdiv"), "fdiv")?.into(),
                BinaryOp::Rem => self.map_builder(self.builder.build_float_rem(l, r, "frem"), "frem")?.into(),
                BinaryOp::Eq => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OEQ, l, r, "feq"),
                    "feq",
                )?
                .into(),
                BinaryOp::NotEq => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::ONE, l, r, "fne"),
                    "fne",
                )?
                .into(),
                BinaryOp::Lt => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OLT, l, r, "flt"),
                    "flt",
                )?
                .into(),
                BinaryOp::LtEq => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OLE, l, r, "fle"),
                    "fle",
                )?
                .into(),
                BinaryOp::Gt => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OGT, l, r, "fgt"),
                    "fgt",
                )?
                .into(),
                BinaryOp::GtEq => self.map_builder(
                    self.builder.build_float_compare(inkwell::FloatPredicate::OGE, l, r, "fge"),
                    "fge",
                )?
                .into(),
                BinaryOp::AndAnd | BinaryOp::OrOr => {
                    return Err(CodegenError { message: "invalid float logical op".to_string() })
                }
            };
            Ok(res)
        } else {
            let l = left.into_int_value();
            let r = right.into_int_value();
            let res = match op {
                BinaryOp::Add => self.map_builder(self.builder.build_int_add(l, r, "add"), "add")?,
                BinaryOp::Sub => self.map_builder(self.builder.build_int_sub(l, r, "sub"), "sub")?,
                BinaryOp::Mul => self.map_builder(self.builder.build_int_mul(l, r, "mul"), "mul")?,
                BinaryOp::Div => self.map_builder(self.builder.build_int_signed_div(l, r, "div"), "div")?,
                BinaryOp::Rem => self.map_builder(self.builder.build_int_signed_rem(l, r, "rem"), "rem")?,
                BinaryOp::Eq => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::EQ, l, r, "eq"), "eq")?,
                BinaryOp::NotEq => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::NE, l, r, "ne"), "ne")?,
                BinaryOp::Lt => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::SLT, l, r, "lt"), "lt")?,
                BinaryOp::LtEq => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::SLE, l, r, "le"), "le")?,
                BinaryOp::Gt => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::SGT, l, r, "gt"), "gt")?,
                BinaryOp::GtEq => self.map_builder(self.builder.build_int_compare(inkwell::IntPredicate::SGE, l, r, "ge"), "ge")?,
                BinaryOp::AndAnd => self.map_builder(self.builder.build_and(l, r, "and"), "and")?,
                BinaryOp::OrOr => self.map_builder(self.builder.build_or(l, r, "or"), "or")?,
            };
            Ok(res.into())
        }
    }

    fn llvm_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, CodegenError> {
        match ty {
            Type::Path(path, _) => {
                let name = path.last().ok_or_else(|| CodegenError { message: "empty type".to_string() })?;
                match name.as_str() {
                    "I64" => Ok(self.context.i64_type().into()),
                    "F32" => Ok(self.context.f32_type().into()),
                    "F64" => Ok(self.context.f64_type().into()),
                    "Bool" => Ok(self.context.bool_type().into()),
                    "Unit" => Ok(self.context.i64_type().into()),
                    "String" => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
                    _ if self.is_gc_ref_type(ty) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
                    _ => Err(CodegenError { message: "unsupported type".to_string() }),
                }
            }
            Type::RawPointer(_) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
            Type::Tuple(_) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
        }
    }

    fn const_i64(&self, text: &str) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        let cleaned = text.replace('_', "");
        let value = cleaned.parse::<i64>().map_err(|_| CodegenError { message: "invalid int literal".to_string() })?;
        Ok(self.context.i64_type().const_int(value as u64, true).into())
    }

    fn create_entry_alloca(
        &self,
        function: FunctionValue<'ctx>,
        name: &str,
        ty: &Type,
    ) -> Result<Local<'ctx>, CodegenError> {
        let builder = self.context.create_builder();
        let entry = function.get_first_basic_block().unwrap();
        match entry.get_first_instruction() {
            Some(inst) => builder.position_before(&inst),
            None => builder.position_at_end(entry),
        }
        let llvm_ty = self.llvm_type(ty)?;
        let ptr = self.map_builder(builder.build_alloca(llvm_ty, name), "alloca")?;
        Ok(Local {
            ptr,
            ty: llvm_ty,
            lang_ty: ty.clone(),
        })
    }

    fn create_local_storage_in_ctx(
        &self,
        function: FunctionValue<'ctx>,
        name: &str,
        ty: &Type,
        root_base: Option<PointerValue<'ctx>>,
        root_index: usize,
    ) -> Result<Local<'ctx>, CodegenError> {
        if self.is_gc_ref_type(ty) {
            let root_base = root_base.ok_or_else(|| CodegenError { message: "missing root base".to_string() })?;
            let idx = self.context.i32_type().const_int(root_index as u64, false);
            let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
            let slot_ptr = unsafe {
                self.map_builder(
                    self.builder.build_gep(ptr_ty, root_base, &[idx], "root_slot"),
                    "gep",
                )?
            };
            Ok(Local {
                ptr: slot_ptr,
                ty: ptr_ty.into(),
                lang_ty: ty.clone(),
            })
        } else {
            self.create_entry_alloca(function, name, ty)
        }
    }

    fn load_local(&self, local: &Local<'ctx>, name: &str) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        let loaded = self.map_builder(self.builder.build_load(local.ty, local.ptr, name), "load")?;
        if self.is_gc_ref_type(&local.lang_ty) {
            if let Some(inst) = loaded.as_instruction_value() {
                inst.set_volatile(true)
                    .map_err(|e| CodegenError { message: format!("volatile load failed: {e}") })?;
            }
        }
        Ok(loaded)
    }

    fn store_local(&self, local: &Local<'ctx>, value: BasicValueEnum<'ctx>) -> Result<(), CodegenError> {
        let inst = self.map_builder(self.builder.build_store(local.ptr, value), "store")?;
        if self.is_gc_ref_type(&local.lang_ty) {
            inst.set_volatile(true)
                .map_err(|e| CodegenError { message: format!("volatile store failed: {e}") })?;
        }
        Ok(())
    }

    fn push_scope(&self, ctx: &mut FnCtx<'ctx>) {
        ctx.scopes.push(Vec::new());
    }

    fn pop_scope(&self, ctx: &mut FnCtx<'ctx>) {
        if let Some(names) = ctx.scopes.pop() {
            for name in names {
                ctx.locals.remove(&name);
            }
        }
    }

    fn map_builder<T>(&self, res: Result<T, BuilderError>, label: &str) -> Result<T, CodegenError> {
        res.map_err(|e| CodegenError { message: format!("{label} failed: {e}") })
    }

    fn is_gc_ref_type(&self, ty: &Type) -> bool {
        match ty {
            Type::RawPointer(_) => false,
            Type::Tuple(_) => true,
            Type::Path(path, _) => {
                let name = path.last().map(|s| s.as_str()).unwrap_or("");
                !matches!(
                    name,
                    "I8" | "I16" | "I32" | "I64" | "U8" | "U16" | "U32" | "U64" | "F32" | "F64" | "Bool" | "Unit" | "String"
                )
            }
        }
    }

    fn count_param_roots(&self, params: &[Param]) -> usize {
        params.iter().filter(|p| self.is_gc_ref_type(&p.ty)).count()
    }

    fn count_roots_in_fn(&self, f: &FnDecl) -> usize {
        let mut count = self.count_param_roots(&f.params);
        count += self.count_roots_in_block(&f.body);
        count
    }

    fn count_roots_in_block(&self, block: &Block) -> usize {
        let mut count = 0;
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => count += self.count_roots_in_expr(expr),
                Stmt::Return(expr, _) => {
                    if let Some(expr) = expr {
                        count += self.count_roots_in_expr(expr);
                    }
                }
            }
        }
        if let Some(tail) = &block.tail {
            count += self.count_roots_in_expr(tail);
        }
        count
    }

    fn count_roots_in_expr(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Let { ty, value, .. } => {
                let mut count = 0;
                if let Some(ty) = ty {
                    if self.is_gc_ref_type(ty) {
                        count += 1;
                    }
                }
                count + self.count_roots_in_expr(value)
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                let mut count = self.count_roots_in_expr(cond);
                count += self.count_roots_in_block(then_branch);
                if let Some(else_branch) = else_branch {
                    count += self.count_roots_in_block(else_branch);
                }
                count
            }
            Expr::While { cond, body, .. } => self.count_roots_in_expr(cond) + self.count_roots_in_block(body),
            Expr::Match { scrutinee, arms, .. } => {
                let mut count = self.count_roots_in_expr(scrutinee);
                for arm in arms {
                    count += self.count_roots_in_pattern(&arm.pattern);
                    count += self.count_roots_in_expr(&arm.body);
                }
                count
            }
            Expr::Assign { target, value, .. } => {
                self.count_roots_in_expr(target) + self.count_roots_in_expr(value)
            }
            Expr::Binary { left, right, .. } => self.count_roots_in_expr(left) + self.count_roots_in_expr(right),
            Expr::Unary { expr, .. } => self.count_roots_in_expr(expr),
            Expr::Call { callee, args, .. } => {
                let mut count = self.count_roots_in_expr(callee);
                for arg in args {
                    count += self.count_roots_in_expr(arg);
                }
                count
            }
            Expr::Field { base, .. } => self.count_roots_in_expr(base),
            Expr::StructLit { fields, .. } => fields.iter().map(|(_, e)| self.count_roots_in_expr(e)).sum(),
            Expr::Tuple { items, .. } => items.iter().map(|e| self.count_roots_in_expr(e)).sum(),
            Expr::Block(block) => self.count_roots_in_block(block),
            _ => 0,
        }
    }

    fn count_roots_in_pattern(&self, pattern: &Pattern) -> usize {
        match pattern {
            Pattern::Struct { path, fields, .. } => {
                if path.len() < 2 {
                    return 0;
                }
                let (enum_name, variant_name) = enum_path_and_variant(path);
                let layout = match self.enums.get(&enum_name) {
                    Some(l) => l,
                    None => return 0,
                };
                let variant = match layout.variants.get(&variant_name) {
                    Some(v) => v,
                    None => return 0,
                };
                let mut count = 0;
                for pat_field in fields {
                    if pat_field.binding.is_none() {
                        continue;
                    }
                    if let Some(field) = variant
                        .fields
                        .iter()
                        .find(|f| f.name.as_deref() == Some(pat_field.name.as_str()))
                    {
                        if field.is_gc_ref {
                            count += 1;
                        }
                    }
                }
                count
            }
            _ => 0,
        }
    }

    fn emit_prologue(
        &mut self,
        thread: PointerValue<'ctx>,
        fn_name: &str,
        num_roots: usize,
        frame_ty: StructType<'ctx>,
    ) -> Result<(PointerValue<'ctx>, PointerValue<'ctx>, Option<PointerValue<'ctx>>), CodegenError> {
        let frame_origin = self.create_frame_origin(fn_name, num_roots)?;
        let frame_ptr = self.map_builder(self.builder.build_alloca(frame_ty, "frame"), "alloca")?;

        let parent_ptr = self.map_builder(
            self.builder.build_struct_gep(frame_ty, frame_ptr, 0, "frame.parent"),
            "gep",
        )?;
        let origin_ptr = self.map_builder(
            self.builder.build_struct_gep(frame_ty, frame_ptr, 1, "frame.origin"),
            "gep",
        )?;

        let top_frame_ptr = self.thread_top_frame_ptr(thread)?;
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let parent = self.map_builder(self.builder.build_load(ptr_ty, top_frame_ptr, "parent"), "load")?;

        self.map_builder(self.builder.build_store(parent_ptr, parent), "store")?;
        self.map_builder(self.builder.build_store(origin_ptr, frame_origin), "store")?;

        let root_base = if num_roots > 0 {
            let roots_ptr = self.map_builder(
                self.builder.build_struct_gep(frame_ty, frame_ptr, 2, "frame.roots"),
                "gep",
            )?;
            let roots_ty = ptr_ty.array_type(num_roots as u32);
            let zero = self.context.i32_type().const_zero();
            let root0 = unsafe {
                self.map_builder(
                    self.builder.build_gep(roots_ty, roots_ptr, &[zero, zero], "root0"),
                    "gep",
                )?
            };
            let ptr_size = std::mem::size_of::<usize>() as u64;
            let size = self.context.i64_type().const_int(ptr_size * num_roots as u64, false);
            let val = self.context.i8_type().const_int(0, false);
            let dest = self.map_builder(
                self.builder.build_bit_cast(root0, ptr_ty, "root0_cast"),
                "bitcast",
            )?;
            let dest_ptr = dest.into_pointer_value();
            let _ = self.map_builder(self.builder.build_memset(dest_ptr, 8, val, size), "memset")?;
            Some(root0)
        } else {
            None
        };

        self.map_builder(self.builder.build_store(top_frame_ptr, frame_ptr), "store")?;

        Ok((frame_ptr, frame_origin, root_base))
    }

    fn emit_epilogue(&self, ctx: &FnCtx<'ctx>) -> Result<(), CodegenError> {
        let parent_ptr = self.map_builder(
            self.builder.build_struct_gep(ctx.frame_ty, ctx.frame_ptr, 0, "frame.parent"),
            "gep",
        )?;
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let parent = self.map_builder(self.builder.build_load(ptr_ty, parent_ptr, "load"), "load")?;
        let top_frame_ptr = self.thread_top_frame_ptr(ctx.thread)?;
        self.map_builder(self.builder.build_store(top_frame_ptr, parent), "store")?;
        Ok(())
    }

    fn emit_pollcheck(&self, ctx: &FnCtx<'ctx>) -> Result<(), CodegenError> {
        let state_ptr = self.thread_state_ptr(ctx.thread)?;
        let state = self.map_builder(
            self.builder.build_load(self.context.i32_type(), state_ptr, "gc_state"),
            "load",
        )?;
        let zero = self.context.i32_type().const_zero();
        let needs = self.map_builder(
            self.builder
                .build_int_compare(inkwell::IntPredicate::NE, state.into_int_value(), zero, "needs_gc"),
            "icmp",
        )?;

        let func = ctx.function;
        let slow_bb = self.context.append_basic_block(func, "poll.slow");
        let cont_bb = self.context.append_basic_block(func, "poll.cont");
        self.map_builder(
            self.builder
                .build_conditional_branch(needs, slow_bb, cont_bb),
            "condbr",
        )?;

        self.builder.position_at_end(slow_bb);
        let args = &[ctx.thread.into(), ctx.frame_origin.into()];
        let _ = self.map_builder(self.builder.build_call(self.gc_pollcheck, args, "poll"), "call")?;
        self.map_builder(self.builder.build_unconditional_branch(cont_bb), "br")?;

        self.builder.position_at_end(cont_bb);
        Ok(())
    }

    fn thread_top_frame_ptr(&self, thread: PointerValue<'ctx>) -> Result<PointerValue<'ctx>, CodegenError> {
        self.map_builder(
            self.builder.build_struct_gep(self.thread_struct_ty, thread, 0, "thread.top"),
            "gep",
        )
    }

    fn thread_state_ptr(&self, thread: PointerValue<'ctx>) -> Result<PointerValue<'ctx>, CodegenError> {
        self.map_builder(
            self.builder.build_struct_gep(self.thread_struct_ty, thread, 1, "thread.state"),
            "gep",
        )
    }

    fn frame_type(&self, num_roots: usize) -> StructType<'ctx> {
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let roots_ty = ptr_ty.array_type(num_roots as u32);
        self.context.struct_type(&[ptr_ty.into(), ptr_ty.into(), roots_ty.into()], false)
    }

    fn create_frame_origin(&mut self, fn_name: &str, num_roots: usize) -> Result<PointerValue<'ctx>, CodegenError> {
        let i32_ty = self.context.i32_type();
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());

        let name_const = self.context.const_string(fn_name.as_bytes(), true);
        let sym = mangle_name(fn_name);
        let name_global = self.module.add_global(name_const.get_type(), None, &format!("__fn_name_{}", sym));
        name_global.set_initializer(&name_const);
        let zero = i32_ty.const_zero();
        let name_ptr = unsafe { name_global.as_pointer_value().const_gep(name_const.get_type(), &[zero, zero]) };

        let origin_ty = self.context.struct_type(&[i32_ty.into(), ptr_ty.into()], false);
        let origin_global = self.module.add_global(origin_ty, None, &format!("__frame_origin_{}", sym));
        let num = i32_ty.const_int(num_roots as u64, false);
        let init = origin_ty.const_named_struct(&[num.into(), name_ptr.into()]);
        origin_global.set_initializer(&init);

        self.compiler_used_globals.push(name_global.as_pointer_value());
        self.compiler_used_globals.push(origin_global.as_pointer_value());

        Ok(origin_global.as_pointer_value())
    }

    fn emit_compiler_used(&self) {
        if self.compiler_used_globals.is_empty() {
            return;
        }
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let arr_ty = ptr_ty.array_type(self.compiler_used_globals.len() as u32);
        let global = self.module.add_global(arr_ty, None, "llvm.compiler.used");
        let arr = ptr_ty.const_array(&self.compiler_used_globals);
        global.set_initializer(&arr);
        global.set_linkage(inkwell::module::Linkage::Appending);
        // Bypass inkwell's set_section which prepends "," on macOS for Mach-O.
        // llvm.compiler.used needs the raw section name "llvm.metadata".
        unsafe {
            use inkwell::values::AsValueRef;
            let section = std::ffi::CString::new("llvm.metadata").unwrap();
            inkwell::llvm_sys::core::LLVMSetSection(global.as_pointer_value().as_value_ref(), section.as_ptr());
        }
    }

    fn field_llvm_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, CodegenError> {
        match ty {
            Type::Path(path, _) => {
                let name = path.last().ok_or_else(|| CodegenError { message: "empty type".to_string() })?;
                match name.as_str() {
                    "I8" | "U8" => Ok(self.context.i8_type().into()),
                    "I16" | "U16" => Ok(self.context.i16_type().into()),
                    "I32" | "U32" => Ok(self.context.i32_type().into()),
                    "I64" | "U64" => Ok(self.context.i64_type().into()),
                    "F32" => Ok(self.context.f32_type().into()),
                    "F64" => Ok(self.context.f64_type().into()),
                    "Bool" => Ok(self.context.i8_type().into()),
                    "String" => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
                    "Unit" => Ok(self.context.i64_type().into()),
                    _ if self.is_gc_ref_type(ty) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
                    _ => Err(CodegenError { message: format!("unsupported field type '{}'", name) }),
                }
            }
            Type::RawPointer(_) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
            Type::Tuple(_) => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
        }
    }

    fn zero_value(&self, ty: BasicTypeEnum<'ctx>) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        match ty {
            BasicTypeEnum::IntType(t) => Ok(t.const_zero().into()),
            BasicTypeEnum::FloatType(t) => Ok(t.const_zero().into()),
            BasicTypeEnum::PointerType(t) => Ok(t.const_null().into()),
            _ => Err(CodegenError { message: "unsupported zero value".to_string() }),
        }
    }

    fn struct_name_from_type(&self, ty: &Type) -> Result<String, CodegenError> {
        match ty {
            Type::Path(path, _) => Ok(path_to_string(path)),
            _ => Err(CodegenError { message: "expected struct type".to_string() }),
        }
    }

    fn layout_for_type(&mut self, ty: &Type) -> Result<StructLayout<'ctx>, CodegenError> {
        match ty {
            Type::Path(path, _) => {
                let name = path_to_string(path);
                self.structs
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| CodegenError { message: "unknown struct layout".to_string() })
            }
            Type::Tuple(items) => {
                let key = self.ensure_tuple_layout(items)?;
                self.tuples
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| CodegenError { message: "unknown tuple layout".to_string() })
            }
            _ => Err(CodegenError { message: "expected struct type".to_string() }),
        }
    }

    fn is_bool_type(&self, ty: &Type) -> bool {
        matches!(ty, Type::Path(path, _) if path.last().map(|s| s.as_str()) == Some("Bool"))
    }

    fn resolve_struct_base(
        &mut self,
        base: &Expr,
        ctx: &FnCtx<'ctx>,
    ) -> Result<(PointerValue<'ctx>, StructLayout<'ctx>), CodegenError> {
        match base {
            Expr::Path(path, _) => {
                let var = path.last().ok_or_else(|| CodegenError { message: "empty base".to_string() })?;
                let local = ctx.locals.get(var).ok_or_else(|| CodegenError { message: "unknown local".to_string() })?;
                let layout = self.layout_for_type(&local.lang_ty)?;
                let obj_ptr = self.load_local(local, "obj")?;
                Ok((obj_ptr.into_pointer_value(), layout))
            }
            Expr::Field { base: inner, name, .. } => {
                let (obj_ptr, _layout, field) = self.resolve_field(inner, name, ctx)?;
                if !self.is_gc_ref_type(&field.ty) {
                    return Err(CodegenError { message: "field base must be gc ref".to_string() });
                }
                let field_ptr = self.field_ptr(obj_ptr, &field)?;
                let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                let loaded = self.map_builder(self.builder.build_load(ptr_ty, field_ptr, "field_obj"), "load")?;
                let layout = self.layout_for_type(&field.ty)?;
                Ok((loaded.into_pointer_value(), layout))
            }
            _ => Err(CodegenError { message: "field base must be path or field in codegen".to_string() }),
        }
    }

    fn resolve_field(
        &mut self,
        base: &Expr,
        name: &str,
        ctx: &FnCtx<'ctx>,
    ) -> Result<(PointerValue<'ctx>, StructLayout<'ctx>, FieldLayout<'ctx>), CodegenError> {
        let (obj_ptr, layout) = self.resolve_struct_base(base, ctx)?;
        let field = layout
            .fields
            .iter()
            .find(|f| f.name == name)
            .cloned()
            .ok_or_else(|| CodegenError { message: "unknown field".to_string() })?;
        Ok((obj_ptr, layout, field))
    }

    fn field_ptr(
        &self,
        obj_ptr: PointerValue<'ctx>,
        field: &FieldLayout<'ctx>,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        self.gc_field_ptr(obj_ptr, field.index)
    }

    /// Compute pointer to field at given index: obj + 8 + index * 8
    fn gc_field_ptr(
        &self,
        obj_ptr: PointerValue<'ctx>,
        index: u64,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        let i8_ty = self.context.i8_type();
        let byte_offset = 8 + index * 8; // HEADER_SIZE=8, each field is 8 bytes
        let offset = self.context.i64_type().const_int(byte_offset, false);
        let field_i8 = unsafe {
            self.map_builder(
                self.builder.build_gep(i8_ty, obj_ptr, &[offset], "field_ptr"),
                "gep",
            )?
        };
        Ok(field_i8)
    }

    /// Compute pointer to enum field at given index: obj + 8 + index * 8
    /// This works for both ptr fields and raw fields since all use 8-byte slots.
    fn enum_field_ptr(&self, obj_ptr: PointerValue<'ctx>, field_index: u64) -> Result<PointerValue<'ctx>, CodegenError> {
        self.gc_field_ptr(obj_ptr, field_index)
    }

    /// Load a value from a GC object field (8-byte slot).
    /// GC ref fields are loaded as ptr, non-GC fields as i64 then truncated/converted.
    fn load_gc_field(&self, field_ptr: PointerValue<'ctx>, field: &FieldLayout<'ctx>) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        if field.is_gc_ref {
            let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
            let loaded = self.map_builder(self.builder.build_load(ptr_ty, field_ptr, "field"), "load")?;
            Ok(loaded)
        } else {
            let i64_ty = self.context.i64_type();
            let loaded = self.map_builder(self.builder.build_load(i64_ty, field_ptr, "field"), "load")?;
            if self.is_bool_type(&field.ty) {
                let zero = i64_ty.const_zero();
                let cmp = self.map_builder(
                    self.builder
                        .build_int_compare(inkwell::IntPredicate::NE, loaded.into_int_value(), zero, "bool"),
                    "icmp",
                )?;
                Ok(cmp.into())
            } else {
                // For I64, F64 etc. the i64 value is the right width.
                // For smaller types (I32, I16, I8, F32) we need to truncate/bitcast.
                match field.llvm_ty {
                    BasicTypeEnum::IntType(t) if t.get_bit_width() < 64 => {
                        let trunc = self.map_builder(
                            self.builder.build_int_truncate(loaded.into_int_value(), t, "trunc"),
                            "trunc",
                        )?;
                        Ok(trunc.into())
                    }
                    BasicTypeEnum::FloatType(t) => {
                        if t == self.context.f64_type() {
                            let cast = self.map_builder(
                                self.builder.build_bit_cast(loaded, self.context.f64_type(), "f64cast"),
                                "bitcast",
                            )?;
                            Ok(cast)
                        } else {
                            // f32: truncate i64 to i32, then bitcast to f32
                            let trunc = self.map_builder(
                                self.builder.build_int_truncate(loaded.into_int_value(), self.context.i32_type(), "trunc32"),
                                "trunc",
                            )?;
                            let cast = self.map_builder(
                                self.builder.build_bit_cast(trunc, self.context.f32_type(), "f32cast"),
                                "bitcast",
                            )?;
                            Ok(cast)
                        }
                    }
                    _ => Ok(loaded),
                }
            }
        }
    }

    /// Store a value to a GC object field (8-byte slot).
    /// GC ref fields are stored as ptr, non-GC fields are extended/converted to i64.
    fn store_gc_field(&self, field_ptr: PointerValue<'ctx>, field: &FieldLayout<'ctx>, value: BasicValueEnum<'ctx>) -> Result<(), CodegenError> {
        if field.is_gc_ref {
            self.map_builder(self.builder.build_store(field_ptr, value), "store")?;
        } else {
            let i64_ty = self.context.i64_type();
            let store_val = if self.is_bool_type(&field.ty) {
                let zext = self.map_builder(
                    self.builder.build_int_z_extend(value.into_int_value(), i64_ty, "boolz"),
                    "zext",
                )?;
                zext.into()
            } else if value.is_float_value() {
                let fv = value.into_float_value();
                if fv.get_type() == self.context.f32_type() {
                    // f32 -> bitcast to i32 -> zext to i64
                    let i32_val = self.map_builder(
                        self.builder.build_bit_cast(fv, self.context.i32_type(), "f32toi32"),
                        "bitcast",
                    )?;
                    let zext = self.map_builder(
                        self.builder.build_int_z_extend(i32_val.into_int_value(), i64_ty, "zext64"),
                        "zext",
                    )?;
                    zext.into()
                } else {
                    // f64 -> bitcast to i64
                    self.map_builder(
                        self.builder.build_bit_cast(fv, i64_ty, "f64toi64"),
                        "bitcast",
                    )?
                }
            } else if value.is_int_value() {
                let iv = value.into_int_value();
                if iv.get_type().get_bit_width() < 64 {
                    let zext = self.map_builder(
                        self.builder.build_int_z_extend(iv, i64_ty, "zext"),
                        "zext",
                    )?;
                    zext.into()
                } else {
                    value
                }
            } else {
                value
            };
            self.map_builder(self.builder.build_store(field_ptr, store_val), "store")?;
        }
        Ok(())
    }

    fn resolve_enum_scrutinee(
        &self,
        scrutinee: &Expr,
        ctx: &FnCtx<'ctx>,
    ) -> Result<(EnumLayout<'ctx>, PointerValue<'ctx>), CodegenError> {
        match scrutinee {
            Expr::Path(path, _) => {
                let name = path.last().ok_or_else(|| CodegenError { message: "empty scrutinee".to_string() })?;
                let local = ctx.locals.get(name).ok_or_else(|| CodegenError { message: "unknown local".to_string() })?;
                let enum_name = self.struct_name_from_type(&local.lang_ty)?;
                let layout = self.enums.get(&enum_name).cloned().ok_or_else(|| CodegenError { message: "unknown enum layout".to_string() })?;
                let obj = self.load_local(local, "enum_obj")?;
                Ok((layout, obj.into_pointer_value()))
            }
            _ => Err(CodegenError { message: "match scrutinee must be path in codegen".to_string() }),
        }
    }

    fn load_enum_tag(&self, obj_ptr: PointerValue<'ctx>, layout: &EnumLayout<'ctx>) -> Result<inkwell::values::IntValue<'ctx>, CodegenError> {
        let tag_index = layout.ptr_field_count; // tag is first raw field
        let tag_ptr = self.gc_field_ptr(obj_ptr, tag_index)?;
        let i64_ty = self.context.i64_type();
        let tag_i64 = self.map_builder(
            self.builder.build_load(i64_ty, tag_ptr, "tag_i64"),
            "load",
        )?;
        // Truncate to i32 for switch instruction
        let tag = self.map_builder(
            self.builder.build_int_truncate(tag_i64.into_int_value(), self.context.i32_type(), "tag"),
            "trunc",
        )?;
        Ok(tag)
    }

}

fn path_to_string(path: &[String]) -> String {
    path.join("::")
}

fn full_item_name(module_path: &[String], name: &str) -> String {
    if module_path.is_empty() {
        name.to_string()
    } else {
        let mut parts = module_path.to_vec();
        parts.push(name.to_string());
        path_to_string(&parts)
    }
}

fn enum_path_and_variant(path: &[String]) -> (String, String) {
    let variant = path.last().cloned().unwrap_or_default();
    let enum_path = path[..path.len() - 1].to_vec();
    (path_to_string(&enum_path), variant)
}

fn last_segment(full: &str) -> &str {
    full.rsplit("::").next().unwrap_or(full)
}

fn extern_symbol_name(full: &str) -> &str {
    last_segment(full)
}

fn mangle_name(name: &str) -> String {
    name.replace("::", "__")
}

fn tuple_key(items: &[Type]) -> String {
    let mut out = String::from("__tuple__");
    for (i, ty) in items.iter().enumerate() {
        if i > 0 {
            out.push('_');
        }
        out.push_str(&type_key(ty));
    }
    out
}

fn type_key(ty: &Type) -> String {
    match ty {
        Type::Path(path, _) => mangle_name(&path_to_string(path)),
        Type::RawPointer(inner) => format!("ptr_{}", type_key(inner)),
        Type::Tuple(items) => {
            let mut out = String::from("tup");
            for item in items {
                out.push('_');
                out.push_str(&type_key(item));
            }
            out
        }
    }
}

struct Local<'ctx> {
    ptr: PointerValue<'ctx>,
    ty: BasicTypeEnum<'ctx>,
    lang_ty: Type,
}

struct FnCtx<'ctx> {
    function: FunctionValue<'ctx>,
    locals: HashMap<String, Local<'ctx>>,
    scopes: Vec<Vec<String>>,
    thread: PointerValue<'ctx>,
    frame_ptr: PointerValue<'ctx>,
    frame_origin: PointerValue<'ctx>,
    frame_ty: StructType<'ctx>,
    root_base: Option<PointerValue<'ctx>>,
    next_root: usize,
    loop_stack: Vec<(inkwell::basic_block::BasicBlock<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::resolve::resolve_module;
    use crate::typecheck::typecheck_module;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn compile_object(src: &str) -> std::path::PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let tokens = Lexer::new(src).lex_all().unwrap();
        let module = Parser::new(tokens).parse_module().unwrap();
        resolve_module(&module).unwrap();
        typecheck_module(&module).unwrap();
        let mut path = std::env::temp_dir();
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        path.push(format!("langc_test_{id}.o"));
        compile_to_object(std::slice::from_ref(&module), &path).unwrap();
        path
    }

    fn compile_link_run(src: &str) -> i32 {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let tokens = Lexer::new(src).lex_all().unwrap();
        let module = Parser::new(tokens).parse_module().unwrap();
        resolve_module(&module).unwrap();
        typecheck_module(&module).unwrap();

        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut exe_path = std::env::temp_dir();
        exe_path.push(format!("langc_e2e_{id}"));

        compile_to_executable(std::slice::from_ref(&module), &exe_path, &[]).unwrap();

        let output = std::process::Command::new(&exe_path)
            .output()
            .expect("failed to run exe");
        let _ = std::fs::remove_file(&exe_path);
        output.status.code().unwrap_or(-1)
    }

    #[test]
    fn codegen_aot_object() {
        let src = r#"
            fn main() -> I64 { 3 }
        "#;
        let path = compile_object(src);
        assert!(path.exists());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn codegen_e2e_aot_all_features() {
        let src = r#"
            extern fn add_i64(a: I64, b: I64) -> I64;
            extern fn null_ptr() -> RawPointer<I8>;
            extern fn ptr_is_null(p: RawPointer<I8>) -> I64;

            struct Pair { a: I64, b: I64 }
            enum Flag { On {}, Off {} }

            fn main() -> I64 {
                let mut i: I64 = 0;
                let mut acc: I64 = 0;
                while i < 4 { acc = acc + i; i = i + 1; };

                let mut p: Pair = Pair { a: acc, b: add_i64(11, 20) };
                p.b = p.b + 0;
                let mut total: I64 = p.a + p.b;

                let f: Flag = Flag::On {};
                total = match f { Flag::On {} => total + 1, Flag::Off {} => total + 2 };

                let extra: I64 = add_i64(1, 2) + 3;
                total = total + extra;

                let ptr: RawPointer<I8> = null_ptr();
                if ptr_is_null(ptr) == 1 { total - 2 } else { 0 }
            }
        "#;
        assert_eq!(compile_link_run(src), 42);
    }

    #[test]
    fn codegen_tree_sum_aot() {
        let src = r#"
            struct Node { left: OptNode, right: OptNode, value: I64 }
            enum OptNode { Some { node: Node }, None {} }

            fn sum(opt: OptNode) -> I64 {
                match opt {
                    OptNode::Some { node } => node.value + sum(node.left) + sum(node.right),
                    OptNode::None {} => 0
                }
            }

            fn main() -> I64 {
                let zero: I64 = 0;
                let one: I64 = 1;
                let two: I64 = 2;
                let three: I64 = 3;
                let four: I64 = 4;

                let leaf1: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: one };
                let leaf2: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: two };
                let leaf3: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: three };
                let leaf4: Node = Node { left: OptNode::None {}, right: OptNode::None {}, value: four };

                let n1: Node = Node {
                    left: OptNode::Some { node: leaf1 },
                    right: OptNode::Some { node: leaf2 },
                    value: zero
                };
                let n2: Node = Node {
                    left: OptNode::Some { node: leaf3 },
                    right: OptNode::Some { node: leaf4 },
                    value: zero
                };
                let root: Node = Node {
                    left: OptNode::Some { node: n1 },
                    right: OptNode::Some { node: n2 },
                    value: zero
                };

                sum(OptNode::Some { node: root })
            }
        "#;
        assert_eq!(compile_link_run(src), 10);
    }
}
