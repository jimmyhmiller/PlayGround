use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType};
use inkwell::values::{AnyValue, BasicMetadataValueEnum, BasicValueEnum, FunctionValue, GlobalValue, PointerValue};
use inkwell::IntPredicate;
use std::collections::HashMap;

use crate::ast::{BinOp, Expr, FunctionDef, Program, Stmt, Type};
use crate::codegen::types::{LLVMTypes, RuntimeFunctions};
use crate::runtime::thread::offsets as thread_offsets;

/// Compiles a program to LLVM IR
pub struct Compiler<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    pub types: LLVMTypes<'ctx>,
    pub runtime: RuntimeFunctions<'ctx>,
    pub program: Program,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context, program: Program) -> Self {
        let module = context.create_module("main");
        let builder = context.create_builder();
        let types = LLVMTypes::new(context, &module, &program);
        let runtime = RuntimeFunctions::declare(&module, &types);

        Self {
            context,
            module,
            builder,
            types,
            runtime,
            program,
        }
    }

    /// Compile all functions in the program
    pub fn compile(&self) {
        // First pass: declare all functions
        let mut function_values: HashMap<String, FunctionValue<'ctx>> = HashMap::new();
        for func in &self.program.functions {
            let fn_value = self.declare_function(func);
            function_values.insert(func.name().to_string(), fn_value);
        }

        // Second pass: compile function bodies
        for func in &self.program.functions {
            let fn_value = function_values[func.name()];
            let mut fn_compiler = FunctionCompiler::new(self, func, fn_value, &function_values);
            fn_compiler.compile();
        }
    }

    /// Declare a function (just the signature, no body)
    fn declare_function(&self, func: &FunctionDef) -> FunctionValue<'ctx> {
        // All functions take thread* as first parameter
        let mut param_types: Vec<BasicMetadataTypeEnum<'ctx>> =
            vec![self.types.ptr_ty.into()]; // thread*

        for param in func.params() {
            param_types.push(self.types.ast_type_to_llvm(&param.typ).into());
        }

        let fn_type = match func.return_type() {
            Type::Void => self.types.void_ty.fn_type(&param_types, false),
            other => self.types.ast_type_to_llvm(other).fn_type(&param_types, false),
        };

        self.module.add_function(func.name(), fn_type, None)
    }

    pub fn print_ir(&self) {
        self.module.print_to_stderr();
    }

    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }
}

/// Per-function compilation state
struct FunctionCompiler<'a, 'ctx> {
    compiler: &'a Compiler<'ctx>,
    func_def: &'a FunctionDef,
    fn_value: FunctionValue<'ctx>,
    functions: &'a HashMap<String, FunctionValue<'ctx>>,

    /// Thread pointer (first parameter)
    thread: PointerValue<'ctx>,

    /// GC frame alloca
    frame_alloca: PointerValue<'ctx>,

    /// FrameOrigin global for this function
    frame_origin: GlobalValue<'ctx>,

    /// Number of root slots in the frame
    num_roots: usize,

    /// Maps variable names to their storage (alloca or root slot index)
    variables: HashMap<String, VarStorage<'ctx>>,

    /// Next available root slot
    next_root_slot: usize,
}

/// How a variable is stored
enum VarStorage<'ctx> {
    /// A primitive value stored in an alloca, with the type for loading
    Alloca(PointerValue<'ctx>, inkwell::types::BasicTypeEnum<'ctx>),
    /// A GC reference stored in a root slot (index into frame.roots)
    RootSlot(usize),
}

impl<'a, 'ctx> FunctionCompiler<'a, 'ctx> {
    fn new(
        compiler: &'a Compiler<'ctx>,
        func_def: &'a FunctionDef,
        fn_value: FunctionValue<'ctx>,
        functions: &'a HashMap<String, FunctionValue<'ctx>>,
    ) -> Self {
        // Count how many GC root slots we need:
        // - One for each GC-typed parameter
        // - One for each GC-typed local variable
        let num_roots = Self::count_roots(func_def);

        Self {
            compiler,
            func_def,
            fn_value,
            functions,
            thread: fn_value.get_nth_param(0).unwrap().into_pointer_value(),
            frame_alloca: compiler.types.ptr_ty.const_null(), // Will be set in compile()
            frame_origin: unsafe { std::mem::zeroed() }, // Will be set in compile()
            num_roots,
            variables: HashMap::new(),
            next_root_slot: 0,
        }
    }

    /// Count how many root slots this function needs
    fn count_roots(func: &FunctionDef) -> usize {
        let mut count = 0;

        // GC-typed parameters need slots
        for param in func.params() {
            if param.typ.is_gc_ref() {
                count += 1;
            }
        }

        // GC-typed locals need slots
        for stmt in &func.body {
            Self::count_roots_in_stmt(stmt, &mut count);
        }

        count
    }

    fn count_roots_in_stmt(stmt: &Stmt, count: &mut usize) {
        match stmt {
            Stmt::Let { typ, .. } if typ.is_gc_ref() => {
                *count += 1;
            }
            Stmt::If { then_block, else_block, .. } => {
                for s in then_block {
                    Self::count_roots_in_stmt(s, count);
                }
                for s in else_block {
                    Self::count_roots_in_stmt(s, count);
                }
            }
            Stmt::While { body, .. } => {
                for s in body {
                    Self::count_roots_in_stmt(s, count);
                }
            }
            _ => {}
        }
    }

    /// Compile the function
    fn compile(&mut self) {
        let entry = self.compiler.context.append_basic_block(self.fn_value, "entry");
        self.compiler.builder.position_at_end(entry);

        // Create FrameOrigin for this function
        self.frame_origin = self.create_frame_origin();

        // Emit GC frame prologue
        self.emit_prologue();

        // Set up parameter variables
        self.setup_parameters();

        // Compile the function body
        for stmt in self.func_def.body.clone() {
            self.compile_stmt(&stmt);
        }

        // If we haven't returned yet, emit a default return
        if self.compiler.builder.get_insert_block().unwrap().get_terminator().is_none() {
            self.emit_epilogue();
            if self.func_def.return_type() == &Type::Void {
                self.compiler.builder.build_return(None).unwrap();
            } else {
                // Return a default value (shouldn't happen in well-formed code)
                let default: BasicValueEnum = match self.func_def.return_type() {
                    Type::I32 | Type::I64 => self.compiler.types.i64_ty.const_zero().into(),
                    Type::Bool => self.compiler.types.i1_ty.const_zero().into(),
                    Type::Struct(_) | Type::Array(_) => self.compiler.types.ptr_ty.const_null().into(),
                    Type::Void => unreachable!(),
                };
                self.compiler.builder.build_return(Some(&default)).unwrap();
            }
        }
    }

    /// Create the FrameOrigin global for this function
    fn create_frame_origin(&self) -> GlobalValue<'ctx> {
        let name_str = self.compiler.context.const_string(self.func_def.name().as_bytes(), true);
        let name_global = self.compiler.module.add_global(
            name_str.get_type(),
            None,
            &format!("__fn_name_{}", self.func_def.name()),
        );
        name_global.set_initializer(&name_str);
        name_global.set_linkage(inkwell::module::Linkage::Private);

        // FrameOrigin: { num_roots: i32, function_name: ptr }
        let origin_value = self.compiler.types.frame_origin_ty.const_named_struct(&[
            self.compiler.types.i32_ty.const_int(self.num_roots as u64, false).into(),
            name_global.as_pointer_value().into(),
        ]);

        let origin_global = self.compiler.module.add_global(
            self.compiler.types.frame_origin_ty,
            None,
            &format!("__frame_origin_{}", self.func_def.name()),
        );
        origin_global.set_initializer(&origin_value);
        origin_global.set_linkage(inkwell::module::Linkage::Private);
        origin_global.set_constant(true);

        origin_global
    }

    /// Emit the GC frame prologue
    fn emit_prologue(&mut self) {
        let types = &self.compiler.types;
        let builder = &self.compiler.builder;

        // Allocate the frame on the stack
        let frame_ty = types.frame_type(self.compiler.context, self.num_roots);
        let frame_alloca = builder.build_alloca(frame_ty, "gc_frame").unwrap();
        self.frame_alloca = frame_alloca;

        // Zero-initialize the frame (important: roots start as null)
        let frame_size = frame_ty.size_of().unwrap();
        builder
            .build_memset(frame_alloca, 8, types.i8_ty.const_zero(), frame_size)
            .unwrap();

        // Store origin pointer: frame.origin = &frame_origin
        let origin_ptr = builder
            .build_struct_gep(frame_ty, frame_alloca, 1, "origin_ptr")
            .unwrap();
        builder
            .build_store(origin_ptr, self.frame_origin.as_pointer_value())
            .unwrap();

        // Push frame onto chain: frame.parent = thread.top_frame
        let top_frame_ptr = self.get_thread_top_frame_ptr();
        let old_top = builder
            .build_load(types.ptr_ty, top_frame_ptr, "old_top")
            .unwrap();

        let parent_ptr = builder
            .build_struct_gep(frame_ty, frame_alloca, 0, "parent_ptr")
            .unwrap();
        builder.build_store(parent_ptr, old_top).unwrap();

        // thread.top_frame = &frame
        builder.build_store(top_frame_ptr, frame_alloca).unwrap();
    }

    /// Emit the GC frame epilogue (before returns)
    fn emit_epilogue(&self) {
        let types = &self.compiler.types;
        let builder = &self.compiler.builder;
        let frame_ty = types.frame_type(self.compiler.context, self.num_roots);

        // Pop frame: thread.top_frame = frame.parent
        let parent_ptr = builder
            .build_struct_gep(frame_ty, self.frame_alloca, 0, "parent_ptr")
            .unwrap();
        let parent = builder
            .build_load(types.ptr_ty, parent_ptr, "parent")
            .unwrap();

        let top_frame_ptr = self.get_thread_top_frame_ptr();
        builder.build_store(top_frame_ptr, parent).unwrap();
    }

    /// Get a pointer to thread.top_frame
    fn get_thread_top_frame_ptr(&self) -> PointerValue<'ctx> {
        let types = &self.compiler.types;
        let builder = &self.compiler.builder;

        unsafe {
            builder
                .build_gep(
                    types.i8_ty,
                    self.thread,
                    &[types.i64_ty.const_int(thread_offsets::TOP_FRAME, false)],
                    "top_frame_ptr",
                )
                .unwrap()
        }
    }

    /// Get a pointer to thread.state
    fn get_thread_state_ptr(&self) -> PointerValue<'ctx> {
        let types = &self.compiler.types;
        let builder = &self.compiler.builder;

        unsafe {
            builder
                .build_gep(
                    types.i8_ty,
                    self.thread,
                    &[types.i64_ty.const_int(thread_offsets::STATE, false)],
                    "state_ptr",
                )
                .unwrap()
        }
    }

    /// Set up variables for function parameters
    fn setup_parameters(&mut self) {
        for (i, param) in self.func_def.params().iter().enumerate() {
            // Parameter values are at index i+1 (0 is thread)
            let param_value = self.fn_value.get_nth_param((i + 1) as u32).unwrap();

            if param.typ.is_gc_ref() {
                // Store in a root slot
                let slot = self.next_root_slot;
                self.next_root_slot += 1;
                self.store_root(slot, param_value.into_pointer_value());
                self.variables.insert(param.name.clone(), VarStorage::RootSlot(slot));
            } else {
                // Store in an alloca
                let ty = self.compiler.types.ast_type_to_llvm(&param.typ);
                let alloca = self
                    .compiler
                    .builder
                    .build_alloca(ty, &param.name)
                    .unwrap();
                self.compiler.builder.build_store(alloca, param_value).unwrap();
                self.variables.insert(param.name.clone(), VarStorage::Alloca(alloca, ty));
            }
        }
    }

    /// Store a value in a root slot
    fn store_root(&self, slot: usize, value: PointerValue<'ctx>) {
        let types = &self.compiler.types;
        let builder = &self.compiler.builder;
        let frame_ty = types.frame_type(self.compiler.context, self.num_roots);

        // GEP to roots[slot]
        let roots_ptr = builder
            .build_struct_gep(frame_ty, self.frame_alloca, 2, "roots_ptr")
            .unwrap();

        let slot_ptr = unsafe {
            builder
                .build_gep(
                    types.ptr_ty.array_type(self.num_roots as u32),
                    roots_ptr,
                    &[
                        types.i32_ty.const_zero(),
                        types.i32_ty.const_int(slot as u64, false),
                    ],
                    "root_slot",
                )
                .unwrap()
        };

        builder.build_store(slot_ptr, value).unwrap();
    }

    /// Load a value from a root slot
    fn load_root(&self, slot: usize) -> PointerValue<'ctx> {
        let types = &self.compiler.types;
        let builder = &self.compiler.builder;
        let frame_ty = types.frame_type(self.compiler.context, self.num_roots);

        let roots_ptr = builder
            .build_struct_gep(frame_ty, self.frame_alloca, 2, "roots_ptr")
            .unwrap();

        let slot_ptr = unsafe {
            builder
                .build_gep(
                    types.ptr_ty.array_type(self.num_roots as u32),
                    roots_ptr,
                    &[
                        types.i32_ty.const_zero(),
                        types.i32_ty.const_int(slot as u64, false),
                    ],
                    "root_slot",
                )
                .unwrap()
        };

        builder
            .build_load(types.ptr_ty, slot_ptr, "root_value")
            .unwrap()
            .into_pointer_value()
    }

    /// Emit a pollcheck (safepoint)
    fn emit_pollcheck(&self) {
        let types = &self.compiler.types;
        let builder = &self.compiler.builder;

        // Load thread.state
        let state_ptr = self.get_thread_state_ptr();
        let state = builder
            .build_load(types.i8_ty, state_ptr, "state")
            .unwrap()
            .into_int_value();

        // Check if any flags are set
        let needs_slow = builder
            .build_int_compare(IntPredicate::NE, state, types.i8_ty.const_zero(), "needs_slow")
            .unwrap();

        // Branch: if flags set, call slow path
        let slow_bb = self
            .compiler
            .context
            .append_basic_block(self.fn_value, "pollcheck_slow");
        let continue_bb = self
            .compiler
            .context
            .append_basic_block(self.fn_value, "pollcheck_continue");

        builder
            .build_conditional_branch(needs_slow, slow_bb, continue_bb)
            .unwrap();

        // Slow path
        builder.position_at_end(slow_bb);
        builder
            .build_call(
                self.compiler.runtime.pollcheck_slow,
                &[
                    self.thread.into(),
                    self.frame_origin.as_pointer_value().into(),
                ],
                "",
            )
            .unwrap();
        builder.build_unconditional_branch(continue_bb).unwrap();

        // Continue
        builder.position_at_end(continue_bb);
    }

    /// Compile a statement
    fn compile_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { name, typ, init } => {
                if typ.is_gc_ref() {
                    let slot = self.next_root_slot;
                    self.next_root_slot += 1;

                    let value = if let Some(expr) = init {
                        self.compile_expr(expr).into_pointer_value()
                    } else {
                        self.compiler.types.ptr_ty.const_null()
                    };

                    self.store_root(slot, value);
                    self.variables.insert(name.clone(), VarStorage::RootSlot(slot));
                } else {
                    let ty = self.compiler.types.ast_type_to_llvm(typ);
                    let alloca = self
                        .compiler
                        .builder
                        .build_alloca(ty, name)
                        .unwrap();

                    if let Some(expr) = init {
                        let value = self.compile_expr(expr);
                        self.compiler.builder.build_store(alloca, value).unwrap();
                    }

                    self.variables.insert(name.clone(), VarStorage::Alloca(alloca, ty));
                }
            }

            Stmt::Assign { name, value } => {
                let compiled_value = self.compile_expr(value);
                match self.variables.get(name).unwrap() {
                    VarStorage::Alloca(alloca, _ty) => {
                        self.compiler.builder.build_store(*alloca, compiled_value).unwrap();
                    }
                    VarStorage::RootSlot(slot) => {
                        self.store_root(*slot, compiled_value.into_pointer_value());
                    }
                }
            }

            Stmt::FieldSet { object, struct_name, field, value } => {
                let obj_ptr = self.compile_expr(object).into_pointer_value();
                let field_value = self.compile_expr(value);

                let struct_def = self.compiler.program.find_struct(struct_name).unwrap();
                let field_idx = struct_def.field_index(field).unwrap();

                // Get payload pointer (skip header)
                let payload_ptr = unsafe {
                    self.compiler.builder.build_gep(
                        self.compiler.types.object_header_ty,
                        obj_ptr,
                        &[self.compiler.types.i32_ty.const_int(1, false)],
                        "payload",
                    ).unwrap()
                };

                // Get field pointer
                let struct_ty = self.compiler.types.struct_types.get(struct_name).unwrap();
                let field_ptr = self.compiler.builder
                    .build_struct_gep(*struct_ty, payload_ptr, field_idx as u32, "field_ptr")
                    .unwrap();

                self.compiler.builder.build_store(field_ptr, field_value).unwrap();
            }

            Stmt::ArraySet { array, index, value } => {
                let arr_ptr = self.compile_expr(array).into_pointer_value();
                let idx = self.compile_expr(index).into_int_value();
                let elem_value = self.compile_expr(value);

                // Get payload pointer (skip header)
                let payload_ptr = unsafe {
                    self.compiler.builder.build_gep(
                        self.compiler.types.object_header_ty,
                        arr_ptr,
                        &[self.compiler.types.i32_ty.const_int(1, false)],
                        "payload",
                    ).unwrap()
                };

                // Get element pointer
                let elem_ptr = unsafe {
                    self.compiler.builder.build_gep(
                        self.compiler.types.ptr_ty,
                        payload_ptr,
                        &[idx],
                        "elem_ptr",
                    ).unwrap()
                };

                self.compiler.builder.build_store(elem_ptr, elem_value).unwrap();
            }

            Stmt::Return(expr) => {
                self.emit_epilogue();
                if let Some(e) = expr {
                    let value = self.compile_expr(e);
                    self.compiler.builder.build_return(Some(&value)).unwrap();
                } else {
                    self.compiler.builder.build_return(None).unwrap();
                }
            }

            Stmt::If { condition, then_block, else_block } => {
                let cond = self.compile_expr(condition).into_int_value();

                let then_bb = self.compiler.context.append_basic_block(self.fn_value, "then");
                let else_bb = self.compiler.context.append_basic_block(self.fn_value, "else");
                let merge_bb = self.compiler.context.append_basic_block(self.fn_value, "merge");

                self.compiler.builder.build_conditional_branch(cond, then_bb, else_bb).unwrap();

                // Then block
                self.compiler.builder.position_at_end(then_bb);
                for stmt in then_block {
                    self.compile_stmt(stmt);
                }
                if self.compiler.builder.get_insert_block().unwrap().get_terminator().is_none() {
                    self.compiler.builder.build_unconditional_branch(merge_bb).unwrap();
                }

                // Else block
                self.compiler.builder.position_at_end(else_bb);
                for stmt in else_block {
                    self.compile_stmt(stmt);
                }
                if self.compiler.builder.get_insert_block().unwrap().get_terminator().is_none() {
                    self.compiler.builder.build_unconditional_branch(merge_bb).unwrap();
                }

                self.compiler.builder.position_at_end(merge_bb);
            }

            Stmt::While { condition, body } => {
                let cond_bb = self.compiler.context.append_basic_block(self.fn_value, "while_cond");
                let body_bb = self.compiler.context.append_basic_block(self.fn_value, "while_body");
                let exit_bb = self.compiler.context.append_basic_block(self.fn_value, "while_exit");

                self.compiler.builder.build_unconditional_branch(cond_bb).unwrap();

                // Condition
                self.compiler.builder.position_at_end(cond_bb);
                let cond = self.compile_expr(condition).into_int_value();
                self.compiler.builder.build_conditional_branch(cond, body_bb, exit_bb).unwrap();

                // Body
                self.compiler.builder.position_at_end(body_bb);
                for stmt in body {
                    self.compile_stmt(stmt);
                }
                // Pollcheck at loop back-edge (safepoint)
                self.emit_pollcheck();
                if self.compiler.builder.get_insert_block().unwrap().get_terminator().is_none() {
                    self.compiler.builder.build_unconditional_branch(cond_bb).unwrap();
                }

                self.compiler.builder.position_at_end(exit_bb);
            }

            Stmt::Expr(expr) => {
                self.compile_expr(expr);
            }
        }
    }

    /// Compile an expression and return its value
    fn compile_expr(&mut self, expr: &Expr) -> BasicValueEnum<'ctx> {
        let types = &self.compiler.types;
        let builder = &self.compiler.builder;

        match expr {
            Expr::IntLit(n) => types.i64_ty.const_int(*n as u64, true).into(),

            Expr::BoolLit(b) => types.i1_ty.const_int(*b as u64, false).into(),

            Expr::Null => types.ptr_ty.const_null().into(),

            Expr::Var(name) => {
                match self.variables.get(name).unwrap() {
                    VarStorage::Alloca(alloca, ty) => {
                        builder.build_load(*ty, *alloca, name).unwrap()
                    }
                    VarStorage::RootSlot(slot) => {
                        self.load_root(*slot).into()
                    }
                }
            }

            Expr::BinOp { op, left, right } => {
                let lhs = self.compile_expr(left);
                let rhs = self.compile_expr(right);

                // Check if we're comparing pointers
                if lhs.is_pointer_value() && rhs.is_pointer_value() {
                    let lhs_ptr = lhs.into_pointer_value();
                    let rhs_ptr = rhs.into_pointer_value();
                    // Convert pointers to integers for comparison
                    let lhs_int = builder.build_ptr_to_int(lhs_ptr, types.i64_ty, "ptr_to_int_l").unwrap();
                    let rhs_int = builder.build_ptr_to_int(rhs_ptr, types.i64_ty, "ptr_to_int_r").unwrap();
                    match op {
                        BinOp::Eq => builder.build_int_compare(IntPredicate::EQ, lhs_int, rhs_int, "eq").unwrap().into(),
                        BinOp::Ne => builder.build_int_compare(IntPredicate::NE, lhs_int, rhs_int, "ne").unwrap().into(),
                        _ => panic!("Cannot use {:?} on pointer values", op),
                    }
                } else {
                    let lhs_int = lhs.into_int_value();
                    let rhs_int = rhs.into_int_value();
                    match op {
                        BinOp::Add => builder.build_int_add(lhs_int, rhs_int, "add").unwrap().into(),
                        BinOp::Sub => builder.build_int_sub(lhs_int, rhs_int, "sub").unwrap().into(),
                        BinOp::Mul => builder.build_int_mul(lhs_int, rhs_int, "mul").unwrap().into(),
                        BinOp::Div => builder.build_int_signed_div(lhs_int, rhs_int, "div").unwrap().into(),
                        BinOp::Eq => builder.build_int_compare(IntPredicate::EQ, lhs_int, rhs_int, "eq").unwrap().into(),
                        BinOp::Ne => builder.build_int_compare(IntPredicate::NE, lhs_int, rhs_int, "ne").unwrap().into(),
                        BinOp::Lt => builder.build_int_compare(IntPredicate::SLT, lhs_int, rhs_int, "lt").unwrap().into(),
                        BinOp::Le => builder.build_int_compare(IntPredicate::SLE, lhs_int, rhs_int, "le").unwrap().into(),
                        BinOp::Gt => builder.build_int_compare(IntPredicate::SGT, lhs_int, rhs_int, "gt").unwrap().into(),
                        BinOp::Ge => builder.build_int_compare(IntPredicate::SGE, lhs_int, rhs_int, "ge").unwrap().into(),
                    }
                }
            }

            Expr::NewStruct(struct_name) => {
                let meta = self.compiler.types.struct_metas.get(struct_name).unwrap();
                let struct_ty = self.compiler.types.struct_types.get(struct_name).unwrap();
                let payload_size = struct_ty.size_of().unwrap();

                let call_site = builder
                    .build_call(
                        self.compiler.runtime.allocate,
                        &[
                            self.thread.into(),
                            meta.as_pointer_value().into(),
                            payload_size.into(),
                        ],
                        "new_struct",
                    )
                    .unwrap();

                let obj = call_site.as_any_value_enum().into_pointer_value();

                // Safepoint after allocation
                self.emit_pollcheck();

                obj.into()
            }

            Expr::NewArray { element_type: _, size } => {
                let length = self.compile_expr(size).into_int_value();
                let meta = self.compiler.types.array_meta.unwrap();

                let call_site = builder
                    .build_call(
                        self.compiler.runtime.allocate_array,
                        &[
                            self.thread.into(),
                            meta.as_pointer_value().into(),
                            length.into(),
                        ],
                        "new_array",
                    )
                    .unwrap();

                let obj = call_site.as_any_value_enum().into_pointer_value();

                // Safepoint after allocation
                self.emit_pollcheck();

                obj.into()
            }

            Expr::FieldGet { object, struct_name, field } => {
                let obj_ptr = self.compile_expr(object).into_pointer_value();

                let struct_def = self.compiler.program.find_struct(struct_name).unwrap();
                let field_idx = struct_def.field_index(field).unwrap();
                let field_type = &struct_def.fields[field_idx].typ;

                // Get payload pointer (skip header)
                let payload_ptr = unsafe {
                    builder.build_gep(
                        types.object_header_ty,
                        obj_ptr,
                        &[types.i32_ty.const_int(1, false)],
                        "payload",
                    ).unwrap()
                };

                // Get field pointer and load
                let struct_ty = types.struct_types.get(struct_name).unwrap();
                let field_ptr = builder
                    .build_struct_gep(*struct_ty, payload_ptr, field_idx as u32, "field_ptr")
                    .unwrap();

                let field_llvm_ty = types.ast_type_to_llvm(field_type);
                builder.build_load(field_llvm_ty, field_ptr, "field_val").unwrap()
            }

            Expr::ArrayGet { array, index } => {
                let arr_ptr = self.compile_expr(array).into_pointer_value();
                let idx = self.compile_expr(index).into_int_value();

                // Get payload pointer (skip header)
                let payload_ptr = unsafe {
                    builder.build_gep(
                        types.object_header_ty,
                        arr_ptr,
                        &[types.i32_ty.const_int(1, false)],
                        "payload",
                    ).unwrap()
                };

                // Get element pointer and load
                let elem_ptr = unsafe {
                    builder.build_gep(
                        types.ptr_ty,
                        payload_ptr,
                        &[idx],
                        "elem_ptr",
                    ).unwrap()
                };

                builder.build_load(types.ptr_ty, elem_ptr, "elem_val").unwrap()
            }

            Expr::ArrayLen(array) => {
                let arr_ptr = self.compile_expr(array).into_pointer_value();

                // Load header.aux (the length)
                let aux_ptr = builder
                    .build_struct_gep(types.object_header_ty, arr_ptr, 2, "aux_ptr")
                    .unwrap();

                let len = builder
                    .build_load(types.i32_ty, aux_ptr, "length")
                    .unwrap()
                    .into_int_value();

                // Zero-extend to i64
                builder.build_int_z_extend(len, types.i64_ty, "length_i64").unwrap().into()
            }

            Expr::Call { function, args } => {
                let fn_value = self.functions.get(function).unwrap();

                let mut call_args: Vec<BasicMetadataValueEnum<'ctx>> = vec![self.thread.into()];
                for arg in args {
                    call_args.push(self.compile_expr(arg).into());
                }

                let call_site = builder
                    .build_call(*fn_value, &call_args, "call")
                    .unwrap();

                // Safepoint after call
                self.emit_pollcheck();

                // Try to get the return value; if void, return a dummy
                let any_val = call_site.as_any_value_enum();
                if any_val.is_pointer_value() {
                    any_val.into_pointer_value().into()
                } else if any_val.is_int_value() {
                    any_val.into_int_value().into()
                } else {
                    // Void call - return dummy value (shouldn't be used)
                    types.i64_ty.const_zero().into()
                }
            }
        }
    }
}
