use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicTypeEnum, IntType, PointerType, StructType, VoidType};
use inkwell::values::{FunctionValue, GlobalValue, PointerValue};
use inkwell::AddressSpace;
use std::collections::HashMap;

use crate::ast::{Program, StructDef, Type};

/// LLVM types used throughout code generation
pub struct LLVMTypes<'ctx> {
    pub void_ty: VoidType<'ctx>,
    pub i1_ty: IntType<'ctx>,
    pub i8_ty: IntType<'ctx>,
    pub i32_ty: IntType<'ctx>,
    pub i64_ty: IntType<'ctx>,
    pub ptr_ty: PointerType<'ctx>,

    /// FrameOrigin: { num_roots: i32, function_name: ptr }
    pub frame_origin_ty: StructType<'ctx>,

    /// Frame header: { parent: ptr, origin: ptr }
    /// Actual frames have roots array following this
    pub frame_header_ty: StructType<'ctx>,

    /// ObjectHeader: { meta: ptr, gc_flags: i32, aux: i32 }
    pub object_header_ty: StructType<'ctx>,

    /// ObjectMeta: { object_type: i32, num_pointer_fields: i32, pointer_field_offsets: ptr, type_name: ptr }
    pub object_meta_ty: StructType<'ctx>,

    /// Maps struct names to their LLVM struct types (just the payload, not including header)
    pub struct_types: HashMap<String, StructType<'ctx>>,

    /// Maps struct names to their ObjectMeta globals
    pub struct_metas: HashMap<String, GlobalValue<'ctx>>,

    /// ObjectMeta for arrays (shared)
    pub array_meta: Option<GlobalValue<'ctx>>,
}

impl<'ctx> LLVMTypes<'ctx> {
    pub fn new(context: &'ctx Context, module: &Module<'ctx>, program: &Program) -> Self {
        let void_ty = context.void_type();
        let i1_ty = context.bool_type();
        let i8_ty = context.i8_type();
        let i32_ty = context.i32_type();
        let i64_ty = context.i64_type();
        let ptr_ty = context.ptr_type(AddressSpace::default());

        // FrameOrigin: { num_roots: i32, function_name: ptr }
        let frame_origin_ty = context.struct_type(
            &[i32_ty.into(), ptr_ty.into()],
            false,
        );

        // Frame header: { parent: ptr, origin: ptr }
        let frame_header_ty = context.struct_type(
            &[ptr_ty.into(), ptr_ty.into()],
            false,
        );

        // ObjectHeader: { meta: ptr, gc_flags: i32, aux: i32 }
        let object_header_ty = context.struct_type(
            &[ptr_ty.into(), i32_ty.into(), i32_ty.into()],
            false,
        );

        // ObjectMeta: { object_type: i32, num_pointer_fields: i32, pointer_field_offsets: ptr, type_name: ptr }
        let object_meta_ty = context.struct_type(
            &[i32_ty.into(), i32_ty.into(), ptr_ty.into(), ptr_ty.into()],
            false,
        );

        let mut types = Self {
            void_ty,
            i1_ty,
            i8_ty,
            i32_ty,
            i64_ty,
            ptr_ty,
            frame_origin_ty,
            frame_header_ty,
            object_header_ty,
            object_meta_ty,
            struct_types: HashMap::new(),
            struct_metas: HashMap::new(),
            array_meta: None,
        };

        // Create LLVM types for all structs in the program
        for struct_def in &program.structs {
            types.create_struct_type(context, module, struct_def);
        }

        // Create array meta
        types.create_array_meta(context, module);

        types
    }

    /// Create an LLVM struct type and its metadata for a struct definition
    fn create_struct_type(
        &mut self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        struct_def: &StructDef,
    ) {
        // Build the payload type (fields only, no header)
        let field_types: Vec<BasicTypeEnum> = struct_def
            .fields
            .iter()
            .map(|f| self.ast_type_to_llvm(&f.typ))
            .collect();

        let struct_ty = context.struct_type(&field_types, false);
        self.struct_types.insert(struct_def.name.clone(), struct_ty);

        // Create metadata for GC
        self.create_struct_meta(context, module, struct_def);
    }

    /// Create ObjectMeta for a struct type
    fn create_struct_meta(
        &mut self,
        context: &'ctx Context,
        module: &Module<'ctx>,
        struct_def: &StructDef,
    ) {
        // Find all GC pointer fields
        let gc_fields = struct_def.gc_field_indices();
        let num_gc_fields = gc_fields.len() as u32;

        // Create type name string
        let name_str = context.const_string(struct_def.name.as_bytes(), true);
        let name_global = module.add_global(
            name_str.get_type(),
            None,
            &format!("__type_name_{}", struct_def.name),
        );
        name_global.set_initializer(&name_str);
        name_global.set_linkage(inkwell::module::Linkage::Private);

        // Create pointer field offsets array
        let offsets_global = if num_gc_fields > 0 {
            // Calculate byte offsets for each GC field
            // We need to compute the offset of each field in the struct
            let struct_ty = self.struct_types.get(&struct_def.name).unwrap();

            let offsets: Vec<_> = gc_fields
                .iter()
                .map(|&idx| self.i32_ty.const_int(idx as u64 * 8, false)) // Simplified: assume 8-byte fields
                .collect();

            let offsets_array = self.i32_ty.const_array(&offsets);
            let offsets_global = module.add_global(
                offsets_array.get_type(),
                None,
                &format!("__gc_offsets_{}", struct_def.name),
            );
            offsets_global.set_initializer(&offsets_array);
            offsets_global.set_linkage(inkwell::module::Linkage::Private);
            offsets_global.as_pointer_value()
        } else {
            self.ptr_ty.const_null()
        };

        // Create ObjectMeta struct
        // { object_type: i32, num_pointer_fields: i32, pointer_field_offsets: ptr, type_name: ptr }
        let meta_value = self.object_meta_ty.const_named_struct(&[
            self.i32_ty.const_int(0, false).into(), // ObjectType::Struct = 0
            self.i32_ty.const_int(num_gc_fields as u64, false).into(),
            offsets_global.into(),
            name_global.as_pointer_value().into(),
        ]);

        let meta_global = module.add_global(
            self.object_meta_ty,
            None,
            &format!("__meta_{}", struct_def.name),
        );
        meta_global.set_initializer(&meta_value);
        meta_global.set_linkage(inkwell::module::Linkage::Private);
        meta_global.set_constant(true);

        self.struct_metas.insert(struct_def.name.clone(), meta_global);
    }

    /// Create ObjectMeta for arrays
    fn create_array_meta(&mut self, context: &'ctx Context, module: &Module<'ctx>) {
        let name_str = context.const_string(b"array", true);
        let name_global = module.add_global(name_str.get_type(), None, "__type_name_array");
        name_global.set_initializer(&name_str);
        name_global.set_linkage(inkwell::module::Linkage::Private);

        // Arrays: object_type = 1, num_pointer_fields = 0 (length is in aux field)
        let meta_value = self.object_meta_ty.const_named_struct(&[
            self.i32_ty.const_int(1, false).into(), // ObjectType::Array = 1
            self.i32_ty.const_zero().into(),
            self.ptr_ty.const_null().into(),
            name_global.as_pointer_value().into(),
        ]);

        let meta_global = module.add_global(self.object_meta_ty, None, "__meta_array");
        meta_global.set_initializer(&meta_value);
        meta_global.set_linkage(inkwell::module::Linkage::Private);
        meta_global.set_constant(true);

        self.array_meta = Some(meta_global);
    }

    /// Convert an AST type to its LLVM representation
    pub fn ast_type_to_llvm(&self, typ: &Type) -> BasicTypeEnum<'ctx> {
        match typ {
            Type::Void => panic!("Void is not a value type"),
            Type::I32 => self.i32_ty.into(),
            Type::I64 => self.i64_ty.into(),
            Type::Bool => self.i1_ty.into(),
            Type::Struct(_) => self.ptr_ty.into(), // All GC refs are pointers
            Type::Array(_) => self.ptr_ty.into(),  // All GC refs are pointers
        }
    }

    /// Get the frame type for a function with a given number of roots
    pub fn frame_type(&self, context: &'ctx Context, num_roots: usize) -> StructType<'ctx> {
        // { parent: ptr, origin: ptr, roots: [ptr x num_roots] }
        context.struct_type(
            &[
                self.ptr_ty.into(),
                self.ptr_ty.into(),
                self.ptr_ty.array_type(num_roots as u32).into(),
            ],
            false,
        )
    }

    /// Get the full object type (header + payload) for a struct
    pub fn full_object_type(&self, context: &'ctx Context, struct_name: &str) -> StructType<'ctx> {
        let payload_ty = self.struct_types.get(struct_name).expect("Unknown struct");
        // { header: ObjectHeader, payload: StructType }
        context.struct_type(
            &[self.object_header_ty.into(), (*payload_ty).into()],
            false,
        )
    }
}

/// Runtime functions that the generated code calls
pub struct RuntimeFunctions<'ctx> {
    /// void gc_pollcheck_slow(thread*, origin*)
    pub pollcheck_slow: FunctionValue<'ctx>,

    /// ptr gc_allocate(thread*, meta*, size)
    pub allocate: FunctionValue<'ctx>,

    /// ptr gc_allocate_array(thread*, meta*, length)
    pub allocate_array: FunctionValue<'ctx>,

    /// void gc_write_barrier(object_ptr, new_value) - notify GC of pointer store (slow path)
    pub write_barrier: FunctionValue<'ctx>,

    /// i64 print_int(i64 value) - returns the value for chaining
    pub print_int: FunctionValue<'ctx>,

    /// External global: young generation start address
    pub young_gen_start: GlobalValue<'ctx>,

    /// External global: young generation end address
    pub young_gen_end: GlobalValue<'ctx>,
}

impl<'ctx> RuntimeFunctions<'ctx> {
    pub fn declare(module: &Module<'ctx>, types: &LLVMTypes<'ctx>) -> Self {
        // void gc_pollcheck_slow(ptr thread, ptr origin)
        let pollcheck_slow_ty = types.void_ty.fn_type(
            &[types.ptr_ty.into(), types.ptr_ty.into()],
            false,
        );
        let pollcheck_slow =
            module.add_function("gc_pollcheck_slow", pollcheck_slow_ty, None);

        // ptr gc_allocate(ptr thread, ptr meta, i64 payload_size)
        let allocate_ty = types.ptr_ty.fn_type(
            &[types.ptr_ty.into(), types.ptr_ty.into(), types.i64_ty.into()],
            false,
        );
        let allocate = module.add_function("gc_allocate", allocate_ty, None);

        // ptr gc_allocate_array(ptr thread, ptr meta, i64 length)
        let allocate_array_ty = types.ptr_ty.fn_type(
            &[types.ptr_ty.into(), types.ptr_ty.into(), types.i64_ty.into()],
            false,
        );
        let allocate_array = module.add_function("gc_allocate_array", allocate_array_ty, None);

        // void gc_write_barrier(ptr object_ptr) - mark card for object if in old gen
        let write_barrier_ty = types.void_ty.fn_type(
            &[types.ptr_ty.into()],
            false,
        );
        let write_barrier = module.add_function("gc_write_barrier", write_barrier_ty, None);

        // i64 print_int(i64 value) - for debugging
        let print_int_ty = types.i64_ty.fn_type(&[types.i64_ty.into()], false);
        let print_int = module.add_function("print_int", print_int_ty, None);

        // External globals for young generation bounds (for inline write barrier check)
        // Mark as constant since they don't change during program execution
        let young_gen_start = module.add_global(types.i64_ty, None, "YOUNG_GEN_START");
        young_gen_start.set_linkage(inkwell::module::Linkage::External);
        young_gen_start.set_constant(true);

        let young_gen_end = module.add_global(types.i64_ty, None, "YOUNG_GEN_END");
        young_gen_end.set_linkage(inkwell::module::Linkage::External);
        young_gen_end.set_constant(true);

        Self {
            pollcheck_slow,
            allocate,
            allocate_array,
            write_barrier,
            print_int,
            young_gen_start,
            young_gen_end,
        }
    }
}
