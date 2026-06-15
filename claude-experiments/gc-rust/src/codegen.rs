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
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;

#[derive(Debug)]
pub struct CodegenError(pub String);

/// A compiled, JIT-ready program.
pub struct Compiled<'ctx> {
    pub module: Module<'ctx>,
    pub entry_name: String,
}

pub fn codegen<'ctx>(
    ctx: &'ctx Context,
    prog: &CoreProgram,
) -> Result<Compiled<'ctx>, CodegenError> {
    let module = ctx.create_module("gcr");
    let builder = ctx.create_builder();
    let mut cg = Codegen { ctx, module, builder, prog, funcs: HashMap::new() };
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

    let entry = prog.entry.ok_or_else(|| CodegenError("no entry".into()))?;
    let entry_name = prog.funcs[entry as usize].name.clone();

    cg.module
        .verify()
        .map_err(|e| CodegenError(format!("LLVM module verify failed: {}", e.to_string())))?;
    Ok(Compiled { module: cg.module, entry_name })
}

struct Codegen<'ctx, 'p> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    prog: &'p CoreProgram,
    funcs: HashMap<FuncId, FunctionValue<'ctx>>,
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
        if v.variants.is_some() {
            // Value enum: tag + opaque payload bytes (size from layout).
            let payload_bytes = v.size.saturating_sub(4) as u32;
            let fields: Vec<BasicTypeEnum> = vec![
                self.ctx.i32_type().as_basic_type_enum(),
                self.ctx.i8_type().array_type(payload_bytes).as_basic_type_enum(),
            ];
            return self.ctx.struct_type(&fields, false);
        }
        let fields: Vec<BasicTypeEnum> = v.fields.iter()
            .filter_map(|r| self.llvm_ty(r))
            .collect();
        self.ctx.struct_type(&fields, false)
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
        }
    }

    fn declare_fn(&self, _id: FuncId, f: &CoreFn) -> FunctionValue<'ctx> {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
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

    fn define_fn(&mut self, id: FuncId, f: &CoreFn) -> Result<(), CodegenError> {
        let func = self.funcs[&id];
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        let ptr = self.ctx.ptr_type(AddressSpace::default());

        // Partition locals: Ref-typed locals become GC frame root slots; all
        // others get plain allocas. Count the refs to size the frame.
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

        // Emit the GC frame `{ parent, origin, [num_roots x ptr] }` when there
        // are any roots. Even with zero roots we keep the chain consistent so
        // the GC always sees a well-formed parent link; but if there are no
        // roots and no allocations, the frame is unnecessary — we still emit it
        // when num_roots > 0.
        let frame = if num_roots > 0 {
            let roots_arr = ptr.array_type(num_roots as u32);
            let frame_ty = self.ctx.struct_type(&[ptr.into(), ptr.into(), roots_arr.into()], false);
            let frame = self.builder.build_alloca(frame_ty, "gcframe").unwrap();
            // origin global
            let origin = self.frame_origin(&f.name, num_roots as u32);
            let origin_field = self.builder.build_struct_gep(frame_ty, frame, 1, "origin.f").unwrap();
            self.builder.build_store(origin_field, origin).unwrap();
            // link: parent = thread.top_frame; thread.top_frame = &frame
            let tf_ptr = self.thread_field_ptr(func, crate::runtime::thread_offsets::TOP_FRAME);
            let prev = self.builder.build_load(ptr, tf_ptr, "prevtop").unwrap();
            let parent_field = self.builder.build_struct_gep(frame_ty, frame, 0, "parent.f").unwrap();
            self.builder.build_store(parent_field, prev).unwrap();
            self.builder.build_store(tf_ptr, frame).unwrap();
            // zero all root slots
            let roots_field = self.builder.build_struct_gep(frame_ty, frame, 2, "roots.f").unwrap();
            for k in 0..num_roots {
                let slot = unsafe {
                    self.builder.build_in_bounds_gep(
                        roots_arr, roots_field,
                        &[self.ctx.i32_type().const_zero(), self.ctx.i32_type().const_int(k as u64, false)],
                        "rinit",
                    ).unwrap()
                };
                self.builder.build_store(slot, ptr.const_null()).unwrap();
            }
            Some((frame, frame_ty, roots_arr, roots_field, tf_ptr))
        } else {
            None
        };

        // Build slots. Ref locals point at their frame root slot; others get
        // plain allocas.
        let mut slots: Vec<Option<PointerValue<'ctx>>> = Vec::with_capacity(f.locals.len());
        for (i, lr) in f.locals.iter().enumerate() {
            if let Some(ri) = root_index[i] {
                let (_, _, roots_arr, roots_field, _) = frame.unwrap();
                let slot = unsafe {
                    self.builder.build_in_bounds_gep(
                        roots_arr, roots_field,
                        &[self.ctx.i32_type().const_zero(), self.ctx.i32_type().const_int(ri as u64, false)],
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

        let unlink = frame.map(|(_, _, _, _, tf_ptr)| {
            let (frame_ptr, frame_ty, ..) = frame.unwrap();
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
        };
        let val = self.gen_block(&mut fcx, &f.body)?;

        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            self.emit_unlink(&fcx);
            match val {
                Some(v) => { self.builder.build_return(Some(&v)).unwrap(); }
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
    fn frame_origin(&self, fn_name: &str, num_roots: u32) -> PointerValue<'ctx> {
        let i32t = self.ctx.i32_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let origin_ty = self.ctx.struct_type(&[i32t.into(), i32t.into(), ptr.into()], false);
        let g = self.module.add_global(origin_ty, None, &format!("__origin_{}", fn_name));
        g.set_constant(true);
        g.set_initializer(&origin_ty.const_named_struct(&[
            i32t.const_int(num_roots as u64, false).into(),
            i32t.const_zero().into(),
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
            CoreExprKind::ConstStr(s) => self.gen_const_str(fcx, s, &e.repr),
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
            CoreExprKind::Cast { value, from, to } => self.gen_cast(fcx, value, from, to),
            CoreExprKind::Call(fid, args) => self.gen_call(fcx, *fid, args),
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
            CoreExprKind::New { layout, fields } => self.gen_alloc(fcx, *layout, None, fields),
            CoreExprKind::MakeVariant { layout, tag, fields } => {
                self.gen_alloc(fcx, *layout, Some(*tag), fields)
            }
            CoreExprKind::Field { base, loc } => self.gen_field(fcx, base, loc),
            CoreExprKind::SetField { base, loc, value } => {
                let obj = self.gen_expr(fcx, base)?.unwrap().into_pointer_value();
                let v = self.gen_expr(fcx, value)?.unwrap();
                let off = match loc {
                    FieldLoc::Ptr { idx } => Self::HEADER + (*idx as u64) * 8,
                    FieldLoc::Raw { offset, .. } => {
                        let lid = match &base.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("setfield on non-ref".into())) };
                        let lay = &self.prog.layouts[lid as usize];
                        Self::HEADER + (lay.ptr_fields as u64) * 8 + *offset as u64
                    }
                    FieldLoc::ValueField { .. } => return Err(CodegenError("value-struct field is immutable".into())),
                };
                let addr = self.obj_addr(obj, off);
                self.builder.build_store(addr, v).unwrap();
                Ok(Some(self.ctx.i64_type().const_zero().into()))
            }
            CoreExprKind::Match { scrutinee, arms } => self.gen_match(fcx, scrutinee, arms, &e.repr),
            CoreExprKind::ArrayNew { layout, len, elem } => self.gen_array_new(fcx, *layout, len, elem),
            CoreExprKind::ArrayLen(arr) => self.gen_array_len(fcx, arr),
            CoreExprKind::ArrayGet { array, index, elem } => self.gen_array_get(fcx, array, index, elem),
            CoreExprKind::ArraySet { array, index, value, elem } => self.gen_array_set(fcx, array, index, value, elem),
            CoreExprKind::MakeClosure { code, env, captures } => self.gen_make_closure(fcx, *code, *env, captures),
            CoreExprKind::CallClosure { callee, args } => self.gen_call_closure(fcx, callee, args, &e.repr),
            other => Err(CodegenError(format!("codegen unsupported in v0 slice: {:?}", core_disc(other)))),
        }
    }

    fn declare_runtime_externs(&self) {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let i32t = self.ctx.i32_type();
        let i64t = self.ctx.i64_type();
        // ptr ai_gc_alloc_fixed(ptr thread, i32 type_id)
        self.module.add_function(
            "ai_gc_alloc_fixed",
            ptr.fn_type(&[ptr.into(), i32t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // ptr ai_gc_alloc_varlen(ptr thread, i32 type_id, i64 n)
        self.module.add_function(
            "ai_gc_alloc_varlen",
            ptr.fn_type(&[ptr.into(), i32t.into(), i64t.into()], false),
            Some(inkwell::module::Linkage::External),
        );
        // void ai_gc_pollcheck_slow(ptr thread)
        self.module.add_function(
            "ai_gc_pollcheck_slow",
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
    }

    /// The byte offset of a heap object's data start (the `Full` header size).
    const HEADER: u64 = 16;

    /// Allocate a heap object of `layout`, store `tag` (for enums) + `fields`,
    /// and return the pointer. The fields are evaluated and stored; pointer
    /// fields go in the leading pointer slots, raw fields at their byte offset.
    fn gen_alloc(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        layout: LayoutId,
        tag: Option<u32>,
        fields: &[CoreExpr],
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let i32t = self.ctx.i32_type();
        let i64t = self.ctx.i64_type();

        // Evaluate field values FIRST (they may allocate; their own roots are
        // already spilled by their sub-expressions). Then allocate, then store.
        let mut vals = Vec::with_capacity(fields.len());
        for fe in fields {
            vals.push((self.gen_expr(fcx, fe)?, fe.repr.clone()));
        }

        let alloc = self.module.get_function("ai_gc_alloc_fixed").unwrap();
        let obj = call_result(
            self.builder.build_call(
                alloc,
                &[fcx.thread.into(), i32t.const_int(layout as u64, false).into()],
                "obj",
            ).unwrap(),
        ).into_pointer_value();

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
                Repr::Value(_) | Repr::Unit => {
                    // value-typed fields: not yet stored field-by-field in v0.
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
        let sty = self.value_struct_ty(value);
        let i32t = self.ctx.i32_type();
        // Evaluate payload field values first.
        let mut vals = Vec::with_capacity(fields.len());
        for f in fields {
            if let Some(v) = self.gen_expr(fcx, f)? {
                vals.push((v, f.repr.clone()));
            }
        }
        // Alloca the aggregate, zero it, store tag + payload, reload.
        let slot = self.builder.build_alloca(sty, "ve").unwrap();
        self.builder.build_store(slot, sty.const_zero()).unwrap();
        // tag = field 0.
        let tag_addr = self.builder.build_struct_gep(sty, slot, 0, "ve.tag").unwrap();
        self.builder.build_store(tag_addr, i32t.const_int(tag as u64, false)).unwrap();
        // payload bytes = field 1; store each field at its running byte offset.
        if sty.count_fields() > 1 {
            let payload_addr = self.builder.build_struct_gep(sty, slot, 1, "ve.pl").unwrap();
            let mut off = 0u64;
            for (v, repr) in &vals {
                let (sz, _) = Self::repr_size_align(repr);
                off = align_up64(off, sz);
                let field_addr = self.payload_field_addr(payload_addr, off);
                self.builder.build_store(field_addr, *v).unwrap();
                off += sz;
            }
        }
        let agg = self.builder.build_load(sty, slot, "ve.val").unwrap();
        Ok(Some(agg))
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

    /// The byte stride of an array element repr, and whether it's a traced
    /// (pointer/Values) array.
    fn elem_stride(elem: &Repr) -> (u64, bool) {
        match elem {
            Repr::Ref(_) => (8, true),
            Repr::Scalar(s) => ((s.bits().max(8) / 8) as u64, false),
            _ => (8, false),
        }
    }

    /// Element address: HEADER + 8 (count word) + index*stride.
    fn array_elem_addr(&self, arr: PointerValue<'ctx>, index: IntValue<'ctx>, stride: u64) -> PointerValue<'ctx> {
        let i64t = self.ctx.i64_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let base = self.builder.build_ptr_to_int(arr, i64t, "a.i").unwrap();
        let hdr = self.builder.build_int_add(base, i64t.const_int(Self::HEADER + 8, false), "a.h").unwrap();
        let idx64 = if index.get_type().get_bit_width() < 64 {
            self.builder.build_int_s_extend(index, i64t, "a.ix").unwrap()
        } else { index };
        let off = self.builder.build_int_mul(idx64, i64t.const_int(stride, false), "a.off").unwrap();
        let addr = self.builder.build_int_add(hdr, off, "a.ea").unwrap();
        self.builder.build_int_to_ptr(addr, ptr, "a.ep").unwrap()
    }

    /// Allocate a `String` varlen object and copy the literal's UTF-8 bytes in.
    fn gen_const_str(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        s: &str,
        repr: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let i32t = self.ctx.i32_type();
        let i64t = self.ctx.i64_type();
        let lid = match repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("string repr".into())) };
        let bytes = s.as_bytes();
        let n = bytes.len() as u64;
        let alloc = self.module.get_function("ai_gc_alloc_varlen").unwrap();
        let obj = call_result(self.builder.build_call(
            alloc,
            &[fcx.thread.into(), i32t.const_int(lid as u64, false).into(), i64t.const_int(n, false).into()],
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
        let alloc = self.module.get_function("ai_gc_alloc_varlen").unwrap();
        let obj = call_result(self.builder.build_call(
            alloc,
            &[fcx.thread.into(), i32t.const_int(layout as u64, false).into(), varlen_len.into()],
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

    fn gen_array_get(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        array: &CoreExpr,
        index: &CoreExpr,
        elem: &Repr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let obj = self.gen_expr(fcx, array)?.unwrap().into_pointer_value();
        let idx = self.gen_expr(fcx, index)?.unwrap().into_int_value();
        let (stride, _) = Self::elem_stride(elem);
        let addr = self.array_elem_addr(obj, idx, stride);
        let lty = self.llvm_ty(elem).ok_or_else(|| CodegenError("array of unit".into()))?;
        let v = self.builder.build_load(lty, addr, "aget").unwrap();
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
        let (stride, _) = Self::elem_stride(elem);
        let addr = self.array_elem_addr(obj, idx, stride);
        self.builder.build_store(addr, val).unwrap();
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
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let i32t = self.ctx.i32_type();
        // Evaluate captures first (they may allocate).
        let mut vals = Vec::with_capacity(captures.len());
        for c in captures {
            vals.push((self.gen_expr(fcx, c)?, c.repr.clone()));
        }
        let alloc = self.module.get_function("ai_gc_alloc_fixed").unwrap();
        let obj = call_result(
            self.builder.build_call(alloc, &[fcx.thread.into(), i32t.const_int(env as u64, false).into()], "clo").unwrap(),
        ).into_pointer_value();

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
                _ => {}
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
        let env_lid = match &callee.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("call on non-closure".into())) };
        let lay = &self.prog.layouts[env_lid as usize];
        let raw_base = Self::HEADER + (lay.ptr_fields as u64) * 8;
        let code_addr = self.obj_addr(env, raw_base);
        let code_ptr = self.builder.build_load(ptr, code_addr, "code").unwrap().into_pointer_value();

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
        let agg = self.gen_expr(fcx, scrutinee)?.unwrap().into_struct_value();
        // Spill the aggregate so we can address its payload bytes for binds.
        let scrut_slot = self.builder.build_alloca(scrut_ty, "vm.scrut").unwrap();
        self.builder.build_store(scrut_slot, agg).unwrap();
        let tag = self.builder.build_extract_value(agg, 0, "vetag").unwrap().into_int_value();
        // Compute the payload pointer while still in the current (entry) block,
        // before we move the builder to the switch/arm blocks.
        let payload_ptr = if scrut_ty.count_fields() > 1 {
            Some(self.builder.build_struct_gep(scrut_ty, scrut_slot, 1, "vm.pl").unwrap())
        } else { None };
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

        for (arm, bb) in arm_blocks {
            self.builder.position_at_end(bb);
            // Bind payload fields by loading them from the byte region.
            if let Some(payload_ptr) = payload_ptr {
                let mut off = 0u64;
                for &local in &arm.binds {
                    let lrepr = fcx.local_reprs[local as usize].clone();
                    if let Some(lty) = self.llvm_ty(&lrepr) {
                        let (sz, _) = Self::repr_size_align(&lrepr);
                        off = align_up64(off, sz);
                        let faddr = self.payload_field_addr(payload_ptr, off);
                        off += sz;
                        let v = self.builder.build_load(lty, faddr, "vm.bind").unwrap();
                        if let Some(slot) = fcx.slots[local as usize] {
                            self.builder.build_store(slot, v).unwrap();
                        }
                    }
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
        let mut cargs: Vec<inkwell::values::BasicMetadataValueEnum> =
            vec![fcx.thread.into()];
        for a in args {
            if let Some(v) = self.gen_expr(fcx, a)? {
                cargs.push(v.into());
            }
        }
        let cs = self.builder.build_call(callee, &cargs, "call").unwrap();
        Ok(match cs.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => Some(v),
            inkwell::values::ValueKind::Instruction(_) => None,
        })
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
            ti
        })
        .collect()
}

/// JIT-compile and run a 0-arg `-> i64` entry with a real GC runtime, returning
/// its result. Sets up a [`RuntimeContext`] over the program's layouts, wires
/// the `ai_gc_*` externs, and passes a live `Thread*`. `stress` forces a
/// collection on every allocation (exercises the GC + precise roots).
pub fn jit_run_i64_gc(prog: &CoreProgram, stress: bool) -> Result<i64, CodegenError> {
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
        ("ai_print_int", runtime::ai_print_int as *const () as usize),
        ("ai_print_float", runtime::ai_print_float as *const () as usize),
    ] {
        if let Some(f) = compiled.module.get_function(name) {
            ee.add_global_mapping(&f, addr);
        }
    }

    let tis = layouts_to_type_infos(prog);
    // A generous semi-space (per space). Large enough for allocation-heavy
    // programs like binary_trees whose live set is several MB; the copying
    // collector reclaims between iterations. Stress mode collects every alloc.
    let space = if stress { 8 << 20 } else { 256 << 20 };
    let mut rt = RuntimeContext::new(space, tis);
    if stress {
        rt.heap().set_gc_every_alloc(true);
    }

    let addr = ee
        .get_function_address(&compiled.entry_name)
        .map_err(|_| CodegenError("entry not found".into()))?;
    type EntryFn = unsafe extern "C" fn(*mut runtime::Thread) -> i64;
    let f: EntryFn = unsafe { std::mem::transmute(addr) };
    let thread = rt.thread_ptr();
    Ok(unsafe { f(thread) })
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
            OptimizationLevel::Aggressive,
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
    use inkwell::targets::FileType;

    let ctx = Context::create();
    let compiled = codegen(&ctx, prog)?;
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

    // ---- Declare gcr_runtime_main -----------------------------------------
    // i64 gcr_runtime_main(ptr layouts, i64 ti_count, ptr entry)
    let runtime_main_ty = i64t.fn_type(&[ptr.into(), i64t.into(), ptr.into()], false);
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
    optimize_module(module);
    let machine = host_target_machine()?;
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
) -> Result<(), CodegenError> {
    // Emit the object next to the output so cleanup is easy and paths are stable.
    let obj_path = out_path.with_extension("o");
    codegen_aot_object(prog, &obj_path)?;

    let staticlib = locate_runtime_staticlib()?;

    // Link: object + runtime staticlib + system libs. The runtime is Rust, so it
    // needs libpthread/libdl/libm/libc + (on glibc) libgcc_s for unwinding. `cc`
    // pulls libc; we add the rest explicitly.
    let status = std::process::Command::new("cc")
        .arg("-o")
        .arg(out_path)
        .arg(&obj_path)
        .arg(&staticlib)
        .args(["-lpthread", "-ldl", "-lm"])
        .status()
        .map_err(|e| CodegenError(format!("failed to invoke linker (cc): {}", e)))?;
    if !status.success() {
        return Err(CodegenError(format!(
            "linker (cc) failed with status {}",
            status
        )));
    }

    // Remove the intermediate object on success.
    let _ = std::fs::remove_file(&obj_path);
    Ok(())
}

/// Locate the gc-rust runtime staticlib (`libgcrust_rt.a`).
///
/// Cargo builds it as a `staticlib` crate-type artifact under the target dir.
/// We search the standard cargo profiles relative to the manifest dir captured
/// at build time, honoring `$GCRUST_RUNTIME_LIB` as an explicit override.
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
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // The current exe lives in target/<profile>/ (or target/<profile>/deps/ for
    // tests); the staticlib is alongside in the same profile dir.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        // .../target/<profile>/gcr  -> dir = .../target/<profile>
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("libgcrust_rt.a"));
            // tests run from .../deps/; the lib is one level up.
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join("libgcrust_rt.a"));
            }
        }
    }
    for profile in ["release", "debug"] {
        candidates.push(manifest.join("target").join(profile).join("libgcrust_rt.a"));
    }
    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }
    // Not found — try to build it on demand so `gcr build` "just works" even
    // after a plain `cargo build` (which only produces the rlib, not the
    // staticlib). Best-effort: if cargo isn't available we fall through to the
    // clear error below.
    let _ = std::process::Command::new("cargo")
        .args(["build", "-p", "gcrust-rt"])
        .current_dir(&manifest)
        .status();
    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }
    Err(CodegenError(format!(
        "could not find the gc-rust runtime staticlib (libgcrust_rt.a), and \
         building it via `cargo build -p gcrust-rt` did not produce it. \
         Set $GCRUST_RUNTIME_LIB to its path. Searched: {}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
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
                     let a: Array<i64> = array_new(4); \
                     a[0] = 10; a[1] = 20; a[3] = 12; \
                     a[0] + a[1] + a[3] \
                   }";
        assert_eq!(run_gc(src, false), 42);
    }

    #[test]
    fn index_in_for_loop() {
        let src = "fn main() -> i64 { \
                     let a: Array<i64> = array_new(50); \
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
                     let a: Array<i64> = array_new(2); \
                     a[0] = 10; a[0] += 5; a[0] *= 2; \
                     a[0] \
                   }";
        assert_eq!(run_gc(src, false), 30);
    }

    // ---- value types (inline, flat) ---------------------------------------

    #[test]
    fn value_enum_with_payloads() {
        let src = "value enum Shape { Circle(i64), Rect(i64, i64), Empty } \
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
        let src = "value enum Opt { None, Some(i64) } \
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
        let src = "value enum Color { Red, Green, Blue } \
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
        let src = "value enum Dir { N, S, E, W } \
                   fn dx(d: Dir) -> i64 { match d { Dir::E => 1, Dir::W => 0 - 1, Dir::N => 0, Dir::S => 0 } } \
                   fn main() -> i64 { dx(Dir::E) + dx(Dir::W) + dx(Dir::E) }";
        assert_eq!(run_gc(src, true), 1);
    }

    #[test]
    fn value_struct_field_access() {
        let src = "value struct Vec3 { x: i64, y: i64, z: i64 } \
                   fn main() -> i64 { \
                       let v = Vec3 { x: 3, y: 4, z: 5 }; \
                       v.x + v.y + v.z \
                   }";
        assert_eq!(run_gc(src, false), 12);
    }

    #[test]
    fn value_struct_passed_by_value() {
        let src = "value struct Vec3 { x: f64, y: f64, z: f64 } \
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
        let src = "value struct P { a: i64, b: i64 } \
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
        let src = "value struct Pt { x: i64, y: i64 } \
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
                     let a: Array<i64> = array_new(5); \
                     array_set(a, 0, 10); \
                     array_set(a, 1, 20); \
                     array_set(a, 4, 12); \
                     array_get(a, 0) + array_get(a, 1) + array_get(a, 4) \
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
                     let a: Array<i64> = array_new(100); \
                     let mut i = 0; \
                     while i < 100 { array_set(a, i, i); i = i + 1; } \
                     let mut sum = 0; \
                     let mut j = 0; \
                     while j < 100 { sum = sum + array_get(a, j); j = j + 1; } \
                     sum \
                   }";
        // sum 0..99 = 4950
        assert_eq!(run_gc(src, false), 4950);
    }

    #[test]
    fn array_float() {
        let src = "fn main() -> i64 { \
                     let a: Array<f64> = array_new(3); \
                     array_set(a, 0, 1.5); \
                     array_set(a, 1, 2.5); \
                     array_set(a, 2, 4.0); \
                     (array_get(a, 0) + array_get(a, 1) + array_get(a, 2)) as i64 \
                   }";
        assert_eq!(run_gc(src, false), 8);
    }

    #[test]
    fn array_survives_gc_stress() {
        // Array of i64 filled, then more allocations under stress; the array is
        // rooted and its scalar contents must survive (untraced Bytes tail).
        let src = "fn mk() -> Array<i64> { array_new(10) } \
                   fn main() -> i64 { \
                       let a: Array<i64> = array_new(3); \
                       array_set(a, 0, 100); array_set(a, 1, 20); array_set(a, 2, 3); \
                       let _x = mk(); let _y = mk(); \
                       array_get(a, 0) + array_get(a, 1) + array_get(a, 2) \
                   }";
        assert_eq!(run_gc(src, true), 123);
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

    const PRELUDE: &str = "enum Option<T> { None, Some(T) } \
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
