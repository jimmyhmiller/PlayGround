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
            // heap refs lower to opaque pointers (wired with the GC phase).
            Repr::Ref(_) | Repr::Value(_) => {
                Some(self.ctx.ptr_type(AddressSpace::default()).as_basic_type_enum())
            }
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
        }
    }

    fn declare_fn(&self, _id: FuncId, f: &CoreFn) -> FunctionValue<'ctx> {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        // First param is the Thread*.
        let mut params: Vec<BasicMetadataTypeEnum> = vec![ptr.into()];
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

        // Store incoming params (LLVM param 0 is Thread*).
        let nparams = f.params.len();
        let mut llvm_idx = 1u32;
        for i in 0..nparams {
            if self.llvm_ty(&f.locals[i]).is_some() {
                if let Some(slot) = slots[i] {
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
            CoreExprKind::New { layout, fields } => self.gen_alloc(fcx, *layout, None, fields),
            CoreExprKind::MakeVariant { layout, tag, fields } => {
                self.gen_alloc(fcx, *layout, Some(*tag), fields)
            }
            CoreExprKind::Field { base, loc } => self.gen_field(fcx, base, loc),
            CoreExprKind::Match { scrutinee, arms } => self.gen_match(fcx, scrutinee, arms, &e.repr),
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

    fn obj_addr(&self, obj: PointerValue<'ctx>, byte_off: u64) -> PointerValue<'ctx> {
        let i64t = self.ctx.i64_type();
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let base = self.builder.build_ptr_to_int(obj, i64t, "o.i").unwrap();
        let addr = self.builder.build_int_add(base, i64t.const_int(byte_off, false), "o.fa").unwrap();
        self.builder.build_int_to_ptr(addr, ptr, "o.fp").unwrap()
    }

    fn gen_field(
        &mut self,
        fcx: &mut FnCtx<'ctx>,
        base: &CoreExpr,
        loc: &FieldLoc,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let obj = self.gen_expr(fcx, base)?.unwrap().into_pointer_value();
        let (off, lty): (u64, BasicTypeEnum) = match loc {
            FieldLoc::Ptr { idx } => (
                Self::HEADER + (*idx as u64) * 8,
                self.ctx.ptr_type(AddressSpace::default()).as_basic_type_enum(),
            ),
            FieldLoc::Raw { offset, repr } => {
                // raw offset is relative to the raw section, which starts after
                // the pointer slots. We need the base layout's ptr_fields.
                let lid = match &base.repr { Repr::Ref(l) => *l, _ => return Err(CodegenError("field on non-ref".into())) };
                let lay = &self.prog.layouts[lid as usize];
                let raw_base = Self::HEADER + (lay.ptr_fields as u64) * 8;
                (raw_base + *offset as u64, self.scalar_ty(*repr))
            }
        };
        let addr = self.obj_addr(obj, off);
        let v = self.builder.build_load(lty, addr, "fld").unwrap();
        Ok(Some(v))
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
    let ee = compiled
        .module
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| CodegenError(e.to_string()))?;

    // Map runtime externs to their Rust implementations.
    for (name, addr) in [
        ("ai_gc_alloc_fixed", runtime::ai_gc_alloc_fixed as *const () as usize),
        ("ai_gc_alloc_varlen", runtime::ai_gc_alloc_varlen as *const () as usize),
        ("ai_gc_pollcheck_slow", runtime::ai_gc_pollcheck_slow as *const () as usize),
    ] {
        if let Some(f) = compiled.module.get_function(name) {
            ee.add_global_mapping(&f, addr);
        }
    }

    let tis = layouts_to_type_infos(prog);
    // A generous space; stress mode collects every alloc regardless.
    let mut rt = RuntimeContext::new(8 << 20, tis);
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
}
