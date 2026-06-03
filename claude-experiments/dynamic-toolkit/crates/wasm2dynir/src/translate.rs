use dynir::builder::ModuleBuilder;
use dynir::*;
use wasmparser::{BinaryReaderError, FuncType, Operator, Parser, Payload, ValType};

#[derive(Debug)]
pub enum TranslateError {
    Wasm(BinaryReaderError),
    UnsupportedType(String),
    UnsupportedOp(String),
    NoFunction,
    NoCode,
}

impl From<BinaryReaderError> for TranslateError {
    fn from(e: BinaryReaderError) -> Self {
        TranslateError::Wasm(e)
    }
}

impl std::fmt::Display for TranslateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranslateError::Wasm(e) => write!(f, "wasm parse error: {}", e),
            TranslateError::UnsupportedType(t) => write!(f, "unsupported type: {}", t),
            TranslateError::UnsupportedOp(op) => write!(f, "unsupported op: {}", op),
            TranslateError::NoFunction => write!(f, "no function found in wasm module"),
            TranslateError::NoCode => write!(f, "no code section found"),
        }
    }
}

fn wasm_ty(vt: &ValType) -> Result<Type, TranslateError> {
    match vt {
        ValType::I32 => Ok(Type::I32),
        ValType::I64 => Ok(Type::I64),
        ValType::F64 => Ok(Type::F64),
        other => Err(TranslateError::UnsupportedType(format!("{:?}", other))),
    }
}

/// Translate the first exported function from a WASM binary into a dynir Function.
pub fn translate_wasm(
    wasm_bytes: &[u8],
) -> Result<(Function, Vec<(String, Signature)>), TranslateError> {
    let parser = Parser::new(0);
    let mut types: Vec<FuncType> = Vec::new();
    let mut func_type_indices: Vec<u32> = Vec::new();
    let mut import_count: u32 = 0;
    let mut import_sigs: Vec<(String, Signature)> = Vec::new();
    let mut target_func_idx: Option<u32> = None;
    let mut export_name: Option<String> = None;
    let wasm_owned: Vec<u8> = wasm_bytes.to_vec();

    for payload in parser.parse_all(&wasm_owned) {
        let payload = payload?;
        match payload {
            Payload::TypeSection(reader) => {
                for rec_group in reader {
                    let rec_group = rec_group?;
                    for sub_type in rec_group.into_types() {
                        if let wasmparser::CompositeInnerType::Func(ft) =
                            sub_type.composite_type.inner
                        {
                            types.push(ft);
                        }
                    }
                }
            }
            Payload::ImportSection(reader) => {
                for import in reader {
                    let import = import?;
                    if let wasmparser::TypeRef::Func(ty_idx) = import.ty {
                        let ft = &types[ty_idx as usize];
                        let params: Result<Vec<Type>, _> =
                            ft.params().iter().map(wasm_ty).collect();
                        let ret = if ft.results().is_empty() {
                            None
                        } else {
                            Some(wasm_ty(&ft.results()[0])?)
                        };
                        import_sigs.push((
                            format!("{}.{}", import.module, import.name),
                            Signature {
                                params: params?,
                                ret,
                            },
                        ));
                        func_type_indices.push(ty_idx);
                        import_count += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for ty_idx in reader {
                    func_type_indices.push(ty_idx?);
                }
            }
            Payload::ExportSection(reader) => {
                for exp in reader {
                    let exp = exp?;
                    if exp.kind == wasmparser::ExternalKind::Func && target_func_idx.is_none() {
                        target_func_idx = Some(exp.index);
                        export_name = Some(exp.name.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    let func_idx = target_func_idx.ok_or(TranslateError::NoFunction)?;
    let name = export_name.unwrap();
    translate_function_inline(
        &wasm_owned,
        &name,
        func_idx,
        import_count,
        &types,
        &func_type_indices,
        &import_sigs,
    )
}

/// Translate an entire WASM module into a dynir [`Module`].
///
/// All internal functions are translated and wired up so they can call each other.
/// Returns the Module and the FuncRef of the first exported function (entry point).
pub fn translate_wasm_module(wasm_bytes: &[u8]) -> Result<(Module, FuncRef), TranslateError> {
    let parser = Parser::new(0);
    let mut types: Vec<FuncType> = Vec::new();
    let mut func_type_indices: Vec<u32> = Vec::new();
    let mut import_count: u32 = 0;
    let mut import_sigs: Vec<(String, Signature)> = Vec::new();
    let mut export_func_idx: Option<u32> = None;
    let wasm_owned: Vec<u8> = wasm_bytes.to_vec();

    // First pass: collect types, imports, function indices, exports
    for payload in parser.parse_all(&wasm_owned) {
        let payload = payload?;
        match payload {
            Payload::TypeSection(reader) => {
                for rec_group in reader {
                    let rec_group = rec_group?;
                    for sub_type in rec_group.into_types() {
                        if let wasmparser::CompositeInnerType::Func(ft) =
                            sub_type.composite_type.inner
                        {
                            types.push(ft);
                        }
                    }
                }
            }
            Payload::ImportSection(reader) => {
                for import in reader {
                    let import = import?;
                    if let wasmparser::TypeRef::Func(ty_idx) = import.ty {
                        let ft = &types[ty_idx as usize];
                        let params: Result<Vec<Type>, _> =
                            ft.params().iter().map(wasm_ty).collect();
                        let ret = if ft.results().is_empty() {
                            None
                        } else {
                            Some(wasm_ty(&ft.results()[0])?)
                        };
                        import_sigs.push((
                            format!("{}.{}", import.module, import.name),
                            Signature {
                                params: params?,
                                ret,
                            },
                        ));
                        func_type_indices.push(ty_idx);
                        import_count += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for ty_idx in reader {
                    func_type_indices.push(ty_idx?);
                }
            }
            Payload::ExportSection(reader) => {
                for exp in reader {
                    let exp = exp?;
                    if exp.kind == wasmparser::ExternalKind::Func && export_func_idx.is_none() {
                        export_func_idx = Some(exp.index);
                    }
                }
            }
            _ => {}
        }
    }

    let entry_func_idx = export_func_idx.ok_or(TranslateError::NoFunction)?;
    let internal_count = func_type_indices.len() as u32 - import_count;

    // Build the module: declare imports, then declare all internal functions
    let mut mb = ModuleBuilder::new();

    // Declare imports
    let mut import_frefs: Vec<FuncRef> = Vec::new();
    for (name, sig) in &import_sigs {
        import_frefs.push(mb.declare_extern(name, sig.clone()));
    }

    // Declare all internal functions
    let mut internal_frefs: Vec<FuncRef> = Vec::new();
    for i in 0..internal_count {
        let abs_idx = (import_count + i) as usize;
        let ft = &types[func_type_indices[abs_idx] as usize];
        let params: Result<Vec<Type>, _> = ft.params().iter().map(wasm_ty).collect();
        let params = params?;
        let ret = if ft.results().is_empty() {
            None
        } else {
            Some(wasm_ty(&ft.results()[0])?)
        };
        let fref = mb.declare_func(&format!("func_{}", i), &params, ret);
        internal_frefs.push(fref);
    }

    // Build func_refs table: all imports + all internals
    let mut func_refs: Vec<Option<FuncRef>> = Vec::new();
    for fref in &import_frefs {
        func_refs.push(Some(*fref));
    }
    for fref in &internal_frefs {
        func_refs.push(Some(*fref));
    }

    // Second pass: translate each internal function's code
    let parser2 = Parser::new(0);
    let mut code_idx: u32 = 0;
    for payload in parser2.parse_all(&wasm_owned) {
        let payload = payload?;
        if let Payload::CodeSectionEntry(body) = payload {
            let abs_idx = (import_count + code_idx) as usize;
            let fref = internal_frefs[code_idx as usize];
            let ft = &types[func_type_indices[abs_idx] as usize];
            let ret = if ft.results().is_empty() {
                None
            } else {
                Some(wasm_ty(&ft.results()[0])?)
            };

            let mut fb = mb.define_func(fref);
            let entry = fb.entry_block();

            let mut local_types: Vec<Type> = Vec::new();
            let mut local_addrs: Vec<Value> = Vec::new();
            for (i, vt) in ft.params().iter().enumerate() {
                let ty = wasm_ty(vt)?;
                let slot = fb.create_stack_slot(8, false);
                let addr = fb.stack_addr(slot);
                let param_val = fb.block_param(entry, i);
                fb.store(param_val, addr, 0);
                local_addrs.push(addr);
                local_types.push(ty);
            }

            let locals_reader = body.get_locals_reader()?;
            for local in locals_reader {
                let (count, ty) = local?;
                let dynir_ty = wasm_ty(&ty)?;
                for _ in 0..count {
                    let slot = fb.create_stack_slot(8, false);
                    let addr = fb.stack_addr(slot);
                    let zero = match dynir_ty {
                        Type::I32 => fb.iconst(Type::I32, 0),
                        Type::I64 => fb.iconst(Type::I64, 0),
                        Type::F64 => fb.f64const(0.0),
                        _ => fb.iconst(Type::I64, 0),
                    };
                    fb.store(zero, addr, 0);
                    local_addrs.push(addr);
                    local_types.push(dynir_ty);
                }
            }

            let mut translator = FuncTranslator {
                builder: &mut fb,
                stack: Vec::new(),
                local_addrs,
                local_types: local_types.clone(),
                control_stack: Vec::new(),
                unreachable: false,
                func_refs: &func_refs,
                types: &types,
                func_type_indices: &func_type_indices,
                ret,
            };

            let result_types: Vec<Type> = ret.into_iter().collect();
            translator.control_stack.push(ControlFrame::Block {
                exit_block: None,
                stack_height: 0,
                under_stack_types: vec![],
                result_types,
                head_unreachable: false,
            });

            let ops_reader = body.get_operators_reader()?;
            for op in ops_reader {
                let op = op?;
                translator.translate_op(&op)?;
            }

            mb.finish_func(fref, fb);
            code_idx += 1;
        }
    }

    let entry_fref = if entry_func_idx >= import_count {
        internal_frefs[(entry_func_idx - import_count) as usize]
    } else {
        return Err(TranslateError::NoFunction);
    };

    let module = mb.build();
    Ok((module, entry_fref))
}

fn translate_function_inline(
    wasm_bytes: &[u8],
    name: &str,
    func_idx: u32,
    import_count: u32,
    types: &[FuncType],
    func_type_indices: &[u32],
    import_sigs: &[(String, Signature)],
) -> Result<(Function, Vec<(String, Signature)>), TranslateError> {
    let local_func_idx = func_idx - import_count;
    let ft = &types[func_type_indices[func_idx as usize] as usize];
    let params: Result<Vec<Type>, _> = ft.params().iter().map(wasm_ty).collect();
    let params = params?;
    let ret = if ft.results().is_empty() {
        None
    } else {
        Some(wasm_ty(&ft.results()[0])?)
    };

    let mut builder = FunctionBuilder::new(name, &params, ret);
    let entry = builder.entry_block();
    let mut local_types: Vec<Type> = Vec::new();
    let mut local_addrs: Vec<Value> = Vec::new();

    // Create stack slots for params and store initial values
    for (i, ty) in params.iter().enumerate() {
        let slot = builder.create_stack_slot(8, false);
        let addr = builder.stack_addr(slot);
        let param_val = builder.block_param(entry, i);
        builder.store(param_val, addr, 0);
        local_addrs.push(addr);
        local_types.push(*ty);
    }

    let mut func_refs: Vec<Option<FuncRef>> = Vec::new();
    for (fname, sig) in import_sigs {
        let fref = builder.declare_func(fname, sig.clone());
        func_refs.push(Some(fref));
    }
    for _ in import_count..func_type_indices.len() as u32 {
        func_refs.push(None);
    }

    let parser = Parser::new(0);
    let mut code_idx: u32 = 0;
    let mut found = false;

    for payload in parser.parse_all(wasm_bytes) {
        let payload = payload?;
        if let Payload::CodeSectionEntry(body) = payload {
            if code_idx == local_func_idx {
                let locals_reader = body.get_locals_reader()?;
                for local in locals_reader {
                    let (count, ty) = local?;
                    let dynir_ty = wasm_ty(&ty)?;
                    for _ in 0..count {
                        let slot = builder.create_stack_slot(8, false);
                        let addr = builder.stack_addr(slot);
                        let zero = match dynir_ty {
                            Type::I32 => builder.iconst(Type::I32, 0),
                            Type::I64 => builder.iconst(Type::I64, 0),
                            Type::F64 => builder.f64const(0.0),
                            _ => builder.iconst(Type::I64, 0),
                        };
                        builder.store(zero, addr, 0);
                        local_addrs.push(addr);
                        local_types.push(dynir_ty);
                    }
                }

                let mut translator = FuncTranslator {
                    builder: &mut builder,
                    stack: Vec::new(),
                    local_addrs,
                    local_types: local_types.clone(),
                    control_stack: Vec::new(),
                    unreachable: false,
                    func_refs: &func_refs,
                    types,
                    func_type_indices,
                    ret,
                };

                let result_types: Vec<Type> = ret.into_iter().collect();
                translator.control_stack.push(ControlFrame::Block {
                    exit_block: None,
                    stack_height: 0,
                    under_stack_types: vec![],
                    result_types,
                    head_unreachable: false,
                });

                let ops_reader = body.get_operators_reader()?;
                for op in ops_reader {
                    let op = op?;
                    translator.translate_op(&op)?;
                }
                found = true;
                break;
            }
            code_idx += 1;
        }
    }

    if !found {
        return Err(TranslateError::NoCode);
    }
    let func = builder.build();
    Ok((func, import_sigs.to_vec()))
}

enum ControlFrame {
    Block {
        exit_block: Option<BlockId>,
        stack_height: usize,
        under_stack_types: Vec<Type>,
        result_types: Vec<Type>,
        head_unreachable: bool,
    },
    Loop {
        header_block: BlockId,
        exit_block: Option<BlockId>,
        stack_height: usize,
        under_stack_types: Vec<Type>,
        param_types: Vec<Type>,
        result_types: Vec<Type>,
        head_unreachable: bool,
    },
    If {
        exit_block: Option<BlockId>,
        else_block: BlockId,
        stack_height: usize,
        under_stack_types: Vec<Type>,
        result_types: Vec<Type>,
        has_else: bool,
        head_unreachable: bool,
    },
}

impl ControlFrame {
    fn stack_height(&self) -> usize {
        match self {
            ControlFrame::Block { stack_height, .. }
            | ControlFrame::Loop { stack_height, .. }
            | ControlFrame::If { stack_height, .. } => *stack_height,
        }
    }

    fn under_stack_types(&self) -> &[Type] {
        match self {
            ControlFrame::Block {
                under_stack_types, ..
            }
            | ControlFrame::Loop {
                under_stack_types, ..
            }
            | ControlFrame::If {
                under_stack_types, ..
            } => under_stack_types,
        }
    }

    fn result_types(&self) -> &[Type] {
        match self {
            ControlFrame::Block { result_types, .. }
            | ControlFrame::Loop { result_types, .. }
            | ControlFrame::If { result_types, .. } => result_types,
        }
    }

    /// Types that a `br` to this label passes as values.
    fn br_types(&self) -> &[Type] {
        match self {
            ControlFrame::Block { result_types, .. } | ControlFrame::If { result_types, .. } => {
                result_types
            }
            ControlFrame::Loop { param_types, .. } => param_types,
        }
    }
}

/// Create or return the exit block for a control frame.
/// Exit block params = under_stack_types + result_types (no locals — they use stack slots).
fn get_or_create_exit(frame: &mut ControlFrame, builder: &mut FunctionBuilder) -> BlockId {
    let (exit_slot, under_stack_types, result_types) = match frame {
        ControlFrame::Block {
            exit_block,
            under_stack_types,
            result_types,
            ..
        }
        | ControlFrame::If {
            exit_block,
            under_stack_types,
            result_types,
            ..
        }
        | ControlFrame::Loop {
            exit_block,
            under_stack_types,
            result_types,
            ..
        } => (exit_block, &*under_stack_types, &*result_types),
    };
    if let Some(bb) = *exit_slot {
        bb
    } else {
        let mut param_tys: Vec<Type> = under_stack_types.to_vec();
        param_tys.extend_from_slice(result_types);
        let bb = builder.create_block(&param_tys);
        *exit_slot = Some(bb);
        bb
    }
}

struct FuncTranslator<'a, 'b> {
    builder: &'a mut FunctionBuilder,
    stack: Vec<Value>,
    /// Stack slot addresses for each wasm local (params + declared locals).
    local_addrs: Vec<Value>,
    local_types: Vec<Type>,
    control_stack: Vec<ControlFrame>,
    unreachable: bool,
    func_refs: &'b [Option<FuncRef>],
    types: &'b [FuncType],
    func_type_indices: &'b [u32],
    ret: Option<Type>,
}

impl<'a, 'b> FuncTranslator<'a, 'b> {
    fn pop(&mut self) -> Value {
        self.stack.pop().expect("stack underflow")
    }

    fn pop2(&mut self) -> (Value, Value) {
        let b = self.pop();
        let a = self.pop();
        (a, b)
    }

    /// Compute stack types for the current stack.
    fn stack_types(&self) -> Vec<Type> {
        self.stack
            .iter()
            .map(|v| self.builder.value_type(*v))
            .collect()
    }

    /// Build the full args for jumping to an exit block:
    /// under_stack_values + result_values (no locals — they use stack slots).
    fn exit_jump_args(&self, stack_height: usize, arity: usize) -> Vec<Value> {
        let mut args: Vec<Value> = self.stack[..stack_height].to_vec();
        args.extend_from_slice(&self.stack[self.stack.len() - arity..]);
        args
    }

    /// After switching to an exit block, restore stack from block params.
    fn restore_from_exit(&mut self, exit: BlockId, under_count: usize, result_count: usize) {
        self.stack.clear();
        for i in 0..under_count {
            self.stack.push(self.builder.block_param(exit, i));
        }
        for i in 0..result_count {
            self.stack
                .push(self.builder.block_param(exit, under_count + i));
        }
    }

    fn br_target_and_types(&mut self, depth: u32) -> (BlockId, Vec<Type>, usize) {
        let idx = self.control_stack.len() - 1 - depth as usize;
        match &mut self.control_stack[idx] {
            ControlFrame::Block { .. } | ControlFrame::If { .. } => {
                let frame = &mut self.control_stack[idx];
                let br_types = frame.br_types().to_vec();
                let sh = frame.stack_height();
                let bb = get_or_create_exit(frame, self.builder);
                (bb, br_types, sh)
            }
            ControlFrame::Loop {
                header_block,
                param_types,
                stack_height,
                ..
            } => {
                let bb = *header_block;
                let pt = param_types.clone();
                let sh = *stack_height;
                (bb, pt, sh)
            }
        }
    }

    fn emit_br(&mut self, depth: u32) {
        let idx = self.control_stack.len() - 1 - depth as usize;
        let is_loop = matches!(&self.control_stack[idx], ControlFrame::Loop { .. });
        let (target, br_types, sh) = self.br_target_and_types(depth);
        let arity = br_types.len();
        if is_loop {
            // Loop header expects under_stack only.
            let args: Vec<Value> = self.stack[..sh].to_vec();
            self.builder.jump(target, &args);
        } else {
            // Block/If: pass under_stack + results.
            let args = self.exit_jump_args(sh, arity);
            self.builder.jump(target, &args);
        }
    }

    fn i32_to_cond(&mut self, val: Value) -> Value {
        let zero = self.builder.iconst(Type::I32, 0);
        self.builder.icmp(CmpOp::Ne, val, zero)
    }

    fn translate_op(&mut self, op: &Operator) -> Result<(), TranslateError> {
        if self.unreachable {
            match op {
                Operator::Block { .. } | Operator::Loop { .. } | Operator::If { .. } => {
                    self.control_stack.push(ControlFrame::Block {
                        exit_block: None,
                        stack_height: 0,
                        under_stack_types: vec![],
                        result_types: vec![],
                        head_unreachable: true,
                    });
                }
                Operator::Else => {
                    if let Some(ControlFrame::If {
                        head_unreachable, ..
                    }) = self.control_stack.last()
                    {
                        if !head_unreachable {
                            if let Some(ControlFrame::If {
                                else_block,
                                stack_height,
                                has_else,
                                under_stack_types,
                                ..
                            }) = self.control_stack.last_mut()
                            {
                                *has_else = true;
                                let _sh = *stack_height;
                                let eb = *else_block;
                                let ust = under_stack_types.clone();
                                self.builder.switch_to_block(eb);
                                // Restore stack to under_stack from else_block params.
                                self.stack.clear();
                                for i in 0..ust.len() {
                                    self.stack.push(self.builder.block_param(eb, i));
                                }
                                self.unreachable = false;
                            }
                        }
                    }
                }
                Operator::End => {
                    let frame = self.control_stack.pop().unwrap();
                    match &frame {
                        ControlFrame::Block {
                            head_unreachable: true,
                            ..
                        }
                        | ControlFrame::Loop {
                            head_unreachable: true,
                            ..
                        }
                        | ControlFrame::If {
                            head_unreachable: true,
                            ..
                        } => {}
                        _ => {
                            self.handle_end(frame)?;
                        }
                    }
                }
                _ => {}
            }
            return Ok(());
        }

        match op {
            Operator::I32Const { value } => {
                let v = self.builder.iconst(Type::I32, *value as i64);
                self.stack.push(v);
            }
            Operator::I64Const { value } => {
                let v = self.builder.iconst(Type::I64, *value);
                self.stack.push(v);
            }
            Operator::F64Const { value } => {
                let v = self.builder.f64const(f64::from_bits(value.bits()));
                self.stack.push(v);
            }

            Operator::LocalGet { local_index } => {
                let addr = self.local_addrs[*local_index as usize];
                let ty = self.local_types[*local_index as usize];
                let val = self.builder.load(ty, addr, 0);
                self.stack.push(val);
            }
            Operator::LocalSet { local_index } => {
                let val = self.pop();
                let addr = self.local_addrs[*local_index as usize];
                self.builder.store(val, addr, 0);
            }
            Operator::LocalTee { local_index } => {
                let val = *self.stack.last().expect("stack underflow");
                let addr = self.local_addrs[*local_index as usize];
                self.builder.store(val, addr, 0);
            }

            Operator::Drop => {
                self.pop();
            }
            Operator::Select => {
                let cond = self.pop();
                let (a, b) = self.pop2();
                let cond_i8 = self.i32_to_cond(cond);
                let result = self.builder.select(cond_i8, a, b);
                self.stack.push(result);
            }

            Operator::I32Add => self.binop(|b, a, bv| b.add(a, bv)),
            Operator::I32Sub => self.binop(|b, a, bv| b.sub(a, bv)),
            Operator::I32Mul => self.binop(|b, a, bv| b.mul(a, bv)),
            Operator::I32DivS => self.binop(|b, a, bv| b.sdiv(a, bv)),
            Operator::I32DivU => self.binop(|b, a, bv| b.udiv(a, bv)),
            Operator::I32And => self.binop(|b, a, bv| b.and(a, bv)),
            Operator::I32Or => self.binop(|b, a, bv| b.or(a, bv)),
            Operator::I32Xor => self.binop(|b, a, bv| b.xor(a, bv)),
            Operator::I32Shl => self.binop(|b, a, bv| b.shl(a, bv)),
            Operator::I32ShrS => self.binop(|b, a, bv| b.ashr(a, bv)),
            Operator::I32ShrU => self.binop(|b, a, bv| b.lshr(a, bv)),
            Operator::I32RemS => {
                let (a, b) = self.pop2();
                let q = self.builder.sdiv(a, b);
                let p = self.builder.mul(q, b);
                let r = self.builder.sub(a, p);
                self.stack.push(r);
            }
            Operator::I32RemU => {
                let (a, b) = self.pop2();
                let q = self.builder.udiv(a, b);
                let p = self.builder.mul(q, b);
                let r = self.builder.sub(a, p);
                self.stack.push(r);
            }

            Operator::I32Eqz => {
                let v = self.pop();
                let zero = self.builder.iconst(Type::I32, 0);
                let cmp = self.builder.icmp(CmpOp::Eq, v, zero);
                let result = self.builder.zext(cmp, Type::I32);
                self.stack.push(result);
            }
            Operator::I32Eq => self.icmp_push(CmpOp::Eq),
            Operator::I32Ne => self.icmp_push(CmpOp::Ne),
            Operator::I32LtS => self.icmp_push(CmpOp::Slt),
            Operator::I32LtU => self.icmp_push(CmpOp::Ult),
            Operator::I32GtS => self.icmp_push(CmpOp::Sgt),
            Operator::I32GtU => self.icmp_push(CmpOp::Ugt),
            Operator::I32LeS => self.icmp_push(CmpOp::Sle),
            Operator::I32LeU => self.icmp_push(CmpOp::Ule),
            Operator::I32GeS => self.icmp_push(CmpOp::Sge),
            Operator::I32GeU => self.icmp_push(CmpOp::Uge),

            Operator::I64Add => self.binop(|b, a, bv| b.add(a, bv)),
            Operator::I64Sub => self.binop(|b, a, bv| b.sub(a, bv)),
            Operator::I64Mul => self.binop(|b, a, bv| b.mul(a, bv)),
            Operator::I64DivS => self.binop(|b, a, bv| b.sdiv(a, bv)),
            Operator::I64DivU => self.binop(|b, a, bv| b.udiv(a, bv)),
            Operator::I64And => self.binop(|b, a, bv| b.and(a, bv)),
            Operator::I64Or => self.binop(|b, a, bv| b.or(a, bv)),
            Operator::I64Xor => self.binop(|b, a, bv| b.xor(a, bv)),
            Operator::I64Shl => self.binop(|b, a, bv| b.shl(a, bv)),
            Operator::I64ShrS => self.binop(|b, a, bv| b.ashr(a, bv)),
            Operator::I64ShrU => self.binop(|b, a, bv| b.lshr(a, bv)),

            Operator::I64Eqz => {
                let v = self.pop();
                let zero = self.builder.iconst(Type::I64, 0);
                let cmp = self.builder.icmp(CmpOp::Eq, v, zero);
                let result = self.builder.zext(cmp, Type::I32);
                self.stack.push(result);
            }
            Operator::I64Eq => self.icmp_push(CmpOp::Eq),
            Operator::I64Ne => self.icmp_push(CmpOp::Ne),
            Operator::I64LtS => self.icmp_push(CmpOp::Slt),
            Operator::I64LtU => self.icmp_push(CmpOp::Ult),
            Operator::I64GtS => self.icmp_push(CmpOp::Sgt),
            Operator::I64GtU => self.icmp_push(CmpOp::Ugt),
            Operator::I64LeS => self.icmp_push(CmpOp::Sle),
            Operator::I64LeU => self.icmp_push(CmpOp::Ule),
            Operator::I64GeS => self.icmp_push(CmpOp::Sge),
            Operator::I64GeU => self.icmp_push(CmpOp::Uge),

            Operator::F64Add => self.binop(|b, a, bv| b.fadd(a, bv)),
            Operator::F64Sub => self.binop(|b, a, bv| b.fsub(a, bv)),
            Operator::F64Mul => self.binop(|b, a, bv| b.fmul(a, bv)),
            Operator::F64Div => self.binop(|b, a, bv| b.fdiv(a, bv)),
            Operator::F64Neg => {
                let v = self.pop();
                let r = self.builder.fneg(v);
                self.stack.push(r);
            }

            Operator::F64Eq => self.fcmp_push(CmpOp::Eq),
            Operator::F64Ne => self.fcmp_push(CmpOp::Ne),
            Operator::F64Lt => self.fcmp_push(CmpOp::Slt),
            Operator::F64Gt => self.fcmp_push(CmpOp::Sgt),
            Operator::F64Le => self.fcmp_push(CmpOp::Sle),
            Operator::F64Ge => self.fcmp_push(CmpOp::Sge),

            Operator::I32WrapI64 => {
                let v = self.pop();
                let r = self.builder.trunc(v, Type::I32);
                self.stack.push(r);
            }
            Operator::I64ExtendI32S => {
                let v = self.pop();
                let r = self.builder.sext(v, Type::I64);
                self.stack.push(r);
            }
            Operator::I64ExtendI32U => {
                let v = self.pop();
                let r = self.builder.zext(v, Type::I64);
                self.stack.push(r);
            }
            Operator::F64ConvertI32S | Operator::F64ConvertI64S => {
                let v = self.pop();
                let r = self.builder.int_to_float(v);
                self.stack.push(r);
            }
            Operator::I32TruncF64S | Operator::I64TruncF64S => {
                let v = self.pop();
                let r = self.builder.float_to_int(v);
                self.stack.push(r);
            }

            Operator::Unreachable => {
                self.builder.unreachable();
                self.unreachable = true;
            }
            Operator::Nop => {}
            Operator::Return => {
                match self.ret {
                    Some(_) => {
                        let val = self.pop();
                        self.builder.ret(val);
                    }
                    None => {
                        self.builder.ret_void();
                    }
                }
                self.unreachable = true;
            }

            Operator::Block { blockty } => {
                let result_types = self.blocktype_results(blockty)?;
                let under_stack_types = self.stack_types();
                self.control_stack.push(ControlFrame::Block {
                    exit_block: None,
                    stack_height: self.stack.len(),
                    under_stack_types,
                    result_types,
                    head_unreachable: false,
                });
            }

            Operator::Loop { blockty } => {
                let result_types = self.blocktype_results(blockty)?;
                let under_stack_types = self.stack_types();
                // Loop header params = under_stack only (locals use stack slots).
                let header_params: Vec<Type> = under_stack_types.clone();
                let header = self.builder.create_block(&header_params);
                let jump_args: Vec<Value> = self.stack.clone();
                self.builder.jump(header, &jump_args);
                self.builder.switch_to_block(header);
                // Restore stack from header params.
                let under_count = self.stack.len();
                for i in 0..under_count {
                    self.stack[i] = self.builder.block_param(header, i);
                }
                self.control_stack.push(ControlFrame::Loop {
                    header_block: header,
                    exit_block: None,
                    stack_height: self.stack.len(),
                    under_stack_types,
                    param_types: vec![],
                    result_types,
                    head_unreachable: false,
                });
            }

            Operator::If { blockty } => {
                let result_types = self.blocktype_results(blockty)?;
                let cond = self.pop();
                let under_stack_types = self.stack_types();
                let cond_i8 = self.i32_to_cond(cond);
                // Then and else blocks get under_stack as params (locals use stack slots).
                let branch_params: Vec<Type> = under_stack_types.clone();
                let then_block = self.builder.create_block(&branch_params);
                let else_block = self.builder.create_block(&branch_params);
                let branch_args: Vec<Value> = self.stack.clone();
                self.builder
                    .br_if(cond_i8, then_block, &branch_args, else_block, &branch_args);
                self.builder.switch_to_block(then_block);
                let under_count = self.stack.len();
                for i in 0..under_count {
                    self.stack[i] = self.builder.block_param(then_block, i);
                }
                self.control_stack.push(ControlFrame::If {
                    exit_block: None,
                    else_block,
                    stack_height: self.stack.len(),
                    under_stack_types,
                    result_types,
                    has_else: false,
                    head_unreachable: false,
                });
            }

            Operator::Else => {
                let frame = self.control_stack.last_mut().unwrap();
                let (else_block, stack_height, under_stack_types) = match frame {
                    ControlFrame::If {
                        else_block,
                        stack_height,
                        has_else,
                        under_stack_types,
                        ..
                    } => {
                        *has_else = true;
                        (*else_block, *stack_height, under_stack_types.clone())
                    }
                    _ => panic!("else without if"),
                };

                if !self.unreachable {
                    let result_types = frame.result_types().to_vec();
                    let arity = result_types.len();
                    let exit = get_or_create_exit(frame, self.builder);
                    let args = self.exit_jump_args(stack_height, arity);
                    self.builder.jump(exit, &args);
                }

                self.builder.switch_to_block(else_block);
                let under_count = under_stack_types.len();
                self.stack.clear();
                for i in 0..under_count {
                    self.stack.push(self.builder.block_param(else_block, i));
                }
                self.unreachable = false;
            }

            Operator::End => {
                let frame = self.control_stack.pop().unwrap();
                self.handle_end(frame)?;
            }

            Operator::Br { relative_depth } => {
                self.emit_br(*relative_depth);
                self.unreachable = true;
            }

            Operator::BrIf { relative_depth } => {
                let cond = self.pop();
                let cond_i8 = self.i32_to_cond(cond);
                let idx = self.control_stack.len() - 1 - *relative_depth as usize;
                let is_loop = matches!(&self.control_stack[idx], ControlFrame::Loop { .. });
                let (target, br_types, sh) = self.br_target_and_types(*relative_depth);
                let arity = br_types.len();

                let target_args = if is_loop {
                    self.stack[..sh].to_vec()
                } else {
                    self.exit_jump_args(sh, arity)
                };

                // Fallthrough: full stack only (no locals).
                let fall_types: Vec<Type> = self.stack_types();
                let fallthrough = self.builder.create_block(&fall_types);
                let fall_args: Vec<Value> = self.stack.clone();

                self.builder
                    .br_if(cond_i8, target, &target_args, fallthrough, &fall_args);
                self.builder.switch_to_block(fallthrough);
                let stack_len = self.stack.len();
                for i in 0..stack_len {
                    self.stack[i] = self.builder.block_param(fallthrough, i);
                }
            }

            Operator::BrTable { targets } => {
                let index = self.pop();
                let index_i64 = self.builder.zext(index, Type::I64);

                let default_depth = targets.default();
                let didx = self.control_stack.len() - 1 - default_depth as usize;
                let default_is_loop =
                    matches!(&self.control_stack[didx], ControlFrame::Loop { .. });
                let (default_target, default_br_types, default_sh) =
                    self.br_target_and_types(default_depth);
                let default_arity = default_br_types.len();
                let default_args = if default_is_loop {
                    self.stack[..default_sh].to_vec()
                } else {
                    self.exit_jump_args(default_sh, default_arity)
                };

                let mut cases: Vec<(i64, BlockId, Vec<Value>)> = Vec::new();
                for (i, depth) in targets.targets().enumerate() {
                    let depth = depth?;
                    let cidx = self.control_stack.len() - 1 - depth as usize;
                    let c_is_loop = matches!(&self.control_stack[cidx], ControlFrame::Loop { .. });
                    let (target, br_types, sh) = self.br_target_and_types(depth);
                    let arity = br_types.len();
                    let args = if c_is_loop {
                        self.stack[..sh].to_vec()
                    } else {
                        self.exit_jump_args(sh, arity)
                    };
                    cases.push((i as i64, target, args));
                }

                let case_refs: Vec<(i64, BlockId, &[Value])> = cases
                    .iter()
                    .map(|(v, b, a)| (*v, *b, a.as_slice()))
                    .collect();
                self.builder
                    .switch(index_i64, &case_refs, default_target, &default_args);
                self.unreachable = true;
            }

            Operator::Call { function_index } => {
                let ft = &self.types[self.func_type_indices[*function_index as usize] as usize];
                let param_count = ft.params().len();
                let args: Vec<Value> = self.stack.split_off(self.stack.len() - param_count);
                let fref = self.func_refs[*function_index as usize].ok_or_else(|| {
                    TranslateError::UnsupportedOp(format!(
                        "call to local function {}",
                        function_index
                    ))
                })?;
                let result = self.builder.call(fref, &args);
                if let Some(val) = result {
                    self.stack.push(val);
                }
            }

            other => {
                return Err(TranslateError::UnsupportedOp(format!("{:?}", other)));
            }
        }
        Ok(())
    }

    fn handle_end(&mut self, frame: ControlFrame) -> Result<(), TranslateError> {
        let stack_height = frame.stack_height();
        let under_count = frame.under_stack_types().len();
        let num_results = frame.result_types().len();

        match frame {
            ControlFrame::Block {
                exit_block: None, ..
            } => {
                if self.control_stack.is_empty() {
                    if !self.unreachable {
                        match self.ret {
                            Some(_) => {
                                let val = self.pop();
                                self.builder.ret(val);
                            }
                            None => {
                                self.builder.ret_void();
                            }
                        }
                    }
                } else if self.unreachable {
                    self.stack.truncate(stack_height);
                }
            }
            ControlFrame::Block {
                exit_block: Some(exit),
                result_types,
                ..
            } => {
                if !self.unreachable {
                    let args = self.exit_jump_args(stack_height, result_types.len());
                    self.builder.jump(exit, &args);
                }
                self.builder.switch_to_block(exit);
                self.restore_from_exit(exit, under_count, num_results);
                self.unreachable = false;
            }
            ControlFrame::Loop {
                exit_block,
                result_types,
                ..
            } => {
                if let Some(exit) = exit_block {
                    if !self.unreachable {
                        let args = self.exit_jump_args(stack_height, result_types.len());
                        self.builder.jump(exit, &args);
                    }
                    self.builder.switch_to_block(exit);
                    self.restore_from_exit(exit, under_count, num_results);
                    self.unreachable = false;
                } else if self.unreachable {
                    self.stack.truncate(stack_height);
                }
            }
            ControlFrame::If {
                exit_block,
                else_block,
                result_types,
                has_else,
                under_stack_types,
                ..
            } => {
                if !has_else {
                    let arity = result_types.len();

                    // Create exit if needed, jump from then-branch.
                    let exit = if let Some(e) = exit_block {
                        if !self.unreachable {
                            let args = self.exit_jump_args(stack_height, arity);
                            self.builder.jump(e, &args);
                        }
                        e
                    } else {
                        let mut param_tys: Vec<Type> = under_stack_types.clone();
                        param_tys.extend_from_slice(&result_types);
                        let e = self.builder.create_block(&param_tys);
                        if !self.unreachable {
                            let args = self.exit_jump_args(stack_height, arity);
                            self.builder.jump(e, &args);
                        }
                        e
                    };

                    // Else block: restore under-stack, jump to exit.
                    self.builder.switch_to_block(else_block);
                    let uc = under_stack_types.len();
                    let mut else_args: Vec<Value> = Vec::new();
                    for i in 0..uc {
                        else_args.push(self.builder.block_param(else_block, i));
                    }
                    // void if: no results to pass, just under_stack.
                    self.builder.jump(exit, &else_args);

                    self.builder.switch_to_block(exit);
                    self.restore_from_exit(exit, under_count, num_results);
                    self.unreachable = false;
                } else {
                    // Has else.
                    if let Some(exit) = exit_block {
                        if !self.unreachable {
                            let arity = result_types.len();
                            let args = self.exit_jump_args(stack_height, arity);
                            self.builder.jump(exit, &args);
                        }
                        self.builder.switch_to_block(exit);
                        self.restore_from_exit(exit, under_count, num_results);
                        self.unreachable = false;
                    } else {
                        self.stack.truncate(stack_height);
                    }
                }
            }
        }
        Ok(())
    }

    fn blocktype_results(
        &self,
        blockty: &wasmparser::BlockType,
    ) -> Result<Vec<Type>, TranslateError> {
        match blockty {
            wasmparser::BlockType::Empty => Ok(vec![]),
            wasmparser::BlockType::Type(vt) => Ok(vec![wasm_ty(vt)?]),
            wasmparser::BlockType::FuncType(idx) => {
                let ft = &self.types[*idx as usize];
                ft.results().iter().map(wasm_ty).collect()
            }
        }
    }

    fn binop(&mut self, f: impl FnOnce(&mut FunctionBuilder, Value, Value) -> Value) {
        let (a, b) = self.pop2();
        let result = f(self.builder, a, b);
        self.stack.push(result);
    }

    fn icmp_push(&mut self, op: CmpOp) {
        let (a, b) = self.pop2();
        let cmp = self.builder.icmp(op, a, b);
        let result = self.builder.zext(cmp, Type::I32);
        self.stack.push(result);
    }

    fn fcmp_push(&mut self, op: CmpOp) {
        let (a, b) = self.pop2();
        let cmp = self.builder.fcmp(op, a, b);
        let result = self.builder.zext(cmp, Type::I32);
        self.stack.push(result);
    }
}
