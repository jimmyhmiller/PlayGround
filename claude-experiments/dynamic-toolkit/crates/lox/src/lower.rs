//! Lower Lox AST → dynir Module using dynlang.
//!
//! All functions take an implicit closure parameter (first arg).
//! Captured variables go through upvalue cells (heap-allocated).
//! Indirect calls use call_indirect (interpreter: func_table_index,
//! JIT: raw code pointer).

use std::collections::HashMap;

use dynlang::*;

use crate::ast::*;
use crate::resolve::{self, FunCaptures, ResolveResult};
use crate::value::{TAG_BOOL, TAG_NIL, TAG_OBJ};

// ── GC object type layout info ───────────────────────────────────

/// Pre-computed field offsets and type info for all GC object types.
/// Passed from lower.rs to vm.rs so both sides use consistent layouts.
pub struct LoxGcTypes {
    /// TypeInfo pointers indexed by ObjTypeId.
    pub type_infos: Vec<&'static TypeInfo>,

    // ObjTypeId indices
    pub string_id: usize,
    pub closure_id: usize,
    pub upvalue_id: usize,
    pub class_id: usize,
    pub instance_id: usize,
    pub bound_method_id: usize,
    pub native_fn_id: usize,
    pub table_id: usize,

    // Table offsets (used for class methods and instance fields)
    pub table_count_off: i32,
    pub table_data_base_off: i32,

    // Upvalue offsets
    pub upvalue_value_off: i32,

    // Closure offsets
    pub closure_func_idx_off: i32,
    pub closure_arity_off: i32,
    pub closure_name_off: i32,
    pub closure_upval_base_off: i32,

    // Class offsets
    pub class_name_off: i32,
    pub class_super_off: i32,
    pub class_methods_off: i32,  // Value: pointer to Table GC object

    // Instance offsets
    pub instance_class_off: i32,
    pub instance_fields_off: i32,  // Value: pointer to Table GC object

    // BoundMethod offsets
    pub bound_receiver_off: i32,
    pub bound_method_off: i32,

    // String offsets
    pub string_len_off: i32,
    pub string_data_base_off: i32,

    // NativeFn offsets
    pub native_name_off: i32,
    pub native_func_ptr_off: i32,
}

// ── Public result ─────────────────────────────────────────────────

pub struct LoweredProgram {
    pub module: Module,
    pub entry: FuncRef,
    pub strings: Vec<String>,
    pub gc_types: LoxGcTypes,
}

// ── String pool ───────────────────────────────────────────────────

struct StringPool {
    strings: Vec<String>,
    map: HashMap<String, u32>,
}

impl StringPool {
    fn new() -> Self {
        StringPool { strings: Vec::new(), map: HashMap::new() }
    }
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) { return id; }
        let id = self.strings.len() as u32;
        self.map.insert(s.to_string(), id);
        self.strings.push(s.to_string());
        id
    }
}

// ── Extern references ─────────────────────────────────────────────

struct Externs {
    print: FuncRef,
    define_global: FuncRef,
    get_global: FuncRef,
    set_global: FuncRef,
    clock: FuncRef,
    // Upvalue / closure ops
    alloc_upvalue: FuncRef,
    get_upvalue: FuncRef,
    set_upvalue: FuncRef,
    make_closure: FuncRef,
    closure_upvalue: FuncRef,
    set_closure_upvalue: FuncRef,
    closure_func_ptr: FuncRef,
    // Object type
    obj_type: FuncRef,
    // Arity checking / call errors
    check_arity: FuncRef,
    call_non_callable: FuncRef,
    get_closure_arity: FuncRef,
    get_class_init_arity: FuncRef,
    get_bound_arity: FuncRef,
    // Class ops
    make_class: FuncRef,
    class_inherit: FuncRef,
    class_add_method: FuncRef,
    construct_instance: FuncRef,
    class_init_ptr: FuncRef,
    class_init_closure: FuncRef,
    // Property ops
    get_property: FuncRef,
    set_property: FuncRef,
    // Super
    get_super: FuncRef,
    // Bound method
    bound_receiver: FuncRef,
    bound_method_closure: FuncRef,
    bound_closure_func_ptr: FuncRef,
    // Native function
    make_native_fn: FuncRef,
    // String resolution
    resolve_string: FuncRef,
}

fn sig(params: &[Type], ret: Option<Type>) -> Signature {
    Signature { params: params.to_vec(), ret }
}

fn declare_externs(dm: &mut DynModule) -> Externs {
    let i = |n: usize| sig(&vec![Type::I64; n], Some(Type::I64));
    let v = |n: usize| sig(&vec![Type::I64; n], None);

    Externs {
        print: dm.declare_extern("lox_print", v(1)),
        define_global: dm.declare_extern("lox_define_global", v(2)),
        get_global: dm.declare_extern("lox_get_global", i(1)),
        set_global: dm.declare_extern("lox_set_global", i(2)),
        clock: dm.declare_extern("lox_clock", i(0)),
        alloc_upvalue: dm.declare_extern("lox_alloc_upvalue", i(1)),
        get_upvalue: dm.declare_extern("lox_get_upvalue", i(1)),
        set_upvalue: dm.declare_extern("lox_set_upvalue", v(2)),
        make_closure: dm.declare_extern("lox_make_closure", i(4)),
        closure_upvalue: dm.declare_extern("lox_closure_upvalue", i(2)),
        set_closure_upvalue: dm.declare_extern("lox_set_closure_upvalue", v(3)),
        closure_func_ptr: dm.declare_extern("lox_closure_func_ptr", i(1)),
        obj_type: dm.declare_extern("lox_obj_type", i(1)),
        check_arity: dm.declare_extern("lox_check_arity", v(3)),
        call_non_callable: dm.declare_extern("lox_call_non_callable", v(0)),
        get_closure_arity: dm.declare_extern("lox_get_closure_arity", i(1)),
        get_class_init_arity: dm.declare_extern("lox_get_class_init_arity", i(1)),
        get_bound_arity: dm.declare_extern("lox_get_bound_arity", i(1)),
        make_class: dm.declare_extern("lox_make_class", i(1)),
        class_inherit: dm.declare_extern("lox_class_inherit", v(2)),
        class_add_method: dm.declare_extern("lox_class_add_method", v(3)),
        construct_instance: dm.declare_extern("lox_construct_instance", i(1)),
        class_init_ptr: dm.declare_extern("lox_class_init_ptr", i(1)),
        class_init_closure: dm.declare_extern("lox_class_init_closure", i(1)),
        get_property: dm.declare_extern("lox_get_property", i(2)),
        set_property: dm.declare_extern("lox_set_property", i(3)),
        get_super: dm.declare_extern("lox_get_super", i(3)),
        bound_receiver: dm.declare_extern("lox_bound_receiver", i(1)),
        bound_method_closure: dm.declare_extern("lox_bound_method_closure", i(1)),
        bound_closure_func_ptr: dm.declare_extern("lox_bound_closure_func_ptr", i(1)),
        make_native_fn: dm.declare_extern("lox_make_native_fn", i(1)),
        resolve_string: dm.declare_extern("lox_resolve_string", i(1)),
    }
}

// ── Lowering context ──────────────────────────────────────────────

struct Ctx<'a> {
    externs: &'a Externs,
    func_names: &'a HashMap<String, FuncRef>,
    func_arities: &'a HashMap<String, usize>,
    resolve: &'a ResolveResult,
    strings: &'a mut StringPool,
    current_func_key: String,
    is_script: bool,
    is_init: bool,
    dead: bool,
    /// Maps upvalue name → local var holding the upvalue cell obj_val.
    upvalue_cells: HashMap<String, String>,
    /// Names of locals in the current function that are captured
    /// (and thus stored in upvalue cells instead of plain vars).
    captured_locals: &'a std::collections::HashSet<String>,
    /// Current class name (for super resolution).
    current_class: Option<String>,
}

// ── Collect all function declarations ─────────────────────────────

fn collect_funs(stmts: &[Stmt]) -> Vec<&FunDecl> {
    let mut out = Vec::new();
    for stmt in stmts { collect_funs_stmt(stmt, &mut out); }
    out
}

fn collect_funs_stmt<'a>(stmt: &'a Stmt, out: &mut Vec<&'a FunDecl>) {
    match stmt {
        Stmt::Fun(decl) => {
            out.push(decl);
            for s in &decl.body { collect_funs_stmt(s, out); }
        }
        Stmt::Block(stmts) => { for s in stmts { collect_funs_stmt(s, out); } }
        Stmt::If(_, t, e) => {
            collect_funs_stmt(t, out);
            if let Some(e) = e { collect_funs_stmt(e, out); }
        }
        Stmt::While(_, b) => collect_funs_stmt(b, out),
        Stmt::Class(decl) => {
            // Collect nested functions inside method bodies
            for method in &decl.methods {
                for s in &method.body { collect_funs_stmt(s, out); }
            }
        }
        _ => {}
    }
}

fn collect_method_funs<'a>(decl: &'a ClassDecl) -> Vec<(&'a FunDecl, String)> {
    decl.methods.iter()
        .map(|m| (m, format!("{}.{}", decl.name, m.name)))
        .collect()
}

// ── Main lowering entry point ─────────────────────────────────────

fn extract_offset(ty: &ObjType, field: &str) -> i32 {
    ty.field_offsets.get(field)
        .unwrap_or_else(|| panic!("unknown field '{}' on type '{}'", field, ty.name))
        .0
}

fn declare_gc_types(dm: &mut DynModule) -> LoxGcTypes {
    let string_ty = dm.obj_type("String")
        .field("len", FieldKind::Raw64)
        .varlen_bytes()
        .build();

    let closure_ty = dm.obj_type("Closure")
        .field("name_ptr", FieldKind::Value)
        .field("func_table_idx", FieldKind::Raw64)
        .field("arity", FieldKind::Raw64)
        .varlen_values()
        .build();

    let upvalue_ty = dm.obj_type("Upvalue")
        .field("value", FieldKind::Value)
        .build();

    let class_ty = dm.obj_type("Class")
        .field("name_ptr", FieldKind::Value)
        .field("superclass", FieldKind::Value)
        .field("methods", FieldKind::Value)  // GC pointer to Table object
        .build();

    let instance_ty = dm.obj_type("Instance")
        .field("class_ptr", FieldKind::Value)
        .field("fields", FieldKind::Value)  // GC pointer to Table object
        .build();

    // Table: used for class methods and instance fields
    // Stores [key, value, key, value, ...] as varlen values
    let table_ty = dm.obj_type("Table")
        .field("count", FieldKind::Raw64)  // number of key-value pairs
        .varlen_values()
        .build();

    let bound_method_ty = dm.obj_type("BoundMethod")
        .field("receiver", FieldKind::Value)
        .field("method", FieldKind::Value)
        .build();

    let native_fn_ty = dm.obj_type("NativeFn")
        .field("name_ptr", FieldKind::Value)
        .field("func_ptr", FieldKind::Raw64)
        .build();

    // Extract all offsets
    let s = dm.get_obj_type(string_ty);
    let c = dm.get_obj_type(closure_ty);
    let u = dm.get_obj_type(upvalue_ty);
    let cl = dm.get_obj_type(class_ty);
    let inst = dm.get_obj_type(instance_ty);
    let bm = dm.get_obj_type(bound_method_ty);
    let nf = dm.get_obj_type(native_fn_ty);
    let tb = dm.get_obj_type(table_ty);

    let type_infos: Vec<&'static TypeInfo> = dm.obj_types.iter()
        .map(|t| t.type_info)
        .collect();

    LoxGcTypes {
        type_infos,
        string_id: string_ty.0,
        closure_id: closure_ty.0,
        upvalue_id: upvalue_ty.0,
        class_id: class_ty.0,
        instance_id: instance_ty.0,
        bound_method_id: bound_method_ty.0,
        native_fn_id: native_fn_ty.0,
        table_id: table_ty.0,

        table_count_off: extract_offset(tb, "count"),
        table_data_base_off: tb.type_info.varlen_element_offset(0) as i32,

        upvalue_value_off: extract_offset(u, "value"),

        closure_func_idx_off: extract_offset(c, "func_table_idx"),
        closure_arity_off: extract_offset(c, "arity"),
        closure_name_off: extract_offset(c, "name_ptr"),
        closure_upval_base_off: c.type_info.varlen_element_offset(0) as i32,

        class_name_off: extract_offset(cl, "name_ptr"),
        class_super_off: extract_offset(cl, "superclass"),
        class_methods_off: extract_offset(cl, "methods"),

        instance_class_off: extract_offset(inst, "class_ptr"),
        instance_fields_off: extract_offset(inst, "fields"),

        bound_receiver_off: extract_offset(bm, "receiver"),
        bound_method_off: extract_offset(bm, "method"),

        string_len_off: extract_offset(s, "len"),
        string_data_base_off: s.type_info.varlen_element_offset(0) as i32,

        native_name_off: extract_offset(nf, "name_ptr"),
        native_func_ptr_off: extract_offset(nf, "func_ptr"),
    }
}

pub fn lower(program: &Program) -> LoweredProgram {
    let mut dm = DynModule::new(
        GcConfig::leak(),
        NanBoxTags { nil: TAG_NIL, bool_tag: TAG_BOOL, ptr: TAG_OBJ },
    );
    dm.register_slow_paths("lox");

    let gc_types = declare_gc_types(&mut dm);
    let externs = declare_externs(&mut dm);
    let mut strings = StringPool::new();
    let resolved = resolve::resolve(program);

    // Collect and declare all functions (+1 for closure param).
    // Methods get +2 (closure + this).
    let fun_decls = collect_funs(&program.stmts);
    let mut func_names: HashMap<String, FuncRef> = HashMap::new();
    let mut func_arities: HashMap<String, usize> = HashMap::new();
    for decl in &fun_decls {
        if func_names.contains_key(&decl.name) { continue; }
        let captures = resolved.functions.get(&decl.name);
        let is_method = captures.map_or(false, |c| c.is_method);
        let extra = if is_method { 2 } else { 1 }; // closure + maybe this
        let fref = dm.declare_func(&decl.name, decl.params.len() + extra);
        func_names.insert(decl.name.clone(), fref);
        func_arities.insert(decl.name.clone(), decl.params.len());
    }
    fn collect_all_classes_stmt<'a>(stmt: &'a Stmt, out: &mut Vec<&'a ClassDecl>) {
        match stmt {
            Stmt::Class(decl) => {
                out.push(decl);
                for method in &decl.methods {
                    for s in &method.body { collect_all_classes_stmt(s, out); }
                }
            }
            Stmt::Block(stmts) => { for s in stmts { collect_all_classes_stmt(s, out); } }
            Stmt::If(_, t, e) => {
                collect_all_classes_stmt(t, out);
                if let Some(e) = e { collect_all_classes_stmt(e, out); }
            }
            Stmt::While(_, b) => collect_all_classes_stmt(b, out),
            Stmt::Fun(decl) => { for s in &decl.body { collect_all_classes_stmt(s, out); } }
            _ => {}
        }
    }
    let mut all_classes = Vec::new();
    for stmt in &program.stmts { collect_all_classes_stmt(stmt, &mut all_classes); }
    for class_decl in &all_classes {
        for (method, key) in collect_method_funs(class_decl) {
            if func_names.contains_key(&key) { continue; }
            let extra = 2; // closure + this
            let fref = dm.declare_func(&key, method.params.len() + extra);
            func_names.insert(key.clone(), fref);
            func_arities.insert(key, method.params.len());
        }
    }

    // Script entry function (takes closure param too, for consistency)
    let script = dm.declare_func("__script__", 1);

    // ── Define script ─────────────────────────────────────────
    {
        let empty_captured = std::collections::HashSet::new();
        let captured = &resolved.script.captured_locals;
        let mut f = dm.start_func(script);
        let mut ctx = Ctx {
            externs: &externs,
            func_names: &func_names,
            func_arities: &func_arities,
            resolve: &resolved,
            strings: &mut strings,
            current_func_key: "__script__".to_string(),
            is_script: true,
            is_init: false,
            dead: false,
            upvalue_cells: HashMap::new(),
            captured_locals: if captured.is_empty() { &empty_captured } else { captured },
            current_class: None,
        };
        // Define clock as a native function global
        {
            let clock_name_id = ctx.strings.intern("clock");
            let clock_name_val = f.fb.iconst(Type::I64, clock_name_id as i64);
            // Make a closure with func_ptr=0, arity=0, name="clock" -- but we need a special NativeFn.
            // Instead, use lox_make_native_fn extern.
            let native_fn = f.fb.call(ctx.externs.make_native_fn, &[clock_name_val]).unwrap();
            f.fb.call(ctx.externs.define_global, &[clock_name_val, native_fn]);
            f.def_var("clock", native_fn);
        }

        // Execute all statements in source order.
        for stmt in &program.stmts {
            if ctx.dead { break; }
            lower_stmt(&mut f, &mut ctx, stmt);
        }
        if !ctx.dead {
            let nil = f.nil();
            f.fb.ret(nil);
        } else {
            f.fb.unreachable();
        }
        dm.finish_func(f);
    }

    // ── Define each function ──────────────────────────────────
    for decl in &fun_decls {
        let key = &decl.name;
        let Some(&fref) = func_names.get(key) else { continue };
        let captures = resolved.functions.get(key);
        let is_method = captures.map_or(false, |c| c.is_method);
        let empty_set = std::collections::HashSet::new();
        let captured_locals = captures.map_or(&empty_set, |c| &c.captured_locals);
        let upvalue_names = captures.map(|c| c.upvalues.as_slice()).unwrap_or(&[]);

        lower_function(&mut dm, fref, decl, key, is_method, upvalue_names, captured_locals,
                        &externs, &func_names, &func_arities, &resolved, &mut strings, None);
    }

    // ── Define methods ────────────────────────────────────────
    for class_decl in &all_classes {
        {
            for (method, key) in collect_method_funs(class_decl) {
                let Some(&fref) = func_names.get(&key) else { continue };
                let captures = resolved.functions.get(&key);
                let empty_set = std::collections::HashSet::new();
                let captured_locals = captures.map_or(&empty_set, |c| &c.captured_locals);
                let upvalue_names = captures.map(|c| c.upvalues.as_slice()).unwrap_or(&[]);

                lower_function(&mut dm, fref, method, &key, true, upvalue_names, captured_locals,
                                &externs, &func_names, &func_arities, &resolved, &mut strings,
                                Some(class_decl.name.clone()));
            }
        }
    }

    let built = dm.build();
    LoweredProgram {
        module: built.module,
        entry: script,
        strings: strings.strings,
        gc_types,
    }
}

fn lower_function(
    dm: &mut DynModule,
    fref: FuncRef,
    decl: &FunDecl,
    key: &str,
    is_method: bool,
    upvalue_names: &[String],
    captured_locals: &std::collections::HashSet<String>,
    externs: &Externs,
    func_names: &HashMap<String, FuncRef>,
    func_arities: &HashMap<String, usize>,
    resolve: &ResolveResult,
    strings: &mut StringPool,
    current_class: Option<String>,
) {
    let mut f = dm.start_func(fref);
    let is_init = is_method && decl.name == "init";
    let mut ctx = Ctx {
        externs,
        func_names,
        func_arities,
        resolve,
        strings,
        current_func_key: key.to_string(),
        is_script: false,
        is_init,
        dead: false,
        upvalue_cells: HashMap::new(),
        captured_locals,
        current_class,
    };

    let entry = f.fb.entry_block();
    let mut param_idx = 0;

    // Param 0: closure
    let closure_val = f.fb.block_param(entry, param_idx);
    param_idx += 1;
    f.def_var("__closure__", closure_val);

    // Load upvalues from closure
    for (i, uv_name) in upvalue_names.iter().enumerate() {
        let idx_val = f.fb.iconst(Type::I64, i as i64);
        let cell = f.fb.call(ctx.externs.closure_upvalue, &[closure_val, idx_val]).unwrap();
        let cell_var = format!("__upval_{}__", uv_name);
        f.def_var(&cell_var, cell);
        ctx.upvalue_cells.insert(uv_name.clone(), cell_var);
    }

    // Param 1 (methods only): this
    if is_method {
        let this_val = f.fb.block_param(entry, param_idx);
        param_idx += 1;
        f.def_var("this", this_val);
    }

    // User params
    for param in &decl.params {
        let val = f.fb.block_param(entry, param_idx);
        param_idx += 1;
        if ctx.captured_locals.contains(param) {
            // This param is captured — wrap in upvalue cell
            let cell = f.fb.call(ctx.externs.alloc_upvalue, &[val]).unwrap();
            let cell_var = format!("__upval_{}__", param);
            f.def_var(&cell_var, cell);
            ctx.upvalue_cells.insert(param.clone(), cell_var);
        } else {
            f.def_var(param, val);
        }
    }

    for stmt in &decl.body {
        if ctx.dead { break; }
        lower_stmt(&mut f, &mut ctx, stmt);
    }

    if !ctx.dead {
        if is_init {
            // init methods implicitly return this
            let this = load_variable(&mut f, &mut ctx, "this");
            f.fb.ret(this);
        } else {
            let nil = f.nil();
            f.fb.ret(nil);
        }
    } else {
        f.fb.unreachable();
    }
    dm.finish_func(f);
}

// ── Statement lowering ────────────────────────────────────────────

fn lower_stmt(f: &mut DynFunc, ctx: &mut Ctx, stmt: &Stmt) {
    if ctx.dead { return; }

    match stmt {
        Stmt::Expr(e) => { lower_expr(f, ctx, e); }

        Stmt::Print(e) => {
            let v = lower_expr(f, ctx, e);
            f.fb.call(ctx.externs.print, &[v]);
        }

        Stmt::Var(name, init) => {
            let val = if let Some(e) = init {
                lower_expr(f, ctx, e)
            } else {
                f.nil()
            };
            if ctx.is_script {
                // Script vars go to global table
                let name_id = ctx.strings.intern(name);
                let id_val = f.fb.iconst(Type::I64, name_id as i64);
                f.fb.call(ctx.externs.define_global, &[id_val, val]);
                // Also local for fast access within script
                if ctx.captured_locals.contains(name) {
                    let cell = f.fb.call(ctx.externs.alloc_upvalue, &[val]).unwrap();
                    let cell_var = format!("__upval_{}__", name);
                    f.def_var(&cell_var, cell);
                    ctx.upvalue_cells.insert(name.clone(), cell_var);
                } else {
                    f.def_var(name, val);
                }
            } else if ctx.captured_locals.contains(name) {
                let cell = f.fb.call(ctx.externs.alloc_upvalue, &[val]).unwrap();
                let cell_var = format!("__upval_{}__", name);
                f.def_var(&cell_var, cell);
                ctx.upvalue_cells.insert(name.clone(), cell_var);
            } else {
                f.def_var(name, val);
            }
        }

        Stmt::Block(stmts) => {
            f.push_scope();
            for s in stmts {
                if ctx.dead { break; }
                lower_stmt(f, ctx, s);
            }
            f.pop_scope();
        }

        Stmt::If(cond, then_branch, else_branch) => {
            let then_bb = f.fb.create_block(&[]);
            let else_bb = f.fb.create_block(&[]);
            let merge_bb = f.fb.create_block(&[]);

            lower_branch(f, ctx, cond, then_bb, &[], else_bb, &[]);

            f.fb.switch_to_block(then_bb);
            ctx.dead = false;
            lower_stmt(f, ctx, then_branch);
            let then_dead = ctx.dead;
            if !then_dead { f.fb.jump(merge_bb, &[]); }
            else { f.fb.unreachable(); }

            f.fb.switch_to_block(else_bb);
            ctx.dead = false;
            if let Some(eb) = else_branch { lower_stmt(f, ctx, eb); }
            let else_dead = ctx.dead;
            if !else_dead { f.fb.jump(merge_bb, &[]); }
            else { f.fb.unreachable(); }

            ctx.dead = then_dead && else_dead;
            if !ctx.dead { f.fb.switch_to_block(merge_bb); }
            else { let d = f.fb.create_block(&[]); f.fb.switch_to_block(d); }
        }

        Stmt::While(cond, body) => {
            let header = f.fb.create_block(&[]);
            let body_bb = f.fb.create_block(&[]);
            let exit = f.fb.create_block(&[]);
            f.fb.jump(header, &[]);
            f.fb.switch_to_block(header);
            lower_branch(f, ctx, cond, body_bb, &[], exit, &[]);
            f.fb.switch_to_block(body_bb);
            ctx.dead = false;
            lower_stmt(f, ctx, body);
            if !ctx.dead { f.fb.jump(header, &[]); }
            else { f.fb.unreachable(); }
            ctx.dead = false;
            f.fb.switch_to_block(exit);
        }

        Stmt::Return(expr) => {
            let val = if let Some(e) = expr {
                lower_expr(f, ctx, e)
            } else if ctx.is_init {
                // Early return in init returns this
                load_variable(f, ctx, "this")
            } else {
                f.nil()
            };
            f.fb.ret(val);
            ctx.dead = true;
            let d = f.fb.create_block(&[]);
            f.fb.switch_to_block(d);
        }

        Stmt::Fun(decl) => {
            // Define the variable first (with nil) so recursive references work.
            // The closure creation may need an upvalue cell for the function
            // itself (self-recursive closures).
            let nil = f.nil();
            if ctx.is_script {
                let name_id = ctx.strings.intern(&decl.name);
                let id_val = f.fb.iconst(Type::I64, name_id as i64);
                f.fb.call(ctx.externs.define_global, &[id_val, nil]);
            }
            if ctx.captured_locals.contains(&decl.name) && !ctx.upvalue_cells.contains_key(&decl.name) {
                let cell = f.fb.call(ctx.externs.alloc_upvalue, &[nil]).unwrap();
                let cell_var = format!("__upval_{}__", decl.name);
                f.def_var(&cell_var, cell);
                ctx.upvalue_cells.insert(decl.name.clone(), cell_var);
            } else if !f.has_var(&decl.name) {
                f.def_var(&decl.name, nil);
            }

            // Now create the closure (can reference itself via the upvalue cell)
            let closure_val = make_closure(f, ctx, &decl.name, &decl.name);

            // Update the variable to hold the actual closure
            store_variable(f, ctx, &decl.name, closure_val);
        }

        Stmt::Class(decl) => {
            lower_class(f, ctx, decl);
        }
    }
}

// ── Expression lowering ───────────────────────────────────────────

fn lower_expr(f: &mut DynFunc, ctx: &mut Ctx, expr: &Expr) -> Value {
    match expr {
        Expr::Number(n) => f.number(*n),
        Expr::Bool(b) => f.bool_val(*b),
        Expr::Nil => f.nil(),

        Expr::String(s) => {
            let id = ctx.strings.intern(s);
            let id_val = f.fb.iconst(Type::I64, id as i64);
            f.fb.call(ctx.externs.resolve_string, &[id_val]).unwrap()
        }

        Expr::Grouping(e) => lower_expr(f, ctx, e),
        Expr::Var(name) => load_variable(f, ctx, name),

        Expr::Assign(name, val) => {
            let v = lower_expr(f, ctx, val);
            store_variable(f, ctx, name, v);
            v
        }

        Expr::Unary(op, e) => {
            let v = lower_expr(f, ctx, e);
            match op {
                UnaryOp::Neg => f.dyn_neg(v),
                UnaryOp::Not => {
                    let falsey = f.is_falsey(v);
                    let t = f.bool_val(true);
                    let fa = f.bool_val(false);
                    f.fb.select(falsey, t, fa)
                }
            }
        }

        Expr::Binary(lhs, op, rhs) => {
            let l = lower_expr(f, ctx, lhs);
            let r = lower_expr(f, ctx, rhs);
            match op {
                BinOp::Add => f.dyn_add(l, r),
                BinOp::Sub => f.dyn_sub(l, r),
                BinOp::Mul => f.dyn_mul(l, r),
                BinOp::Div => f.dyn_div(l, r),
                BinOp::Eq => f.dyn_eq(l, r),
                BinOp::Ne => {
                    let eq = f.dyn_eq(l, r);
                    let falsey = f.is_falsey(eq);
                    let t = f.bool_val(true);
                    let fa = f.bool_val(false);
                    f.fb.select(falsey, t, fa)
                }
                BinOp::Lt => f.dyn_lt(l, r),
                BinOp::Gt => f.dyn_gt(l, r),
                BinOp::Le => {
                    // a <= b is !(a > b)
                    let gt = f.dyn_gt(l, r);
                    let falsey = f.is_falsey(gt);
                    let t = f.bool_val(true);
                    let fa = f.bool_val(false);
                    f.fb.select(falsey, t, fa)
                }
                BinOp::Ge => {
                    // a >= b is !(a < b)
                    let lt = f.dyn_lt(l, r);
                    let falsey = f.is_falsey(lt);
                    let t = f.bool_val(true);
                    let fa = f.bool_val(false);
                    f.fb.select(falsey, t, fa)
                }
            }
        }

        Expr::Logical(lhs, op, rhs) => {
            let a = lower_expr(f, ctx, lhs);
            let short_bb = f.fb.create_block(&[]);
            let eval_bb = f.fb.create_block(&[]);
            let merge_bb = f.fb.create_block(&[Type::I64]);
            match op {
                LogicalOp::And => {
                    let falsey = f.is_falsey(a);
                    f.fb.br_if(falsey, short_bb, &[], eval_bb, &[]);
                }
                LogicalOp::Or => {
                    let truthy = f.is_truthy(a);
                    f.fb.br_if(truthy, short_bb, &[], eval_bb, &[]);
                }
            }
            f.fb.switch_to_block(short_bb);
            f.fb.jump(merge_bb, &[a]);
            f.fb.switch_to_block(eval_bb);
            let b = lower_expr(f, ctx, rhs);
            f.fb.jump(merge_bb, &[b]);
            f.fb.switch_to_block(merge_bb);
            f.fb.block_param(merge_bb, 0)
        }

        Expr::Call(callee, args) => {
            lower_call(f, ctx, callee, args)
        }

        Expr::Get(obj_expr, name) => {
            let obj = lower_expr(f, ctx, obj_expr);
            let name_id = ctx.strings.intern(name);
            let name_val = f.fb.iconst(Type::I64, name_id as i64);
            f.fb.call(ctx.externs.get_property, &[obj, name_val]).unwrap()
        }

        Expr::Set(obj_expr, name, val_expr) => {
            let obj = lower_expr(f, ctx, obj_expr);
            let val = lower_expr(f, ctx, val_expr);
            let name_id = ctx.strings.intern(name);
            let name_val = f.fb.iconst(Type::I64, name_id as i64);
            f.fb.call(ctx.externs.set_property, &[obj, name_val, val]).unwrap()
        }

        Expr::This => load_variable(f, ctx, "this"),

        Expr::Super(method_name) => {
            let this = load_variable(f, ctx, "this");
            // Load the superclass: try 'super' variable first, then fall back to class name
            let class_val = if ctx.upvalue_cells.contains_key("super") || f.has_var("super") {
                load_variable(f, ctx, "super")
            } else {
                let class_name = ctx.current_class.as_ref()
                    .expect("'super' outside of class").clone();
                load_variable(f, ctx, &class_name)
            };
            let method_id = ctx.strings.intern(method_name);
            let method_val = f.fb.iconst(Type::I64, method_id as i64);
            // lox_get_super(this, class_val, method_name_id) -> bound method
            f.fb.call(ctx.externs.get_super, &[this, class_val, method_val]).unwrap()
        }
    }
}

// ── Function call lowering ────────────────────────────────────────

fn lower_call(f: &mut DynFunc, ctx: &mut Ctx, callee: &Expr, args: &[Expr]) -> Value {
    let arg_vals: Vec<Value> = args.iter()
        .map(|a| lower_expr(f, ctx, a))
        .collect();

    // Special cases: known direct calls
    match callee {
        Expr::Var(name) if name == "clock" => {
            return f.fb.call(ctx.externs.clock, &[]).unwrap();
        }
        Expr::Var(name) if ctx.func_names.contains_key(name.as_str()) && ctx.is_script => {
            // Direct call to a known global function. Check arity first.
            let expected_arity = ctx.func_arities.get(name.as_str()).copied().unwrap_or(0);
            let num_args = arg_vals.len();
            if expected_arity != num_args {
                let callee_val = load_variable(f, ctx, name);
                let arity_v = f.fb.iconst(Type::I64, expected_arity as i64);
                let num_args_v = f.fb.iconst(Type::I64, num_args as i64);
                f.fb.call(ctx.externs.check_arity, &[callee_val, arity_v, num_args_v]);
                return f.nil();
            }
            let fref = ctx.func_names[name.as_str()];
            let closure_val = load_variable(f, ctx, name);
            let mut all_args = vec![closure_val];
            all_args.extend(arg_vals);
            return f.fb.call(fref, &all_args).unwrap();
        }
        _ => {}
    }

    // General case: indirect call through a value
    let callee_val = lower_expr(f, ctx, callee);
    emit_indirect_call(f, ctx, callee_val, &arg_vals)
}

/// Emit an indirect call: dispatch based on object type (closure, class, bound method).
fn emit_indirect_call(f: &mut DynFunc, ctx: &mut Ctx, callee: Value, args: &[Value]) -> Value {
    let obj_type = f.fb.call(ctx.externs.obj_type, &[callee]).unwrap();

    let closure_bb = f.fb.create_block(&[]);
    let class_bb = f.fb.create_block(&[]);
    let bound_bb = f.fb.create_block(&[]);
    let error_bb = f.fb.create_block(&[]);
    let merge_bb = f.fb.create_block(&[Type::I64]);

    // ObjType::Closure = 1, Class = 3, BoundMethod = 5
    let one = f.fb.iconst(Type::I64, 1);
    let three = f.fb.iconst(Type::I64, 3);
    let five = f.fb.iconst(Type::I64, 5);

    let num_args = args.len();
    let num_args_val = f.fb.iconst(Type::I64, num_args as i64);

    let is_closure = f.fb.icmp(CmpOp::Eq, obj_type, one);
    f.fb.br_if(is_closure, closure_bb, &[], class_bb, &[]);

    // ── Closure call
    f.fb.switch_to_block(closure_bb);
    {
        let arity = f.fb.call(ctx.externs.get_closure_arity, &[callee]).unwrap();
        let arity_ok = f.fb.icmp(CmpOp::Eq, arity, num_args_val);
        let closure_call_bb = f.fb.create_block(&[]);
        let closure_err_bb = f.fb.create_block(&[]);
        f.fb.br_if(arity_ok, closure_call_bb, &[], closure_err_bb, &[]);

        f.fb.switch_to_block(closure_err_bb);
        f.fb.call(ctx.externs.check_arity, &[callee, arity, num_args_val]);
        let nil = f.nil();
        f.fb.jump(merge_bb, &[nil]);

        f.fb.switch_to_block(closure_call_bb);
        let func_ptr = f.fb.call(ctx.externs.closure_func_ptr, &[callee]).unwrap();
        let mut all_args = vec![callee];
        all_args.extend_from_slice(args);
        let result = f.fb.call_indirect(func_ptr, &all_args, Some(Type::I64)).unwrap();
        f.fb.jump(merge_bb, &[result]);
    }

    // ── Class call (construction)
    f.fb.switch_to_block(class_bb);
    {
        let is_class = f.fb.icmp(CmpOp::Eq, obj_type, three);
        let real_class_bb = f.fb.create_block(&[]);
        f.fb.br_if(is_class, real_class_bb, &[], bound_bb, &[]);

        f.fb.switch_to_block(real_class_bb);
        // Check init arity
        let init_arity = f.fb.call(ctx.externs.get_class_init_arity, &[callee]).unwrap();
        let sentinel = f.fb.iconst(Type::I64, 255);
        let has_init_check = f.fb.icmp(CmpOp::Ne, init_arity, sentinel);
        let has_init_arity_bb = f.fb.create_block(&[]);
        let no_init_arity_bb = f.fb.create_block(&[]);
        let after_arity_bb = f.fb.create_block(&[]);
        f.fb.br_if(has_init_check, has_init_arity_bb, &[], no_init_arity_bb, &[]);

        f.fb.switch_to_block(has_init_arity_bb);
        {
            let init_arity_ok = f.fb.icmp(CmpOp::Eq, init_arity, num_args_val);
            let init_ok_bb = f.fb.create_block(&[]);
            let init_err_bb = f.fb.create_block(&[]);
            f.fb.br_if(init_arity_ok, init_ok_bb, &[], init_err_bb, &[]);

            f.fb.switch_to_block(init_err_bb);
            f.fb.call(ctx.externs.check_arity, &[callee, init_arity, num_args_val]);
            let nil = f.nil();
            f.fb.jump(merge_bb, &[nil]);

            f.fb.switch_to_block(init_ok_bb);
            f.fb.jump(after_arity_bb, &[]);
        }

        f.fb.switch_to_block(no_init_arity_bb);
        if num_args > 0 {
            let zero_val = f.fb.iconst(Type::I64, 0);
            f.fb.call(ctx.externs.check_arity, &[callee, zero_val, num_args_val]);
            let nil = f.nil();
            f.fb.jump(merge_bb, &[nil]);
        } else {
            f.fb.jump(after_arity_bb, &[]);
        }

        f.fb.switch_to_block(after_arity_bb);
        let instance = f.fb.call(ctx.externs.construct_instance, &[callee]).unwrap();
        let init_ptr = f.fb.call(ctx.externs.class_init_ptr, &[callee]).unwrap();
        let zero = f.fb.iconst(Type::I64, 0);
        let has_init = f.fb.icmp(CmpOp::Ne, init_ptr, zero);
        let call_init_bb = f.fb.create_block(&[]);
        let no_init_bb = f.fb.create_block(&[]);
        f.fb.br_if(has_init, call_init_bb, &[], no_init_bb, &[]);

        f.fb.switch_to_block(call_init_bb);
        let init_closure = f.fb.call(ctx.externs.class_init_closure, &[callee]).unwrap();
        let mut init_args = vec![init_closure, instance];
        init_args.extend_from_slice(args);
        f.fb.call_indirect(init_ptr, &init_args, Some(Type::I64));
        f.fb.jump(merge_bb, &[instance]);

        f.fb.switch_to_block(no_init_bb);
        f.fb.jump(merge_bb, &[instance]);
    }

    // ── Bound method call
    f.fb.switch_to_block(bound_bb);
    {
        let is_bound = f.fb.icmp(CmpOp::Eq, obj_type, five);
        let real_bound_bb = f.fb.create_block(&[]);
        f.fb.br_if(is_bound, real_bound_bb, &[], error_bb, &[]);

        f.fb.switch_to_block(real_bound_bb);
        let bound_arity = f.fb.call(ctx.externs.get_bound_arity, &[callee]).unwrap();
        let bound_ok = f.fb.icmp(CmpOp::Eq, bound_arity, num_args_val);
        let bound_call_bb = f.fb.create_block(&[]);
        let bound_err_bb = f.fb.create_block(&[]);
        f.fb.br_if(bound_ok, bound_call_bb, &[], bound_err_bb, &[]);

        f.fb.switch_to_block(bound_err_bb);
        f.fb.call(ctx.externs.check_arity, &[callee, bound_arity, num_args_val]);
        let nil = f.nil();
        f.fb.jump(merge_bb, &[nil]);

        f.fb.switch_to_block(bound_call_bb);
        let receiver = f.fb.call(ctx.externs.bound_receiver, &[callee]).unwrap();
        let method_closure = f.fb.call(ctx.externs.bound_method_closure, &[callee]).unwrap();
        let func_ptr = f.fb.call(ctx.externs.bound_closure_func_ptr, &[callee]).unwrap();
        let mut method_args = vec![method_closure, receiver];
        method_args.extend_from_slice(args);
        let result = f.fb.call_indirect(func_ptr, &method_args, Some(Type::I64)).unwrap();
        f.fb.jump(merge_bb, &[result]);
    }

    // ── Error: not callable
    f.fb.switch_to_block(error_bb);
    {
        f.fb.call(ctx.externs.call_non_callable, &[]);
        let nil = f.nil();
        f.fb.jump(merge_bb, &[nil]);
    }

    f.fb.switch_to_block(merge_bb);
    f.fb.block_param(merge_bb, 0)
}

// ── Class lowering ────────────────────────────────────────────────

fn lower_class(f: &mut DynFunc, ctx: &mut Ctx, decl: &ClassDecl) {
    let name_id = ctx.strings.intern(&decl.name);
    let name_val = f.fb.iconst(Type::I64, name_id as i64);
    let class_val = f.fb.call(ctx.externs.make_class, &[name_val]).unwrap();

    // Define class variable early so methods can reference it
    if ctx.is_script {
        let id_val = f.fb.iconst(Type::I64, name_id as i64);
        f.fb.call(ctx.externs.define_global, &[id_val, class_val]);
    }
    if !f.has_var(&decl.name) {
        f.def_var(&decl.name, class_val);
    } else {
        f.set_var(&decl.name, class_val);
    }

    // Inherit
    if let Some(super_name) = &decl.superclass {
        let super_val = load_variable(f, ctx, super_name);
        f.fb.call(ctx.externs.class_inherit, &[class_val, super_val]);

        // Define 'super' as a variable holding the DECLARING class value
        // (lox_get_super looks up the superclass from this class's .superclass field)
        let super_scope_key = format!("{}__super", decl.name);
        let super_captures = ctx.resolve.functions.get(&super_scope_key);
        if super_captures.map_or(false, |c| c.captured_locals.contains("super")) {
            // 'super' is captured by a method — wrap in upvalue cell
            let cell = f.fb.call(ctx.externs.alloc_upvalue, &[class_val]).unwrap();
            let cell_var = "__upval_super__".to_string();
            f.def_var(&cell_var, cell);
            ctx.upvalue_cells.insert("super".to_string(), cell_var);
        } else {
            f.def_var("super", class_val);
        }
    }

    // Methods
    for method in &decl.methods {
        let key = format!("{}.{}", decl.name, method.name);
        let method_closure = make_closure(f, ctx, &key, &key);
        let method_name_id = ctx.strings.intern(&method.name);
        let method_name_val = f.fb.iconst(Type::I64, method_name_id as i64);
        f.fb.call(ctx.externs.class_add_method, &[class_val, method_name_val, method_closure]);
    }
}

// ── Closure creation ──────────────────────────────────────────────

fn make_closure(f: &mut DynFunc, ctx: &mut Ctx, func_key: &str, decl_name: &str) -> Value {
    let fref = ctx.func_names[func_key];
    let func_idx = f.fb.iconst(Type::I64, fref.index() as i64);

    let captures = ctx.resolve.functions.get(func_key);
    let upvalue_names = captures.map(|c| c.upvalues.as_slice()).unwrap_or(&[]);
    let arity = ctx.func_arities.get(func_key).copied().unwrap_or(0);
    let num_upvalues = f.fb.iconst(Type::I64, upvalue_names.len() as i64);

    // Compute display name: for methods "Class.method", just use the method part
    let display_name = if decl_name.contains('.') {
        decl_name.split('.').last().unwrap_or(decl_name)
    } else {
        decl_name
    };
    let name_id = ctx.strings.intern(display_name);

    // lox_make_closure(func_table_idx, num_upvalues, arity, name_id) -> closure_val
    let arity_val = f.fb.iconst(Type::I64, arity as i64);
    let name_id_val = f.fb.iconst(Type::I64, name_id as i64);
    let closure_val = f.fb.call(ctx.externs.make_closure, &[func_idx, num_upvalues, arity_val, name_id_val]).unwrap();

    // Set each upvalue
    for (i, uv_name) in upvalue_names.iter().enumerate() {
        let cell = if let Some(cell_var) = ctx.upvalue_cells.get(uv_name) {
            f.get_var(cell_var)
        } else {
            // The variable isn't in an upvalue cell in our scope — it must be
            // a captured local. Allocate a cell for it now.
            let val = if f.has_var(uv_name) {
                f.get_var(uv_name)
            } else {
                // Try global
                let name_id = ctx.strings.intern(uv_name);
                let id_val = f.fb.iconst(Type::I64, name_id as i64);
                f.fb.call(ctx.externs.get_global, &[id_val]).unwrap()
            };
            let cell = f.fb.call(ctx.externs.alloc_upvalue, &[val]).unwrap();
            let cell_var = format!("__upval_{}__", uv_name);
            f.def_var(&cell_var, cell);
            ctx.upvalue_cells.insert(uv_name.clone(), cell_var);
            f.get_var(&format!("__upval_{}__", uv_name))
        };
        let idx = f.fb.iconst(Type::I64, i as i64);
        f.fb.call(ctx.externs.set_closure_upvalue, &[closure_val, idx, cell]);
    }

    closure_val
}

// ── Helpers ───────────────────────────────────────────────────────

fn lower_branch(
    f: &mut DynFunc, ctx: &mut Ctx, cond: &Expr,
    then_bb: BlockId, then_args: &[Value],
    else_bb: BlockId, else_args: &[Value],
) {
    if let Some(cmp_op) = as_num_cmp(cond) {
        let Expr::Binary(lhs, _, rhs) = cond else { unreachable!() };
        let l = lower_expr(f, ctx, lhs);
        let r = lower_expr(f, ctx, rhs);
        let fa = f.fb.bitcast(l, Type::F64);
        let fb_val = f.fb.bitcast(r, Type::F64);
        let cmp = f.fb.fcmp(cmp_op, fa, fb_val);
        f.fb.br_if(cmp, then_bb, then_args, else_bb, else_args);
        return;
    }
    let v = lower_expr(f, ctx, cond);
    f.br_if_truthy(v, then_bb, then_args, else_bb, else_args);
}

fn as_num_cmp(expr: &Expr) -> Option<CmpOp> {
    match expr {
        Expr::Binary(_, op, _) => match op {
            BinOp::Lt => Some(CmpOp::Slt),
            BinOp::Le => Some(CmpOp::Sle),
            BinOp::Gt => Some(CmpOp::Sgt),
            BinOp::Ge => Some(CmpOp::Sge),
            _ => None,
        },
        _ => None,
    }
}

fn expr_might_be_string(expr: &Expr) -> bool {
    match expr {
        Expr::String(_) => true,
        Expr::Binary(l, BinOp::Add, r) => expr_might_be_string(l) || expr_might_be_string(r),
        Expr::Grouping(e) => expr_might_be_string(e),
        Expr::Assign(_, e) => expr_might_be_string(e),
        // Variables/calls/properties could be strings
        Expr::Var(_) | Expr::Call(_, _) | Expr::Get(_, _) => true,
        _ => false,
    }
}

fn load_variable(f: &mut DynFunc, ctx: &mut Ctx, name: &str) -> Value {
    // Check locals first (may shadow upvalue cells)
    if f.has_var(name) {
        return f.get_var(name);
    }
    // Check upvalue cells
    if let Some(cell_var) = ctx.upvalue_cells.get(name) {
        let cell = f.get_var(cell_var);
        return f.fb.call(ctx.externs.get_upvalue, &[cell]).unwrap();
    }
    // Fall back to global
    let name_id = ctx.strings.intern(name);
    let id_val = f.fb.iconst(Type::I64, name_id as i64);
    f.fb.call(ctx.externs.get_global, &[id_val]).unwrap()
}

fn store_variable(f: &mut DynFunc, ctx: &mut Ctx, name: &str, val: Value) {
    // Check upvalue cells first
    if let Some(cell_var) = ctx.upvalue_cells.get(name) {
        let cell = f.get_var(cell_var);
        f.fb.call(ctx.externs.set_upvalue, &[cell, val]);
        // Also sync to global if in script
        if ctx.is_script {
            let name_id = ctx.strings.intern(name);
            let id_val = f.fb.iconst(Type::I64, name_id as i64);
            f.fb.call(ctx.externs.set_global, &[id_val, val]);
        }
        return;
    }
    // Check locals
    if f.has_var(name) {
        f.set_var(name, val);
        // Sync to global if in script
        if ctx.is_script {
            let name_id = ctx.strings.intern(name);
            let id_val = f.fb.iconst(Type::I64, name_id as i64);
            f.fb.call(ctx.externs.set_global, &[id_val, val]);
        }
        return;
    }
    // Global
    let name_id = ctx.strings.intern(name);
    let id_val = f.fb.iconst(Type::I64, name_id as i64);
    f.fb.call(ctx.externs.set_global, &[id_val, val]);
}
