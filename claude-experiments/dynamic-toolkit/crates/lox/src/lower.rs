/// Lower Lox AST → dynir Module.
///
/// All values are NanBox-encoded I64. Dynamic operations dispatch through
/// extern runtime functions. Functions are first-class via CallIndirect.
use std::collections::HashMap;

use dynir::builder::{FunctionBuilder, ModuleBuilder};
use dynir::ir::{FuncRef, Module, Value};
use dynir::types::{Signature, Type};

use crate::ast::*;
use crate::value;

pub struct LoweredProgram {
    pub module: Module,
    pub entry: FuncRef,
}

// All extern runtime functions
struct Externs {
    add: FuncRef,
    sub: FuncRef,
    mul: FuncRef,
    div: FuncRef,
    negate: FuncRef,
    not: FuncRef,
    equal: FuncRef,
    greater: FuncRef,
    less: FuncRef,
    is_falsey: FuncRef,
    print: FuncRef,
    define_global: FuncRef,
    get_global: FuncRef,
    set_global: FuncRef,
    get_property: FuncRef,
    set_property: FuncRef,
    get_super: FuncRef,
    call_value: FuncRef,
    invoke: FuncRef,
    super_invoke: FuncRef,
    make_class: FuncRef,
    inherit: FuncRef,
    define_method: FuncRef,
    make_string: FuncRef,
}

fn sig(p: &[Type], r: Option<Type>) -> Signature {
    Signature { params: p.to_vec(), ret: r }
}

fn declare_externs(mb: &mut ModuleBuilder) -> Externs {
    use Type::I64;
    Externs {
        add: mb.declare_extern("lox_add", sig(&[I64, I64], Some(I64))),
        sub: mb.declare_extern("lox_sub", sig(&[I64, I64], Some(I64))),
        mul: mb.declare_extern("lox_mul", sig(&[I64, I64], Some(I64))),
        div: mb.declare_extern("lox_div", sig(&[I64, I64], Some(I64))),
        negate: mb.declare_extern("lox_negate", sig(&[I64], Some(I64))),
        not: mb.declare_extern("lox_not", sig(&[I64], Some(I64))),
        equal: mb.declare_extern("lox_equal", sig(&[I64, I64], Some(I64))),
        greater: mb.declare_extern("lox_greater", sig(&[I64, I64], Some(I64))),
        less: mb.declare_extern("lox_less", sig(&[I64, I64], Some(I64))),
        is_falsey: mb.declare_extern("lox_is_falsey", sig(&[I64], Some(I64))),
        print: mb.declare_extern("lox_print", sig(&[I64], None)),
        define_global: mb.declare_extern("lox_define_global", sig(&[I64, I64], None)),
        get_global: mb.declare_extern("lox_get_global", sig(&[I64], Some(I64))),
        set_global: mb.declare_extern("lox_set_global", sig(&[I64, I64], Some(I64))),
        get_property: mb.declare_extern("lox_get_property", sig(&[I64, I64], Some(I64))),
        set_property: mb.declare_extern("lox_set_property", sig(&[I64, I64, I64], Some(I64))),
        get_super: mb.declare_extern("lox_get_super", sig(&[I64, I64, I64], Some(I64))),
        // call_value: callee + up to 8 args encoded; we use a variadic approach:
        // pass (callee, arg_count, arg0, arg1, ..., arg7)
        // Actually, we'll use CallIndirect for this. The handler receives (callee, args...).
        // For simplicity, let's cap at 8 args inline for now.
        call_value: mb.declare_extern("lox_call_value", sig(&[I64; 10], Some(I64))),
        invoke: mb.declare_extern("lox_invoke", sig(&[I64; 10], Some(I64))),
        super_invoke: mb.declare_extern("lox_super_invoke", sig(&[I64; 11], Some(I64))),
        make_class: mb.declare_extern("lox_make_class", sig(&[I64], Some(I64))),
        inherit: mb.declare_extern("lox_inherit", sig(&[I64, I64], None)),
        define_method: mb.declare_extern("lox_define_method", sig(&[I64, I64, I64], None)),
        make_string: mb.declare_extern("lox_make_string", sig(&[I64], Some(I64))),
    }
}

pub fn lower(program: &Program) -> LoweredProgram {
    let mut mb = ModuleBuilder::new();
    let externs = declare_externs(&mut mb);

    // The script body becomes a function "lox_script" with no params
    let entry = mb.declare_func("lox_script", &[], Some(Type::I64));

    // First pass: collect all function declarations (for forward references)
    // This is tricky because Lox functions are defined with statements, not top-level.
    // For now, we'll handle this during lowering by using globals for function values.

    // Define the script function
    let fb = mb.define_func(entry);
    let mut lowerer = FuncLowerer::new(fb, &externs);

    for stmt in &program.stmts {
        lowerer.lower_stmt(stmt);
        if lowerer.dead {
            break;
        }
    }

    if !lowerer.dead {
        let nil = lowerer.fb.iconst(Type::I64, value::nil_val() as i64);
        lowerer.fb.ret(nil);
    }

    mb.finish_func(entry, lowerer.fb);
    let module = mb.build();

    LoweredProgram { module, entry }
}

struct FuncLowerer<'a> {
    fb: FunctionBuilder,
    externs: &'a Externs,
    vars: Vec<HashMap<String, Value>>,
    dead: bool,
}

impl<'a> FuncLowerer<'a> {
    fn new(fb: FunctionBuilder, externs: &'a Externs) -> Self {
        FuncLowerer {
            fb,
            externs,
            vars: vec![HashMap::new()],
            dead: false,
        }
    }

    fn push_scope(&mut self) {
        self.vars.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.vars.pop();
    }

    fn def_var(&mut self, name: &str, val: Value) {
        self.vars.last_mut().unwrap().insert(name.to_string(), val);
    }

    fn lookup_var(&self, name: &str) -> Option<Value> {
        for scope in self.vars.iter().rev() {
            if let Some(&v) = scope.get(name) {
                return Some(v);
            }
        }
        None
    }

    fn set_var(&mut self, name: &str, val: Value) {
        for scope in self.vars.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), val);
                return;
            }
        }
    }

    fn nil(&mut self) -> Value {
        self.fb.iconst(Type::I64, value::nil_val() as i64)
    }

    fn lower_stmt(&mut self, stmt: &Stmt) {
        if self.dead { return; }
        match stmt {
            Stmt::Expr(expr) => {
                self.lower_expr(expr);
            }
            Stmt::Print(expr) => {
                let val = self.lower_expr(expr);
                self.fb.call(self.externs.print, &[val]);
            }
            Stmt::Var(name, init) => {
                let val = if let Some(e) = init {
                    self.lower_expr(e)
                } else {
                    self.nil()
                };
                self.def_var(name, val);
            }
            Stmt::Block(stmts) => {
                self.push_scope();
                for s in stmts {
                    self.lower_stmt(s);
                    if self.dead { break; }
                }
                self.pop_scope();
            }
            Stmt::If(cond, then_branch, else_branch) => {
                self.lower_if(cond, then_branch, else_branch.as_deref());
            }
            Stmt::While(cond, body) => {
                self.lower_while(cond, body);
            }
            Stmt::Return(expr) => {
                let val = if let Some(e) = expr {
                    self.lower_expr(e)
                } else {
                    self.nil()
                };
                self.fb.ret(val);
                self.dead = true;
            }
            Stmt::Fun(decl) => {
                // For now, Lox functions are stored as global values via the runtime.
                // This is a simplification — proper closures need more work.
                // We'll define them as globals through the runtime.
                let name_val = self.make_string_constant(&decl.name);
                // For now, just define a nil placeholder — full function support needs
                // CallIndirect and closure objects.
                let nil = self.nil();
                self.fb.call(self.externs.define_global, &[name_val, nil]);
                self.def_var(&decl.name, nil);
                // TODO: proper function compilation
            }
            Stmt::Class(decl) => {
                let name_val = self.make_string_constant(&decl.name);
                let class = self.fb.call(self.externs.make_class, &[name_val]).unwrap();
                self.def_var(&decl.name, class);
                self.fb.call(self.externs.define_global, &[name_val, class]);
                // TODO: methods, inheritance
            }
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> Value {
        match expr {
            Expr::Number(n) => {
                self.fb.iconst(Type::I64, value::number_val(*n) as i64)
            }
            Expr::String(s) => {
                self.make_string_constant(s)
            }
            Expr::Bool(b) => {
                self.fb.iconst(Type::I64, value::bool_val(*b) as i64)
            }
            Expr::Nil => self.nil(),
            Expr::Var(name) => {
                if let Some(val) = self.lookup_var(name) {
                    val
                } else {
                    // Must be a global
                    let name_val = self.make_string_constant(name);
                    self.fb.call(self.externs.get_global, &[name_val]).unwrap()
                }
            }
            Expr::Assign(name, value) => {
                let val = self.lower_expr(value);
                if self.lookup_var(name).is_some() {
                    self.set_var(name, val);
                } else {
                    let name_val = self.make_string_constant(name);
                    self.fb.call(self.externs.set_global, &[name_val, val]);
                }
                val
            }
            Expr::Binary(lhs, op, rhs) => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                let func = match op {
                    BinOp::Add => self.externs.add,
                    BinOp::Sub => self.externs.sub,
                    BinOp::Mul => self.externs.mul,
                    BinOp::Div => self.externs.div,
                    BinOp::Eq => self.externs.equal,
                    BinOp::Ne => {
                        let eq = self.fb.call(self.externs.equal, &[l, r]).unwrap();
                        return self.fb.call(self.externs.not, &[eq]).unwrap();
                    }
                    BinOp::Lt => self.externs.less,
                    BinOp::Gt => self.externs.greater,
                    BinOp::Le => {
                        let gt = self.fb.call(self.externs.greater, &[l, r]).unwrap();
                        return self.fb.call(self.externs.not, &[gt]).unwrap();
                    }
                    BinOp::Ge => {
                        let lt = self.fb.call(self.externs.less, &[l, r]).unwrap();
                        return self.fb.call(self.externs.not, &[lt]).unwrap();
                    }
                };
                self.fb.call(func, &[l, r]).unwrap()
            }
            Expr::Unary(op, operand) => {
                let val = self.lower_expr(operand);
                match op {
                    UnaryOp::Neg => self.fb.call(self.externs.negate, &[val]).unwrap(),
                    UnaryOp::Not => self.fb.call(self.externs.not, &[val]).unwrap(),
                }
            }
            Expr::Logical(lhs, op, rhs) => {
                self.lower_logical(lhs, *op, rhs)
            }
            Expr::Call(callee, args) => {
                // Build arg array: [callee, arg_count, arg0..arg7]
                let callee_val = self.lower_expr(callee);
                let arg_vals: Vec<Value> = args.iter().map(|a| self.lower_expr(a)).collect();
                let arg_count = self.fb.iconst(Type::I64, args.len() as i64);
                let nil = self.nil();

                // Pack into fixed 10-arg extern: (callee, argc, a0, a1, ..., a7)
                let mut call_args = vec![callee_val, arg_count];
                for i in 0..8 {
                    if i < arg_vals.len() {
                        call_args.push(arg_vals[i]);
                    } else {
                        call_args.push(nil);
                    }
                }
                self.fb.call(self.externs.call_value, &call_args).unwrap()
            }
            Expr::Get(object, name) => {
                let obj = self.lower_expr(object);
                let name_val = self.make_string_constant(name);
                self.fb.call(self.externs.get_property, &[obj, name_val]).unwrap()
            }
            Expr::Set(object, name, value) => {
                let obj = self.lower_expr(object);
                let name_val = self.make_string_constant(name);
                let val = self.lower_expr(value);
                self.fb.call(self.externs.set_property, &[obj, name_val, val]).unwrap()
            }
            Expr::This => {
                if let Some(val) = self.lookup_var("this") {
                    val
                } else {
                    let name_val = self.make_string_constant("this");
                    self.fb.call(self.externs.get_global, &[name_val]).unwrap()
                }
            }
            Expr::Super(method) => {
                let this = if let Some(v) = self.lookup_var("this") { v } else { self.nil() };
                let superclass = if let Some(v) = self.lookup_var("super") { v } else { self.nil() };
                let name_val = self.make_string_constant(method);
                self.fb.call(self.externs.get_super, &[this, name_val, superclass]).unwrap()
            }
            Expr::Grouping(inner) => self.lower_expr(inner),
        }
    }

    fn lower_if(&mut self, cond: &Expr, then_branch: &Stmt, else_branch: Option<&Stmt>) {
        let c = self.lower_expr(cond);
        let falsey = self.fb.call(self.externs.is_falsey, &[c]).unwrap();
        let cond_i8 = self.fb.trunc(falsey, Type::I8);

        let then_bb = self.fb.create_block(&[]);
        let else_bb = self.fb.create_block(&[]);
        let merge_bb = self.fb.create_block(&[]);

        // If falsey → else, otherwise → then
        self.fb.br_if(cond_i8, else_bb, &[], then_bb, &[]);

        // Then
        self.fb.switch_to_block(then_bb);
        self.dead = false;
        self.push_scope();
        self.lower_stmt(then_branch);
        self.pop_scope();
        let then_dead = self.dead;
        if !then_dead {
            self.fb.jump(merge_bb, &[]);
        } else {
            self.fb.unreachable();
        }

        // Else
        self.fb.switch_to_block(else_bb);
        self.dead = false;
        if let Some(eb) = else_branch {
            self.push_scope();
            self.lower_stmt(eb);
            self.pop_scope();
        }
        let else_dead = self.dead;
        if !else_dead {
            self.fb.jump(merge_bb, &[]);
        } else {
            self.fb.unreachable();
        }

        self.dead = then_dead && else_dead;
        self.fb.switch_to_block(merge_bb);
    }

    fn lower_while(&mut self, cond: &Expr, body: &Stmt) {
        let header_bb = self.fb.create_block(&[]);
        let body_bb = self.fb.create_block(&[]);
        let exit_bb = self.fb.create_block(&[]);

        self.fb.jump(header_bb, &[]);

        self.fb.switch_to_block(header_bb);
        let c = self.lower_expr(cond);
        let falsey = self.fb.call(self.externs.is_falsey, &[c]).unwrap();
        let cond_i8 = self.fb.trunc(falsey, Type::I8);
        self.fb.br_if(cond_i8, exit_bb, &[], body_bb, &[]);

        self.fb.switch_to_block(body_bb);
        self.push_scope();
        self.lower_stmt(body);
        self.pop_scope();
        if !self.dead {
            self.fb.jump(header_bb, &[]);
        } else {
            self.fb.unreachable();
        }

        self.dead = false;
        self.fb.switch_to_block(exit_bb);
    }

    fn lower_logical(&mut self, lhs: &Expr, op: LogicalOp, rhs: &Expr) -> Value {
        let left = self.lower_expr(lhs);
        let falsey = self.fb.call(self.externs.is_falsey, &[left]).unwrap();
        let cond_i8 = self.fb.trunc(falsey, Type::I8);

        let rhs_bb = self.fb.create_block(&[]);
        let merge_bb = self.fb.create_block(&[Type::I64]);

        match op {
            LogicalOp::And => {
                // If left is falsey, short-circuit with left; else evaluate right
                self.fb.br_if(cond_i8, merge_bb, &[left], rhs_bb, &[]);
            }
            LogicalOp::Or => {
                // If left is truthy (not falsey), short-circuit with left; else evaluate right
                self.fb.br_if(cond_i8, rhs_bb, &[], merge_bb, &[left]);
            }
        }

        self.fb.switch_to_block(rhs_bb);
        let right = self.lower_expr(rhs);
        self.fb.jump(merge_bb, &[right]);

        self.fb.switch_to_block(merge_bb);
        self.fb.block_param(merge_bb, 0)
    }

    fn make_string_constant(&mut self, s: &str) -> Value {
        // Encode string pointer as a constant. The runtime will intern it.
        // For now, use the make_string extern to create/intern the string at runtime.
        // We pass a unique ID that the runtime maps to the actual string.
        // Simpler: just store the string hash + length as an I64 constant
        // and let the runtime look it up.
        //
        // Actually, the cleanest approach: we store the raw string bytes in
        // the constant pool and pass an index. But dynir doesn't have a
        // string constant pool. So we'll use the extern to create strings.
        //
        // For now, encode a unique ID for each string. The runtime maintains
        // a table mapping ID → string content.
        let id = string_id(s);
        let id_val = self.fb.iconst(Type::I64, id as i64);
        self.fb.call(self.externs.make_string, &[id_val]).unwrap()
    }
}

/// Deterministic string → u64 ID mapping.
pub fn string_id(s: &str) -> u64 {
    // FNV-1a hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
