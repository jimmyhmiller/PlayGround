use crate::ast::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    Struct(Rc<RefCell<StructVal>>),
    Enum(EnumVal),
    Tuple(Vec<Value>),
}

#[derive(Debug, Clone)]
pub struct StructVal {
    pub name: String,
    pub fields: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct EnumVal {
    pub name: String,
    pub variant: String,
    pub payload: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Program {
    funcs: HashMap<String, FnDecl>,
    externs: HashMap<String, ExternFnDecl>,
    structs: HashMap<String, StructDecl>,
    enums: HashMap<String, EnumDecl>,
}

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub message: String,
}

pub fn build_program(module: &Module) -> Program {
    let mut funcs = HashMap::new();
    let mut externs = HashMap::new();
    let mut structs = HashMap::new();
    let mut enums = HashMap::new();

    for item in &module.items {
        match item {
            Item::Fn(f) => {
                funcs.insert(f.name.clone(), f.clone());
            }
            Item::ExternFn(f) => {
                externs.insert(f.name.clone(), f.clone());
            }
            Item::Struct(s) => {
                structs.insert(s.name.clone(), s.clone());
            }
            Item::Enum(e) => {
                enums.insert(e.name.clone(), e.clone());
            }
            Item::Use(_) => {}
        }
    }

    Program {
        funcs,
        externs,
        structs,
        enums,
    }
}

pub fn run_main(program: &Program) -> Result<Value, RuntimeError> {
    let main = program
        .funcs
        .get("main")
        .ok_or_else(|| RuntimeError {
            message: "missing main function".to_string(),
        })?
        .clone();
    let mut eval = Evaluator::new(program);
    eval.call_fn(&main, vec![])
}

struct Evaluator<'a> {
    program: &'a Program,
    scopes: Vec<HashMap<String, Value>>,
}

enum Control {
    Value(Value),
    Return(Value),
    Break,
    Continue,
}

impl<'a> Evaluator<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            scopes: vec![HashMap::new()],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn set_local(&mut self, name: String, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, value);
        }
    }

    fn assign_local(&mut self, name: &str, value: Value) -> Result<(), RuntimeError> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return Ok(());
            }
        }
        Err(RuntimeError {
            message: format!("unknown local '{}'", name),
        })
    }

    fn get_local(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(val.clone());
            }
        }
        None
    }

    fn call_fn(&mut self, f: &FnDecl, args: Vec<Value>) -> Result<Value, RuntimeError> {
        self.push_scope();
        for (param, arg) in f.params.iter().zip(args.into_iter()) {
            self.set_local(param.name.clone(), arg);
        }
        let result = match self.eval_block(&f.body)? {
            Control::Value(v) | Control::Return(v) => v,
            Control::Break | Control::Continue => {
                return Err(RuntimeError {
                    message: "break/continue outside of loop".to_string(),
                });
            }
        };
        self.pop_scope();
        Ok(result)
    }

    fn eval_block(&mut self, block: &Block) -> Result<Control, RuntimeError> {
        self.push_scope();
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, _) => {
                    match self.eval_expr_control(expr)? {
                        Control::Value(_) => {}
                        ctrl => {
                            self.pop_scope();
                            return Ok(ctrl);
                        }
                    }
                }
                Stmt::Return(expr, _) => {
                    let value = if let Some(expr) = expr {
                        self.eval_expr(expr)?
                    } else {
                        Value::Unit
                    };
                    self.pop_scope();
                    return Ok(Control::Return(value));
                }
            }
        }
        let result = if let Some(tail) = &block.tail {
            match self.eval_expr_control(tail)? {
                Control::Value(v) => v,
                ctrl => {
                    self.pop_scope();
                    return Ok(ctrl);
                }
            }
        } else {
            Value::Unit
        };
        self.pop_scope();
        Ok(Control::Value(result))
    }

    fn eval_expr_control(&mut self, expr: &Expr) -> Result<Control, RuntimeError> {
        match expr {
            Expr::Break { .. } => Ok(Control::Break),
            Expr::Continue { .. } => Ok(Control::Continue),
            _ => Ok(Control::Value(self.eval_expr(expr)?)),
        }
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, RuntimeError> {
        match expr {
            Expr::Let {
                name,
                value,
                ..
            } => {
                let v = self.eval_expr(value)?;
                self.set_local(name.clone(), v);
                Ok(Value::Unit)
            }
            Expr::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                let c = self.eval_expr(cond)?;
                match c {
                    Value::Bool(true) => match self.eval_block(then_branch)? {
                        Control::Value(v) | Control::Return(v) => Ok(v),
                        Control::Break | Control::Continue => Ok(Value::Unit),
                    },
                    Value::Bool(false) => {
                        if let Some(else_branch) = else_branch {
                            match self.eval_block(else_branch)? {
                                Control::Value(v) | Control::Return(v) => Ok(v),
                                Control::Break | Control::Continue => Ok(Value::Unit),
                            }
                        } else {
                            Ok(Value::Unit)
                        }
                    }
                    _ => Err(RuntimeError {
                        message: "if condition must be bool".to_string(),
                    }),
                }
            }
            Expr::While { cond, body, .. } => {
                loop {
                    let c = self.eval_expr(cond)?;
                    match c {
                        Value::Bool(true) => match self.eval_block(body)? {
                            Control::Value(_) | Control::Continue => {}
                            Control::Return(v) => return Ok(v),
                            Control::Break => break,
                        },
                        Value::Bool(false) => break,
                        _ => {
                            return Err(RuntimeError {
                                message: "while condition must be bool".to_string(),
                            })
                        }
                    }
                }
                Ok(Value::Unit)
            }
            Expr::Match { scrutinee, arms, .. } => {
                let scrut = self.eval_expr(scrutinee)?;
                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &scrut) {
                        return self.eval_expr(&arm.body);
                    }
                }
                Ok(Value::Unit)
            }
            Expr::Assign { target, value, .. } => {
                let v = self.eval_expr(value)?;
                match &**target {
                    Expr::Path(path, _) => {
                        if let Some(name) = path.last() {
                            self.assign_local(name, v)?;
                            Ok(Value::Unit)
                        } else {
                            Err(RuntimeError {
                                message: "invalid assignment target".to_string(),
                            })
                        }
                    }
                    Expr::Field { base, name, .. } => {
                        let base_val = self.eval_expr(base)?;
                        match base_val {
                            Value::Struct(s) => {
                                s.borrow_mut().fields.insert(name.clone(), v);
                                Ok(Value::Unit)
                            }
                            _ => Err(RuntimeError {
                                message: "field assignment requires struct".to_string(),
                            }),
                        }
                    }
                    _ => Err(RuntimeError {
                        message: "invalid assignment target".to_string(),
                    }),
                }
            }
            Expr::Binary { op, left, right, .. } => {
                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                self.eval_binary(op, l, r)
            }
            Expr::Unary { op, expr, .. } => {
                let v = self.eval_expr(expr)?;
                match op {
                    UnaryOp::Neg => match v {
                        Value::Int(i) => Ok(Value::Int(-i)),
                        Value::Float(f) => Ok(Value::Float(-f)),
                        _ => Err(RuntimeError {
                            message: "negation requires number".to_string(),
                        }),
                    },
                    UnaryOp::Not => match v {
                        Value::Bool(b) => Ok(Value::Bool(!b)),
                        _ => Err(RuntimeError {
                            message: "not requires bool".to_string(),
                        }),
                    },
                }
            }
            Expr::Call { callee, args, .. } => {
                if let Expr::Path(path, _) = &**callee {
                    let arg_vals = args
                        .iter()
                        .map(|a| self.eval_expr(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    return self.call_path(path, arg_vals);
                }
                Err(RuntimeError {
                    message: "call target must be path".to_string(),
                })
            }
            Expr::Field { base, name, .. } => {
                let base_val = self.eval_expr(base)?;
                match base_val {
                    Value::Struct(s) => match s.borrow().fields.get(name) {
                        Some(v) => Ok(v.clone()),
                        None => Err(RuntimeError {
                            message: "unknown field".to_string(),
                        }),
                    },
                    Value::Tuple(items) => {
                        if let Ok(idx) = name.parse::<usize>() {
                            if idx < items.len() {
                                return Ok(items[idx].clone());
                            }
                        }
                        Err(RuntimeError {
                            message: "unknown tuple field".to_string(),
                        })
                    }
                    _ => Err(RuntimeError {
                        message: "field access requires struct".to_string(),
                    }),
                }
            }
            Expr::Path(path, _) => {
                if let Some(name) = path.last() {
                    if let Some(v) = self.get_local(name) {
                        return Ok(v);
                    }
                }
                Err(RuntimeError {
                    message: "unknown value".to_string(),
                })
            }
            Expr::StructLit { path, fields, .. } => {
                let name = path.last().cloned().unwrap_or_default();
                if !self.program.structs.contains_key(&name) {
                    return Err(RuntimeError {
                        message: "unknown struct".to_string(),
                    });
                }
                let mut map = HashMap::new();
                for (field, expr) in fields {
                    map.insert(field.clone(), self.eval_expr(expr)?);
                }
                Ok(Value::Struct(Rc::new(RefCell::new(StructVal { name, fields: map }))))
            }
            Expr::Tuple { items, .. } => {
                let mut out = Vec::new();
                for item in items {
                    out.push(self.eval_expr(item)?);
                }
                Ok(Value::Tuple(out))
            }
            Expr::Literal(lit, _) => match lit {
                Literal::Int(s) => Ok(Value::Int(parse_i64(s)?)),
                Literal::Char(b) => Ok(Value::Int(*b as i64)),
                Literal::Float(s) => Ok(Value::Float(parse_f64(s)?)),
                Literal::Str(s) => Ok(Value::Str(s.clone())),
                Literal::Bool(b) => Ok(Value::Bool(*b)),
                Literal::Unit => Ok(Value::Unit),
            },
            Expr::Block(block) => match self.eval_block(block)? {
                Control::Value(v) | Control::Return(v) => Ok(v),
                Control::Break | Control::Continue => Ok(Value::Unit),
            },
            Expr::Break { .. } => Err(RuntimeError {
                message: "break outside of loop".to_string(),
            }),
            Expr::Continue { .. } => Err(RuntimeError {
                message: "continue outside of loop".to_string(),
            }),
        }
    }

    fn call_path(&mut self, path: &[String], args: Vec<Value>) -> Result<Value, RuntimeError> {
        if path.len() >= 2 {
            let enum_name = &path[0];
            let variant = &path[1];
            if self.program.enums.contains_key(enum_name) {
                return Ok(Value::Enum(EnumVal {
                    name: enum_name.clone(),
                    variant: variant.clone(),
                    payload: args,
                }));
            }
        }
        if let Some(name) = path.last() {
            if let Some(f) = self.program.funcs.get(name) {
                return self.call_fn(f, args);
            }
            if self.program.externs.contains_key(name) {
                return self.call_extern(name, args);
            }
        }
        Err(RuntimeError {
            message: "unknown function".to_string(),
        })
    }

    fn call_extern(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        match name {
            "print_int" => {
                if let Some(Value::Int(i)) = args.get(0) {
                    println!("{}", i);
                    return Ok(Value::Unit);
                }
                Err(RuntimeError {
                    message: "print_int expects int".to_string(),
                })
            }
            "print_str" => {
                if let Some(Value::Str(s)) = args.get(0) {
                    println!("{}", s);
                    return Ok(Value::Unit);
                }
                Err(RuntimeError {
                    message: "print_str expects string".to_string(),
                })
            }
            _ => Err(RuntimeError {
                message: format!("extern '{}' not implemented", name),
            }),
        }
    }

    fn eval_binary(&self, op: &BinaryOp, l: Value, r: Value) -> Result<Value, RuntimeError> {
        match op {
            BinaryOp::Add => num_bin(l, r, |a, b| a + b),
            BinaryOp::Sub => num_bin(l, r, |a, b| a - b),
            BinaryOp::Mul => num_bin(l, r, |a, b| a * b),
            BinaryOp::Div => num_bin(l, r, |a, b| a / b),
            BinaryOp::Rem => int_bin(l, r, |a, b| a % b),
            BinaryOp::Eq => Ok(Value::Bool(values_equal(&l, &r))),
            BinaryOp::NotEq => Ok(Value::Bool(!values_equal(&l, &r))),
            BinaryOp::Lt => cmp_bin(l, r, |a, b| a < b),
            BinaryOp::LtEq => cmp_bin(l, r, |a, b| a <= b),
            BinaryOp::Gt => cmp_bin(l, r, |a, b| a > b),
            BinaryOp::GtEq => cmp_bin(l, r, |a, b| a >= b),
            BinaryOp::AndAnd => bool_bin(l, r, |a, b| a && b),
            BinaryOp::OrOr => bool_bin(l, r, |a, b| a || b),
        }
    }

    fn pattern_matches(&self, pattern: &Pattern, value: &Value) -> bool {
        match pattern {
            Pattern::Wildcard(_) => true,
            Pattern::Path(path, _) => {
                if path.len() >= 2 {
                    if let Value::Enum(e) = value {
                        return e.name == path[0] && e.variant == path[1];
                    }
                }
                false
            }
            Pattern::Struct { path, .. } => {
                if path.len() >= 2 {
                    if let Value::Enum(e) = value {
                        return e.name == path[0] && e.variant == path[1];
                    }
                }
                false
            }
        }
    }
}

fn parse_i64(text: &str) -> Result<i64, RuntimeError> {
    let cleaned = text.replace('_', "");
    cleaned.parse::<i64>().map_err(|_| RuntimeError {
        message: "invalid int literal".to_string(),
    })
}

fn parse_f64(text: &str) -> Result<f64, RuntimeError> {
    let cleaned = text.replace('_', "");
    cleaned.parse::<f64>().map_err(|_| RuntimeError {
        message: "invalid float literal".to_string(),
    })
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Tuple(xs), Value::Tuple(ys)) => {
            if xs.len() != ys.len() {
                return false;
            }
            for (x, y) in xs.iter().zip(ys.iter()) {
                if !values_equal(x, y) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

fn num_bin<F>(l: Value, r: Value, f: F) -> Result<Value, RuntimeError>
where
    F: Fn(f64, f64) -> f64,
{
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(f(a as f64, b as f64) as i64)),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(f(a, b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(f(a as f64, b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(f(a, b as f64))),
        _ => Err(RuntimeError {
            message: "numeric op requires numbers".to_string(),
        }),
    }
}

fn int_bin<F>(l: Value, r: Value, f: F) -> Result<Value, RuntimeError>
where
    F: Fn(i64, i64) -> i64,
{
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(f(a, b))),
        _ => Err(RuntimeError {
            message: "int op requires ints".to_string(),
        }),
    }
}

fn cmp_bin<F>(l: Value, r: Value, f: F) -> Result<Value, RuntimeError>
where
    F: Fn(f64, f64) -> bool,
{
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(f(a as f64, b as f64))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(f(a, b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Bool(f(a as f64, b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(f(a, b as f64))),
        _ => Err(RuntimeError {
            message: "comparison requires numbers".to_string(),
        }),
    }
}

fn bool_bin<F>(l: Value, r: Value, f: F) -> Result<Value, RuntimeError>
where
    F: Fn(bool, bool) -> bool,
{
    match (l, r) {
        (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(f(a, b))),
        _ => Err(RuntimeError {
            message: "boolean op requires bools".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::resolve::resolve_module;
    use crate::typecheck::typecheck_module;

    fn run(src: &str) -> Value {
        let tokens = Lexer::new(src).lex_all().unwrap();
        let module = Parser::new(tokens).parse_module().unwrap();
        resolve_module(&module).unwrap();
        typecheck_module(&module).unwrap();
        let program = build_program(&module);
        run_main(&program).unwrap()
    }

    #[test]
    fn eval_arithmetic() {
        let src = r#"
            fn main() -> I64 { let x: I64 = 4; x * 2 + 1 }
        "#;
        match run(src) {
            Value::Int(v) => assert_eq!(v, 9),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn eval_struct_field() {
        let src = r#"
            struct User { id: I64 }
            fn main() -> I64 { let u = User { id: 3 }; u.id }
        "#;
        match run(src) {
            Value::Int(v) => assert_eq!(v, 3),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn eval_match_enum() {
        let src = r#"
            enum Result { Ok(I64), Err(I64) }
            fn main() -> I64 {
                let r: Result = Result::Ok(1);
                match r { Result::Ok => 7, Result::Err => 2 }
            }
        "#;
        match run(src) {
            Value::Int(v) => assert_eq!(v, 7),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn eval_while_loop() {
        let src = r#"
            fn main() -> I64 {
                let mut i: I64 = 0;
                let mut sum: I64 = 0;
                while i < 3 { sum = sum + i; i = i + 1; };
                sum
            }
        "#;
        match run(src) {
            Value::Int(v) => assert_eq!(v, 3),
            _ => panic!("expected int"),
        }
    }
}
