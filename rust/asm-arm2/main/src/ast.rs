use ir::{Ir, Value, VirtualRegister};
use std::collections::HashMap;

use crate::ir::{self, Condition};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ast {
    Function {
        name: String,
        args: Vec<String>,
        body: Vec<Ast>,
    },
    If {
        condition: Box<Ast>,
        then: Box<Ast>,
        else_: Box<Ast>,
    },
    Condition {
        operator: Condition,
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Add {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Sub {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Recurse {
        args: Vec<Ast>,
    },
    Constant(i64),
    Variable(String),
}

impl Ast {
    pub fn compile(&self) -> Ir {
        let mut compiler = AstCompiler {
            ast: self.clone(),
            variables: HashMap::new(),
        };
        compiler.compile()
    }
}

pub struct AstCompiler {
    pub ast: Ast,
    pub variables: HashMap<String, VirtualRegister>,
}

impl AstCompiler {
    pub fn compile(&mut self) -> Ir {
        let mut ir = Ir::new();
        self.compile_to_ir(&Box::new(self.ast.clone()), &mut ir);
        ir
    }

    pub fn compile_to_ir(&mut self, ast: &Ast, ir: &mut Ir) -> Value {
        match ast.clone() {
            Ast::Function {
                name: _,
                args,
                body,
            } => {
                for (index, arg) in args.iter().enumerate() {
                    let reg = ir.arg(index);
                    self.variables.insert(arg.clone(), reg);
                }

                for ast in body[..body.len() - 1].iter() {
                    self.compile_to_ir(&Box::new(ast), ir);
                }
                let last = body.last().unwrap();
                let return_value = self.compile_to_ir(&Box::new(last), ir);
                ir.ret(return_value)
            }
            Ast::If {
                condition,
                then,
                else_,
            } => {
                // TODO: My condition system is a bit ugly
                // Mostly because I don't have booleans
                if let Ast::Condition {
                    operator,
                    left,
                    right,
                } = condition.as_ref()
                {
                    let a = self.compile_to_ir(left, ir);
                    let b = self.compile_to_ir(right, ir);
                    let end_if_label = ir.label("end_if");

                    let result_reg = ir.volatile_register();

                    let then_label = ir.label("then");
                    ir.jump_if(then_label, *operator, a, b);

                    let else_result = self.compile_to_ir(&else_, ir);
                    ir.assign(result_reg, else_result);
                    ir.jump(end_if_label);

                    ir.write_label(then_label);
                    let then_result = self.compile_to_ir(&then, ir);
                    ir.assign(result_reg, then_result);

                    ir.write_label(end_if_label);

                    result_reg.into()
                } else {
                    panic!("Expected condition")
                }
            }
            Ast::Add { left, right } => {
                let left = self.compile_to_ir(&left, ir);
                let right = self.compile_to_ir(&right, ir);
                ir.add(left, right)
            }
            Ast::Sub { left, right } => {
                let left = self.compile_to_ir(&left, ir);
                let right = self.compile_to_ir(&right, ir);
                ir.sub(left, right)
            }
            Ast::Recurse { args } => {
                let args = args
                    .iter()
                    .map(|arg| self.compile_to_ir(&Box::new(arg.clone()), ir))
                    .collect();
                ir.recurse(args)
            }
            Ast::Constant(n) => Value::SignedConstant(n as isize),
            Ast::Variable(name) => {
                let reg = self.variables.get(&name).unwrap();
                Value::Register(*reg)
            }
            Ast::Condition { .. } => {
                panic!("Condition should be handled by if")
            }
        }
    }
}

#[macro_export]
macro_rules! ast {
    ((fn $name:ident[]
        $body:tt
     )) => {
        Ast::Func{
            name: stringify!($name).to_string(),
            args: vec![],
            body: vec![Ast::Return(Box::new(ast!($body)))]
        }
    };
    ((fn $name:ident[$arg:ident]
        $body:tt
     )) => {
        Ast::Function {
            name: stringify!($name).to_string(),
            args: vec![stringify!($arg).to_string()],
            body: vec![ast!($body)]
        }
    };
    ((fn $name:ident[$arg1:ident $arg2:ident]
        $body:tt
     )) => {
        Ast::Func{
            name: stringify!($name).to_string(),
            args: vec![stringify!($arg1).to_string(), stringify!($arg2).to_string()],
            body: vec![ast!($body)]
        }
    };
    ((fn $name:ident[$arg1:ident $arg2:ident $arg3:ident]
        $body:tt
     )) => {
        Ast::Func{
            name: stringify!($name).to_string(),
            args: vec![stringify!($arg1).to_string(), stringify!($arg2).to_string(), stringify!($arg3).to_string()],
            body: vec![Ast::Return(Box::new(ast!($body)))]
        }
    };
    ((let [$name:tt $val:tt]
        $body:tt
    )) => {
        Ast::Do(vec![
            Ast::Let(stringify!($name).to_string(), Box::new(ast!($val))),
            ast!($body)]);
    };
    ((if ($cond:tt $arg:tt $val:tt)
        $result1:tt
        $result2:tt
    )) => {
        Ast::If{
            condition: Box::new(Ast::Condition {
                operator: ast!($cond),
                left: Box::new(ast!($arg)),
                right: Box::new(ast!($val))
            }),
            then: Box::new(ast!($result1)),
            else_: Box::new(ast!($result2))
        }
    };
    ((+ $arg1:tt $arg2:tt)) => {
        Ast::Add {
            left: Box::new(ast!($arg1)),
            right: Box::new(ast!($arg2))
        }
    };
    ((+ $arg1:tt $arg2:tt $($args:tt)+)) => {
            Ast::Add(Box::new(ast!($arg1)),
                     Box::new(ast!((+ $arg2 $($args)+))))
    };
    ((- $arg1:tt $arg2:tt)) => {
        Ast::Sub {
            left: Box::new(ast!($arg1)),
            right: Box::new(ast!($arg2))
        }
    };

    ((do $($arg1:tt)+)) => {
        Ast::Do(vec![$(ast!($arg1)),+])
    };
    ((return $arg:tt)) => {
        Ast::Return(Box::new(ast!($arg)))
    };
    (($f:ident $arg:tt)) => {
        Ast::Recurse {
            args: vec![ast!($arg)]
        }
    };
    (<=) => {
        Condition::LessThanOrEqual
    };
    (=) => {
        Condition::Equal
    };
    // (($f:ident $arg1:tt $arg2:tt)) => {
    //     Ast::Call2(stringify!($f).to_string(), Box::new(ast!($arg1)), Box::new(ast!($arg2)))
    // };
    // (($f:ident $arg1:tt $arg2:tt $arg3:tt)) => {
    //     Ast::Call3(stringify!($f).to_string(), Box::new(ast!($arg1)), Box::new(ast!($arg2)), Box::new(ast!($arg3)))
    // };
    ($int:literal) => {
        Ast::Constant($int)
    };
    ($var:ident) => {
        Ast::Variable(stringify!($var).to_string())
    }
}


pub fn fib() -> Ast {
    ast! {
        (fn fib [n]
            (if (<= n 1)
                n
                (+ (fib (- n 1)) (fib (- n 2)))))
    }
}




pub fn fib2() -> Ast {
    ast! {
        (fn fib [n]
            (if (= n 0)
                0
                (if (= n 1)
                    1
                    (+ (fib (- n 1)) (fib (- n 2))))))
    }
}
