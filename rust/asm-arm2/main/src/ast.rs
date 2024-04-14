use ir::{Ir, Value, VirtualRegister};
use std::collections::HashMap;

use crate::{
    compiler::Compiler,
    ir::{self, BuiltInTypes, Condition},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ast {
    Program {
        elements: Vec<Ast>,
    },
    Function {
        name: String,
        // TODO: Change this to a Vec<Ast>
        args: Vec<String>,
        body: Vec<Ast>,
    },
    Struct {
        name: String,
        fields: Vec<Ast>
    },
    If {
        condition: Box<Ast>,
        then: Vec<Ast>,
        else_: Vec<Ast>,
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
    Mul {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Div {
        left: Box<Ast>,
        right: Box<Ast>,
    },
    Recurse {
        args: Vec<Ast>,
    },
    TailRecurse {
        args: Vec<Ast>,
    },
    Call {
        name: String,
        args: Vec<Ast>,
    },
    Let(Box<Ast>, Box<Ast>),
    NumberLiteral(i64),
    Identifier(String),
    Variable(String),
    String(String),
    True,
    False,

}

impl Ast {
    pub fn compile(&self, compiler: &mut Compiler) -> Ir {
        let mut compiler = AstCompiler {
            ast: self.clone(),
            ir: Ir::new(),
            name: "".to_string(),
            compiler,
            context: vec![],
            current_context: Context {
                tail_position: true,
                in_function: false,
            },
            next_context: Context {
                tail_position: true,
                in_function: false,
            },
            environment_stack: vec![Environment::new()],
        };
        compiler.compile()
    }

    pub fn nodes(&self) -> &Vec<Ast> {
        match self {
            Ast::Program { elements } => elements,
            _ => panic!("Only works on program"),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Ast::Function { name, .. } => name.as_str(),
            _ => panic!("Only works on function"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum VariableLocation {
    Register(VirtualRegister),
    Local(usize),
    FreeVariable(usize),
}

impl From<&VariableLocation> for Value {
    fn from(location: &VariableLocation) -> Self {
        match location {
            VariableLocation::Register(reg) => Value::Register(*reg),
            VariableLocation::Local(index) => Value::Local(*index),
            VariableLocation::FreeVariable(index) => Value::FreeVariable(*index),

        }
    }
}

#[derive(Debug, Clone)]
pub struct Context {
    pub tail_position: bool,
    pub in_function: bool,
}

#[derive(Debug, Clone)]
pub struct Environment {
    pub local_variables: Vec<String>,
    pub variables: HashMap<String, VariableLocation>,
    pub free_variables: Vec<String>,
}

impl Environment {
    fn new() -> Self {
        Environment { 
            local_variables: vec![],
            variables: HashMap::new(),
            free_variables: vec![],
         }
    }
}

pub struct AstCompiler<'a> {
    pub ast: Ast,
    pub ir: Ir,
    pub name: String,
    pub compiler: &'a mut Compiler,
    // This feels dumb and complicated. But my brain
    // won't let me think of a better way
    // I know there is one.
    pub context: Vec<Context>,
    pub current_context: Context,
    pub next_context: Context,
    pub environment_stack: Vec<Environment>,
}

impl<'a> AstCompiler<'a> {

    pub fn tail_position(&mut self) {
        self.next_context.tail_position = true;
    }

    pub fn not_tail_position(&mut self) {
        self.next_context.tail_position = false;
    }

    pub fn in_function(&mut self) {
        self.next_context.in_function = true;
    }

    pub fn is_tail_position(&self) -> bool {
        self.current_context.tail_position
    }

    pub fn call_compile(&mut self, ast: &Ast) -> Value {
        // Does this even work?
        self.context.push(self.current_context.clone());
        self.current_context = self.next_context.clone();
        let result = self.compile_to_ir(ast);
        self.next_context = self.current_context.clone();
        self.current_context = self.context.pop().unwrap();
        result
    }

    pub fn compile(&mut self) -> Ir {
        self.tail_position();
        self.call_compile(&Box::new(self.ast.clone()));
        let mut ir = Ir::new();
        std::mem::swap(&mut ir, &mut self.ir);
        ir
    }
    pub fn compile_to_ir(&mut self, ast: &Ast) -> Value {
        match ast.clone() {
            Ast::Program { elements } => {
                let mut last = Value::SignedConstant(0);
                for ast in elements.iter() {
                    self.tail_position();
                    last = self.call_compile(ast);
                }
                last
            }
            Ast::Function { name, args, body } => {
                self.create_new_environment();
                let old_ir = std::mem::replace(&mut self.ir, Ir::new());
                assert!(!name.is_empty());
                // self.ir.breakpoint();
                self.name = name.clone();
                for (index, arg) in args.iter().enumerate() {
                    let reg = self.ir.arg(index);
                    self.ir.register_argument(reg);
                    self.insert_variable(arg.clone(), VariableLocation::Register(reg));
                }

                for ast in body[..body.len().saturating_sub(1)].iter() {
                    self.call_compile(&Box::new(ast));
                }
                // TODO: Need some concept of nil I think
                let last = body.last().unwrap();
                let return_value = self.call_compile(&Box::new(last));
                self.ir.ret(return_value);


                let mut code = self.ir.compile(&name);
                let function_pointer = self.compiler.upsert_function(&name, &code.compile_to_bytes()).unwrap();

                code.share_label_info_debug(function_pointer);

                self.ir = old_ir;

                if self.has_free_variables() {
                    // When I get those free variables, I'd need to 
                    // make sure that the variables they refer to are
                    // heap allocated. How am I going to do that?
                    // I actually probably need to think about this more
                    // If they are already heap allocated, then I just 
                    // store the pointer. If they are immutable variables,
                    // I just take the value
                    // If they are mutable, then I'd need to heap allocate
                    // but maybe I just heap allocate all mutable variables?
                    // What about functions that change overtime?
                    // Not 100% sure about all of this

                    for free_variable in self.get_current_env().free_variables.clone().iter(){
                        let variable = self.get_variable(free_variable).unwrap();
                        // we are now going to push these variables onto the stack
                        
                        match variable {
                            VariableLocation::Register(reg) => {
                                let reg = self.ir.volatile_register();
                                self.ir.assign(reg, reg);
                                self.ir.push_to_stack(reg);
                            }
                            VariableLocation::Local(index) => {
                                let reg = self.ir.volatile_register();
                                self.ir.load_local(reg, index);
                                self.ir.push_to_stack(reg);
                            }
                            VariableLocation::FreeVariable(index) => {
                                panic!("We are trying to find this variable concretely and found a free variable")
                            }
                        }
                        // load count of free variables
                        let num_free = self.get_current_env().free_variables.len();

                        // get a pointer to the start of the free variables on the stack
                        let free_variable_pointer = self.ir.get_stack_pointer_imm(num_free as isize + 1);
                        

                        let num_free = Value::SignedConstant(num_free as isize);
                        let num_free_reg = self.ir.volatile_register();
                        self.ir.assign(num_free_reg, num_free);
                        let num_free_reg = self.ir.tag(num_free_reg.into(), BuiltInTypes::Int.get_tag());
                        // Call make_closure
                        let make_closure = self.compiler.find_function("make_closure").unwrap();
                        let make_closure = self.compiler.get_function_pointer(make_closure).unwrap();
                        let make_closure_reg = self.ir.volatile_register();
                        self.ir.assign(make_closure_reg, make_closure);
                        let function_pointer_reg = self.ir.volatile_register();
                        self.ir.assign(function_pointer_reg, function_pointer);

                        let compiler_pointer_reg = self.ir.volatile_register();
                        let compiler_pointer: Value = self.compiler.get_compiler_ptr().into();
                        self.ir.assign(compiler_pointer_reg, compiler_pointer);

                        let closure = self.ir.call(
                            make_closure_reg.into(),
                            vec![
                                compiler_pointer_reg.into(),
                                function_pointer_reg.into(),
                                num_free_reg.into(),
                                free_variable_pointer.into(),
                            ]
                        );
                        self.pop_environment();
                        return closure;
                        
                    }
                }

                let function = self.ir.function(function_pointer);

                self.pop_environment();
                function
            }

            Ast::Struct { name, fields } => {
                // TODO: Do something with the struct
                Value::Null
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
                    let a = self.call_compile(left);
                    let b = self.call_compile(right);
                    let end_if_label = self.ir.label("end_if");

                    let result_reg = self.ir.volatile_register();

                    let then_label = self.ir.label("then");
                    self.ir.jump_if(then_label, *operator, a, b);

                    let mut else_result = Value::SignedConstant(0);
                    for ast in else_.iter() {
                        else_result = self.call_compile(&Box::new(ast));
                    }
                    self.ir.assign(result_reg, else_result);
                    self.ir.jump(end_if_label);

                    self.ir.write_label(then_label);

                    let mut then_result = Value::SignedConstant(0);
                    for ast in then.iter() {
                        then_result = self.call_compile(&Box::new(ast));
                    }
                    self.ir.assign(result_reg, then_result);

                    self.ir.write_label(end_if_label);

                    result_reg.into()
                } else {
                    panic!("Expected condition")
                }
            }
            Ast::Add { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.add(left, right)
            }
            Ast::Sub { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.sub(left, right)
            }
            Ast::Mul { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.mul(left, right)
            }
            Ast::Div { left, right } => {
                self.not_tail_position();
                let left = self.call_compile(&left);
                self.not_tail_position();
                let right = self.call_compile(&right);
                self.ir.div(left, right)
            }
            Ast::Recurse { args } | Ast::TailRecurse { args } => {
                let args = args
                    .iter()
                    .map(|arg| {
                        self.not_tail_position();
                        self.call_compile(&Box::new(arg.clone()))
                    })
                    .collect();
                if matches!(ast, Ast::TailRecurse { .. }) {
                    self.ir.tail_recurse(args)
                } else {
                    self.ir.recurse(args)
                }
            }
            // TODO: Have an idea of built in functions to call
            // These functions should be passed as the first argument context
            // (maybe the compiler struct), so they can do things with the system
            // like heap allocate and such
            Ast::Call { name, args } => {
                if name == self.name {
                    // TODO: I'm guessing I can have tail recursive closures that I will need to deal with
                    if self.is_tail_position() {
                        return self.call_compile(&Ast::TailRecurse { args });
                    } else {
                        return self.call_compile(&Ast::Recurse { args });
                    }
                }

                let mut args: Vec<Value> = args
                    .iter()
                    .map(|arg| {
                        self.not_tail_position();
                        let value = self.call_compile(&Box::new(arg.clone()));
                        match value {
                            Value::Register(_) => value,
                            _ => {
                                let reg = self.ir.volatile_register();
                                self.ir.assign(reg, value);
                                reg.into()
                            }
                        }
                    })
                    .collect();

                if let Some(function) = self.get_variable_current_env(&name) {
                    // TODO: Right now, this would only find variables in the current environment
                    // I also need to deal wiht functions vs closures
                    let function_register = self.ir.volatile_register();


                    let closure_register = self.ir.volatile_register();
                    self.ir.assign(closure_register, &function);
                    // Check if the tag is a closure
                    let tag = self.ir.get_tag(closure_register.into());
                    let closure_tag = BuiltInTypes::Closure.get_tag();
                    let closure_tag = Value::RawValue(closure_tag as usize);
                    let call_function = self.ir.label("call_function");
                    let skip_load_function = self.ir.label("skip_load_function");
                    // TODO: It might be better to change the layout of these jumps
                    // so that the non-closure case is the fall through
                    // I just have to think about the correct way to do that
                    self.ir.jump_if(call_function, Condition::NotEqual, tag, closure_tag);
                    // I need to grab the function pointer
                    // Closures are a pointer to a structure like this
                    // struct Closure {
                    //     function_pointer: *const u8,
                    //     num_free_variables: usize,
                    //     ree_variables: *const Value,
                    // }
                    let closure_register = self.ir.untag(closure_register.into());
                    // TODO:
                    // I'm currently not getting the correct data. Need to figure out why.
                    // probably should build a heap viewer
                    let function_pointer = self.ir.load_from_memory(closure_register, 0);
                    self.ir.assign(function_register, function_pointer);
                    let num_free_variables = self.ir.load_from_memory(closure_register, 8);
                    // for each variable I need to push them onto the stack after the prelude
                    let loop_start = self.ir.label("loop_start");
                    let counter = self.ir.volatile_register();
                    self.ir.assign(counter, Value::SignedConstant(0));
                    self.ir.write_label(loop_start);
                    self.ir.jump_if(skip_load_function, Condition::GreaterThanOrEqual, counter, num_free_variables);
                    // TODO: This needs to change based on counter
                    let free_variable = self.ir.load_from_memory(closure_register, 16);
                    let offset = self.ir.volatile_register();
                    self.ir.assign(offset, counter);
                    let free_variable_offset = self.ir.add(offset, 0);
                    let free_variable_slot_pointer = self.ir.get_stack_pointer(free_variable_offset);
                    // TODO: Hardcoded 4 for prelude. Need to actually figure out that value correctly
                    let free_variable_slot_pointer = self.ir.sub(free_variable_slot_pointer, 4);
                    self.ir.heap_store(free_variable_slot_pointer, free_variable);
                    let counter_increment = self.ir.add(1, counter);
                    self.ir.assign(counter, counter_increment);
                    self.ir.jump(loop_start);
                    self.ir.write_label(call_function);
                    self.ir.assign(function_register, &function);
                    self.ir.write_label(skip_load_function);
                    self.ir.call(function_register.into(), args)
                } else {
                    // TODO: I shouldn't just assume the function will exist
                    // unless I have a good plan for dealing with when it doesn't
                    let function = self.compiler.reserve_function(name.as_str()).unwrap();

                    if function.is_builtin {
                        let pointer_reg = self.ir.volatile_register();
                        let pointer: Value = self.compiler.get_compiler_ptr().into();
                        self.ir.assign(pointer_reg, pointer);
                        args.insert(0, pointer_reg.into());
                    }
                    // TODO: Do an indirect call via jump table
                    let function_pointer = self.compiler.get_function_pointer(function).unwrap();
    
                    let function = self.ir.function(function_pointer);
                    self.ir.call(function, args)
                }
               
            }
            Ast::NumberLiteral(n) => Value::SignedConstant(n as isize),
            Ast::Variable(name) => {
                let reg = &self.get_variable_alloc_free_variable(&name);
                reg.into()
            }
            Ast::Identifier(_) => {
                todo!()
            }
            Ast::Let(name, value) => {
                if let Ast::Variable(name) = name.as_ref() {
                    self.not_tail_position();
                    let value = self.call_compile(&value);
                    self.not_tail_position();
                    let reg = self.ir.volatile_register();
                    self.ir.assign(reg, value);
                    let local_index = self.find_or_insert_local(name);
                    self.ir.store_local(local_index, reg);
                    self.insert_variable(name.to_string(), VariableLocation::Local(local_index));
                    reg.into()
                } else {
                    panic!("Expected variable")
                }
            }
            Ast::Condition {
                operator,
                left,
                right,
            } => {

                self.not_tail_position();
                let a = self.call_compile(&left);
                self.not_tail_position();
                let b = self.call_compile(&right);
                self.ir.compare(a, b, operator)
            }
            Ast::String(str) => {
                let constant_ptr = self.string_constant(str);
                self.ir.load_string_constant(constant_ptr)
            }
            Ast::True => Value::True,
            Ast::False => Value::False,
        }
    }

    fn find_or_insert_local(&mut self, name: &str) -> usize {
        let current_env = self.environment_stack.last_mut().unwrap();
        if let Some(index) = current_env.local_variables.iter().position(|n| n == name) {
            index
        } else {
            current_env.local_variables.push(name.to_string());
            current_env.local_variables.len() - 1
        }
    }

    fn insert_variable(&mut self, clone: String, reg: VariableLocation) {
        let current_env = self.environment_stack.last_mut().unwrap();
        current_env.variables.insert(clone, reg);
    }

    // TODO: Need to walk the environment stack
    fn get_variable_current_env(&self, name: &str) -> Option<VariableLocation> {
        self.environment_stack.last().unwrap().variables.get(name).cloned()
    }

    fn get_variable_alloc_free_variable(&mut self, name: &str) -> VariableLocation {
        // TODO: Should walk the environment stack
        if let Some(variable) = self.environment_stack.last().unwrap().variables.get(name) {
            variable.clone()
        } else {
            let current_env = self.environment_stack.last_mut().unwrap();
            current_env.free_variables.push(name.to_string());
            let index = current_env.free_variables.len() - 1;
            current_env.variables.insert(name.to_string(), VariableLocation::FreeVariable(index));
            let current_env = self.environment_stack.last().unwrap();
            current_env.variables.get(name).unwrap().clone()
        }
    }

    fn get_variable(&self, name: &str) -> Option<VariableLocation> {
        for env in self.environment_stack.iter().rev() {
            if let Some(variable) = env.variables.get(name) {
                if !matches!(&variable, VariableLocation::FreeVariable(_)) {
                    return Some(variable.clone());
                }
            }
        }
        None
    }

    fn string_constant(&mut self, str: String) -> Value {
        self.compiler.add_string(ir::StringValue { str })
    }

    fn create_new_environment(&mut self) {
        self.environment_stack.push(Environment::new());
    }

    fn pop_environment(&mut self) {
        self.environment_stack.pop();
    }

    fn get_current_env(&self) -> &Environment {
        self.environment_stack.last().unwrap()
    }

    fn has_free_variables(&self) -> bool {
        let current_env = self.get_current_env();
        !current_env.free_variables.is_empty()
    }

   
}

impl From<i64> for Ast {
    fn from(val: i64) -> Self {
        Ast::NumberLiteral(val)
    }
}

impl From<&'static str> for Ast {
    fn from(val: &'static str) -> Self {
        Ast::String(val.to_string())
    }
}

#[macro_export]
macro_rules! ast {
    ((fn $name:ident[]
        $body:tt
     )) => {
        Ast::Function {
            name: stringify!($name).to_string(),
            args: vec![],
            body: vec![ast!($body)]
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
        Ast::Function {
            name: stringify!($name).to_string(),
            args: vec![stringify!($arg1).to_string(), stringify!($arg2).to_string()],
            body: vec![ast!($body)]
        }
    };
    ((fn $name:ident[$arg1:ident $arg2:ident $arg3:ident]
        $body:tt
     )) => {
        Ast::Function {
            name: stringify!($name).to_string(),
            args: vec![stringify!($arg1).to_string(), stringify!($arg2).to_string(), stringify!($arg3).to_string()],
            body: vec![ast!($body)]
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
        Ast::If {
            condition: Box::new(Ast::Condition {
                operator: ast!($cond),
                left: Box::new(ast!($arg)),
                right: Box::new(ast!($val))
            }),
            then: vec![ast!($result1)],
            else_: vec![ast!($result2)]
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
    (($f:ident)) => {
        Ast::Call {
            name: stringify!($f).to_string(),
            args: vec![]
        }
    };
    (($f:ident $arg:tt)) => {
        Ast::Call {
            name: stringify!($f).to_string(),
            args: vec![ast!($arg)]
        }
    };
    (($f:ident $($arg:tt)+)) => {
        Ast::Call {
            name: stringify!($f).to_string(),
            args: vec![$(ast!($arg)),+]
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
    ($lit:literal) => {
        $lit.into()
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

pub fn hello_world() -> Ast {
    ast! {
        (fn hello []
            (print "Hello World!"))
    }
}

pub fn hello_world2() -> Ast {
    ast! {
        (fn hello []
            (test))
    }
}




#[cfg(test)]
fn check_tail_recursion(ast: Ast) -> bool {
    let ir = ast.compile(&mut Compiler::new());
    ir.instructions.iter().any(|instruction| {
        matches!(instruction, ir::Instruction::TailRecurse { .. })
    })
}

#[test]
fn tail_position() {
    

    // simple case
    let my_ast = ast! {
        (fn tail_recursive [n]
            (tail_recursive (- n 1)))
    };

    assert!(check_tail_recursion(my_ast));

    let my_ast = ast! {
        (fn tail_recursive [n]
            (if (= n 0)
                0
                (tail_recursive (- n 1))))
    };

    assert!(check_tail_recursion(my_ast));

    let my_ast = ast! {
        (fn tail_recursive [n]
            (if (= n 0)
                0
                (if (= n 1)
                    1
                    (tail_recursive (- n 1)))))
    };

    assert!(check_tail_recursion(my_ast));

    // my ast macro didn't let me write this correctly
    // but I just want the check the general syntactic form
    let reduce = ast! {
        (fn reduce [f acc list]
            (if (= list list)
                acc
                (reduce f (f acc (head list)) (tail list))))
    };

    assert!(check_tail_recursion(reduce));
    

    // not tail recursive
    let my_ast = ast! {
        (fn not_tail_recursive [n]
            (if (= n 0)
                0
                (if (= n 1)
                    1
                    (+ (not_tail_recursive (- n 1)) (not_tail_recursive (- n 2))))))

    };

    assert!(!check_tail_recursion(my_ast));

    // fib is not tail
    let my_ast = fib();
    assert!(!check_tail_recursion(my_ast));
}