use ir::{Ir, Value, VirtualRegister};
use std::collections::HashMap;

use crate::{
    arm::LowLevelArm,
    compiler::{Allocator, Compiler, Struct},
    ir::{self, BuiltInTypes, Condition},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ast {
    Program {
        elements: Vec<Ast>,
    },
    Function {
        name: Option<String>,
        // TODO: Change this to a Vec<Ast>
        args: Vec<String>,
        body: Vec<Ast>,
    },
    Struct {
        name: String,
        fields: Vec<Ast>,
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
    // TODO: Should I have both identifier and variable?
    // When should I have them?
    Identifier(String),
    Variable(String),
    String(String),
    True,
    False,
    StructCreation {
        name: String,
        fields: Vec<(String, Ast)>,
    },
    PropertyAccess {
        object: Box<Ast>,
        property: Box<Ast>,
    },
    Null,
}

impl Ast {
    pub fn compile<Alloc: Allocator>(&self, compiler: &mut Compiler<Alloc>) -> Ir {
        let mut ast_compiler = AstCompiler {
            ast: self.clone(),
            ir: Ir::new(),
            name: None,
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

        // println!("{:#?}", compiler);
        ast_compiler.compile()
    }

    pub fn nodes(&self) -> &Vec<Ast> {
        match self {
            Ast::Program { elements } => elements,
            _ => panic!("Only works on program"),
        }
    }

    pub fn name(&self) -> Option<String> {
        match self {
            Ast::Function { name, .. } => name.clone(),
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

// TODO: I have a global kind of compiler thing with functions
// I think structs should maybe go there?
// I also need to deal with namespacing
// Or maybe I should just put everything in this environment and then
// things in the top level are global/namespaced
// in fact, I don't think I will have "global",
// just namesapces
// With "core" being included by default

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

#[derive(Debug)]
pub struct AstCompiler<'a, Alloc: Allocator> {
    pub ast: Ast,
    pub ir: Ir,
    pub name: Option<String>,
    pub compiler: &'a mut Compiler<Alloc>,
    // This feels dumb and complicated. But my brain
    // won't let me think of a better way
    // I know there is one.
    pub context: Vec<Context>,
    pub current_context: Context,
    pub next_context: Context,
    pub environment_stack: Vec<Environment>,
}

impl<'a, Alloc: Allocator> AstCompiler<'a, Alloc> {
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
        self.context.push(self.current_context.clone());
        self.current_context = self.next_context.clone();
        let result = self.compile_to_ir(ast);
        self.next_context = self.current_context.clone();
        self.current_context = self.context.pop().unwrap();
        result
    }

    pub fn compile(&mut self) -> Ir {
        // TODO: Get rid of clone
        self.first_pass(&self.ast.clone());

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
                self.name = name.clone();
                for (index, arg) in args.iter().enumerate() {
                    let reg = self.ir.arg(index);
                    self.ir.register_argument(reg);
                    self.insert_variable(arg.clone(), VariableLocation::Register(reg));
                }

                for ast in body[..body.len().saturating_sub(1)].iter() {
                    self.call_compile(&Box::new(ast));
                }
                let last = body.last().unwrap_or(&Ast::Null);
                let return_value = self.call_compile(&Box::new(last));
                self.ir.ret(return_value);

                let lang = LowLevelArm::new();

                let error_fn_pointer = self.compiler.find_function("throw_error").unwrap();
                let error_fn_pointer = self
                    .compiler
                    .get_function_pointer(error_fn_pointer)
                    .unwrap();

                let compiler_ptr = self.compiler.get_compiler_ptr() as usize;

                let mut code = self.ir.compile(lang, error_fn_pointer, compiler_ptr);

                let function_pointer = self
                    .compiler
                    .upsert_function(name.as_deref(), &mut code, self.ir.num_locals)
                    .unwrap();

                code.share_label_info_debug(function_pointer);

                self.ir = old_ir;

                
                if let Some(value) = self.compile_closure(function_pointer) {
                    return value;
                }

                let function = self.ir.function(Value::Function(function_pointer));

                self.pop_environment();
                function
            }

            Ast::Struct { name: _, fields: _ } => {
                // TODO: This should probably return the struct value
                // A concept I don't yet have
                Value::Null
            }
            Ast::StructCreation { name, fields } => {
                let field_results = fields
                    .iter()
                    .map(|field| {
                        self.not_tail_position();
                        self.call_compile(&field.1)
                    })
                    .collect::<Vec<_>>();

                let (struct_id, struct_type) = self
                    .compiler
                    .get_struct(&name)
                    .unwrap_or_else(|| panic!("Struct not found {}", name));

                for field in fields.iter() {
                    let mut found = false;
                    for defined_field in struct_type.fields.iter() {
                        if &field.0 == defined_field {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        panic!("Struct field not defined {}", field.0);
                    }
                }

                let compiler_pointer_reg = self.ir.assign_new(self.compiler.get_compiler_ptr());

                let allocate_struct = self.compiler.find_function("allocate_struct").unwrap();
                let allocate_struct = self.compiler.get_function_pointer(allocate_struct).unwrap();
                let allocate_struct = self.ir.assign_new(allocate_struct);

                // Shift size left one so we can use it to mark
                let size_reg = self.ir.assign_new(struct_type.size() + 1);
                let stack_pointer = self.ir.get_stack_pointer_imm(0);
                // TODO: I need store the struct type here, so I know things about what data is here.

                let struct_ptr = self.ir.call_builtin(
                    allocate_struct.into(),
                    vec![compiler_pointer_reg.into(), size_reg.into(), stack_pointer],
                );

                // TODO: I want a better way to make clear the structure here.
                // Maybe I just make a struct and have a way of generating code
                // based on that struct?

                let struct_ptr = self.ir.assign_new(struct_ptr);
                let mut offset = 1;
                self.ir.heap_store_offset(
                    struct_ptr,
                    Value::SignedConstant(struct_id as isize),
                    offset,
                );
                offset += 1;

                for (i, reg) in field_results.iter().enumerate() {
                    self.ir.heap_store_offset(struct_ptr, *reg, offset + i);
                }

                self.ir
                    .tag(struct_ptr.into(), BuiltInTypes::Struct.get_tag())
            }
            Ast::PropertyAccess { object, property } => {
                let object = self.call_compile(object.as_ref());
                let object = self.ir.assign_new(object);
                let property = if let Ast::Identifier(name) = property.as_ref() {
                    name.clone()
                } else {
                    panic!("Expected identifier")
                };
                let constant_ptr = self.string_constant(property);
                let constant_ptr = self.ir.assign_new(constant_ptr);
                self.call_builtin("property_access", vec![object.into(), constant_ptr.into()])
            }
            Ast::If {
                condition,
                then,
                else_,
            } => {
                
                let condition = self.call_compile(&condition);

                let end_if_label = self.ir.label("end_if");

                let result_reg = self.ir.volatile_register();

                let then_label = self.ir.label("then");
                self.ir.jump_if(then_label, Condition::Equal, condition, Value::True);

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
            Ast::Call { name, args } => {
                if Some(name.clone()) == self.name {
                    if self.is_tail_position() {
                        return self.call_compile(&Ast::TailRecurse { args });
                    } else {
                        return self.call_compile(&Ast::Recurse { args });
                    }
                }

                let args: Vec<Value> = args
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

                
                // TODO: Should the arguments be evaluated first?
                // I think so, this will matter once I think about macros
                // though
                if self.compiler.is_inline_primitive_function(&name) {
                    return self.compile_inline_primitive_function(&name, args);
                }
                
                // TODO: This isn't they way to handle this
                // I am activing as if all closures are assign to a variable when they aren't.
                // Need to have negative test cases for this
                if let Some(function) = self.get_variable_current_env(&name) {
                    self.compile_closure_call(function, args)
                } else {
                    self.compile_standard_function_call(name, args)
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
            Ast::Null => Value::Null,
        }
    }

    fn compile_standard_function_call(&mut self, name: String, mut args: Vec<Value>) -> Value {
        // TODO: I shouldn't just assume the function will exist
        // unless I have a good plan for dealing with when it doesn't
        let function = self.compiler.reserve_function(name.as_str()).unwrap();
    
        let builtin = function.is_builtin;
        let needs_stack_pointer = function.needs_stack_pointer;
        if builtin {
            let pointer_reg = self.ir.volatile_register();
            let pointer: Value = self.compiler.get_compiler_ptr().into();
            self.ir.assign(pointer_reg, pointer);
            args.insert(0, pointer_reg.into());
        }
        if needs_stack_pointer {
            let stack_pointer_reg = self.ir.volatile_register();
            let stack_pointer = self.ir.get_stack_pointer_imm(0);
            self.ir.assign(stack_pointer_reg, stack_pointer);
            args.insert(1, stack_pointer);
        }
    
        let jump_table_pointer =
            self.compiler.get_jump_table_pointer(function).unwrap();
        let jump_table_point_reg =
            self.ir.assign_new(Value::Pointer(jump_table_pointer));
        let function_pointer = self.ir.load_from_memory(jump_table_point_reg.into(), 0);
    
        let function = self.ir.function(function_pointer);
        if builtin {
            // self.ir.breakpoint();
            self.ir.call_builtin(function, args)
        } else {
            self.ir.call(function, args)
        }
    }
    
    fn compile_closure_call(&mut self, function: VariableLocation, args: Vec<Value>) -> Value {
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
        self.ir
            .jump_if(call_function, Condition::NotEqual, tag, closure_tag);
        // I need to grab the function pointer
        // Closures are a pointer to a structure like this
        // struct Closure {
        //     function_pointer: *const u8,
        //     num_free_variables: usize,
        //     ree_variables: *const Value,
        // }
        let closure_register = self.ir.untag(closure_register.into());
        let function_pointer = self.ir.load_from_memory(closure_register, 0);
    
        self.ir.assign(function_register, function_pointer);
    
        // TODO: I need to fix how these are stored on the stack
    
        let num_free_variables = self.ir.load_from_memory(closure_register, 1);
        let num_free_variables =
            self.ir.tag(num_free_variables, BuiltInTypes::Int.get_tag());
        // for each variable I need to push them onto the stack after the prelude
        let loop_start = self.ir.label("loop_start");
        let counter = self.ir.volatile_register();
        // self.ir.breakpoint();
        self.ir.assign(counter, Value::SignedConstant(0));
        self.ir.write_label(loop_start);
        self.ir.jump_if(
            skip_load_function,
            Condition::GreaterThanOrEqual,
            counter,
            num_free_variables,
        );
        let free_variable_offset = self.ir.add(counter, Value::SignedConstant(3));
        let free_variable_offset =
            self.ir.mul(free_variable_offset, Value::SignedConstant(8));
        // TODO: This needs to change based on counter
        let free_variable_offset = self.ir.untag(free_variable_offset);
        let free_variable = self
            .ir
            .heap_load_with_reg_offset(closure_register, free_variable_offset);
    
        let free_variable_offset = self.ir.sub(num_free_variables, counter);
        let num_local = self.ir.load_from_memory(closure_register, 2);
        let num_local = self.ir.tag(num_local, BuiltInTypes::Int.get_tag());
        let free_variable_offset = self.ir.add(free_variable_offset, num_local);
        // // TODO: Make this better
        let free_variable_offset =
            self.ir.mul(free_variable_offset, Value::SignedConstant(-8));
        let free_variable_offset = self.ir.untag(free_variable_offset);
        let free_variable_slot_pointer = self.ir.get_stack_pointer_imm(2);
        self.ir.heap_store_with_reg_offset(
            free_variable_slot_pointer,
            free_variable,
            free_variable_offset,
        );
    
        let label = self.ir.label("increment_counter");
        self.ir.write_label(label);
        let counter_increment = self.ir.add(Value::SignedConstant(1), counter);
        self.ir.assign(counter, counter_increment);
    
        self.ir.jump(loop_start);
        self.ir.extend_register_life(num_free_variables);
        self.ir.extend_register_life(counter.into());
        self.ir.extend_register_life(closure_register);
        self.ir.write_label(call_function);
        self.ir.assign(function_register, &function);
        self.ir.write_label(skip_load_function);
        self.ir.call(function_register.into(), args)
    }
    
    fn compile_closure(&mut self, function_pointer: usize) -> Option<Value> {
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
            let label = self.ir.label("closure");
    
            // self.ir.breakpoint();
            // get a pointer to the start of the free variables on the stack
            let free_variable_pointer = self.ir.get_current_stack_position();
    
            self.ir.write_label(label);
            for free_variable in self.get_current_env().free_variables.clone().iter() {
                let variable = self
                    .get_variable(free_variable)
                    .unwrap_or_else(|| panic!("Can't find variable {}", free_variable));
                // we are now going to push these variables onto the stack
    
                match variable {
                    VariableLocation::Register(reg) => {
                        self.ir.push_to_stack(reg.into());
                    }
                    VariableLocation::Local(index) => {
                        let reg = self.ir.volatile_register();
                        self.ir.load_local(reg, index);
                        self.ir.push_to_stack(reg.into());
                    }
                    VariableLocation::FreeVariable(_) => {
                        panic!("We are trying to find this variable concretely and found a free variable")
                    }
                }
            }
            // load count of free variables
            let num_free = self.get_current_env().free_variables.len();
    
            let num_free = Value::SignedConstant(num_free as isize);
            let num_free_reg = self.ir.volatile_register();
            self.ir.assign(num_free_reg, num_free);
            // Call make_closure
            let make_closure = self.compiler.find_function("make_closure").unwrap();
            let make_closure = self.compiler.get_function_pointer(make_closure).unwrap();
            let make_closure_reg = self.ir.volatile_register();
            self.ir.assign(make_closure_reg, make_closure);
            let function_pointer_reg = self.ir.volatile_register();
            self.ir.assign(function_pointer_reg, function_pointer);
    
            let compiler_pointer_reg = self.ir.assign_new(self.compiler.get_compiler_ptr());
    
            let closure = self.ir.call(
                make_closure_reg.into(),
                vec![
                    compiler_pointer_reg.into(),
                    function_pointer_reg.into(),
                    num_free_reg.into(),
                    free_variable_pointer,
                ],
            );
            self.pop_environment();
            return Some(closure);
        }
        None
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
        self.environment_stack
            .last()
            .unwrap()
            .variables
            .get(name)
            .cloned()
    }

    fn get_variable_alloc_free_variable(&mut self, name: &str) -> VariableLocation {
        // TODO: Should walk the environment stack
        if let Some(variable) = self.environment_stack.last().unwrap().variables.get(name) {
            variable.clone()
        } else {
            let current_env = self.environment_stack.last_mut().unwrap();
            current_env.free_variables.push(name.to_string());
            let index = current_env.free_variables.len() - 1;
            current_env
                .variables
                .insert(name.to_string(), VariableLocation::FreeVariable(index));
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

    fn call_builtin(&mut self, arg: &str, args: Vec<Value>) -> Value {
        let mut args = args;
        let function = self.compiler.find_function(arg).unwrap();
        assert!(function.is_builtin);
        let function = self.compiler.get_function_pointer(function).unwrap();
        let function = self.ir.assign_new(function);
        let pointer_reg = self.ir.volatile_register();
        let pointer: Value = self.compiler.get_compiler_ptr().into();
        self.ir.assign(pointer_reg, pointer);
        args.insert(0, pointer_reg.into());
        self.ir.call(function.into(), args)
    }

    fn first_pass(&mut self, ast: &Ast) {
        match ast {
            Ast::Program { elements } => {
                for ast in elements.iter() {
                    self.first_pass(ast);
                }
            }
            Ast::Function {
                name,
                args: _,
                body: _,
            } => {
                if name.is_some() {
                    self.compiler
                        .reserve_function(name.as_deref().unwrap())
                        .unwrap();
                } else {
                    panic!("Why do we have a top level function without a name? Is that allowed?");
                }
            }
            Ast::Struct { name, fields } => {
                self.compiler.add_struct(Struct {
                    name: name.clone(),
                    fields: fields
                        .iter()
                        .map(|field| {
                            if let Ast::Identifier(name) = field {
                                name.clone()
                            } else {
                                panic!("Expected identifier got {:?}", field)
                            }
                        })
                        .collect(),
                });
            }
            Ast::Let(_, _) => todo!(),
            _ => {}
        }
    }
    
    fn compile_inline_primitive_function<>(&mut self, name: &str, args: Vec<Value>) -> Value {
        match name {
            "primitive_deref" => {
                // self.ir.breakpoint();
                let pointer = args[0];
                let untagged = self.ir.untag(pointer.into());
                // TODO: I need a raw add that doesn't check for tags
                let offset = self.ir.add(untagged, Value::RawValue(16));
                let reg = self.ir.volatile_register();
                self.ir.atomic_load(reg.into(), offset)
            },
            "primitive_reset!" => {
                let pointer = args[0];
                let untagged = self.ir.untag(pointer.into());
                // TODO: I need a raw add that doesn't check for tags
                let offset = self.ir.add(untagged, Value::RawValue(16));
                let value = args[1];
                self.call_builtin("gc_add_root", vec![pointer, value]);
                self.ir.atomic_store(offset, value.into());
                args[1]
            },
            "primitive_compare_and_swap!" => {
                // self.ir.breakpoint();
                let pointer = args[0];
                let untagged = self.ir.untag(pointer.into());
                let offset = self.ir.add(untagged, Value::RawValue(16));
                let expected = args[1];
                let new = args[2];
                let expected_and_result = self.ir.assign_new_force(expected);
                self.ir.compare_and_swap(expected_and_result.into(), new.into(), offset);
                // TODO: I should do a conditional move here instead of a jump
                let label = self.ir.label("compare_and_swap");
                let result = self.ir.assign_new(Value::True);
                self.ir.jump_if(label, Condition::Equal, expected_and_result, expected);
                self.ir.assign(result, Value::False);
                self.ir.write_label(label);
                result.into()
            }
            "primitive_breakpoint!" => {
                self.ir.breakpoint();
                Value::Null
            }
            _ => panic!("Unknown inline primitive function {}", name)
        }
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
