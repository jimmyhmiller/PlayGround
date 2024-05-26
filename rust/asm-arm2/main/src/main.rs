use crate::{compiler::Compiler, ir::BuiltInTypes, parser::Parser};
use arm::LowLevelArm;
use asm::arm::{SP, X0, X1, X10, X2, X3, X4};
use bincode::{config::standard, Decode, Encode};
use compiler::StackMapDetails;
use std::{error::Error, mem, slice::from_raw_parts, time::Instant};

mod arm;
pub mod ast;
pub mod common;
pub mod compiler;
pub mod ir;
pub mod parser;

#[derive(Debug, Encode, Decode)]
pub struct Message {
    kind: String,
    data: Data,
}

// TODO: This should really live on the debugger side of things
#[derive(Debug, Encode, Decode)]
enum Data {
    ForeignFunction {
        name: String,
        pointer: usize,
    },
    BuiltinFunction {
        name: String,
        pointer: usize,
    },
    HeapSegmentPointer {
        pointer: usize,
    },
    UserFunction {
        name: String,
        pointer: usize,
        len: usize,
    },
    Label {
        label: String,
        function_pointer: usize,
        label_index: usize,
        label_location: usize,
    },
    StackMap {
        pc: usize,
        name: String,
        stack_map: Vec<(usize, StackMapDetails)>,
    },
}

trait Serialize {
    fn to_binary(&self) -> Vec<u8>;
    fn from_binary(data: &[u8]) -> Self;
}

impl<T: Encode + Decode> Serialize for T {
    fn to_binary(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, standard()).unwrap()
    }
    fn from_binary(data: &[u8]) -> T {
        let (data, _) = bincode::decode_from_slice(data, standard()).unwrap();
        data
    }
}

#[allow(unused)]
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn debugger_info(buffer: *const u8, length: usize) {
    // Hack to make sure this isn't inlined
    let x = 2;
}

pub fn debugger(message: Message) {
    // println!("{:?}", message);
    let message = message.to_binary();
    let ptr = message.as_ptr();
    let length = message.len();
    mem::forget(message);
    unsafe {
        debugger_info(ptr, length);
    }
    #[allow(unused)]
    let message = unsafe { from_raw_parts(ptr, length) };
    // Should make it is so we clean up this memory
}

fn fib_rust(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    fib_rust(n - 1) + fib_rust(n - 2)
}

// Do these need to be extern "C"?
fn allocate_array(compiler: *mut Compiler, value: usize) -> usize {
    let value = BuiltInTypes::untag(value);
    let compiler = unsafe { &mut *compiler };
    // TODO: Stack pointer should be passed in
    let pointer = compiler.allocate(value, 0, BuiltInTypes::Array).unwrap();
    
    BuiltInTypes::Array.tag(pointer as isize) as usize
}

fn allocate_struct(compiler: *mut Compiler, value: usize, stack_pointer: usize) -> usize {
    let value = BuiltInTypes::untag(value);
    let compiler = unsafe { &mut *compiler };
    
    compiler
        .allocate(value, stack_pointer, BuiltInTypes::Struct)
        .unwrap()
}

fn array_store(compiler: *mut Compiler, array: usize, index: usize, value: usize) -> usize {
    let compiler = unsafe { &mut *compiler };
    compiler.array_store(array, index, value).unwrap()
}

fn array_get(compiler: *mut Compiler, array: usize, index: usize) -> usize {
    let compiler = unsafe { &mut *compiler };
    compiler.array_get(array, index).unwrap()
}

fn make_closure(
    compiler: *mut Compiler,
    function: usize,
    num_free: usize,
    free_variable_pointer: usize,
) -> usize {
    let compiler = unsafe { &mut *compiler };
    let num_free = BuiltInTypes::untag(num_free);
    let free_variables = unsafe { from_raw_parts(free_variable_pointer as *const usize, num_free) };
    compiler.make_closure(function, free_variables).unwrap()
}

fn property_access(
    compiler: *mut Compiler,
    struct_pointer: usize,
    str_constant_ptr: usize,
) -> usize {
    let compiler = unsafe { &mut *compiler };
    compiler.property_access(struct_pointer, str_constant_ptr)
}

fn compile_trampoline(compiler: &mut Compiler) {
    let mut lang = LowLevelArm::new();
    // lang.breakpoint();
    lang.prelude(-2);

    // set SP to equal the first argument

    // Should I store or push?
    for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
        lang.store_on_stack(*reg, -((i + 2) as i32));
    }


    lang.mov_reg(X10, SP);
    lang.mov_reg(SP, X0);
    lang.push_to_stack(X10, 0);

    lang.mov_reg(X10, X1);
    lang.mov_reg(X0, X2);
    lang.mov_reg(X1, X3);
    lang.mov_reg(X2, X4);


    lang.call(X10);
    // lang.breakpoint();
    lang.pop_from_stack(X10, 0);
    lang.mov_reg(SP, X10);
    for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate().rev() {
        lang.load_from_stack(*reg, -((i + 2) as i32));
    }
    lang.epilogue(2);
    lang.ret();

    compiler
        .add_function("trampoline", &lang.compile_directly())
        .unwrap();
    let function = compiler.get_function_by_name_mut("trampoline").unwrap();
    function.is_builtin = true;
}



fn main() -> Result<(), Box<dyn Error>> {
    let mut compiler = Compiler::new();

    compile_trampoline(&mut compiler);

    let heap_pointer = compiler.get_heap_pointer();

    debugger(Message {
        kind: "HeapPointer".to_string(),
        data: Data::HeapSegmentPointer {
            pointer: heap_pointer,
        },
    });
    // Very inefficient way to do array stuff
    // but working
    compiler.add_builtin_function("println", ir::println_value as *const u8)?;
    compiler.add_builtin_function("print", ir::print_value as *const u8)?;
    compiler.add_builtin_function("allocate_array", allocate_array as *const u8)?;
    compiler.add_builtin_function("allocate_struct", allocate_struct as *const u8)?;
    compiler.add_builtin_function("array_store", array_store as *const u8)?;
    compiler.add_builtin_function("array_get", array_get as *const u8)?;
    compiler.add_builtin_function("make_closure", make_closure as *const u8)?;
    compiler.add_builtin_function("property_access", property_access as *const u8)?;
    // compiler.add_builtin_function("gc", gc as *const u8)?;

    // TODO: getting no free registers in MainThread!
    let hello_ast = Parser::from_file(
        "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/asm-arm2/main/resources/examples.bg",
    )?;

    // println!("{:#?}", hello_ast);

    compiler.compile_ast(hello_ast)?;

    compiler.check_functions();

    // let hello_result = compiler.run_function("hello", vec![1]);
    // compiler.print(hello_result as usize);
    // let countdown_result = compiler.run_function("count_down", vec![10000000]);
    // compiler.print(countdown_result as usize);

    // let hello2_result = compiler.run_function("hello2", vec![]);
    // compiler.print(hello2_result as usize);

    // let time = Instant::now();
    // let result = compiler.run_function("mainThread", vec![10]);
    // println!("Our time {:?}", time.elapsed());
    // compiler.println(result as usize);

    let time = Instant::now();
    let result = compiler.run_function("mainThread", vec![21]);
    println!("Our time {:?}", time.elapsed());
    compiler.println(result as usize);

    // let result = compiler.run_function("simpleFunctionWithLocals", vec![]);
    // println!("Our time {:?}", time.elapsed());
    // compiler.println(result as usize);

    // let time = Instant::now();
    // let result = compiler.run_function("testGcNested", vec![]);
    // println!("Our time {:?}", time.elapsed());
    // compiler.println(result as usize);

    // TODO: As I'm compiling an ast to ir,
    // I need to separate out functions into their own units
    // of code.

    // let n = 32;
    // let time = Instant::now();
    // let result1 = compiler.run_function("fib", vec![n]);
    // println!("Our time {:?}", time.elapsed());
    // let time = Instant::now();
    // let result2 = fib_rust(n as usize);
    // println!("Rust time {:?}", time.elapsed());
    // println!("{} {}", BuiltInTypes::untag(result1 as usize), result2);
    Ok(())
}

// TODO:
// Runtime?
//     Function in our language names and calling them.
//     Built-in functions
//     Stack
//     Heap
// Parser
// Debugging

// TODO: Make variables
// Should we allow reassignment?
// Need to add guards against type errors
// Need to do some looping construct
// Need to do load and store in the actual
// compiler instead of cheating
// Think about protocols
// Think about how to implementing interesting
// data structures in the language itself
// Consider checked and uncheck stuff

// Bugs:
