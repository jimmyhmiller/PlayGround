use std::{error::Error, time::Instant, mem, slice::from_raw_parts};
use bincode::{Encode, Decode, config::standard};
use crate::{compiler::Compiler, ir::BuiltInTypes, parser::Parser};

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
    ForeignFunction { name: String, pointer: usize },
    BuiltinFunction {name: String, pointer: usize},
    HeapPointer { pointer: usize },
    UserFunction { name: String, pointer: usize, len: usize },
    Label { label: String, function_pointer: usize, label_index: usize, label_location: usize },
}

trait Serialize {
    fn to_binary(&self) -> Vec<u8>;
    fn from_binary(data: &[u8]) -> Self;
}

impl<T : Encode + Decode> Serialize for T {
    fn to_binary(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, standard()).unwrap()
    }
    fn from_binary(data: &[u8]) -> T {
        let (data, _ ) = bincode::decode_from_slice(data, standard()).unwrap();
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


fn test_fib(compiler: &mut Compiler, n: u64) -> Result<(), Box<dyn Error>> {

    compiler.compile_ast(parser::fib())?;

    let time = Instant::now();

    let result1 = compiler.run_function("fib", vec![n as i32]);
    println!("Our time {:?}", time.elapsed());
    let time = Instant::now();
    let result2 = fib_rust(n as usize);
    println!("Rust time {:?}", time.elapsed());
    println!("{} {}", BuiltInTypes::untag(result1 as usize), result2);

    Ok(())
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
    let pointer = compiler.allocate(value).unwrap();
    let pointer = BuiltInTypes::Array.tag(pointer as isize) as usize;
    pointer
}

fn array_store(compiler: *mut Compiler, array: usize, index: usize, value: usize) -> usize {
    let compiler = unsafe { &mut *compiler };
    compiler.array_store(array, index, value).unwrap()
}

fn array_get(compiler: *mut Compiler, array: usize, index: usize) -> usize {
    let compiler = unsafe { &mut *compiler };
    compiler.array_get(array, index).unwrap()
}

fn make_closure(compiler: *mut Compiler, function: usize, num_free: usize, free_variable_pointer: usize) -> usize {
    let compiler = unsafe { &mut *compiler };
    let num_free = BuiltInTypes::untag(num_free);
    let free_variables = unsafe { from_raw_parts(free_variable_pointer as *const usize, num_free) };
    compiler.make_closure(function, free_variables).unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut compiler = Compiler::new();

    let heap_pointer = compiler.get_heap_pointer();

    debugger(Message {
        kind: "HeapPointer".to_string(),
        data: Data::HeapPointer { pointer: heap_pointer },
    });
    // Very inefficient way to do array stuff
    // but working
    compiler.add_builtin_function("print", ir::print_value as *const u8)?;
    compiler.add_builtin_function("allocate_array", allocate_array as *const u8)?;
    compiler.add_builtin_function("array_store", array_store as *const u8)?;
    compiler.add_builtin_function("array_get", array_get as *const u8)?;
    compiler.add_builtin_function("make_closure", make_closure as *const u8)?;

    let hello_ast = parse! {
        fn hello(x) {
            let array = allocate_array(16);
            array_store(array, 0, 42);
            array_store(array, x, "hello");
            let result = array_get(array, x)
            print(result)
        }

        fn count_down(x) {
            if x == 0 {
                0
            } else {
                count_down(x - 1)
            }
        }

        fn hello2() {
            let y = fn thing() {
                42
            }
            print(y)
            print(y())
        }

        fn hello_closure() {
            let x = 42;
            let y = fn closure_fn() {
                x
            }
            print(y())
        }

    };

    compiler.compile_ast(hello_ast)?;

    // let hello_result = compiler.run_function("hello", vec![1]);
    // compiler.print(hello_result as usize);
    // let countdown_result = compiler.run_function("count_down", vec![10000000]);
    // compiler.print(countdown_result as usize);

    // let hello2_result = compiler.run_function("hello2", vec![]);
    // compiler.print(hello2_result as usize);


    let hello_closure_result = compiler.run_function("hello_closure", vec![]);
    compiler.print(hello_closure_result as usize);

   

    // let top_level = parse!(
    //     let x = 1;
    //     let y = 2;
    //     let z = x + y;
    //     function print_z(z) {
    //         print(z)
    //     }
    //     print_z(z)
    // );

    // If i want something like the over to work, I need to
    // 1. Compile the whole thing as a function that I can call
    // 2. Return the location of funtions nested
    // For this I am ignoring closures for now
    // I need to make the Compiler deal with this rather
    // than doing everything piecemeal like I am now
    // I also need a place to store variables
    // Probably a concept of namespaces



    // TODO: As I'm compiling an ast to ir,
    // I need to separate out functions into their own units
    // of code.

  
    test_fib(&mut compiler, 32)?;
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
