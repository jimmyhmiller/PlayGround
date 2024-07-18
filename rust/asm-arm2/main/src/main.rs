#![allow(clippy::match_like_matches_macro)]
use crate::{compiler::Compiler, ir::BuiltInTypes, parser::Parser};
use arm::LowLevelArm;
use asm::arm::{SP, X0, X1, X10, X2, X3, X4};
use bincode::{config::standard, Decode, Encode};
use clap::{command, Parser as ClapParser};
use compiler::{Allocator, DefaultPrinter, Printer, StackMapDetails, TestPrinter};
#[allow(unused)]
use gc::{
    compacting::CompactingHeap, simple_generation::SimpleGeneration,
    simple_mark_and_sweep::SimpleMarkSweepHeap,
};

use std::{error::Error, mem, slice::from_raw_parts, sync::RwLock, time::Instant};

mod arm;
pub mod ast;
pub mod common;
pub mod compiler;
mod gc;
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
    Allocate {
        bytes: usize,
        stack_pointer: usize,
        kind: String,
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
/// # Safety
///
/// This does nothing
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

pub extern "C" fn println_value<Alloc: Allocator>(
    compiler: *const RwLock<Compiler<Alloc>>,
    value: usize,
) -> usize {
    let compiler = unsafe { & *compiler };
    let mut compiler = compiler.write().unwrap();
    compiler.println(value);
    0b111
}

pub extern "C" fn print_value<Alloc: Allocator>(
    compiler: *const RwLock<Compiler<Alloc>>,
    value: usize,
) -> usize {
    let compiler = unsafe { & *compiler };
    let mut compiler = compiler.write().unwrap();
    compiler.print(value);
    0b111
}


extern "C" fn allocate_struct<Alloc: Allocator>(
    compiler: *const RwLock<Compiler<Alloc>>,
    value: usize,
    stack_pointer: usize,
) -> usize {
    let value = BuiltInTypes::untag(value);
    let compiler = unsafe { & *compiler };

    let mut compiler = compiler.write().unwrap();

    compiler
        .allocate(value, stack_pointer, BuiltInTypes::Struct)
        .unwrap()
}

extern "C" fn make_closure<Alloc: Allocator>(
    compiler: *const RwLock<Compiler<Alloc>>,
    function: usize,
    num_free: usize,
    free_variable_pointer: usize,
) -> usize {
    let compiler = unsafe { & *compiler };
    let mut compiler = compiler.write().unwrap();
    let num_free = BuiltInTypes::untag(num_free);
    let free_variable_pointer = free_variable_pointer as *const usize;
    let start = unsafe { free_variable_pointer.sub(num_free - 1) };
    let free_variables = unsafe { from_raw_parts(start, num_free) };
    compiler.make_closure(function, free_variables).unwrap()
}

extern "C" fn property_access<Alloc: Allocator>(
    compiler: *const RwLock<Compiler<Alloc>>,
    struct_pointer: usize,
    str_constant_ptr: usize,
) -> usize {
    let compiler = unsafe { & *compiler };
    let compiler = compiler.read().unwrap();
    compiler.property_access(struct_pointer, str_constant_ptr)
}

pub extern "C" fn throw_error<Alloc: Allocator>(
    _compiler: *const RwLock<Compiler<Alloc>>,
    _stack_pointer: usize,
) -> usize {
    // let compiler = unsafe { &mut *compiler };
    panic!("Error!");
}

pub extern "C" fn gc<Alloc: Allocator>(
    compiler: *const RwLock<Compiler<Alloc>>,
    stack_pointer: usize,
) -> usize {
    let compiler = unsafe { & *compiler };
    let mut compiler = compiler.write().unwrap();
    compiler.gc(stack_pointer);
    BuiltInTypes::null_value() as usize
}

pub extern "C" fn gc_add_root<Alloc: Allocator>(
    compiler: *const RwLock<Compiler<Alloc>>,
    old: usize,
    young: usize,
) -> usize {
    let compiler = unsafe { & *compiler };
    let mut compiler = compiler.write().unwrap();
    compiler.gc_add_root(old, young);
    BuiltInTypes::null_value() as usize
}

pub extern "C" fn new_thread<Alloc: Allocator>(
    compiler: *const RwLock<Compiler<Alloc>>,
    function: usize,
) -> usize {
    let compiler = unsafe { & *compiler };
    let mut compiler = compiler.write().unwrap();
    compiler.new_thread(function);
    BuiltInTypes::null_value() as usize
}

fn compile_trampoline<Alloc: Allocator>(compiler: &mut Compiler<Alloc>) {
    let mut lang = LowLevelArm::new();
    // lang.breakpoint();
    lang.prelude(-2);

    // Should I store or push?
    for (i, reg) in lang.canonical_volatile_registers.clone().iter().enumerate() {
        lang.store_on_stack(*reg, -((i + 2) as i32));
    }

    lang.mov_reg(X10, SP);
    lang.mov_reg(SP, X0);
    lang.push_to_stack(X10);

    lang.mov_reg(X10, X1);
    lang.mov_reg(X0, X2);
    lang.mov_reg(X1, X3);
    lang.mov_reg(X2, X4);

    lang.call(X10);
    // lang.breakpoint();
    lang.pop_from_stack(X10, 0);
    lang.mov_reg(SP, X10);
    for (i, reg) in lang
        .canonical_volatile_registers
        .clone()
        .iter()
        .enumerate()
        .rev()
    {
        lang.load_from_stack(*reg, -((i + 2) as i32));
    }
    lang.epilogue(2);
    lang.ret();

    compiler
        .add_function(Some("trampoline"), &lang.compile_directly(), 0)
        .unwrap();
    let function = compiler.get_function_by_name_mut("trampoline").unwrap();
    function.is_builtin = true;
}


#[derive(ClapParser, Debug, Clone)]
#[command(version, about, long_about = None)]
#[command(name = "beag")]
#[command(bin_name = "beag")]
pub struct CommandLineArguments {
    program: Option<String>,
    #[clap(long, default_value = "false")]
    show_times: bool,
    #[clap(long, default_value = "false")]
    show_gc_times: bool,
    #[clap(long, default_value = "false")]
    no_gc: bool,
    #[clap(long, default_value = "false")]
    gc_always: bool,
    #[clap(long, default_value = "false")]
    all_tests: bool,
    #[clap(long, default_value = "false")]
    test: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = CommandLineArguments::parse();
    if args.all_tests {
        run_all_tests(args)
    } else {
        main_inner(args)
    }
}

fn run_all_tests(args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    for entry in std::fs::read_dir("resources")? {
        let entry = entry?;
        let path = entry.path();
        let path = path.to_str().unwrap();
        let source = std::fs::read_to_string(path)?;
        if !source.contains("// Expect") {
            continue;
        }
        println!("Running test: {}", path);
        let args = CommandLineArguments {
            program: Some(path.to_string()),
            show_times: args.show_times,
            show_gc_times: args.show_gc_times,
            no_gc: args.no_gc,
            gc_always: args.gc_always,
            all_tests: false,
            test: true,
        };
        main_inner(args)?;
    }
    Ok(())
}

extern "C" fn placeholder() -> usize {
    BuiltInTypes::null_value() as usize
}

fn main_inner(args: CommandLineArguments) -> Result<(), Box<dyn Error>> {
    if args.program.is_none() {
        println!("No program provided");
        return Ok(());
    }
    let program = args.program.clone().unwrap();
    let parse_time = Instant::now();
    let source = std::fs::read_to_string(program.clone())?;
    // TODO: This is very ad-hoc
    // I should make it real functionality later
    // but right now I just want something working
    let has_expect = args.test && source.contains("// Expect");
    let mut parser = Parser::new(source);
    let ast = parser.parse();
    if args.show_times {
        println!("Parse time {:?}", parse_time.elapsed());
    }

    cfg_if::cfg_if! {
        if #[cfg(feature = "compacting")] {
            type Alloc = CompactingHeap;
        } else if #[cfg(feature = "simple-mark-and-sweep")] {
            type Alloc = SimpleMarkSweepHeap;
        } else if #[cfg(feature = "simple-generation")] {
            type Alloc = SimpleGeneration;
        } else {
            type Alloc = SimpleMarkSweepHeap;
        }
    }

    let allocator = Alloc::new();
    let printer: Box<dyn Printer> = if has_expect {
        Box::new(TestPrinter::new(Box::new(DefaultPrinter)))
    } else {
        Box::new(DefaultPrinter)
    };

    let compiler = Compiler::new(args.clone(), allocator, printer);
    let compiler_with_lock = RwLock::new(compiler);

    let mut borrowed_compiler = compiler_with_lock.write().unwrap();

    borrowed_compiler.set_lock_pointer(&compiler_with_lock as *const _);

    compile_trampoline(&mut borrowed_compiler);

    borrowed_compiler.add_builtin_function("println", println_value::<Alloc> as *const u8, false)?;
    borrowed_compiler.add_builtin_function("print", print_value::<Alloc> as *const u8, false)?;
    borrowed_compiler.add_builtin_function("allocate_struct", allocate_struct::<Alloc> as *const u8, true)?;
        // TODO: Probably needs true
    borrowed_compiler.add_builtin_function("make_closure", make_closure::<Alloc> as *const u8, false)?;
    borrowed_compiler.add_builtin_function("property_access", property_access::<Alloc> as *const u8, false)?;
    borrowed_compiler.add_builtin_function("throw_error", throw_error::<Alloc> as *const u8, false)?;
    borrowed_compiler.add_builtin_function("assert!", placeholder as *const u8, false)?;
    borrowed_compiler.add_builtin_function("gc", gc::<Alloc> as *const u8, true)?;
    borrowed_compiler.add_builtin_function("gc_add_root", gc_add_root::<Alloc> as *const u8, false)?;
    borrowed_compiler.add_builtin_function("thread", new_thread::<Alloc> as *const u8, false)?;

    let compile_time = Instant::now();
    borrowed_compiler.compile_ast(ast)?;

    borrowed_compiler.check_functions();
    if args.show_times {
        println!("Compile time {:?}", compile_time.elapsed());
    }

    drop(borrowed_compiler);
    

    let time = Instant::now();
    let borrowed_compiler = compiler_with_lock.read().unwrap();
    let f = borrowed_compiler.get_function0("main");
    drop(borrowed_compiler);
    let result = f();
    if args.show_times {
        println!("Time {:?}", time.elapsed());
    }
    let mut borrowed_compiler = compiler_with_lock.write().unwrap();
    borrowed_compiler.println(result as usize);

    if has_expect {
        let source = std::fs::read_to_string(program)?;
        let expected = get_expect(&source);
        let expected = expected.trim();
        let printed = borrowed_compiler.printer.get_output().join("").trim().to_string();
        if printed != expected {
            println!("Expected: \n{}\n", expected);
            println!("Got: \n{}\n", printed);
            panic!("Test failed");
        }
        println!("Test passed");
    }
    drop(borrowed_compiler);

    loop {
       // take the list of threads so we are not holding a borrow on the compiler
       // use mem::replace to swap out the threads with an empty vec
       let threads = mem::replace(&mut compiler_with_lock.write().unwrap().threads, Vec::new());
        if threads.is_empty() {
            break;
        }
        for thread in threads {
            thread.join().unwrap();
        }
    }
    

    Ok(())
}

fn get_expect(source: &str) -> String {
    let start = source.find("// Expect").unwrap();
    // get each line as long as they start with //
    let lines = source[start..]
        .lines()
        .skip(1)
        .take_while(|line| line.starts_with("//"))
        .map(|line| line.trim_start_matches("//").trim())
        .collect::<Vec<_>>()
        .join("\n");
    lines
}

#[test]
fn try_all_examples() -> Result<(), Box<dyn Error>> {
    let args = CommandLineArguments {
        program: None,
        show_times: false,
        show_gc_times: false,
        no_gc: false,
        gc_always: false,
        all_tests: true,
        test: false,
    };
    run_all_tests(args)?;
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
