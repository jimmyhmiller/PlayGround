use std::collections::VecDeque;
use std::process::Command;
use std::fs::File;
use std::io::prelude::{Write};


// https://en.wikipedia.org/wiki/X86_calling_conventions#x86-64_calling_conventions
// If the callee wishes to use registers RBX, RSP, RBP, and R12–R15,
// it must restore their original values before returning control to
// the caller. All other registers must be saved by the caller if
// it wishes to preserve their values.

#[allow(dead_code)]
#[derive(Debug)]
enum Register {
    RBP,
    RSP,
    RBX,
    RAX,
    RDI,
    RSI,
    RCX,
    R9,
    R15,
    EAX,
    DerefData(String),
    Const(i64),
    Deref(Box<Register>, i64),
    StackPointerOffset(i64),
    StackBaseOffset(i64),
}

#[allow(dead_code)]
#[derive(Debug)]
enum Op {
    Push(Register),
    Mov(Register, Register),
    Lea(Register, Register),
    Pop(Register),
    Leave,
    Ret,
    Inc(Register),
    Add(Register, Register),
    Sub(Register, Register),
    Cmp(Register, Register),
    Je(String),
    Jne(String),
    Jmp(String),
    SysCall,
    Label(String),
    Section(String),
    Global(String),
    Call(String),
    DefaultRel,
    Db(String),
    Extern(String),
    Comment(String),
}

trait Emittable {
    fn emit(&self, buffer: &mut String);
}

impl Emittable for Register {
    fn emit(&self, buffer : &mut String) {
        match self {
            Register::RBX => {
                buffer.push_str("rbx");
            }
            Register::R9 => {
                buffer.push_str("r9");
            }
            Register::R15 => {
                buffer.push_str("r15");
            }
            Register::RAX => {
                buffer.push_str("rax");
            }
            Register::RBP => {
                buffer.push_str("rbp");
            }
            Register::RSP => {
                buffer.push_str("rsp");
            }
            Register::EAX => {
                buffer.push_str("eax");
            }
            Register::RDI => {
                buffer.push_str("rdi");
            }
            Register::RSI => {
                buffer.push_str("rsi");
            }
            Register::RCX => {
                buffer.push_str("rcx");
            }
            Register::Const(x) => {
                buffer.push_str(x.to_string().as_str());
            }
            Register::DerefData(label) => {
                buffer.push_str("[");
                buffer.push_str(label);
                buffer.push_str("]");
            }
            Register::Deref(register, offset) => {
                buffer.push_str("qword ");
                buffer.push_str("[");
                register.emit(buffer);
                // don't do anything if 0
                if *offset > 0 {
                    buffer.push_str("+");
                    buffer.push_str(&offset.to_string());
                } else if *offset < 0 {
                    buffer.push_str("-");
                    buffer.push_str(&offset.to_string());
                }
                buffer.push_str("]");
            }
            Register::StackPointerOffset(offset) => {
                buffer.push_str("qword ");
                buffer.push_str("[");
                buffer.push_str("rsp");
                if *offset > 0 {
                    buffer.push_str("+");
                    buffer.push_str(&offset.to_string());
                } else if *offset < 0 {
                    buffer.push_str(&offset.to_string());
                }
                buffer.push_str("]");
            }
            Register::StackBaseOffset(offset) => {
                buffer.push_str("qword ");
                buffer.push_str("[");
                buffer.push_str("rbp");
                if *offset > 0 {
                    buffer.push_str("+");
                    buffer.push_str(&offset.to_string());
                } else if *offset < 0 {
                    buffer.push_str(&offset.to_string());
                }
                buffer.push_str("]");
            }
        }
    }
}

impl Emittable for Op {
    fn emit(&self, buffer : &mut String) {
        match self {
            Op::Push(r) => {
                buffer.push_str("push "); 
                r.emit(buffer);
            }
            Op::Mov(a, b) => {
                buffer.push_str("mov "); 
                a.emit(buffer);
                buffer.push_str(", ");
                b.emit(buffer);
            }
            Op::Inc(a) => {
                buffer.push_str("inc "); 
                a.emit(buffer);
            }
            Op::Add(a, b) => {
                buffer.push_str("add "); 
                a.emit(buffer);
                buffer.push_str(", ");
                b.emit(buffer);
            }
            Op::Sub(a, b) => {
                buffer.push_str("sub "); 
                a.emit(buffer);
                buffer.push_str(", ");
                b.emit(buffer);
            }
            Op::Cmp(a, b) => {
                buffer.push_str("cmp "); 
                a.emit(buffer);
                buffer.push_str(", ");
                b.emit(buffer);
            }
            Op::Lea(a, b) => {
                buffer.push_str("lea "); 
                a.emit(buffer);
                buffer.push_str(", ");
                b.emit(buffer);
            }
            Op::Je(label) => {
                buffer.push_str("je ");
                buffer.push_str(label); 
            }
            Op::Jne(label) => {
                buffer.push_str("jne ");
                buffer.push_str(label); 
            }
            Op::Jmp(label) => {
                buffer.push_str("jmp ");
                buffer.push_str(label); 
            }
            Op::Pop(r) => {
                buffer.push_str("pop "); 
                r.emit(buffer);
            }
            Op::Ret => {
                buffer.push_str("ret"); 
            }
            Op::Leave => {
                buffer.push_str("leave"); 
            }
            Op::Label(label) => {
                buffer.push('\n');
                buffer.push_str(label);
                buffer.push_str(":");
            }
            Op::Section(title) => {
                buffer.push_str("section .");
                buffer.push_str(title);
            }
            Op::Extern(label) => {
                buffer.push_str("extern ");
                buffer.push_str(label);
            }
            Op::Global(label) => {
                buffer.push_str("global ");
                buffer.push_str(label);
                buffer.push('\n');
            }
            Op::Call(label) => {
                buffer.push_str("call "); 
                buffer.push_str(label);
            }
            Op::DefaultRel => {
                buffer.push_str("default rel");
            } 
            Op::Db(data) => {
                let bytes : Vec<String> = data.chars().map(|c| (c as u32).to_string()).collect();
                buffer.push_str("db ");
                buffer.push_str(&bytes.join(", ").to_string());
                buffer.push_str(", 0");
            }
            Op::SysCall => {
                buffer.push_str("syscall"); 
            }
            Op::Comment(comment) => {
                buffer.push_str("; ");
                buffer.push_str(comment);
            }
        }
        buffer.push('\n');
    }
}




// fn generate_function(name: String, instructions: Vec<Op>)

use Op::*;
use Register::*;



// I'm transitioning from a generic stack machine to thinking
// about function calls and how to properly have locals and things
// like that. So I am allowing myself a way to refer to things
// on the stack in this general way, but also to have args and locals.
// Right now I am assuming that args are on the stack,
// but with this setup, I could put the args on registers like the c abi
// expects, which would be more performant.
#[allow(dead_code)]
#[derive(Debug)]
enum Location {
    // Nth element from top of stack
    Stack(i64),
    // Nth element above base stack pointer
    // TODO: Probably change to registers
    Arg(i64),
    // Nth element below base stack pointer
    Local(i64),
    // Not actually a location, just the value
    Const(i64),
}

// Need to add print to this language
// Need to add a few more operators
// Then I need to make a little higher level language
// that compiles to this language.
// Need to add functions and function calls
#[allow(dead_code)]
#[derive(Debug)]
enum Lang {
    Int(i64),
    Plus,
    GetArg(i64),
    SetArg(i64, Location),
    GetLocal(i64),
    SetLocal(i64, Location),
    Add(Location, Location),
    Label(String),
    Equal,
    JumpEqual(String),
    JumpNotEqual(String),
    Jump(String),
    Print,
    Store(i64),
    Read(i64),
    Func(String),
    FuncEnd,
    Call(String),

}


// Need to think about values on the stack from a call, vs locals on the stack.
// Maybe I have the base point vs the stack pointer?
// This is where liveness analysis could make things faster
fn loc_to_register(location: Location, current_offset: i64) -> Register {
    match location {
        Location::Stack(i) => StackBaseOffset(current_offset + i * 8),
        // Plus 2 because of the return value and base pointer
        // if we weren't aligned, we push another
        Location::Arg(i) => StackBaseOffset((i + 2) * 8),
        Location::Local(i) => StackBaseOffset(i * -8),
        Location::Const(i) => Const(i),
    }
}




#[allow(dead_code)]
#[allow(unused_variables)]
fn to_asm(lang: Vec<Lang>) -> VecDeque<Op> {



    let mut instructions : VecDeque<Op> = VecDeque::new();
    macro_rules! comment {
        ( $($x:expr),+ ) => {
            instructions.push_back(Comment(format!($($x),+)));
        }
    }
    macro_rules! back {
        ( $x:expr ) => {
            instructions.push_back($x);
        }
    }
    let mut offset : i64 = 0;
    // Max here is a bit weird because it is negative
    let mut max_offset : i64 = 0;
    let mut args = 0;
    let mut current_function_start = 0;

    // Consider a macro??
    // Seems weird, but it worked well for me before.
    // Really takes the ugliness out of some of this code.
    // Is it worth it though?

    offset -= 8;
    for e in lang {
        match e {
            Lang::GetArg(i) => {
                offset -= 8;
                max_offset = offset;
                comment!("Get Arg {}", i);
                back!(Mov(RDI, loc_to_register(Location::Arg(i), offset)));
                back!(Mov(StackBaseOffset(offset), RDI));
            },
            Lang::SetArg(i, location) => {
                args += 1;
                comment!("Pushing arg {} with value {:?}", i, location);
                back!(Mov(RDI, loc_to_register(location, offset)));
                back!(Mov(StackPointerOffset(i * 8), RDI));
            }
            Lang::GetLocal(i) => {
                offset -= 8;
                max_offset = offset;
                comment!("Get Local {}", i);
                back!(Mov(RDI, loc_to_register(Location::Local(i), offset)));
                back!(Mov(StackBaseOffset(offset), RDI));
            },
            Lang::SetLocal(i, location) => {
                
            }
            Lang::Func(name) => {
                back!(Label(name));
                back!(Push(RBP));
                back!(Mov(RBP, RSP));
                offset = 0;
                max_offset = 0;
                current_function_start = instructions.len();
            },
            Lang::FuncEnd => {
                // floor because it is negative
                let reserved_space = (max_offset as f64/16.0).floor() as i64 * 16 - 8;
                if reserved_space != 0 {
                    instructions.insert(current_function_start, Sub(RSP, Const(reserved_space.abs())));
                }
                back!(Leave);
                back!(Ret);
            },
            Lang::Call(name) => {
                offset -= 8 * args;
                max_offset = offset;
                back!(Call(name));
            },
            Lang::Int(i) => {
                offset -= 8;
                max_offset = offset;
                comment!("Int {}", i);
                back!(Mov(StackBaseOffset(offset), Const(i)));
            },
            Lang::Plus => {
                comment!("Plus");
                back!(Mov(RAX, StackBaseOffset(offset+8)));
                back!(Add(RAX, StackBaseOffset(offset)));
                offset += 8;
                back!(Mov(StackBaseOffset(offset), RAX));
            },
            Lang::Label(name) => {
                back!(Label(name));
            },
            Lang::Equal => {
                back!(Mov(RDI, StackBaseOffset(offset + 8)));
                back!(Cmp(StackBaseOffset(offset), RDI));
                back!(Mov(StackBaseOffset(offset), RCX));
            },
            Lang::JumpEqual(label)=> {
                back!(Mov(RCX, StackBaseOffset(offset)));
                offset += 8;
                back!(Je(label)); 
            }
            Lang::JumpNotEqual(label) => {
                back!(Mov(RCX, StackBaseOffset(offset)));
                offset += 8;
                back!(Jne(label));
            }
            Lang::Jump(label) => {
                back!(Jmp(label));
            }
            Lang::Print => {
                back!(Lea(RDI, DerefData("format".to_string())));
                back!(Mov(RSI, StackBaseOffset(offset)));
                back!(Push(RAX));
                // have better way to deal with alignment
                if offset % 16 != 0 {
                   back!(Push(RAX));  
                }
                // Printf cares about rax for some weird reason
                back!(Mov(RAX, Const(0)));
                back!(Call("_printf".to_string()));
                back!(Pop(RAX));
                if offset % 16 != 0 {
                   back!(Pop(RAX));  
                }
            }
            Lang::Read(index) => {
                offset -= 8;
                max_offset = offset;
                back!(Mov(RDI, Deref(Box::new(R15), index*8)));
                back!(Mov(StackBaseOffset(offset), RDI));
            }
            // Should store pop?
            Lang::Store(index) => {
                back!(Mov(RDI, StackBaseOffset(offset)));
                back!(Mov(Deref(Box::new(R15), index*8), RDI));
            }
            Lang::Add(loc1, loc2) => {
                comment!("Add {:?}, {:?}", loc1, loc2);
                back!(Mov(RAX, loc_to_register(loc1, offset)));
                back!(Add(RAX, loc_to_register(loc2, offset)));
                offset += 8;
                back!(Mov(StackBaseOffset(offset), RAX));
            }
        }
    }

    // Think about the fact that we are pushing here.
    // We push in the reverse order of what we want to run.

    // if args != 0 {
    //     instructions.push_front(Mov(RSP, RBP));
    //     // ceil because it is positive
    // }

    // instructions.push_back(Mov(RAX, StackBaseOffset(offset)));
    return instructions;
}

fn main() -> std::io::Result<()> {
    let buffer = &mut "".to_string();
    let mut prelude = vec![
        Global("_main".to_string()),
        Extern("_printf".to_string()),
        Extern("_malloc".to_string()),
        Extern("_exit".to_string()),
        Section("data".to_string()),
        Label("format".to_string()),
        DefaultRel,
        Db("%d\n".to_string()),
        Section("text".to_string()),
        Label("_main".to_string()),
        Call("main".to_string()),
        Label("exit".to_string()),
        Mov(RDI, Const(0)),
        // So I knew that things had to be aligned.
        // But I had thought it was 16 bit alignment.
        // When it was 16 BYTE alignment. Which means
        // 128 bit alignment. 
        // So when you do call, you push the return address
        // onto the stack. Which means you are always
        // aligned on 8 bytes instead of 16.
        // So before another call you need an odd number
        // of pushes.
        // This is really annoying and weird. Will have to
        // figure out how to deal with this.
        Push(RBP),
        Call("_exit".to_string()),
        Label("main".to_string()),
        Mov(RDI, Const(8*1000)),
        Push(RBP),
        Mov(RBP, RSP),

        Comment("Push to align stack".to_string()),
        Push(RBP),
        Call("_malloc".to_string()),
        Pop(RBP),
        // Should probably be in bss or something?
        Mov(R15, RAX),
        Comment("Push to align stack".to_string()),
        Push(RBP),
        Call("start".to_string()),
        Pop(RBP),
        Leave,
        Ret,
    ];

    let main = to_asm(vec![
        Lang::Func("start".to_string()),
        Lang::Int(42),
        Lang::Store(0),
        Lang::SetArg(0, Location::Const(0)),
        Lang::SetArg(1, Location::Const(20)),
        Lang::Call("body".to_string()),
        Lang::FuncEnd,


        Lang::Func("body".to_string()),
        // This causes a segfault. Need to fix.
        // alignment?
        // Lang::Int(42),
        Lang::GetArg(0),
        Lang::Label("loop".to_string()),
        Lang::GetArg(1),
        Lang::Equal,
        Lang::JumpEqual("done".to_string()),
        Lang::Print,
        Lang::Int(1),
        Lang::Add(Location::Stack(1), Location::Stack(0)),
        Lang::Jump("loop".to_string()),
        Lang::Label("done".to_string()),
        Lang::Read(0),
        Lang::Print,
        Lang::FuncEnd,
    ]);

    let mut postlude = vec![
        // Mov(RSP, RBP),
        // Pop(RBP),
        // Ret,
    ];
    prelude.append(&mut main.into_iter().collect());
    prelude.append(&mut postlude);
    let instructions = prelude;

    for instruction in instructions.iter() {
        instruction.emit(buffer);
    }

    println!("{}", buffer);
    let mut file = File::create("run_prog.s")?;
    file.write_all(buffer.as_bytes())?;

    let result1 = Command::new("nasm")
            .arg("-f")
            .arg("macho64")
            .arg("run_prog.s")
            .output();

    println!("{:?}", result1);

    let dylib_path = std::str::from_utf8(Command::new("xcrun")
        .args(&["-sdk", "macosx", "--show-sdk-path"])
        .output()?.stdout.as_slice()).unwrap().to_string();

    let result2 = Command::new("/usr/bin/ld")
            .arg("-macosx_version_min")
            .arg("10.15.0")
            .arg("-syslibroot")
            .arg(dylib_path.trim())
            .arg("-lSystem")
            .arg("-o")
            .arg("run_prog")
            .arg("run_prog.o")
            .output()?;

    println!("{:?}", result2);
            
    let result3 = Command::new("./run_prog").output().unwrap();
    println!("{} {:?}", String::from_utf8_lossy(&result3.stdout), result3.status.code());
    Ok(())
}
