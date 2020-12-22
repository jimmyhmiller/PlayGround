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
    R9,
    EAX,
    DerefData(String),
    Const(i64),
    StackPointerOffset(i64)
}

#[allow(dead_code)]
#[derive(Debug)]
enum Op {
    Push(Register),
    Mov(Register, Register),
    Lea(Register, Register),
    Pop(Register),
    Ret,
    Inc(Register),
    Add(Register, Register),
    Sub(Register, Register),
    SysCall,
    Label(String),
    Section(String),
    Global(String),
    Call(String),
    DefaultRel,
    Db(String),
    Extern(String),
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
            Register::Const(x) => {
                buffer.push_str(x.to_string().as_str());
            }
            Register::DerefData(label) => {
                buffer.push_str("[");
                buffer.push_str(label);
                buffer.push_str("]");
            }
            Register::StackPointerOffset(offset) => {
                buffer.push_str("qword ");
                buffer.push_str("[");
                buffer.push_str("rbp");
                buffer.push_str(&offset.to_string());
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
            Op::Lea(a, b) => {
                buffer.push_str("lea "); 
                a.emit(buffer);
                buffer.push_str(", ");
                b.emit(buffer);
            }
            Op::Pop(r) => {
                buffer.push_str("pop "); 
                r.emit(buffer);
            }
            Op::Ret => {
                buffer.push_str("ret"); 
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
            SysCall => {
                buffer.push_str("syscall"); 
            }
        }
        buffer.push('\n');
    }
}




// fn generate_function(name: String, instructions: Vec<Op>)

use Op::*;
use Register::*;

#[allow(dead_code)]
#[derive(Debug)]
enum Lang {
    Int(i64),
    Plus,
}

#[allow(dead_code)]
fn to_asm(lang: Vec<Lang>) -> VecDeque<Op> {
    let mut instructions : VecDeque<Op> = VecDeque::new();
    let mut offset : i64 = 0;
    offset -= 8;
    instructions.push_front(Mov(StackPointerOffset(offset), Const(0)));
    for e in lang {
        match e {
            Lang::Int(i) => {
                offset -= 8;
                instructions.push_back(Mov(StackPointerOffset(offset), Const(i)));
            },
            Lang::Plus => {
                // super inefficient
                instructions.push_back(Mov(RAX, Const(0)));
                instructions.push_back(Add(RAX, StackPointerOffset(offset+8)));
                instructions.push_back(Add(RAX, StackPointerOffset(offset)));
                offset += 8;
                instructions.push_back(Mov(StackPointerOffset(offset), RAX));
            },
        }
    }
    instructions.push_front(Sub(RSP, Const(offset.abs())));
    return instructions;
}

fn main() -> std::io::Result<()> {
    let buffer = &mut "".to_string();
    let mut instructions = vec![
        Global("_main".to_string()),
        Extern("_printf".to_string()),
        Extern("_exit".to_string()),
        Section("data".to_string()),
        Label("format".to_string()),
        DefaultRel,
        Db("Hello %d\n".to_string()),
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
        Push(RBP),
        Mov(RBP, RSP),
    ];
    let add_things = to_asm(vec![
        Lang::Int(20),
        Lang::Int(2),
        Lang::Plus,
        Lang::Int(3),
        Lang::Plus,
        Lang::Int(30),
        Lang::Plus,
    ]);

    let mut end = vec![
        Lea(RDI, DerefData("format".to_string())),
        Mov(RSI, RAX),
        Push(RAX),
        Mov(RAX, Const(0)),
        Call("_printf".to_string()),
        Pop(RAX),
        Mov(RSP, RBP),
        Pop(RBP),
        Ret,
    ];
    instructions.append(&mut add_things.into_iter().collect());
    instructions.append(&mut end);

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
    println!("{:?} {:?}", String::from_utf8_lossy(&result3.stdout), result3.status.code());
    Ok(())
}
