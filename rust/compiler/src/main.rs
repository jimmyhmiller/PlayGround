use std::process::Command;
use std::fs::File;
use std::io::prelude::{Write};

// Come up with better name
enum Register {
    RBP,
    RSP,
    EAX,
    RBX,
    RAX,
    RDI,
    RSI,
    R9,
    DerefData(String),
    Const(usize),
}

enum Op {
    Push(Register),
    Mov(Register, Register),
    Lea(Register, Register),
    Pop(Register),
    Ret,
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

fn main() -> std::io::Result<()> {
    let buffer = &mut "".to_string();
    let instructions = [
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
        // Guessing this has to do with alignment?
        // Need to understand that more.
        Push(RBP),
        Call("_exit".to_string()),
        Label("main".to_string()),
        Push(RBP),
        Mov(RBP, RSP),
        Mov(RAX, Const(17)),
        Lea(RDI, DerefData("format".to_string())),
        Mov(RSI, RAX),
        Push(RAX),
        Mov(RAX, Const(0)),
        Call("_printf".to_string()),
        Pop(RAX),
        Pop(RBP),
        Ret,
    ];
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

    let result2 = Command::new("ld")
            .arg("-macosx_version_min")
            .arg("10.15.0")
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
