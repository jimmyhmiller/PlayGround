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
    R9,
    Const(usize),
}

enum Op {
    Push(Register),
    Mov(Register, Register),
    Pop(Register),
    Ret,
    SysCall,
    Label(String),
    Section(String),
    Global(String),
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
            Register::Const(x) => {
                buffer.push_str(x.to_string().as_str());
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
            Op::Global(label) => {
                buffer.push_str("global ");
                buffer.push_str(label);
                buffer.push('\n');
            }
            SysCall => {
                buffer.push_str("syscall"); 
            }
        }
        buffer.push('\n');
    }
}

use Op::*;
use Register::*;

fn main() -> std::io::Result<()> {
    let buffer = &mut "".to_string();
    let instructions = [
        Global("_main".to_string()),
        Section("text".to_string()),
        Label("_main".to_string()),
        Mov(RBP, RSP),
        Mov(RAX, Const(17)),
        Push(RAX),
        Label("exit".to_string()),
        Pop(R9),
        Mov(RAX, Const(0x2000001)),
        Mov(RDI, R9),
        SysCall
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
            
    let result3 = Command::new("./run_prog")
            .output().unwrap().status.code();

    println!("{:?}", result3);
    Ok(())
}
