
use dynasmrt::x64::Assembler;
use std::ops::Deref;
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

use std::{io, slice, mem};
use std::io::Write;

#[derive(Debug)]
struct Compiler {
    assembler: Assembler
}

impl Compiler {
    #[allow(dead_code, unused_variables)]
    pub extern "C" fn add_more_than_once(&mut self, i : i32) {
        println!("{:?} {}", self, i);
        let here = self.assembler.offset();
        if i <= 0 {
            println!("path");
            dynasm!(self.assembler
                // ; add rsp, BYTE 0x28
                ; mov r9, 10
                ; mov rax, 42
                ; ret
            );
             println!("path done");
        } else {
            dynasm!(self.assembler
                ; add r9, 1
            );
        }
        println!("Here2");
        let _ = self.assembler.commit().unwrap();
        let reader =  self.assembler.reader();
        let lock = reader.lock();
        let buf = lock.deref();
        println!("Here3");

        let hello_fn: extern "C" fn() -> i32 = unsafe { mem::transmute(buf.ptr(here)) };
        drop(lock);
        println!("Here4");
        println!("returned {}", hello_fn());
        println!("Here5");
        
        return;
    }
}




fn main() {
    let mut compiler = Compiler{
        assembler: dynasmrt::x64::Assembler::new().unwrap(),
    };

    let printit = compiler.assembler.offset();
    dynasm!(compiler.assembler
        ; .arch x64
        ; mov r9, 42
        ; mov rax, 32
        ; mov rsi, 0
        ; mov rax, QWORD Compiler::add_more_than_once as _
        ; sub rsp, BYTE 0x28
        ; call rax
        ; add rsp, BYTE 0x28
        ; ret
    );
    let _ = compiler.assembler.commit().unwrap();
    let reader = compiler.assembler.reader();
    let lock = reader.lock();
    let buf = lock.deref();

    let hello_fn: extern "C" fn(&mut Compiler) -> i32 = unsafe { mem::transmute(buf.ptr(printit)) };
    drop(lock);

    println!("{}", hello_fn(&mut compiler));
    println!("here exit");
}
