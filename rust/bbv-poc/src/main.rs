

use std::ops::Deref;
use dynasmrt::DynasmApi;
use dynasmrt::x64::Assembler;
use dynasmrt::AssemblyOffset;
use dynasmrt::{dynasm, Pointer};

use std::{mem};


#[derive(Debug)]
struct Compiler {
    code: Assembler,
    stubs: Assembler,
}

impl Compiler {
    #[allow(dead_code, unused_variables)]

    // Instead of assuming I know where that code was,
    // I can get passed the pointer to that code. (or maybe the offset?)
    // Use that the base pointer, I can find our offset.
    // if the code is on the frontier, we can do fall through
    // optimizations.
    // Need to go back and look at how others are doing this.
    // In order to add more stubs here I think I need to do
    // alter uncommitted.
    // The only problem there is I'm not sure if this idea plays
    // nice with gcing our stubs.
    // Because we make a call, the code is executing.
    // I'm not quite sure how to get around that yet.

    pub extern "C" fn the_answer(&mut self) -> *const u8 {
        println!("Call to compiler");
        let offset = self.code.offset();
        self.code.alter(|op| {
            op.goto(AssemblyOffset(offset.0 - 10));
            dynasm!(op
                ; mov rax, 42
                ; ret
            );
            op.check(offset).unwrap();
        }).unwrap();
        let reader = self.code.reader();
        let lock = reader.lock();
        let buf = lock.deref();
        let ptr = buf.ptr(AssemblyOffset(offset.0 - 10));
        return ptr;
    }
}




fn main() {
    let mut compiler = Compiler{
        code: dynasmrt::x64::Assembler::new().unwrap(),
        stubs: dynasmrt::x64::Assembler::new().unwrap(),
    };
    let here = compiler.stubs.offset();
    dynasm!(compiler.stubs
        ; mov rax, QWORD Compiler::the_answer as _
        ; sub rsp, BYTE 0x28
        ; call rax
        ; add rsp, BYTE 0x28
        ; jmp rax
     );

    compiler.stubs.commit().unwrap();
    let reader = compiler.stubs.reader();
    let lock = reader.lock();
    let buf = lock.deref();
    let ptr = buf.ptr(here);
    drop(lock);

    let main = compiler.code.offset();
    dynasm!(compiler.code
        ; .arch x64
        ; mov rax, QWORD Pointer!(ptr)
        // ; mov rdi, compiler.code.offset().0 as _
        ; jmp rax
        ; mov rax, 56
        ; ret
    );

    compiler.code.commit().unwrap();
    let reader = compiler.code.reader();
    let lock = reader.lock();
    let buf = lock.deref();

    let hello_fn: extern "C" fn(&mut Compiler) -> i32 = unsafe { mem::transmute(buf.ptr(main)) };
    drop(lock);

    // Second time we will not compile back into the compiler
    println!("{}", hello_fn(&mut compiler));
    println!("{}", hello_fn(&mut compiler));
    println!("exit");
}
