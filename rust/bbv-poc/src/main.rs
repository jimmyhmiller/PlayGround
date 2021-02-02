

use std::ops::Deref;
use dynasmrt::DynasmApi;
use dynasmrt::x64::Assembler;
use dynasmrt::AssemblyOffset;
use dynasmrt::{dynasm, Pointer};

use std::{mem};
use dynasmrt::x64::Rq;


#[derive(Debug)]
struct Compiler {
    code: Assembler,
    stubs: Assembler,
    context: Vec<(Type, Location)>,
    available_registers: Vec<Rq>,
}


#[derive(Debug)]
#[allow(dead_code, unused_variables)]
enum Lang {
    If(Box<Lang>, Box<Lang>, Box<Lang>),
    Bool(bool),
    Int(i32),
    Equal(Box<Lang>, Box<Lang>),
    Add(Box<Lang>, Box<Lang>),
}

#[derive(Debug)]
#[allow(dead_code, unused_variables)]
enum Type {
    Int,
    Bool,
}


#[derive(Debug)]
#[allow(dead_code, unused_variables)]
enum Register {
    RAX,
    RDX,
    RDI,
    RSI,
}


#[derive(Debug)]
#[allow(dead_code, unused_variables)]
enum Location {
    Register(Rq),
    Stack(i64),
    Immediate(i32),
}


// I really am not liking dynasms macro based way of doing things.
// I need to consider another library.



impl Compiler {

    fn add_to_location(&mut self, reg: u8, loc: Location) {
        match loc {
            Location::Register(r) => {
                dynasm!(self.code
                    ; add Rq(reg), Rq(r as u8)
                )
            }
            Location::Stack(_) => {

            }
            Location::Immediate(i) => {
                dynasm!(self.code
                    ; add Rq(reg), i
                )
            }
        }
    }

    #[allow(dead_code, unused_variables)]
    fn compile(&mut self, expr: Lang) {

        match expr {
            Lang::If(pred, t_expr, f_expr) => {
                // This is the actually interesting part and I should
                // actually implement this.
            }
            Lang::Bool(b) => {
                self.context.push((Type::Bool, Location::Immediate(b as i32)))
            }
            Lang::Int(i) => {
                self.context.push((Type::Int, Location::Immediate(i)))
            }
            Lang::Equal(left, right) => {

            }
            Lang::Add(left, right) => {
                self.compile(*left);
                self.compile(*right);
                let right_ctx = self.context.pop().unwrap();
                let left_ctx = self.context.pop().unwrap();
                let reg = self.available_registers.pop().unwrap();

                dynasm!(self.code
                    // This suboptimal. Should be mov and add?
                    ; xor Rq(reg as u8), Rq(reg as u8)
                    ;; self.add_to_location(reg as u8, left_ctx.1)
                    ;; self.add_to_location(reg as u8, right_ctx.1)
                    ; mov rax, Rq(reg as u8)
                );
                self.context.push((Type::Int, Location::Register(reg)));
                // Need to de-alloc registers based on if they are free
                // in the context
            }
        }
    }

    // Add(Add(1, 2) 1)

    // #[allow(dead_code, unused_variables)]
    // // Instead of assuming I know where that code was,
    // // I can get passed the pointer to that code. (or maybe the offset?)
    // // Use that the base pointer, I can find our offset.
    // // if the code is on the frontier, we can do fall through
    // // optimizations.
    // // Need to go back and look at how others are doing this.
    // // In order to add more stubs here I think I need to do
    // // alter uncommitted.
    // // The only problem there is I'm not sure if this idea plays
    // // nice with gcing our stubs.
    // // Because we make a call, the code is executing.
    // // I'm not quite sure how to get around that yet.

    // pub extern "C" fn the_answer(&mut self) -> *const u8 {
    //     println!("Call to compiler");
    //     let offset = self.code.offset();
    //     self.code.alter(|op| {
    //         op.goto(AssemblyOffset(offset.0 - 10));
    //         dynasm!(op
    //             ; mov rax, 42
    //             ; ret
    //         );
    //         op.check(offset).unwrap();
    //     }).unwrap();
    //     let reader = self.code.reader();
    //     let lock = reader.lock();
    //     let buf = lock.deref();
    //     let ptr = buf.ptr(AssemblyOffset(offset.0 - 10));
    //     return ptr;
    // }
}




fn main() {
    let mut compiler = Compiler{
        code: dynasmrt::x64::Assembler::new().unwrap(),
        stubs: dynasmrt::x64::Assembler::new().unwrap(),
        context: vec![],
        available_registers: vec![Rq::RDX, Rq::RSI, Rq::RDI, Rq::R9],
    };

    let main = compiler.code.offset();

    compiler.compile(Lang::Add(Box::new(Lang::Int(2)), Box::new(Lang::Add(Box::new(Lang::Int(2)), Box::new(Lang::Int(2))))));
    dynasm!(compiler.code
        ; ret
    );

    compiler.code.commit().unwrap();
    let reader = compiler.code.reader();
    let lock = reader.lock();
    let buf = lock.deref();

    let main_fn: extern "C" fn() -> i64 = unsafe { mem::transmute(buf.ptr(main)) };
    drop(lock);


    println!("{}", main_fn());
    // println!("{}", hello_fn(&mut compiler));
    println!("exit");

    // let here = compiler.stubs.offset();
    // dynasm!(compiler.stubs
    //     ; mov rax, QWORD Compiler::the_answer as _
    //     ; sub rsp, BYTE 0x28
    //     ; call rax
    //     ; add rsp, BYTE 0x28
    //     ; jmp rax
    //  );

    // compiler.stubs.commit().unwrap();
    // let reader = compiler.stubs.reader();
    // let lock = reader.lock();
    // let buf = lock.deref();
    // let ptr = buf.ptr(here);
    // drop(lock);

    // let main = compiler.code.offset();
    // dynasm!(compiler.code
    //     ; .arch x64
    //     ; mov rax, QWORD Pointer!(ptr)
    //     // ; mov rdi, compiler.code.offset().0 as _
    //     ; jmp rax
    //     ; mov rax, 56
    //     ; ret
    // );

    // compiler.code.commit().unwrap();
    // let reader = compiler.code.reader();
    // let lock = reader.lock();
    // let buf = lock.deref();

    // let hello_fn: extern "C" fn(&mut Compiler) -> i32 = unsafe { mem::transmute(buf.ptr(main)) };
    // drop(lock);

    // // Second time we will not compile back into the compiler
    // println!("{}", hello_fn(&mut compiler));
    // println!("{}", hello_fn(&mut compiler));
    // println!("exit");
}
