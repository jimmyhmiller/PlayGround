use std::collections::HashMap;
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
#[derive(Debug, Clone)]
enum Register {
    RBP,
    RSP,
    RBX,
    RAX,
    RDX,
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
            Register::RDX => {
                buffer.push_str("rdx");
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
#[derive(Debug, Clone, Copy)]
enum Location {
    // Nth element from top of stack
    Stack(i64),
    // Nth element above base stack pointer
    // TODO: Probably change to registers
    Arg(i64),
    // Nth element below base stack pointer
    Local(i64),
    ReturnRegister,
    // Not actually a location, just the value
    Const(i64),
}

impl Location {
    fn is_stack(&self) -> bool {
        match self {
            Location::Stack(_) => true,
            _ => false
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum Expr {
    Function1(String, String, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Equals(Box<Expr>, Box<Expr>),
    Var(String),
    Let(String, Box<Expr>),
    Int(i64),
    Return(Box<Expr>),
    Print(Box<Expr>),
    // While
    Call1(String, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>)
}

#[allow(dead_code)]
#[derive(Debug)]
enum ExprOrLang {
    IsLang(Lang),
    IsExpr(Expr),
}



#[allow(dead_code)]
#[allow(unused_variables)]
fn expr_to_lang(expr: Expr) -> Vec<Lang> {
    let mut queue : VecDeque<ExprOrLang> = VecDeque::new();
    let mut instructions : Vec<Lang> = vec![];
    // Need to do scopes properly
    let mut label_index = 0;
    let mut environment : HashMap<String, i64> = HashMap::new();
    // This could be wrong. Only really dealing with one function right now
    let mut current_func_name = "".to_string();
    queue.push_front(ExprOrLang::IsExpr(expr));
    while !(queue.is_empty()) {
        match queue.pop_front().unwrap() {
            ExprOrLang::IsLang(l) => {
                match (&l, &instructions.pop()) {
                    (Lang::Sub(Location::Stack(n), Location::Stack(0)), Some(Lang::Int(m))) => {
                        instructions.push(Lang::Sub(Location::Stack(n-1), Location::Const(*m)))
                    }
                    (Lang::Add(Location::Stack(n), Location::Stack(0)), Some(Lang::Int(m))) => {
                        instructions.push(Lang::Add(Location::Stack(n-1), Location::Const(*m)))
                    }
                    (Lang::Add(s, Location::Stack(0)), Some(c@Lang::Call1(_, _))) => {
                        instructions.push(c.clone());
                        instructions.push(Lang::Add(*s, Location::ReturnRegister))
                    }
                    (Lang::Sub(s, Location::Stack(0)), Some(c@Lang::Call1(_, _))) => {
                        instructions.push(c.clone());
                        instructions.push(Lang::Sub(*s, Location::ReturnRegister))
                    }
                    (Lang::Call1(f, Location::Stack(0)), Some(Lang::Int(n))) => {
                        instructions.push(Lang::Call1(f.to_string(), Location::Const(*n)))
                    }
                    (_, last_instruction) => {
                        println!("instructions {:?}\n{:?}",  last_instruction, l);
                        if last_instruction.is_some() {
                             instructions.push(last_instruction.as_ref().unwrap().clone());
                        }
                        instructions.push(l)
                    }
                }

            }
            ExprOrLang::IsExpr(expr) => {
                 match &expr {
                    Expr::Function1(name, arg, body) => {
                        current_func_name = name.to_string();
                        environment.clear();
                        instructions.push(Lang::Func(name.to_string()));
                        environment.insert(arg.to_string(), 0);
                        queue.push_front(ExprOrLang::IsExpr((**body).clone()));
                        queue.push_back(ExprOrLang::IsLang(Lang::Label(format!("{}_exit", name))));
                        queue.push_back(ExprOrLang::IsLang(Lang::FuncEnd));
                    },
                    Expr::If(pred, then, otherwise) => {
                        label_index +=1;
                        // backwards order because push_front
                        queue.push_front(ExprOrLang::IsExpr((**then).clone()));
                        // Need to have a counter for new labels scoped to function
                        queue.push_front(ExprOrLang::IsLang(Lang::Label(format!("then{}", label_index))));
                        queue.push_front(ExprOrLang::IsExpr((**otherwise).clone()));
                        queue.push_front(ExprOrLang::IsLang(Lang::JumpEqual(format!("then{}", label_index))));
                        queue.push_front(ExprOrLang::IsExpr((**pred).clone()));
                    },
                    // This only works in if for right now.
                    // Need to make this better and actually have booleans
                    Expr::Equals(v1, v2) => {
                         // backwards order because push_front
                        queue.push_front(ExprOrLang::IsExpr((**v2).clone()));
                        queue.push_front(ExprOrLang::IsExpr((**v1).clone()));
                    },
                    Expr::Add(v1, v2) => {
                        // backwards order because push_front
                        queue.push_front(ExprOrLang::IsLang(Lang::Add(Location::Stack(1), Location::Stack(0))));
                        queue.push_front(ExprOrLang::IsExpr((**v2).clone()));
                        queue.push_front(ExprOrLang::IsExpr((**v1).clone()));
                    },
                    Expr::Sub(v1, v2) => {
                        // backwards order because push_front
                        queue.push_front(ExprOrLang::IsLang(Lang::Sub(Location::Stack(1), Location::Stack(0))));
                        queue.push_front(ExprOrLang::IsExpr((**v2).clone()));
                        queue.push_front(ExprOrLang::IsExpr((**v1).clone()));
                    },
                    Expr::Var(s) => {
                        match environment.get(s) {
                            // This is wrong I don't distinguish between locals
                            // or args. Need to do that.
                            Some(n) => instructions.push(Lang::GetArg(*n)),
                            None => panic!("Variable not found {}", s),
                        }
                    },
                    Expr::Let(_, _) => {},
                    Expr::Int(n) => {
                        instructions.push(Lang::Int(*n));
                    },
                    Expr::Return(e) => {
                        // backwards order because push_front
                        queue.push_front(ExprOrLang::IsLang(Lang::Jump(format!("{}_exit", current_func_name))));
                        queue.push_front(ExprOrLang::IsLang(Lang::SetReturn));
                        queue.push_front(ExprOrLang::IsExpr((**e).clone()));
                    },
                    Expr::Print(e) => {
                        // backwards order because push_front
                        queue.push_front(ExprOrLang::IsLang(Lang::Print));
                        queue.push_front(ExprOrLang::IsExpr((**e).clone()));
                    },
                    Expr::Call1(name, arg) => {
                        queue.push_front(ExprOrLang::IsLang(Lang::Call1(name.to_string(), Location::Stack(0))));
                        queue.push_front(ExprOrLang::IsExpr((**arg).clone()));
                    }
                };
            }
        }
       
    }

    return instructions;
}





// Need to add print to this language
// Need to add a few more operators
// Then I need to make a little higher level language
// that compiles to this language.
// Need to add functions and function calls
#[allow(dead_code)]
#[derive(Debug, Clone)]
enum Lang {
    Int(i64),
    GetArg(i64),
    GetLocal(i64),
    SetLocal(i64, Location),
    Add(Location, Location),
    Sub(Location, Location),
    Label(String),
    // Equal,
    // Need to convert jump equal to location version
    // JumpEqualL(String, Location, Location),
    JumpEqual(String),
    JumpNotEqual(String),
    Jump(String),
    Print,
    Store(i64),
    Read(i64),
    Func(String),
    FuncEnd,
    SetReturn,
    Pop,
    // i64 is the number of arguments
    Call0(String),
    Call1(String, Location),
    Call2(String, Location, Location),

}


// Need to think about values on the stack from a call, vs locals on the stack.
// Maybe I have the base point vs the stack pointer?
// This is where liveness analysis could make things faster
fn loc_to_register(location: & Location, current_offset: i64) -> Register {
    match location {
        Location::Stack(i) => StackBaseOffset(current_offset + i * 8),
        Location::Arg(i) => ARGUMENT_REGISTERS[*i as usize].clone(),
        Location::Local(i) => StackBaseOffset(i * -8),
        Location::Const(i) => Const(*i),
        Location::ReturnRegister => RAX,
    }
}


const ARGUMENT_REGISTERS: [Register; 4] = [RDI, RSI, RDX, RCX];


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
    let mut _max_offset = 0;
    let mut _function_start_index : usize = 0;
    // Need to implement allocating stack at the beginning.
    // That means I need to keep track of how much stack I use.
    // And I need to allocate that at the beginning of the function.
    // Right now I generate multiple returns. Which does kind of complicate
    // the way I track the stack.
    // Maybe don't do that?

    // If I don't, then I can keep track of max_offset
    // make sure that it is a multple of 16, then all my calls
    // will be aligned and I can reference things offset from the base pointer.

    macro_rules! move_stack_pointer {
        ( $n:expr ) => {
            offset -= $n * 8;
            if ($n > 0) {
                _max_offset = offset;
            }
            // back!(Add(RSP, Const($n * -8)));
        };
        () => {
           move_stack_pointer!(1);
        }
    }

    macro_rules! fix_alignment {
        ( $n:expr ) => {{
            false
            // let new_offset = offset + -8 * $n;
            // if new_offset % 16 != 0 {
            //     back!(Push(RBP));
            //     offset -= 8;
            //     true
            // } else {
            //     false
            // }
        }};
        () => {
            fix_alignment!(0)
        }
    }

    // Consider keeping track of where last value is.
    // If we know that stack(0) is really r9, then let's use that.
    for e in lang {
        match e {
            Lang::GetArg(i) => {
                comment!("Get Arg {}", i);
                move_stack_pointer!();
                back!(Mov(StackBaseOffset(offset), ARGUMENT_REGISTERS[i as usize].clone()));
            },
            Lang::GetLocal(i) => {
                move_stack_pointer!();
                comment!("Get Local {}", i);
                back!(Mov(R9, loc_to_register(&Location::Local(i), offset)));
                back!(Mov(StackBaseOffset(offset), R9));
            },
            Lang::SetLocal(i, location) => {
                // TODO fill out
            }
            Lang::Func(name) => {
                back!(Label(name));
                back!(Push(RBP));
                back!(Mov(RBP, RSP));
                _function_start_index = instructions.len();
                offset = 0;
                _max_offset = 0;
            },
            Lang::SetReturn => {
                back!(Mov(RAX, StackBaseOffset(offset)));
            }
            Lang::FuncEnd => {
                let reserved_stack = ((_max_offset as f64 /16.0).floor() as i64).abs() * 16;
                if reserved_stack != 0 {
                    instructions.insert(_function_start_index, Sub(RSP, Register::Const(reserved_stack)));
                }
                _max_offset = 0;
                back!(Leave);
                back!(Ret);
            },
            Lang::Call0(name) => {
                let fixed = fix_alignment!();
                back!(Call(name));
                if fixed {
                    offset -= 8;
                    back!(Pop(RBP));
                }
                move_stack_pointer!();
                back!(Mov(StackBaseOffset(offset), RAX));
            },
            Lang::Call1(name, arg1) => {
                comment!("Arg {} with value {:?}", 0, arg1);

                let i = 0;
                let reg = &ARGUMENT_REGISTERS[i as usize];
                back!(Mov(R9, loc_to_register(&arg1, offset)));
                back!(Push(reg.clone()));
                offset -= 8;
                back!(Mov(reg.clone(), R9));

                back!(Push(RBP));
                back!(Call(name));
                back!(Pop(RBP));
                back!(Pop(reg.clone()));
                offset += 8;
                back!(Mov(StackBaseOffset(offset), RAX)); 
            },
            Lang::Call2(name, arg1, arg2) => {
                comment!("Arg {} with value {:?}", 0, arg1);

                let i = 0;
                back!(Mov(ARGUMENT_REGISTERS[i as usize].clone(), loc_to_register(&arg1, offset)));

                let i = 1;
                comment!("Pushing arg {} with value {:?}", i, arg2);
                back!(Mov(ARGUMENT_REGISTERS[i as usize].clone(), loc_to_register(&arg2, offset)));
                let fixed = fix_alignment!();
                back!(Call(name));
                if fixed {
                    offset -= 8;
                    back!(Pop(RBP));
                }
                move_stack_pointer!();
                back!(Mov(StackBaseOffset(offset), RAX));
            },
            Lang::Int(i) => {
                comment!("Int {}", i);
                move_stack_pointer!();
                back!(Mov(StackBaseOffset(offset), Const(i)));
            },
            Lang::Label(name) => {
                back!(Label(name));
            },
            Lang::JumpEqual(label)=> {
                comment!("Jump Equal");
                back!(Mov(R9, StackBaseOffset(offset)));
                move_stack_pointer!(-1);
                back!(Cmp(StackBaseOffset(offset), R9));
                back!(Je(label)); 
            }
            Lang::JumpNotEqual(label) => {
                back!(Mov(R9, StackBaseOffset(offset)));
                move_stack_pointer!(-1);
                back!(Cmp(StackBaseOffset(offset), R9));
                back!(Jne(label)); 
            }
            Lang::Jump(label) => {
                back!(Jmp(label));
            }
            Lang::Print => {
                comment!("Print!");
                back!(Mov(R9, StackBaseOffset(offset)));
                back!(Push(RDI));
                back!(Push(RSI));
                back!(Lea(RDI, DerefData("format".to_string())));
                back!(Mov(RSI, R9));
                back!(Push(RAX));
                // We just did three pushes hence 3.
                // Can of course simplify, but for clairity leaving it.
                 // Not sure about this code now.
                if (offset + 8*3) % 16 != 0 {
                   back!(Push(RAX));  
                }
                // Printf cares about rax for some weird reason
                back!(Mov(RAX, Const(0)));
                back!(Call("_printf".to_string()));
                back!(Pop(RAX));
                // Not sure about this code now.
                if (offset + 8*3) % 16 != 0 {
                   back!(Pop(RAX));  
                }
                back!(Pop(RSI));
                back!(Pop(RDI));
            }
            Lang::Read(index) => {
                move_stack_pointer!();
                back!(Mov(R9, Deref(Box::new(R15), index*8)));
                back!(Mov(StackBaseOffset(offset), R9));
            }
            // Should store pop?
            Lang::Store(index) => {
                comment!("Store {}", index);
                back!(Mov(R9, StackBaseOffset(offset)));
                back!(Mov(Deref(Box::new(R15), index*8), R9));
            }
            Lang::Add(loc1, loc2) => {
                comment!("Add {:?}, {:?}", loc1, loc2);
                back!(Mov(R9, loc_to_register(&loc1, offset)));
                back!(Add(R9, loc_to_register(&loc2, offset)));
                if loc1.is_stack() && loc2.is_stack()  {
                    move_stack_pointer!(-1);
                } else if loc1.is_stack() || loc2.is_stack()  {
                    // don't move stack
                } else {
                     move_stack_pointer!();
                }
                back!(Mov(StackBaseOffset(offset), R9));
            }
            Lang::Sub(loc1, loc2) => {
                comment!("Sub {:?}, {:?}", loc1, loc2);
                back!(Mov(R9, loc_to_register(&loc1, offset)));
                back!(Sub(R9, loc_to_register(&loc2, offset)));
                if loc1.is_stack() && loc2.is_stack()  {
                    move_stack_pointer!(-1);
                } else if loc1.is_stack() || loc2.is_stack()  {
                    // don't move stack
                } else {
                     move_stack_pointer!();
                }
                back!(Mov(StackBaseOffset(offset), R9));
            }
            Lang::Pop => {
                move_stack_pointer!(-1);
            }
        }
    }

    return instructions;
}

// Figure out how to reduce changes to rsp

fn main() -> std::io::Result<()> {

    macro_rules! lang {
        ((defn $name:ident[$arg:ident] 
            $body:tt
         )) => {
            Expr::Function1(stringify!($name).to_string(), stringify!($arg).to_string(),
                Box::new(lang!($body)))
        };
        ((if (= $arg:ident $val:expr)
            $result1:tt
            $result2:tt
        )) => {
            Expr::If(Box::new(
                    Expr::Equals(
                        Box::new(Expr::Var(stringify!($arg).to_string())),
                        Box::new(Expr::Int($val)))),
                    Box::new(Expr::Return(Box::new(lang!($result1)))),
                    Box::new(Expr::Return(Box::new(lang!($result2)))))
        };
        ((+ $arg1:tt $arg2:tt)) => {
            Expr::Add(Box::new(lang!($arg1)),
                      Box::new(lang!($arg2)))
        };
        ((- $arg1:tt $arg2:tt)) => {
            Expr::Sub(Box::new(lang!($arg1)),
                      Box::new(lang!($arg2)))
        };
        (($f:ident $arg:tt)) => {
            Expr::Call1(stringify!($f).to_string(), Box::new(lang!($arg)))
        };
        ($int:literal) => {
            Expr::Int($int)
        };
        ($var:ident) => {
            Expr::Var(stringify!($var).to_string())
        }
    }

    let prog = lang! {

        (defn fibonacci [n]
          (if (= n 0)
            0
            (if (= n 1) 
              1
              (+ (fibonacci (- n 1)) 
                 (fibonacci (- n 2))))))

    };
    println!("{:#?}", prog);


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
        Lang::Call1("fibonacci".to_string(), Location::Const(40)),
        Lang::Print,
        Lang::FuncEnd,

    ]);
    prelude.append(&mut main.into_iter().collect());
    prelude.append(&mut to_asm(expr_to_lang(prog)).into_iter().collect());
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
