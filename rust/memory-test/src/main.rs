mod codegen;

use codegen::{compile_directly, jump, mov_imm, ret, ArmAsm, X0};
use libc::{
    mmap, mprotect, sysconf, MAP_ANONYMOUS, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE,
    _SC_PAGESIZE,
};
use mach2::exception_types::{
    exception_behavior_t, exception_mask_t, EXCEPTION_DEFAULT, EXC_MASK_BAD_ACCESS,
};
use mach2::kern_return::kern_return_t;
use mach2::mach_types::task_t;
use mach2::port::{mach_port_t, MACH_PORT_NULL};
use mach2::thread_status::thread_state_flavor_t;
use mach2::traps::mach_task_self;
use nix::sys::signal::{sigaction, SaFlags, SigAction, SigHandler, Signal};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

extern "C" {
    /// Provided by LLVM to clear the instruction cache for the specified range.
    fn __clear_cache(start: *mut core::ffi::c_void, end: *mut core::ffi::c_void);
}

extern "C" {
    fn task_set_exception_ports(
        task: task_t,
        exception_mask: exception_mask_t,
        new_port: mach_port_t,
        behavior: exception_behavior_t,
        new_flavor: thread_state_flavor_t,
    ) -> kern_return_t;
}
extern "C" fn handle_sigbus(
    signal: libc::c_int,
    info: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) {
    unsafe {
        if signal == libc::SIGBUS {
            let addr = (*info).si_addr as *mut u8;
            println!("SIGBUS caught at address: {:?}", addr);

            // Wait for the signal to resume execution
            let signal_pair = SIGNAL_PAIR.clone();
            let (lock, cvar) = &*signal_pair;
            let mut signaled = lock.lock().unwrap();
            while !*signaled {
                signaled = cvar.wait(signaled).unwrap();
            }

            // Resume execution by advancing the program counter
            let ucontext = &mut *(context as *mut libc::ucontext_t);
            println!("PC: {:x}", (*ucontext.uc_mcontext).__ss.__pc);
            let pc = (*ucontext.uc_mcontext).__ss.__pc;
            if pc % 8 == 0 {
                (*ucontext.uc_mcontext).__ss.__pc = pc.wrapping_add(8);
            } else {
                (*ucontext.uc_mcontext).__ss.__pc = pc.wrapping_add(4);
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref SIGNAL_PAIR: Arc<(Mutex<bool>, Condvar)> = Arc::new((Mutex::new(false), Condvar::new()));
}

fn main() {
    unsafe {
        task_set_exception_ports(
            mach_task_self(),
            EXC_MASK_BAD_ACCESS,
            MACH_PORT_NULL,
            EXCEPTION_DEFAULT as exception_behavior_t,
            0,
        );
    }

    let sigbus_action = SigAction::new(
        SigHandler::SigAction(handle_sigbus),
        SaFlags::SA_RESTART,
        nix::sys::signal::SigSet::empty(),
    );
    unsafe {
        sigaction(Signal::SIGBUS, &sigbus_action).expect("Failed to set SIGBUS handler");
    }

    let endless_loop: Vec<ArmAsm> = vec![mov_imm(X0, 42), jump(-1), ret()];

    let compiled = compile_directly(endless_loop);

    let page_size = unsafe { sysconf(_SC_PAGESIZE) } as usize;
    let memory = unsafe {
        mmap(
            std::ptr::null_mut(),
            page_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        ) as *mut u8
    };

    if memory.is_null() {
        panic!("Failed to allocate executable memory");
    }

    unsafe {
        std::ptr::copy_nonoverlapping(compiled.as_ptr(), memory, compiled.len());

        mprotect(
            memory as *mut core::ffi::c_void,
            page_size,
            PROT_READ | PROT_EXEC,
        );
        __clear_cache(
            memory as *mut core::ffi::c_void,
            (memory as usize + page_size) as *mut core::ffi::c_void,
        );
    }

    let f: extern "C" fn() -> i32 = unsafe { std::mem::transmute(memory) };
    let thread = thread::spawn(move || {
        f() // This will trigger SIGBUS
    });

    thread::sleep(Duration::from_secs(1)); // Allow thread to trigger SIGBUS

    println!("Marking memory non-executable...");
    unsafe {
        mprotect(
            memory as *mut libc::c_void,
            page_size,
            PROT_READ | PROT_WRITE,
        );
    }
    println!("Marking memory executable again...");
    unsafe {
        mprotect(
            memory as *mut libc::c_void,
            page_size,
            PROT_READ | PROT_EXEC,
        );
    }

    {
        let signal_pair = SIGNAL_PAIR.clone();
        let (lock, cvar) = &*signal_pair;
        let mut signaled = lock.lock().unwrap();
        *signaled = true;
        cvar.notify_all();
    }

    println!("{:?}", thread.join().unwrap());
}
