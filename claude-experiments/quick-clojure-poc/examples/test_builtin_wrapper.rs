/// Test that the builtin wrappers work correctly when called directly

use quick_clojure_poc::trampoline::{generate_builtin_wrappers, Trampoline};

fn main() {
    // First test: simple function that returns 42
    println!("=== Test 1: Simple return 42 ===");
    let code42: [u32; 2] = [
        0xD2800540,  // mov x0, #42
        0xD65F03C0,  // ret
    ];
    let ptr42 = allocate_and_copy(&code42);
    println!("Code at 0x{:x}: [{:08x}, {:08x}]", ptr42,
             unsafe { *(ptr42 as *const u32) },
             unsafe { *((ptr42 + 4) as *const u32) });

    let fn42: extern "C" fn() -> usize = unsafe { std::mem::transmute(ptr42) };
    let result42 = fn42();
    println!("Result: {} (expected 42)", result42);

    // Test 1b: Function that returns first arg
    println!("\n=== Test 1b: Return first arg ===");
    let code_ret_arg: [u32; 1] = [
        0xD65F03C0,  // ret (just return x0)
    ];
    let ptr_ret_arg = allocate_and_copy(&code_ret_arg);
    let fn_ret_arg: extern "C" fn(usize) -> usize = unsafe { std::mem::transmute(ptr_ret_arg) };
    let result_ret_arg = fn_ret_arg(42);
    println!("fn_ret_arg(42) = {} (expected 42)", result_ret_arg);

    // Test 1c: Function that returns second arg
    println!("\n=== Test 1c: Return second arg ===");
    let code_ret_arg2: [u32; 2] = [
        0xAA0103E0,  // mov x0, x1 (copy x1 to x0)
        0xD65F03C0,  // ret
    ];
    let ptr_ret_arg2 = allocate_and_copy(&code_ret_arg2);
    let fn_ret_arg2: extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(ptr_ret_arg2) };
    let result_ret_arg2 = fn_ret_arg2(11, 22);
    println!("fn_ret_arg2(11, 22) = {} (expected 22)", result_ret_arg2);

    // Test 1d: Simple add (no tagging)
    println!("\n=== Test 1d: Simple add (no tagging) ===");
    let code_add_simple: [u32; 2] = [
        0x8B010000,  // add x0, x0, x1
        0xD65F03C0,  // ret
    ];
    let ptr_add_simple = allocate_and_copy(&code_add_simple);
    let fn_add_simple: extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(ptr_add_simple) };
    let result_add_simple = fn_add_simple(3, 4);
    println!("fn_add_simple(3, 4) = {} (expected 7)", result_add_simple);

    // Test 1e: Add with tagging
    println!("\n=== Test 1e: Add with tagging ===");
    let code_add: [u32; 5] = [
        0xD343FC00,  // lsr x0, x0, #3  (untag arg0)
        0xD343FC21,  // lsr x1, x1, #3  (untag arg1)
        0x8B010000,  // add x0, x0, x1
        0xD37DF000,  // lsl x0, x0, #3  (retag) - correct encoding
        0xD65F03C0,  // ret
    ];
    let ptr_add = allocate_and_copy(&code_add);
    println!("Add function at 0x{:x}", ptr_add);
    unsafe {
        let code = ptr_add as *const u32;
        for i in 0..5 {
            println!("  {:04}: 0x{:08X}", i * 4, *code.add(i));
        }
    }
    let fn_add: extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(ptr_add) };
    let result_add = fn_add(24, 32);  // tagged 3 and 4
    println!("Result: {} (expected 56, untagged {})", result_add, result_add >> 3);

    // Test 1f: Just lsr (untag first arg)
    println!("\n=== Test 1f: Just lsr x0, x0, #3 ===");
    let code_lsr: [u32; 2] = [
        0xD343FC00,  // lsr x0, x0, #3  (untag)
        0xD65F03C0,  // ret
    ];
    let ptr_lsr = allocate_and_copy(&code_lsr);
    let fn_lsr: extern "C" fn(usize) -> usize = unsafe { std::mem::transmute(ptr_lsr) };
    let result_lsr = fn_lsr(24);  // tagged 3
    println!("fn_lsr(24) = {} (expected 3)", result_lsr);

    // Test 1g: lsr both then add
    println!("\n=== Test 1g: lsr + lsr + add ===");
    let code_lsr_add: [u32; 4] = [
        0xD343FC00,  // lsr x0, x0, #3  (untag arg0)
        0xD343FC21,  // lsr x1, x1, #3  (untag arg1)
        0x8B010000,  // add x0, x0, x1
        0xD65F03C0,  // ret
    ];
    let ptr_lsr_add = allocate_and_copy(&code_lsr_add);
    let fn_lsr_add: extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(ptr_lsr_add) };
    let result_lsr_add = fn_lsr_add(24, 32);  // tagged 3 and 4
    println!("fn_lsr_add(24, 32) = {} (expected 7)", result_lsr_add);

    // Test 1h: Just lsl
    println!("\n=== Test 1h: Just lsl x0, x0, #3 ===");
    let code_lsl: [u32; 2] = [
        0xD37DF000,  // lsl x0, x0, #3  (retag) - correct encoding
        0xD65F03C0,  // ret
    ];
    let ptr_lsl = allocate_and_copy(&code_lsl);
    let fn_lsl: extern "C" fn(usize) -> usize = unsafe { std::mem::transmute(ptr_lsl) };
    let result_lsl = fn_lsl(7);
    println!("fn_lsl(7) = {} (expected 56)", result_lsl);

    println!("\n=== Test 2: Builtin wrappers ===");

    let wrappers = generate_builtin_wrappers();

    // Dump the first few instructions
    let plus_ptr = *wrappers.get("+").expect("+ wrapper should exist");
    println!("Dumping first 5 instructions at 0x{:x}:", plus_ptr);
    unsafe {
        let code = plus_ptr as *const u32;
        for i in 0..5 {
            let instr = *code.add(i);
            println!("  {:04}: 0x{:08X}", i * 4, instr);
        }
    }

    // Get the + wrapper
    let plus_ptr = *wrappers.get("+").expect("+ wrapper should exist");
    println!("+ wrapper at address: 0x{:x}", plus_ptr);

    // Cast to function pointer
    let plus_fn: extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(plus_ptr) };

    // Test: (+ 3 4) with tagged args
    // Tagged 3 = 3 << 3 = 24
    // Tagged 4 = 4 << 3 = 32
    let arg0 = 3usize << 3;  // 24
    let arg1 = 4usize << 3;  // 32

    println!("Calling + wrapper with args: {} (tagged 3), {} (tagged 4)", arg0, arg1);
    let result = plus_fn(arg0, arg1);
    println!("Result via fn ptr: {} (tag={}, untagged: {})", result, result & 7, result >> 3);

    // Try calling via inline asm
    let result_asm: usize;
    unsafe {
        std::arch::asm!(
            "blr {ptr}",
            ptr = in(reg) plus_ptr,
            in("x0") arg0,
            in("x1") arg1,
            lateout("x0") result_asm,
            clobber_abi("C"),
        );
    }
    println!("Result via asm:    {} (tag={}, untagged: {})", result_asm, result_asm & 7, result_asm >> 3);

    // Try calling through the trampoline (generates JIT code that calls wrapper)
    println!("\n--- Test via Trampoline ---");
    let result_tramp = test_via_trampoline(plus_ptr, arg0, arg1);
    println!("Result via trampoline: {} (tag={}, untagged: {})", result_tramp, result_tramp & 7, result_tramp >> 3);

    if result >> 3 == 7 && (result & 7) == 0 {
        println!("SUCCESS! 3 + 4 = 7");
    } else {
        println!("FAILURE! Expected 56 (tagged 7), got {}", result);
    }

    // Test a few more operations
    println!("\n--- Testing other builtins ---");

    // Test -
    let minus_ptr = *wrappers.get("-").expect("- wrapper should exist");
    let minus_fn: extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(minus_ptr) };
    let result = minus_fn(10usize << 3, 3usize << 3);
    println!("10 - 3 = {} (expected 7)", result >> 3);

    // Test *
    let mul_ptr = *wrappers.get("*").expect("* wrapper should exist");
    let mul_fn: extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(mul_ptr) };
    let result = mul_fn(6usize << 3, 7usize << 3);
    println!("6 * 7 = {} (expected 42)", result >> 3);

    // Test <
    let lt_ptr = *wrappers.get("<").expect("< wrapper should exist");
    let lt_fn: extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(lt_ptr) };
    let result = lt_fn(3usize << 3, 5usize << 3);
    // true = 11 (0b1011), false = 3 (0b0011)
    println!("(< 3 5) = {} (expected 11=true)", result);
    let result = lt_fn(5usize << 3, 3usize << 3);
    println!("(< 5 3) = {} (expected 3=false)", result);
}

/// Allocate code and make it executable
fn allocate_and_copy(code: &[u32]) -> usize {
    use std::ptr;

    unsafe {
        let code_size = code.len() * 4;
        let mem = libc::mmap(
            ptr::null_mut(),
            code_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if mem == libc::MAP_FAILED {
            panic!("mmap failed");
        }

        ptr::copy_nonoverlapping(code.as_ptr() as *const u8, mem as *mut u8, code_size);

        if libc::mprotect(mem, code_size, libc::PROT_READ | libc::PROT_EXEC) != 0 {
            panic!("mprotect failed");
        }

        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn sys_icache_invalidate(start: *const libc::c_void, size: libc::size_t);
            }
            sys_icache_invalidate(mem, code_size);
        }

        mem as usize
    }
}

/// Generate JIT code that loads args, calls the wrapper, and returns
fn test_via_trampoline(wrapper_ptr: usize, arg0: usize, arg1: usize) -> usize {
    use std::ptr;

    let mut code: Vec<u32> = Vec::new();

    // Prologue - save link register (we're going to call the wrapper)
    // stp x29, x30, [sp, #-16]!
    code.push(0xa9bf7bfd);
    // mov x29, sp
    code.push(0x910003fd);

    // Load arg0 into x0
    // movz x0, #(arg0 & 0xffff)
    code.push(0xd2800000 | ((arg0 as u32 & 0xffff) << 5));
    if arg0 > 0xffff {
        // movk x0, #((arg0 >> 16) & 0xffff), lsl #16
        code.push(0xf2a00000 | (((arg0 as u32 >> 16) & 0xffff) << 5));
    }

    // Load arg1 into x1
    // movz x1, #(arg1 & 0xffff)
    code.push(0xd2800001 | ((arg1 as u32 & 0xffff) << 5));
    if arg1 > 0xffff {
        // movk x1, #((arg1 >> 16) & 0xffff), lsl #16
        code.push(0xf2a00001 | (((arg1 as u32 >> 16) & 0xffff) << 5));
    }

    // Load wrapper address into x2
    // movz x2, #(low 16 bits)
    code.push(0xd2800002 | (((wrapper_ptr as u32) & 0xffff) << 5));
    // movk x2, #(next 16 bits), lsl #16
    code.push(0xf2a00002 | ((((wrapper_ptr >> 16) as u32) & 0xffff) << 5));
    // movk x2, #(next 16 bits), lsl #32
    code.push(0xf2c00002 | ((((wrapper_ptr >> 32) as u32) & 0xffff) << 5));
    // movk x2, #(high 16 bits), lsl #48
    code.push(0xf2e00002 | ((((wrapper_ptr >> 48) as u32) & 0xffff) << 5));

    // Call the wrapper
    // blr x2
    code.push(0xd63f0040);

    // Result is now in x0

    // Epilogue - restore link register and return
    // ldp x29, x30, [sp], #16
    code.push(0xa8c17bfd);
    // ret
    code.push(0xd65f03c0);

    // Allocate and execute
    unsafe {
        let code_size = code.len() * 4;
        let mem = libc::mmap(
            ptr::null_mut(),
            code_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if mem == libc::MAP_FAILED {
            panic!("mmap failed");
        }

        ptr::copy_nonoverlapping(code.as_ptr() as *const u8, mem as *mut u8, code_size);

        if libc::mprotect(mem, code_size, libc::PROT_READ | libc::PROT_EXEC) != 0 {
            panic!("mprotect failed");
        }

        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn sys_icache_invalidate(start: *const libc::c_void, size: libc::size_t);
            }
            sys_icache_invalidate(mem, code_size);
        }

        let trampoline = Trampoline::new(64 * 1024);
        let result = trampoline.execute(mem as *const u8);

        libc::munmap(mem, code_size);

        result as usize
    }
}
