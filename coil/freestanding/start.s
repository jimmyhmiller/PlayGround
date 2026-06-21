// crt0 boot stub for bare-metal aarch64 (qemu virt). The standard startup a freestanding
// program needs and that Coil functions can't express (they have prologues that spill —
// before any stack exists). qemu jumps here; we set up the C runtime preconditions, then
// call the Coil entry `bare.start`. Lives in the RECIPE, not the compiler — freestanding
// startup composes on top, no core change.
.section .text.boot, "ax"
.globl _start
_start:
    // 1. stack pointer = top of the reserved stack (the linker script provides it).
    ldr     x0, =_stack_top
    mov     sp, x0

    // 1b. enable FP/SIMD at EL1 (CPACR_EL1.FPEN = 0b11). The optimizer emits SIMD
    //     instructions for struct/array init + memcpy; FP/SIMD is TRAPPED at reset,
    //     so without this those instructions fault with an Undefined Instruction
    //     (ESR EC=0x7). The stdlib's allocator setup hits this; hello.coil dodged it.
    mov     x0, #(3 << 20)
    msr     cpacr_el1, x0
    isb

    // 2. zero .bss (uninitialized globals — e.g. the stdlib's alloc-static state).
    //    No loader does this bare-metal; without it those globals hold garbage.
    ldr     x0, =__bss_start
    ldr     x1, =__bss_end
1:  cmp     x0, x1
    b.hs    2f
    str     xzr, [x0], #8
    b       1b

    // 3. into Coil. bare.start runs the program and never returns (it halts).
2:  bl      "bare.start"

    // 4. belt-and-suspenders: if it ever returns, park the core.
3:  wfe
    b       3b
