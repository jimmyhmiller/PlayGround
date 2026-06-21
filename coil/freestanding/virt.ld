/* Linker script for the qemu-system-aarch64 `virt` machine (-kernel).
   RAM starts at 0x40000000; qemu loads the ELF and enters at `_start` (the crt0 boot
   stub, kept FIRST at the load base). No crt0/libc from a toolchain — start.s is the
   entire runtime. Provides the symbols start.s needs: __bss_start/__bss_end + a
   reserved stack with _stack_top. */
ENTRY(_start)
SECTIONS {
  . = 0x40000000;
  .text : {
    KEEP(*(.text.boot))    /* the boot stub first, at the entry/load base */
    *(.text*)
  }
  .rodata : { *(.rodata*) }   /* the c"…" string literals */
  .data   : { *(.data*) }
  . = ALIGN(16);
  __bss_start = .;
  .bss    : { *(.bss*) }      /* zero-init globals (incl. stdlib alloc-static state) */
  . = ALIGN(16);
  __bss_end = .;
  . = . + 0x10000;            /* 64 KiB stack */
  . = ALIGN(16);
  _stack_top = .;
}
