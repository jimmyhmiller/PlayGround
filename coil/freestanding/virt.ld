/* Linker script for the qemu-system-aarch64 `virt` machine (-kernel).
   RAM starts at 0x40000000; qemu loads the ELF and jumps to its entry (bare.start,
   set via `ld.lld -e`). No crt0, no libc — a single flat image. */
SECTIONS {
  . = 0x40000000;
  .text   : { *(.text*) }
  .rodata : { *(.rodata*) }   /* the c"…" string literals live here */
  .data   : { *(.data*) }
  .bss    : { *(.bss*) }      /* the static UART/arena buffers live here */
}
