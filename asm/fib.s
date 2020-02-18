; fib test
; Took an example and changed the registers so we are never
; pushing onto the stack at all
global start
extern _printf

section .text
start:
  push    rbx

  mov     r12, 90
  xor     r13, r13
  xor     r14, r14
  xor     rax, rax
  inc     r14

print:

  lea     rdi, [format]
  mov     rsi, r13

  call    _printf

  mov     r15, r13
  mov     r13, r14
  add     r14, r15
  dec     r12
  jnz     print

  pop     rbx

exit:
  mov rax, 0x2000001 ; magic MacOS exit thing
  mov rdi, 0
  syscall

section .data
format:
  default rel
    db    "%20ld", 10, 0