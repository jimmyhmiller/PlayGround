global start
extern _printf

section .text
start:
start:
  push    rbx

  xor     r15,  r15 ; sum
  xor     r13, r13 ; num1
  mov     r14, 1 ; num 2

main:
  test r14, 1 ; How does this work? Why is this odd? I think because of bits comparison?
  jne odd

even:
  add r15, r14

odd:
  xadd r14, r13 ; add and swap
  cmp r13, 4000000
  jle main


print_answer:
  lea     rdi, [format]
  mov     rsi, r15
  call    _printf

exit:
  mov rax, 0x2000001 ; magic MacOS exit thing
  mov rdi, 0
  syscall

section .data
format:
  default rel
    db    "%20ld", 10, 0