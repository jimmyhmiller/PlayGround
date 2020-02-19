global start
extern _printf

section .text
start:
  push    rbx ; something said I need to do this on mac, not sure 
 
  mov     r15,  0  ; sum
  mov     r12,  3  ; counter
  mov     r13d, 3  ; const for division
  mov     r14d, 5  ; const for division

main:
  mov eax, r12d
  xor edx, edx
  div r13d

  test edx, edx
  je sum

  mov eax, r12d
  xor edx, edx
  div r14d

  test edx, edx
  jne next

sum:
  add r15, r12

next:
  inc r12
  cmp r12, 1000
  jne main


print_answer:
  lea     rdi, [format]
  mov     rsi, r15
  call    _printf

exit:
  pop rbx
  mov rax, 0x2000001 ; magic MacOS exit thing
  mov rdi, 0
  syscall

section .data
format:
  default rel
    db    "%20ld", 10, 0