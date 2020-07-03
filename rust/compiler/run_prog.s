global _main

section .text

_main:
mov rbp, rsp
mov rax, 17
push rax

exit:
pop r9
mov rax, 33554433
mov rdi, r9
syscall
