global _main

extern _printf
extern _exit
section .data

format:
default rel
db 72, 101, 108, 108, 111, 32, 37, 100, 10, 0
section .text

_main:
call main

exit:
mov rdi, 0
push rbp
call _exit

main:
push rbp
mov rbp, rsp
mov rax, 19
lea rdi, [format]
mov rsi, rax
push rax
mov rax, 0
call _printf
pop rax
pop rbp
ret
