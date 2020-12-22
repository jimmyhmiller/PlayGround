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
sub rsp, 16
mov qword [rbp-8], 0
mov qword [rbp-16], 20
mov qword [rbp-24], 2
mov rax, 0
add rax, qword [rbp-16]
add rax, qword [rbp-24]
mov qword [rbp-16], rax
mov qword [rbp-24], 3
mov rax, 0
add rax, qword [rbp-16]
add rax, qword [rbp-24]
mov qword [rbp-16], rax
mov qword [rbp-24], 30
mov rax, 0
add rax, qword [rbp-16]
add rax, qword [rbp-24]
mov qword [rbp-16], rax
lea rdi, [format]
mov rsi, rax
push rax
mov rax, 0
call _printf
pop rax
mov rsp, rbp
pop rbp
ret
