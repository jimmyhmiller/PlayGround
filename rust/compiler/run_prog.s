global _main

extern _printf
extern _malloc
extern _exit
section .data

format:
default rel
db 37, 100, 10, 0
section .text

_main:
call main

exit:
mov rdi, 0
push rbp
call _exit

main:
mov rdi, 8000
call _malloc
mov r15, rax
push rbp
mov rbp, rsp
sub rsp, 32
mov qword [rbp-8], 0
mov rax, 0
mov qword [rbp-16], 42
mov rdi, qword [rbp-16]
mov qword [r15], rdi
mov qword [rbp-24], 0

loop:
mov qword [rbp-32], 21
mov rdi, qword [rbp-24]
cmp qword [rbp-32], rdi
mov qword [rbp-32], rcx
mov rcx, qword [rbp-32]
je done
lea rdi, [format]
mov rsi, qword [rbp-24]
push rax
mov rax, 0
call _printf
pop rax
mov qword [rbp-32], 1
mov rax, 0
add rax, qword [rbp-24]
add rax, qword [rbp-32]
mov qword [rbp-24], rax
jmp loop

done:
mov rdi, qword [r15]
mov qword [rbp-32], rdi
lea rdi, [format]
mov rsi, qword [rbp-32]
push rax
mov rax, 0
call _printf
pop rax
mov rax, qword [rbp-32]
mov rsp, rbp
pop rbp
ret
