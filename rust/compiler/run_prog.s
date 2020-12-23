global _main

extern _printf
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
push rbp
mov rbp, rsp
sub rsp, 32
mov qword [rbp-8], 0
mov rax, 0
mov qword [rbp-16], 0

loop:
mov qword [rbp-24], 21
mov rdi, qword [rbp-16]
cmp qword [rbp-24], rdi
mov qword [rbp-24], rcx
mov rcx, qword [rbp-24]
je done
lea rdi, [format]
mov rsi, qword [rbp-16]
push rax
mov rax, 0
call _printf
pop rax
mov qword [rbp-24], 1
mov rax, 0
add rax, qword [rbp-16]
add rax, qword [rbp-24]
mov qword [rbp-16], rax
jmp loop

done:
mov rax, qword [rbp-16]
mov rsp, rbp
pop rbp
ret
