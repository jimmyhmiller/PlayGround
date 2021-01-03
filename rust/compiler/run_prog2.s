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
push rbp
mov rbp, rsp
; Push to align stack
push rbp
call _malloc
pop rbp
mov r15, rax
; Push to align stack
push rbp
call start
pop rbp
leave
ret

start:
push rbp
mov rbp, rsp
; Arg 0 with value Const(40)
mov r9, 40
push rdi
mov rdi, r9
push rbp
call fibonacci
pop rbp
pop rdi
mov qword [rbp], rax
; Print!
mov r9, qword [rbp]
push rdi
push rsi
lea rdi, [format]
mov rsi, r9
push rax
push rax
mov rax, 0
call _printf
pop rax
pop rax
pop rsi
pop rdi
leave
ret

fibonacci:
push rbp
mov rbp, rsp
sub rsp, 48
; Get Arg 0
mov qword [rbp-8], rdi
; Int 0
mov qword [rbp-16], 0
; Jump Equal
mov r9, qword [rbp-16]
cmp qword [rbp-8], r9
je then1
; Get Arg 0
mov qword [rbp-16], rdi
; Int 1
mov qword [rbp-24], 1
; Jump Equal
mov r9, qword [rbp-24]
cmp qword [rbp-16], r9
je then2
; Sub Arg(0), Const(1)
mov r9, rdi
sub r9, 1
mov qword [rbp-24], r9
; Arg 0 with value Stack(0)
mov r9, qword [rbp-24]
push rdi
mov rdi, r9
push rbp
call fibonacci
pop rbp
pop rdi
mov qword [rbp-24], rax
; Sub Arg(0), Const(2)
mov r9, rdi
sub r9, 2
mov qword [rbp-32], r9
; Arg 0 with value Stack(0)
mov r9, qword [rbp-32]
push rdi
mov rdi, r9
push rbp
call fibonacci
pop rbp
pop rdi
mov qword [rbp-32], rax
; Add Stack(1), ReturnRegister
mov r9, qword [rbp-24]
add r9, rax
mov qword [rbp-32], r9
mov rax, qword [rbp-32]
jmp fibonacci_exit

then2:
; Int 1
mov qword [rbp-40], 1
mov rax, qword [rbp-40]
jmp fibonacci_exit
mov rax, qword [rbp-40]
jmp fibonacci_exit

then1:
; Int 0
mov qword [rbp-48], 0
mov rax, qword [rbp-48]
jmp fibonacci_exit

fibonacci_exit:
leave
ret
