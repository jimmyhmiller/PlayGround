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
push rdi
mov rdi, 40
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
sub rsp, 16
; Jump Equal
mov r9, rdi
cmp r9, 0
je then1
; Jump Equal
mov r9, rdi
cmp r9, 1
je then2
; Sub Arg(0), Const(1)
mov r9, rdi
sub r9, 1
mov qword [rbp-8], r9
; Arg 0 with value Stack(0)
push rdi
mov rdi, qword [rbp-8]
push rbp
call fibonacci
pop rbp
pop rdi
mov qword [rbp-8], rax
; Sub Arg(0), Const(2)
mov r9, rdi
sub r9, 2
mov qword [rbp-16], r9
; Arg 0 with value Stack(0)
push rdi
mov rdi, qword [rbp-16]
push rbp
call fibonacci
pop rbp
pop rdi
mov qword [rbp-16], rax
; Add Stack(1), Stack(0)
mov r9, qword [rbp-8]
add r9, qword [rbp-16]
mov qword [rbp-8], r9
mov rax, qword [rbp-8]
jmp fibonacci_exit

then2:
mov rax, 1
jmp fibonacci_exit
mov rax, qword [rbp-8]
jmp fibonacci_exit

then1:
mov rax, 0
jmp fibonacci_exit

fibonacci_exit:
leave
ret
