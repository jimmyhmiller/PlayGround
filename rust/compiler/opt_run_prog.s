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
sub rsp, 32
; Get Arg 0
mov qword [rbp-8], rdi
; Int 0
cmp qword [rbp-8], 0
jne then1
mov rax, 0
jmp fibonacci_exit
then1:
cmp qword [rbp-8], 1
jne body
mov rax, 1
jmp fibonacci_exit
body:
mov r9, rdi
sub r9, 1
mov rdi, r9
call fibonacci
mov qword [rbp-16], rax
mov r9, qword [rbp-8]
sub r9, 2
mov rdi, r9
call fibonacci
; Add Stack(1), ReturnRegister
mov r9, qword [rbp-16]
add r9, rax
mov rax, r9
jmp fibonacci_exit


fibonacci_exit:
leave
ret
