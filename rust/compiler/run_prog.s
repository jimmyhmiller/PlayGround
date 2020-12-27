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
; Int 42
add rsp, -8
mov qword [rsp], 42
; Store 0
mov r9, qword [rsp]
mov qword [r15], r9
; Arg 0 with value Const(1)
mov rdi, 1
push rbp
call fib
pop rbp
add rsp, -8
mov qword [rsp], rax
; Print!
mov r9, qword [rsp]
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
mov rax, qword [rsp]
leave
ret

fib:
push rbp
mov rbp, rsp
; Get Arg 0
add rsp, -8
mov qword [rsp], rdi

loop_fib:
; Int 0
add rsp, -8
mov qword [rsp], 0
; Jump Equal
mov r9, qword [rsp]
add rsp, 8
cmp qword [rsp], r9
je done_fib
; Int 1
add rsp, -8
mov qword [rsp], 1
; Jump Equal
mov r9, qword [rsp]
add rsp, 8
cmp qword [rsp], r9
je done_fib
; Add Arg(0), Const(-1)
mov rax, rdi
add rax, -1
mov qword [rsp], rax
; Arg 0 with value Stack(0)
mov rdi, qword [rsp]
push rbp
call fib
pop rbp
add rsp, -8
mov qword [rsp], rax
; Add Arg(0), Const(-2)
mov rax, rdi
add rax, -2
mov qword [rsp], rax
; Arg 0 with value Stack(0)
mov rdi, qword [rsp]
call fib
add rsp, -8
mov qword [rsp], rax
; Add Stack(0), Stack(1)
mov rax, qword [rsp]
add rax, qword [rsp+8]
add rsp, 8
mov qword [rsp], rax

done_fib:
; Int 1
add rsp, -8
mov qword [rsp], 1
mov rax, qword [rsp]
leave
ret
