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
add rsp, -8
; Int 42
mov qword [rsp], 42
mov rdi, qword [rsp]
mov qword [r15], rdi
push rbp
add rsp, -16
; Pushing arg 0 with value Const(0)
mov rdi, 0
mov qword [rsp], rdi
; Pushing arg 1 with value Const(20)
mov rdi, 20
mov qword [rsp+8], rdi
call body
leave
ret

body:
push rbp
mov rbp, rsp
add rsp, -8
; Int 42
mov qword [rsp], 42
add rsp, -8
; Get Arg 0
mov rdi, qword [rbp+16]
mov qword [rsp], rdi

loop:
add rsp, -8
; Get Arg 1
mov rdi, qword [rbp+24]
mov qword [rsp], rdi
mov rdi, qword [rsp-8]
add rsp, 8
cmp qword [rsp-8], rdi
je done
lea rdi, [format]
mov rsi, qword [rsp]
push rax
push rax
mov rax, 0
call _printf
pop rax
pop rax
add rsp, -8
; Int 1
mov qword [rsp], 1
; Add Stack(1), Stack(0)
mov rax, qword [rsp+8]
add rax, qword [rsp]
add rsp, 8
mov qword [rsp], rax
jmp loop

done:
add rsp, -8
mov rdi, qword [r15]
mov qword [rsp], rdi
lea rdi, [format]
mov rsi, qword [rsp]
push rax
mov rax, 0
call _printf
pop rax
leave
ret
