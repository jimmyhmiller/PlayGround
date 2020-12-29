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

fibonacci:
push rbp
mov rbp, rsp
; Get Arg 0
add rsp, -8
mov qword [rsp], rdi
; Int 0
add rsp, -8
mov qword [rsp], 0
; Jump Equal
mov r9, qword [rsp]
add rsp, 8
cmp qword [rsp], r9
je then1
; Get Arg 0
add rsp, -8
mov qword [rsp], rdi
; Int 1
add rsp, -8
mov qword [rsp], 1
; Jump Equal
mov r9, qword [rsp]
add rsp, 8
cmp qword [rsp], r9
je then2
; Get Arg 0
add rsp, -8
mov qword [rsp], rdi
; Int 1
add rsp, -8
mov qword [rsp], 1
; Sub Stack(1), Stack(0)
mov rax, qword [rsp+8]
sub rax, qword [rsp]
add rsp, 8
mov qword [rsp], rax
; Arg 0 with value Stack(0)
mov r9, qword [rsp]
push rdi
mov rdi, r9
call fibonacci
pop rdi
mov qword [rsp], rax
; Get Arg 0
add rsp, -8
mov qword [rsp], rdi
; Int 2
add rsp, -8
mov qword [rsp], 2
; Sub Stack(1), Stack(0)
mov rax, qword [rsp+8]
sub rax, qword [rsp]
add rsp, 8
mov qword [rsp], rax
; Arg 0 with value Stack(0)
mov r9, qword [rsp]
push rdi
mov rdi, r9
push rbp
call fibonacci
pop rbp
pop rdi
mov qword [rsp], rax
; Add Stack(1), Stack(0)
mov rax, qword [rsp+8]
add rax, qword [rsp]
add rsp, 8
mov qword [rsp], rax
mov rax, qword [rsp]
leave
ret

then2:
; Int 1
add rsp, -8
mov qword [rsp], 1
mov rax, qword [rsp]
leave
ret
mov rax, qword [rsp]
leave
ret

then1:
; Int 0
add rsp, -8
mov qword [rsp], 0
mov rax, qword [rsp]
leave
ret
