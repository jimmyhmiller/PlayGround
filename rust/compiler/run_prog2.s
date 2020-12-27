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
; Reserving args space
add rsp, -24
; Pushing arg 0 with value Const(0)
mov rdi, 0
mov qword [rsp+8], rdi
; Pushing arg 1 with value Const(20)
mov rdi, 20
mov qword [rsp+16], rdi
call body
leave
ret

body:
push rbp
mov rbp, rsp
sub rsp, 16
; Get Arg 0
mov rdi, qword [rbp+24]
mov qword [rbp-8], rdi

loop:
; Get Arg 1
mov rdi, qword [rbp+32]
mov qword [rbp-16], rdi
mov rdi, qword [rbp-8]
cmp qword [rbp-16], rdi
mov qword [rbp-16], rcx
mov rcx, qword [rbp-16]
je done
lea rdi, [format]
mov rsi, qword [rbp-8]
push rax
mov rax, 0
call _printf
pop rax
; Int 1
mov qword [rbp-16], 1
; Add Stack(1), Stack(0)
mov rax, qword [rbp-8]
add rax, qword [rbp-16]
mov qword [rbp-8], rax
jmp loop

done:
leave
ret
mov rax, qword [rbp-8]
