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
; Arg 0 with value Const(0)
push rbp
mov rdi, 0
; Pushing arg 1 with value Const(20)
mov rsi, 20
call body
leave
ret

body:
push rbp
mov rbp, rsp
; Int 42
add rsp, -8
mov qword [rsp], 42
; Get Arg 0
add rsp, -8
mov qword [rsp], rdi

loop:
; Get Arg 1
add rsp, -8
mov qword [rsp], rsi
; Jump Equal
mov r9, qword [rsp]
add rsp, 8
cmp qword [rsp], r9
je done
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
; Int 1
add rsp, -8
mov qword [rsp], 1
; Add Stack(1), Stack(0)
mov rax, qword [rsp+8]
add rax, qword [rsp]
add rsp, 8
mov qword [rsp], rax
jmp loop

done:
add rsp, -8
mov r9, qword [r15]
mov qword [rsp], r9
; Print!
mov r9, qword [rsp]
push rdi
push rsi
lea rdi, [format]
mov rsi, r9
push rax
mov rax, 0
call _printf
pop rax
pop rsi
pop rdi
leave
ret
