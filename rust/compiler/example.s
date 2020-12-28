	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 11, 1
	.intel_syntax noprefix
	.globl	_fib                    ## -- Begin function fib
	.p2align	4, 0x90
_fib:                                   ## @fib
	.cfi_startproc
## %bb.0:
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset rbp, -16
	mov	rbp, rsp
	.cfi_def_cfa_register rbp
	sub	rsp, 32
	mov	qword ptr [rbp - 16], rdi
	cmp	qword ptr [rbp - 16], 0
	jne	LBB0_2
## %bb.1:
	mov	dword ptr [rbp - 4], 0
	jmp	LBB0_5
LBB0_2:
	cmp	qword ptr [rbp - 16], 1
	jne	LBB0_4
## %bb.3:
	mov	dword ptr [rbp - 4], 1
	jmp	LBB0_5
LBB0_4:
	mov	rax, qword ptr [rbp - 16]
	sub	rax, 1
	mov	rdi, rax
	call	_fib
	mov	rcx, qword ptr [rbp - 16]
	sub	rcx, 2
	mov	rdi, rcx
	mov	dword ptr [rbp - 20], eax ## 4-byte Spill
	call	_fib
	mov	edx, dword ptr [rbp - 20] ## 4-byte Reload
	add	edx, eax
	mov	dword ptr [rbp - 4], edx
LBB0_5:
	mov	eax, dword ptr [rbp - 4]
	add	rsp, 32
	pop	rbp
	ret
	.cfi_endproc
                                        ## -- End function
	.globl	_main                   ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset rbp, -16
	mov	rbp, rsp
	.cfi_def_cfa_register rbp
	sub	rsp, 16
	mov	dword ptr [rbp - 4], 0
	mov	edi, 40
	call	_fib
	mov	dword ptr [rbp - 8], eax
	mov	esi, dword ptr [rbp - 8]
	lea	rdi, [rip + L_.str]
	mov	al, 0
	call	_printf
	xor	ecx, ecx
	mov	dword ptr [rbp - 12], eax ## 4-byte Spill
	mov	eax, ecx
	add	rsp, 16
	pop	rbp
	ret
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"%d\n"

.subsections_via_symbols
