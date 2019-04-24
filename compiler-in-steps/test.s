	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14	sdk_version 10, 14
	.globl	_scheme_entry           ## -- Begin function scheme_entry
	.p2align	4, 0x90
_scheme_entry:                          ## @scheme_entry
	.cfi_startproc
## %bb.0:
	movl	$42, %eax
	retq
	.cfi_endproc
                                        ## -- End function

.subsections_via_symbols
