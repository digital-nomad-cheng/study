	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 15	sdk_version 10, 15
	.globl	_plus                   ## -- Begin function plus
	.p2align	4, 0x90
_plus:                                  ## @plus
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	leaq	(%rdi,%rsi), %rax
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	_sumstore               ## -- Begin function sumstore
	.p2align	4, 0x90
_sumstore:                              ## @sumstore
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%rbx
	pushq	%rax
	.cfi_offset %rbx, -24
	movq	%rdx, %rbx
	callq	_plus
	movq	%rax, (%rbx)
	addq	$8, %rsp
	popq	%rbx
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.globl	_main                   ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r14
	pushq	%rbx
	subq	$16, %rsp
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	movq	%rsi, %rbx
	movq	8(%rsi), %rdi
	callq	_atoi
	movslq	%eax, %r14
	movq	16(%rbx), %rdi
	callq	_atoi
	movslq	%eax, %rbx
	leaq	-24(%rbp), %rdx
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	_sumstore
	movq	-24(%rbp), %rcx
	leaq	L_.str(%rip), %rdi
	movq	%r14, %rsi
	movq	%rbx, %rdx
	xorl	%eax, %eax
	callq	_printf
	xorl	%eax, %eax
	addq	$16, %rsp
	popq	%rbx
	popq	%r14
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"%ld + %ld --> %ld\n"


.subsections_via_symbols
