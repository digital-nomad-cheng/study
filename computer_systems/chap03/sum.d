
sum:	file format Mach-O 64-bit x86-64

Disassembly of section __TEXT,__text:
__text:
100000ed0:	55 	pushq	%rbp
100000ed1:	48 89 e5 	movq	%rsp, %rbp
100000ed4:	48 8d 04 37 	leaq	(%rdi,%rsi), %rax
100000ed8:	5d 	popq	%rbp
100000ed9:	c3 	retq
100000eda:	66 0f 1f 44 00 00 	nopw	(%rax,%rax)
100000ee0:	55 	pushq	%rbp
100000ee1:	48 89 e5 	movq	%rsp, %rbp
100000ee4:	53 	pushq	%rbx
100000ee5:	50 	pushq	%rax
100000ee6:	48 89 d3 	movq	%rdx, %rbx
100000ee9:	e8 e2 ff ff ff 	callq	-30 <_plus>
100000eee:	48 89 03 	movq	%rax, (%rbx)
100000ef1:	48 83 c4 08 	addq	$8, %rsp
100000ef5:	5b 	popq	%rbx
100000ef6:	5d 	popq	%rbp
100000ef7:	c3 	retq
100000ef8:	0f 1f 84 00 00 00 00 00 	nopl	(%rax,%rax)
100000f00:	55 	pushq	%rbp
100000f01:	48 89 e5 	movq	%rsp, %rbp
100000f04:	41 56 	pushq	%r14
100000f06:	53 	pushq	%rbx
100000f07:	48 83 ec 10 	subq	$16, %rsp
100000f0b:	48 89 f3 	movq	%rsi, %rbx
100000f0e:	48 8b 7e 08 	movq	8(%rsi), %rdi
100000f12:	e8 41 00 00 00 	callq	65 <dyld_stub_binder+0x100000f58>
100000f17:	4c 63 f0 	movslq	%eax, %r14
100000f1a:	48 8b 7b 10 	movq	16(%rbx), %rdi
100000f1e:	e8 35 00 00 00 	callq	53 <dyld_stub_binder+0x100000f58>
100000f23:	48 63 d8 	movslq	%eax, %rbx
100000f26:	48 8d 55 e8 	leaq	-24(%rbp), %rdx
100000f2a:	4c 89 f7 	movq	%r14, %rdi
100000f2d:	48 89 de 	movq	%rbx, %rsi
100000f30:	e8 ab ff ff ff 	callq	-85 <_sumstore>
100000f35:	48 8b 4d e8 	movq	-24(%rbp), %rcx
100000f39:	48 8d 3d 48 00 00 00 	leaq	72(%rip), %rdi
100000f40:	4c 89 f6 	movq	%r14, %rsi
100000f43:	48 89 da 	movq	%rbx, %rdx
100000f46:	31 c0 	xorl	%eax, %eax
100000f48:	e8 11 00 00 00 	callq	17 <dyld_stub_binder+0x100000f5e>
100000f4d:	31 c0 	xorl	%eax, %eax
100000f4f:	48 83 c4 10 	addq	$16, %rsp
100000f53:	5b 	popq	%rbx
100000f54:	41 5e 	popq	%r14
100000f56:	5d 	popq	%rbp
100000f57:	c3 	retq

_plus:
100000ed0:	55 	pushq	%rbp
100000ed1:	48 89 e5 	movq	%rsp, %rbp
100000ed4:	48 8d 04 37 	leaq	(%rdi,%rsi), %rax
100000ed8:	5d 	popq	%rbp
100000ed9:	c3 	retq
100000eda:	66 0f 1f 44 00 00 	nopw	(%rax,%rax)

_sumstore:
100000ee0:	55 	pushq	%rbp
100000ee1:	48 89 e5 	movq	%rsp, %rbp
100000ee4:	53 	pushq	%rbx
100000ee5:	50 	pushq	%rax
100000ee6:	48 89 d3 	movq	%rdx, %rbx
100000ee9:	e8 e2 ff ff ff 	callq	-30 <_plus>
100000eee:	48 89 03 	movq	%rax, (%rbx)
100000ef1:	48 83 c4 08 	addq	$8, %rsp
100000ef5:	5b 	popq	%rbx
100000ef6:	5d 	popq	%rbp
100000ef7:	c3 	retq
100000ef8:	0f 1f 84 00 00 00 00 00 	nopl	(%rax,%rax)

_main:
100000f00:	55 	pushq	%rbp
100000f01:	48 89 e5 	movq	%rsp, %rbp
100000f04:	41 56 	pushq	%r14
100000f06:	53 	pushq	%rbx
100000f07:	48 83 ec 10 	subq	$16, %rsp
100000f0b:	48 89 f3 	movq	%rsi, %rbx
100000f0e:	48 8b 7e 08 	movq	8(%rsi), %rdi
100000f12:	e8 41 00 00 00 	callq	65 <dyld_stub_binder+0x100000f58>
100000f17:	4c 63 f0 	movslq	%eax, %r14
100000f1a:	48 8b 7b 10 	movq	16(%rbx), %rdi
100000f1e:	e8 35 00 00 00 	callq	53 <dyld_stub_binder+0x100000f58>
100000f23:	48 63 d8 	movslq	%eax, %rbx
100000f26:	48 8d 55 e8 	leaq	-24(%rbp), %rdx
100000f2a:	4c 89 f7 	movq	%r14, %rdi
100000f2d:	48 89 de 	movq	%rbx, %rsi
100000f30:	e8 ab ff ff ff 	callq	-85 <_sumstore>
100000f35:	48 8b 4d e8 	movq	-24(%rbp), %rcx
100000f39:	48 8d 3d 48 00 00 00 	leaq	72(%rip), %rdi
100000f40:	4c 89 f6 	movq	%r14, %rsi
100000f43:	48 89 da 	movq	%rbx, %rdx
100000f46:	31 c0 	xorl	%eax, %eax
100000f48:	e8 11 00 00 00 	callq	17 <dyld_stub_binder+0x100000f5e>
100000f4d:	31 c0 	xorl	%eax, %eax
100000f4f:	48 83 c4 10 	addq	$16, %rsp
100000f53:	5b 	popq	%rbx
100000f54:	41 5e 	popq	%r14
100000f56:	5d 	popq	%rbp
100000f57:	c3 	retq
Disassembly of section __TEXT,__stubs:
__stubs:
100000f58:	ff 25 a2 10 00 00 	jmpq	*4258(%rip)
100000f5e:	ff 25 a4 10 00 00 	jmpq	*4260(%rip)
Disassembly of section __TEXT,__stub_helper:
__stub_helper:
100000f64:	4c 8d 1d a5 10 00 00 	leaq	4261(%rip), %r11
100000f6b:	41 53 	pushq	%r11
100000f6d:	ff 25 8d 00 00 00 	jmpq	*141(%rip)
100000f73:	90 	nop
100000f74:	68 00 00 00 00 	pushq	$0
100000f79:	e9 e6 ff ff ff 	jmp	-26 <__stub_helper>
100000f7e:	68 0c 00 00 00 	pushq	$12
100000f83:	e9 dc ff ff ff 	jmp	-36 <__stub_helper>
