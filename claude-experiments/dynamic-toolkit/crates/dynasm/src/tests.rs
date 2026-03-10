#[cfg(test)]
mod arm64_tests {
    use crate::arm64::*;
    use crate::arm64::inst::*;
    use crate::buffer::CodeBuffer;
    #[test]
    fn test_ret() {
        let inst = Arm64Inst::ret();
        let encoded = inst.encode();
        // RET X30: 0xD65F03C0
        assert_eq!(encoded, 0xD65F03C0);
    }

    #[test]
    fn test_add_shifted_register() {
        // ADD X0, X1, X2
        let inst = Arm64Inst::add(X0, X1, X2);
        let encoded = inst.encode();
        // sf=1, op=0, S=0, shift=00, Rm=X2, imm6=0, Rn=X1, Rd=X0
        // 1 00 01011 00 0 00010 000000 00001 00000
        assert_eq!(encoded, 0x8B020020);
    }

    #[test]
    fn test_sub_imm() {
        // SUB SP, SP, #16
        let inst = Arm64Inst::sub_imm(SP, SP, 16);
        let encoded = inst.encode();
        // sf=1, op=1, S=0, 100010, sh=0, imm12=16, Rn=SP(31), Rd=SP(31)
        // 1 10 100010 0 000000010000 11111 11111
        assert_eq!(encoded, 0xD10043FF);
    }

    #[test]
    fn test_movz() {
        // MOVZ X0, #42
        let inst = Arm64Inst::movz(X0, 42, 0);
        let encoded = inst.encode();
        // sf=1, opc=10, 100101, hw=00, imm16=42, Rd=X0
        assert_eq!(encoded, 0xD2800540);
    }

    #[test]
    fn test_bl() {
        // BL +4 (offset in instruction words)
        let inst = Arm64Inst::bl(4);
        let encoded = inst.encode();
        // 1 00101 00000000000000000000000100
        assert_eq!(encoded, 0x94000004);
    }

    #[test]
    fn test_ldr_unsigned_offset() {
        // LDR X0, [X1, #8]
        let inst = Arm64Inst::ldr(X0, X1, 8);
        let encoded = inst.encode();
        // size=11, 111 0 01 01 imm12=1 Rn=X1 Rt=X0
        assert_eq!(encoded, 0xF9400420);
    }

    #[test]
    fn test_str_unsigned_offset() {
        // STR X0, [X1, #16]
        let inst = Arm64Inst::str(X0, X1, 16);
        let encoded = inst.encode();
        // size=11, 111 0 01 00, imm12=2, Rn=X1, Rt=X0
        assert_eq!(encoded, 0xF9000820);
    }

    #[test]
    fn test_stp_pre_index() {
        // STP X29, X30, [SP, #-16]!
        let inst = Arm64Inst::stp(X29, X30, SP, -16, StpMode::PreIndex);
        let encoded = inst.encode();
        // opc=10, 101 0 011 0 imm7=-2(7bit) Rt2=X30 Rn=SP Rt=X29
        assert_eq!(encoded, 0xA9BF7BFD);
    }

    #[test]
    fn test_ldp_post_index() {
        // LDP X29, X30, [SP], #16
        let inst = Arm64Inst::ldp(X29, X30, SP, 16, LdpMode::PostIndex);
        let encoded = inst.encode();
        // opc=10, 101 0 001 1 imm7=2 Rt2=X30 Rn=SP Rt=X29
        assert_eq!(encoded, 0xA8C17BFD);
    }

    #[test]
    fn test_cbz() {
        // CBZ X0, +8 (2 instructions forward)
        let inst = Arm64Inst::cbz(X0, 2);
        let encoded = inst.encode();
        // sf=1, 011010 0 imm19=2 Rt=X0
        assert_eq!(encoded, 0xB4000040);
    }

    #[test]
    fn test_cbnz() {
        // CBNZ X0, +8 (2 instructions forward)
        let inst = Arm64Inst::cbnz(X0, 2);
        let encoded = inst.encode();
        // sf=1, 011010 1 imm19=2 Rt=X0
        assert_eq!(encoded, 0xB5000040);
    }

    #[test]
    fn test_ldrb() {
        // LDRB W0, [X1, #0]
        let inst = Arm64Inst::ldrb(W0, X1, 0);
        let encoded = inst.encode();
        // size=00, 111 0 01 01, imm12=0, Rn=X1, Rt=W0
        assert_eq!(encoded, 0x39400020);
    }

    #[test]
    fn test_b_cond() {
        // B.EQ +4 (1 instruction forward)
        let inst = Arm64Inst::b_cond(Arm64Cond::EQ, 1);
        let encoded = inst.encode();
        // 0101010 0 imm19=1 0 cond=0000
        assert_eq!(encoded, 0x54000020);
    }

    #[test]
    fn test_cmp_register() {
        // CMP X1, X2
        let inst = Arm64Inst::cmp(X1, X2);
        let encoded = inst.encode();
        // sf=1, 11 01011 00 0 Rm=X2 000000 Rn=X1 Rd=11111(XZR)
        assert_eq!(encoded, 0xEB02003F);
    }

    #[test]
    fn test_cset() {
        // CSET X0, EQ
        let inst = Arm64Inst::cset(X0, Arm64Cond::EQ);
        let encoded = inst.encode();
        // CSINC X0, XZR, XZR, NE (inverted condition)
        // sf=1, 00 11010100 11111 cond=NE(0001) 0 1 Rm=11111 Rd=X0
        assert_eq!(encoded, 0x9A9F17E0);
    }

    #[test]
    fn test_mov_imm64() {
        // Load 0x1234_5678_9ABC_DEF0 into X0
        let insts = Arm64Inst::mov_imm64(X0, 0x1234_5678_9ABC_DEF0);
        assert_eq!(insts.len(), 4);
        // MOVZ X0, #0xDEF0
        // MOVK X0, #0x9ABC, LSL #16
        // MOVK X0, #0x5678, LSL #32
        // MOVK X0, #0x1234, LSL #48
    }

    #[test]
    fn test_mov_imm64_zero() {
        let insts = Arm64Inst::mov_imm64(X0, 0);
        assert_eq!(insts.len(), 1);
    }

    #[test]
    fn test_codebuffer_label_forward_branch() {
        let mut buf: CodeBuffer<Arm64> = CodeBuffer::new();
        let label = buf.create_label();

        // Emit B.EQ with placeholder offset
        let branch_offset = buf.emit(Arm64Inst::b_cond(Arm64Cond::EQ, 0));
        buf.add_reloc(branch_offset, label, Arm64RelocKind::Cond19);

        // Emit a NOP
        buf.emit(Arm64Inst::add_imm(X0, X0, 0)); // NOP-like

        // Bind label here (offset = 8)
        buf.bind_label(label);
        buf.emit(Arm64Inst::ret());

        buf.finalize();

        let code = buf.code();
        // The B.EQ should now have imm19 = 2 (8 bytes / 4 = 2 instructions)
        let branch_word = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        let imm19 = (branch_word >> 5) & 0x7FFFF;
        assert_eq!(imm19, 2);
    }

    #[test]
    fn test_codebuffer_bl_reloc() {
        let mut buf: CodeBuffer<Arm64> = CodeBuffer::new();
        let label = buf.create_label();

        let bl_offset = buf.emit(Arm64Inst::bl(0));
        buf.add_reloc(bl_offset, label, Arm64RelocKind::Branch26);

        // 3 NOPs
        buf.emit(Arm64Inst::add_imm(X0, X0, 0));
        buf.emit(Arm64Inst::add_imm(X0, X0, 0));
        buf.emit(Arm64Inst::add_imm(X0, X0, 0));

        buf.bind_label(label);
        buf.emit(Arm64Inst::ret());

        buf.finalize();

        let code = buf.code();
        let bl_word = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        let imm26 = bl_word & 0x03FF_FFFF;
        assert_eq!(imm26, 4); // 4 instructions forward
    }
}

#[cfg(test)]
mod x86_64_tests {
    use crate::x86_64::*;
    use crate::x86_64::inst::X64Inst;
    use crate::buffer::CodeBuffer;

    #[test]
    fn test_mov_rr() {
        // MOV RAX, RBX
        let instr = X64Inst::MovRR { dest: RAX, src: RBX };
        assert_eq!(instr.encode().as_slice(), &[0x48, 0x89, 0xD8]);

        // MOV R8, R9
        let instr = X64Inst::MovRR { dest: R8, src: R9 };
        assert_eq!(instr.encode().as_slice(), &[0x4D, 0x89, 0xC8]);
    }

    #[test]
    fn test_add_rr() {
        let instr = X64Inst::AddRR { dest: RAX, src: RBX };
        assert_eq!(instr.encode().as_slice(), &[0x48, 0x01, 0xD8]);
    }

    #[test]
    fn test_push_pop() {
        assert_eq!(X64Inst::Push { reg: RAX }.encode().as_slice(), &[0x50]);
        assert_eq!(X64Inst::Push { reg: R8 }.encode().as_slice(), &[0x41, 0x50]);
        assert_eq!(X64Inst::Pop { reg: RBX }.encode().as_slice(), &[0x5B]);
    }

    #[test]
    fn test_ret() {
        assert_eq!(X64Inst::Ret.encode().as_slice(), &[0xC3]);
    }

    #[test]
    fn test_lea_rsp_offset() {
        let instr = X64Inst::Lea { dest: RDX, base: RSP, offset: -8 };
        assert_eq!(instr.encode().as_slice(), &[0x48, 0x8D, 0x54, 0x24, 0xF8]);
    }

    #[test]
    fn test_sub_ri() {
        let instr = X64Inst::SubRI { dest: RSP, imm: 8 };
        let encoded = instr.encode();
        // REX.W + 81 /5 + imm32
        assert_eq!(encoded[0], 0x48); // REX.W
        assert_eq!(encoded[1], 0x81);
    }

    #[test]
    fn test_jmp_rel32() {
        let instr = X64Inst::Jmp { offset: 0 };
        let encoded = instr.encode();
        assert_eq!(encoded.as_slice(), &[0xE9, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_call_rel32() {
        let instr = X64Inst::CallRel { offset: 0 };
        let encoded = instr.encode();
        assert_eq!(encoded.as_slice(), &[0xE8, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_codebuffer_forward_jump() {
        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        let label = buf.create_label();

        // JMP with placeholder
        let jmp_offset = buf.emit(X64Inst::Jmp { offset: 0 });
        // The rel32 field starts at jmp_offset + 1 (after the E9 opcode)
        buf.add_reloc(jmp_offset + 1, label, X64RelocKind::Rel32);

        // NOP
        buf.emit(X64Inst::Nop);

        // Bind label
        buf.bind_label(label);
        buf.emit(X64Inst::Ret);

        buf.finalize();

        let code = buf.code();
        // JMP should jump over the NOP to RET
        // reloc at offset 1, target at offset 6
        // disp = 6 - (1 + 4) = 1
        let disp = i32::from_le_bytes([code[1], code[2], code[3], code[4]]);
        assert_eq!(disp, 1);
    }

    #[test]
    fn test_codebuffer_jcc_forward() {
        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        let label = buf.create_label();

        // JE with placeholder
        let jcc_offset = buf.emit(X64Inst::Jcc { offset: 0, cond: Condition::E });
        // rel32 starts after 0F 8x (2 bytes)
        buf.add_reloc(jcc_offset + 2, label, X64RelocKind::Rel32);

        // Two NOPs
        buf.emit(X64Inst::Nop);
        buf.emit(X64Inst::Nop);

        buf.bind_label(label);
        buf.emit(X64Inst::Ret);

        buf.finalize();

        let code = buf.code();
        // JE rel32 = 6 bytes, then 2 NOPs = 2 bytes. Target = offset 8
        // disp = 8 - (2 + 4) = 2
        let disp = i32::from_le_bytes([code[2], code[3], code[4], code[5]]);
        assert_eq!(disp, 2);
    }

    #[test]
    fn test_codebuffer_backward_jump() {
        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        let label = buf.create_label();

        // Bind label at start
        buf.bind_label(label);
        buf.emit(X64Inst::Nop); // offset 0, 1 byte

        // JMP back to label
        let jmp_offset = buf.emit(X64Inst::Jmp { offset: 0 });
        buf.add_reloc(jmp_offset + 1, label, X64RelocKind::Rel32);

        buf.finalize();

        let code = buf.code();
        // reloc at offset 2, target at offset 0
        // disp = 0 - (2 + 4) = -6
        let disp = i32::from_le_bytes([code[2], code[3], code[4], code[5]]);
        assert_eq!(disp, -6);
    }

    #[test]
    fn test_int3() {
        assert_eq!(X64Inst::Int3.encode().as_slice(), &[0xCC]);
    }

    #[test]
    fn test_nop() {
        assert_eq!(X64Inst::Nop.encode().as_slice(), &[0x90]);
    }

    #[test]
    fn test_cqo() {
        assert_eq!(X64Inst::Cqo.encode().as_slice(), &[0x48, 0x99]);
    }

    #[test]
    fn test_mfence() {
        assert_eq!(X64Inst::Mfence.encode().as_slice(), &[0x0F, 0xAE, 0xF0]);
    }

    #[test]
    fn test_mov_ri() {
        // MOV RAX, 0x1234567890ABCDEF
        let instr = X64Inst::MovRI { dest: RAX, imm: 0x1234567890ABCDEFi64 };
        let encoded = instr.encode();
        assert_eq!(encoded.len(), 10);
        assert_eq!(encoded[0], 0x48); // REX.W
        assert_eq!(encoded[1], 0xB8); // B8+0
    }
}

#[cfg(test)]
mod executable_tests {
    use crate::buffer::CodeBuffer;
    use crate::code_memory::{CodeMemory, PagedCodeMemory};

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_arm64_execute_return_42() {
        use crate::arm64::*;
        use crate::arm64::inst::Arm64Inst;

        let mut buf: CodeBuffer<Arm64> = CodeBuffer::new();
        buf.emit(Arm64Inst::movz(X0, 42, 0));
        buf.emit(Arm64Inst::ret());
        buf.finalize();

        let mut mem = PagedCodeMemory::new();
        mem.push(buf.code());
        mem.finalize();
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(), 42);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_arm64_execute_add() {
        use crate::arm64::*;
        use crate::arm64::inst::Arm64Inst;

        let mut buf: CodeBuffer<Arm64> = CodeBuffer::new();
        buf.emit(Arm64Inst::add(X0, X0, X1));
        buf.emit(Arm64Inst::ret());
        buf.finalize();

        let mut mem = PagedCodeMemory::new();
        mem.push(buf.code());
        mem.finalize();
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(10, 32), 42);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_arm64_execute_forward_branch() {
        use crate::arm64::*;
        use crate::arm64::inst::Arm64Inst;

        let mut buf: CodeBuffer<Arm64> = CodeBuffer::new();
        let skip = buf.create_label();

        let cbz_off = buf.emit(Arm64Inst::cbz(X0, 0));
        buf.add_reloc(cbz_off, skip, Arm64RelocKind::Cond19);
        buf.emit(Arm64Inst::ret());

        buf.bind_label(skip);
        buf.emit(Arm64Inst::movz(X0, 99, 0));
        buf.emit(Arm64Inst::ret());
        buf.finalize();

        let mut mem = PagedCodeMemory::new();
        mem.push(buf.code());
        mem.finalize();
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(0), 99);
        assert_eq!(f(5), 5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x64_execute_return_42() {
        use crate::x86_64::*;
        use crate::x86_64::inst::X64Inst;

        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        buf.emit(X64Inst::MovRI32 { dest: RAX, imm: 42 });
        buf.emit(X64Inst::Ret);
        buf.finalize();

        let mut mem = PagedCodeMemory::new();
        mem.push(buf.code());
        mem.finalize();
        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(), 42);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x64_execute_add() {
        use crate::x86_64::*;
        use crate::x86_64::inst::X64Inst;

        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        buf.emit(X64Inst::MovRR { dest: RAX, src: RDI });
        buf.emit(X64Inst::AddRR { dest: RAX, src: RSI });
        buf.emit(X64Inst::Ret);
        buf.finalize();

        let mut mem = PagedCodeMemory::new();
        mem.push(buf.code());
        mem.finalize();
        let f: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(10, 32), 42);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x64_execute_forward_jump() {
        use crate::x86_64::*;
        use crate::x86_64::inst::X64Inst;

        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        let skip = buf.create_label();

        buf.emit(X64Inst::TestRR { a: RDI, b: RDI });
        let jcc_off = buf.emit(X64Inst::Jcc { offset: 0, cond: Condition::E });
        buf.add_reloc(jcc_off + 2, skip, X64RelocKind::Rel32);

        buf.emit(X64Inst::MovRR { dest: RAX, src: RDI });
        buf.emit(X64Inst::Ret);

        buf.bind_label(skip);
        buf.emit(X64Inst::MovRI32 { dest: RAX, imm: 99 });
        buf.emit(X64Inst::Ret);
        buf.finalize();

        let mut mem = PagedCodeMemory::new();
        mem.push(buf.code());
        mem.finalize();
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(0), 99);
        assert_eq!(f(5), 5);
    }
}
