//! ARM64 instruction encoders - generated from official ARM XML spec
//!
//! This module provides two levels of API:
//! - Simple functions: `add(rd, rn, rm)` for common 64-bit operations
//! - Raw functions in `raw` module: Full ARM encoding parameters when needed
//!
//! Register parameters are u8 (0-30 for X0-X30, 31 for SP/XZR depending on instruction).

#![allow(clippy::identity_op)]
#![allow(clippy::unusual_byte_groupings)]
#![allow(dead_code)]

/// Truncate an immediate to the specified bit width, asserting no data is lost
#[inline]
fn truncate_imm<const WIDTH: usize>(imm: i32) -> u32 {
    let masked = (imm as u32) & ((1 << WIDTH) - 1);
    // In debug builds, verify we didn't lose significant bits
    debug_assert!(
        imm >= 0 && imm as u32 == masked || imm < 0 && imm as u32 == masked | (u32::MAX << WIDTH),
        "Immediate {} doesn't fit in {} bits",
        imm,
        WIDTH
    );
    masked
}

// =============================================================================
// Simple 64-bit convenience functions
// =============================================================================

/// ADD Xd, Xn, Xm - 64-bit register add
#[inline]
pub fn add(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::add_addsub_shift(1, 0, rm, 0, rn, rd)
}

/// SUB Xd, Xn, Xm - 64-bit register subtract
#[inline]
pub fn sub(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::sub_addsub_shift(1, 0, rm, 0, rn, rd)
}

/// ADD Xd, Xn, #imm - 64-bit immediate add
#[inline]
pub fn add_imm(rd: u8, rn: u8, imm: i32) -> u32 {
    raw::add_addsub_imm(1, 0, imm, rn, rd)
}

/// SUB Xd, Xn, #imm - 64-bit immediate subtract
#[inline]
pub fn sub_imm(rd: u8, rn: u8, imm: i32) -> u32 {
    raw::sub_addsub_imm(1, 0, imm, rn, rd)
}

/// MOV Xd, Xm - 64-bit register move (ORR with XZR)
#[inline]
pub fn mov(rd: u8, rm: u8) -> u32 {
    raw::orr_log_shift(1, 0, rm, 0, 31, rd) // ORR Xd, XZR, Xm
}

/// MOV Xd, SP or MOV SP, Xn - move involving stack pointer
/// Uses ADD Xd, Xn, #0 which treats reg 31 as SP
#[inline]
pub fn mov_sp(rd: u8, rn: u8) -> u32 {
    raw::add_addsub_imm(1, 0, 0, rn, rd)
}

/// MOVZ Xd, #imm16 - move wide immediate (zero other bits)
#[inline]
pub fn movz(rd: u8, imm16: i32, shift: i32) -> u32 {
    raw::movz(1, shift, imm16, rd)
}

/// MOVK Xd, #imm16, LSL #shift - move wide immediate, keep other bits
#[inline]
pub fn movk(rd: u8, imm16: i32, shift: i32) -> u32 {
    raw::movk(1, shift, imm16, rd)
}

/// MUL Xd, Xn, Xm - 64-bit multiply (MADD with XZR)
#[inline]
pub fn mul(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::madd(1, rm, 31, rn, rd) // MADD Xd, Xn, Xm, XZR
}

/// SDIV Xd, Xn, Xm - 64-bit signed divide
#[inline]
pub fn sdiv(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::sdiv(1, rm, rn, rd)
}

/// AND Xd, Xn, Xm - 64-bit bitwise AND
#[inline]
pub fn and(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::and_log_shift(1, 0, rm, 0, rn, rd)
}

/// ORR Xd, Xn, Xm - 64-bit bitwise OR
#[inline]
pub fn orr(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::orr_log_shift(1, 0, rm, 0, rn, rd)
}

/// EOR Xd, Xn, Xm - 64-bit bitwise XOR
#[inline]
pub fn eor(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::eor_log_shift(1, 0, rm, 0, rn, rd)
}

/// MVN Xd, Xm - 64-bit bitwise NOT (ORN with XZR)
#[inline]
pub fn mvn(rd: u8, rm: u8) -> u32 {
    raw::orn_log_shift(1, 0, rm, 0, 31, rd) // ORN Xd, XZR, Xm
}

/// LSL Xd, Xn, Xm - 64-bit logical shift left (variable)
#[inline]
pub fn lsl(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::lslv(1, rm, rn, rd)
}

/// ASR Xd, Xn, Xm - 64-bit arithmetic shift right (variable)
#[inline]
pub fn asr(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::asrv(1, rm, rn, rd)
}

/// LSR Xd, Xn, Xm - 64-bit logical shift right (variable)
#[inline]
pub fn lsr(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::lsrv(1, rm, rn, rd)
}

/// LSL Xd, Xn, #shift - 64-bit logical shift left (immediate)
/// Encoded as UBFM Xd, Xn, #(-shift mod 64), #(63-shift)
#[inline]
pub fn lsl_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    let immr = (64 - shift) & 63;
    let imms = 63 - shift;
    raw::ubfm(1, 1, immr as i32, imms as i32, rn, rd)
}

/// ASR Xd, Xn, #shift - 64-bit arithmetic shift right (immediate)
/// Encoded as SBFM Xd, Xn, #shift, #63
#[inline]
pub fn asr_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    raw::sbfm(1, 1, shift as i32, 63, rn, rd)
}

/// LSR Xd, Xn, #shift - 64-bit logical shift right (immediate)
/// Encoded as UBFM Xd, Xn, #shift, #63
#[inline]
pub fn lsr_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    raw::ubfm(1, 1, shift as i32, 63, rn, rd)
}

/// CMP Xn, Xm - 64-bit compare (SUBS with XZR destination)
#[inline]
pub fn cmp(rn: u8, rm: u8) -> u32 {
    raw::subs_addsub_shift(1, 0, rm, 0, rn, 31) // SUBS XZR, Xn, Xm
}

/// CMP Xn, #imm - 64-bit compare immediate
#[inline]
pub fn cmp_imm(rn: u8, imm: i32) -> u32 {
    raw::subs_addsub_imm(1, 0, imm, rn, 31) // SUBS XZR, Xn, #imm
}

/// LDR Xt, [Xn, #offset] - load 64-bit from base + unsigned offset
#[inline]
pub fn ldr(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::ldr_imm_gen(
        0b11,
        0,
        rn,
        rt,
        offset / 8,
        raw::LdrImmGenSelector::UnsignedOffset,
    )
}

/// STR Xt, [Xn, #offset] - store 64-bit to base + unsigned offset
#[inline]
pub fn str(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::str_imm_gen(
        0b11,
        0,
        rn,
        rt,
        offset / 8,
        raw::StrImmGenSelector::UnsignedOffset,
    )
}

/// LDRB Wt, [Xn, #offset] - load byte from base + unsigned offset
#[inline]
pub fn ldrb(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::ldrb_imm(0, rn, rt, offset, raw::LdrbImmSelector::UnsignedOffset)
}

/// STRB Wt, [Xn, #offset] - store byte to base + unsigned offset
#[inline]
pub fn strb(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::strb_imm(0, rn, rt, offset, raw::StrbImmSelector::UnsignedOffset)
}

/// LDUR Xt, [Xn, #simm9] - load 64-bit from base + signed 9-bit offset (unscaled)
#[inline]
pub fn ldur(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::ldur_gen(0b11, offset, rn, rt) // size=11 for 64-bit
}

/// STUR Xt, [Xn, #simm9] - store 64-bit to base + signed 9-bit offset (unscaled)
#[inline]
pub fn stur(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::stur_gen(0b11, offset, rn, rt) // size=11 for 64-bit
}

/// STR Xt, [Xn, #simm9]! - store 64-bit with pre-index (decrement then store)
#[inline]
pub fn str_pre(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::str_imm_gen(0b11, offset, rn, rt, 0, raw::StrImmGenSelector::PreIndex)
}

/// LDR Xt, [Xn], #simm9 - load 64-bit with post-index (load then increment)
#[inline]
pub fn ldr_post(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::ldr_imm_gen(0b11, offset, rn, rt, 0, raw::LdrImmGenSelector::PostIndex)
}

/// BRK #imm16 - breakpoint instruction
#[inline]
pub fn brk(imm16: u16) -> u32 {
    0xd4200000 | ((imm16 as u32) << 5)
}

/// STP Xt1, Xt2, [Xn, #offset]! - store pair with pre-index
#[inline]
pub fn stp_pre(rt: u8, rt2: u8, rn: u8, offset: i32) -> u32 {
    raw::stp_gen(0b10, offset / 8, rt2, rn, rt, raw::StpGenSelector::PreIndex)
}

/// LDP Xt1, Xt2, [Xn], #offset - load pair with post-index
#[inline]
pub fn ldp_post(rt: u8, rt2: u8, rn: u8, offset: i32) -> u32 {
    raw::ldp_gen(
        0b10,
        offset / 8,
        rt2,
        rn,
        rt,
        raw::LdpGenSelector::PostIndex,
    )
}

/// B label - unconditional branch (offset in instructions)
#[inline]
pub fn b(offset: i32) -> u32 {
    raw::buncond(offset)
}

/// B.cond label - conditional branch (offset in instructions)
/// Condition codes: EQ=0, NE=1, HS/CS=2, LO/CC=3, MI=4, PL=5, VS=6, VC=7,
///                  HI=8, LS=9, GE=10, LT=11, GT=12, LE=13, AL=14
#[inline]
pub fn b_cond(offset: i32, cond: u8) -> u32 {
    raw::bcond(offset, cond as i32)
}

/// BLR Xn - branch with link to register
#[inline]
pub fn blr(rn: u8) -> u32 {
    raw::blr(rn)
}

/// RET - return (branch to X30)
#[inline]
pub fn ret() -> u32 {
    raw::ret(30)
}

/// RET Xn - return to address in Xn
#[inline]
pub fn ret_reg(rn: u8) -> u32 {
    raw::ret(rn)
}

/// FADD Dd, Dn, Dm - double-precision floating-point add
#[inline]
pub fn fadd(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::fadd_float(0b01, rm, rn, rd) // ftype=01 for double
}

/// FSUB Dd, Dn, Dm - double-precision floating-point subtract
#[inline]
pub fn fsub(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::fsub_float(0b01, rm, rn, rd)
}

/// FMUL Dd, Dn, Dm - double-precision floating-point multiply
#[inline]
pub fn fmul(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::fmul_float(0b01, rm, rn, rd)
}

/// FDIV Dd, Dn, Dm - double-precision floating-point divide
#[inline]
pub fn fdiv(rd: u8, rn: u8, rm: u8) -> u32 {
    raw::fdiv_float(0b01, rm, rn, rd)
}

/// FMOV Dd, Xn - move from general register to FP register (as bits)
#[inline]
pub fn fmov_to_fp(rd: u8, rn: u8) -> u32 {
    raw::fmov_float_gen(1, 0b01, 0b00, 0b111, rn, rd)
}

/// FMOV Xd, Dn - move from FP register to general register (as bits)
#[inline]
pub fn fmov_from_fp(rd: u8, rn: u8) -> u32 {
    raw::fmov_float_gen(1, 0b01, 0b00, 0b110, rn, rd)
}

/// SCVTF Dd, Xn - signed 64-bit integer to double
#[inline]
pub fn scvtf(rd: u8, rn: u8) -> u32 {
    raw::scvtf_float_int(1, 0b01, rn, rd)
}

/// ADR Xd, label - form PC-relative address
/// immlo is bits [1:0], immhi is bits [20:2] of the offset
#[inline]
pub fn adr(rd: u8, immlo: i32, immhi: i32) -> u32 {
    raw::adr(immlo, immhi, rd)
}

/// CSET Xd, cond - conditional set (sets to 1 if cond is true, 0 otherwise)
/// This is CSINC Xd, XZR, XZR, invert(cond)
#[inline]
pub fn cset(rd: u8, cond: u8) -> u32 {
    // CSET is an alias for CSINC with Rn=Rm=XZR and inverted condition
    raw::csinc(1, 31, (cond ^ 1) as i32, 31, rd) // sf=1 for 64-bit, invert condition
}

/// CSINC Xd, Xn, Xm, cond - conditional select increment
#[inline]
pub fn csinc(rd: u8, rn: u8, rm: u8, cond: u8) -> u32 {
    raw::csinc(1, rm, cond as i32, rn, rd)
}

/// NOP - no operation
#[inline]
pub fn nop() -> u32 {
    raw::nop()
}

/// AND Xd, Xn, #0b111 - extract lowest 3 bits (for tag extraction)
/// This uses the ARM64 logical immediate encoding for the pattern 0b111
#[inline]
pub fn and_imm_0b111(rd: u8, rn: u8) -> u32 {
    // For 64-bit AND with 0b111: sf=1, N=1, immr=0, imms=2
    // imms=2 means (2+1)=3 consecutive ones = 0b111
    raw::and_log_imm(1, 1, 0, 2, rn, rd)
}

/// ORR Xd, Xn, #0b100 - set bit 2 (for Function tag)
/// This uses the ARM64 logical immediate encoding for the pattern 0b100
#[inline]
pub fn orr_imm_0b100(rd: u8, rn: u8) -> u32 {
    // For 64-bit ORR with 0b100: sf=1, N=1, immr=61, imms=0
    // This encodes a single 1 bit at position 2
    raw::orr_log_imm(1, 1, 61, 0, rn, rd)
}

// =============================================================================
// Raw instruction encoders (full ARM parameters)
// =============================================================================
pub mod raw {
    use super::truncate_imm;

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum LdpGenSelector {
        PostIndex,
        PreIndex,
        SignedOffset,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum LdrImmGenSelector {
        PostIndex,
        PreIndex,
        UnsignedOffset,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum LdrbImmSelector {
        PostIndex,
        PreIndex,
        UnsignedOffset,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum StpGenSelector {
        PostIndex,
        PreIndex,
        SignedOffset,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum StrImmGenSelector {
        PostIndex,
        PreIndex,
        UnsignedOffset,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum StrbImmSelector {
        PostIndex,
        PreIndex,
        UnsignedOffset,
    }

    /// ADD (immediate) -- A64
    /// Add (immediate)
    /// ADD  <Wd|WSP>, <Wn|WSP>, #<imm>{, <shift>}
    /// ADD  <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
    #[inline]
    pub fn add_addsub_imm(sf: i32, sh: i32, imm12: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_100010_0_000000000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (sh as u32) << 22;
        result |= truncate_imm::<12>(imm12) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// ADD (shifted register) -- A64
    /// Add (shifted register)
    /// ADD  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// ADD  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    #[inline]
    pub fn add_addsub_shift(sf: i32, shift: i32, rm: u8, imm6: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_01011_00_0_00000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (shift as u32) << 22;
        result |= (rm as u32) << 16;
        result |= truncate_imm::<6>(imm6) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// ADR -- A64
    /// Form PC-relative address
    /// ADR  <Xd>, <label>
    #[inline]
    pub fn adr(immlo: i32, immhi: i32, rd: u8) -> u32 {
        let mut result = 0b0_00_10000_0000000000000000000_00000;
        result |= (immlo as u32) << 29;
        result |= (immhi as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// AND (immediate) -- A64
    /// Bitwise AND (immediate)
    /// AND  <Wd|WSP>, <Wn>, #<imm>
    /// AND  <Xd|SP>, <Xn>, #<imm>
    #[inline]
    pub fn and_log_imm(sf: i32, n: i32, immr: i32, imms: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_00_100100_0_000000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (n as u32) << 22;
        result |= (immr as u32) << 16;
        result |= (imms as u32) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// AND (shifted register) -- A64
    /// Bitwise AND (shifted register)
    /// AND  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// AND  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    #[inline]
    pub fn and_log_shift(sf: i32, shift: i32, rm: u8, imm6: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_00_01010_00_0_00000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (shift as u32) << 22;
        result |= (rm as u32) << 16;
        result |= truncate_imm::<6>(imm6) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// ASRV -- A64
    /// Arithmetic Shift Right Variable
    /// ASRV  <Wd>, <Wn>, <Wm>
    /// ASRV  <Xd>, <Xn>, <Xm>
    #[inline]
    pub fn asrv(sf: i32, rm: u8, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11010110_00000_0010_10_00000_00000;
        result |= (sf as u32) << 31;
        result |= (rm as u32) << 16;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// B.cond -- A64
    /// Branch conditionally
    /// B.<cond>  <label>
    #[inline]
    pub fn bcond(imm19: i32, cond: i32) -> u32 {
        let mut result = 0b0101010_0_0000000000000000000_0_0000;
        result |= truncate_imm::<19>(imm19) << 5;
        result |= (cond as u32) << 0;
        result
    }

    /// B -- A64
    /// Branch
    /// B  <label>
    #[inline]
    pub fn buncond(imm26: i32) -> u32 {
        let mut result = 0b0_00101_00000000000000000000000000;
        result |= truncate_imm::<26>(imm26) << 0;
        result
    }

    /// BLR -- A64
    /// Branch with Link to Register
    /// BLR  <Xn>
    #[inline]
    pub fn blr(rn: u8) -> u32 {
        let mut result = 0b1101011_0_0_01_11111_0000_0_0_00000_00000;
        result |= (rn as u32) << 5;
        result
    }

    /// CSINC -- A64
    /// Conditional Select Increment
    /// CSINC  <Wd>, <Wn>, <Wm>, <cond>
    /// CSINC  <Xd>, <Xn>, <Xm>, <cond>
    #[inline]
    pub fn csinc(sf: i32, rm: u8, cond: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11010100_00000_0000_0_1_00000_00000;
        result |= (sf as u32) << 31;
        result |= (rm as u32) << 16;
        result |= (cond as u32) << 12;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// EOR (shifted register) -- A64
    /// Bitwise Exclusive OR (shifted register)
    /// EOR  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// EOR  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    #[inline]
    pub fn eor_log_shift(sf: i32, shift: i32, rm: u8, imm6: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_10_01010_00_0_00000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (shift as u32) << 22;
        result |= (rm as u32) << 16;
        result |= truncate_imm::<6>(imm6) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// FADD (scalar) -- A64
    /// Floating-point Add (scalar)
    /// FADD  <Hd>, <Hn>, <Hm>
    /// FADD  <Sd>, <Sn>, <Sm>
    /// FADD  <Dd>, <Dn>, <Dm>
    #[inline]
    pub fn fadd_float(ftype: i32, rm: u8, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11110_00_1_00000_001_0_10_00000_00000;
        result |= (ftype as u32) << 22;
        result |= (rm as u32) << 16;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// FDIV (scalar) -- A64
    /// Floating-point Divide (scalar)
    /// FDIV  <Hd>, <Hn>, <Hm>
    /// FDIV  <Sd>, <Sn>, <Sm>
    /// FDIV  <Dd>, <Dn>, <Dm>
    #[inline]
    pub fn fdiv_float(ftype: i32, rm: u8, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11110_00_1_00000_0001_10_00000_00000;
        result |= (ftype as u32) << 22;
        result |= (rm as u32) << 16;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// FMOV (general) -- A64
    /// Floating-point Move to or from general-purpose register without conversion
    /// FMOV  <Wd>, <Hn>
    /// FMOV  <Xd>, <Hn>
    /// FMOV  <Hd>, <Wn>
    /// FMOV  <Sd>, <Wn>
    /// FMOV  <Wd>, <Sn>
    /// FMOV  <Hd>, <Xn>
    /// FMOV  <Dd>, <Xn>
    /// FMOV  <Vd>.D[1], <Xn>
    /// FMOV  <Xd>, <Dn>
    /// FMOV  <Xd>, <Vn>.D[1]
    #[inline]
    pub fn fmov_float_gen(sf: i32, ftype: i32, rmode: i32, opcode: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11110_00_1_00_000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (ftype as u32) << 22;
        result |= (rmode as u32) << 19;
        result |= (opcode as u32) << 16;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// FMUL (scalar) -- A64
    /// Floating-point Multiply (scalar)
    /// FMUL  <Hd>, <Hn>, <Hm>
    /// FMUL  <Sd>, <Sn>, <Sm>
    /// FMUL  <Dd>, <Dn>, <Dm>
    #[inline]
    pub fn fmul_float(ftype: i32, rm: u8, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11110_00_1_00000_0_000_10_00000_00000;
        result |= (ftype as u32) << 22;
        result |= (rm as u32) << 16;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// FSUB (scalar) -- A64
    /// Floating-point Subtract (scalar)
    /// FSUB  <Hd>, <Hn>, <Hm>
    /// FSUB  <Sd>, <Sn>, <Sm>
    /// FSUB  <Dd>, <Dn>, <Dm>
    #[inline]
    pub fn fsub_float(ftype: i32, rm: u8, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11110_00_1_00000_001_1_10_00000_00000;
        result |= (ftype as u32) << 22;
        result |= (rm as u32) << 16;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// LDP -- A64
    /// Load Pair of Registers
    /// LDP  <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
    /// LDP  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
    /// LDP  <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!
    /// LDP  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
    /// LDP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
    /// LDP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
    #[inline]
    pub fn ldp_gen(
        opc: i32,
        imm7: i32,
        rt2: u8,
        rn: u8,
        rt: u8,
        class_selector: LdpGenSelector,
    ) -> u32 {
        match class_selector {
            LdpGenSelector::PostIndex => {
                let mut result = 0b00_101_0_001_1_0000000_00000_00000_00000;
                result |= (opc as u32) << 30;
                result |= truncate_imm::<7>(imm7) << 15;
                result |= (rt2 as u32) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            LdpGenSelector::PreIndex => {
                let mut result = 0b00_101_0_011_1_0000000_00000_00000_00000;
                result |= (opc as u32) << 30;
                result |= truncate_imm::<7>(imm7) << 15;
                result |= (rt2 as u32) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            LdpGenSelector::SignedOffset => {
                let mut result = 0b00_101_0_010_1_0000000_00000_00000_00000;
                result |= (opc as u32) << 30;
                result |= truncate_imm::<7>(imm7) << 15;
                result |= (rt2 as u32) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
        }
    }

    /// LDR (immediate) -- A64
    /// Load Register (immediate)
    /// LDR  <Wt>, [<Xn|SP>], #<simm>
    /// LDR  <Xt>, [<Xn|SP>], #<simm>
    /// LDR  <Wt>, [<Xn|SP>, #<simm>]!
    /// LDR  <Xt>, [<Xn|SP>, #<simm>]!
    /// LDR  <Wt>, [<Xn|SP>{, #<pimm>}]
    /// LDR  <Xt>, [<Xn|SP>{, #<pimm>}]
    #[inline]
    pub fn ldr_imm_gen(
        size: i32,
        imm9: i32,
        rn: u8,
        rt: u8,
        imm12: i32,
        class_selector: LdrImmGenSelector,
    ) -> u32 {
        match class_selector {
            LdrImmGenSelector::PostIndex => {
                let mut result = 0b00_111_0_00_01_0_000000000_01_00000_00000;
                result |= (size as u32) << 30;
                result |= truncate_imm::<9>(imm9) << 12;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            LdrImmGenSelector::PreIndex => {
                let mut result = 0b00_111_0_00_01_0_000000000_11_00000_00000;
                result |= (size as u32) << 30;
                result |= truncate_imm::<9>(imm9) << 12;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            LdrImmGenSelector::UnsignedOffset => {
                let mut result = 0b00_111_0_01_01_000000000000_00000_00000;
                result |= (size as u32) << 30;
                result |= truncate_imm::<12>(imm12) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
        }
    }

    /// LDRB (immediate) -- A64
    /// Load Register Byte (immediate)
    /// LDRB  <Wt>, [<Xn|SP>], #<simm>
    /// LDRB  <Wt>, [<Xn|SP>, #<simm>]!
    /// LDRB  <Wt>, [<Xn|SP>{, #<pimm>}]
    #[inline]
    pub fn ldrb_imm(imm9: i32, rn: u8, rt: u8, imm12: i32, class_selector: LdrbImmSelector) -> u32 {
        match class_selector {
            LdrbImmSelector::PostIndex => {
                let mut result = 0b00_111_0_00_01_0_000000000_01_00000_00000;
                result |= truncate_imm::<9>(imm9) << 12;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            LdrbImmSelector::PreIndex => {
                let mut result = 0b00_111_0_00_01_0_000000000_11_00000_00000;
                result |= truncate_imm::<9>(imm9) << 12;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            LdrbImmSelector::UnsignedOffset => {
                let mut result = 0b00_111_0_01_01_000000000000_00000_00000;
                result |= truncate_imm::<12>(imm12) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
        }
    }

    /// LDUR -- A64
    /// Load Register (unscaled)
    /// LDUR  <Wt>, [<Xn|SP>{, #<simm>}]
    /// LDUR  <Xt>, [<Xn|SP>{, #<simm>}]
    #[inline]
    pub fn ldur_gen(size: i32, imm9: i32, rn: u8, rt: u8) -> u32 {
        let mut result = 0b00_111_0_00_01_0_000000000_00_00000_00000;
        result |= (size as u32) << 30;
        result |= truncate_imm::<9>(imm9) << 12;
        result |= (rn as u32) << 5;
        result |= (rt as u32) << 0;
        result
    }

    /// LSLV -- A64
    /// Logical Shift Left Variable
    /// LSLV  <Wd>, <Wn>, <Wm>
    /// LSLV  <Xd>, <Xn>, <Xm>
    #[inline]
    pub fn lslv(sf: i32, rm: u8, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11010110_00000_0010_00_00000_00000;
        result |= (sf as u32) << 31;
        result |= (rm as u32) << 16;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// LSRV -- A64
    /// Logical Shift Right Variable
    /// LSRV  <Wd>, <Wn>, <Wm>
    /// LSRV  <Xd>, <Xn>, <Xm>
    #[inline]
    pub fn lsrv(sf: i32, rm: u8, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11010110_00000_0010_01_00000_00000;
        result |= (sf as u32) << 31;
        result |= (rm as u32) << 16;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// MADD -- A64
    /// Multiply-Add
    /// MADD  <Wd>, <Wn>, <Wm>, <Wa>
    /// MADD  <Xd>, <Xn>, <Xm>, <Xa>
    #[inline]
    pub fn madd(sf: i32, rm: u8, ra: u8, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_00_11011_000_00000_0_00000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (rm as u32) << 16;
        result |= (ra as u32) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// MOVK -- A64
    /// Move wide with keep
    /// MOVK  <Wd>, #<imm>{, LSL #<shift>}
    /// MOVK  <Xd>, #<imm>{, LSL #<shift>}
    #[inline]
    pub fn movk(sf: i32, hw: i32, imm16: i32, rd: u8) -> u32 {
        let mut result = 0b0_11_100101_00_0000000000000000_00000;
        result |= (sf as u32) << 31;
        result |= (hw as u32) << 21;
        result |= (imm16 as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// MOVZ -- A64
    /// Move wide with zero
    /// MOVZ  <Wd>, #<imm>{, LSL #<shift>}
    /// MOVZ  <Xd>, #<imm>{, LSL #<shift>}
    #[inline]
    pub fn movz(sf: i32, hw: i32, imm16: i32, rd: u8) -> u32 {
        let mut result = 0b0_10_100101_00_0000000000000000_00000;
        result |= (sf as u32) << 31;
        result |= (hw as u32) << 21;
        result |= (imm16 as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// NOP -- A64
    /// No Operation
    /// NOP
    #[inline]
    pub fn nop() -> u32 {
        let mut result = 0b1101010100_0_00_011_0010_0000_000_11111;
        result
    }

    /// ORN (shifted register) -- A64
    /// Bitwise OR NOT (shifted register)
    /// ORN  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// ORN  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    #[inline]
    pub fn orn_log_shift(sf: i32, shift: i32, rm: u8, imm6: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_01_01010_00_1_00000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (shift as u32) << 22;
        result |= (rm as u32) << 16;
        result |= truncate_imm::<6>(imm6) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// ORR (immediate) -- A64
    /// Bitwise OR (immediate)
    /// ORR  <Wd|WSP>, <Wn>, #<imm>
    /// ORR  <Xd|SP>, <Xn>, #<imm>
    #[inline]
    pub fn orr_log_imm(sf: i32, n: i32, immr: i32, imms: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_01_100100_0_000000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (n as u32) << 22;
        result |= (immr as u32) << 16;
        result |= (imms as u32) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// ORR (shifted register) -- A64
    /// Bitwise OR (shifted register)
    /// ORR  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// ORR  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    #[inline]
    pub fn orr_log_shift(sf: i32, shift: i32, rm: u8, imm6: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_01_01010_00_0_00000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (shift as u32) << 22;
        result |= (rm as u32) << 16;
        result |= truncate_imm::<6>(imm6) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// RET -- A64
    /// Return from subroutine
    /// RET  {<Xn>}
    #[inline]
    pub fn ret(rn: u8) -> u32 {
        let mut result = 0b1101011_0_0_10_11111_0000_0_0_00000_00000;
        result |= (rn as u32) << 5;
        result
    }

    /// SBFM -- A64
    /// Signed Bitfield Move
    /// SBFM  <Wd>, <Wn>, #<immr>, #<imms>
    /// SBFM  <Xd>, <Xn>, #<immr>, #<imms>
    #[inline]
    pub fn sbfm(sf: i32, n: i32, immr: i32, imms: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_00_100110_0_000000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (n as u32) << 22;
        result |= (immr as u32) << 16;
        result |= (imms as u32) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// SCVTF (scalar, integer) -- A64
    /// Signed integer Convert to Floating-point (scalar)
    /// SCVTF  <Hd>, <Wn>
    /// SCVTF  <Sd>, <Wn>
    /// SCVTF  <Dd>, <Wn>
    /// SCVTF  <Hd>, <Xn>
    /// SCVTF  <Sd>, <Xn>
    /// SCVTF  <Dd>, <Xn>
    #[inline]
    pub fn scvtf_float_int(sf: i32, ftype: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11110_00_1_00_010_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (ftype as u32) << 22;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// SDIV -- A64
    /// Signed Divide
    /// SDIV  <Wd>, <Wn>, <Wm>
    /// SDIV  <Xd>, <Xn>, <Xm>
    #[inline]
    pub fn sdiv(sf: i32, rm: u8, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_0_0_11010110_00000_00001_1_00000_00000;
        result |= (sf as u32) << 31;
        result |= (rm as u32) << 16;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// STP -- A64
    /// Store Pair of Registers
    /// STP  <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
    /// STP  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
    /// STP  <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!
    /// STP  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
    /// STP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
    /// STP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
    #[inline]
    pub fn stp_gen(
        opc: i32,
        imm7: i32,
        rt2: u8,
        rn: u8,
        rt: u8,
        class_selector: StpGenSelector,
    ) -> u32 {
        match class_selector {
            StpGenSelector::PostIndex => {
                let mut result = 0b00_101_0_001_0_0000000_00000_00000_00000;
                result |= (opc as u32) << 30;
                result |= truncate_imm::<7>(imm7) << 15;
                result |= (rt2 as u32) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            StpGenSelector::PreIndex => {
                let mut result = 0b00_101_0_011_0_0000000_00000_00000_00000;
                result |= (opc as u32) << 30;
                result |= truncate_imm::<7>(imm7) << 15;
                result |= (rt2 as u32) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            StpGenSelector::SignedOffset => {
                let mut result = 0b00_101_0_010_0_0000000_00000_00000_00000;
                result |= (opc as u32) << 30;
                result |= truncate_imm::<7>(imm7) << 15;
                result |= (rt2 as u32) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
        }
    }

    /// STR (immediate) -- A64
    /// Store Register (immediate)
    /// STR  <Wt>, [<Xn|SP>], #<simm>
    /// STR  <Xt>, [<Xn|SP>], #<simm>
    /// STR  <Wt>, [<Xn|SP>, #<simm>]!
    /// STR  <Xt>, [<Xn|SP>, #<simm>]!
    /// STR  <Wt>, [<Xn|SP>{, #<pimm>}]
    /// STR  <Xt>, [<Xn|SP>{, #<pimm>}]
    #[inline]
    pub fn str_imm_gen(
        size: i32,
        imm9: i32,
        rn: u8,
        rt: u8,
        imm12: i32,
        class_selector: StrImmGenSelector,
    ) -> u32 {
        match class_selector {
            StrImmGenSelector::PostIndex => {
                let mut result = 0b00_111_0_00_00_0_000000000_01_00000_00000;
                result |= (size as u32) << 30;
                result |= truncate_imm::<9>(imm9) << 12;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            StrImmGenSelector::PreIndex => {
                let mut result = 0b00_111_0_00_00_0_000000000_11_00000_00000;
                result |= (size as u32) << 30;
                result |= truncate_imm::<9>(imm9) << 12;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            StrImmGenSelector::UnsignedOffset => {
                let mut result = 0b00_111_0_01_00_000000000000_00000_00000;
                result |= (size as u32) << 30;
                result |= truncate_imm::<12>(imm12) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
        }
    }

    /// STRB (immediate) -- A64
    /// Store Register Byte (immediate)
    /// STRB  <Wt>, [<Xn|SP>], #<simm>
    /// STRB  <Wt>, [<Xn|SP>, #<simm>]!
    /// STRB  <Wt>, [<Xn|SP>{, #<pimm>}]
    #[inline]
    pub fn strb_imm(imm9: i32, rn: u8, rt: u8, imm12: i32, class_selector: StrbImmSelector) -> u32 {
        match class_selector {
            StrbImmSelector::PostIndex => {
                let mut result = 0b00_111_0_00_00_0_000000000_01_00000_00000;
                result |= truncate_imm::<9>(imm9) << 12;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            StrbImmSelector::PreIndex => {
                let mut result = 0b00_111_0_00_00_0_000000000_11_00000_00000;
                result |= truncate_imm::<9>(imm9) << 12;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
            StrbImmSelector::UnsignedOffset => {
                let mut result = 0b00_111_0_01_00_000000000000_00000_00000;
                result |= truncate_imm::<12>(imm12) << 10;
                result |= (rn as u32) << 5;
                result |= (rt as u32) << 0;
                result
            }
        }
    }

    /// STUR -- A64
    /// Store Register (unscaled)
    /// STUR  <Wt>, [<Xn|SP>{, #<simm>}]
    /// STUR  <Xt>, [<Xn|SP>{, #<simm>}]
    #[inline]
    pub fn stur_gen(size: i32, imm9: i32, rn: u8, rt: u8) -> u32 {
        let mut result = 0b00_111_0_00_00_0_000000000_00_00000_00000;
        result |= (size as u32) << 30;
        result |= truncate_imm::<9>(imm9) << 12;
        result |= (rn as u32) << 5;
        result |= (rt as u32) << 0;
        result
    }

    /// SUB (immediate) -- A64
    /// Subtract (immediate)
    /// SUB  <Wd|WSP>, <Wn|WSP>, #<imm>{, <shift>}
    /// SUB  <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
    #[inline]
    pub fn sub_addsub_imm(sf: i32, sh: i32, imm12: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_1_0_100010_0_000000000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (sh as u32) << 22;
        result |= truncate_imm::<12>(imm12) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// SUB (shifted register) -- A64
    /// Subtract (shifted register)
    /// SUB  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// SUB  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    #[inline]
    pub fn sub_addsub_shift(sf: i32, shift: i32, rm: u8, imm6: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_1_0_01011_00_0_00000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (shift as u32) << 22;
        result |= (rm as u32) << 16;
        result |= truncate_imm::<6>(imm6) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// SUBS (immediate) -- A64
    /// Subtract (immediate), setting flags
    /// SUBS  <Wd>, <Wn|WSP>, #<imm>{, <shift>}
    /// SUBS  <Xd>, <Xn|SP>, #<imm>{, <shift>}
    #[inline]
    pub fn subs_addsub_imm(sf: i32, sh: i32, imm12: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_1_1_100010_0_000000000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (sh as u32) << 22;
        result |= truncate_imm::<12>(imm12) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// SUBS (shifted register) -- A64
    /// Subtract (shifted register), setting flags
    /// SUBS  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
    /// SUBS  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
    #[inline]
    pub fn subs_addsub_shift(sf: i32, shift: i32, rm: u8, imm6: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_1_1_01011_00_0_00000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (shift as u32) << 22;
        result |= (rm as u32) << 16;
        result |= truncate_imm::<6>(imm6) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }

    /// UBFM -- A64
    /// Unsigned Bitfield Move
    /// UBFM  <Wd>, <Wn>, #<immr>, #<imms>
    /// UBFM  <Xd>, <Xn>, #<immr>, #<imms>
    #[inline]
    pub fn ubfm(sf: i32, n: i32, immr: i32, imms: i32, rn: u8, rd: u8) -> u32 {
        let mut result = 0b0_10_100110_0_000000_000000_00000_00000;
        result |= (sf as u32) << 31;
        result |= (n as u32) << 22;
        result |= (immr as u32) << 16;
        result |= (imms as u32) << 10;
        result |= (rn as u32) << 5;
        result |= (rd as u32) << 0;
        result
    }
} // mod raw
