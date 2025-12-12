//! Simple Rust code generator that produces ergonomic functions with u8 registers.
//!
//! This generator produces:
//! 1. Raw instruction encoders with full ARM parameters (in `raw` module)
//! 2. Simple convenience functions for common 64-bit operations
//!
//! Example usage:
//! - `add(rd, rn, rm)` - ADD Xd, Xn, Xm (64-bit register add)
//! - `raw::add_addsub_shift(sf, shift, rm, imm6, rn, rd)` - Full control

use crate::{CodeGenerator, FieldKind, Instruction};

pub struct SimpleRustGenerator;

impl CodeGenerator for SimpleRustGenerator {
    fn generate_prefix(&self) -> String {
        r#"//! ARM64 instruction encoders - generated from official ARM XML spec
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
        imm >= 0 && imm as u32 == masked ||
        imm < 0 && imm as u32 == masked | (u32::MAX << WIDTH),
        "Immediate {} doesn't fit in {} bits", imm, WIDTH
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
    raw::orr_log_shift(1, 0, rm, 0, 31, rd)  // ORR Xd, XZR, Xm
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
    raw::madd(1, rm, 31, rn, rd)  // MADD Xd, Xn, Xm, XZR
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
    raw::orn_log_shift(1, 0, rm, 0, 31, rd)  // ORN Xd, XZR, Xm
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
    raw::subs_addsub_shift(1, 0, rm, 0, rn, 31)  // SUBS XZR, Xn, Xm
}

/// CMP Xn, #imm - 64-bit compare immediate
#[inline]
pub fn cmp_imm(rn: u8, imm: i32) -> u32 {
    raw::subs_addsub_imm(1, 0, imm, rn, 31)  // SUBS XZR, Xn, #imm
}

/// LDR Xt, [Xn, #offset] - load 64-bit from base + unsigned offset
#[inline]
pub fn ldr(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::ldr_imm_gen(0b11, 0, rn, rt, offset / 8, raw::LdrImmGenSelector::UnsignedOffset)
}

/// STR Xt, [Xn, #offset] - store 64-bit to base + unsigned offset
#[inline]
pub fn str(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::str_imm_gen(0b11, 0, rn, rt, offset / 8, raw::StrImmGenSelector::UnsignedOffset)
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
    raw::ldur_gen(0b11, offset, rn, rt)  // size=11 for 64-bit
}

/// STUR Xt, [Xn, #simm9] - store 64-bit to base + signed 9-bit offset (unscaled)
#[inline]
pub fn stur(rt: u8, rn: u8, offset: i32) -> u32 {
    raw::stur_gen(0b11, offset, rn, rt)  // size=11 for 64-bit
}

/// STP Xt1, Xt2, [Xn, #offset]! - store pair with pre-index
#[inline]
pub fn stp_pre(rt: u8, rt2: u8, rn: u8, offset: i32) -> u32 {
    raw::stp_gen(0b10, offset / 8, rt2, rn, rt, raw::StpGenSelector::PreIndex)
}

/// LDP Xt1, Xt2, [Xn], #offset - load pair with post-index
#[inline]
pub fn ldp_post(rt: u8, rt2: u8, rn: u8, offset: i32) -> u32 {
    raw::ldp_gen(0b10, offset / 8, rt2, rn, rt, raw::LdpGenSelector::PostIndex)
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
    raw::fadd_float(0b01, rm, rn, rd)  // ftype=01 for double
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
    raw::csinc(1, 31, (cond ^ 1) as i32, 31, rd)  // sf=1 for 64-bit, invert condition
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

"#
        .to_string()
    }

    fn generate_registers(&self) -> String {
        // No register constants needed - we use plain u8
        String::new()
    }

    fn generate_instruction_enum(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();

        for instruction in instructions {
            // Generate function signature with simplified types
            let function_name = to_snake_case(&instruction.name);
            let mut params = Vec::new();

            // Add instruction-specific parameters with simplified types
            for field in instruction.fields.iter().filter(|f| f.is_arg) {
                let param_type = match &field.kind {
                    FieldKind::Register => "u8",
                    FieldKind::Immediate => "i32",
                    FieldKind::ClassSelector(name) => name,
                    FieldKind::NonPowerOfTwoImm(_) => "i32",
                };
                params.push(format!("{}: {}", field.name, param_type));
            }

            // Generate function documentation
            result.push_str(&format!("/// {}\n", instruction.title));
            if !instruction.description.is_empty() {
                for line in instruction.description.lines() {
                    result.push_str(&format!("/// {}\n", line));
                }
            }
            for comment in &instruction.comments {
                if !comment.is_empty() {
                    result.push_str(&format!("/// {}\n", comment));
                }
            }

            // Mark as inline for performance
            result.push_str("#[inline]\n");

            // Generate function signature
            if params.is_empty() {
                result.push_str(&format!("pub fn {}() -> u32 {{\n", function_name));
            } else {
                result.push_str(&format!(
                    "pub fn {}({}) -> u32 {{\n",
                    function_name,
                    params.join(", ")
                ));
            }

            // Generate function body based on instruction diagrams
            if instruction.diagrams.len() == 1 {
                let diagram = &instruction.diagrams[0];
                let bits = diagram
                    .boxes
                    .iter()
                    .filter_map(|x| x.bits.render())
                    .collect::<Vec<String>>()
                    .join("_");
                result.push_str(&format!("    let mut result = 0b{};\n", bits));

                for armbox in diagram.boxes.iter() {
                    if !armbox.is_arg() {
                        continue;
                    }
                    match armbox.kind() {
                        FieldKind::Register => {
                            result.push_str(&format!(
                                "    result |= ({} as u32) << {};\n",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::Immediate => {
                            result.push_str(&format!(
                                "    result |= ({} as u32) << {};\n",
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                        FieldKind::ClassSelector(_) => {}
                        FieldKind::NonPowerOfTwoImm(n) => {
                            result.push_str(&format!(
                                "    result |= truncate_imm::<{}>({}) << {};\n",
                                n,
                                armbox.name.to_ascii_lowercase(),
                                armbox.shift()
                            ));
                        }
                    }
                }
                result.push_str("    result\n");
            } else {
                // Multiple diagrams - use match on class_selector
                result.push_str("    match class_selector {\n");
                for diagram in instruction.diagrams.iter() {
                    let selector_variant = to_camel_case(&diagram.name);
                    result.push_str(&format!(
                        "        {}Selector::{} => {{\n",
                        instruction.name, selector_variant
                    ));

                    let bits = diagram
                        .boxes
                        .iter()
                        .filter_map(|x| x.bits.render())
                        .collect::<Vec<String>>()
                        .join("_");
                    result.push_str(&format!("            let mut result = 0b{};\n", bits));

                    for armbox in diagram.boxes.iter() {
                        if !armbox.is_arg() {
                            continue;
                        }
                        match armbox.kind() {
                            FieldKind::Register => {
                                result.push_str(&format!(
                                    "            result |= ({} as u32) << {};\n",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::Immediate => {
                                result.push_str(&format!(
                                    "            result |= ({} as u32) << {};\n",
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                            FieldKind::ClassSelector(_) => {}
                            FieldKind::NonPowerOfTwoImm(n) => {
                                result.push_str(&format!(
                                    "            result |= truncate_imm::<{}>({}) << {};\n",
                                    n,
                                    armbox.name.to_ascii_lowercase(),
                                    armbox.shift()
                                ));
                            }
                        }
                    }
                    result.push_str("            result\n");
                    result.push_str("        }\n");
                }
                result.push_str("    }\n");
            }

            result.push_str("}\n\n");
        }

        // Close the raw module
        result.push_str("} // mod raw\n");

        result
    }

    fn generate_encoding_impl(&self, _instructions: &[Instruction]) -> String {
        String::new()
    }

    fn generate_class_selector_enums(&self, instructions: &[Instruction]) -> String {
        let mut result = String::new();

        for instruction in instructions {
            if instruction.diagrams.len() == 1 {
                continue;
            }

            result.push_str("    #[derive(Debug, Copy, Clone, PartialEq, Eq)]\n");
            result.push_str(&format!("    pub enum {}Selector {{\n", instruction.name));
            for diagram in instruction.diagrams.iter() {
                result.push_str(&format!("        {},\n", to_camel_case(&diagram.name)));
            }
            result.push_str("    }\n\n");
        }

        result
    }

    fn generate(&self, instructions: &[Instruction]) -> String {
        let mut output = String::new();
        output.push_str(&self.generate_prefix());
        // Selectors go inside raw module, so put them after the "pub mod raw {" line
        output.push_str(&self.generate_class_selector_enums(instructions));
        output.push_str("\n");
        output.push_str(&self.generate_instruction_enum(instructions));
        output
    }
}

fn to_camel_case(name: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for c in name.chars() {
        if c == '_' || c == '-' || c == ' ' || c == ',' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c.to_ascii_lowercase());
        }
    }

    result
}

fn to_snake_case(name: &str) -> String {
    let mut result = String::new();
    let mut prev_was_lower = false;

    for c in name.chars() {
        if c.is_uppercase() && prev_was_lower {
            result.push('_');
            result.push(c.to_ascii_lowercase());
        } else if c == '-' || c == ' ' || c == ',' {
            result.push('_');
        } else {
            result.push(c.to_ascii_lowercase());
        }
        prev_was_lower = c.is_lowercase();
    }

    result
}
