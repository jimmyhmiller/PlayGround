/// Generate REX prefix.
/// W = 1 for 64-bit operand size
/// R = extension of ModR/M reg field
/// X = extension of SIB index field
/// B = extension of ModR/M r/m field, SIB base field, or opcode reg field
#[inline]
pub fn rex(w: bool, r: bool, x: bool, b: bool) -> u8 {
    0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8)
}

/// Generate REX.W prefix for 64-bit operand size.
#[inline]
pub fn rex_w(reg: u8, rm: u8) -> u8 {
    rex(true, reg >= 8, false, rm >= 8)
}

/// Generate REX prefix without W bit (only if needed for extended registers).
#[inline]
pub fn rex_opt(reg: u8, rm: u8) -> Option<u8> {
    let r = reg >= 8;
    let b = rm >= 8;
    if r || b {
        Some(rex(false, r, false, b))
    } else {
        None
    }
}

/// Generate ModR/M byte.
/// mod: 0b11 for register-register, 0b00 for [reg], 0b01 for [reg+disp8], 0b10 for [reg+disp32]
/// reg: register operand or opcode extension (3 bits)
/// rm: r/m operand (3 bits)
#[inline]
pub fn modrm(mod_: u8, reg: u8, rm: u8) -> u8 {
    (mod_ << 6) | ((reg & 0x7) << 3) | (rm & 0x7)
}

/// Generate SIB byte.
/// scale: 0=1, 1=2, 2=4, 3=8
/// index: index register (3 bits)
/// base: base register (3 bits)
#[inline]
pub fn sib(scale: u8, index: u8, base: u8) -> u8 {
    (scale << 6) | ((index & 0x7) << 3) | (base & 0x7)
}

/// Encode ModRM byte and displacement for a memory operand.
pub fn encode_modrm_mem(bytes: &mut Vec<u8>, reg: u8, base: u8, offset: i32) {
    let base_low = base & 0x7;
    let needs_sib = base_low == 4; // RSP/R12
    let rbp_base = base_low == 5; // RBP/R13

    if offset == 0 && !rbp_base {
        bytes.push(modrm(0b00, reg, base));
        if needs_sib {
            bytes.push(sib(0, 4, base_low));
        }
    } else if (-128..=127).contains(&offset) {
        bytes.push(modrm(0b01, reg, base));
        if needs_sib {
            bytes.push(sib(0, 4, base_low));
        }
        bytes.push(offset as i8 as u8);
    } else {
        bytes.push(modrm(0b10, reg, base));
        if needs_sib {
            bytes.push(sib(0, 4, base_low));
        }
        bytes.extend_from_slice(&offset.to_le_bytes());
    }
}

/// Encode a memory operand with REX.W.
pub fn encode_mem_op(opcode: u8, reg: u8, base: u8, offset: i32, rex_w_needed: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    if rex_w_needed {
        bytes.push(rex_w(reg, base));
    } else if reg >= 8 || base >= 8 {
        bytes.push(rex(false, reg >= 8, false, base >= 8));
    }
    bytes.push(opcode);
    encode_modrm_mem(&mut bytes, reg, base, offset);
    bytes
}

/// Encode a memory operand without automatic REX (for SSE with 0F prefix).
pub fn encode_mem_op_no_rex(opcode: u8, reg: u8, base: u8, offset: i32) -> Vec<u8> {
    let mut bytes = Vec::new();
    if reg >= 8 || base >= 8 {
        bytes.push(rex(false, reg >= 8, false, base >= 8));
    }
    bytes.push(0x0F);
    bytes.push(opcode);
    encode_modrm_mem(&mut bytes, reg, base, offset);
    bytes
}
