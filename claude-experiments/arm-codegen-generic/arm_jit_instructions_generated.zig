const std = @import("std");

pub const Size = enum(u8) {
    S32 = 0,
    S64 = 1,
};

pub const Register = extern struct {
    size: Size,
    index: u8,

    pub fn init(idx: u8, sz: Size) Register {
        return .{ .size = sz, .index = idx };
    }

    pub fn sf(self: Register) u32 {
        return @intFromEnum(self.size);
    }

    pub fn encode(self: Register) u32 {
        return self.index;
    }
};

// Register constants
pub const SP = Register.init(31, .S64);
pub const ZERO_REGISTER = Register.init(31, .S64);


// 64-bit registers
pub const X0 = Register.init(0, .S64);
pub const X1 = Register.init(1, .S64);
pub const X2 = Register.init(2, .S64);
pub const X3 = Register.init(3, .S64);
pub const X4 = Register.init(4, .S64);
pub const X5 = Register.init(5, .S64);
pub const X6 = Register.init(6, .S64);
pub const X7 = Register.init(7, .S64);
pub const X8 = Register.init(8, .S64);
pub const X9 = Register.init(9, .S64);
pub const X10 = Register.init(10, .S64);
pub const X11 = Register.init(11, .S64);
pub const X12 = Register.init(12, .S64);
pub const X13 = Register.init(13, .S64);
pub const X14 = Register.init(14, .S64);
pub const X15 = Register.init(15, .S64);
pub const X16 = Register.init(16, .S64);
pub const X17 = Register.init(17, .S64);
pub const X18 = Register.init(18, .S64);
pub const X19 = Register.init(19, .S64);
pub const X20 = Register.init(20, .S64);
pub const X21 = Register.init(21, .S64);
pub const X22 = Register.init(22, .S64);
pub const X23 = Register.init(23, .S64);
pub const X24 = Register.init(24, .S64);
pub const X25 = Register.init(25, .S64);
pub const X26 = Register.init(26, .S64);
pub const X27 = Register.init(27, .S64);
pub const X28 = Register.init(28, .S64);
pub const X29 = Register.init(29, .S64);
pub const X30 = Register.init(30, .S64);

// 32-bit registers
pub const W0 = Register.init(0, .S32);
pub const W1 = Register.init(1, .S32);
pub const W2 = Register.init(2, .S32);
pub const W3 = Register.init(3, .S32);
pub const W4 = Register.init(4, .S32);
pub const W5 = Register.init(5, .S32);
pub const W6 = Register.init(6, .S32);
pub const W7 = Register.init(7, .S32);
pub const W8 = Register.init(8, .S32);
pub const W9 = Register.init(9, .S32);
pub const W10 = Register.init(10, .S32);
pub const W11 = Register.init(11, .S32);
pub const W12 = Register.init(12, .S32);
pub const W13 = Register.init(13, .S32);
pub const W14 = Register.init(14, .S32);
pub const W15 = Register.init(15, .S32);
pub const W16 = Register.init(16, .S32);
pub const W17 = Register.init(17, .S32);
pub const W18 = Register.init(18, .S32);
pub const W19 = Register.init(19, .S32);
pub const W20 = Register.init(20, .S32);
pub const W21 = Register.init(21, .S32);
pub const W22 = Register.init(22, .S32);
pub const W23 = Register.init(23, .S32);
pub const W24 = Register.init(24, .S32);
pub const W25 = Register.init(25, .S32);
pub const W26 = Register.init(26, .S32);
pub const W27 = Register.init(27, .S32);
pub const W28 = Register.init(28, .S32);
pub const W29 = Register.init(29, .S32);
pub const W30 = Register.init(30, .S32);



pub const StrImmGenSelector = enum(u32) {
    PostIndex,
    PreIndex,
    UnsignedOffset,
};

// Condition codes for conditional branches
pub const Condition = enum(u32) {
    EQ = 0b0000, // Equal
    NE = 0b0001, // Not equal
    CS = 0b0010, // Carry set (unsigned higher or same)
    CC = 0b0011, // Carry clear (unsigned lower)
    MI = 0b0100, // Minus (negative)
    PL = 0b0101, // Plus (positive or zero)
    VS = 0b0110, // Overflow set
    VC = 0b0111, // Overflow clear
    HI = 0b1000, // Unsigned higher
    LS = 0b1001, // Unsigned lower or same
    GE = 0b1010, // Signed greater than or equal
    LT = 0b1011, // Signed less than
    GT = 0b1100, // Signed greater than
    LE = 0b1101, // Signed less than or equal
    AL = 0b1110, // Always
};

// Shift types for shifted register operands
pub const ShiftType = enum(u32) {
    LSL = 0b00, // Logical shift left
    LSR = 0b01, // Logical shift right
    ASR = 0b10, // Arithmetic shift right
    ROR = 0b11, // Rotate right
};

// Extend types for extended register operands
pub const ExtendType = enum(u32) {
    UXTB = 0b000, // Unsigned extend byte
    UXTH = 0b001, // Unsigned extend halfword
    UXTW = 0b010, // Unsigned extend word
    UXTX = 0b011, // Unsigned extend doubleword
    SXTB = 0b100, // Signed extend byte
    SXTH = 0b101, // Signed extend halfword
    SXTW = 0b110, // Signed extend word
    SXTX = 0b111, // Signed extend doubleword
};



/// ADD (immediate) -- A64 -- A64
/// Add (immediate)
/// ADD  <Wd|WSP>, <Wn|WSP>, #<imm>{, <shift>}
/// ADD  <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
pub fn add_addsub_imm(sf: u8, sh: u8, imm12: i12, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00010001000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(sh)))) << 22;
    result |= (@as(u32, @as(u12, @bitCast(imm12)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for add_addsub_imm
export fn add_addsub_imm_c(sf: u8, sh: u8, imm12: i16, rn: Register, rd: Register) u32 {
    return add_addsub_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(sh)), @as(i12, @truncate(imm12)), rn, rd);
}

/// ADD (shifted register) -- A64 -- A64
/// Add (shifted register)
/// ADD  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// ADD  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn add_addsub_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00001011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for add_addsub_shift
export fn add_addsub_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return add_addsub_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
}

/// BL -- A64 -- A64
/// Branch with Link
/// BL  <label>
pub fn bl(imm26: i26) u32 {
    var result: u32 = 0b10010100000000000000000000000000;
    result |= (@as(u32, @as(u26, @bitCast(imm26)))) << 0;
    return result;
}

/// C-compatible wrapper for bl
export fn bl_c(imm26: i32) u32 {
    return bl(@as(i26, @truncate(imm26)));
}

/// LDR (register) -- A64 -- A64
/// Load Register (register)
/// LDR  <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
/// LDR  <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
pub fn ldr_reg_gen(size: u8, rm: Register, option: u8, s: u8, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000011000000000100000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u3, @truncate(option)))) << 13;
    result |= (@as(u32, @as(u1, @truncate(s)))) << 12;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldr_reg_gen
export fn ldr_reg_gen_c(size: u8, rm: Register, option: u8, s: u8, rn: Register, rt: Register) u32 {
    return ldr_reg_gen(@as(u2, @truncate(size)), rm, @as(u3, @truncate(option)), @as(u1, @truncate(s)), rn, rt);
}

/// MOV (to/from SP) -- A64 -- A64
/// MOV  <Wd|WSP>, <Wn|WSP>
/// ADD <Wd|WSP>, <Wn|WSP>, #0
/// MOV  <Xd|SP>, <Xn|SP>
/// ADD <Xd|SP>, <Xn|SP>, #0
pub fn mov_add_addsub_imm(sf: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00010001000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for mov_add_addsub_imm
export fn mov_add_addsub_imm_c(sf: u8, rn: Register, rd: Register) u32 {
    return mov_add_addsub_imm(@as(u1, @truncate(sf)), rn, rd);
}

/// RET -- A64 -- A64
/// Return from subroutine
/// RET  {<Xn>}
pub fn ret(rn: Register) u32 {
    var result: u32 = 0b11010110010111110000000000000000;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for ret
export fn ret_c(rn: Register) u32 {
    return ret(rn);
}

/// STR (immediate) -- A64 -- A64
/// Store Register (immediate)
/// STR  <Wt>, [<Xn|SP>], #<simm>
/// STR  <Xt>, [<Xn|SP>], #<simm>
/// STR  <Wt>, [<Xn|SP>, #<simm>]!
/// STR  <Xt>, [<Xn|SP>, #<simm>]!
/// STR  <Wt>, [<Xn|SP>{, #<pimm>}]
/// STR  <Xt>, [<Xn|SP>{, #<pimm>}]
pub fn str_imm_gen(size: u8, imm9: i9, rn: Register, rt: Register, imm12: i12, class_selector: StrImmGenSelector) u32 {
    return switch (class_selector) {
        .PostIndex => blk: {
            var result: u32 = 0b00111000000000000000010000000000;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
            result |= (@as(u32, @as(u9, @bitCast(imm9)))) << 12;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
        .PreIndex => blk: {
            var result: u32 = 0b00111000000000000000110000000000;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
            result |= (@as(u32, @as(u9, @bitCast(imm9)))) << 12;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
        .UnsignedOffset => blk: {
            var result: u32 = 0b00111001000000000000000000000000;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
            result |= (@as(u32, @as(u12, @bitCast(imm12)))) << 10;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for str_imm_gen
export fn str_imm_gen_c(size: u8, imm9: i16, rn: Register, rt: Register, imm12: i16, class_selector: u32) u32 {
    return str_imm_gen(@as(u2, @truncate(size)), @as(i9, @truncate(imm9)), rn, rt, @as(i12, @truncate(imm12)), @as(StrImmGenSelector, @enumFromInt(class_selector)));
}

/// SUB (immediate) -- A64 -- A64
/// Subtract (immediate)
/// SUB  <Wd|WSP>, <Wn|WSP>, #<imm>{, <shift>}
/// SUB  <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
pub fn sub_addsub_imm(sf: u8, sh: u8, imm12: i12, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01010001000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(sh)))) << 22;
    result |= (@as(u32, @as(u12, @bitCast(imm12)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for sub_addsub_imm
export fn sub_addsub_imm_c(sf: u8, sh: u8, imm12: i16, rn: Register, rd: Register) u32 {
    return sub_addsub_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(sh)), @as(i12, @truncate(imm12)), rn, rd);
}

