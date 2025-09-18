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



pub const AddAdvsimdSelector = enum(u32) {
    Scalar,
    Vector,
};

pub const DsbSelector = enum(u32) {
    MemoryBarrier,
    MemoryNxsBarrier,
};

pub const FaddAdvsimdSelector = enum(u32) {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
};

pub const FmulAdvsimdVecSelector = enum(u32) {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
};

pub const FsubAdvsimdSelector = enum(u32) {
    HalfPrecision,
    SinglePrecisionAndDoublePrecision,
};

pub const LdpGenSelector = enum(u32) {
    PostIndex,
    PreIndex,
    SignedOffset,
};

pub const LdrImmGenSelector = enum(u32) {
    PostIndex,
    PreIndex,
    UnsignedOffset,
};

pub const StpGenSelector = enum(u32) {
    PostIndex,
    PreIndex,
    SignedOffset,
};

pub const StrImmGenSelector = enum(u32) {
    PostIndex,
    PreIndex,
    UnsignedOffset,
};

pub const SubAdvsimdSelector = enum(u32) {
    Scalar,
    Vector,
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



/// ADD (extended register) -- A64 -- A64
/// Add (extended register)
/// ADD  <Wd|WSP>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
/// ADD  <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
pub fn add_addsub_ext(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00001011001000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u3, @truncate(option)))) << 13;
    result |= (@as(u32, @as(u3, @bitCast(@as(i3, @truncate(imm3)))))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for add_addsub_ext
export fn add_addsub_ext_c(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register, rd: Register) u32 {
    return add_addsub_ext(@as(u1, @truncate(sf)), rm, @as(u3, @truncate(option)), @as(i3, @truncate(imm3)), rn, rd);
}

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

/// ADD (vector) -- A64 -- A64
/// Add (vector)
/// ADD  <V><d>, <V><n>, <V><m>
/// ADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
pub fn add_advsimd(size: u8, rm: Register, rn: Register, rd: Register, q: u8, class_selector: AddAdvsimdSelector) u32 {
    return switch (class_selector) {
        .Scalar => blk: {
            var result: u32 = 0b01011110001000001000010000000000;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 22;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
        .Vector => blk: {
            var result: u32 = 0b00001110001000001000010000000000;
            result |= (@as(u32, @as(u1, @truncate(q)))) << 30;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 22;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for add_advsimd
export fn add_advsimd_c(size: u8, rm: Register, rn: Register, rd: Register, q: u8, class_selector: u32) u32 {
    return add_advsimd(@as(u2, @truncate(size)), rm, rn, rd, @as(u1, @truncate(q)), @as(AddAdvsimdSelector, @enumFromInt(class_selector)));
}

/// ADDS (extended register) -- A64 -- A64
/// Add (extended register), setting flags
/// ADDS  <Wd>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
/// ADDS  <Xd>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
pub fn adds_addsub_ext(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00101011001000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u3, @truncate(option)))) << 13;
    result |= (@as(u32, @as(u3, @bitCast(@as(i3, @truncate(imm3)))))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for adds_addsub_ext
export fn adds_addsub_ext_c(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register, rd: Register) u32 {
    return adds_addsub_ext(@as(u1, @truncate(sf)), rm, @as(u3, @truncate(option)), @as(i3, @truncate(imm3)), rn, rd);
}

/// ADDS (immediate) -- A64 -- A64
/// Add (immediate), setting flags
/// ADDS  <Wd>, <Wn|WSP>, #<imm>{, <shift>}
/// ADDS  <Xd>, <Xn|SP>, #<imm>{, <shift>}
pub fn adds_addsub_imm(sf: u8, sh: u8, imm12: i12, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00110001000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(sh)))) << 22;
    result |= (@as(u32, @as(u12, @bitCast(imm12)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for adds_addsub_imm
export fn adds_addsub_imm_c(sf: u8, sh: u8, imm12: i16, rn: Register, rd: Register) u32 {
    return adds_addsub_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(sh)), @as(i12, @truncate(imm12)), rn, rd);
}

/// ADDS (shifted register) -- A64 -- A64
/// Add (shifted register), setting flags
/// ADDS  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// ADDS  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn adds_addsub_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00101011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for adds_addsub_shift
export fn adds_addsub_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return adds_addsub_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
}

/// ADR -- A64 -- A64
/// Form PC-relative address
/// ADR  <Xd>, <label>
pub fn adr(immlo: u8, immhi: u19, rd: Register) u32 {
    var result: u32 = 0b00010000000000000000000000000000;
    result |= (@as(u32, @as(u2, @truncate(immlo)))) << 29;
    result |= (@as(u32, immhi)) << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for adr
export fn adr_c(immlo: u8, immhi: u32, rd: Register) u32 {
    return adr(@as(u2, @truncate(immlo)), @as(u19, @truncate(immhi)), rd);
}

/// ADRP -- A64 -- A64
/// Form PC-relative address to 4KB page
/// ADRP  <Xd>, <label>
pub fn adrp(immlo: u8, immhi: u19, rd: Register) u32 {
    var result: u32 = 0b10010000000000000000000000000000;
    result |= (@as(u32, @as(u2, @truncate(immlo)))) << 29;
    result |= (@as(u32, immhi)) << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for adrp
export fn adrp_c(immlo: u8, immhi: u32, rd: Register) u32 {
    return adrp(@as(u2, @truncate(immlo)), @as(u19, @truncate(immhi)), rd);
}

/// AND (immediate) -- A64 -- A64
/// Bitwise AND (immediate)
/// AND  <Wd|WSP>, <Wn>, #<imm>
/// AND  <Xd|SP>, <Xn>, #<imm>
pub fn and_log_imm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00010010000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for and_log_imm
export fn and_log_imm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return and_log_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// AND (shifted register) -- A64 -- A64
/// Bitwise AND (shifted register)
/// AND  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// AND  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn and_log_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00001010000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for and_log_shift
export fn and_log_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return and_log_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
}

/// ANDS (immediate) -- A64 -- A64
/// Bitwise AND (immediate), setting flags
/// ANDS  <Wd>, <Wn>, #<imm>
/// ANDS  <Xd>, <Xn>, #<imm>
pub fn ands_log_imm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01110010000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for ands_log_imm
export fn ands_log_imm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return ands_log_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// ANDS (shifted register) -- A64 -- A64
/// Bitwise AND (shifted register), setting flags
/// ANDS  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// ANDS  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn ands_log_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01101010000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for ands_log_shift
export fn ands_log_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return ands_log_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
}

/// ASR (register) -- A64 -- A64
/// Arithmetic Shift Right (register)
/// ASR  <Wd>, <Wn>, <Wm>
/// ASRV <Wd>, <Wn>, <Wm>
/// ASR  <Xd>, <Xn>, <Xm>
/// ASRV <Xd>, <Xn>, <Xm>
pub fn asr_asrv(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011010110000000010100000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for asr_asrv
export fn asr_asrv_c(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    return asr_asrv(@as(u1, @truncate(sf)), rm, rn, rd);
}

/// ASR (immediate) -- A64 -- A64
/// Arithmetic Shift Right (immediate)
/// ASR  <Wd>, <Wn>, #<shift>
/// SBFM <Wd>, <Wn>, #<shift>, #31
/// ASR  <Xd>, <Xn>, #<shift>
/// SBFM <Xd>, <Xn>, #<shift>, #63
pub fn asr_sbfm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00010011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for asr_sbfm
export fn asr_sbfm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return asr_sbfm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// BFM -- A64 -- A64
/// Bitfield Move
/// BFM  <Wd>, <Wn>, #<immr>, #<imms>
/// BFM  <Xd>, <Xn>, #<immr>, #<imms>
pub fn bfm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00110011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for bfm
export fn bfm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return bfm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// BIC (shifted register) -- A64 -- A64
/// Bitwise Bit Clear (shifted register)
/// BIC  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// BIC  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn bic_log_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00001010001000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for bic_log_shift
export fn bic_log_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return bic_log_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
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

/// BLR -- A64 -- A64
/// Branch with Link to Register
/// BLR  <Xn>
pub fn blr(rn: Register) u32 {
    var result: u32 = 0b11010110001111110000000000000000;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for blr
export fn blr_c(rn: Register) u32 {
    return blr(rn);
}

/// BR -- A64 -- A64
/// Branch to Register
/// BR  <Xn>
pub fn br(rn: Register) u32 {
    var result: u32 = 0b11010110000111110000000000000000;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for br
export fn br_c(rn: Register) u32 {
    return br(rn);
}

/// BRK -- A64 -- A64
/// Breakpoint instruction
/// BRK  #<imm>
pub fn brk(imm16: u16) u32 {
    var result: u32 = 0b11010100001000000000000000000000;
    result |= (@as(u32, imm16)) << 5;
    return result;
}

/// C-compatible wrapper for brk
export fn brk_c(imm16: u16) u32 {
    return brk(@as(u16, @truncate(imm16)));
}

/// CAS, CASA, CASAL, CASL -- A64 -- A64
/// Compare and Swap word or doubleword in memory
/// CAS  <Ws>, <Wt>, [<Xn|SP>{,#0}]
/// CASA  <Ws>, <Wt>, [<Xn|SP>{,#0}]
/// CASAL  <Ws>, <Wt>, [<Xn|SP>{,#0}]
/// CASL  <Ws>, <Wt>, [<Xn|SP>{,#0}]
/// CAS  <Xs>, <Xt>, [<Xn|SP>{,#0}]
/// CASA  <Xs>, <Xt>, [<Xn|SP>{,#0}]
/// CASAL  <Xs>, <Xt>, [<Xn|SP>{,#0}]
/// CASL  <Xs>, <Xt>, [<Xn|SP>{,#0}]
pub fn cas(size: u8, l: u8, rs: Register, o0: u8, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00001000101000000111110000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(l)))) << 22;
    result |= rs.encode() << 16;
    result |= (@as(u32, @as(u1, @truncate(o0)))) << 15;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for cas
export fn cas_c(size: u8, l: u8, rs: Register, o0: u8, rn: Register, rt: Register) u32 {
    return cas(@as(u2, @truncate(size)), @as(u1, @truncate(l)), rs, @as(u1, @truncate(o0)), rn, rt);
}

/// CASP, CASPA, CASPAL, CASPL -- A64 -- A64
/// Compare and Swap Pair of words or doublewords in memory
/// CASP  <Ws>, <W(s+1)>, <Wt>, <W(t+1)>, [<Xn|SP>{,#0}]
/// CASPA  <Ws>, <W(s+1)>, <Wt>, <W(t+1)>, [<Xn|SP>{,#0}]
/// CASPAL  <Ws>, <W(s+1)>, <Wt>, <W(t+1)>, [<Xn|SP>{,#0}]
/// CASPL  <Ws>, <W(s+1)>, <Wt>, <W(t+1)>, [<Xn|SP>{,#0}]
/// CASP  <Xs>, <X(s+1)>, <Xt>, <X(t+1)>, [<Xn|SP>{,#0}]
/// CASPA  <Xs>, <X(s+1)>, <Xt>, <X(t+1)>, [<Xn|SP>{,#0}]
/// CASPAL  <Xs>, <X(s+1)>, <Xt>, <X(t+1)>, [<Xn|SP>{,#0}]
/// CASPL  <Xs>, <X(s+1)>, <Xt>, <X(t+1)>, [<Xn|SP>{,#0}]
pub fn casp(sz: u8, l: u8, rs: Register, o0: u8, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00001000001000000111110000000000;
    result |= (@as(u32, @as(u1, @truncate(sz)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(l)))) << 22;
    result |= rs.encode() << 16;
    result |= (@as(u32, @as(u1, @truncate(o0)))) << 15;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for casp
export fn casp_c(sz: u8, l: u8, rs: Register, o0: u8, rn: Register, rt: Register) u32 {
    return casp(@as(u1, @truncate(sz)), @as(u1, @truncate(l)), rs, @as(u1, @truncate(o0)), rn, rt);
}

/// CBNZ -- A64 -- A64
/// Compare and Branch on Nonzero
/// CBNZ  <Wt>, <label>
/// CBNZ  <Xt>, <label>
pub fn cbnz(sf: u8, imm19: i19, rt: Register) u32 {
    var result: u32 = 0b00110101000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u19, @bitCast(imm19)))) << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for cbnz
export fn cbnz_c(sf: u8, imm19: i32, rt: Register) u32 {
    return cbnz(@as(u1, @truncate(sf)), @as(i19, @truncate(imm19)), rt);
}

/// CBZ -- A64 -- A64
/// Compare and Branch on Zero
/// CBZ  <Wt>, <label>
/// CBZ  <Xt>, <label>
pub fn cbz(sf: u8, imm19: i19, rt: Register) u32 {
    var result: u32 = 0b00110100000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u19, @bitCast(imm19)))) << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for cbz
export fn cbz_c(sf: u8, imm19: i32, rt: Register) u32 {
    return cbz(@as(u1, @truncate(sf)), @as(i19, @truncate(imm19)), rt);
}

/// CINC -- A64 -- A64
/// Conditional Increment
/// CINC  <Wd>, <Wn>, <cond>
/// CSINC <Wd>, <Wn>, <Wn>, invert(<cond>)
/// CINC  <Xd>, <Xn>, <cond>
/// CSINC <Xd>, <Xn>, <Xn>, invert(<cond>)
pub fn cinc_csinc(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011010100000000000010000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u4, @truncate(cond)))) << 12;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for cinc_csinc
export fn cinc_csinc_c(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    return cinc_csinc(@as(u1, @truncate(sf)), rm, @as(u4, @truncate(cond)), rn, rd);
}

/// CMN (extended register) -- A64 -- A64
/// Compare Negative (extended register)
/// CMN  <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
/// ADDS WZR, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
/// CMN  <Xn|SP>, <R><m>{, <extend> {#<amount>}}
/// ADDS XZR, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
pub fn cmn_adds_addsub_ext(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register) u32 {
    var result: u32 = 0b00101011001000000000000000011111;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u3, @truncate(option)))) << 13;
    result |= (@as(u32, @as(u3, @bitCast(@as(i3, @truncate(imm3)))))) << 10;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for cmn_adds_addsub_ext
export fn cmn_adds_addsub_ext_c(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register) u32 {
    return cmn_adds_addsub_ext(@as(u1, @truncate(sf)), rm, @as(u3, @truncate(option)), @as(i3, @truncate(imm3)), rn);
}

/// CMN (immediate) -- A64 -- A64
/// Compare Negative (immediate)
/// CMN  <Wn|WSP>, #<imm>{, <shift>}
/// ADDS WZR, <Wn|WSP>, #<imm> {, <shift>}
/// CMN  <Xn|SP>, #<imm>{, <shift>}
/// ADDS XZR, <Xn|SP>, #<imm> {, <shift>}
pub fn cmn_adds_addsub_imm(sf: u8, sh: u8, imm12: i12, rn: Register) u32 {
    var result: u32 = 0b00110001000000000000000000011111;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(sh)))) << 22;
    result |= (@as(u32, @as(u12, @bitCast(imm12)))) << 10;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for cmn_adds_addsub_imm
export fn cmn_adds_addsub_imm_c(sf: u8, sh: u8, imm12: i16, rn: Register) u32 {
    return cmn_adds_addsub_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(sh)), @as(i12, @truncate(imm12)), rn);
}

/// CMN (shifted register) -- A64 -- A64
/// Compare Negative (shifted register)
/// CMN  <Wn>, <Wm>{, <shift> #<amount>}
/// ADDS WZR, <Wn>, <Wm> {, <shift> #<amount>}
/// CMN  <Xn>, <Xm>{, <shift> #<amount>}
/// ADDS XZR, <Xn>, <Xm> {, <shift> #<amount>}
pub fn cmn_adds_addsub_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register) u32 {
    var result: u32 = 0b00101011000000000000000000011111;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for cmn_adds_addsub_shift
export fn cmn_adds_addsub_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register) u32 {
    return cmn_adds_addsub_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn);
}

/// CMP (extended register) -- A64 -- A64
/// Compare (extended register)
/// CMP  <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
/// SUBS WZR, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
/// CMP  <Xn|SP>, <R><m>{, <extend> {#<amount>}}
/// SUBS XZR, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
pub fn cmp_subs_addsub_ext(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register) u32 {
    var result: u32 = 0b01101011001000000000000000011111;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u3, @truncate(option)))) << 13;
    result |= (@as(u32, @as(u3, @bitCast(@as(i3, @truncate(imm3)))))) << 10;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for cmp_subs_addsub_ext
export fn cmp_subs_addsub_ext_c(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register) u32 {
    return cmp_subs_addsub_ext(@as(u1, @truncate(sf)), rm, @as(u3, @truncate(option)), @as(i3, @truncate(imm3)), rn);
}

/// CMP (immediate) -- A64 -- A64
/// Compare (immediate)
/// CMP  <Wn|WSP>, #<imm>{, <shift>}
/// SUBS WZR, <Wn|WSP>, #<imm> {, <shift>}
/// CMP  <Xn|SP>, #<imm>{, <shift>}
/// SUBS XZR, <Xn|SP>, #<imm> {, <shift>}
pub fn cmp_subs_addsub_imm(sf: u8, sh: u8, imm12: i12, rn: Register) u32 {
    var result: u32 = 0b01110001000000000000000000011111;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(sh)))) << 22;
    result |= (@as(u32, @as(u12, @bitCast(imm12)))) << 10;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for cmp_subs_addsub_imm
export fn cmp_subs_addsub_imm_c(sf: u8, sh: u8, imm12: i16, rn: Register) u32 {
    return cmp_subs_addsub_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(sh)), @as(i12, @truncate(imm12)), rn);
}

/// CMP (shifted register) -- A64 -- A64
/// Compare (shifted register)
/// CMP  <Wn>, <Wm>{, <shift> #<amount>}
/// SUBS WZR, <Wn>, <Wm> {, <shift> #<amount>}
/// CMP  <Xn>, <Xm>{, <shift> #<amount>}
/// SUBS XZR, <Xn>, <Xm> {, <shift> #<amount>}
pub fn cmp_subs_addsub_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register) u32 {
    var result: u32 = 0b01101011000000000000000000011111;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for cmp_subs_addsub_shift
export fn cmp_subs_addsub_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register) u32 {
    return cmp_subs_addsub_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn);
}

/// CNEG -- A64 -- A64
/// Conditional Negate
/// CNEG  <Wd>, <Wn>, <cond>
/// CSNEG <Wd>, <Wn>, <Wn>, invert(<cond>)
/// CNEG  <Xd>, <Xn>, <cond>
/// CSNEG <Xd>, <Xn>, <Xn>, invert(<cond>)
pub fn cneg_csneg(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01011010100000000000010000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u4, @truncate(cond)))) << 12;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for cneg_csneg
export fn cneg_csneg_c(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    return cneg_csneg(@as(u1, @truncate(sf)), rm, @as(u4, @truncate(cond)), rn, rd);
}

/// CSEL -- A64 -- A64
/// Conditional Select
/// CSEL  <Wd>, <Wn>, <Wm>, <cond>
/// CSEL  <Xd>, <Xn>, <Xm>, <cond>
pub fn csel(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011010100000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u4, @truncate(cond)))) << 12;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for csel
export fn csel_c(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    return csel(@as(u1, @truncate(sf)), rm, @as(u4, @truncate(cond)), rn, rd);
}

/// CSET -- A64 -- A64
/// Conditional Set
/// CSET  <Wd>, <cond>
/// CSINC <Wd>, WZR, WZR, invert(<cond>)
/// CSET  <Xd>, <cond>
/// CSINC <Xd>, XZR, XZR, invert(<cond>)
pub fn cset_csinc(sf: u8, cond: u8, rd: Register) u32 {
    var result: u32 = 0b00011010100111110000011111100000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u4, @truncate(cond)))) << 12;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for cset_csinc
export fn cset_csinc_c(sf: u8, cond: u8, rd: Register) u32 {
    return cset_csinc(@as(u1, @truncate(sf)), @as(u4, @truncate(cond)), rd);
}

/// CSETM -- A64 -- A64
/// Conditional Set Mask
/// CSETM  <Wd>, <cond>
/// CSINV <Wd>, WZR, WZR, invert(<cond>)
/// CSETM  <Xd>, <cond>
/// CSINV <Xd>, XZR, XZR, invert(<cond>)
pub fn csetm_csinv(sf: u8, cond: u8, rd: Register) u32 {
    var result: u32 = 0b01011010100111110000001111100000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u4, @truncate(cond)))) << 12;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for csetm_csinv
export fn csetm_csinv_c(sf: u8, cond: u8, rd: Register) u32 {
    return csetm_csinv(@as(u1, @truncate(sf)), @as(u4, @truncate(cond)), rd);
}

/// CSINC -- A64 -- A64
/// Conditional Select Increment
/// CSINC  <Wd>, <Wn>, <Wm>, <cond>
/// CSINC  <Xd>, <Xn>, <Xm>, <cond>
pub fn csinc(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011010100000000000010000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u4, @truncate(cond)))) << 12;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for csinc
export fn csinc_c(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    return csinc(@as(u1, @truncate(sf)), rm, @as(u4, @truncate(cond)), rn, rd);
}

/// CSINV -- A64 -- A64
/// Conditional Select Invert
/// CSINV  <Wd>, <Wn>, <Wm>, <cond>
/// CSINV  <Xd>, <Xn>, <Xm>, <cond>
pub fn csinv(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01011010100000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u4, @truncate(cond)))) << 12;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for csinv
export fn csinv_c(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    return csinv(@as(u1, @truncate(sf)), rm, @as(u4, @truncate(cond)), rn, rd);
}

/// CSNEG -- A64 -- A64
/// Conditional Select Negation
/// CSNEG  <Wd>, <Wn>, <Wm>, <cond>
/// CSNEG  <Xd>, <Xn>, <Xm>, <cond>
pub fn csneg(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01011010100000000000010000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u4, @truncate(cond)))) << 12;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for csneg
export fn csneg_c(sf: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    return csneg(@as(u1, @truncate(sf)), rm, @as(u4, @truncate(cond)), rn, rd);
}

/// DMB -- A64 -- A64
/// Data Memory Barrier
/// DMB  <option>|#<imm>
pub fn dmb(crm: u8) u32 {
    var result: u32 = 0b11010101000000110011000010111111;
    result |= (@as(u32, @as(u4, @truncate(crm)))) << 8;
    return result;
}

/// C-compatible wrapper for dmb
export fn dmb_c(crm: u8) u32 {
    return dmb(@as(u4, @truncate(crm)));
}

/// DSB -- A64 -- A64
/// Data Synchronization Barrier
/// DSB  <option>|#<imm>
/// DSB  <option>nXS|#<imm>
pub fn dsb(crm: u8, imm2: u8, class_selector: DsbSelector) u32 {
    return switch (class_selector) {
        .MemoryBarrier => blk: {
            var result: u32 = 0b11010101000000110011000010011111;
            result |= (@as(u32, @as(u4, @truncate(crm)))) << 8;
            break :blk result;
        },
        .MemoryNxsBarrier => blk: {
            var result: u32 = 0b11010101000000110011001000111111;
            result |= (@as(u32, @as(u2, @truncate(imm2)))) << 10;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for dsb
export fn dsb_c(crm: u8, imm2: u8, class_selector: u32) u32 {
    return dsb(@as(u4, @truncate(crm)), @as(u2, @truncate(imm2)), @as(DsbSelector, @enumFromInt(class_selector)));
}

/// EOR (immediate) -- A64 -- A64
/// Bitwise Exclusive OR (immediate)
/// EOR  <Wd|WSP>, <Wn>, #<imm>
/// EOR  <Xd|SP>, <Xn>, #<imm>
pub fn eor_log_imm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01010010000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for eor_log_imm
export fn eor_log_imm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return eor_log_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// EOR (shifted register) -- A64 -- A64
/// Bitwise Exclusive OR (shifted register)
/// EOR  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// EOR  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn eor_log_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01001010000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for eor_log_shift
export fn eor_log_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return eor_log_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
}

/// EXTR -- A64 -- A64
/// Extract register
/// EXTR  <Wd>, <Wn>, <Wm>, #<lsb>
/// EXTR  <Xd>, <Xn>, <Xm>, #<lsb>
pub fn extr(sf: u8, n: u8, rm: Register, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00010011100000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for extr
export fn extr_c(sf: u8, n: u8, rm: Register, imms: u8, rn: Register, rd: Register) u32 {
    return extr(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), rm, @as(u6, @truncate(imms)), rn, rd);
}

/// FABS (scalar) -- A64 -- A64
/// Floating-point Absolute value (scalar)
/// FABS  <Hd>, <Hn>
/// FABS  <Sd>, <Sn>
/// FABS  <Dd>, <Dn>
pub fn fabs_float(ftype: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000001100000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fabs_float
export fn fabs_float_c(ftype: u8, rn: Register, rd: Register) u32 {
    return fabs_float(@as(u2, @truncate(ftype)), rn, rd);
}

/// FADD (vector) -- A64 -- A64
/// Floating-point Add (vector)
/// FADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
/// FADD  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
pub fn fadd_advsimd(q: u8, rm: Register, rn: Register, rd: Register, sz: u8, class_selector: FaddAdvsimdSelector) u32 {
    return switch (class_selector) {
        .HalfPrecision => blk: {
            var result: u32 = 0b00001110010000000001010000000000;
            result |= (@as(u32, @as(u1, @truncate(q)))) << 30;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
        .SinglePrecisionAndDoublePrecision => blk: {
            var result: u32 = 0b00001110001000001101010000000000;
            result |= (@as(u32, @as(u1, @truncate(q)))) << 30;
            result |= (@as(u32, @as(u1, @truncate(sz)))) << 22;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for fadd_advsimd
export fn fadd_advsimd_c(q: u8, rm: Register, rn: Register, rd: Register, sz: u8, class_selector: u32) u32 {
    return fadd_advsimd(@as(u1, @truncate(q)), rm, rn, rd, @as(u1, @truncate(sz)), @as(FaddAdvsimdSelector, @enumFromInt(class_selector)));
}

/// FADD (scalar) -- A64 -- A64
/// Floating-point Add (scalar)
/// FADD  <Hd>, <Hn>, <Hm>
/// FADD  <Sd>, <Sn>, <Sm>
/// FADD  <Dd>, <Dn>, <Dm>
pub fn fadd_float(ftype: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000000010100000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fadd_float
export fn fadd_float_c(ftype: u8, rm: Register, rn: Register, rd: Register) u32 {
    return fadd_float(@as(u2, @truncate(ftype)), rm, rn, rd);
}

/// FCMP -- A64 -- A64
/// Floating-point quiet Compare (scalar)
/// FCMP  <Hn>, <Hm>
/// FCMP  <Hn>, #0.0
/// FCMP  <Sn>, <Sm>
/// FCMP  <Sn>, #0.0
/// FCMP  <Dn>, <Dm>
/// FCMP  <Dn>, #0.0
pub fn fcmp_float(ftype: u8, rm: Register, rn: Register, opc: u8) u32 {
    var result: u32 = 0b00011110001000000010000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= (@as(u32, @as(u2, @truncate(opc)))) << 3;
    return result;
}

/// C-compatible wrapper for fcmp_float
export fn fcmp_float_c(ftype: u8, rm: Register, rn: Register, opc: u8) u32 {
    return fcmp_float(@as(u2, @truncate(ftype)), rm, rn, @as(u2, @truncate(opc)));
}

/// FCMPE -- A64 -- A64
/// Floating-point signaling Compare (scalar)
/// FCMPE  <Hn>, <Hm>
/// FCMPE  <Hn>, #0.0
/// FCMPE  <Sn>, <Sm>
/// FCMPE  <Sn>, #0.0
/// FCMPE  <Dn>, <Dm>
/// FCMPE  <Dn>, #0.0
pub fn fcmpe_float(ftype: u8, rm: Register, rn: Register, opc: u8) u32 {
    var result: u32 = 0b00011110001000000010000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= (@as(u32, @as(u2, @truncate(opc)))) << 3;
    return result;
}

/// C-compatible wrapper for fcmpe_float
export fn fcmpe_float_c(ftype: u8, rm: Register, rn: Register, opc: u8) u32 {
    return fcmpe_float(@as(u2, @truncate(ftype)), rm, rn, @as(u2, @truncate(opc)));
}

/// FCSEL -- A64 -- A64
/// Floating-point Conditional Select (scalar)
/// FCSEL  <Hd>, <Hn>, <Hm>, <cond>
/// FCSEL  <Sd>, <Sn>, <Sm>, <cond>
/// FCSEL  <Dd>, <Dn>, <Dm>, <cond>
pub fn fcsel_float(ftype: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000000000110000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u4, @truncate(cond)))) << 12;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fcsel_float
export fn fcsel_float_c(ftype: u8, rm: Register, cond: u8, rn: Register, rd: Register) u32 {
    return fcsel_float(@as(u2, @truncate(ftype)), rm, @as(u4, @truncate(cond)), rn, rd);
}

/// FCVT -- A64 -- A64
/// Floating-point Convert precision (scalar)
/// FCVT  <Sd>, <Hn>
/// FCVT  <Dd>, <Hn>
/// FCVT  <Hd>, <Sn>
/// FCVT  <Dd>, <Sn>
/// FCVT  <Hd>, <Dn>
/// FCVT  <Sd>, <Dn>
pub fn fcvt_float(ftype: u8, opc: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000100100000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= (@as(u32, @as(u2, @truncate(opc)))) << 15;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fcvt_float
export fn fcvt_float_c(ftype: u8, opc: u8, rn: Register, rd: Register) u32 {
    return fcvt_float(@as(u2, @truncate(ftype)), @as(u2, @truncate(opc)), rn, rd);
}

/// FCVTZS (scalar, integer) -- A64 -- A64
/// Floating-point Convert to Signed integer, rounding toward Zero (scalar)
/// FCVTZS  <Wd>, <Hn>
/// FCVTZS  <Xd>, <Hn>
/// FCVTZS  <Wd>, <Sn>
/// FCVTZS  <Xd>, <Sn>
/// FCVTZS  <Wd>, <Dn>
/// FCVTZS  <Xd>, <Dn>
pub fn fcvtzs_float_int(sf: u8, ftype: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001110000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fcvtzs_float_int
export fn fcvtzs_float_int_c(sf: u8, ftype: u8, rn: Register, rd: Register) u32 {
    return fcvtzs_float_int(@as(u1, @truncate(sf)), @as(u2, @truncate(ftype)), rn, rd);
}

/// FCVTZU (scalar, integer) -- A64 -- A64
/// Floating-point Convert to Unsigned integer, rounding toward Zero (scalar)
/// FCVTZU  <Wd>, <Hn>
/// FCVTZU  <Xd>, <Hn>
/// FCVTZU  <Wd>, <Sn>
/// FCVTZU  <Xd>, <Sn>
/// FCVTZU  <Wd>, <Dn>
/// FCVTZU  <Xd>, <Dn>
pub fn fcvtzu_float_int(sf: u8, ftype: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001110010000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fcvtzu_float_int
export fn fcvtzu_float_int_c(sf: u8, ftype: u8, rn: Register, rd: Register) u32 {
    return fcvtzu_float_int(@as(u1, @truncate(sf)), @as(u2, @truncate(ftype)), rn, rd);
}

/// FDIV (scalar) -- A64 -- A64
/// Floating-point Divide (scalar)
/// FDIV  <Hd>, <Hn>, <Hm>
/// FDIV  <Sd>, <Sn>, <Sm>
/// FDIV  <Dd>, <Dn>, <Dm>
pub fn fdiv_float(ftype: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000000001100000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fdiv_float
export fn fdiv_float_c(ftype: u8, rm: Register, rn: Register, rd: Register) u32 {
    return fdiv_float(@as(u2, @truncate(ftype)), rm, rn, rd);
}

/// FMADD -- A64 -- A64
/// Floating-point fused Multiply-Add (scalar)
/// FMADD  <Hd>, <Hn>, <Hm>, <Ha>
/// FMADD  <Sd>, <Sn>, <Sm>, <Sa>
/// FMADD  <Dd>, <Dn>, <Dm>, <Da>
pub fn fmadd_float(ftype: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011111000000000000000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fmadd_float
export fn fmadd_float_c(ftype: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return fmadd_float(@as(u2, @truncate(ftype)), rm, ra, rn, rd);
}

/// FMOV (general) -- A64 -- A64
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
pub fn fmov_float_gen(sf: u8, ftype: u8, rmode: u8, opcode: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= (@as(u32, @as(u2, @truncate(rmode)))) << 19;
    result |= (@as(u32, @as(u3, @truncate(opcode)))) << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fmov_float_gen
export fn fmov_float_gen_c(sf: u8, ftype: u8, rmode: u8, opcode: u8, rn: Register, rd: Register) u32 {
    return fmov_float_gen(@as(u1, @truncate(sf)), @as(u2, @truncate(ftype)), @as(u2, @truncate(rmode)), @as(u3, @truncate(opcode)), rn, rd);
}

/// FMOV (scalar, immediate) -- A64 -- A64
/// Floating-point move immediate (scalar)
/// FMOV  <Hd>, #<imm>
/// FMOV  <Sd>, #<imm>
/// FMOV  <Dd>, #<imm>
pub fn fmov_float_imm(ftype: u8, imm8: u8, rd: Register) u32 {
    var result: u32 = 0b00011110001000000001000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= (@as(u32, imm8)) << 13;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fmov_float_imm
export fn fmov_float_imm_c(ftype: u8, imm8: u8, rd: Register) u32 {
    return fmov_float_imm(@as(u2, @truncate(ftype)), @as(u8, @truncate(imm8)), rd);
}

/// FMSUB -- A64 -- A64
/// Floating-point Fused Multiply-Subtract (scalar)
/// FMSUB  <Hd>, <Hn>, <Hm>, <Ha>
/// FMSUB  <Sd>, <Sn>, <Sm>, <Sa>
/// FMSUB  <Dd>, <Dn>, <Dm>, <Da>
pub fn fmsub_float(ftype: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011111000000001000000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fmsub_float
export fn fmsub_float_c(ftype: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return fmsub_float(@as(u2, @truncate(ftype)), rm, ra, rn, rd);
}

/// FMUL (vector) -- A64 -- A64
/// Floating-point Multiply (vector)
/// FMUL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
/// FMUL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
pub fn fmul_advsimd_vec(q: u8, rm: Register, rn: Register, rd: Register, sz: u8, class_selector: FmulAdvsimdVecSelector) u32 {
    return switch (class_selector) {
        .HalfPrecision => blk: {
            var result: u32 = 0b00101110010000000001110000000000;
            result |= (@as(u32, @as(u1, @truncate(q)))) << 30;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
        .SinglePrecisionAndDoublePrecision => blk: {
            var result: u32 = 0b00101110001000001101110000000000;
            result |= (@as(u32, @as(u1, @truncate(q)))) << 30;
            result |= (@as(u32, @as(u1, @truncate(sz)))) << 22;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for fmul_advsimd_vec
export fn fmul_advsimd_vec_c(q: u8, rm: Register, rn: Register, rd: Register, sz: u8, class_selector: u32) u32 {
    return fmul_advsimd_vec(@as(u1, @truncate(q)), rm, rn, rd, @as(u1, @truncate(sz)), @as(FmulAdvsimdVecSelector, @enumFromInt(class_selector)));
}

/// FMUL (scalar) -- A64 -- A64
/// Floating-point Multiply (scalar)
/// FMUL  <Hd>, <Hn>, <Hm>
/// FMUL  <Sd>, <Sn>, <Sm>
/// FMUL  <Dd>, <Dn>, <Dm>
pub fn fmul_float(ftype: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000000000100000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fmul_float
export fn fmul_float_c(ftype: u8, rm: Register, rn: Register, rd: Register) u32 {
    return fmul_float(@as(u2, @truncate(ftype)), rm, rn, rd);
}

/// FNEG (scalar) -- A64 -- A64
/// Floating-point Negate (scalar)
/// FNEG  <Hd>, <Hn>
/// FNEG  <Sd>, <Sn>
/// FNEG  <Dd>, <Dn>
pub fn fneg_float(ftype: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000010100000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fneg_float
export fn fneg_float_c(ftype: u8, rn: Register, rd: Register) u32 {
    return fneg_float(@as(u2, @truncate(ftype)), rn, rd);
}

/// FNMADD -- A64 -- A64
/// Floating-point Negated fused Multiply-Add (scalar)
/// FNMADD  <Hd>, <Hn>, <Hm>, <Ha>
/// FNMADD  <Sd>, <Sn>, <Sm>, <Sa>
/// FNMADD  <Dd>, <Dn>, <Dm>, <Da>
pub fn fnmadd_float(ftype: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011111001000000000000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fnmadd_float
export fn fnmadd_float_c(ftype: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return fnmadd_float(@as(u2, @truncate(ftype)), rm, ra, rn, rd);
}

/// FNMSUB -- A64 -- A64
/// Floating-point Negated fused Multiply-Subtract (scalar)
/// FNMSUB  <Hd>, <Hn>, <Hm>, <Ha>
/// FNMSUB  <Sd>, <Sn>, <Sm>, <Sa>
/// FNMSUB  <Dd>, <Dn>, <Dm>, <Da>
pub fn fnmsub_float(ftype: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011111001000001000000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fnmsub_float
export fn fnmsub_float_c(ftype: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return fnmsub_float(@as(u2, @truncate(ftype)), rm, ra, rn, rd);
}

/// FSQRT (scalar) -- A64 -- A64
/// Floating-point Square Root (scalar)
/// FSQRT  <Hd>, <Hn>
/// FSQRT  <Sd>, <Sn>
/// FSQRT  <Dd>, <Dn>
pub fn fsqrt_float(ftype: u8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000011100000000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fsqrt_float
export fn fsqrt_float_c(ftype: u8, rn: Register, rd: Register) u32 {
    return fsqrt_float(@as(u2, @truncate(ftype)), rn, rd);
}

/// FSUB (vector) -- A64 -- A64
/// Floating-point Subtract (vector)
/// FSUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
/// FSUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
pub fn fsub_advsimd(q: u8, rm: Register, rn: Register, rd: Register, sz: u8, class_selector: FsubAdvsimdSelector) u32 {
    return switch (class_selector) {
        .HalfPrecision => blk: {
            var result: u32 = 0b00001110110000000001010000000000;
            result |= (@as(u32, @as(u1, @truncate(q)))) << 30;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
        .SinglePrecisionAndDoublePrecision => blk: {
            var result: u32 = 0b00001110101000001101010000000000;
            result |= (@as(u32, @as(u1, @truncate(q)))) << 30;
            result |= (@as(u32, @as(u1, @truncate(sz)))) << 22;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for fsub_advsimd
export fn fsub_advsimd_c(q: u8, rm: Register, rn: Register, rd: Register, sz: u8, class_selector: u32) u32 {
    return fsub_advsimd(@as(u1, @truncate(q)), rm, rn, rd, @as(u1, @truncate(sz)), @as(FsubAdvsimdSelector, @enumFromInt(class_selector)));
}

/// FSUB (scalar) -- A64 -- A64
/// Floating-point Subtract (scalar)
/// FSUB  <Hd>, <Hn>, <Hm>
/// FSUB  <Sd>, <Sn>, <Sm>
/// FSUB  <Dd>, <Dn>, <Dm>
pub fn fsub_float(ftype: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011110001000000011100000000000;
    result |= (@as(u32, @as(u2, @truncate(ftype)))) << 22;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for fsub_float
export fn fsub_float_c(ftype: u8, rm: Register, rn: Register, rd: Register) u32 {
    return fsub_float(@as(u2, @truncate(ftype)), rm, rn, rd);
}

/// HLT -- A64 -- A64
/// Halt instruction
/// HLT  #<imm>
pub fn hlt(imm16: u16) u32 {
    var result: u32 = 0b11010100010000000000000000000000;
    result |= (@as(u32, imm16)) << 5;
    return result;
}

/// C-compatible wrapper for hlt
export fn hlt_c(imm16: u16) u32 {
    return hlt(@as(u16, @truncate(imm16)));
}

/// HVC -- A64 -- A64
/// Hypervisor Call
/// HVC  #<imm>
pub fn hvc(imm16: u16) u32 {
    var result: u32 = 0b11010100000000000000000000000010;
    result |= (@as(u32, imm16)) << 5;
    return result;
}

/// C-compatible wrapper for hvc
export fn hvc_c(imm16: u16) u32 {
    return hvc(@as(u16, @truncate(imm16)));
}

/// ISB -- A64 -- A64
/// Instruction Synchronization Barrier
/// ISB  {<option>|#<imm>}
pub fn isb(crm: u8) u32 {
    var result: u32 = 0b11010101000000110011000011011111;
    result |= (@as(u32, @as(u4, @truncate(crm)))) << 8;
    return result;
}

/// C-compatible wrapper for isb
export fn isb_c(crm: u8) u32 {
    return isb(@as(u4, @truncate(crm)));
}

/// LDADD, LDADDA, LDADDAL, LDADDL -- A64 -- A64
/// Atomic add on word or doubleword in memory
/// LDADD  <Ws>, <Wt>, [<Xn|SP>]
/// LDADDA  <Ws>, <Wt>, [<Xn|SP>]
/// LDADDAL  <Ws>, <Wt>, [<Xn|SP>]
/// LDADDL  <Ws>, <Wt>, [<Xn|SP>]
/// LDADD  <Xs>, <Xt>, [<Xn|SP>]
/// LDADDA  <Xs>, <Xt>, [<Xn|SP>]
/// LDADDAL  <Xs>, <Xt>, [<Xn|SP>]
/// LDADDL  <Xs>, <Xt>, [<Xn|SP>]
pub fn ldadd(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000001000000000000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(a)))) << 23;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldadd
export fn ldadd_c(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    return ldadd(@as(u2, @truncate(size)), @as(u1, @truncate(a)), r, rs, rn, rt);
}

/// LDAR -- A64 -- A64
/// Load-Acquire Register
/// LDAR  <Wt>, [<Xn|SP>{,#0}]
/// LDAR  <Xt>, [<Xn|SP>{,#0}]
pub fn ldar(size: u8, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00001000110111111111110000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldar
export fn ldar_c(size: u8, rn: Register, rt: Register) u32 {
    return ldar(@as(u2, @truncate(size)), rn, rt);
}

/// LDCLR, LDCLRA, LDCLRAL, LDCLRL -- A64 -- A64
/// Atomic bit clear on word or doubleword in memory
/// LDCLR  <Ws>, <Wt>, [<Xn|SP>]
/// LDCLRA  <Ws>, <Wt>, [<Xn|SP>]
/// LDCLRAL  <Ws>, <Wt>, [<Xn|SP>]
/// LDCLRL  <Ws>, <Wt>, [<Xn|SP>]
/// LDCLR  <Xs>, <Xt>, [<Xn|SP>]
/// LDCLRA  <Xs>, <Xt>, [<Xn|SP>]
/// LDCLRAL  <Xs>, <Xt>, [<Xn|SP>]
/// LDCLRL  <Xs>, <Xt>, [<Xn|SP>]
pub fn ldclr(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000001000000001000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(a)))) << 23;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldclr
export fn ldclr_c(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    return ldclr(@as(u2, @truncate(size)), @as(u1, @truncate(a)), r, rs, rn, rt);
}

/// LDEOR, LDEORA, LDEORAL, LDEORL -- A64 -- A64
/// Atomic exclusive OR on word or doubleword in memory
/// LDEOR  <Ws>, <Wt>, [<Xn|SP>]
/// LDEORA  <Ws>, <Wt>, [<Xn|SP>]
/// LDEORAL  <Ws>, <Wt>, [<Xn|SP>]
/// LDEORL  <Ws>, <Wt>, [<Xn|SP>]
/// LDEOR  <Xs>, <Xt>, [<Xn|SP>]
/// LDEORA  <Xs>, <Xt>, [<Xn|SP>]
/// LDEORAL  <Xs>, <Xt>, [<Xn|SP>]
/// LDEORL  <Xs>, <Xt>, [<Xn|SP>]
pub fn ldeor(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000001000000010000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(a)))) << 23;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldeor
export fn ldeor_c(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    return ldeor(@as(u2, @truncate(size)), @as(u1, @truncate(a)), r, rs, rn, rt);
}

/// LDP -- A64 -- A64
/// Load Pair of Registers
/// LDP  <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
/// LDP  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
/// LDP  <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!
/// LDP  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
/// LDP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
/// LDP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
pub fn ldp_gen(opc: u8, imm7: i7, rt2: Register, rn: Register, rt: Register, class_selector: LdpGenSelector) u32 {
    return switch (class_selector) {
        .PostIndex => blk: {
            var result: u32 = 0b00101000110000000000000000000000;
            result |= (@as(u32, @as(u2, @truncate(opc)))) << 30;
            result |= (@as(u32, @as(u7, @bitCast(imm7)))) << 15;
            result |= rt2.encode() << 10;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
        .PreIndex => blk: {
            var result: u32 = 0b00101001110000000000000000000000;
            result |= (@as(u32, @as(u2, @truncate(opc)))) << 30;
            result |= (@as(u32, @as(u7, @bitCast(imm7)))) << 15;
            result |= rt2.encode() << 10;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
        .SignedOffset => blk: {
            var result: u32 = 0b00101001010000000000000000000000;
            result |= (@as(u32, @as(u2, @truncate(opc)))) << 30;
            result |= (@as(u32, @as(u7, @bitCast(imm7)))) << 15;
            result |= rt2.encode() << 10;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for ldp_gen
export fn ldp_gen_c(opc: u8, imm7: i8, rt2: Register, rn: Register, rt: Register, class_selector: u32) u32 {
    return ldp_gen(@as(u2, @truncate(opc)), @as(i7, @truncate(imm7)), rt2, rn, rt, @as(LdpGenSelector, @enumFromInt(class_selector)));
}

/// LDR (immediate) -- A64 -- A64
/// Load Register (immediate)
/// LDR  <Wt>, [<Xn|SP>], #<simm>
/// LDR  <Xt>, [<Xn|SP>], #<simm>
/// LDR  <Wt>, [<Xn|SP>, #<simm>]!
/// LDR  <Xt>, [<Xn|SP>, #<simm>]!
/// LDR  <Wt>, [<Xn|SP>{, #<pimm>}]
/// LDR  <Xt>, [<Xn|SP>{, #<pimm>}]
pub fn ldr_imm_gen(size: u8, imm9: i9, rn: Register, rt: Register, imm12: i12, class_selector: LdrImmGenSelector) u32 {
    return switch (class_selector) {
        .PostIndex => blk: {
            var result: u32 = 0b00111000010000000000010000000000;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
            result |= (@as(u32, @as(u9, @bitCast(imm9)))) << 12;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
        .PreIndex => blk: {
            var result: u32 = 0b00111000010000000000110000000000;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
            result |= (@as(u32, @as(u9, @bitCast(imm9)))) << 12;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
        .UnsignedOffset => blk: {
            var result: u32 = 0b00111001010000000000000000000000;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
            result |= (@as(u32, @as(u12, @bitCast(imm12)))) << 10;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for ldr_imm_gen
export fn ldr_imm_gen_c(size: u8, imm9: i16, rn: Register, rt: Register, imm12: i16, class_selector: u32) u32 {
    return ldr_imm_gen(@as(u2, @truncate(size)), @as(i9, @truncate(imm9)), rn, rt, @as(i12, @truncate(imm12)), @as(LdrImmGenSelector, @enumFromInt(class_selector)));
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

/// LDSET, LDSETA, LDSETAL, LDSETL -- A64 -- A64
/// Atomic bit set on word or doubleword in memory
/// LDSET  <Ws>, <Wt>, [<Xn|SP>]
/// LDSETA  <Ws>, <Wt>, [<Xn|SP>]
/// LDSETAL  <Ws>, <Wt>, [<Xn|SP>]
/// LDSETL  <Ws>, <Wt>, [<Xn|SP>]
/// LDSET  <Xs>, <Xt>, [<Xn|SP>]
/// LDSETA  <Xs>, <Xt>, [<Xn|SP>]
/// LDSETAL  <Xs>, <Xt>, [<Xn|SP>]
/// LDSETL  <Xs>, <Xt>, [<Xn|SP>]
pub fn ldset(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000001000000011000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(a)))) << 23;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldset
export fn ldset_c(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    return ldset(@as(u2, @truncate(size)), @as(u1, @truncate(a)), r, rs, rn, rt);
}

/// LDSMAX, LDSMAXA, LDSMAXAL, LDSMAXL -- A64 -- A64
/// Atomic signed maximum on word or doubleword in memory
/// LDSMAX  <Ws>, <Wt>, [<Xn|SP>]
/// LDSMAXA  <Ws>, <Wt>, [<Xn|SP>]
/// LDSMAXAL  <Ws>, <Wt>, [<Xn|SP>]
/// LDSMAXL  <Ws>, <Wt>, [<Xn|SP>]
/// LDSMAX  <Xs>, <Xt>, [<Xn|SP>]
/// LDSMAXA  <Xs>, <Xt>, [<Xn|SP>]
/// LDSMAXAL  <Xs>, <Xt>, [<Xn|SP>]
/// LDSMAXL  <Xs>, <Xt>, [<Xn|SP>]
pub fn ldsmax(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000001000000100000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(a)))) << 23;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldsmax
export fn ldsmax_c(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    return ldsmax(@as(u2, @truncate(size)), @as(u1, @truncate(a)), r, rs, rn, rt);
}

/// LDSMIN, LDSMINA, LDSMINAL, LDSMINL -- A64 -- A64
/// Atomic signed minimum on word or doubleword in memory
/// LDSMIN  <Ws>, <Wt>, [<Xn|SP>]
/// LDSMINA  <Ws>, <Wt>, [<Xn|SP>]
/// LDSMINAL  <Ws>, <Wt>, [<Xn|SP>]
/// LDSMINL  <Ws>, <Wt>, [<Xn|SP>]
/// LDSMIN  <Xs>, <Xt>, [<Xn|SP>]
/// LDSMINA  <Xs>, <Xt>, [<Xn|SP>]
/// LDSMINAL  <Xs>, <Xt>, [<Xn|SP>]
/// LDSMINL  <Xs>, <Xt>, [<Xn|SP>]
pub fn ldsmin(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000001000000101000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(a)))) << 23;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldsmin
export fn ldsmin_c(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    return ldsmin(@as(u2, @truncate(size)), @as(u1, @truncate(a)), r, rs, rn, rt);
}

/// LDUMAX, LDUMAXA, LDUMAXAL, LDUMAXL -- A64 -- A64
/// Atomic unsigned maximum on word or doubleword in memory
/// LDUMAX  <Ws>, <Wt>, [<Xn|SP>]
/// LDUMAXA  <Ws>, <Wt>, [<Xn|SP>]
/// LDUMAXAL  <Ws>, <Wt>, [<Xn|SP>]
/// LDUMAXL  <Ws>, <Wt>, [<Xn|SP>]
/// LDUMAX  <Xs>, <Xt>, [<Xn|SP>]
/// LDUMAXA  <Xs>, <Xt>, [<Xn|SP>]
/// LDUMAXAL  <Xs>, <Xt>, [<Xn|SP>]
/// LDUMAXL  <Xs>, <Xt>, [<Xn|SP>]
pub fn ldumax(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000001000000110000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(a)))) << 23;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldumax
export fn ldumax_c(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    return ldumax(@as(u2, @truncate(size)), @as(u1, @truncate(a)), r, rs, rn, rt);
}

/// LDUMIN, LDUMINA, LDUMINAL, LDUMINL -- A64 -- A64
/// Atomic unsigned minimum on word or doubleword in memory
/// LDUMIN  <Ws>, <Wt>, [<Xn|SP>]
/// LDUMINA  <Ws>, <Wt>, [<Xn|SP>]
/// LDUMINAL  <Ws>, <Wt>, [<Xn|SP>]
/// LDUMINL  <Ws>, <Wt>, [<Xn|SP>]
/// LDUMIN  <Xs>, <Xt>, [<Xn|SP>]
/// LDUMINA  <Xs>, <Xt>, [<Xn|SP>]
/// LDUMINAL  <Xs>, <Xt>, [<Xn|SP>]
/// LDUMINL  <Xs>, <Xt>, [<Xn|SP>]
pub fn ldumin(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000001000000111000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u1, @truncate(a)))) << 23;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldumin
export fn ldumin_c(size: u8, a: u8, r: Register, rs: Register, rn: Register, rt: Register) u32 {
    return ldumin(@as(u2, @truncate(size)), @as(u1, @truncate(a)), r, rs, rn, rt);
}

/// LDUR -- A64 -- A64
/// Load Register (unscaled)
/// LDUR  <Wt>, [<Xn|SP>{, #<simm>}]
/// LDUR  <Xt>, [<Xn|SP>{, #<simm>}]
pub fn ldur_gen(size: u8, imm9: i9, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000010000000000000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u9, @bitCast(imm9)))) << 12;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for ldur_gen
export fn ldur_gen_c(size: u8, imm9: i16, rn: Register, rt: Register) u32 {
    return ldur_gen(@as(u2, @truncate(size)), @as(i9, @truncate(imm9)), rn, rt);
}

/// LSL (register) -- A64 -- A64
/// Logical Shift Left (register)
/// LSL  <Wd>, <Wn>, <Wm>
/// LSLV <Wd>, <Wn>, <Wm>
/// LSL  <Xd>, <Xn>, <Xm>
/// LSLV <Xd>, <Xn>, <Xm>
pub fn lsl_lslv(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011010110000000010000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for lsl_lslv
export fn lsl_lslv_c(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    return lsl_lslv(@as(u1, @truncate(sf)), rm, rn, rd);
}

/// LSL (immediate) -- A64 -- A64
/// Logical Shift Left (immediate)
/// LSL  <Wd>, <Wn>, #<shift>
/// UBFM <Wd>, <Wn>, #(-<shift> MOD 32), #(31-<shift>)
/// LSL  <Xd>, <Xn>, #<shift>
/// UBFM <Xd>, <Xn>, #(-<shift> MOD 64), #(63-<shift>)
pub fn lsl_ubfm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01010011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for lsl_ubfm
export fn lsl_ubfm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return lsl_ubfm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// LSR (register) -- A64 -- A64
/// Logical Shift Right (register)
/// LSR  <Wd>, <Wn>, <Wm>
/// LSRV <Wd>, <Wn>, <Wm>
/// LSR  <Xd>, <Xn>, <Xm>
/// LSRV <Xd>, <Xn>, <Xm>
pub fn lsr_lsrv(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011010110000000010010000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for lsr_lsrv
export fn lsr_lsrv_c(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    return lsr_lsrv(@as(u1, @truncate(sf)), rm, rn, rd);
}

/// LSR (immediate) -- A64 -- A64
/// Logical Shift Right (immediate)
/// LSR  <Wd>, <Wn>, #<shift>
/// UBFM <Wd>, <Wn>, #<shift>, #31
/// LSR  <Xd>, <Xn>, #<shift>
/// UBFM <Xd>, <Xn>, #<shift>, #63
pub fn lsr_ubfm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01010011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for lsr_ubfm
export fn lsr_ubfm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return lsr_ubfm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// MADD -- A64 -- A64
/// Multiply-Add
/// MADD  <Wd>, <Wn>, <Wm>, <Wa>
/// MADD  <Xd>, <Xn>, <Xm>, <Xa>
pub fn madd(sf: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for madd
export fn madd_c(sf: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return madd(@as(u1, @truncate(sf)), rm, ra, rn, rd);
}

/// MNEG -- A64 -- A64
/// Multiply-Negate
/// MNEG  <Wd>, <Wn>, <Wm>
/// MSUB <Wd>, <Wn>, <Wm>, WZR
/// MNEG  <Xd>, <Xn>, <Xm>
/// MSUB <Xd>, <Xn>, <Xm>, XZR
pub fn mneg_msub(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011011000000001111110000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for mneg_msub
export fn mneg_msub_c(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    return mneg_msub(@as(u1, @truncate(sf)), rm, rn, rd);
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

/// MOV (register) -- A64 -- A64
/// Move (register)
/// MOV  <Wd>, <Wm>
/// ORR <Wd>, WZR, <Wm>
/// MOV  <Xd>, <Xm>
/// ORR <Xd>, XZR, <Xm>
pub fn mov_orr_log_shift(sf: u8, rm: Register, rd: Register) u32 {
    var result: u32 = 0b00101010000000000000001111100000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for mov_orr_log_shift
export fn mov_orr_log_shift_c(sf: u8, rm: Register, rd: Register) u32 {
    return mov_orr_log_shift(@as(u1, @truncate(sf)), rm, rd);
}

/// MOVK -- A64 -- A64
/// Move wide with keep
/// MOVK  <Wd>, #<imm>{, LSL #<shift>}
/// MOVK  <Xd>, #<imm>{, LSL #<shift>}
pub fn movk(sf: u8, hw: u8, imm16: u16, rd: Register) u32 {
    var result: u32 = 0b01110010100000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(hw)))) << 21;
    result |= (@as(u32, imm16)) << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for movk
export fn movk_c(sf: u8, hw: u8, imm16: u16, rd: Register) u32 {
    return movk(@as(u1, @truncate(sf)), @as(u2, @truncate(hw)), @as(u16, @truncate(imm16)), rd);
}

/// MOVN -- A64 -- A64
/// Move wide with NOT
/// MOVN  <Wd>, #<imm>{, LSL #<shift>}
/// MOVN  <Xd>, #<imm>{, LSL #<shift>}
pub fn movn(sf: u8, hw: u8, imm16: u16, rd: Register) u32 {
    var result: u32 = 0b00010010100000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(hw)))) << 21;
    result |= (@as(u32, imm16)) << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for movn
export fn movn_c(sf: u8, hw: u8, imm16: u16, rd: Register) u32 {
    return movn(@as(u1, @truncate(sf)), @as(u2, @truncate(hw)), @as(u16, @truncate(imm16)), rd);
}

/// MOVZ -- A64 -- A64
/// Move wide with zero
/// MOVZ  <Wd>, #<imm>{, LSL #<shift>}
/// MOVZ  <Xd>, #<imm>{, LSL #<shift>}
pub fn movz(sf: u8, hw: u8, imm16: u16, rd: Register) u32 {
    var result: u32 = 0b01010010100000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(hw)))) << 21;
    result |= (@as(u32, imm16)) << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for movz
export fn movz_c(sf: u8, hw: u8, imm16: u16, rd: Register) u32 {
    return movz(@as(u1, @truncate(sf)), @as(u2, @truncate(hw)), @as(u16, @truncate(imm16)), rd);
}

/// MRS -- A64 -- A64
/// Move System Register
/// MRS  <Xt>, (<systemreg>|S<op0>_<op1>_<Cn>_<Cm>_<op2>)
pub fn mrs(o0: u8, op1: u8, crn: u8, crm: u8, op2: u8, rt: Register) u32 {
    var result: u32 = 0b11010101001100000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(o0)))) << 19;
    result |= (@as(u32, @as(u3, @truncate(op1)))) << 16;
    result |= (@as(u32, @as(u4, @truncate(crn)))) << 12;
    result |= (@as(u32, @as(u4, @truncate(crm)))) << 8;
    result |= (@as(u32, @as(u3, @truncate(op2)))) << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for mrs
export fn mrs_c(o0: u8, op1: u8, crn: u8, crm: u8, op2: u8, rt: Register) u32 {
    return mrs(@as(u1, @truncate(o0)), @as(u3, @truncate(op1)), @as(u4, @truncate(crn)), @as(u4, @truncate(crm)), @as(u3, @truncate(op2)), rt);
}

/// MSUB -- A64 -- A64
/// Multiply-Subtract
/// MSUB  <Wd>, <Wn>, <Wm>, <Wa>
/// MSUB  <Xd>, <Xn>, <Xm>, <Xa>
pub fn msub(sf: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011011000000001000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for msub
export fn msub_c(sf: u8, rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return msub(@as(u1, @truncate(sf)), rm, ra, rn, rd);
}

/// MUL (vector) -- A64 -- A64
/// Multiply (vector)
/// MUL  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
pub fn mul_advsimd_vec(q: u8, size: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00001110001000001001110000000000;
    result |= (@as(u32, @as(u1, @truncate(q)))) << 30;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 22;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for mul_advsimd_vec
export fn mul_advsimd_vec_c(q: u8, size: u8, rm: Register, rn: Register, rd: Register) u32 {
    return mul_advsimd_vec(@as(u1, @truncate(q)), @as(u2, @truncate(size)), rm, rn, rd);
}

/// MUL -- A64 -- A64
/// MUL  <Wd>, <Wn>, <Wm>
/// MADD <Wd>, <Wn>, <Wm>, WZR
/// MUL  <Xd>, <Xn>, <Xm>
/// MADD <Xd>, <Xn>, <Xm>, XZR
pub fn mul_madd(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011011000000000111110000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for mul_madd
export fn mul_madd_c(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    return mul_madd(@as(u1, @truncate(sf)), rm, rn, rd);
}

/// NOP -- A64 -- A64
/// No Operation
/// NOP
pub fn nop() u32 {
    const result: u32 = 0b11010101000000110010000000011111;
    return result;
}

/// C-compatible wrapper for nop
export fn nop_c() u32 {
    return nop();
}

/// ORN (shifted register) -- A64 -- A64
/// Bitwise OR NOT (shifted register)
/// ORN  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// ORN  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn orn_log_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00101010001000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for orn_log_shift
export fn orn_log_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return orn_log_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
}

/// ORR (immediate) -- A64 -- A64
/// Bitwise OR (immediate)
/// ORR  <Wd|WSP>, <Wn>, #<imm>
/// ORR  <Xd|SP>, <Xn>, #<imm>
pub fn orr_log_imm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00110010000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for orr_log_imm
export fn orr_log_imm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return orr_log_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// ORR (shifted register) -- A64 -- A64
/// Bitwise OR (shifted register)
/// ORR  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// ORR  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn orr_log_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00101010000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for orr_log_shift
export fn orr_log_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return orr_log_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
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

/// ROR (register) -- A64 -- A64
/// Rotate Right (register)
/// ROR  <Wd>, <Wn>, <Wm>
/// RORV <Wd>, <Wn>, <Wm>
/// ROR  <Xd>, <Xn>, <Xm>
/// RORV <Xd>, <Xn>, <Xm>
pub fn ror_rorv(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011010110000000010110000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for ror_rorv
export fn ror_rorv_c(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    return ror_rorv(@as(u1, @truncate(sf)), rm, rn, rd);
}

/// SBFM -- A64 -- A64
/// Signed Bitfield Move
/// SBFM  <Wd>, <Wn>, #<immr>, #<imms>
/// SBFM  <Xd>, <Xn>, #<immr>, #<imms>
pub fn sbfm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00010011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for sbfm
export fn sbfm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return sbfm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// SDIV -- A64 -- A64
/// Signed Divide
/// SDIV  <Wd>, <Wn>, <Wm>
/// SDIV  <Xd>, <Xn>, <Xm>
pub fn sdiv(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011010110000000000110000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for sdiv
export fn sdiv_c(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    return sdiv(@as(u1, @truncate(sf)), rm, rn, rd);
}

/// SEV -- A64 -- A64
/// Send Event
/// SEV
pub fn sev() u32 {
    const result: u32 = 0b11010101000000110010000010011111;
    return result;
}

/// C-compatible wrapper for sev
export fn sev_c() u32 {
    return sev();
}

/// SEVL -- A64 -- A64
/// Send Event Local
/// SEVL
pub fn sevl() u32 {
    const result: u32 = 0b11010101000000110010000010111111;
    return result;
}

/// C-compatible wrapper for sevl
export fn sevl_c() u32 {
    return sevl();
}

/// SMADDL -- A64 -- A64
/// Signed Multiply-Add Long
/// SMADDL  <Xd>, <Wn>, <Wm>, <Xa>
pub fn smaddl(rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b10011011001000000000000000000000;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for smaddl
export fn smaddl_c(rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return smaddl(rm, ra, rn, rd);
}

/// SMC -- A64 -- A64
/// Secure Monitor Call
/// SMC  #<imm>
pub fn smc(imm16: u16) u32 {
    var result: u32 = 0b11010100000000000000000000000011;
    result |= (@as(u32, imm16)) << 5;
    return result;
}

/// C-compatible wrapper for smc
export fn smc_c(imm16: u16) u32 {
    return smc(@as(u16, @truncate(imm16)));
}

/// SMNEGL -- A64 -- A64
/// Signed Multiply-Negate Long
/// SMNEGL  <Xd>, <Wn>, <Wm>
/// SMSUBL <Xd>, <Wn>, <Wm>, XZR
pub fn smnegl_smsubl(rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b10011011001000001111110000000000;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for smnegl_smsubl
export fn smnegl_smsubl_c(rm: Register, rn: Register, rd: Register) u32 {
    return smnegl_smsubl(rm, rn, rd);
}

/// SMSUBL -- A64 -- A64
/// Signed Multiply-Subtract Long
/// SMSUBL  <Xd>, <Wn>, <Wm>, <Xa>
pub fn smsubl(rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b10011011001000001000000000000000;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for smsubl
export fn smsubl_c(rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return smsubl(rm, ra, rn, rd);
}

/// SMULL -- A64 -- A64
/// Signed Multiply Long
/// SMULL  <Xd>, <Wn>, <Wm>
/// SMADDL <Xd>, <Wn>, <Wm>, XZR
pub fn smull_smaddl(rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b10011011001000000111110000000000;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for smull_smaddl
export fn smull_smaddl_c(rm: Register, rn: Register, rd: Register) u32 {
    return smull_smaddl(rm, rn, rd);
}

/// STADD, STADDL -- A64 -- A64
/// Atomic add on word or doubleword in memory, without return
/// STADD  <Ws>, [<Xn|SP>]
/// LDADD <Ws>, WZR, [<Xn|SP>]
/// STADDL  <Ws>, [<Xn|SP>]
/// LDADDL <Ws>, WZR, [<Xn|SP>]
/// STADD  <Xs>, [<Xn|SP>]
/// LDADD <Xs>, XZR, [<Xn|SP>]
/// STADDL  <Xs>, [<Xn|SP>]
/// LDADDL <Xs>, XZR, [<Xn|SP>]
pub fn stadd_ldadd(size: u8, r: Register, rs: Register, rn: Register) u32 {
    var result: u32 = 0b00111000001000000000000000011111;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for stadd_ldadd
export fn stadd_ldadd_c(size: u8, r: Register, rs: Register, rn: Register) u32 {
    return stadd_ldadd(@as(u2, @truncate(size)), r, rs, rn);
}

/// STCLR, STCLRL -- A64 -- A64
/// Atomic bit clear on word or doubleword in memory, without return
/// STCLR  <Ws>, [<Xn|SP>]
/// LDCLR <Ws>, WZR, [<Xn|SP>]
/// STCLRL  <Ws>, [<Xn|SP>]
/// LDCLRL <Ws>, WZR, [<Xn|SP>]
/// STCLR  <Xs>, [<Xn|SP>]
/// LDCLR <Xs>, XZR, [<Xn|SP>]
/// STCLRL  <Xs>, [<Xn|SP>]
/// LDCLRL <Xs>, XZR, [<Xn|SP>]
pub fn stclr_ldclr(size: u8, r: Register, rs: Register, rn: Register) u32 {
    var result: u32 = 0b00111000001000000001000000011111;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for stclr_ldclr
export fn stclr_ldclr_c(size: u8, r: Register, rs: Register, rn: Register) u32 {
    return stclr_ldclr(@as(u2, @truncate(size)), r, rs, rn);
}

/// STEOR, STEORL -- A64 -- A64
/// Atomic exclusive OR on word or doubleword in memory, without return
/// STEOR  <Ws>, [<Xn|SP>]
/// LDEOR <Ws>, WZR, [<Xn|SP>]
/// STEORL  <Ws>, [<Xn|SP>]
/// LDEORL <Ws>, WZR, [<Xn|SP>]
/// STEOR  <Xs>, [<Xn|SP>]
/// LDEOR <Xs>, XZR, [<Xn|SP>]
/// STEORL  <Xs>, [<Xn|SP>]
/// LDEORL <Xs>, XZR, [<Xn|SP>]
pub fn steor_ldeor(size: u8, r: Register, rs: Register, rn: Register) u32 {
    var result: u32 = 0b00111000001000000010000000011111;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for steor_ldeor
export fn steor_ldeor_c(size: u8, r: Register, rs: Register, rn: Register) u32 {
    return steor_ldeor(@as(u2, @truncate(size)), r, rs, rn);
}

/// STLR -- A64 -- A64
/// Store-Release Register
/// STLR  <Wt>, [<Xn|SP>{,#0}]
/// STLR  <Xt>, [<Xn|SP>{,#0}]
pub fn stlr(size: u8, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00001000100111111111110000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for stlr
export fn stlr_c(size: u8, rn: Register, rt: Register) u32 {
    return stlr(@as(u2, @truncate(size)), rn, rt);
}

/// STP -- A64 -- A64
/// Store Pair of Registers
/// STP  <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
/// STP  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
/// STP  <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!
/// STP  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
/// STP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
/// STP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
pub fn stp_gen(opc: u8, imm7: i7, rt2: Register, rn: Register, rt: Register, class_selector: StpGenSelector) u32 {
    return switch (class_selector) {
        .PostIndex => blk: {
            var result: u32 = 0b00101000100000000000000000000000;
            result |= (@as(u32, @as(u2, @truncate(opc)))) << 30;
            result |= (@as(u32, @as(u7, @bitCast(imm7)))) << 15;
            result |= rt2.encode() << 10;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
        .PreIndex => blk: {
            var result: u32 = 0b00101001100000000000000000000000;
            result |= (@as(u32, @as(u2, @truncate(opc)))) << 30;
            result |= (@as(u32, @as(u7, @bitCast(imm7)))) << 15;
            result |= rt2.encode() << 10;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
        .SignedOffset => blk: {
            var result: u32 = 0b00101001000000000000000000000000;
            result |= (@as(u32, @as(u2, @truncate(opc)))) << 30;
            result |= (@as(u32, @as(u7, @bitCast(imm7)))) << 15;
            result |= rt2.encode() << 10;
            result |= rn.encode() << 5;
            result |= rt.encode() << 0;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for stp_gen
export fn stp_gen_c(opc: u8, imm7: i8, rt2: Register, rn: Register, rt: Register, class_selector: u32) u32 {
    return stp_gen(@as(u2, @truncate(opc)), @as(i7, @truncate(imm7)), rt2, rn, rt, @as(StpGenSelector, @enumFromInt(class_selector)));
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

/// STR (register) -- A64 -- A64
/// Store Register (register)
/// STR  <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
/// STR  <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
pub fn str_reg_gen(size: u8, rm: Register, option: u8, s: u8, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000001000000000100000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u3, @truncate(option)))) << 13;
    result |= (@as(u32, @as(u1, @truncate(s)))) << 12;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for str_reg_gen
export fn str_reg_gen_c(size: u8, rm: Register, option: u8, s: u8, rn: Register, rt: Register) u32 {
    return str_reg_gen(@as(u2, @truncate(size)), rm, @as(u3, @truncate(option)), @as(u1, @truncate(s)), rn, rt);
}

/// STSET, STSETL -- A64 -- A64
/// Atomic bit set on word or doubleword in memory, without return
/// STSET  <Ws>, [<Xn|SP>]
/// LDSET <Ws>, WZR, [<Xn|SP>]
/// STSETL  <Ws>, [<Xn|SP>]
/// LDSETL <Ws>, WZR, [<Xn|SP>]
/// STSET  <Xs>, [<Xn|SP>]
/// LDSET <Xs>, XZR, [<Xn|SP>]
/// STSETL  <Xs>, [<Xn|SP>]
/// LDSETL <Xs>, XZR, [<Xn|SP>]
pub fn stset_ldset(size: u8, r: Register, rs: Register, rn: Register) u32 {
    var result: u32 = 0b00111000001000000011000000011111;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for stset_ldset
export fn stset_ldset_c(size: u8, r: Register, rs: Register, rn: Register) u32 {
    return stset_ldset(@as(u2, @truncate(size)), r, rs, rn);
}

/// STSMAX, STSMAXL -- A64 -- A64
/// Atomic signed maximum on word or doubleword in memory, without return
/// STSMAX  <Ws>, [<Xn|SP>]
/// LDSMAX <Ws>, WZR, [<Xn|SP>]
/// STSMAXL  <Ws>, [<Xn|SP>]
/// LDSMAXL <Ws>, WZR, [<Xn|SP>]
/// STSMAX  <Xs>, [<Xn|SP>]
/// LDSMAX <Xs>, XZR, [<Xn|SP>]
/// STSMAXL  <Xs>, [<Xn|SP>]
/// LDSMAXL <Xs>, XZR, [<Xn|SP>]
pub fn stsmax_ldsmax(size: u8, r: Register, rs: Register, rn: Register) u32 {
    var result: u32 = 0b00111000001000000100000000011111;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for stsmax_ldsmax
export fn stsmax_ldsmax_c(size: u8, r: Register, rs: Register, rn: Register) u32 {
    return stsmax_ldsmax(@as(u2, @truncate(size)), r, rs, rn);
}

/// STSMIN, STSMINL -- A64 -- A64
/// Atomic signed minimum on word or doubleword in memory, without return
/// STSMIN  <Ws>, [<Xn|SP>]
/// LDSMIN <Ws>, WZR, [<Xn|SP>]
/// STSMINL  <Ws>, [<Xn|SP>]
/// LDSMINL <Ws>, WZR, [<Xn|SP>]
/// STSMIN  <Xs>, [<Xn|SP>]
/// LDSMIN <Xs>, XZR, [<Xn|SP>]
/// STSMINL  <Xs>, [<Xn|SP>]
/// LDSMINL <Xs>, XZR, [<Xn|SP>]
pub fn stsmin_ldsmin(size: u8, r: Register, rs: Register, rn: Register) u32 {
    var result: u32 = 0b00111000001000000101000000011111;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for stsmin_ldsmin
export fn stsmin_ldsmin_c(size: u8, r: Register, rs: Register, rn: Register) u32 {
    return stsmin_ldsmin(@as(u2, @truncate(size)), r, rs, rn);
}

/// STUMAX, STUMAXL -- A64 -- A64
/// Atomic unsigned maximum on word or doubleword in memory, without return
/// STUMAX  <Ws>, [<Xn|SP>]
/// LDUMAX <Ws>, WZR, [<Xn|SP>]
/// STUMAXL  <Ws>, [<Xn|SP>]
/// LDUMAXL <Ws>, WZR, [<Xn|SP>]
/// STUMAX  <Xs>, [<Xn|SP>]
/// LDUMAX <Xs>, XZR, [<Xn|SP>]
/// STUMAXL  <Xs>, [<Xn|SP>]
/// LDUMAXL <Xs>, XZR, [<Xn|SP>]
pub fn stumax_ldumax(size: u8, r: Register, rs: Register, rn: Register) u32 {
    var result: u32 = 0b00111000001000000110000000011111;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for stumax_ldumax
export fn stumax_ldumax_c(size: u8, r: Register, rs: Register, rn: Register) u32 {
    return stumax_ldumax(@as(u2, @truncate(size)), r, rs, rn);
}

/// STUMIN, STUMINL -- A64 -- A64
/// Atomic unsigned minimum on word or doubleword in memory, without return
/// STUMIN  <Ws>, [<Xn|SP>]
/// LDUMIN <Ws>, WZR, [<Xn|SP>]
/// STUMINL  <Ws>, [<Xn|SP>]
/// LDUMINL <Ws>, WZR, [<Xn|SP>]
/// STUMIN  <Xs>, [<Xn|SP>]
/// LDUMIN <Xs>, XZR, [<Xn|SP>]
/// STUMINL  <Xs>, [<Xn|SP>]
/// LDUMINL <Xs>, XZR, [<Xn|SP>]
pub fn stumin_ldumin(size: u8, r: Register, rs: Register, rn: Register) u32 {
    var result: u32 = 0b00111000001000000111000000011111;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= r.encode() << 22;
    result |= rs.encode() << 16;
    result |= rn.encode() << 5;
    return result;
}

/// C-compatible wrapper for stumin_ldumin
export fn stumin_ldumin_c(size: u8, r: Register, rs: Register, rn: Register) u32 {
    return stumin_ldumin(@as(u2, @truncate(size)), r, rs, rn);
}

/// STUR -- A64 -- A64
/// Store Register (unscaled)
/// STUR  <Wt>, [<Xn|SP>{, #<simm>}]
/// STUR  <Xt>, [<Xn|SP>{, #<simm>}]
pub fn stur_gen(size: u8, imm9: i9, rn: Register, rt: Register) u32 {
    var result: u32 = 0b00111000000000000000000000000000;
    result |= (@as(u32, @as(u2, @truncate(size)))) << 30;
    result |= (@as(u32, @as(u9, @bitCast(imm9)))) << 12;
    result |= rn.encode() << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for stur_gen
export fn stur_gen_c(size: u8, imm9: i16, rn: Register, rt: Register) u32 {
    return stur_gen(@as(u2, @truncate(size)), @as(i9, @truncate(imm9)), rn, rt);
}

/// SUB (extended register) -- A64 -- A64
/// Subtract (extended register)
/// SUB  <Wd|WSP>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
/// SUB  <Xd|SP>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
pub fn sub_addsub_ext(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01001011001000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u3, @truncate(option)))) << 13;
    result |= (@as(u32, @as(u3, @bitCast(@as(i3, @truncate(imm3)))))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for sub_addsub_ext
export fn sub_addsub_ext_c(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register, rd: Register) u32 {
    return sub_addsub_ext(@as(u1, @truncate(sf)), rm, @as(u3, @truncate(option)), @as(i3, @truncate(imm3)), rn, rd);
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

/// SUB (shifted register) -- A64 -- A64
/// Subtract (shifted register)
/// SUB  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// SUB  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn sub_addsub_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01001011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for sub_addsub_shift
export fn sub_addsub_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return sub_addsub_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
}

/// SUB (vector) -- A64 -- A64
/// Subtract (vector)
/// SUB  <V><d>, <V><n>, <V><m>
/// SUB  <Vd>.<T>, <Vn>.<T>, <Vm>.<T>
pub fn sub_advsimd(size: u8, rm: Register, rn: Register, rd: Register, q: u8, class_selector: SubAdvsimdSelector) u32 {
    return switch (class_selector) {
        .Scalar => blk: {
            var result: u32 = 0b01111110001000001000010000000000;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 22;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
        .Vector => blk: {
            var result: u32 = 0b00101110001000001000010000000000;
            result |= (@as(u32, @as(u1, @truncate(q)))) << 30;
            result |= (@as(u32, @as(u2, @truncate(size)))) << 22;
            result |= rm.encode() << 16;
            result |= rn.encode() << 5;
            result |= rd.encode() << 0;
            break :blk result;
        },
    };
}

/// C-compatible wrapper for sub_advsimd
export fn sub_advsimd_c(size: u8, rm: Register, rn: Register, rd: Register, q: u8, class_selector: u32) u32 {
    return sub_advsimd(@as(u2, @truncate(size)), rm, rn, rd, @as(u1, @truncate(q)), @as(SubAdvsimdSelector, @enumFromInt(class_selector)));
}

/// SUBS (extended register) -- A64 -- A64
/// Subtract (extended register), setting flags
/// SUBS  <Wd>, <Wn|WSP>, <Wm>{, <extend> {#<amount>}}
/// SUBS  <Xd>, <Xn|SP>, <R><m>{, <extend> {#<amount>}}
pub fn subs_addsub_ext(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01101011001000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u3, @truncate(option)))) << 13;
    result |= (@as(u32, @as(u3, @bitCast(@as(i3, @truncate(imm3)))))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for subs_addsub_ext
export fn subs_addsub_ext_c(sf: u8, rm: Register, option: u8, imm3: i8, rn: Register, rd: Register) u32 {
    return subs_addsub_ext(@as(u1, @truncate(sf)), rm, @as(u3, @truncate(option)), @as(i3, @truncate(imm3)), rn, rd);
}

/// SUBS (immediate) -- A64 -- A64
/// Subtract (immediate), setting flags
/// SUBS  <Wd>, <Wn|WSP>, #<imm>{, <shift>}
/// SUBS  <Xd>, <Xn|SP>, #<imm>{, <shift>}
pub fn subs_addsub_imm(sf: u8, sh: u8, imm12: i12, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01110001000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(sh)))) << 22;
    result |= (@as(u32, @as(u12, @bitCast(imm12)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for subs_addsub_imm
export fn subs_addsub_imm_c(sf: u8, sh: u8, imm12: i16, rn: Register, rd: Register) u32 {
    return subs_addsub_imm(@as(u1, @truncate(sf)), @as(u1, @truncate(sh)), @as(i12, @truncate(imm12)), rn, rd);
}

/// SUBS (shifted register) -- A64 -- A64
/// Subtract (shifted register), setting flags
/// SUBS  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
/// SUBS  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
pub fn subs_addsub_shift(sf: u8, shift: u8, rm: Register, imm6: i6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01101011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u2, @truncate(shift)))) << 22;
    result |= rm.encode() << 16;
    result |= (@as(u32, @as(u6, @bitCast(imm6)))) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for subs_addsub_shift
export fn subs_addsub_shift_c(sf: u8, shift: u8, rm: Register, imm6: i8, rn: Register, rd: Register) u32 {
    return subs_addsub_shift(@as(u1, @truncate(sf)), @as(u2, @truncate(shift)), rm, @as(i6, @truncate(imm6)), rn, rd);
}

/// SVC -- A64 -- A64
/// Supervisor Call
/// SVC  #<imm>
pub fn svc(imm16: u16) u32 {
    var result: u32 = 0b11010100000000000000000000000001;
    result |= (@as(u32, imm16)) << 5;
    return result;
}

/// C-compatible wrapper for svc
export fn svc_c(imm16: u16) u32 {
    return svc(@as(u16, @truncate(imm16)));
}

/// SYS -- A64 -- A64
/// System instruction
/// SYS  #<op1>, <Cn>, <Cm>, #<op2>{, <Xt>}
pub fn sys(op1: u8, crn: u8, crm: u8, op2: u8, rt: Register) u32 {
    var result: u32 = 0b11010101000010000000000000000000;
    result |= (@as(u32, @as(u3, @truncate(op1)))) << 16;
    result |= (@as(u32, @as(u4, @truncate(crn)))) << 12;
    result |= (@as(u32, @as(u4, @truncate(crm)))) << 8;
    result |= (@as(u32, @as(u3, @truncate(op2)))) << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for sys
export fn sys_c(op1: u8, crn: u8, crm: u8, op2: u8, rt: Register) u32 {
    return sys(@as(u3, @truncate(op1)), @as(u4, @truncate(crn)), @as(u4, @truncate(crm)), @as(u3, @truncate(op2)), rt);
}

/// SYSL -- A64 -- A64
/// System instruction with result
/// SYSL  <Xt>, #<op1>, <Cn>, <Cm>, #<op2>
pub fn sysl(op1: u8, crn: u8, crm: u8, op2: u8, rt: Register) u32 {
    var result: u32 = 0b11010101001010000000000000000000;
    result |= (@as(u32, @as(u3, @truncate(op1)))) << 16;
    result |= (@as(u32, @as(u4, @truncate(crn)))) << 12;
    result |= (@as(u32, @as(u4, @truncate(crm)))) << 8;
    result |= (@as(u32, @as(u3, @truncate(op2)))) << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for sysl
export fn sysl_c(op1: u8, crn: u8, crm: u8, op2: u8, rt: Register) u32 {
    return sysl(@as(u3, @truncate(op1)), @as(u4, @truncate(crn)), @as(u4, @truncate(crm)), @as(u3, @truncate(op2)), rt);
}

/// TBNZ -- A64 -- A64
/// Test bit and Branch if Nonzero
/// TBNZ  <R><t>, #<imm>, <label>
pub fn tbnz(b5: u8, b40: u5, imm14: i14, rt: Register) u32 {
    var result: u32 = 0b00110111000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(b5)))) << 31;
    result |= (@as(u32, b40)) << 19;
    result |= (@as(u32, @as(u14, @bitCast(imm14)))) << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for tbnz
export fn tbnz_c(b5: u8, b40: u8, imm14: i16, rt: Register) u32 {
    return tbnz(@as(u1, @truncate(b5)), @as(u5, @truncate(b40)), @as(i14, @truncate(imm14)), rt);
}

/// TBZ -- A64 -- A64
/// Test bit and Branch if Zero
/// TBZ  <R><t>, #<imm>, <label>
pub fn tbz(b5: u8, b40: u5, imm14: i14, rt: Register) u32 {
    var result: u32 = 0b00110110000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(b5)))) << 31;
    result |= (@as(u32, b40)) << 19;
    result |= (@as(u32, @as(u14, @bitCast(imm14)))) << 5;
    result |= rt.encode() << 0;
    return result;
}

/// C-compatible wrapper for tbz
export fn tbz_c(b5: u8, b40: u8, imm14: i16, rt: Register) u32 {
    return tbz(@as(u1, @truncate(b5)), @as(u5, @truncate(b40)), @as(i14, @truncate(imm14)), rt);
}

/// UBFM -- A64 -- A64
/// Unsigned Bitfield Move
/// UBFM  <Wd>, <Wn>, #<immr>, #<imms>
/// UBFM  <Xd>, <Xn>, #<immr>, #<imms>
pub fn ubfm(sf: u8, n: u8, immr: u6, imms: u6, rn: Register, rd: Register) u32 {
    var result: u32 = 0b01010011000000000000000000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= (@as(u32, @as(u1, @truncate(n)))) << 22;
    result |= (@as(u32, immr)) << 16;
    result |= (@as(u32, imms)) << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for ubfm
export fn ubfm_c(sf: u8, n: u8, immr: u8, imms: u8, rn: Register, rd: Register) u32 {
    return ubfm(@as(u1, @truncate(sf)), @as(u1, @truncate(n)), @as(u6, @truncate(immr)), @as(u6, @truncate(imms)), rn, rd);
}

/// UDIV -- A64 -- A64
/// Unsigned Divide
/// UDIV  <Wd>, <Wn>, <Wm>
/// UDIV  <Xd>, <Xn>, <Xm>
pub fn udiv(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b00011010110000000000100000000000;
    result |= (@as(u32, @as(u1, @truncate(sf)))) << 31;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for udiv
export fn udiv_c(sf: u8, rm: Register, rn: Register, rd: Register) u32 {
    return udiv(@as(u1, @truncate(sf)), rm, rn, rd);
}

/// UMADDL -- A64 -- A64
/// Unsigned Multiply-Add Long
/// UMADDL  <Xd>, <Wn>, <Wm>, <Xa>
pub fn umaddl(rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b10011011101000000000000000000000;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for umaddl
export fn umaddl_c(rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return umaddl(rm, ra, rn, rd);
}

/// UMNEGL -- A64 -- A64
/// Unsigned Multiply-Negate Long
/// UMNEGL  <Xd>, <Wn>, <Wm>
/// UMSUBL <Xd>, <Wn>, <Wm>, XZR
pub fn umnegl_umsubl(rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b10011011101000001111110000000000;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for umnegl_umsubl
export fn umnegl_umsubl_c(rm: Register, rn: Register, rd: Register) u32 {
    return umnegl_umsubl(rm, rn, rd);
}

/// UMSUBL -- A64 -- A64
/// Unsigned Multiply-Subtract Long
/// UMSUBL  <Xd>, <Wn>, <Wm>, <Xa>
pub fn umsubl(rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b10011011101000001000000000000000;
    result |= rm.encode() << 16;
    result |= ra.encode() << 10;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for umsubl
export fn umsubl_c(rm: Register, ra: Register, rn: Register, rd: Register) u32 {
    return umsubl(rm, ra, rn, rd);
}

/// UMULL -- A64 -- A64
/// Unsigned Multiply Long
/// UMULL  <Xd>, <Wn>, <Wm>
/// UMADDL <Xd>, <Wn>, <Wm>, XZR
pub fn umull_umaddl(rm: Register, rn: Register, rd: Register) u32 {
    var result: u32 = 0b10011011101000000111110000000000;
    result |= rm.encode() << 16;
    result |= rn.encode() << 5;
    result |= rd.encode() << 0;
    return result;
}

/// C-compatible wrapper for umull_umaddl
export fn umull_umaddl_c(rm: Register, rn: Register, rd: Register) u32 {
    return umull_umaddl(rm, rn, rd);
}

/// WFE -- A64 -- A64
/// Wait For Event
/// WFE
pub fn wfe() u32 {
    const result: u32 = 0b11010101000000110010000001011111;
    return result;
}

/// C-compatible wrapper for wfe
export fn wfe_c() u32 {
    return wfe();
}

/// WFI -- A64 -- A64
/// Wait For Interrupt
/// WFI
pub fn wfi() u32 {
    const result: u32 = 0b11010101000000110010000001111111;
    return result;
}

/// C-compatible wrapper for wfi
export fn wfi_c() u32 {
    return wfi();
}

/// YIELD -- A64 -- A64
/// YIELD
/// YIELD
pub fn yield() u32 {
    const result: u32 = 0b11010101000000110010000000111111;
    return result;
}

/// C-compatible wrapper for yield
export fn yield_c() u32 {
    return yield();
}

