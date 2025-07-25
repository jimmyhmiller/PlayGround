
#pragma once
#include <cstdint>
#include <cassert>

namespace arm_asm {

enum class Size : uint8_t {
    S32 = 0,
    S64 = 1
};

struct Register {
    Size size;
    uint8_t index;
    
    constexpr Register(uint8_t idx, Size sz) : size(sz), index(idx) {}
    
    constexpr int sf() const {
        return static_cast<int>(size);
    }
    
    constexpr uint8_t encode() const {
        return index;
    }
};

// Register constants
constexpr Register SP{31, Size::S64};
constexpr Register ZERO_REGISTER{31, Size::S64};

template<int WIDTH>
constexpr uint32_t truncate_imm(int32_t imm) {
    static_assert(WIDTH > 0 && WIDTH <= 32, "Width must be between 1 and 32");
    const uint32_t masked = static_cast<uint32_t>(imm) & ((1U << WIDTH) - 1);
    
    // Assert that we didn't drop any bits by truncating
    if (imm >= 0) {
        assert(static_cast<uint32_t>(imm) == masked);
    } else {
        assert(static_cast<uint32_t>(imm) == (masked | (0xFFFFFFFFU << WIDTH)));
    }
    
    return masked;
}


constexpr Register X0{0U, Size::S64};
constexpr Register X1{1U, Size::S64};
constexpr Register X2{2U, Size::S64};
constexpr Register X3{3U, Size::S64};
constexpr Register X4{4U, Size::S64};
constexpr Register X5{5U, Size::S64};
constexpr Register X6{6U, Size::S64};
constexpr Register X7{7U, Size::S64};
constexpr Register X8{8U, Size::S64};
constexpr Register X9{9U, Size::S64};
constexpr Register X10{10U, Size::S64};
constexpr Register X11{11U, Size::S64};
constexpr Register X12{12U, Size::S64};
constexpr Register X13{13U, Size::S64};
constexpr Register X14{14U, Size::S64};
constexpr Register X15{15U, Size::S64};
constexpr Register X16{16U, Size::S64};
constexpr Register X17{17U, Size::S64};
constexpr Register X18{18U, Size::S64};
constexpr Register X19{19U, Size::S64};
constexpr Register X20{20U, Size::S64};
constexpr Register X21{21U, Size::S64};
constexpr Register X22{22U, Size::S64};
constexpr Register X23{23U, Size::S64};
constexpr Register X24{24U, Size::S64};
constexpr Register X25{25U, Size::S64};
constexpr Register X26{26U, Size::S64};
constexpr Register X27{27U, Size::S64};
constexpr Register X28{28U, Size::S64};
constexpr Register X29{29U, Size::S64};
constexpr Register X30{30U, Size::S64};



enum class LdpGenSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
};

enum class StpGenSelector {
    PostIndex,
    PreIndex,
    SignedOffset,
};

enum class StrImmGenSelector {
    PostIndex,
    PreIndex,
    UnsignedOffset,
};



/**
 * ADD (immediate) -- A64
 * Add (immediate)
 * ADD  <Wd|WSP>, <Wn|WSP>, #<imm>{, <shift>}
 * ADD  <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
 */
constexpr uint32_t add_addsub_imm(int32_t sf, int32_t sh, int32_t imm12, Register rn, Register rd) noexcept {
    uint32_t result = 0b00010001000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(sh) << 22U;
    result |= truncate_imm<12>(imm12) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * ADD (shifted register) -- A64
 * Add (shifted register)
 * ADD  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
 * ADD  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
 */
constexpr uint32_t add_addsub_shift(int32_t sf, int32_t shift, Register rm, int32_t imm6, Register rn, Register rd) noexcept {
    uint32_t result = 0b00001011000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(shift) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= truncate_imm<6>(imm6) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * ADR -- A64
 * Form PC-relative address
 * ADR  <Xd>, <label>
 */
constexpr uint32_t adr(int32_t immlo, int32_t immhi, Register rd) noexcept {
    uint32_t result = 0b00010000000000000000000000000000U;
    result |= static_cast<uint32_t>(immlo) << 29U;
    result |= static_cast<uint32_t>(immhi) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * AND (immediate) -- A64
 * Bitwise AND (immediate)
 * AND  <Wd|WSP>, <Wn>, #<imm>
 * AND  <Xd|SP>, <Xn>, #<imm>
 */
constexpr uint32_t and_log_imm(int32_t sf, int32_t n, int32_t immr, int32_t imms, Register rn, Register rd) noexcept {
    uint32_t result = 0b00010010000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(n) << 22U;
    result |= static_cast<uint32_t>(immr) << 16U;
    result |= static_cast<uint32_t>(imms) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * AND (shifted register) -- A64
 * Bitwise AND (shifted register)
 * AND  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
 * AND  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
 */
constexpr uint32_t and_log_shift(int32_t sf, int32_t shift, Register rm, int32_t imm6, Register rn, Register rd) noexcept {
    uint32_t result = 0b00001010000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(shift) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= truncate_imm<6>(imm6) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * ASR (register) -- A64
 * Arithmetic Shift Right (register)
 * ASR  <Wd>, <Wn>, <Wm>
 * ASRV <Wd>, <Wn>, <Wm>
 * ASR  <Xd>, <Xn>, <Xm>
 * ASRV <Xd>, <Xn>, <Xm>
 */
constexpr uint32_t asr_asrv(int32_t sf, Register rm, Register rn, Register rd) noexcept {
    uint32_t result = 0b00011010110000000010100000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * ASR (immediate) -- A64
 * Arithmetic Shift Right (immediate)
 * ASR  <Wd>, <Wn>, #<shift>
 * SBFM <Wd>, <Wn>, #<shift>, #31
 * ASR  <Xd>, <Xn>, #<shift>
 * SBFM <Xd>, <Xn>, #<shift>, #63
 */
constexpr uint32_t asr_sbfm(int32_t sf, int32_t n, int32_t immr, int32_t imms, Register rn, Register rd) noexcept {
    uint32_t result = 0b00010011000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(n) << 22U;
    result |= static_cast<uint32_t>(immr) << 16U;
    result |= static_cast<uint32_t>(imms) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * B.cond -- A64
 * Branch conditionally
 * B.<cond>  <label>
 */
constexpr uint32_t bcond(int32_t imm19, int32_t cond) noexcept {
    uint32_t result = 0b01010100000000000000000000000000U;
    result |= truncate_imm<19>(imm19) << 5U;
    result |= static_cast<uint32_t>(cond) << 0U;
    return result;
}

/**
 * BL -- A64
 * Branch with Link
 * BL  <label>
 */
constexpr uint32_t bl(int32_t imm26) noexcept {
    uint32_t result = 0b10010100000000000000000000000000U;
    result |= truncate_imm<26>(imm26) << 0U;
    return result;
}

/**
 * BLR -- A64
 * Branch with Link to Register
 * BLR  <Xn>
 */
constexpr uint32_t blr(Register rn) noexcept {
    uint32_t result = 0b11010110001111110000000000000000U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    return result;
}

/**
 * BRK -- A64
 * Breakpoint instruction
 * BRK  #<imm>
 */
constexpr uint32_t brk(int32_t imm16) noexcept {
    uint32_t result = 0b11010100001000000000000000000000U;
    result |= static_cast<uint32_t>(imm16) << 5U;
    return result;
}

/**
 * CAS, CASA, CASAL, CASL -- A64
 * Compare and Swap word or doubleword in memory
 * CAS  <Ws>, <Wt>, [<Xn|SP>{,#0}]
 * CASA  <Ws>, <Wt>, [<Xn|SP>{,#0}]
 * CASAL  <Ws>, <Wt>, [<Xn|SP>{,#0}]
 * CASL  <Ws>, <Wt>, [<Xn|SP>{,#0}]
 * CAS  <Xs>, <Xt>, [<Xn|SP>{,#0}]
 * CASA  <Xs>, <Xt>, [<Xn|SP>{,#0}]
 * CASAL  <Xs>, <Xt>, [<Xn|SP>{,#0}]
 * CASL  <Xs>, <Xt>, [<Xn|SP>{,#0}]
 */
constexpr uint32_t cas(int32_t size, int32_t l, Register rs, int32_t o0, Register rn, Register rt) noexcept {
    uint32_t result = 0b00001000101000000111110000000000U;
    result |= static_cast<uint32_t>(size) << 30U;
    result |= static_cast<uint32_t>(l) << 22U;
    result |= static_cast<uint32_t>(rs.encode()) << 16U;
    result |= static_cast<uint32_t>(o0) << 15U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rt.encode()) << 0U;
    return result;
}

/**
 * CMP (shifted register) -- A64
 * Compare (shifted register)
 * CMP  <Wn>, <Wm>{, <shift> #<amount>}
 * SUBS WZR, <Wn>, <Wm> {, <shift> #<amount>}
 * CMP  <Xn>, <Xm>{, <shift> #<amount>}
 * SUBS XZR, <Xn>, <Xm> {, <shift> #<amount>}
 */
constexpr uint32_t cmp_subs_addsub_shift(int32_t sf, int32_t shift, Register rm, int32_t imm6, Register rn) noexcept {
    uint32_t result = 0b01101011000000000000000000011111U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(shift) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= truncate_imm<6>(imm6) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    return result;
}

/**
 * CSET -- A64
 * Conditional Set
 * CSET  <Wd>, <cond>
 * CSINC <Wd>, WZR, WZR, invert(<cond>)
 * CSET  <Xd>, <cond>
 * CSINC <Xd>, XZR, XZR, invert(<cond>)
 */
constexpr uint32_t cset_csinc(int32_t sf, int32_t cond, Register rd) noexcept {
    uint32_t result = 0b00011010100111110000011111100000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(cond) << 12U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * EOR (shifted register) -- A64
 * Bitwise Exclusive OR (shifted register)
 * EOR  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
 * EOR  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
 */
constexpr uint32_t eor_log_shift(int32_t sf, int32_t shift, Register rm, int32_t imm6, Register rn, Register rd) noexcept {
    uint32_t result = 0b01001010000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(shift) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= truncate_imm<6>(imm6) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * FADD (scalar) -- A64
 * Floating-point Add (scalar)
 * FADD  <Hd>, <Hn>, <Hm>
 * FADD  <Sd>, <Sn>, <Sm>
 * FADD  <Dd>, <Dn>, <Dm>
 */
constexpr uint32_t fadd_float(int32_t ftype, Register rm, Register rn, Register rd) noexcept {
    uint32_t result = 0b00011110001000000010100000000000U;
    result |= static_cast<uint32_t>(ftype) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * FDIV (scalar) -- A64
 * Floating-point Divide (scalar)
 * FDIV  <Hd>, <Hn>, <Hm>
 * FDIV  <Sd>, <Sn>, <Sm>
 * FDIV  <Dd>, <Dn>, <Dm>
 */
constexpr uint32_t fdiv_float(int32_t ftype, Register rm, Register rn, Register rd) noexcept {
    uint32_t result = 0b00011110001000000001100000000000U;
    result |= static_cast<uint32_t>(ftype) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * FMOV (general) -- A64
 * Floating-point Move to or from general-purpose register without conversion
 * FMOV  <Wd>, <Hn>
 * FMOV  <Xd>, <Hn>
 * FMOV  <Hd>, <Wn>
 * FMOV  <Sd>, <Wn>
 * FMOV  <Wd>, <Sn>
 * FMOV  <Hd>, <Xn>
 * FMOV  <Dd>, <Xn>
 * FMOV  <Vd>.D[1], <Xn>
 * FMOV  <Xd>, <Dn>
 * FMOV  <Xd>, <Vn>.D[1]
 */
constexpr uint32_t fmov_float_gen(int32_t sf, int32_t ftype, int32_t rmode, int32_t opcode, Register rn, Register rd) noexcept {
    uint32_t result = 0b00011110001000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(ftype) << 22U;
    result |= static_cast<uint32_t>(rmode) << 19U;
    result |= static_cast<uint32_t>(opcode) << 16U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * FMUL (scalar) -- A64
 * Floating-point Multiply (scalar)
 * FMUL  <Hd>, <Hn>, <Hm>
 * FMUL  <Sd>, <Sn>, <Sm>
 * FMUL  <Dd>, <Dn>, <Dm>
 */
constexpr uint32_t fmul_float(int32_t ftype, Register rm, Register rn, Register rd) noexcept {
    uint32_t result = 0b00011110001000000000100000000000U;
    result |= static_cast<uint32_t>(ftype) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * FSUB (scalar) -- A64
 * Floating-point Subtract (scalar)
 * FSUB  <Hd>, <Hn>, <Hm>
 * FSUB  <Sd>, <Sn>, <Sm>
 * FSUB  <Dd>, <Dn>, <Dm>
 */
constexpr uint32_t fsub_float(int32_t ftype, Register rm, Register rn, Register rd) noexcept {
    uint32_t result = 0b00011110001000000011100000000000U;
    result |= static_cast<uint32_t>(ftype) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * LDAR -- A64
 * Load-Acquire Register
 * LDAR  <Wt>, [<Xn|SP>{,#0}]
 * LDAR  <Xt>, [<Xn|SP>{,#0}]
 */
constexpr uint32_t ldar(int32_t size, Register rn, Register rt) noexcept {
    uint32_t result = 0b00001000110111111111110000000000U;
    result |= static_cast<uint32_t>(size) << 30U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rt.encode()) << 0U;
    return result;
}

/**
 * LDP -- A64
 * Load Pair of Registers
 * LDP  <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
 * LDP  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
 * LDP  <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!
 * LDP  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
 * LDP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
 * LDP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
 */
constexpr uint32_t ldp_gen(int32_t opc, int32_t imm7, Register rt2, Register rn, Register rt, LdpGenSelector class_selector) noexcept {
    switch (class_selector) {
    case LdpGenSelector::PostIndex: {
        uint32_t result = 0b00101000110000000000000000000000U;
        result |= static_cast<uint32_t>(opc) << 30U;
        result |= truncate_imm<7>(imm7) << 15U;
        result |= static_cast<uint32_t>(rt2.encode()) << 10U;
        result |= static_cast<uint32_t>(rn.encode()) << 5U;
        result |= static_cast<uint32_t>(rt.encode()) << 0U;
        return result;
    }
    case LdpGenSelector::PreIndex: {
        uint32_t result = 0b00101001110000000000000000000000U;
        result |= static_cast<uint32_t>(opc) << 30U;
        result |= truncate_imm<7>(imm7) << 15U;
        result |= static_cast<uint32_t>(rt2.encode()) << 10U;
        result |= static_cast<uint32_t>(rn.encode()) << 5U;
        result |= static_cast<uint32_t>(rt.encode()) << 0U;
        return result;
    }
    case LdpGenSelector::SignedOffset: {
        uint32_t result = 0b00101001010000000000000000000000U;
        result |= static_cast<uint32_t>(opc) << 30U;
        result |= truncate_imm<7>(imm7) << 15U;
        result |= static_cast<uint32_t>(rt2.encode()) << 10U;
        result |= static_cast<uint32_t>(rn.encode()) << 5U;
        result |= static_cast<uint32_t>(rt.encode()) << 0U;
        return result;
    }
    default:
        // Should never reach here if all cases are handled
        return 0U;
    }
}

/**
 * LDR (register) -- A64
 * Load Register (register)
 * LDR  <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
 * LDR  <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
 */
constexpr uint32_t ldr_reg_gen(int32_t size, Register rm, int32_t option, int32_t s, Register rn, Register rt) noexcept {
    uint32_t result = 0b00111000011000000000100000000000U;
    result |= static_cast<uint32_t>(size) << 30U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(option) << 13U;
    result |= static_cast<uint32_t>(s) << 12U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rt.encode()) << 0U;
    return result;
}

/**
 * LDUR -- A64
 * Load Register (unscaled)
 * LDUR  <Wt>, [<Xn|SP>{, #<simm>}]
 * LDUR  <Xt>, [<Xn|SP>{, #<simm>}]
 */
constexpr uint32_t ldur_gen(int32_t size, int32_t imm9, Register rn, Register rt) noexcept {
    uint32_t result = 0b00111000010000000000000000000000U;
    result |= static_cast<uint32_t>(size) << 30U;
    result |= truncate_imm<9>(imm9) << 12U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rt.encode()) << 0U;
    return result;
}

/**
 * LSL (register) -- A64
 * Logical Shift Left (register)
 * LSL  <Wd>, <Wn>, <Wm>
 * LSLV <Wd>, <Wn>, <Wm>
 * LSL  <Xd>, <Xn>, <Xm>
 * LSLV <Xd>, <Xn>, <Xm>
 */
constexpr uint32_t lsl_lslv(int32_t sf, Register rm, Register rn, Register rd) noexcept {
    uint32_t result = 0b00011010110000000010000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * LSL (immediate) -- A64
 * Logical Shift Left (immediate)
 * LSL  <Wd>, <Wn>, #<shift>
 * UBFM <Wd>, <Wn>, #(-<shift> MOD 32), #(31-<shift>)
 * LSL  <Xd>, <Xn>, #<shift>
 * UBFM <Xd>, <Xn>, #(-<shift> MOD 64), #(63-<shift>)
 */
constexpr uint32_t lsl_ubfm(int32_t sf, int32_t n, int32_t immr, int32_t imms, Register rn, Register rd) noexcept {
    uint32_t result = 0b01010011000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(n) << 22U;
    result |= static_cast<uint32_t>(immr) << 16U;
    result |= static_cast<uint32_t>(imms) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * MADD -- A64
 * Multiply-Add
 * MADD  <Wd>, <Wn>, <Wm>, <Wa>
 * MADD  <Xd>, <Xn>, <Xm>, <Xa>
 */
constexpr uint32_t madd(int32_t sf, Register rm, Register ra, Register rn, Register rd) noexcept {
    uint32_t result = 0b00011011000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(ra.encode()) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * MOV (to/from SP) -- A64
 * MOV  <Wd|WSP>, <Wn|WSP>
 * ADD <Wd|WSP>, <Wn|WSP>, #0
 * MOV  <Xd|SP>, <Xn|SP>
 * ADD <Xd|SP>, <Xn|SP>, #0
 */
constexpr uint32_t mov_add_addsub_imm(int32_t sf, Register rn, Register rd) noexcept {
    uint32_t result = 0b00010001000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * MOV (register) -- A64
 * Move (register)
 * MOV  <Wd>, <Wm>
 * ORR <Wd>, WZR, <Wm>
 * MOV  <Xd>, <Xm>
 * ORR <Xd>, XZR, <Xm>
 */
constexpr uint32_t mov_orr_log_shift(int32_t sf, Register rm, Register rd) noexcept {
    uint32_t result = 0b00101010000000000000001111100000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * MOVK -- A64
 * Move wide with keep
 * MOVK  <Wd>, #<imm>{, LSL #<shift>}
 * MOVK  <Xd>, #<imm>{, LSL #<shift>}
 */
constexpr uint32_t movk(int32_t sf, int32_t hw, int32_t imm16, Register rd) noexcept {
    uint32_t result = 0b01110010100000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(hw) << 21U;
    result |= static_cast<uint32_t>(imm16) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * MOVZ -- A64
 * Move wide with zero
 * MOVZ  <Wd>, #<imm>{, LSL #<shift>}
 * MOVZ  <Xd>, #<imm>{, LSL #<shift>}
 */
constexpr uint32_t movz(int32_t sf, int32_t hw, int32_t imm16, Register rd) noexcept {
    uint32_t result = 0b01010010100000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(hw) << 21U;
    result |= static_cast<uint32_t>(imm16) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * ORR (shifted register) -- A64
 * Bitwise OR (shifted register)
 * ORR  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
 * ORR  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
 */
constexpr uint32_t orr_log_shift(int32_t sf, int32_t shift, Register rm, int32_t imm6, Register rn, Register rd) noexcept {
    uint32_t result = 0b00101010000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(shift) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= truncate_imm<6>(imm6) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * RET -- A64
 * Return from subroutine
 * RET  {<Xn>}
 */
constexpr uint32_t ret(Register rn) noexcept {
    uint32_t result = 0b11010110010111110000000000000000U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    return result;
}

/**
 * SDIV -- A64
 * Signed Divide
 * SDIV  <Wd>, <Wn>, <Wm>
 * SDIV  <Xd>, <Xn>, <Xm>
 */
constexpr uint32_t sdiv(int32_t sf, Register rm, Register rn, Register rd) noexcept {
    uint32_t result = 0b00011010110000000000110000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * STLR -- A64
 * Store-Release Register
 * STLR  <Wt>, [<Xn|SP>{,#0}]
 * STLR  <Xt>, [<Xn|SP>{,#0}]
 */
constexpr uint32_t stlr(int32_t size, Register rn, Register rt) noexcept {
    uint32_t result = 0b00001000100111111111110000000000U;
    result |= static_cast<uint32_t>(size) << 30U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rt.encode()) << 0U;
    return result;
}

/**
 * STP -- A64
 * Store Pair of Registers
 * STP  <Wt1>, <Wt2>, [<Xn|SP>], #<imm>
 * STP  <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
 * STP  <Wt1>, <Wt2>, [<Xn|SP>, #<imm>]!
 * STP  <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
 * STP  <Wt1>, <Wt2>, [<Xn|SP>{, #<imm>}]
 * STP  <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
 */
constexpr uint32_t stp_gen(int32_t opc, int32_t imm7, Register rt2, Register rn, Register rt, StpGenSelector class_selector) noexcept {
    switch (class_selector) {
    case StpGenSelector::PostIndex: {
        uint32_t result = 0b00101000100000000000000000000000U;
        result |= static_cast<uint32_t>(opc) << 30U;
        result |= truncate_imm<7>(imm7) << 15U;
        result |= static_cast<uint32_t>(rt2.encode()) << 10U;
        result |= static_cast<uint32_t>(rn.encode()) << 5U;
        result |= static_cast<uint32_t>(rt.encode()) << 0U;
        return result;
    }
    case StpGenSelector::PreIndex: {
        uint32_t result = 0b00101001100000000000000000000000U;
        result |= static_cast<uint32_t>(opc) << 30U;
        result |= truncate_imm<7>(imm7) << 15U;
        result |= static_cast<uint32_t>(rt2.encode()) << 10U;
        result |= static_cast<uint32_t>(rn.encode()) << 5U;
        result |= static_cast<uint32_t>(rt.encode()) << 0U;
        return result;
    }
    case StpGenSelector::SignedOffset: {
        uint32_t result = 0b00101001000000000000000000000000U;
        result |= static_cast<uint32_t>(opc) << 30U;
        result |= truncate_imm<7>(imm7) << 15U;
        result |= static_cast<uint32_t>(rt2.encode()) << 10U;
        result |= static_cast<uint32_t>(rn.encode()) << 5U;
        result |= static_cast<uint32_t>(rt.encode()) << 0U;
        return result;
    }
    default:
        // Should never reach here if all cases are handled
        return 0U;
    }
}

/**
 * STR (immediate) -- A64
 * Store Register (immediate)
 * STR  <Wt>, [<Xn|SP>], #<simm>
 * STR  <Xt>, [<Xn|SP>], #<simm>
 * STR  <Wt>, [<Xn|SP>, #<simm>]!
 * STR  <Xt>, [<Xn|SP>, #<simm>]!
 * STR  <Wt>, [<Xn|SP>{, #<pimm>}]
 * STR  <Xt>, [<Xn|SP>{, #<pimm>}]
 */
constexpr uint32_t str_imm_gen(int32_t size, int32_t imm9, Register rn, Register rt, int32_t imm12, StrImmGenSelector class_selector) noexcept {
    switch (class_selector) {
    case StrImmGenSelector::PostIndex: {
        uint32_t result = 0b00111000000000000000010000000000U;
        result |= static_cast<uint32_t>(size) << 30U;
        result |= truncate_imm<9>(imm9) << 12U;
        result |= static_cast<uint32_t>(rn.encode()) << 5U;
        result |= static_cast<uint32_t>(rt.encode()) << 0U;
        return result;
    }
    case StrImmGenSelector::PreIndex: {
        uint32_t result = 0b00111000000000000000110000000000U;
        result |= static_cast<uint32_t>(size) << 30U;
        result |= truncate_imm<9>(imm9) << 12U;
        result |= static_cast<uint32_t>(rn.encode()) << 5U;
        result |= static_cast<uint32_t>(rt.encode()) << 0U;
        return result;
    }
    case StrImmGenSelector::UnsignedOffset: {
        uint32_t result = 0b00111001000000000000000000000000U;
        result |= static_cast<uint32_t>(size) << 30U;
        result |= truncate_imm<12>(imm12) << 10U;
        result |= static_cast<uint32_t>(rn.encode()) << 5U;
        result |= static_cast<uint32_t>(rt.encode()) << 0U;
        return result;
    }
    default:
        // Should never reach here if all cases are handled
        return 0U;
    }
}

/**
 * STR (register) -- A64
 * Store Register (register)
 * STR  <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
 * STR  <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
 */
constexpr uint32_t str_reg_gen(int32_t size, Register rm, int32_t option, int32_t s, Register rn, Register rt) noexcept {
    uint32_t result = 0b00111000001000000000100000000000U;
    result |= static_cast<uint32_t>(size) << 30U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= static_cast<uint32_t>(option) << 13U;
    result |= static_cast<uint32_t>(s) << 12U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rt.encode()) << 0U;
    return result;
}

/**
 * STUR -- A64
 * Store Register (unscaled)
 * STUR  <Wt>, [<Xn|SP>{, #<simm>}]
 * STUR  <Xt>, [<Xn|SP>{, #<simm>}]
 */
constexpr uint32_t stur_gen(int32_t size, int32_t imm9, Register rn, Register rt) noexcept {
    uint32_t result = 0b00111000000000000000000000000000U;
    result |= static_cast<uint32_t>(size) << 30U;
    result |= truncate_imm<9>(imm9) << 12U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rt.encode()) << 0U;
    return result;
}

/**
 * SUB (immediate) -- A64
 * Subtract (immediate)
 * SUB  <Wd|WSP>, <Wn|WSP>, #<imm>{, <shift>}
 * SUB  <Xd|SP>, <Xn|SP>, #<imm>{, <shift>}
 */
constexpr uint32_t sub_addsub_imm(int32_t sf, int32_t sh, int32_t imm12, Register rn, Register rd) noexcept {
    uint32_t result = 0b01010001000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(sh) << 22U;
    result |= truncate_imm<12>(imm12) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * SUB (shifted register) -- A64
 * Subtract (shifted register)
 * SUB  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
 * SUB  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
 */
constexpr uint32_t sub_addsub_shift(int32_t sf, int32_t shift, Register rm, int32_t imm6, Register rn, Register rd) noexcept {
    uint32_t result = 0b01001011000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(shift) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= truncate_imm<6>(imm6) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}

/**
 * SUBS (shifted register) -- A64
 * Subtract (shifted register), setting flags
 * SUBS  <Wd>, <Wn>, <Wm>{, <shift> #<amount>}
 * SUBS  <Xd>, <Xn>, <Xm>{, <shift> #<amount>}
 */
constexpr uint32_t subs_addsub_shift(int32_t sf, int32_t shift, Register rm, int32_t imm6, Register rn, Register rd) noexcept {
    uint32_t result = 0b01101011000000000000000000000000U;
    result |= static_cast<uint32_t>(sf) << 31U;
    result |= static_cast<uint32_t>(shift) << 22U;
    result |= static_cast<uint32_t>(rm.encode()) << 16U;
    result |= truncate_imm<6>(imm6) << 10U;
    result |= static_cast<uint32_t>(rn.encode()) << 5U;
    result |= static_cast<uint32_t>(rd.encode()) << 0U;
    return result;
}





} // namespace arm_asm
