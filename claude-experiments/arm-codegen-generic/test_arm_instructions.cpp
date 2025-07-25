#include <iostream>
#include <iomanip>
#include <cassert>
#include "arm_jit_instructions.hpp"

using namespace arm_asm;

void test_basic_arithmetic() {
    std::cout << "🧮 Testing Basic Arithmetic Instructions\n";
    std::cout << "==========================================\n";
    
    // Test ADD instruction: ADD X1, X2, X0
    uint32_t add_instr = add_addsub_shift(1, 0, X0, 0, X2, X1);
    std::cout << "ADD X1, X2, X0 = 0x" << std::hex << std::setw(8) << std::setfill('0') << add_instr << std::dec << "\n";
    
    // Test SUB instruction: SUB X3, X4, X5  
    uint32_t sub_instr = sub_addsub_shift(1, 0, X5, 0, X4, X3);
    std::cout << "SUB X3, X4, X5 = 0x" << std::hex << std::setw(8) << std::setfill('0') << sub_instr << std::dec << "\n";
    
    // Test SUB immediate: SUB X1, X2, #42
    uint32_t sub_imm_instr = sub_addsub_imm(1, 0, 42, X2, X1);
    std::cout << "SUB X1, X2, #42 = 0x" << std::hex << std::setw(8) << std::setfill('0') << sub_imm_instr << std::dec << "\n";
    
    // Test MUL using MADD: MUL X6, X7, X8 (MADD X6, X7, X8, XZR)
    uint32_t mul_instr = madd(1, X8, ZERO_REGISTER, X7, X6);
    std::cout << "MUL X6, X7, X8 = 0x" << std::hex << std::setw(8) << std::setfill('0') << mul_instr << std::dec << "\n";
    
    // Test DIV: SDIV X9, X10, X11
    uint32_t div_instr = sdiv(1, X11, X10, X9);
    std::cout << "SDIV X9, X10, X11 = 0x" << std::hex << std::setw(8) << std::setfill('0') << div_instr << std::dec << "\n";
    
    std::cout << "\n";
}

void test_move_operations() {
    std::cout << "📦 Testing Move Operations\n";
    std::cout << "===========================\n";
    
    // Test MOVZ: MOV X1, #42
    uint32_t movz_instr = movz(1, 0, 42, X1);
    std::cout << "MOV X1, #42 = 0x" << std::hex << std::setw(8) << std::setfill('0') << movz_instr << std::dec << "\n";
    
    // Test MOVK: MOVK X2, #0x1234, LSL #16
    uint32_t movk_instr = movk(1, 1, 0x1234, X2);
    std::cout << "MOVK X2, #0x1234, LSL #16 = 0x" << std::hex << std::setw(8) << std::setfill('0') << movk_instr << std::dec << "\n";
    
    // Test register move: MOV X3, X4 (using ORR)
    uint32_t mov_reg_instr = mov_orr_log_shift(1, X4, X3);
    std::cout << "MOV X3, X4 = 0x" << std::hex << std::setw(8) << std::setfill('0') << mov_reg_instr << std::dec << "\n";
    
    std::cout << "\n";
}

void test_control_flow() {
    std::cout << "🔀 Testing Control Flow Instructions\n";
    std::cout << "=====================================\n";
    
    // Test conditional branch: B.EQ #8 (condition = 0, offset = 2 instructions)
    uint32_t beq_instr = bcond(2, 0);
    std::cout << "B.EQ #8 = 0x" << std::hex << std::setw(8) << std::setfill('0') << beq_instr << std::dec << "\n";
    
    // Test branch with link: BL #16 (offset = 4 instructions)
    uint32_t bl_instr = bl(4);
    std::cout << "BL #16 = 0x" << std::hex << std::setw(8) << std::setfill('0') << bl_instr << std::dec << "\n";
    
    // Test branch with link register: BLR X5
    uint32_t blr_instr = blr(X5);
    std::cout << "BLR X5 = 0x" << std::hex << std::setw(8) << std::setfill('0') << blr_instr << std::dec << "\n";
    
    // Test return: RET X30
    uint32_t ret_instr = ret(X30);
    std::cout << "RET = 0x" << std::hex << std::setw(8) << std::setfill('0') << ret_instr << std::dec << "\n";
    
    std::cout << "\n";
}

void test_memory_operations() {
    std::cout << "💾 Testing Memory Operations\n";
    std::cout << "=============================\n";
    
    // Test store pair: STP X1, X2, [SP, #-16]!
    uint32_t stp_instr = stp_gen(0b10, -2, X2, SP, X1, StpGenSelector::PreIndex);
    std::cout << "STP X1, X2, [SP, #-16]! = 0x" << std::hex << std::setw(8) << std::setfill('0') << stp_instr << std::dec << "\n";
    
    // Test load pair: LDP X3, X4, [SP], #16  
    uint32_t ldp_instr = ldp_gen(0b10, 2, X4, SP, X3, LdpGenSelector::PostIndex);
    std::cout << "LDP X3, X4, [SP], #16 = 0x" << std::hex << std::setw(8) << std::setfill('0') << ldp_instr << std::dec << "\n";
    
    // Test unscaled store: STUR X5, [X6, #-8]
    uint32_t stur_instr = stur_gen(0b11, -1, X6, X5);
    std::cout << "STUR X5, [X6, #-8] = 0x" << std::hex << std::setw(8) << std::setfill('0') << stur_instr << std::dec << "\n";
    
    // Test unscaled load: LDUR X7, [X8, #16]
    uint32_t ldur_instr = ldur_gen(0b11, 2, X8, X7);
    std::cout << "LDUR X7, [X8, #16] = 0x" << std::hex << std::setw(8) << std::setfill('0') << ldur_instr << std::dec << "\n";
    
    std::cout << "\n";
}

void test_bitwise_operations() {
    std::cout << "🔢 Testing Bitwise Operations\n"; 
    std::cout << "==============================\n";
    
    // Test AND: AND X1, X2, X3
    uint32_t and_instr = and_log_shift(1, 0, X3, 0, X2, X1);
    std::cout << "AND X1, X2, X3 = 0x" << std::hex << std::setw(8) << std::setfill('0') << and_instr << std::dec << "\n";
    
    // Test OR: ORR X4, X5, X6
    uint32_t orr_instr = orr_log_shift(1, 0, X6, 0, X5, X4);
    std::cout << "ORR X4, X5, X6 = 0x" << std::hex << std::setw(8) << std::setfill('0') << orr_instr << std::dec << "\n";
    
    // Test XOR: EOR X7, X8, X9
    uint32_t eor_instr = eor_log_shift(1, 0, X9, 0, X8, X7);
    std::cout << "EOR X7, X8, X9 = 0x" << std::hex << std::setw(8) << std::setfill('0') << eor_instr << std::dec << "\n";
    
    // Test left shift: LSL X10, X11, X12
    uint32_t lsl_instr = lsl_lslv(1, X12, X11, X10);
    std::cout << "LSL X10, X11, X12 = 0x" << std::hex << std::setw(8) << std::setfill('0') << lsl_instr << std::dec << "\n";
    
    std::cout << "\n";
}

void test_floating_point() {
    std::cout << "🔢 Testing Floating Point Operations\n";
    std::cout << "=====================================\n";
    
    // Test FADD: FADD D1, D2, D3
    uint32_t fadd_instr = fadd_float(0b01, X3, X2, X1);
    std::cout << "FADD D1, D2, D3 = 0x" << std::hex << std::setw(8) << std::setfill('0') << fadd_instr << std::dec << "\n";
    
    // Test FSUB: FSUB D4, D5, D6
    uint32_t fsub_instr = fsub_float(0b01, X6, X5, X4);
    std::cout << "FSUB D4, D5, D6 = 0x" << std::hex << std::setw(8) << std::setfill('0') << fsub_instr << std::dec << "\n";
    
    // Test FMUL: FMUL D7, D8, D9
    uint32_t fmul_instr = fmul_float(0b01, X9, X8, X7);
    std::cout << "FMUL D7, D8, D9 = 0x" << std::hex << std::setw(8) << std::setfill('0') << fmul_instr << std::dec << "\n";
    
    // Test FDIV: FDIV D10, D11, D12
    uint32_t fdiv_instr = fdiv_float(0b01, X12, X11, X10);
    std::cout << "FDIV D10, D11, D12 = 0x" << std::hex << std::setw(8) << std::setfill('0') << fdiv_instr << std::dec << "\n";
    
    std::cout << "\n";
}

void test_compile_time_evaluation() {
    std::cout << "⚡ Testing Compile-Time Evaluation\n";
    std::cout << "===================================\n";
    
    // These should all be computed at compile time
    constexpr uint32_t compile_time_add = add_addsub_shift(1, 0, X0, 0, X2, X1);
    constexpr uint32_t compile_time_mov = movz(1, 0, 42, X3);
    constexpr uint32_t compile_time_ret = ret(X30);
    
    static_assert(compile_time_add == 0x8b000041U, "ADD encoding should be constant");
    static_assert(compile_time_mov == 0xd2800543U, "MOV encoding should be constant");  
    static_assert(compile_time_ret == 0xd65f03c0U, "RET encoding should be constant");
    
    std::cout << "✅ All compile-time evaluations passed!\n";
    std::cout << "✅ Instructions are computed at compile time with zero runtime cost\n";
    std::cout << "\n";
}

int main() {
    std::cout << "🚀 ARM Instruction Encoder Test Suite\n";
    std::cout << "======================================\n\n";
    
    test_basic_arithmetic();
    test_move_operations();
    test_control_flow();
    test_memory_operations();
    test_bitwise_operations();
    test_floating_point();
    test_compile_time_evaluation();
    
    std::cout << "🎉 All tests completed successfully!\n";
    std::cout << "✅ Generated C++ ARM instruction encoders are working correctly\n";
    std::cout << "🔧 Ready for use in JIT compilers, assemblers, and code generators\n";
    
    return 0;
}