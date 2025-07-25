use arm_codegen_generic::{
    ArmCodeGen,
    cpp_function_generator::CppFunctionGenerator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Generating C++ ARM instructions for JIT compiler");
    println!("===================================================");
    
    // These are common ARM instructions used by JIT compilers and code generators
    let jit_instructions = vec![
        // Basic arithmetic and logic
        "AddAddsubShift",      // add(destination, a, b)
        "SubAddsubShift",      // sub(destination, a, b) 
        "SubAddsubImm",        // sub_imm(destination, a, b)
        "Madd",                // mul(destination, a, b) - using MADD with zero register
        "Sdiv",                // div(destination, a, b)
        
        // Bitwise operations
        "AsrSbfm",             // shift_right_imm(destination, a, b)
        "LslUbfm",             // shift_left_imm(destination, a, b) 
        "LslLslv",             // shift_left(dest, a, b)
        "AsrAsrv",             // shift_right(dest, a, b)
        "EorLogShift",         // xor(dest, a, b)
        "OrrLogShift",         // or(destination, a, b)
        "AndLogImm",           // and_imm(destination, a, b) and get_tag(destination, value)
        "AndLogShift",         // and(destination, a, b)
        
        // Move operations
        "Movz",                // mov_imm(destination, input) and mov_64_bit_num
        "Movk",                // mov_64_bit_num (move with keep)
        "MovOrrLogShift",      // mov_reg(destination, source)
        "MovAddAddsubImm",     // mov_sp(destination, source)
        
        // Control flow
        "Ret",                 // ret()
        "BCond",               // jump_equal, jump_not_equal, etc.
        "Bl",                  // branch_with_link(destination)
        "Blr",                 // branch_with_link_register(register)
        "Brk",                 // breakpoint()
        "Adr",                 // adr(destination, label_index)
        
        // Load/Store operations
        "StpGen",              // store_pair(reg1, reg2, destination, offset)
        "LdpGen",              // load_pair(reg1, reg2, destination, offset)
        "SturGen",             // store_on_stack/push_to_stack
        "LdurGen",             // load_from_stack/pop_from_stack
        "StrImmGen",           // store_on_heap
        "LdrRegGen",           // load_from_heap_with_reg_offset
        "StrRegGen",           // store_to_heap_with_reg_offset
        
        // Atomic operations
        "Ldar",                // atomic_load
        "Stlr",                // atomic_store
        "Cas",                 // compare_and_swap
        
        // Comparison operations
        "CmpSubsAddsubShift",  // compare(a, b)
        "SubsAddsubShift",     // compare_bool (first instruction)
        "CsetCsinc",           // compare_bool (second instruction)
        
        // Stack pointer operations
        "AddAddsubImm",        // add_stack_pointer
        
        // Floating point operations
        "FmovFloatGen",        // fmov
        "FaddFloat",           // fadd
        "FsubFloat",           // fsub
        "FmulFloat",           // fmul
        "FdivFloat",           // fdiv
    ];
    
    let arm = ArmCodeGen::new()?;
    
    // Generate C++ code for all JIT instructions
    let cpp_code = arm.generate(CppFunctionGenerator, jit_instructions.clone());
    
    // Count generated functions
    let func_count = cpp_code.lines()
        .filter(|line| line.trim().starts_with("constexpr uint32_t ") && !line.contains("truncate_imm"))
        .count();
    
    println!("📊 Statistics:");
    println!("  Instructions requested: {}", jit_instructions.len());
    println!("  C++ functions generated: {}", func_count);
    println!("  Total code size: {} KB", cpp_code.len() / 1024);
    
    println!("\n🎯 Generated C++ function signatures for JIT compiler:");
    println!("```cpp");
    
    // Show function signatures
    for line in cpp_code.lines() {
        if line.trim().starts_with("constexpr uint32_t ") && !line.contains("truncate_imm") {
            println!("{}", line.trim());
        }
    }
    
    println!("```");
    
    println!("\n💾 Writing C++ code to arm_jit_instructions.hpp...");
    std::fs::write("arm_jit_instructions.hpp", cpp_code)?;
    
    println!("\n✅ C++ ARM instruction encoders generated for JIT compiler!");
    println!("📁 File: arm_jit_instructions.hpp");
    println!("🔧 Include this file in your C++ project to use the ARM encoders");
    
    Ok(())
}