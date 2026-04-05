use arm_codegen_generic::{ArmCodeGen, rust_function_generator::RustFunctionGenerator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;
    
    let instructions = vec![
        // Integer arithmetic  
        "AddAddsubImm",     // ADD Xd, Xn, #imm
        "AddAddsubShift",   // ADD Xd, Xn, Xm, shift
        "SubAddsubImm",     // SUB Xd, Xn, #imm
        "SubAddsubShift",   // SUB Xd, Xn, Xm, shift
        "MsubMsub",         // MSUB (for computing offsets)
        "Madd",             // MADD (multiply-add integer)
        
        // Move/immediate
        "MovMovzImm",       // MOV Xd, #imm16 (alias of MOVZ)
        "Movk",             // MOVK Xd, #imm16, LSL #shift
        "Movz",             // MOVZ
        "MovOrrLogShift",   // MOV Rd, Rm (register)
        "MovAddAddsubImm",  // MOV Rd, Rn (from SP)
        
        // Compare & branch
        "CmpSubsAddsubImm", // CMP Xn, #imm (alias of SUBS)
        "SubsAddsubImm",    // SUBS with immediate
        "SubsAddsubShift",  // SUBS with shift
        "BCond",            // B.cond
        "Bl",               // BL (call)
        "Blr",              // BLR (call register)
        "Br",               // BR (jump register)
        "Ret",              // RET
        
        // Load/store integer
        "LdrImmGen",        // LDR Xt, [Xn, #imm]
        "LdurGen",          // LDUR Xt, [Xn, #simm9]
        "StrImmGen",        // STR Xt, [Xn, #imm]
        "SturGen",          // STUR Xt, [Xn, #simm9]
        "LdpGen",           // LDP (load pair)
        "StpGen",           // STP (store pair)
        
        // Load/store SIMD/FP
        "LdrImmFpsimd",     // LDR Qt, [Xn, #imm]
        "LdrRegFpsimd",     // LDR Qt, [Xn, Xm]
        "StrImmFpsimd",     // STR Qt, [Xn, #imm]
        "StrRegFpsimd",     // STR Qt, [Xn, Xm]
        "Ld1AdvsimdMult",   // LD1 {Vt.4S}, [Xn]
        "St1AdvsimdMult",   // ST1 {Vt.4S}, [Xn]
        
        // NEON f32x4 arithmetic
        "FaddAdvsimd",      // FADD Vd.4S, Vn.4S, Vm.4S
        "FsubAdvsimd",      // FSUB
        "FmulAdvsimdVec",   // FMUL Vd.4S, Vn.4S, Vm.4S
        "FmlaAdvsimdVec",   // FMLA Vd.4S, Vn.4S, Vm.4S (FMA!)
        "FmaxAdvsimd",      // FMAX
        "FnegAdvsimd",      // FNEG Vd.4S, Vn.4S
        "FsqrtAdvsimd",     // FSQRT Vd.4S, Vn.4S
        "FcmpFloat",        // FCMP
        
        // NEON utility
        "DupAdvsimdGen",    // DUP Vd.4S, Wn (broadcast scalar to vector)
        "DupAdvsimdElt",    // DUP Vd.4S, Vn.S[idx] (broadcast lane)
        "FmovAdvsimd",      // FMOV Vd.4S, #imm (vector immediate)
        "FmovFloat",        // FMOV (register)
        "FmovFloatGen",     // FMOV between GP and FP regs
        
        // Shifts
        "LslLslv",          // LSL Xd, Xn, Xm
        "LslUbfm",          // LSL Xd, Xn, #shift
        "AsrSbfm",          // ASR
    ];
    
    let code = arm.generate(RustFunctionGenerator, instructions);
    print!("{}", code);
    
    Ok(())
}
