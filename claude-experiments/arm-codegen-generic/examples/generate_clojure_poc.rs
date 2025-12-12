use arm_codegen_generic::{
    ArmCodeGen,
    simple_rust_generator::SimpleRustGenerator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arm = ArmCodeGen::new()?;

    // Instructions needed by quick-clojure-poc
    let instructions = vec![
        // Arithmetic (register)
        "AddAddsubShift",   // ADD Xd, Xn, Xm
        "SubAddsubShift",   // SUB Xd, Xn, Xm

        // Arithmetic (immediate)
        "AddAddsubImm",     // ADD Xd, Xn, #imm
        "SubAddsubImm",     // SUB Xd, Xn, #imm

        // Compare (SUBS with XZR destination)
        "SubsAddsubShift",  // CMP Xn, Xm (SUBS XZR, Xn, Xm)
        "SubsAddsubImm",    // CMP Xn, #imm (SUBS XZR, Xn, #imm)

        // Logical (register)
        "OrrLogShift",      // ORR/MOV Xd, Xn, Xm
        "AndLogShift",      // AND Xd, Xn, Xm
        "EorLogShift",      // EOR Xd, Xn, Xm
        "OrnLogShift",      // ORN/MVN Xd, Xn, Xm

        // Logical (immediate)
        "AndLogImm",        // AND Xd, Xn, #imm
        "OrrLogImm",        // ORR Xd, Xn, #imm

        // Move (wide immediate)
        "Movz",             // MOVZ Xd, #imm, LSL #shift
        "Movk",             // MOVK Xd, #imm, LSL #shift

        // Multiply/Divide
        "Madd",             // MADD/MUL Xd, Xn, Xm, Xa
        "Sdiv",             // SDIV Xd, Xn, Xm

        // Shifts (variable)
        "Lslv",             // LSL Xd, Xn, Xm
        "Asrv",             // ASR Xd, Xn, Xm
        "Lsrv",             // LSR Xd, Xn, Xm

        // Shifts (immediate) - these are bitfield instructions
        "Ubfm",             // UBFM/LSL/LSR (immediate)
        "Sbfm",             // SBFM/ASR (immediate)

        // Load/Store (immediate offset)
        "LdrImmGen",        // LDR Xt, [Xn, #imm]
        "StrImmGen",        // STR Xt, [Xn, #imm]
        "LdrbImm",          // LDRB Wt, [Xn, #imm]
        "StrbImm",          // STRB Wt, [Xn, #imm]

        // Load/Store (unscaled immediate - signed 9-bit offset)
        "LdurGen",          // LDUR Xt, [Xn, #simm9]
        "SturGen",          // STUR Xt, [Xn, #simm9]

        // Load/Store pair
        "LdpGen",           // LDP Xt1, Xt2, [Xn, #imm]
        "StpGen",           // STP Xt1, Xt2, [Xn, #imm]

        // Branches
        "BCond",            // B.cond label
        "BUncond",          // B label
        "Blr",              // BLR Xn
        "Ret",              // RET

        // Floating point
        "FaddFloat",        // FADD Dd, Dn, Dm
        "FsubFloat",        // FSUB Dd, Dn, Dm
        "FmulFloat",        // FMUL Dd, Dn, Dm
        "FdivFloat",        // FDIV Dd, Dn, Dm
        "FmovFloatGen",     // FMOV between GP and FP registers
        "ScvtfFloatInt",    // SCVTF Dd, Xn (int to float)

        // PC-relative address
        "Adr",              // ADR Xd, label

        // Conditional select
        "Csinc",            // CSINC Xd, Xn, Xm, cond (CSET when Rn=Rm=XZR)

        // System
        "Nop",              // NOP
    ];

    let code = arm.generate(SimpleRustGenerator, instructions);
    print!("{}", code);

    Ok(())
}
