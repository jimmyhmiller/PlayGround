use arm_codegen_generic::{ArmCodeGen, zig_function_generator::ZigFunctionGenerator};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let codegen = ArmCodeGen::new()?;

    // Get all basic ARM64 instructions (excluding advanced SIMD/crypto/system)
    let basic_instructions = vec![
        // Arithmetic - immediate
        "AddAddsubImm", "AddsAddsubImm", "SubAddsubImm", "SubsAddsubImm",
        "CmpSubsAddsubImm", "CmnAddsAddsubImm",

        // Arithmetic - register
        "AddAddsubShift", "AddsAddsubShift", "SubAddsubShift", "SubsAddsubShift",
        "CmpSubsAddsubShift", "CmnAddsAddsubShift",

        // Arithmetic - extended register
        "AddAddsubExt", "AddsAddsubExt", "SubAddsubExt", "SubsAddsubExt",
        "CmpSubsAddsubExt", "CmnAddsAddsubExt",

        // Logical - immediate
        "AndLogImm", "OrrLogImm", "EorLogImm", "AndsLogImm",

        // Logical - register
        "AndLogShift", "OrrLogShift", "EorLogShift", "AndsLogShift",
        "BicLogShift", "OrnLogShift", "EonLogShift", "BicsLogShift",

        // Move wide
        "Movz", "Movn", "Movk",

        // Move - register/SP
        "MovAddAddsubImm", "MovOrrLogShift",

        // Bitfield
        "Sbfm", "Bfm", "Ubfm",
        "AsrSbfm", "LslUbfm", "LsrUbfm",

        // Extract
        "Extr",

        // Multiply
        "Madd", "Msub", "Smaddl", "Smsubl", "Umaddl", "Umsubl",
        "MulMadd", "MnegMsub", "SmullSmaddl", "SmneglSmsubl",
        "UmullUmaddl", "UmneglUmsubl",

        // Divide
        "Sdiv", "Udiv",

        // Variable shifts
        "LslLslv", "LsrLsrv", "AsrAsrv", "RorRorv",

        // CRC32
        "Crc32b", "Crc32h", "Crc32w", "Crc32x",
        "Crc32cb", "Crc32ch", "Crc32cw", "Crc32cx",

        // Conditional select
        "Csel", "Csinc", "Csinv", "Csneg",
        "CsetCsinc", "CsetmCsinv", "CincCsinc", "CnegCsneg",

        // Conditional compare
        "Ccmn", "Ccmp",

        // Branches
        "Bl", "Br", "Blr", "Ret",

        // Conditional branches
        "Bcond",

        // Compare and branch
        "Cbz", "Cbnz", "Tbz", "Tbnz",

        // Unconditional branch
        "B",

        // PC-relative addressing
        "Adr", "Adrp",

        // Load/store - register offset
        "LdrRegGen", "LdrhRegGen", "LdrbRegGen", "LdrswRegGen", "LdrshRegGen", "LdrsbRegGen",
        "StrRegGen", "StrhRegGen", "StrbRegGen",

        // Load/store - immediate offset
        "LdrImmGen", "LdrhImmGen", "LdrbImmGen", "LdrswImmGen", "LdrshImmGen", "LdrsbImmGen",
        "StrImmGen", "StrhImmGen", "StrbImmGen",

        // Load/store - unscaled immediate
        "LdurGen", "LdurhGen", "LdurbGen", "LdurswGen", "LdurshGen", "LdursbGen",
        "SturGen", "SturhGen", "SturbGen",

        // Load/store pair
        "LdpGen", "StpGen",

        // Load/store exclusive
        "LdxrGen", "StxrGen", "LdaxrGen", "StlxrGen",
        "LdxpGen", "StxpGen", "LdaxpGen", "StlxpGen",

        // Load/store acquire/release
        "Ldar", "Stlr",

        // Atomic operations
        "Ldadd", "Ldclr", "Ldeor", "Ldset", "Ldsmax", "Ldsmin", "Ldumax", "Ldumin",
        "StaddLdadd", "StclrLdclr", "SteorLdeor", "StsetLdset",
        "StsmaxLdsmax", "StsminLdsmin", "StumaxLdumax", "StuminLdumin",

        // Compare and swap
        "Cas", "Casp",

        // Barriers
        "Dsb", "Dmb", "Isb",

        // System
        "Mrs", "Msr", "Sys", "Sysl",

        // Hints
        "Nop", "Yield", "Wfe", "Wfi", "Sev", "Sevl",

        // Exception generation
        "Svc", "Hvc", "Smc", "Brk", "Hlt",

        // Pointer authentication (if available)
        "Paciasp", "Autiasp", "Paciaz", "Autiaz",

        // Floating point - basic arithmetic
        "FaddFloat", "FsubFloat", "FmulFloat", "FdivFloat",
        "FnegFloat", "FabsFloat", "FsqrtFloat",

        // Floating point - multiply-add
        "FmaddFloat", "FmsubFloat", "FnmaddFloat", "FnmsubFloat",

        // Floating point - conversion
        "FcvtFloat", "FcvtzuFloatInt", "FcvtzsFloatInt",
        "ScvtfIntFloat", "UcvtfIntFloat",

        // Floating point - compare
        "FcmpFloat", "FcmpeFloat",

        // Floating point - conditional select
        "FcselFloat",

        // Floating point - move
        "FmovFloatGen", "FmovFloatImm",

        // SIMD - basic (selected common ones)
        "AddAdvsimd", "SubAdvsimd", "MulAdvsimdElem", "MulAdvsimdVec",
        "FaddAdvsimd", "FsubAdvsimd", "FmulAdvsimdVec",
    ];

    println!("Generating Zig code for {} ARM64 instructions...", basic_instructions.len());

    let generator = ZigFunctionGenerator;
    let zig_code = codegen.generate(generator, basic_instructions);

    // Write to file
    fs::write("arm64_instructions_complete.zig", &zig_code)?;

    println!("Generated complete ARM64 Zig library saved to 'arm64_instructions_complete.zig'");
    println!("File size: {} bytes", zig_code.len());

    // Count functions generated
    let function_count = zig_code.matches("pub fn ").count();
    let export_count = zig_code.matches("export fn ").count();

    println!("Generated {} Zig-native functions and {} C-export wrappers", function_count, export_count);

    Ok(())
}