//! AArch64 (Apple Silicon / ARM64) target description.
//!
//! Provides a ready-to-use `Target` implementation so users only need
//! to implement the `Function` trait for their IR.
//!
//! # Example
//! ```ignore
//! use regalloc::aarch64::AArch64Target;
//! use regalloc::aarch64;
//!
//! let target = AArch64Target::default();
//! let mut allocator = LinearScanAllocator;
//! let result = allocator.allocate(&my_function, &target)?;
//! ```

use crate::target::{CallingConvention, RegInfo};
use crate::types::*;

// ============================================================
// Register numbering
// ============================================================

// GPRs: x0-x30 → PReg(0)..=PReg(30)
// FP/SIMD: v0-v31 → PReg(32)..=PReg(63)
//
// PReg(31) is intentionally skipped — it's the zero register / SP
// encoding in hardware, so we reserve that slot.

/// GPR physical register from architectural number (0-30).
pub const fn gpr(n: u16) -> PReg {
    assert!(n <= 30, "GPR number must be 0-30");
    PReg(n)
}

/// FP/SIMD physical register from architectural number (0-31).
pub const fn fp(n: u16) -> PReg {
    assert!(n <= 31, "FP number must be 0-31");
    PReg(32 + n)
}

// Well-known GPRs
pub const X0: PReg = gpr(0);
pub const X1: PReg = gpr(1);
pub const X2: PReg = gpr(2);
pub const X3: PReg = gpr(3);
pub const X4: PReg = gpr(4);
pub const X5: PReg = gpr(5);
pub const X6: PReg = gpr(6);
pub const X7: PReg = gpr(7);
pub const X8: PReg = gpr(8);   // indirect result location
pub const X9: PReg = gpr(9);
pub const X10: PReg = gpr(10);
pub const X11: PReg = gpr(11);
pub const X12: PReg = gpr(12);
pub const X13: PReg = gpr(13);
pub const X14: PReg = gpr(14);
pub const X15: PReg = gpr(15);
pub const X16: PReg = gpr(16); // IP0 — intra-procedure scratch
pub const X17: PReg = gpr(17); // IP1 — intra-procedure scratch
pub const X18: PReg = gpr(18); // platform register (reserved on Apple)
pub const X19: PReg = gpr(19);
pub const X20: PReg = gpr(20);
pub const X21: PReg = gpr(21);
pub const X22: PReg = gpr(22);
pub const X23: PReg = gpr(23);
pub const X24: PReg = gpr(24);
pub const X25: PReg = gpr(25);
pub const X26: PReg = gpr(26);
pub const X27: PReg = gpr(27);
pub const X28: PReg = gpr(28);
pub const FP: PReg = gpr(29);  // frame pointer
pub const LR: PReg = gpr(30);  // link register
pub const SP: PReg = PReg(31); // stack pointer (special encoding)

// Well-known FP/SIMD regs
pub const V0: PReg = fp(0);
pub const V1: PReg = fp(1);
pub const V2: PReg = fp(2);
pub const V3: PReg = fp(3);
pub const V4: PReg = fp(4);
pub const V5: PReg = fp(5);
pub const V6: PReg = fp(6);
pub const V7: PReg = fp(7);

// ============================================================
// Register classes
// ============================================================

/// General-purpose register class (x0-x30, 64-bit integers / pointers).
pub const GPR: RegClass = RegClass(0);

/// Floating-point / SIMD register class (v0-v31, 128-bit NEON).
pub const FP_SIMD: RegClass = RegClass(1);

// ============================================================
// Target
// ============================================================

/// AArch64 target following the AAPCS64 calling convention
/// (as used on macOS / Apple Silicon).
///
/// Differences from the generic AAPCS64:
/// - x18 is reserved (platform register on Apple).
///
/// Call `AArch64Target::default()` for standard Apple Silicon settings,
/// or use the builder methods to customize.
#[derive(Clone, Debug)]
pub struct AArch64Target {
    /// Whether to reserve x29 as frame pointer (default: true).
    pub use_frame_pointer: bool,
    /// Whether x18 is reserved (default: true on Apple).
    pub reserve_x18: bool,
}

impl Default for AArch64Target {
    fn default() -> Self {
        AArch64Target {
            use_frame_pointer: true,
            reserve_x18: true,
        }
    }
}

impl AArch64Target {
    /// Linux-style: x18 is not reserved, frame pointer optional.
    pub fn linux() -> Self {
        AArch64Target {
            use_frame_pointer: false,
            reserve_x18: false,
        }
    }

    pub fn with_frame_pointer(mut self, fp: bool) -> Self {
        self.use_frame_pointer = fp;
        self
    }

    pub fn with_reserve_x18(mut self, r: bool) -> Self {
        self.reserve_x18 = r;
        self
    }
}

// ---- Static register tables ----

// GPR names indexed by architectural number 0-30, then SP at index 31.
static GPR_NAMES: [&str; 32] = [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
    "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24", "x25", "x26", "x27", "x28", "fp", "lr", "sp",
];

static FP_NAMES: [&str; 32] = [
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
];

// All GPR PRegs (x0-x30).
static ALL_GPRS: [PReg; 31] = {
    let mut regs = [PReg(0); 31];
    let mut i = 0u16;
    while i < 31 {
        regs[i as usize] = PReg(i);
        i += 1;
    }
    regs
};

// All FP PRegs (v0-v31).
static ALL_FPS: [PReg; 32] = {
    let mut regs = [PReg(0); 32];
    let mut i = 0u16;
    while i < 32 {
        regs[i as usize] = PReg(32 + i);
        i += 1;
    }
    regs
};

// Argument GPRs: x0-x7.
static ARG_GPRS: [PReg; 8] = [
    PReg(0), PReg(1), PReg(2), PReg(3), PReg(4), PReg(5), PReg(6), PReg(7),
];

// Argument FP regs: v0-v7.
static ARG_FPS: [PReg; 8] = [
    PReg(32), PReg(33), PReg(34), PReg(35), PReg(36), PReg(37), PReg(38), PReg(39),
];

// Return GPRs: x0, x1 (for 128-bit returns).
static RET_GPRS: [PReg; 2] = [PReg(0), PReg(1)];

// Return FP regs: v0-v3 (for HFA returns).
static RET_FPS: [PReg; 4] = [PReg(32), PReg(33), PReg(34), PReg(35)];

static REG_CLASSES: [RegClass; 2] = [GPR, FP_SIMD];

impl RegInfo for AArch64Target {
    type RegIter<'a> = std::iter::Copied<std::slice::Iter<'a, PReg>>;

    fn reg_classes(&self) -> &[RegClass] {
        &REG_CLASSES
    }

    fn class_regs(&self, class: RegClass) -> Self::RegIter<'_> {
        match class.0 {
            0 => ALL_GPRS.iter().copied(),
            1 => ALL_FPS.iter().copied(),
            _ => [].iter().copied(),
        }
    }

    fn class_size(&self, class: RegClass) -> usize {
        match class.0 {
            0 => 31, // x0-x30
            1 => 32, // v0-v31
            _ => 0,
        }
    }

    fn reg_class_of(&self, reg: PReg) -> RegClass {
        if reg.0 < 32 {
            GPR
        } else {
            FP_SIMD
        }
    }

    fn reg_name(&self, reg: PReg) -> &str {
        if reg.0 < 32 {
            GPR_NAMES[reg.0 as usize]
        } else {
            FP_NAMES[(reg.0 - 32) as usize]
        }
    }

    fn class_name(&self, class: RegClass) -> &str {
        match class.0 {
            0 => "GPR",
            1 => "FP/SIMD",
            _ => "unknown",
        }
    }

    fn spill_size(&self, class: RegClass) -> u32 {
        match class.0 {
            0 => 8,  // 64-bit GPR
            1 => 16, // 128-bit NEON
            _ => 8,
        }
    }

    fn spill_align(&self, class: RegClass) -> u32 {
        match class.0 {
            0 => 8,
            1 => 16,
            _ => 8,
        }
    }
}

impl CallingConvention for AArch64Target {
    fn callee_saved(&self) -> &[PReg] {
        // Combine GPR and FP callee-saved. We return a static slice
        // for efficiency — the combined list is always the same.
        static CALLEE_SAVED_ALL: [PReg; 18] = [
            // GPR: x19-x28
            PReg(19), PReg(20), PReg(21), PReg(22), PReg(23),
            PReg(24), PReg(25), PReg(26), PReg(27), PReg(28),
            // FP: v8-v15
            PReg(40), PReg(41), PReg(42), PReg(43),
            PReg(44), PReg(45), PReg(46), PReg(47),
        ];
        &CALLEE_SAVED_ALL
    }

    fn caller_saved(&self) -> &[PReg] {
        static CALLER_SAVED_ALL: [PReg; 40] = [
            // GPR: x0-x15
            PReg(0), PReg(1), PReg(2), PReg(3), PReg(4), PReg(5), PReg(6), PReg(7),
            PReg(8), PReg(9), PReg(10), PReg(11), PReg(12), PReg(13), PReg(14), PReg(15),
            // FP: v0-v7, v16-v31
            PReg(32), PReg(33), PReg(34), PReg(35), PReg(36), PReg(37), PReg(38), PReg(39),
            PReg(48), PReg(49), PReg(50), PReg(51), PReg(52), PReg(53), PReg(54), PReg(55),
            PReg(56), PReg(57), PReg(58), PReg(59), PReg(60), PReg(61), PReg(62), PReg(63),
        ];
        &CALLER_SAVED_ALL
    }

    fn arg_regs(&self, class: RegClass) -> &[PReg] {
        match class.0 {
            0 => &ARG_GPRS,
            1 => &ARG_FPS,
            _ => &[],
        }
    }

    fn ret_regs(&self, class: RegClass) -> &[PReg] {
        match class.0 {
            0 => &RET_GPRS,
            1 => &RET_FPS,
            _ => &[],
        }
    }

    fn stack_pointer(&self) -> Option<PReg> {
        Some(SP)
    }

    fn frame_pointer(&self) -> Option<PReg> {
        if self.use_frame_pointer {
            Some(FP)
        } else {
            None
        }
    }

    fn reserved_regs(&self) -> &[PReg] {
        // We build a few static variants to avoid allocation.
        static RESERVED_FP_X18: [PReg; 5] = [
            PReg(16), PReg(17), // IP0, IP1
            PReg(18),           // platform register
            PReg(29),           // FP
            PReg(31),           // SP
        ];
        static RESERVED_FP_NO_X18: [PReg; 4] = [
            PReg(16), PReg(17),
            PReg(29),
            PReg(31),
        ];
        static RESERVED_NO_FP_X18: [PReg; 4] = [
            PReg(16), PReg(17),
            PReg(18),
            PReg(31),
        ];
        static RESERVED_NO_FP_NO_X18: [PReg; 3] = [
            PReg(16), PReg(17),
            PReg(31),
        ];

        match (self.use_frame_pointer, self.reserve_x18) {
            (true, true) => &RESERVED_FP_X18,
            (true, false) => &RESERVED_FP_NO_X18,
            (false, true) => &RESERVED_NO_FP_X18,
            (false, false) => &RESERVED_NO_FP_NO_X18,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::Target;

    #[test]
    fn default_target_sanity() {
        let t = AArch64Target::default();

        // 31 GPRs, 32 FP regs
        assert_eq!(t.class_size(GPR), 31);
        assert_eq!(t.class_size(FP_SIMD), 32);

        // Reserved: IP0, IP1, x18, FP, SP = 5
        assert_eq!(t.reserved_regs().len(), 5);

        // Allocatable GPRs: 31 - 5 reserved = 26
        // (x16, x17, x18, x29 are GPRs that are reserved; SP is PReg(31))
        let alloc_gprs = t.allocatable_regs(GPR);
        assert_eq!(alloc_gprs.len(), 27); // 31 - 4 (SP is not in ALL_GPRS)

        // Allocatable FP: all 32 (no FP regs are reserved)
        let alloc_fps = t.allocatable_regs(FP_SIMD);
        assert_eq!(alloc_fps.len(), 32);

        // Names
        assert_eq!(t.reg_name(X0), "x0");
        assert_eq!(t.reg_name(FP), "fp");
        assert_eq!(t.reg_name(LR), "lr");
        assert_eq!(t.reg_name(SP), "sp");
        assert_eq!(t.reg_name(V0), "v0");

        // Arg regs
        assert_eq!(t.arg_regs(GPR).len(), 8);
        assert_eq!(t.arg_regs(FP_SIMD).len(), 8);

        // Ret regs
        assert_eq!(t.ret_regs(GPR).len(), 2);
        assert_eq!(t.ret_regs(FP_SIMD).len(), 4);
    }

    #[test]
    fn linux_target_no_x18_reserved() {
        let t = AArch64Target::linux();
        // x18 is not reserved, FP is not reserved
        assert_eq!(t.reserved_regs().len(), 3); // IP0, IP1, SP
        assert!(!t.reserved_regs().contains(&PReg(18)));
        assert!(!t.reserved_regs().contains(&PReg(29)));
    }

    #[test]
    fn allocatable_excludes_reserved() {
        let t = AArch64Target::default();
        let alloc = t.allocatable_regs(GPR);
        for r in t.reserved_regs() {
            assert!(
                !alloc.contains(r),
                "reserved {:?} ({}) should not be allocatable",
                r,
                t.reg_name(*r)
            );
        }
    }

    #[test]
    fn reg_classes_correct() {
        let t = AArch64Target::default();
        // GPRs are class 0
        for i in 0..31u16 {
            assert_eq!(t.reg_class_of(PReg(i)), GPR);
        }
        assert_eq!(t.reg_class_of(SP), GPR); // SP is PReg(31), technically GPR encoding
        // FP regs are class 1
        for i in 32..64u16 {
            assert_eq!(t.reg_class_of(PReg(i)), FP_SIMD);
        }
    }

    /// End-to-end: use the AArch64 target with the linear scan allocator.
    #[test]
    fn aarch64_with_linear_scan() {
        use crate::allocator::RegisterAllocator;
        use crate::linear_scan::LinearScanAllocator;
        use crate::testing::*;

        let t = AArch64Target::default();

        // Build a simple function using GPR class
        let mut b = TestFunctionBuilder::new();
        let v0 = b.vreg(GPR);
        let v1 = b.vreg(GPR);
        let v2 = b.vreg(GPR);

        let _bb0 = b.block();
        b.inst(vec![def_reg(v0, GPR)]);
        b.inst(vec![def_reg(v1, GPR)]);
        b.inst(vec![def_reg(v2, GPR), use_reg(v0, GPR), use_reg(v1, GPR)]);
        b.ret(vec![use_reg(v2, GPR)]);

        let func = b.build();
        let alloc = LinearScanAllocator.allocate(&func, &t).unwrap();

        // Verify
        let verify_result = crate::verify::verify(&func, &t, &alloc);
        assert!(verify_result.is_ok(), "verify failed: {:?}", verify_result.err());

        // All allocations should be in allocatable GPRs.
        let allocatable = t.allocatable_regs(GPR);
        for (&(_inst, _op), &preg) in &alloc.inst_allocs {
            assert!(
                allocatable.contains(&preg),
                "allocated {:?} ({}) which is not allocatable",
                preg,
                t.reg_name(preg)
            );
        }
    }

    /// Mixed GPR + FP allocation on AArch64.
    #[test]
    fn aarch64_mixed_gpr_fp() {
        use crate::allocator::RegisterAllocator;
        use crate::linear_scan::LinearScanAllocator;
        use crate::testing::*;

        let t = AArch64Target::default();

        let mut b = TestFunctionBuilder::new();
        let vi = b.vreg(GPR);
        let vf = b.vreg(FP_SIMD);
        let vr = b.vreg(GPR);

        let _bb0 = b.block();
        b.inst(vec![def_reg(vi, GPR)]);
        b.inst(vec![def_reg(vf, FP_SIMD)]);
        b.inst(vec![def_reg(vr, GPR), use_reg(vi, GPR)]);
        b.ret(vec![use_reg(vr, GPR)]);

        let func = b.build();
        let alloc = LinearScanAllocator.allocate(&func, &t).unwrap();

        let verify_result = crate::verify::verify(&func, &t, &alloc);
        assert!(verify_result.is_ok(), "verify failed: {:?}", verify_result.err());

        // FP vreg should be in FP class (PReg 32-63).
        let fp_preg = alloc.get(InstId(1), 0).unwrap();
        assert!(
            fp_preg.0 >= 32 && fp_preg.0 < 64,
            "FP vreg should be in FP/SIMD class, got {:?} ({})",
            fp_preg,
            t.reg_name(fp_preg)
        );
    }
}
