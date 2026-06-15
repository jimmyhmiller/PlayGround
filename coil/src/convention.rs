//! Calling conventions as first-class data (`defcc`).
//!
//! In M1 a convention records its register/stack intent (informational) plus a
//! **lowering strategy**. Only the `:native <llvm-cc>` strategy is implemented:
//! it maps onto one of LLVM's built-in calling conventions and is applied to
//! both the function and every call site. Conventions whose register layout
//! LLVM's closed CC enum can't express are declared `:shim` and are rejected by
//! the checker until M2 implements the naked+inline-asm trampoline path.

#[derive(Debug, Clone)]
pub enum Lowering {
    /// Coincides with a built-in LLVM calling convention.
    Native(NativeCc),
    /// Exotic register layout; requires a naked + inline-asm shim (M2, not yet).
    Shim,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NativeCc {
    C,
    Fast,
    Cold,
}

impl NativeCc {
    /// Numeric LLVM calling-convention id (stable in LLVM's CallingConv.h).
    pub fn id(self) -> u32 {
        match self {
            NativeCc::C => 0,
            NativeCc::Fast => 8,
            NativeCc::Cold => 9,
        }
    }

    pub fn parse(s: &str) -> Option<NativeCc> {
        match s {
            "c" | "ccc" => Some(NativeCc::C),
            "fast" | "fastcc" => Some(NativeCc::Fast),
            "cold" | "coldcc" => Some(NativeCc::Cold),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Convention {
    pub name: String,
    pub params: Vec<String>,   // register intent (informational in M1)
    pub ret: Option<String>,   // register intent (informational in M1)
    pub clobber: Vec<String>,
    pub preserve: Vec<String>,
    pub lowering: Lowering,
}

impl Convention {
    /// The built-in `c` convention, available without a `defcc`.
    pub fn default_c() -> Convention {
        Convention {
            name: "c".to_string(),
            params: vec![],
            ret: None,
            clobber: vec![],
            preserve: vec![],
            lowering: Lowering::Native(NativeCc::C),
        }
    }

    /// LLVM call-conv id if this convention has a native lowering.
    pub fn native_id(&self) -> Option<u32> {
        match &self.lowering {
            Lowering::Native(cc) => Some(cc.id()),
            Lowering::Shim => None,
        }
    }

    pub fn is_shim(&self) -> bool {
        matches!(self.lowering, Lowering::Shim)
    }
}
