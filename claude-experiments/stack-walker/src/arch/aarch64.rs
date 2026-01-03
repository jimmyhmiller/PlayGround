//! aarch64 stack walking implementation
//!
//! Frame pointer-based unwinding for ARM64 architecture, including
//! Pointer Authentication Code (PAC) support for Apple Silicon.
//!
//! ## Stack Frame Layout
//!
//! ```text
//! Higher addresses
//! ┌─────────────────┐
//! │   arguments     │
//! ├─────────────────┤
//! │  saved LR (x30) │  [fp + 8] - return address
//! ├─────────────────┤
//! │  saved FP (x29) │  [fp] ← current fp points here
//! ├─────────────────┤
//! │  saved regs     │
//! ├─────────────────┤
//! │  local vars     │
//! ├─────────────────┤
//! │                 │  ← current sp
//! └─────────────────┘
//! Lower addresses
//! ```

use crate::arch::{StackWalker, UnwindRegisters, UnwindResult};
use crate::error::StackWalkError;
use crate::frame::{Frame, FrameAddress, UnwindMethod};
use crate::memory::MemoryReader;

/// Pointer authentication mask for ARM64
///
/// On Apple Silicon with PAC, the upper bits of pointers may contain
/// authentication codes that must be stripped before use.
#[derive(Debug, Clone, Copy)]
pub struct PtrAuthMask(u64);

impl PtrAuthMask {
    /// Create a new pointer authentication mask
    ///
    /// The mask should have 1s in the bits that are valid address bits,
    /// and 0s in the bits used for PAC.
    pub fn new(mask: u64) -> Self {
        Self(mask)
    }

    /// Strip pointer authentication bits from an address
    #[inline]
    pub fn strip(&self, address: u64) -> u64 {
        address & self.0
    }

    /// No masking (all bits are valid address bits)
    ///
    /// Use this for systems without PAC or when PAC is disabled.
    pub fn none() -> Self {
        Self(!0u64)
    }

    /// Apple Silicon user-space mask
    ///
    /// On Apple Silicon, user-space addresses use the lower 48 bits.
    /// The upper 16 bits may contain PAC codes.
    pub fn apple_silicon_user() -> Self {
        // Mask for 48-bit addresses (bits 0-47)
        Self(0x0000_FFFF_FFFF_FFFF)
    }

    /// Get the raw mask value
    pub fn mask(&self) -> u64 {
        self.0
    }
}

impl Default for PtrAuthMask {
    fn default() -> Self {
        Self::none()
    }
}

/// aarch64 registers needed for unwinding
#[derive(Debug, Clone, Default)]
pub struct UnwindRegsAarch64 {
    /// Program counter
    pub pc: u64,
    /// Stack pointer (SP)
    pub sp: u64,
    /// Frame pointer (X29)
    pub fp: u64,
    /// Link register (X30) - contains return address
    pub lr: u64,
}

impl UnwindRegsAarch64 {
    /// Create a new register set
    pub fn new(pc: u64, sp: u64, fp: u64, lr: u64) -> Self {
        Self { pc, sp, fp, lr }
    }

    /// Create registers with only PC and FP (common case)
    pub fn from_pc_fp(pc: u64, fp: u64) -> Self {
        Self {
            pc,
            sp: 0,
            fp,
            lr: 0,
        }
    }
}

impl UnwindRegisters for UnwindRegsAarch64 {
    #[inline]
    fn instruction_pointer(&self) -> u64 {
        self.pc
    }

    #[inline]
    fn set_instruction_pointer(&mut self, value: u64) {
        self.pc = value;
    }

    #[inline]
    fn stack_pointer(&self) -> u64 {
        self.sp
    }

    #[inline]
    fn set_stack_pointer(&mut self, value: u64) {
        self.sp = value;
    }

    #[inline]
    fn frame_pointer(&self) -> Option<u64> {
        if self.fp != 0 {
            Some(self.fp)
        } else {
            None
        }
    }

    #[inline]
    fn set_frame_pointer(&mut self, value: Option<u64>) {
        self.fp = value.unwrap_or(0);
    }

    fn arch_name() -> &'static str {
        "aarch64"
    }
}

/// aarch64 stack walker using frame pointer chain
#[derive(Debug, Clone)]
pub struct Aarch64StackWalker {
    /// Pointer authentication mask for stripping PAC codes
    ptr_auth_mask: PtrAuthMask,
}

impl Aarch64StackWalker {
    /// Create a new aarch64 stack walker with no PAC masking
    pub fn new() -> Self {
        Self {
            ptr_auth_mask: PtrAuthMask::none(),
        }
    }

    /// Create a stack walker with the specified PAC mask
    pub fn with_ptr_auth_mask(mask: PtrAuthMask) -> Self {
        Self { ptr_auth_mask: mask }
    }

    /// Create a stack walker configured for Apple Silicon
    pub fn apple_silicon() -> Self {
        Self {
            ptr_auth_mask: PtrAuthMask::apple_silicon_user(),
        }
    }

    /// Get the current PAC mask
    pub fn ptr_auth_mask(&self) -> PtrAuthMask {
        self.ptr_auth_mask
    }

    /// Set the PAC mask
    pub fn set_ptr_auth_mask(&mut self, mask: PtrAuthMask) {
        self.ptr_auth_mask = mask;
    }
}

impl Default for Aarch64StackWalker {
    fn default() -> Self {
        Self::new()
    }
}

impl StackWalker for Aarch64StackWalker {
    type Registers = UnwindRegsAarch64;

    fn unwind_one<M: MemoryReader>(
        &self,
        current_regs: &Self::Registers,
        memory: &mut M,
        frame_index: usize,
        is_first_frame: bool,
    ) -> UnwindResult<Self::Registers> {
        let address = if is_first_frame {
            FrameAddress::InstructionPointer(current_regs.pc)
        } else {
            FrameAddress::ReturnAddress(current_regs.pc)
        };

        let frame = Frame::new(
            frame_index,
            address,
            current_regs.sp,
            Some(current_regs.fp),
            if is_first_frame {
                UnwindMethod::InitialFrame
            } else {
                UnwindMethod::FramePointer
            },
        );

        let fp = current_regs.fp;

        // End of stack: frame pointer is null
        if fp == 0 {
            return UnwindResult::EndOfStack { frame };
        }

        // Read caller's frame pointer from [fp]
        let new_fp = match memory.read_u64(fp) {
            Ok(v) => v,
            Err(e) => return UnwindResult::Failed { frame, error: e },
        };

        // Read return address (saved LR) from [fp + 8]
        let return_address_raw = match memory.read_u64(fp.wrapping_add(8)) {
            Ok(v) => v,
            Err(e) => return UnwindResult::Failed { frame, error: e },
        };

        // Strip PAC bits from return address
        let return_address = self.ptr_auth_mask.strip(return_address_raw);

        // New stack pointer = old frame pointer + 16 (after saved FP and LR)
        let new_sp = fp.wrapping_add(16);

        // Validate monotonic progress
        if new_fp != 0 && new_fp <= fp {
            return UnwindResult::Failed {
                frame,
                error: StackWalkError::InvalidFramePointer {
                    fp: new_fp,
                    previous_fp: fp,
                },
            };
        }

        if new_sp <= current_regs.sp {
            return UnwindResult::Failed {
                frame,
                error: StackWalkError::InvalidStackPointer {
                    sp: new_sp,
                    previous_sp: current_regs.sp,
                },
            };
        }

        // Check for end of stack
        if new_fp == 0 {
            let next_regs = UnwindRegsAarch64 {
                pc: return_address,
                sp: new_sp,
                fp: 0,
                lr: return_address,
            };
            return UnwindResult::Unwound {
                frame,
                next_registers: next_regs,
            };
        }

        let next_regs = UnwindRegsAarch64 {
            pc: return_address,
            sp: new_sp,
            fp: new_fp,
            lr: return_address,
        };

        UnwindResult::Unwound {
            frame,
            next_registers: next_regs,
        }
    }
}

/// Capture current aarch64 registers
///
/// This function uses inline assembly to capture the current register state.
#[cfg(target_arch = "aarch64")]
pub fn capture_current_registers() -> UnwindRegsAarch64 {
    let pc: u64;
    let sp: u64;
    let fp: u64;
    let lr: u64;

    unsafe {
        std::arch::asm!(
            "adr {pc}, .",
            "mov {sp}, sp",
            "mov {fp}, x29",
            "mov {lr}, x30",
            pc = out(reg) pc,
            sp = out(reg) sp,
            fp = out(reg) fp,
            lr = out(reg) lr,
            options(nostack, nomem),
        );
    }

    UnwindRegsAarch64 { pc, sp, fp, lr }
}

/// Capture current registers (stub for non-aarch64)
#[cfg(not(target_arch = "aarch64"))]
pub fn capture_current_registers() -> UnwindRegsAarch64 {
    panic!("capture_current_registers() is only available on aarch64")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::WalkConfig;
    use crate::memory::SliceMemoryReader;

    fn build_test_stack() -> (Vec<u8>, u64, UnwindRegsAarch64) {
        let stack_base = 0x7fff_0000u64;
        let mut stack = vec![0u8; 0x200];

        // Frame 0 at fp=0x7fff0080, points to Frame 1 at 0x7fff0100
        let frame0_offset = 0x80usize;
        stack[frame0_offset..frame0_offset + 8]
            .copy_from_slice(&(stack_base + 0x100).to_le_bytes());
        stack[frame0_offset + 8..frame0_offset + 16]
            .copy_from_slice(&0x4000_2000u64.to_le_bytes());

        // Frame 1 at fp=0x7fff0100, points to Frame 2 at 0x7fff0180
        let frame1_offset = 0x100usize;
        stack[frame1_offset..frame1_offset + 8]
            .copy_from_slice(&(stack_base + 0x180).to_le_bytes());
        stack[frame1_offset + 8..frame1_offset + 16]
            .copy_from_slice(&0x4000_3000u64.to_le_bytes());

        // Frame 2 at fp=0x7fff0180, end of stack (fp = 0)
        let frame2_offset = 0x180usize;
        stack[frame2_offset..frame2_offset + 8].copy_from_slice(&0u64.to_le_bytes());
        stack[frame2_offset + 8..frame2_offset + 16]
            .copy_from_slice(&0x4000_4000u64.to_le_bytes());

        let regs = UnwindRegsAarch64 {
            pc: 0x4000_1000,
            sp: stack_base + 0x70,
            fp: stack_base + 0x80,
            lr: 0x4000_2000,
        };

        (stack, stack_base, regs)
    }

    #[test]
    fn test_simple_frame_chain() {
        let (stack, stack_base, regs) = build_test_stack();
        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = Aarch64StackWalker::new();
        let config = WalkConfig::no_validation();

        let trace = walker.walk(&regs, &mut memory, &config);

        assert_eq!(trace.len(), 4, "Expected 4 frames");
        assert!(!trace.truncated);

        assert_eq!(trace.frames[0].raw_address(), 0x4000_1000);
        assert_eq!(trace.frames[1].raw_address(), 0x4000_2000);
        assert_eq!(trace.frames[2].raw_address(), 0x4000_3000);
        assert_eq!(trace.frames[3].raw_address(), 0x4000_4000);
    }

    #[test]
    fn test_pac_stripping() {
        let stack_base = 0x7fff_0000u64;
        let mut stack = vec![0u8; 0x100];

        // Frame with PAC-signed return address
        let frame_offset = 0x80usize;
        stack[frame_offset..frame_offset + 8].copy_from_slice(&0u64.to_le_bytes()); // End of stack
        // Return address with PAC bits in upper 16 bits
        let signed_addr = 0xABCD_0000_4000_2000u64;
        stack[frame_offset + 8..frame_offset + 16].copy_from_slice(&signed_addr.to_le_bytes());

        let regs = UnwindRegsAarch64 {
            pc: 0x4000_1000,
            sp: stack_base + 0x70,
            fp: stack_base + 0x80,
            lr: 0,
        };

        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = Aarch64StackWalker::apple_silicon();
        let config = WalkConfig::no_validation();

        let trace = walker.walk(&regs, &mut memory, &config);

        assert_eq!(trace.len(), 2);
        // Second frame should have stripped address
        assert_eq!(trace.frames[1].raw_address(), 0x0000_4000_2000);
    }

    #[test]
    fn test_ptr_auth_mask() {
        let mask = PtrAuthMask::apple_silicon_user();

        // Address with PAC in upper bits
        let signed = 0xABCD_0000_1234_5678u64;
        assert_eq!(mask.strip(signed), 0x0000_0000_1234_5678);

        // Address without PAC
        let unsigned = 0x0000_0000_1234_5678u64;
        assert_eq!(mask.strip(unsigned), unsigned);
    }

    #[test]
    fn test_walk_with_callback() {
        let (stack, stack_base, regs) = build_test_stack();
        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = Aarch64StackWalker::new();
        let config = WalkConfig::no_validation();

        let mut addresses = Vec::new();
        walker.walk_with(&regs, &mut memory, &config, |frame| {
            addresses.push(frame.raw_address());
            true
        });

        assert_eq!(addresses, vec![0x4000_1000, 0x4000_2000, 0x4000_3000, 0x4000_4000]);
    }
}
