//! x86_64 stack walking implementation
//!
//! Frame pointer-based unwinding for x86_64 architecture.
//!
//! ## Stack Frame Layout
//!
//! ```text
//! Higher addresses
//! ┌─────────────────┐
//! │   arguments     │
//! ├─────────────────┤
//! │ return address  │  [rbp + 8]
//! ├─────────────────┤
//! │  caller's rbp   │  [rbp] ← current rbp points here
//! ├─────────────────┤
//! │  saved regs     │
//! ├─────────────────┤
//! │  local vars     │
//! ├─────────────────┤
//! │                 │  ← current rsp
//! └─────────────────┘
//! Lower addresses
//! ```

use crate::arch::{StackWalker, UnwindRegisters, UnwindResult};
use crate::error::StackWalkError;
use crate::frame::{Frame, FrameAddress, UnwindMethod};
use crate::memory::MemoryReader;

/// x86_64 registers needed for unwinding
#[derive(Debug, Clone, Default)]
pub struct UnwindRegsX86_64 {
    /// Instruction pointer (RIP)
    pub rip: u64,
    /// Stack pointer (RSP)
    pub rsp: u64,
    /// Frame/base pointer (RBP)
    pub rbp: u64,
}

impl UnwindRegsX86_64 {
    /// Create a new register set
    pub fn new(rip: u64, rsp: u64, rbp: u64) -> Self {
        Self { rip, rsp, rbp }
    }
}

impl UnwindRegisters for UnwindRegsX86_64 {
    #[inline]
    fn instruction_pointer(&self) -> u64 {
        self.rip
    }

    #[inline]
    fn set_instruction_pointer(&mut self, value: u64) {
        self.rip = value;
    }

    #[inline]
    fn stack_pointer(&self) -> u64 {
        self.rsp
    }

    #[inline]
    fn set_stack_pointer(&mut self, value: u64) {
        self.rsp = value;
    }

    #[inline]
    fn frame_pointer(&self) -> Option<u64> {
        if self.rbp != 0 {
            Some(self.rbp)
        } else {
            None
        }
    }

    #[inline]
    fn set_frame_pointer(&mut self, value: Option<u64>) {
        self.rbp = value.unwrap_or(0);
    }

    fn arch_name() -> &'static str {
        "x86_64"
    }
}

/// x86_64 stack walker using frame pointer chain
#[derive(Debug, Clone, Default)]
pub struct X86_64StackWalker;

impl X86_64StackWalker {
    /// Create a new x86_64 stack walker
    pub fn new() -> Self {
        Self
    }
}

impl StackWalker for X86_64StackWalker {
    type Registers = UnwindRegsX86_64;

    fn unwind_one<M: MemoryReader>(
        &self,
        current_regs: &Self::Registers,
        memory: &mut M,
        frame_index: usize,
        is_first_frame: bool,
    ) -> UnwindResult<Self::Registers> {
        let address = if is_first_frame {
            FrameAddress::InstructionPointer(current_regs.rip)
        } else {
            FrameAddress::ReturnAddress(current_regs.rip)
        };

        let frame = Frame::new(
            frame_index,
            address,
            current_regs.rsp,
            Some(current_regs.rbp),
            if is_first_frame {
                UnwindMethod::InitialFrame
            } else {
                UnwindMethod::FramePointer
            },
        );

        let fp = current_regs.rbp;

        // End of stack: frame pointer is null
        if fp == 0 {
            return UnwindResult::EndOfStack { frame };
        }

        // Read caller's frame pointer from [rbp]
        let new_fp = match memory.read_u64(fp) {
            Ok(v) => v,
            Err(e) => return UnwindResult::Failed { frame, error: e },
        };

        // Read return address from [rbp + 8]
        let return_address = match memory.read_u64(fp.wrapping_add(8)) {
            Ok(v) => v,
            Err(e) => return UnwindResult::Failed { frame, error: e },
        };

        // New stack pointer = old frame pointer + 16 (after saved RBP and return address)
        let new_sp = fp.wrapping_add(16);

        // Validate monotonic progress (stack grows down, so addresses increase as we unwind)
        if new_fp != 0 && new_fp <= fp {
            return UnwindResult::Failed {
                frame,
                error: StackWalkError::InvalidFramePointer {
                    fp: new_fp,
                    previous_fp: fp,
                },
            };
        }

        if new_sp <= current_regs.rsp {
            return UnwindResult::Failed {
                frame,
                error: StackWalkError::InvalidStackPointer {
                    sp: new_sp,
                    previous_sp: current_regs.rsp,
                },
            };
        }

        // Check for end of stack (null frame pointer in caller)
        if new_fp == 0 {
            // We have one more frame with the return address, but no further frames after
            let next_regs = UnwindRegsX86_64::new(return_address, new_sp, 0);
            return UnwindResult::Unwound {
                frame,
                next_registers: next_regs,
            };
        }

        let next_regs = UnwindRegsX86_64::new(return_address, new_sp, new_fp);
        UnwindResult::Unwound {
            frame,
            next_registers: next_regs,
        }
    }
}

/// Capture current x86_64 registers
///
/// This function uses inline assembly to capture the current register state.
/// It's primarily useful for testing and debugging.
///
/// # Safety
///
/// This function is safe to call but returns the register state at the point
/// of the call, which includes this function's own frame.
#[cfg(target_arch = "x86_64")]
pub fn capture_current_registers() -> UnwindRegsX86_64 {
    let rip: u64;
    let rsp: u64;
    let rbp: u64;

    unsafe {
        std::arch::asm!(
            "lea {rip}, [rip]",
            "mov {rsp}, rsp",
            "mov {rbp}, rbp",
            rip = out(reg) rip,
            rsp = out(reg) rsp,
            rbp = out(reg) rbp,
            options(nostack, nomem),
        );
    }

    UnwindRegsX86_64 { rip, rsp, rbp }
}

/// Capture current registers (stub for non-x86_64)
#[cfg(not(target_arch = "x86_64"))]
pub fn capture_current_registers() -> UnwindRegsX86_64 {
    panic!("capture_current_registers() is only available on x86_64")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::WalkConfig;
    use crate::memory::SliceMemoryReader;

    /// Build a simulated stack for testing
    fn build_test_stack() -> (Vec<u8>, u64, UnwindRegsX86_64) {
        let stack_base = 0x7fff_0000u64;
        let mut stack = vec![0u8; 0x200];

        // Frame 0 (current) at rbp=0x7fff0080
        // Points to Frame 1 at 0x7fff0100
        let frame0_offset = 0x80usize;
        // [rbp] = caller's rbp
        stack[frame0_offset..frame0_offset + 8]
            .copy_from_slice(&(stack_base + 0x100).to_le_bytes());
        // [rbp + 8] = return address
        stack[frame0_offset + 8..frame0_offset + 16]
            .copy_from_slice(&0x4000_2000u64.to_le_bytes());

        // Frame 1 at rbp=0x7fff0100
        // Points to Frame 2 at 0x7fff0180
        let frame1_offset = 0x100usize;
        stack[frame1_offset..frame1_offset + 8]
            .copy_from_slice(&(stack_base + 0x180).to_le_bytes());
        stack[frame1_offset + 8..frame1_offset + 16]
            .copy_from_slice(&0x4000_3000u64.to_le_bytes());

        // Frame 2 at rbp=0x7fff0180
        // End of stack (rbp = 0)
        let frame2_offset = 0x180usize;
        stack[frame2_offset..frame2_offset + 8].copy_from_slice(&0u64.to_le_bytes());
        stack[frame2_offset + 8..frame2_offset + 16]
            .copy_from_slice(&0x4000_4000u64.to_le_bytes());

        let regs = UnwindRegsX86_64::new(
            0x4000_1000,            // RIP - current instruction
            stack_base + 0x70,      // RSP - below frame 0
            stack_base + 0x80,      // RBP - frame 0
        );

        (stack, stack_base, regs)
    }

    #[test]
    fn test_simple_frame_chain() {
        let (stack, stack_base, regs) = build_test_stack();
        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = X86_64StackWalker::new();
        let config = WalkConfig::no_validation();

        let trace = walker.walk(&regs, &mut memory, &config);

        assert_eq!(trace.len(), 4, "Expected 4 frames");
        assert!(!trace.truncated, "Trace should not be truncated");

        // Frame 0: current instruction
        assert_eq!(trace.frames[0].raw_address(), 0x4000_1000);
        assert!(matches!(trace.frames[0].address, FrameAddress::InstructionPointer(_)));

        // Frame 1: first return address
        assert_eq!(trace.frames[1].raw_address(), 0x4000_2000);
        assert!(matches!(trace.frames[1].address, FrameAddress::ReturnAddress(_)));

        // Frame 2: second return address
        assert_eq!(trace.frames[2].raw_address(), 0x4000_3000);

        // Frame 3: final frame (before the null rbp)
        assert_eq!(trace.frames[3].raw_address(), 0x4000_4000);
    }

    #[test]
    fn test_walk_with_callback() {
        let (stack, stack_base, regs) = build_test_stack();
        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = X86_64StackWalker::new();
        let config = WalkConfig::no_validation();

        let mut frame_count = 0;
        let trace = walker.walk_with(&regs, &mut memory, &config, |_frame| {
            frame_count += 1;
            true // continue
        });

        assert_eq!(frame_count, 4);
        assert_eq!(trace.len(), 4);
    }

    #[test]
    fn test_early_stop_callback() {
        let (stack, stack_base, regs) = build_test_stack();
        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = X86_64StackWalker::new();
        let config = WalkConfig::no_validation();

        let trace = walker.walk_with(&regs, &mut memory, &config, |frame| {
            frame.index < 2 // Stop after 2 frames
        });

        assert_eq!(trace.len(), 3); // Frames 0, 1, 2 (stopped after processing 2)
        assert!(trace.truncated);
    }

    #[test]
    fn test_invalid_frame_pointer() {
        let stack_base = 0x7fff_0000u64;
        let mut stack = vec![0u8; 0x100];

        // Frame 0 at rbp=0x7fff0080, points BACKWARDS to 0x7fff0040
        let frame0_offset = 0x80usize;
        stack[frame0_offset..frame0_offset + 8]
            .copy_from_slice(&(stack_base + 0x40).to_le_bytes()); // Invalid: going backwards!
        stack[frame0_offset + 8..frame0_offset + 16]
            .copy_from_slice(&0x4000_2000u64.to_le_bytes());

        let regs = UnwindRegsX86_64::new(0x4000_1000, stack_base + 0x70, stack_base + 0x80);

        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = X86_64StackWalker::new();
        let config = WalkConfig::no_validation();

        let trace = walker.walk(&regs, &mut memory, &config);

        assert!(trace.truncated);
        assert!(trace.truncation_reason.is_some());
        assert!(trace.truncation_reason.unwrap().contains("Invalid frame pointer"));
    }

    #[test]
    fn test_max_frames_limit() {
        // Create a circular frame chain (would loop forever)
        let stack_base = 0x7fff_0000u64;
        let mut stack = vec![0u8; 0x100];

        // Frame at rbp=0x7fff0080, points to itself
        let frame_offset = 0x80usize;
        stack[frame_offset..frame_offset + 8]
            .copy_from_slice(&(stack_base + 0x90).to_le_bytes()); // Points forward but will loop
        stack[frame_offset + 8..frame_offset + 16]
            .copy_from_slice(&0x4000_2000u64.to_le_bytes());

        // Frame at 0x90 points back to 0x80 (circular)
        let frame2_offset = 0x90usize;
        stack[frame2_offset..frame2_offset + 8]
            .copy_from_slice(&(stack_base + 0x80).to_le_bytes()); // Would go backwards
        stack[frame2_offset + 8..frame2_offset + 16]
            .copy_from_slice(&0x4000_3000u64.to_le_bytes());

        let regs = UnwindRegsX86_64::new(0x4000_1000, stack_base + 0x70, stack_base + 0x80);

        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = X86_64StackWalker::new();
        let mut config = WalkConfig::no_validation();
        config.max_frames = 5;

        let trace = walker.walk(&regs, &mut memory, &config);

        // Should stop due to invalid frame pointer, not max frames
        assert!(trace.truncated);
        assert!(trace.len() <= 5);
    }

    #[test]
    fn test_frame_iterator() {
        let (stack, stack_base, regs) = build_test_stack();
        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = X86_64StackWalker::new();
        let config = WalkConfig::no_validation();

        let frames: Vec<_> = walker
            .iter_frames(regs, &mut memory, &config)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(frames.len(), 4);
    }

    #[test]
    fn test_frame_processor_validates_return_addresses() {
        use crate::arch::FrameProcessor;

        let (stack, stack_base, regs) = build_test_stack();
        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = X86_64StackWalker::new();

        // Configure to only accept addresses in 0x4000_1000..0x4000_1FFF
        // This should reject 0x4000_2000 and beyond
        let config = WalkConfig {
            max_frames: 256,
            min_code_address: 0x4000_1000,
            max_code_address: 0x4000_1FFF,
            validate_return_addresses: true,
        };

        let mut processor = FrameProcessor::new(walker, |_frame, _regs| true);
        let trace = processor.walk(&regs, &mut memory, &config);

        // Should stop at frame 1 which has return address 0x4000_2000 (out of range)
        assert!(trace.truncated, "Trace should be truncated due to invalid return address");
        assert!(trace.truncation_reason.is_some());
        assert!(
            trace.truncation_reason.as_ref().unwrap().contains("Invalid return address"),
            "Truncation reason should mention invalid return address, got: {:?}",
            trace.truncation_reason
        );
    }

    #[test]
    fn test_frame_processor_marks_max_frames_truncation() {
        use crate::arch::FrameProcessor;

        let (stack, stack_base, regs) = build_test_stack();
        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let walker = X86_64StackWalker::new();

        // Set max_frames to 2, less than the 4 frames in our test stack
        let config = WalkConfig {
            max_frames: 2,
            validate_return_addresses: false,
            ..Default::default()
        };

        let mut processor = FrameProcessor::new(walker, |_frame, _regs| true);
        let trace = processor.walk(&regs, &mut memory, &config);

        assert_eq!(trace.len(), 2, "Should have exactly 2 frames");
        assert!(trace.truncated, "Trace should be marked as truncated");
        assert!(trace.truncation_reason.is_some());
        assert!(
            trace.truncation_reason.as_ref().unwrap().contains("Max frames"),
            "Truncation reason should mention max frames, got: {:?}",
            trace.truncation_reason
        );
    }
}
