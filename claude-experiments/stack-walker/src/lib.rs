//! Stack Walker Library
//!
//! A modular library for walking stack frames, designed for GC root scanning
//! in JIT-compiled code. Supports x86_64 and aarch64 architectures with
//! pluggable abstractions for symbol resolution and slot parsing.
//!
//! # Quick Start
//!
//! ```ignore
//! use stack_walker::{
//!     Aarch64StackWalker, StackWalker, WalkConfig,
//!     UnsafeDirectReader, capture_current_aarch64,
//! };
//!
//! // Create a walker for the current architecture
//! let walker = Aarch64StackWalker::apple_silicon();
//!
//! // Capture current registers
//! let regs = capture_current_aarch64();
//!
//! // Create a memory reader for the current process
//! let mut reader = UnsafeDirectReader::new();
//!
//! // Walk the stack
//! let trace = walker.walk(&regs, &mut reader, &WalkConfig::default());
//!
//! for frame in &trace {
//!     println!("Frame {}: 0x{:016x}", frame.index, frame.raw_address());
//! }
//! ```
//!
//! # GC Integration
//!
//! For GC root scanning, use `walk_with` to process each frame as it's unwound:
//!
//! ```ignore
//! use stack_walker::{StackWalker, SlotProvider, ConservativeSlotProvider};
//!
//! walker.walk_with(&regs, &mut memory, &config, |frame| {
//!     // Scan this frame for GC roots
//!     let slot_provider = ConservativeSlotProvider;
//!     for slot in frame.scan_slots(&mut memory, &slot_provider) {
//!         if let Ok(slot) = slot {
//!             if is_gc_pointer(slot.value) {
//!                 mark_root(slot.value);
//!             }
//!         }
//!     }
//!     true // continue walking
//! });
//! ```
//!
//! # Architecture Support
//!
//! The library provides separate implementations for each architecture:
//!
//! - [`X86_64StackWalker`] - x86_64 frame pointer walking
//! - [`Aarch64StackWalker`] - ARM64 with PAC (Pointer Authentication) support
//!
//! Use the `NativeStackWalker` type alias for the current platform.

pub mod arch;
pub mod display;
pub mod error;
pub mod frame;
pub mod memory;
pub mod slot;
pub mod symbol;

// Re-export core types
pub use error::{Result, StackWalkError};
pub use frame::{Frame, FrameAddress, StackTrace, UnwindMethod};
pub use memory::{ClosureMemoryReader, MemoryReader, SliceMemoryReader, UnsafeDirectReader};

// Re-export architecture types
pub use arch::{FrameProcessor, StackWalker, UnwindRegisters, UnwindResult, WalkConfig};
pub use arch::x86_64::{UnwindRegsX86_64, X86_64StackWalker};
pub use arch::aarch64::{Aarch64StackWalker, PtrAuthMask, UnwindRegsAarch64};

// Re-export slot types
pub use slot::{ConservativeSlotProvider, FrameSlotScanner, NoSlotProvider, SlotInfo, SlotProvider, SlotRange};

// Re-export symbol types
pub use symbol::{
    ChainedResolver, MapSymbolResolver, NullSymbolResolver, RangeSymbolResolver,
    SymbolError, SymbolInfo, SymbolResolver,
};

// Re-export display types
pub use display::{AddressFormatter, BacktraceFormatter, CompactFormatter, FrameFormatter, JsonFormatter};

// Platform-specific type aliases
#[cfg(target_arch = "x86_64")]
pub type NativeStackWalker = X86_64StackWalker;
#[cfg(target_arch = "x86_64")]
pub type NativeUnwindRegs = UnwindRegsX86_64;

#[cfg(target_arch = "aarch64")]
pub type NativeStackWalker = Aarch64StackWalker;
#[cfg(target_arch = "aarch64")]
pub type NativeUnwindRegs = UnwindRegsAarch64;

/// Capture current x86_64 registers
#[cfg(target_arch = "x86_64")]
pub use arch::x86_64::capture_current_registers as capture_current_x86_64;

/// Capture current aarch64 registers
#[cfg(target_arch = "aarch64")]
pub use arch::aarch64::capture_current_registers as capture_current_aarch64;

/// Capture current registers for the native platform
#[cfg(target_arch = "x86_64")]
pub fn capture_current_native() -> NativeUnwindRegs {
    arch::x86_64::capture_current_registers()
}

/// Capture current registers for the native platform
#[cfg(target_arch = "aarch64")]
pub fn capture_current_native() -> NativeUnwindRegs {
    arch::aarch64::capture_current_registers()
}

/// Create a native stack walker for the current platform
#[cfg(target_arch = "x86_64")]
pub fn native_walker() -> NativeStackWalker {
    X86_64StackWalker::new()
}

/// Create a native stack walker for the current platform
#[cfg(target_arch = "aarch64")]
pub fn native_walker() -> NativeStackWalker {
    Aarch64StackWalker::apple_silicon()
}

/// Walk the current stack and return a trace
///
/// This is a convenience function that captures the current registers
/// and walks the stack using the native walker.
///
/// # Safety
///
/// This function reads memory from the current process using raw pointers.
/// It should only be called when the stack is in a consistent state.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn walk_current_stack() -> StackTrace {
    walk_current_stack_with_config(&WalkConfig::default())
}

/// Walk the current stack with custom configuration
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn walk_current_stack_with_config(config: &WalkConfig) -> StackTrace {
    let walker = native_walker();
    let regs = capture_current_native();
    let mut memory = UnsafeDirectReader::new();
    walker.walk(&regs, &mut memory, config)
}

/// Walk the current stack, calling a callback for each frame
///
/// Returns early if the callback returns false.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn walk_current_stack_with<F>(config: &WalkConfig, on_frame: F) -> StackTrace
where
    F: FnMut(&Frame) -> bool,
{
    let walker = native_walker();
    let regs = capture_current_native();
    let mut memory = UnsafeDirectReader::new();
    walker.walk_with(&regs, &mut memory, config, on_frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_memory_reader() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        let mut reader = SliceMemoryReader::new(0x1000, &data);

        let value = reader.read_u64(0x1000).unwrap();
        assert_eq!(value, 0xf0debc9a78563412);
    }

    #[test]
    fn test_x86_64_walker_creation() {
        let walker = X86_64StackWalker::new();
        assert_eq!(walker.arch_name(), "x86_64");
    }

    #[test]
    fn test_aarch64_walker_creation() {
        let walker = Aarch64StackWalker::new();
        assert_eq!(walker.arch_name(), "aarch64");

        let apple_walker = Aarch64StackWalker::apple_silicon();
        assert_eq!(apple_walker.ptr_auth_mask().mask(), 0x0000_FFFF_FFFF_FFFF);
    }

    #[test]
    fn test_config_defaults() {
        let config = WalkConfig::default();
        assert_eq!(config.max_frames, 256);
        assert!(config.validate_return_addresses);
    }

    #[test]
    fn test_stack_trace_iteration() {
        let mut trace = StackTrace::new();
        trace.push(Frame::new(
            0,
            FrameAddress::InstructionPointer(0x1000),
            0x7fff0000,
            Some(0x7fff0010),
            UnwindMethod::InitialFrame,
        ));

        assert_eq!(trace.len(), 1);
        assert!(!trace.is_empty());

        for frame in &trace {
            assert_eq!(frame.raw_address(), 0x1000);
        }
    }

    // Integration test that actually walks the current stack
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn test_walk_current_stack() {
        let trace = walk_current_stack();

        // Should have at least a few frames (test function + test harness)
        assert!(trace.len() >= 2, "Expected at least 2 frames, got {}", trace.len());

        // First frame should be the current instruction
        assert!(matches!(trace.frames[0].address, FrameAddress::InstructionPointer(_)));

        // All frames should have valid addresses
        for frame in &trace {
            assert!(frame.raw_address() > 0x1000, "Invalid address: 0x{:x}", frame.raw_address());
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn test_walk_with_callback() {
        let config = WalkConfig::default();
        let mut count = 0;

        let trace = walk_current_stack_with(&config, |_frame| {
            count += 1;
            count < 5 // Stop after 5 frames
        });

        assert!(count <= 5);
        assert!(trace.len() <= 5);
    }
}
