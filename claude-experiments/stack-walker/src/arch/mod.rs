//! Architecture-specific stack walking implementations
//!
//! This module provides the core traits and implementations for different CPU architectures.

pub mod x86_64;
pub mod aarch64;

use crate::error::Result;
use crate::frame::{Frame, StackTrace};
use crate::memory::MemoryReader;

/// Configuration for stack walking
#[derive(Debug, Clone)]
pub struct WalkConfig {
    /// Maximum number of frames to walk (prevents infinite loops)
    pub max_frames: usize,

    /// Minimum valid code address (for validation)
    pub min_code_address: u64,

    /// Maximum valid code address (for validation)
    pub max_code_address: u64,

    /// Whether to validate return addresses are in code range
    pub validate_return_addresses: bool,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            max_frames: 256,
            min_code_address: 0x1000, // Skip null page
            max_code_address: u64::MAX,
            validate_return_addresses: true,
        }
    }
}

impl WalkConfig {
    /// Create a config with no address validation (for testing)
    pub fn no_validation() -> Self {
        Self {
            validate_return_addresses: false,
            ..Default::default()
        }
    }

    /// Check if an address is in the valid code range
    #[inline]
    pub fn is_valid_code_address(&self, address: u64) -> bool {
        !self.validate_return_addresses
            || (address >= self.min_code_address && address <= self.max_code_address)
    }
}

/// Trait for architecture-specific register sets
pub trait UnwindRegisters: Clone + Default + std::fmt::Debug {
    /// Get the instruction pointer / program counter
    fn instruction_pointer(&self) -> u64;

    /// Set the instruction pointer
    fn set_instruction_pointer(&mut self, value: u64);

    /// Get the stack pointer
    fn stack_pointer(&self) -> u64;

    /// Set the stack pointer
    fn set_stack_pointer(&mut self, value: u64);

    /// Get the frame pointer (if available)
    fn frame_pointer(&self) -> Option<u64>;

    /// Set the frame pointer
    fn set_frame_pointer(&mut self, value: Option<u64>);

    /// Check if a frame pointer is available and valid
    fn has_valid_frame_pointer(&self) -> bool {
        self.frame_pointer().map_or(false, |fp| fp != 0)
    }

    /// Architecture name (e.g., "x86_64", "aarch64")
    fn arch_name() -> &'static str;
}

/// Result of unwinding a single frame
#[derive(Debug)]
pub enum UnwindResult<R: UnwindRegisters> {
    /// Successfully unwound to the next frame
    Unwound {
        /// The current frame that was just processed
        frame: Frame,
        /// Registers for the next (caller) frame
        next_registers: R,
    },
    /// This was the last frame (end of stack)
    EndOfStack {
        /// The final frame
        frame: Frame,
    },
    /// Unwinding failed
    Failed {
        /// The frame where we failed (partial info)
        frame: Frame,
        /// The error
        error: crate::error::StackWalkError,
    },
}

/// Core trait for stack walking implementations
///
/// This trait is designed for composability - you can wrap one walker with another
/// to add custom behavior (like GC root scanning) at each frame.
pub trait StackWalker {
    /// Register type for this architecture
    type Registers: UnwindRegisters;

    /// Unwind a single frame, returning the result
    ///
    /// This is the core method that processes one frame at a time.
    /// Use this for fine-grained control over the unwinding process.
    fn unwind_one<M: MemoryReader>(
        &self,
        current_regs: &Self::Registers,
        memory: &mut M,
        frame_index: usize,
        is_first_frame: bool,
    ) -> UnwindResult<Self::Registers>;

    /// Walk the entire stack, calling a closure for each frame
    ///
    /// The closure receives each frame and returns `true` to continue
    /// walking or `false` to stop early. This is the recommended API
    /// for GC integration where you need to process each frame.
    ///
    /// # Example
    ///
    /// ```ignore
    /// walker.walk_with(regs, memory, config, |frame| {
    ///     // Process frame for GC roots
    ///     scan_frame_for_roots(frame);
    ///     true // continue walking
    /// });
    /// ```
    fn walk_with<M, F>(
        &self,
        initial_regs: &Self::Registers,
        memory: &mut M,
        config: &WalkConfig,
        mut on_frame: F,
    ) -> StackTrace
    where
        M: MemoryReader,
        F: FnMut(&Frame) -> bool,
    {
        let mut trace = StackTrace::with_capacity(32);
        let mut current_regs = initial_regs.clone();
        let mut is_first = true;

        for frame_index in 0..config.max_frames {
            match self.unwind_one(&current_regs, memory, frame_index, is_first) {
                UnwindResult::Unwound { frame, next_registers } => {
                    // Validate return address if configured
                    if !is_first && !config.is_valid_code_address(frame.raw_address()) {
                        trace.truncate_with_reason(format!(
                            "Invalid return address: 0x{:016x}",
                            frame.raw_address()
                        ));
                        trace.push(frame);
                        break;
                    }

                    let should_continue = on_frame(&frame);
                    trace.push(frame);

                    if !should_continue {
                        trace.truncate_with_reason("Stopped by callback");
                        break;
                    }

                    current_regs = next_registers;
                    is_first = false;
                }
                UnwindResult::EndOfStack { frame } => {
                    on_frame(&frame);
                    trace.push(frame);
                    break;
                }
                UnwindResult::Failed { frame, error } => {
                    trace.truncate_with_reason(format!("{}", error));
                    trace.push(frame);
                    break;
                }
            }
        }

        if trace.len() >= config.max_frames {
            trace.truncate_with_reason(format!("Max frames ({}) reached", config.max_frames));
        }

        trace
    }

    /// Walk the entire stack, collecting all frames
    ///
    /// This is a convenience method that collects all frames into a StackTrace.
    fn walk<M: MemoryReader>(
        &self,
        initial_regs: &Self::Registers,
        memory: &mut M,
        config: &WalkConfig,
    ) -> StackTrace {
        self.walk_with(initial_regs, memory, config, |_| true)
    }

    /// Create an iterator over frames
    ///
    /// This allows using standard iterator methods for frame processing.
    fn iter_frames<'a, M: MemoryReader + 'a>(
        &'a self,
        initial_regs: Self::Registers,
        memory: &'a mut M,
        config: &'a WalkConfig,
    ) -> FrameIterator<'a, Self, M>
    where
        Self: Sized,
    {
        FrameIterator::new(self, initial_regs, memory, config)
    }

    /// Architecture name
    fn arch_name(&self) -> &'static str {
        Self::Registers::arch_name()
    }
}

/// Iterator over stack frames
///
/// This provides an iterator interface for frame-by-frame processing.
pub struct FrameIterator<'a, W, M>
where
    W: StackWalker,
    M: MemoryReader,
{
    walker: &'a W,
    memory: &'a mut M,
    config: &'a WalkConfig,
    current_regs: Option<W::Registers>,
    frame_index: usize,
    is_first: bool,
    done: bool,
}

impl<'a, W, M> FrameIterator<'a, W, M>
where
    W: StackWalker,
    M: MemoryReader,
{
    fn new(
        walker: &'a W,
        initial_regs: W::Registers,
        memory: &'a mut M,
        config: &'a WalkConfig,
    ) -> Self {
        Self {
            walker,
            memory,
            config,
            current_regs: Some(initial_regs),
            frame_index: 0,
            is_first: true,
            done: false,
        }
    }
}

impl<'a, W, M> Iterator for FrameIterator<'a, W, M>
where
    W: StackWalker,
    M: MemoryReader,
{
    type Item = Result<Frame>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let regs = self.current_regs.take()?;

        if self.frame_index >= self.config.max_frames {
            self.done = true;
            return Some(Err(crate::error::StackWalkError::MaxFramesExceeded {
                max: self.config.max_frames,
            }));
        }

        match self.walker.unwind_one(&regs, self.memory, self.frame_index, self.is_first) {
            UnwindResult::Unwound { frame, next_registers } => {
                self.current_regs = Some(next_registers);
                self.frame_index += 1;
                self.is_first = false;
                Some(Ok(frame))
            }
            UnwindResult::EndOfStack { frame } => {
                self.done = true;
                Some(Ok(frame))
            }
            UnwindResult::Failed { frame: _, error } => {
                self.done = true;
                Some(Err(error))
            }
        }
    }
}

/// A wrapper that allows custom processing at each frame
///
/// This is useful for GC integration where you need to scan each frame
/// for roots while still leveraging the underlying walker's logic.
///
/// # Example
///
/// ```ignore
/// let gc_walker = FrameProcessor::new(base_walker, |frame, regs| {
///     // Scan this frame for GC roots
///     for slot in frame.stack_pointer..frame.frame_pointer.unwrap_or(frame.stack_pointer) {
///         if let Some(ptr) = maybe_gc_pointer(slot) {
///             mark_root(ptr);
///         }
///     }
///     // Return true to continue, false to stop
///     true
/// });
/// ```
pub struct FrameProcessor<W, F>
where
    W: StackWalker,
    F: FnMut(&Frame, &W::Registers) -> bool,
{
    inner: W,
    processor: F,
}

impl<W, F> FrameProcessor<W, F>
where
    W: StackWalker,
    F: FnMut(&Frame, &W::Registers) -> bool,
{
    /// Create a new frame processor wrapping an existing walker
    pub fn new(inner: W, processor: F) -> Self {
        Self { inner, processor }
    }

    /// Get a reference to the inner walker
    pub fn inner(&self) -> &W {
        &self.inner
    }

    /// Walk the stack, calling the processor for each frame
    pub fn walk<M: MemoryReader>(
        &mut self,
        initial_regs: &W::Registers,
        memory: &mut M,
        config: &WalkConfig,
    ) -> StackTrace {
        let mut trace = StackTrace::with_capacity(32);
        let mut current_regs = initial_regs.clone();
        let mut is_first = true;

        for frame_index in 0..config.max_frames {
            match self.inner.unwind_one(&current_regs, memory, frame_index, is_first) {
                UnwindResult::Unwound { frame, next_registers } => {
                    // Validate return address if configured
                    if !is_first && !config.is_valid_code_address(frame.raw_address()) {
                        trace.truncate_with_reason(format!(
                            "Invalid return address: 0x{:016x}",
                            frame.raw_address()
                        ));
                        trace.push(frame);
                        break;
                    }

                    let should_continue = (self.processor)(&frame, &current_regs);
                    trace.push(frame);

                    if !should_continue {
                        trace.truncate_with_reason("Stopped by processor");
                        break;
                    }

                    current_regs = next_registers;
                    is_first = false;
                }
                UnwindResult::EndOfStack { frame } => {
                    (self.processor)(&frame, &current_regs);
                    trace.push(frame);
                    break;
                }
                UnwindResult::Failed { frame, error } => {
                    trace.truncate_with_reason(format!("{}", error));
                    trace.push(frame);
                    break;
                }
            }
        }

        if trace.len() >= config.max_frames {
            trace.truncate_with_reason(format!("Max frames ({}) reached", config.max_frames));
        }

        trace
    }
}
