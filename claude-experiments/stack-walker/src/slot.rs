//! Stack slot parsing for GC root scanning
//!
//! This module provides traits for parsing and interpreting stack slots
//! within stack frames. This is primarily useful for garbage collection
//! where you need to identify potential GC roots on the stack.

use crate::frame::Frame;
use crate::memory::MemoryReader;

/// A range of stack slots to scan
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotRange {
    /// Start address (inclusive, lower address)
    pub start: u64,
    /// End address (exclusive, higher address)
    pub end: u64,
}

impl SlotRange {
    /// Create a new slot range
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Get the number of u64 slots in this range
    pub fn slot_count(&self) -> usize {
        ((self.end - self.start) / 8) as usize
    }

    /// Check if the range is empty
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Iterate over slot addresses
    pub fn iter_addresses(&self) -> impl Iterator<Item = u64> {
        let start = self.start;
        let end = self.end;
        (0..).map(move |i| start + i * 8).take_while(move |addr| *addr < end)
    }
}

/// Information about a stack slot
#[derive(Debug, Clone)]
pub struct SlotInfo {
    /// Address of this slot
    pub address: u64,
    /// Value stored in this slot
    pub value: u64,
    /// Offset from frame pointer (if known)
    pub fp_offset: Option<i64>,
}

impl SlotInfo {
    /// Create a new slot info
    pub fn new(address: u64, value: u64) -> Self {
        Self {
            address,
            value,
            fp_offset: None,
        }
    }

    /// Create with frame pointer offset
    pub fn with_fp_offset(address: u64, value: u64, fp_offset: i64) -> Self {
        Self {
            address,
            value,
            fp_offset: Some(fp_offset),
        }
    }
}

/// Trait for determining which slots to scan in a frame
///
/// Implement this trait to provide custom slot scanning logic based on
/// your JIT's calling convention and stack layout.
///
/// # Example
///
/// ```ignore
/// struct MyJitSlotProvider {
///     // Map from return address to frame layout info
///     frame_layouts: HashMap<u64, FrameLayout>,
/// }
///
/// impl SlotProvider for MyJitSlotProvider {
///     fn get_scannable_range(&self, frame: &Frame) -> Option<SlotRange> {
///         // Look up the frame layout for this return address
///         if let Some(layout) = self.frame_layouts.get(&frame.lookup_address()) {
///             let fp = frame.frame_pointer?;
///             Some(SlotRange::new(
///                 fp - layout.locals_size,
///                 fp,
///             ))
///         } else {
///             // Unknown frame, scan conservatively
///             let fp = frame.frame_pointer?;
///             let sp = frame.stack_pointer;
///             Some(SlotRange::new(sp, fp))
///         }
///     }
/// }
/// ```
pub trait SlotProvider {
    /// Get the range of slots to scan for this frame
    ///
    /// Returns None if this frame should be skipped (e.g., not a JIT frame).
    fn get_scannable_range(&self, frame: &Frame) -> Option<SlotRange>;

    /// Get specific slot offsets to scan (alternative to range)
    ///
    /// If implemented, this takes precedence over `get_scannable_range`.
    /// Returns a list of offsets from the frame pointer.
    fn get_slot_offsets(&self, _frame: &Frame) -> Option<Vec<i64>> {
        None
    }
}

/// A slot provider that scans all slots between SP and FP
#[derive(Debug, Clone, Copy, Default)]
pub struct ConservativeSlotProvider;

impl SlotProvider for ConservativeSlotProvider {
    fn get_scannable_range(&self, frame: &Frame) -> Option<SlotRange> {
        let fp = frame.frame_pointer?;
        let sp = frame.stack_pointer;

        if sp >= fp {
            return None; // Invalid frame
        }

        Some(SlotRange::new(sp, fp))
    }
}

/// A slot provider that returns no slots (for skipping frames)
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSlotProvider;

impl SlotProvider for NoSlotProvider {
    fn get_scannable_range(&self, _frame: &Frame) -> Option<SlotRange> {
        None
    }
}

/// Iterator over slots in a frame
pub struct SlotIterator<'a, M: MemoryReader, P: SlotProvider> {
    memory: &'a mut M,
    provider: &'a P,
    frame: &'a Frame,
    current_address: u64,
    end_address: u64,
    fp: Option<u64>,
    initialized: bool,
}

impl<'a, M: MemoryReader, P: SlotProvider> SlotIterator<'a, M, P> {
    /// Create a new slot iterator for a frame
    pub fn new(memory: &'a mut M, provider: &'a P, frame: &'a Frame) -> Self {
        Self {
            memory,
            provider,
            frame,
            current_address: 0,
            end_address: 0,
            fp: frame.frame_pointer,
            initialized: false,
        }
    }

    fn initialize(&mut self) {
        if let Some(range) = self.provider.get_scannable_range(self.frame) {
            self.current_address = range.start;
            self.end_address = range.end;
        }
        self.initialized = true;
    }
}

impl<'a, M: MemoryReader, P: SlotProvider> Iterator for SlotIterator<'a, M, P> {
    type Item = crate::error::Result<SlotInfo>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.initialized {
            self.initialize();
        }

        if self.current_address >= self.end_address {
            return None;
        }

        let address = self.current_address;
        self.current_address += 8;

        match self.memory.read_u64(address) {
            Ok(value) => {
                let fp_offset = self.fp.map(|fp| address as i64 - fp as i64);
                Some(Ok(SlotInfo {
                    address,
                    value,
                    fp_offset,
                }))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Extension trait for scanning frame slots
pub trait FrameSlotScanner {
    /// Scan slots in this frame using the given provider
    fn scan_slots<'a, M: MemoryReader, P: SlotProvider>(
        &'a self,
        memory: &'a mut M,
        provider: &'a P,
    ) -> SlotIterator<'a, M, P>;
}

impl FrameSlotScanner for Frame {
    fn scan_slots<'a, M: MemoryReader, P: SlotProvider>(
        &'a self,
        memory: &'a mut M,
        provider: &'a P,
    ) -> SlotIterator<'a, M, P> {
        SlotIterator::new(memory, provider, self)
    }
}

/// Convenience function to scan all slots in a frame conservatively
pub fn scan_frame_slots<M: MemoryReader>(
    frame: &Frame,
    memory: &mut M,
) -> Vec<crate::error::Result<SlotInfo>> {
    let provider = ConservativeSlotProvider;
    let mut iter = SlotIterator::new(memory, &provider, frame);
    iter.by_ref().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameAddress, UnwindMethod};
    use crate::memory::SliceMemoryReader;

    #[test]
    fn test_slot_range() {
        let range = SlotRange::new(0x1000, 0x1020);
        assert_eq!(range.slot_count(), 4);
        assert!(!range.is_empty());

        let addresses: Vec<_> = range.iter_addresses().collect();
        assert_eq!(addresses, vec![0x1000, 0x1008, 0x1010, 0x1018]);
    }

    #[test]
    fn test_conservative_scanning() {
        let stack_base = 0x7fff_0000u64;
        let mut stack = vec![0u8; 0x100];

        // Fill some slots with values
        stack[0x40..0x48].copy_from_slice(&0xAAAAAAAAu64.to_le_bytes());
        stack[0x48..0x50].copy_from_slice(&0xBBBBBBBBu64.to_le_bytes());
        stack[0x50..0x58].copy_from_slice(&0xCCCCCCCCu64.to_le_bytes());

        let frame = Frame::new(
            0,
            FrameAddress::InstructionPointer(0x1000),
            stack_base + 0x40, // SP
            Some(stack_base + 0x58), // FP
            UnwindMethod::InitialFrame,
        );

        let mut memory = SliceMemoryReader::new(stack_base, &stack);
        let provider = ConservativeSlotProvider;

        let slots: Vec<_> = SlotIterator::new(&mut memory, &provider, &frame)
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].value, 0xAAAAAAAA);
        assert_eq!(slots[1].value, 0xBBBBBBBB);
        assert_eq!(slots[2].value, 0xCCCCCCCC);
    }
}
