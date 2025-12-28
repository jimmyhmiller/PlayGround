// Stack Walker - Stack root enumeration for GC
//
// Provides utilities to walk the stack and find heap pointers
// using stack map information.
// Note: Some functions are only used by feature-gated GC implementations.
#![allow(dead_code)]

use super::StackMap;
use super::types::BuiltInTypes;

/// A simple abstraction for walking the stack and finding heap pointers
pub struct StackWalker;

impl StackWalker {
    /// Get the live portion of the stack as a slice
    pub fn get_live_stack(stack_base: usize, stack_pointer: usize) -> &'static [usize] {
        let distance_till_end = stack_base - stack_pointer;
        let num_words = (distance_till_end / 8) + 1;

        // Start one word before the stack pointer to match original behavior
        let start_ptr = stack_pointer - 8;

        unsafe { std::slice::from_raw_parts(start_ptr as *const usize, num_words) }
    }

    /// Walk the stack and call a callback for each heap pointer found
    /// The callback receives (stack_offset, heap_pointer_value)
    ///
    /// This uses a conservative approach: we scan ALL values on the stack that look
    /// like heap pointers, not just those within recognized stack frames. This is
    /// necessary because:
    /// 1. Registers saved by ExternalCallWithSaves (via STP) are pushed below the
    ///    stack frame boundary and wouldn't be scanned otherwise
    /// 2. The trampoline's return address may not be in our stack map, so frame-based
    ///    scanning would miss values in the current call context
    pub fn walk_stack_roots<F>(
        stack_base: usize,
        stack_pointer: usize,
        _stack_map: &StackMap, // Kept for API compatibility but not used in conservative scan
        mut callback: F,
    ) where
        F: FnMut(usize, usize),
    {
        let stack = Self::get_live_stack(stack_base, stack_pointer);

        // Conservative scan: check EVERY word on the stack for heap pointers
        // This catches:
        // - Values in recognized stack frames
        // - Registers saved via STP before trampoline calls
        // - Any other live values that might be heap references
        for (i, &value) in stack.iter().enumerate() {
            if BuiltInTypes::is_heap_pointer(value) {
                let untagged = BuiltInTypes::untag(value);
                // Verify alignment (garbage values may have heap pointer tags but wrong alignment)
                if untagged % 8 == 0 {
                    callback(i, value);
                }
            }
        }
    }

    /// Collect all heap pointers from the stack into a vector
    /// Returns (stack_offset, heap_pointer_value) pairs
    pub fn collect_stack_roots(
        stack_base: usize,
        stack_pointer: usize,
        stack_map: &StackMap,
    ) -> Vec<(usize, usize)> {
        let mut roots = Vec::with_capacity(32);
        Self::walk_stack_roots(stack_base, stack_pointer, stack_map, |offset, pointer| {
            roots.push((offset, pointer));
        });
        roots
    }

    /// Get a mutable slice of the live stack for updating pointers after GC
    pub fn get_live_stack_mut(stack_base: usize, stack_pointer: usize) -> &'static mut [usize] {
        let distance_till_end = stack_base - stack_pointer;
        let num_words = (distance_till_end / 8) + 1;

        // Start one word before the stack pointer to match original behavior
        let start_ptr = stack_pointer - 8;

        unsafe { std::slice::from_raw_parts_mut(start_ptr as *mut usize, num_words) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_slice_calculation() {
        // Test the calculation logic with known values
        let stack_base = 0x1000;
        let stack_pointer = 0x0F00; // 256 bytes down from base

        let slice = StackWalker::get_live_stack(stack_base, stack_pointer);

        // Distance is 256 bytes = 32 words, +1 = 33 words
        assert_eq!(slice.len(), 33);
        // Should start 8 bytes before stack_pointer
        assert_eq!(slice.as_ptr() as usize, stack_pointer - 8);
    }
}
