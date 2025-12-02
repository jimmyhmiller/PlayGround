use crate::types::BuiltInTypes;

use super::StackMap;

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
    pub fn walk_stack_roots<F>(
        stack_base: usize,
        stack_pointer: usize,
        stack_map: &StackMap,
        mut callback: F,
    ) where
        F: FnMut(usize, usize),
    {
        let stack = Self::get_live_stack(stack_base, stack_pointer);

        let mut i = 0;
        while i < stack.len() {
            let value = stack[i];

            if let Some(details) = stack_map.find_stack_data(value) {
                let mut frame_size = details.max_stack_size + details.number_of_locals;
                if frame_size % 2 != 0 {
                    frame_size += 1;
                }

                let bottom_of_frame = i + frame_size + 1;
                let active_frame = details.current_stack_size + details.number_of_locals;

                i = bottom_of_frame;

                for (j, slot) in stack
                    .iter()
                    .enumerate()
                    .take(bottom_of_frame)
                    .skip(bottom_of_frame - active_frame)
                {
                    if BuiltInTypes::is_heap_pointer(*slot) {
                        let untagged = BuiltInTypes::untag(*slot);
                        debug_assert!(untagged % 8 == 0, "Pointer is not aligned");
                        callback(j, *slot);
                    }
                }
                continue;
            }
            i += 1;
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
