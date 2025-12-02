use std::error::Error;

#[derive(Debug, Clone)]
pub struct StackSegment {
    pub id: usize,
    pub data: Vec<u8>,
    pub base: usize,
    pub size: usize,
}

impl StackSegment {
    pub fn new(id: usize, data: Vec<u8>) -> Self {
        let base = data.as_ptr() as usize;
        let size = data.len();
        Self {
            id,
            data,
            base,
            size,
        }
    }

    pub fn as_stack_pointer_pair(&self) -> (usize, usize) {
        (self.base + self.size, self.base)
    }
}

pub struct StackSegmentAllocator {
    segments: Vec<StackSegment>,
    next_id: usize,
}

impl StackSegmentAllocator {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            next_id: 0,
        }
    }

    pub fn add_segment(&mut self, stack_data: &[u8]) -> Result<usize, Box<dyn Error>> {
        let id = self.next_id;
        self.next_id += 1;

        let data = stack_data.to_vec();
        let segment = StackSegment::new(id, data);

        self.segments.push(segment);
        Ok(id)
    }

    pub fn remove_segment(&mut self, id: usize) -> Result<(), Box<dyn Error>> {
        let index = self
            .segments
            .iter()
            .position(|s| s.id == id)
            .ok_or_else(|| format!("Stack segment with id {} not found", id))?;

        self.segments.remove(index);
        Ok(())
    }

    pub fn get_segment(&self, id: usize) -> Option<&StackSegment> {
        self.segments.iter().find(|s| s.id == id)
    }

    pub fn get_all_stack_pointers(&self) -> Vec<(usize, usize)> {
        self.segments
            .iter()
            .map(|segment| segment.as_stack_pointer_pair())
            .collect()
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn clear_all_segments(&mut self) {
        self.segments.clear();
    }

    pub fn restore_segment(&self, id: usize, target_ptr: *mut u8) -> Result<usize, Box<dyn Error>> {
        let segment = self
            .get_segment(id)
            .ok_or_else(|| format!("Stack segment with id {} not found", id))?;

        unsafe {
            std::ptr::copy_nonoverlapping(segment.data.as_ptr(), target_ptr, segment.size);
        }

        Ok(segment.size)
    }
}

impl Default for StackSegmentAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_allocator() {
        let allocator = StackSegmentAllocator::new();
        assert_eq!(allocator.segment_count(), 0);
        assert_eq!(allocator.next_id, 0);
    }

    #[test]
    fn test_add_segment() {
        let mut allocator = StackSegmentAllocator::new();
        let stack_data = vec![1, 2, 3, 4, 5];

        let id = allocator.add_segment(&stack_data).unwrap();
        assert_eq!(id, 0);
        assert_eq!(allocator.segment_count(), 1);
        assert_eq!(allocator.next_id, 1);

        let segment = allocator.get_segment(id).unwrap();
        assert_eq!(segment.id, id);
        assert_eq!(segment.data, stack_data);
        assert_eq!(segment.size, 5);
    }

    #[test]
    fn test_add_multiple_segments() {
        let mut allocator = StackSegmentAllocator::new();

        let id1 = allocator.add_segment(&[1, 2, 3]).unwrap();
        let id2 = allocator.add_segment(&[4, 5, 6, 7]).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(allocator.segment_count(), 2);

        let segment1 = allocator.get_segment(id1).unwrap();
        let segment2 = allocator.get_segment(id2).unwrap();

        assert_eq!(segment1.data, vec![1, 2, 3]);
        assert_eq!(segment2.data, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_remove_segment() {
        let mut allocator = StackSegmentAllocator::new();

        let id1 = allocator.add_segment(&[1, 2, 3]).unwrap();
        let id2 = allocator.add_segment(&[4, 5, 6]).unwrap();

        assert_eq!(allocator.segment_count(), 2);

        allocator.remove_segment(id1).unwrap();
        assert_eq!(allocator.segment_count(), 1);
        assert!(allocator.get_segment(id1).is_none());
        assert!(allocator.get_segment(id2).is_some());
    }

    #[test]
    fn test_remove_nonexistent_segment() {
        let mut allocator = StackSegmentAllocator::new();

        let result = allocator.remove_segment(999);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_get_all_stack_pointers() {
        let mut allocator = StackSegmentAllocator::new();

        allocator.add_segment(&[1, 2, 3]).unwrap();
        allocator.add_segment(&[4, 5]).unwrap();

        let stack_pointers = allocator.get_all_stack_pointers();
        assert_eq!(stack_pointers.len(), 2);

        // Each tuple should be (base + size, base) representing (stack_top, stack_bottom)
        for (stack_top, stack_bottom) in &stack_pointers {
            assert!(stack_top > stack_bottom);
        }
    }

    #[test]
    fn test_clear_all_segments() {
        let mut allocator = StackSegmentAllocator::new();

        allocator.add_segment(&[1, 2, 3]).unwrap();
        allocator.add_segment(&[4, 5, 6]).unwrap();
        assert_eq!(allocator.segment_count(), 2);

        allocator.clear_all_segments();
        assert_eq!(allocator.segment_count(), 0);
        assert!(allocator.get_all_stack_pointers().is_empty());
    }

    #[test]
    fn test_stack_pointer_pair_format() {
        let mut allocator = StackSegmentAllocator::new();
        let stack_data = vec![0; 100]; // 100 bytes

        let id = allocator.add_segment(&stack_data).unwrap();
        let segment = allocator.get_segment(id).unwrap();
        let (stack_top, stack_bottom) = segment.as_stack_pointer_pair();

        assert_eq!(stack_top - stack_bottom, 100);
        assert_eq!(stack_bottom, segment.base);
        assert_eq!(stack_top, segment.base + segment.size);
    }

    #[test]
    fn test_restore_segment() {
        let mut allocator = StackSegmentAllocator::new();
        let original_data = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let id = allocator.add_segment(&original_data).unwrap();

        // Create a target buffer to restore into
        let mut target_buffer = vec![0u8; 10]; // Larger buffer
        let target_ptr = target_buffer.as_mut_ptr();

        // Restore the segment
        let bytes_copied = allocator.restore_segment(id, target_ptr).unwrap();

        assert_eq!(bytes_copied, 8);
        assert_eq!(&target_buffer[0..8], &original_data[..]);
        assert_eq!(&target_buffer[8..], &[0u8, 0u8]); // Remaining bytes should be untouched
    }

    #[test]
    fn test_restore_nonexistent_segment() {
        let allocator = StackSegmentAllocator::new();
        let mut target_buffer = vec![0u8; 10];
        let target_ptr = target_buffer.as_mut_ptr();

        let result = allocator.restore_segment(999, target_ptr);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_restore_multiple_segments() {
        let mut allocator = StackSegmentAllocator::new();

        let data1 = vec![10, 20, 30];
        let data2 = vec![40, 50, 60, 70];

        let id1 = allocator.add_segment(&data1).unwrap();
        let id2 = allocator.add_segment(&data2).unwrap();

        // Restore first segment
        let mut buffer1 = vec![0u8; 5];
        let bytes1 = allocator
            .restore_segment(id1, buffer1.as_mut_ptr())
            .unwrap();
        assert_eq!(bytes1, 3);
        assert_eq!(&buffer1[0..3], &[10, 20, 30]);

        // Restore second segment
        let mut buffer2 = vec![0u8; 6];
        let bytes2 = allocator
            .restore_segment(id2, buffer2.as_mut_ptr())
            .unwrap();
        assert_eq!(bytes2, 4);
        assert_eq!(&buffer2[0..4], &[40, 50, 60, 70]);
    }

    #[test]
    fn test_restore_segment_exact_size() {
        let mut allocator = StackSegmentAllocator::new();
        let data = vec![100, 101, 102, 103];

        let id = allocator.add_segment(&data).unwrap();

        // Target buffer is exactly the same size
        let mut target_buffer = vec![0u8; 4];
        let bytes_copied = allocator
            .restore_segment(id, target_buffer.as_mut_ptr())
            .unwrap();

        assert_eq!(bytes_copied, 4);
        assert_eq!(target_buffer, data);
    }

    #[test]
    fn test_restore_preserves_original_segment() {
        let mut allocator = StackSegmentAllocator::new();
        let original_data = vec![1, 2, 3, 4];

        let id = allocator.add_segment(&original_data).unwrap();

        // Restore multiple times to different buffers
        let mut buffer1 = vec![0u8; 4];
        let mut buffer2 = vec![0u8; 4];

        allocator.restore_segment(id, buffer1.as_mut_ptr()).unwrap();
        allocator.restore_segment(id, buffer2.as_mut_ptr()).unwrap();

        // Both should have the same data
        assert_eq!(buffer1, original_data);
        assert_eq!(buffer2, original_data);

        // Original segment should still exist and be unchanged
        let segment = allocator.get_segment(id).unwrap();
        assert_eq!(segment.data, original_data);
    }
}
