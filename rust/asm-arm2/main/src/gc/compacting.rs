use std::{error::Error, mem};

use mmap_rs::{MmapMut, MmapOptions};

use crate::{
    compiler::{AllocateAction, Allocator, AllocatorOptions, StackMap, STACK_SIZE},
    ir::BuiltInTypes,
};

struct Segment {
    memory: MmapMut,
    offset: usize,
    size: usize,
    memory_range: std::ops::Range<*const u8>,
}

impl Segment {
    fn new(size: usize) -> Self {
        let memory = MmapOptions::new(size)
            .unwrap()
            .map_mut()
            .unwrap()
            .make_mut()
            .unwrap_or_else(|(_map, e)| {
                panic!("Failed to make mmap executable: {}", e);
            });
        let memory_range = memory.as_ptr_range();
        Self {
            memory,
            offset: 0,
            size,
            memory_range,
        }
    }
}

struct Space {
    segments: Vec<Segment>,
    segment_offset: usize,
    segment_size: usize,
    scale_factor: usize,
}

struct ObjectIterator {
    space: *const Space,
    segment_index: usize,
    offset: usize,
}

impl Iterator for ObjectIterator {
    type Item = *const u8;

    fn next(&mut self) -> Option<Self::Item> {
        let space = unsafe { &*self.space };
        if self.offset >= space.segments[self.segment_index].offset {
            self.segment_index += 1;
            self.offset = 0;
        }
        if self.segment_index == space.segments.len() {
            return None;
        }
        let segment = &space.segments[self.segment_index];
        if segment.offset == 0 {
            return None;
        }
        let pointer = unsafe { segment.memory.as_ptr().add(self.offset) };
        let size = unsafe { *pointer.cast::<usize>() };

        self.offset += (size >> 1) + 8;
        Some(pointer)
    }
}

impl Space {
    fn new(segment_size: usize, scale_factor: usize) -> Self {
        let space = vec![Segment::new(segment_size)];
        Self {
            segments: space,
            segment_offset: 0,
            segment_size,
            scale_factor,
        }
    }

    fn object_iter_from_position(
        &self,
        segment_index: usize,
        offset: usize,
    ) -> impl Iterator<Item = *const u8> {
        ObjectIterator {
            space: self,
            segment_index,
            offset,
        }
    }

    fn current_position(&self) -> (usize, usize) {
        (
            self.segment_offset,
            self.segments[self.segment_offset].offset,
        )
    }

    fn contains(&self, pointer: *const u8) -> bool {
        for segment in self.segments.iter() {
            if segment.memory_range.contains(&pointer) {
                return true;
            }
        }
        false
    }

    fn copy_data_to_offset(&mut self, data: &[u8]) -> isize {
        if !self.can_allocate(data.len()) {
            self.resize();
        }
        let segment = self.segments.get_mut(self.segment_offset).unwrap();
        let buffer = &mut segment.memory[segment.offset..segment.offset + data.len()];
        buffer.copy_from_slice(data);
        let pointer = buffer.as_ptr() as isize;
        self.increment_current_offset(data.len());
        pointer
    }

    fn write_object(&mut self, segment_offset: usize, offset: usize, shifted_size: usize) -> usize {
        let memory = &mut self.segments[segment_offset].memory;

        let buffer = &mut memory[offset..offset + 8];

        // write the size of the object to the first 8 bytes
        buffer[..shifted_size.to_le_bytes().len()].copy_from_slice(&shifted_size.to_le_bytes());

        buffer.as_ptr() as usize
    }

    fn increment_current_offset(&mut self, size: usize) {
        self.segments[self.segment_offset].offset += size;
        // align to 8 bytes
        self.segments[self.segment_offset].offset =
            (self.segments[self.segment_offset].offset + 7) & !7;
        debug_assert!(
            self.segments[self.segment_offset].offset % 8 == 0,
            "Heap offset is not aligned"
        );
    }

    fn can_allocate(&mut self, size: usize) -> bool {
        let segment = self.segments.get(self.segment_offset).unwrap();
        let current_segment = segment.offset + size + 8 < segment.size;
        if current_segment {
            return true;
        }
        while self.segment_offset < self.segments.len() {
            let segment = self.segments.get(self.segment_offset).unwrap();
            if segment.offset + size + 8 < segment.size {
                return true;
            }
            self.segment_offset += 1;
        }
        if self.segment_offset == self.segments.len() {
            self.segment_offset = self.segments.len() - 1;
        }
        false
    }

    fn allocate(&mut self, bytes: usize) -> Result<usize, Box<dyn Error>> {
        let segment = self.segments.get_mut(self.segment_offset).unwrap();
        let mut offset = segment.offset;
        let size = (bytes + 1) * 8;
        if offset + size > segment.size {
            self.segment_offset += 1;
            if self.segment_offset == self.segments.len() {
                self.segments.push(Segment::new(self.segment_size));
            }
            offset = 0;
        }
        let shifted_size = (bytes * 8) << 1;
        let pointer = self.write_object(self.segment_offset, offset, shifted_size);
        self.increment_current_offset(size);
        assert!(pointer % 8 == 0, "Pointer is not aligned");
        Ok(pointer)
    }

    fn clear(&mut self) {
        for segment in self.segments.iter_mut() {
            segment.offset = 0;
        }
        self.segment_offset = 0;
    }

    fn resize(&mut self) {
        let offset = self.segment_offset;
        for _ in 0..self.scale_factor {
            self.segments.push(Segment::new(self.segment_size));
        }
        self.segment_offset = offset + 1;
        self.scale_factor *= 2;
        self.scale_factor = self.scale_factor.min(64);
    }
}

pub struct CompactingHeap {
    from_space: Space,
    to_space: Space,
}

impl Allocator for CompactingHeap {
    fn new() -> Self {
        let segment_size = MmapOptions::page_size() * 100;
        let from_space = Space::new(segment_size, 1);
        let to_space = Space::new(segment_size, 1);
        Self {
            from_space,
            to_space,
        }
    }

    fn allocate(
        &mut self,
        bytes: usize,
        kind: BuiltInTypes,
        options: AllocatorOptions,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        let pointer = self.allocate_inner(bytes, kind, options)?;

        Ok(pointer)
    }

    // TODO: Still got bugs here
    // Simple cases work, but not all cases
    fn gc(
        &mut self,
        stack_map: &StackMap,
        stack_pointers: &[(usize, usize)],
        options: AllocatorOptions,
    ) {
        if !options.gc {
            return;
        }
        let start = std::time::Instant::now();
        for (stack_base, stack_pointer) in stack_pointers.iter() {
            let roots = self.gather_roots(*stack_base, stack_map, *stack_pointer);
            let new_roots = unsafe { self.copy_all(roots.iter().map(|x| x.1).collect()) };

            let stack_buffer = get_live_stack(*stack_base, *stack_pointer);
            for (i, (stack_offset, _)) in roots.iter().enumerate() {
                debug_assert!(
                    BuiltInTypes::untag(new_roots[i]) % 8 == 0,
                    "Pointer is not aligned"
                );
                stack_buffer[*stack_offset] = new_roots[i];
            }
        }
        mem::swap(&mut self.from_space, &mut self.to_space);

        self.to_space.clear();
        if options.print_stats {
            println!("GC took: {:?}", start.elapsed());
        }
    }

    fn gc_add_root(&mut self, _old: usize, _young: usize) {
        // We don't need to do anything because all roots are gathered
        // from the stack.
        // Maybe we should do something though?
        // I guess this could be useful for c stuff,
        // but for right now I'm not going to do anything.
    }

    fn grow(&mut self, _options: AllocatorOptions) {
        self.from_space.resize();
    }
}

impl CompactingHeap {
    #[allow(clippy::too_many_arguments)]
    fn allocate_inner(
        &mut self,
        bytes: usize,
        _kind: BuiltInTypes,
        _options: AllocatorOptions,
    ) -> Result<AllocateAction, Box<dyn Error>> {
        if self.from_space.can_allocate(bytes) {
            Ok(AllocateAction::Allocated(self.from_space.allocate(bytes)?))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
        let (start_segment, start_offset) = self.to_space.current_position();
        // TODO: Is this vec the best way? Probably not
        // I could hand this the pointers to the stack location
        // then resolve what they point to and update them?
        // I should think about how to get rid of this allocation at the very least.
        let mut new_roots = vec![];
        for root in roots.iter() {
            new_roots.push(self.copy_using_cheneys_algorithm(*root));
        }

        for object in self
            .to_space
            .object_iter_from_position(start_segment, start_offset)
        {
            let size: usize = *(object as *const usize) >> 1;
            let marked = size & 1 == 1;
            if marked {
                panic!("We are copying to this space, nothing should be marked");
            }

            let data = std::slice::from_raw_parts_mut(object.add(8) as *mut usize, size / 8);

            for datum in data.iter_mut() {
                if BuiltInTypes::is_heap_pointer(*datum) {
                    *datum = self.copy_using_cheneys_algorithm(*datum);
                }
            }
        }

        new_roots
    }

    // TODO: Finish this
    unsafe fn copy_using_cheneys_algorithm(&mut self, root: usize) -> usize {
        // I could make this check the memory range.
        // In the original it does. But I don't think I have to?

        let untagged = BuiltInTypes::untag(root);
        let pointer = untagged as *mut u8;

        // If it is marked, we have copied it already
        // the first 8 bytes are a tagged forward pointer
        let first_field = *(pointer.add(8).cast::<usize>());
        if BuiltInTypes::is_heap_pointer(first_field) {
            let untagged_data = BuiltInTypes::untag(first_field);
            if self.to_space.contains(untagged_data as *const u8) {
                debug_assert!(untagged_data % 8 == 0, "Pointer is not aligned");
                return first_field;
            }
        }

        let size = *(pointer as *const usize) >> 1;
        let data = std::slice::from_raw_parts(pointer as *const u8, size + 8);
        let new_pointer = self.to_space.copy_data_to_offset(data);
        debug_assert!(new_pointer % 8 == 0, "Pointer is not aligned");
        // update header of original object to now be the forwarding pointer
        let tagged_new = BuiltInTypes::get_kind(root).tag(new_pointer as isize) as usize;
        let untagged = BuiltInTypes::untag(root);
        let pointer = untagged as *mut u8;
        let pointer = pointer.add(8);
        *pointer.cast::<usize>() = tagged_new;
        let size = *(untagged as *const usize) >> 1;
        assert!(size % 8 == 0 && size < 100);
        tagged_new
    }

    // Stolen from simple mark and sweep
    pub fn gather_roots(
        &mut self,
        stack_base: usize,
        stack_map: &StackMap,
        stack_pointer: usize,
    ) -> Vec<(usize, usize)> {
        // I'm adding to the end of the stack I've allocated so I only need to go from the end
        // til the current stack
        let stack = get_live_stack(stack_base, stack_pointer);

        let mut to_mark: Vec<usize> = Vec::with_capacity(128);
        let mut roots: Vec<(usize, usize)> = Vec::with_capacity(36);

        let mut i = 0;
        while i < stack.len() {
            let value = stack[i];

            if let Some(details) = stack_map.find_stack_data(value) {
                let mut frame_size = details.max_stack_size + details.number_of_locals;
                if frame_size % 2 != 0 {
                    frame_size += 1;
                }

                let bottom_of_frame = i + frame_size + 1;
                let _top_of_frame = i + 1;

                let active_frame = details.current_stack_size + details.number_of_locals;

                i = bottom_of_frame;

                for (j, slot) in stack
                    .iter()
                    .enumerate()
                    .take(bottom_of_frame)
                    .skip(bottom_of_frame - active_frame)
                {
                    if BuiltInTypes::is_heap_pointer(*slot) {
                        roots.push((j, *slot));
                        let untagged = BuiltInTypes::untag(*slot);
                        debug_assert!(untagged % 8 == 0, "Pointer is not aligned");
                        to_mark.push(*slot);
                    }
                }
                continue;
            }
            i += 1;
        }
        roots
    }
}

fn get_live_stack<'a>(stack_base: usize, stack_pointer: usize) -> &'a mut [usize] {
    let stack_end = stack_base;
    // let current_stack_pointer = current_stack_pointer & !0b111;
    let distance_till_end = stack_end - stack_pointer;
    let num_64_till_end = (distance_till_end / 8) + 1;
    let len = STACK_SIZE / 8;
    let stack_begin = stack_end - STACK_SIZE;
    let stack =
        unsafe { std::slice::from_raw_parts_mut(stack_begin as *mut usize, STACK_SIZE / 8) };

    (&mut stack[len - num_64_till_end..]) as _
}

// TODO: I can borrow the code here to get to a generational gc
// That should make a significant difference in performance
// I think to get there, I just need to mark things when I compact them
// Then those those that are marked get copied to the old generation
// I should probably read more about a proper setup for this
// to try and get the details right.
