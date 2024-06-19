use std::error::Error;
use mmap_rs::{MmapMut, MmapOptions};
use crate::{compiler::{Allocator, StackMap, GC_ALWAYS, GC_NEVER}, debugger, ir::BuiltInTypes, Data, Message};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct FreeListEntry {
    segment: usize,
    offset: usize,
    size: usize,
}

impl FreeListEntry {
    fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.size
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SegmentAction {
    Increment,
    AllocateMore,
}

struct Segment {
    memory: MmapMut,
    offset: usize,
    size: usize,
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
        Self {
            memory,
            offset: 0,
            size,
        }
    }
    #[allow(unused)]
    fn copy_data_to_offset(&mut self, data: &[u8]) -> *const u8 {
        let buffer = &mut self.memory[self.offset..];
        buffer.copy_from_slice(data);
        self.offset += data.len();
        buffer.as_ptr()
    }
}


pub struct SimpleMarkSweepHeap {
    segments: Vec<Segment>,
    segment_offset: usize,
    segment_size: usize,
    free_list: Vec<FreeListEntry>,
    scale_factor: usize,
}

impl Allocator for SimpleMarkSweepHeap {
    fn allocate(&mut self, stack: &MmapMut, stack_map: &StackMap, stack_pointer: usize, bytes: usize, kind: BuiltInTypes) -> Result<usize, Box<dyn Error>> {
        self.allocate_inner(stack, stack_map, stack_pointer, bytes, kind, 0)
    }

    fn gc(&mut self, stack: &MmapMut, stack_map: &StackMap, stack_pointer: usize) {
        self.mark_and_sweep(stack, stack_map, stack_pointer);
    }
}

impl SimpleMarkSweepHeap {
    pub fn new() -> Self {
        let segment_size = MmapOptions::page_size() * 100;
        Self {
            segments: vec![Segment::new(segment_size), Segment::new(segment_size)],
            segment_offset: 0,
            segment_size,
            scale_factor: 2,
            free_list: vec![],
        }
    }

    fn segment_pointer(&self, arg: usize) -> usize {
        let segment = self.segments.get(arg).unwrap();
        segment.memory.as_ptr() as usize
    }

    fn switch_to_available_segment(&mut self, size: usize) -> bool {
        for (segment_index, segment) in self.segments.iter().enumerate() {
            if segment.size - segment.offset > size {
                self.segment_offset = segment_index;
                return true;
            }
        }
        false
    }

    fn create_more_segments(&mut self, size: usize) -> SegmentAction {
        if self.switch_to_available_segment(size) {
            return SegmentAction::Increment;
        }

        for (segment_index, segment) in self.segments.iter().enumerate() {
            if segment.offset + size < segment.size {
                self.segment_offset = segment_index;
                return SegmentAction::Increment;
            }
        }

        self.segment_offset = self.segments.len();

        for i in 0..self.scale_factor {
            self.segments.push(Segment::new(self.segment_size));
            let segment_pointer = self.segment_pointer(self.segment_offset + i);
            debugger(Message {
                kind: "HeapSegmentPointer".to_string(),
                data: Data::HeapSegmentPointer {
                    pointer: segment_pointer,
                },
            });
        }

        self.scale_factor *= 2;
        self.scale_factor = self.scale_factor.min(64);
        SegmentAction::AllocateMore
    }

    fn write_object(&mut self, segment_offset: usize, offset: usize, shifted_size: usize) -> usize {
        let memory = &mut self.segments[segment_offset].memory;

        let buffer = &mut memory[offset..offset + 8];

        // write the size of the object to the first 8 bytes
        buffer[..shifted_size.to_le_bytes().len()].copy_from_slice(&shifted_size.to_le_bytes());

        buffer.as_ptr() as usize
    }

    fn free_are_disjoint(entry1: &FreeListEntry, entry2: &FreeListEntry) -> bool {
        entry1.segment != entry2.segment
            || entry1.offset + entry1.size <= entry2.offset
            || entry2.offset + entry2.size <= entry1.offset
    }

    fn all_disjoint(&self) -> bool {
        for i in 0..self.free_list.len() {
            for j in 0..self.free_list.len() {
                if i == j {
                    continue;
                }
                if !Self::free_are_disjoint(&self.free_list[i], &self.free_list[j]) {
                    return false;
                }
            }
        }
        true
    }

    fn add_free(&mut self, entry: FreeListEntry) {
        // TODO: If a whole segment is free
        // I need a fast path where I don't have to update free list

        for current_entry in self.free_list.iter_mut() {
            if *current_entry == entry {
                println!("Double free!");
            }

            if current_entry.segment == entry.segment
                && current_entry.offset + current_entry.size == entry.offset
            {
                current_entry.size += entry.size;
                return;
            }
            if current_entry.segment == entry.segment
                && entry.offset + entry.size == current_entry.offset
            {
                current_entry.offset = entry.offset;
                current_entry.size += entry.size;
                return;
            }
        }
        if entry.offset == 0 && entry.size == self.segments[entry.segment].offset {
            self.segments[entry.segment].offset = 0;
        } else {
            self.free_list.push(entry);
        }

        debug_assert!(self.all_disjoint(), "Free list is not disjoint");
    }

    fn current_offset(&self) -> usize {
        self.segments[self.segment_offset].offset
    }

    fn current_segment_size(&self) -> usize {
        self.segments[self.segment_offset].size
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

    pub fn mark_and_sweep(&mut self, stack: &MmapMut, stack_map: &StackMap, stack_pointer: usize) {
        let start = std::time::Instant::now();
        self.mark(stack, stack_map, stack_pointer);
        self.sweep();
        println!("Mark and sweep took {:?}", start.elapsed());
    }

    pub fn mark(&mut self, stack: &MmapMut, stack_map: &StackMap, current_stack_pointer: usize) {
        // I'm adding to the end of the stack I've allocated so I only need to go from the end
        // til the current stack
        let stack_end = stack.as_ptr() as usize + stack.size();
        // let current_stack_pointer = current_stack_pointer & !0b111;
        let distance_till_end = stack_end - current_stack_pointer;
        let num_64_till_end = (distance_till_end / 8) + 1;
        let stack =
            unsafe { std::slice::from_raw_parts(stack.as_ptr() as *const usize, stack.size() / 8) };
        let stack = &stack[stack.len() - num_64_till_end..];

        let mut to_mark: Vec<usize> = Vec::with_capacity(128);

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

                for j in (bottom_of_frame - active_frame)..bottom_of_frame {
                    if BuiltInTypes::is_heap_pointer(stack[j]) {
                        let untagged = BuiltInTypes::untag(stack[j]);
                        if untagged as usize % 8 != 0 {
                            println!("Not aligned");
                        }
                        // println!("Pushing mark 0x{:?}", stack[j]);
                        to_mark.push(stack[j]);
                    }
                }
                continue;
            }
            i += 1;
        }

        while let Some(value) = to_mark.pop() {
            let _tagged = value;
            let untagged = BuiltInTypes::untag(value);
            let pointer = untagged as *mut u8;
            if pointer as usize % 8 != 0 {
                panic!("Not aligned {:x}", pointer as usize);
            }
            unsafe {
                let mut data: usize = *pointer.cast::<usize>();
                // check right most bit
                if (data & 1) == 1 {
                    continue;
                }
                data |= 1;
                *pointer.cast::<usize>() = data;

                // println!("Marking 0x{:x}", tagged);

                let size = *(pointer as *const usize) >> 1;
                let data = std::slice::from_raw_parts(pointer.add(8) as *const usize, size / 8);
                for datum in data.iter() {
                    if BuiltInTypes::is_heap_pointer(*datum) {
                        to_mark.push(*datum)
                    }
                }
            }
        }
    }

    fn sweep(&mut self) {
        // println!("Sweeping");
        let mut free_entries: Vec<FreeListEntry> = Vec::with_capacity(128);
        for (segment_index, segment) in self.segments.iter_mut().enumerate() {
            if segment.offset == 0 {
                continue;
            }
            let mut free_in_segment: Vec<&FreeListEntry> = self
                .free_list
                .iter()
                .filter(|x| x.segment == segment_index)
                .collect();

            free_in_segment.sort_by_key(|x| x.offset);
            let mut offset = 0;
            let segment_range = segment.offset;
            // TODO: I'm scanning whole segment even if unused
            let pointer = segment.memory.as_mut_ptr();
            while offset < segment_range {
                for free in free_in_segment.iter() {
                    if free.range().contains(&offset) {
                        offset = free.range().end;
                    }
                    if free.offset > offset {
                        break;
                    }
                }
                if offset >= segment_range {
                    break;
                }
                unsafe {
                    let pointer = pointer.add(offset);
                    let mut data: usize = *pointer.cast::<usize>();

                    // check right most bit
                    if (data & 1) == 1 {
                        // println!("marked!");
                        data &= !1;
                    } else {
                        let entry = FreeListEntry {
                            segment: segment_index,
                            offset,
                            size: (data >> 1) + 8,
                        };
                        let mut entered = false;
                        for current_entry in free_entries.iter_mut().rev() {
                            if current_entry.segment == entry.segment
                                && current_entry.offset + current_entry.size == entry.offset
                            {
                                current_entry.size += entry.size;
                                entered = true;
                                break;
                            }
                            if current_entry.segment == entry.segment
                                && entry.offset + entry.size == current_entry.offset
                            {
                                current_entry.offset = entry.offset;
                                current_entry.size += entry.size;
                                entered = true;
                                break;
                            }
                        }
                        if !entered {
                            free_entries.push(entry);
                        }
                        // println!("Found garbage!");
                    }

                    *pointer.cast::<usize>() = data;
                    let size = (data >> 1) + 8;
                    // println!("size: {}", size);
                    offset += size;
                    offset = (offset + 7) & !7;
                }
            }
        }

        for entry in free_entries {
            self.add_free(entry);
        }
    }
    fn allocate_inner(&mut self, stack: &MmapMut, stack_map: &StackMap, stack_pointer: usize, bytes: usize, kind: BuiltInTypes, depth: usize) ->  Result<usize, Box<dyn Error>>  {
        if GC_ALWAYS && !GC_NEVER {
            self.mark_and_sweep(stack, stack_map, stack_pointer);
        }

        if depth > 1 {
            // This might feel a bit dumb
            // But I do think it is reasonable to recurse
            // to get to the state I want
            // But I really should catch that depth.
            // never exceeds 1
            panic!("Recursed more than once in allocate")
        }

        let size = (bytes + 1) * 8;
        let shifted_size = (bytes * 8) << 1;

        if self.current_offset() + size < self.current_segment_size() {
            let pointer = self.write_object(
                self.segment_offset,
                self.current_offset(),
                shifted_size,
            );
            self.increment_current_offset(size);
            return Ok(pointer);
        }

        if self.switch_to_available_segment(size) {
            return self.allocate_inner(stack, stack_map, stack_pointer, bytes, kind, depth + 1);
        }

        debug_assert!(
            !self.segments.iter().any(|x| x.offset == 0),
            "Available segment not being used"
        );

        let mut spot = self
            .free_list
            .iter_mut()
            .enumerate()
            .find(|(_, x)| x.size >= size);

        if spot.is_none() {
            if !GC_NEVER {
                self.mark_and_sweep(stack, stack_map, stack_pointer);
            }
            if self.switch_to_available_segment(size) {
                return self.allocate_inner(stack, stack_map, stack_pointer, bytes, kind, depth + 1);
            }

            spot = self
                .free_list
                .iter_mut()
                .enumerate()
                .find(|(_, x)| x.size >= size);

            if spot.is_none() {
                self.create_more_segments(size);
                return self.allocate_inner(stack, stack_map, stack_pointer, bytes, kind, depth + 1);
            }
        }

        let (spot_index, spot) = spot.unwrap();

        let mut spot_clone = *spot;
        spot_clone.size = size;
        spot.size -= size;
        spot.offset += size;
        if spot.size == 0 {
            self.free_list.remove(spot_index);
        }

        let pointer = self
            .write_object(spot_clone.segment, spot_clone.offset, shifted_size);
        Ok(pointer)
    }



}