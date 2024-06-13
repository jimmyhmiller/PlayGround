use core::fmt;
use std::{collections::HashMap, error::Error, ops::Range, slice::from_raw_parts_mut};

use bincode::{Decode, Encode};
use mmap_rs::{Mmap, MmapMut, MmapOptions};

use crate::{
    arm::LowLevelArm,
    debugger,
    ir::{BuiltInTypes, StringValue, Value},
    parser::Parser,
    Data, Message,
};

// TODO: Make better
const GC_ALWAYS: bool = false;
const GC_NEVER: bool = false;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    name: String,
    offset_or_pointer: usize,
    jump_table_offset: usize,
    is_foreign: bool,
    pub is_builtin: bool,
    is_defined: bool,
}

#[derive(Debug)]
struct HeapObject<'a> {
    size: usize,
    data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<String>,
}

impl Struct {
    pub fn size(&self) -> usize {
        self.fields.len()
    }
}

struct StructManager {
    name_to_id: HashMap<String, usize>,
    structs: Vec<Struct>,
}

impl StructManager {
    fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            structs: Vec::new(),
        }
    }

    fn insert(&mut self, name: String, s: Struct) {
        let id = self.structs.len();
        self.name_to_id.insert(name.clone(), id);
        self.structs.push(s);
    }

    fn get(&self, name: &str) -> Option<(usize, &Struct)> {
        let id = self.name_to_id.get(name)?;
        self.structs.get(*id).map(|x| (*id, x))
    }

    pub fn get_by_id(&self, type_id: usize) -> Option<&Struct> {
        self.structs.get(type_id)
    }
}

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
    UseFree,
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
}

// TODO: It might be faster
// to have all the bitmaps together
// and then keep track of the indexes

#[derive(Debug, PartialEq, Eq)]
enum Mode {
    FillFromStart(usize),
    FreeList,
    Full,
}

// TODO:
// If i want to do this quickly, I need
// a fast way to get from pointer to reference bitmap
// I could store the bitmap in the segment itself.
// Then I could back into the pointer what the slot is and the size
// Then given that I can find the beginning of the segment
// and then update the bitmap
// I think that would be sufficiently fast


struct Segment2 {
    memory: MmapMut,
    size: usize,
    object_size: usize,
    allocated_bitmap: Vec<u64>,
    reference_bitmap: Vec<u64>,
    mode: Mode,
}

impl Segment2 {
    fn new(size: usize, object_size: usize) -> Self {
        assert!(
            size % object_size == 0,
            "Size must be a multiple of object size"
        );
        assert!(object_size % 8 == 0, "Object size must be a multiple of 8");
        assert!(size % 64 == 0, "Size must be a multiple of 64");
        let memory = MmapOptions::new(size)
            .unwrap()
            .map_mut()
            .unwrap()
            .make_mut()
            .unwrap_or_else(|(_map, e)| {
                panic!("Failed to make mmap executable: {}", e);
            });
        let bitmap_size = size / object_size / 64;
        let bitmap = vec![0; bitmap_size];
        Self {
            memory,
            size,
            object_size,
            allocated_bitmap: bitmap.clone(),
            reference_bitmap: bitmap,
            mode: Mode::FillFromStart(0),
        }
    }

    fn number_of_slots(&self) -> usize {
        self.size / self.object_size
    }

    fn find_empty_slot(&mut self) -> Option<usize> {
        match self.mode {
            Mode::FillFromStart(index) => {
                if index + 1 > self.number_of_slots() {
                    self.mode = Mode::FreeList;
                    self.reset_mode();
                    return self.find_empty_slot();
                }
                self.mode = if index + 1 == self.number_of_slots() {
                    Mode::Full
                } else {
                    Mode::FillFromStart(index + 1)
                };
                // println!("{:?}, {}", self.mode, index);
                return Some(index);
            }
            Mode::FreeList => {
                for (index, word) in self.allocated_bitmap.iter().enumerate() {
                    if *word != u64::MAX {
                        for i in 0..64 {
                            if (word & (1 << i)) == 0 {
                                let slot = index * 64 + i;
                                return Some(slot);
                            }
                        }
                    }
                }
            }
            Mode::Full => return None,
        }

        None
    }

    fn set_allocated_slot(&mut self, index: usize) {
        let word_index = index / 64;
        let bit_index = index % 64;
        assert!(self.allocated_bitmap[word_index] & (1 << bit_index) == 0);
        self.allocated_bitmap[word_index] |= 1 << bit_index;
    }

    fn set_reference_slot(&mut self, index: usize) {
        let word_index = index / 64;
        let bit_index = index % 64;
        self.reference_bitmap[word_index] |= 1 << bit_index;
    }

    fn get_reference_slot(&self, slot: usize) -> usize {
        let word_index = slot / 64;
        let bit_index = slot % 64;
        let bit = ((self.reference_bitmap[word_index] & (1 << bit_index)) >> bit_index) as usize;
        assert!(bit == 0 || bit == 1);
        bit
    }

    fn write_object(&mut self, index: usize, size: usize) {
        let start = index * self.object_size;
        let end = start + self.object_size;
        let memory = &mut self.memory;
        let shifted_size = (size << 1) as u32;
        let buffer = &mut memory[start..end];
        // write the size of the object to the first 8 bytes
        buffer[0..4].copy_from_slice(&shifted_size.to_le_bytes());
        // I don't technically need to do this
        // write zeros for the rest of object_size
        for i in 8..buffer.len() {
            buffer[i] = 0;
        }
    }

    fn is_full(&self) -> bool {
        if self.mode == Mode::Full {
            return true;
        }
        if matches!(self.mode, Mode::FillFromStart(_)) {
            return false;
        }
        self.allocated_bitmap.iter().all(|x| *x == u64::MAX)
    }

    fn set_full(&mut self) {
        if self.is_full() {
            self.mode = Mode::Full;
        }
    }

    fn deallocate_unreferenced(&mut self) {
        // For all the bits in reference_bitmap
        // we want any that are 0 to be 0 in allocated_bitmap
        // but keep the ones that are 1
        self.allocated_bitmap
            .iter_mut()
            .zip(self.reference_bitmap.iter())
            .for_each(|(allocated, reference)| {
                *allocated &= *reference;
            });
        self.reference_bitmap.iter_mut().for_each(|x| *x = 0);

        self.reset_mode();
    }

    fn reset_mode(&mut self) {
        if self.is_full() {
            self.mode = Mode::Full;
            return;
        }
        if self.allocated_bitmap.iter().all(|x| *x == 0) {
            self.mode = Mode::FillFromStart(0);
            return;
        }
    }
}

struct Heap2 {
    segments: Vec<(usize, Vec<Segment2>)>,
    segment_offsets: HashMap<usize, usize>,
    pending_marks: Vec<usize>,
    pointer_to_segment: Vec<(usize, (usize, usize))>,
    scale_factor: usize,
}

impl Heap2 {
    fn get_segment_or_create(&mut self, size: usize) -> &mut Segment2 {
        // TODO: Need a large object heap
        let nearest_power_of_two = size.next_power_of_two();
        let segment_offset = *self
            .segment_offsets
            .get(&nearest_power_of_two)
            .unwrap_or(&0);
        let mut position = self
            .segments
            .iter()
            .position(|x| x.0 == nearest_power_of_two);

        if position.is_none() {
            let segment = Segment2::new(size * 64 * 100, size);
            let start = segment.memory.as_ptr() as usize;
            self.pointer_to_segment
                .push((start, (nearest_power_of_two, 0)));
            self.pointer_to_segment.sort();
            self.segments.push((nearest_power_of_two, vec![segment]));
            self.segment_offsets.insert(nearest_power_of_two, 0);
            position = Some(self.segments.len() - 1);
        }
        self.segments
            .get_mut(position.unwrap())
            .unwrap()
            .1
            .get_mut(segment_offset)
            .unwrap()
    }

    fn all_segments_full(&self, size: usize) -> bool {
        let nearest_power_of_two = size.next_power_of_two();
        if let Some((_, segments)) = self.segments.iter().find(|x| x.0 == nearest_power_of_two) {
            return segments.iter().all(|x| x.is_full());
        }
        false
    }

    fn create_new_segment(&mut self, size: usize) {
        let nearest_power_of_two = size.next_power_of_two();
        if let Some((_, segments)) = self.segments.iter().find(|x| x.0 == nearest_power_of_two) {
            for (index, segment) in segments.iter().enumerate() {
                if !segment.is_full() {
                    self.segment_offsets.insert(nearest_power_of_two, index);
                    return;
                }
            }
        }

        let segment_count = self
            .segments
            .iter()
            .find(|x| x.0 == nearest_power_of_two)
            .unwrap()
            .1
            .len();

        if let Some((_, segments)) = self
            .segments
            .iter_mut()
            .find(|x| x.0 == nearest_power_of_two)
        {
            for _ in 0..self.scale_factor {
                let segment = Segment2::new(size * 64 * 100, size);
                self.pointer_to_segment.push((
                    segment.memory.as_ptr() as usize,
                    (nearest_power_of_two, segments.len()),
                ));
                segments.push(segment);
            }
        }
        self.pointer_to_segment.sort();

        self.segment_offsets
            .insert(nearest_power_of_two, segment_count);
        self.scale_factor *= 2;
        self.scale_factor = self.scale_factor.min(64);
    }

    fn new() -> Self {
        Self {
            segments: Vec::new(),
            segment_offsets: HashMap::new(),
            pending_marks: vec![],
            pointer_to_segment: vec![],
            scale_factor: 2,
        }
    }
}

struct Heap {
    segments: Vec<Segment>,
    segment_offset: usize,
    segment_size: usize,
    free_list: Vec<FreeListEntry>,
    scale_factor: usize,
}

impl Heap {
    fn new() -> Self {
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
        return SegmentAction::AllocateMore;
    }

    fn write_object(&mut self, segment_offset: usize, offset: usize, shifted_size: usize) -> usize {
        let memory = &mut self.segments[segment_offset].memory;

        let buffer = &mut memory[offset..offset + 8];

        // write the size of the object to the first 8 bytes
        buffer[..shifted_size.to_le_bytes().len()].copy_from_slice(&shifted_size.to_le_bytes());

        let pointer = buffer.as_ptr() as usize;
        pointer
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
        assert!(
            self.segments[self.segment_offset].offset % 8 == 0,
            "Heap offset is not aligned"
        );
    }
}

#[derive(Debug, Encode, Decode, Clone)]
pub struct StackMapDetails {
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

const STACK_SIZE: usize = 1024 * 1024 * 32;

pub struct Compiler {
    code_offset: usize,
    code_memory: Option<Mmap>,
    // TODO: Think about the jump table more
    jump_table: Option<Mmap>,
    // Do I need this offset?
    jump_table_offset: usize,
    structs: StructManager,
    functions: Vec<Function>,
    heap: Heap,
    heap2: Heap2,
    stack: Option<MmapMut>,
    string_constants: Vec<StringValue>,
    stack_map: Vec<(usize, StackMapDetails)>,
}

impl fmt::Debug for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Make this better
        f.debug_struct("Compiler")
            .field("code_offset", &self.code_offset)
            .field("jump_table_offset", &self.jump_table_offset)
            .field("functions", &self.functions)
            .finish()
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            code_memory: Some(
                MmapOptions::new(MmapOptions::page_size())
                    .unwrap()
                    .map()
                    .unwrap(),
            ),
            code_offset: 0,
            jump_table: Some(
                MmapOptions::new(MmapOptions::page_size())
                    .unwrap()
                    .map()
                    .unwrap(),
            ),
            jump_table_offset: 0,
            structs: StructManager::new(),
            functions: Vec::new(),
            heap: Heap::new(),
            heap2: Heap2::new(),
            stack: Some(MmapOptions::new(STACK_SIZE).unwrap().map_mut().unwrap()),
            string_constants: vec![],
            stack_map: vec![],
        }
    }

    pub fn find_stack_data(&self, pointer: usize) -> Option<&StackMapDetails> {
        for (key, value) in self.stack_map.iter() {
            if *key == pointer.saturating_sub(4) {
                return Some(value);
            }
        }
        None
    }

    pub fn mark2(&mut self, current_stack_pointer: usize) {
        let start = std::time::Instant::now();
        // println!("Marking");
        // Get the stack as a [usize]
        let stack = self.stack.as_ref().unwrap();
        // I'm adding to the end of the stack I've allocated so I only need to go from the end
        // til the current stack

        let stack_end = stack.as_ptr() as usize + (STACK_SIZE);
        // let current_stack_pointer = current_stack_pointer & !0b111;
        let distance_till_end = stack_end - current_stack_pointer;
        let num_64_till_end = (distance_till_end / 8) + 1;
        let stack =
            unsafe { std::slice::from_raw_parts(stack.as_ptr() as *const usize, STACK_SIZE / 8) };
        let stack = &stack[stack.len() - num_64_till_end..];

        // for value in stack.iter() {
        //    println!("0x{:x}", value)
        // }
        // Walk the stack

        let mut to_mark: Vec<usize> = Vec::with_capacity(128);

        let mut i = 0;
        while i < stack.len() {
            let value = stack[i];

            if let Some(details) = self.find_stack_data(value) {
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
                let mut data = pointer.cast::<usize>();

                let mut address = data as usize;
                let size = unsafe { *data >> 1 };
                let mut segment_pointer_belongs_to = None;
                for (start, (size, segment_index)) in self.heap2.pointer_to_segment.iter().rev() {
                    
                    if address >= *start {
                        segment_pointer_belongs_to = self
                            .heap2
                            .segments
                            .iter_mut()
                            .find(|x| x.0 == *size)
                            .unwrap()
                            .1
                            .get_mut(*segment_index);
                        break;
                    }
                }

                if segment_pointer_belongs_to.is_none() {
                    panic!("Pointer not in any segment");
                }
                let segment_pointer_belongs_to = segment_pointer_belongs_to.unwrap();

                let offset = address - segment_pointer_belongs_to.memory.as_ptr() as usize;
                let slot = offset / size;

                let referenced = segment_pointer_belongs_to.get_reference_slot(slot);
                if referenced == 1 {
                    continue;
                }

                segment_pointer_belongs_to.set_reference_slot(slot);
                let referenced = segment_pointer_belongs_to.get_reference_slot(slot);
                assert!(referenced == 1);

                let size = *(pointer as *const usize) >> 1;
                let data = std::slice::from_raw_parts(pointer.add(8) as *const usize, size / 8);
                for datum in data.iter() {
                    if BuiltInTypes::is_heap_pointer(*datum) {
                        to_mark.push(*datum)
                    }
                }
            }
        }

        // self.sweep();

        println!("Marking took {:?}", start.elapsed());

        // println!("marked {}", marked);
    }

    pub fn mark(
        &mut self,
        current_stack_pointer: usize,
        mark_fn: impl Fn(&mut Compiler, *mut usize) -> bool,
    ) {
        let start = std::time::Instant::now();
        // println!("Marking");
        // Get the stack as a [usize]
        let stack = self.stack.as_ref().unwrap();
        // I'm adding to the end of the stack I've allocated so I only need to go from the end
        // til the current stack

        let stack_end = stack.as_ptr() as usize + (STACK_SIZE);
        // let current_stack_pointer = current_stack_pointer & !0b111;
        let distance_till_end = stack_end - current_stack_pointer;
        let num_64_till_end = (distance_till_end / 8) + 1;
        let stack =
            unsafe { std::slice::from_raw_parts(stack.as_ptr() as *const usize, STACK_SIZE / 8) };
        let stack = &stack[stack.len() - num_64_till_end..];

        // for value in stack.iter() {
        //    println!("0x{:x}", value)
        // }
        // Walk the stack

        let mut to_mark: Vec<usize> = Vec::with_capacity(128);

        let mut i = 0;
        while i < stack.len() {
            let value = stack[i];

            if let Some(details) = self.find_stack_data(value) {
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
                let mut data = pointer.cast::<usize>();
                let is_already_marked = mark_fn(self, data);
                if is_already_marked {
                    continue;
                }
                // *pointer.cast::<usize>() = data;

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

        // self.sweep();

        println!("Marking took {:?}", start.elapsed());

        // println!("marked {}", marked);
    }

    fn sweep(&mut self) {
        // println!("Sweeping");
        let mut free_entries: Vec<FreeListEntry> = Vec::with_capacity(128);
        for (segment_index, segment) in self.heap.segments.iter_mut().enumerate() {
            if segment.offset == 0 {
                continue;
            }
            let mut free_in_segment: Vec<&FreeListEntry> = self
                .heap
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

        // for pointer in freed_pointers {
        //     let tagged = BuiltInTypes::Struct.tag(pointer as isize) as usize;
        //     println!("Freeing 0x{:x} {:?}", tagged, self.get_repr(tagged, 0));
        // }

        for entry in free_entries {
            self.heap.add_free(entry);
        }

        // println!("======\n\n");
    }

    pub fn get_heap_pointer(&self) -> usize {
        // TODO: I need to tell my debugger about all the heap pointers
        self.heap.segment_pointer(0)
    }

    pub fn mark_bits(&mut self, data: *mut usize) -> bool {
        unsafe {
            let value = *data;
            // check right most bit
            if (value & 1) == 1 {
                return true;
            }
            *data = value | 1;
            return false;
        }
    }

    fn sweep2(&mut self) {
        for (_, segments) in self.heap2.segments.iter_mut() {
            for segment in segments.iter_mut() {
                if segment.mode == Mode::FillFromStart(0) {
                    continue;
                }
                segment.deallocate_unreferenced();
            }
        }
    }

    fn allocate2(&mut self, stack_pointer: usize, size: usize) -> usize {
        // TODO: Better
        if GC_ALWAYS {
            self.mark2(stack_pointer);
            self.sweep2();
        }

        // if self.heap2.all_segments_full(size) {
        //     self.mark(stack_pointer, Self::mark_in_bitmap);
        //     self.sweep2();
        // }
        let mut segment = self.heap2.get_segment_or_create(size);
        if segment.is_full() {
            self.heap2.create_new_segment(size);
            self.mark2(stack_pointer);
            self.sweep2();
            segment = self.heap2.get_segment_or_create(size);
        }
        let slot = segment.find_empty_slot();
        if slot.is_none() {
            panic!(
                "No empty slot found {:?} {:?}",
                segment.mode,
                segment.is_full()
            );
        }
        let slot = slot.unwrap();
        segment.set_allocated_slot(slot);
        segment.write_object(slot, size);
        let pointer = segment.memory.as_ptr() as usize + slot * size;
        pointer
    }

    pub fn allocate(
        &mut self,
        bytes: usize,
        stack_pointer: usize,
        kind: BuiltInTypes,
        depth: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let result = self.allocate2(stack_pointer, (bytes + 1) * 8);
        // self.mark( stack_pointer, Self::mark_in_bitmap);
        return Ok(result);

        if GC_ALWAYS && !GC_NEVER {
            self.mark(stack_pointer, Self::mark_bits);
        }

        if depth > 1 {
            println!("Huh?");
        }

        let size = (bytes + 1) * 8;
        let shifted_size = (bytes * 8) << 1;

        if self.heap.current_offset() + size < self.heap.current_segment_size() {
            let pointer = self.heap.write_object(
                self.heap.segment_offset,
                self.heap.current_offset(),
                shifted_size,
            );
            self.heap.increment_current_offset(size);
            return Ok(pointer);
        }

        if self.heap.switch_to_available_segment(size) {
            return self.allocate(bytes, stack_pointer, kind, depth + 1);
        }

        assert!(
            !self.heap.segments.iter().any(|x| x.offset == 0),
            "Available segment not being used"
        );

        let mut spot = self
            .heap
            .free_list
            .iter_mut()
            .enumerate()
            .find(|(_, x)| x.size >= size);

        if spot.is_none() {
            if !GC_NEVER {
                self.mark(stack_pointer, Self::mark_bits);
            }
            if self.heap.switch_to_available_segment(size) {
                return self.allocate(bytes, stack_pointer, kind, depth + 1);
            }

            spot = self
                .heap
                .free_list
                .iter_mut()
                .enumerate()
                .find(|(_, x)| x.size >= size);

            if spot.is_none() {
                self.heap.create_more_segments(size);
                return self.allocate(bytes, stack_pointer, kind, depth + 1);
            }
        }

        let (spot_index, spot) = spot.unwrap();

        let mut spot_clone = spot.clone();
        spot_clone.size = size;
        spot.size -= size;
        spot.offset += size;
        if spot.size == 0 {
            self.heap.free_list.remove(spot_index);
        }

        let pointer = self
            .heap
            .write_object(spot_clone.segment, spot_clone.offset, shifted_size);
        Ok(pointer)
    }

    fn get_stack_pointer(&self) -> usize {
        // I think I want the end of the stack
        (self.stack.as_ref().unwrap().as_ptr() as usize) + (STACK_SIZE)
    }

    pub fn add_foreign_function(
        &mut self,
        name: &str,
        function: *const u8,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let offset = function as usize;
        let jump_table_offset = self.add_jump_table_entry(index, offset)?;
        self.functions.push(Function {
            name: name.to_string(),
            offset_or_pointer: offset,
            jump_table_offset,
            is_foreign: true,
            is_builtin: false,
            is_defined: true,
        });
        debugger(Message {
            kind: "foreign_function".to_string(),
            data: Data::ForeignFunction {
                name: name.to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone())
                    .unwrap(),
            },
        });
        Ok(self.functions.len() - 1)
    }
    pub fn add_builtin_function(
        &mut self,
        name: &str,
        function: *const u8,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let offset = function as usize;

        let jump_table_offset = self.add_jump_table_entry(index, offset)?;
        self.functions.push(Function {
            name: name.to_string(),
            offset_or_pointer: offset,
            jump_table_offset,
            is_foreign: true,
            is_builtin: true,
            is_defined: true,
        });
        debugger(Message {
            kind: "builtin_function".to_string(),
            data: Data::BuiltinFunction {
                name: name.to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone())
                    .unwrap(),
            },
        });
        Ok(self.functions.len() - 1)
    }

    pub fn upsert_function(
        &mut self,
        name: &str,
        code: &mut LowLevelArm,
    ) -> Result<usize, Box<dyn Error>> {
        let bytes = &(code.compile_to_bytes());
        for (index, function) in self.functions.iter_mut().enumerate() {
            if function.name == name {
                self.overwrite_function(index, bytes)?;
                break;
            }
        }
        self.add_function(name, bytes)?;

        // TODO: Make this better
        let function = self.find_function(name).unwrap();
        let function_pointer = Self::get_function_pointer(self, function.clone()).unwrap();

        let translated_stack_map = code.translate_stack_map(function_pointer);
        let translated_stack_map: Vec<(usize, StackMapDetails)> = translated_stack_map
            .iter()
            .map(|(key, value)| {
                (
                    *key,
                    StackMapDetails {
                        current_stack_size: *value,
                        number_of_locals: code.max_locals as usize,
                        max_stack_size: code.max_stack_size as usize,
                    },
                )
            })
            .collect();

        debugger(Message {
            kind: "stack_map".to_string(),
            data: Data::StackMap {
                pc: function_pointer,
                name: name.to_string(),
                stack_map: translated_stack_map.clone(),
            },
        });
        self.stack_map.extend(translated_stack_map);

        debugger(Message {
            kind: "user_function".to_string(),
            data: Data::UserFunction {
                name: name.to_string(),
                pointer: function_pointer,
                len: bytes.len(),
            },
        });
        Ok(function_pointer)
    }

    pub fn reserve_function(&mut self, name: &str) -> Result<Function, Box<dyn Error>> {
        for function in self.functions.iter_mut() {
            if function.name == name {
                return Ok(function.clone());
            }
        }
        let index = self.functions.len();
        let jump_table_offset = self.add_jump_table_entry(index, 0)?;
        let function = Function {
            name: name.to_string(),
            offset_or_pointer: 0,
            jump_table_offset,
            is_foreign: false,
            is_builtin: false,
            is_defined: false,
        };
        self.functions.push(function.clone());
        Ok(function)
    }

    pub fn add_function(&mut self, name: &str, code: &[u8]) -> Result<usize, Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let index = self.functions.len();
        self.functions.push(Function {
            name: name.to_string(),
            offset_or_pointer: offset,
            jump_table_offset: 0,
            is_foreign: false,
            is_builtin: false,
            is_defined: true,
        });
        let function_pointer =
            Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        let jump_table_offset = self.add_jump_table_entry(index, function_pointer)?;

        self.functions[index].jump_table_offset = jump_table_offset;
        Ok(jump_table_offset)
    }

    pub fn overwrite_function(
        &mut self,
        index: usize,
        code: &[u8],
    ) -> Result<usize, Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let function = &mut self.functions[index];
        function.offset_or_pointer = offset;
        let jump_table_offset = function.jump_table_offset;
        let function_clone = function.clone();
        let function_pointer = self.get_function_pointer(function_clone).unwrap();
        self.modify_jump_table_entry(jump_table_offset, function_pointer)?;
        let function = &mut self.functions[index];
        function.is_defined = true;
        Ok(function.jump_table_offset)
    }

    pub fn get_function_pointer(&self, function: Function) -> Result<usize, Box<dyn Error>> {
        // Gets the absolute pointer to a function
        // if it is a foreign function, return the offset
        // if it is a local function, return the offset + the start of code_memory
        if function.is_foreign {
            Ok(function.offset_or_pointer)
        } else {
            Ok(function.offset_or_pointer + self.code_memory.as_ref().unwrap().as_ptr() as usize)
        }
    }

    pub fn get_jump_table_pointer(&self, function: Function) -> Result<usize, Box<dyn Error>> {
        Ok(function.jump_table_offset * 8 + self.jump_table.as_ref().unwrap().as_ptr() as usize)
    }

    pub fn add_jump_table_entry(
        &mut self,
        _index: usize,
        pointer: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let jump_table_offset = self.jump_table_offset;
        let memory = self.jump_table.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[jump_table_offset * 8..];
        let pointer = BuiltInTypes::Function.tag(pointer as isize) as usize;
        // Write full usize to buffer
        for (index, byte) in pointer.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        let mem = memory.make_read_only().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap read_only: {}", e);
        });
        self.jump_table_offset += 1;
        self.jump_table = Some(mem);
        Ok(jump_table_offset)
    }

    fn modify_jump_table_entry(
        &mut self,
        jump_table_offset: usize,
        function_pointer: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let memory = self.jump_table.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[jump_table_offset * 8..];

        let function_pointer = BuiltInTypes::Function.tag(function_pointer as isize) as usize;
        // Write full usize to buffer
        for (index, byte) in function_pointer.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        let mem = memory.make_read_only().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap read_only: {}", e);
        });
        self.jump_table = Some(mem);
        Ok(jump_table_offset)
    }

    pub fn add_code(&mut self, code: &[u8]) -> Result<usize, Box<dyn Error>> {
        let start = self.code_offset;
        let memory = self.code_memory.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[self.code_offset..];

        for (index, byte) in code.iter().enumerate() {
            buffer[index] = *byte;
        }

        let size: usize = memory.size();
        memory.flush(0..size)?;
        memory.flush_icache()?;
        self.code_offset += code.len();

        let exec = memory.make_exec().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap executable: {}", e);
        });

        self.code_memory = Some(exec);

        Ok(start)
    }

    pub fn get_compiler_ptr(&self) -> *const Compiler {
        self as *const Compiler
    }

    // TODO: Make this good
    pub fn run(&self, jump_table_offset: usize) -> Result<u64, Box<dyn Error>> {
        // get offset stored in jump table as a usize
        let offset =
            &self.jump_table.as_ref().unwrap()[jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let trampoline = self
            .functions
            .iter()
            .find(|f| f.name == "trampoline")
            .unwrap();
        let trampoline_jump_table_offset = trampoline.jump_table_offset;
        let trampoline_offset = &self.jump_table.as_ref().unwrap()
            [trampoline_jump_table_offset * 8..trampoline_jump_table_offset * 8 + 8];

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(trampoline_offset);
        let trampoline_start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let f: fn(u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        let stack_pointer = self.get_stack_pointer();
        let result = f(stack_pointer as u64, start as u64);
        Ok(result)
    }

    pub fn run1(&self, jump_table_offset: usize, arg: u64) -> Result<u64, Box<dyn Error>> {
        // get offset stored in jump table as a usize
        let offset =
            &self.jump_table.as_ref().unwrap()[jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let trampoline = self
            .functions
            .iter()
            .find(|f| f.name == "trampoline")
            .unwrap();
        let trampoline_jump_table_offset = trampoline.jump_table_offset;
        let trampoline_offset = &self.jump_table.as_ref().unwrap()
            [trampoline_jump_table_offset * 8..trampoline_jump_table_offset * 8 + 8];

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(trampoline_offset);
        let trampoline_start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let f: fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        let arg = BuiltInTypes::Int.tag(arg as isize) as u64;
        let stack_pointer = self.get_stack_pointer();
        let result = f(stack_pointer as u64, start as u64, arg);
        Ok(result)
    }

    pub fn run2(
        &self,
        _jump_table_offset: usize,
        _arg1: u64,
        _arg2: u64,
    ) -> Result<u64, Box<dyn Error>> {
        unimplemented!("Add trampoline");
    }

    pub fn get_repr(&self, value: usize, depth: usize) -> Option<String> {
        if depth > 1000 {
            return Some("...".to_string());
        }
        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Null => Some("null".to_string()),
            BuiltInTypes::Int => {
                let value = BuiltInTypes::untag(value);
                Some(value.to_string())
            }
            BuiltInTypes::Float => todo!(),
            BuiltInTypes::String => {
                let value = BuiltInTypes::untag(value);
                let string = &self.string_constants[value];
                Some(string.str.clone())
            }
            BuiltInTypes::Bool => {
                let value = BuiltInTypes::untag(value);
                if value == 0 {
                    Some("false".to_string())
                } else {
                    Some("true".to_string())
                }
            }
            BuiltInTypes::Function => Some("function".to_string()),
            BuiltInTypes::Closure => todo!(),
            BuiltInTypes::Struct => {
                unsafe {
                    let value = BuiltInTypes::untag(value);
                    let pointer = value as *const u8;

                    if pointer as usize % 8 != 0 {
                        panic!("Not aligned");
                    }
                    // get first 8 bytes as size le encoded
                    let size = *(pointer as *const usize) >> 1;
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts(pointer, size);
                    // type id is the first 8 bytes of data
                    let type_id = usize::from_le_bytes(data[0..8].try_into().unwrap());
                    let type_id = BuiltInTypes::untag(type_id);
                    let struct_value = self.structs.get_by_id(type_id as usize);
                    Some(self.get_struct_repr(struct_value?, data[8..].to_vec(), depth + 1)?)
                }
            }
            BuiltInTypes::Array => {
                unsafe {
                    let value = BuiltInTypes::untag(value);
                    let pointer = value as *const u8;
                    // get first 8 bytes as size le encoded
                    let size = *(pointer as *const usize) >> 1;
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts(pointer, size);
                    let heap_object = HeapObject { size, data };

                    let mut repr = "[".to_string();
                    for i in 0..heap_object.size {
                        let value = &heap_object.data[i * 8..i * 8 + 8];
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(value);
                        let value = usize::from_le_bytes(bytes);
                        repr.push_str(&self.get_repr(value, depth + 1)?);
                        if i != heap_object.size - 1 {
                            repr.push_str(", ");
                        }
                    }
                    repr.push(']');
                    Some(repr)
                }
            }
        }
    }

    pub fn println(&self, value: usize) {
        println!("{}", self.get_repr(value, 0).unwrap());
    }

    pub fn print(&self, value: usize) {
        print!("{}", self.get_repr(value, 0).unwrap());
    }

    pub fn array_store(
        &mut self,
        array: usize,
        index: usize,
        value: usize,
    ) -> Result<usize, Box<dyn Error>> {
        unsafe {
            let tag = BuiltInTypes::get_kind(array);
            match tag {
                BuiltInTypes::Array => {
                    // TODO: Bounds check
                    let index = BuiltInTypes::untag(index);
                    let index = index * 8;
                    let array = BuiltInTypes::untag(array);
                    let pointer = array as *mut u8;
                    // get first 8 bytes as size
                    let size = *(pointer as *const usize) >> 1;
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts_mut(pointer, size);
                    // store all 8 bytes of value in data
                    for (offset, byte) in value.to_le_bytes().iter().enumerate() {
                        data[index + offset] = *byte;
                    }
                    Ok(BuiltInTypes::Array.tag(array as isize) as usize)
                }
                _ => panic!("Not an array"),
            }
        }
    }

    pub(crate) fn array_get(
        &mut self,
        array: usize,
        index: usize,
    ) -> Result<usize, Box<dyn Error>> {
        unsafe {
            let tag = BuiltInTypes::get_kind(array);
            match tag {
                BuiltInTypes::Array => {
                    let index = BuiltInTypes::untag(index);
                    let array = BuiltInTypes::untag(array);
                    let pointer = array as *mut u8;
                    // get first 8 bytes as size
                    let size = *(pointer as *const usize);
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts_mut(pointer, size);
                    // get next 8 bytes as usize
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&data[index * 8..index * 8 + 8]);
                    let data = usize::from_le_bytes(bytes);
                    Ok(data)
                }
                _ => panic!("Not an array"),
            }
        }
    }

    pub fn compile(&mut self, code: String) -> Result<(), Box<dyn Error>> {
        let mut parser = Parser::new(code);
        let ast = parser.parse();
        self.compile_ast(ast)
    }

    // TODO: Strings disappear because I am holding on to them only in IR

    pub fn compile_ast(&mut self, ast: crate::ast::Ast) -> Result<(), Box<dyn Error>> {
        ast.compile(self);
        Ok(())
    }

    // TODO: Make less ugly
    pub(crate) fn run_function(&self, arg: &str, vec: Vec<i32>) -> u64 {
        match vec.len() {
            0 => {
                let function = self.functions.iter().find(|f| f.name == arg).unwrap();
                self.run(function.jump_table_offset).unwrap()
            }
            1 => {
                let function = self.functions.iter().find(|f| f.name == arg).unwrap();

                self.run1(function.jump_table_offset, vec[0] as u64)
                    .unwrap()
            }
            2 => {
                let function = self.functions.iter().find(|f| f.name == arg).unwrap();
                self.run2(function.jump_table_offset, vec[0] as u64, vec[1] as u64)
                    .unwrap()
            }
            _ => panic!("Too many arguments"),
        }
    }

    pub fn add_string(&mut self, string_value: StringValue) -> Value {
        self.string_constants.push(string_value);
        let offset = self.string_constants.len() - 1;
        Value::StringConstantPtr(offset)
    }

    pub(crate) fn find_function(&self, name: &str) -> Option<Function> {
        self.functions.iter().find(|f| f.name == name).cloned()
    }

    pub fn get_function_by_pointer(&self, value: usize) -> Option<&Function> {
        let offset = value.saturating_sub(self.code_memory.as_ref().unwrap().as_ptr() as usize);
        self.functions
            .iter()
            .find(|f| f.offset_or_pointer == offset)
    }

    pub fn get_function_by_pointer_mut(&mut self, value: usize) -> Option<&mut Function> {
        let offset = value - self.code_memory.as_ref().unwrap().as_ptr() as usize;
        self.functions
            .iter_mut()
            .find(|f| f.offset_or_pointer == offset)
    }

    // TODO: Call this
    // After I make a function, I need to check if there are free variables
    // If so I need to grab those variables from the evironment
    // pop them on the stack
    // call this funciton with a pointer to the base of the stack
    // and the number of variables
    // then I need to pop the stack
    // and get the closure
    // When it comes to function calls
    // if it is a function, do direct call.
    // if it is a closure, grab the closure data
    // and put the free variables on the stack before the locals

    pub fn make_closure(
        &mut self,
        function: usize,
        free_variables: &[usize],
    ) -> Result<usize, Box<dyn Error>> {
        let len = 8 + 8 + free_variables.len() * 8;
        // TODO: Stack pointer should be passed in
        let heap_pointer = self.allocate(len, 0, BuiltInTypes::Closure, 0)?;
        let pointer = heap_pointer as *mut u8;
        let num_free = free_variables.len();
        let function = function.to_le_bytes();
        let free_variables = free_variables.iter().flat_map(|v| v.to_le_bytes());
        let buffer = unsafe { from_raw_parts_mut(pointer, len) };
        // write function pointer
        for (index, byte) in function.iter().enumerate() {
            buffer[index] = *byte;
        }
        let num_free = num_free.to_le_bytes();
        // Write number of free variables
        for (index, byte) in num_free.iter().enumerate() {
            buffer[8 + index] = *byte;
        }
        // write free variables
        for (index, byte) in free_variables.enumerate() {
            buffer[16 + index] = byte;
        }
        Ok(BuiltInTypes::Closure.tag(heap_pointer as isize) as usize)
    }

    pub fn property_access(&self, struct_pointer: usize, str_constant_ptr: usize) -> usize {
        unsafe {
            let struct_pointer = BuiltInTypes::untag(struct_pointer);
            let struct_pointer = struct_pointer as *const u8;
            let size = *(struct_pointer as *const usize) >> 1;
            let str_constant_ptr: usize = BuiltInTypes::untag(str_constant_ptr);
            let string_value = &self.string_constants[str_constant_ptr];
            let string = &string_value.str;
            let struct_pointer = struct_pointer.add(8);
            let data = std::slice::from_raw_parts(struct_pointer, size);
            // type id is the first 8 bytes of data
            let type_id = usize::from_le_bytes(data[0..8].try_into().unwrap());
            let type_id = BuiltInTypes::untag(type_id);
            let struct_value = self.structs.get_by_id(type_id as usize).unwrap();
            let field_index = struct_value
                .fields
                .iter()
                .position(|f| f == string)
                .unwrap();
            let field_index = (field_index + 1) * 8;
            let field = &data[field_index..field_index + 8];
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(field);

            usize::from_le_bytes(bytes)
        }
    }

    pub fn add_struct(&mut self, s: Struct) {
        self.structs.insert(s.name.clone(), s);
    }

    pub fn get_struct(&self, name: &str) -> Option<(usize, &Struct)> {
        self.structs.get(name)
    }

    fn get_struct_repr(
        &self,
        struct_value: &Struct,
        to_vec: Vec<u8>,
        depth: usize,
    ) -> Option<String> {
        // It should look like this
        // struct_name { field1: value1, field2: value2 }
        let mut repr = struct_value.name.clone();
        repr.push_str(" { ");
        for (index, field) in struct_value.fields.iter().enumerate() {
            repr.push_str(field);
            repr.push_str(": ");
            let value = &to_vec[index * 8..index * 8 + 8];
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(value);
            let value = usize::from_le_bytes(bytes);
            repr.push_str(&self.get_repr(value, depth + 1)?);
            if index != struct_value.fields.len() - 1 {
                repr.push_str(", ");
            }
        }
        repr.push_str(" }");
        Some(repr)
    }

    pub fn check_functions(&self) {
        let undefined_functions: Vec<&Function> =
            self.functions.iter().filter(|f| !f.is_defined).collect();
        if !undefined_functions.is_empty() {
            panic!(
                "Undefined functions: {:?}",
                undefined_functions
                    .iter()
                    .map(|f| f.name.clone())
                    .collect::<Vec<String>>()
            );
        }
    }

    pub fn get_function_by_name_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.iter_mut().find(|f| f.name == name)
    }
}
