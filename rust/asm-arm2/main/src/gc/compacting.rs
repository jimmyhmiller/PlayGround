

use mmap_rs::{MmapMut, MmapOptions};

use crate::ir::BuiltInTypes;

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

struct Space {
    space: Vec<Segment>,
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
        let segment = &space.space[self.segment_index];
        if segment.offset == 0 {
            return None;
        }
        let pointer = unsafe { segment.memory.as_ptr().add(self.offset) };
        let size = unsafe { *(pointer as *const usize) >> 1 };

        self.offset += size + 8;
        if self.offset == segment.offset {
            self.segment_index += 1;
            self.offset = 0;
        }
        Some(pointer)
    }
}

impl Space {
    fn new(segment_size: usize, scale_factor: usize) -> Self {
        let mut space = vec![];
        space.push(Segment::new(segment_size));
        Self {
            space,
            segment_offset: 0,
            segment_size,
            scale_factor,
        }
    }

    fn object_iter(&self) -> impl Iterator<Item = *const u8> {
        ObjectIterator {
            space: self,
            segment_index: 0,
            offset: 0,
        }
    }
}

pub struct CompactingHeap {
    from_space: Space,
    to_space: Space,
}


// impl CompactingHeap {
//     // TODO: I need to change this into a copy from roots to heap
//     // not a segment.
//     // That means I need be able to capture the state before I start adding objects
//     // and then be able to iterate over the new ones added.
//     // Right now, this would cause problems, because the objects alive from the roots
//     // will probably not fit in one segment.

//     // I also should move this to a new struct

//     // I really want to experiment more with gc, but it feels so bogged down in the implementation
//     // details right now.
//     #[allow(unused)]
//     unsafe fn copy_all(&mut self, roots: Vec<usize>) -> Vec<usize> {
//         // TODO: Is this vec the best way? Probably not
//         // I could hand this the pointers to the stack location
//         // then resolve what they point to and update them?
//         // I should think about how to get rid of this allocation at the very least.
//         let mut new_roots = vec![];
//         for root in roots {
//             new_roots.push(self.copy_using_cheneys_algorithm(root, to_space));
//         }

//         let mut offset = 0;
//         let memory = to_space.memory.as_mut_ptr();
//         let memory_range = to_space.memory.as_ptr_range();
//         // I could maybe operate on this as a slice rather than pointers
//         while offset < to_space.offset {
//             let pointer = memory.add(offset);
//             let size: usize = *pointer.cast::<usize>() >> 1;
//             let data = std::slice::from_raw_parts(pointer.add(8) as *const usize, size / 8);
//             for datum in data.iter() {
//                 if BuiltInTypes::is_heap_pointer(*datum) {
//                     let untagged = BuiltInTypes::untag(*datum);
//                     let pointer = untagged as *const u8;
//                     if memory_range.contains(&pointer) {
//                         continue;
//                     }
//                     self.copy_using_cheneys_algorithm(pointer as usize, to_space);
//                 }
//             }
//             offset += size + 8;

//         }

//         new_roots
//     }
    

//     // TODO: Finish this
//     unsafe fn copy_using_cheneys_algorithm(&mut self, root: usize, to_segment: &mut Segment) -> usize {
//         // I could make this check the memory range.
//         // In the original it does. But I don't think I have to?

//         let untagged = BuiltInTypes::untag(root);
//         let pointer = untagged as *const u8;
//         // If it is marked, we have copied it already
//         // the first 8 bytes are a tagged forward pointer
//         let data: usize = *pointer.cast::<usize>();
//         if (data & 1) == 1 {
//             return data >> 1;
//         }

//         let size = *(pointer as *const usize) >> 1;
//         let data = std::slice::from_raw_parts(pointer as *const u8, size + 8);
//         let new_pointer = to_segment.copy_data_to_offset(data);
//         // update header of original object to now be the forwarding pointer
//         let pointer = pointer as *mut usize;
//         let tagged_new = BuiltInTypes::get_kind(root).tag(new_pointer as isize) as usize;
//         *pointer = tagged_new | 1;
//         tagged_new
//     }
    
// }