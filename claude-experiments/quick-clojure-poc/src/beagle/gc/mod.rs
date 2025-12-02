use std::{error::Error, thread::ThreadId};

use bincode::{Decode, Encode};
use nanoserde::SerJson;

use crate::{CommandLineArguments, types::BuiltInTypes};

pub mod compacting;
pub mod generational;
pub mod mark_and_sweep;
pub mod mutex_allocator;
pub mod stack_segments;
pub mod stack_walker;

#[derive(Debug, Encode, Decode, SerJson, Clone)]
pub struct StackMapDetails {
    pub function_name: Option<String>,
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

pub const STACK_SIZE: usize = 1024 * 1024 * 128;

#[derive(Debug, Clone)]
pub struct StackMap {
    details: Vec<(usize, StackMapDetails)>,
}

impl Default for StackMap {
    fn default() -> Self {
        Self::new()
    }
}

impl StackMap {
    pub fn new() -> Self {
        Self { details: vec![] }
    }

    pub fn find_stack_data(&self, pointer: usize) -> Option<&StackMapDetails> {
        for (key, value) in self.details.iter() {
            if *key == pointer.saturating_sub(4) {
                return Some(value);
            }
        }
        None
    }

    pub fn extend(&mut self, translated_stack_map: Vec<(usize, StackMapDetails)>) {
        self.details.extend(translated_stack_map);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AllocatorOptions {
    pub gc: bool,
    pub print_stats: bool,
    pub gc_always: bool,
}

pub enum AllocateAction {
    Allocated(*const u8),
    Gc,
}

pub fn get_allocate_options(command_line_arguments: &CommandLineArguments) -> AllocatorOptions {
    AllocatorOptions {
        gc: !command_line_arguments.no_gc,
        print_stats: command_line_arguments.show_gc_times,
        gc_always: command_line_arguments.gc_always,
    }
}

pub trait Allocator {
    fn new(options: AllocatorOptions) -> Self;

    // TODO: I probably want something like kind, but not actually kind
    // I might need to allocate things differently based on type
    fn try_allocate(
        &mut self,
        words: usize,
        kind: BuiltInTypes,
    ) -> Result<AllocateAction, Box<dyn Error>>;

    fn gc(&mut self, stack_map: &StackMap, stack_pointers: &[(usize, usize)]);

    fn grow(&mut self);
    fn gc_add_root(&mut self, old: usize);
    fn register_temporary_root(&mut self, root: usize) -> usize;
    fn unregister_temporary_root(&mut self, id: usize) -> usize;
    fn add_namespace_root(&mut self, namespace_id: usize, root: usize);
    // TODO: Get rid of allocation
    fn get_namespace_relocations(&mut self) -> Vec<(usize, Vec<(usize, usize)>)>;

    #[allow(unused)]
    fn get_pause_pointer(&self) -> usize {
        0
    }

    fn register_thread(&mut self, _thread_id: ThreadId) {}

    fn remove_thread(&mut self, _thread_id: ThreadId) {}

    fn register_parked_thread(&mut self, _thread_id: ThreadId, _stack_pointer: usize) {}

    fn get_allocation_options(&self) -> AllocatorOptions;
}
