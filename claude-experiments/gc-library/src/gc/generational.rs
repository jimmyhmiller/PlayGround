//! Generational garbage collector.
//!
//! This collector uses a young generation for new allocations and promotes
//! surviving objects to an old generation. Write barriers track old-to-young
//! pointers using a card table.

use alloc::vec::Vec;
use core::marker::PhantomData;

use super::usdt_probes;
use crate::traits::{ForwardingSupport, GcObject, GcTypes, RootProvider, TaggedPointer};

use super::{mark_and_sweep::MarkAndSweep, AllocError, AllocateAction, Allocator, AllocatorOptions, MemoryProvider};

const DEFAULT_PAGE_COUNT: usize = 1024;
const MAX_PAGE_COUNT: usize = 1000000;

/// Card size in bytes (512 = 2^9)
const CARD_SIZE_LOG2: usize = 9;
const CARD_SIZE: usize = 1 << CARD_SIZE_LOG2;

// =============================================================================
// Card Table
// =============================================================================

/// Card table for write barrier tracking.
///
/// Each byte in the table represents one 512-byte "card" of the old generation heap.
/// When a heap store occurs, the card containing the destination is marked dirty.
/// During minor GC, only dirty cards need to be scanned for old-to-young references.
pub struct CardTable {
    cards: Vec<u8>,
    heap_start: usize,
    card_count: usize,
    biased_ptr: *mut u8,
    dirty_card_indices: Vec<usize>,
}

unsafe impl Send for CardTable {}
unsafe impl Sync for CardTable {}

impl CardTable {
    fn new(heap_start: usize, heap_size: usize) -> Self {
        let card_count = heap_size.div_ceil(CARD_SIZE);
        let mut cards = alloc::vec![0u8; card_count];
        let biased_ptr = unsafe { cards.as_mut_ptr().sub(heap_start >> CARD_SIZE_LOG2) };
        Self {
            cards,
            heap_start,
            card_count,
            biased_ptr,
            dirty_card_indices: Vec::with_capacity(64),
        }
    }

    #[inline]
    pub fn mark_dirty(&mut self, addr: usize) {
        let card_index = (addr - self.heap_start) >> CARD_SIZE_LOG2;
        if card_index < self.card_count {
            if self.cards[card_index] == 0 {
                self.cards[card_index] = 1;
                self.dirty_card_indices.push(card_index);
            }
        }
    }

    pub fn resize(&mut self, new_heap_size: usize) {
        let new_card_count = new_heap_size.div_ceil(CARD_SIZE);
        if new_card_count > self.card_count {
            self.cards.resize(new_card_count, 0);
            self.card_count = new_card_count;
            self.biased_ptr = unsafe {
                self.cards.as_mut_ptr().sub(self.heap_start >> CARD_SIZE_LOG2)
            };
        }
    }

    pub fn biased_ptr(&self) -> *mut u8 {
        self.biased_ptr
    }

    pub fn dirty_card_indices(&self) -> &[usize] {
        &self.dirty_card_indices
    }

    pub fn clear(&mut self) {
        for &card_index in &self.dirty_card_indices {
            self.cards[card_index] = 0;
        }
        self.dirty_card_indices.clear();
    }

    pub fn has_dirty_cards(&self) -> bool {
        !self.dirty_card_indices.is_empty()
    }
}

// =============================================================================
// Young Generation Space
// =============================================================================

struct Space<M: MemoryProvider> {
    start: *const u8,
    page_count: usize,
    allocation_offset: usize,
    memory: M,
}

unsafe impl<M: MemoryProvider + Send> Send for Space<M> {}
unsafe impl<M: MemoryProvider + Sync> Sync for Space<M> {}

impl<M: MemoryProvider> Space<M> {
    fn byte_count(&self) -> usize {
        self.page_count * self.memory.page_size()
    }

    fn contains(&self, pointer: *const u8) -> bool {
        let start = self.start as usize;
        let end = start + self.byte_count();
        let pointer = pointer as usize;
        pointer >= start && pointer < end
    }

    fn write_object<T: GcTypes>(&mut self, offset: usize, size_bytes: usize) -> *const u8 {
        let ptr = unsafe { self.start.add(offset) };
        let mut heap_object = T::ObjectHandle::from_untagged(ptr);
        assert!(self.contains(heap_object.get_pointer()));
        heap_object.write_header(size_bytes);
        heap_object.get_pointer()
    }

    fn write_object_zeroed<T: GcTypes>(&mut self, offset: usize, size_bytes: usize) -> *const u8 {
        let ptr = unsafe { self.start.add(offset) };
        let mut heap_object = T::ObjectHandle::from_untagged(ptr);
        assert!(self.contains(heap_object.get_pointer()));

        let header_size = 8;
        let full_size = size_bytes + header_size;
        unsafe {
            core::ptr::write_bytes(self.start.add(offset) as *mut u8, 0, full_size);
        }

        heap_object.write_header(size_bytes);
        heap_object.get_pointer()
    }

    fn allocate<T: GcTypes>(&mut self, size_bytes: usize) -> *const u8 {
        let offset = self.allocation_offset;
        let header_size = 8;
        let full_size = size_bytes + header_size;
        let pointer = self.write_object::<T>(offset, size_bytes);
        self.increment_current_offset(full_size);
        pointer
    }

    fn allocate_zeroed<T: GcTypes>(&mut self, size_bytes: usize) -> *const u8 {
        let offset = self.allocation_offset;
        let header_size = 8;
        let full_size = size_bytes + header_size;
        let pointer = self.write_object_zeroed::<T>(offset, size_bytes);
        self.increment_current_offset(full_size);
        pointer
    }

    fn increment_current_offset(&mut self, size: usize) {
        self.allocation_offset += size;
    }

    fn clear(&mut self) {
        self.allocation_offset = 0;
    }

    fn new(mut memory: M, default_page_count: usize) -> Result<Self, AllocError> {
        let page_size = memory.page_size();
        let total_size = page_size * MAX_PAGE_COUNT;

        let start = memory
            .allocate_region(total_size)
            .ok_or(AllocError::ProviderFailed)?;

        let initial_size = default_page_count * page_size;
        if !memory.commit(start, initial_size) {
            return Err(AllocError::ProviderFailed);
        }

        Ok(Self {
            start: start as *const u8,
            page_count: default_page_count,
            allocation_offset: 0,
            memory,
        })
    }

    fn can_allocate(&self, size_bytes: usize) -> bool {
        let header_size = 8;
        let alloc_size = size_bytes + header_size;
        let new_offset = self.allocation_offset + alloc_size;
        new_offset <= self.byte_count()
    }
}

// =============================================================================
// Generational GC
// =============================================================================

/// Generational garbage collector.
///
/// Uses a young generation for new allocations and an old generation
/// (mark-and-sweep) for long-lived objects. Write barriers track
/// old-to-young pointers using a card table.
///
/// # Type Parameters
/// - `T`: The runtime's type system implementing [`GcTypes`]
/// - `M`: The memory provider implementing [`MemoryProvider`]
///
/// # Requirements
/// - `T::ObjectHandle` must implement [`ForwardingSupport`] for object promotion
pub struct GenerationalGC<T: GcTypes, M: MemoryProvider>
where
    T::ObjectHandle: ForwardingSupport,
{
    young: Space<M>,
    old: MarkAndSweep<T, M>,
    copied: Vec<T::ObjectHandle>,
    gc_count: usize,
    full_gc_frequency: usize,
    atomic_pause: [u8; 8],
    options: AllocatorOptions,
    remembered_set: Vec<usize>,
    card_table: CardTable,
    _phantom: PhantomData<T>,
}

impl<T: GcTypes, M: MemoryProvider> Allocator<T, M> for GenerationalGC<T, M>
where
    T::ObjectHandle: ForwardingSupport,
    M: Clone,
{
    fn new(options: AllocatorOptions, memory: M) -> Self {
        let young = Space::new(memory.clone(), DEFAULT_PAGE_COUNT * 10)
            .expect("Failed to create young generation");
        let old = MarkAndSweep::new_with_page_count(memory, DEFAULT_PAGE_COUNT * 100, options)
            .expect("Failed to create old generation");
        let card_table = CardTable::new(old.heap_start(), old.heap_size());
        Self {
            young,
            old,
            copied: Vec::new(),
            gc_count: 0,
            full_gc_frequency: 100,
            atomic_pause: [0; 8],
            options,
            remembered_set: Vec::with_capacity(64),
            card_table,
            _phantom: PhantomData,
        }
    }

    fn try_allocate(
        &mut self,
        words: usize,
        kind: T::ObjectKind,
    ) -> Result<AllocateAction, AllocError> {
        self.allocate_inner::<false>(words, kind)
    }

    fn try_allocate_zeroed(
        &mut self,
        words: usize,
        kind: T::ObjectKind,
    ) -> Result<AllocateAction, AllocError> {
        self.allocate_inner::<true>(words, kind)
    }

    fn gc(&mut self, roots: &dyn RootProvider<T::TaggedValue>) {
        if !self.options.gc {
            return;
        }
        if self.gc_count != 0 && self.gc_count % self.full_gc_frequency == 0 {
            self.gc_count = 0;
            self.full_gc(roots);
        } else {
            self.minor_gc(roots);
        }
        self.gc_count += 1;
    }

    fn grow(&mut self) {
        self.old.grow();
        self.card_table.resize(self.old.heap_size());
    }

    fn allocate_for_runtime(
        &mut self,
        words: usize,
        kind: T::ObjectKind,
    ) -> Result<T::TaggedValue, AllocError> {
        // Allocate in old generation - runtime objects are long-lived
        match self.old.try_allocate(words, kind)? {
            AllocateAction::Allocated(ptr) => Ok(T::TaggedValue::tag(ptr, kind)),
            AllocateAction::Gc => Err(AllocError::OutOfMemory),
        }
    }

    fn get_allocation_options(&self) -> AllocatorOptions {
        self.options
    }

    fn write_barrier(&mut self, object_ptr: usize, new_value: usize) {
        let new_tagged = T::TaggedValue::from_usize(new_value);
        if !new_tagged.is_heap_pointer() {
            return;
        }

        let new_value_untagged = new_tagged.untag();
        if !self.young.contains(new_value_untagged) {
            return;
        }

        let object_tagged = T::TaggedValue::from_usize(object_ptr);
        if !object_tagged.is_heap_pointer() {
            return;
        }

        let object_untagged = object_tagged.untag();
        if !self.old.contains(object_untagged) {
            return;
        }

        self.card_table.mark_dirty(object_untagged as usize);

        if !self.remembered_set.contains(&object_ptr) {
            self.remembered_set.push(object_ptr);
        }
    }

    fn get_card_table_biased_ptr(&self) -> *mut u8 {
        self.card_table.biased_ptr()
    }

    fn mark_card_unconditional(&mut self, object_ptr: usize) {
        let tagged = T::TaggedValue::from_usize(object_ptr);
        if !tagged.is_heap_pointer() {
            return;
        }

        let object_untagged = tagged.untag();
        if self.old.contains(object_untagged) {
            self.card_table.mark_dirty(object_untagged as usize);
        }
    }

    fn get_young_gen_bounds(&self) -> (usize, usize) {
        let start = self.young.start as usize;
        let end = start + self.young.byte_count();
        (start, end)
    }
}

impl<T: GcTypes, M: MemoryProvider> GenerationalGC<T, M>
where
    T::ObjectHandle: ForwardingSupport,
{
    /// Get a pointer used for thread pause synchronization.
    pub fn get_pause_pointer(&self) -> usize {
        self.atomic_pause.as_ptr() as usize
    }

    fn allocate_inner<const ZEROED: bool>(
        &mut self,
        words: usize,
        _kind: T::ObjectKind,
    ) -> Result<AllocateAction, AllocError> {
        let size_bytes = words * 8;
        if self.young.can_allocate(size_bytes) {
            let ptr = if ZEROED {
                self.young.allocate_zeroed::<T>(size_bytes)
            } else {
                self.young.allocate::<T>(size_bytes)
            };
            Ok(AllocateAction::Allocated(ptr))
        } else {
            Ok(AllocateAction::Gc)
        }
    }

    fn minor_gc(&mut self, roots: &dyn RootProvider<T::TaggedValue>) {
        #[cfg(feature = "std")]
        let start = std::time::Instant::now();
        usdt_probes::fire_gc_minor_start(self.gc_count);

        self.gc_count += 1;

        // Gather roots and separate young vs old
        let mut young_roots: Vec<(usize, T::TaggedValue)> = Vec::new();
        let mut old_gen_objects: Vec<usize> = Vec::new();

        roots.enumerate_roots(&mut |slot_addr, tagged| {
            if !tagged.is_heap_pointer() {
                return;
            }
            let untagged = tagged.untag();
            if self.young.contains(untagged) {
                young_roots.push((slot_addr, tagged));
            } else if self.old.contains(untagged) {
                old_gen_objects.push(tagged.as_usize());
            }
        });

        // Process young roots - copy to old gen
        for (slot_addr, tagged) in &young_roots {
            let new_value = self.copy(tagged.as_usize());
            unsafe {
                *(*slot_addr as *mut usize) = new_value;
            }
        }

        // Process old gen objects found in roots
        for old_root in old_gen_objects {
            self.process_old_gen_object(old_root);
        }

        // Process remembered set
        let remembered = core::mem::take(&mut self.remembered_set);
        for old_object in remembered {
            self.process_old_gen_object(old_object);
        }

        // Process dirty cards
        self.process_dirty_cards();

        // Copy remaining reachable objects
        self.copy_remaining();

        self.young.clear();
        self.card_table.clear();

        usdt_probes::fire_gc_minor_end(self.gc_count);
        #[cfg(feature = "std")]
        if self.options.print_stats {
            println!("Minor gc took {:?}", start.elapsed());
        }
    }

    fn process_dirty_cards(&mut self) {
        if !self.card_table.has_dirty_cards() {
            return;
        }

        // Collect dirty card indices into a set for fast lookup
        let dirty_cards: alloc::collections::BTreeSet<usize> = self
            .card_table
            .dirty_card_indices()
            .iter()
            .copied()
            .collect();

        let old_start = self.old.heap_start();
        let mut objects_to_process: Vec<usize> = Vec::new();

        self.old.walk_objects_mut(|obj_addr, _heap_obj| {
            let card_index = (obj_addr - old_start) >> CARD_SIZE_LOG2;
            if dirty_cards.contains(&card_index) {
                let obj = T::ObjectHandle::from_untagged(obj_addr as *const u8);
                if let Some(kind) = obj.get_object_kind() {
                    let tagged = T::TaggedValue::tag(obj_addr as *const u8, kind);
                    objects_to_process.push(tagged.as_usize());
                }
            }
        });

        for old_object in objects_to_process {
            self.process_old_gen_object(old_object);
        }
    }

    fn process_old_gen_object(&mut self, old_object: usize) {
        let tagged = T::TaggedValue::from_usize(old_object);
        let mut heap_obj = T::ObjectHandle::from_tagged(tagged);

        let fields = heap_obj.get_fields_mut();
        for field in fields.iter_mut() {
            let field_tagged = T::TaggedValue::from_usize(*field);
            if field_tagged.is_heap_pointer() {
                let field_ptr = field_tagged.untag();
                if self.young.contains(field_ptr) {
                    let new_value = self.copy(*field);
                    *field = new_value;
                }
            }
        }
    }

    fn copy(&mut self, root: usize) -> usize {
        let tagged = T::TaggedValue::from_usize(root);
        if !tagged.is_heap_pointer() {
            return root;
        }

        let mut heap_object = T::ObjectHandle::from_tagged(tagged);

        // Skip if not in young gen
        if !self.young.contains(heap_object.get_pointer()) {
            return root;
        }

        // Check if already forwarded
        if heap_object.is_forwarded() {
            return heap_object.get_forwarding_pointer().as_usize();
        }

        // Copy object data to old generation
        let data = heap_object.get_full_object_data();
        let new_pointer = self.old.copy_data_to_offset(data);

        // Get the new object and add to processing queue
        let new_object = T::ObjectHandle::from_untagged(new_pointer);
        self.copied.push(new_object);

        // Store forwarding pointer
        let kind = heap_object.get_object_kind().expect("object must have kind");
        let tagged_new = T::TaggedValue::tag(new_pointer, kind);
        heap_object.set_forwarding_pointer(tagged_new);

        tagged_new.as_usize()
    }

    fn copy_remaining(&mut self) {
        while let Some(mut object) = self.copied.pop() {
            for field in object.get_fields_mut().iter_mut() {
                let field_tagged = T::TaggedValue::from_usize(*field);
                if field_tagged.is_heap_pointer() {
                    let field_ptr = field_tagged.untag();
                    if self.young.contains(field_ptr) {
                        *field = self.copy(*field);
                    }
                }
            }
        }
    }

    fn full_gc(&mut self, roots: &dyn RootProvider<T::TaggedValue>) {
        usdt_probes::fire_gc_full_start(self.gc_count);
        self.minor_gc(roots);
        self.old.gc(roots);
        usdt_probes::fire_gc_full_end(self.gc_count);
    }
}
