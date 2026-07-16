//! The REAL heap (Stage D): objects are raw memory with an 8-byte header,
//! fields inline after it, allocated by atomic bump out of a semi-space.
//!
//! Ported from `gc-rust`'s `gcrust-rt/src/gc/{header,type_info,scan,alloc,
//! semi_space}.rs`, folded to microlang's needs:
//!
//! - ONE header word: `[type_id u16 | spare u16 | aux u32]`. `aux` carries the
//!   varlen length (element count for Values, byte count for Bytes) — no
//!   separate count word. Bit 63 of the header word is the FORWARDING bit;
//!   when set the low 63 bits are the to-space address (so `aux` is capped at
//!   31 bits — asserted at allocation, never a real constraint).
//! - A `TypeInfo` table describes every type's shape ONCE (traced value
//!   fields, raw bytes, varlen tail) and a single generic `scan_object` walks
//!   any object from it — the GC has no per-type code.
//! - The GC is tag-scheme-agnostic through `PtrPolicy` (gc-rust's trait; it IS
//!   microlang's value-model axis — each `Repr` implements it in D2).
//! - Real semi-space: two equal regions, Cheney copy, flip and reuse. Today's
//!   append-and-poison heap becomes the VERIFY mode here: when armed
//!   (`MICROLANG_GC_VERIFY=1`, or any debug build), the collector poisons the
//!   evacuated space so a stale dereference reads a loud out-of-range type_id
//!   instead of silently-recycled memory, and every traced slot is checked to
//!   point at a real header before it is followed.
//! - `AtomicBumpAllocator` + the three-word `AllocWindow` mirror give the JIT
//!   (D5) an inline allocation fast path; `limit = 0` is gc-stress mode
//!   (every allocation takes the out-of-line slow path).
//!
//! Object layout (all offsets from the object start):
//!
//! ```text
//! ┌──────────────────────┐ 0
//! │ header (u64)         │ type_id | spare | aux
//! ├──────────────────────┤ 8
//! │ value fields (u64×n) │ value_field_count slots, GC-traced
//! ├──────────────────────┤ 8 + vfc*8
//! │ raw bytes            │ raw_byte_count bytes, padded to 8
//! ├──────────────────────┤ align8(...)
//! │ varlen tail          │ aux × 8 bytes (Values) or aux bytes (Bytes)
//! └──────────────────────┘
//! ```

use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicU8, AtomicUsize, Ordering};

// ─── Header ──────────────────────────────────────────────────────────────

/// High bit of the header word: the object has been evacuated; the low 63
/// bits hold the to-space address. A valid (non-forwarded) header never has
/// this bit set because `aux` is capped at 31 bits.
pub const FORWARDING_BIT: u64 = 1 << 63;

pub const HEADER_SIZE: usize = 8;

/// Maximum varlen length representable in the header's aux field (31 bits —
/// bit 63 of the word is the forwarding bit).
pub const MAX_AUX: u32 = (1 << 31) - 1;

#[inline(always)]
pub const fn make_header(type_id: u16, spare: u16, aux: u32) -> u64 {
    (type_id as u64) | ((spare as u64) << 16) | ((aux as u64) << 32)
}

#[inline(always)]
pub const fn header_type_id(hdr: u64) -> u16 {
    hdr as u16
}

#[inline(always)]
pub const fn header_spare(hdr: u64) -> u16 {
    (hdr >> 16) as u16
}

#[inline(always)]
pub const fn header_aux(hdr: u64) -> u32 {
    (hdr >> 32) as u32
}

// ─── TypeInfo ────────────────────────────────────────────────────────────

/// Whether a heap object has a variable-length tail section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarLenKind {
    /// Fixed-size object — no tail. `aux` is free per-type metadata.
    None,
    /// Tail of GC-traced value words; `aux` = element count.
    Values,
    /// Tail of untraced bytes (string contents, bignum limbs); `aux` = byte count.
    Bytes,
}

/// Round `n` up to the next multiple of 8.
#[inline(always)]
pub const fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// Describes the shape of a heap object so allocation sizes, field offsets,
/// and GC scan boundaries all come from ONE place.
#[derive(Debug, Clone, Copy)]
pub struct TypeInfo {
    /// Index into the runtime's type table; stored in every object header.
    pub type_id: u16,
    /// Debug name (verify-mode diagnostics).
    pub name: &'static str,
    /// Number of GC-traced value slots (each 8 bytes) after the header.
    pub value_field_count: u16,
    /// Number of untraced raw bytes after the value fields.
    pub raw_byte_count: u16,
    /// Variable-length tail kind (length lives in the header's aux field).
    pub varlen: VarLenKind,
    /// Fixed-size type whose aux field carries per-type METADATA rather than
    /// a varlen length (e.g. a growable-array handle's logical element count).
    pub aux_metadata: bool,
}

impl TypeInfo {
    pub const fn new(type_id: u16, name: &'static str) -> Self {
        TypeInfo {
            type_id,
            name,
            value_field_count: 0,
            raw_byte_count: 0,
            varlen: VarLenKind::None,
            aux_metadata: false,
        }
    }
    pub const fn with_fields(mut self, count: u16) -> Self {
        self.value_field_count = count;
        self
    }
    pub const fn with_raw_bytes(mut self, count: u16) -> Self {
        self.raw_byte_count = count;
        self
    }
    pub const fn with_varlen_values(mut self) -> Self {
        self.varlen = VarLenKind::Values;
        self
    }
    pub const fn with_varlen_bytes(mut self) -> Self {
        self.varlen = VarLenKind::Bytes;
        self
    }

    /// Byte offset of value field `index` from the object start.
    #[inline(always)]
    pub const fn value_field_offset(&self, index: u16) -> usize {
        HEADER_SIZE + (index as usize) * 8
    }

    /// Byte offset of the raw-bytes section from the object start.
    #[inline(always)]
    pub const fn raw_data_offset(&self) -> usize {
        HEADER_SIZE + (self.value_field_count as usize) * 8
    }

    /// Byte offset of the varlen tail from the object start.
    #[inline(always)]
    pub const fn varlen_offset(&self) -> usize {
        align8(self.raw_data_offset() + self.raw_byte_count as usize)
    }

    /// Total allocation size in bytes for an object whose header aux is `aux`.
    /// Always a multiple of 8.
    #[inline(always)]
    pub const fn allocation_size(&self, aux: u32) -> usize {
        match self.varlen {
            VarLenKind::None => self.varlen_offset(),
            VarLenKind::Values => self.varlen_offset() + (aux as usize) * 8,
            VarLenKind::Bytes => align8(self.varlen_offset() + aux as usize),
        }
    }
}

// ─── Gc pointer + object accessors ───────────────────────────────────────

/// A raw pointer to a heap object's header. The `Gc` newtype exists so heap
/// addresses never travel as bare integers inside the runtime; encoding into
/// a tagged value word is the `Repr`'s job (D2).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Gc(pub *mut u8);

impl Gc {
    #[inline(always)]
    pub fn addr(self) -> usize {
        self.0 as usize
    }
    #[inline(always)]
    pub fn from_addr(a: usize) -> Self {
        Gc(a as *mut u8)
    }

    /// Read the header word.
    ///
    /// # Safety
    /// `self` must point at a live heap object.
    #[inline(always)]
    pub unsafe fn header(self) -> u64 {
        unsafe { *(self.0 as *const u64) }
    }
    #[inline(always)]
    pub unsafe fn type_id(self) -> u16 {
        header_type_id(unsafe { self.header() })
    }
    #[inline(always)]
    pub unsafe fn spare(self) -> u16 {
        header_spare(unsafe { self.header() })
    }
    #[inline(always)]
    pub unsafe fn aux(self) -> u32 {
        header_aux(unsafe { self.header() })
    }
    /// Rewrite the header's aux field (a growable handle's logical length).
    ///
    /// # Safety
    /// `self` must point at a live heap object; `aux <= MAX_AUX`. For a
    /// varlen object the new aux must not exceed the allocated tail.
    #[inline(always)]
    pub unsafe fn set_aux(self, aux: u32) {
        debug_assert!(aux <= MAX_AUX);
        unsafe {
            let hdr = self.header();
            *(self.0 as *mut u64) = (hdr & 0xffff_ffff) | ((aux as u64) << 32);
        }
    }
    #[inline(always)]
    pub unsafe fn set_spare(self, spare: u16) {
        unsafe {
            let hdr = self.header();
            *(self.0 as *mut u64) = (hdr & !0xffff_0000) | ((spare as u64) << 16);
        }
    }

    /// Read traced value field `i` (layout per this object's `TypeInfo`;
    /// fields always start right after the header).
    #[inline(always)]
    pub unsafe fn field(self, i: usize) -> u64 {
        unsafe { *(self.0.add(HEADER_SIZE + i * 8) as *const u64) }
    }
    #[inline(always)]
    pub unsafe fn set_field(self, i: usize, v: u64) {
        unsafe { *(self.0.add(HEADER_SIZE + i * 8) as *mut u64) = v }
    }
    /// Value field `i` as an atomic (Atom CAS; frame-published slots).
    #[inline(always)]
    pub unsafe fn field_atomic(self, i: usize) -> &'static AtomicU64 {
        unsafe { &*(self.0.add(HEADER_SIZE + i * 8) as *const AtomicU64) }
    }

    /// Pointer to the raw-bytes section.
    #[inline(always)]
    pub unsafe fn raw_ptr(self, info: &TypeInfo) -> *mut u8 {
        unsafe { self.0.add(info.raw_data_offset()) }
    }
    /// Read a raw u64 at raw-section word `w`.
    #[inline(always)]
    pub unsafe fn raw_word(self, info: &TypeInfo, w: usize) -> u64 {
        unsafe { *(self.raw_ptr(info).add(w * 8) as *const u64) }
    }
    #[inline(always)]
    pub unsafe fn set_raw_word(self, info: &TypeInfo, w: usize, v: u64) {
        unsafe { *(self.raw_ptr(info).add(w * 8) as *mut u64) = v }
    }
    /// Read a raw u32 at the start of the raw section (Char codepoints).
    #[inline(always)]
    pub unsafe fn raw_word_u32(self, info: &TypeInfo) -> u32 {
        unsafe { *(self.raw_ptr(info) as *const u32) }
    }
    #[inline(always)]
    pub unsafe fn set_raw_word_u32(self, info: &TypeInfo, v: u32) {
        unsafe { *(self.raw_ptr(info) as *mut u32) = v }
    }
    /// Read/write an i128 at raw-section byte offset `off` (BigInt, Ratio).
    #[inline(always)]
    pub unsafe fn raw_i128(self, info: &TypeInfo, off: usize) -> i128 {
        unsafe { (self.raw_ptr(info).add(off) as *const i128).read_unaligned() }
    }
    #[inline(always)]
    pub unsafe fn set_raw_i128(self, info: &TypeInfo, off: usize, v: i128) {
        unsafe { (self.raw_ptr(info).add(off) as *mut i128).write_unaligned(v) }
    }

    /// The varlen tail as a value-word slice (`varlen == Values`); length is
    /// the header's aux.
    #[inline(always)]
    pub unsafe fn values(self, info: &TypeInfo) -> &'static [u64] {
        unsafe {
            std::slice::from_raw_parts(self.0.add(info.varlen_offset()) as *const u64, self.aux() as usize)
        }
    }
    #[inline(always)]
    pub unsafe fn values_mut(self, info: &TypeInfo) -> &'static mut [u64] {
        unsafe {
            std::slice::from_raw_parts_mut(self.0.add(info.varlen_offset()) as *mut u64, self.aux() as usize)
        }
    }
    /// The varlen tail as a byte slice (`varlen == Bytes`).
    #[inline(always)]
    pub unsafe fn bytes(self, info: &TypeInfo) -> &'static [u8] {
        unsafe { std::slice::from_raw_parts(self.0.add(info.varlen_offset()), self.aux() as usize) }
    }
    #[inline(always)]
    pub unsafe fn bytes_mut(self, info: &TypeInfo) -> &'static mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(info.varlen_offset()), self.aux() as usize) }
    }
}

// ─── PtrPolicy ───────────────────────────────────────────────────────────

/// How the GC identifies and rewrites heap pointers inside value slots. The
/// collector is tag-scheme-agnostic: it works on raw `u64`s and delegates to
/// this trait — which is exactly microlang's value-model (`Repr`) axis.
pub trait PtrPolicy {
    /// If `bits` encodes a heap reference, the raw object address; `None` for
    /// every immediate (fixnum, bool, nil, sym, float).
    fn try_decode_ptr(bits: u64) -> Option<*mut u8>;
    /// Encode a raw object address back into a tagged value word. Must invert
    /// `try_decode_ptr`.
    fn encode_ptr(ptr: *mut u8) -> u64;
}

// ─── scan_object ─────────────────────────────────────────────────────────

/// Call `visitor` on every GC-traceable slot of the object: the fixed value
/// fields, then the varlen Values tail (length from the header's aux). This
/// is the ONLY object walker — there is no per-type GC code.
///
/// # Safety
/// `obj` must point at a live object whose shape matches `info`.
#[inline]
pub unsafe fn scan_object(obj: Gc, info: &TypeInfo, mut visitor: impl FnMut(*mut u64)) {
    unsafe {
        for i in 0..info.value_field_count as usize {
            visitor(obj.0.add(HEADER_SIZE + i * 8) as *mut u64);
        }
        if info.varlen == VarLenKind::Values {
            let n = obj.aux() as usize;
            let base = obj.0.add(info.varlen_offset());
            for i in 0..n {
                visitor(base.add(i * 8) as *mut u64);
            }
        }
    }
}

// ─── AtomicBumpAllocator (one space) ─────────────────────────────────────

/// One bump region: a fixed range of zero-initialized memory and an atomic
/// cursor. Allocation is a CAS bump — no malloc, no lock. The region is
/// reserved up front and committed lazily by the OS as it is first touched,
/// so a large virtual size costs nothing until used.
pub struct AtomicBumpAllocator {
    base: *mut u8,
    cursor: AtomicUsize,
    size: usize,
}

// SAFETY: base is immutable after construction; the cursor is atomic. Object
// initialization races are governed by the runtime's existing heap discipline
// (allocation returns exclusively-owned memory; publication is via tagged
// value words with release/acquire slot ordering).
unsafe impl Send for AtomicBumpAllocator {}
unsafe impl Sync for AtomicBumpAllocator {}

impl AtomicBumpAllocator {
    pub fn new(size: usize) -> Self {
        let size = align8(size);
        // Alignment 8 (all an object needs), NOT the page size: keeping the
        // align at or under the allocator's MIN_ALIGN routes `alloc_zeroed`
        // to `calloc`, whose large-allocation path is fresh mmap'd zero pages
        // — reserved up front, COMMITTED LAZILY as touched. A larger align
        // takes Rust's aligned-alloc + eager-memset path instead, which would
        // commit (and zero) the whole space at construction — gigabytes per
        // Runtime the design promises never to touch.
        let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null(), "heap: reserving a {size}-byte space failed");
        AtomicBumpAllocator { base, cursor: AtomicUsize::new(0), size }
    }

    #[inline(always)]
    pub fn base(&self) -> *mut u8 {
        self.base
    }
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }
    #[inline(always)]
    pub fn used(&self) -> usize {
        // The JIT's inline fast path (D5) bumps with fetch_add and can
        // overshoot when it loses the race for the last bytes; clamp so
        // walkers never see a cursor past the space.
        self.cursor.load(Ordering::Acquire).min(self.size)
    }
    #[inline(always)]
    pub fn remaining(&self) -> usize {
        self.size - self.used()
    }
    #[inline(always)]
    pub fn contains(&self, ptr: *const u8) -> bool {
        let a = ptr as usize;
        let b = self.base as usize;
        a >= b && a < b + self.size
    }
    /// Reset the cursor (after evacuation) and re-zero the previously used
    /// prefix so the next fill starts from zeroed memory (bump allocation
    /// relies on it for cheap object init). In verify mode the caller poisons
    /// instead; see `Heap::collect`.
    pub fn reset_zeroed(&self) {
        let used = self.used();
        unsafe { std::ptr::write_bytes(self.base, 0, used) };
        self.cursor.store(0, Ordering::Release);
    }
    /// Reset the cursor, filling the previously used prefix with the POISON
    /// pattern (verify mode): any stale read sees type_id 0x5A5A — far out of
    /// table range — and panics loudly instead of touching recycled memory.
    pub fn reset_poisoned(&self) {
        let used = self.used();
        unsafe { std::ptr::write_bytes(self.base, POISON_BYTE, used) };
        self.cursor.store(0, Ordering::Release);
    }

    /// Bump-allocate `size` bytes (already 8-aligned). Returns null when the
    /// space is exhausted. Memory is NOT re-zeroed here: the space is zeroed
    /// at construction and re-zeroed by `reset_zeroed`, so a fresh claim is
    /// always already zero (poisoned resets re-zero eagerly too — see
    /// `Heap::collect`).
    #[inline]
    pub fn alloc_raw(&self, size: usize) -> *mut u8 {
        debug_assert_eq!(size & 7, 0);
        let mut cur = self.cursor.load(Ordering::Relaxed);
        loop {
            let end = cur + size;
            if end > self.size {
                return std::ptr::null_mut();
            }
            match self.cursor.compare_exchange_weak(cur, end, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => return unsafe { self.base.add(cur) },
                Err(c) => cur = c,
            }
        }
    }
}

impl Drop for AtomicBumpAllocator {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.size, 8).unwrap();
        unsafe { std::alloc::dealloc(self.base, layout) };
    }
}

/// Verify-mode poison byte. 0x5A keeps bit 63 of a poisoned header word CLEAR
/// (so a stale pointer is not mistaken for a forwarded one) while producing a
/// type_id (0x5A5A) no real table contains.
pub const POISON_BYTE: u8 = 0x5A;

// ─── AllocWindow (JIT-facing mirror, D5) ─────────────────────────────────

/// The three words the JIT's inline allocation fast path reads, mirrored from
/// the active from-space and updated only under STW (at space flips). Layout
/// is ABI: emitted code reads the fields at fixed offsets.
///
///   offset 0: cursor — pointer to the active space's atomic cursor word
///   offset 8: base   — the active space's base address
///   offset 16: limit — allocation limit in bytes; 0 forces EVERY allocation
///             through the out-of-line slow path (gc-stress mode)
#[repr(C)]
pub struct AllocWindow {
    pub cursor: AtomicPtr<u8>,
    pub base: AtomicPtr<u8>,
    pub limit: AtomicUsize,
}

impl AllocWindow {
    pub fn empty() -> Self {
        AllocWindow {
            cursor: AtomicPtr::new(std::ptr::null_mut()),
            base: AtomicPtr::new(std::ptr::null_mut()),
            limit: AtomicUsize::new(0),
        }
    }
    /// Point the window at `space` (at construction and at every flip, both
    /// under STW or before any mutator runs).
    pub fn point_at(&self, space: &AtomicBumpAllocator, limit: usize) {
        self.cursor
            .store(&space.cursor as *const AtomicUsize as *mut u8, Ordering::Release);
        self.base.store(space.base, Ordering::Release);
        self.limit.store(limit, Ordering::Release);
    }
}

// ─── The heap: two spaces + Cheney evacuation ────────────────────────────

/// Default per-space size in MiB when `MICROLANG_HEAP_MB` is unset. Virtual
/// reservation (lazily committed), so big is cheap; this is a CAP, and
/// exhausting it is a loud panic (allocation never triggers GC here — the
/// runtime's GC is explicit/safepoint-driven, unchanged from before).
const DEFAULT_SPACE_MB: usize = 4096;

/// Bits of the ONE cheap poll word (`Heap::poll`) every tier checks at its
/// safepoints and the JIT polls from emitted code (Stage E). Bit 0 mirrors the
/// runtime's `gc_requested` (a sibling wants to collect: park). Bit 1 is
/// allocation PRESSURE: the bump cursor crossed the soft threshold, so the
/// next safepoint should trigger a collection itself.
pub const POLL_REQUESTED: u8 = 1;
pub const POLL_PRESSURE: u8 = 2;

/// The Stage-D heap: two equal bump spaces, explicit Cheney evacuation, flip
/// and reuse. Shared across mutator threads (allocation is an atomic bump);
/// `collect` must run under the runtime's existing STW discipline (all other
/// mutators parked, heap_lock held).
pub struct Heap {
    spaces: [AtomicBumpAllocator; 2],
    /// Index of the active (from) space. Flipped only under STW.
    active: AtomicUsize,
    /// Armed: poison evacuated space, check every traced slot's target header.
    verify: bool,
    pub collections: AtomicU64,
    /// Live bytes copied by the last collection.
    pub last_live_bytes: AtomicUsize,
    /// The JIT's inline-allocation mirror (D5); re-pointed at every flip.
    pub window: AllocWindow,
    /// The safepoint poll word (`POLL_*` bits). Allocation sets PRESSURE when
    /// it crosses `soft_limit`; the STW rendezvous mirrors REQUESTED here so
    /// one byte answers "should this safepoint do anything".
    pub poll: AtomicU8,
    /// Soft trigger in bytes (`MICROLANG_GC_TRIGGER_PCT` of a space, default
    /// 50%). Atomic so tests can lower it on a live heap. Allocation NEVER
    /// collects and never fails before the hard wall — crossing this only
    /// raises the pressure bit for the next safepoint.
    soft_limit: AtomicUsize,
    /// gc-stress (`MICROLANG_GC_STRESS=1`): the pressure bit is permanently
    /// set, so every safepoint collects. The bug hammer.
    stress: bool,
}

fn verify_armed_default() -> bool {
    if let Ok(v) = std::env::var("MICROLANG_GC_VERIFY") {
        return v != "0" && !v.is_empty();
    }
    cfg!(debug_assertions)
}

impl Heap {
    pub fn new() -> Self {
        let mb = std::env::var("MICROLANG_HEAP_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SPACE_MB);
        Self::with_space_size(mb << 20)
    }

    pub fn with_space_size(bytes: usize) -> Self {
        let pct = std::env::var("MICROLANG_GC_TRIGGER_PCT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|p| (1..=100).contains(p))
            .unwrap_or(50);
        let stress = std::env::var("MICROLANG_GC_STRESS").is_ok_and(|v| v != "0" && !v.is_empty());
        // The window stays EMPTY (limit 0 = closed, every inline allocation
        // takes the slow path) until `arm_window` — the cursor mirror is a
        // pointer INTO this struct's inline spaces, so arming here would
        // dangle the moment the constructed Heap is moved to its final home.
        Heap {
            spaces: [AtomicBumpAllocator::new(bytes), AtomicBumpAllocator::new(bytes)],
            active: AtomicUsize::new(0),
            verify: verify_armed_default(),
            collections: AtomicU64::new(0),
            last_live_bytes: AtomicUsize::new(0),
            window: AllocWindow::empty(),
            poll: AtomicU8::new(if stress { POLL_PRESSURE } else { 0 }),
            soft_limit: AtomicUsize::new(bytes / 100 * pct),
            stress,
        }
    }

    /// Open the JIT's inline-allocation window. Must be called (once) AFTER
    /// the heap has reached its final address — `self`'s spaces are inline, so
    /// the window's cursor pointer is only stable from then on. (Collections
    /// re-point it at every flip.)
    pub fn arm_window(&self) {
        self.window.point_at(self.from_space(), self.window_limit());
    }

    /// The window's allocation limit: the SOFT trigger, not the space size —
    /// an inline allocation crossing it falls to the out-of-line shim, which
    /// raises the pressure bit (Stage E). That is how JIT-allocated garbage
    /// drives collections without any extra emitted code per allocation.
    fn window_limit(&self) -> usize {
        self.soft_limit.load(Ordering::Relaxed).min(self.from_space().size())
    }

    #[inline(always)]
    pub fn from_space(&self) -> &AtomicBumpAllocator {
        &self.spaces[self.active.load(Ordering::Acquire)]
    }
    #[inline(always)]
    fn to_space(&self) -> &AtomicBumpAllocator {
        &self.spaces[1 - self.active.load(Ordering::Acquire)]
    }

    pub fn used(&self) -> usize {
        self.from_space().used()
    }
    pub fn verify_armed(&self) -> bool {
        self.verify
    }
    /// Arm/disarm verify mode (tests; the CLI leaves the env-var default).
    pub fn set_verify(&mut self, on: bool) {
        self.verify = on;
    }

    /// Does `ptr` lie in the active space? (Verify-mode stale-pointer check.)
    pub fn contains(&self, ptr: *const u8) -> bool {
        self.from_space().contains(ptr)
    }

    /// Allocate an object of type `info` with varlen length `aux`; the header
    /// is written, everything else is zero. Panics loudly on exhaustion —
    /// allocation NEVER triggers a collection (the runtime's GC is explicit /
    /// safepoint-driven; stack maps are future work).
    #[inline]
    pub fn alloc(&self, info: &TypeInfo, aux: u32) -> Gc {
        assert!(aux <= MAX_AUX, "heap: varlen length {aux} exceeds MAX_AUX");
        debug_assert!(
            (info.varlen != VarLenKind::None) || aux == 0 || info.aux_is_metadata(),
            "heap: aux={aux} on fixed-size type {} (set aux_is_metadata in TypeInfo if intended)",
            info.name
        );
        let size = info.allocation_size(aux);
        let space = self.from_space();
        let p = space.alloc_raw(size);
        if p.is_null() {
            panic!(
                "heap space exhausted ({} MiB used of {} MiB): raise MICROLANG_HEAP_MB \
                 (allocation never collects; collections run at safepoints)",
                space.used() >> 20,
                space.size() >> 20
            );
        }
        // Allocation-driven collection (Stage E): crossing the soft threshold
        // raises the PRESSURE bit; the NEXT safepoint collects. Never here.
        if space.used() > self.soft_limit.load(Ordering::Relaxed)
            && self.poll.load(Ordering::Relaxed) & POLL_PRESSURE == 0
        {
            self.poll.fetch_or(POLL_PRESSURE, Ordering::Relaxed);
        }
        unsafe { *(p as *mut u64) = make_header(info.type_id, 0, aux) };
        Gc(p)
    }

    /// Lower/raise the soft pressure trigger (tests drive collections without
    /// filling a multi-GiB space). Bytes, per space. Re-gates the inline
    /// allocation window at the new trigger too.
    pub fn set_trigger_bytes(&self, bytes: usize) {
        self.soft_limit.store(bytes, Ordering::Relaxed);
        self.window.limit.store(self.window_limit(), Ordering::Release);
    }

    /// Cheney evacuation. `roots` must enumerate EVERY live root slot (the
    /// runtime's shadow stacks, globals, consts, frames, published mutator
    /// roots …); each slot is rewritten in place when its target moves.
    ///
    /// # Safety
    /// Stop-the-world: no other thread may read or mutate the heap during
    /// this call. Every object in from-space must have a valid header.
    pub unsafe fn collect<P: PtrPolicy>(
        &self,
        types: &[TypeInfo],
        enumerate_roots: &mut dyn FnMut(&mut dyn FnMut(*mut u64)),
    ) {
        let from = self.from_space();
        let to = self.to_space();
        debug_assert_eq!(to.used(), 0, "to-space not empty at collection start");

        // Phase 1: forward the roots.
        enumerate_roots(&mut |slot| unsafe { self.process_slot::<P>(types, from, to, slot) });

        // Phase 2: Cheney scan — walk to-space linearly; copying appends.
        let mut scan = 0usize;
        while scan < to.used() {
            let obj = Gc(unsafe { to.base().add(scan) });
            let hdr = unsafe { obj.header() };
            debug_assert_eq!(hdr & FORWARDING_BIT, 0, "forwarded header in to-space");
            let tid = header_type_id(hdr) as usize;
            assert!(
                tid != 0 && tid < types.len(),
                "GC: to-space object at {:p} has type_id {tid} out of range — heap corruption",
                obj.0
            );
            let info = &types[tid];
            unsafe {
                scan_object(obj, info, |slot| self.process_slot::<P>(types, from, to, slot));
            }
            scan += info.allocation_size(header_aux(hdr));
        }

        // Phase 3: flip. The old from-space is re-zeroed (or poisoned when
        // verify is armed) and becomes the next to-space.
        self.last_live_bytes.store(to.used(), Ordering::Relaxed);
        if self.verify {
            from.reset_poisoned();
            // A poisoned space must still hand out ZEROED memory when it next
            // becomes from-space; re-zero now (still under STW, so this is
            // simply eager instead of lazy).
            from.reset_zeroed();
        } else {
            from.reset_zeroed();
        }
        self.active.store(1 - self.active.load(Ordering::Acquire), Ordering::Release);
        self.collections.fetch_add(1, Ordering::Relaxed);
        self.window.point_at(self.from_space(), self.window_limit());
        // Pressure is spent: this collection was the response. It re-arms when
        // an allocation crosses the soft threshold again (immediately, if the
        // live set alone exceeds it — the heap is genuinely tight then). In
        // stress mode the bit stays up so EVERY safepoint keeps collecting.
        if !self.stress {
            self.poll.fetch_and(!POLL_PRESSURE, Ordering::Relaxed);
        }
    }

    /// Forward one traced slot: if it holds a pointer into from-space, copy
    /// or follow the target and rewrite the slot.
    unsafe fn process_slot<P: PtrPolicy>(
        &self,
        types: &[TypeInfo],
        from: &AtomicBumpAllocator,
        to: &AtomicBumpAllocator,
        slot: *mut u64,
    ) {
        let bits = unsafe { *slot };
        if let Some(ptr) = P::try_decode_ptr(bits) {
            if from.contains(ptr) {
                let new = unsafe { self.copy_or_forward(types, to, ptr) };
                unsafe { *slot = P::encode_ptr(new) };
            } else if self.verify && !to.contains(ptr) {
                // A traced slot pointing at neither space is a stale pointer
                // from a previous cycle (or a scalar that decoded as a ref).
                panic!(
                    "GC verify: traced slot holds {bits:#x} -> {ptr:p}, outside both spaces \
                     (stale pointer or non-pointer decoded as a ref)"
                );
            }
        }
    }

    /// Copy `old` to to-space (or return its existing forwarding target).
    unsafe fn copy_or_forward(&self, types: &[TypeInfo], to: &AtomicBumpAllocator, old: *mut u8) -> *mut u8 {
        let hdr = unsafe { *(old as *const u64) };
        if hdr & FORWARDING_BIT != 0 {
            return (hdr & !FORWARDING_BIT) as *mut u8;
        }
        let tid = header_type_id(hdr) as usize;
        // PRECISE-LAYOUT DETECTOR (gc-rust's armed pattern): a traced slot
        // pointing into from-space always targets a real object; an
        // out-of-range type_id means a scalar leaked into a traced slot.
        // Loud panic, never a silent conservative skip.
        assert!(
            tid != 0 && tid < types.len(),
            "GC precise-layout violation: traced slot points at {old:p} whose header \
             type_id={tid} is out of range (table len {}). A non-pointer reached a traced slot.",
            types.len()
        );
        let info = &types[tid];
        let size = info.allocation_size(header_aux(hdr));
        let new = to.alloc_raw(size);
        assert!(!new.is_null(), "to-space exhausted during collection (live set exceeds space)");
        unsafe {
            std::ptr::copy_nonoverlapping(old, new, size);
            *(old as *mut u64) = (new as u64) | FORWARDING_BIT;
        }
        new
    }

    /// Walk every live object in the active space: `visitor(obj, info)`.
    ///
    /// # Safety
    /// All objects must have valid headers; no concurrent mutation.
    pub unsafe fn walk(&self, types: &[TypeInfo], visitor: &mut dyn FnMut(Gc, &TypeInfo)) {
        let space = self.from_space();
        let mut off = 0usize;
        let used = space.used();
        while off < used {
            let obj = Gc(unsafe { space.base().add(off) });
            let hdr = unsafe { obj.header() };
            let tid = header_type_id(hdr) as usize;
            assert!(tid != 0 && tid < types.len(), "heap walk: bad type_id {tid} at offset {off}");
            let info = &types[tid];
            visitor(obj, info);
            off += info.allocation_size(header_aux(hdr));
        }
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeInfo {
    /// Fixed-size types whose aux field carries per-type metadata rather than
    /// a varlen length (a growable-array HANDLE's logical element count lives
    /// in aux while the elements live in a separate data blob).
    pub const fn aux_is_metadata(&self) -> bool {
        self.aux_metadata
    }
    pub const fn with_aux_metadata(mut self) -> Self {
        self.aux_metadata = true;
        self
    }
}

// ─── Closure object ABI ──────────────────────────────────────────────────
// The JIT's emitted call sequence reads these offsets directly (tag check →
// header type_id check → meta arity check → code-word call), which is what
// retires the old heap-id-keyed fast-target table. Keep in sync with
// `kind::CLOSURE`'s TypeInfo (raw16 + varlen values).

/// Byte offset of the closure META word: `[template u32 | nparams u16 |
/// nslots u15 | variadic bit63]`.
pub const CLOSURE_META_OFF: usize = 8;
/// Byte offset of the closure's native CODE pointer (0 = not yet compiled —
/// callers take the slow/shim path).
pub const CLOSURE_CODE_OFF: usize = 16;
/// Byte offset of the inline capture array (varlen values; length = aux).
pub const CLOSURE_CAPS_OFF: usize = 24;
/// Byte offset of a MultiFn's fixed-clause table, indexed by arity (varlen
/// values; length = header aux): `[hdr | variadic closure | raw8 vmin |
/// clauses…]`. The JIT's call sites select the clause for a compile-time
/// arity with one bounds check + one load.
pub const MULTIFN_FIXED_OFF: usize = 24;
/// Byte offset of a RECORD's inline field array (varlen values; length = header
/// aux): `[hdr | raw8 type sym | fields…]`. The JIT's inline `(field r i)` arm
/// indexes off this. Keep in sync with `kind::RECORD`'s TypeInfo (raw8 + varlen
/// values) — `record_fields_off_matches_type_info` pins it.
pub const RECORD_FIELDS_OFF: usize = 16;

pub const META_VARIADIC_BIT: u64 = 1 << 63;
const META_NSLOTS_SHIFT: u64 = 48;
const META_NPARAMS_SHIFT: u64 = 32;

#[inline(always)]
pub fn closure_meta(template: u32, nparams: u16, nslots: u16, variadic: bool) -> u64 {
    assert!(nslots < (1 << 15), "closure nslots {nslots} exceeds the 15-bit meta field");
    (template as u64)
        | ((nparams as u64) << META_NPARAMS_SHIFT)
        | ((nslots as u64) << META_NSLOTS_SHIFT)
        | if variadic { META_VARIADIC_BIT } else { 0 }
}
#[inline(always)]
pub const fn meta_template(m: u64) -> u32 {
    m as u32
}
#[inline(always)]
pub const fn meta_nparams(m: u64) -> usize {
    ((m >> META_NPARAMS_SHIFT) & 0xffff) as usize
}
#[inline(always)]
pub const fn meta_nslots(m: u64) -> u16 {
    ((m >> META_NSLOTS_SHIFT) & 0x7fff) as u16
}
#[inline(always)]
pub const fn meta_variadic(m: u64) -> bool {
    m & META_VARIADIC_BIT != 0
}

// ─── microlang's type table (kinds) ──────────────────────────────────────

/// The concrete object kinds of the microlang runtime, as header type ids.
/// Type id 0 is RESERVED-INVALID so zeroed or stale memory read as an object
/// panics loudly instead of masquerading as a real type.
pub mod kind {
    pub const INVALID: u16 = 0;
    /// `[hdr | head | tail]`
    pub const CONS: u16 = 1;
    /// `[hdr]` — the `()` singleton's shape (still a real allocation so its
    /// identity is an address like everything else).
    pub const EMPTY_LIST: u16 = 2;
    /// `[hdr | bytes…]` aux = byte length (UTF-8).
    pub const STR: u16 = 3;
    /// `[hdr | raw4 codepoint]`
    pub const CHAR: u16 = 4;
    /// Growable array HANDLE: `[hdr(aux = logical len) | dataref]` — identity
    /// lives here; growth allocates a new DATA blob (the ArrayList shape).
    pub const ARRAY: u16 = 5;
    /// Array DATA blob: `[hdr | values…]` aux = capacity. Slots past the
    /// handle's logical len are zero (never decoded as refs by any Repr).
    pub const ARRAY_DATA: u16 = 6;
    /// `(values …)` packet: `[hdr | values…]` aux = count.
    pub const VALUES: u16 = 7;
    /// Promoted i128: `[hdr | raw16]`.
    pub const BIGINT: u16 = 8;
    /// Arbitrary precision: `[hdr | raw8 sign | limb bytes…]` aux = limb bytes.
    pub const HUGEINT: u16 = 9;
    /// Exact rational: `[hdr | raw32 (num i128, den i128)]`.
    pub const RATIO: u16 = 10;
    /// Boxed f64: `[hdr | raw8]`.
    pub const BOXFLOAT: u16 = 11;
    /// Flat closure: `[hdr(aux = ncaps) | raw16: (template_id u32 | nparams
    /// u16 | nslots u16), code ptr | cap values…]`. spare bit 0 = variadic.
    /// The code pointer lives IN the object — the call site loads it directly
    /// (this retires the fast-target side table).
    pub const CLOSURE: u16 = 12;
    /// Per-arity multifn: `[hdr(aux = nfixed) | variadic closure (or 0) |
    /// raw8 variadic-min (u64::MAX = none) | fixed arity closures…]`.
    pub const MULTIFN: u16 = 13;
    /// User record: `[hdr(aux = nfields) | raw8: type Sym | field values…]`.
    pub const RECORD: u16 = 14;
    /// Escape continuation: `[hdr | raw8 tag]`.
    pub const ESCAPE: u16 = 15;
    /// Full continuation: `[hdr | raw8: kont-registry index]` (CEK Konts are
    /// execution-machine state, never heap data; the registry is the root).
    pub const CONT: u16 = 16;
    /// Delimited continuation: same shape as CONT.
    pub const PARTIAL_CONT: u16 = 17;
    /// Atom: `[hdr | slot]` — CAS on the slot word (STW keeps moving safe).
    pub const ATOM: u16 = 18;
    /// Future: `[hdr | raw8: future-registry index]` (join handle + cached
    /// result live in the registry, an OS-resource table).
    pub const FUTURE: u16 = 19;

    pub const COUNT: usize = 20;
}

/// Build microlang's `TypeInfo` table, indexed by `kind::*`.
pub fn type_table() -> Vec<TypeInfo> {
    use kind::*;
    let mut t = vec![TypeInfo::new(INVALID, "INVALID"); COUNT];
    let mut set = |ti: TypeInfo| t[ti.type_id as usize] = ti;
    set(TypeInfo::new(CONS, "cons").with_fields(2));
    set(TypeInfo::new(EMPTY_LIST, "empty-list"));
    set(TypeInfo::new(STR, "string").with_varlen_bytes());
    set(TypeInfo::new(CHAR, "char").with_raw_bytes(4));
    set(TypeInfo::new(ARRAY, "array").with_fields(1).with_aux_metadata());
    set(TypeInfo::new(ARRAY_DATA, "array-data").with_varlen_values());
    set(TypeInfo::new(VALUES, "values").with_varlen_values());
    set(TypeInfo::new(BIGINT, "bigint").with_raw_bytes(16));
    set(TypeInfo::new(HUGEINT, "hugeint").with_raw_bytes(8).with_varlen_bytes());
    set(TypeInfo::new(RATIO, "ratio").with_raw_bytes(32));
    set(TypeInfo::new(BOXFLOAT, "boxfloat").with_raw_bytes(8));
    set(TypeInfo::new(CLOSURE, "closure").with_raw_bytes(16).with_varlen_values());
    set(TypeInfo::new(MULTIFN, "multifn").with_fields(1).with_raw_bytes(8).with_varlen_values());
    set(TypeInfo::new(RECORD, "record").with_raw_bytes(8).with_varlen_values());
    set(TypeInfo::new(ESCAPE, "escape").with_raw_bytes(8));
    set(TypeInfo::new(CONT, "cont").with_raw_bytes(8));
    set(TypeInfo::new(PARTIAL_CONT, "partial-cont").with_raw_bytes(8));
    set(TypeInfo::new(ATOM, "atom").with_fields(1));
    set(TypeInfo::new(FUTURE, "future").with_raw_bytes(8));
    t
}

// ─── tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The test tag scheme: refs are `addr | 0b001` (LowBit's shape — real
    /// addresses are 8-aligned so the low 3 bits are free).
    struct TestPolicy;
    impl PtrPolicy for TestPolicy {
        fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
            if bits & 0b111 == 0b001 {
                Some((bits & !0b111) as *mut u8)
            } else {
                None
            }
        }
        fn encode_ptr(ptr: *mut u8) -> u64 {
            ptr as u64 | 0b001
        }
    }
    fn enc(g: Gc) -> u64 {
        TestPolicy::encode_ptr(g.0)
    }
    fn dec(bits: u64) -> Gc {
        Gc(TestPolicy::try_decode_ptr(bits).expect("not a ref"))
    }
    fn imm(i: u64) -> u64 {
        i << 3 // tag 0b000 = test immediate
    }

    fn small_heap() -> (Heap, Vec<TypeInfo>) {
        let mut h = Heap::with_space_size(1 << 16);
        h.set_verify(true);
        (h, type_table())
    }

    #[test]
    fn header_roundtrip() {
        let h = make_header(kind::STR, 7, 12345);
        assert_eq!(header_type_id(h), kind::STR);
        assert_eq!(header_spare(h), 7);
        assert_eq!(header_aux(h), 12345);
        assert_eq!(h & FORWARDING_BIT, 0);
        // aux at MAX never sets the forwarding bit
        let h = make_header(u16::MAX, u16::MAX, MAX_AUX);
        assert_eq!(h & FORWARDING_BIT, 0);
        assert_eq!(header_aux(h), MAX_AUX);
    }

    #[test]
    fn type_info_layout() {
        let t = type_table();
        let cons = &t[kind::CONS as usize];
        assert_eq!(cons.value_field_offset(0), 8);
        assert_eq!(cons.value_field_offset(1), 16);
        assert_eq!(cons.allocation_size(0), 24);

        let s = &t[kind::STR as usize];
        assert_eq!(s.varlen_offset(), 8);
        assert_eq!(s.allocation_size(0), 8);
        assert_eq!(s.allocation_size(1), 16); // 1 byte pads to 8
        assert_eq!(s.allocation_size(8), 16);
        assert_eq!(s.allocation_size(9), 24);

        let clo = &t[kind::CLOSURE as usize];
        assert_eq!(clo.raw_data_offset(), 8); // no traced fixed fields
        assert_eq!(clo.varlen_offset(), 24); // hdr + raw16
        assert_eq!(clo.allocation_size(3), 24 + 24);

        let mf = &t[kind::MULTIFN as usize];
        assert_eq!(mf.value_field_offset(0), 8);
        assert_eq!(mf.raw_data_offset(), 16);
        assert_eq!(mf.varlen_offset(), 24);

        let rec = &t[kind::RECORD as usize];
        assert_eq!(rec.varlen_offset(), 16);
        assert_eq!(rec.allocation_size(2), 32);
    }

    /// The JIT's inline `(field r i)` arm bakes `RECORD_FIELDS_OFF` as an
    /// immediate; if RECORD's TypeInfo ever grows a fixed field or a wider raw
    /// section, the emitted load would read the wrong word. Pin them together.
    #[test]
    fn record_fields_off_matches_type_info() {
        let t = type_table();
        assert_eq!(t[kind::RECORD as usize].varlen_offset(), RECORD_FIELDS_OFF);
    }

    #[test]
    fn alloc_writes_header_and_zeroes() {
        let (h, t) = small_heap();
        let c = h.alloc(&t[kind::CONS as usize], 0);
        unsafe {
            assert_eq!(c.type_id(), kind::CONS);
            assert_eq!(c.aux(), 0);
            assert_eq!(c.field(0), 0);
            assert_eq!(c.field(1), 0);
            c.set_field(0, imm(41));
            c.set_field(1, imm(42));
            assert_eq!(c.field(0), imm(41));
            assert_eq!(c.field(1), imm(42));
        }
        // 8-aligned addresses, always.
        assert_eq!(c.addr() & 7, 0);
    }

    #[test]
    fn str_bytes_roundtrip() {
        let (h, t) = small_heap();
        let info = &t[kind::STR as usize];
        let s = h.alloc(info, 5);
        unsafe {
            s.bytes_mut(info).copy_from_slice(b"hello");
            assert_eq!(s.bytes(info), b"hello");
            assert_eq!(s.aux(), 5);
        }
    }

    #[test]
    fn scan_visits_fields_and_varlen() {
        let (h, t) = small_heap();
        let info = &t[kind::MULTIFN as usize];
        let m = h.alloc(info, 3);
        unsafe {
            m.set_field(0, imm(1));
            m.values_mut(info).copy_from_slice(&[imm(2), imm(3), imm(4)]);
            let mut seen = Vec::new();
            scan_object(m, info, |slot| seen.push(*slot));
            assert_eq!(seen, vec![imm(1), imm(2), imm(3), imm(4)]);
        }
        // Bytes tails are NOT scanned.
        let sinfo = &t[kind::STR as usize];
        let s = h.alloc(sinfo, 16);
        let mut n = 0;
        unsafe { scan_object(s, sinfo, |_| n += 1) };
        assert_eq!(n, 0);
    }

    #[test]
    fn collect_preserves_live_graph_and_reclaims_garbage() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        // list (1 2 3) with nil-ish immediate terminator
        let mut tail = imm(0);
        for i in (1..=3).rev() {
            let c = h.alloc(cons, 0);
            unsafe {
                c.set_field(0, imm(i));
                c.set_field(1, tail);
            }
            tail = enc(c);
        }
        // plus garbage
        for _ in 0..100 {
            h.alloc(cons, 0);
        }
        let used_before = h.used();
        let mut root = tail;
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        assert_ne!(root, tail, "root should be rewritten to the new address");
        // Live set: exactly 3 cons cells.
        assert_eq!(h.used(), 3 * cons.allocation_size(0));
        assert!(h.used() < used_before);
        // Graph intact.
        unsafe {
            let mut cur = root;
            for i in 1..=3 {
                let c = dec(cur);
                assert_eq!(c.type_id(), kind::CONS);
                assert_eq!(c.field(0), imm(i));
                cur = c.field(1);
            }
            assert_eq!(cur, imm(0));
        }
        assert_eq!(h.collections.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn shared_object_copied_once() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let shared = h.alloc(cons, 0);
        unsafe { shared.set_field(0, imm(9)) };
        let a = h.alloc(cons, 0);
        let b = h.alloc(cons, 0);
        unsafe {
            a.set_field(0, enc(shared));
            b.set_field(0, enc(shared));
        }
        let mut ra = enc(a);
        let mut rb = enc(b);
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| {
                visit(&mut ra as *mut u64);
                visit(&mut rb as *mut u64);
            });
        }
        unsafe {
            let sa = dec(dec(ra).field(0));
            let sb = dec(dec(rb).field(0));
            assert_eq!(sa, sb, "shared child must be copied exactly once");
            assert_eq!(sa.field(0), imm(9));
        }
        assert_eq!(h.used(), 3 * cons.allocation_size(0));
    }

    #[test]
    fn cycles_survive() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let a = h.alloc(cons, 0);
        let b = h.alloc(cons, 0);
        unsafe {
            a.set_field(0, imm(1));
            a.set_field(1, enc(b));
            b.set_field(0, imm(2));
            b.set_field(1, enc(a)); // cycle
        }
        let mut root = enc(a);
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        unsafe {
            let na = dec(root);
            let nb = dec(na.field(1));
            assert_eq!(na.field(0), imm(1));
            assert_eq!(nb.field(0), imm(2));
            assert_eq!(dec(nb.field(1)), na, "cycle closes back on the relocated a");
        }
        assert_eq!(h.used(), 2 * cons.allocation_size(0));
    }

    #[test]
    fn varlen_values_object_moves_intact() {
        let (h, t) = small_heap();
        let vinfo = &t[kind::VALUES as usize];
        let cons = &t[kind::CONS as usize];
        let child = h.alloc(cons, 0);
        unsafe { child.set_field(0, imm(77)) };
        let v = h.alloc(vinfo, 4);
        unsafe {
            v.values_mut(vinfo).copy_from_slice(&[imm(1), enc(child), imm(3), imm(4)]);
        }
        let mut root = enc(v);
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        unsafe {
            let nv = dec(root);
            assert_eq!(nv.aux(), 4);
            let vals = nv.values(vinfo);
            assert_eq!(vals[0], imm(1));
            assert_eq!(vals[2], imm(3));
            let nchild = dec(vals[1]);
            assert_eq!(nchild.field(0), imm(77));
        }
    }

    #[test]
    fn strings_move_with_bytes() {
        let (h, t) = small_heap();
        let sinfo = &t[kind::STR as usize];
        let s = h.alloc(sinfo, 11);
        unsafe { s.bytes_mut(sinfo).copy_from_slice(b"hello world") };
        let mut root = enc(s);
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        unsafe {
            assert_eq!(dec(root).bytes(sinfo), b"hello world");
        }
    }

    #[test]
    fn flip_reuses_spaces_across_many_collections() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let mut root;
        {
            let c = h.alloc(cons, 0);
            unsafe { c.set_field(0, imm(5)) };
            root = enc(c);
        }
        let space = h.from_space().size();
        // Enough allocation+collection that an append-only heap would blow a
        // 64 KiB space many times over; flip+reuse never exceeds one space.
        for round in 0..50 {
            for _ in 0..500 {
                h.alloc(cons, 0); // garbage
            }
            unsafe {
                h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
            }
            assert_eq!(h.used(), cons.allocation_size(0), "round {round}: only the root lives");
            assert!(h.used() <= space);
        }
        unsafe { assert_eq!(dec(root).field(0), imm(5)) };
        assert_eq!(h.collections.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn growable_array_handle_shape() {
        let (h, t) = small_heap();
        let ainfo = &t[kind::ARRAY as usize];
        let dinfo = &t[kind::ARRAY_DATA as usize];
        // capacity-4 blob, logical len 2
        let blob = h.alloc(dinfo, 4);
        unsafe { blob.values_mut(dinfo)[..2].copy_from_slice(&[imm(10), imm(20)]) };
        let handle = h.alloc(ainfo, 2); // aux = logical len (metadata)
        unsafe { handle.set_field(0, enc(blob)) };
        let mut root = enc(handle);
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        unsafe {
            let nh = dec(root);
            assert_eq!(nh.aux(), 2, "logical length rides the header");
            let nblob = dec(nh.field(0));
            assert_eq!(nblob.aux(), 4, "capacity rides the blob header");
            assert_eq!(&nblob.values(dinfo)[..2], &[imm(10), imm(20)]);
            // grow in place: len < cap
            nh.set_aux(3);
            nblob.values_mut(dinfo)[2] = imm(30);
            assert_eq!(nh.aux(), 3);
        }
    }

    #[test]
    fn atom_slot_is_cas_able_and_traced() {
        let (h, t) = small_heap();
        let ainfo = &t[kind::ATOM as usize];
        let cons = &t[kind::CONS as usize];
        let boxed = h.alloc(cons, 0);
        unsafe { boxed.set_field(0, imm(123)) };
        let atom = h.alloc(ainfo, 0);
        unsafe {
            let slot = atom.field_atomic(0);
            slot.store(enc(boxed), Ordering::Release);
            // CAS like the runtime's swap! would
            let old = slot.load(Ordering::Acquire);
            assert!(slot.compare_exchange(old, old, Ordering::AcqRel, Ordering::Acquire).is_ok());
        }
        let mut root = enc(atom);
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        unsafe {
            let na = dec(root);
            let v = na.field_atomic(0).load(Ordering::Acquire);
            assert_eq!(dec(v).field(0), imm(123), "atom contents traced + forwarded");
        }
    }

    #[test]
    #[should_panic(expected = "precise-layout violation")]
    fn verify_catches_scalar_in_traced_slot() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let c = h.alloc(cons, 0);
        // Forge a "pointer" into from-space that does NOT target an object
        // header: point at the cons's second field word, which stays zero —
        // read as a header that is type_id 0 = INVALID.
        let bogus = TestPolicy::encode_ptr(unsafe { c.0.add(16) });
        unsafe { c.set_field(0, bogus) };
        let mut root = enc(c);
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
    }

    #[test]
    fn poisoned_space_reads_loud_type_id() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let stale = h.alloc(cons, 0);
        let mut root = imm(0); // nothing live
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        // verify mode re-zeroes after poisoning so the space is reusable; the
        // stale object's memory is at minimum no longer a valid CONS header.
        unsafe {
            assert_ne!(stale.type_id(), kind::CONS, "stale from-space read must not look live");
        }
    }

    #[test]
    #[should_panic(expected = "heap space exhausted")]
    fn exhaustion_is_loud() {
        let h = Heap::with_space_size(1 << 12);
        let t = type_table();
        for _ in 0..10_000 {
            h.alloc(&t[kind::CONS as usize], 0);
        }
    }

    #[test]
    fn alloc_window_mirrors_active_space() {
        let (h, t) = small_heap();
        h.arm_window(); // the heap is at its final address now
        let base0 = h.window.base.load(Ordering::Acquire);
        assert_eq!(base0, h.from_space().base());
        let mut root = imm(0);
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        let base1 = h.window.base.load(Ordering::Acquire);
        assert_eq!(base1, h.from_space().base(), "window re-pointed at flip");
        assert_ne!(base0, base1, "flip changed the active space");
        // gc-stress: limit 0 forces the slow path in emitted code (D5 uses
        // this; here we just pin the ABI field).
        h.window.limit.store(0, Ordering::Release);
        assert_eq!(h.window.limit.load(Ordering::Acquire), 0);
    }

    #[test]
    fn multithreaded_alloc_is_disjoint() {
        let h = std::sync::Arc::new(Heap::with_space_size(1 << 22));
        let t = std::sync::Arc::new(type_table());
        let mut handles = Vec::new();
        for tid in 0..8u64 {
            let h = h.clone();
            let t = t.clone();
            handles.push(std::thread::spawn(move || {
                let cons = &t[kind::CONS as usize];
                let mut mine = Vec::new();
                for i in 0..2000u64 {
                    let c = h.alloc(cons, 0);
                    unsafe {
                        c.set_field(0, (tid << 32 | i) << 3);
                    }
                    mine.push(c);
                }
                for (i, c) in mine.iter().enumerate() {
                    unsafe {
                        assert_eq!(c.field(0), (tid << 32 | i as u64) << 3, "another thread scribbled");
                    }
                }
            }));
        }
        for jh in handles {
            jh.join().unwrap();
        }
        assert_eq!(
            h.used(),
            8 * 2000 * type_table()[kind::CONS as usize].allocation_size(0)
        );
    }
}
