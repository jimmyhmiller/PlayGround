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
//! Stage I made the heap GENERATIONAL; Stage M gave the young generation a
//! SURVIVOR SPACE + AGING. Everything is born in EDEN (the only space `alloc`
//! and the JIT's window ever bump). The young gen is now three regions —
//! eden plus two SURVIVOR bump spaces (one active FROM, one idle TO) — over the
//! old semi-space pair, unchanged. Two collections:
//!
//! - a MINOR (`collect_minor`) evacuates the live young set (eden ∪
//!   survivor-FROM) into survivor-TO, ageing each object by one; an object that
//!   has survived `tenure` minors — or that overflows survivor-TO — is PROMOTED
//!   to the old gen instead. Its root set is the caller's enumeration PLUS a
//!   scan of the old gen's DIRTY CARDS, because an old object's field can name
//!   a young object (in eden or survivor-FROM) and no root does. That is the
//!   whole reason the write barrier exists. Stage I promoted on FIRST survival,
//!   which over-tenured mid-iteration transients into the old gen where they
//!   died at once; aging is the fix — a short-lived object now dies in a
//!   survivor space and never reaches the old gen.
//! - a MAJOR (`collect`) is Stage D's Cheney over the old gen — extended to
//!   treat the WHOLE young gen (eden + both survivors) as extra from-spaces, so
//!   it is a complete collection in any heap state and needs no minor first.
//!
//! The invariant everything else rests on: **a minor fully evacuates eden and
//! survivor-FROM** (an exact evacuation still — a survivor now lives in
//! survivor-TO rather than the old gen), so afterwards the only old→young edges
//! are old→survivor-TO ones and clearing EVERY card is correct (survivor-TO is
//! itself fully scanned as from-space every minor). It is also checkable, which
//! is what makes this shippable: in verify mode a minor ends by walking the
//! whole old gen and asserting that no object holds a pointer into eden or
//! survivor-FROM. A missed barrier is otherwise silent and corrupts the heap
//! arbitrarily later (beagle shipped exactly this bug and found it via a crash
//! in unrelated code); here it dies naming the object, its type, the slot, and
//! the target — at the collection that caused it.
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

// ─── Object age (Stage M: survivor + aging) ───────────────────────────────
//
// A young object's AGE counts the minor collections it has survived. It rides
// in the low bits of the header's `spare` u16 — free today: the runtime's
// transient stamps deliberately went to a value slot, not here (see the note
// on `hamt` transients in runtime.rs), and the closure variadic bit lives in
// the meta word, not spare. At `TENURE_THRESHOLD` an object is promoted to the
// old gen and the age goes inert (minors never move old objects again).

/// Bits of `spare` the age uses. 8 is far more than any tenure threshold
/// needs, and leaves the high half of `spare` for a future use.
pub const AGE_BITS: u16 = 8;
/// Mask selecting the age within `spare`.
pub const AGE_MASK: u16 = (1 << AGE_BITS) - 1;

/// The age recorded in a (non-forwarded) header.
#[inline(always)]
pub const fn header_age(hdr: u64) -> u16 {
    header_spare(hdr) & AGE_MASK
}

/// The header word with its age field replaced, every other bit — type_id,
/// aux, and any non-age spare bit — preserved. Used to stamp the NEW copy of a
/// surviving object; it must NEVER be written to the forwarding word, which
/// lives in the OLD header after evacuation.
#[inline(always)]
pub const fn header_with_age(hdr: u64, age: u16) -> u64 {
    (hdr & !((AGE_MASK as u64) << 16)) | (((age & AGE_MASK) as u64) << 16)
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
    /// The object's age — minors survived (Stage M). Only meaningful for young
    /// objects; inert once promoted.
    #[inline(always)]
    pub unsafe fn age(self) -> u16 {
        header_age(unsafe { self.header() })
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

    /// Why the NURSERY is only ever zeroed, never poisoned:
    ///
    /// Poison earns its keep on a semi-space, which sits idle until the next
    /// flip and is refilled only by whole-object copies — the pattern survives
    /// the whole idle window, so a stale read lands on it. The nursery is the
    /// opposite: `alloc` bumps into it the instant we return and writes only
    /// the header, promising the rest is zero (`alloc_closure` leans on that
    /// directly — an unwritten code word MUST read 0, or the JIT's `code != 0`
    /// guard would call into the pattern). Poisoning it would therefore have to
    /// be undone by a zeroing pass in the same breath, under STW, with no
    /// observation point in between: the stamp could never be read by anyone,
    /// and it would cost a SECOND full-nursery memset on the hottest GC path.
    ///
    /// A stale nursery read is still loud without it: the zeroed header is
    /// `type_id` 0 = `kind::INVALID`, which every reader already panics on.
    /// So the nursery gets exactly one zeroing pass, in both modes.
    ///
    /// (Historical trap, fixed in Stage I: `reset_poisoned(); reset_zeroed();`
    /// does NOT poison-then-zero — the first call clears the cursor, so the
    /// second sees `used() == 0` and zeroes nothing. The old semi-space reset
    /// read as "poison, then re-zero" and in fact left the space poisoned;
    /// `alloc` then handed out non-zero memory after a flip.)
    #[inline]
    fn reset_nursery_space(&self) {
        self.reset_zeroed();
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

// ─── Card table (Stage I) ────────────────────────────────────────────────

/// 512 bytes of old gen per card — gc-rust's `card_table.rs` shift, and
/// HotSpot's.
pub const CARD_SHIFT: usize = 9;
pub const CARD_SIZE: usize = 1 << CARD_SHIFT;
pub const CARD_CLEAN: u8 = 0;
pub const CARD_DIRTY: u8 = 1;

/// A zeroed `Vec<AtomicU8>`, allocated the way the spaces are.
///
/// `vec![0u8; n]` routes to `alloc_zeroed`/calloc, whose large-allocation path
/// is fresh mmap'd zero pages — reserved up front, COMMITTED LAZILY as touched.
/// That matters because the tables are sized against the OLD GEN, which is
/// multi-GiB by default (4 GiB/space = 8 MiB of cards + 64 MiB of start
/// offsets). Building them element-by-element — the only way to `collect()` a
/// `Vec` of atomics — would touch, and therefore commit, every one of those
/// pages per `Runtime` before a single card is ever marked.
fn zeroed_atomic_u8s(n: usize) -> Vec<AtomicU8> {
    let mut v = std::mem::ManuallyDrop::new(vec![0u8; n]);
    // SAFETY: AtomicU8 has the size and alignment of u8 and an all-zero byte
    // is a valid AtomicU8, so the buffer — and the Layout it will be freed
    // with — are identical under either element type.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr().cast::<AtomicU8>(), v.len(), v.capacity()) }
}

/// The same trick for the object-start index. See `zeroed_atomic_u8s`.
fn zeroed_atomic_usizes(n: usize) -> Vec<AtomicUsize> {
    let mut v = std::mem::ManuallyDrop::new(vec![0usize; n]);
    // SAFETY: AtomicUsize has the size and alignment of usize and an all-zero
    // word is a valid AtomicUsize.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr().cast::<AtomicUsize>(), v.len(), v.capacity()) }
}

/// The three words the write barrier reads, mirrored into emitted code by the
/// JIT (I3) exactly as `AllocWindow` is by the inline allocator. Layout is ABI:
/// emitted code reads the fields at fixed offsets.
///
///   offset 0: base  — base of the ACTIVE old space; the barrier's bounds
///             compare is relative to it, so it MUST follow a major flip
///   offset 8: size  — size of one old space (both are equal, so one table
///             serves whichever is active)
///   offset 16: cards — base of the card byte array
///
/// This is not a *copy* of the barrier's inputs, it is the only one: `mark`
/// reads these same words. A mirror the Rust barrier did not itself read could
/// drift from it silently, and "the JIT marks a card in the wrong table" is
/// precisely the class of bug this stage exists to make impossible.
///
/// All three are atomics because a `*const u8` field would cost `Heap` its
/// auto `Sync` — the same reason `AllocWindow` is all-atomic. Only `base`
/// actually changes (at flips, under STW); `size` and `cards` are set once, so
/// their relaxed loads are plain loads.
#[repr(C)]
struct CardWindow {
    base: AtomicUsize,
    size: AtomicUsize,
    cards: AtomicPtr<u8>,
}

/// The remembered set — and there is no other one.
///
/// beagle's post-mortem is the design here: its barrier pushed to a
/// `dirty_card_indices` Vec AND a `remembered_set` Vec on every old→young
/// store, unsynchronized, which is a real data race under ≥2 mutators. Lost
/// edges → a young object referenced only from the old gen is never promoted →
/// a stale/forwarded pointer surfaces as a crash in unrelated code. So:
///
/// - the card BYTE array is the single source of truth,
/// - marked with ONE idempotent `Relaxed` store (concurrent same-value marks
///   are benign; visibility comes from the STW safepoint, not a per-barrier
///   fence — and NOT from a mutex, which Jimmy explicitly rejected),
/// - dirty cards are DISCOVERED by scanning the array at STW, word-at-a-time.
///   The table is 1/512 of the old gen; we never walk the old gen itself.
///
/// `starts` is what keeps the dirty-card scan O(dirty objects) instead of
/// O(old gen). beagle needed a full Block Offset Table because its old gen is
/// a free-list mark-and-sweep; ours is copying/compact, so objects run
/// contiguously from base to `used` and "the object starts in this card" is a
/// forward walk from one known offset. It is written ONLY at the placement
/// chokepoint (`copy_or_forward`) and read ONLY at STW — the barrier never
/// touches it, which is the property beagle's design lacked.
pub struct CardTable {
    /// The three words the barrier needs — and the ONLY copy of them, read by
    /// `mark` and by the JIT's inline mirror alike. See `CardWindow`.
    win: CardWindow,
    cards: Vec<AtomicU8>,
    /// Per card: 1 + the byte offset of the FIRST object that BEGINS in it, or
    /// 0 for "no object begins here". The +1 bias is what lets the table be
    /// `alloc_zeroed` (offset 0 is a real object start), and a dirty card whose
    /// entry is 0 is a HARD PANIC — never a silent skip. That is sound only
    /// because the barrier marks the OBJECT BASE, not the field address: an
    /// object spanning cards 3..7 is only ever marked on card 3, so cards 4..6
    /// are never dirtied without a start.
    starts: Vec<AtomicUsize>,
    /// Exclusive high-water of every offset that has a `starts` entry. Nothing
    /// above `scan_cards()` has ever been written, so the scans and clears
    /// never touch — and so never commit — the lazily-reserved pages past it.
    indexed: AtomicUsize,
}

impl CardTable {
    fn new(base: usize, size: usize) -> Self {
        let ncards = size.div_ceil(CARD_SIZE);
        let cards = zeroed_atomic_u8s(ncards);
        CardTable {
            // The window's `cards` pointer names the Vec's buffer, which is
            // allocated once and never grown — so this is stable for the
            // table's life, and stable across the `Heap` move that `arm_window`
            // waits for (moving a Vec does not move its buffer, unlike the
            // AllocWindow's cursor, which points INTO the Heap).
            win: CardWindow {
                base: AtomicUsize::new(base),
                size: AtomicUsize::new(size),
                cards: AtomicPtr::new(cards.as_ptr() as *mut u8),
            },
            cards,
            starts: zeroed_atomic_usizes(ncards),
            indexed: AtomicUsize::new(0),
        }
    }

    /// THE write barrier. Mark the card holding `addr` — an OBJECT BASE —
    /// dirty, if `addr` is in the old gen at all.
    ///
    /// One unsigned compare does both bounds: a nursery (or any non-old)
    /// address wraps `addr - base` to a huge offset and fails `< size`. That is
    /// why no generation test is needed.
    ///
    /// The JIT mirrors this sequence instruction-for-instruction in
    /// `emit_card_mark` — load base/size/cards from the `CardWindow`, sub,
    /// unsigned compare, shift, byte store — reading the SAME `win` words this
    /// does. Keep the two in step: they are one barrier with two emitters.
    #[inline(always)]
    pub fn mark(&self, addr: usize) {
        let off = addr.wrapping_sub(self.win.base.load(Ordering::Relaxed));
        if off < self.win.size.load(Ordering::Relaxed) {
            // SAFETY: `off < size` and `cards.len() == size.div_ceil(CARD_SIZE)`,
            // so `off >> CARD_SHIFT < cards.len()`. The single compare above IS
            // the bounds check; a second one is pure cost on the hot path.
            unsafe { self.cards.get_unchecked(off >> CARD_SHIFT) }.store(CARD_DIRTY, Ordering::Relaxed);
        }
    }

    /// Point the table at `space` (the active old space) after a flip. STW.
    fn point_at(&self, space: &AtomicBumpAllocator) {
        debug_assert_eq!(
            space.size(),
            self.win.size.load(Ordering::Relaxed),
            "card table sized for a different old space"
        );
        self.win.base.store(space.base() as usize, Ordering::Release);
    }

    /// Address of the `CardWindow` — what the JIT holds in its RunCtx so its
    /// inline mark reads the live base at every flip (I3).
    pub fn window(&self) -> *const u8 {
        &self.win as *const CardWindow as *const u8
    }

    /// Record that an object BEGINS at old-gen byte `off`. Called once per
    /// placement, from the collector only.
    ///
    /// Placement is bump allocation, so offsets ascend and the first object
    /// placed in a card IS the first one that begins in it — the entry is
    /// written once and never lowered.
    #[inline]
    fn record_start(&self, off: usize) {
        let c = off >> CARD_SHIFT;
        if self.starts[c].load(Ordering::Relaxed) == 0 {
            self.starts[c].store(off + 1, Ordering::Relaxed);
        }
        if off + 1 > self.indexed.load(Ordering::Relaxed) {
            self.indexed.store(off + 1, Ordering::Relaxed);
        }
    }

    /// Byte offset of the first object beginning in card `c`, if any.
    #[inline]
    fn start_of(&self, c: usize) -> Option<usize> {
        match self.starts[c].load(Ordering::Relaxed) {
            0 => None,
            biased => Some(biased - 1),
        }
    }

    /// Is the card holding old-gen address `addr` dirty? (Verify-mode check that
    /// a retained old→survivor edge kept its card — the same unsigned-bounds
    /// trick as `mark`.) A non-old address is trivially not dirty.
    #[inline]
    fn is_dirty_addr(&self, addr: usize) -> bool {
        let off = addr.wrapping_sub(self.win.base.load(Ordering::Relaxed));
        off < self.win.size.load(Ordering::Relaxed)
            && self.cards[off >> CARD_SHIFT].load(Ordering::Relaxed) != CARD_CLEAN
    }

    /// Every card index that could be dirty or indexed: nothing above the
    /// active cursor or the recorded high-water has ever been written, because
    /// the barrier only marks a real old object's BASE, which is below the
    /// cursor by construction, and `record_start` only runs below it too.
    /// Scanning or clearing the full table instead would touch — and so commit
    /// — all 8 MiB of cards and 64 MiB of starts on every collection.
    ///
    /// A barrier called with a bogus address ABOVE this bound is therefore not
    /// scanned. That loses nothing: there is no object up there to find an edge
    /// from, and the edge that the call was *supposed* to remember is by
    /// construction still unmarked — which the missed-barrier walk catches.
    /// (A bogus address that lands below the bound hits `start_of`'s hard
    /// panic.) Both bad-barrier shapes stay loud; neither is skipped silently.
    fn scan_cards(&self, active_used: usize) -> usize {
        active_used
            .max(self.indexed.load(Ordering::Relaxed))
            .div_ceil(CARD_SIZE)
            .min(self.cards.len())
    }

    /// Call `f` with every dirty card index, ascending.
    ///
    /// WORD-AT-A-TIME: eight card bytes are read as one `u64` and the whole run
    /// is skipped on zero — one compare instead of eight branches. beagle's
    /// first cut scanned byte-by-byte into a per-GC `HashSet` and went
    /// pathological under gc-stress (a collection per allocation).
    ///
    /// # Safety
    /// STW: no mutator may be marking. The `u64` reads alias the `AtomicU8`
    /// buffer, which is only race-free because nothing else is running.
    unsafe fn for_each_dirty(&self, active_used: usize, mut f: impl FnMut(usize)) {
        let n = self.scan_cards(active_used);
        let p = self.cards.as_ptr().cast::<u8>();
        let mut i = 0;
        while i + 8 <= n {
            // Unaligned: the table is a byte array, and on every target we
            // care about this is one ordinary load.
            if unsafe { p.add(i).cast::<u64>().read_unaligned() } == 0 {
                i += 8;
                continue;
            }
            for k in i..i + 8 {
                if unsafe { *p.add(k) } != CARD_CLEAN {
                    f(k);
                }
            }
            i += 8;
        }
        while i < n {
            if unsafe { *p.add(i) } != CARD_CLEAN {
                f(i);
            }
            i += 1;
        }
    }

    /// Clean every card. A minor calls this and then RE-ARMS (via `mark`) the
    /// cards of old objects that still point into survivor-TO: Stage M made
    /// survivors that move across minors, so HotSpot-style card cleaning — moot
    /// under Stage I's promote-straight-to-old — is now load-bearing. (A major
    /// calls this too, and rebuilds from scratch, so no card survives it.)
    ///
    /// # Safety
    /// STW.
    unsafe fn clear_cards(&self, active_used: usize) {
        let n = self.scan_cards(active_used);
        unsafe { std::ptr::write_bytes(self.cards.as_ptr() as *mut u8, CARD_CLEAN, n) };
    }

    /// Drop the object-start index. Only a MAJOR does this: it relocates every
    /// object, so it rebuilds the index from scratch as it copies.
    ///
    /// # Safety
    /// STW, and the caller must re-record every surviving object.
    unsafe fn clear_starts(&self, active_used: usize) {
        let n = self.scan_cards(active_used);
        unsafe { std::ptr::write_bytes(self.starts.as_ptr() as *mut usize, 0, n) };
        self.indexed.store(0, Ordering::Relaxed);
    }
}

// SAFETY: `cards`/`starts` are atomics; `base` is atomic and written only at
// STW flips. Everything except `mark` is STW-only.
unsafe impl Send for CardTable {}
unsafe impl Sync for CardTable {}

// ─── The heap: nursery + old semi-spaces + Cheney evacuation ─────────────

/// Default per-space size in MiB when `MICROLANG_HEAP_MB` is unset. Virtual
/// reservation (lazily committed), so big is cheap; this is a CAP, and
/// exhausting it is a loud panic (allocation never triggers GC here — the
/// runtime's GC is explicit/safepoint-driven, unchanged from before).
const DEFAULT_SPACE_MB: usize = 4096;

/// Default NURSERY size in MiB when `MICROLANG_NURSERY_MB` is unset. Every
/// object is born here; a minor evacuates it whole. Stage I first shipped 32
/// with a 50%-of-nursery trigger — but that is TWO mistakes for a fully
/// evacuated space (not a flip semi-space, where 50% is right): it fires a
/// minor every 16 MiB and leaves half the nursery cold. A minor's cost is the
/// LIVE set it copies, not the dead garbage it skips, so a bigger nursery with
/// a fill-it-up trigger cuts minor FREQUENCY without making any single minor
/// more expensive — the win Stage I was supposed to deliver but the tuning
/// undid. 64 MiB fills to `size - HEADROOM` before collecting.
const DEFAULT_NURSERY_MB: usize = 64;

/// Default SURVIVOR size in MiB per space when `MICROLANG_SURVIVOR_MB` is unset
/// — ~1/8 of the default eden (64 MiB). A survivor holds only what actually
/// survived a minor (a small fraction of the eden it drained) and is copied
/// whole every minor, so it is far smaller than eden. It is a TUNING knob, not
/// a correctness bound: overflow spills to the old gen (see `evacuate_young`),
/// so an undersized survivor only tenures things a little early. Sized as a
/// FRACTION of the nursery so the tiny test heaps get a proportionally tiny
/// survivor rather than one larger than the nursery they share a heap with.
const DEFAULT_SURVIVOR_DIVISOR: usize = 8;

/// Default tenuring threshold (`MICROLANG_TENURE` overrides). An object is
/// promoted to the old gen only after surviving this many minors as a
/// survivor. The whole point of Stage M: a transient that dies within its
/// first few minors — the mid-iteration garbage that promote-on-first-survival
/// over-tenured — NEVER reaches the old gen. 3 is HotSpot's small-heap order.
pub const TENURE_THRESHOLD: u16 = 3;

/// Reserved tail of the nursery kept BELOW the soft trigger: the most a single
/// expression can allocate between two safepoints (which is where a
/// pressure-raised collection actually fires). A `%pv-from-array` / big-literal
/// burst must still fit here after the trigger trips, or `alloc` hits the hard
/// wall mid-expression. 16 MiB matches the headroom the original 32 MiB@50%
/// default happened to leave, so this raises the fill fraction without
/// shrinking the safety margin.
const NURSERY_HEADROOM_MB: usize = 16;

/// The nursery's soft trigger in bytes. An explicit `MICROLANG_GC_TRIGGER_PCT`
/// (the stress/low-trigger tests) wins and applies as a percentage. Otherwise
/// fill to `size - HEADROOM`, but never above half of a nursery too small to
/// hold the headroom (the tiny test heaps), so it still collects.
fn nursery_soft_limit(nursery_bytes: usize, pct_explicit: Option<usize>) -> usize {
    if let Some(pct) = pct_explicit {
        return nursery_bytes / 100 * pct;
    }
    let headroom = NURSERY_HEADROOM_MB << 20;
    if nursery_bytes > headroom * 2 {
        nursery_bytes - headroom
    } else {
        nursery_bytes / 2
    }
}

/// Bits of the ONE cheap poll word (`Heap::poll`) every tier checks at its
/// safepoints and the JIT polls from emitted code (Stage E). Bit 0 mirrors the
/// runtime's `gc_requested` (a sibling wants to collect: park). Bit 1 is
/// allocation PRESSURE: the bump cursor crossed the soft threshold, so the
/// next safepoint should trigger a collection itself.
pub const POLL_REQUESTED: u8 = 1;
pub const POLL_PRESSURE: u8 = 2;

/// What a minor GC did, and what the caller must do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MinorOutcome {
    /// Bytes copied out of the nursery into the old gen.
    pub promoted_bytes: usize,
    /// The old gen crossed its own soft threshold, so promotion is now the
    /// thing filling the heap: follow with `collect` (a major). Safe to do
    /// immediately — the nursery is empty, so the major only sees old objects.
    pub needs_major: bool,
}

/// The Stage-I heap: a nursery every object is born in, plus Stage D's two
/// equal old bump spaces with Cheney evacuation, flip and reuse. Shared across
/// mutator threads (allocation is an atomic bump); `collect`/`collect_minor`
/// must run under the runtime's existing STW discipline (all other mutators
/// parked, heap_lock held).
pub struct Heap {
    /// EDEN: the ONE space `alloc` and the JIT's inline window bump into.
    /// Everything is born here. Stage M made the young generation three
    /// regions (eden + two survivor spaces); eden is still the only one any
    /// allocation touches, and the window still points here, unchanged.
    nursery: AtomicBumpAllocator,
    /// The two SURVIVOR spaces (Stage M): one active FROM, one empty TO. A
    /// minor evacuates the live young set (eden ∪ survivor-FROM) into
    /// survivor-TO, ageing each object by one; an object that reaches
    /// `tenure` (or overflows survivor-TO) promotes to the old gen instead.
    /// This is what stops the over-promotion of promote-on-first-survival: a
    /// transient that dies within a few minors never leaves the young gen.
    survivor: [AtomicBumpAllocator; 2],
    /// Index of the active survivor FROM-space (the one holding last minor's
    /// survivors). Swapped with TO at the end of every minor, under STW.
    survivor_active: AtomicUsize,
    /// Tenuring threshold: age at which a survivor is promoted to the old gen.
    /// `MICROLANG_TENURE`, default `TENURE_THRESHOLD`.
    tenure: u16,
    /// The OLD gen: Stage D's semi-space pair, unchanged. Only the collector
    /// fills it — a minor appends promoted (tenured or spilled) objects to the
    /// active space, a major Cheney-copies that space into the other and flips.
    spaces: [AtomicBumpAllocator; 2],
    /// Index of the active (from) space. Flipped only under STW.
    active: AtomicUsize,
    /// Old→young edges, one byte per 512-byte card of the ACTIVE old space.
    cards: CardTable,
    /// Armed: poison evacuated space, check every traced slot's target header,
    /// and walk the old gen after every minor hunting a missed barrier.
    verify: bool,
    /// Total collections, minor + major. (Stage D/E callers count `collect`
    /// calls with this; it keeps meaning "how many times did the GC run".)
    pub collections: AtomicU64,
    pub minor_collections: AtomicU64,
    pub major_collections: AtomicU64,
    /// Bytes copied out of the nursery into the old gen, all time.
    pub promoted_bytes: AtomicU64,
    /// Live bytes copied by the last MAJOR collection.
    pub last_live_bytes: AtomicUsize,
    /// The JIT's inline-allocation mirror (D5). It points at the NURSERY, which
    /// never flips — so unlike Stage D it is armed once and never re-pointed.
    /// The D5 inline sequence is unchanged; it just bumps a different space.
    pub window: AllocWindow,
    /// The safepoint poll word (`POLL_*` bits). Allocation sets PRESSURE when
    /// it crosses `soft_limit`; the STW rendezvous mirrors REQUESTED here so
    /// one byte answers "should this safepoint do anything".
    pub poll: AtomicU8,
    /// Soft trigger in bytes (`MICROLANG_GC_TRIGGER_PCT` of the NURSERY,
    /// default 50%). Atomic so tests can lower it on a live heap. Allocation
    /// NEVER collects and never fails before the hard wall — crossing this only
    /// raises the pressure bit for the next safepoint.
    soft_limit: AtomicUsize,
    /// The same trigger for the OLD gen (percent of one old space). A minor
    /// promotes into the old gen, so the old gen fills too; crossing this is
    /// what `MinorOutcome::needs_major` reports.
    old_soft_limit: AtomicUsize,
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
        let nmb = std::env::var("MICROLANG_NURSERY_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NURSERY_MB);
        Self::with_sizes(mb << 20, nmb << 20)
    }

    /// The test constructor: `bytes` sizes the nursery AND each old semi-space
    /// alike, so a small heap collects (and exhausts) at a predictable point.
    pub fn with_space_size(bytes: usize) -> Self {
        Self::with_sizes(bytes, bytes)
    }

    /// Survivor size is derived (env or a fraction of the nursery). This
    /// forwards to `with_sizes_survivor` with that default; tests that want to
    /// force survivor-TO overflow pass an explicit tiny survivor there.
    pub fn with_sizes(old_bytes: usize, nursery_bytes: usize) -> Self {
        let survivor_bytes = std::env::var("MICROLANG_SURVIVOR_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb << 20)
            .unwrap_or_else(|| (nursery_bytes / DEFAULT_SURVIVOR_DIVISOR).max(8));
        Self::with_sizes_survivor(old_bytes, nursery_bytes, survivor_bytes)
    }

    pub fn with_sizes_survivor(old_bytes: usize, nursery_bytes: usize, survivor_bytes: usize) -> Self {
        let pct_explicit = std::env::var("MICROLANG_GC_TRIGGER_PCT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|p| (1..=100).contains(p));
        let pct = pct_explicit.unwrap_or(50);
        let tenure = std::env::var("MICROLANG_TENURE")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .filter(|&t| t <= AGE_MASK)
            .unwrap_or(TENURE_THRESHOLD);
        let stress = std::env::var("MICROLANG_GC_STRESS").is_ok_and(|v| v != "0" && !v.is_empty());
        let spaces = [AtomicBumpAllocator::new(old_bytes), AtomicBumpAllocator::new(old_bytes)];
        // Sized for ONE old space (both are equal) and pointed at the active
        // one; the flip re-points it. `AtomicBumpAllocator::new` rounds, so
        // take the size it actually reserved rather than the request.
        let cards = CardTable::new(spaces[0].base() as usize, spaces[0].size());
        // The window stays EMPTY (limit 0 = closed, every inline allocation
        // takes the slow path) until `arm_window` — the cursor mirror is a
        // pointer INTO this struct's inline spaces, so arming here would
        // dangle the moment the constructed Heap is moved to its final home.
        Heap {
            nursery: AtomicBumpAllocator::new(nursery_bytes),
            survivor: [
                AtomicBumpAllocator::new(survivor_bytes),
                AtomicBumpAllocator::new(survivor_bytes),
            ],
            survivor_active: AtomicUsize::new(0),
            tenure,
            spaces,
            active: AtomicUsize::new(0),
            cards,
            verify: verify_armed_default(),
            collections: AtomicU64::new(0),
            minor_collections: AtomicU64::new(0),
            major_collections: AtomicU64::new(0),
            promoted_bytes: AtomicU64::new(0),
            last_live_bytes: AtomicUsize::new(0),
            window: AllocWindow::empty(),
            poll: AtomicU8::new(if stress { POLL_PRESSURE } else { 0 }),
            // The NURSERY fills to `size - headroom` before a minor (it is
            // evacuated whole, so filling it costs nothing extra — see
            // DEFAULT_NURSERY_MB). An EXPLICIT `MICROLANG_GC_TRIGGER_PCT` still
            // wins (the stress/low-trigger tests set it to 1), and a nursery too
            // small to hold the headroom (the tiny test heaps) falls back to
            // half-full so it still collects. The OLD gen keeps the percentage
            // trigger: its threshold gates when a minor defers to a major, and
            // leaving real headroom there is correct.
            soft_limit: AtomicUsize::new(nursery_soft_limit(nursery_bytes, pct_explicit)),
            old_soft_limit: AtomicUsize::new(old_bytes / 100 * pct),
            stress,
        }
    }

    /// Open the JIT's inline-allocation window. Must be called (once) AFTER
    /// the heap has reached its final address — `self`'s spaces are inline, so
    /// the window's cursor pointer is only stable from then on. (Collections
    /// re-point it at every flip.)
    pub fn arm_window(&self) {
        self.window.point_at(&self.nursery, self.window_limit());
    }

    /// The window's allocation limit: the SOFT trigger, not the space size —
    /// an inline allocation crossing it falls to the out-of-line shim, which
    /// raises the pressure bit (Stage E). That is how JIT-allocated garbage
    /// drives collections without any extra emitted code per allocation.
    fn window_limit(&self) -> usize {
        self.soft_limit.load(Ordering::Relaxed).min(self.nursery.size())
    }

    /// The nursery — where every object is born and the window points.
    #[inline(always)]
    pub fn nursery(&self) -> &AtomicBumpAllocator {
        &self.nursery
    }
    /// Base of the active old space.
    #[inline(always)]
    pub fn old_base(&self) -> *mut u8 {
        self.from_space().base()
    }
    /// Address of the card table's `CardWindow` — the barrier's three words, at
    /// ABI offsets. The JIT holds this in its RunCtx and marks inline against
    /// the very words `write_barrier` reads (I3).
    #[inline(always)]
    pub fn card_window(&self) -> *const u8 {
        self.cards.window()
    }

    /// THE WRITE BARRIER. Call this AFTER storing a heap pointer into a field
    /// of `obj` — `obj` being the OBJECT BASE, never the field address.
    ///
    /// Needed on every store into an object that might be OLD: `Gc::set_field`,
    /// `values_mut`/`arr_slice_mut` element stores, the atom store/CAS (a
    /// long-lived atom `swap!`-ed to a fresh young value every frame is
    /// beagle's exact crash shape), `arr_extend`, the transient in-place edits,
    /// and the JIT's inline `aset` arm. (I2/I3 route those; I1 only provides
    /// the entry point.)
    ///
    /// NOT needed for the initializing stores of a freshly allocated object: it
    /// is in the nursery by construction, so those are young→anything. That is
    /// what keeps the D5 inline allocation sequences barrier-free — the hot
    /// paths pay nothing.
    ///
    /// Marking the base rather than the field is what makes the dirty-card scan
    /// tractable: a dirty card then always has an object starting in it, so
    /// `starts` needs one entry per card and a missing one is a real bug.
    #[inline(always)]
    pub fn write_barrier(&self, obj: Gc) {
        self.cards.mark(obj.addr());
    }

    #[inline(always)]
    pub fn from_space(&self) -> &AtomicBumpAllocator {
        &self.spaces[self.active.load(Ordering::Acquire)]
    }
    #[inline(always)]
    fn to_space(&self) -> &AtomicBumpAllocator {
        &self.spaces[1 - self.active.load(Ordering::Acquire)]
    }
    /// The active survivor FROM-space: last minor's survivors live here, and a
    /// minor drains it alongside eden.
    #[inline(always)]
    fn survivor_from(&self) -> &AtomicBumpAllocator {
        &self.survivor[self.survivor_active.load(Ordering::Acquire)]
    }
    /// The idle survivor TO-space: empty between minors, filled with this
    /// minor's aged survivors, then swapped to become the next FROM.
    #[inline(always)]
    fn survivor_to(&self) -> &AtomicBumpAllocator {
        &self.survivor[1 - self.survivor_active.load(Ordering::Acquire)]
    }

    /// Bytes held across the WHOLE heap: eden + the active survivor + old gen.
    pub fn used(&self) -> usize {
        self.nursery.used() + self.survivor_from().used() + self.from_space().used()
    }
    /// Bytes allocated in EDEN since the last collection.
    pub fn nursery_used(&self) -> usize {
        self.nursery.used()
    }
    /// Bytes of survivors held in the active survivor space (Stage M).
    pub fn survivor_used(&self) -> usize {
        self.survivor_from().used()
    }
    /// Bytes of promoted objects in the active old space.
    pub fn old_used(&self) -> usize {
        self.from_space().used()
    }
    pub fn verify_armed(&self) -> bool {
        self.verify
    }
    /// gc-stress (`MICROLANG_GC_STRESS=1`). The runtime's collection policy
    /// reads this to force a periodic MAJOR on top of the minor it runs at
    /// every safepoint — a minor never flips the semi-spaces, so without that
    /// the hammer would leave half the collector untouched.
    pub fn stress_mode(&self) -> bool {
        self.stress
    }
    /// Arm/disarm verify mode (tests; the CLI leaves the env-var default).
    pub fn set_verify(&mut self, on: bool) {
        self.verify = on;
    }
    /// Set the tenuring threshold (tests; the CLI leaves the env-var default).
    pub fn set_tenure(&mut self, tenure: u16) {
        assert!(tenure <= AGE_MASK, "tenure {tenure} exceeds the age field's {AGE_MASK}");
        self.tenure = tenure;
    }
    /// The tenuring threshold in effect.
    pub fn tenure(&self) -> u16 {
        self.tenure
    }

    /// Does `ptr` name live heap — any young region or the active old space?
    /// (Verify-mode stale-pointer check.)
    pub fn contains(&self, ptr: *const u8) -> bool {
        self.young_contains(ptr) || self.from_space().contains(ptr)
    }
    /// Is `ptr` young — in EDEN specifically? (The historical `nursery`
    /// question, kept for callers/tests that mean "born-here, not yet copied".)
    #[inline(always)]
    pub fn nursery_contains(&self, ptr: *const u8) -> bool {
        self.nursery.contains(ptr)
    }
    /// Is `ptr` in the young generation at all — eden or EITHER survivor space?
    /// This is the from-space test a MAJOR uses, and it is why a store into a
    /// survivor object still needs no write barrier: survivor addresses are not
    /// in the old gen, so the barrier's unsigned compare skips them exactly as
    /// it skips eden. Both survivors are covered because between collections
    /// only the active one holds objects, but a check that spans both is robust
    /// and costs one more compare on a cold (verify/major) path.
    #[inline(always)]
    pub fn young_contains(&self, ptr: *const u8) -> bool {
        self.nursery.contains(ptr) || self.survivor[0].contains(ptr) || self.survivor[1].contains(ptr)
    }

    /// Allocate an object of type `info` with varlen length `aux` IN THE
    /// NURSERY; the header is written, everything else is zero. Panics loudly
    /// on exhaustion — allocation NEVER triggers a collection (the runtime's GC
    /// is explicit / safepoint-driven).
    ///
    /// Nothing is ever allocated straight into the old gen: promotion is the
    /// only way in. That is what lets the initializing stores of a fresh object
    /// skip the write barrier.
    #[inline]
    pub fn alloc(&self, info: &TypeInfo, aux: u32) -> Gc {
        assert!(aux <= MAX_AUX, "heap: varlen length {aux} exceeds MAX_AUX");
        debug_assert!(
            (info.varlen != VarLenKind::None) || aux == 0 || info.aux_is_metadata(),
            "heap: aux={aux} on fixed-size type {} (set aux_is_metadata in TypeInfo if intended)",
            info.name
        );
        let size = info.allocation_size(aux);
        let space = &self.nursery;
        let p = space.alloc_raw(size);
        if p.is_null() {
            panic!(
                "heap space exhausted: nursery full ({} MiB of {} MiB): raise \
                 MICROLANG_NURSERY_MB (allocation never collects; collections run at safepoints)",
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

    /// Lower/raise the OLD gen's soft threshold — what `MinorOutcome`
    /// reports against. Bytes, per old space.
    pub fn set_old_trigger_bytes(&self, bytes: usize) {
        self.old_soft_limit.store(bytes, Ordering::Relaxed);
    }

    /// Can a minor complete without exhausting the old space?
    ///
    /// The WORST case a minor can force into the old gen is EVERY live young
    /// byte: survivor-TO could be too small, so all of it spills. Since a minor
    /// copies each object at exactly the size it already occupies, that worst
    /// case is bounded by `eden.used() + survivor-FROM.used()` (the two regions
    /// a minor drains). When the old space has room for that, the minor cannot
    /// exhaust it however the aging/spill split falls out — which is why
    /// `collect_minor` needs no mid-evacuation escape hatch (there is no safe
    /// one: the object graph is half-forwarded at that point). Whatever
    /// survivor-TO does absorb only lowers the old-gen demand below this bound.
    pub fn minor_will_fit(&self) -> bool {
        self.nursery.used() + self.survivor_from().used() <= self.from_space().remaining()
    }

    /// A MAJOR collection: Cheney evacuation of the whole live set into the old
    /// to-space, then flip. `roots` must enumerate EVERY live root slot (the
    /// runtime's shadow stacks, globals, consts, frames, published mutator
    /// roots …); each slot is rewritten in place when its target moves.
    ///
    /// This is Stage D/E's `collect` and still means "collect EVERYTHING": it
    /// treats ALL THREE young regions — eden and BOTH survivor spaces — as
    /// additional from-spaces (Stage M), so it is complete in any heap state
    /// and needs no minor to run first. That matters twice — it is what the
    /// explicit `(gc)` prim wants, and it is the answer when `minor_will_fit`
    /// says no (a minor into a nearly-full old space would be pointless anyway:
    /// this reclaims the old garbage AND the whole young gen in one pass,
    /// rather than promoting survivors twice). A major does not preserve age or
    /// the survivor split — everything that lives goes straight to the old gen.
    ///
    /// Afterwards eden and both survivor spaces are empty, every card is clean,
    /// and the object start index is rebuilt against the new active space.
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

        // A major relocates every object, so the start index it inherits is
        // about to be meaningless: drop it now and let `copy_or_forward` rebuild
        // it against to-space as it places. The dirty marks go too — a major
        // scans everything, so no card can survive it.
        unsafe {
            self.cards.clear_cards(from.used());
            self.cards.clear_starts(from.used());
        }

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

        // Phase 3: flip. The evacuated old from-space becomes the next
        // to-space; the nursery restarts at zero.
        self.last_live_bytes.store(to.used(), Ordering::Relaxed);
        if self.verify {
            // Stage I: the poison now STAYS. `alloc` no longer touches this
            // space — the nursery serves every allocation and only whole-object
            // copies ever land here — so the space can sit poisoned for its
            // entire idle window, which is what the poison is for: a stale
            // pointer into it reads type_id 0x5A5A instead of recycled memory.
            from.reset_poisoned();
        } else {
            from.reset_zeroed();
        }
        self.reset_nursery();
        // Both survivor spaces were from-spaces for this major (all live young
        // objects went to the old gen), so both are empty now — reset them like
        // the old from-space. `survivor_active` is untouched: both are empty, so
        // the next minor's TO (the inactive one) is empty as it requires.
        self.reset_survivor(&self.survivor[0]);
        self.reset_survivor(&self.survivor[1]);
        self.active.store(1 - self.active.load(Ordering::Acquire), Ordering::Release);
        // The card table follows the flip; the start index it now holds was
        // rebuilt above, in to-space offsets, and to-space is the active space
        // from this store on. Offsets are base-agnostic, which is why the
        // rebuild could run before the flip.
        self.cards.point_at(self.from_space());
        self.collections.fetch_add(1, Ordering::Relaxed);
        self.major_collections.fetch_add(1, Ordering::Relaxed);
        // The window is NOT re-pointed: it names the nursery, which does not
        // flip. Only the limit can move (`set_trigger_bytes`).
        // Pressure is spent: this collection was the response. It re-arms when
        // an allocation crosses the soft threshold again (immediately, if the
        // live set alone exceeds it — the heap is genuinely tight then). In
        // stress mode the bit stays up so EVERY safepoint keeps collecting.
        if !self.stress {
            self.poll.fetch_and(!POLL_PRESSURE, Ordering::Relaxed);
        }
    }

    /// A MINOR collection (Stage M: survivor + aging): evacuate the live young
    /// set — eden ∪ survivor-FROM — into survivor-TO, ageing each object by one;
    /// an object that has reached `tenure` (or that overflows survivor-TO)
    /// promotes to the ACTIVE old space instead. This is the hot path: it
    /// copies only what actually survived a young collection, and — the whole
    /// point of Stage M — a transient that dies within its first `tenure`
    /// minors NEVER reaches the old gen, so promotion stops filling the old gen
    /// with garbage that dies immediately.
    ///
    /// The root set is `enumerate_roots` PLUS the old gen's DIRTY CARDS: an old
    /// object's field can be the only reference to a young object (in eden OR in
    /// survivor-FROM), and no root names it. Missing that edge is the bug this
    /// whole family of stages is built around.
    ///
    /// Afterwards: eden and survivor-FROM are empty (an exact evacuation still),
    /// the survivor roles are swapped so this minor's survivor-TO is the next
    /// minor's survivor-FROM, every card is clean, and the start index covers
    /// the promoted range.
    ///
    /// The caller owns the policy, in two pieces:
    /// - BEFORE: if `minor_will_fit()` is false, call `collect` instead. A
    ///   minor cannot grow the old space, and there is no safe way to bail out
    ///   half-way through an evacuation. Calling anyway is a loud panic.
    /// - AFTER: if the returned `needs_major` is set, follow with `collect`.
    ///   The young gen is empty at that point, so the major only sees old
    ///   objects — this is the spec's "minor first, then major if over
    ///   threshold".
    ///
    /// # Safety
    /// Stop-the-world, as `collect`.
    pub unsafe fn collect_minor<P: PtrPolicy>(
        &self,
        types: &[TypeInfo],
        enumerate_roots: &mut dyn FnMut(&mut dyn FnMut(*mut u64)),
    ) -> MinorOutcome {
        let old = self.from_space();
        let survivor_from = self.survivor_from();
        let survivor_to = self.survivor_to();
        debug_assert_eq!(
            survivor_to.used(),
            0,
            "survivor-TO must be empty at the start of a minor (the previous minor reset it)"
        );
        assert!(
            self.minor_will_fit(),
            "minor GC: {} live young bytes (eden {} + survivor {}) could not all spill into the \
             old space's {} remaining bytes. Evacuation cannot fail half-way (the graph is already \
             part-forwarded), so the caller must check `minor_will_fit()` and run `collect` (a \
             major) instead.",
            self.nursery.used() + survivor_from.used(),
            self.nursery.used(),
            survivor_from.used(),
            old.remaining()
        );
        // Everything already in the old gen sits below this; everything the
        // minor PROMOTES (tenured or spilled) lands at or above it. It bounds
        // the dirty-card walk (promoted objects are scanned by the Cheney phase,
        // not re-scanned as pre-existing old objects) and starts the old Cheney
        // scan.
        let promote_from = old.used();

        // Old-gen object bases that still point into survivor-TO after this
        // minor. Their cards must be RE-ARMED (see the card-cleaning note at the
        // clear below): a GC move creates no store, so the barrier cannot know a
        // survivor shifted, and blindly clearing the card would lose an edge
        // whose target keeps moving until it tenures. Collector scratch, not a
        // barrier structure — the beagle "no aux Vec" rule is about the mutator
        // barrier, and this runs STW.
        let mut remembered: Vec<usize> = Vec::new();

        // Phase 1: evacuate from the roots. (Roots are re-scanned every minor,
        // so a root→survivor edge needs no card — only old→survivor does.)
        enumerate_roots(&mut |slot| unsafe {
            self.evacuate_young_slot::<P>(types, old, survivor_from, survivor_to, slot)
        });

        // Phase 2: the OTHER root set — old objects written since the last
        // collection. This is the only thing standing between a young object
        // reachable solely from the old gen and oblivion.
        unsafe {
            self.scan_dirty_cards::<P>(types, old, survivor_from, survivor_to, promote_from, &mut remembered)
        };

        // Phase 3: Cheney over BOTH to-regions to a fixpoint. A survivor's field
        // can name an object that tenures to the old gen, and a freshly-promoted
        // object's field can name one that stays in survivor-TO, so appending to
        // either region can create more work in either — neither scan is
        // complete until both are drained. Each object is copied at most once
        // (the forwarding bit), so the totals are bounded and the loop
        // terminates when a full pass appends nothing.
        let mut sscan = 0usize;
        let mut oscan = promote_from;
        loop {
            let mut progressed = false;
            while sscan < survivor_to.used() {
                let obj = Gc(unsafe { survivor_to.base().add(sscan) });
                let hdr = unsafe { obj.header() };
                debug_assert_eq!(hdr & FORWARDING_BIT, 0, "forwarded header among survivors");
                let info = self.checked_info(types, hdr, obj, "survivor-TO");
                unsafe {
                    scan_object(obj, info, |slot| {
                        self.evacuate_young_slot::<P>(types, old, survivor_from, survivor_to, slot)
                    });
                }
                sscan += info.allocation_size(header_aux(hdr));
                progressed = true;
            }
            while oscan < old.used() {
                let obj = Gc(unsafe { old.base().add(oscan) });
                let hdr = unsafe { obj.header() };
                debug_assert_eq!(hdr & FORWARDING_BIT, 0, "forwarded header among promoted objects");
                let info = self.checked_info(types, hdr, obj, "promoted");
                unsafe {
                    scan_object(obj, info, |slot| {
                        self.evacuate_young_slot::<P>(types, old, survivor_from, survivor_to, slot)
                    });
                    // A freshly-promoted object that ended up holding a survivor
                    // must remember it (see `remembered`).
                    if self.holds_pointer_into::<P>(obj, info, survivor_to) {
                        remembered.push(obj.addr());
                    }
                }
                oscan += info.allocation_size(header_aux(hdr));
                progressed = true;
            }
            if !progressed {
                break;
            }
        }

        let promoted = old.used() - promote_from;

        // Empty the just-evacuated young regions, then swap survivor roles: the
        // survivor-TO we just filled becomes the next minor's survivor-FROM, and
        // the now-empty survivor-FROM becomes the next minor's (empty) TO. (The
        // reset only zeroes/poisons; the address RANGES the detector checks are
        // unchanged, so it can still run afterwards.)
        self.reset_nursery();
        self.reset_survivor(survivor_from);
        self.survivor_active
            .store(1 - self.survivor_active.load(Ordering::Acquire), Ordering::Release);

        // CARD CLEANING. Eden and survivor-FROM are empty, so a card is needed
        // now ONLY for an old object that points into survivor-TO (this minor's
        // survivors, which move again next minor). Clear everything, then RE-ARM
        // exactly those. beagle's note called HotSpot-style card cleaning "moot"
        // because Stage I promoted young straight to old — but survivors keep an
        // old→young edge live across minors WITHOUT any new store, so the
        // collector, not the barrier, must retain it. Every such edge was
        // written this minor (survivor-TO is all new), so re-arming the objects
        // touched in phases 2-3 is complete; the verify walk below proves it.
        unsafe { self.cards.clear_cards(old.used()) };
        for &base in &remembered {
            self.cards.mark(base);
        }

        // THE missed-barrier detector, re-aimed: no old object may point into
        // the just-EVACUATED eden or survivor-FROM (both empty now), and every
        // old→survivor-TO edge MUST have a live card (the card-cleaning above).
        // A survivor now lives in the post-swap active survivor space
        // (`survivor_to` here), which is legal to hold.
        if self.verify {
            unsafe { self.verify_no_old_to_young::<P>(types, survivor_from, survivor_to) };
        }

        self.promoted_bytes.fetch_add(promoted as u64, Ordering::Relaxed);
        self.collections.fetch_add(1, Ordering::Relaxed);
        self.minor_collections.fetch_add(1, Ordering::Relaxed);
        if !self.stress {
            self.poll.fetch_and(!POLL_PRESSURE, Ordering::Relaxed);
        }
        MinorOutcome {
            promoted_bytes: promoted,
            needs_major: old.used() > self.old_soft_limit.load(Ordering::Relaxed),
        }
    }

    /// Read + range-check an object's `TypeInfo` from its header, with a loud
    /// panic naming the region on a corrupt type_id. Shared by the two Cheney
    /// scan loops so their bounds check reads identically.
    #[inline]
    fn checked_info<'a>(&self, types: &'a [TypeInfo], hdr: u64, obj: Gc, region: &str) -> &'a TypeInfo {
        let tid = header_type_id(hdr) as usize;
        assert!(
            tid != 0 && tid < types.len(),
            "GC: {region} object at {:p} has type_id {tid} out of range — heap corruption",
            obj.0
        );
        &types[tid]
    }

    /// Does `obj` hold a traced pointer into `space`? Used to decide whether an
    /// old object must retain a card for the survivor it points at.
    ///
    /// # Safety
    /// `obj` matches `info`; no concurrent mutation.
    unsafe fn holds_pointer_into<P: PtrPolicy>(&self, obj: Gc, info: &TypeInfo, space: &AtomicBumpAllocator) -> bool {
        let mut found = false;
        unsafe {
            scan_object(obj, info, |slot| {
                if !found {
                    if let Some(p) = P::try_decode_ptr(*slot) {
                        found = space.contains(p);
                    }
                }
            });
        }
        found
    }

    /// Empty the nursery: one zeroing pass, in both modes. See
    /// `reset_nursery_space` for why poison would be unobservable here (and
    /// would cost a second full-nursery memset on the hottest GC path).
    fn reset_nursery(&self) {
        self.nursery.reset_nursery_space();
    }

    /// Empty a just-evacuated survivor space. Unlike the nursery, a survivor is
    /// filled ONLY by whole-object copies and sits idle until the next minor
    /// re-fills it — the same shape as the old semi-space — so poison earns its
    /// keep here: a stale pointer into it reads type_id 0x5A5A for the whole
    /// idle window. No `alloc` ever touches a survivor, so nothing relies on it
    /// being zero the way the nursery does.
    fn reset_survivor(&self, s: &AtomicBumpAllocator) {
        if self.verify {
            s.reset_poisoned();
        } else {
            s.reset_zeroed();
        }
    }

    /// Evacuate one traced slot's young target (Stage M). If it points into
    /// eden OR survivor-FROM, copy/age/tenure the target and rewrite the slot.
    /// A pointer into survivor-TO is left alone — it is a slot in an already-
    /// evacuated object pointing at another already-evacuated one, in its final
    /// place for this minor. Old-gen targets stay put — a minor moves nothing
    /// that is already old.
    unsafe fn evacuate_young_slot<P: PtrPolicy>(
        &self,
        types: &[TypeInfo],
        old: &AtomicBumpAllocator,
        survivor_from: &AtomicBumpAllocator,
        survivor_to: &AtomicBumpAllocator,
        slot: *mut u64,
    ) {
        let bits = unsafe { *slot };
        if let Some(ptr) = P::try_decode_ptr(bits) {
            if self.nursery.contains(ptr) || survivor_from.contains(ptr) {
                let new = unsafe { self.evacuate_young(types, old, survivor_to, ptr) };
                unsafe { *slot = P::encode_ptr(new) };
            } else if self.verify && !old.contains(ptr) && !survivor_to.contains(ptr) {
                panic!(
                    "GC verify: traced slot holds {bits:#x} -> {ptr:p}, in none of eden, \
                     survivor-FROM, survivor-TO, or the old gen (stale pointer or non-pointer \
                     decoded as a ref)"
                );
            }
        }
    }

    /// Copy one live young object out of the from-set and return its new
    /// address, installing forwarding so shared referrers and cycles converge.
    /// The aging decision:
    /// - already forwarded → return the existing target.
    /// - `age >= tenure` → PROMOTE to the old gen (age goes inert).
    /// - else copy to survivor-TO with `age += 1`; if survivor-TO is FULL,
    ///   SPILL to the old gen instead (a minor must always complete — there is
    ///   no safe bail-out from a half-forwarded graph, and `minor_will_fit`
    ///   guaranteed the old gen can absorb the whole young set if it must).
    unsafe fn evacuate_young(
        &self,
        types: &[TypeInfo],
        old: &AtomicBumpAllocator,
        survivor_to: &AtomicBumpAllocator,
        ptr: *mut u8,
    ) -> *mut u8 {
        let hdr = unsafe { *(ptr as *const u64) };
        if hdr & FORWARDING_BIT != 0 {
            return (hdr & !FORWARDING_BIT) as *mut u8;
        }
        let tid = header_type_id(hdr) as usize;
        // PRECISE-LAYOUT DETECTOR: a traced slot pointing into the young gen
        // always targets a real object; an out-of-range type_id means a scalar
        // leaked into a traced slot. Loud, never a silent conservative skip.
        assert!(
            tid != 0 && tid < types.len(),
            "GC precise-layout violation: traced slot points at {ptr:p} whose header \
             type_id={tid} is out of range (table len {}). A non-pointer reached a traced slot.",
            types.len()
        );
        let info = &types[tid];
        let size = info.allocation_size(header_aux(hdr));
        let age = header_age(hdr);

        // Tenured, or spilled because survivor-TO is full → the old gen.
        // `copy_or_forward` copies, forwards the OLD header, and records the
        // start; the age rides along in the copy and is thereafter inert.
        if age >= self.tenure {
            return unsafe { self.copy_or_forward(types, old, ptr) };
        }
        let dest = survivor_to.alloc_raw(size);
        if dest.is_null() {
            return unsafe { self.copy_or_forward(types, old, ptr) };
        }
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, dest, size);
            // Forwarding goes in the OLD header. This must happen only after the
            // copy, and it must not be confused with the age bump below: the
            // forwarding word names `dest`, the aged header IS `dest`'s header.
            *(ptr as *mut u64) = (dest as u64) | FORWARDING_BIT;
            // Bump the age on the NEW object's header (dest currently holds a
            // verbatim copy of the original, non-forwarded header). The masked
            // write preserves type_id and aux, so a varlen object keeps its
            // length; bit 63 stays clear because age lives in bits 16..24.
            *(dest as *mut u64) = header_with_age(hdr, age + 1);
        }
        dest
    }

    /// Walk the old objects living in dirty cards and evacuate whatever young
    /// objects they name (eden or survivor-FROM). `old_end` is the old space's
    /// fill BEFORE this minor promoted anything — objects above it are this
    /// minor's promotions, and the Cheney phase owns those.
    unsafe fn scan_dirty_cards<P: PtrPolicy>(
        &self,
        types: &[TypeInfo],
        old: &AtomicBumpAllocator,
        survivor_from: &AtomicBumpAllocator,
        survivor_to: &AtomicBumpAllocator,
        old_end: usize,
        remembered: &mut Vec<usize>,
    ) {
        let base = old.base();
        unsafe {
            self.cards.for_each_dirty(old_end, |c| {
                let Some(start) = self.cards.start_of(c) else {
                    // Never a silent skip. The barrier marks object BASES, so a
                    // dirty card must contain one; if it does not, someone
                    // barriered a field address, an interior pointer, or an
                    // address outside the old gen — and some other card that
                    // SHOULD be dirty probably is not.
                    panic!(
                        "GC: dirty card {c} (old-gen bytes {}..{}) has no object start in the \
                         index. The barrier marks the OBJECT BASE, so every dirty card must have \
                         one — a barrier was called with a field address or a non-old-gen address.",
                        c << CARD_SHIFT,
                        (c + 1) << CARD_SHIFT
                    );
                };
                // The card's objects, and only them: an object BEGINNING in this
                // card is scanned whole even if it runs past the card's end.
                let card_end = ((c + 1) << CARD_SHIFT).min(old_end);
                let mut off = start;
                while off < card_end {
                    let obj = Gc(base.add(off));
                    let hdr = obj.header();
                    let tid = header_type_id(hdr) as usize;
                    assert!(
                        tid != 0 && tid < types.len(),
                        "GC: dirty-card walk found type_id {tid} at old-gen offset {off} — the \
                         object start index does not match the heap"
                    );
                    let info = &types[tid];
                    scan_object(obj, info, |slot| {
                        self.evacuate_young_slot::<P>(types, old, survivor_from, survivor_to, slot)
                    });
                    // If this old object still points into survivor-TO after
                    // evacuation, its card must survive the coming clear.
                    if self.holds_pointer_into::<P>(obj, info, survivor_to) {
                        remembered.push(obj.addr());
                    }
                    off += info.allocation_size(header_aux(hdr));
                }
            })
        }
    }

    /// After a minor, NO old object may hold a pointer into either JUST-
    /// EVACUATED young region — eden or survivor-FROM. The minor drained both
    /// whole, so such a pointer is a dangling reference to memory we are about
    /// to recycle, and it exists only because the store that created the edge
    /// never marked the card. (An old→survivor-TO pointer is NOT a violation:
    /// survivor-TO holds this minor's fresh survivors and the edge is legal,
    /// card-marked like any old→young store.)
    ///
    /// This is O(old gen), hence verify-only — but the gc-stress battery runs
    /// with verify armed, which is where it earns its keep. Without it a missed
    /// barrier is silent and surfaces later as corruption somewhere unrelated
    /// (beagle's "Struct not found by ID", hunted through a crash in another
    /// subsystem entirely).
    unsafe fn verify_no_old_to_young<P: PtrPolicy>(
        &self,
        types: &[TypeInfo],
        survivor_from: &AtomicBumpAllocator,
        survivor_to: &AtomicBumpAllocator,
    ) {
        let old = self.from_space();
        let nbase = self.nursery.base() as usize;
        let nsize = self.nursery.size();
        let sbase = survivor_from.base() as usize;
        let ssize = survivor_from.size();
        let mut off = 0usize;
        let used = old.used();
        while off < used {
            let obj = Gc(unsafe { old.base().add(off) });
            let hdr = unsafe { obj.header() };
            let tid = header_type_id(hdr) as usize;
            assert!(
                tid != 0 && tid < types.len(),
                "GC verify: old-gen object at offset {off} has type_id {tid} out of range"
            );
            let info = &types[tid];
            let mut points_at_survivor = false;
            unsafe {
                scan_object(obj, info, |slot| {
                    let bits = *slot;
                    if let Some(p) = P::try_decode_ptr(bits) {
                        let a = p as usize;
                        let region = if a.wrapping_sub(nbase) < nsize {
                            Some("EDEN")
                        } else if a.wrapping_sub(sbase) < ssize {
                            Some("SURVIVOR-FROM")
                        } else {
                            None
                        };
                        if let Some(region) = region {
                            panic!(
                                "GC verify: MISSED WRITE BARRIER — old-gen {} at {:p} (offset \
                                 {off}) still holds a {region} pointer {bits:#x} -> {p:p} in slot \
                                 +{} after a minor GC. A minor evacuates eden and survivor-FROM \
                                 whole, so this edge was never card-marked: the store that created \
                                 it did not call Heap::write_barrier.",
                                info.name,
                                obj.0,
                                slot as usize - obj.addr()
                            );
                        }
                        if survivor_to.contains(p) {
                            points_at_survivor = true;
                        }
                    }
                });
            }
            // Card-cleaning completeness: an old→survivor-TO edge that lost its
            // card would be found by NO minor next time (the survivor keeps
            // moving), corrupting the heap exactly as a missed barrier does.
            // This makes the retention loud instead of silent.
            if points_at_survivor {
                assert!(
                    self.cards.is_dirty_addr(obj.addr()),
                    "GC verify: DROPPED REMEMBERED-SET ENTRY — old-gen {} at {:p} (offset {off}) \
                     points into survivor-TO but its card is CLEAN after a minor. The card-cleaning \
                     re-arm missed it; the next minor will not scan this object and its survivor \
                     will be lost.",
                    info.name,
                    obj.0
                );
            }
            off += info.allocation_size(header_aux(hdr));
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
            // A major has FOUR from-spaces: the old one and the whole young gen
            // (eden + both survivors). All evacuate into the same old to-space,
            // which is what makes a major complete on its own and safe with a
            // non-empty young gen.
            if from.contains(ptr) || self.young_contains(ptr) {
                let new = unsafe { self.copy_or_forward(types, to, ptr) };
                unsafe { *slot = P::encode_ptr(new) };
            } else if self.verify && !to.contains(ptr) {
                // A traced slot pointing at no space at all is a stale pointer
                // from a previous cycle (or a scalar that decoded as a ref).
                panic!(
                    "GC verify: traced slot holds {bits:#x} -> {ptr:p}, outside both spaces \
                     (stale pointer or non-pointer decoded as a ref)"
                );
            }
        }
    }

    /// Copy `old` into the old gen at `to` (or return its existing forwarding
    /// target). `to` is the old to-space for a major and the ACTIVE old space
    /// for a minor's promotion — either way the destination is the old gen, so
    /// every placement extends the per-card object start index.
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
        // Keyed by OFFSET, not address, so the entry survives the flip that a
        // major ends with — and so one table serves both semi-spaces.
        self.cards.record_start(new as usize - to.base() as usize);
        new
    }

    /// Walk every live object — eden, the active survivor space, then the
    /// active old space: `visitor(obj, info)`. (The idle survivor-TO is empty
    /// between collections, so it is skipped.)
    ///
    /// # Safety
    /// All objects must have valid headers; no concurrent mutation.
    pub unsafe fn walk(&self, types: &[TypeInfo], visitor: &mut dyn FnMut(Gc, &TypeInfo)) {
        unsafe {
            self.walk_space(&self.nursery, types, visitor);
            self.walk_space(self.survivor_from(), types, visitor);
            self.walk_space(self.from_space(), types, visitor);
        }
    }

    unsafe fn walk_space(
        &self,
        space: &AtomicBumpAllocator,
        types: &[TypeInfo],
        visitor: &mut dyn FnMut(Gc, &TypeInfo),
    ) {
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

    /// Stage I re-points the window at the NURSERY. It was Stage D's active
    /// semi-space, and the D5 inline sequence is unchanged — it just bumps a
    /// different space. The consequence worth pinning: the nursery does not
    /// flip, so unlike Stage D the window is armed once and stays put across
    /// collections of both kinds. If the window ever tracked the old spaces
    /// again, emitted code would allocate straight into the old gen and every
    /// barrier-free initializing store would become a missed edge.
    #[test]
    fn alloc_window_follows_the_nursery() {
        let (h, t) = small_heap();
        h.arm_window(); // the heap is at its final address now
        let base0 = h.window.base.load(Ordering::Acquire);
        assert_eq!(base0, h.nursery().base(), "window points at the nursery");
        assert_ne!(base0, h.from_space().base(), "…and not at an old space");
        let mut root = imm(0);
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        assert_eq!(
            h.window.base.load(Ordering::Acquire),
            base0,
            "a major flips the OLD spaces; the nursery — and so the window — stays put"
        );
        unsafe {
            h.collect_minor::<TestPolicy>(&t, &mut |visit| visit(&mut root as *mut u64));
        }
        assert_eq!(h.window.base.load(Ordering::Acquire), base0, "a minor does not move the nursery");
        // The window's cursor is the nursery's, so emitted code and `alloc`
        // bump the same word.
        assert_eq!(
            h.window.cursor.load(Ordering::Acquire) as usize,
            &h.nursery().cursor as *const AtomicUsize as usize
        );
        // gc-stress: limit 0 forces the slow path in emitted code (D5 uses
        // this; here we just pin the ABI field).
        h.window.limit.store(0, Ordering::Release);
        assert_eq!(h.window.limit.load(Ordering::Acquire), 0);
    }

    // ─── Stage I: generational ───────────────────────────────────────────

    /// Run a minor with `roots` as the entire root set, rewriting them.
    fn minor(h: &Heap, t: &[TypeInfo], roots: &mut [u64]) -> MinorOutcome {
        unsafe {
            h.collect_minor::<TestPolicy>(t, &mut |visit| {
                for r in roots.iter_mut() {
                    visit(r as *mut u64);
                }
            })
        }
    }

    /// Every dirty card, ascending — the collector's own view of the
    /// remembered set.
    fn dirty_cards(h: &Heap) -> Vec<usize> {
        let mut v = Vec::new();
        unsafe { h.cards.for_each_dirty(h.old_used(), |c| v.push(c)) };
        v
    }

    /// Stage M: a first-survival object ages into the SURVIVOR space, NOT the
    /// old gen (that is the whole fix — promote-on-first-survival over-tenured
    /// mid-iteration transients). The garbage still dies; the survivor is
    /// evacuated out of eden whole, now carrying age 1.
    #[test]
    fn minor_moves_survivors_to_the_survivor_space_and_drops_the_rest() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let live = h.alloc(cons, 0);
        unsafe { live.set_field(0, imm(7)) };
        for _ in 0..50 {
            h.alloc(cons, 0); // garbage: reachable from nothing
        }
        assert!(h.nursery_used() > 50 * cons.allocation_size(0));
        let mut roots = [enc(live)];
        let out = minor(&h, &t, &mut roots);

        assert_eq!(out.promoted_bytes, 0, "a first-survival object is NOT promoted to old");
        assert_eq!(h.old_used(), 0, "the old gen is untouched");
        assert_eq!(h.survivor_used(), cons.allocation_size(0), "only the live cons survives, in survivor");
        assert_ne!(roots[0], enc(live), "the root is rewritten to the survivor address");
        assert!(h.nursery_contains(live.0), "the original address was in eden…");
        assert!(!h.nursery_contains(dec(roots[0]).0), "…and the survivor lives elsewhere");
        assert_eq!(h.nursery_used(), 0, "a minor evacuates eden WHOLE");
        unsafe {
            assert_eq!(dec(roots[0]).field(0), imm(7), "contents came along");
            assert_eq!(dec(roots[0]).age(), 1, "and its age advanced to 1");
        }
        assert_eq!(h.minor_collections.load(Ordering::Relaxed), 1);
        assert_eq!(h.major_collections.load(Ordering::Relaxed), 0);
        assert_eq!(h.promoted_bytes.load(Ordering::Relaxed), 0);
    }

    /// Evacuation is transitive: a survivor's fields still name eden objects,
    /// and the Cheney scan over survivor-TO must chase them. Without it only `a`
    /// moves and `b`/`c` are left in a space that is about to be recycled.
    #[test]
    fn minor_evacuates_transitively() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        // c is 3 hops from the only root, and each hop is discovered only by
        // scanning the object copied on the hop before.
        let c = h.alloc(cons, 0);
        let b = h.alloc(cons, 0);
        let a = h.alloc(cons, 0);
        unsafe {
            c.set_field(0, imm(3));
            b.set_field(0, imm(2));
            b.set_field(1, enc(c));
            a.set_field(0, imm(1));
            a.set_field(1, enc(b));
        }
        let mut roots = [enc(a)];
        let out = minor(&h, &t, &mut roots);
        assert_eq!(out.promoted_bytes, 0, "first survival: the chain ages into survivor, not old");
        assert_eq!(h.survivor_used(), 3 * cons.allocation_size(0), "the whole chain survived");
        assert_eq!(h.nursery_used(), 0);
        unsafe {
            let na = dec(roots[0]);
            let nb = dec(na.field(1));
            let nc = dec(nb.field(1));
            for (o, want) in [(na, imm(1)), (nb, imm(2)), (nc, imm(3))] {
                assert!(!h.nursery_contains(o.0), "every hop is a survivor now");
                assert_eq!(o.field(0), want);
            }
        }
    }

    /// THE reason the write barrier exists. `young` is reachable ONLY from an
    /// old object's field — no root names it, so the root enumeration cannot
    /// find it and the minor does not scan old objects. The dirty card is the
    /// only path to it.
    #[test]
    fn dirty_card_finds_an_edge_no_root_names() {
        // tenure 0 = promote-on-first-survival: this test is about the OLD→young
        // dirty-card path, so it needs the holder (and the found edge) in the
        // old gen, not the survivor space. Aging is exercised elsewhere.
        let (mut h, t) = small_heap();
        h.set_tenure(0);
        let cons = &t[kind::CONS as usize];
        let mut roots = [enc(h.alloc(cons, 0))];
        minor(&h, &t, &mut roots);
        let old = dec(roots[0]);
        assert!(!h.nursery_contains(old.0), "the holder is old now");

        let young = h.alloc(cons, 0);
        unsafe {
            young.set_field(0, imm(99));
            old.set_field(1, enc(young)); // the old→young store…
        }
        h.write_barrier(old); // …and the barrier that remembers it
        assert_eq!(dirty_cards(&h), vec![0], "the barrier marked the holder's card");

        let out = minor(&h, &t, &mut roots);
        assert_eq!(out.promoted_bytes, cons.allocation_size(0), "the young object was promoted");
        unsafe {
            let promoted = dec(dec(roots[0]).field(1));
            assert!(!h.nursery_contains(promoted.0), "…into the old gen");
            assert_eq!(promoted.field(0), imm(99), "…with its contents intact");
        }
    }

    /// The missed-barrier detector, on an old→EDEN store: the SAME store as
    /// above with the barrier left out. This is beagle's shipped bug, and it
    /// must die at the collection that caused it rather than as corruption
    /// somewhere later.
    #[test]
    #[should_panic(expected = "EDEN pointer")]
    fn missed_barrier_detector_fires_on_an_eden_store() {
        let (mut h, t) = small_heap();
        h.set_tenure(0); // the holder must be OLD for the store to be old→young
        let cons = &t[kind::CONS as usize];
        let mut roots = [enc(h.alloc(cons, 0))];
        minor(&h, &t, &mut roots);
        let old = dec(roots[0]);

        let young = h.alloc(cons, 0); // in eden
        unsafe { old.set_field(1, enc(young)) }; // no write_barrier: the bug
        minor(&h, &t, &mut roots);
    }

    /// The re-aimed detector also fires on an old→SURVIVOR-FROM store. Stage M
    /// made old→survivor-TO edges LEGAL (a fresh survivor held by an old
    /// object, card-marked), but an old object pointing into the JUST-EVACUATED
    /// survivor-FROM is the same missed-barrier bug: that region is empty now.
    #[test]
    #[should_panic(expected = "SURVIVOR-FROM pointer")]
    fn missed_barrier_detector_fires_on_a_survivor_from_store() {
        let (mut h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        // An OLD holder (promote it directly with tenure 0)…
        h.set_tenure(0);
        let mut roots = [enc(h.alloc(cons, 0))];
        minor(&h, &t, &mut roots);
        let old = dec(roots[0]);
        assert_eq!(h.old_used(), cons.allocation_size(0), "holder is old");
        // …and a young object that becomes a SURVIVOR (default aging).
        h.set_tenure(TENURE_THRESHOLD);
        let mut sroots = vec![roots[0], enc(h.alloc(cons, 0))];
        minor(&h, &t, &mut sroots);
        let survivor = dec(sroots[1]);
        assert_eq!(h.survivor_used(), cons.allocation_size(0), "the young object is a survivor now");

        // The old→survivor store, WITHOUT the barrier: the bug. The survivor is
        // reachable ONLY through this unbarriered edge, so the next minor cannot
        // find it and leaves it in the evacuated survivor-FROM → detector fires.
        unsafe { old.set_field(1, enc(survivor)) };
        let mut froots = [sroots[0]];
        minor(&h, &t, &mut froots);
    }

    /// The scan reads 8 card bytes as one word and skips the run on zero, so
    /// the indices that can go missing are the ones at word edges — and the
    /// ragged tail, which the word loop never reaches. 100 cards = 12 whole
    /// words + a 4-card tail.
    #[test]
    fn dirty_scan_finds_cards_at_word_boundaries_and_in_the_tail() {
        let ct = CardTable::new(0x1000, 100 * CARD_SIZE);
        let want = [0usize, 7, 8, 15, 16, 88, 95, 96, 99];
        for &c in &want {
            ct.cards[c].store(CARD_DIRTY, Ordering::Relaxed);
        }
        let mut seen = Vec::new();
        unsafe { ct.for_each_dirty(100 * CARD_SIZE, |c| seen.push(c)) };
        assert_eq!(seen, want, "every dirty card, ascending, across word edges and the tail");

        // …and an all-clean table yields nothing (the skip-the-run path).
        let ct = CardTable::new(0x1000, 100 * CARD_SIZE);
        let mut n = 0;
        unsafe { ct.for_each_dirty(100 * CARD_SIZE, |_| n += 1) };
        assert_eq!(n, 0);
    }

    /// The barrier's single unsigned compare must reject every address that is
    /// not in the old gen — below the base (which wraps to a huge offset) and
    /// at or past the limit alike. I2 routes EVERY field store through it, so
    /// the overwhelmingly common call is a store into a young object, which has
    /// nothing to remember; a compare that let those through would dirty
    /// arbitrary cards and turn `starts` misses into spurious hard panics.
    #[test]
    fn barrier_bounds_are_one_unsigned_compare() {
        let base = 0x10_000usize;
        let ct = CardTable::new(base, 64 * CARD_SIZE);
        ct.mark(base - 8); // below: wraps
        ct.mark(base + 64 * CARD_SIZE); // at the limit
        ct.mark(base + 64 * CARD_SIZE + 4096); // past it
        ct.mark(0);
        ct.mark(usize::MAX);
        let mut n = 0;
        unsafe { ct.for_each_dirty(64 * CARD_SIZE, |_| n += 1) };
        assert_eq!(n, 0, "only old-gen addresses may dirty a card");

        ct.mark(base); // first byte in range
        ct.mark(base + 64 * CARD_SIZE - 1); // last
        let mut seen = Vec::new();
        unsafe { ct.for_each_dirty(64 * CARD_SIZE, |c| seen.push(c)) };
        assert_eq!(seen, vec![0, 63], "…and both ends of the range do");
    }

    /// The same property on a real heap: barriering a young object (eden OR
    /// survivor) is a no-op, barriering the promoted one is not. tenure 0 puts
    /// the object in the old gen after one minor so the second half has teeth.
    #[test]
    fn barrier_on_a_young_object_remembers_nothing() {
        let (mut h, t) = small_heap();
        h.set_tenure(0);
        let cons = &t[kind::CONS as usize];
        let young = h.alloc(cons, 0);
        h.write_barrier(young);
        assert_eq!(dirty_cards(&h), Vec::<usize>::new(), "a young store has nothing to remember");

        let mut roots = [enc(young)];
        minor(&h, &t, &mut roots);
        h.write_barrier(dec(roots[0]));
        assert_eq!(dirty_cards(&h), vec![0], "the same object, once old, does");
    }

    /// The same boundaries, end to end: real old objects spread over cards
    /// either side of the scan's word edges, each holding the only reference
    /// to a young object.
    #[test]
    fn dirty_cards_across_word_boundaries_promote_their_edges() {
        // tenure 0: the 400 holders and the edges they name all live in the OLD
        // gen, which is what the card-address arithmetic below depends on.
        let (mut h, t) = small_heap();
        h.set_tenure(0);
        let cons = &t[kind::CONS as usize];
        // ~9.6 KiB of old gen = cards 0..18, enough to straddle two word edges.
        let mut roots: Vec<u64> = (0..400)
            .map(|i| {
                let c = h.alloc(cons, 0);
                unsafe { c.set_field(0, imm(i)) };
                enc(c)
            })
            .collect();
        minor(&h, &t, &mut roots);
        assert_eq!(h.nursery_used(), 0);

        let base = h.old_base() as usize;
        let card_of = |bits: u64| (dec(bits).addr() - base) >> CARD_SHIFT;
        // Cards 7/8 and 15/16 straddle the scan's word edges; 18 is the last.
        let targets: Vec<u64> = [0usize, 1, 7, 8, 9, 15, 16, 18]
            .iter()
            .filter_map(|&want| roots.iter().copied().find(|&r| card_of(r) == want))
            .collect();
        assert_eq!(targets.len(), 8, "test needs an old object in each probed card");

        for (i, &tgt) in targets.iter().enumerate() {
            let young = h.alloc(cons, 0);
            unsafe {
                young.set_field(0, imm(1000 + i as u64));
                dec(tgt).set_field(1, enc(young));
            }
            h.write_barrier(dec(tgt));
        }
        assert_eq!(
            dirty_cards(&h),
            vec![0, 1, 7, 8, 9, 15, 16, 18],
            "one dirty card per barriered object's base"
        );

        let out = minor(&h, &t, &mut roots);
        assert_eq!(out.promoted_bytes, targets.len() * cons.allocation_size(0));
        for (i, &tgt) in targets.iter().enumerate() {
            unsafe {
                let child = dec(dec(tgt).field(1));
                assert!(!h.nursery_contains(child.0), "card {} edge missed", card_of(tgt));
                assert_eq!(child.field(0), imm(1000 + i as u64));
            }
        }
    }

    #[test]
    fn minor_leaves_the_nursery_empty_and_every_card_clean() {
        // tenure 0 so the 100 objects are OLD after one minor and can be
        // barriered (a survivor store has nothing to remember).
        let (mut h, t) = small_heap();
        h.set_tenure(0);
        let cons = &t[kind::CONS as usize];
        let mut roots: Vec<u64> = (0..100).map(|_| enc(h.alloc(cons, 0))).collect();
        minor(&h, &t, &mut roots);
        for &r in &roots {
            h.write_barrier(dec(r));
        }
        assert!(!dirty_cards(&h).is_empty(), "the barrier marked cards to begin with");

        minor(&h, &t, &mut roots);
        assert_eq!(h.nursery_used(), 0, "eden empty");
        assert_eq!(dirty_cards(&h), Vec::<usize>::new(), "and every card clean");
        // A major clears them too, and rebuilds the index against the new space.
        for &r in &roots {
            h.write_barrier(dec(r));
        }
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| {
                for r in roots.iter_mut() {
                    visit(r as *mut u64);
                }
            });
        }
        assert_eq!(dirty_cards(&h), Vec::<usize>::new(), "a major scans everything: no card survives");
        assert_eq!(h.nursery_used(), 0);
    }

    /// The card→object lookup rests on the barrier marking OBJECT BASES. Mark
    /// a card in the middle of a multi-card object — what barriering a field
    /// address would do — and the scan must panic rather than silently skip a
    /// card that might have held a real edge.
    #[test]
    #[should_panic(expected = "has no object start in the index")]
    fn dirty_card_without_an_object_start_is_a_hard_panic() {
        // tenure 0 so the big object promotes to the OLD gen (where cards and
        // the object-start index live) on its first minor.
        let (mut h, t) = small_heap();
        h.set_tenure(0);
        let vinfo = &t[kind::VALUES as usize];
        let big = h.alloc(vinfo, 200); // 1608 bytes: begins in card 0, ends in card 3
        let mut roots = [enc(big)];
        minor(&h, &t, &mut roots);
        assert!(h.old_used() > 3 * CARD_SIZE, "the object must span cards for this to mean anything");

        h.cards.mark(h.old_base() as usize + 2 * CARD_SIZE + 8);
        minor(&h, &t, &mut roots);
    }

    #[test]
    fn minor_preserves_cycles() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let a = h.alloc(cons, 0);
        let b = h.alloc(cons, 0);
        unsafe {
            a.set_field(0, imm(1));
            a.set_field(1, enc(b));
            b.set_field(0, imm(2));
            b.set_field(1, enc(a)); // cycle: the forwarding bit is what stops it
        }
        let mut roots = [enc(a)];
        let out = minor(&h, &t, &mut roots);
        assert_eq!(out.promoted_bytes, 0, "a first-survival cycle goes to survivor, not old");
        assert_eq!(h.survivor_used(), 2 * cons.allocation_size(0), "each object copied once");
        unsafe {
            let na = dec(roots[0]);
            let nb = dec(na.field(1));
            assert_eq!(na.field(0), imm(1));
            assert_eq!(nb.field(0), imm(2));
            assert_eq!(dec(nb.field(1)), na, "the cycle closes back on the copied a");
        }
    }

    /// A shared child reached twice must be copied once — the forwarding
    /// header, through the survivor-evacuation path this time.
    #[test]
    fn minor_evacuates_a_shared_child_once() {
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
        let mut roots = [enc(a), enc(b)];
        let out = minor(&h, &t, &mut roots);
        assert_eq!(out.promoted_bytes, 0, "first survival: to survivor, not old");
        assert_eq!(h.survivor_used(), 3 * cons.allocation_size(0), "shared child copied exactly once");
        unsafe {
            let sa = dec(dec(roots[0]).field(0));
            let sb = dec(dec(roots[1]).field(0));
            assert_eq!(sa, sb, "shared child copied exactly once");
            assert_eq!(sa.field(0), imm(9));
        }
    }

    #[test]
    fn minor_moves_varlen_objects_intact() {
        let (h, t) = small_heap();
        let vinfo = &t[kind::VALUES as usize];
        let sinfo = &t[kind::STR as usize];
        let cons = &t[kind::CONS as usize];
        let child = h.alloc(cons, 0);
        unsafe { child.set_field(0, imm(77)) };
        let s = h.alloc(sinfo, 11);
        unsafe { s.bytes_mut(sinfo).copy_from_slice(b"hello world") };
        let v = h.alloc(vinfo, 4);
        unsafe { v.values_mut(vinfo).copy_from_slice(&[imm(1), enc(child), enc(s), imm(4)]) };

        let mut roots = [enc(v)];
        minor(&h, &t, &mut roots);
        assert_eq!(h.nursery_used(), 0);
        unsafe {
            let nv = dec(roots[0]);
            assert_eq!(nv.aux(), 4, "varlen length rides the header through promotion");
            let vals = nv.values(vinfo);
            assert_eq!(vals[0], imm(1));
            assert_eq!(vals[3], imm(4));
            // The Values tail is traced: both refs promoted…
            assert_eq!(dec(vals[1]).field(0), imm(77));
            // …and the Bytes tail is not, but still copies verbatim.
            assert_eq!(dec(vals[2]).bytes(sinfo), b"hello world");
        }
    }

    /// Promotion cannot fail half-way — the graph is already part-forwarded —
    /// so the caller checks `minor_will_fit` first. `collect` is the answer
    /// when it says no: it reclaims the old garbage AND the nursery in one
    /// pass, so it makes progress where a minor could not.
    #[test]
    fn a_minor_that_would_not_fit_defers_to_a_major() {
        // 4 KiB old spaces, 64 KiB nursery: the nursery outgrows the old gen.
        let h = {
            let mut h = Heap::with_sizes(1 << 12, 1 << 16);
            h.set_verify(true);
            h
        };
        let t = type_table();
        let cons = &t[kind::CONS as usize];
        assert!(h.minor_will_fit(), "an empty nursery always fits");
        let live = h.alloc(cons, 0);
        unsafe { live.set_field(0, imm(5)) };
        for _ in 0..300 {
            h.alloc(cons, 0); // 7.2 KiB of garbage: more than the old space holds
        }
        assert!(!h.minor_will_fit(), "the bound is nursery USED, not live — we cannot know live yet");

        let mut roots = [enc(live)];
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| visit(&mut roots[0] as *mut u64));
        }
        assert_eq!(h.nursery_used(), 0, "the major evacuated the nursery too");
        assert_eq!(h.old_used(), cons.allocation_size(0), "…keeping only what was live");
        unsafe { assert_eq!(dec(roots[0]).field(0), imm(5)) };
        assert!(h.minor_will_fit(), "and minors are viable again");
    }

    #[test]
    #[should_panic(expected = "could not all spill")]
    fn calling_a_minor_that_cannot_fit_is_loud() {
        let h = Heap::with_sizes(1 << 12, 1 << 16);
        let t = type_table();
        let cons = &t[kind::CONS as usize];
        let mut roots = [enc(h.alloc(cons, 0))];
        for _ in 0..300 {
            h.alloc(cons, 0);
        }
        minor(&h, &t, &mut roots);
    }

    /// The other half of the policy: a minor that fits still reports when
    /// promotion has filled the old gen past its threshold, so the caller can
    /// follow with a major. The young gen is empty by then, so it is safe.
    /// `needs_major` keys on the OLD gen, so this needs promotion to old —
    /// tenure 0 gives it (a survivor that never promotes could never trip it).
    #[test]
    fn minor_reports_when_the_old_gen_needs_a_major() {
        let (mut h, t) = small_heap();
        h.set_tenure(0);
        let cons = &t[kind::CONS as usize];
        h.set_old_trigger_bytes(1 << 20); // far away
        let mut roots = [enc(h.alloc(cons, 0))];
        assert!(!minor(&h, &t, &mut roots).needs_major);

        h.set_old_trigger_bytes(0); // anything promoted is over
        let mut roots = vec![roots[0], enc(h.alloc(cons, 0))];
        assert!(minor(&h, &t, &mut roots).needs_major, "old gen over threshold");
        // And the major it asks for runs cleanly on the (now empty) nursery.
        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| {
                for r in roots.iter_mut() {
                    visit(r as *mut u64);
                }
            });
        }
        assert_eq!(h.old_used(), 2 * cons.allocation_size(0));
    }

    /// A minor moves nothing that is ALREADY OLD — the old gen is untouched
    /// until a major. (A SURVIVOR does move across minors, until it tenures;
    /// that is covered by the aging tests. Here tenure 0 puts the object in the
    /// old gen after one minor so its address must then stay fixed.)
    #[test]
    fn minors_do_not_move_old_objects() {
        let (mut h, t) = small_heap();
        h.set_tenure(0);
        let cons = &t[kind::CONS as usize];
        let mut roots = [enc(h.alloc(cons, 0))];
        minor(&h, &t, &mut roots);
        let settled = roots[0];
        for _ in 0..5 {
            h.alloc(cons, 0);
            minor(&h, &t, &mut roots);
            assert_eq!(roots[0], settled, "an old object keeps its address across minors");
        }
        assert_eq!(h.old_used(), cons.allocation_size(0), "and nothing re-promotes it");
        assert_eq!(h.minor_collections.load(Ordering::Relaxed), 6);
    }

    /// Many minors against a small nursery: the old gen grows only by what
    /// actually survives, and the nursery is reused every time — an
    /// append-only nursery would have blown 64 KiB many times over.
    #[test]
    fn nursery_is_reused_across_many_minors() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let mut roots = [enc(h.alloc(cons, 0))];
        unsafe { dec(roots[0]).set_field(0, imm(5)) };
        for round in 0..50 {
            for _ in 0..500 {
                h.alloc(cons, 0); // 12 KiB of garbage per round
            }
            minor(&h, &t, &mut roots);
            assert_eq!(h.nursery_used(), 0, "round {round}");
            // Only the root ever survives — early rounds in the survivor space,
            // later ones (once it tenures) in the old gen, but always exactly
            // one cons live somewhere.
            assert_eq!(
                h.survivor_used() + h.old_used(),
                cons.allocation_size(0),
                "round {round}: only the root ever survived, wherever it now lives"
            );
        }
        unsafe { assert_eq!(dec(roots[0]).field(0), imm(5)) };
        // By now it long since crossed TENURE and settled in the old gen.
        assert_eq!(h.old_used(), cons.allocation_size(0), "tenured to old and stays put");
        assert_eq!(h.survivor_used(), 0);
        assert_eq!(
            h.promoted_bytes.load(Ordering::Relaxed),
            cons.allocation_size(0) as u64,
            "promoted exactly once, when it crossed TENURE"
        );
    }

    /// The precise-layout detector covers the promotion path too, not just the
    /// major's.
    #[test]
    #[should_panic(expected = "precise-layout violation")]
    fn minor_catches_a_scalar_in_a_traced_slot() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let c = h.alloc(cons, 0);
        let bogus = TestPolicy::encode_ptr(unsafe { c.0.add(16) });
        unsafe { c.set_field(0, bogus) };
        let mut roots = [enc(c)];
        minor(&h, &t, &mut roots);
    }

    /// The nursery is recycled, so a stale young pointer must not read as a
    /// live object — the zeroed header reads `kind::INVALID`, which every
    /// reader panics on — AND `alloc`'s zeroed-fields contract must still hold
    /// for the next object to claim that memory. `alloc_closure` leans on the
    /// second half directly: an unwritten code word must read 0 or the JIT's
    /// `code != 0` guard calls into whatever is there (the Stage-D reset bug
    /// this pins against; see `reset_nursery_space`).
    #[test]
    fn recycled_nursery_reads_invalid_and_stays_zeroed() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let stale = h.alloc(cons, 0);
        unsafe { stale.set_field(0, imm(1234)) };
        let mut roots = [imm(0)]; // nothing live
        minor(&h, &t, &mut roots);
        unsafe {
            // Read the recycled header BEFORE anything reclaims the address.
            assert_eq!(stale.type_id(), kind::INVALID, "a recycled header reads INVALID");
        }
        // The next object claims that same memory and must find it zeroed.
        let fresh = h.alloc(cons, 0);
        assert_eq!(fresh.addr(), stale.addr(), "the nursery restarted at zero");
        unsafe {
            assert_eq!(fresh.type_id(), kind::CONS);
            assert_eq!(fresh.field(0), 0, "unwritten fields read 0 (alloc's contract)");
            assert_eq!(fresh.field(1), 0);
        }
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

    // ─── Stage M: survivor space + aging ─────────────────────────────────

    /// An object is promoted to the old gen ONLY after surviving `tenure`
    /// minors — the age threshold. It ages one step per minor in the survivor
    /// space and never touches the old gen until then. Break `age >= tenure`
    /// (e.g. to `>`) and the final promotion never happens: this fails.
    #[test]
    fn an_object_is_promoted_only_after_surviving_tenure_minors() {
        let (h, t) = small_heap();
        assert_eq!(h.tenure(), TENURE_THRESHOLD, "the default tenure is in effect");
        let cons = &t[kind::CONS as usize];
        let live = h.alloc(cons, 0);
        unsafe { live.set_field(0, imm(7)) };
        let mut roots = [enc(live)];

        // Minors 1..=TENURE: survives in the survivor space, ageing each time.
        for age in 1..=TENURE_THRESHOLD {
            for _ in 0..10 {
                h.alloc(cons, 0); // fresh garbage each round
            }
            let out = minor(&h, &t, &mut roots);
            assert_eq!(out.promoted_bytes, 0, "age {age}: still a survivor, not promoted");
            assert_eq!(h.old_used(), 0, "age {age}: the old gen is still empty");
            assert_eq!(h.survivor_used(), cons.allocation_size(0), "age {age}: held in survivor");
            unsafe { assert_eq!(dec(roots[0]).age(), age, "age advances by one per minor") };
        }

        // The next minor finds it at age == TENURE and promotes it.
        let out = minor(&h, &t, &mut roots);
        assert_eq!(out.promoted_bytes, cons.allocation_size(0), "promoted once age reached TENURE");
        assert_eq!(h.old_used(), cons.allocation_size(0), "now in the old gen");
        assert_eq!(h.survivor_used(), 0, "and no longer a survivor");
        unsafe { assert_eq!(dec(roots[0]).field(0), imm(7), "contents intact through it all") };
    }

    /// A short-lived object survives one minor (into the survivor space), then
    /// is dropped — it dies in survivor WITHOUT ever reaching the old gen.
    /// That is the whole win: promote-on-first-survival would have tenured it.
    #[test]
    fn a_short_lived_object_dies_in_survivor_without_reaching_old() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let mut root = [enc(h.alloc(cons, 0))]; // a long-lived root, always kept

        let transient = h.alloc(cons, 0);
        unsafe { transient.set_field(0, imm(42)) };
        let mut both = [root[0], enc(transient)];
        minor(&h, &t, &mut both); // transient → survivor (age 1)
        assert_eq!(h.survivor_used(), 2 * cons.allocation_size(0), "root + transient both survive");
        assert_eq!(h.old_used(), 0);
        root[0] = both[0];

        // Drop the transient (not in the root set) and collect again.
        minor(&h, &t, &mut root);
        assert_eq!(h.survivor_used(), cons.allocation_size(0), "only the root survives the second minor");
        assert_eq!(h.old_used(), 0, "the transient NEVER reached the old gen — it died in survivor");
        assert_eq!(h.promoted_bytes.load(Ordering::Relaxed), 0, "nothing was ever promoted");
    }

    /// survivor-TO overflow SPILLS to the old gen: a minor must always complete,
    /// so when the survivor space fills the remaining survivors promote early
    /// rather than the collector getting stuck on a half-forwarded graph. With
    /// a 64-byte survivor only a couple of conses fit; the rest spill. Nothing
    /// is lost, and every object is intact wherever it landed.
    #[test]
    fn survivor_overflow_spills_to_the_old_gen() {
        let mut h = Heap::with_sizes_survivor(1 << 16, 1 << 16, 64);
        h.set_verify(true);
        let t = type_table();
        let cons = &t[kind::CONS as usize];
        let n = 20usize;
        let sz = cons.allocation_size(0);
        let mut roots: Vec<u64> = (0..n)
            .map(|i| {
                let c = h.alloc(cons, 0);
                unsafe { c.set_field(0, imm(i as u64)) };
                enc(c)
            })
            .collect();
        let out = minor(&h, &t, &mut roots);

        assert!(h.survivor_used() > 0, "survivor took some");
        assert!(h.survivor_used() < n * sz, "…but not all — the rest had to spill");
        assert_eq!(out.promoted_bytes, n * sz - h.survivor_used(), "the overflow spilled to old");
        assert_eq!(h.survivor_used() + h.old_used(), n * sz, "every live object landed somewhere");
        assert_eq!(h.nursery_used(), 0, "eden emptied whole");
        for (i, &r) in roots.iter().enumerate() {
            unsafe { assert_eq!(dec(r).field(0), imm(i as u64), "contents intact across the split") };
        }
    }

    /// Age is PRESERVED and INCREMENTED across a minor — read it back — and the
    /// bump must not disturb the rest of the header (type_id, and a varlen
    /// object's aux length). The age write lands on the NEW copy, never the
    /// forwarding word.
    #[test]
    fn age_is_preserved_and_incremented_across_a_minor() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let live = h.alloc(cons, 0);
        unsafe { assert_eq!(live.age(), 0, "born at age 0") };
        let mut roots = [enc(live)];
        minor(&h, &t, &mut roots);
        unsafe { assert_eq!(dec(roots[0]).age(), 1, "the first minor incremented age to 1") };
        minor(&h, &t, &mut roots);
        unsafe { assert_eq!(dec(roots[0]).age(), 2, "and the second to 2") };

        // A varlen object: its aux (length) must ride through the age bump.
        let vinfo = &t[kind::VALUES as usize];
        let v = h.alloc(vinfo, 5);
        unsafe { v.values_mut(vinfo).copy_from_slice(&[imm(1), imm(2), imm(3), imm(4), imm(5)]) };
        let mut vroots = [roots[0], enc(v)];
        minor(&h, &t, &mut vroots);
        unsafe {
            let nv = dec(vroots[1]);
            assert_eq!(nv.age(), 1, "the varlen object aged too");
            assert_eq!(nv.type_id(), kind::VALUES, "type_id preserved across the age bump");
            assert_eq!(nv.aux(), 5, "aux (varlen length) preserved across the age bump");
            assert_eq!(nv.values(vinfo), &[imm(1), imm(2), imm(3), imm(4), imm(5)], "contents intact");
        }
    }

    /// A MAJOR evacuates all three young/old regions at once — eden, the
    /// survivor space, and the old gen — into the fresh old space. Build one
    /// live object in each, then collect: all three end up old, intact, and
    /// eden + survivor are empty.
    #[test]
    fn a_major_evacuates_eden_survivor_and_old_together() {
        let (mut h, t) = small_heap();
        let cons = &t[kind::CONS as usize];

        // An OLD object (promote directly with tenure 0).
        h.set_tenure(0);
        let old_obj = h.alloc(cons, 0);
        unsafe { old_obj.set_field(0, imm(100)) };
        let mut roots = vec![enc(old_obj)];
        minor(&h, &t, &mut roots);
        assert_eq!(h.old_used(), cons.allocation_size(0), "old_obj is in the old gen");

        // A SURVIVOR object (default aging).
        h.set_tenure(TENURE_THRESHOLD);
        let surv = h.alloc(cons, 0);
        unsafe { surv.set_field(0, imm(200)) };
        roots.push(enc(surv));
        minor(&h, &t, &mut roots);
        assert_eq!(h.survivor_used(), cons.allocation_size(0), "surv is in the survivor space");

        // An EDEN object (no collection since).
        let eden = h.alloc(cons, 0);
        unsafe { eden.set_field(0, imm(300)) };
        roots.push(enc(eden));
        assert!(
            h.nursery_used() > 0 && h.survivor_used() > 0 && h.old_used() > 0,
            "one live object in each of the three regions"
        );

        unsafe {
            h.collect::<TestPolicy>(&t, &mut |visit| {
                for r in roots.iter_mut() {
                    visit(r as *mut u64);
                }
            });
        }
        assert_eq!(h.nursery_used(), 0, "eden emptied by the major");
        assert_eq!(h.survivor_used(), 0, "survivor emptied by the major");
        assert_eq!(h.old_used(), 3 * cons.allocation_size(0), "all three are old now");
        unsafe {
            assert_eq!(dec(roots[0]).field(0), imm(100), "old object intact");
            assert_eq!(dec(roots[1]).field(0), imm(200), "survivor evacuated intact");
            assert_eq!(dec(roots[2]).field(0), imm(300), "eden object evacuated intact");
        }
    }

    /// Cycles AND varlen objects survive AGING: several minors re-copy the whole
    /// live survivor set, and on the last one the graph tenures to the old gen.
    /// A broken forwarding or age path corrupts the cycle on any pass.
    #[test]
    fn cycles_and_varlen_survive_aging() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        let vinfo = &t[kind::VALUES as usize];
        let a = h.alloc(cons, 0);
        let b = h.alloc(cons, 0);
        let v = h.alloc(vinfo, 3);
        unsafe {
            a.set_field(0, imm(1));
            a.set_field(1, enc(b));
            b.set_field(0, imm(2));
            b.set_field(1, enc(a)); // cycle a <-> b
            v.values_mut(vinfo).copy_from_slice(&[enc(a), imm(9), enc(b)]);
        }
        let mut roots = [enc(v)];
        for round in 1..=TENURE_THRESHOLD + 1 {
            for _ in 0..5 {
                h.alloc(cons, 0); // garbage
            }
            minor(&h, &t, &mut roots);
            assert_eq!(h.nursery_used(), 0, "round {round}: eden empty");
            unsafe {
                let nv = dec(roots[0]);
                let vals = nv.values(vinfo);
                let na = dec(vals[0]);
                let nb = dec(vals[2]);
                assert_eq!(vals[1], imm(9), "round {round}: untraced slot intact");
                assert_eq!(na.field(0), imm(1));
                assert_eq!(nb.field(0), imm(2));
                assert_eq!(dec(na.field(1)), nb, "round {round}: a→b");
                assert_eq!(dec(nb.field(1)), na, "round {round}: b→a closes on the re-copied a");
            }
        }
        // On the (TENURE+1)-th minor the whole graph reached age TENURE and
        // tenured to the old gen; it is intact there.
        assert_eq!(h.survivor_used(), 0, "tenured out of survivor");
        assert_eq!(
            h.old_used(),
            2 * cons.allocation_size(0) + vinfo.allocation_size(3),
            "the whole graph is in the old gen"
        );
    }

    /// The Stage M invariant: after a minor, eden and the JUST-EVACUATED
    /// survivor-FROM are BOTH empty (an exact evacuation still — a survivor is
    /// simply in survivor-TO now rather than the old gen). The survivor roles
    /// swapped, so the space we drained is the idle survivor-TO afterwards.
    #[test]
    fn eden_and_survivor_from_empty_after_a_minor() {
        let (h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        // Fill eden with survivors + garbage; keep some as roots.
        let mut roots: Vec<u64> = (0..10).map(|_| enc(h.alloc(cons, 0))).collect();
        for _ in 0..40 {
            h.alloc(cons, 0);
        }
        minor(&h, &t, &mut roots); // survivors now populate survivor-FROM
        assert!(h.survivor_used() > 0, "a survivor set exists to re-evacuate");
        for _ in 0..40 {
            h.alloc(cons, 0); // refill eden
        }

        let active_before = h.survivor_active.load(Ordering::Acquire);
        minor(&h, &t, &mut roots);
        // Eden empty, and the survivor space we evacuated FROM (now the idle TO
        // after the swap) empty. The survivors live in the new active FROM.
        assert_eq!(h.nursery_used(), 0, "eden empty");
        assert_eq!(
            h.survivor[active_before].used(),
            0,
            "the just-evacuated survivor-FROM is empty (exact evacuation)"
        );
        assert_ne!(
            h.survivor_active.load(Ordering::Acquire),
            active_before,
            "survivor roles swapped"
        );
        assert_eq!(h.survivor_used(), 10 * cons.allocation_size(0), "survivors are in the new FROM");
    }

    /// CARD CLEANING: an old→SURVIVOR edge must survive REPEATED minors. The
    /// survivor moves survivor→survivor on every minor until it tenures, and no
    /// STORE happens on a GC move, so the barrier cannot re-arm the card — the
    /// COLLECTOR must. Blind card-clearing (correct only when survivors went
    /// straight to old, as in Stage I) would lose this edge on the SECOND minor.
    /// The child is reachable ONLY through the old holder, so a lost card is a
    /// lost object — caught here by the value read-back and the detector alike.
    #[test]
    fn an_old_to_survivor_edge_is_kept_across_repeated_minors() {
        let (mut h, t) = small_heap();
        let cons = &t[kind::CONS as usize];
        // An OLD holder (tenure 0 to promote it directly)…
        h.set_tenure(0);
        let mut roots = [enc(h.alloc(cons, 0))];
        minor(&h, &t, &mut roots);
        let old = dec(roots[0]);
        assert_eq!(h.old_used(), cons.allocation_size(0), "holder is old");
        // …and a child that AGES in survivor for many minors (high tenure), held
        // ONLY by the old object, remembered by ONE barrier at store time.
        h.set_tenure(100);
        let young = h.alloc(cons, 0);
        unsafe {
            young.set_field(0, imm(55));
            old.set_field(1, enc(young)); // the old→young store…
        }
        h.write_barrier(old); // …remembered once; the collector re-arms it after

        for round in 0..5 {
            minor(&h, &t, &mut roots);
            assert_eq!(dec(roots[0]), old, "the old holder itself never moves");
            unsafe {
                let child = dec(old.field(1));
                assert!(!h.nursery_contains(child.0), "round {round}: child evacuated out of eden");
                assert_eq!(child.field(0), imm(55), "round {round}: child found via the retained card");
                assert_eq!(child.age(), round as u16 + 1, "round {round}: it is ageing in survivor");
            }
            assert_eq!(h.old_used(), cons.allocation_size(0), "round {round}: only the holder is old");
            assert!(h.survivor_used() > 0, "round {round}: the child lives on in survivor");
        }
    }
}
