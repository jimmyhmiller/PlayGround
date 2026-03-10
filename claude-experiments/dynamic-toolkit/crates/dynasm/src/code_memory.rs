use std::ptr;

/// Trait for managing regions of executable code.
///
/// Lifecycle:
/// 1. `push()` — append code bytes (memory is writable, not executable)
/// 2. `finalize()` — make all pushed code executable (W→X transition)
/// 3. `patch()` — modify already-executable code (temporarily W, then back to X)
///
/// Implementations **must** enforce W^X: no page is ever simultaneously
/// writable and executable.
pub trait CodeMemory {
    /// Append code bytes. Returns the offset where they were placed.
    /// The memory is writable but not yet executable.
    fn push(&mut self, code: &[u8]) -> usize;

    /// Make all pending code executable. Must be called before any pushed
    /// code can be run.
    fn finalize(&mut self);

    /// Pointer to the start of the code region.
    fn base_ptr(&self) -> *const u8;

    /// Number of bytes used.
    fn len(&self) -> usize;

    /// Whether no code has been pushed.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pointer to code at the given offset.
    fn ptr_at(&self, offset: usize) -> *const u8 {
        assert!(offset < self.len(), "offset {offset} out of bounds (len {})", self.len());
        unsafe { self.base_ptr().add(offset) }
    }

    /// Patch already-finalized code at `offset`. Temporarily makes the
    /// affected pages writable, writes the bytes, then re-protects as
    /// executable. Flushes the instruction cache on architectures that
    /// require it.
    fn patch(&mut self, offset: usize, bytes: &[u8]);
}

// ─── Default implementation: PagedCodeMemory ────────────────────────

const DEFAULT_CAPACITY: usize = 64 * 1024 * 1024; // 64 MB of virtual address space

/// A growable, W^X-enforcing code memory region.
///
/// Reserves a large virtual address range upfront (no physical memory
/// committed) and commits pages on demand as code is pushed. Pages
/// transition strictly between states:
///
/// ```text
///   PROT_NONE  →  PROT_READ|PROT_WRITE  →  PROT_READ|PROT_EXEC
///   (reserved)       (writable)                (executable)
/// ```
///
/// Patching temporarily moves pages back to RW, writes, then restores
/// RX. At no point is any page both writable and executable.
pub struct PagedCodeMemory {
    base: *mut u8,
    /// Total reserved virtual address space.
    capacity: usize,
    /// How many bytes of the reserved region are committed (accessible).
    /// Always page-aligned.
    committed: usize,
    /// How many bytes have been written via `push`.
    used: usize,
    /// Everything in `[0, executable_end)` is currently PROT_READ|PROT_EXEC.
    /// Always page-aligned.
    executable_end: usize,
    /// OS page size.
    page_size: usize,
}

impl PagedCodeMemory {
    /// Create a new code memory region with the default capacity (64 MB).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a new code memory region reserving `capacity` bytes of
    /// virtual address space. No physical memory is committed until
    /// code is pushed. `capacity` is rounded up to a page boundary.
    pub fn with_capacity(capacity: usize) -> Self {
        let page_size = Self::query_page_size();
        let capacity = page_align_up(capacity, page_size);
        assert!(capacity > 0, "capacity must be > 0");

        // Reserve address space with no access permissions.
        // This consumes virtual address space but no physical memory.
        let base = unsafe {
            libc::mmap(
                ptr::null_mut(),
                capacity,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert!(base != libc::MAP_FAILED, "mmap reservation failed");

        Self {
            base: base as *mut u8,
            capacity,
            committed: 0,
            used: 0,
            executable_end: 0,
            page_size,
        }
    }

    /// The OS page size.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Total reserved virtual address space.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes of committed (accessible) memory.
    pub fn committed(&self) -> usize {
        self.committed
    }

    /// Ensure at least `needed` bytes are committed as writable.
    fn ensure_committed(&mut self, needed: usize) {
        if needed <= self.committed {
            return;
        }
        assert!(needed <= self.capacity, "code memory exhausted: need {needed}, capacity {}", self.capacity);

        let new_committed = page_align_up(needed, self.page_size);
        let grow_start = self.committed;
        let grow_len = new_committed - grow_start;

        let ret = unsafe {
            libc::mprotect(
                self.base.add(grow_start) as *mut libc::c_void,
                grow_len,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        assert_eq!(ret, 0, "mprotect(RW) failed for new pages");

        self.committed = new_committed;
    }

    /// If `used` falls within the executable region, re-protect the tail
    /// pages as writable so we can continue pushing code.
    fn ensure_writable_tail(&mut self) {
        if self.used >= self.executable_end {
            return;
        }
        // The page containing `used` (and everything after it up to
        // committed) needs to be writable.
        let rw_start = page_align_down(self.used, self.page_size);
        let rw_end = self.committed;
        if rw_start < rw_end {
            let ret = unsafe {
                libc::mprotect(
                    self.base.add(rw_start) as *mut libc::c_void,
                    rw_end - rw_start,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
            assert_eq!(ret, 0, "mprotect(RW) failed for writable tail");
        }
        self.executable_end = rw_start;
    }

    fn query_page_size() -> usize {
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        assert!(ps > 0, "sysconf(_SC_PAGESIZE) failed");
        ps as usize
    }
}

impl Default for PagedCodeMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeMemory for PagedCodeMemory {
    fn push(&mut self, code: &[u8]) -> usize {
        if code.is_empty() {
            return self.used;
        }
        let offset = self.used;
        let needed = offset + code.len();

        self.ensure_committed(needed);
        self.ensure_writable_tail();

        unsafe {
            ptr::copy_nonoverlapping(code.as_ptr(), self.base.add(offset), code.len());
        }
        self.used = needed;
        offset
    }

    fn finalize(&mut self) {
        if self.used == self.executable_end {
            return; // nothing new to finalize
        }

        // Protect from the current executable boundary up through all
        // pages containing written code.
        let rx_start = self.executable_end; // already page-aligned
        let rx_end = page_align_up(self.used, self.page_size);

        if rx_end > rx_start {
            let ret = unsafe {
                libc::mprotect(
                    self.base.add(rx_start) as *mut libc::c_void,
                    rx_end - rx_start,
                    libc::PROT_READ | libc::PROT_EXEC,
                )
            };
            assert_eq!(ret, 0, "mprotect(RX) failed during finalize");

            flush_icache(unsafe { self.base.add(rx_start) }, rx_end - rx_start);
        }

        self.executable_end = rx_end;
    }

    fn base_ptr(&self) -> *const u8 {
        self.base as *const u8
    }

    fn len(&self) -> usize {
        self.used
    }

    fn patch(&mut self, offset: usize, bytes: &[u8]) {
        assert!(
            offset + bytes.len() <= self.executable_end,
            "patch region [{offset}, {}) is not fully finalized (executable_end = {})",
            offset + bytes.len(),
            self.executable_end,
        );

        let page_start = page_align_down(offset, self.page_size);
        let page_end = page_align_up(offset + bytes.len(), self.page_size);

        // Temporarily make writable.
        let ret = unsafe {
            libc::mprotect(
                self.base.add(page_start) as *mut libc::c_void,
                page_end - page_start,
                libc::PROT_READ | libc::PROT_WRITE,
            )
        };
        assert_eq!(ret, 0, "mprotect(RW) failed during patch");

        // Write patch bytes.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(offset), bytes.len());
        }

        // Restore executable.
        let ret = unsafe {
            libc::mprotect(
                self.base.add(page_start) as *mut libc::c_void,
                page_end - page_start,
                libc::PROT_READ | libc::PROT_EXEC,
            )
        };
        assert_eq!(ret, 0, "mprotect(RX) failed during patch");

        flush_icache(unsafe { self.base.add(offset) }, bytes.len());
    }
}

impl Drop for PagedCodeMemory {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                libc::munmap(self.base as *mut libc::c_void, self.capacity);
            }
        }
    }
}

// PagedCodeMemory is Send: the raw pointer is exclusively owned and
// the memory region is only accessed through &mut self.
unsafe impl Send for PagedCodeMemory {}

// ─── Helpers ────────────────────────────────────────────────────────

fn page_align_up(n: usize, page_size: usize) -> usize {
    (n + page_size - 1) & !(page_size - 1)
}

fn page_align_down(n: usize, page_size: usize) -> usize {
    n & !(page_size - 1)
}

/// Flush the instruction cache for the given range.
///
/// Required on architectures with non-coherent instruction caches
/// (e.g., ARM64). A no-op on x86-64 where caches are coherent.
#[cfg(target_arch = "aarch64")]
fn flush_icache(ptr: *const u8, len: usize) {
    unsafe extern "C" {
        fn sys_icache_invalidate(start: *const u8, len: usize);
    }
    unsafe {
        sys_icache_invalidate(ptr, len);
    }
}

#[cfg(target_arch = "x86_64")]
fn flush_icache(_ptr: *const u8, _len: usize) {
    // x86-64 has coherent instruction caches; no flush needed.
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn flush_icache(_ptr: *const u8, _len: usize) {
    // Unknown architecture — conservative no-op.
    // Implementors targeting other architectures should add
    // the appropriate cache flush here.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_align() {
        assert_eq!(page_align_up(0, 4096), 0);
        assert_eq!(page_align_up(1, 4096), 4096);
        assert_eq!(page_align_up(4096, 4096), 4096);
        assert_eq!(page_align_up(4097, 4096), 8192);
        assert_eq!(page_align_down(0, 4096), 0);
        assert_eq!(page_align_down(4095, 4096), 0);
        assert_eq!(page_align_down(4096, 4096), 4096);
        assert_eq!(page_align_down(8191, 4096), 4096);
    }

    #[test]
    fn test_new_default() {
        let mem = PagedCodeMemory::new();
        assert_eq!(mem.len(), 0);
        assert!(mem.is_empty());
        assert_eq!(mem.capacity(), page_align_up(DEFAULT_CAPACITY, mem.page_size()));
        assert_eq!(mem.committed(), 0);
    }

    #[test]
    fn test_push_and_finalize() {
        let mut mem = PagedCodeMemory::new();
        let code = [0x90; 16]; // 16 NOPs

        let offset = mem.push(&code);
        assert_eq!(offset, 0);
        assert_eq!(mem.len(), 16);
        assert!(mem.committed() >= 16);

        mem.finalize();

        // Verify the bytes are readable
        for i in 0..16 {
            let byte = unsafe { *mem.base_ptr().add(i) };
            assert_eq!(byte, 0x90);
        }
    }

    #[test]
    fn test_incremental_push() {
        let mut mem = PagedCodeMemory::new();

        let off1 = mem.push(&[0xAA; 8]);
        let off2 = mem.push(&[0xBB; 8]);

        assert_eq!(off1, 0);
        assert_eq!(off2, 8);
        assert_eq!(mem.len(), 16);
    }

    #[test]
    fn test_push_after_finalize() {
        let mut mem = PagedCodeMemory::new();

        mem.push(&[0x90; 8]);
        mem.finalize();

        // Push more code after finalizing
        let off = mem.push(&[0xCC; 8]);
        assert_eq!(off, 8);
        assert_eq!(mem.len(), 16);

        mem.finalize();

        // Both regions should be readable
        assert_eq!(unsafe { *mem.base_ptr().add(0) }, 0x90);
        assert_eq!(unsafe { *mem.base_ptr().add(8) }, 0xCC);
    }

    #[test]
    fn test_push_empty() {
        let mut mem = PagedCodeMemory::new();
        let off = mem.push(&[]);
        assert_eq!(off, 0);
        assert_eq!(mem.len(), 0);
    }

    #[test]
    fn test_grows_page_at_a_time() {
        let mut mem = PagedCodeMemory::new();
        let ps = mem.page_size();

        // Push less than a page
        mem.push(&vec![0x90; 100]);
        assert_eq!(mem.committed(), ps);

        // Push up to just over one page
        mem.push(&vec![0x90; ps]);
        assert_eq!(mem.committed(), 2 * ps);
    }

    #[test]
    fn test_patch() {
        let mut mem = PagedCodeMemory::new();

        mem.push(&[0x90; 16]);
        mem.finalize();

        // Patch bytes 4..8
        mem.patch(4, &[0xCC, 0xCC, 0xCC, 0xCC]);

        assert_eq!(unsafe { *mem.base_ptr().add(0) }, 0x90);
        assert_eq!(unsafe { *mem.base_ptr().add(4) }, 0xCC);
        assert_eq!(unsafe { *mem.base_ptr().add(7) }, 0xCC);
        assert_eq!(unsafe { *mem.base_ptr().add(8) }, 0x90);
    }

    #[test]
    #[should_panic(expected = "not fully finalized")]
    fn test_patch_unfinalized_panics() {
        let mut mem = PagedCodeMemory::new();
        mem.push(&[0x90; 16]);
        // Not finalized — patching should panic
        mem.patch(0, &[0xCC]);
    }

    #[test]
    #[should_panic(expected = "code memory exhausted")]
    fn test_exceeds_capacity() {
        let mut mem = PagedCodeMemory::with_capacity(4096);
        let ps = mem.page_size();
        // Push more than capacity
        mem.push(&vec![0x90; ps + 1]);
    }

    #[test]
    fn test_multiple_finalize_cycles() {
        let mut mem = PagedCodeMemory::new();

        // Cycle 1
        mem.push(&[0x11; 8]);
        mem.finalize();

        // Cycle 2
        mem.push(&[0x22; 8]);
        mem.finalize();

        // Cycle 3
        mem.push(&[0x33; 8]);
        mem.finalize();

        assert_eq!(mem.len(), 24);
        assert_eq!(unsafe { *mem.base_ptr().add(0) }, 0x11);
        assert_eq!(unsafe { *mem.base_ptr().add(8) }, 0x22);
        assert_eq!(unsafe { *mem.base_ptr().add(16) }, 0x33);
    }

    #[test]
    fn test_finalize_idempotent() {
        let mut mem = PagedCodeMemory::new();
        mem.push(&[0x90; 8]);
        mem.finalize();
        mem.finalize(); // should be a no-op
        assert_eq!(mem.len(), 8);
    }

    #[test]
    fn test_ptr_at() {
        let mut mem = PagedCodeMemory::new();
        mem.push(&[0xAA, 0xBB, 0xCC, 0xDD]);
        mem.finalize();

        let p = mem.ptr_at(2);
        assert_eq!(unsafe { *p }, 0xCC);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_ptr_at_out_of_bounds() {
        let mut mem = PagedCodeMemory::new();
        mem.push(&[0x90; 4]);
        mem.ptr_at(4);
    }

    // ─── Execution tests ────────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_execute_return_42() {
        use crate::arm64::*;
        use crate::arm64::inst::Arm64Inst;
        use crate::buffer::CodeBuffer;

        let mut buf: CodeBuffer<Arm64> = CodeBuffer::new();
        buf.emit(Arm64Inst::movz(X0, 42, 0));
        buf.emit(Arm64Inst::ret());
        buf.finalize();

        let mut mem = PagedCodeMemory::new();
        mem.push(buf.code());
        mem.finalize();

        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(), 42);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_execute_incremental() {
        use crate::arm64::*;
        use crate::arm64::inst::Arm64Inst;
        use crate::buffer::CodeBuffer;

        let mut mem = PagedCodeMemory::new();

        // First function: return 10
        let mut buf: CodeBuffer<Arm64> = CodeBuffer::new();
        buf.emit(Arm64Inst::movz(X0, 10, 0));
        buf.emit(Arm64Inst::ret());
        buf.finalize();
        let off1 = mem.push(buf.code());
        mem.finalize();

        // Second function: return 20
        let mut buf: CodeBuffer<Arm64> = CodeBuffer::new();
        buf.emit(Arm64Inst::movz(X0, 20, 0));
        buf.emit(Arm64Inst::ret());
        buf.finalize();
        let off2 = mem.push(buf.code());
        mem.finalize();

        let f1: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.ptr_at(off1)) };
        let f2: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.ptr_at(off2)) };
        assert_eq!(f1(), 10);
        assert_eq!(f2(), 20);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_execute_after_patch() {
        use crate::arm64::*;
        use crate::arm64::inst::Arm64Inst;
        use crate::buffer::CodeBuffer;

        let mut mem = PagedCodeMemory::new();

        // Emit: return 1
        let mut buf: CodeBuffer<Arm64> = CodeBuffer::new();
        buf.emit(Arm64Inst::movz(X0, 1, 0));
        buf.emit(Arm64Inst::ret());
        buf.finalize();
        mem.push(buf.code());
        mem.finalize();

        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(), 1);

        // Patch: change to return 99
        let mut patch_buf: CodeBuffer<Arm64> = CodeBuffer::new();
        patch_buf.emit(Arm64Inst::movz(X0, 99, 0));
        patch_buf.finalize();
        mem.patch(0, patch_buf.code());

        assert_eq!(f(), 99);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_execute_return_42() {
        use crate::x86_64::*;
        use crate::x86_64::inst::X64Inst;
        use crate::buffer::CodeBuffer;

        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        buf.emit(X64Inst::MovRI32 { dest: RAX, imm: 42 });
        buf.emit(X64Inst::Ret);
        buf.finalize();

        let mut mem = PagedCodeMemory::new();
        mem.push(buf.code());
        mem.finalize();

        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(), 42);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_execute_incremental() {
        use crate::x86_64::*;
        use crate::x86_64::inst::X64Inst;
        use crate::buffer::CodeBuffer;

        let mut mem = PagedCodeMemory::new();

        // First function: return 10
        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        buf.emit(X64Inst::MovRI32 { dest: RAX, imm: 10 });
        buf.emit(X64Inst::Ret);
        buf.finalize();
        let off1 = mem.push(buf.code());
        mem.finalize();

        // Second function: return 20
        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        buf.emit(X64Inst::MovRI32 { dest: RAX, imm: 20 });
        buf.emit(X64Inst::Ret);
        buf.finalize();
        let off2 = mem.push(buf.code());
        mem.finalize();

        let f1: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.ptr_at(off1)) };
        let f2: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.ptr_at(off2)) };
        assert_eq!(f1(), 10);
        assert_eq!(f2(), 20);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_execute_after_patch() {
        use crate::x86_64::*;
        use crate::x86_64::inst::X64Inst;
        use crate::buffer::CodeBuffer;

        let mut mem = PagedCodeMemory::new();

        // Emit: return 1
        let mut buf: CodeBuffer<X64> = CodeBuffer::new();
        buf.emit(X64Inst::MovRI32 { dest: RAX, imm: 1 });
        buf.emit(X64Inst::Ret);
        buf.finalize();
        mem.push(buf.code());
        mem.finalize();

        let f: extern "C" fn() -> u64 = unsafe { std::mem::transmute(mem.base_ptr()) };
        assert_eq!(f(), 1);

        // Patch: change to return 99
        let mut patch_buf: CodeBuffer<X64> = CodeBuffer::new();
        patch_buf.emit(X64Inst::MovRI32 { dest: RAX, imm: 99 });
        patch_buf.finalize();
        mem.patch(0, patch_buf.code());

        assert_eq!(f(), 99);
    }
}
