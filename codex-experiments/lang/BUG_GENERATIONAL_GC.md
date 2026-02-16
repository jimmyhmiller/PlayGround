# Bug Report: Generational GC Crashes in Custom Callback API

## Summary

The generational GC implementation in `gc-library`'s custom callback API crashes with `SIGBUS` when used by the Lang compiler. The issue is that the generational collector uses a **copying/moving strategy** which requires forwarding pointer support, but the callback API doesn't support object relocation.

## Severity

**High** - Prevents using generational GC, which would provide better performance than mark-and-sweep.

## Current Status

- ✅ Mark-and-sweep GC works perfectly
- ✅ Generational GC works with the standard FFI API (non-callback)
- ❌ Generational GC crashes with custom callback API
- ❌ JIT tests crash with SIGBUS (all 20 JIT tests fail)

## How to Reproduce

### 1. Configure for Generational GC

In `runtime/gc_bridge.c`:
```c
void gc_init(void) {
    GcCustomConfig cfg;
    // ... other config ...
    cfg.strategy = GC_STRATEGY_GENERATIONAL;  // Use generational
    gc = gc_lib_custom_create(cfg);
}
```

### 2. Run Tests

```bash
cargo test
```

**Expected:** Tests pass
**Actual:** SIGBUS crash in all JIT tests

```
running 40 tests
error: test failed, to rerun pass `--bin langc`

Caused by:
  process didn't exit successfully: ... (signal: 10, SIGBUS: access to undefined memory)
```

### 3. Minimal Reproducer

The gc-library's own tests pass, but the Lang compiler crashes:

```bash
# gc-library tests - PASS
cd ../gc-library && cargo test
# All 74 tests pass

# Lang compiler tests - FAIL
cd lang && cargo test
# SIGBUS crash
```

## Root Cause

### The Problem: Forwarding Pointers Not Supported

In `gc-library/src/lang_gc.rs` lines 202-215:

```rust
impl ForwardingSupport for CallbackObject {
    fn is_forwarded(&self) -> bool {
        // Callback API doesn't support moving GC yet
        false
    }

    fn get_forwarding_pointer(&self) -> Self::TaggedValue {
        panic!("Callback API does not support forwarding")  // ❌ PANICS!
    }

    fn set_forwarding_pointer(&mut self, _new_location: Self::TaggedValue) {
        panic!("Callback API does not support forwarding")  // ❌ PANICS!
    }
}
```

### Why This Happens

1. **Generational GC uses copying collection** - Objects are moved from young to old generation
2. **Moving objects requires updating all references** - This needs forwarding pointers
3. **Callback API uses custom object layouts** - The caller (Lang compiler) manages headers, not the GC
4. **Can't relocate objects without cooperation** - The GC doesn't know how to update references in the custom layout

### The Object Layout Issue

Lang compiler's object layout (`gc_bridge.c` lines 7-32):
```c
/* Object header layout (8 bytes):
 *   offset 0: u8  mark_flag
 *   offset 1: u8  pad
 *   offset 2: u16 type_id
 *   offset 4: u16 num_ptr_fields
 *   offset 6: u16 num_total_fields
 * After header: num_total_fields * 8 bytes of field data.
 */
```

The GC can't insert a forwarding pointer without corrupting this layout.

## Potential Solutions

### Option 1: Non-Moving Generational GC (Recommended)

Implement a **non-copying generational collector** that uses:
- Write barriers to track old→young references
- Separate young and old heaps (no moving)
- Major/minor collections without object relocation

**Pros:**
- Works with callback API's custom layouts
- Still gets generational performance benefits
- No changes needed to Lang compiler

**Cons:**
- More fragmentation than copying collector
- Slightly less performance than copying

**Implementation:**
```rust
pub struct NonMovingGenerationalGC<T: GcTypes, M: MemoryProvider> {
    young: MarkAndSweep<T, M>,  // Young generation (mark-sweep)
    old: MarkAndSweep<T, M>,    // Old generation (mark-sweep)
    remembered_set: Vec<usize>, // Old→young pointers
}
```

### Option 2: Add Forwarding Support to Callback API

Add a `forwarding_pointer` callback to let the user manage forwarding:

```c
typedef struct {
    GcCustomRootEnumerator root_enumerator;
    GcCustomTracer         tracer;
    GcCustomIsMarked       is_marked;
    GcCustomSetMark        set_mark;
    GcCustomGetSize        get_size;
    // NEW:
    GcCustomGetForwarding  get_forwarding;  // Read forwarding ptr from object
    GcCustomSetForwarding  set_forwarding;  // Write forwarding ptr to object
    size_t                 initial_heap;
    GcStrategy             strategy;
} GcCustomConfig;
```

**Pros:**
- Enables full copying collector
- Better performance (less fragmentation)

**Cons:**
- Requires changes to Lang compiler's object layout
- More complex for users of callback API
- Need to reserve space in object header for forwarding pointer

### Option 3: Hybrid Approach

Use different strategies based on object age:
- Young generation: In-place mark-sweep (no moving)
- Old generation: Copying collector (objects are tenured infrequently)
- Only move during old→old compaction (rare)

**Pros:**
- Balance of performance and simplicity
- Most objects die young (never moved)

**Cons:**
- Still needs some forwarding support for old generation

## Recommended Fix

**Implement Option 1: Non-Moving Generational GC**

This provides the best balance:
1. ✅ Works with existing callback API (no breaking changes)
2. ✅ Better performance than mark-sweep
3. ✅ No object relocation complexity
4. ✅ Write barriers still help performance

### Implementation Sketch

```rust
// In gc-library/src/gc/non_moving_generational.rs
pub struct NonMovingGenerationalGC<T: GcTypes, M: MemoryProvider> {
    young_space: Space<M>,
    young_free_list: FreeList,
    old: MarkAndSweep<T, M>,
    card_table: CardTable,
    options: AllocatorOptions,
}

impl<T: GcTypes, M: MemoryProvider> Allocator<T, M> for NonMovingGenerationalGC<T, M> {
    fn try_allocate(&mut self, field_count: usize, kind: T::ObjectKind)
        -> Result<AllocateAction<T>, AllocError>
    {
        // Allocate in young generation (no moving)
        // Minor GC when young is full
        // Promote survivors to old generation (still no moving)
        // Major GC when old is full
    }
}
```

## Testing Plan

After fix:

1. **Unit tests:** Verify non-moving generational works
   ```bash
   cd gc-library && cargo test
   ```

2. **Integration tests:** Lang compiler tests should pass
   ```bash
   cd lang && cargo test
   # Should pass all 40 tests
   ```

3. **Bootstrap:** Self-hosting should work
   ```bash
   cargo run --release -- bootstrap
   # Should output: typechecked ok
   ```

4. **Performance:** Benchmark vs mark-sweep
   ```bash
   cargo run --release -- run examples/binarytrees.lang
   # Should be faster than mark-sweep
   ```

## Workaround

Currently using **mark-and-sweep** which works correctly:

```c
// In runtime/gc_bridge.c
cfg.strategy = GC_STRATEGY_MARK_SWEEP;  // Use mark-sweep (works)
```

All production code paths work with mark-and-sweep:
- ✅ Bootstrap
- ✅ Run mode
- ✅ Build mode
- ✅ Self-hosting compiler

## Files Involved

### gc-library:
- `src/lang_gc.rs` - Callback API implementation (lines 202-215 have forwarding panics)
- `src/gc/generational.rs` - Current copying generational GC
- `src/gc/non_moving_generational.rs` - **NEW FILE** (recommended solution)
- `include/gc_library.h` - Public C API

### lang:
- `runtime/gc_bridge.c` - Bridge to gc-library (line 106 sets strategy)
- `src/codegen.rs` - JIT tests crash here (lines 3466+)

## References

- Memory about bootstrap issues: `~/.claude/projects/.../memory/bootstrap-lessons.md`
- GC library location: `~/Documents/Code/PlayGround/claude-experiments/gc-library/`
- Lang compiler location: `~/Documents/Code/PlayGround/codex-experiments/lang/`

## Priority

**Medium-High**

- Not blocking: Mark-and-sweep works fine for now
- Important: Generational GC would improve performance significantly
- Self-hosting works: Not affecting core functionality

---

**Created:** 2026-02-11
**Status:** Open
**Assigned to:** gc-library maintainer
