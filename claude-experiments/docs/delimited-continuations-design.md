# Delimited Continuations with Stack-Walker

## Design Space Exploration

**Goal**: Multi-shot delimited continuations using raw stack copying

**Context**: The stack-walker library provides inspection (walking, slot scanning, register capture). Continuations need both inspection AND manipulation (copying/restoring stack segments).

---

## What the Library Currently Provides

1. **Stack Walking** - Walk from current frame to outer frames via frame pointer chain
2. **Register Capture** - Capture IP, SP, FP for x86_64/aarch64
3. **Slot Scanning** - Read stack slot values between SP and FP
4. **Memory Abstraction** - `MemoryReader` trait for reading stack memory

## Delimited Continuation Architecture

### Core Concepts

```
┌─────────────────────────────┐
│  current computation        │  ← shift/control happens here
├─────────────────────────────┤
│  frame n                    │
│  frame n-1                  │  ← captured continuation
│  ...                        │
│  frame 1                    │
├─────────────────────────────┤
│  prompt/reset frame         │  ← delimiter (capture stops here)
├─────────────────────────────┤
│  outer frames               │
└─────────────────────────────┘
```

### Key Components Needed

#### 1. Prompt Marking
Mark stack frames as "prompt" delimiters:
```rust
struct PromptRegistry {
    // Map from return address -> prompt tag
    prompts: HashMap<u64, PromptTag>,
}

// Alternatively: detect prompts by known function addresses
fn is_prompt_frame(frame: &Frame, resolver: &impl SymbolResolver) -> Option<PromptTag> {
    let symbol = resolver.resolve(frame.lookup_address())?;
    if symbol.name.starts_with("prompt_") {
        // Parse tag from symbol name or use frame data
    }
}
```

#### 2. Continuation Capture (shift/control)
```rust
struct CapturedContinuation {
    /// Frames from innermost to prompt (exclusive)
    frames: Vec<CapturedFrame>,
    /// Raw stack memory between current SP and prompt SP
    stack_segment: Vec<u8>,
    /// Original base addresses for relocation
    original_sp: u64,
    original_fp: u64,
    /// Registers at capture point
    registers: NativeUnwindRegs,
    /// The prompt tag this was captured against
    prompt_tag: PromptTag,
}

struct CapturedFrame {
    frame: Frame,
    slots: Vec<SlotInfo>,  // For GC root scanning
}
```

#### 3. Capture Implementation
```rust
fn capture_continuation<M: MemoryReader>(
    walker: &impl StackWalker,
    memory: &mut M,
    prompt_tag: PromptTag,
) -> Result<CapturedContinuation> {
    let regs = capture_current_native();
    let mut frames = Vec::new();
    let mut prompt_sp = None;

    // Walk to find the prompt and collect frames
    walker.walk_with(&regs, memory, &config, |frame| {
        if is_prompt_frame(frame, prompt_tag) {
            prompt_sp = Some(frame.stack_pointer);
            return false; // Stop at prompt
        }

        // Capture frame with slots (for GC)
        let slots: Vec<_> = frame.scan_slots(memory, &ConservativeSlotProvider)
            .filter_map(|r| r.ok())
            .collect();

        frames.push(CapturedFrame { frame: frame.clone(), slots });
        true
    });

    let prompt_sp = prompt_sp.ok_or(Error::NoPromptFound)?;

    // Copy the raw stack segment
    let stack_size = (prompt_sp - regs.stack_pointer()) as usize;
    let mut stack_segment = vec![0u8; stack_size];
    memory.read_bytes(regs.stack_pointer(), &mut stack_segment)?;

    Ok(CapturedContinuation {
        frames,
        stack_segment,
        original_sp: regs.stack_pointer(),
        original_fp: regs.frame_pointer().unwrap_or(0),
        registers: regs,
        prompt_tag,
    })
}
```

#### 4. Continuation Invocation (Resume)

This requires assembly/unsafe code since we're manipulating the stack:

```rust
// Conceptual - actual impl needs arch-specific assembly
unsafe fn invoke_continuation(
    cont: &CapturedContinuation,
    value: Value,  // Value passed to continuation
) -> ! {
    // 1. Ensure sufficient stack space below current SP

    // 2. Copy stack segment to new location
    let new_sp = current_sp() - cont.stack_segment.len();
    std::ptr::copy_nonoverlapping(
        cont.stack_segment.as_ptr(),
        new_sp as *mut u8,
        cont.stack_segment.len()
    );

    // 3. Optionally: relocate any stack-internal pointers
    //    (Often not needed if pointers are to heap objects)

    // 4. Set up registers and jump
    // - Load continuation's return address
    // - Set SP/FP to new locations (adjusted for relocation)
    // - Put return value in appropriate register
    // - Jump to continuation point
}
```

### GC Integration (Key Strength)

The existing slot scanning infrastructure makes captured continuations GC-traceable:

```rust
impl GcTraceable for CapturedContinuation {
    fn trace(&self, gc: &mut GcTracer) {
        // Each captured slot that might be a GC pointer
        for captured_frame in &self.frames {
            for slot in &captured_frame.slots {
                if gc.is_managed_pointer(slot.value) {
                    gc.trace(slot.value);
                    // If we need to update pointers (moving GC):
                    // Remember location for updating in stack_segment
                }
            }
        }
    }
}
```

## Implementation Approaches

### Approach A: Raw Stack Copying (Efficient)
- Copy raw bytes between SP and prompt SP
- Minimal transformation, fast capture/restore
- Challenges: pointer relocation for moving GC

### Approach B: Frame-by-Frame Reconstruction (Portable)
- Store each frame's return address + slot values
- Rebuild stack from scratch on resume
- More portable, easier to serialize
- Challenges: must understand exact calling convention

### Approach C: Segmented Stacks (Fibers/Green Threads)
- Allocate separate stack segments per prompt
- Capture = just save the segment reference
- Natural fit for many continuations
- Challenges: more complex stack switching

## New Modules to Add

```
src/
├── continuation/
│   ├── mod.rs           # Module definition
│   ├── prompt.rs        # Prompt marking/detection
│   ├── capture.rs       # Continuation capture logic
│   ├── invoke.rs        # Restoration/invocation (arch-specific)
│   ├── x86_64.rs        # x86_64 assembly for invoke
│   └── aarch64.rs       # aarch64 assembly for invoke
```

## Key Files to Modify

- `src/lib.rs` - Export continuation module
- `src/frame.rs` - Possibly add continuation-related metadata
- `src/arch/mod.rs` - Extend `UnwindRegisters` trait if needed

---

## Deep Dive: Multi-Shot Raw Stack Copying

### The Core Challenge

Multi-shot means the **same captured continuation can be invoked multiple times**. Each invocation needs its own execution context:

```
Capture at T0:
  Stack: [frame1][frame2][frame3][prompt]
         ^-- captured segment --^

Invoke #1 at T1:
  New stack: [frame1'][frame2'][frame3'][prompt']
             ^-- copy of captured segment --^

Invoke #2 at T2:  (original capture still valid!)
  Another new stack: [frame1''][frame2''][frame3''][prompt'']
```

### Design Decision Tree

#### 1. When to Copy?

**Option A: Copy-on-Capture**
- Make a heap copy when `shift` runs
- Every invocation copies from this heap version
- Pro: Captured continuation is immutable
- Con: Extra copy even if never invoked

**Option B: Copy-on-Invoke**
- Keep continuation as reference to original stack location
- Copy when actually invoked
- Pro: Lazy, avoids unnecessary copying
- Con: Original stack may be mutated/gone before invoke

**Recommended**: Copy-on-Capture for safety. The capture point is well-defined.

#### 2. Where to Put Copies?

**Option A: Below Current SP (Stack Extension)**
```
Before invoke:          After invoke:
[current frame]         [current frame]
[...]                   [...]
                        [continuation copy] ← new SP points here
                        [gap or guard]
```
- Uses existing stack
- Limited by stack size
- Natural for single-threaded

**Option B: Separate Heap-Allocated Stack Segment**
```
Main stack:             Continuation heap segment:
[current frame]         [continuation copy]
[...]                   ^
     invoke jumps to ───┘
```
- Unlimited size
- Easier for multi-shot (each invoke gets fresh allocation)
- Must handle stack switching

**Option C: Cactus Stack / Spaghetti Stack**
```
     ┌─[continuation copy A]
     │
[prompt]─┼─[continuation copy B]
     │
     └─[continuation copy C]
```
- Each branch is a separate invocation
- Natural for exploring computation trees
- Classic Scheme implementation technique

#### 3. Pointer Relocation Problem

When stack is copied to new location, any pointers INTO the stack become invalid:

```rust
fn foo() {
    let x = 42;
    let ptr = &x;  // Points to location on stack!
    shift(|k| {
        k(());     // When k invoked, ptr points to OLD location
    });
    *ptr  // Use after relocation - undefined behavior!
}
```

**Solutions:**

**A: Forbid Stack Pointers**
- Language/runtime ensures no pointers into stack
- All values either immediate or heap-allocated
- This is what many Scheme implementations do

**B: Relocate Pointers During Copy**
- Use the slot scanning to find stack-internal pointers
- Adjust them by the relocation offset
- Requires distinguishing stack pointers from heap pointers

**C: Indirection Through Handles**
- All "stack references" go through an indirection table
- Table updated on relocation
- Performance cost but fully general

**D: Pin Continuation's Original Location**
- Don't allow reuse of original stack region
- Wastes memory, works for few continuations

#### 4. Frame Pointer Chain Relocation

The FP chain must be adjusted after copying:

```
Original:                After copy:
FP₀ → FP₁ → FP₂         FP₀' → FP₁' → FP₂'
│                       │
└─ absolute addresses   └─ must be adjusted by offset
```

```rust
fn relocate_fp_chain(segment: &mut [u8], old_base: u64, new_base: u64) {
    let offset = new_base as i64 - old_base as i64;

    // Walk through the captured frames
    for frame in &captured_frames {
        if let Some(old_fp) = frame.frame_pointer {
            // Is this FP pointing into the copied segment?
            if old_fp >= old_base && old_fp < old_base + segment.len() {
                let slot_offset = (old_fp - old_base) as usize;
                let stored_fp = read_u64(&segment[slot_offset..]);
                let new_fp = (stored_fp as i64 + offset) as u64;
                write_u64(&mut segment[slot_offset..], new_fp);
            }
        }
    }
}
```

#### 5. Return Address Handling

Return addresses point into CODE, not stack, so they don't need relocation. However:

**Consideration: Stack Canaries / Return Address Protection**
- Some systems store cookies relative to return addresses
- Must handle if using stack protection

**Consideration: Exception Unwinding**
- Exception handlers may use absolute stack addresses
- Captured exception state needs care

### GC Integration for Multi-Shot

Each live copy of a continuation is a set of GC roots:

```rust
struct CapturedContinuation {
    stack_segment: Vec<u8>,      // Raw bytes
    slot_map: Vec<SlotLocation>, // Where GC pointers live in segment
}

struct SlotLocation {
    offset: usize,   // Offset into stack_segment
    is_gc_ptr: bool, // Whether this slot holds a GC pointer
}

impl GcTraceable for CapturedContinuation {
    fn trace(&self, gc: &mut Gc) {
        for slot in &self.slot_map {
            if slot.is_gc_ptr {
                let ptr = read_u64(&self.stack_segment[slot.offset..]);
                if let Some(new_ptr) = gc.trace_and_maybe_move(ptr) {
                    // For moving GC: update the pointer in our copy
                    write_u64(&mut self.stack_segment[slot.offset..], new_ptr);
                }
            }
        }
    }
}
```

### Architecture-Specific Concerns

#### x86_64
- Red zone (128 bytes below RSP) - must account for in capture
- Shadow space on Windows (32 bytes)
- RBP-based frame chain is standard

#### aarch64
- No red zone in standard ABI
- PAC (Pointer Authentication) - return addresses are signed!
- Must strip PAC before copying, re-sign after restoration
- The stack-walker already has `PtrAuthMask` support

```rust
// For aarch64 with PAC
fn capture_with_pac(frame: &Frame, mask: PtrAuthMask) -> CapturedFrame {
    // Strip authentication before storing
    let raw_return_addr = mask.strip(frame.raw_address());
    // ... capture ...
}

fn restore_with_pac(frame: &CapturedFrame, mask: PtrAuthMask) {
    // Re-sign when restoring
    let signed_return_addr = mask.sign(frame.return_addr);
    // ... restore ...
}
```

### Usage Pattern: shift/reset

```rust
// User code:
reset(|| {
    let x = 1;
    let y = shift(|k| {
        k(10) + k(20)  // Multi-shot: k invoked twice!
    });
    x + y
})
// Result: (1 + 10) + (1 + 20) = 32

// What happens:
// 1. reset pushes a prompt marker
// 2. shift walks stack to find prompt
// 3. shift captures everything between current frame and prompt
// 4. shift calls the lambda with k (the captured continuation)
// 5. Each k(v) invocation:
//    a. Copies the captured segment to new location
//    b. Sets up return value v
//    c. Jumps into the copy
//    d. When copy returns, we're back in the shift body
```

### Alternative: Delimited by Function Pointer

Instead of tagged prompts, delimit by function address:

```rust
fn find_delimiter(walker: &impl StackWalker, target_fn: *const ()) -> Option<u64> {
    let mut delimiter_sp = None;

    walker.walk_with(&regs, &mut memory, &config, |frame| {
        // Use symbol resolution or JIT metadata to check
        if frame.lookup_address() == target_fn as u64 {
            delimiter_sp = Some(frame.stack_pointer);
            return false;
        }
        true
    });

    delimiter_sp
}
```

### Performance Considerations

| Operation | Cost |
|-----------|------|
| Capture | O(n) where n = bytes between SP and prompt |
| Invoke (first time) | O(n) copy + O(frames) for FP relocation |
| Invoke (multi-shot) | Same as first time |
| GC trace | O(slots) per live continuation |

**Optimizations:**
- Lazy FP relocation (only fix what's accessed)
- Copy-on-write for multi-shot (share until mutation)
- Generational copying (only copy young frames)

---

## Summary

For multi-shot raw stack copying with the stack-walker:

1. **Capture**: Walk stack to prompt, copy raw bytes, record slot map for GC
2. **Store**: Heap-allocate the captured segment
3. **Invoke**: Copy to new location, relocate FP chain, handle PAC on aarch64
4. **GC**: Trace all live continuations using slot map

The stack-walker provides the inspection layer. You'd add:
- Prompt marking system
- Stack segment copying logic
- Architecture-specific restoration assembly
- GC integration for captured continuations
