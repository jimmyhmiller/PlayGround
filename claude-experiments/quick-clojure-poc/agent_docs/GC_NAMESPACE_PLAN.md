# GC-Allocated Namespace Implementation Plan

## UPDATED APPROACH
We're NOT importing Beagle as a library. Instead, we'll:
1. Keep namespaces as HashMap for now (TEMPORARY but working)
2. Structure the code to treat namespaces as first-class values
3. Later: steal specific GC code from Beagle when we're ready to implement heap allocation

## Original Goal
Implement namespaces as first-class heap-allocated objects with proper garbage collection, not as Rust HashMaps.

## Current State
- ✅ AST supports namespace operations (Ns, Use, qualified symbols)
- ✅ Compiler has namespace tracking with Rust HashMaps (TEMPORARY - must be replaced)
- ❌ No heap allocation for namespaces
- ❌ No GC integration

## What We're Copying from Beagle
Already copied:
- `src/beagle/types.rs` - Header, HeapObject, Tagged types
- `src/beagle/gc/` - GC infrastructure (mark-and-sweep, generational, etc.)

Need to copy/integrate:
- Memory allocator interface
- String allocation (as reference for namespace allocation)
- GC root management for namespaces

## Namespace Object Design

### Heap Layout
```
Namespace Object:
┌─────────────────────────────────┐
│ Header (8 bytes)                │
│  - type_id: 10 (Namespace)      │
│  - size: 1 + (num_bindings * 2) │
│  - opaque: false (has pointers!)│
└─────────────────────────────────┘
│ Field 0: name (String pointer)  │ ──→ "user" or "clojure.core"
├─────────────────────────────────┤
│ Field 1: symbol_0_name          │ ──→ String pointer to "pi"
│ Field 2: symbol_0_value         │ ──→ Tagged int: 314
├─────────────────────────────────┤
│ Field 3: symbol_1_name          │ ──→ String pointer to "e"
│ Field 4: symbol_1_value         │ ──→ Tagged int: 271
└─────────────────────────────────┘
```

**Why alternating key/value pairs?**
- Simple to implement
- No hash table needed (linear scan is fine for small namespaces)
- Easy to reallocate when adding bindings
- GC can scan all fields uniformly

### Type ID
```rust
const TYPE_ID_NAMESPACE: u8 = 10;
```

## Implementation Steps

### 1. Minimal GC Runtime (New File: `src/gc_runtime.rs`)

```rust
use crate::beagle::types::{Header, HeapObject, BuiltInTypes};
use crate::beagle::gc::{Allocator, AllocateAction};

pub struct GCRuntime {
    heap: Box<dyn Allocator>,  // Use Beagle's allocator
    namespace_roots: Vec<usize>,  // Pointers to namespace objects
}

impl GCRuntime {
    pub fn new() -> Self {
        // Create allocator (mark-and-sweep for simplicity)
        let heap = Box::new(MarkAndSweep::new(options));
        GCRuntime {
            heap,
            namespace_roots: vec![],
        }
    }

    /// Allocate a namespace object on the heap
    pub fn allocate_namespace(&mut self, name: &str) -> Result<usize, String> {
        // 1. Allocate name string
        let name_ptr = self.allocate_string(name)?;

        // 2. Allocate namespace: 1 word for name, start with 0 bindings
        let size_words = 1;
        let ns_ptr = self.allocate_heap_object(size_words, TYPE_ID_NAMESPACE)?;

        // 3. Write namespace data
        let mut heap_obj = HeapObject::from_untagged(ns_ptr);
        heap_obj.write_field(0, name_ptr);  // Store name pointer

        // 4. Tag and return
        Ok(BuiltInTypes::HeapObject.tagged(ns_ptr))
    }

    /// Add a binding to a namespace (may reallocate!)
    pub fn namespace_add_binding(
        &mut self,
        ns_ptr: usize,
        symbol_name: &str,
        value: usize,
    ) -> Result<usize, String> {
        let ns_untagged = BuiltInTypes::HeapObject.untag(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged);
        let header = heap_obj.read_header();
        let current_size = header.size as usize;

        // Allocate symbol name string
        let symbol_ptr = self.allocate_string(symbol_name)?;

        // Reallocate with +2 words (name + value)
        let new_size = current_size + 2;
        let new_ns_ptr = self.allocate_heap_object(new_size, TYPE_ID_NAMESPACE)?;

        // Copy existing fields
        let mut new_heap_obj = HeapObject::from_untagged(new_ns_ptr);
        for i in 0..current_size {
            let field = heap_obj.get_field(i);
            new_heap_obj.write_field(i, field);
        }

        // Add new binding
        new_heap_obj.write_field(current_size, symbol_ptr);
        new_heap_obj.write_field(current_size + 1, value);

        Ok(BuiltInTypes::HeapObject.tagged(new_ns_ptr))
    }

    /// Look up a binding in a namespace
    pub fn namespace_lookup(&self, ns_ptr: usize, symbol_name: &str) -> Option<usize> {
        let ns_untagged = BuiltInTypes::HeapObject.untag(ns_ptr);
        let heap_obj = HeapObject::from_untagged(ns_untagged);
        let header = heap_obj.read_header();
        let size = header.size as usize;
        let num_bindings = (size - 1) / 2;

        // Linear search through bindings
        for i in 0..num_bindings {
            let name_ptr = heap_obj.get_field(1 + i * 2);
            let stored_name = self.read_string(name_ptr);
            if stored_name == symbol_name {
                return Some(heap_obj.get_field(1 + i * 2 + 1));
            }
        }
        None
    }

    /// Register namespace as GC root
    pub fn add_namespace_root(&mut self, ns_ptr: usize) {
        self.namespace_roots.push(ns_ptr);
    }

    // Helper methods
    fn allocate_heap_object(&mut self, size: usize, type_id: u8) -> Result<usize, String> {
        match self.heap.try_allocate(size, BuiltInTypes::HeapObject)? {
            AllocateAction::Allocated(ptr) => {
                let mut heap_obj = HeapObject::from_untagged(ptr as usize);
                heap_obj.write_header_direct(Header {
                    type_id,
                    type_data: 0,
                    size: size as u8,
                    opaque: false,  // Contains pointers!
                    marked: false,
                });
                Ok(ptr as usize)
            }
            AllocateAction::Gc => {
                // Trigger GC and retry
                self.gc()?;
                self.allocate_heap_object(size, type_id)
            }
        }
    }

    fn allocate_string(&mut self, s: &str) -> Result<usize, String> {
        let bytes = s.as_bytes();
        let words = (bytes.len() + 7) / 8;  // Round up to 8-byte words
        let ptr = self.allocate_heap_object(words, TYPE_ID_STRING)?;

        let mut heap_obj = HeapObject::from_untagged(ptr);
        // Update header with actual byte length
        let mut header = heap_obj.read_header();
        header.type_data = bytes.len() as u32;
        header.opaque = true;  // Strings have no pointers
        heap_obj.write_header_direct(header);

        // Write string data
        heap_obj.write_bytes(bytes);

        Ok(BuiltInTypes::String.tagged(ptr))
    }

    fn read_string(&self, tagged_ptr: usize) -> String {
        let ptr = BuiltInTypes::String.untag(tagged_ptr);
        let heap_obj = HeapObject::from_untagged(ptr);
        let header = heap_obj.read_header();
        let byte_len = header.type_data as usize;
        heap_obj.read_string(byte_len)
    }

    fn gc(&mut self) -> Result<(), String> {
        // Collect namespace roots for GC
        let stack_pointers = vec![];  // No stack scanning yet
        self.heap.gc(&StackMap::new(), &stack_pointers);

        // Update namespace roots after GC (they may have moved)
        let relocations = self.heap.get_namespace_relocations();
        for (ns_id, moves) in relocations {
            for (old_ptr, new_ptr) in moves {
                if let Some(pos) = self.namespace_roots.iter().position(|&p| p == old_ptr) {
                    self.namespace_roots[pos] = new_ptr;
                }
            }
        }

        Ok(())
    }
}
```

### 2. Update Compiler to Use GCRuntime

```rust
pub struct Compiler {
    /// Runtime with heap allocator
    runtime: Arc<Mutex<GCRuntime>>,

    /// Current namespace (HEAP POINTER, not string!)
    current_namespace_ptr: usize,

    /// Namespace registry: name → heap pointer
    namespace_registry: HashMap<String, usize>,

    /// IR builder
    builder: IrBuilder,
}

impl Compiler {
    pub fn new(runtime: Arc<Mutex<GCRuntime>>) -> Self {
        let mut rt = runtime.lock().unwrap();

        // Bootstrap namespaces
        let core_ns_ptr = rt.allocate_namespace("clojure.core").unwrap();
        let user_ns_ptr = rt.allocate_namespace("user").unwrap();

        // Register as GC roots
        rt.add_namespace_root(core_ns_ptr);
        rt.add_namespace_root(user_ns_ptr);

        let mut namespace_registry = HashMap::new();
        namespace_registry.insert("clojure.core".to_string(), core_ns_ptr);
        namespace_registry.insert("user".to_string(), user_ns_ptr);

        drop(rt);

        Compiler {
            runtime,
            current_namespace_ptr: user_ns_ptr,
            namespace_registry,
            builder: IrBuilder::new(),
        }
    }

    fn compile_def(&mut self, name: &str, value_expr: &Expr) -> Result<IrValue, String> {
        let value_reg = self.compile(value_expr)?;

        // After execution, add binding to current namespace
        // This requires executing the IR first, then calling runtime
        // We'll need to change the execution flow

        Ok(value_reg)
    }

    pub fn set_global(&mut self, name: String, value: usize) {
        let mut rt = self.runtime.lock().unwrap();

        // Add binding to current namespace (may reallocate!)
        let new_ns_ptr = rt
            .namespace_add_binding(self.current_namespace_ptr, &name, value)
            .unwrap();

        // Update current namespace pointer if it moved
        if new_ns_ptr != self.current_namespace_ptr {
            self.current_namespace_ptr = new_ns_ptr;

            // Update in registry
            if let Some(ns_name) = self.get_current_namespace_name() {
                self.namespace_registry.insert(ns_name, new_ns_ptr);
            }
        }
    }

    fn compile_var(&mut self, namespace: &Option<String>, name: &str) -> Result<IrValue, String> {
        let rt = self.runtime.lock().unwrap();

        let ns_ptr = if let Some(ns_name) = namespace {
            self.namespace_registry
                .get(ns_name)
                .copied()
                .ok_or_else(|| format!("Undefined namespace: {}", ns_name))?
        } else {
            self.current_namespace_ptr
        };

        if let Some(value) = rt.namespace_lookup(ns_ptr, name) {
            let result = self.builder.new_register();
            self.builder.emit(Instruction::LoadConstant(
                result,
                IrValue::TaggedConstant(value as isize),
            ));
            Ok(result)
        } else {
            Err(format!("Undefined variable: {}", name))
        }
    }
}
```

### 3. Update Main REPL

```rust
fn main() {
    let runtime = Arc::new(Mutex::new(GCRuntime::new()));
    let mut repl_compiler = Compiler::new(runtime.clone());

    loop {
        // ... REPL loop
        // After execution, call repl_compiler.set_global() as before
    }
}
```

## Type IDs
```rust
const TYPE_ID_STRING: u8 = 2;
const TYPE_ID_NAMESPACE: u8 = 10;
```

## Key Differences from Current Implementation

| Current (HashMap) | New (GC Heap) |
|------------------|---------------|
| `globals: HashMap<String, isize>` | `GCRuntime` manages namespaces |
| `current_namespace: String` | `current_namespace_ptr: usize` (heap pointer) |
| Symbol lookup in HashMap | `namespace_lookup()` via runtime |
| No GC | Full GC with namespace roots |
| Namespaces in Rust memory | Namespaces in heap, GC'd |

## Benefits
1. **True first-class namespaces** - can be passed around, introspected
2. **Proper GC** - namespaces are collected when unreferenced
3. **Matches real Clojure** - namespaces are objects, not metadata
4. **Foundation for future** - enables `*ns*` var, namespace metadata, etc.

## Migration Strategy
1. ✅ Copy minimal GC code from Beagle (already done)
2. Create `gc_runtime.rs` with allocator and namespace operations
3. Update `Compiler::new()` to take runtime
4. Replace all HashMap operations with runtime calls
5. Test thoroughly

## Files to Create/Modify
- **NEW**: `src/gc_runtime.rs` - Runtime with namespace allocation
- **MODIFY**: `src/compiler.rs` - Use runtime instead of HashMap
- **MODIFY**: `src/main.rs` - Create runtime, pass to compiler
- **VERIFY**: GC files in `src/beagle/gc/` work for our use case

## Success Criteria
- [ ] Namespaces allocated on heap with type_id=10
- [ ] Symbol lookup works via heap namespace objects
- [ ] GC collects unreferenced namespaces
- [ ] All namespace tests still pass
- [ ] No Rust HashMap for namespace storage
