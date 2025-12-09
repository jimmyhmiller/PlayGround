# Plan: Implement throw/try/catch/finally for Clojure

## Overview

This plan adds exception handling following **Beagle's approach**: save SP/FP/LR on handler setup, restore and jump on throw. This is simpler and more efficient than setjmp/longjmp.

## Architecture (Based on Beagle)

### Exception Handler Structure

```rust
#[repr(C)]
pub struct ExceptionHandler {
    pub handler_address: usize,   // Label address to jump to (catch block)
    pub stack_pointer: usize,     // Saved SP
    pub frame_pointer: usize,     // Saved FP (x29)
    pub link_register: usize,     // Saved LR (x30)
    pub result_local: usize,      // Where to store exception (FP-relative offset)
}
```

### Control Flow Mechanism

1. **try** block: Push handler with saved SP/FP/LR and catch label
2. **throw**: Pop handler, store exception, restore SP/FP/LR, jump to catch
3. **catch**: Exception is already in result_local, just execute handler
4. **finally**: Run on both normal and exception paths (re-throw if needed)

### Runtime State

Add to `GCRuntime`:
```rust
/// Stack of exception handlers (per-thread in future)
exception_handlers: Vec<ExceptionHandler>,
```

## Implementation Steps

### Step 1: AST Support

Add new AST nodes in `clojure_ast.rs`:

```rust
/// (throw expr)
Throw {
    exception: Box<Expr>,
}

/// (try expr* catch-clause* finally-clause?)
Try {
    body: Vec<Expr>,
    catches: Vec<CatchClause>,
    finally: Option<Vec<Expr>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatchClause {
    pub exception_type: String,  // e.g., "Exception" (ignored for now - catch all)
    pub binding: String,         // Local binding for caught exception
    pub body: Vec<Expr>,
}
```

Parsing in `analyze()`:
- `(throw expr)` → `Expr::Throw`
- `(try body... (catch ExType e handler...) (finally cleanup...))` → `Expr::Try`

### Step 2: IR Instructions

Add new instructions in `ir.rs`:

```rust
/// Push exception handler
/// Arguments: catch_label, result_local (where to store exception)
/// At codegen time, also captures SP, FP, LR
PushExceptionHandler(Label, IrValue),

/// Remove the current exception handler (normal exit from try)
PopExceptionHandler,

/// Throw an exception - never returns
/// Pops handler, stores exception at result_local, restores SP/FP/LR, jumps to catch
Throw(IrValue),
```

### Step 3: Runtime Trampolines

Add to `trampoline.rs`:

```rust
/// Push exception handler
/// Args: handler_address, result_local_offset, link_register, stack_pointer, frame_pointer
#[no_mangle]
pub extern "C" fn trampoline_push_exception_handler(
    handler_address: usize,
    result_local: usize,
    link_register: usize,
    stack_pointer: usize,
    frame_pointer: usize,
) -> usize {
    let handler = ExceptionHandler {
        handler_address,
        stack_pointer,
        frame_pointer,
        link_register,
        result_local,
    };
    unsafe {
        let rt = &mut *RUNTIME.as_ref().unwrap().get();
        rt.push_exception_handler(handler);
    }
    7 // nil
}

/// Pop exception handler (normal exit)
#[no_mangle]
pub extern "C" fn trampoline_pop_exception_handler() -> usize {
    unsafe {
        let rt = &mut *RUNTIME.as_ref().unwrap().get();
        rt.pop_exception_handler();
    }
    7 // nil
}

/// Throw exception - never returns
/// Args: stack_pointer (for stack trace), exception_value
#[no_mangle]
pub extern "C" fn trampoline_throw(stack_pointer: usize, exception_value: usize) -> ! {
    unsafe {
        let rt = &mut *RUNTIME.as_ref().unwrap().get();

        if let Some(handler) = rt.pop_exception_handler() {
            // Store exception at result_local (FP-relative)
            let result_ptr = (handler.frame_pointer.wrapping_add(handler.result_local)) as *mut usize;
            *result_ptr = exception_value;

            // Restore SP, FP, LR and jump to handler
            asm!(
                "mov sp, {0}",
                "mov x29, {1}",
                "mov x30, {2}",
                "br {3}",
                in(reg) handler.stack_pointer,
                in(reg) handler.frame_pointer,
                in(reg) handler.link_register,
                in(reg) handler.handler_address,
                options(noreturn)
            );
        } else {
            // No handler - print and abort
            eprintln!("Uncaught exception!");
            std::process::abort();
        }
    }
}
```

### Step 4: Runtime Implementation

Add to `gc_runtime.rs`:

```rust
impl GCRuntime {
    pub fn push_exception_handler(&mut self, handler: ExceptionHandler) {
        self.exception_handlers.push(handler);
    }

    pub fn pop_exception_handler(&mut self) -> Option<ExceptionHandler> {
        self.exception_handlers.pop()
    }
}
```

### Step 5: Compiler Changes

Add to `compiler.rs`:

```rust
fn compile_throw(&mut self, exception: &Expr) -> Result<IrValue, String> {
    let exc_val = self.compile(exception)?;
    self.builder.emit(Instruction::Throw(exc_val));

    // Throw never returns, but return dummy for type consistency
    let dummy = self.builder.new_register();
    self.builder.emit(Instruction::LoadConstant(dummy, IrValue::Null));
    Ok(dummy)
}

fn compile_try(
    &mut self,
    body: &[Expr],
    catches: &[CatchClause],
    finally: &Option<Vec<Expr>>,
) -> Result<IrValue, String> {
    let result = self.builder.new_register();
    let catch_label = self.builder.new_label();
    let after_catch_label = self.builder.new_label();

    // Allocate a local for the exception (will be filled by throw)
    let exception_local = self.builder.new_register();
    self.builder.emit(Instruction::LoadConstant(exception_local, IrValue::Null));

    // 1. Push exception handler
    self.builder.emit(Instruction::PushExceptionHandler(
        catch_label.clone(),
        exception_local,
    ));

    // 2. Compile try body
    let body_result = self.compile_do(body)?;
    self.builder.emit(Instruction::Assign(result, body_result));

    // 3. Pop handler (normal exit)
    self.builder.emit(Instruction::PopExceptionHandler);

    // 4. Handle finally for normal path
    if let Some(finally_body) = finally {
        self.compile_do(finally_body)?;
    }
    self.builder.emit(Instruction::Jump(after_catch_label.clone()));

    // 5. Catch block (throw jumps here with exception in exception_local)
    self.builder.emit(Instruction::Label(catch_label));

    // Bind exception to catch variable
    if !catches.is_empty() {
        let catch = &catches[0];  // For now, single catch
        self.push_scope();
        self.bind_local(catch.binding.clone(), exception_local);

        let catch_result = self.compile_do(&catch.body)?;
        self.builder.emit(Instruction::Assign(result, catch_result));

        self.pop_scope();
    }

    // 6. Handle finally for catch path
    if let Some(finally_body) = finally {
        self.compile_do(finally_body)?;
    }

    // 7. End
    self.builder.emit(Instruction::Label(after_catch_label));
    Ok(result)
}
```

### Step 6: ARM64 Code Generation

Add to `arm_codegen.rs`:

```rust
Instruction::PushExceptionHandler(catch_label, result_local) => {
    // Calculate result_local offset (FP-relative)
    let result_local_offset = self.get_local_offset(result_local)?;

    // Get label address (will be fixed up)
    // For now, use ADR to get address
    let handler_addr_reg = 15;  // x15
    self.emit_adr(handler_addr_reg, catch_label);

    // Call: push_exception_handler(handler_addr, result_local, LR, SP, FP)
    self.emit_mov(0, handler_addr_reg);           // x0 = handler address
    self.emit_mov_imm(1, result_local_offset);    // x1 = result local offset
    self.emit_mov(2, 30);                         // x2 = LR (x30)
    self.emit_mov(3, 31);                         // x3 = SP
    self.emit_mov(4, 29);                         // x4 = FP (x29)

    let func_addr = trampoline_push_exception_handler as usize;
    self.emit_mov_imm(15, func_addr as i64);
    self.emit_blr(15);
}

Instruction::PopExceptionHandler => {
    let func_addr = trampoline_pop_exception_handler as usize;
    self.emit_mov_imm(15, func_addr as i64);
    self.emit_blr(15);
}

Instruction::Throw(exc) => {
    let exc_reg = self.get_physical_reg(exc)?;

    // Call: throw(SP, exception_value)
    self.emit_mov(1, exc_reg);                    // x1 = exception value
    // Get stack pointer into x0
    self.emit_mov(0, 31);                         // x0 = SP

    let func_addr = trampoline_throw as usize;
    self.emit_mov_imm(15, func_addr as i64);
    self.emit_blr(15);  // Never returns
}
```

### Step 7: Built-in Functions

For MVP, just support throwing any value. Later add:

```clojure
;; Create exception with message and data
(ex-info "message" {:key "value"})

;; Get exception message
(ex-message ex)

;; Get exception data
(ex-data ex)
```

## Testing Strategy

1. **Basic throw/catch**:
   ```clojure
   (try (throw 42) (catch Exception e e))
   ;; => 42
   ```

2. **Finally always runs**:
   ```clojure
   (def x 0)
   (try (throw 1) (catch Exception e e) (finally (def x 1)))
   x ;; => 1
   ```

3. **Nested try/catch**:
   ```clojure
   (try
     (try (throw 1) (catch Exception e (throw 2)))
     (catch Exception e e))
   ;; => 2
   ```

4. **Normal path with finally**:
   ```clojure
   (def y 0)
   (try 42 (finally (def y 1)))
   ;; => 42, y = 1
   ```

## Files to Modify

1. `src/clojure_ast.rs` - Add `Throw`, `Try`, `CatchClause` AST nodes + parsing
2. `src/ir.rs` - Add `PushExceptionHandler`, `PopExceptionHandler`, `Throw` instructions
3. `src/compiler.rs` - Add `compile_throw`, `compile_try`
4. `src/arm_codegen.rs` - Generate ARM64 for exception instructions
5. `src/trampoline.rs` - Add `trampoline_push_exception_handler`, `trampoline_pop_exception_handler`, `trampoline_throw`
6. `src/gc_runtime.rs` - Add `ExceptionHandler` struct and handler stack

## Key Differences from setjmp/longjmp

Beagle's approach (which we're following):
- **Simpler**: Just save/restore SP/FP/LR - no jmp_buf
- **More efficient**: No need to save all registers, just the ones needed for stack frame
- **Explicit**: The handler address is a known label, not a magic return point
- **GC-friendly**: No opaque jmp_buf to worry about

## Notes

- **GC Integration**: Exception values can be any tagged value (int, closure, heap object)
- **Stack Unwinding**: Manual restore of SP/FP/LR jumps directly to catch block
- **Dynamic Bindings**: For MVP, dynamic bindings won't be auto-restored (can add later)
- **Performance**: Minimal overhead - just a runtime call to push/pop handlers
- **finally semantics**: Duplicated code on both paths (normal and exception) - simple but correct
