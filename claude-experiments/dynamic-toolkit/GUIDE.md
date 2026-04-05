# Dynamic Toolkit: Implementing a Language

This guide explains how to implement a programming language using the dynamic-toolkit
ecosystem. The core idea: you write a **frontend** (parser) and a **lowering pass** that
translates your AST into **dynir**, the toolkit's intermediate representation. The toolkit
handles interpretation, optimization, JIT compilation, GC, and more.

## Architecture Overview

```
Your Language
  ┌──────────────────────────────────────────────────┐
  │  source → lex → parse → AST                      │  You write this
  │  AST → lower using DynModule/DynFunc (dynlang)    │  You write this
  └──────────────────────────────────────────────────┘
                        ↓
                    dynir Module
                        ↓
  ┌──────────────────────────────────────────────────┐
  │  ModuleInterpreter (reference interpreter)        │  Toolkit provides
  │  dynlower → dynasm (JIT to native code)           │  Toolkit provides
  │  dynruntime ExecutionEngine (tiered interp + JIT) │  Toolkit provides
  │  dynalloc (GC), dynobj (heap objects)              │  Toolkit provides
  └──────────────────────────────────────────────────┘
```

## The Crate Ecosystem

| Crate | What it does |
|-------|-------------|
| **dynlang** | **Start here.** High-level builder for dynamic languages: mutable variables, NanBox constants, inline fast paths for arithmetic, string pool, truthiness checks. Wraps dynir. |
| **dynir** | The IR: SSA instructions, `FunctionBuilder`/`ModuleBuilder`, reference interpreter, verifier, optimizer. dynlang wraps this; use directly for advanced needs. |
| **dynvalue** | Tagged value encoding. Two schemes: `LowBit<N>` (tag in low bits) and `NanBox` (NaN-boxing for unboxed floats). |
| **dynexec** | Execution semantics: delimited continuations, frame capture/resume, calling conventions, root management strategies. |
| **dynlower** | Lowers dynir → native machine code (ARM64/x86-64) with register allocation and GC stack maps. |
| **dynasm** | Architecture-specific code emission (ARM64, x86-64). Used by dynlower. |
| **dynalloc** | Semi-space copying GC with generational support, write barriers, safepoints. |
| **dynobj** | Heap object layout: headers, field access, GC root scanning. |
| **dynruntime** | Execution engine: coordinates interpreter + JIT, tiering policies, deoptimization. |

**For most language implementations, you only need `dynlang` + `dynir` + `dynvalue`.**

## Quick Start with dynlang

The `dynlang` crate provides `DynModule` and `DynFunc` — high-level wrappers that
handle NanBox encoding, mutable variables, inline fast paths, and more. This is the
**recommended** way to implement a dynamically-typed language.

### Step 1: Write Your Frontend

Parse source code into an AST. This is standard compiler stuff — the toolkit doesn't
provide a parser.

### Step 2: Set Up a DynModule

```rust
use dynlang::*;

let mut dm = DynModule::new();

// Register slow-path externs for operations that can't be handled inline.
// This declares: lox_add, lox_sub, lox_mul, lox_div, lox_neg,
//                lox_eq, lox_lt, lox_gt, lox_not
// All take/return I64 (NanBox-encoded values).
dm.register_slow_paths("lox");

// You can also register individual slow paths:
// dm.register_slow_path(DynOp::Add, "my_custom_add");

// Declare additional externs for language-specific operations:
let rt_print = dm.declare_extern("lox_print", Signature {
    params: vec![Type::I64],
    ret: None,
});
let rt_call = dm.declare_extern("lox_call", Signature {
    params: vec![Type::I64, Type::I64],
    ret: Some(Type::I64),
});

// String constant pool
let hello_id = dm.add_string("hello");
let world_id = dm.add_string("world");
```

### Step 3: Declare All Functions

Declare all functions before defining any (enables mutual recursion):

```rust
let main_ref = dm.declare_func("main", 0);      // 0 params
let fib_ref = dm.declare_func("fib", 1);         // 1 param
let helper_ref = dm.declare_func("helper", 2);   // 2 params
```

All parameters and return values are `Type::I64` (NanBox-encoded).

### Step 4: Define Functions with DynFunc

```rust
let mut f = dm.start_func(main_ref);

// ── NanBox constants ──────────────────────────────────────
let x = f.number(42.0);          // float → NanBox I64
let n = f.nil();                  // nil constant
let t = f.bool_val(true);        // boolean constant
let s = f.tagged_const(2, hello_id as u64);  // tagged value (e.g. string ID)

// ── Mutable variables (backed by stack slots) ─────────────
f.def_var("counter", f.number(0.0));
f.set_var("counter", f.number(1.0));
let val = f.get_var("counter");   // loads current value

// ── Scoping ───────────────────────────────────────────────
f.push_scope();
f.def_var("x", f.number(10.0));  // shadows outer "x" if any
// ... use x ...
f.pop_scope();                    // inner "x" no longer visible

// ── Dynamic arithmetic (inline float fast path) ──────────
// If both args are NanBox floats → fadd/fsub/fmul/fdiv inline.
// Otherwise → calls the registered slow-path extern.
let sum  = f.dyn_add(x, val);
let diff = f.dyn_sub(x, val);
let prod = f.dyn_mul(x, val);
let quot = f.dyn_div(x, val);
let neg  = f.dyn_neg(x);

// ── Dynamic comparison (inline float fast path) ──────────
// Returns a NanBox bool.
let eq = f.dyn_eq(x, val);       // float: NaN != NaN, -0 == +0
let lt = f.dyn_lt(x, val);
let gt = f.dyn_gt(x, val);

// ── Type checks (return I8) ──────────────────────────────
let is_num   = f.is_number(val);
let is_nil   = f.is_nil(val);
let is_bool  = f.is_bool(val);
let is_ptr   = f.is_ptr(val);
let falsey   = f.is_falsey(val);  // true for nil and false
let truthy   = f.is_truthy(val);

// ── Truthiness branching ─────────────────────────────────
let then_bb = f.fb.create_block(&[]);
let else_bb = f.fb.create_block(&[]);
f.br_if_truthy(val, then_bb, &[], else_bb, &[]);

// ── Unbox/box for manual float operations ─────────────────
let raw_f64 = f.unbox_number(x);       // bitcast I64 → F64
let boxed   = f.box_number(raw_f64);   // bitcast F64 → I64 + NaN canonicalization

// ── Access the raw FunctionBuilder for anything else ──────
// f.fb gives you full access to dynir's FunctionBuilder:
f.fb.switch_to_block(then_bb);
let result = f.fb.call(fib_ref, &[x]).unwrap();
f.fb.ret(result);

f.fb.switch_to_block(else_bb);
f.fb.ret(n);

dm.finish_func(f);
```

### Step 5: Build and Run

```rust
let built = dm.build();
// built.module  — the dynir Module
// built.strings — the string constant pool (Vec<String>)
// built.func_refs — name → FuncRef map

// Run with the reference interpreter:
use dynir::interp::{ExternCallResult, InterpResult, ModuleInterpreter, NoGcRoots};
use dynvalue::NanBox;

let roots = NoGcRoots;
let mut interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &roots);

// Bind slow-path externs
interp.bind_by_name("lox_add", |args| {
    // Handle non-float add (e.g. string concatenation)
    // args[0] and args[1] are NanBox-encoded u64 values
    ExternCallResult::Value(Some(result_bits))
});

// Bind other externs
interp.bind_by_name("lox_print", |args| {
    let val = args[0];
    // Decode and print the value
    ExternCallResult::Value(None)
});

// Run
match interp.run(main_ref, &[]) {
    Ok(InterpResult::Value(v)) => println!("Result: {}", f64::from_bits(v)),
    Ok(InterpResult::Void) => {},
    other => panic!("unexpected: {:?}", other),
}
```

## Complete Example: Lowering an AST

Here's a realistic pattern for lowering an expression-based AST:

```rust
use dynlang::*;

struct Lowerer<'a> {
    dm: &'a DynModule,
    func_refs: &'a std::collections::HashMap<String, FuncRef>,
    rt_print: FuncRef,
}

fn lower_program(program: &Program) -> BuiltModule {
    let mut dm = DynModule::new();
    dm.register_slow_paths("rt");
    let rt_print = dm.declare_extern("rt_print",
        Signature { params: vec![Type::I64], ret: None });

    // Phase 1: declare all functions
    for func in &program.functions {
        dm.declare_func(&func.name, func.params.len());
    }

    // Phase 2: define all functions
    for func in &program.functions {
        let fref = dm.func_ref(&func.name);
        let mut f = dm.start_func(fref);

        // Bind parameters as variables
        let entry = f.fb.entry_block();
        for (i, param_name) in func.params.iter().enumerate() {
            let val = f.fb.block_param(entry, i);
            f.def_var(param_name, val);
        }

        // Lower the body
        let result = lower_expr(&mut f, &func.body, &dm);
        f.fb.ret(result);

        dm.finish_func(f);
    }

    dm.build()
}

fn lower_expr(f: &mut DynFunc, expr: &Expr, dm: &DynModule) -> Value {
    match expr {
        Expr::Number(n) => f.number(*n),
        Expr::Nil => f.nil(),
        Expr::Bool(b) => f.bool_val(*b),
        Expr::Var(name) => f.get_var(name),

        Expr::Add(l, r) => {
            let lv = lower_expr(f, l, dm);
            let rv = lower_expr(f, r, dm);
            f.dyn_add(lv, rv)     // inline float fast path + slow path
        }
        Expr::Sub(l, r) => {
            let lv = lower_expr(f, l, dm);
            let rv = lower_expr(f, r, dm);
            f.dyn_sub(lv, rv)
        }
        Expr::Neg(e) => {
            let v = lower_expr(f, e, dm);
            f.dyn_neg(v)
        }
        Expr::Less(l, r) => {
            let lv = lower_expr(f, l, dm);
            let rv = lower_expr(f, r, dm);
            f.dyn_lt(lv, rv)      // returns NanBox bool
        }
        Expr::Equal(l, r) => {
            let lv = lower_expr(f, l, dm);
            let rv = lower_expr(f, r, dm);
            f.dyn_eq(lv, rv)
        }

        Expr::If(cond, then_body, else_body) => {
            let c = lower_expr(f, cond, dm);
            let then_bb = f.fb.create_block(&[]);
            let else_bb = f.fb.create_block(&[]);
            let merge_bb = f.fb.create_block(&[Type::I64]);

            f.br_if_truthy(c, then_bb, &[], else_bb, &[]);

            f.fb.switch_to_block(then_bb);
            let then_val = lower_expr(f, then_body, dm);
            f.fb.jump(merge_bb, &[then_val]);

            f.fb.switch_to_block(else_bb);
            let else_val = lower_expr(f, else_body, dm);
            f.fb.jump(merge_bb, &[else_val]);

            f.fb.switch_to_block(merge_bb);
            f.fb.block_param(merge_bb, 0)
        }

        Expr::While(cond, body) => {
            let header = f.fb.create_block(&[]);
            let body_bb = f.fb.create_block(&[]);
            let exit = f.fb.create_block(&[]);

            f.fb.jump(header, &[]);

            f.fb.switch_to_block(header);
            let c = lower_expr(f, cond, dm);
            f.br_if_truthy(c, body_bb, &[], exit, &[]);

            f.fb.switch_to_block(body_bb);
            f.push_scope();
            let _ = lower_expr(f, body, dm);
            f.pop_scope();
            f.fb.jump(header, &[]);

            f.fb.switch_to_block(exit);
            f.nil()
        }

        Expr::Let(name, init, body) => {
            let val = lower_expr(f, init, dm);
            f.push_scope();
            f.def_var(name, val);
            let result = lower_expr(f, body, dm);
            f.pop_scope();
            result
        }

        Expr::Assign(name, val) => {
            let v = lower_expr(f, val, dm);
            f.set_var(name, v);
            v
        }

        Expr::Call(name, args) => {
            let fref = dm.func_ref(name);
            let arg_vals: Vec<Value> = args.iter()
                .map(|a| lower_expr(f, a, dm))
                .collect();
            f.fb.call(fref, &arg_vals).unwrap()
        }
    }
}
```

## What DynFunc Gives You (vs raw dynir)

| Pain point | Raw dynir | With dynlang |
|------------|-----------|-------------|
| **Mutable variables** | Manual stack slot allocation, `stack_addr` + `load`/`store` for every access | `f.def_var("x", val)` / `f.get_var("x")` / `f.set_var("x", val)` |
| **NanBox encoding** | Must know bit patterns: `fb.iconst(Type::I64, 0x7FFC_0000_0000_0000)` | `f.number(3.14)`, `f.nil()`, `f.bool_val(true)` |
| **Dynamic add** | 20+ lines: check tags, branch, bitcast, fadd, canonicalize NaN, merge | `f.dyn_add(a, b)` — generates the same IR automatically |
| **Truthiness** | Manual nil + false check, OR, branch | `f.is_falsey(v)` or `f.br_if_truthy(v, then, else)` |
| **String constants** | No built-in support; hack with extern calls | `dm.add_string("hello")` → ID, accessible at runtime via `built.strings` |
| **Variable scoping** | N/A (SSA has no scopes) | `f.push_scope()` / `f.pop_scope()` with automatic shadowing |

**When to drop to raw dynir**: The `f.fb` field gives you full access to the underlying
`FunctionBuilder`. Use it for control flow (`br_if`, `jump`, `switch`), calling functions,
and anything the high-level API doesn't cover.

## Tagged Values (NanBox)

dynlang uses NanBox encoding by default. All values are `Type::I64`:

- **Floats**: Stored directly as IEEE 754 bits (zero overhead)
- **Tagged values**: Use the NaN payload space. Default tags:
  - Tag 0 = nil
  - Tag 1 = bool (payload 0=false, 1=true)
  - Tag 2 = heap pointer (payload = pointer bits)
  - Tag 3 = available for your language

Customize tag assignments: `DynModule::new().with_tags(NanBoxTags { nil: 0, bool_tag: 1, ptr: 2 })`

The `dyn_add`, `dyn_lt`, etc. methods generate inline fast paths that check "are both
values floats?" using the NanBox bit pattern. If so, they operate directly on the float
bits. Otherwise, they call the registered slow-path extern.

## Advanced: Raw dynir API

For advanced use cases or statically-typed languages, you can use the raw `dynir` API
directly. The `FunctionBuilder` provides full SSA construction:

### Constants and Arithmetic

```rust
let n = fb.iconst(Type::I64, 42);
let f = fb.f64const(3.14);
let sum = fb.add(a, b);          // integer add
let fsum = fb.fadd(fa, fb);      // float add
```

### Comparisons (return I8)

```rust
let eq = fb.icmp(CmpOp::Eq, a, b);
let lt = fb.icmp(CmpOp::Slt, a, b);
let feq = fb.fcmp(CmpOp::Eq, fa, fb);
```

### Control Flow

```rust
// Branching
let then_bb = fb.create_block(&[]);
let else_bb = fb.create_block(&[]);
let merge_bb = fb.create_block(&[Type::I64]);

fb.br_if(cond_i8, then_bb, &[], else_bb, &[]);

fb.switch_to_block(then_bb);
fb.jump(merge_bb, &[result1]);

fb.switch_to_block(else_bb);
fb.jump(merge_bb, &[result2]);

fb.switch_to_block(merge_bb);
let merged = fb.block_param(merge_bb, 0);
```

### Loops

```rust
let header = fb.create_block(&[Type::I64, Type::I64]); // (i, acc)
let body = fb.create_block(&[Type::I64, Type::I64]);
let exit = fb.create_block(&[Type::I64]);

let zero = fb.iconst(Type::I64, 0);
fb.jump(header, &[zero, zero]);

fb.switch_to_block(header);
let i = fb.block_param(header, 0);
let acc = fb.block_param(header, 1);
let cmp = fb.icmp(CmpOp::Slt, i, n);
fb.br_if(cmp, body, &[i, acc], exit, &[acc]);

fb.switch_to_block(body);
let bi = fb.block_param(body, 0);
let bacc = fb.block_param(body, 1);
let new_acc = fb.add(bacc, bi);
let one = fb.iconst(Type::I64, 1);
let new_i = fb.add(bi, one);
fb.jump(header, &[new_i, new_acc]);

fb.switch_to_block(exit);
let result = fb.block_param(exit, 0);
fb.ret(result);
```

### Stack Slots (mutable storage)

```rust
let slot = fb.create_stack_slot(8, false);
let addr = fb.stack_addr(slot);
fb.store(value, addr, 0);
let loaded = fb.load(Type::I64, addr, 0);
```

### Tagged Value IR Operations

```rust
let tag = fb.tag_of(val);                    // extract tag → I32
let payload = fb.payload(val);               // extract payload → I64
let tagged = fb.make_tagged(1, payload_val); // create tagged value → I64
let is_int = fb.is_tag(val, 1);             // check tag → I8
```

### Guards and Deoptimization

```rust
let deopt = fb.create_deopt(bytecode_offset, "expected integer");
let is_int = fb.is_tag(val, TAG_INT);
fb.guard(is_int, deopt, &[val, other_val]);
// If we get here, val is definitely an integer
```

### Delimited Continuations

```rust
let prompt = fb.create_prompt();
let handler_bb = fb.create_block(&[Type::I64]);
fb.push_prompt(prompt, handler_bb);
// ... code that might abort ...
fb.pop_prompt(prompt);
fb.jump(handler_bb, &[normal_result]);

fb.switch_to_block(handler_bb);
let result = fb.block_param(handler_bb, 0);

// Capture, abort, resume:
let k = fb.capture_slice(prompt, &[live_val1, live_val2]);
fb.abort_to_prompt(prompt, &[abort_value]);
fb.resume_slice(k, &[resume_value]);
let copy = fb.clone_slice(k); // multi-shot
```

## Reference Implementations

- **`contlang`** — Cleanest example. Small language with delimited continuations.
  AST-based lowering. ~343 lines of lowering code. Start here.
- **`lox`** — Lox language (from Crafting Interpreters). Bytecode-based lowering
  with NanBox encoding and extern runtime functions.
- **`lua2dynir`** — Lua 5.1 bytecode translator. More complex, shows register-based
  bytecode translation with NanBox constants.

## Key Principles

1. **Use dynlang for dynamic languages**. `DynModule` + `DynFunc` handle NanBox encoding,
   mutable variables, and inline fast paths. Drop to raw `f.fb` only when needed.

2. **Everything goes through dynir**. Don't write your own interpreter loop or code
   generator. Lower to dynir and let the toolkit handle execution.

3. **Dynamic operations → extern functions**. Type checking, string ops, object property
   access — anything polymorphic gets delegated to extern functions that you bind at
   runtime. The `dyn_add` etc. methods handle the common case (float fast path) inline
   and fall back to your extern for everything else.

4. **All values are I64**. In a dynamically-typed language, every value in the IR is
   `Type::I64` holding a NanBox-encoded value. The tag scheme is chosen when you set
   up the interpreter (`ModuleInterpreter::<NanBox, _>`).

5. **SSA form with block parameters**. There are no phi nodes. Pass values as arguments
   when jumping to blocks. The target block receives them via `block_param`. Or just use
   `def_var`/`get_var`/`set_var` and let stack slots + mem2reg handle it.

6. **Declare first, define later**. Declare all functions with `declare_func` before
   defining any of them. This allows mutual recursion.

7. **Terminators end blocks**. Every block must end with exactly one terminator:
   `ret`, `ret_void`, `jump`, `br_if`, `switch`, `unreachable`, etc. After a terminator,
   call `f.fb.switch_to_block(next_bb)` before emitting more instructions.

## DynFunc Quick Reference

| Category | Methods |
|----------|---------|
| **Variables** | `def_var(name, val)`, `get_var(name)`, `set_var(name, val)`, `push_scope()`, `pop_scope()` |
| **Constants** | `number(f64)`, `nil()`, `bool_val(bool)`, `tagged_const(tag, payload)` |
| **Type checks** | `is_number(v)`, `is_nil(v)`, `is_bool(v)`, `is_ptr(v)`, `is_tagged(v)`, `is_falsey(v)`, `is_truthy(v)` |
| **Branching** | `br_if_truthy(v, then, then_args, else, else_args)` |
| **Arithmetic** | `dyn_add`, `dyn_sub`, `dyn_mul`, `dyn_div`, `dyn_neg` |
| **Comparison** | `dyn_eq`, `dyn_lt`, `dyn_gt` |
| **Box/unbox** | `unbox_number(v)` (I64→F64), `box_number(v)` (F64→I64 + NaN canon) |
| **Raw builder** | `f.fb` — full `FunctionBuilder` access for control flow, calls, etc. |

## FunctionBuilder Quick Reference (via f.fb)

| Category | Methods |
|----------|---------|
| **Blocks** | `entry_block()`, `create_block(&[Type])`, `switch_to_block(id)`, `block_param(block, idx)` |
| **Constants** | `iconst(Type, i64)`, `f64const(f64)` |
| **Arithmetic** | `add`, `sub`, `mul`, `sdiv`, `udiv`, `fadd`, `fsub`, `fmul`, `fdiv` |
| **Bitwise** | `and`, `or`, `xor`, `shl`, `lshr`, `ashr`, `neg`, `not` |
| **Compare** | `icmp(CmpOp, a, b)`, `fcmp(CmpOp, a, b)` → returns I8 |
| **Convert** | `sext`, `zext`, `trunc`, `int_to_float`, `float_to_int`, `bitcast` |
| **Memory** | `create_stack_slot(size, is_gc_root)`, `stack_addr(slot)`, `load(ty, addr, offset)`, `store(val, addr, offset)` |
| **Tagged** | `tag_of`, `payload`, `make_tagged`, `is_tag`, `select` |
| **Calls** | `call(func_ref, &[args])`, `call_indirect(ptr, &[args], ret_ty)` |
| **Control** | `ret(val)`, `ret_void()`, `jump(block, &[args])`, `br_if(cond, then, then_args, else, else_args)`, `switch(val, cases, default)`, `unreachable()` |
| **Guards** | `create_deopt(resume_point, desc)`, `guard(cond, deopt_id, &[live_vals])` |
| **Continuations** | `create_prompt()`, `push_prompt`, `pop_prompt`, `capture_slice`, `clone_slice`, `abort_to_prompt`, `resume_slice` |
| **GC** | `safepoint(&[live_gc_ptrs])` |
