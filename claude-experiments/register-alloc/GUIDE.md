# regalloc Integration Guide

This guide explains how to integrate the `regalloc` library with your compiler.
There are two sides to the integration: **describing your IR** and **describing
your target machine**. Once both are done, you call an allocator and apply the
result.

## Overview

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  Your IR      │     │  Your Target │     │  Allocator          │
│  impl Function│────▶│  impl RegInfo│────▶│  impl RegisterAlloc │
│               │     │  impl CC     │     │  (linear_scan, ...) │
└──────────────┘     └──────────────┘     └──────────┬──────────┘
                                                      │
                                                      ▼
                                               ┌──────────────┐
                                               │  Allocation   │
                                               │  - reg map    │
                                               │  - moves      │
                                               │  - spill slots│
                                               └──────────────┘
```

You implement traits. The allocator reads your IR through those traits, never
mutating it. The allocator produces an `Allocation` — a map from virtual
register operands to physical registers, plus a list of moves/spills you need
to insert. You apply those results to your IR however you like.

---

## Step 1: Describe your IR — `Function`

The `Function` trait is how the allocator reads your program. You don't need to
convert your IR into some intermediate format; you just teach the allocator how
to traverse what you already have.

### Core types you'll use

```rust
use regalloc::types::*;

// VReg(u32)  — your virtual/temporary register names
// PReg(u16)  — physical register numbers
// RegClass(u8) — register class IDs (you decide the numbering)
// BlockId(u32) — basic block IDs
// InstId(u32)  — instruction IDs (must be unique across the whole function)
```

### What `Function` looks like

```rust
pub trait Function {
    // Iterator types — use whatever your IR naturally gives you.
    type BlockIter<'a>: Iterator<Item = BlockId> where Self: 'a;
    type InstIter<'a>: Iterator<Item = InstId> where Self: 'a;
    type OperandIter<'a>: Iterator<Item = Operand> where Self: 'a;
    type SuccIter<'a>: Iterator<Item = BlockId> where Self: 'a;
    type PredIter<'a>: Iterator<Item = BlockId> where Self: 'a;

    fn num_vregs(&self) -> usize;
    fn vreg_class(&self, vreg: VReg) -> RegClass;

    fn blocks(&self) -> Self::BlockIter<'_>;       // layout order
    fn num_blocks(&self) -> usize;
    fn entry_block(&self) -> BlockId;

    fn block_insts(&self, block: BlockId) -> Self::InstIter<'_>;
    fn block_succs(&self, block: BlockId) -> Self::SuccIter<'_>;
    fn block_preds(&self, block: BlockId) -> Self::PredIter<'_>;

    fn inst_operands(&self, inst: InstId) -> Self::OperandIter<'_>;
    fn is_branch(&self, inst: InstId) -> bool;
    fn is_return(&self, inst: InstId) -> bool;
    fn is_call(&self, inst: InstId) -> bool;

    fn inst_clobbers(&self, inst: InstId) -> &[PReg]; // regs trashed by this inst
    fn block_params(&self, block: BlockId) -> &[VReg]; // SSA phi defs
    fn branch_args(&self, inst: InstId, succ_idx: usize) -> &[VReg]; // SSA phi args
    fn num_insts(&self) -> usize;
}
```

### Example implementation

Suppose your IR looks like this:

```rust
struct MyFunc {
    blocks: Vec<MyBlock>,
    insts: Vec<MyInst>,
}
struct MyBlock {
    id: u32,
    inst_range: Range<u32>,   // indices into MyFunc.insts
    succs: Vec<u32>,
    preds: Vec<u32>,
    params: Vec<u32>,         // vreg IDs for block params (phis)
}
struct MyInst {
    opcode: Opcode,
    operands: Vec<MyOperand>,
    clobbers: Vec<u16>,       // PReg IDs
}
```

Your `Function` impl translates between your representation and the trait's
types:

```rust
impl Function for MyFunc {
    type BlockIter<'a> = impl Iterator<Item = BlockId> + 'a;
    // ... etc. Use map/copied adapters on your slices.

    fn num_vregs(&self) -> usize { self.vreg_count }
    fn vreg_class(&self, vreg: VReg) -> RegClass {
        self.vreg_classes[vreg.0 as usize]
    }

    fn blocks(&self) -> Self::BlockIter<'_> {
        (0..self.blocks.len()).map(|i| BlockId(i as u32))
    }

    fn block_insts(&self, block: BlockId) -> Self::InstIter<'_> {
        let b = &self.blocks[block.0 as usize];
        (b.inst_range.start..b.inst_range.end).map(InstId)
    }

    fn inst_operands(&self, inst: InstId) -> Self::OperandIter<'_> {
        self.insts[inst.0 as usize].operands.iter().map(|op| {
            // Convert your MyOperand to regalloc::Operand
            Operand {
                reg: Reg::Virtual(VReg(op.vreg)),
                kind: convert_kind(op.kind),
                constraint: convert_constraint(op.constraint),
            }
        })
    }

    fn inst_clobbers(&self, inst: InstId) -> &[PReg] {
        // Safety: PReg is repr-transparent over u16, but you'll
        // likely just store Vec<PReg> directly.
        &self.insts[inst.0 as usize].clobbers_as_pregs
    }

    // ... fill in the rest
}
```

### Operand details

Each operand has three parts:

```rust
pub struct Operand {
    pub reg: Reg,                   // which vreg (or preg) this refers to
    pub kind: OperandKind,          // how the instruction uses it
    pub constraint: OperandConstraint, // where it can live
}
```

**`OperandKind`** — how the value flows:

| Kind       | Meaning                                                              |
|------------|----------------------------------------------------------------------|
| `Use`      | Read. The value must be in a register before the instruction.        |
| `Def`      | Write. The value is produced by this instruction (may reuse an input register). |
| `UseDef`   | Read then write. Same register for both (two-address).               |
| `EarlyDef` | Write, but the register must be different from all inputs. Used when the output is written before inputs are fully consumed. |

**`OperandConstraint`** — where it can go:

| Constraint       | Meaning                                                         |
|------------------|-----------------------------------------------------------------|
| `RegClass(rc)`   | Any register in class `rc`.                                     |
| `FixedReg(preg)` | Must be exactly this physical register. Use for instructions like x86 `idiv` that mandate specific registers. |
| `Tied(idx)`      | Must be the same register as operand `idx` in the same instruction. |
| `Reuse(idx)`     | Like Tied, but the allocator may insert a move to satisfy it.   |
| `RegOrStack(rc)` | Can be a register or a spill slot (memory operand).             |

### Modeling common instruction patterns

**Normal 3-operand instruction** (`add dst, a, b`):
```rust
vec![
    Operand { reg: Reg::Virtual(dst), kind: Def, constraint: RegClass(GPR) },
    Operand { reg: Reg::Virtual(a),   kind: Use, constraint: RegClass(GPR) },
    Operand { reg: Reg::Virtual(b),   kind: Use, constraint: RegClass(GPR) },
]
```

**x86-style 2-address instruction** (`add dst/a, b` — dst overwrites a):
```rust
vec![
    Operand { reg: Reg::Virtual(dst), kind: Def,    constraint: Reuse(1) },
    Operand { reg: Reg::Virtual(a),   kind: Use,    constraint: RegClass(GPR) },
    Operand { reg: Reg::Virtual(b),   kind: Use,    constraint: RegClass(GPR) },
]
```
The allocator will ensure `dst` and `a` are in the same physical register,
inserting a move beforehand if needed.

**x86 `idiv`** (quotient in rax, remainder in rdx, dividend must be in rax):
```rust
vec![
    Operand { reg: Reg::Virtual(quot), kind: Def, constraint: FixedReg(RAX) },
    Operand { reg: Reg::Virtual(rem),  kind: Def, constraint: FixedReg(RDX) },
    Operand { reg: Reg::Virtual(dvd),  kind: Use, constraint: FixedReg(RAX) },
    Operand { reg: Reg::Virtual(dvs),  kind: Use, constraint: RegClass(GPR) },
]
```

**Call instruction** — a call clobbers all caller-saved registers:
```rust
// Operands for args/returns as usual, plus:
fn inst_clobbers(&self, inst: InstId) -> &[PReg] {
    if self.is_call(inst) {
        &self.caller_saved_regs  // everything not preserved across calls
    } else {
        &[]
    }
}
```

### SSA block parameters (phi nodes)

If your IR uses SSA with block parameters instead of phi instructions:

```rust
fn block_params(&self, block: BlockId) -> &[VReg] {
    // The virtual registers defined at block entry (phi outputs)
    &self.blocks[block.0 as usize].params
}

fn branch_args(&self, inst: InstId, succ_idx: usize) -> &[VReg] {
    // The values passed to the target block's params
    &self.branch_inst(inst).args[succ_idx]
}
```

If your IR does NOT use SSA block parameters, return empty slices for both.

---

## Step 2: Describe your target — `RegInfo` + `CallingConvention`

Two traits describe the machine. Implementing both gives you a `Target`
automatically via a blanket impl.

### `RegInfo` — the register file

```rust
pub trait RegInfo {
    type RegIter<'a>: Iterator<Item = PReg> where Self: 'a;

    fn reg_classes(&self) -> &[RegClass];
    fn class_regs(&self, class: RegClass) -> Self::RegIter<'_>;
    fn class_size(&self, class: RegClass) -> usize;
    fn reg_class_of(&self, reg: PReg) -> RegClass;

    fn reg_name(&self, reg: PReg) -> &str;    // "rax", "x0", etc.
    fn class_name(&self, class: RegClass) -> &str;  // "GPR", "FP"

    fn spill_size(&self, class: RegClass) -> u32;   // bytes to save one reg
    fn spill_align(&self, class: RegClass) -> u32;

    // Optional: sub-register relationships (x86 rax/eax/ax/al)
    fn is_sub_reg(&self, sub: PReg, sup: PReg) -> bool { false }
    fn overlapping_regs(&self, reg: PReg) -> &[PReg] { &[] }
}
```

**Key design decision**: `class_regs` returns registers in *allocation
preference order*. Put the registers you'd like the allocator to use first at
the front. For example, on AArch64 you might put callee-saved registers last so
the allocator avoids them unless necessary (reducing prologue/epilogue cost).

### `CallingConvention` — what survives calls

```rust
pub trait CallingConvention {
    fn callee_saved(&self) -> &[PReg];   // preserved across calls
    fn caller_saved(&self) -> &[PReg];   // trashed by calls
    fn arg_regs(&self, class: RegClass) -> &[PReg];  // argument passing order
    fn ret_regs(&self, class: RegClass) -> &[PReg];  // return value registers
    fn reserved_regs(&self) -> &[PReg];  // never allocate (sp, fp, etc.)

    // Optional:
    fn stack_pointer(&self) -> Option<PReg> { None }
    fn frame_pointer(&self) -> Option<PReg> { None }
}
```

`reserved_regs` is critical: these registers are **never** assigned to any
virtual register. Always include your stack pointer here. Include the frame
pointer if it's reserved.

### Example: AArch64

```rust
struct AArch64Target;

const GPR: RegClass = RegClass(0);
const FPR: RegClass = RegClass(1);

// x0-x30 are GPRs; x31 is sp/zr (reserved)
// d0-d31 are FPRs

impl RegInfo for AArch64Target {
    type RegIter<'a> = std::iter::Copied<std::slice::Iter<'a, PReg>>;

    fn reg_classes(&self) -> &[RegClass] { &[GPR, FPR] }

    fn class_regs(&self, class: RegClass) -> Self::RegIter<'_> {
        match class {
            GPR => GPR_REGS.iter().copied(),
            FPR => FPR_REGS.iter().copied(),
            _ => unreachable!(),
        }
    }

    fn class_size(&self, class: RegClass) -> usize {
        match class { GPR => 30, FPR => 32, _ => unreachable!() }
    }

    fn reg_class_of(&self, reg: PReg) -> RegClass {
        if reg.0 < 31 { GPR } else { FPR }
    }

    fn reg_name(&self, reg: PReg) -> &str {
        NAMES[reg.0 as usize]  // "x0", "x1", ..., "d0", "d1", ...
    }
    fn class_name(&self, class: RegClass) -> &str {
        match class { GPR => "GPR", FPR => "FPR", _ => unreachable!() }
    }
    fn spill_size(&self, _class: RegClass) -> u32 { 8 }
    fn spill_align(&self, _class: RegClass) -> u32 { 8 }
}

impl CallingConvention for AArch64Target {
    fn callee_saved(&self) -> &[PReg] {
        &[PReg(19), PReg(20), /* ... */ PReg(28)] // x19-x28
    }
    fn caller_saved(&self) -> &[PReg] {
        &[PReg(0), PReg(1), /* ... */ PReg(18)]   // x0-x18
    }
    fn arg_regs(&self, class: RegClass) -> &[PReg] {
        match class {
            GPR => &[PReg(0), PReg(1), PReg(2), PReg(3),
                     PReg(4), PReg(5), PReg(6), PReg(7)],
            _ => &[/* d0-d7 */],
        }
    }
    fn ret_regs(&self, _class: RegClass) -> &[PReg] {
        &[PReg(0), PReg(1)]  // x0, x1
    }
    fn reserved_regs(&self) -> &[PReg] {
        &[PReg(31)]  // sp
    }
}
// Target is auto-implemented: impl<T: RegInfo + CallingConvention> Target for T {}
```

---

## Step 3: Run the allocator

```rust
use regalloc::linear_scan::LinearScanAllocator;
use regalloc::allocator::RegisterAllocator;

let func: MyFunc = /* ... */;
let target = AArch64Target;

let mut allocator = LinearScanAllocator;
let allocation = allocator.allocate(&func, &target)?;
```

That's it. The allocator reads your IR through the `Function` trait, reads
machine info through `RegInfo`/`CallingConvention`, and returns an `Allocation`.

---

## Step 4: Apply the allocation

The `Allocation` tells you three things:

### 1. Register assignments

```rust
// For each instruction, for each operand index:
let preg: Option<PReg> = allocation.get(inst_id, operand_index);
```

Walk your instructions and rewrite virtual registers to the physical registers
the allocator chose.

### 2. Inserted moves

```rust
for mv in &allocation.moves {
    match &mv.at {
        MovePosition::Before(inst) => {
            // Insert a move BEFORE this instruction.
        }
        MovePosition::After(inst) => {
            // Insert a move AFTER this instruction.
        }
        MovePosition::BlockEdge { from, to } => {
            // Insert on the CFG edge from block `from` to block `to`.
            // Typically at the end of `from` (before its terminator)
            // or at the start of `to` (if `to` has one predecessor).
        }
    }

    match (&mv.from, &mv.to) {
        (MoveOperand::Reg(src), MoveOperand::Reg(dst)) => {
            // emit: mov dst, src
        }
        (MoveOperand::Reg(src), MoveOperand::SpillSlot(slot)) => {
            // emit: store src -> stack[slot]
        }
        (MoveOperand::SpillSlot(slot), MoveOperand::Reg(dst)) => {
            // emit: load stack[slot] -> dst
        }
        (MoveOperand::SpillSlot(s1), MoveOperand::SpillSlot(s2)) => {
            // emit: load s1 -> scratch; store scratch -> s2
        }
    }
}
```

You decide how spill slots map to actual stack offsets. The allocator tells you
how many slots were used (`allocation.num_spill_slots`) and each slot's class
(from the move's `class` field), so you can compute the frame size:

```rust
let frame_size: u32 = (0..allocation.num_spill_slots)
    .map(|i| /* spill_size for this slot's class */)
    .sum();
```

### 3. Spill slot map

```rust
// Which vregs were spilled (useful for debugging):
for (vreg, slot) in &allocation.spill_slots {
    println!("{:?} lives at stack slot {:?}", vreg, slot);
}
```

---

## Step 5: Verify (testing)

The library includes a verifier that checks allocations for correctness:

```rust
use regalloc::verify;

match verify::verify(&func, &target, &allocation) {
    Ok(()) => { /* allocation is correct */ }
    Err(errors) => {
        for e in &errors {
            eprintln!("verify: {}", e);
        }
    }
}
```

The verifier checks:
- Every virtual register operand got assigned a physical register
- Fixed-register constraints are satisfied
- Tied/Reuse constraints are satisfied
- Reserved registers were never assigned
- No two definitions at the same instruction write the same register

Run this in your test suite on every allocation. It catches bugs fast.

---

## Step 6: Measure quality — `CostModel`

You can evaluate how good an allocation is:

```rust
use regalloc::cost::{CostModel, UniformCostModel, AllocationCost};

let cost: AllocationCost = UniformCostModel.evaluate(&func, &target, &allocation);
println!("spill loads: {}", cost.spill_loads);
println!("spill stores: {}", cost.spill_stores);
println!("reg moves: {}", cost.reg_moves);
println!("total cost: {}", cost.total_cost);
```

To define your own cost model:

```rust
struct MyCostModel {
    block_freqs: HashMap<BlockId, f64>,  // from profiling
}

impl CostModel for MyCostModel {
    fn spill_store_cost(&self, class: RegClass) -> f64 {
        match class {
            GPR => 4.0,   // 4-cycle store
            FPR => 6.0,   // wider store is more expensive
            _ => 4.0,
        }
    }
    fn spill_load_cost(&self, class: RegClass) -> f64 {
        match class {
            GPR => 3.0,
            FPR => 5.0,
            _ => 3.0,
        }
    }
    fn reg_move_cost(&self, _class: RegClass) -> f64 { 1.0 }
    fn block_frequency(&self, block: BlockId) -> f64 {
        *self.block_freqs.get(&block).unwrap_or(&1.0)
    }
}
```

This is useful for comparing allocator algorithms or tuning your IR to reduce
pressure in hot loops.

---

## Step 7: Plug in a different allocator

Every allocator implements the same trait:

```rust
pub trait RegisterAllocator {
    fn allocate<F: Function, T: Target>(
        &mut self, func: &F, target: &T,
    ) -> Result<Allocation, AllocError>;

    fn name(&self) -> &str;
}
```

To swap allocators:

```rust
use regalloc::linear_scan::LinearScanAllocator;

// Just change this line to use a different algorithm:
let mut alloc = LinearScanAllocator;
let result = alloc.allocate(&func, &target)?;
```

To implement your own:

```rust
pub struct MyGraphColoringAllocator { /* ... */ }

impl RegisterAllocator for MyGraphColoringAllocator {
    fn name(&self) -> &str { "graph-coloring" }

    fn allocate<F: Function, T: Target>(
        &mut self, func: &F, target: &T,
    ) -> Result<Allocation, AllocError> {
        let mut alloc = Allocation::new();
        // ... build interference graph ...
        // ... color it ...
        // ... for each vreg, call alloc.set(inst, op_idx, preg) ...
        // ... for spilled vregs, push InsertedMoves ...
        Ok(alloc)
    }
}
```

Then test it against the built-in test suite:

```rust
use regalloc::testing::*;
use regalloc::cost::UniformCostModel;

let target = TestTarget::with_gpr(4);
let cm = UniformCostModel;

for func in &[
    TestSuite::straight_line_no_pressure(),
    TestSuite::high_pressure(8),
    TestSuite::diamond(),
    TestSuite::simple_loop(),
    TestSuite::fixed_constraints(),
    TestSuite::with_call(),
    TestSuite::two_address(),
    TestSuite::early_def_test(),
] {
    let result = run_test(&mut MyGraphColoringAllocator::new(), func, &target, Some(&cm));
    assert!(result.is_ok(), "failed on test: {:?}", result.error);
}
```

Compare two allocators:

```rust
let (r1, r2) = compare_allocators(
    &mut LinearScanAllocator,
    &mut MyGraphColoringAllocator::new(),
    &func,
    &target,
    &cm,
);
println!("{}: cost={}", r1.allocator_name, r1.cost.unwrap().total_cost);
println!("{}: cost={}", r2.allocator_name, r2.cost.unwrap().total_cost);
```

---

## Quick reference: what to implement

| You have...                      | Implement...                        |
|----------------------------------|-------------------------------------|
| A compiler IR                    | `Function`                          |
| A target architecture            | `RegInfo` + `CallingConvention`     |
| A new allocation algorithm       | `RegisterAllocator`                 |
| Custom performance metrics       | `CostModel`                         |

Each of these is independent. You can use the built-in `TestFunction` +
`TestTarget` to test a new allocator without writing any IR/target code. You can
use the built-in `LinearScanAllocator` + `TestTarget` to test a new IR
integration without writing an allocator. Mix and match.
