# ARM64 Instructions for CPython JIT

Generated using `arm-codegen-generic` library.

## Summary

Successfully generated ARM64 instruction encoders for all requested CPython JIT instructions:

1. **ADD (register)**: Add two 64-bit registers (Rd = Rn + Rm)
2. **SUB (register)**: Subtract two 64-bit registers (Rd = Rn - Rm)
3. **CMP (register)**: Compare two 64-bit registers (sets flags, like SUBS with XZR destination)
4. **CBNZ**: Compare and Branch on Non-Zero (64-bit)
5. **B.LT**: Branch if Less Than (signed comparison, condition code 0xB)
6. **B.GE**: Branch if Greater or Equal (signed, condition code 0xA)
7. **B.LE**: Branch if Less or Equal (signed, condition code 0xD)
8. **B.GT**: Branch if Greater Than (signed, condition code 0xC)
9. **B.EQ**: Branch if Equal (condition code 0x0)
10. **B.NE**: Branch if Not Equal (condition code 0x1)

## Generated Files

- **cpython_jit_arm64.rs**: Raw generated code with encoding functions
- **cpython_jit_arm64_enhanced.rs**: Enhanced version with enum variants matching CPython JIT style

## Instruction Encodings

### Low-Level Encoding Functions

```rust
/// ADD X0, X1, X2 (64-bit register addition)
let encoding = add_addsub_shift(1, 0, X2, 0, X1, X0);

/// SUB X0, X1, X2 (64-bit register subtraction)
let encoding = sub_addsub_shift(1, 0, X2, 0, X1, X0);

/// CMP X1, X2 (64-bit register comparison)
let encoding = cmp_subs_addsub_shift(1, 0, X2, 0, X1);

/// CBNZ X0, offset (64-bit compare and branch on non-zero)
let encoding = cbnz(1, offset, X0);

/// Conditional branches (with condition codes)
let b_eq = bcond(offset, 0);  // B.EQ (equal)
let b_ne = bcond(offset, 1);  // B.NE (not equal)
let b_ge = bcond(offset, 10); // B.GE (signed >=)
let b_lt = bcond(offset, 11); // B.LT (signed <)
let b_gt = bcond(offset, 12); // B.GT (signed >)
let b_le = bcond(offset, 13); // B.LE (signed <=)
```

### High-Level Enum Variants

The enhanced version includes enum variants matching the CPython JIT style:

```rust
pub enum ArmAsm {
    AddReg { sf: i32, shift: i32, rm: Register, imm6: i32, rn: Register, rd: Register },
    SubReg { sf: i32, shift: i32, rm: Register, imm6: i32, rn: Register, rd: Register },
    CmpReg { sf: i32, shift: i32, rm: Register, imm6: i32, rn: Register },
    Cbnz { imm19: i32, rt: Register },
    BLt { imm19: i32 },
    BGe { imm19: i32 },
    BLe { imm19: i32 },
    BGt { imm19: i32 },
    BEq { imm19: i32 },
    BNe { imm19: i32 },
}
```

## Integration with CPython JIT

### Step 1: Add Enum Variants to `ArmAsm` enum

Add these variants to the existing `ArmAsm` enum in `/Users/jimmyhmiller/Documents/Code/open-source/cpython/cpython-rust-jit/src/arm64.rs`:

```rust
// ADD (shifted register) - Add two registers
AddReg {
    sf: i32,
    shift: i32,
    rm: Register,
    imm6: i32,
    rn: Register,
    rd: Register,
},

// SUB (shifted register) - Subtract two registers (already exists, but verify encoding)
SubReg {
    sf: i32,
    shift: i32,
    rm: Register,
    imm6: i32,
    rn: Register,
    rd: Register,
},

// CMP (shifted register) - Compare registers
CmpReg {
    sf: i32,
    shift: i32,
    rm: Register,
    imm6: i32,
    rn: Register,
},

// CBNZ - Compare and Branch on Non-Zero (64-bit)
Cbnz { imm19: i32, rt: Register },

// B.LT - Branch if less than (signed)
BLt { imm19: i32 },

// B.GE - Branch if greater than or equal (signed)
BGe { imm19: i32 },

// B.LE - Branch if less than or equal (signed)
BLe { imm19: i32 },

// B.GT - Branch if greater than (signed)
BGt { imm19: i32 },

// B.EQ - Branch if equal
BEq { imm19: i32 },

// B.NE - Branch if not equal
BNe { imm19: i32 },
```

### Step 2: Add Encoding Cases to `ArmAsm::encode()`

Add these encoding cases:

```rust
ArmAsm::AddReg { sf, shift, rm, imm6, rn, rd } => {
    // ADD (shifted register): sf 0 0 01011 shift 0 Rm imm6 Rn Rd
    0b0_00_01011_00_0_00000_000000_00000_00000
        | ((*sf as u32) << 31)
        | ((*shift as u32) << 22)
        | (rm << 16)
        | (truncate_imm::<_, 6>(*imm6) << 10)
        | (rn << 5)
        | (rd << 0)
}

ArmAsm::SubReg { sf, shift, rm, imm6, rn, rd } => {
    // SUB (shifted register): sf 1 0 01011 shift 0 Rm imm6 Rn Rd
    0b0_10_01011_00_0_00000_000000_00000_00000
        | ((*sf as u32) << 31)
        | ((*shift as u32) << 22)
        | (rm << 16)
        | (truncate_imm::<_, 6>(*imm6) << 10)
        | (rn << 5)
        | (rd << 0)
}

ArmAsm::CmpReg { sf, shift, rm, imm6, rn } => {
    // CMP is SUBS with XZR as Rd: sf 1 1 01011 shift 0 Rm imm6 Rn 11111
    0b0_11_01011_00_0_00000_000000_00000_11111
        | ((*sf as u32) << 31)
        | ((*shift as u32) << 22)
        | (rm << 16)
        | (truncate_imm::<_, 6>(*imm6) << 10)
        | (rn << 5)
}

ArmAsm::Cbnz { imm19, rt } => {
    // 64-bit CBNZ: 0b10110101 ..... Rt
    0b10110101_0000000000000000000_00000 | (truncate_imm::<_, 19>(*imm19) << 5) | (rt << 0)
}

ArmAsm::BLt { imm19 } => {
    // B.LT: signed less than, condition code 0b1011 (11)
    0b0101010_0_0000000000000000000_0_1011
        | (truncate_imm::<_, 19>(*imm19) << 5)
}

ArmAsm::BGe { imm19 } => {
    // B.GE: signed greater or equal, condition code 0b1010 (10)
    0b0101010_0_0000000000000000000_0_1010
        | (truncate_imm::<_, 19>(*imm19) << 5)
}

ArmAsm::BLe { imm19 } => {
    // B.LE: signed less or equal, condition code 0b1101 (13)
    0b0101010_0_0000000000000000000_0_1101
        | (truncate_imm::<_, 19>(*imm19) << 5)
}

ArmAsm::BGt { imm19 } => {
    // B.GT: signed greater than, condition code 0b1100 (12)
    0b0101010_0_0000000000000000000_0_1100
        | (truncate_imm::<_, 19>(*imm19) << 5)
}

ArmAsm::BEq { imm19 } => {
    // B.EQ: equal, condition code 0b0000 (0)
    0b0101010_0_0000000000000000000_0_0000
        | (truncate_imm::<_, 19>(*imm19) << 5)
}

ArmAsm::BNe { imm19 } => {
    // B.NE: not equal, condition code 0b0001 (1)
    0b0101010_0_0000000000000000000_0_0001
        | (truncate_imm::<_, 19>(*imm19) << 5)
}
```

### Step 3: Add Builder Helper Methods to `Arm64Builder`

Add these helper methods for convenient instruction generation:

```rust
/// Add two registers: rd = rn + rm
pub fn add_reg(&mut self, rd: Register, rn: Register, rm: Register) {
    self.instructions.push(ArmAsm::AddReg {
        sf: rd.sf(),
        shift: 0,
        rm,
        imm6: 0,
        rn,
        rd,
    });
}

/// Subtract two registers: rd = rn - rm (already exists, verify it uses SubReg variant)
pub fn sub_reg(&mut self, rd: Register, rn: Register, rm: Register) {
    self.instructions.push(ArmAsm::SubReg {
        sf: rd.sf(),
        shift: 0,
        rm,
        imm6: 0,
        rn,
        rd,
    });
}

/// Compare two registers: CMP rn, rm
pub fn cmp_reg(&mut self, rn: Register, rm: Register) {
    self.instructions.push(ArmAsm::CmpReg {
        sf: rn.sf(),
        shift: 0,
        rm,
        imm6: 0,
        rn,
    });
}

/// Compare and branch on non-zero: CBNZ rt, label
pub fn cbnz_label(&mut self, rt: Register, label_id: usize) {
    let at = self.instructions.len();
    self.instructions.push(ArmAsm::Cbnz { imm19: 0, rt });
    self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::Cbnz });
}

/// Branch if less than (signed): B.LT label
pub fn b_lt_label(&mut self, label_id: usize) {
    let at = self.instructions.len();
    self.instructions.push(ArmAsm::BLt { imm19: 0 });
    self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
}

/// Branch if greater or equal (signed): B.GE label
pub fn b_ge_label(&mut self, label_id: usize) {
    let at = self.instructions.len();
    self.instructions.push(ArmAsm::BGe { imm19: 0 });
    self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
}

/// Branch if less or equal (signed): B.LE label
pub fn b_le_label(&mut self, label_id: usize) {
    let at = self.instructions.len();
    self.instructions.push(ArmAsm::BLe { imm19: 0 });
    self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
}

/// Branch if greater than (signed): B.GT label
pub fn b_gt_label(&mut self, label_id: usize) {
    let at = self.instructions.len();
    self.instructions.push(ArmAsm::BGt { imm19: 0 });
    self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
}

/// Branch if equal: B.EQ label
pub fn b_eq_label(&mut self, label_id: usize) {
    let at = self.instructions.len();
    self.instructions.push(ArmAsm::BEq { imm19: 0 });
    self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
}

/// Branch if not equal: B.NE label
pub fn b_ne_label(&mut self, label_id: usize) {
    let at = self.instructions.len();
    self.instructions.push(ArmAsm::BNe { imm19: 0 });
    self.patches.push(BranchPatch { at, target_label: label_id, kind: BranchKind::BCond });
}
```

### Step 4: Update `BranchKind` enum

```rust
#[derive(Debug, Copy, Clone)]
enum BranchKind {
    B,
    Cbz,
    Cbnz,    // Add this
    BCond,   // Add this (for all B.cond variants)
}
```

### Step 5: Update `compile()` method

Update the branch patching logic in the `compile()` method:

```rust
pub fn compile(&self) -> Vec<u8> {
    let mut insts = self.instructions.clone();
    for patch in &self.patches {
        let at = patch.at;
        let target_index = self.labels[patch.target_label];
        let delta = target_index as isize - at as isize;
        match (&mut insts[at], patch.kind) {
            (ArmAsm::B { imm26 }, BranchKind::B) => {
                *imm26 = delta as i32;
            }
            (ArmAsm::Cbz { imm19, .. }, BranchKind::Cbz) => {
                *imm19 = delta as i32;
            }
            // Add these new cases:
            (ArmAsm::Cbnz { imm19, .. }, BranchKind::Cbnz) => {
                *imm19 = delta as i32;
            }
            (ArmAsm::BLt { imm19 }, BranchKind::BCond) |
            (ArmAsm::BGe { imm19 }, BranchKind::BCond) |
            (ArmAsm::BLe { imm19 }, BranchKind::BCond) |
            (ArmAsm::BGt { imm19 }, BranchKind::BCond) |
            (ArmAsm::BEq { imm19 }, BranchKind::BCond) |
            (ArmAsm::BNe { imm19 }, BranchKind::BCond) => {
                *imm19 = delta as i32;
            }
            _ => {}
        }
    }
    insts.iter().flat_map(|inst| inst.to_bytes()).collect()
}
```

## ARM64 Condition Codes Reference

For reference, the ARM64 condition codes used in conditional branches:

| Code | Mnemonic | Meaning (integer) | Flags |
|------|----------|-------------------|-------|
| 0000 | EQ | Equal | Z == 1 |
| 0001 | NE | Not equal | Z == 0 |
| 1010 | GE | Signed greater than or equal | N == V |
| 1011 | LT | Signed less than | N != V |
| 1100 | GT | Signed greater than | Z == 0 && N == V |
| 1101 | LE | Signed less than or equal | Z == 1 || N != V |

## Testing

All generated encodings have been verified to compile and pass tests. Run the included tests with:

```bash
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/arm-codegen-generic
rustc --test cpython_jit_arm64_enhanced.rs -o /tmp/test && /tmp/test
```

## Usage Example

Here's a complete example of using these instructions in the CPython JIT:

```rust
use crate::arm64::*;

let mut builder = Arm64Builder::new();

// Function prologue
builder.prologue();

// Compare X0 with X1
builder.cmp_reg(X0, X1);

// Branch if less than
let less_label = builder.new_label();
builder.b_lt_label(less_label);

// X0 >= X1: Add X2 = X0 + X1
builder.add_reg(X2, X0, X1);
let done_label = builder.new_label();
builder.b_label(done_label);

// X0 < X1: Subtract X2 = X1 - X0
builder.place_label(less_label);
builder.sub_reg(X2, X1, X0);

// Done
builder.place_label(done_label);
builder.mov_reg(X0, X2);  // Return result in X0

// Function epilogue
builder.epilogue();
builder.ret();

// Compile to machine code
let machine_code = builder.compile();
```

## Files Generated

All files are located in `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/arm-codegen-generic/`:

1. **cpython_jit_arm64.rs** - Raw generated encoding functions
2. **cpython_jit_arm64_enhanced.rs** - Enhanced version with enum variants and builder helpers
3. **examples/cpython_jit_instructions.rs** - Generator script
4. **CPYTHON_JIT_INSTRUCTIONS.md** - This documentation

## Next Steps

1. Copy the enum variants and encoding implementations to `cpython-rust-jit/src/arm64.rs`
2. Add the builder helper methods to `Arm64Builder`
3. Update `BranchKind` and `compile()` method as shown above
4. Test the new instructions with the CPython JIT compiler
5. Build and verify with `./scratch/build_cpython.sh`
