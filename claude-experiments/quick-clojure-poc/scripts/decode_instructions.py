#!/usr/bin/env python3
"""Decode ARM64 instructions from hex"""

instructions = [
    (0x0000, 0xa9bf7bfd, "stp x29, x30, [sp, #-16]! - Save FP and LR"),
    (0x0004, 0xaa1f03fd, "mov x29, sp - Set frame pointer"),
    (0x0008, 0xd10063ff, "sub sp, sp, #24 - Allocate stack space"),
    (0x000c, 0xd2800109, "mov x9, #8 - Load constant 8"),
    (0x0010, 0xf81e83a9, "stur x9, [x29, #-24] - SPILL STORE slot 4"),
    (0x0014, 0xd2800209, "mov x9, #16 - Load constant 16"),
    (0x0018, 0xf81e03a9, "stur x9, [x29, #-32] - SPILL STORE slot 5"),
    (0x001c, 0xd2800314, "mov x20, #24 - Load constant 24"),
    (0x0020, 0xd2800413, "mov x19, #32 - Load constant 32"),
    (0x0024, 0xd2800516, "mov x22, #40 - Load constant 40"),
    (0x0028, 0x9343fe75, "asr x21, x19, #3 - Untag x19"),
    (0x002c, 0x9343fed3, "asr x19, x22, #3 - Untag x22"),
    (0x0030, 0x8b1302b6, "add x22, x21, x19 - Add"),
    (0x0034, 0xd37df2d3, "lsl x19, x22, #3 - Tag result"),
    (0x0038, 0x9343fe96, "asr x22, x20, #3 - Untag x20"),
    (0x003c, 0x9343fe74, "asr x20, x19, #3 - Untag x19"),
    (0x0040, 0x8b1402d3, "add x19, x22, x20 - Add"),
    (0x0044, 0xd37df274, "lsl x20, x19, #3 - Tag result"),
    (0x0048, 0xf85e03a9, "ldur x9, [x29, #-32] - SPILL LOAD slot 5"),
    (0x004c, 0x9343fd33, "asr x19, x9, #3 - Untag spilled value"),
    (0x0050, 0x9343fe96, "asr x22, x20, #3 - Untag x20"),
    (0x0054, 0x8b160274, "add x20, x19, x22 - Add"),
    (0x0058, 0xd37df296, "lsl x22, x20, #3 - Tag result"),
    (0x005c, 0xf85e83a9, "ldur x9, [x29, #-24] - SPILL LOAD slot 4"),
    (0x0060, 0x9343fd34, "asr x20, x9, #3 - Untag spilled value"),
    (0x0064, 0x9343fed3, "asr x19, x22, #3 - Untag x22"),
    (0x0068, 0x8b130296, "add x22, x20, x19 - Add"),
    (0x006c, 0xd37df2d3, "lsl x19, x22, #3 - Tag result"),
    (0x0070, 0xaa1303e0, "mov x0, x19 - Move result to x0"),
    (0x0074, 0x9343fc00, "asr x0, x0, #3 - Untag for return"),
    (0x0078, 0x910063ff, "add sp, sp, #24 - Deallocate stack"),
    (0x007c, 0xa8c17bfd, "ldp x29, x30, [sp], #16 - Restore FP and LR"),
    (0x0080, 0xd65f03c0, "ret - Return"),
]

print("ARM64 JIT Code Disassembly")
print("=" * 80)
print()

for addr, instr, desc in instructions:
    print(f"0x{addr:04x}: 0x{instr:08x}  {desc}")

print()
print("=" * 80)
print("ANALYSIS:")
print("=" * 80)
print()
print("Prologue (0x0000-0x0008):")
print("  1. Save x29, x30 to stack")
print("  2. Set x29 = sp (frame pointer)")
print("  3. Allocate 24 bytes of stack space")
print()
print("Expected stack layout after prologue:")
print("  [x29 + 0]: saved x29, x30")
print("  [x29 - 16]: (gap)")
print("  [x29 - 24]: spill slot 4  (instruction 0x0010)")
print("  [x29 - 32]: spill slot 5  (instruction 0x0018)")
print()
print("Issue Analysis:")
print("  The prologue DOES set x29 = sp at instruction 0x0004")
print("  Then allocates stack space at 0x0008")
print("  After that, x29 should be valid for all spill operations")
print()
print("WAIT - I see the problem!")
print("  Instruction 0x0008: sub sp, sp, #24")
print("  This should be sub sp, sp, #48 or more!")
print()
print("  We have 6 spill slots (0-5) according to the debug output.")
print("  That's 6 * 8 = 48 bytes needed.")
print("  But we only allocated 24 bytes!")
print()
print("  Stack calculation is wrong in the codegen!")
