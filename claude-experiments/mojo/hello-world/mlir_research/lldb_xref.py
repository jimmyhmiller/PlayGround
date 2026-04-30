"""LLDB Python script: break at every "MLIR23.0.0git" xref site, dump backtrace.

Use:
    lldb --batch -o "command script import lldb_xref.py" \
         -o "run vector_add.mojo" \
         -o "quit" \
         /path/to/mojo
"""

# All adrp+add sites that load the producer string "MLIR23.0.0git"
# (computed from llvm-objdump in the un-slid binary)
XREF_SITES = [
    0x1004230b0,   # adrp x1, 0x104200000  + add x1, x1, #0xa9f
    0x100425698,
    0x1008f4ccc,
    0x100900574,
    0x10090073c,
    0x1009040a8,
    0x101759b7c,
]

# The 248-byte ctor that consumes (dst, "MLIR23.0.0git", 13)
CTOR_SITE = 0x1017792e4

import lldb

def __lldb_init_module(debugger, internal_dict):
    target = debugger.GetSelectedTarget()
    print(f"target: {target}")
    # Use file addresses (un-slid). LLDB resolves to load addresses automatically.
    for i, file_addr in enumerate(XREF_SITES):
        bp = target.BreakpointCreateByAddress(file_addr)
        bp.SetScriptCallbackFunction("lldb_xref.on_xref_hit")
        bp.SetAutoContinue(True)
        print(f"breakpoint #{bp.GetID()} at file address {file_addr:#x}")
    bp = target.BreakpointCreateByAddress(CTOR_SITE)
    bp.SetScriptCallbackFunction("lldb_xref.on_ctor_hit")
    bp.SetAutoContinue(True)
    print(f"breakpoint #{bp.GetID()} at ctor file address {CTOR_SITE:#x}")

_HIT_COUNT = {}

def on_xref_hit(frame, bp_loc, internal_dict):
    addr = frame.GetPC()
    _HIT_COUNT[addr] = _HIT_COUNT.get(addr, 0) + 1
    n = _HIT_COUNT[addr]
    print(f"\n==== XREF site {addr:#x} hit #{n} ====")
    thread = frame.GetThread()
    for i in range(min(8, thread.GetNumFrames())):
        f = thread.GetFrameAtIndex(i)
        print(f"  #{i:>2} {f.GetPC():#x}  {f.GetFunctionName() or f.GetSymbol().GetName() or '<unknown>'}")
    return False  # auto-continue

def on_ctor_hit(frame, bp_loc, internal_dict):
    pc = frame.GetPC()
    _HIT_COUNT[pc] = _HIT_COUNT.get(pc, 0) + 1
    n = _HIT_COUNT[pc]
    if n > 5:
        return False  # auto-continue, don't spam
    print(f"\n==== CTOR site {pc:#x} hit #{n} ====")
    # x1 should still hold "MLIR23.0.0git", x2 the length
    error = lldb.SBError()
    process = frame.GetThread().GetProcess()
    x1 = frame.FindRegister("x1").GetValueAsUnsigned()
    x2 = frame.FindRegister("x2").GetValueAsUnsigned()
    s = process.ReadCStringFromMemory(x1, min(x2 + 1, 64), error)
    print(f"  arg1 (str at x1)={s!r}, arg2 (len)={x2}")
    return False
