"""LLDB Python script: break on mlirOperationWriteBytecode, dump the MLIR
operation as text by calling mlirOperationDump in the target process.

This captures the MLIR module as it's about to be serialized to disk — i.e.,
the post-transform MLIR including any kernel functions if they haven't been
elaborated yet.

Usage:
    lldb --batch -s commands.txt mojo
    where commands.txt does:
        command script import /path/to/lldb_dump_mlir.py
        run run my_program.mojo
"""
import lldb

def __lldb_init_module(debugger, internal_dict):
    target = debugger.GetSelectedTarget()
    print(f"target: {target}")
    bp = target.BreakpointCreateByName("mlirOperationWriteBytecode", "mojo")
    bp.SetScriptCallbackFunction("lldb_dump_mlir.on_write_bytecode")
    bp.SetAutoContinue(False)  # we want to stop, dump, then continue
    print(f"breakpoint #{bp.GetID()} on mlirOperationWriteBytecode, "
          f"locations={bp.GetNumLocations()}")

_HIT = 0

def on_write_bytecode(frame, bp_loc, internal_dict):
    global _HIT
    _HIT += 1
    n = _HIT
    process = frame.GetThread().GetProcess()
    target = process.GetTarget()
    debugger = target.GetDebugger()
    op_ptr = frame.FindRegister("x0").GetValueAsUnsigned()
    print(f"\n==== mlirOperationWriteBytecode hit #{n} (MlirOperation.ptr = {op_ptr:#x}) ====")
    # Call mlirOperationDump(op) in the target process.
    # The C signature: void mlirOperationDump(MlirOperation op);
    # MlirOperation is a struct { void* ptr } passed by value, ABI = a single x0 pointer.
    expr = f"((void(*)(void*)){target.FindSymbols('mlirOperationDump')[0].GetSymbol().GetStartAddress().GetLoadAddress(target):#x})((void*){op_ptr:#x})"
    print(f"expr: {expr}")
    options = lldb.SBExpressionOptions()
    options.SetTryAllThreads(False)
    options.SetIgnoreBreakpoints(True)
    options.SetUnwindOnError(True)
    options.SetTimeoutInMicroSeconds(30 * 1000 * 1000)
    result = frame.EvaluateExpression(expr, options)
    err = result.GetError()
    if err.Fail():
        print(f"  expr failed: {err}")
    else:
        print(f"  dump invoked OK (output was sent to stderr)")
    # Save raw bytes of the operation pointer for offline analysis
    return False  # stop — caller can `c` to continue
