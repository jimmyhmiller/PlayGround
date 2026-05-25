"""op_text_tokenize: BPE tokenize a UTF-8 string to int32 token ids.

Stateful: init_op(vocab_path, merges_path) -> handle holding the Tokenizer.
forward(handle, text) -> Python list of int token ids. The orchestrator
then materializes that list into a Mojo-side Buffer if needed.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from bpe_tokenizer import Tokenizer, load_tokenizer, tokenize


@fieldwise_init
struct OpState(Movable):
    var tok: Tokenizer


def init_op(
    vocab_path_obj: PythonObject, merges_path_obj: PythonObject
) raises -> PythonObject:
    var vocab = String(py=vocab_path_obj)
    var merges = String(py=merges_path_obj)
    var tok = load_tokenizer(vocab, merges)
    var ptr = alloc[OpState](1)
    ptr.init_pointee_move(OpState(tok^))
    return PythonObject(Int(ptr))


def tokenize_text(
    handle: PythonObject, text_obj: PythonObject
) raises -> PythonObject:
    """Tokenize the given text. Returns a Python list[int]."""
    var addr = Int(py=handle)
    if addr == 0:
        raise Error("op_text_tokenize: null handle")
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](unsafe_from_address=addr)
    var text = String(py=text_obj)
    var ids = tokenize(text, state_ptr[].tok)
    var out = Python.list()
    for i in range(len(ids)):
        out.append(ids[i])
    return out


def destroy_op(handle: PythonObject) raises -> PythonObject:
    var addr = Int(py=handle)
    if addr == 0:
        return PythonObject(None)
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](unsafe_from_address=addr)
    state_ptr.destroy_pointee()
    state_ptr.free()
    return PythonObject(None)


@export
def PyInit_op_text_tokenize() -> PythonObject:
    try:
        var b = PythonModuleBuilder("op_text_tokenize")
        b.def_function[init_op](
            "init_op", docstring="init_op(vocab_path, merges_path) -> handle"
        )
        b.def_function[tokenize_text](
            "tokenize", docstring="tokenize(handle, text) -> list[int]"
        )
        b.def_function[destroy_op](
            "destroy_op", docstring="destroy_op(handle)"
        )
        return b.finalize()
    except e:
        abort(String("failed to create op_text_tokenize module: ", e))
