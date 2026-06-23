"""gc-rust lldb pretty-printers (debugger P3, design §3).

Renders gc-rust heap objects as language values — `Point { x: 3, y: 4 }`,
`Shape::Rect(3, 4)` (enum PAYLOADS, which native DWARF can't express), nested
refs inline — by decoding the reflection blob the compiler bakes into every
binary (`gcrust_type_meta`) and reading object memory at a stop. Mirrors the
runtime's `gc::dump::render_object`.

Usage:
    (lldb) command script import tools/gcr_lldb.py
Then `frame variable` / `v` auto-render gc-rust structs and enums. The
`gcrv <expr>` command prints any expression's value the same way.

Object model (must match gcrust-rt): the Full GC header is 16 bytes; the u16
`type_id` lives at byte offset 8; field offsets in the reflection metadata are
ABSOLUTE (header included), so a field reads at `object_base + offset`.
"""

import struct
import lldb

TYPE_ID_OFFSET = 8       # u16 type_id within the 16-byte Full header
BLOB_SYMBOL = "gcrust_type_meta"
MAX_DEPTH = 4

# Scalar kind tag (reflect::ScalarKind::to_tag) -> (size_bytes, category).
# category: 's' signed int, 'u' unsigned int, 'f' float, 'b' bool, 'c' char,
# 'p' raw pointer.
_SCALAR = {
    0: (1, "s"), 1: (2, "s"), 2: (4, "s"), 3: (8, "s"),
    4: (1, "u"), 5: (2, "u"), 6: (4, "u"), 7: (8, "u"),
    8: (4, "f"), 9: (8, "f"),
    10: (1, "b"), 11: (4, "c"), 12: (8, "p"),
}


# ---------------------------------------------------------------------------
# Reflection blob decoding (mirrors gc::reflect format, see reflect.rs).
# ---------------------------------------------------------------------------
class _Cursor:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def u8(self):
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def u16(self):
        v = struct.unpack_from("<H", self.buf, self.pos)[0]
        self.pos += 2
        return v

    def u32(self):
        v = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def s(self):
        n = self.u16()
        v = self.buf[self.pos:self.pos + n].decode("utf-8", "replace")
        self.pos += n
        return v


def _read_field(c):
    name = c.s()
    offset = c.u16()
    tag = c.u8()
    if tag == 0:       # Ref(target type_id)
        return {"name": name, "off": offset, "ty": ("ref", c.u16())}
    if tag == 1:       # Scalar(kind tag)
        return {"name": name, "off": offset, "ty": ("scalar", c.u8())}
    # Value(value_id)
    return {"name": name, "off": offset, "ty": ("value", c.u16())}


def _read_field_list(c):
    return [_read_field(c) for _ in range(c.u16())]


def _read_kind(c):
    tag = c.u8()
    if tag == 0:       # Struct
        return {"kind": "struct", "fields": _read_field_list(c)}
    if tag == 1:       # Enum
        tag_offset = c.u16()
        nvariants = c.u16()
        variants = []
        for _ in range(nvariants):
            vname = c.s()
            vtag = c.u32()
            variants.append({"name": vname, "tag": vtag, "fields": _read_field_list(c)})
        return {"kind": "enum", "tag_offset": tag_offset, "variants": variants}
    return {"kind": "opaque"}


def _decode(buf):
    """Decode the blob into {types: {type_id: entry}, values: {value_id: entry}}."""
    c = _Cursor(buf)
    types = {}
    for _ in range(c.u32()):
        tid = c.u16()
        name = c.s()
        entry = _read_kind(c)
        entry["name"] = name
        types[tid] = entry
    values = {}
    for _ in range(c.u32()):
        vid = c.u16()
        name = c.s()
        entry = _read_kind(c)
        entry["name"] = name
        values[vid] = entry
    return {"types": types, "values": values}


# ---------------------------------------------------------------------------
# Object rendering.
# ---------------------------------------------------------------------------
def _read_mem(process, addr, size):
    err = lldb.SBError()
    data = process.ReadMemory(addr, size, err)
    if not err.Success() or data is None:
        return None
    return bytes(data)


def _render_scalar(raw, kind_tag):
    size, cat = _SCALAR[kind_tag]
    if cat == "f":
        return repr(struct.unpack("<f" if size == 4 else "<d", raw)[0])
    val = int.from_bytes(raw, "little", signed=(cat == "s"))
    if cat == "b":
        return "true" if val else "false"
    if cat == "c":
        try:
            return "'" + chr(val) + "'"
        except (ValueError, OverflowError):
            return str(val)
    if cat == "p":
        return "0x%x" % val
    return str(val)


def _all_numeric(fields):
    return fields and all(f["name"].isdigit() for f in fields)


def _render_fields(process, base, fields, table, depth):
    """Render a field list at `base`; returns (positional?, [strings])."""
    out = []
    for f in fields:
        off = f["off"]
        kind, arg = f["ty"]
        if kind == "scalar":
            raw = _read_mem(process, base + off, _SCALAR[arg][0])
            out.append(_render_scalar(raw, arg) if raw else "?")
        elif kind == "ref":
            ptr = _read_mem(process, base + off, 8)
            child = int.from_bytes(ptr, "little") if ptr else 0
            out.append(_render_ref(process, child, table, depth - 1))
        else:  # value aggregate (inline) — render in place at base+off
            entry = table["values"].get(arg)
            if entry is None:
                out.append("…")
            else:
                out.append(_render_entry(process, base + off, entry, table, depth - 1))
    return out


def _render_ref(process, addr, table, depth):
    if addr == 0:
        return "null"
    if depth <= 0:
        return "0x%x" % addr
    tid_raw = _read_mem(process, addr + TYPE_ID_OFFSET, 2)
    if not tid_raw:
        return "0x%x" % addr
    tid = int.from_bytes(tid_raw, "little")
    return _render_type(process, addr, tid, table, depth)


def _render_type(process, base, tid, table, depth):
    entry = table["types"].get(tid)
    if entry is None:
        return "<type %d @ 0x%x>" % (tid, base)
    return _render_entry(process, base, entry, table, depth)


def _render_entry(process, base, entry, table, depth):
    """Render a struct/enum entry whose fields sit at `base + field.off`. Shared
    by heap objects (offsets header-absolute) and inline value aggregates
    (offsets value-relative) — same shape, same code."""
    name = entry["name"]
    if entry["kind"] == "struct":
        fields = entry["fields"]
        parts = _render_fields(process, base, fields, table, depth)
        if _all_numeric(fields):  # tuple struct
            return "%s(%s)" % (name, ", ".join(parts))
        body = ", ".join("%s: %s" % (f["name"], p) for f, p in zip(fields, parts))
        return "%s { %s }" % (name, body)
    if entry["kind"] == "enum":
        tag_raw = _read_mem(process, base + entry["tag_offset"], 4)
        tag = int.from_bytes(tag_raw, "little") if tag_raw else None
        variant = next((v for v in entry["variants"] if v["tag"] == tag), None)
        if variant is None:
            return "%s::<tag %s>" % (name, tag)
        if not variant["fields"]:
            return "%s::%s" % (name, variant["name"])
        parts = _render_fields(process, base, variant["fields"], table, depth)
        return "%s::%s(%s)" % (name, variant["name"], ", ".join(parts))
    return "%s{…}" % name  # opaque


# ---------------------------------------------------------------------------
# lldb integration.
# ---------------------------------------------------------------------------
_TABLE = None


def _table_for(target):
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    # The blob is a plain data symbol with NO debug-info DIE, so
    # FindFirstGlobalVariable can't see it. Locate it in the symbol table and
    # read its bytes from the target image (works before `run`, file-backed).
    syms = target.FindSymbols(BLOB_SYMBOL)
    if syms.GetSize() == 0:
        return None
    sym = syms.GetContextAtIndex(0).GetSymbol()
    if not sym or not sym.IsValid():
        return None
    start = sym.GetStartAddress()
    size = sym.GetEndAddress().GetFileAddress() - start.GetFileAddress()
    if size <= 0:
        return None
    err = lldb.SBError()
    buf = target.ReadMemory(start, int(size), err)
    if not err.Success() or buf is None:
        return None
    _TABLE = _decode(bytes(buf))
    return _TABLE


def gcr_summary(valobj, internal_dict):
    """type-summary callback: render a gc-rust object value."""
    target = valobj.GetTarget()
    table = _table_for(target)
    if table is None:
        return ""
    process = valobj.GetProcess()
    base = valobj.GetLoadAddress()
    if base == lldb.LLDB_INVALID_ADDRESS:
        return ""
    # The variable's DWARF type name identifies the gc-rust type; find its id.
    tname = valobj.GetType().GetUnqualifiedType().GetName()
    tid = next((i for i, e in table["types"].items() if e["name"] == tname), None)
    if tid is None:
        return ""
    return _render_type(process, base, tid, table, MAX_DEPTH)


def _register(debugger):
    target = debugger.GetSelectedTarget()
    if not target or not target.IsValid():
        return 0
    table = _table_for(target)
    if table is None:
        return 0
    ci = debugger.GetCommandInterpreter()
    res = lldb.SBCommandReturnObject()
    n = 0
    for entry in table["types"].values():
        if entry["kind"] in ("struct", "enum"):
            ci.HandleCommand(
                'type summary add -F gcr_lldb.gcr_summary "%s"' % entry["name"], res
            )
            n += 1
    return n


def gcrv(debugger, command, result, internal_dict):
    """`gcrv <expr>` — print a gc-rust value via reflection rendering."""
    target = debugger.GetSelectedTarget()
    frame = target.GetProcess().GetSelectedThread().GetSelectedFrame()
    val = frame.EvaluateExpression(command.strip())
    if not val or not val.IsValid():
        result.AppendMessage("gcrv: could not evaluate %r" % command)
        return
    result.AppendMessage(gcr_summary(val, internal_dict) or "<not a gc-rust value>")


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        "command script add -f gcr_lldb.gcrv gcrv"
    )
    n = _register(debugger)
    print("gcr_lldb: registered reflection pretty-printers for %d gc-rust types" % n)
