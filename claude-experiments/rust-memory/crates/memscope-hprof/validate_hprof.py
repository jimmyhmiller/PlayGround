#!/usr/bin/env python3
"""Independent structural validator for an HPROF 1.0.2 heap dump.

Parses the file per the HPROF binary spec and checks that it is internally
consistent — record framing consumes the whole file, every object reference and
GC root resolves to a defined object (or null), and every instance's field bytes
match its class's declared field count. This is deliberately a separate
implementation from the Rust writer, so a shared bug can't pass both.

Usage: validate_hprof.py <file.hprof>
"""
import struct
import sys

U1 = lambda b, p: b[p]
U2 = lambda b, p: struct.unpack_from(">H", b, p)[0]
U4 = lambda b, p: struct.unpack_from(">I", b, p)[0]
U8 = lambda b, p: struct.unpack_from(">Q", b, p)[0]

# Field/element type -> size in bytes (idSize for object).
PRIM = {2: 8, 4: 1, 5: 2, 6: 4, 7: 8, 8: 1, 9: 2, 10: 4, 11: 8}


def main(path):
    b = memoryview(open(path, "rb").read())
    assert bytes(b[:18]) == b"JAVA PROFILE 1.0.2", "bad magic"
    p = b.index(0) + 1
    idsize = U4(b, p); p += 4
    assert idsize == 8, f"idSize {idsize} (expected 8)"
    p += 8  # timestamp

    classes = {}      # class id -> [field type codes]
    strings = {}      # string id -> bytes
    loaded = set()    # class ids from LOAD_CLASS
    objects = set()   # all object ids
    refs = []         # (referrer, referenced id)
    roots = []
    insts = []        # (obj, class id, field-bytes len)

    n_records = 0
    while p < len(b):
        tag = U1(b, p)
        ln = U4(b, p + 5)
        body = p + 9
        end = body + ln
        n_records += 1
        if tag == 0x01:  # UTF8
            sid = U8(b, body)
            strings[sid] = bytes(b[body + 8:end])
        elif tag == 0x02:  # LOAD_CLASS
            loaded.add(U8(b, body + 4))
        elif tag in (0x1C, 0x0C):  # HEAP_DUMP_SEGMENT / HEAP_DUMP
            q = body
            while q < end:
                sub = U1(b, q); q += 1
                if sub == 0x20:  # CLASS_DUMP
                    cid = U8(b, q); q += 8 + 4 + 8 * 6 + 4
                    cp = U2(b, q); q += 2
                    for _ in range(cp):  # constant pool
                        q += 2; t = U1(b, q); q += 1 + PRIM[t]
                    ns = U2(b, q); q += 2
                    for _ in range(ns):  # static fields
                        q += 8; t = U1(b, q); q += 1 + PRIM[t]
                    ni = U2(b, q); q += 2
                    ftypes = []
                    for _ in range(ni):
                        q += 8  # field name id
                        ftypes.append(U1(b, q)); q += 1
                    classes[cid] = ftypes
                elif sub == 0x21:  # INSTANCE_DUMP
                    oid = U8(b, q); q += 8 + 4
                    cid = U8(b, q); q += 8
                    nb = U4(b, q); q += 4
                    objects.add(oid)
                    insts.append((oid, cid, nb))
                    # field values are references (object fields only in our writer)
                    for k in range(nb // 8):
                        refs.append((oid, U8(b, q + k * 8)))
                    q += nb
                elif sub == 0x22:  # OBJ_ARRAY_DUMP
                    oid = U8(b, q); q += 8 + 4
                    ne = U4(b, q); q += 4 + 8
                    objects.add(oid)
                    for k in range(ne):
                        refs.append((oid, U8(b, q + k * 8)))
                    q += ne * 8
                elif sub == 0x23:  # PRIM_ARRAY_DUMP
                    oid = U8(b, q); q += 8 + 4
                    ne = U4(b, q); q += 4
                    t = U1(b, q); q += 1
                    objects.add(oid)
                    q += ne * PRIM[t]
                elif sub == 0xFF:  # ROOT_UNKNOWN
                    roots.append(U8(b, q)); q += 8
                else:
                    raise SystemExit(f"FAIL: unknown heap sub-record {sub:#x}")
            assert q == end, f"segment overran: {q} != {end}"
        p = end
    assert p == len(b), f"trailing bytes: consumed {p} of {len(b)}"

    # --- consistency checks ---
    errors = []
    for oid, cid, nb in insts:
        if cid not in classes:
            errors.append(f"instance {oid:#x} has undefined class {cid:#x}")
        else:
            want = sum(PRIM[t] for t in classes[cid])
            if nb != want:
                errors.append(f"instance {oid:#x}: {nb} field bytes, class declares {want}")
    dangling = [(a, r) for (a, r) in refs if r != 0 and r not in objects]
    bad_roots = [r for r in roots if r not in objects]
    for cid in classes:
        if cid not in loaded:
            errors.append(f"class {cid:#x} dumped but never LOAD_CLASS'd")

    if dangling:
        errors.append(f"{len(dangling)} dangling references, e.g. {dangling[0][1]:#x}")
    if bad_roots:
        errors.append(f"{len(bad_roots)} roots are not objects")

    print(f"records={n_records} classes={len(classes)} objects={len(objects)} "
          f"refs={len(refs)} roots={len(roots)} strings={len(strings)}")
    if errors:
        for e in errors:
            print("  FAIL:", e)
        sys.exit(1)
    print("OK: structurally valid and internally consistent")


if __name__ == "__main__":
    main(sys.argv[1])
