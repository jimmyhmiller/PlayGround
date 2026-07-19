#!/usr/bin/env python3
# Make a Mach-O executable's LC_UUID deterministic by overwriting it with a fixed
# value. The system linker stamps a RANDOM LC_UUID into every executable it
# produces (dyld requires the load command to be PRESENT, so it can't simply be
# dropped); two byte-identical objects therefore link to executables that differ
# only in those 16 UUID bytes (and the ad-hoc code-signature that hashes them).
# This is pure link-tool nondeterminism, outside the compiler. We canonicalize the
# UUID to a fixed constant so a reproducible-build check sees identical bytes; the
# caller must then re-sign (codesign -f -s -) so the ad-hoc signature matches.
import sys, struct

LC_UUID = 0x1B
MH_MAGIC_64 = 0xFEEDFACF
MH_CIGAM_64 = 0xCFFAEDFE
FIXED = bytes(range(16))  # deterministic, non-zero, present

def main(path):
    with open(path, "r+b") as f:
        data = bytearray(f.read())
        magic = struct.unpack_from("<I", data, 0)[0]
        if magic == MH_MAGIC_64:
            endi = "<"
        elif magic == MH_CIGAM_64:
            endi = ">"
        else:
            sys.exit(f"canon-uuid: not a thin 64-bit Mach-O: {path} (magic {magic:#x})")
        ncmds = struct.unpack_from(endi + "I", data, 16)[0]
        off = 32  # mach_header_64 is 32 bytes
        patched = 0
        for _ in range(ncmds):
            cmd, cmdsize = struct.unpack_from(endi + "II", data, off)
            if cmd == LC_UUID:
                data[off + 8: off + 24] = FIXED
                patched += 1
            off += cmdsize
        if patched != 1:
            sys.exit(f"canon-uuid: expected exactly 1 LC_UUID, found {patched} in {path}")
        f.seek(0)
        f.write(data)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: canon-uuid.py <macho-executable>")
    main(sys.argv[1])
