#!/usr/bin/env bash
# Object-writer gate for the x86-64 backend: elf.coil must emit an object that
# the SYSTEM toolchain accepts, not merely one that looks right.
#
# selfhost/src/elf_selftest.coil hand-assembles
#     int main(void) { puts("elf ok"); return 42; }
# and writes it through elf.coil. Here we:
#   1. check readelf can parse it and reports the right class/type/machine;
#   2. check the symbol table obeys ELF's local-before-global rule (sh_info);
#   3. check both relocations are present with the right types;
#   4. LINK it with cc and RUN it — the end-to-end proof. A bad section header,
#      a wrong sh_info, or a mis-signed addend shows up here as a link error or
#      a wrong exit code.
#   5. check the link produced no warnings (a missing .note.GNU-stack silently
#      gives the whole program an executable stack).
#
# Usage: selfhost/oracle/x64/gate-elf.sh <coil-binary>
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN=${1:?usage: gate-elf.sh <coil-binary>}
[ -x "$BIN" ] || { echo "GATE FAIL: binary not executable: $BIN"; exit 2; }

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
OBJ="$WORK/elf_selftest.o"
EXE="$WORK/elf_selftest"
fail=0
check() {  # check <description> <condition-output> <expected-substring>
  if printf '%s' "$2" | grep -q "$3"; then
    echo "  ok   — $1"
  else
    echo "  FAIL — $1 (wanted /$3/, got: $(printf '%s' "$2" | head -1))"
    fail=$((fail+1))
  fi
}

"$BIN" run selfhost/src/elf_selftest.coil -- "$OBJ" >/dev/null || {
  echo "GATE FAIL: elf_selftest.coil did not run"; exit 2; }
[ -f "$OBJ" ] || { echo "GATE FAIL: no object written"; exit 2; }

# ---- 1. header ----
hdr=$(readelf -h "$OBJ" 2>&1) || { echo "GATE FAIL: readelf rejected the object"; echo "$hdr"; exit 1; }
check "ELF64 class"           "$hdr" "ELF64"
check "relocatable (ET_REL)"  "$hdr" "REL (Relocatable file)"
check "machine is x86-64"     "$hdr" "X86-64"

# ---- 2. symbol table: locals before globals, sh_info marks the boundary ----
syms=$(readelf -sW "$OBJ" 2>&1)
check "main is a global FUNC" "$syms" "GLOBAL DEFAULT.*main"
check "puts is undefined"     "$syms" "puts"
# readelf validates sh_info itself: a wrong value makes it print this warning.
badinfo=$(readelf -sW "$OBJ" 2>&1 | grep -ci "symbol table .* has a sh_info of\|local symbol.* at index" || true)
if [ "$badinfo" = 0 ]; then
  echo "  ok   — symtab sh_info is consistent (locals precede globals)"
else
  echo "  FAIL — readelf complains about the local/global boundary"; fail=$((fail+1))
fi

# ---- 3. relocations ----
rels=$(readelf -rW "$OBJ" 2>&1)
check "PC32 reloc for the string"  "$rels" "R_X86_64_PC32"
check "PLT32 reloc for puts"       "$rels" "R_X86_64_PLT32"

# ---- 4+5. the real test: link it and run it, with no linker warnings ----
lnk=$(cc "$OBJ" -o "$EXE" 2>&1); lrc=$?
if [ $lrc -ne 0 ]; then
  echo "  FAIL — cc could not link the object:"; echo "$lnk" | head -5; fail=$((fail+1))
else
  echo "  ok   — cc linked the object"
  if [ -n "$lnk" ]; then
    echo "  FAIL — linker emitted warnings:"; echo "$lnk" | head -3; fail=$((fail+1))
  else
    echo "  ok   — link produced no warnings"
  fi
  out=$("$EXE"); rc=$?
  [ "$out" = "elf ok" ] && echo "  ok   — program printed 'elf ok'" \
                        || { echo "  FAIL — program printed '$out'"; fail=$((fail+1)); }
  [ "$rc" = 42 ] && echo "  ok   — program exited 42" \
                 || { echo "  FAIL — program exited $rc, want 42"; fail=$((fail+1)); }
fi

echo
[ "$fail" = 0 ] && { echo "x64 gate-elf: PASS"; exit 0; } || { echo "x64 gate-elf: $fail check(s) FAILED"; exit 1; }
