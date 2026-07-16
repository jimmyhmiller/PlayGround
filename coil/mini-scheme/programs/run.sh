#!/bin/bash
# Compile each Scheme program through the GC metaprogram (scm2coil -> coil -> native)
# and check its output against Chez Scheme running the identical source.
cd "$(dirname "$0")"
pass=0; fail=0
for f in *.scm; do
  name="${f%.scm}"
  exp=$(chez -q --script "$f" 2>/dev/null)
  python3 ../meta/scm2coil.py "$f" > "gen_$name.coil" 2>/dev/null
  if ../../coil build "gen_$name.coil" -o "/tmp/prog_$name" >/tmp/b.err 2>&1; then
    got=$(/tmp/prog_$name 2>/dev/null)
    if [ "$got" = "$exp" ]; then echo "  PASS  $(printf %-12s $name) $got"; pass=$((pass+1))
    else echo "  FAIL  $name ours=[$got] chez=[$exp]"; fail=$((fail+1)); fi
  else echo "  BUILD-FAIL $name"; fail=$((fail+1)); fi
  rm -f "gen_$name.coil"
done
echo "=== $pass/$((pass+fail)) match Chez exactly ==="
