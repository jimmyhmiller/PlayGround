#!/bin/bash
# Compile each Scheme program through the GC metaprogram (scm2coil -> coil -> native),
# check output against Chez on identical source, and print the program's own COMPUTE time
# (internal monotonic timer, separate from process startup).
cd "$(dirname "$0")"
pass=0; fail=0
for f in *.scm; do
  name="${f%.scm}"
  exp=$(chez -q --script "$f" 2>/dev/null)
  python3 ../meta/scm2coil.py "$f" > "gen_$name.coil" 2>/dev/null
  if ../../coil build "gen_$name.coil" -o "/tmp/prog_$name" >/tmp/b.err 2>&1; then
    got=$(/tmp/prog_$name 2>/tmp/e.err)
    comp=$(grep -oE 'compute_ns=[0-9]+' /tmp/e.err | grep -oE '[0-9]+')
    ms=$(python3 -c "print(f'{${comp:-0}/1e6:.3f}ms')")
    if [ "$got" = "$exp" ]; then printf "  PASS  %-12s %-12s compute=%s\n" "$name" "$got" "$ms"; pass=$((pass+1))
    else echo "  FAIL  $name ours=[$got] chez=[$exp]"; fail=$((fail+1)); fi
  else echo "  BUILD-FAIL $name"; fail=$((fail+1)); fi
  rm -f "gen_$name.coil"
done
echo "=== $pass/$((pass+fail)) match Chez exactly ==="
