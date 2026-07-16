#!/usr/bin/env bash
# Compile evalcore.scm two ways and compare to Chez/Petite on the SAME source.
#   ./run.sh            correctness + which fib the target runs
#   ./run.sh bench      hyperfine benchmark (edit (fib N) in evalcore.scm to scale)
set -e
python3 scm2coil.py evalcore.scm > evalcore.coil
../../coil build evalcore.coil -o /tmp/evalcore-coil >/dev/null
echo "same evalcore.scm source, three hosts:"
echo "  our coil (scheme metaprogram -> native + GC): $(/tmp/evalcore-coil 2>/dev/null)"
echo "  chez  (native compiler):                       $(chez  -q --script evalcore.scm 2>/dev/null)"
echo "  petite (interpreter):                          $(petite -q --script evalcore.scm 2>/dev/null)"
if [ "$1" = "bench" ]; then
  echo; echo "benchmark:"; /tmp/evalcore-coil 2>&1 >/dev/null
  hyperfine -w2 -r10 "/tmp/evalcore-coil" "chez -q --script evalcore.scm" "petite -q --script evalcore.scm"
fi
