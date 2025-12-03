#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          Dynamic Binding Test Suite                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

cargo run < tests/test_dynamic_bindings.txt 2>/dev/null | awk '
BEGIN {
    test_num = 0
    in_test = 0
}

/^;; Test/ {
    test_num++
    test_name = substr($0, 4)
    print ""
    print "Test " test_num ": " test_name
    print "─────────────────────────────────────────────────────────"
    in_test = 1
    next
}

/^user=>/ {
    if (in_test) {
        line = $0
        getline
        if ($0 !~ /^user=>/ && $0 !~ /^$/ && $0 !~ /^;/) {
            # Remove user=> prompt and get expression
            gsub(/^user=> */, "", line)
            if (line != "" && line != "0") {
                printf "  %-40s => %s\n", line, $0
            }
        }
    }
    next
}

/^:quit/ {
    print ""
    print "✓ All tests completed!"
    exit
}
'
