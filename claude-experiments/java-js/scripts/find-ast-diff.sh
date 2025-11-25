#!/bin/bash

FILE="$1"

if [ -z "$FILE" ]; then
    echo "Usage: $0 <javascript-file>"
    exit 1
fi

# Parse with Acorn and save
node scripts/parse-with-acorn.js "$FILE" > /tmp/acorn.json 2>&1

# Parse with Java and save
mvn exec:java -Dexec.mainClass="com.jsparser.ASTComparator" -Dexec.args="$FILE" -q 2>&1 | \
    awk '/=== Acorn AST ===/,/=== Java AST ===/' | \
    grep -v "=== Acorn AST ===" | \
    grep -v "=== Java AST ===" > /tmp/acorn2.json

mvn exec:java -Dexec.mainClass="com.jsparser.ASTComparator" -Dexec.args="$FILE" -q 2>&1 | \
    awk '/=== Java AST ===/,/✗ ASTs differ/' | \
    grep -v "=== Java AST ===" | \
    grep -v "✗ ASTs differ" | \
    grep -v "✓ ASTs match" > /tmp/java.json

echo "=== First differences ==="
diff -u /tmp/acorn2.json /tmp/java.json | head -50
