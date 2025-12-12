#!/bin/bash
# Cold-Start Benchmark: Time to parse TypeScript compiler ONCE (including process startup)
#
# This measures real-world "one-shot" performance - what you'd experience
# running a parser from the command line on a single file.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TS_FILE="$PROJECT_DIR/benchmarks/real-world-libs/typescript.js"
TS_SIZE=$(wc -c < "$TS_FILE" | tr -d ' ')
TS_SIZE_MB=$(echo "scale=1; $TS_SIZE / 1048576" | bc)

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Cold-Start Benchmark: TypeScript Compiler ($TS_SIZE_MB MB)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Measures: Process startup + parse (single run, no warmup)"
echo ""

# Create temporary one-shot parser scripts
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Java one-shot parser
cat > "$TEMP_DIR/JavaParse.java" << 'JAVA'
import com.jsparser.Parser;
import java.nio.file.Files;
import java.nio.file.Paths;
public class JavaParse {
    public static void main(String[] args) throws Exception {
        String code = Files.readString(Paths.get(args[0]));
        Parser parser = new Parser(code, false);
        parser.parse();
    }
}
JAVA

# Node.js one-shot parsers
cat > "$TEMP_DIR/parse-acorn.mjs" << 'JS'
import * as acorn from 'acorn';
import { readFileSync } from 'fs';
const code = readFileSync(process.argv[2], 'utf-8');
acorn.parse(code, { ecmaVersion: 'latest' });
JS

cat > "$TEMP_DIR/parse-babel.mjs" << 'JS'
import { parse } from '@babel/parser';
import { readFileSync } from 'fs';
const code = readFileSync(process.argv[2], 'utf-8');
parse(code);
JS

cat > "$TEMP_DIR/parse-meriyah.mjs" << 'JS'
import { parseScript } from 'meriyah';
import { readFileSync } from 'fs';
const code = readFileSync(process.argv[2], 'utf-8');
parseScript(code, { next: true });
JS

# Results array
declare -a RESULTS

# Time a command and return milliseconds
time_cmd() {
    local start=$(python3 -c 'import time; print(int(time.time() * 1000))')
    eval "$1" > /dev/null 2>&1
    local end=$(python3 -c 'import time; print(int(time.time() * 1000))')
    echo $((end - start))
}

echo "Running benchmarks..."
echo ""

# Java
echo -n "  Java (Our Parser)...      "
cd "$PROJECT_DIR"
mvn compile -q -DskipTests 2>/dev/null
JAVA_TIME=$(time_cmd "mvn exec:java -q -Dexec.mainClass='com.jsparser.benchmarks.OneShotParse' -Dexec.args='$TS_FILE' 2>/dev/null || java -cp target/classes com.jsparser.benchmarks.OneShotParse '$TS_FILE' 2>/dev/null" || echo "0")
# Fallback: create and run inline
if [ "$JAVA_TIME" = "0" ] || [ -z "$JAVA_TIME" ]; then
    JAVA_TIME=$(time_cmd "java -cp target/classes:. -Dfile.encoding=UTF-8 --enable-preview -XX:+UseParallelGC -e \"
        import com.jsparser.Parser;
        import java.nio.file.Files;
        import java.nio.file.Paths;
        String code = Files.readString(Paths.get(\\\"$TS_FILE\\\"));
        new Parser(code, false).parse();
    \" 2>/dev/null" || echo "N/A")
fi
if [ "$JAVA_TIME" = "0" ] || [ -z "$JAVA_TIME" ]; then
    # Direct execution
    JAVA_TIME=$( { time java -cp target/classes com.jsparser.Parser "$TS_FILE" 2>/dev/null; } 2>&1 | grep real | awk '{print $2}' | sed 's/m/*60000+/;s/s/*1000/' | bc 2>/dev/null || echo "N/A" )
fi
# Simple approach - just time mvn exec
START=$(python3 -c 'import time; print(int(time.time() * 1000))')
mvn exec:java -q -Dexec.mainClass="com.jsparser.benchmarks.SimpleBenchmark" -Dexec.args="0 1" 2>/dev/null | grep -q "TypeScript" || true
END=$(python3 -c 'import time; print(int(time.time() * 1000))')
JAVA_TIME=$((END - START))
echo "${JAVA_TIME} ms"
RESULTS+=("Java (Our Parser)|$JAVA_TIME")

# Rust - OXC
echo -n "  Rust (OXC)...             "
cd "$PROJECT_DIR/benchmarks/rust"
cargo build --release -q 2>/dev/null
START=$(python3 -c 'import time; print(int(time.time() * 1000))')
./target/release/benchmark-real-world 0 1 2>/dev/null | grep -q "TypeScript" || true
END=$(python3 -c 'import time; print(int(time.time() * 1000))')
RUST_TIME=$((END - START))
echo "${RUST_TIME} ms"
RESULTS+=("Rust (OXC+SWC)|$RUST_TIME")

# Go - esbuild
echo -n "  Go (esbuild)...           "
cd "$PROJECT_DIR/benchmarks/go"
if [ -f "./benchmark-go" ]; then
    START=$(python3 -c 'import time; print(int(time.time() * 1000))')
    ./benchmark-go 0 1 2>/dev/null | grep -q "TypeScript" || true
    END=$(python3 -c 'import time; print(int(time.time() * 1000))')
    GO_TIME=$((END - START))
    echo "${GO_TIME} ms"
    RESULTS+=("Go (esbuild)|$GO_TIME")
else
    echo "N/A (not built)"
    RESULTS+=("Go (esbuild)|999999")
fi

# Node.js - Acorn
echo -n "  Node.js (Acorn)...        "
cd "$PROJECT_DIR/benchmarks/javascript"
START=$(python3 -c 'import time; print(int(time.time() * 1000))')
node -e "
const acorn = require('acorn');
const fs = require('fs');
const code = fs.readFileSync('$TS_FILE', 'utf-8');
acorn.parse(code, { ecmaVersion: 'latest' });
" 2>/dev/null
END=$(python3 -c 'import time; print(int(time.time() * 1000))')
ACORN_TIME=$((END - START))
echo "${ACORN_TIME} ms"
RESULTS+=("Node.js (Acorn)|$ACORN_TIME")

# Node.js - Babel
echo -n "  Node.js (Babel)...        "
START=$(python3 -c 'import time; print(int(time.time() * 1000))')
node -e "
const { parse } = require('@babel/parser');
const fs = require('fs');
const code = fs.readFileSync('$TS_FILE', 'utf-8');
parse(code);
" 2>/dev/null
END=$(python3 -c 'import time; print(int(time.time() * 1000))')
BABEL_TIME=$((END - START))
echo "${BABEL_TIME} ms"
RESULTS+=("Node.js (Babel)|$BABEL_TIME")

# Node.js - Meriyah
echo -n "  Node.js (Meriyah)...      "
START=$(python3 -c 'import time; print(int(time.time() * 1000))')
node -e "
const { parseScript } = require('meriyah');
const fs = require('fs');
const code = fs.readFileSync('$TS_FILE', 'utf-8');
parseScript(code, { next: true });
" 2>/dev/null
END=$(python3 -c 'import time; print(int(time.time() * 1000))')
MERIYAH_TIME=$((END - START))
echo "${MERIYAH_TIME} ms"
RESULTS+=("Node.js (Meriyah)|$MERIYAH_TIME")

# Sort and display results
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Results (sorted by total time, including process startup)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
printf "  %-25s | %12s | %12s\n" "Parser" "Time (ms)" "vs Fastest"
echo "  ---------------------------------------------------------------"

# Sort results
IFS=$'\n' SORTED=($(for r in "${RESULTS[@]}"; do echo "$r"; done | sort -t'|' -k2 -n))

FASTEST=$(echo "${SORTED[0]}" | cut -d'|' -f2)

RANK=1
for result in "${SORTED[@]}"; do
    NAME=$(echo "$result" | cut -d'|' -f1)
    TIME=$(echo "$result" | cut -d'|' -f2)
    if [ "$TIME" = "999999" ]; then
        continue
    fi
    RATIO=$(echo "scale=2; $TIME / $FASTEST" | bc)
    MEDAL=""
    case $RANK in
        1) MEDAL="🥇" ;;
        2) MEDAL="🥈" ;;
        3) MEDAL="🥉" ;;
    esac
    printf "  %s %-23s | %12s | %11sx\n" "$MEDAL" "$NAME" "$TIME" "$RATIO"
    RANK=$((RANK + 1))
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
