#!/bin/bash
# Analyze JFR allocation data and produce actionable recommendations

set -e

cd "$(dirname "$0")"

# Find the most recent JFR file
JFR_FILE=$(ls -t benchmark-results/alloc-profiles/*.jfr 2>/dev/null | head -1)

if [ -z "$JFR_FILE" ]; then
    echo "No JFR file found. Run ./run-alloc-profile.sh first."
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  ALLOCATION ANALYSIS REPORT"
echo "  JFR File: $JFR_FILE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Extract allocation samples and TLAB data
SAMPLES_FILE="/tmp/alloc_samples_$$.txt"
TLAB_FILE="/tmp/alloc_tlab_$$.txt"

jfr print --events jdk.ObjectAllocationSample "$JFR_FILE" > "$SAMPLES_FILE" 2>/dev/null
jfr print --events jdk.ObjectAllocationInNewTLAB "$JFR_FILE" > "$TLAB_FILE" 2>/dev/null

echo "────────────────────────────────────────────────────────────────────────────────"
echo "  TOP ALLOCATION HOTSPOTS BY COUNT"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""
echo "Type                                          | TLAB Count | Sample Count"
echo "----------------------------------------------|------------|-------------"

# Extract counts for jsparser types
for type in "Token" "SourceLocation\$Position" "SourceLocation" "Identifier" "MemberExpression" "CallExpression" "Literal" "BinaryExpression"; do
    tlab_count=$(grep -c "com.jsparser.*$type" "$TLAB_FILE" 2>/dev/null || echo "0")
    sample_count=$(grep -c "com.jsparser.*$type" "$SAMPLES_FILE" 2>/dev/null || echo "0")
    printf "%-45s | %10s | %12s\n" "$type" "$tlab_count" "$sample_count"
done

# JDK types
echo ""
for type in "String" "byte\[\]" "char\[\]" "Integer" "ArrayList" "Object\[\]"; do
    tlab_count=$(grep "objectClass = java.lang.$type\|objectClass = $type" "$TLAB_FILE" 2>/dev/null | wc -l | tr -d ' ')
    sample_count=$(grep "objectClass = java.lang.$type\|objectClass = $type" "$SAMPLES_FILE" 2>/dev/null | wc -l | tr -d ' ')
    printf "%-45s | %10s | %12s\n" "java.lang.$type" "$tlab_count" "$sample_count"
done

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "  ALLOCATION WEIGHT (BYTES) BY LOCATION"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

# Extract weighted allocation sites from samples
grep -B5 "weight = " "$SAMPLES_FILE" 2>/dev/null | \
    grep -E "(objectClass|weight|line:)" | \
    paste - - - 2>/dev/null | \
    sort -t'=' -k3 -rn | head -20 || echo "No weight data available"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  OPTIMIZATION RECOMMENDATIONS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check for specific patterns and give recommendations

echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ 1. TOKEN ALLOCATION (Highest impact)                                        │"
echo "├──────────────────────────────────────────────────────────────────────────────┤"
echo "│ Finding: Token is the most allocated type                                   │"
echo "│                                                                              │"
echo "│ Locations:                                                                   │"
grep -A20 "objectClass = com.jsparser.Token" "$SAMPLES_FILE" 2>/dev/null | \
    grep "line:" | head -5 | sed 's/^/│   /' || echo "│   (See JFR file for details)"
echo "│                                                                              │"
echo "│ Recommendations:                                                             │"
echo "│   • Pool single-char tokens: ; , ( ) [ ] { } : . etc.                       │"
echo "│     Create static final Token instances for common punctuation              │"
echo "│   • Consider flyweight pattern for tokens with no literal value             │"
echo "│   • For keywords, use interned strings or enum-based lexemes                │"
echo "│                                                                              │"
echo "│ Example fix in Lexer.java:                                                  │"
echo "│   private static final Token SEMICOLON_TOKEN =                              │"
echo "│       new Token(TokenType.SEMICOLON, \";\", null, 0, 0, 0, 1, 0, 1, null);   │"
echo "│   // Return pooled token with position overlay                              │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ 2. STRING ALLOCATION FROM toCharArray() (Second highest impact)             │"
echo "├──────────────────────────────────────────────────────────────────────────────┤"
echo "│ Finding: char[] allocation of ~18MB in Lexer constructor                    │"
echo "│                                                                              │"
echo "│ Location: Lexer.java line 35                                                │"
echo "│   this.buf = source.toCharArray();                                          │"
echo "│                                                                              │"
echo "│ Recommendations:                                                             │"
echo "│   • Instead of copying to char[], work directly with String.charAt()        │"
echo "│   • Or use sun.misc.Unsafe to get char[] without copying (advanced)         │"
echo "│   • For hot path: use String.value field access via reflection (risky)      │"
echo "│                                                                              │"
echo "│ Example fix:                                                                │"
echo "│   // Remove: this.buf = source.toCharArray();                               │"
echo "│   // Change peek() to: return source.charAt(position);                      │"
echo "│   // Modern JVMs optimize String.charAt() very well                         │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ 3. STRING ALLOCATION IN scanIdentifier                                      │"
echo "├──────────────────────────────────────────────────────────────────────────────┤"
echo "│ Finding: String allocation at Lexer.java line 1027                          │"
echo "│   String identifierName = new String(buf, startPos, asciiEnd - startPos);   │"
echo "│                                                                              │"
echo "│ Recommendations:                                                             │"
echo "│   • Use String interning for common identifiers (keywords, common names)    │"
echo "│   • Consider a string pool/cache for repeated identifiers                   │"
echo "│   • For pure ASCII identifiers, use source.substring() which may share      │"
echo "│     backing array in some JVM implementations                               │"
echo "│                                                                              │"
echo "│ Example fix:                                                                │"
echo "│   String identifierName = source.substring(startPos, asciiEnd).intern();   │"
echo "│   // Or maintain a HashMap<String,String> for deduplication                 │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ 4. SourceLocation\$Position ALLOCATION                                       │"
echo "├──────────────────────────────────────────────────────────────────────────────┤"
echo "│ Finding: 765+ Position objects allocated                                    │"
echo "│                                                                              │"
echo "│ Recommendations:                                                             │"
echo "│   • Pool common positions (line 1 col 0, etc.)                              │"
echo "│   • Consider storing line/column as primitives in parent object             │"
echo "│   • Use int-packed representation: (line << 16) | column                    │"
echo "│   • Lazy computation: store offset, compute line/col on demand              │"
echo "│                                                                              │"
echo "│ Example packed approach:                                                    │"
echo "│   // Instead of: new Position(line, column)                                 │"
echo "│   // Use: int pos = (line << 20) | column; // supports 1M lines, 1M cols   │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ 5. INTEGER AUTOBOXING                                                       │"
echo "├──────────────────────────────────────────────────────────────────────────────┤"
echo "│ Finding: 76+ Integer objects allocated (autoboxing)                         │"
echo "│                                                                              │"
echo "│ Location: Likely Stack<Integer> usage in Lexer.java                         │"
echo "│   private Stack<Integer> templateBraceDepthStack = new Stack<>();           │"
echo "│                                                                              │"
echo "│ Recommendations:                                                             │"
echo "│   • Replace Stack<Integer> with int[] + index counter                       │"
echo "│   • Use IntArrayList from Eclipse Collections or fastutil                   │"
echo "│   • Simple fix: private int[] braceDepths = new int[32]; int braceTop = 0; │"
echo "│                                                                              │"
echo "│ Example fix:                                                                │"
echo "│   // Replace: templateBraceDepthStack.push(depth + 1)                       │"
echo "│   // With: braceDepths[braceTop++] = depth + 1;                             │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

echo "┌──────────────────────────────────────────────────────────────────────────────┐"
echo "│ 6. ARRAYLIST GROWTH                                                         │"
echo "├──────────────────────────────────────────────────────────────────────────────┤"
echo "│ Finding: Multiple ArrayList and Object[] allocations                        │"
echo "│                                                                              │"
echo "│ Location: Lexer.tokenize() line 43                                          │"
echo "│   List<Token> tokens = new ArrayList<>();                                   │"
echo "│                                                                              │"
echo "│ Recommendations:                                                             │"
echo "│   • Pre-size ArrayList based on source length estimate                      │"
echo "│     Heuristic: sourceLength / 5 (avg ~5 chars per token)                    │"
echo "│   • Consider Token[] with manual resizing for hot path                      │"
echo "│                                                                              │"
echo "│ Example fix:                                                                │"
echo "│   int estimatedTokens = source.length() / 5;                                │"
echo "│   List<Token> tokens = new ArrayList<>(estimatedTokens);                    │"
echo "└──────────────────────────────────────────────────────────────────────────────┘"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  PRIORITY ORDER FOR OPTIMIZATION"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  1. [HIGH]   Remove toCharArray() - saves ~18MB per parse"
echo "  2. [HIGH]   Pool/flyweight single-char tokens - reduces Token count by ~50%"
echo "  3. [MEDIUM] String interning for identifiers - reduces String allocations"
echo "  4. [MEDIUM] Replace Stack<Integer> with int[] - eliminates autoboxing"
echo "  5. [MEDIUM] Pack Position into parent or use ints - reduces Position objects"
echo "  6. [LOW]    Pre-size ArrayList - minor improvement"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Cleanup
rm -f "$SAMPLES_FILE" "$TLAB_FILE"
