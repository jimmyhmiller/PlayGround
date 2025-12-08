#!/bin/bash
# Allocation profiling script using JFR (Java Flight Recorder)
# Records allocations and analyzes where memory is being allocated

set -e

cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════════════════"
echo "  Allocation Profiling with JFR"
echo "════════════════════════════════════════════════════════════════"
echo ""

RESULTS_DIR="benchmark-results/alloc-profiles"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JFR_FILE="$RESULTS_DIR/alloc_${TIMESTAMP}.jfr"
REPORT_FILE="$RESULTS_DIR/alloc_report_${TIMESTAMP}.txt"

# Build first
echo "📦 Building project..."
mvn compile -q -DskipTests 2>/dev/null || mvn compile -DskipTests

echo ""
echo "🔬 Running allocation profiling..."
echo "   This will parse TypeScript.js multiple times to gather data..."
echo ""

# Create a simple Java program to run parsing multiple times with JFR
cat > /tmp/AllocProfile.java << 'JAVA_EOF'
import com.jsparser.Parser;
import com.jsparser.ast.Program;
import java.nio.file.Files;
import java.nio.file.Path;

public class AllocProfile {
    public static void main(String[] args) throws Exception {
        // Read the TypeScript source
        String source = Files.readString(Path.of("benchmarks/real-world-libs/typescript.js"));
        System.out.println("Source size: " + source.length() + " bytes");

        // Warm up
        System.out.println("Warming up (5 iterations)...");
        for (int i = 0; i < 5; i++) {
            Parser parser = new Parser(source);
            Program program = parser.parse();
            System.out.print(".");
        }
        System.out.println(" done");

        // Force GC before profiling
        System.gc();
        Thread.sleep(100);

        // Profile runs
        System.out.println("Profiling (20 iterations)...");
        long start = System.nanoTime();
        for (int i = 0; i < 20; i++) {
            Parser parser = new Parser(source);
            Program program = parser.parse();
            System.out.print(".");
        }
        long elapsed = System.nanoTime() - start;
        System.out.println(" done");
        System.out.printf("Average parse time: %.2f ms%n", (elapsed / 20.0) / 1_000_000.0);
    }
}
JAVA_EOF

# Compile the profiler
echo "Compiling profiler..."
javac --enable-preview --source 25 \
    -cp target/classes:/Users/jimmyhmiller/.m2/repository/com/fasterxml/jackson/core/jackson-databind/2.16.1/jackson-databind-2.16.1.jar:/Users/jimmyhmiller/.m2/repository/com/fasterxml/jackson/core/jackson-core/2.16.1/jackson-core-2.16.1.jar:/Users/jimmyhmiller/.m2/repository/com/fasterxml/jackson/core/jackson-annotations/2.16.1/jackson-annotations-2.16.1.jar \
    -d /tmp \
    /tmp/AllocProfile.java

echo ""
echo "Running with JFR allocation profiling..."
echo ""

# Create custom JFR settings for detailed allocation tracking
JFR_SETTINGS="/tmp/alloc-profile.jfc"
cat > "$JFR_SETTINGS" << 'JFC_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration version="2.0" label="Allocation Profile" description="Low overhead config for allocation profiling">
  <event name="jdk.ObjectAllocationInNewTLAB">
    <setting name="enabled">true</setting>
    <setting name="stackTrace">true</setting>
  </event>
  <event name="jdk.ObjectAllocationOutsideTLAB">
    <setting name="enabled">true</setting>
    <setting name="stackTrace">true</setting>
  </event>
  <event name="jdk.ObjectAllocationSample">
    <setting name="enabled">true</setting>
    <setting name="throttle">150/s</setting>
    <setting name="stackTrace">true</setting>
  </event>
  <event name="jdk.OldObjectSample">
    <setting name="enabled">true</setting>
    <setting name="stackTrace">true</setting>
    <setting name="cutoff">0 ns</setting>
  </event>
  <event name="jdk.ExecutionSample">
    <setting name="enabled">true</setting>
    <setting name="period">10 ms</setting>
  </event>
  <event name="jdk.NativeMethodSample">
    <setting name="enabled">true</setting>
    <setting name="period">10 ms</setting>
  </event>
</configuration>
JFC_EOF

# Run with JFR - enable allocation profiling with custom settings
java --enable-preview \
    -XX:+UnlockDiagnosticVMOptions \
    -XX:+DebugNonSafepoints \
    -XX:StartFlightRecording=filename="$JFR_FILE",settings="$JFR_SETTINGS",dumponexit=true \
    -XX:FlightRecorderOptions=stackdepth=128 \
    -cp /tmp:target/classes:/Users/jimmyhmiller/.m2/repository/com/fasterxml/jackson/core/jackson-databind/2.16.1/jackson-databind-2.16.1.jar:/Users/jimmyhmiller/.m2/repository/com/fasterxml/jackson/core/jackson-core/2.16.1/jackson-core-2.16.1.jar:/Users/jimmyhmiller/.m2/repository/com/fasterxml/jackson/core/jackson-annotations/2.16.1/jackson-annotations-2.16.1.jar \
    AllocProfile

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Analyzing Allocations"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Generate allocation report using jfr command
echo "JFR file: $JFR_FILE"
echo ""

# Create comprehensive report
{
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  ALLOCATION PROFILE REPORT - $(date)"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo ""

    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  TOP ALLOCATION SITES (by total bytes allocated)"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""
    jfr print --events jdk.ObjectAllocationInNewTLAB,jdk.ObjectAllocationOutsideTLAB "$JFR_FILE" 2>/dev/null | \
        grep -E "(objectClass|allocationSize|stackTrace)" | head -200 || echo "No TLAB allocation events found"

    echo ""
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "  ALLOCATION SUMMARY BY CLASS"
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo ""

    # Extract and summarize allocations by class
    jfr summary "$JFR_FILE" 2>/dev/null || echo "Summary not available"

} > "$REPORT_FILE" 2>&1

echo "Basic report saved to: $REPORT_FILE"
echo ""

# Now do detailed analysis with jfr print
echo "────────────────────────────────────────────────────────────────────────────────"
echo "  DETAILED ALLOCATION ANALYSIS"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

# Parse the JFR and extract allocation data
echo "Extracting allocation events..."
jfr print --events jdk.ObjectAllocationSample "$JFR_FILE" 2>/dev/null > /tmp/alloc_samples.txt || true
jfr print --events jdk.ObjectAllocationInNewTLAB "$JFR_FILE" 2>/dev/null > /tmp/alloc_tlab.txt || true

# Count allocations by type
echo ""
echo "Top allocated types (from allocation samples):"
echo ""
grep "objectClass" /tmp/alloc_samples.txt 2>/dev/null | \
    sed 's/.*objectClass = //' | \
    sort | uniq -c | sort -rn | head -20 || echo "  No allocation samples captured"

echo ""
echo "Top allocated types (from TLAB allocations):"
echo ""
grep "objectClass" /tmp/alloc_tlab.txt 2>/dev/null | \
    sed 's/.*objectClass = //' | \
    sort | uniq -c | sort -rn | head -20 || echo "  No TLAB allocation events"

# Extract stack traces for top allocations
echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "  STACK TRACES FOR KEY ALLOCATIONS"
echo "────────────────────────────────────────────────────────────────────────────────"

# Look for our parser's allocations specifically
echo ""
echo "Parser-related allocations (com.jsparser):"
echo ""
grep -A 30 "com.jsparser" /tmp/alloc_samples.txt 2>/dev/null | head -100 || \
grep -A 30 "com.jsparser" /tmp/alloc_tlab.txt 2>/dev/null | head -100 || \
echo "  No parser-specific allocations found in samples"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Files generated:"
echo "   JFR Recording: $JFR_FILE"
echo "   Text Report:   $REPORT_FILE"
echo ""
echo "💡 For detailed visual analysis, open $JFR_FILE in:"
echo "   - JDK Mission Control (jmc)"
echo "   - IntelliJ IDEA Profiler"
echo "   - VisualVM with JFR plugin"
echo ""
echo "   Or run: jfr print --events jdk.ObjectAllocationSample $JFR_FILE"
echo ""

# Cleanup
rm -f /tmp/AllocProfile.java /tmp/AllocProfile.class /tmp/alloc_samples.txt /tmp/alloc_tlab.txt
