package com.jsparser.benchmarks;

import com.jsparser.Lexer;
import com.jsparser.SIMDIdentifierScanner;
import com.jsparser.Token;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * JMH microbenchmark for Lexer.scanIdentifier performance.
 *
 * Run:
 *   mvn clean package -DskipTests
 *   java --enable-preview --add-modules jdk.incubator.vector -jar target/benchmarks.jar ScanIdentifierBenchmark
 *
 * Quick run:
 *   java --enable-preview --add-modules jdk.incubator.vector -jar target/benchmarks.jar ScanIdentifierBenchmark -f 1 -wi 3 -i 5
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(value = 1, jvmArgs = {"--enable-preview", "--add-modules", "jdk.incubator.vector"})
public class ScanIdentifierBenchmark {

    // === Test Data: Short Identifiers (most common case) ===
    // Single-letter and short variable names like i, j, x, fn, id
    private static final String SHORT_IDENTIFIERS;
    static {
        StringBuilder sb = new StringBuilder();
        String[] shortNames = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                               "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                               "id", "fn", "cb", "el", "ev", "xs", "ys"};
        for (int i = 0; i < 1000; i++) {
            sb.append(shortNames[i % shortNames.length]);
            sb.append(" ");
        }
        SHORT_IDENTIFIERS = sb.toString();
    }

    // === Test Data: Medium Identifiers (typical variable names) ===
    // Common JavaScript naming patterns: camelCase, common prefixes
    private static final String MEDIUM_IDENTIFIERS;
    static {
        StringBuilder sb = new StringBuilder();
        String[] mediumNames = {"value", "count", "index", "result", "data", "item", "event",
                                "element", "callback", "response", "request", "options",
                                "config", "handler", "context", "target", "source", "params",
                                "userId", "userName", "itemId", "dataList", "eventType"};
        for (int i = 0; i < 1000; i++) {
            sb.append(mediumNames[i % mediumNames.length]);
            sb.append(" ");
        }
        MEDIUM_IDENTIFIERS = sb.toString();
    }

    // === Test Data: Long Identifiers (descriptive names) ===
    // Long descriptive names common in enterprise code
    private static final String LONG_IDENTIFIERS;
    static {
        StringBuilder sb = new StringBuilder();
        String[] longNames = {
            "calculateTotalAmountWithDiscount",
            "fetchUserProfileDataFromServer",
            "handleFormSubmissionEvent",
            "processPaymentTransaction",
            "validateEmailAddressFormat",
            "initializeApplicationState",
            "updateComponentRendering",
            "transformResponsePayload",
            "authenticateUserCredentials",
            "synchronizeDataWithBackend"
        };
        for (int i = 0; i < 500; i++) {
            sb.append(longNames[i % longNames.length]);
            sb.append(" ");
        }
        LONG_IDENTIFIERS = sb.toString();
    }

    // === Test Data: Keywords (tests the switch statement) ===
    // Mix of keywords to exercise keyword detection
    private static final String KEYWORDS;
    static {
        StringBuilder sb = new StringBuilder();
        String[] keywords = {"var", "let", "const", "function", "return", "if", "else",
                            "for", "while", "do", "break", "continue", "switch", "case",
                            "default", "try", "catch", "finally", "throw", "new", "typeof",
                            "void", "delete", "this", "true", "false", "null", "in",
                            "instanceof", "class", "import", "export", "with", "debugger"};
        for (int i = 0; i < 1000; i++) {
            sb.append(keywords[i % keywords.length]);
            sb.append(" ");
        }
        KEYWORDS = sb.toString();
    }

    // === Test Data: Mixed (realistic distribution) ===
    // Mix of short, medium, long identifiers and keywords
    private static final String MIXED_REALISTIC;
    static {
        StringBuilder sb = new StringBuilder();
        // Realistic JS code pattern: ~40% short, ~30% medium, ~20% keywords, ~10% long
        String[] mixed = {
            // Short (40%)
            "i", "j", "x", "y", "n", "a", "b", "c", "fn", "cb", "el", "id",
            "i", "j", "x", "y", "n", "a", "b", "c", "fn", "cb", "el", "id",
            "i", "j", "x", "y", "n", "a", "b", "c", "fn", "cb", "el", "id",
            "i", "j", "x", "y",
            // Medium (30%)
            "value", "count", "index", "result", "data", "item", "event",
            "element", "callback", "response", "request", "options", "config",
            "handler", "context", "target", "source", "params", "userId",
            "userName", "itemId", "dataList", "eventType", "startIndex",
            "endIndex", "maxCount", "minValue", "prevState", "nextState", "tmpVar",
            // Keywords (20%)
            "var", "let", "const", "function", "return", "if", "else", "for",
            "while", "this", "true", "false", "null", "new", "typeof", "break",
            "continue", "switch", "case", "try",
            // Long (10%)
            "calculateTotalAmount", "fetchUserProfile", "handleFormSubmit",
            "processPayment", "validateEmail", "initializeState",
            "updateComponent", "transformData", "authenticateUser", "syncData"
        };
        for (int i = 0; i < 1000; i++) {
            sb.append(mixed[i % mixed.length]);
            sb.append(" ");
        }
        MIXED_REALISTIC = sb.toString();
    }

    // === Test Data: Underscore/Dollar prefixed (common patterns) ===
    private static final String UNDERSCORE_DOLLAR;
    static {
        StringBuilder sb = new StringBuilder();
        String[] names = {"_private", "_internal", "_cache", "_data", "_state",
                         "$el", "$scope", "$http", "$timeout", "$q",
                         "__proto__", "__dirname", "__filename",
                         "_id", "_value", "_count", "$emit", "$on", "$watch"};
        for (int i = 0; i < 1000; i++) {
            sb.append(names[i % names.length]);
            sb.append(" ");
        }
        UNDERSCORE_DOLLAR = sb.toString();
    }

    // === Test Data: Unicode escapes (edge case) ===
    // Identifiers with unicode escape sequences
    private static final String UNICODE_ESCAPES;
    static {
        StringBuilder sb = new StringBuilder();
        // Mix of regular identifiers with some unicode escapes
        String[] names = {
            "\\u0061bc", // "abc" with unicode escape
            "\\u0076alue", // "value" with unicode escape
            "normal",
            "\\u0069ndex", // "index" with unicode escape
            "data",
            "\\u0072esult", // "result" with unicode escape
            "item",
            "\\u0063ount" // "count" with unicode escape
        };
        for (int i = 0; i < 200; i++) {
            sb.append(names[i % names.length]);
            sb.append(" ");
        }
        UNICODE_ESCAPES = sb.toString();
    }

    // === Benchmarks ===

    @Benchmark
    public List<Token> shortIdentifiers() {
        Lexer lexer = new Lexer(SHORT_IDENTIFIERS);
        return lexer.tokenize();
    }

    @Benchmark
    public List<Token> mediumIdentifiers() {
        Lexer lexer = new Lexer(MEDIUM_IDENTIFIERS);
        return lexer.tokenize();
    }

    @Benchmark
    public List<Token> longIdentifiers() {
        Lexer lexer = new Lexer(LONG_IDENTIFIERS);
        return lexer.tokenize();
    }

    @Benchmark
    public List<Token> keywordsOnly() {
        Lexer lexer = new Lexer(KEYWORDS);
        return lexer.tokenize();
    }

    @Benchmark
    public List<Token> mixedRealistic() {
        Lexer lexer = new Lexer(MIXED_REALISTIC);
        return lexer.tokenize();
    }

    @Benchmark
    public List<Token> underscoreDollar() {
        Lexer lexer = new Lexer(UNDERSCORE_DOLLAR);
        return lexer.tokenize();
    }

    @Benchmark
    public List<Token> unicodeEscapes() {
        Lexer lexer = new Lexer(UNICODE_ESCAPES);
        return lexer.tokenize();
    }

    // =========================================================================
    // SIMD Scanner Raw Benchmarks - Direct comparison of scanning algorithms
    // =========================================================================

    // Pre-converted char arrays for raw scanning benchmarks
    private static final char[] SHORT_CHARS = SHORT_IDENTIFIERS.toCharArray();
    private static final char[] MEDIUM_CHARS = MEDIUM_IDENTIFIERS.toCharArray();
    private static final char[] LONG_CHARS = LONG_IDENTIFIERS.toCharArray();
    private static final char[] MIXED_CHARS = MIXED_REALISTIC.toCharArray();

    // === Short Identifiers - Raw Scanning ===

    @Benchmark
    public int shortIdentifiers_Scalar() {
        int total = 0;
        int pos = 0;
        while (pos < SHORT_CHARS.length) {
            // Skip whitespace
            while (pos < SHORT_CHARS.length && SHORT_CHARS[pos] == ' ') pos++;
            if (pos >= SHORT_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierScalar(SHORT_CHARS, pos, SHORT_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    @Benchmark
    public int shortIdentifiers_ShortVector() {
        int total = 0;
        int pos = 0;
        while (pos < SHORT_CHARS.length) {
            // Skip whitespace
            while (pos < SHORT_CHARS.length && SHORT_CHARS[pos] == ' ') pos++;
            if (pos >= SHORT_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierShortVector(SHORT_CHARS, pos, SHORT_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    @Benchmark
    public int shortIdentifiers_ByteVector() {
        int total = 0;
        int pos = 0;
        while (pos < SHORT_CHARS.length) {
            // Skip whitespace
            while (pos < SHORT_CHARS.length && SHORT_CHARS[pos] == ' ') pos++;
            if (pos >= SHORT_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierByteVector(SHORT_CHARS, pos, SHORT_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    // === Medium Identifiers - Raw Scanning ===

    @Benchmark
    public int mediumIdentifiers_Scalar() {
        int total = 0;
        int pos = 0;
        while (pos < MEDIUM_CHARS.length) {
            while (pos < MEDIUM_CHARS.length && MEDIUM_CHARS[pos] == ' ') pos++;
            if (pos >= MEDIUM_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierScalar(MEDIUM_CHARS, pos, MEDIUM_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    @Benchmark
    public int mediumIdentifiers_ShortVector() {
        int total = 0;
        int pos = 0;
        while (pos < MEDIUM_CHARS.length) {
            while (pos < MEDIUM_CHARS.length && MEDIUM_CHARS[pos] == ' ') pos++;
            if (pos >= MEDIUM_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierShortVector(MEDIUM_CHARS, pos, MEDIUM_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    @Benchmark
    public int mediumIdentifiers_ByteVector() {
        int total = 0;
        int pos = 0;
        while (pos < MEDIUM_CHARS.length) {
            while (pos < MEDIUM_CHARS.length && MEDIUM_CHARS[pos] == ' ') pos++;
            if (pos >= MEDIUM_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierByteVector(MEDIUM_CHARS, pos, MEDIUM_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    // === Long Identifiers - Raw Scanning ===

    @Benchmark
    public int longIdentifiers_Scalar() {
        int total = 0;
        int pos = 0;
        while (pos < LONG_CHARS.length) {
            while (pos < LONG_CHARS.length && LONG_CHARS[pos] == ' ') pos++;
            if (pos >= LONG_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierScalar(LONG_CHARS, pos, LONG_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    @Benchmark
    public int longIdentifiers_ShortVector() {
        int total = 0;
        int pos = 0;
        while (pos < LONG_CHARS.length) {
            while (pos < LONG_CHARS.length && LONG_CHARS[pos] == ' ') pos++;
            if (pos >= LONG_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierShortVector(LONG_CHARS, pos, LONG_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    @Benchmark
    public int longIdentifiers_ByteVector() {
        int total = 0;
        int pos = 0;
        while (pos < LONG_CHARS.length) {
            while (pos < LONG_CHARS.length && LONG_CHARS[pos] == ' ') pos++;
            if (pos >= LONG_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierByteVector(LONG_CHARS, pos, LONG_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    // === Mixed Realistic - Raw Scanning ===

    @Benchmark
    public int mixedRealistic_Scalar() {
        int total = 0;
        int pos = 0;
        while (pos < MIXED_CHARS.length) {
            while (pos < MIXED_CHARS.length && MIXED_CHARS[pos] == ' ') pos++;
            if (pos >= MIXED_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierScalar(MIXED_CHARS, pos, MIXED_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    @Benchmark
    public int mixedRealistic_ShortVector() {
        int total = 0;
        int pos = 0;
        while (pos < MIXED_CHARS.length) {
            while (pos < MIXED_CHARS.length && MIXED_CHARS[pos] == ' ') pos++;
            if (pos >= MIXED_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierShortVector(MIXED_CHARS, pos, MIXED_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    @Benchmark
    public int mixedRealistic_ByteVector() {
        int total = 0;
        int pos = 0;
        while (pos < MIXED_CHARS.length) {
            while (pos < MIXED_CHARS.length && MIXED_CHARS[pos] == ' ') pos++;
            if (pos >= MIXED_CHARS.length) break;

            int end = SIMDIdentifierScanner.scanIdentifierByteVector(MIXED_CHARS, pos, MIXED_CHARS.length);
            total += (end - pos);
            pos = end;
        }
        return total;
    }

    // Main method
    public static void main(String[] args) throws Exception {
        // Print sizes and vector species info
        System.out.println("SIMD Identifier Scanner Benchmark");
        System.out.println("==================================");
        System.out.println();
        System.out.println("Vector Species:");
        System.out.println("  ShortVector: " + SIMDIdentifierScanner.getShortSpecies());
        System.out.println("  ByteVector:  " + SIMDIdentifierScanner.getByteSpecies());
        System.out.println();
        System.out.println("Test data sizes:");
        System.out.println("  SHORT_IDENTIFIERS: " + SHORT_IDENTIFIERS.length() + " chars");
        System.out.println("  MEDIUM_IDENTIFIERS: " + MEDIUM_IDENTIFIERS.length() + " chars");
        System.out.println("  LONG_IDENTIFIERS: " + LONG_IDENTIFIERS.length() + " chars");
        System.out.println("  KEYWORDS: " + KEYWORDS.length() + " chars");
        System.out.println("  MIXED_REALISTIC: " + MIXED_REALISTIC.length() + " chars");
        System.out.println("  UNDERSCORE_DOLLAR: " + UNDERSCORE_DOLLAR.length() + " chars");
        System.out.println("  UNICODE_ESCAPES: " + UNICODE_ESCAPES.length() + " chars");
        System.out.println();

        org.openjdk.jmh.Main.main(args);
    }
}
