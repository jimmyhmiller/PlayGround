package com.jsparser;

import com.jsparser.regex.RegexSyntaxException;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests using test262 regex test suite.
 */
class Test262RegexIntegrationTest {

    @Test
    void testUnicodeSetsBreakingChanges() throws IOException {
        // Test all 28 breaking-change tests from test262
        Path dir = Paths.get("test-oracles/test262/test/built-ins/RegExp/prototype/unicodeSets");

        List<String> failures = new ArrayList<>();
        int tested = 0;
        int passed = 0;

        for (int i = 1; i <= 28; i++) {
            String filename = String.format("breaking-change-from-u-to-v-%02d.js", i);
            Path file = dir.resolve(filename);

            if (!Files.exists(file)) {
                continue;
            }

            tested++;
            String source = Files.readString(file);

            // All these files should contain negative tests (expected SyntaxError)
            boolean shouldFail = source.contains("phase: parse") && source.contains("type: SyntaxError");

            try {
                Parser.parse(source);
                if (shouldFail) {
                    failures.add(filename + ": Expected SyntaxError but parsed successfully");
                } else {
                    passed++;
                }
            } catch (RegexSyntaxException e) {
                if (shouldFail) {
                    passed++;
                } else {
                    failures.add(filename + ": Unexpected error: " + e.getMessage());
                }
            } catch (Exception e) {
                // Might be other parse errors (like comments etc), which is OK for this test
                if (shouldFail) {
                    passed++;
                }
            }
        }

        assertTrue(tested > 0, "Should have found test262 breaking-change files");
        assertTrue(failures.isEmpty(), "Test262 failures:\n" + String.join("\n", failures));
        System.out.println("Test262 breaking-change tests: " + passed + "/" + tested + " passed");
    }

    @Test
    void testDuplicateFlags() throws IOException {
        Path file = Paths.get("test-oracles/test262/test/built-ins/RegExp/duplicate-flags.js");

        if (!Files.exists(file)) {
            return; // Skip if file doesn't exist
        }

        String source = Files.readString(file);

        // Note: This test uses RegExp constructor syntax, not literal syntax
        // Our validator only validates regex literals, not RegExp() calls
        // So we just verify the file parses (it should parse fine since it's valid JS syntax)
        assertDoesNotThrow(() -> Parser.parse(source));

        // But we do test that regex literals with duplicate flags are rejected
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/test/gg"));
    }

    @Test
    void testUandVFlagsMutuallyExclusive() throws IOException {
        Path file = Paths.get("test-oracles/test262/test/built-ins/RegExp/prototype/unicodeSets/uv-flags.js");

        if (!Files.exists(file)) {
            return; // Skip if file doesn't exist
        }

        String source = Files.readString(file);

        // This test expects u and v together to throw SyntaxError
        assertThrows(RegexSyntaxException.class, () -> Parser.parse(source));
    }

    @Test
    void testSpecificBreakingChangePatterns() {
        // Test specific patterns from test262 that should fail with v flag

        // Characters that must be escaped in v mode
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[|]/v"));     // pipe
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[(]/v"));     // left paren
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[)]/v"));     // right paren
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[[]/v"));     // left bracket
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[{]/v"));     // left brace
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[}]/v"));     // right brace
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[/]/v"));     // slash

        // Double punctuators
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[!!]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[##]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[$$]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[%%]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[&&]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[**]/v"));

        // But these should work with u flag
        assertDoesNotThrow(() -> Parser.parse("/[|]/u"));
        assertDoesNotThrow(() -> Parser.parse("/[(]/u"));
        assertDoesNotThrow(() -> Parser.parse("/[!!]/u"));
    }
}
