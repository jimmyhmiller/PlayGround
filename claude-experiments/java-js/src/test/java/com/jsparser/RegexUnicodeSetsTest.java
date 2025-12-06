package com.jsparser;

import com.jsparser.regex.RegexSyntaxException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for regex v-flag (unicode sets mode) validation.
 * The v flag is more restrictive than u flag in character classes.
 */
class RegexUnicodeSetsTest {

    @Test
    void testPipeInCharacterClass() {
        // This is the specific test case from the user's question!
        // /[|]/v should throw SyntaxError
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/[|]/v"));
        assertTrue(ex.getMessage().contains("must be escaped"));

        // But escaped version is OK
        assertDoesNotThrow(() -> Parser.parse("/[\\|]/v"));
    }

    @Test
    void testParenthesesInCharacterClass() {
        // Unescaped parentheses are invalid in v mode
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[(]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[)]/v"));

        // But escaped versions are OK
        assertDoesNotThrow(() -> Parser.parse("/[\\(]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[\\)]/v"));
    }

    @Test
    void testBracketsInCharacterClass() {
        // Unescaped brackets are invalid in v mode
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[[]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[{]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[}]/v"));

        // But escaped versions are OK
        assertDoesNotThrow(() -> Parser.parse("/[\\[]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[\\{]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[\\}]/v"));
    }

    @Test
    void testSlashInCharacterClass() {
        // Unescaped slash is invalid in v mode
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[/]/v"));

        // But escaped version is OK
        assertDoesNotThrow(() -> Parser.parse("/[\\/]/v"));
    }

    @Test
    void testHyphenInCharacterClass() {
        // Unescaped hyphen is invalid in v mode
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[-]/v"));

        // But escaped version is OK
        assertDoesNotThrow(() -> Parser.parse("/[\\-]/v"));
    }

    @Test
    void testDoublePunctuators() {
        // Double punctuators are invalid in v mode
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[!!]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[##]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[$$]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[%%]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[**]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[++]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[,,]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[..]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[::]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[;;]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[<<]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[==]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[>>]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[??]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[@@]/v"));
    }

    @Test
    void testSinglePunctuatorsAllowed() {
        // Single instances of these punctuators are OK in v mode
        assertDoesNotThrow(() -> Parser.parse("/[!]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[#]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[$]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[%]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[*]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[+]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[,]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[.]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[:]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[;]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[<]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[=]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[>]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[?]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[@]/v"));
    }

    @Test
    void testVFlagInheritsUFlagRestrictions() {
        // All u flag restrictions also apply to v flag
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/\\M/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/\\c1/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/\\01/v"));
    }

    @Test
    void testComplexValidPatterns() {
        // Test some complex but valid v-flag patterns
        assertDoesNotThrow(() -> Parser.parse("/[a-z]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[\\d]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[abc]/v"));
        assertDoesNotThrow(() -> Parser.parse("/[a-z0-9_]/v"));
    }

    @Test
    void testBreakingChangeFromU() {
        // These patterns work with u flag but fail with v flag
        // (This tests the breaking changes mentioned in test262)

        // Works with u flag
        assertDoesNotThrow(() -> Parser.parse("/[|]/u"));
        assertDoesNotThrow(() -> Parser.parse("/[(]/u"));
        assertDoesNotThrow(() -> Parser.parse("/[!!]/u"));

        // Fails with v flag
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[|]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[(]/v"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/[!!]/v"));
    }
}
