package com.jsparser;

import com.jsparser.regex.RegexSyntaxException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for regex u-flag (unicode mode) validation.
 */
class RegexUnicodeModeTest {

    @Test
    void testInvalidIdentityEscape() {
        // In unicode mode, only syntax characters can be escaped
        // \M is invalid because M is not a syntax character
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/\\M/u"));
        assertTrue(ex.getMessage().contains("Invalid identity escape"));

        // More invalid identity escapes
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/\\a/u"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/\\#/u"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/\\%/u"));
    }

    @Test
    void testValidIdentityEscape() {
        // Syntax characters can be escaped in unicode mode
        assertDoesNotThrow(() -> Parser.parse("/\\[/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\]/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\(/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\)/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\/\\//u")); // escaped slash
    }

    @Test
    void testInvalidControlEscape() {
        // \c must be followed by a letter in unicode mode
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/\\c1/u"));
        assertTrue(ex.getMessage().contains("Invalid control escape"));

        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/\\c#/u"));
    }

    @Test
    void testValidControlEscape() {
        assertDoesNotThrow(() -> Parser.parse("/\\cA/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\cZ/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\ca/u"));
    }

    @Test
    void testInvalidDecimalEscape() {
        // \0 followed by digit is invalid in unicode mode
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/\\01/u"));
        assertTrue(ex.getMessage().contains("Invalid decimal escape"));

        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/\\07/u"));
    }

    @Test
    void testValidNullEscape() {
        // \0 not followed by digit is valid
        assertDoesNotThrow(() -> Parser.parse("/\\0/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\0a/u"));
    }

    @Test
    void testInvalidHexEscape() {
        // \x must be followed by exactly 2 hex digits in unicode mode
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/\\x1/u"));
        assertTrue(ex.getMessage().contains("Invalid hex escape"));

        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/\\xG/u"));
    }

    @Test
    void testValidHexEscape() {
        assertDoesNotThrow(() -> Parser.parse("/\\x00/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\xFF/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\x1a/u"));
    }

    @Test
    void testInvalidUnicodeEscape() {
        // backslash-u must be followed by 4 hex digits or {hex+} in unicode mode
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/\\u123/u"));
        assertTrue(ex.getMessage().contains("Invalid unicode escape"));
    }

    @Test
    void testValidUnicodeEscape() {
        assertDoesNotThrow(() -> Parser.parse("/\\u0041/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\u{41}/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\u{1F600}/u"));
    }

    @Test
    void testIncompleteQuantifier() {
        // Incomplete quantifiers are invalid in unicode mode
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/a{/u"));
        assertTrue(ex.getMessage().contains("Incomplete quantifier"));

        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/a{1/u"));
        assertThrows(RegexSyntaxException.class, () -> Parser.parse("/a{1,/u"));
    }

    @Test
    void testValidQuantifier() {
        assertDoesNotThrow(() -> Parser.parse("/a{1}/u"));
        assertDoesNotThrow(() -> Parser.parse("/a{1,}/u"));
        assertDoesNotThrow(() -> Parser.parse("/a{1,3}/u"));
    }

    @Test
    void testQuantifierRequiresAtom() {
        // Quantifiers at the start require an atom
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("var x = /+/u"));
        assertTrue(ex.getMessage().contains("Quantifier requires an atom"));

        assertThrows(RegexSyntaxException.class, () -> Parser.parse("var x = /?/u"));
        // Note: can't test /*/u because /* starts a comment in the lexer
    }

    @Test
    void testValidQuantifierAfterAtom() {
        assertDoesNotThrow(() -> Parser.parse("/a*/u"));
        assertDoesNotThrow(() -> Parser.parse("/a+/u"));
        assertDoesNotThrow(() -> Parser.parse("/a?/u"));
        assertDoesNotThrow(() -> Parser.parse("/(abc)*/u"));
    }

    @Test
    void testUnclosedGroup() {
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/(abc/u"));
        assertTrue(ex.getMessage().contains("Unclosed group"));
    }

    @Test
    void testUnclosedCharacterClass() {
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/[abc/u"));
        assertTrue(ex.getMessage().contains("Unclosed character class"));
    }

    @Test
    void testComplexValidPatterns() {
        // Test some complex but valid patterns
        assertDoesNotThrow(() -> Parser.parse("/^[a-z]+$/u"));
        assertDoesNotThrow(() -> Parser.parse("/(foo|bar)+/u"));
        assertDoesNotThrow(() -> Parser.parse("/\\d{2,4}/u"));
        assertDoesNotThrow(() -> Parser.parse("/(?:non)(?=capture)/u"));
        assertDoesNotThrow(() -> Parser.parse("/(?<name>[a-z]+)/u"));
    }
}
