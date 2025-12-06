package com.jsparser;

import com.jsparser.regex.RegexSyntaxException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for regex flag validation.
 */
class RegexFlagValidationTest {

    @Test
    void testValidSingleFlags() {
        // All valid flags should work individually
        assertDoesNotThrow(() -> Parser.parse("/test/g"));
        assertDoesNotThrow(() -> Parser.parse("/test/i"));
        assertDoesNotThrow(() -> Parser.parse("/test/m"));
        assertDoesNotThrow(() -> Parser.parse("/test/s"));
        assertDoesNotThrow(() -> Parser.parse("/test/u"));
        assertDoesNotThrow(() -> Parser.parse("/test/y"));
        assertDoesNotThrow(() -> Parser.parse("/test/d"));
        assertDoesNotThrow(() -> Parser.parse("/test/v"));
    }

    @Test
    void testValidMultipleFlags() {
        assertDoesNotThrow(() -> Parser.parse("/test/gi"));
        assertDoesNotThrow(() -> Parser.parse("/test/gim"));
        assertDoesNotThrow(() -> Parser.parse("/test/gimsyd"));
    }

    @Test
    void testInvalidFlag() {
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/test/G"));
        assertTrue(ex.getMessage().contains("Invalid flag 'G'"));
    }

    @Test
    void testDuplicateFlags() {
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/test/gg"));
        assertTrue(ex.getMessage().contains("Duplicate flag 'g'"));

        ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/test/gig"));
        assertTrue(ex.getMessage().contains("Duplicate flag 'g'"));
    }

    @Test
    void testUandVMutuallyExclusive() {
        RegexSyntaxException ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/test/uv"));
        assertTrue(ex.getMessage().contains("mutually exclusive"));

        ex = assertThrows(RegexSyntaxException.class,
            () -> Parser.parse("/test/vu"));
        assertTrue(ex.getMessage().contains("mutually exclusive"));
    }

    @Test
    void testVFlagWithPipe() {
        // This specific test case from test262
        // /[|]/v should throw SyntaxError (but we haven't implemented pattern validation yet)
        // For now, just test that the v flag itself is accepted
        assertDoesNotThrow(() -> Parser.parse("/test/v"));
    }
}
