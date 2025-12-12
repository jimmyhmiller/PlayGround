package com.jsparser;

import com.jsparser.regex.RegexValidator;
import com.jsparser.regex.RegexSyntaxException;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class RegexRangeTest {
    @Test
    void testPropertyEscapeAtEndOfRange() {
        // [--\p{Hex}] should be invalid: range from '-' to '\p{Hex}' where \p{Hex} is a character class
        assertThrows(RegexSyntaxException.class, () -> {
            RegexValidator.validate("[--\\p{Hex}]", "u", 0, 1, 0);
        }, "Character class range ending with property escape should throw");
    }
}
