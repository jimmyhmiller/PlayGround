package com.jsparser;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

public class BacktickInRegexTest {

    @Test
    @DisplayName("Backtick inside regex literal should not be treated as template literal")
    void backtickInsideRegex() {
        String source = "const re = /^`$/;";

        assertDoesNotThrow(() -> {
            var program = Parser.parse(source, false);
            assertEquals(1, program.body().size());
        });
    }

    @Test
    @DisplayName("Multiple special chars including backtick in regex")
    void complexRegexWithBacktick() {
        String source = "const isValidHeaderName = (str) => /^[-_a-zA-Z0-9^`|~,!#$%&'*+.]+$/.test(str.trim());";

        assertDoesNotThrow(() -> {
            var program = Parser.parse(source, false);
            assertEquals(1, program.body().size());
        });
    }

    @Test
    @DisplayName("Backtick in character class in regex")
    void backtickInCharacterClass() {
        String source = "const re = /[`]/;";

        assertDoesNotThrow(() -> {
            var program = Parser.parse(source, false);
            assertEquals(1, program.body().size());
        });
    }
}
