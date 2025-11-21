package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class UnicodeEscapeTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "var \\u0041 = 1;",      // \u0041 is 'A'
        "var \\u0061bc = 2;",    // \u0061 is 'a', so 'abc'
        "var x\\u0079 = 3;",     // \u0079 is 'y', so 'xy'
        "var \\u0024 = 4;",      // \u0024 is '$'
        "var \\u005F = 5;",      // \u005F is '_'
        "var \\u{41} = 6;",      // ES6+ braced format: \\u{41} is 'A'
        "var \\u{61}bc = 7;",    // \\u{61} is 'a', so 'abc'
        "var \\u{24} = 8;",      // \\u{24} is '$'
        "var \\u{5F} = 9;",      // \\u{5F} is '_'
        "var \\u{30} = 0;",      // \\u{30} is '0' (digit)
    })
    void testUnicodeEscapesParse(String source) throws Exception {
        System.out.println("Testing: " + source);
        assertDoesNotThrow(() -> Parser.parse(source), "Should parse without errors: " + source);
    }
}
