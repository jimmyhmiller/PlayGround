package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class ForStatementTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "for (;;) x;",
        "for (var i = 0; i < 10; i++) x;",
        "for (let i = 0; i < 10; i++) x;",
        "for (const i = 0; i < 10; i++) x;",
        "for (i = 0; i < 10; i++) x;",
        "for (var i = 0, j = 0; i < 10; i++) x;",
        "for (var i = 0; i < 10; ) x;",
        "for (var i = 0; ; i++) x;",
        "for (; i < 10; i++) x;",
        "for (var i = 0; i < 10; i++) { }",
        "for (var i = 0; i < 10; i++) { x++; }",
        "for (x = 0; x < 10; x++) y;",
        "for (var i = 0; i < 10; i++) for (var j = 0; j < 10; j++) x;",
    })
    void testForStatementsAgainstOracle(String source) throws Exception {
        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source);

        String expectedJson = mapper.writeValueAsString(expected);
        String actualJson = mapper.writeValueAsString(actual);

        System.out.println("Testing: " + source);
        if (!expectedJson.equals(actualJson)) {
            System.out.println("EXPECTED:");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expected));
            System.out.println("\nACTUAL:");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actual));
        }

        assertEquals(expectedJson, actualJson, "AST mismatch for: " + source);
    }
}
