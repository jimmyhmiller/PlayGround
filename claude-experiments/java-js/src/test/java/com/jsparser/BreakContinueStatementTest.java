package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class BreakContinueStatementTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "while (true) break;",
        "while (x) { break; }",
        "for (;;) break;",
        "for (var i = 0; i < 10; i++) { if (x) break; }",
        "do break; while (x);",
        "while (true) continue;",
        "while (x) { continue; }",
        "for (;;) continue;",
        "for (var i = 0; i < 10; i++) { if (x) continue; }",
        "do continue; while (x);",
        "while (true) { if (x) break; else continue; }",
        "for (var i = 0; i < 10; i++) { for (var j = 0; j < 10; j++) { if (x) break; } }",
    })
    void testBreakContinueStatementsAgainstOracle(String source) throws Exception {
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
