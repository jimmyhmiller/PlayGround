package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class IfStatementTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "if (true) x;",
        "if (false) x;",
        "if (x) { }",
        "if (x) { y; }",
        "if (x) y; else z;",
        "if (x) { y; } else { z; }",
        "if (x > 5) x;",
        "if (x === 0) x++; else x--;",
        "if (x && y) z;",
        "if (!x) y;",
        "if (x) if (y) z;",
        "if (x) y; else if (z) w;",
        "if (x) { var a = 1; }",
        "if (typeof x === \"string\") x.length;",
    })
    void testIfStatementsAgainstOracle(String source) throws Exception {
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
