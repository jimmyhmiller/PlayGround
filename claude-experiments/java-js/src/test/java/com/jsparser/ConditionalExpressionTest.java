package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class ConditionalExpressionTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "x ? y : z;",
        "true ? 1 : 2;",
        "x > 0 ? x : -x;",
        "a ? b ? c : d : e;",
        "a ? b : c ? d : e;",
        "x && y ? 1 : 0;",
        "var a = x ? y : z;",
        "x ? y++ : y--;",
        "x ? { a: 1 } : { b: 2 };",
        "x ? [1, 2] : [3, 4];",
        "x ? func() : other();",
    })
    void testConditionalExpressionsAgainstOracle(String source) throws Exception {
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
