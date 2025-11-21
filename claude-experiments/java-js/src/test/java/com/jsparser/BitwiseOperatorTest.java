package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class BitwiseOperatorTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "var x = a & b;",
        "var x = a | b;",
        "var x = a ^ b;",
        "var x = a << b;",
        "var x = a >> b;",
        "var x = a >>> b;",
        "var x = 5 & 3;",
        "var x = 5 | 3;",
        "var x = 5 ^ 3;",
        "var x = 8 << 2;",
        "var x = 8 >> 2;",
        "var x = -8 >>> 2;",
        "var x = a & b | c;",
        "var x = a | b & c;",
        "var x = a ^ b & c;",
    })
    void testBitwiseOperatorsAgainstOracle(String source) throws Exception {
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
