package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class DestructuringTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        // Object destructuring
        "var {x} = obj;",
        "var {x, y} = obj;",
        "var {x, y, z} = obj;",
        "let {a} = foo;",
        "const {b} = bar;",

        // Nested object destructuring
        "var {x: y} = obj;",
        "var {a: b, c: d} = obj;",

        // Array destructuring
        "var [a] = arr;",
        "var [a, b] = arr;",
        "var [a, b, c] = arr;",
        "let [x] = foo;",
        "const [y] = bar;",

        // Mixed
        "var {x} = obj, y = 1;",
        "var [a] = arr, b = 2;",
    })
    void testDestructuringAgainstOracle(String source) throws Exception {
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
