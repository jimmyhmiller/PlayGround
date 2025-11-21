package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class FunctionDeclarationTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "function foo() {}",
        "function bar() { }",
        "function add(a, b) { return a + b; }",
        "function greet(name) { var msg = \"Hello\"; return msg; }",
        "function noParams() { return 42; }",
        "function multiParam(x, y, z) { return x + y + z; }",
        "function withIf(x) { if (x > 0) return x; return -x; }",
        "function withLoop(n) { for (var i = 0; i < n; i++) { x++; } }",
        "function withWhile(x) { while (x > 0) { x--; } return x; }",
        "function nested() { var x = 1; if (true) { var y = 2; } }",
        "function breakLoop(arr) { for (var i = 0; i < 10; i++) { if (arr[i]) break; } }",
    })
    void testFunctionDeclarationsAgainstOracle(String source) throws Exception {
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
