package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class FunctionExpressionTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "var f = function() {};",
        "var add = function(a, b) { return a + b; };",
        "var fact = function factorial(n) { return n <= 1 ? 1 : n * factorial(n - 1); };",
        "(function() {})();",
        "var x = function(a, b, c) { var sum = a + b + c; return sum; };",
        "array.map(function(x) { return x * 2; });",
        "setTimeout(function() { x++; }, 1000);",
        "var obj = { method: function(x) { return x; } };",
        "[1, 2].forEach(function(n) { console.log(n); });",
        "var f = function() { if (true) return 1; };",
        "var nested = function() { return function() { return 42; }; };",
    })
    void testFunctionExpressionsAgainstOracle(String source) throws Exception {
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
