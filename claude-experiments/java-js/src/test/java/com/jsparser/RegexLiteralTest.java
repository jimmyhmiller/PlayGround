package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class RegexLiteralTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "var re = /abc/;",
        "var re = /test/g;",
        "var re = /pattern/i;",
        "var re = /foo/gi;",
        "var re = /bar/gim;",
        "var re = /a+b*/;",
        "var re = /[0-9]+/;",
        "var re = /[a-z]/i;",
        "var re = /(foo|bar)/;",
        "var re = /\\d+/;",
        "var re = /\\/path/;",
        "var re = /\\w+/g;",
        "var re = /\\s*/;",
        "var x = /test/.test(str);",
        "if (/abc/.test(s)) {}",
    })
    void testRegexLiteralsAgainstOracle(String source) throws Exception {
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
