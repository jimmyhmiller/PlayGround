package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.ser.FilterProvider;
import com.fasterxml.jackson.databind.ser.impl.SimpleBeanPropertyFilter;
import com.fasterxml.jackson.databind.ser.impl.SimpleFilterProvider;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

class ParserTest {
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final ObjectMapper mapperIgnoreLoc = new ObjectMapper()
            .addMixIn(Object.class, IgnoreLocMixin.class)
            .setFilterProvider(new SimpleFilterProvider()
                    .addFilter("ignoreLoc", SimpleBeanPropertyFilter.serializeAllExcept("loc")));

    @com.fasterxml.jackson.annotation.JsonFilter("ignoreLoc")
    private static class IgnoreLocMixin {}

    @BeforeAll
    static void setup() throws Exception {
        // Ensure oracle dependencies are installed
        ProcessBuilder pb = new ProcessBuilder("npm", "install");
        pb.directory(new java.io.File("src/test/resources"));
        pb.inheritIO();
        Process process = pb.start();
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("Failed to install oracle dependencies");
        }
    }

    @Test
    @DisplayName("Parser should match oracle on simple literal")
    void testSimpleLiteral() throws Exception {
        String source = "42;";

        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source);

        assertNotNull(expected);
        assertEquals("Program", expected.type());

        assertNotNull(actual);
        assertEquals("Program", actual.type());

        System.out.println("Oracle AST:");
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expected));
        System.out.println("\nActual AST:");
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actual));
    }

    @ParameterizedTest
    @ValueSource(strings = {
        "42;",
        "\"hello\";",
        "true;",
        "x;",
        "1 + 2;",
        "x = 5;",
    })
    @DisplayName("Parser should match oracle on various expressions")
    void testOracleComparison(String source) throws Exception {
        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source);

        assertNotNull(expected);
        assertEquals("Program", expected.type());

        assertNotNull(actual);
        assertEquals("Program", actual.type());

        // Compare ASTs including source locations
        String expectedJson = mapper.writeValueAsString(expected);
        String actualJson = mapper.writeValueAsString(actual);

        System.out.println("Testing: " + source);
        if (!expectedJson.equals(actualJson)) {
            System.out.println("Expected: " + expectedJson);
            System.out.println("Actual:   " + actualJson);
        }

        assertEquals(expectedJson, actualJson, "AST mismatch for: " + source);
    }
}
