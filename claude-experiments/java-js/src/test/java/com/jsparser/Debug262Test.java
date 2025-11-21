package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class Debug262Test {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void debugNans() throws Exception {
        String source = Files.readString(Paths.get("test-oracles/test262/test/harness/nans.js"));

        System.out.println("=== SOURCE ===");
        System.out.println(source);

        // Parse with our parser
        Program actual = Parser.parse(source);
        String actualJson = mapper.writeValueAsString(actual);

        // Parse with oracle
        Program expected = OracleParser.parse(source);
        String expectedJson = mapper.writeValueAsString(expected);

        System.out.println("\n=== OUR JSON ===");
        System.out.println(actualJson);
        System.out.println("\n=== ORACLE JSON ===");
        System.out.println(expectedJson);

        System.out.println("\n=== MATCH? ===");
        System.out.println(actualJson.equals(expectedJson));
    }
}
