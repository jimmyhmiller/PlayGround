package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class DebugCompareTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void debugCompare() throws Exception {
        String source = "assert(true);";

        // Parse with our parser
        Program actual = Parser.parse(source);
        String actualJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actual);

        // Parse with oracle
        Program expected = OracleParser.parse(source);
        String expectedJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expected);

        System.out.println("=== OUR PARSER ===");
        System.out.println(actualJson);
        System.out.println("\n=== ACORN (via oracle) ===");
        System.out.println(expectedJson);

        // Also write to files for easier diffing
        Files.writeString(Paths.get("/tmp/ours.json"), actualJson);
        Files.writeString(Paths.get("/tmp/acorn.json"), expectedJson);

        System.out.println("\nWritten to /tmp/ours.json and /tmp/acorn.json");
        System.out.println("Run: diff /tmp/acorn.json /tmp/ours.json");
    }
}
