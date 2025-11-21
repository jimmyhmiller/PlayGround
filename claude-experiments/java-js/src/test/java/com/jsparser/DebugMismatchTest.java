package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DebugMismatchTest {
    @Test
    void debugAsyncGenerators() throws Exception {
        Path testFile = Paths.get("test-oracles/test262/test/staging/sm/AsyncGenerators/for-await-bad-syntax.js");
        String source = Files.readString(testFile);

        System.out.println("Line 15:");
        String[] lines = source.split("\n");
        System.out.println(lines[14]); // 0-indexed

        Program program = Parser.parse(source);
        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(program);

        // Navigate to the declarator
        var tree = mapper.readTree(json);
        var declarator = tree.get("body").get(1).get("body").get("body").get(0).get("declarations").get(0);
        System.out.println("\nDeclarator end: " + declarator.get("end").asInt());
        System.out.println("Init end: " + declarator.get("init").get("end").asInt());
    }
}
