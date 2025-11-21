package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class InvestigateLargeNumber {
    @Test
    void investigate() throws Exception {
        Path testFile = Paths.get("test-oracles/test262/test/built-ins/Temporal/PlainTime/prototype/add/add-large-subseconds.js");
        String source = Files.readString(testFile);

        Program program = Parser.parse(source);
        ObjectMapper mapper = new ObjectMapper();

        var tree = mapper.readTree(mapper.writeValueAsString(program));
        var decl = tree.get("body").get(9).get("declarations").get(0);

        System.out.println("Variable name: " + decl.get("id").get("name").asText());
        System.out.println("Init value: " + decl.get("init").get("value"));
        System.out.println("Init raw: " + decl.get("init").get("raw").asText());
        System.out.println("Expected value: 9007199254740991000");

        // Find the line
        String[] lines = source.split("\n");
        System.out.println("\nLine 33: " + lines[32]);
    }
}
