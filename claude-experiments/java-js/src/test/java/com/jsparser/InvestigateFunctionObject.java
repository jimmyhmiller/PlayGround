package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class InvestigateFunctionObject {
    @Test
    void investigate() throws Exception {
        Path testFile = Paths.get("test-oracles/test262/test/built-ins/Function/prototype/toString/built-in-function-object.js");
        String source = Files.readString(testFile);

        Program program = Parser.parse(source);
        ObjectMapper mapper = new ObjectMapper();
        
        var tree = mapper.readTree(mapper.writeValueAsString(program));
        var decl = tree.get("body").get(1).get("body").get("body").get(5).get("expression")
            .get("arguments").get(0).get("body").get("body").get(1).get("declarations").get(0);
        
        System.out.println("Declarator end: " + decl.get("end").asInt());
        System.out.println("Expected: 1295");
        System.out.println("Init end: " + decl.get("init").get("end").asInt());
        System.out.println("Init type: " + decl.get("init").get("type").asText());
        
        // Find what's at those positions
        System.out.println("\nCharacter at 1293: '" + source.charAt(1293) + "'");
        System.out.println("Character at 1294: '" + source.charAt(1294) + "'");
        System.out.println("Character at 1295: '" + source.charAt(1295) + "'");
        
        // Find the line
        int lineNum = 1;
        for (int i = 0; i < 1293 && i < source.length(); i++) {
            if (source.charAt(i) == '\n') lineNum++;
        }
        System.out.println("Line number: " + lineNum);
        
        String[] lines = source.split("\n");
        if (lineNum <= lines.length) {
            System.out.println("Line: " + lines[lineNum - 1]);
        }
    }
}
