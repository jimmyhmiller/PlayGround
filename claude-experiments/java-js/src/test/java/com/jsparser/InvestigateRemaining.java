package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class InvestigateRemaining {
    @Test
    void investigateTemporal() throws Exception {
        Path testFile = Paths.get("test-oracles/test262/test/built-ins/Temporal/PlainTime/compare/argument-string-time-designator-required-for-disambiguation.js");
        String source = Files.readString(testFile);

        Program program = Parser.parse(source);
        ObjectMapper mapper = new ObjectMapper();
        
        var tree = mapper.readTree(mapper.writeValueAsString(program));
        var expr = tree.get("body").get(1).get("expression").get("arguments").get(0).get("body").get("body").get(3).get("expression");
        
        System.out.println("Expression type: " + expr.get("type").asText());
        System.out.println("Expression end: " + expr.get("end").asInt());
        System.out.println("Expected: 845");
        
        // Find what's at those positions
        System.out.println("\nCharacter at 843: '" + source.charAt(843) + "'");
        System.out.println("Character at 844: '" + source.charAt(844) + "'");
        System.out.println("Character at 845: '" + source.charAt(845) + "'");
        
        // Find the line
        int lineNum = 1;
        for (int i = 0; i < 843 && i < source.length(); i++) {
            if (source.charAt(i) == '\n') lineNum++;
        }
        System.out.println("Line number: " + lineNum);
        
        String[] lines = source.split("\n");
        if (lineNum <= lines.length) {
            System.out.println("Line: " + lines[lineNum - 1]);
        }
    }
}
