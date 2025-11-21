package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class FindJSONParseIssue {
    @Test
    void findIssue() throws Exception {
        Path testFile = Paths.get("test-oracles/test262/test/staging/sm/JSON/parse.js");
        String source = Files.readString(testFile);

        Program program = Parser.parse(source);
        ObjectMapper mapper = new ObjectMapper();
        
        var tree = mapper.readTree(mapper.writeValueAsString(program));
        var stmt = tree.get("body").get(112);
        var expr = stmt.get("expression");
        var arg = expr.get("arguments").get(0);
        
        System.out.println("Statement type: " + stmt.get("type").asText());
        System.out.println("Expression type: " + expr.get("type").asText());
        System.out.println("Argument type: " + arg.get("type").asText());
        System.out.println("Argument end: " + arg.get("end").asInt());
        System.out.println("Expected end: 4406");
        System.out.println("Difference: " + (4406 - arg.get("end").asInt()));
        
        if (arg.has("property")) {
            System.out.println("Property type: " + arg.get("property").get("type").asText());
            System.out.println("Property: " + arg.get("property"));
        }
        
        // Find what's at position 4401 and 4406 in source
        System.out.println("\nCharacters around position 4401-4406:");
        System.out.println("4401: '" + source.charAt(4401) + "'");
        System.out.println("4402: '" + source.charAt(4402) + "'");
        System.out.println("4403: '" + source.charAt(4403) + "'");
        System.out.println("4404: '" + source.charAt(4404) + "'");
        System.out.println("4405: '" + source.charAt(4405) + "'");
        System.out.println("4406: '" + source.charAt(4406) + "'");
        
        // Find the line
        int lineNum = 1;
        for (int i = 0; i < 4401 && i < source.length(); i++) {
            if (source.charAt(i) == '\n') lineNum++;
        }
        System.out.println("Line number around position 4401: " + lineNum);
        
        String[] lines = source.split("\n");
        System.out.println("Line content: " + lines[lineNum - 1]);
    }
}
