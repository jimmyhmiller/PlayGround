package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class TestLocEnd {

    @Test
    void testRegressFile() throws Exception {
        String code = Files.readString(Paths.get("test-oracles/test262/test/staging/sm/regress/regress-593256.js"));
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        JsonNode loc = tree.get("loc");

        System.out.println("File lines: " + code.split("\n").length);
        System.out.println("File length: " + code.length());
        System.out.println("Last char: " + (int)code.charAt(code.length() - 1));
        System.out.println();
        System.out.println("Program loc: " + loc);
        System.out.println("Expected end: line 28, column 0");
        System.out.println("Actual end: line " + loc.get("end").get("line") + ", column " + loc.get("end").get("column"));
    }

    @Test
    void testSimpleFile() throws Exception {
        // Test with simple content
        String code = "var x = 1;\n";
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        JsonNode loc = tree.get("loc");

        System.out.println("\nSimple file test:");
        System.out.println("Code: " + code.replace("\n", "\\n"));
        System.out.println("Length: " + code.length());
        System.out.println("Program loc: " + loc);
    }

    @Test
    void testLineTerminatorFile() throws Exception {
        String code = Files.readString(Paths.get("test-oracles/test262/test/built-ins/Function/prototype/toString/line-terminator-normalisation-CR.js"));
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        JsonNode loc = tree.get("loc");

        System.out.println("\nLine terminator file:");
        System.out.println("File length: " + code.length());
        System.out.println("Program loc: " + loc);

        // Show the expected
        String expectedJson = Files.readString(Paths.get("test-oracles/test262-cache/built-ins/Function/prototype/toString/line-terminator-normalisation-CR.js.json"));
        JsonNode expected = mapper.readTree(expectedJson);
        System.out.println("Expected loc: " + expected.get("loc"));
    }
}
