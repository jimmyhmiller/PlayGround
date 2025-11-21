package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class TestSourceType {

    @Test
    void testShadowRealmImportValue() throws Exception {
        String code = Files.readString(Paths.get("test-oracles/test262/test/built-ins/ShadowRealm/prototype/importValue/import-value.js"));
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        
        System.out.println("File content (first 200 chars):");
        System.out.println(code.substring(0, Math.min(200, code.length())));
        System.out.println("\nOur sourceType: " + tree.get("sourceType"));
        
        // Check expected
        String expectedJson = Files.readString(Paths.get("test-oracles/test262-cache/built-ins/ShadowRealm/prototype/importValue/import-value.js.json"));
        JsonNode expected = mapper.readTree(expectedJson);
        System.out.println("Expected sourceType: " + expected.get("sourceType"));
        
        // Check if file contains import/export
        boolean hasImport = code.contains("import ");
        boolean hasExport = code.contains("export ");
        System.out.println("\nFile contains 'import ': " + hasImport);
        System.out.println("File contains 'export ': " + hasExport);
    }
    
    @Test
    void testFutureReservedWords() throws Exception {
        String code = Files.readString(Paths.get("test-oracles/test262/test/staging/sm/misc/future-reserved-words.js"));
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        
        System.out.println("\n=== Future Reserved Words Test ===");
        System.out.println("File content (first 300 chars):");
        System.out.println(code.substring(0, Math.min(300, code.length())));
        System.out.println("\nOur sourceType: " + tree.get("sourceType"));
        
        // Check expected
        String expectedJson = Files.readString(Paths.get("test-oracles/test262-cache/staging/sm/misc/future-reserved-words.js.json"));
        JsonNode expected = mapper.readTree(expectedJson);
        System.out.println("Expected sourceType: " + expected.get("sourceType"));
        
        // Check if file contains import/export AS KEYWORDS (not in strings)
        System.out.println("\nSearching for actual import/export keywords...");
        // The word "import" appears in the string array
        System.out.println("'\"import\"' in string: " + code.contains("\"import\""));
        System.out.println("'\"export\"' in string: " + code.contains("\"export\""));
    }
}
