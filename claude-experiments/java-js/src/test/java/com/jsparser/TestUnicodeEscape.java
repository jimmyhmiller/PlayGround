package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class TestUnicodeEscape {

    @Test
    void testConstEscaped() throws Exception {
        String code = Files.readString(Paths.get("test-oracles/test262/test/language/expressions/object/ident-name-method-def-const-escaped.js"));
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        
        // Navigate to the property key
        JsonNode propertyKey = tree.get("body").get(0).get("declarations").get(0).get("init").get("properties").get(0).get("key");
        
        System.out.println("Property key: " + propertyKey);
        System.out.println("Type: " + propertyKey.get("type"));
        System.out.println("Name: " + propertyKey.get("name"));
        
        // Load expected
        String expectedJson = Files.readString(Paths.get("test-oracles/test262-cache/language/expressions/object/ident-name-method-def-const-escaped.js.json"));
        JsonNode expected = mapper.readTree(expectedJson);
        JsonNode expectedKey = expected.get("body").get(0).get("declarations").get(0).get("init").get("properties").get(0).get("key");
        
        System.out.println("\nExpected name: " + expectedKey.get("name"));
        System.out.println("Actual name: " + propertyKey.get("name"));
    }
}
