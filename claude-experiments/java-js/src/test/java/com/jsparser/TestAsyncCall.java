package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestAsyncCall {

    @Test
    void testAsyncCallExpression() throws Exception {
        // async(obj) - should be CallExpression (calling a variable named 'async')
        String code = "async = function() { return 42; };\nvar z = async(obj);";
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        JsonNode init = tree.get("body").get(1).get("declarations").get(0).get("init");

        assertEquals("CallExpression", init.get("type").asText(),
            "async(obj) should be parsed as CallExpression, not async arrow function");
        assertEquals("async", init.get("callee").get("name").asText());
    }

    @Test
    void testAsyncArrowFunction() throws Exception {
        // async(obj)=>{} - should be ArrowFunctionExpression
        String code = "var w = async(obj)=>{};";
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        JsonNode init = tree.get("body").get(0).get("declarations").get(0).get("init");

        assertEquals("ArrowFunctionExpression", init.get("type").asText());
        assertTrue(init.get("async").asBoolean());
    }

    @Test
    void testFullAsyncFile() throws Exception {
        // Test the actual failing file
        String code = Files.readString(Paths.get("test-oracles/test262/test/staging/sm/async-functions/async-contains-unicode-escape.js"));
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        JsonNode body = tree.get("body");

        System.out.println("Body has " + body.size() + " statements");

        // Find the statement "var z = async(obj);" - search for it
        boolean foundCallExpr = false;
        boolean foundArrowExpr = false;

        for (int i = 0; i < body.size(); i++) {
            JsonNode stmt = body.get(i);
            if (stmt.get("type").asText().equals("VariableDeclaration")) {
                JsonNode decls = stmt.get("declarations");
                if (decls != null && decls.size() > 0) {
                    JsonNode decl = decls.get(0);
                    JsonNode id = decl.get("id");
                    JsonNode init = decl.get("init");

                    if (id != null && init != null) {
                        String varName = id.get("name").asText();

                        // Check for "var z = async(obj);"
                        if ("z".equals(varName)) {
                            System.out.println("Found 'var z' at index " + i);
                            System.out.println("  Type: " + init.get("type").asText());
                            assertEquals("CallExpression", init.get("type").asText(),
                                "var z = async(obj); should be CallExpression");
                            foundCallExpr = true;
                        }

                        // Check for "var w = async(obj)=>{};"
                        if ("w".equals(varName)) {
                            System.out.println("Found 'var w' at index " + i);
                            System.out.println("  Type: " + init.get("type").asText());
                            assertEquals("ArrowFunctionExpression", init.get("type").asText(),
                                "var w = async(obj)=>{}; should be ArrowFunctionExpression");
                            assertTrue(init.get("async").asBoolean());
                            foundArrowExpr = true;
                        }
                    }
                }
            }
        }

        assertTrue(foundCallExpr, "Should have found 'var z = async(obj);'");
        assertTrue(foundArrowExpr, "Should have found 'var w = async(obj)=>{};'");
    }
}
