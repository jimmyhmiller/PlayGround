package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.jsparser.ast.Node;
import org.junit.jupiter.api.Test;

public class SimpleBigIntTest {
    @Test
    public void testBigIntInPattern() throws Exception {
        String source = "let { 1n: a } = { \"1\": \"foo\" };";
        Parser parser = new Parser(source);
        Node ast = parser.parse();

        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(ast);
        System.out.println(json);

        JsonNode root = mapper.readTree(json);
        JsonNode key = root.get("body").get(0).get("declarations").get(0).get("id").get("properties").get(0).get("key");
        System.out.println("\n=== Key node ===");
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(key));
    }
}
