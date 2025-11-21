package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class DebugAsyncArrow {
    @Test
    public void test() throws Exception {
        String code = "async\nidentifier => {}";
        Parser parser = new Parser(code);
        Program program = parser.parse();
        
        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(program);
        JsonNode root = mapper.readTree(json);
        
        JsonNode expr = root.get("body").get(0).get("expression");
        System.out.println("Expression type: " + expr.get("type").asText());
        System.out.println("Full expression:");
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expr));
    }
}
