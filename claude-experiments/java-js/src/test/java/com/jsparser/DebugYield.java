package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class DebugYield {
    @Test
    public void test() throws Exception {
        String code = """
        var yield = 23;
        function *g() {
          function f(x = yield) {
            paramValue = x;
          }
          f();
        }
        """;
        Parser parser = new Parser(code);
        Program program = parser.parse();
        
        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(program);
        JsonNode root = mapper.readTree(json);
        
        // Navigate to the yield in the parameter default
        JsonNode yieldNode = root.get("body").get(1).get("body").get("body").get(0).get("params").get(0).get("right");
        System.out.println("Yield node type: " + yieldNode.get("type").asText());
        System.out.println("Full node:");
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(yieldNode));
    }
}
