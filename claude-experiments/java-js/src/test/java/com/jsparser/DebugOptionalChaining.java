package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class DebugOptionalChaining {
    @Test
    public void testOptionalChaining() throws Exception {
        String code = "const value = true ?.30 : false;";
        Parser parser = new Parser(code);
        Program program = parser.parse();
        
        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(program);
        JsonNode root = mapper.readTree(json);
        
        JsonNode consequent = root.get("body").get(0).get("declarations").get(0).get("init").get("consequent");
        System.out.println("Consequent loc: " + mapper.writerWithDefaultPrettyPrinter().writeValueAsString(consequent.get("loc")));
    }
}
