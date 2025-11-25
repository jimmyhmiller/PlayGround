package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class TestDebugSeqEnd {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    @Test
    public void testSimpleCases() throws Exception {
        String[] cases = {
            "(a, b);",           // No paren on last expr: should end at 'b'
            "(a, (b));",         // Paren on last expr: should include first )
            "((a), (b));",       // Parens on both: should include first )
        };

        for (String code : cases) {
            System.out.println("\n=== Testing: " + code + " ===");
            Program program = Parser.parse(code, false);
            String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(program);

            // Extract just the sequence expression
            @SuppressWarnings("unchecked")
            java.util.Map<String, Object> obj = mapper.readValue(json, java.util.Map.class);
            @SuppressWarnings("unchecked")
            java.util.List<Object> body = (java.util.List<Object>) obj.get("body");
            @SuppressWarnings("unchecked")
            java.util.Map<String, Object> stmt = (java.util.Map<String, Object>) body.get(0);
            @SuppressWarnings("unchecked")
            java.util.Map<String, Object> seq = (java.util.Map<String, Object>) stmt.get("expression");

            System.out.println("Sequence: start=" + seq.get("start") + ", end=" + seq.get("end"));
        }
    }
}
