package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestSequenceParens {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    @Test
    public void testSequenceInTernary() throws Exception {
        String code = "x ? a : ((y = 1), y);";

        // Expected from Acorn:
        // SequenceExpression at positions 9-19 (includes its parens, not outer parens)

        Program program = Parser.parse(code, false);
        String javaJson = mapper.writeValueAsString(program);

        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> javaObj = mapper.readValue(javaJson, java.util.Map.class);
        @SuppressWarnings("unchecked")
        java.util.List<Object> body = (java.util.List<Object>) javaObj.get("body");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> stmt = (java.util.Map<String, Object>) body.get(0);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> ternary = (java.util.Map<String, Object>) stmt.get("expression");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> alternate = (java.util.Map<String, Object>) ternary.get("alternate");

        Integer start = (Integer) alternate.get("start");
        Integer end = (Integer) alternate.get("end");
        String type = (String) alternate.get("type");

        System.out.println("Alternate (sequence) type: " + type);
        System.out.println("  start: " + start + " (expected: 9)");
        System.out.println("  end: " + end + " (expected: 19)");

        System.out.println("\nCode positions:");
        for (int i = 0; i < code.length(); i++) {
            System.out.println(i + ": " + code.charAt(i));
        }

        assertEquals("SequenceExpression", type);
        assertEquals(9, start, "Should start at inner '(', position 9");
        assertEquals(19, end, "Should end at matching ')', position 19");
    }
}
