package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestParenthesizedPosition {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    @Test
    public void testParenthesizedAssignmentInSequence() throws Exception {
        String code = "console.log(1), (x[y] = !0);";

        // Expected positions from Acorn:
        // Assignment expression (x[y] = !0) without parens:
        //   start: 17 (position of 'x')
        //   end: 26 (position after '0')
        //   loc.end.column: 26

        Program program = Parser.parse(code, false);
        String javaJson = mapper.writeValueAsString(program);

        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> javaObj = mapper.readValue(javaJson, java.util.Map.class);
        @SuppressWarnings("unchecked")
        java.util.List<Object> body = (java.util.List<Object>) javaObj.get("body");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> stmt = (java.util.Map<String, Object>) body.get(0);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> seq = (java.util.Map<String, Object>) stmt.get("expression");
        @SuppressWarnings("unchecked")
        java.util.List<Object> expressions = (java.util.List<Object>) seq.get("expressions");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> assignment = (java.util.Map<String, Object>) expressions.get(1);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> loc = (java.util.Map<String, Object>) assignment.get("loc");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> end = (java.util.Map<String, Object>) loc.get("end");

        Integer start = (Integer) assignment.get("start");
        Integer assignmentEnd = (Integer) assignment.get("end");
        Integer endColumn = (Integer) end.get("column");

        System.out.println("Assignment expression positions:");
        System.out.println("  start: " + start + " (expected: 17)");
        System.out.println("  end: " + assignmentEnd + " (expected: 26)");
        System.out.println("  loc.end.column: " + endColumn + " (expected: 26)");

        // The assignment should NOT include the surrounding parentheses
        assertEquals(17, start, "Start should be at 'x', not '('");
        assertEquals(26, assignmentEnd, "End should be after '0', not ')'");
        assertEquals(26, endColumn, "End column should be 26");
    }

    @Test
    public void testParenthesizedConditionalStart() throws Exception {
        String code = "x, (a ? b : c);";

        // Expected from Acorn:
        // Conditional expression (a ? b : c) without parens:
        //   start: 4 (position of 'a')

        Program program = Parser.parse(code, false);
        String javaJson = mapper.writeValueAsString(program);

        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> javaObj = mapper.readValue(javaJson, java.util.Map.class);
        @SuppressWarnings("unchecked")
        java.util.List<Object> body = (java.util.List<Object>) javaObj.get("body");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> stmt = (java.util.Map<String, Object>) body.get(0);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> seq = (java.util.Map<String, Object>) stmt.get("expression");
        @SuppressWarnings("unchecked")
        java.util.List<Object> expressions = (java.util.List<Object>) seq.get("expressions");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> cond = (java.util.Map<String, Object>) expressions.get(1);

        Integer start = (Integer) cond.get("start");

        System.out.println("Conditional expression start: " + start + " (expected: 4)");
        assertEquals(4, start, "Start should be at 'a', not '('");
    }
}
