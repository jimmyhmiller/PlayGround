package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class TestDebugSequence {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    @Test
    public void testSimpleSequence() throws Exception {
        String code = "(a, b);";
        System.out.println("=== Testing: " + code + " ===");
        Program program = Parser.parse(code, false);
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(program);
        System.out.println(json);
    }

    @Test
    public void testNestedSequence() throws Exception {
        String code = "((a), b);";
        System.out.println("=== Testing: " + code + " ===");
        Program program = Parser.parse(code, false);
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(program);
        System.out.println(json);
    }

    @Test
    public void testDoubleNestedSequence() throws Exception {
        String code = "x ? a : ((y = 1), y);";
        System.out.println("=== Testing: " + code + " ===");
        Program program = Parser.parse(code, false);
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(program);
        System.out.println(json);
    }
}
