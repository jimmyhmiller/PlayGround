package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class TestSimpleParse {

    @Test
    void testSimpleVar() throws Exception {
        String code = "var x = 1;";
        Program ast = Parser.parse(code);
        
        ObjectMapper mapper = new ObjectMapper();
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(ast));
    }
}
