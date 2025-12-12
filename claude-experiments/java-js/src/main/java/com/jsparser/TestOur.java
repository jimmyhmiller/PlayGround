package com.jsparser;

import com.jsparser.ast.Program;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.nio.file.*;

public class TestOur {
    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        String code = "let o = { 1n() { return \"bar\"; } };";
        Program result = Parser.parse(code, false); // script mode
        System.out.println(mapper.writeValueAsString(result));
    }
}
