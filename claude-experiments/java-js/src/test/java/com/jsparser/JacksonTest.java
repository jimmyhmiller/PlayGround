package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.module.paramnames.ParameterNamesModule;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class JacksonTest {
    @Test
    void testJacksonDeserialization() throws Exception {
        String json = """
            {
              "type": "Program",
              "body": [],
              "sourceType": "script"
            }
            """;

        ObjectMapper mapper = new ObjectMapper()
                .registerModule(new ParameterNamesModule());

        Program program = mapper.readValue(json, Program.class);
        System.out.println("Program: " + program);
        System.out.println("Type: " + program.type());
        System.out.println("SourceType: " + program.sourceType());
        System.out.println("Body: " + program.body());
    }
}
