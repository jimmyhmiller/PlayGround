package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.*;
import org.junit.jupiter.api.Test;

public class JsonOrderTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void testVariableDeclaratorOrder() throws Exception {
        Identifier id = new Identifier(247, 250,
            new SourceLocation(new SourceLocation.Position(10, 4), new SourceLocation.Position(10, 7)), "obj");
        VariableDeclarator vd = new VariableDeclarator(247, 250,
            new SourceLocation(new SourceLocation.Position(10, 4), new SourceLocation.Position(10, 7)), id, null);

        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(vd);
        System.out.println("VariableDeclarator JSON:");
        System.out.println(json);

        String[] lines = json.split("\n");
        for (int i = 0; i < Math.min(10, lines.length); i++) {
            System.out.println("Line " + (i+1) + ": " + lines[i]);
        }
    }
}
