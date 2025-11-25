package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class TestColumnOffset {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    @Test
    public void testColumnOffset() throws Exception {
        String source = "const x = 1;\nconst y = 2;\n";

        System.out.println("=== Source ===");
        System.out.println(source);
        System.out.println("\n=== Parsing with Java ===");

        Program ast = Parser.parse(source, false);
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(ast);

        // Look for column values
        System.out.println(json);
    }
}
