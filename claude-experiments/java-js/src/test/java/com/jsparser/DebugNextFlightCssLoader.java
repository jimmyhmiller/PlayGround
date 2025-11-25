package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Debug test for next-flight-css-loader.js parsing issue
 */
public class DebugNextFlightCssLoader {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    @Test
    public void testNextFlightCssLoader() throws Exception {
        String filePath = "../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/esm/build/webpack/loaders/next-flight-css-loader.js";
        String source = Files.readString(Paths.get(filePath));

        System.out.println("=== Source ===");
        System.out.println(source);
        System.out.println("\n=== Parsing with Java ===");

        Program ast = Parser.parse(source, true);
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(ast);

        System.out.println(json);
    }
}
