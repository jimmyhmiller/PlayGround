package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Node;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

public class DebugMismatchesTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    public void testCarriageReturnASI() throws IOException {
        String testFile = "test-oracles/test262/test/language/comments/multi-line-asi-carriage-return.js";
        debugFile(testFile);
    }

    @Test
    public void testLineSeparatorASI() throws IOException {
        String testFile = "test-oracles/test262/test/language/comments/multi-line-asi-line-separator.js";
        debugFile(testFile);
    }

    @Test
    public void testLineFeedASI() throws IOException {
        String testFile = "test-oracles/test262/test/language/comments/multi-line-asi-line-feed.js";
        debugFile(testFile);
    }

    @Test
    public void testParagraphSeparatorASI() throws IOException {
        String testFile = "test-oracles/test262/test/language/comments/multi-line-asi-paragraph-separator.js";
        debugFile(testFile);
    }

    @Test
    public void testWithStrictMode() throws IOException {
        String testFile = "test-oracles/test262/test/language/statements/with/12.10.1-13-s.js";
        debugFile(testFile);
    }

    @Test
    public void testClassFieldSetGeneratorASI() throws IOException {
        String testFile = "test-oracles/test262/test/language/statements/class/elements/syntax/valid/grammar-field-named-set-followed-by-generator-asi.js";
        debugFile(testFile);
    }

    @Test
    public void testClassFieldGetGeneratorASI() throws IOException {
        String testFile = "test-oracles/test262/test/language/statements/class/elements/syntax/valid/grammar-field-named-get-followed-by-generator-asi.js";
        debugFile(testFile);
    }

    @Test
    public void testBigIntPropertyName() throws IOException {
        String testFile = "test-oracles/test262/test/language/expressions/object/literal-property-name-bigint.js";
        debugFile(testFile);
    }

    @Test
    public void testNewTargetASI() throws IOException {
        String testFile = "test-oracles/test262/test/language/expressions/new.target/asi.js";
        debugFile(testFile);
    }

    private void debugFile(String testFile) throws IOException {
        Path sourcePath = Paths.get(testFile);
        String source = Files.readString(sourcePath);

        // Get cache path
        String relativePath = testFile.replace("test-oracles/test262/test/", "");
        Path cachePath = Paths.get("test-oracles/test262-cache", relativePath + ".json");

        System.out.println("\n=== Debugging: " + testFile + " ===");
        System.out.println("Source length: " + source.length());

        // Parse with Java parser
        Parser parser = new Parser(source);
        Node ast = parser.parse();
        String javaJson = objectMapper.writeValueAsString(ast);

        // Read cached JSON (from Acorn)
        String cachedJson = Files.readString(cachePath);

        // Compare
        JsonNode javaNode = objectMapper.readTree(javaJson);
        JsonNode cachedNode = objectMapper.readTree(cachedJson);

        boolean matches = javaNode.equals(cachedNode);
        System.out.println("Matches: " + matches);

        if (!matches) {
            System.out.println("\n--- Java AST ---");
            System.out.println(objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(javaNode));
            System.out.println("\n--- Acorn AST (cached) ---");
            System.out.println(objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(cachedNode));

            // Find differences
            findDifferences(javaNode, cachedNode, "");
        }
    }

    private void findDifferences(JsonNode java, JsonNode acorn, String path) {
        if (java == null && acorn == null) return;
        if (java == null) {
            System.out.println("DIFF at " + path + ": Java is null, Acorn has value: " + acorn);
            return;
        }
        if (acorn == null) {
            System.out.println("DIFF at " + path + ": Acorn is null, Java has value: " + java);
            return;
        }

        if (java.isObject() && acorn.isObject()) {
            java.fieldNames().forEachRemaining(field -> {
                if (!acorn.has(field)) {
                    System.out.println("DIFF at " + path + "." + field + ": Only in Java AST");
                } else {
                    findDifferences(java.get(field), acorn.get(field), path + "." + field);
                }
            });
            acorn.fieldNames().forEachRemaining(field -> {
                if (!java.has(field)) {
                    System.out.println("DIFF at " + path + "." + field + ": Only in Acorn AST");
                }
            });
        } else if (java.isArray() && acorn.isArray()) {
            if (java.size() != acorn.size()) {
                System.out.println("DIFF at " + path + ": Array size mismatch - Java: " + java.size() + ", Acorn: " + acorn.size());
            }
            for (int i = 0; i < Math.min(java.size(), acorn.size()); i++) {
                findDifferences(java.get(i), acorn.get(i), path + "[" + i + "]");
            }
        } else if (!java.equals(acorn)) {
            System.out.println("DIFF at " + path + ": Java=" + java + ", Acorn=" + acorn);
        }
    }
}
