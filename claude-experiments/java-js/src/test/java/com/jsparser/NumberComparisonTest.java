package com.jsparser;

import com.fasterxml.jackson.databind.*;
import org.junit.jupiter.api.Test;
import java.nio.file.*;

public class NumberComparisonTest {
    @Test
    void compareNumbers() throws Exception {
        String source = Files.readString(Path.of("tests/test-number-cases.js"));
        var prog = Parser.parse(source, false);

        ObjectMapper m = new ObjectMapper();
        m.enable(SerializationFeature.INDENT_OUTPUT);

        String actualJson = m.writeValueAsString(prog);
        String expectedJson = Files.readString(Path.of("tests/test-number-cases-acorn.json"));

        JsonNode expected = m.readTree(expectedJson);
        JsonNode actual = m.readTree(actualJson);

        // Extract all literals and compare
        System.out.println("=== Number Comparison ===");
        System.out.println(String.format("%-40s | %-25s | %-25s | %s", "Variable", "Expected Value", "Actual Value", "Match"));
        System.out.println("-".repeat(120));

        JsonNode expBody = expected.get("body");
        JsonNode actBody = actual.get("body");

        for (int i = 0; i < expBody.size() && i < actBody.size(); i++) {
            JsonNode expStmt = expBody.get(i);
            JsonNode actStmt = actBody.get(i);

            if ("VariableDeclaration".equals(expStmt.path("type").asText())) {
                JsonNode expDecl = expStmt.path("declarations").get(0);
                JsonNode actDecl = actStmt.path("declarations").get(0);

                String varName = expDecl.path("id").path("name").asText();
                JsonNode expInit = expDecl.path("init");
                JsonNode actInit = actDecl.path("init");

                if ("Literal".equals(expInit.path("type").asText())) {
                    String expValue = expInit.path("value").toString();
                    String actValue = actInit.path("value").toString();
                    String expRaw = expInit.path("raw").asText();
                    boolean matches = expValue.equals(actValue);

                    System.out.println(String.format("%-40s | %-25s | %-25s | %s",
                        varName + " (raw: " + expRaw + ")",
                        expValue,
                        actValue,
                        matches ? "✓" : "✗"));
                }
            }
        }
    }
}
