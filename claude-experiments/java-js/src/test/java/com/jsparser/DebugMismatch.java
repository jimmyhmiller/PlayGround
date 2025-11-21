package com.jsparser;

import com.fasterxml.jackson.databind.*;
import org.junit.jupiter.api.Test;
import java.nio.file.*;

public class DebugMismatch {
    @Test
    void test() throws Exception {
        String file = "test-oracles/test262/test/built-ins/parseInt/S15.1.2.2_A7.2_T3.js";
        String source = Files.readString(Path.of(file));
        var prog = Parser.parse(source, false);

        ObjectMapper m = new ObjectMapper();
        m.enable(SerializationFeature.INDENT_OUTPUT);
        String actualJson = m.writeValueAsString(prog);

        String expectedJson = Files.readString(Path.of(file.replace("test262/test/", "test262-cache/") + ".json"));

        JsonNode expected = m.readTree(expectedJson);
        JsonNode actual = m.readTree(actualJson);

        System.out.println("Are they equal? " + expected.equals(actual));

        // Show first difference
        System.out.println("\nExpected body[0]:");
        System.out.println(expected.get("body").get(0).toPrettyString().substring(0, 500));
        System.out.println("\nActual body[0]:");
        System.out.println(actual.get("body").get(0).toPrettyString().substring(0, 500));
    }

    private void findLargeNumber(JsonNode node, String label) {
        if (node.isObject()) {
            if ("Literal".equals(node.path("type").asText()) && node.has("value")) {
                JsonNode value = node.get("value");
                JsonNode raw = node.get("raw");
                if (value.isNumber() && raw != null) {
                    System.out.println(label + " - raw: " + raw.asText() + ", value: " + value);
                }
            }
            node.fields().forEachRemaining(entry -> findLargeNumber(entry.getValue(), label));
        } else if (node.isArray()) {
            for (JsonNode child : node) {
                findLargeNumber(child, label);
            }
        }
    }
}
