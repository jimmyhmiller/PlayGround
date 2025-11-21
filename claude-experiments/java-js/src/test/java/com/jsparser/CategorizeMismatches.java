package com.jsparser;

import com.fasterxml.jackson.databind.*;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

public class CategorizeMismatches {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void categorize() throws IOException {
        Path test262Dir = Paths.get("test-oracles/test262/test");
        Path cacheDir = Paths.get("test-oracles/test262-cache");

        Map<String, Integer> categories = new HashMap<>();
        List<String> otherFiles = new ArrayList<>();
        List<String> positionFiles = new ArrayList<>();
        List<String> valueRawFiles = new ArrayList<>();

        try (Stream<Path> paths = Files.walk(test262Dir)) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> p.toString().endsWith(".js"))
                 .forEach(path -> {
                     try {
                         String source = Files.readString(path);
                         Path relativePath = test262Dir.relativize(path);
                         Path cacheFile = cacheDir.resolve(relativePath.toString() + ".json");

                         if (!Files.exists(cacheFile)) return;

                         String expectedJson = Files.readString(cacheFile);
                         boolean isModule = Parser.hasModuleFlag(source) ||
                             path.toString().endsWith("_FIXTURE.js");
                         Program actualProgram = Parser.parse(source, isModule);
                         String actualJson = mapper.writeValueAsString(actualProgram);

                         Object expectedObj = mapper.readValue(expectedJson, Object.class);
                         Object actualObj = mapper.readValue(actualJson, Object.class);

                         // Normalize regex value differences
                         normalizeRegexValues(expectedObj, actualObj);

                         if (!Objects.deepEquals(expectedObj, actualObj)) {
                             JsonNode expected = mapper.readTree(expectedJson);
                             JsonNode actual = mapper.readTree(actualJson);
                             String cat = findCategory(expected, actual);
                             categories.merge(cat, 1, Integer::sum);
                             if (cat.equals("other") && otherFiles.size() < 10) {
                                 otherFiles.add(path.toString());
                             }
                             if (cat.equals("position") && positionFiles.size() < 10) {
                                 positionFiles.add(path.toString());
                             }
                             if (cat.equals("value/raw") && valueRawFiles.size() < 10) {
                                 valueRawFiles.add(path.toString());
                             }
                         }
                     } catch (Exception e) {}
                 });
        }

        categories.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .limit(20)
            .forEach(e -> System.out.printf("%-50s: %4d\n", e.getKey(), e.getValue()));

        if (!otherFiles.isEmpty()) {
            System.out.println("\nExample 'other' mismatch files:");
            otherFiles.forEach(System.out::println);
        }

        if (!positionFiles.isEmpty()) {
            System.out.println("\nExample 'position' mismatch files:");
            positionFiles.forEach(System.out::println);
        }

        if (!valueRawFiles.isEmpty()) {
            System.out.println("\nExample 'value/raw' mismatch files:");
            valueRawFiles.forEach(System.out::println);
        }
    }
    
    private String findCategory(JsonNode expected, JsonNode actual) {
        // Check sourceType first
        if (!expected.path("sourceType").equals(actual.path("sourceType"))) {
            return "sourceType";
        }
        return findDiff(expected, actual, "");
    }
    
    private String findDiff(JsonNode e, JsonNode a, String path) {
        if (e.isObject() && a.isObject()) {
            for (var it = e.fieldNames(); it.hasNext();) {
                String field = it.next();
                if (!e.get(field).equals(a.path(field))) {
                    if (field.equals("start") || field.equals("end") || field.equals("loc")) {
                        return "position";
                    }
                    if (field.equals("value") || field.equals("raw")) {
                        return "value/raw";
                    }
                    return findDiff(e.get(field), a.path(field), path + "." + field);
                }
            }
        }
        if (e.isArray() && a.isArray()) {
            for (int i = 0; i < e.size() && i < a.size(); i++) {
                if (!e.get(i).equals(a.get(i))) {
                    return findDiff(e.get(i), a.get(i), path + "[" + i + "]");
                }
            }
            if (e.size() != a.size()) return "array-size";
        }
        return "other";
    }

    @SuppressWarnings("unchecked")
    private void normalizeRegexValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            if ("Literal".equals(expMap.get("type")) && expMap.containsKey("regex")) {
                if (expMap.get("value") == null && actMap.get("value") instanceof Map) {
                    Map<?, ?> actValue = (Map<?, ?>) actMap.get("value");
                    if (actValue.isEmpty()) {
                        actMap.put("value", null);
                    }
                }
            }

            for (String key : expMap.keySet()) {
                if (actMap.containsKey(key)) {
                    normalizeRegexValues(expMap.get(key), actMap.get(key));
                }
            }
        } else if (expected instanceof List && actual instanceof List) {
            List<Object> expList = (List<Object>) expected;
            List<Object> actList = (List<Object>) actual;
            for (int i = 0; i < Math.min(expList.size(), actList.size()); i++) {
                normalizeRegexValues(expList.get(i), actList.get(i));
            }
        }
    }
}
