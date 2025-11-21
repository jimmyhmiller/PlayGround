package com.jsparser;

import com.fasterxml.jackson.databind.*;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

public class RemainingMismatches {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void find() throws Exception {
        Path test262Dir = Paths.get("test-oracles/test262/test");
        Path cacheDir = Paths.get("test-oracles/test262-cache");
        
        Map<String, List<String>> byCategory = new HashMap<>();
        
        try (Stream<Path> paths = Files.walk(test262Dir)) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> p.toString().endsWith(".js"))
                 .forEach(path -> {
                     try {
                         String source = Files.readString(path);
                         Path relativePath = test262Dir.relativize(path);
                         Path cacheFile = cacheDir.resolve(relativePath.toString() + ".json");
                         if (!Files.exists(cacheFile)) return;
                         
                         boolean isModule = Parser.hasModuleFlag(source) || path.toString().endsWith("_FIXTURE.js");
                         Program prog = Parser.parse(source, isModule);
                         
                         Object expectedObj = mapper.readValue(Files.readString(cacheFile), Object.class);
                         Object actualObj = mapper.readValue(mapper.writeValueAsString(prog), Object.class);
                         normalizeRegexValues(expectedObj, actualObj);
                         
                         if (!Objects.deepEquals(expectedObj, actualObj)) {
                             JsonNode expected = mapper.readTree(Files.readString(cacheFile));
                             JsonNode actual = mapper.readTree(mapper.writeValueAsString(prog));
                             String cat = findCategory(expected, actual);
                             byCategory.computeIfAbsent(cat, k -> new ArrayList<>()).add(relativePath.toString());
                         }
                     } catch (Exception e) {}
                 });
        }
        
        for (var entry : byCategory.entrySet()) {
            System.out.println("=== " + entry.getKey() + " (" + entry.getValue().size() + " files) ===");
            for (String f : entry.getValue().subList(0, Math.min(3, entry.getValue().size()))) {
                System.out.println("  " + f);
            }
            System.out.println();
        }
    }
    
    String findCategory(JsonNode e, JsonNode a) {
        return findDiff(e, a, "");
    }
    
    String findDiff(JsonNode e, JsonNode a, String path) {
        if (e.isObject() && a.isObject()) {
            for (var it = e.fieldNames(); it.hasNext();) {
                String field = it.next();
                if (!e.get(field).equals(a.path(field))) {
                    if (field.equals("start") || field.equals("end") || field.equals("loc")) {
                        return "position";
                    }
                    if (field.equals("value") || field.equals("raw")) {
                        return "value/raw: " + path + "." + field;
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
            if (e.size() != a.size()) return "array-size: " + path;
        }
        return "other: " + path + " exp=" + e.getNodeType() + " act=" + a.getNodeType();
    }
    
    @SuppressWarnings("unchecked")
    void normalizeRegexValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;
            if ("Literal".equals(expMap.get("type")) && expMap.containsKey("regex")) {
                if (expMap.get("value") == null && actMap.get("value") instanceof Map) {
                    if (((Map<?, ?>) actMap.get("value")).isEmpty()) {
                        actMap.put("value", null);
                    }
                }
            }
            for (String key : expMap.keySet()) {
                if (actMap.containsKey(key)) normalizeRegexValues(expMap.get(key), actMap.get(key));
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
