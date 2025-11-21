package com.jsparser;

import com.fasterxml.jackson.databind.*;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

public class ValueRawMismatches {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void find() throws Exception {
        Path test262Dir = Paths.get("test-oracles/test262/test");
        Path cacheDir = Paths.get("test-oracles/test262-cache");
        
        List<String> samples = new ArrayList<>();
        
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
                         
                         JsonNode expected = mapper.readTree(Files.readString(cacheFile));
                         JsonNode actual = mapper.readTree(mapper.writeValueAsString(prog));
                         
                         String diff = findValueDiff(expected, actual, "");
                         if (diff != null && samples.size() < 15) {
                             samples.add(relativePath + "\n  " + diff);
                         }
                     } catch (Exception e) {}
                 });
        }
        samples.forEach(System.out::println);
    }
    
    String findValueDiff(JsonNode e, JsonNode a, String path) {
        if (e.isObject() && a.isObject()) {
            for (var it = e.fieldNames(); it.hasNext();) {
                String f = it.next();
                if (!e.get(f).equals(a.path(f))) {
                    if (f.equals("value") || f.equals("raw")) {
                        return path + "." + f + ": expected=" + e.get(f) + ", actual=" + a.path(f);
                    }
                    String sub = findValueDiff(e.get(f), a.path(f), path + "." + f);
                    if (sub != null) return sub;
                }
            }
        }
        if (e.isArray() && a.isArray()) {
            for (int i = 0; i < Math.min(e.size(), a.size()); i++) {
                String sub = findValueDiff(e.get(i), a.get(i), path + "[" + i + "]");
                if (sub != null) return sub;
            }
        }
        return null;
    }
}
