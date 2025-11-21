package com.jsparser;

import com.fasterxml.jackson.databind.*;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

public class PositionMismatchDetail {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void find() throws Exception {
        Path test262Dir = Paths.get("test-oracles/test262/test");
        Path cacheDir = Paths.get("test-oracles/test262-cache");
        
        int count = 0;
        try (Stream<Path> paths = Files.walk(test262Dir)) {
            for (Path path : (Iterable<Path>) paths.filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".js"))::iterator) {
                if (count >= 3) break;
                
                try {
                    String source = Files.readString(path);
                    Path relativePath = test262Dir.relativize(path);
                    Path cacheFile = cacheDir.resolve(relativePath.toString() + ".json");
                    if (!Files.exists(cacheFile)) continue;
                    
                    boolean isModule = Parser.hasModuleFlag(source) || path.toString().endsWith("_FIXTURE.js");
                    Program prog = Parser.parse(source, isModule);
                    
                    JsonNode expected = mapper.readTree(Files.readString(cacheFile));
                    JsonNode actual = mapper.readTree(mapper.writeValueAsString(prog));
                    
                    String diff = findPositionDiff(expected, actual, "");
                    if (diff != null) {
                        System.out.println("=== " + relativePath + " ===");
                        System.out.println(diff);
                        System.out.println();
                        count++;
                    }
                } catch (Exception e) {}
            }
        }
    }
    
    String findPositionDiff(JsonNode e, JsonNode a, String path) {
        if (e.isObject() && a.isObject()) {
            for (var it = e.fieldNames(); it.hasNext();) {
                String f = it.next();
                if (!e.get(f).equals(a.path(f))) {
                    if (f.equals("start") || f.equals("end") || f.equals("loc")) {
                        return String.format("%s.%s:\n  expected: %s\n  actual:   %s", 
                            path, f, e.get(f), a.path(f));
                    }
                    String sub = findPositionDiff(e.get(f), a.path(f), path + "." + f);
                    if (sub != null) return sub;
                }
            }
        }
        if (e.isArray() && a.isArray()) {
            for (int i = 0; i < Math.min(e.size(), a.size()); i++) {
                String sub = findPositionDiff(e.get(i), a.get(i), path + "[" + i + "]");
                if (sub != null) return sub;
            }
        }
        return null;
    }
}
