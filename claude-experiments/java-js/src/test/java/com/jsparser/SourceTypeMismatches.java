package com.jsparser;

import com.fasterxml.jackson.databind.*;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

public class SourceTypeMismatches {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void find() throws IOException {
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
                         
                         String expectedJson = Files.readString(cacheFile);
                         boolean isModule = Parser.hasModuleFlag(source);
                         Program actualProgram = Parser.parse(source, isModule);
                         String actualJson = mapper.writeValueAsString(actualProgram);
                         
                         JsonNode expected = mapper.readTree(expectedJson);
                         JsonNode actual = mapper.readTree(actualJson);
                         
                         String expType = expected.path("sourceType").asText();
                         String actType = actual.path("sourceType").asText();
                         
                         if (!expType.equals(actType) && samples.size() < 10) {
                             samples.add(String.format("%s: expected=%s, actual=%s, hasModuleFlag=%s, isFixture=%s",
                                 relativePath, expType, actType, isModule, path.toString().endsWith("_FIXTURE.js")));
                         }
                     } catch (Exception e) {}
                 });
        }
        
        System.out.println("Sample sourceType mismatches:");
        samples.forEach(System.out::println);
    }
}
