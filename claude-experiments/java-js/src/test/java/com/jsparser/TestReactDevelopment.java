package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class TestReactDevelopment {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    @Test
    public void testReactDevelopmentFile() throws Exception {
        String filePath = "../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/react-experimental/cjs/react.development.js";
        String source = Files.readString(Paths.get(filePath));

        // Parse with Acorn
        String acornJson = parseWithAcorn(source, filePath);
        if (acornJson == null) {
            System.out.println("Acorn failed to parse");
            return;
        }

        // Parse with Java
        Program javaAst = Parser.parse(source, false);
        String javaJson = mapper.writeValueAsString(javaAst);

        // Parse both for structural comparison
        Object acornObj = mapper.readValue(acornJson, Object.class);
        Object javaObj = mapper.readValue(javaJson, Object.class);

        // Find differences
        List<String> diffs = findDifferences(acornObj, javaObj, "");

        System.out.println("Found " + diffs.size() + " differences");
        if (!diffs.isEmpty()) {
            System.out.println("First 30 differences:");
            for (int i = 0; i < Math.min(30, diffs.size()); i++) {
                System.out.println("  " + diffs.get(i));
            }
        }
    }

    private String parseWithAcorn(String source, String filePath) throws Exception {
        Path tempFile = Files.createTempFile("acorn-ast-", ".json");
        try {
            String[] cmd = {
                    "node",
                    "-e",
                    "const acorn = require('acorn'); " +
                            "const fs = require('fs'); " +
                            "const source = fs.readFileSync(process.argv[1], 'utf-8'); " +
                            "try { const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: 'script'}); " +
                            "fs.writeFileSync(process.argv[2], JSON.stringify(ast, (k,v) => typeof v === 'bigint' ? null : v)); " +
                            "process.exit(0); } catch(e) { console.error(e.message); process.exit(1); }",
                    filePath,
                    tempFile.toAbsolutePath().toString()
            };

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                return null;
            }

            return Files.readString(tempFile);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    @SuppressWarnings("unchecked")
    private List<String> findDifferences(Object expected, Object actual, String path) {
        List<String> diffs = new ArrayList<>();

        if (expected == null && actual == null) {
            return diffs;
        }

        if (expected == null || actual == null) {
            diffs.add(path + ": null mismatch");
            return diffs;
        }

        if (!expected.getClass().equals(actual.getClass())) {
            diffs.add(path + ": type mismatch");
            return diffs;
        }

        if (expected instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            Set<String> allKeys = new HashSet<>();
            allKeys.addAll(expMap.keySet());
            allKeys.addAll(actMap.keySet());

            for (String key : allKeys) {
                if (!expMap.containsKey(key)) {
                    diffs.add(path + "." + key + ": missing in Acorn");
                } else if (!actMap.containsKey(key)) {
                    diffs.add(path + "." + key + ": missing in Java");
                } else {
                    diffs.addAll(findDifferences(expMap.get(key), actMap.get(key), path + "." + key));
                }
            }
        } else if (expected instanceof List) {
            List<Object> expList = (List<Object>) expected;
            List<Object> actList = (List<Object>) actual;

            if (expList.size() != actList.size()) {
                diffs.add(path + ": length mismatch (" + expList.size() + " vs " + actList.size() + ")");
            }

            int len = Math.min(expList.size(), actList.size());
            for (int i = 0; i < len; i++) {
                diffs.addAll(findDifferences(expList.get(i), actList.get(i), path + "[" + i + "]"));
            }
        } else {
            if (!expected.equals(actual)) {
                diffs.add(path + ": " + expected + " != " + actual);
            }
        }

        return diffs;
    }
}
