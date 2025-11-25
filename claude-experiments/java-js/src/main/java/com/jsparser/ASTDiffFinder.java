package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Find specific differences between Java parser and Acorn ASTs
 */
public class ASTDiffFinder {
    private static final ObjectMapper mapper = new ObjectMapper();

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Usage: java ASTDiffFinder <file-path>");
            System.exit(1);
        }

        String filePath = args[0];
        String source = Files.readString(Paths.get(filePath));

        // Parse with Acorn
        String acornJson = parseWithAcorn(source, filePath);
        if (acornJson == null) {
            System.err.println("Failed to parse with Acorn");
            System.exit(1);
        }

        // Parse with Java
        Program javaAst;
        try {
            javaAst = Parser.parse(source, true);
        } catch (Exception e) {
            try {
                javaAst = Parser.parse(source, false);
            } catch (Exception e2) {
                System.err.println("Failed to parse with Java parser: " + e2.getMessage());
                System.exit(1);
                return;
            }
        }

        String javaJson = mapper.writeValueAsString(javaAst);

        // Parse JSONs
        @SuppressWarnings("unchecked")
        Map<String, Object> acornMap = mapper.readValue(acornJson, Map.class);
        @SuppressWarnings("unchecked")
        Map<String, Object> javaMap = mapper.readValue(javaJson, Map.class);

        // Find differences
        List<String> diffs = new ArrayList<>();
        findDifferences("", acornMap, javaMap, diffs, 0, 10); // max 10 differences

        if (diffs.isEmpty()) {
            System.out.println("✓ ASTs match!");
        } else {
            System.out.println("✗ Found " + diffs.size() + " differences:\n");
            for (int i = 0; i < diffs.size(); i++) {
                System.out.println((i + 1) + ". " + diffs.get(i));
                System.out.println();
            }
        }
    }

    private static void findDifferences(String path, Object acorn, Object java, List<String> diffs, int depth, int maxDiffs) {
        if (diffs.size() >= maxDiffs || depth > 50) return;

        if (acorn == null && java == null) return;
        if (acorn == null || java == null) {
            diffs.add("At " + path + ":\n  Acorn: " + acorn + "\n  Java:  " + java);
            return;
        }

        if (acorn.getClass() != java.getClass()) {
            diffs.add("At " + path + " (type mismatch):\n  Acorn: " + acorn.getClass().getSimpleName() + " = " + acorn + "\n  Java:  " + java.getClass().getSimpleName() + " = " + java);
            return;
        }

        if (acorn instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> acornMap = (Map<String, Object>) acorn;
            @SuppressWarnings("unchecked")
            Map<String, Object> javaMap = (Map<String, Object>) java;

            // Check for keys only in Acorn
            for (String key : acornMap.keySet()) {
                if (!javaMap.containsKey(key)) {
                    diffs.add("At " + path + "." + key + ":\n  Present in Acorn but missing in Java\n  Acorn value: " + acornMap.get(key));
                    if (diffs.size() >= maxDiffs) return;
                }
            }

            // Check for keys only in Java
            for (String key : javaMap.keySet()) {
                if (!acornMap.containsKey(key)) {
                    diffs.add("At " + path + "." + key + ":\n  Present in Java but missing in Acorn\n  Java value: " + javaMap.get(key));
                    if (diffs.size() >= maxDiffs) return;
                }
            }

            // Check for differences in common keys
            for (String key : acornMap.keySet()) {
                if (javaMap.containsKey(key)) {
                    String newPath = path.isEmpty() ? key : path + "." + key;
                    findDifferences(newPath, acornMap.get(key), javaMap.get(key), diffs, depth + 1, maxDiffs);
                    if (diffs.size() >= maxDiffs) return;
                }
            }
        } else if (acorn instanceof List) {
            @SuppressWarnings("unchecked")
            List<Object> acornList = (List<Object>) acorn;
            @SuppressWarnings("unchecked")
            List<Object> javaList = (List<Object>) java;

            if (acornList.size() != javaList.size()) {
                diffs.add("At " + path + " (array length mismatch):\n  Acorn length: " + acornList.size() + "\n  Java length:  " + javaList.size());
                return;
            }

            for (int i = 0; i < acornList.size(); i++) {
                findDifferences(path + "[" + i + "]", acornList.get(i), javaList.get(i), diffs, depth + 1, maxDiffs);
                if (diffs.size() >= maxDiffs) return;
            }
        } else if (!acorn.equals(java)) {
            // Special handling for numeric comparisons across types
            if (acorn instanceof Number && java instanceof Number) {
                double acornNum = ((Number) acorn).doubleValue();
                double javaNum = ((Number) java).doubleValue();
                if (Math.abs(acornNum - javaNum) < 1e-9) {
                    // Numbers are equal within tolerance
                    return;
                }
            }
            diffs.add("At " + path + ":\n  Acorn: " + acorn + "\n  Java:  " + java);
        }
    }

    private static String parseWithAcorn(String source, String filePath) {
        try {
            ProcessBuilder pb = new ProcessBuilder("node", "scripts/parse-with-acorn.js", filePath);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                return output.toString();
            }
            return null;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
