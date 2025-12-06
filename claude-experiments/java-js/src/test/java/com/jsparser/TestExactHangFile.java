package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

public class TestExactHangFile {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void testS15_10_5_1_A1_with_normalization() throws Exception {
        Path sourceFile = Paths.get("test-oracles/test262/test/built-ins/RegExp/prototype/S15.10.5.1_A1.js");
        Path cacheFile = Paths.get("test-oracles/test262-cache/built-ins/RegExp/prototype/S15.10.5.1_A1.js.json");

        String source = Files.readString(sourceFile);
        String expectedJson = Files.readString(cacheFile);

        System.out.println("Parsing...");
        Program actualProgram = Parser.parse(source, false);
        String actualJson = mapper.writeValueAsString(actualProgram);

        System.out.println("Reading JSON objects...");
        Object expectedObj = mapper.readValue(expectedJson, Object.class);
        Object actualObj = mapper.readValue(actualJson, Object.class);

        if (expectedObj instanceof Map) {
            ((Map<?, ?>)expectedObj).remove("_metadata");
        }

        System.out.println("Normalizing regex values...");
        normalizeRegexValues(expectedObj, actualObj);

        System.out.println("Normalizing bigint values...");
        normalizeBigIntValues(expectedObj, actualObj);

        System.out.println("Comparing...");
        boolean matches = Objects.deepEquals(expectedObj, actualObj);

        System.out.println("Result: " + (matches ? "MATCH" : "MISMATCH"));
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
        } else if (expected instanceof java.util.List && actual instanceof java.util.List) {
            java.util.List<Object> expList = (java.util.List<Object>) expected;
            java.util.List<Object> actList = (java.util.List<Object>) actual;
            for (int i = 0; i < Math.min(expList.size(), actList.size()); i++) {
                normalizeRegexValues(expList.get(i), actList.get(i));
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void normalizeBigIntValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            if ("Literal".equals(expMap.get("type")) && expMap.containsKey("bigint")) {
                Object expBigint = expMap.get("bigint");
                Object actBigint = actMap.get("bigint");

                if (expBigint != null && actBigint != null) {
                    String expStr = expBigint.toString();
                    String actStr = actBigint.toString();

                    if (!expStr.equals(actStr)) {
                        try {
                            java.math.BigInteger expBI = new java.math.BigInteger(expStr);
                            java.math.BigInteger actBI = new java.math.BigInteger(actStr);
                            if (expBI.equals(actBI)) {
                                actMap.put("bigint", expStr);
                            }
                        } catch (NumberFormatException e) {
                            // If can't parse, leave as is
                        }
                    }
                }
            }

            for (String key : expMap.keySet()) {
                if (actMap.containsKey(key)) {
                    normalizeBigIntValues(expMap.get(key), actMap.get(key));
                }
            }
        } else if (expected instanceof java.util.List && actual instanceof java.util.List) {
            java.util.List<Object> expList = (java.util.List<Object>) expected;
            java.util.List<Object> actList = (java.util.List<Object>) actual;
            for (int i = 0; i < Math.min(expList.size(), actList.size()); i++) {
                normalizeBigIntValues(expList.get(i), actList.get(i));
            }
        }
    }
}
