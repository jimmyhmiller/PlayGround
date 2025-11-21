package com.jsparser;

import com.fasterxml.jackson.databind.*;
import org.junit.jupiter.api.Test;
import java.nio.file.*;

public class PositionDebugTest {
    @Test
    void testLineTerminators() throws Exception {
        String file = "test-oracles/test262/test/language/expressions/tagged-template/constructor-invocation.js";
        String source = Files.readString(Path.of(file));
        var prog = Parser.parse(source, false);

        ObjectMapper m = new ObjectMapper();
        m.enable(SerializationFeature.INDENT_OUTPUT);
        String actualJson = m.writeValueAsString(prog);

        String expectedJson = Files.readString(Path.of(file.replace("test262/test/", "test262-cache/") + ".json"));

        JsonNode expected = m.readTree(expectedJson);
        JsonNode actual = m.readTree(actualJson);

        // Find first difference
        compareNodes(expected, actual, "");
    }

    private void compareNodes(JsonNode exp, JsonNode act, String path) {
        if (exp.isObject() && act.isObject()) {
            var expFields = exp.fields();
            while (expFields.hasNext()) {
                var entry = expFields.next();
                String field = entry.getKey();
                JsonNode expValue = entry.getValue();
                JsonNode actValue = act.get(field);

                if (actValue == null) {
                    System.out.println("Missing field: " + path + "." + field);
                } else if (!expValue.equals(actValue)) {
                    if ((field.equals("line") || field.equals("column") || field.equals("start") || field.equals("end"))) {
                        System.out.println(path + "." + field + ": expected=" + expValue + ", actual=" + actValue);
                    } else {
                        compareNodes(expValue, actValue, path + "." + field);
                    }
                }
            }
        } else if (exp.isArray() && act.isArray()) {
            for (int i = 0; i < Math.min(exp.size(), act.size()); i++) {
                compareNodes(exp.get(i), act.get(i), path + "[" + i + "]");
            }
        } else if (!exp.equals(act)) {
            System.out.println(path + ": expected=" + exp + ", actual=" + act);
        }
    }
}
