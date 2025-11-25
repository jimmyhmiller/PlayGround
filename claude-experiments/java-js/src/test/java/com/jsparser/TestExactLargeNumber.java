package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class TestExactLargeNumber {

    @Test
    public void testExactNumber() throws Exception {
        String code = "var x = 9223372036854776000;";
        Program ast = Parser.parse(code, false);

        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(ast);

        System.out.println("=== Java Parser Output ===");
        System.out.println(json);
        System.out.println();

        if (json.contains("9223372036854776000")) {
            System.out.println("✓ Contains exact number");
        } else if (json.contains("9223372036854775807")) {
            System.out.println("✗ WRONG: Truncated to Long.MAX_VALUE");
        } else if (json.contains("9.223372036854776E18")) {
            System.out.println("✓ Stored as double (acceptable)");
        } else {
            System.out.println("? Unknown representation");
        }
    }
}
