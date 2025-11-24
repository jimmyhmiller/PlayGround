package com.jsparser;

import org.junit.jupiter.api.Test;

public class DebugBigIntParserTest {
    @Test
    public void testBigIntParsing() {
        String source = "let { 1n: a } = { \"1\": \"foo\" };";
        Parser parser = new Parser(source);

        System.out.println("Testing BigInt in destructuring pattern:");
        System.out.println("Source: " + source);

        try {
            var ast = parser.parse();
            System.out.println("Parsed successfully");
        } catch (Exception e) {
            System.out.println("Parse error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
