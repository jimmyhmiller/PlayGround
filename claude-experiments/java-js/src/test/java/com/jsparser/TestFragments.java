package com.jsparser;

import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Path;

public class TestFragments {

    @Test
    public void testFragments() throws Exception {
        String[] sizes = {"500", "1000", "2000", "5000", "10000", "20000"};

        for (String size : sizes) {
            String filename = "/tmp/fragment-" + size + ".js";
            System.out.println("=== Testing fragment-" + size + ".js ===");
            try {
                String source = Files.readString(Path.of(filename));
                Lexer lexer = new Lexer(source);
                var tokens = lexer.tokenize();
                System.out.println("SUCCESS: Tokenized " + tokens.size() + " tokens");
            } catch (Exception e) {
                System.out.println("FAILURE: " + e.getMessage());
            }
        }
    }
}
