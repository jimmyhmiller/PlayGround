package com.jsparser;

import org.junit.jupiter.api.Test;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class MinimalStringBugTest {

    @Test
    public void testExtractedFragment() throws IOException {
        // This is the extracted fragment from position 71558-72600 that causes the lexer to fail
        String source = Files.readString(Path.of("/tmp/minimal-string-bug.js"));

        System.out.println("Testing fragment of length: " + source.length());
        System.out.println("First 100 chars: " + source.substring(0, Math.min(100, source.length())));

        try {
            Lexer lexer = new Lexer(source);
            var tokens = lexer.tokenize();
            System.out.println("SUCCESS: Tokenized " + tokens.size() + " tokens");

            // Print first few tokens
            for (int i = 0; i < Math.min(10, tokens.size()); i++) {
                System.out.println("Token " + i + ": " + tokens.get(i));
            }
        } catch (Exception e) {
            System.out.println("FAILURE: " + e.getMessage());
            e.printStackTrace();
            fail("Should tokenize successfully: " + e.getMessage());
        }
    }

    @Test
    public void testFullFileFromStart() throws IOException {
        // Test the full file from the beginning to see where it fails
        String source = Files.readString(Path.of("../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/@vercel/nft/index.js"));

        try {
            Lexer lexer = new Lexer(source);
            var tokens = lexer.tokenize();
            System.out.println("SUCCESS: Full file tokenized " + tokens.size() + " tokens");
        } catch (Exception e) {
            System.out.println("FAILURE at: " + e.getMessage());
            // This is expected to fail
            assertTrue(e.getMessage().contains("Unterminated string") ||
                      e.getMessage().contains("Unexpected character"),
                      "Expected string tokenization error");
        }
    }
}
