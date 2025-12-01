package com.jsparser;

import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Path;

public class TestExactFailingFile {

    @Test
    public void testFullFile() throws Exception {
        // Test the actual failing file
        String source = Files.readString(Path.of("../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/@vercel/nft/index.js"));

        System.out.println("File length: " + source.length());

        try {
            Lexer lexer = new Lexer(source);
            var tokens = lexer.tokenize();
            System.out.println("SUCCESS: Tokenized " + tokens.size() + " tokens");
        } catch (Exception e) {
            System.out.println("FAILED: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }
}
