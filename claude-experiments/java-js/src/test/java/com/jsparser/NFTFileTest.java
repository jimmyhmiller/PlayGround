package com.jsparser;

import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Test parsing the @vercel/nft/index.js file that's currently failing
 */
public class NFTFileTest {

    @Test
    void testNFTFile() throws Exception {
        String file = "../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/@vercel/nft/index.js";
        String source = Files.readString(Paths.get(file));

        System.out.println("File size: " + source.length());
        System.out.println("Attempting to parse...");

        try {
            Parser.parse(source, false);
            System.out.println("✓ Parsed successfully!");
        } catch (ParseException e) {
            System.out.println("✗ Parse error: " + e.getMessage());
            int position = e.getToken().position();
            int line = e.getToken().line();
            int column = e.getToken().column();
            System.out.println("Position: " + position);
            System.out.println("Line: " + line);
            System.out.println("Column: " + column);

            // Show context around the error
            if (position > 0 && position < source.length()) {
                int start = Math.max(0, position - 50);
                int end = Math.min(source.length(), position + 50);
                String context = source.substring(start, end);
                System.out.println("\nContext:");
                System.out.println(context.replace("\n", "\\n").replace("\r", "\\r"));
                System.out.println(" ".repeat(Math.min(50, position - start)) + "^");
            }

            throw e;
        }
    }
}
