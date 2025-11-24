package com.jsparser;

import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DebugAxiosTest {

    @Test
    void debugAxiosHeadersFile() throws Exception {
        String source = Files.readString(
            Paths.get("temp-packages/axios/node_modules/axios/lib/core/AxiosHeaders.js")
        );

        System.out.println("File length: " + source.length());
        System.out.println("First 500 chars:\n" + source.substring(0, Math.min(500, source.length())));

        try {
            var program = Parser.parse(source, true); // It's a module (uses import)
            System.out.println("✓ Parsed successfully!");
            System.out.println("Number of statements: " + program.body().size());
        } catch (ParseException e) {
            System.err.println("✗ Parse failed:");
            System.err.println("Message: " + e.getMessage());
            System.err.println("Error type: " + e.getErrorType());

            Token token = e.getToken();
            if (token != null) {
                System.err.println("Token: " + token);
                System.err.println("Line: " + token.line());
                System.err.println("Column: " + token.column());

                // Print context around the error
                String[] lines = source.split("\n");
                int errorLine = token.line() - 1;

                System.err.println("\nContext around error:");
                for (int i = Math.max(0, errorLine - 3); i < Math.min(lines.length, errorLine + 3); i++) {
                    String marker = (i == errorLine) ? ">>> " : "    ";
                    System.err.println(marker + (i + 1) + ": " + lines[i]);
                }
            }

            throw e;
        }
    }
}
