package com.jsparser;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

public class TestRegexpModifiers {

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void testRegexpModifiersFile() throws Exception {
        Path file = Paths.get("test-oracles/test262/test/built-ins/RegExp/regexp-modifiers/remove-ignoreCase-does-not-affect-ignoreCase-property.js");
        String source = Files.readString(file);

        System.out.println("File content (first 500 chars):");
        System.out.println(source.substring(0, Math.min(500, source.length())));

        System.out.println("\nParsing...");
        try {
            Parser.parse(source, false);
            System.out.println("✓ Parsed successfully!");
        } catch (Exception e) {
            System.out.println("✗ Parse failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void testSimpleRegexpModifier() throws Exception {
        String source = "var re1 = /(?-i:)/i;";

        System.out.println("Parsing: " + source);
        try {
            Parser.parse(source, false);
            System.out.println("✓ Parsed successfully!");
        } catch (Exception e) {
            System.out.println("✗ Parse failed: " + e.getMessage());
            throw e;
        }
    }
}
