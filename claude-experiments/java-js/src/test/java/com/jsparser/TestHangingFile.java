package com.jsparser;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

public class TestHangingFile {

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void testS15_10_5_1_A2() throws Exception {
        Path file = Paths.get("test-oracles/test262/test/built-ins/RegExp/prototype/S15.10.5.1_A2.js");
        String source = Files.readString(file);

        System.out.println("Parsing file: " + file.getFileName());
        System.out.println("File size: " + source.length() + " chars");

        long start = System.currentTimeMillis();
        Parser.parse(source, false);
        long end = System.currentTimeMillis();

        System.out.println("✓ Parsed successfully in " + (end - start) + "ms");
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void testS15_10_5_1_A3() throws Exception {
        Path file = Paths.get("test-oracles/test262/test/built-ins/RegExp/prototype/S15.10.5.1_A3.js");
        String source = Files.readString(file);

        System.out.println("Parsing file: " + file.getFileName());
        System.out.println("File size: " + source.length() + " chars");

        long start = System.currentTimeMillis();
        Parser.parse(source, false);
        long end = System.currentTimeMillis();

        System.out.println("✓ Parsed successfully in " + (end - start) + "ms");
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void testS15_10_5_1_A4() throws Exception {
        Path file = Paths.get("test-oracles/test262/test/built-ins/RegExp/prototype/S15.10.5.1_A4.js");
        String source = Files.readString(file);

        System.out.println("Parsing file: " + file.getFileName());
        System.out.println("File size: " + source.length() + " chars");

        long start = System.currentTimeMillis();
        Parser.parse(source, false);
        long end = System.currentTimeMillis();

        System.out.println("✓ Parsed successfully in " + (end - start) + "ms");
    }
}
