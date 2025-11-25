package com.jsparser;

import java.nio.file.Files;
import java.nio.file.Paths;

public class ProfileLargeFile {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: ProfileLargeFile <file-path>");
            System.exit(1);
        }

        String filePath = args[0];
        System.out.println("Reading file: " + filePath);
        String source = Files.readString(Paths.get(filePath));
        System.out.println("File size: " + source.length() + " characters");

        System.out.println("\nParsing (warm-up run)...");
        var ast1 = Parser.parse(source, false);
        System.out.println("Warm-up complete. AST has " + ast1.body().size() + " statements");

        System.out.println("\nStarting profiled run...");
        System.out.println("(Attach YourKit profiler now and press Enter to continue)");
        System.in.read();

        long start = System.currentTimeMillis();
        var ast2 = Parser.parse(source, false);
        long elapsed = System.currentTimeMillis() - start;

        System.out.println("\nParsing complete!");
        System.out.println("Time: " + elapsed + "ms");
        System.out.println("AST has " + ast2.body().size() + " statements");

        System.out.println("\nPress Enter to exit (so you can capture the profile)...");
        System.in.read();
    }
}
