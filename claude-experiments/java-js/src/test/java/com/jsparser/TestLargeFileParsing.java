package com.jsparser;

import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Paths;

public class TestLargeFileParsing {
    @Test
    public void testLargeFile() throws Exception {
        String file = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/simple-nextjs-demo/simple-nextjs-demo/node_modules/styled-jsx/dist/babel/index.js";
        System.out.println("Reading file: " + file);
        String source = Files.readString(Paths.get(file));
        System.out.println("File size: " + source.length() + " characters");

        System.out.println("Parsing with Java parser...");
        long start = System.currentTimeMillis();
        var ast = Parser.parse(source, false);
        long elapsed = System.currentTimeMillis() - start;
        System.out.println("Parse successful! Took " + elapsed + "ms");
        System.out.println("AST has " + ast.body().size() + " top-level statements");
    }
}
