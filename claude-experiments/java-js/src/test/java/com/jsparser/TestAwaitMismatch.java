package com.jsparser;

import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Path;

public class TestAwaitMismatch {

    @Test
    public void testFileAsScript() throws Exception {
        Path filePath = Path.of("test-oracles/test262/test/staging/sm/fields/await-identifier-module-2.js");
        String source = Files.readString(filePath);

        // Extract just the code (skip frontmatter)
        String code = source.substring(source.lastIndexOf("---*/") + 5).trim();

        System.out.println("Code: " + code);
        System.out.println("\nParsing as SCRIPT (isModule=false):");

        try {
            Program ast = Parser.parse(code, false);  // script context
            System.out.println("SUCCESS as script");
            System.out.println("AST: " + ast);
        } catch (Exception e) {
            System.out.println("FAILED as script: " + e.getMessage());
        }

        System.out.println("\nParsing as MODULE (isModule=true):");

        try {
            Program ast = Parser.parse(code, true);  // module context
            System.out.println("SUCCESS as module (INCORRECT - should fail!)");
            System.out.println("AST: " + ast);
        } catch (Exception e) {
            System.out.println("FAILED as module (CORRECT): " + e.getMessage());
        }
    }
}
