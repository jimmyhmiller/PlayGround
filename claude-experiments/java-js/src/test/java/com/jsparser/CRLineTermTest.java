package com.jsparser;

import org.junit.jupiter.api.Test;
import java.nio.file.*;

public class CRLineTermTest {
    @Test
    void test() throws Exception {
        String source = Files.readString(Path.of("test-oracles/test262/test/built-ins/Function/prototype/toString/line-terminator-normalisation-CR.js"));
        System.out.println("Source length: " + source.length());
        System.out.println("Contains \\r: " + source.contains("\r"));
        System.out.println("Contains \\n: " + source.contains("\n"));
        
        var prog = Parser.parse(source);
        System.out.println("Body count: " + prog.body().size());
        for (var stmt : prog.body()) {
            System.out.println("  " + stmt.getClass().getSimpleName());
        }
    }
}
