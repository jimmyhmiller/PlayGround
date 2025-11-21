package com.jsparser;

import com.fasterxml.jackson.databind.*;
import org.junit.jupiter.api.Test;
import java.nio.file.*;

public class ExportDfltGenTest {
    @Test
    void test() throws Exception {
        String source = Files.readString(Path.of("test-oracles/test262/test/language/module-code/eval-export-dflt-gen-anon-semi.js"));
        var prog = Parser.parse(source, true); // module mode
        
        var body1 = prog.body().get(1);
        System.out.println("body[1].type: " + body1.type());
        if (body1 instanceof com.jsparser.ast.ExportDefaultDeclaration ed) {
            System.out.println("body[1].declaration.type: " + ed.declaration().type());
            System.out.println("body[1].declaration class: " + ed.declaration().getClass().getSimpleName());
        }
    }
}
