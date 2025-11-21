package com.jsparser;

import com.fasterxml.jackson.databind.*;
import org.junit.jupiter.api.Test;
import java.nio.file.*;

public class EscapedLetTest {
    @Test
    void test() throws Exception {
        String source = Files.readString(Path.of("test-oracles/test262/test/language/statements/let/syntax/escaped-let.js"));
        var prog = Parser.parse(source);
        ObjectMapper m = new ObjectMapper();
        
        System.out.println("body[1].type: " + prog.body().get(1).getClass().getSimpleName());
        System.out.println("body[1]:");
        System.out.println(m.writeValueAsString(prog.body().get(1)));
    }
}
