package com.jsparser;

import com.fasterxml.jackson.databind.*;
import org.junit.jupiter.api.Test;
import java.nio.file.*;

public class AwaitGlobalTest {
    @Test
    void test() throws Exception {
        String source = Files.readString(Path.of("test-oracles/test262/test/language/expressions/await/await-in-global.js"));
        var prog = Parser.parse(source);
        ObjectMapper m = new ObjectMapper();
        m.enable(SerializationFeature.INDENT_OUTPUT);
        
        var body1 = prog.body().get(1);
        System.out.println("body[1]:");
        System.out.println(m.writeValueAsString(body1));
    }
}
