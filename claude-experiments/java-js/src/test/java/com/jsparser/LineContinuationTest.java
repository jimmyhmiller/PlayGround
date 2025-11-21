package com.jsparser;

import com.fasterxml.jackson.databind.*;
import org.junit.jupiter.api.Test;
import java.nio.file.*;

public class LineContinuationTest {
    @Test
    void test() throws Exception {
        String source = Files.readString(Path.of("test-oracles/test262/test/built-ins/String/prototype/trim/15.5.4.20-4-59.js"));
        var prog = Parser.parse(source);
        ObjectMapper m = new ObjectMapper();
        m.enable(SerializationFeature.INDENT_OUTPUT);
        
        // body[1] is assert.sameValue(...)
        var stmt = prog.body().get(1);
        System.out.println("Our loc: " + m.writeValueAsString(stmt).substring(0, 200));
        
        // Expected
        var expected = m.readTree(Files.readString(Path.of("test-oracles/test262-cache/built-ins/String/prototype/trim/15.5.4.20-4-59.js.json")));
        System.out.println("Expected loc: " + expected.get("body").get(1).get("loc"));
    }
}
