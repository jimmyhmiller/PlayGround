package com.jsparser;

import com.fasterxml.jackson.databind.*;
import org.junit.jupiter.api.Test;

public class RegexValueTest {
    @Test
    void test() throws Exception {
        ObjectMapper m = new ObjectMapper();
        var prog = Parser.parse("/foo/g");
        String json = m.writeValueAsString(prog);
        System.out.println(json);
    }
}
