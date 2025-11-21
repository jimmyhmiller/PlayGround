package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestPrivate {
    @Test
    void testPrivateMemberAccess() throws Exception {
        String source = "obj.#x;";
        Parser.parse(source);
        System.out.println("✓ Private member access works");
    }
}
