package com.jsparser.ast;

import java.util.List;

public record ClassBody(
    int start,
    int end,
    SourceLocation loc,
    List<Node> body  // List of MethodDefinition or PropertyDefinition
) implements Node {
    @Override
    public String type() {
        return "ClassBody";
    }
}
