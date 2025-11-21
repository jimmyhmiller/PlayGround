package com.jsparser.ast;

import java.util.List;

public record ObjectPattern(
    int start,
    int end,
    SourceLocation loc,
    List<Node> properties  // Can be Property or RestElement
) implements Pattern {
    @Override
    public String type() {
        return "ObjectPattern";
    }
}
