package com.jsparser.ast;

import java.util.List;

public record ArrayPattern(
    int start,
    int end,
    SourceLocation loc,
    List<Pattern> elements
) implements Pattern {
    @Override
    public String type() {
        return "ArrayPattern";
    }
}
