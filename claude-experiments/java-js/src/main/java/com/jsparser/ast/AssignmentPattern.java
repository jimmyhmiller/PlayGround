package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record AssignmentPattern(
    int start,
    int end,
    SourceLocation loc,
    Pattern left,
    Expression right
) implements Pattern {
    @Override
    public String type() {
        return "AssignmentPattern";
    }
}
