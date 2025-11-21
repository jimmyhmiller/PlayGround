package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ReturnStatement(
    int start,
    int end,
    SourceLocation loc,
    Expression argument  // Can be null for "return;"
) implements Statement {
    @Override
    public String type() {
        return "ReturnStatement";
    }
}
