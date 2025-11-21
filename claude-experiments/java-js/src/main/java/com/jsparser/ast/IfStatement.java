package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record IfStatement(
    int start,
    int end,
    SourceLocation loc,
    Expression test,
    Statement consequent,
    Statement alternate  // Can be null
) implements Statement {
    @Override
    public String type() {
        return "IfStatement";
    }
}
