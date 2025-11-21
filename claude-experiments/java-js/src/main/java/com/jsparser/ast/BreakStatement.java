package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record BreakStatement(
    int start,
    int end,
    SourceLocation loc,
    Identifier label  // Can be null for unlabeled break
) implements Statement {
    @Override
    public String type() {
        return "BreakStatement";
    }
}
