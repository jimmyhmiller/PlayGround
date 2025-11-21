package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ContinueStatement(
    int start,
    int end,
    SourceLocation loc,
    Identifier label  // Can be null for unlabeled continue
) implements Statement {
    @Override
    public String type() {
        return "ContinueStatement";
    }
}
