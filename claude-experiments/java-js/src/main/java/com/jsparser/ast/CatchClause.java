package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record CatchClause(
    int start,
    int end,
    SourceLocation loc,
    Pattern param,           // The exception parameter (can be null in ES2019+)
    BlockStatement body
) implements Node {
    @Override
    public String type() {
        return "CatchClause";
    }
}
