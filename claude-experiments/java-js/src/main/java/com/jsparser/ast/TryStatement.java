package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonProperty;

public record TryStatement(
    int start,
    int end,
    SourceLocation loc,
    BlockStatement block,
    CatchClause handler,      // Can be null
    BlockStatement finalizer  // Can be null
) implements Statement {
    @Override
    public String type() {
        return "TryStatement";
    }
}
