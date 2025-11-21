package com.jsparser.ast;

import java.util.List;
import com.fasterxml.jackson.annotation.JsonProperty;

public record MethodDefinition(
    int start,
    int end,
    SourceLocation loc,
    Expression key,        // Property name (Identifier or PrivateIdentifier)
    FunctionExpression value,
    String kind,          // "constructor" | "method" | "get" | "set"
    boolean computed,
    @JsonProperty("static") boolean isStatic
) implements Node {
    @Override
    public String type() {
        return "MethodDefinition";
    }
}
