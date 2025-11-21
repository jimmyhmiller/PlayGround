package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonProperty;

public record PropertyDefinition(
    int start,
    int end,
    SourceLocation loc,
    Expression key,        // Property name (Identifier or PrivateIdentifier)
    Expression value,      // Can be null for class fields without initializer
    boolean computed,
    @JsonProperty("static") boolean isStatic
) implements Node {
    @Override
    public String type() {
        return "PropertyDefinition";
    }
}
