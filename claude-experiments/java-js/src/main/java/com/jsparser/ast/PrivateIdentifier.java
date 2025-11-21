package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record PrivateIdentifier(
    int start,
    int end,
    SourceLocation loc,
    String name  // Name without the # prefix
) implements Expression {
    @Override
    public String type() {
        return "PrivateIdentifier";
    }
}
