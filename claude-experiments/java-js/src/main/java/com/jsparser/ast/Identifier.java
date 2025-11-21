package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Identifier(
    int start,
    int end,
    SourceLocation loc,
    String name
) implements Expression, Pattern {
    public Identifier(String name) {
        this(0, 0, null, name);
    }

    @Override
    public String type() {
        return "Identifier";
    }
}
