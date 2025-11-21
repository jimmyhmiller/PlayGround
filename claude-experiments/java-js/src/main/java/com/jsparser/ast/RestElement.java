package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record RestElement(
    int start,
    int end,
    SourceLocation loc,
    Pattern argument
) implements Pattern {
    @Override
    public String type() {
        return "RestElement";
    }
}
