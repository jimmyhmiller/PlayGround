package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record SpreadElement(
    int start,
    int end,
    SourceLocation loc,
    Expression argument
) implements Expression {
    @Override
    public String type() {
        return "SpreadElement";
    }
}
