package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonProperty;

public record AwaitExpression(
    @JsonProperty("start") int start,
    @JsonProperty("end") int end,
    @JsonProperty("loc") SourceLocation loc,
    @JsonProperty("argument") Expression argument
) implements Expression {
    @Override
    @JsonProperty("type")
    public String type() {
        return "AwaitExpression";
    }
}
