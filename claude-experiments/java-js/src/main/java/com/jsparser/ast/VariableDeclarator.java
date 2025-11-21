package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record VariableDeclarator(
    int start,
    int end,
    SourceLocation loc,
    Pattern id,
    Expression init     // Can be null
) {
    @JsonProperty(value = "type", index = 0)
    public String type() {
        return "VariableDeclarator";
    }
}
