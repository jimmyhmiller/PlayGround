package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WithStatement(
    int start,
    int end,
    SourceLocation loc,
    Expression object,
    Statement body
) implements Statement {
    @Override
    public String type() {
        return "WithStatement";
    }
}
