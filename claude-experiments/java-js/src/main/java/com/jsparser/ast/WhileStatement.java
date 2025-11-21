package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record WhileStatement(
    int start,
    int end,
    SourceLocation loc,
    Expression test,
    Statement body
) implements Statement {
    @Override
    public String type() {
        return "WhileStatement";
    }
}
