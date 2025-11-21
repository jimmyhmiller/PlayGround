package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record DoWhileStatement(
    int start,
    int end,
    SourceLocation loc,
    Statement body,
    Expression test
) implements Statement {
    @Override
    public String type() {
        return "DoWhileStatement";
    }
}
