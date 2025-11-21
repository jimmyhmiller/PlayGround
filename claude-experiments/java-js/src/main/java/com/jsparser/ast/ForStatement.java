package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ForStatement(
    int start,
    int end,
    SourceLocation loc,
    Node init,        // Can be VariableDeclaration | Expression | null
    Expression test,  // Can be null
    Expression update, // Can be null
    Statement body
) implements Statement {
    @Override
    public String type() {
        return "ForStatement";
    }
}
