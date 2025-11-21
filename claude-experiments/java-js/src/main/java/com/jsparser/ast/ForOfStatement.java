package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ForOfStatement(
    int start,
    int end,
    SourceLocation loc,
    boolean await,
    Node left,   // VariableDeclaration or Expression
    Expression right,
    Statement body
) implements Statement {
    @Override
    public String type() {
        return "ForOfStatement";
    }
}
