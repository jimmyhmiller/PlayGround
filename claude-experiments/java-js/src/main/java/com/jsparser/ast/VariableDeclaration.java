package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record VariableDeclaration(
    int start,
    int end,
    SourceLocation loc,
    List<VariableDeclarator> declarations,
    String kind  // "var" | "let" | "const"
) implements Statement {
    @Override
    public String type() {
        return "VariableDeclaration";
    }
}
