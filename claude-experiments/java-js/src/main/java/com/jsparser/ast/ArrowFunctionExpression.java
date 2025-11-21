package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ArrowFunctionExpression(
    int start,
    int end,
    SourceLocation loc,
    Identifier id,       // Always null for arrow functions
    boolean expression,  // true if body is expression, false if block
    boolean generator,   // Always false for arrows
    boolean async,
    List<Pattern> params,
    Node body            // Can be Expression or BlockStatement
) implements Expression {
    @Override
    public String type() {
        return "ArrowFunctionExpression";
    }
}
