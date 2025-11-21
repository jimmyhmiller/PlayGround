package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record FunctionExpression(
    int start,
    int end,
    SourceLocation loc,
    Identifier id,           // Can be null for anonymous functions
    boolean expression,
    boolean generator,
    boolean async,
    List<Pattern> params,
    BlockStatement body
) implements Expression {
    @Override
    public String type() {
        return "FunctionExpression";
    }
}
