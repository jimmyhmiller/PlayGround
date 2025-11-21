package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record FunctionDeclaration(
    int start,
    int end,
    SourceLocation loc,
    Identifier id,
    boolean expression,
    boolean generator,
    boolean async,
    List<Pattern> params,
    BlockStatement body
) implements Statement {
    @Override
    public String type() {
        return "FunctionDeclaration";
    }
}
