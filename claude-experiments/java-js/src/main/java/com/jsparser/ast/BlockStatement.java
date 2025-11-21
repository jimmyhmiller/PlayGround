package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record BlockStatement(
    int start,
    int end,
    SourceLocation loc,
    List<Statement> body
) implements Statement {
    @Override
    public String type() {
        return "BlockStatement";
    }
}
