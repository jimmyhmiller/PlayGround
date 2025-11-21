package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ArrayExpression(
    int start,
    int end,
    SourceLocation loc,
    List<Expression> elements
) implements Expression {
    public ArrayExpression(List<Expression> elements) {
        this(0, 0, null, elements);
    }

    @Override
    public String type() {
        return "ArrayExpression";
    }
}
