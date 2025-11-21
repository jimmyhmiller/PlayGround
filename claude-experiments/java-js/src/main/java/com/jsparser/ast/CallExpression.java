package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record CallExpression(
    int start,
    int end,
    SourceLocation loc,
    Expression callee,
    List<Expression> arguments,
    boolean optional
) implements Expression {
    public CallExpression(Expression callee, List<Expression> arguments) {
        this(0, 0, null, callee, arguments, false);
    }

    @Override
    public String type() {
        return "CallExpression";
    }
}
