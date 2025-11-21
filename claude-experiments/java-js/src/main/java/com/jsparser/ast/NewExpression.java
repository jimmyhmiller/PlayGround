package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record NewExpression(
    int start,
    int end,
    SourceLocation loc,
    Expression callee,
    List<Expression> arguments
) implements Expression {
    public NewExpression(Expression callee, List<Expression> arguments) {
        this(0, 0, null, callee, arguments);
    }

    @Override
    public String type() {
        return "NewExpression";
    }
}
