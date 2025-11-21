package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ObjectExpression(
    int start,
    int end,
    SourceLocation loc,
    List<Node> properties  // Can be Property or SpreadElement
) implements Expression {
    public ObjectExpression(List<Node> properties) {
        this(0, 0, null, properties);
    }

    @Override
    public String type() {
        return "ObjectExpression";
    }
}
