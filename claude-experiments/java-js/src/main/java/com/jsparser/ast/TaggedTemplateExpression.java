package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record TaggedTemplateExpression(
    int start,
    int end,
    SourceLocation loc,
    Expression tag,
    TemplateLiteral quasi
) implements Expression {
    @Override
    public String type() {
        return "TaggedTemplateExpression";
    }
}
