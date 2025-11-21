package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record TemplateLiteral(
    int start,
    int end,
    SourceLocation loc,
    List<Expression> expressions,
    List<TemplateElement> quasis
) implements Expression {
    @Override
    public String type() {
        return "TemplateLiteral";
    }
}
