package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record TemplateElement(
    int start,
    int end,
    SourceLocation loc,
    TemplateElementValue value,
    boolean tail
) implements Node {
    @Override
    public String type() {
        return "TemplateElement";
    }

    public record TemplateElementValue(
        String raw,
        String cooked
    ) {}
}
