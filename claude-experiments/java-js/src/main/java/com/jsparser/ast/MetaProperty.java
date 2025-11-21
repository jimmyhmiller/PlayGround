package com.jsparser.ast;

public record MetaProperty(
    int start,
    int end,
    SourceLocation loc,
    Identifier meta,     // 'new' or 'import'
    Identifier property  // 'target' or 'meta'
) implements Expression {
    @Override
    public String type() {
        return "MetaProperty";
    }
}
