package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record YieldExpression(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    boolean delegate,  // true for yield*, false for yield
    Expression argument  // null for standalone yield
) implements Expression {
    public YieldExpression(boolean delegate, Expression argument) {
        this(0, 0, 0, 0, 0, 0, delegate, argument);
    }

    @JsonCreator
    public YieldExpression(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("delegate") boolean delegate,
        @JsonProperty("argument") Expression argument
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             delegate,
             argument);
    }

    @Override
    @JsonProperty("loc")
    public SourceLocation loc() {
        return new SourceLocation(
            new SourceLocation.Position(startLine, startCol),
            new SourceLocation.Position(endLine, endCol)
        );
    }

    @Override
    public String type() {
        return "YieldExpression";
    }
}
