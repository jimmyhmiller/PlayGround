package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record UnaryExpression(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    String operator,  // "!", "-", "+", "~", "typeof", "void", "delete"
    boolean prefix,   // Always true for prefix unary operators
    Expression argument
) implements Expression {
    // Constructor with prefix defaulting to true
    public UnaryExpression(int startLine, int startCol, int endLine, int endCol, String operator, Expression argument) {
        this(0, 0, startLine, startCol, endLine, endCol, operator, true, argument);
    }

    @JsonCreator
    public UnaryExpression(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("operator") String operator,
        @JsonProperty("prefix") boolean prefix,
        @JsonProperty("argument") Expression argument
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             operator,
             prefix,
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
        return "UnaryExpression";
    }
}
