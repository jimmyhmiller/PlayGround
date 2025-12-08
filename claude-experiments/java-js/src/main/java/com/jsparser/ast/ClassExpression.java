package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record ClassExpression(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Identifier id,         // Class name (can be null for anonymous class)
    Expression superClass, // Can be null if no extends
    ClassBody body
) implements Expression {
    @JsonCreator
    public ClassExpression(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("id") Identifier id,
        @JsonProperty("superClass") Expression superClass,
        @JsonProperty("body") ClassBody body
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             id,
             superClass,
             body);
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
        return "ClassExpression";
    }
}
