package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record MemberExpression(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Expression object,
    Expression property,
    boolean computed,
    boolean optional
) implements Expression, Pattern {
    public MemberExpression(Expression object, Expression property, boolean computed) {
        this(0, 0, 0, 0, 0, 0, object, property, computed, false);
    }

    @JsonCreator
    public MemberExpression(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("object") Expression object,
        @JsonProperty("property") Expression property,
        @JsonProperty("computed") boolean computed,
        @JsonProperty("optional") boolean optional
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             object,
             property,
             computed,
             optional);
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
        return "MemberExpression";
    }
}
