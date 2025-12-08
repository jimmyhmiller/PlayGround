package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record UpdateExpression(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    String operator,  // "++" | "--"
    boolean prefix,   // true for ++x, false for x++
    Expression argument
) implements Expression {
    @JsonCreator
    public UpdateExpression(
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
        return "UpdateExpression";
    }
}
