package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonInclude;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record ExpressionStatement(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Expression expression,
    @JsonInclude(JsonInclude.Include.NON_NULL) String directive
) implements Statement {
    public ExpressionStatement(Expression expression) {
        this(0, 0, 0, 0, 0, 0, expression, null);
    }

    public ExpressionStatement(int start, int end, int startLine, int startCol, int endLine, int endCol, Expression expression) {
        this(start, end, startLine, startCol, endLine, endCol, expression, null);
    }

    @JsonCreator
    public ExpressionStatement(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("expression") Expression expression,
        @JsonProperty("directive") String directive
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             expression,
             directive);
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
        return "ExpressionStatement";
    }
}
