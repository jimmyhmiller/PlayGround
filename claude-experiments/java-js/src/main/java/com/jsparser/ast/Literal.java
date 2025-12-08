package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record Literal(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    @JsonSerialize(using = JavaScriptNumberSerializer.class)
    Object value,
    String raw,
    @JsonInclude(JsonInclude.Include.NON_NULL) RegexInfo regex,
    @JsonInclude(JsonInclude.Include.NON_NULL) String bigint
) implements Expression {
    public Literal(Object value, String raw) {
        this(0, 0, 0, 0, 0, 0, value, raw, null, null);
    }

    public Literal(int start, int end, int startLine, int startCol, int endLine, int endCol, Object value, String raw) {
        this(start, end, startLine, startCol, endLine, endCol, value, raw, null, null);
    }

    public Literal(int start, int end, int startLine, int startCol, int endLine, int endCol, Object value, String raw, RegexInfo regex) {
        this(start, end, startLine, startCol, endLine, endCol, value, raw, regex, null);
    }

    @JsonCreator
    public Literal(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("value") Object value,
        @JsonProperty("raw") String raw,
        @JsonProperty("regex") RegexInfo regex,
        @JsonProperty("bigint") String bigint
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             value,
             raw,
             regex,
             bigint);
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
        return "Literal";
    }

    @JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
    public record RegexInfo(String pattern, String flags) {}
}
