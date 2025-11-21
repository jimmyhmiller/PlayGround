package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Literal(
    int start,
    int end,
    SourceLocation loc,
    @JsonSerialize(using = JavaScriptNumberSerializer.class)
    Object value,
    String raw,
    @JsonInclude(JsonInclude.Include.NON_NULL) RegexInfo regex,
    @JsonInclude(JsonInclude.Include.NON_NULL) String bigint
) implements Expression {
    public Literal(Object value, String raw) {
        this(0, 0, null, value, raw, null, null);
    }

    public Literal(int start, int end, SourceLocation loc, Object value, String raw) {
        this(start, end, loc, value, raw, null, null);
    }

    public Literal(int start, int end, SourceLocation loc, Object value, String raw, RegexInfo regex) {
        this(start, end, loc, value, raw, regex, null);
    }

    @Override
    public String type() {
        return "Literal";
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record RegexInfo(String pattern, String flags) {}
}
