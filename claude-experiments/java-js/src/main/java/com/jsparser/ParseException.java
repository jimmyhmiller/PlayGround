package com.jsparser;

import java.util.HashMap;
import java.util.Map;

/**
 * Base exception for all parsing errors.
 * Contains structured data that can be serialized to JSON for analysis.
 */
public class ParseException extends RuntimeException {
    private final String errorType;
    private final Token token;
    private final String expected;
    private final String context;

    public ParseException(String errorType, Token token, String expected, String context, String message) {
        super(message);
        this.errorType = errorType;
        this.token = token;
        this.expected = expected;
        this.context = context;
    }

    public String getErrorType() {
        return errorType;
    }

    public Token getToken() {
        return token;
    }

    public String getExpected() {
        return expected;
    }

    public String getContext() {
        return context;
    }

    /**
     * Convert exception to JSON-serializable map for analysis.
     */
    public Map<String, Object> toJson() {
        Map<String, Object> json = new HashMap<>();
        json.put("errorType", errorType);
        json.put("message", getMessage());

        if (token != null) {
            json.put("tokenType", token.type().name());
            json.put("tokenLexeme", token.lexeme());
            json.put("line", token.line());
            json.put("column", token.column());
            json.put("position", token.position());
        }

        if (expected != null) {
            json.put("expected", expected);
        }

        if (context != null) {
            json.put("context", context);
        }

        return json;
    }
}
