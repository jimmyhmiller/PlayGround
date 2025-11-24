package com.jsparser;

public record Token(
    TokenType type,
    String lexeme,
    Object literal,
    int line,
    int column,
    int position,
    int endPosition,
    String raw  // For template literals: the raw string value (with escape sequences, line continuations)
) {
    public Token(TokenType type, String lexeme, int line, int column, int position) {
        this(type, lexeme, null, line, column, position, position + (lexeme != null ? lexeme.length() : 0), null);
    }

    public Token(TokenType type, String lexeme, Object literal, int line, int column, int position) {
        this(type, lexeme, literal, line, column, position, position + (lexeme != null ? lexeme.length() : 0), null);
    }

    public Token(TokenType type, String lexeme, Object literal, int line, int column, int position, int endPosition) {
        this(type, lexeme, literal, line, column, position, endPosition, null);
    }
}
