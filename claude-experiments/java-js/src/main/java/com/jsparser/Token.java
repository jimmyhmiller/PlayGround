package com.jsparser;

public record Token(
    TokenType type,
    String lexeme,
    Object literal,
    int line,
    int column,
    int position,
    int endPosition
) {
    public Token(TokenType type, String lexeme, int line, int column, int position) {
        this(type, lexeme, null, line, column, position, position + (lexeme != null ? lexeme.length() : 0));
    }

    public Token(TokenType type, String lexeme, Object literal, int line, int column, int position) {
        this(type, lexeme, literal, line, column, position, position + (lexeme != null ? lexeme.length() : 0));
    }
}
