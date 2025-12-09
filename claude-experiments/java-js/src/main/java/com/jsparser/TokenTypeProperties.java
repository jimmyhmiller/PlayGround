package com.jsparser;

import java.util.EnumMap;
import java.util.Map;

/**
 * Properties for each TokenType, based on Acorn's token type system.
 * The `beforeExpr` property indicates whether a token can be followed by an expression
 * (thus `/` after it would be a regex, not division).
 */
public class TokenTypeProperties {
    private static final Map<TokenType, Boolean> BEFORE_EXPR = new EnumMap<>(TokenType.class);

    static {
        // Tokens that can be followed by an expression (beforeExpr = true)
        // After these, `/` is a regex, not division

        // Opening delimiters
        BEFORE_EXPR.put(TokenType.LPAREN, true);
        BEFORE_EXPR.put(TokenType.LBRACKET, true);
        BEFORE_EXPR.put(TokenType.LBRACE, true);

        // Separators
        BEFORE_EXPR.put(TokenType.COMMA, true);
        BEFORE_EXPR.put(TokenType.SEMICOLON, true);
        BEFORE_EXPR.put(TokenType.COLON, true);

        // Assignment and compound assignment
        BEFORE_EXPR.put(TokenType.ASSIGN, true);
        BEFORE_EXPR.put(TokenType.PLUS_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.MINUS_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.STAR_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.STAR_STAR_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.SLASH_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.PERCENT_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.LEFT_SHIFT_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.RIGHT_SHIFT_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.UNSIGNED_RIGHT_SHIFT_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.BIT_AND_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.BIT_OR_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.BIT_XOR_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.AND_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.OR_ASSIGN, true);
        BEFORE_EXPR.put(TokenType.QUESTION_QUESTION_ASSIGN, true);

        // Binary operators
        BEFORE_EXPR.put(TokenType.PLUS, true);
        BEFORE_EXPR.put(TokenType.MINUS, true);
        BEFORE_EXPR.put(TokenType.STAR, true);
        BEFORE_EXPR.put(TokenType.STAR_STAR, true);
        BEFORE_EXPR.put(TokenType.SLASH, true);
        BEFORE_EXPR.put(TokenType.PERCENT, true);
        BEFORE_EXPR.put(TokenType.BIT_AND, true);
        BEFORE_EXPR.put(TokenType.BIT_OR, true);
        BEFORE_EXPR.put(TokenType.BIT_XOR, true);
        BEFORE_EXPR.put(TokenType.LEFT_SHIFT, true);
        BEFORE_EXPR.put(TokenType.RIGHT_SHIFT, true);
        BEFORE_EXPR.put(TokenType.UNSIGNED_RIGHT_SHIFT, true);

        // Comparison operators
        BEFORE_EXPR.put(TokenType.EQ, true);
        BEFORE_EXPR.put(TokenType.NE, true);
        BEFORE_EXPR.put(TokenType.EQ_STRICT, true);
        BEFORE_EXPR.put(TokenType.NE_STRICT, true);
        BEFORE_EXPR.put(TokenType.LT, true);
        BEFORE_EXPR.put(TokenType.LE, true);
        BEFORE_EXPR.put(TokenType.GT, true);
        BEFORE_EXPR.put(TokenType.GE, true);

        // Logical operators
        BEFORE_EXPR.put(TokenType.AND, true);
        BEFORE_EXPR.put(TokenType.OR, true);
        BEFORE_EXPR.put(TokenType.QUESTION_QUESTION, true);

        // Unary operators
        BEFORE_EXPR.put(TokenType.BANG, true);
        BEFORE_EXPR.put(TokenType.TILDE, true);

        // Other operators
        BEFORE_EXPR.put(TokenType.QUESTION, true);
        BEFORE_EXPR.put(TokenType.ARROW, true);
        BEFORE_EXPR.put(TokenType.DOT_DOT_DOT, true);

        // Keywords that can be followed by expressions
        BEFORE_EXPR.put(TokenType.RETURN, true);
        BEFORE_EXPR.put(TokenType.THROW, true);
        BEFORE_EXPR.put(TokenType.NEW, true);
        BEFORE_EXPR.put(TokenType.TYPEOF, true);
        BEFORE_EXPR.put(TokenType.VOID, true);
        BEFORE_EXPR.put(TokenType.DELETE, true);
        // Note: YIELD removed - now tokenized as IDENTIFIER
        BEFORE_EXPR.put(TokenType.CASE, true);
        BEFORE_EXPR.put(TokenType.ELSE, true);
        BEFORE_EXPR.put(TokenType.DEFAULT, true);  // export default /regex/

        // All other tokens default to beforeExpr = false
    }

    public static boolean beforeExpr(TokenType type) {
        return BEFORE_EXPR.getOrDefault(type, false);
    }
}
