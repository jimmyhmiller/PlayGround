package com.jsparser;

/**
 * Represents a lexer context for tracking whether we're in a statement or expression context.
 * Used to determine whether `/` is division or regex.
 * Based on Acorn's token context approach.
 */
public class LexerContext {
    private final String token;
    private final boolean isExpr;

    public LexerContext(String token, boolean isExpr) {
        this.token = token;
        this.isExpr = isExpr;
    }

    public String getToken() {
        return token;
    }

    public boolean isExpr() {
        return isExpr;
    }

    // Predefined contexts
    public static final LexerContext B_STAT = new LexerContext("{", false);  // block statement
    public static final LexerContext B_EXPR = new LexerContext("{", true);   // object literal
    public static final LexerContext P_STAT = new LexerContext("(", false);  // statement parens (if, while, etc)
    public static final LexerContext P_EXPR = new LexerContext("(", true);   // expression parens
    public static final LexerContext F_STAT = new LexerContext("function", false);  // function declaration
    public static final LexerContext F_EXPR = new LexerContext("function", true);   // function expression
    public static final LexerContext Q_TMPL = new LexerContext("`", true);   // template literal
}
