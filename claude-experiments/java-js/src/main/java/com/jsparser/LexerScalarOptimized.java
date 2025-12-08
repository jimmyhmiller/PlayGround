package com.jsparser;

import com.jsparser.ast.Literal;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

/**
 * Lexer variant with optimized scalar identifier scanning.
 *
 * Uses inline ASCII checks instead of Character.isUnicodeIdentifierPart()
 * for the common case of ASCII identifiers.
 */
public class LexerScalarOptimized {
    private final String source;
    private final char[] buf;
    private final int length;
    private int position = 0;
    private int line = 1;
    private int column = 0;
    private Stack<Integer> templateBraceDepthStack = new Stack<>();
    private TokenType lastTokenType = null;
    private boolean atLineStart = true;

    private Stack<LexerContext> contextStack = new Stack<>();
    private boolean exprAllowed = true;
    private int generatorDepth = 0;
    private Stack<Boolean> functionIsGeneratorStack = new Stack<>();

    private boolean strictMode;

    public LexerScalarOptimized(String source) {
        this(source, false);
    }

    public LexerScalarOptimized(String source, boolean strictMode) {
        this.source = source;
        this.buf = source.toCharArray();
        this.length = buf.length;
        this.strictMode = strictMode;
        contextStack.push(LexerContext.B_STAT);
    }

    public List<Token> tokenize() {
        // Delegate to standard Lexer for full tokenization
        // Only scanIdentifier is optimized
        Lexer delegate = new Lexer(source, strictMode);
        return delegate.tokenize();
    }

    /**
     * Optimized scanIdentifier using fast ASCII path.
     *
     * This demonstrates the optimized approach:
     * 1. Use fast ASCII identifier scanning
     * 2. Fall back to unicode handling only when needed
     */
    public Token scanIdentifierOptimized(int startLine, int startColumn, int startPos) {
        // Fast path: scan ASCII identifier chars
        int identEnd = SIMDIdentifierScanner.scanIdentifierScalar(buf, position, length);

        // Check what stopped us
        if (identEnd < length) {
            char stopChar = buf[identEnd];

            // If we hit backslash or non-ASCII, we need full unicode handling
            if (stopChar == '\\' || stopChar > 127) {
                return scanIdentifierWithUnicode(startLine, startColumn, startPos, identEnd);
            }
        }

        // Pure ASCII identifier - fast path
        if (identEnd == startPos) {
            // Empty identifier - shouldn't happen if called correctly
            throw new RuntimeException("Invalid identifier start at position " + startPos);
        }

        String name = new String(buf, startPos, identEnd - startPos);
        position = identEnd;
        column += (identEnd - startPos);

        // Keyword lookup
        TokenType type = lookupKeyword(name);
        Object literal = literalFor(type);

        return new Token(type, literal, startLine, startColumn, startPos, identEnd);
    }

    /**
     * Handle identifiers that contain unicode escapes or non-ASCII characters.
     */
    private Token scanIdentifierWithUnicode(int startLine, int startColumn, int startPos, int asciiEnd) {
        StringBuilder actualName = new StringBuilder();
        boolean hasEscapes = false;

        // First, append the ASCII part we already scanned
        if (asciiEnd > startPos) {
            actualName.append(buf, startPos, asciiEnd - startPos);
            position = asciiEnd;
            column += (asciiEnd - startPos);
        }

        // Now handle the rest with unicode support
        while (!isAtEnd()) {
            char c = peek();

            if (c == '\\' && peekNext() == 'u') {
                hasEscapes = true;
                advance(); // consume \
                advance(); // consume u
                actualName.append(scanUnicodeEscape());
            } else if (isAlphaNumeric(c)) {
                actualName.append(advance());
                // Handle surrogate pairs
                if (Character.isHighSurrogate(c) && !isAtEnd() && Character.isLowSurrogate(peek())) {
                    actualName.append(advance());
                }
            } else {
                break;
            }
        }

        String identifierName = actualName.toString();

        // Keywords with escape sequences should be treated as identifiers
        TokenType type = TokenType.IDENTIFIER;
        if (!hasEscapes) {
            type = lookupKeyword(identifierName);
        }

        Object literal = literalFor(type);
        return new Token(type, literal, startLine, startColumn, startPos, position);
    }

    /**
     * Fast keyword lookup using switch expression.
     */
    private TokenType lookupKeyword(String name) {
        return switch (name) {
            case "true" -> TokenType.TRUE;
            case "false" -> TokenType.FALSE;
            case "null" -> TokenType.NULL;
            case "var" -> TokenType.VAR;
            case "let" -> TokenType.LET;
            case "const" -> TokenType.CONST;
            case "function" -> TokenType.FUNCTION;
            case "class" -> TokenType.CLASS;
            case "return" -> TokenType.RETURN;
            case "if" -> TokenType.IF;
            case "else" -> TokenType.ELSE;
            case "for" -> TokenType.FOR;
            case "while" -> TokenType.WHILE;
            case "do" -> TokenType.DO;
            case "break" -> TokenType.BREAK;
            case "continue" -> TokenType.CONTINUE;
            case "switch" -> TokenType.SWITCH;
            case "case" -> TokenType.CASE;
            case "default" -> TokenType.DEFAULT;
            case "try" -> TokenType.TRY;
            case "catch" -> TokenType.CATCH;
            case "finally" -> TokenType.FINALLY;
            case "throw" -> TokenType.THROW;
            case "new" -> TokenType.NEW;
            case "typeof" -> TokenType.TYPEOF;
            case "void" -> TokenType.VOID;
            case "delete" -> TokenType.DELETE;
            case "this" -> TokenType.THIS;
            case "super" -> TokenType.SUPER;
            case "in" -> TokenType.IN;
            case "instanceof" -> TokenType.INSTANCEOF;
            case "import" -> TokenType.IMPORT;
            case "export" -> TokenType.EXPORT;
            case "with" -> TokenType.WITH;
            case "debugger" -> TokenType.DEBUGGER;
            default -> TokenType.IDENTIFIER;
        };
    }

    private Object literalFor(TokenType type) {
        return switch (type) {
            case TRUE -> true;
            case FALSE -> false;
            case NULL -> null;
            default -> null;
        };
    }

    // Helper methods (same as Lexer)
    private char peek() {
        if (isAtEnd()) return '\0';
        return buf[position];
    }

    private char peekNext() {
        if (position + 1 >= length) return '\0';
        return buf[position + 1];
    }

    private char advance() {
        column++;
        return buf[position++];
    }

    private boolean isAtEnd() {
        return position >= length;
    }

    private boolean isAlphaNumeric(char c) {
        return Character.isUnicodeIdentifierPart(c) || c == '$' || Character.isHighSurrogate(c);
    }

    private String scanUnicodeEscape() {
        if (peek() == '{') {
            advance(); // consume {
            StringBuilder hex = new StringBuilder();
            while (!isAtEnd() && peek() != '}') {
                if (isHexDigit(peek())) {
                    hex.append(advance());
                } else {
                    throw new RuntimeException("Invalid unicode escape sequence");
                }
            }
            if (peek() != '}') {
                throw new RuntimeException("Invalid unicode escape sequence");
            }
            advance(); // consume }
            if (hex.length() == 0) {
                throw new RuntimeException("Invalid unicode escape sequence");
            }
            int codePoint = Integer.parseInt(hex.toString(), 16);
            return Character.toString(codePoint);
        } else {
            StringBuilder hex = new StringBuilder();
            for (int i = 0; i < 4; i++) {
                if (!isAtEnd() && isHexDigit(peek())) {
                    hex.append(advance());
                } else {
                    throw new RuntimeException("Invalid unicode escape sequence");
                }
            }
            int codePoint = Integer.parseInt(hex.toString(), 16);
            return Character.toString(codePoint);
        }
    }

    private boolean isHexDigit(char c) {
        return (c >= '0' && c <= '9') ||
               (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
    }
}
