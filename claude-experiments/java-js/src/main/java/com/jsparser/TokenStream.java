package com.jsparser;

import java.util.Stack;

/**
 * On-demand token stream with a ring buffer for lazy tokenization.
 * Replaces the upfront List<Token> approach to reduce allocations.
 *
 * Buffer layout: maintains a small window of tokens for lookahead.
 */
public class TokenStream {
    private static final int BUFFER_SIZE = 3;

    private final Lexer lexer;
    private final char[] sourceBuf;
    private final int sourceLength;

    // Ring buffer for tokens
    private final Token[] buffer = new Token[BUFFER_SIZE];
    private int head = 0;  // Index of current token
    private int count = 0; // Valid tokens in buffer

    private boolean lexerExhausted = false;
    private Token previousToken = null;  // Track the actual previous token (before current)
    private TokenType lastScannedType = null;  // Track the type of the last token we scanned for context

    // Checkpoint support for arrow function lookahead
    private LexerCheckpoint checkpoint = null;

    public TokenStream(String source, boolean strictMode) {
        this.lexer = new Lexer(source, strictMode);
        this.sourceBuf = lexer.getSource();
        this.sourceLength = source.length();

        // Handle hashbang at the start
        lexer.skipHashbang();

        // Pre-fill buffer with first tokens
        fillBuffer();
    }

    /**
     * Get the current token without consuming it.
     */
    public Token peek() {
        ensureBuffer(1);
        return buffer[head];
    }

    /**
     * Get a token ahead of current position.
     * @param offset 0 = current, 1 = next, 2 = next+1
     */
    public Token peekAhead(int offset) {
        if (offset < 0 || offset >= BUFFER_SIZE) {
            throw new IllegalArgumentException("Lookahead offset must be 0, 1, or 2, got: " + offset);
        }
        ensureBuffer(offset + 1);
        int idx = (head + offset) % BUFFER_SIZE;
        return buffer[idx];
    }

    /**
     * Get the previous token (before current).
     */
    public Token previous() {
        return previousToken;
    }

    /**
     * Consume current token and advance to next.
     * @return the consumed token
     */
    public Token advance() {
        Token current = peek();
        Token oldPrevious = previousToken;
        previousToken = current;

        // Save for potential unskip - store the token itself since the buffer slot may be reused
        lastSkippedToken = current;
        lastSkippedPrevious = oldPrevious;

        // Shift head forward (circular)
        head = (head + 1) % BUFFER_SIZE;
        count--;

        // Note: Don't refill here - let peek/peekAhead do it lazily to support unskip
        return current;
    }

    // Track for unskip support
    private Token lastSkippedToken = null;
    private Token lastSkippedPrevious = null;

    /**
     * Undo the last advance() call. Only valid immediately after advance().
     * Used by prefixTemplate which needs to back up one token.
     */
    public void unskip() {
        // Put the token back into the buffer
        head = (head - 1 + BUFFER_SIZE) % BUFFER_SIZE;
        buffer[head] = lastSkippedToken;
        count++;
        previousToken = lastSkippedPrevious;
        lastSkippedToken = null;
        lastSkippedPrevious = null;
    }

    /**
     * Check if we've reached EOF.
     */
    public boolean isAtEnd() {
        Token current = peek();
        return current != null && current.type() == TokenType.EOF;
    }

    /**
     * Get the source buffer for lazy lexeme access.
     */
    public char[] getSource() {
        return sourceBuf;
    }

    /**
     * Get the source length.
     */
    public int getSourceLength() {
        return sourceLength;
    }

    // ========================================================================
    // Checkpoint/Restore for arrow function lookahead
    // ========================================================================

    /**
     * Save current state for potential restore.
     * Used by isArrowFunctionParameters() which scans ahead without consuming.
     */
    public void checkpoint() {
        Token[] bufferCopy = new Token[BUFFER_SIZE];
        System.arraycopy(buffer, 0, bufferCopy, 0, BUFFER_SIZE);

        this.checkpoint = new LexerCheckpoint(
            lexer.saveState(),
            head,
            count,
            bufferCopy,
            previousToken,
            lexerExhausted,
            lastScannedType
        );
    }

    /**
     * Restore to the last checkpoint.
     */
    public void restore() {
        if (checkpoint == null) {
            throw new IllegalStateException("No checkpoint to restore");
        }

        lexer.restoreState(checkpoint.lexerState());
        this.head = checkpoint.head();
        this.count = checkpoint.count();
        System.arraycopy(checkpoint.buffer(), 0, this.buffer, 0, BUFFER_SIZE);
        this.previousToken = checkpoint.previousToken();
        this.lexerExhausted = checkpoint.lexerExhausted();
        this.lastScannedType = checkpoint.lastScannedType();
        this.checkpoint = null;
    }

    /**
     * Clear the checkpoint (commit changes).
     */
    public void clearCheckpoint() {
        this.checkpoint = null;
    }

    // ========================================================================
    // Internal buffer management
    // ========================================================================

    private void fillBuffer() {
        while (count < BUFFER_SIZE && !lexerExhausted) {
            Token token = scanNextToken();
            if (token != null) {
                int idx = (head + count) % BUFFER_SIZE;
                buffer[idx] = token;
                count++;
            }
            // If EOF, mark exhausted
            if (token != null && token.type() == TokenType.EOF) {
                lexerExhausted = true;
            }
        }
    }

    private void ensureBuffer(int needed) {
        while (count < needed && !lexerExhausted) {
            Token token = scanNextToken();
            if (token != null) {
                int idx = (head + count) % BUFFER_SIZE;
                buffer[idx] = token;
                count++;
            }
            if (token != null && token.type() == TokenType.EOF) {
                lexerExhausted = true;
            }
        }
    }

    /**
     * Scan the next token from the lexer, handling template state.
     * This mirrors the logic from Lexer.tokenize()'s main loop.
     */
    private Token scanNextToken() {
        lexer.skipWhitespace();

        if (lexer.isAtEnd()) {
            return new Token(TokenType.EOF, lexer.getLine(), lexer.getColumn(),
                           lexer.getPosition(), lexer.getPosition());
        }

        // Get the Lexer's template stack for tracking interpolation
        Stack<Integer> templateStack = lexer.getTemplateBraceDepthStack();

        // Handle template continuation
        if (!templateStack.isEmpty()) {
            int depth = templateStack.peek();
            char c = lexer.peekChar();

            if (c == '}' && depth == 0) {
                // End of interpolation, scan template continuation
                Token token = lexer.scanTemplateContinuation();
                if (token != null) {
                    lexer.updateContext(lastScannedType, token);
                    lastScannedType = token.type();
                }
                return token;
            } else if (c == '{' && depth >= 0) {
                // Nested brace inside interpolation - increment depth
                templateStack.push(templateStack.pop() + 1);
            } else if (c == '}' && depth > 0) {
                // Closing brace but not the interpolation close - decrement depth
                templateStack.push(templateStack.pop() - 1);
            }
        }

        // Scan regular token
        Token token = lexer.nextToken();
        if (token != null) {
            lexer.updateContext(lastScannedType, token);
            lastScannedType = token.type();
        }
        return token;
    }

    // ========================================================================
    // Checkpoint record
    // ========================================================================

    private record LexerCheckpoint(
        Lexer.State lexerState,
        int head,
        int count,
        Token[] buffer,
        Token previousToken,
        boolean lexerExhausted,
        TokenType lastScannedType
    ) {}
}
