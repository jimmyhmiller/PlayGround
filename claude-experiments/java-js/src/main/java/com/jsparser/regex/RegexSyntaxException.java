package com.jsparser.regex;

import com.jsparser.ParseException;
import com.jsparser.Token;
import com.jsparser.TokenType;

/**
 * Exception thrown when regex validation fails.
 * Provides detailed error messages with position information.
 */
public class RegexSyntaxException extends ParseException {
    private final String pattern;
    private final String flags;
    private final int patternPosition;

    /**
     * Creates a new RegexSyntaxException.
     *
     * @param message Human-readable error description
     * @param pattern The regex pattern (without delimiters)
     * @param flags The regex flags
     * @param patternPosition Position in the pattern where error occurred (-1 for flag errors)
     * @param sourcePosition Position in source file
     * @param line Line number in source file
     * @param column Column number in source file
     */
    public RegexSyntaxException(String message, String pattern, String flags,
                                int patternPosition, int sourcePosition,
                                int line, int column) {
        super("RegexSyntax",
              createToken(sourcePosition, line, column),
              null,
              "regex validation",
              buildMessage(message, pattern, flags, patternPosition));
        this.pattern = pattern;
        this.flags = flags;
        this.patternPosition = patternPosition;
    }

    /**
     * Builds a detailed error message.
     */
    private static String buildMessage(String message, String pattern,
                                       String flags, int patternPosition) {
        if (patternPosition >= 0) {
            return String.format(
                "Invalid regular expression: /%s/%s: %s at position %d",
                pattern, flags, message, patternPosition
            );
        } else {
            return String.format(
                "Invalid regular expression: /%s/%s: %s",
                pattern, flags, message
            );
        }
    }

    /**
     * Creates a synthetic token for error reporting.
     */
    private static Token createToken(int position, int line, int column) {
        return new Token(TokenType.REGEX, "", null, line, column, position, position);
    }

    public String getPattern() {
        return pattern;
    }

    public String getFlags() {
        return flags;
    }

    public int getPatternPosition() {
        return patternPosition;
    }
}
