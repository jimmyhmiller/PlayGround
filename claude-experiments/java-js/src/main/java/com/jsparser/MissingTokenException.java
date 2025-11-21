package com.jsparser;

/**
 * Thrown when the parser expected a token but none was available (e.g., unexpected EOF).
 */
public class MissingTokenException extends ParseException {
    public MissingTokenException(String expected) {
        this(expected, null, null);
    }

    public MissingTokenException(String expected, Token lastToken) {
        this(expected, lastToken, null);
    }

    public MissingTokenException(String expected, Token lastToken, String context) {
        super(
            "MissingToken",
            lastToken,
            expected,
            context,
            buildMessage(expected, lastToken, context)
        );
    }

    private static String buildMessage(String expected, Token lastToken, String context) {
        StringBuilder msg = new StringBuilder("Missing ");
        msg.append(expected);

        if (lastToken != null) {
            msg.append(" (last token: ").append(lastToken).append(")");
        }

        if (context != null) {
            msg.append(" (in ").append(context).append(")");
        }

        return msg.toString();
    }
}
