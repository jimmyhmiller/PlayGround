package com.jsparser;

/**
 * Thrown when the parser encounters a token it didn't expect.
 */
public class UnexpectedTokenException extends ParseException {
    public UnexpectedTokenException(Token token) {
        this(token, null, null);
    }

    public UnexpectedTokenException(Token token, String expected) {
        this(token, expected, null);
    }

    public UnexpectedTokenException(Token token, String expected, String context) {
        super(
            "UnexpectedToken",
            token,
            expected,
            context,
            buildMessage(token, expected, context)
        );
    }

    private static String buildMessage(Token token, String expected, String context) {
        StringBuilder msg = new StringBuilder("Unexpected token: ");
        msg.append(token);

        if (expected != null) {
            msg.append(", expected: ").append(expected);
        }

        if (context != null) {
            msg.append(" (in ").append(context).append(")");
        }

        return msg.toString();
    }
}
