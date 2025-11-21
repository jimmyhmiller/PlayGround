package com.jsparser;

/**
 * Thrown when the parser expected a specific token but found something else.
 */
public class ExpectedTokenException extends ParseException {
    public ExpectedTokenException(String expected, Token found) {
        this(expected, found, null);
    }

    public ExpectedTokenException(String expected, Token found, String context) {
        super(
            "ExpectedToken",
            found,
            expected,
            context,
            buildMessage(expected, found, context)
        );
    }

    private static String buildMessage(String expected, Token found, String context) {
        StringBuilder msg = new StringBuilder("Expected ");
        msg.append(expected);

        if (found != null) {
            msg.append(" at token: ").append(found);
        }

        if (context != null) {
            msg.append(" (in ").append(context).append(")");
        }

        return msg.toString();
    }
}
