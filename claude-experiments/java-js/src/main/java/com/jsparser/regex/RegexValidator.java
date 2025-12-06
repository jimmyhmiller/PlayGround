package com.jsparser.regex;

import java.util.HashSet;
import java.util.Set;

/**
 * Validates ECMAScript regular expression literals according to specification.
 * Validates both flags and pattern syntax.
 *
 * <p>Following Acorn's architecture, this validator is separate from the lexer
 * and is called after the regex has been tokenized.
 *
 * @see <a href="https://tc39.es/ecma262/#sec-patterns">ECMAScript Patterns</a>
 */
public class RegexValidator {

    /**
     * Valid regex flags per ECMAScript specification.
     * - g: global
     * - i: ignoreCase
     * - m: multiline
     * - s: dotAll (ES2018)
     * - u: unicode (ES2015)
     * - y: sticky (ES2015)
     * - d: hasIndices (ES2022)
     * - v: unicodeSets (ES2024)
     */
    private static final Set<Character> VALID_FLAGS = Set.of('g', 'i', 'm', 's', 'u', 'y', 'd', 'v');

    /**
     * Validates a regex literal pattern and flags.
     *
     * @param pattern The regex pattern (without delimiters)
     * @param flags The regex flags
     * @param position Position in source (for error reporting)
     * @param line Line number (for error reporting)
     * @param column Column number (for error reporting)
     * @throws RegexSyntaxException if validation fails
     */
    public static void validate(String pattern, String flags, int position, int line, int column) {
        // 1. Validate flags first (simpler, fail faster)
        validateFlags(flags, pattern, position, line, column);

        // 2. Determine unicode mode
        boolean unicodeMode = flags.contains("u");
        boolean unicodeSetsMode = flags.contains("v");

        // 3. Validate pattern with appropriate mode
        RegexValidationState state = new RegexValidationState(
            pattern, unicodeMode, unicodeSetsMode, position, line, column
        );
        state.validatePattern();
    }

    /**
     * Validates regex flags according to ECMAScript specification.
     *
     * Validation rules:
     * - Only characters g, i, m, s, u, y, d, v are allowed
     * - No duplicate flags allowed
     * - u and v flags are mutually exclusive (ES2024)
     *
     * @param flags The flags string to validate
     * @param pattern The pattern (for error messages)
     * @param position Position in source
     * @param line Line number
     * @param column Column number
     * @throws RegexSyntaxException if validation fails
     */
    private static void validateFlags(String flags, String pattern, int position, int line, int column) {
        Set<Character> seen = new HashSet<>();
        boolean hasU = false;
        boolean hasV = false;

        for (int i = 0; i < flags.length(); i++) {
            char flag = flags.charAt(i);

            // Check if valid
            if (!VALID_FLAGS.contains(flag)) {
                throw new RegexSyntaxException(
                    "Invalid flag '" + flag + "'",
                    pattern, flags, -1, position, line, column
                );
            }

            // Check for duplicates
            if (seen.contains(flag)) {
                throw new RegexSyntaxException(
                    "Duplicate flag '" + flag + "'",
                    pattern, flags, -1, position, line, column
                );
            }

            seen.add(flag);
            if (flag == 'u') hasU = true;
            if (flag == 'v') hasV = true;
        }

        // Check u/v exclusivity (ES2024)
        if (hasU && hasV) {
            throw new RegexSyntaxException(
                "Flags 'u' and 'v' are mutually exclusive",
                pattern, flags, -1, position, line, column
            );
        }
    }
}
