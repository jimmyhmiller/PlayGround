package com.jsparser;

import jdk.incubator.vector.*;

/**
 * SIMD-optimized identifier scanning utilities using Java's Vector API.
 *
 * Provides multiple implementations for benchmarking:
 * - ShortVector: Works directly on char[] (16 chars per 256-bit vector)
 * - ByteVector: Processes ASCII bytes (32 chars per 256-bit vector)
 * - Scalar: Optimized scalar baseline for comparison
 */
public final class SIMDIdentifierScanner {

    // Use preferred species for best performance on current hardware
    private static final VectorSpecies<Short> SHORT_SPECIES = ShortVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED;

    // ASCII identifier character boundaries
    private static final short LOWER_A = 'a';
    private static final short LOWER_Z = 'z';
    private static final short UPPER_A = 'A';
    private static final short UPPER_Z = 'Z';
    private static final short DIGIT_0 = '0';
    private static final short DIGIT_9 = '9';
    private static final short UNDERSCORE = '_';
    private static final short DOLLAR = '$';
    private static final short BACKSLASH = '\\';
    private static final short ASCII_MAX = 127;

    private SIMDIdentifierScanner() {}

    /**
     * Approach A: ShortVector directly on char[].
     *
     * Scans for ASCII identifier characters [a-zA-Z0-9_$] starting from 'start'.
     * Returns the index of the first non-identifier character, or the index where
     * we hit a non-ASCII character or backslash (unicode escape).
     *
     * @param buf The source char array
     * @param start Starting position (must be valid identifier start)
     * @param length Total length of buffer
     * @return Index of first non-identifier char (or length if identifier extends to end)
     */
    public static int scanIdentifierShortVector(char[] buf, int start, int length) {
        int pos = start;
        int vectorLength = SHORT_SPECIES.length();
        int loopBound = length - vectorLength;

        // SIMD loop - process vectorLength chars at a time
        while (pos <= loopBound) {
            ShortVector chars = ShortVector.fromCharArray(SHORT_SPECIES, buf, pos);

            // Check for lowercase letters (a-z)
            VectorMask<Short> isLower = chars.compare(VectorOperators.GE, LOWER_A)
                    .and(chars.compare(VectorOperators.LE, LOWER_Z));

            // Check for uppercase letters (A-Z)
            VectorMask<Short> isUpper = chars.compare(VectorOperators.GE, UPPER_A)
                    .and(chars.compare(VectorOperators.LE, UPPER_Z));

            // Check for digits (0-9)
            VectorMask<Short> isDigit = chars.compare(VectorOperators.GE, DIGIT_0)
                    .and(chars.compare(VectorOperators.LE, DIGIT_9));

            // Check for underscore and dollar
            VectorMask<Short> isUnderscore = chars.compare(VectorOperators.EQ, UNDERSCORE);
            VectorMask<Short> isDollar = chars.compare(VectorOperators.EQ, DOLLAR);

            // Check for backslash (potential unicode escape)
            VectorMask<Short> isBackslash = chars.compare(VectorOperators.EQ, BACKSLASH);

            // Check for non-ASCII (need scalar fallback)
            VectorMask<Short> isNonAscii = chars.compare(VectorOperators.GT, ASCII_MAX);

            // Combine: valid identifier char = letter OR digit OR _ OR $
            VectorMask<Short> isIdentChar = isLower.or(isUpper).or(isDigit).or(isUnderscore).or(isDollar);

            // Stop conditions: not identifier char OR backslash OR non-ASCII
            VectorMask<Short> shouldStop = isIdentChar.not().or(isBackslash).or(isNonAscii);

            // Find first position where we should stop
            int stopIndex = shouldStop.firstTrue();
            if (stopIndex < vectorLength) {
                return pos + stopIndex;
            }

            pos += vectorLength;
        }

        // Scalar tail - handle remaining chars
        return scanIdentifierScalarFrom(buf, pos, length);
    }

    /**
     * Approach B: ByteVector with char-to-short-to-byte extraction.
     *
     * For pure ASCII identifiers, we can check if all chars are < 128 and then
     * process them as bytes for 2x throughput.
     *
     * @param buf The source char array
     * @param start Starting position
     * @param length Total length of buffer
     * @return Index of first non-identifier char
     */
    public static int scanIdentifierByteVector(char[] buf, int start, int length) {
        int pos = start;
        int vectorLength = BYTE_SPECIES.length();

        // First, try to find a run of ASCII chars we can process as bytes
        // We need to load chars, verify they're ASCII, then process

        int shortVectorLength = SHORT_SPECIES.length();
        int loopBound = length - shortVectorLength;

        while (pos <= loopBound) {
            // Load as shorts (chars)
            ShortVector chars = ShortVector.fromCharArray(SHORT_SPECIES, buf, pos);

            // Quick check: any non-ASCII or special chars?
            VectorMask<Short> isNonAscii = chars.compare(VectorOperators.GT, ASCII_MAX);
            VectorMask<Short> isBackslash = chars.compare(VectorOperators.EQ, BACKSLASH);

            if (isNonAscii.anyTrue() || isBackslash.anyTrue()) {
                // Found non-ASCII or escape - find exact position and stop
                VectorMask<Short> shouldStopEarly = isNonAscii.or(isBackslash);
                int earlyStop = shouldStopEarly.firstTrue();

                // Scan the ASCII portion with identifier check
                return scanIdentifierScalarFrom(buf, pos, Math.min(pos + earlyStop, length));
            }

            // All ASCII - now check if they're valid identifier chars
            // Convert to bytes by casting (safe since we verified ASCII)
            // For now, use the same short-based logic since Java Vector API
            // doesn't have direct char->byte narrowing that preserves semantics

            // Check for lowercase letters (a-z)
            VectorMask<Short> isLower = chars.compare(VectorOperators.GE, LOWER_A)
                    .and(chars.compare(VectorOperators.LE, LOWER_Z));

            // Check for uppercase letters (A-Z)
            VectorMask<Short> isUpper = chars.compare(VectorOperators.GE, UPPER_A)
                    .and(chars.compare(VectorOperators.LE, UPPER_Z));

            // Check for digits (0-9)
            VectorMask<Short> isDigit = chars.compare(VectorOperators.GE, DIGIT_0)
                    .and(chars.compare(VectorOperators.LE, DIGIT_9));

            // Check for underscore and dollar
            VectorMask<Short> isUnderscore = chars.compare(VectorOperators.EQ, UNDERSCORE);
            VectorMask<Short> isDollar = chars.compare(VectorOperators.EQ, DOLLAR);

            // Combine
            VectorMask<Short> isIdentChar = isLower.or(isUpper).or(isDigit).or(isUnderscore).or(isDollar);
            VectorMask<Short> notIdentChar = isIdentChar.not();

            int stopIndex = notIdentChar.firstTrue();
            if (stopIndex < shortVectorLength) {
                return pos + stopIndex;
            }

            pos += shortVectorLength;
        }

        // Scalar tail
        return scanIdentifierScalarFrom(buf, pos, length);
    }

    /**
     * Optimized scalar implementation for comparison baseline.
     * Uses inline ASCII range checks instead of Character.isUnicodeIdentifierPart().
     *
     * @param buf The source char array
     * @param start Starting position
     * @param length Total length of buffer
     * @return Index of first non-identifier char
     */
    public static int scanIdentifierScalar(char[] buf, int start, int length) {
        return scanIdentifierScalarFrom(buf, start, length);
    }

    /**
     * Scalar scanning helper - scans from given position.
     */
    private static int scanIdentifierScalarFrom(char[] buf, int pos, int length) {
        while (pos < length) {
            char c = buf[pos];

            // Fast ASCII check first
            if (c <= 127) {
                // ASCII identifier chars: a-z, A-Z, 0-9, _, $
                if ((c >= 'a' && c <= 'z') ||
                    (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') ||
                    c == '_' || c == '$') {
                    pos++;
                    continue;
                }
                // Backslash might start unicode escape - let caller handle
                if (c == '\\') {
                    return pos;
                }
                // Other ASCII char - not identifier
                return pos;
            }

            // Non-ASCII - could be unicode identifier or surrogate
            // Let caller handle with Character.isUnicodeIdentifierPart()
            return pos;
        }
        return pos;
    }

    /**
     * Check if a character is a valid ASCII identifier start character.
     * (a-z, A-Z, _, $)
     */
    public static boolean isAsciiIdentifierStart(char c) {
        return (c >= 'a' && c <= 'z') ||
               (c >= 'A' && c <= 'Z') ||
               c == '_' || c == '$';
    }

    /**
     * Check if a character is a valid ASCII identifier continuation character.
     * (a-z, A-Z, 0-9, _, $)
     */
    public static boolean isAsciiIdentifierPart(char c) {
        return (c >= 'a' && c <= 'z') ||
               (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') ||
               c == '_' || c == '$';
    }

    /**
     * Get the vector species being used for ShortVector operations.
     * Useful for debugging/logging.
     */
    public static VectorSpecies<Short> getShortSpecies() {
        return SHORT_SPECIES;
    }

    /**
     * Get the vector species being used for ByteVector operations.
     */
    public static VectorSpecies<Byte> getByteSpecies() {
        return BYTE_SPECIES;
    }
}
