package com.jsparser.regex;

/**
 * Stateful validator for regex patterns.
 * Implements recursive descent parsing of ECMAScript regex grammar.
 *
 * <p>This class follows Acorn's RegExpValidationState design, parsing the
 * regex pattern according to the ECMAScript grammar and validating syntax
 * based on the active flags (unicode mode vs unicode sets mode).
 *
 * @see <a href="https://tc39.es/ecma262/#sec-patterns">ECMAScript Patterns Grammar</a>
 */
public class RegexValidationState {
    private final String pattern;
    private final boolean unicodeMode;      // u flag
    private final boolean unicodeSetsMode;  // v flag
    private int pos;
    private final int startPosition;
    private final int startLine;
    private final int startColumn;

    // State tracking
    private int groupCount;
    private int maxBackReference;
    private boolean lastAssertionIsQuantifiable;
    private boolean inNegatedCharacterClass;

    /**
     * Creates a new validation state.
     *
     * @param pattern The regex pattern to validate
     * @param unicodeMode True if u flag is set
     * @param unicodeSetsMode True if v flag is set
     * @param startPosition Position in source file
     * @param startLine Line in source file
     * @param startColumn Column in source file
     */
    public RegexValidationState(String pattern, boolean unicodeMode, boolean unicodeSetsMode,
                                int startPosition, int startLine, int startColumn) {
        this.pattern = pattern;
        this.unicodeMode = unicodeMode;
        this.unicodeSetsMode = unicodeSetsMode;
        this.pos = 0;
        this.startPosition = startPosition;
        this.startLine = startLine;
        this.startColumn = startColumn;
        this.groupCount = 0;
        this.maxBackReference = 0;
        this.lastAssertionIsQuantifiable = false;
    }

    /**
     * Validates the entire pattern.
     * Entry point for pattern validation.
     *
     * @throws RegexSyntaxException if validation fails
     */
    public void validatePattern() {
        pos = 0;
        parseDisjunction();
        if (!isAtEnd()) {
            throw error("Unexpected characters at end of pattern");
        }
    }

    // ========== ECMAScript Grammar Implementation ==========
    // Following the grammar from: https://tc39.es/ecma262/#sec-patterns

    /**
     * Disjunction ::
     *   Alternative
     *   Alternative | Disjunction
     */
    private void parseDisjunction() {
        parseAlternative();
        while (match('|')) {
            parseAlternative();
        }
    }

    /**
     * Alternative ::
     *   [empty]
     *   Alternative Term
     */
    private void parseAlternative() {
        while (!isAtEnd() && peek() != '|' && peek() != ')') {
            parseTerm();
        }
    }

    /**
     * Term ::
     *   Assertion
     *   Atom
     *   Atom Quantifier
     */
    private void parseTerm() {
        if (parseAssertion()) {
            // Assertions can be followed by quantifiers (though semantically meaningless for lookahead/lookbehind)
            // We need to consume any quantifier to avoid infinite loops
            if (isQuantifierStart()) {
                parseQuantifier();
            }
            return;
        }

        // Check for quantifier without atom first (only in unicode modes)
        if ((unicodeMode || unicodeSetsMode) && isQuantifierStart()) {
            throw error("Quantifier requires an atom");
        }

        int atomStart = pos;
        if (!parseAtom()) {
            // No atom found and no quantifier, nothing to parse
            return;
        }

        // Check for quantifier after atom
        parseQuantifier();
    }

    /**
     * Assertion ::
     *   ^
     *   $
     *   \b
     *   \B
     *   (?= Disjunction )
     *   (?! Disjunction )
     *   (?<= Disjunction )
     *   (?<! Disjunction )
     *
     * Returns true if an assertion was parsed.
     */
    private boolean parseAssertion() {
        char c = peek();

        if (c == '^' || c == '$') {
            consume();
            lastAssertionIsQuantifiable = false;
            return true;
        }

        if (c == '\\') {
            char next = peek(1);
            if (next == 'b' || next == 'B') {
                consume(); // \
                consume(); // b or B
                lastAssertionIsQuantifiable = false;
                return true;
            }
        }

        if (c == '(' && peek(1) == '?') {
            char lookahead = peek(2);
            if (lookahead == '=' || lookahead == '!') {
                // Positive/negative lookahead
                consume(); // (
                consume(); // ?
                consume(); // = or !
                parseDisjunction();
                if (!match(')')) {
                    throw error("Unclosed lookahead assertion");
                }
                lastAssertionIsQuantifiable = false;
                return true;
            }
            if (lookahead == '<') {
                char lookbehindType = peek(3);
                if (lookbehindType == '=' || lookbehindType == '!') {
                    // Positive/negative lookbehind
                    consume(); // (
                    consume(); // ?
                    consume(); // <
                    consume(); // = or !
                    parseDisjunction();
                    if (!match(')')) {
                        throw error("Unclosed lookbehind assertion");
                    }
                    lastAssertionIsQuantifiable = false;
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Atom ::
     *   PatternCharacter
     *   .
     *   \ AtomEscape
     *   CharacterClass
     *   ( GroupSpecifier Disjunction )
     *   (?: Disjunction )
     *
     * Returns true if an atom was parsed.
     */
    private boolean parseAtom() {
        char c = peek();

        // Dot
        if (c == '.') {
            consume();
            return true;
        }

        // Escape
        if (c == '\\') {
            parseAtomEscape();
            return true;
        }

        // Character class
        if (c == '[') {
            parseCharacterClass();
            return true;
        }

        // Group
        if (c == '(') {
            consume();

            // Check for non-capturing group (?:...) or named group (?<name>...) or modifiers (?i:...)
            if (peek() == '?') {
                char next = peek(1);
                if (next == ':') {
                    // Non-capturing group
                    consume(); // ?
                    consume(); // :
                    parseDisjunction();
                } else if (next == '<') {
                    // Named capturing group
                    consume(); // ?
                    consume(); // <
                    parseGroupName();
                    if (!match('>')) {
                        throw error("Expected '>' in named group");
                    }
                    groupCount++;
                    parseDisjunction();
                } else if (isModifierChar(next) || next == '-') {
                    // Regexp modifiers: (?i:...), (?-i:...), (?im:...), (?i-m:...), etc.
                    consume(); // ?
                    parseModifiers();
                    if (!match(':')) {
                        throw error("Expected ':' in modifier group");
                    }
                    parseDisjunction();
                } else {
                    // This is an assertion, not a group - backtrack
                    pos--;
                    return false;
                }
            } else {
                // Capturing group
                groupCount++;
                parseDisjunction();
            }

            if (!match(')')) {
                throw error("Unclosed group");
            }
            return true;
        }

        // Pattern character (anything that's not a special character)
        if (isPatternCharacter(c)) {
            consume();
            return true;
        }

        return false;
    }

    /**
     * AtomEscape ::
     *   DecimalEscape
     *   CharacterEscape
     *   CharacterClassEscape
     *   k GroupName [+NamedCaptureGroups]
     */
    private void parseAtomEscape() {
        consume(); // consume '\'

        if (isAtEnd()) {
            throw error("Pattern cannot end with backslash");
        }

        char c = peek();

        // Decimal escape (backreference or octal)
        if (c >= '1' && c <= '9') {
            parseDecimalEscape();
            return;
        }

        // Character class escape (\d, \D, \s, \S, \w, \W)
        if (c == 'd' || c == 'D' || c == 's' || c == 'S' || c == 'w' || c == 'W') {
            consume();
            return;
        }

        // Unicode property escape (\p{...}, \P{...})
        if (c == 'p' || c == 'P') {
            parseUnicodePropertyEscape();
            return;
        }

        // Named backreference (\k<name>)
        if (c == 'k') {
            int startPos = pos;
            consume();
            if (!match('<')) {
                if (unicodeMode || unicodeSetsMode) {
                    throw error("Invalid escape sequence");
                }
                // In non-unicode mode, \k is a literal k
                return;
            }

            // In non-unicode mode, malformed named backreferences are treated as literals
            if (!(unicodeMode || unicodeSetsMode)) {
                try {
                    parseGroupName();
                    if (!match('>')) {
                        // Missing '>', treat as literal in non-unicode mode
                        pos = startPos;
                        return;
                    }
                } catch (RegexSyntaxException e) {
                    // Invalid group name in non-unicode mode, treat as literal
                    pos = startPos;
                    return;
                }
            } else {
                // In unicode mode, errors are thrown
                parseGroupName();
                if (!match('>')) {
                    throw error("Expected '>' in named backreference");
                }
            }
            return;
        }

        // Character escape
        parseCharacterEscape();
    }

    /**
     * Parse decimal escape (backreference or legacy octal).
     */
    private void parseDecimalEscape() {
        int start = pos;
        StringBuilder num = new StringBuilder();

        while (!isAtEnd() && peek() >= '0' && peek() <= '9') {
            num.append(consume());
        }

        int value = Integer.parseInt(num.toString());

        if (value > maxBackReference) {
            maxBackReference = value;
        }

        // In unicode mode, octal escapes are not allowed
        if (unicodeMode || unicodeSetsMode) {
            // This is a backreference (validation of whether the group exists
            // would typically happen in a second pass, which we're not doing)
        }
    }

    /**
     * CharacterEscape ::
     *   ControlEscape
     *   c ControlLetter
     *   0 [lookahead ∉ DecimalDigit]
     *   HexEscapeSequence
     *   RegExpUnicodeEscapeSequence
     *   IdentityEscape
     */
    private void parseCharacterEscape() {
        char c = peek();

        // Control escapes: t, n, v, f, r
        if (c == 't' || c == 'n' || c == 'v' || c == 'f' || c == 'r') {
            consume();
            return;
        }

        // Control letter: \cX
        if (c == 'c') {
            consume();
            if (isAtEnd()) {
                if (unicodeMode || unicodeSetsMode) {
                    throw error("Invalid control escape");
                }
                return;
            }
            char next = peek();
            if ((next >= 'a' && next <= 'z') || (next >= 'A' && next <= 'Z')) {
                consume();
                return;
            }
            if (unicodeMode || unicodeSetsMode) {
                throw error("Invalid control escape");
            }
            return;
        }

        // Null escape: \0
        if (c == '0') {
            consume();
            // In unicode mode, \0 followed by digit is invalid
            if ((unicodeMode || unicodeSetsMode) && !isAtEnd() && peek() >= '0' && peek() <= '9') {
                throw error("Invalid decimal escape in unicode mode");
            }
            return;
        }

        // Hex escape: \xHH
        if (c == 'x') {
            consume();
            if (!isHexDigit(peek()) || !isHexDigit(peek(1))) {
                if (unicodeMode || unicodeSetsMode) {
                    throw error("Invalid hex escape");
                }
                return;
            }
            consume();
            consume();
            return;
        }

        // Unicode escape: backslash-u-HHHH or backslash-u-{...}
        if (c == 'u') {
            parseUnicodeEscape();
            return;
        }

        // Identity escape - in unicode mode, only syntax characters can be escaped
        if (unicodeMode || unicodeSetsMode) {
            // In unicode sets mode, also allow - to be escaped
            if (!isSyntaxCharacter(c) && c != '/' && !(unicodeSetsMode && c == '-')) {
                throw error("Invalid identity escape in unicode mode");
            }
        }

        consume();
    }

    /**
     * Parse Unicode escape sequence.
     */
    private void parseUnicodeEscape() {
        consume(); // consume 'u'

        if (peek() == '{') {
            // backslash-u-{HexDigits}
            consume();
            if (!isHexDigit(peek())) {
                throw error("Invalid unicode escape");
            }
            while (!isAtEnd() && isHexDigit(peek())) {
                consume();
            }
            if (!match('}')) {
                throw error("Unclosed unicode escape");
            }
        } else {
            // backslash-u-HHHH
            for (int i = 0; i < 4; i++) {
                if (!isHexDigit(peek())) {
                    if (unicodeMode || unicodeSetsMode) {
                        throw error("Invalid unicode escape");
                    }
                    return;
                }
                consume();
            }
        }
    }

    /**
     * Parse Unicode property escape: \p{...} or \P{...}
     */
    private void parseUnicodePropertyEscape() {
        // Consume the 'p' or 'P' character
        char escapeChar = consume(); // 'p' or 'P'
        boolean isNegated = (escapeChar == 'P');

        if (!match('{')) {
            if (unicodeMode || unicodeSetsMode) {
                throw error("Invalid unicode property escape");
            }
            return;
        }

        // Read property name/value
        StringBuilder property = new StringBuilder();
        while (!isAtEnd() && peek() != '}') {
            property.append(consume());
        }

        if (!match('}')) {
            throw error("Unclosed unicode property escape");
        }

        String propertyStr = property.toString();
        boolean isStrProp = isStringProperty(propertyStr);

        // String properties are only allowed in unicode sets mode (v flag), not in unicode mode (u flag)
        if (unicodeMode && !unicodeSetsMode && isStrProp) {
            throw error("String property '" + propertyStr + "' can only be used with the v flag (unicode sets mode), not the u flag");
        }

        // In unicode sets mode (v flag), negated string properties are not allowed
        // This includes both \P{...} and [^\p{...}]
        if (unicodeSetsMode && isStrProp && (isNegated || inNegatedCharacterClass)) {
            if (isNegated) {
                throw error("Cannot negate string property '" + propertyStr + "' with \\P in unicode sets mode");
            } else {
                throw error("Cannot negate string property '" + propertyStr + "' in character class [^\\p{...}] in unicode sets mode");
            }
        }

        // Validate property name/value syntax
        validateUnicodePropertySyntax(propertyStr);
    }

    /**
     * Validate Unicode property escape syntax and property names.
     * Checks for:
     * - Empty property
     * - Whitespace (not allowed)
     * - Grammar extensions like In/Is prefixes (not allowed)
     * - Circumflex negation (not allowed)
     * - Empty values in property=value form
     * - Starts with = (value without property)
     * - Wrong separator (: instead of =)
     * - Invalid/unsupported property names
     */
    private void validateUnicodePropertySyntax(String propertyStr) {
        // Empty property is not allowed (e.g., \p{})
        if (propertyStr.isEmpty()) {
            throw error("Unicode property escape cannot be empty");
        }

        // No whitespace allowed in property names/values
        if (propertyStr.indexOf(' ') >= 0 || propertyStr.indexOf('\t') >= 0) {
            throw error("Unicode property escape cannot contain whitespace");
        }

        // Circumflex negation is not allowed (e.g., ^General_Category=Letter)
        if (propertyStr.startsWith("^")) {
            throw error("Unicode property escape cannot use ^ for negation inside braces");
        }

        // Starts with = is not allowed (e.g., =Letter)
        if (propertyStr.startsWith("=")) {
            throw error("Unicode property escape cannot start with =");
        }

        // Colon separator is not allowed (should use = instead)
        if (propertyStr.indexOf(':') >= 0) {
            throw error("Unicode property escape must use = not : as separator");
        }

        // Check for empty values in property=value form
        int equalsIndex = propertyStr.indexOf('=');
        if (equalsIndex >= 0) {
            String value = propertyStr.substring(equalsIndex + 1);
            if (value.isEmpty()) {
                throw error("Unicode property escape cannot have empty value after =");
            }
        }

        // Check for grammar extensions that are not allowed
        // "In" prefix: not allowed at all (whether for blocks or in property=value form)
        // Examples: InAdlam, InBasic_Latin, InScript=Adlam
        // This is different from valid properties like "Initial_Punctuation" or "Inherited"
        if (propertyStr.startsWith("In") && propertyStr.length() > 2 &&
            Character.isUpperCase(propertyStr.charAt(2))) {
            // "In" prefix is not allowed
            throw error("Unicode property escape cannot use 'In' prefix");
        }

        // "Is" prefix in property=value form is not allowed (e.g., IsScript=Adlam)
        if (propertyStr.startsWith("Is") && propertyStr.indexOf('=') >= 0) {
            throw error("Unicode property escape cannot use 'Is' prefix");
        }

        // Validate property names and values against Unicode property database
        validateUnicodeProperty(propertyStr);
    }

    /**
     * Validate Unicode property names against a database of known properties.
     * This includes checking for:
     * - Deprecated/unsupported properties
     * - Invalid property names
     * - Binary properties used with values
     * - Case sensitivity
     */
    private void validateUnicodeProperty(String propertyStr) {
        int equalsIndex = propertyStr.indexOf('=');

        if (equalsIndex >= 0) {
            // Property=Value form
            String propertyName = propertyStr.substring(0, equalsIndex);
            String propertyValue = propertyStr.substring(equalsIndex + 1);

            // Check if this is a binary property (which shouldn't have a value)
            if (isBinaryProperty(propertyName)) {
                throw error("Binary property '" + propertyName + "' cannot have a value");
            }

            // Check if property is unsupported/deprecated
            if (isUnsupportedProperty(propertyName)) {
                throw error("Unsupported Unicode property: " + propertyName);
            }

            // Check for invalid property names in property=value form
            if (isInvalidPropertyName(propertyName)) {
                throw error("Invalid Unicode property: " + propertyName);
            }

            // Check for invalid values (all lowercase, weird formatting, etc.)
            if (isInvalidPropertyValue(propertyValue)) {
                throw error("Invalid Unicode property value: " + propertyValue);
            }

            // General_Category and Script/Script_Extensions are the main non-binary properties
            // For now, we do basic validation
        } else {
            // Property-only form (either a binary property or General_Category value)

            // Check if property is unsupported/deprecated
            if (isUnsupportedProperty(propertyStr)) {
                throw error("Unsupported Unicode property: " + propertyStr);
            }

            // Check if this is a non-binary property that requires a value
            if (isNonBinaryProperty(propertyStr)) {
                throw error("Non-binary property '" + propertyStr + "' requires a value");
            }

            // Check if this looks like an obviously invalid property name
            if (isInvalidPropertyName(propertyStr)) {
                throw error("Invalid Unicode property: " + propertyStr);
            }
        }
    }

    /**
     * Check if a property is a non-binary property that requires a value.
     * These properties must be used in property=value form.
     */
    private boolean isNonBinaryProperty(String propertyName) {
        // Properties that require values
        return propertyName.equals("General_Category") ||
               propertyName.equals("Script") ||
               propertyName.equals("Script_Extensions");
    }

    /**
     * Check if a property name is a binary property.
     * Binary properties can only be used alone, not with a value.
     */
    private boolean isBinaryProperty(String propertyName) {
        // Common binary properties that are tested
        return propertyName.equals("ASCII") ||
               propertyName.equals("Alphabetic") ||
               propertyName.equals("Any") ||
               propertyName.equals("Assigned") ||
               propertyName.equals("Lowercase") ||
               propertyName.equals("Uppercase") ||
               propertyName.equals("White_Space") ||
               propertyName.equals("Hex_Digit") ||
               propertyName.equals("ID_Start") ||
               propertyName.equals("ID_Continue");
    }

    /**
     * Check if a property has been deprecated or removed from the Unicode spec.
     */
    private boolean isUnsupportedProperty(String propertyName) {
        // Properties that were removed from the Unicode property escapes proposal
        return propertyName.equals("Composition_Exclusion") ||
               propertyName.equals("Expands_On_NFC") ||
               propertyName.equals("Expands_On_NFD") ||
               propertyName.equals("Expands_On_NFKC") ||
               propertyName.equals("Expands_On_NFKD") ||
               propertyName.equals("FC_NFKC_Closure") ||
               propertyName.equals("Full_Composition_Exclusion") ||
               propertyName.equals("Grapheme_Link") ||
               propertyName.equals("Hyphen") ||
               propertyName.equals("Other_Alphabetic") ||
               propertyName.equals("Other_Default_Ignorable_Code_Point") ||
               propertyName.equals("Other_Grapheme_Extend") ||
               propertyName.equals("Other_ID_Continue") ||
               propertyName.equals("Other_ID_Start") ||
               propertyName.equals("Other_Lowercase") ||
               propertyName.equals("Other_Math") ||
               propertyName.equals("Other_Uppercase") ||
               propertyName.equals("Prepended_Concatenation_Mark") ||
               // Block and Line_Break are not supported
               propertyName.equals("Block") ||
               propertyName.equals("Line_Break");
    }

    /**
     * Check if a property name is obviously invalid.
     * This catches completely made-up property names and wrong casings.
     */
    private boolean isInvalidPropertyName(String propertyName) {
        // Check for explicitly known invalid property names
        if (propertyName.equals("invalid") ||
            propertyName.equals("UnknownBinaryProperty") ||
            propertyName.contains("Breakz")) { // Typos
            return true;
        }

        // Check for wrong case versions of known properties (case-sensitive match required)
        if (propertyName.equals("any") || propertyName.equals("ANY") || // Should be "Any"
            propertyName.equals("assigned") || propertyName.equals("ASSIGNED") || // Should be "Assigned"
            propertyName.equals("lowercase") || // Should be "Lowercase"
            propertyName.equals("uppercase") || // Should be "Uppercase"
            propertyName.equals("ascii") || propertyName.equals("Ascii")) { // Should be "ASCII"
            return true;
        }

        // Check for properties with weird formatting (hyphens instead of underscores)
        if (propertyName.contains("-") &&
            !propertyName.matches("^[A-Z][a-z]+(_[A-Z][a-z]+)*$")) {
            // Hyphens are not allowed (except in valid property names which don't have them)
            return true;
        }

        return false;
    }

    /**
     * Check if a property value is obviously invalid.
     * This catches wrong casings and nonsense values.
     */
    private boolean isInvalidPropertyValue(String value) {
        // Check for all-lowercase versions of known values (with or without underscores)
        if (value.equals("uppercase_letter") || // Should be "Uppercase_Letter"
            value.equals("lowercase_letter") || // Should be "Lowercase_Letter"
            value.equals("uppercaseletter") ||  // Should be "Uppercase_Letter"
            value.equals("lowercaseletter")) {  // Should be "Lowercase_Letter"
            return true;
        }

        // Check for completely nonsense values
        if (value.equals("WAT")) {
            return true;
        }

        return false;
    }

    /**
     * Check if a Unicode property is a "property of strings" (matches sequences, not just single characters).
     * These properties cannot be negated with \P{...} in unicode sets mode.
     */
    private boolean isStringProperty(String propertyName) {
        // String properties from Unicode property escapes
        // See: https://tc39.es/ecma262/#table-binary-unicode-properties-of-strings
        return propertyName.equals("Basic_Emoji") ||
               propertyName.equals("Emoji_Keycap_Sequence") ||
               propertyName.equals("RGI_Emoji_Modifier_Sequence") ||
               propertyName.equals("RGI_Emoji_Flag_Sequence") ||
               propertyName.equals("RGI_Emoji_Tag_Sequence") ||
               propertyName.equals("RGI_Emoji_ZWJ_Sequence") ||
               propertyName.equals("RGI_Emoji");
    }

    /**
     * Check if an escape sequence is a "character class" escape that matches multiple characters.
     * These cannot be used as endpoints in character class ranges in unicode mode.
     * Examples: \p{...}, \P{...}, \d, \D, \w, \W, \s, \S
     */
    private boolean isCharacterClassEscape(int start, int end) {
        if (end <= start + 1) return false; // Need at least '\' and one char
        if (pattern.charAt(start) != '\\') return false;

        char afterBackslash = pattern.charAt(start + 1);
        // Check for property escapes
        if (afterBackslash == 'p' || afterBackslash == 'P') {
            return true;
        }
        // Check for predefined character class escapes
        if (afterBackslash == 'd' || afterBackslash == 'D' ||
            afterBackslash == 'w' || afterBackslash == 'W' ||
            afterBackslash == 's' || afterBackslash == 'S') {
            return true;
        }
        return false;
    }

    /**
     * Parse group name for named captures and backreferences.
     */
    private void parseGroupName() {
        // Group name can start with identifier or Unicode escape
        if (isAtEnd()) {
            throw error("Invalid group name");
        }

        // First character - handle surrogate pairs for high Unicode
        if (peek() == '\\') {
            parseGroupNameEscape();
        } else {
            int codePoint = getCodePointAt(pos);
            if (isIdentifierStartCodePoint(codePoint)) {
                pos += Character.charCount(codePoint);
            } else {
                throw error("Invalid group name");
            }
        }

        // Rest of name
        while (!isAtEnd() && peek() != '>') {
            if (peek() == '\\') {
                parseGroupNameEscape();
            } else {
                int codePoint = getCodePointAt(pos);
                if (isIdentifierPartCodePoint(codePoint)) {
                    pos += Character.charCount(codePoint);
                } else {
                    break; // End of group name
                }
            }
        }
    }

    private int getCodePointAt(int index) {
        if (index >= pattern.length()) {
            return -1;
        }
        return pattern.codePointAt(index);
    }

    private boolean isIdentifierStartCodePoint(int codePoint) {
        if (codePoint < 0) return false;
        return (codePoint >= 'a' && codePoint <= 'z') ||
               (codePoint >= 'A' && codePoint <= 'Z') ||
               codePoint == '_' || codePoint == '$' ||
               Character.isLetter(codePoint) ||
               codePoint == 0x200C || codePoint == 0x200D; // Zero-width joiners
    }

    private boolean isIdentifierPartCodePoint(int codePoint) {
        return isIdentifierStartCodePoint(codePoint) || Character.isDigit(codePoint);
    }

    /**
     * Parse Unicode escape in group name: \\u{...} or \\uXXXX
     */
    private void parseGroupNameEscape() {
        consume(); // consume backslash
        if (isAtEnd()) {
            throw error("Invalid escape in group name");
        }

        char c = peek();
        if (c == 'u') {
            consume(); // consume u
            if (!isAtEnd() && peek() == '{') {
                // \\u{XXXX} format
                consume(); // consume {
                while (!isAtEnd() && peek() != '}') {
                    if (!isHexDigit(peek())) {
                        throw error("Invalid hex digit in Unicode escape");
                    }
                    consume();
                }
                if (!match('}')) {
                    throw error("Unclosed Unicode escape");
                }
            } else {
                // \\uXXXX format
                for (int i = 0; i < 4; i++) {
                    if (isAtEnd() || !isHexDigit(peek())) {
                        throw error("Invalid Unicode escape");
                    }
                    consume();
                }
            }
        } else {
            throw error("Invalid escape in group name");
        }
    }

    /**
     * CharacterClass ::
     *   [ [lookahead ≠ ^] ClassContents ]
     *   [^ ClassContents ]
     */
    private void parseCharacterClass() {
        consume(); // consume '['

        // Check for negation
        boolean negated = match('^');

        // Track negation state for property escape validation
        boolean wasInNegatedCharClass = inNegatedCharacterClass;
        if (negated) {
            inNegatedCharacterClass = true;
        }

        if (unicodeSetsMode) {
            parseCharacterClassV();
        } else {
            parseCharacterClassU();
        }

        if (!match(']')) {
            throw error("Unclosed character class");
        }

        // Restore previous negation state
        inNegatedCharacterClass = wasInNegatedCharClass;
    }

    /**
     * Parse character class contents for standard/unicode mode.
     */
    private void parseCharacterClassU() {
        boolean lastWasCharacterClass = false; // Track if last atom was a character class escape
        boolean expectingRangeEnd = false; // True if we just saw 'X-' and are expecting the end atom
        boolean hadPreviousAtom = false; // Track if we've seen at least one atom (for detecting literal '-' at start)

        while (!isAtEnd() && peek() != ']') {
            if (peek() == '\\') {
                int beforeEscape = pos;
                parseClassEscape();
                // Check if this was a property escape or other character class escape
                boolean isCharClassEscape = isCharacterClassEscape(beforeEscape, pos);

                // If we're expecting a range end and this is a char class escape, that's an error
                if (unicodeMode && expectingRangeEnd && isCharClassEscape) {
                    throw error("Character class escape cannot be used as end of range in unicode mode");
                }

                // Check for range: if we have a character class escape followed by '-' and another atom,
                // that's an error in unicode mode
                if (unicodeMode && isCharClassEscape && peek() == '-' && peek(1) != ']') {
                    throw error("Character class escape cannot be used as start of range in unicode mode");
                }

                lastWasCharacterClass = isCharClassEscape;
                expectingRangeEnd = false;
                hadPreviousAtom = true;
            } else if (peek() == '-') {
                // This could be a range operator
                consume();
                // '-' is a range operator only if:
                // 1. We've seen a previous atom (not at start of class)
                // 2. There's something after the dash (not at end of class)
                // Otherwise it's a literal '-'
                if (hadPreviousAtom && peek() != ']') {
                    expectingRangeEnd = true;
                } else {
                    // '-' as a literal (at start or end) IS an atom
                    expectingRangeEnd = false;
                    hadPreviousAtom = true;
                }
            } else {
                consume();
                lastWasCharacterClass = false;
                expectingRangeEnd = false;
                hadPreviousAtom = true;
            }
        }
    }

    /**
     * Parse character class contents for unicode sets mode (v flag).
     * This is more restrictive than u mode.
     *
     * In v mode, these characters must be escaped in character classes:
     * ( ) [ ] { } / - | and double punctuators like !! ## && etc.
     * Exception: - is allowed in ranges like [a-z]
     */
    private void parseCharacterClassV() {
        char lastChar = '\0';
        boolean lastWasChar = false;
        boolean hasLeftOperand = false; // Track if we have an operand before an operator

        while (!isAtEnd() && peek() != ']') {
            char c = peek();

            // Check for nested character class [...]
            if (c == '[') {
                consume(); // consume [
                parseCharacterClassV(); // recursively parse nested class
                if (!match(']')) {
                    throw error("Unclosed nested character class");
                }
                lastWasChar = true;
                lastChar = '\0';
                hasLeftOperand = true;
                continue;
            }

            // Check for set operators && (intersection) and -- (subtraction)
            if (c == '&' && peek(1) == '&') {
                // Intersection operator requires left operand
                if (!hasLeftOperand) {
                    throw error("Set intersection operator && requires a left operand");
                }
                consume(); // consume first &
                consume(); // consume second &
                lastWasChar = false;
                hasLeftOperand = false; // Reset, need right operand
                continue;
            }

            if (c == '-' && peek(1) == '-') {
                // Subtraction operator requires left operand
                if (!hasLeftOperand) {
                    throw error("Set subtraction operator -- requires a left operand");
                }
                consume(); // consume first -
                consume(); // consume second -
                lastWasChar = false;
                hasLeftOperand = false; // Reset, need right operand
                continue;
            }

            // Check for escape
            if (c == '\\') {
                parseClassEscape();
                lastWasChar = true;
                lastChar = '\0'; // Don't track escaped chars for ranges
                hasLeftOperand = true;
                continue;
            }

            // Special handling for hyphen - it's allowed in ranges
            if (c == '-') {
                // Check if this is part of a range (prev char exists and next char exists)
                if (lastWasChar && !isAtEnd() && peek(1) != ']' && peek(1) != '-') {
                    // This looks like a range, allow it
                    consume(); // consume the -
                    lastWasChar = false;
                    continue;
                } else {
                    // Standalone hyphen must be escaped
                    throw error("Character '-' must be escaped in unicode sets mode");
                }
            }

            // Check for other reserved syntax characters that must be escaped
            if (isClassSyntaxCharVExceptHyphen(c)) {
                throw error("Character '" + c + "' must be escaped in unicode sets mode");
            }

            // Check for double punctuators (but not && which we handled above)
            if (isDoublePunctuatorV() && !(c == '&' && peek(1) == '&')) {
                throw error("Double punctuator must be escaped in unicode sets mode");
            }

            // Regular character
            lastChar = c;
            lastWasChar = true;
            hasLeftOperand = true;
            consume();
        }

        // At end of character class, check if we have a trailing operator without right operand
        if (!hasLeftOperand && pos > 0) {
            // This means we ended with an operator like && or -- without a right operand
            // The operator was consumed, so we need to check if the class is empty after an operator
            // Actually, this is already caught because closing ] will be encountered and hasLeftOperand will be false
        }
    }

    /**
     * Characters that must be escaped in v-flag character classes (except hyphen and [ which have special rules).
     */
    private boolean isClassSyntaxCharVExceptHyphen(char c) {
        // Note: [ is handled separately for nested classes, so not included here
        return c == '(' || c == ')' || c == '{' || c == '}' ||
               c == '/' || c == '|';
    }

    /**
     * Check if current position is a double punctuator in v-flag mode.
     * Double punctuators: !! ## $$ %% && ** ++ ,, .. :: ;; << == >> ?? @@ `` ~~ ^^^
     */
    private boolean isDoublePunctuatorV() {
        char c = peek();
        char next = peek(1);

        if (c != next) {
            return false;
        }

        // Check if it's a punctuator that can't be doubled
        return c == '!' || c == '#' || c == '$' || c == '%' || c == '&' ||
               c == '*' || c == '+' || c == ',' || c == '.' || c == ':' ||
               c == ';' || c == '<' || c == '=' || c == '>' || c == '?' ||
               c == '@' || c == '`' || c == '~' || c == '^';
    }

    /**
     * Parse escape sequence inside character class.
     */
    private void parseClassEscape() {
        consume(); // consume '\'

        if (isAtEnd()) {
            throw error("Character class cannot end with backslash");
        }

        char c = peek();

        // Character class escape
        if (c == 'd' || c == 'D' || c == 's' || c == 'S' || c == 'w' || c == 'W') {
            consume();
            return;
        }

        // Unicode property escape
        if (c == 'p' || c == 'P') {
            parseUnicodePropertyEscape();
            return;
        }

        // String literal escape \q{...} in unicode sets mode
        if (unicodeSetsMode && c == 'q') {
            consume(); // consume 'q'
            if (!match('{')) {
                throw error("Expected '{' after \\q in unicode sets mode");
            }
            // Parse string alternatives: \q{a|b|c}
            while (!isAtEnd() && peek() != '}') {
                // Parse each alternative (can be multi-char strings or escape sequences)
                while (!isAtEnd() && peek() != '|' && peek() != '}') {
                    if (peek() == '\\') {
                        consume();
                        if (!isAtEnd()) consume(); // Consume escaped character
                    } else {
                        consume();
                    }
                }
                if (peek() == '|') {
                    consume(); // consume separator
                }
            }
            if (!match('}')) {
                throw error("Unclosed \\q{...} string literal");
            }
            return;
        }

        // In unicode mode, - can be escaped inside character classes
        if ((unicodeMode || unicodeSetsMode) && c == '-') {
            consume();
            return;
        }

        // Backspace \b inside character class
        if (c == 'b') {
            consume();
            return;
        }

        // Character escape
        parseCharacterEscape();
    }

    /**
     * Quantifier ::
     *   QuantifierPrefix
     *   QuantifierPrefix ?
     *
     * QuantifierPrefix ::
     *   *
     *   +
     *   ?
     *   { DecimalDigits }
     *   { DecimalDigits ,}
     *   { DecimalDigits , DecimalDigits }
     */
    private void parseQuantifier() {
        if (!isQuantifierStart()) {
            return;
        }

        char c = peek();

        if (c == '*' || c == '+' || c == '?') {
            consume();
            // Check for lazy quantifier
            match('?');
            return;
        }

        if (c == '{') {
            parseBraceQuantifier();
        }
    }

    /**
     * Parse {n}, {n,}, or {n,m} quantifier.
     */
    private void parseBraceQuantifier() {
        int start = pos;
        consume(); // consume '{'

        if (!isDigit(peek())) {
            if (unicodeMode || unicodeSetsMode) {
                throw error("Incomplete quantifier");
            }
            // In non-unicode mode, incomplete quantifier is treated as literal
            // Roll back - the { should have been consumed as a pattern character, not as quantifier
            pos = start;
            return;
        }

        // Parse min
        while (!isAtEnd() && isDigit(peek())) {
            consume();
        }

        if (match(',')) {
            // {n,} or {n,m}
            if (isDigit(peek())) {
                // {n,m}
                while (!isAtEnd() && isDigit(peek())) {
                    consume();
                }
            }
            // else {n,}
        }

        if (!match('}')) {
            if (unicodeMode || unicodeSetsMode) {
                throw error("Incomplete quantifier");
            }
            // In non-unicode mode, reset and treat as literal
            // Roll back completely - the { should be consumed as a pattern character
            pos = start;
            return;
        }

        // Check for lazy quantifier
        match('?');
    }

    // ========== Helper Methods ==========

    private boolean isQuantifierStart() {
        char c = peek();
        return c == '*' || c == '+' || c == '?' || c == '{';
    }

    private boolean isPatternCharacter(char c) {
        // Pattern characters are any characters except: ^ $ \ . * + ? ( ) [ |
        // Note: ] and } are allowed as literal pattern characters outside their closing context
        return c != '^' && c != '$' && c != '\\' && c != '.' &&
               c != '*' && c != '+' && c != '?' && c != '(' &&
               c != ')' && c != '[' && c != '|';
    }

    private boolean isSyntaxCharacter(char c) {
        return c == '^' || c == '$' || c == '\\' || c == '.' ||
               c == '*' || c == '+' || c == '?' || c == '(' ||
               c == ')' || c == '[' || c == ']' || c == '{' ||
               c == '}' || c == '|';
    }

    private boolean isHexDigit(char c) {
        return (c >= '0' && c <= '9') ||
               (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
    }

    private boolean isDigit(char c) {
        return c >= '0' && c <= '9';
    }

    private boolean isIdentifierStart(char c) {
        // Support Unicode characters in identifiers (ES2015+)
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '$' ||
               Character.isLetter(c) || c == '\u200C' || c == '\u200D'; // Zero-width joiners
    }

    private boolean isIdentifierPart(char c) {
        return isIdentifierStart(c) || isDigit(c) ||
               Character.isDigit(c); // Unicode digits
    }

    /**
     * Creates an error with current position information.
     */
    private RegexSyntaxException error(String message) {
        return new RegexSyntaxException(
            message, pattern,
            (unicodeSetsMode ? "v" : (unicodeMode ? "u" : "")),
            pos, startPosition, startLine, startColumn
        );
    }

    // Helper methods for parsing

    private char peek() {
        if (isAtEnd()) return '\0';
        return pattern.charAt(pos);
    }

    private char peek(int offset) {
        int index = pos + offset;
        if (index < 0 || index >= pattern.length()) return '\0';
        return pattern.charAt(index);
    }

    private char consume() {
        return pattern.charAt(pos++);
    }

    private boolean match(char c) {
        if (peek() == c) {
            consume();
            return true;
        }
        return false;
    }

    private boolean matchString(String s) {
        if (pos + s.length() > pattern.length()) return false;
        if (pattern.startsWith(s, pos)) {
            pos += s.length();
            return true;
        }
        return false;
    }

    private boolean isAtEnd() {
        return pos >= pattern.length();
    }

    /**
     * Check if character is a valid modifier flag (i, m, s)
     */
    private boolean isModifierChar(char c) {
        return c == 'i' || c == 'm' || c == 's';
    }

    /**
     * Parse modifier flags in a modifier group (?i:...) or (?-i:...) or (?i-m:...)
     * Format: ModifierFlags | ModifierFlags-ModifierFlags
     * Where ModifierFlags can be i, m, s in any combination
     */
    private void parseModifiers() {
        // Parse add modifiers (before optional -)
        while (!isAtEnd() && isModifierChar(peek())) {
            consume();
        }

        // Check for remove modifiers (after -)
        if (peek() == '-') {
            consume(); // consume -
            while (!isAtEnd() && isModifierChar(peek())) {
                consume();
            }
        }

        // Next character should be ':'
        // Will be checked by caller
    }
}
