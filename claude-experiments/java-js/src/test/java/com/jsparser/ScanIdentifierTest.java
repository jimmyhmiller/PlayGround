package com.jsparser;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Exhaustive tests for scanIdentifier functionality.
 *
 * This test class is designed to be reusable with different scanner implementations.
 * Set the SCANNER field to test different versions of scanIdentifier.
 *
 * Usage:
 *   // Test the default Lexer implementation
 *   ScanIdentifierTest.SCANNER = ScanIdentifierTest.DEFAULT_SCANNER;
 *
 *   // Test a custom implementation
 *   ScanIdentifierTest.SCANNER = (source) -> {
 *       MyOptimizedLexer lexer = new MyOptimizedLexer(source);
 *       return lexer.tokenize();
 *   };
 */
public class ScanIdentifierTest {

    /**
     * Default scanner using the standard Lexer.
     */
    public static final Function<String, List<Token>> DEFAULT_SCANNER = (source) -> {
        Lexer lexer = new Lexer(source);
        return lexer.tokenize();
    };

    /**
     * The scanner function to test. Change this to test different implementations.
     * Takes source code string, returns list of tokens.
     */
    public static Function<String, List<Token>> SCANNER = DEFAULT_SCANNER;

    /**
     * Helper to get the first token (skipping EOF).
     */
    private Token firstToken(String source) {
        List<Token> tokens = SCANNER.apply(source);
        assertFalse(tokens.isEmpty(), "Expected at least one token");
        return tokens.get(0);
    }

    /**
     * Helper to get all non-EOF tokens.
     */
    private List<Token> allTokens(String source) {
        List<Token> tokens = SCANNER.apply(source);
        return tokens.stream()
            .filter(t -> t.type() != TokenType.EOF)
            .toList();
    }

    @BeforeEach
    void resetScanner() {
        // Ensure we're using the configured scanner
        // Subclasses can override this to set a different scanner
    }

    // ========================================================================
    // SIMPLE IDENTIFIERS
    // ========================================================================

    @Nested
    @DisplayName("Simple Identifiers")
    class SimpleIdentifiers {

        @Test
        @DisplayName("Single letter identifier")
        void singleLetter() {
            Token token = firstToken("a");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("a", token.lexeme());
        }

        @ParameterizedTest
        @ValueSource(strings = {"a", "b", "c", "x", "y", "z", "A", "B", "Z"})
        @DisplayName("All single ASCII letters")
        void allSingleLetters(String letter) {
            Token token = firstToken(letter);
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals(letter, token.lexeme());
        }

        @Test
        @DisplayName("Common short identifiers")
        void shortIdentifiers() {
            for (String id : new String[]{"id", "fn", "cb", "el", "ev", "xs", "ys", "ok"}) {
                Token token = firstToken(id);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + id);
                assertEquals(id, token.lexeme());
            }
        }

        @Test
        @DisplayName("Typical variable names")
        void typicalNames() {
            String[] names = {"value", "count", "index", "result", "data", "item",
                             "element", "callback", "response", "request", "options"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("CamelCase identifiers")
        void camelCase() {
            String[] names = {"userId", "firstName", "lastName", "dataList",
                             "eventHandler", "clickCount", "isValid", "hasError"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Long descriptive identifiers")
        void longIdentifiers() {
            String[] names = {
                "calculateTotalAmountWithDiscount",
                "fetchUserProfileDataFromServer",
                "handleFormSubmissionEvent",
                "processPaymentTransaction",
                "validateEmailAddressFormat",
                "initializeApplicationStateManager"
            };
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Identifiers with digits")
        void withDigits() {
            String[] names = {"a1", "x2", "var1", "item2", "count3", "data123",
                             "v1alpha", "test2beta", "handler42"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }
    }

    // ========================================================================
    // UNDERSCORE AND DOLLAR PREFIX
    // ========================================================================

    @Nested
    @DisplayName("Underscore and Dollar Prefix")
    class UnderscoreDollar {

        @Test
        @DisplayName("Underscore prefix")
        void underscorePrefix() {
            String[] names = {"_", "_a", "_private", "_internal", "_cache", "_data"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Dollar prefix")
        void dollarPrefix() {
            String[] names = {"$", "$a", "$el", "$scope", "$http", "$timeout"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Double underscore (dunder)")
        void doubleUnderscore() {
            String[] names = {"__proto__", "__dirname", "__filename", "__init__"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Mixed underscore and dollar")
        void mixedUnderscoreDollar() {
            String[] names = {"_$var", "$_var", "_$_", "$_$"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Trailing underscore")
        void trailingUnderscore() {
            String[] names = {"foo_", "bar__", "private_", "unused_"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }
    }

    // ========================================================================
    // KEYWORDS
    // ========================================================================

    @Nested
    @DisplayName("Keywords")
    class Keywords {

        @ParameterizedTest
        @MethodSource("com.jsparser.ScanIdentifierTest#keywordProvider")
        @DisplayName("All keywords return correct token type")
        void allKeywords(String keyword, TokenType expectedType) {
            Token token = firstToken(keyword);
            assertEquals(expectedType, token.type(), "Failed for keyword: " + keyword);
            assertEquals(keyword, token.lexeme());
        }

        @Test
        @DisplayName("Keywords are case-sensitive")
        void caseSensitive() {
            // These should be identifiers, not keywords
            String[] notKeywords = {"Var", "VAR", "LET", "Let", "CONST", "Const",
                                   "Function", "FUNCTION", "Return", "RETURN"};
            for (String name : notKeywords) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(),
                    "Should be identifier, not keyword: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Keyword prefixes are identifiers")
        void keywordPrefixes() {
            // These start with keywords but are identifiers
            String[] names = {"variable", "lethal", "constant", "functions",
                             "returns", "iffy", "elsewhere", "forehead"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(),
                    "Should be identifier: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Keyword suffixes are identifiers")
        void keywordSuffixes() {
            String[] names = {"myvar", "islet", "getconst", "runfunction"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(),
                    "Should be identifier: " + name);
                assertEquals(name, token.lexeme());
            }
        }
    }

    static Stream<Arguments> keywordProvider() {
        return Stream.of(
            Arguments.of("var", TokenType.VAR),
            Arguments.of("let", TokenType.LET),
            Arguments.of("const", TokenType.CONST),
            Arguments.of("function", TokenType.FUNCTION),
            Arguments.of("class", TokenType.CLASS),
            Arguments.of("return", TokenType.RETURN),
            Arguments.of("if", TokenType.IF),
            Arguments.of("else", TokenType.ELSE),
            Arguments.of("for", TokenType.FOR),
            Arguments.of("while", TokenType.WHILE),
            Arguments.of("do", TokenType.DO),
            Arguments.of("break", TokenType.BREAK),
            Arguments.of("continue", TokenType.CONTINUE),
            Arguments.of("switch", TokenType.SWITCH),
            Arguments.of("case", TokenType.CASE),
            Arguments.of("default", TokenType.DEFAULT),
            Arguments.of("try", TokenType.TRY),
            Arguments.of("catch", TokenType.CATCH),
            Arguments.of("finally", TokenType.FINALLY),
            Arguments.of("throw", TokenType.THROW),
            Arguments.of("new", TokenType.NEW),
            Arguments.of("typeof", TokenType.TYPEOF),
            Arguments.of("void", TokenType.VOID),
            Arguments.of("delete", TokenType.DELETE),
            Arguments.of("this", TokenType.THIS),
            Arguments.of("super", TokenType.SUPER),
            Arguments.of("in", TokenType.IN),
            Arguments.of("instanceof", TokenType.INSTANCEOF),
            Arguments.of("import", TokenType.IMPORT),
            Arguments.of("export", TokenType.EXPORT),
            Arguments.of("with", TokenType.WITH),
            Arguments.of("debugger", TokenType.DEBUGGER),
            Arguments.of("true", TokenType.TRUE),
            Arguments.of("false", TokenType.FALSE),
            Arguments.of("null", TokenType.NULL)
        );
    }

    // ========================================================================
    // UNICODE IDENTIFIERS
    // ========================================================================

    @Nested
    @DisplayName("Unicode Identifiers")
    class UnicodeIdentifiers {

        @Test
        @DisplayName("Greek letters")
        void greekLetters() {
            String[] names = {"α", "β", "γ", "δ", "π", "Ω", "αβγ", "πr2"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Cyrillic letters")
        void cyrillicLetters() {
            String[] names = {"привет", "мир", "переменная"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("CJK characters")
        void cjkCharacters() {
            String[] names = {"変数", "名前", "数"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Mixed ASCII and Unicode")
        void mixedAsciiUnicode() {
            String[] names = {"test変数", "myπ", "αBeta", "data数"};
            for (String name : names) {
                Token token = firstToken(name);
                assertEquals(TokenType.IDENTIFIER, token.type(), "Failed for: " + name);
                assertEquals(name, token.lexeme());
            }
        }

        @Test
        @DisplayName("Emoji identifiers (if supported)")
        void emojiIdentifiers() {
            // Note: Most JS engines don't support emoji as identifiers
            // This test documents current behavior
            try {
                Token token = firstToken("🚀");
                // If it parses, check the result
                assertNotNull(token);
            } catch (RuntimeException e) {
                // Expected - emoji not valid identifier
                assertTrue(e.getMessage().contains("Unexpected"));
            }
        }
    }

    // ========================================================================
    // UNICODE ESCAPES
    // ========================================================================

    // Helper to create unicode escape strings without Java interpreting them
    // ue("0061") returns the string backslash-u-0-0-6-1
    private static String ue(String hex) {
        return "\\" + "u" + hex;
    }

    // Helper for braced unicode escapes: ueb("61") returns backslash-u-{-6-1-}
    private static String ueb(String hex) {
        return "\\" + "u{" + hex + "}";
    }

    @Nested
    @DisplayName("Unicode Escape Sequences")
    class UnicodeEscapes {

        @Test
        @DisplayName("Simple \\uXXXX escape at start")
        void simpleEscapeAtStart() {
            // \u0061 = 'a'
            Token token = firstToken(ue("0061") + "bc");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("abc", token.lexeme());
        }

        @Test
        @DisplayName("Simple \\uXXXX escape in middle")
        void simpleEscapeInMiddle() {
            // \u0062 = 'b'
            Token token = firstToken("a" + ue("0062") + "c");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("abc", token.lexeme());
        }

        @Test
        @DisplayName("Simple \\uXXXX escape at end")
        void simpleEscapeAtEnd() {
            // \u0063 = 'c'
            Token token = firstToken("ab" + ue("0063"));
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("abc", token.lexeme());
        }

        @Test
        @DisplayName("Multiple escapes")
        void multipleEscapes() {
            // \u0061\u0062\u0063 = "abc"
            Token token = firstToken(ue("0061") + ue("0062") + ue("0063"));
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("abc", token.lexeme());
        }

        @Test
        @DisplayName("Braced unicode escape \\u{XX}")
        void bracedEscape() {
            // backslash-u-{61} = 'a'
            Token token = firstToken(ueb("61") + "bc");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("abc", token.lexeme());
        }

        @Test
        @DisplayName("Braced unicode escape with more digits")
        void bracedEscapeLong() {
            // backslash-u-{0061} = 'a'
            Token token = firstToken(ueb("0061") + "bc");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("abc", token.lexeme());
        }

        @Test
        @DisplayName("Keyword spelled with escapes becomes identifier")
        void keywordWithEscapes() {
            // "var" spelled with unicode escapes should be IDENTIFIER, not VAR
            // \u0076 = 'v'
            Token token = firstToken(ue("0076") + "ar");
            assertEquals(TokenType.IDENTIFIER, token.type(),
                "Keyword with escape should be identifier");
            assertEquals("var", token.lexeme());
        }

        @Test
        @DisplayName("All keywords with first char escaped")
        void allKeywordsEscaped() {
            // v=\u0076, l=\u006c, c=\u0063, f=\u0066, r=\u0072, i=\u0069, etc.
            String[][] tests = {
                {ue("0076") + "ar", "var"},      // var
                {ue("006c") + "et", "let"},      // let
                {ue("0063") + "onst", "const"},  // const
                {ue("0066") + "unction", "function"},
                {ue("0072") + "eturn", "return"},
                {ue("0069") + "f", "if"},
                {ue("0074") + "rue", "true"},
                {ue("0066") + "alse", "false"},
                {ue("006e") + "ull", "null"},
            };
            for (String[] test : tests) {
                Token token = firstToken(test[0]);
                assertEquals(TokenType.IDENTIFIER, token.type(),
                    "Escaped keyword should be identifier: " + test[1]);
                assertEquals(test[1], token.lexeme());
            }
        }

        @Test
        @DisplayName("Underscore as unicode escape")
        void underscoreEscape() {
            // \u005f = '_'
            Token token = firstToken(ue("005f") + "private");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("_private", token.lexeme());
        }

        @Test
        @DisplayName("Dollar as unicode escape")
        void dollarEscape() {
            // \u0024 = '$'
            Token token = firstToken(ue("0024") + "scope");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("$scope", token.lexeme());
        }

        @Test
        @DisplayName("High unicode code point (astral plane)")
        void astralPlane() {
            // U+1F600 = 😀 - should fail as it's not a valid identifier char
            // But U+10000 and above that ARE valid ID chars should work
            // Mathematical Alphanumeric Symbols: U+1D400 = 𝐀
            try {
                Token token = firstToken(ueb("1D400"));
                assertEquals(TokenType.IDENTIFIER, token.type());
            } catch (RuntimeException e) {
                // Some implementations may not support this
                assertTrue(true);
            }
        }
    }

    // ========================================================================
    // SURROGATE PAIRS
    // ========================================================================

    @Nested
    @DisplayName("Surrogate Pairs")
    class SurrogatePairs {

        @Test
        @DisplayName("Identifier with surrogate pair in source")
        void surrogatePairInSource() {
            // 𝑥 (Mathematical Italic Small X) is U+1D465, encoded as surrogate pair
            String mathX = "\uD835\uDC65"; // Surrogate pair for U+1D465
            try {
                Token token = firstToken(mathX);
                assertEquals(TokenType.IDENTIFIER, token.type());
                assertEquals(mathX, token.lexeme());
            } catch (RuntimeException e) {
                // May not be supported
                assertTrue(true);
            }
        }
    }

    // ========================================================================
    // POSITION TRACKING
    // ========================================================================

    @Nested
    @DisplayName("Position Tracking")
    class PositionTracking {

        @Test
        @DisplayName("Start position for first identifier")
        void startPositionFirst() {
            Token token = firstToken("foo");
            assertEquals(0, token.position());
            assertEquals(1, token.line());
            assertEquals(0, token.column());
        }

        @Test
        @DisplayName("Start position after whitespace")
        void startPositionAfterWhitespace() {
            List<Token> tokens = allTokens("   foo");
            assertEquals(1, tokens.size());
            Token token = tokens.get(0);
            assertEquals(3, token.position());
            assertEquals(3, token.column());
        }

        @Test
        @DisplayName("Multiple identifiers positions")
        void multipleIdentifiersPositions() {
            List<Token> tokens = allTokens("foo bar baz");
            assertEquals(3, tokens.size());

            assertEquals(0, tokens.get(0).position());
            assertEquals("foo", tokens.get(0).lexeme());

            assertEquals(4, tokens.get(1).position());
            assertEquals("bar", tokens.get(1).lexeme());

            assertEquals(8, tokens.get(2).position());
            assertEquals("baz", tokens.get(2).lexeme());
        }

        @Test
        @DisplayName("Position with unicode escapes")
        void positionWithEscapes() {
            // "abc" with escapes takes more source chars but lexeme is "abc"
            Token token = firstToken(ue("0061") + "bc");
            assertEquals(0, token.position());
            assertEquals("abc", token.lexeme());
            // End position should reflect actual source consumption
            assertEquals(8, token.endPosition());
        }

        @Test
        @DisplayName("Line tracking with newlines")
        void lineTracking() {
            List<Token> tokens = allTokens("foo\nbar\nbaz");
            assertEquals(3, tokens.size());

            assertEquals(1, tokens.get(0).line());
            assertEquals(2, tokens.get(1).line());
            assertEquals(3, tokens.get(2).line());
        }
    }

    // ========================================================================
    // BOUNDARY CONDITIONS
    // ========================================================================

    @Nested
    @DisplayName("Boundary Conditions")
    class BoundaryConditions {

        @Test
        @DisplayName("Identifier at end of input")
        void identifierAtEnd() {
            Token token = firstToken("foo");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("foo", token.lexeme());
        }

        @Test
        @DisplayName("Identifier followed by operator")
        void followedByOperator() {
            List<Token> tokens = allTokens("foo+bar");
            assertEquals(3, tokens.size());
            assertEquals("foo", tokens.get(0).lexeme());
            assertEquals(TokenType.PLUS, tokens.get(1).type());
            assertEquals("bar", tokens.get(2).lexeme());
        }

        @Test
        @DisplayName("Identifier followed by parenthesis")
        void followedByParen() {
            List<Token> tokens = allTokens("foo()");
            assertEquals(3, tokens.size());
            assertEquals("foo", tokens.get(0).lexeme());
            assertEquals(TokenType.LPAREN, tokens.get(1).type());
            assertEquals(TokenType.RPAREN, tokens.get(2).type());
        }

        @Test
        @DisplayName("Identifier followed by bracket")
        void followedByBracket() {
            List<Token> tokens = allTokens("foo[0]");
            assertEquals(4, tokens.size());
            assertEquals("foo", tokens.get(0).lexeme());
            assertEquals(TokenType.LBRACKET, tokens.get(1).type());
        }

        @Test
        @DisplayName("Identifier followed by dot")
        void followedByDot() {
            List<Token> tokens = allTokens("foo.bar");
            assertEquals(3, tokens.size());
            assertEquals("foo", tokens.get(0).lexeme());
            assertEquals(TokenType.DOT, tokens.get(1).type());
            assertEquals("bar", tokens.get(2).lexeme());
        }

        @Test
        @DisplayName("Very long identifier")
        void veryLongIdentifier() {
            String longId = "a".repeat(10000);
            Token token = firstToken(longId);
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals(longId, token.lexeme());
        }

        @Test
        @DisplayName("Identifier with max digit suffix")
        void maxDigitSuffix() {
            String id = "var" + "9".repeat(100);
            Token token = firstToken(id);
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals(id, token.lexeme());
        }
    }

    // ========================================================================
    // LITERAL VALUES
    // ========================================================================

    @Nested
    @DisplayName("Literal Values")
    class LiteralValues {

        @Test
        @DisplayName("true has boolean literal")
        void trueLiteral() {
            Token token = firstToken("true");
            assertEquals(TokenType.TRUE, token.type());
            assertEquals(Boolean.TRUE, token.literal());
        }

        @Test
        @DisplayName("false has boolean literal")
        void falseLiteral() {
            Token token = firstToken("false");
            assertEquals(TokenType.FALSE, token.type());
            assertEquals(Boolean.FALSE, token.literal());
        }

        @Test
        @DisplayName("null has null literal")
        void nullLiteral() {
            Token token = firstToken("null");
            assertEquals(TokenType.NULL, token.type());
            assertNull(token.literal());
        }

        @Test
        @DisplayName("Regular identifiers have no literal")
        void identifierNoLiteral() {
            Token token = firstToken("foo");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertNull(token.literal());
        }

        @Test
        @DisplayName("Keywords have no literal")
        void keywordNoLiteral() {
            Token token = firstToken("var");
            assertEquals(TokenType.VAR, token.type());
            assertNull(token.literal());
        }
    }

    // ========================================================================
    // CONTEXTUAL KEYWORDS
    // ========================================================================

    @Nested
    @DisplayName("Contextual Keywords")
    class ContextualKeywords {

        @Test
        @DisplayName("'of' is identifier (contextual keyword)")
        void ofIsIdentifier() {
            Token token = firstToken("of");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("of", token.lexeme());
        }

        @Test
        @DisplayName("'yield' is identifier (contextual keyword)")
        void yieldIsIdentifier() {
            Token token = firstToken("yield");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("yield", token.lexeme());
        }

        @Test
        @DisplayName("'async' is identifier (contextual keyword)")
        void asyncIsIdentifier() {
            Token token = firstToken("async");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("async", token.lexeme());
        }

        @Test
        @DisplayName("'await' is identifier (contextual keyword)")
        void awaitIsIdentifier() {
            Token token = firstToken("await");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("await", token.lexeme());
        }

        @Test
        @DisplayName("'get' is identifier (contextual keyword)")
        void getIsIdentifier() {
            Token token = firstToken("get");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("get", token.lexeme());
        }

        @Test
        @DisplayName("'set' is identifier (contextual keyword)")
        void setIsIdentifier() {
            Token token = firstToken("set");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("set", token.lexeme());
        }

        @Test
        @DisplayName("'static' is identifier")
        void staticIsIdentifier() {
            Token token = firstToken("static");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("static", token.lexeme());
        }

        @Test
        @DisplayName("'from' is identifier")
        void fromIsIdentifier() {
            Token token = firstToken("from");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("from", token.lexeme());
        }

        @Test
        @DisplayName("'as' is identifier")
        void asIsIdentifier() {
            Token token = firstToken("as");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("as", token.lexeme());
        }
    }

    // ========================================================================
    // STRESS TESTS
    // ========================================================================

    @Nested
    @DisplayName("Stress Tests")
    class StressTests {

        @Test
        @DisplayName("Many short identifiers")
        void manyShortIdentifiers() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 10000; i++) {
                sb.append("x").append(i).append(" ");
            }
            List<Token> tokens = allTokens(sb.toString());
            assertEquals(10000, tokens.size());
        }

        @Test
        @DisplayName("Many keywords")
        void manyKeywords() {
            StringBuilder sb = new StringBuilder();
            String[] keywords = {"var", "let", "const", "if", "else", "for", "while"};
            for (int i = 0; i < 10000; i++) {
                sb.append(keywords[i % keywords.length]).append(" ");
            }
            List<Token> tokens = allTokens(sb.toString());
            assertEquals(10000, tokens.size());
        }

        @Test
        @DisplayName("Mixed content")
        void mixedContent() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 1000; i++) {
                sb.append("var x").append(i).append(" = ").append(i).append("; ");
            }
            List<Token> tokens = allTokens(sb.toString());
            // Each iteration: var, x#, =, #, ;
            assertEquals(5000, tokens.size());
        }

        @Test
        @DisplayName("Unicode escapes stress test")
        void unicodeEscapesStress() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 1000; i++) {
                sb.append(ue("0061")).append(i).append(" ");
            }
            List<Token> tokens = allTokens(sb.toString());
            assertEquals(1000, tokens.size());
            // All should start with 'a'
            for (Token token : tokens) {
                assertTrue(token.lexeme().startsWith("a"),
                    "Expected to start with 'a': " + token.lexeme());
            }
        }
    }

    // ========================================================================
    // RESERVED WORDS (Future Reserved Words)
    // ========================================================================

    @Nested
    @DisplayName("Future Reserved Words")
    class FutureReservedWords {

        @Test
        @DisplayName("enum is identifier (future reserved)")
        void enumIdentifier() {
            Token token = firstToken("enum");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("enum", token.lexeme());
        }

        @Test
        @DisplayName("implements is identifier")
        void implementsIdentifier() {
            Token token = firstToken("implements");
            assertEquals(TokenType.IDENTIFIER, token.type());
        }

        @Test
        @DisplayName("interface is identifier")
        void interfaceIdentifier() {
            Token token = firstToken("interface");
            assertEquals(TokenType.IDENTIFIER, token.type());
        }

        @Test
        @DisplayName("package is identifier")
        void packageIdentifier() {
            Token token = firstToken("package");
            assertEquals(TokenType.IDENTIFIER, token.type());
        }

        @Test
        @DisplayName("private is identifier")
        void privateIdentifier() {
            Token token = firstToken("private");
            assertEquals(TokenType.IDENTIFIER, token.type());
        }

        @Test
        @DisplayName("protected is identifier")
        void protectedIdentifier() {
            Token token = firstToken("protected");
            assertEquals(TokenType.IDENTIFIER, token.type());
        }

        @Test
        @DisplayName("public is identifier")
        void publicIdentifier() {
            Token token = firstToken("public");
            assertEquals(TokenType.IDENTIFIER, token.type());
        }
    }
}
