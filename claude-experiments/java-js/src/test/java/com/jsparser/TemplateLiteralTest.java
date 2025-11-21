package com.jsparser;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class TemplateLiteralTest {

    @Test
    void testTemplateLiteralInFunctionCall() {
        // This is exact content from the file - template literal with spaces in interpolation
        String source = "test(`${ dateTimeString }${ zoneString }`, components);";

        System.out.println("Source: " + source);

        try {
            Lexer lexer = new Lexer(source);
            List<Token> tokens = lexer.tokenize();

            System.out.println("\nTokens:");
            for (Token token : tokens) {
                System.out.println("  " + token);
            }

            System.out.println("\nSUCCESS! Total tokens: " + tokens.size());
        } catch (Exception e) {
            System.out.println("\nERROR at position: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }
}
