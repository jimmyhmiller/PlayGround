package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestEndsWithN {
    @Test
    public void testEndsWithN() {
        String lexeme = "1n";
        System.out.println("Lexeme: '" + lexeme + "'");
        System.out.println("Length: " + lexeme.length());
        System.out.println("Ends with 'n': " + lexeme.endsWith("n"));
        System.out.println("Last char: '" + lexeme.charAt(lexeme.length() - 1) + "'");
        System.out.println("Last char code: " + (int)lexeme.charAt(lexeme.length() - 1));
    }
}
