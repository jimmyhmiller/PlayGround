package com.jsparser;

import org.junit.jupiter.api.Test;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class DiagnosticStringBugTest {

    private static final String FAILING_FILE =
        "../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/@vercel/nft/index.js";

    @Test
    public void binarySearchForFailurePoint() throws IOException {
        String source = Files.readString(Path.of(FAILING_FILE));

        // We know failure occurs around position 96751
        // Work backwards to find the root cause
        int knownFailure = 96751;
        int searchStart = 0;

        System.out.println("Starting binary search from 0 to " + knownFailure);
        int failurePoint = binarySearch(source, searchStart, knownFailure);

        System.out.println("\n=== FAILURE POINT FOUND ===");
        System.out.println("Position: " + failurePoint);
        printContext(source, failurePoint, 100);
        analyzeEscapes(source, failurePoint);
    }

    private int binarySearch(String source, int start, int end) {
        if (end - start <= 100) {
            // Narrow enough, do linear search
            for (int i = start; i < end; i += 10) {
                if (!canParse(source.substring(0, i))) {
                    System.out.println("Found failure at position: " + i);
                    return i;
                }
            }
            return end;
        }

        int mid = (start + end) / 2;
        String chunk = source.substring(0, mid);

        if (canParse(chunk)) {
            System.out.println("Position " + mid + " parses OK, searching " + mid + " to " + end);
            return binarySearch(source, mid, end);
        } else {
            System.out.println("Position " + mid + " FAILS, searching " + start + " to " + mid);
            return binarySearch(source, start, mid);
        }
    }

    private boolean canParse(String chunk) {
        try {
            new Lexer(chunk).tokenize();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private void printContext(String source, int pos, int range) {
        int start = Math.max(0, pos - range);
        int end = Math.min(source.length(), pos + range);

        System.out.println("\nContext (showing " + range + " chars before and after):");
        System.out.println("BEFORE: " + escape(source.substring(start, pos)));
        System.out.println("AT: [" + escape(source.substring(pos, Math.min(pos + 1, end))) + "]");
        System.out.println("AFTER: " + escape(source.substring(Math.min(pos + 1, end), end)));
    }

    private void analyzeEscapes(String source, int pos) {
        // Look backwards for unclosed string
        int searchBack = Math.max(0, pos - 1000);
        String searchArea = source.substring(searchBack, pos);

        System.out.println("\n=== ESCAPE SEQUENCE ANALYSIS ===");

        // Find all escape sequences
        for (int i = searchArea.length() - 1; i >= 0; i--) {
            if (searchArea.charAt(i) == '\\') {
                int absPos = searchBack + i;
                String escapeSeq = extractEscapeSequence(source, absPos);
                System.out.println("Position " + absPos + ": \\" + escapeSeq);
            }
        }

        // Find unclosed quotes
        int lastQuote = -1;
        char quoteChar = '\0';
        for (int i = searchArea.length() - 1; i >= 0; i--) {
            char c = searchArea.charAt(i);
            if (c == '"' || c == '\'' || c == '`') {
                lastQuote = searchBack + i;
                quoteChar = c;
                break;
            }
        }

        if (lastQuote >= 0) {
            System.out.println("\nLast quote found: '" + quoteChar + "' at position " + lastQuote);
            printContext(source, lastQuote, 50);
        }
    }

    private String extractEscapeSequence(String source, int pos) {
        if (pos >= source.length() - 1) return "";

        char next = source.charAt(pos + 1);
        switch (next) {
            case 'u':
                if (pos + 2 < source.length() && source.charAt(pos + 2) == '{') {
                    // Unicode escape: \\u{...}
                    int end = source.indexOf('}', pos + 2);
                    if (end > 0) {
                        return source.substring(pos + 1, end + 1);
                    }
                } else if (pos + 5 < source.length()) {
                    // Unicode escape: \\uXXXX
                    return source.substring(pos + 1, pos + 6);
                }
                return "u???";
            case 'x':
                if (pos + 3 < source.length()) {
                    return source.substring(pos + 1, pos + 4);
                }
                return "x??";
            case '0': case '1': case '2': case '3':
            case '4': case '5': case '6': case '7':
                // Octal - up to 3 digits
                int end = pos + 1;
                while (end < Math.min(source.length(), pos + 4) &&
                       source.charAt(end) >= '0' && source.charAt(end) <= '7') {
                    end++;
                }
                return source.substring(pos + 1, end);
            default:
                return String.valueOf(next);
        }
    }

    private String escape(String s) {
        return s.replace("\\", "\\\\")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
                .replace("\"", "\\\"");
    }
}
