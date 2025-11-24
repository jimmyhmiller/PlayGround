package com.jsparser;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class CommentAnnotationTest {

    @Test
    public void testPureAnnotation() {
        String code = "const x = /* @__PURE__ */ new Map();";
        Parser parser = new Parser(code);
        try {
            Object result = parser.parse();
            System.out.println("✓ Parsed @__PURE__ annotation");
            assertNotNull(result);
        } catch (Exception e) {
            System.out.println("✗ Failed: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    @Test
    public void testLicenseComment() {
        String code = "/**\n * @license React\n * react.production.js\n */\nvar x = 1;";
        Parser parser = new Parser(code);
        try {
            Object result = parser.parse();
            System.out.println("✓ Parsed @license comment");
            assertNotNull(result);
        } catch (Exception e) {
            System.out.println("✗ Failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    public void testNoCapture() {
        String code = "const map = /* @__PURE__ */new Map();";
        Parser parser = new Parser(code);
        try {
            Object result = parser.parse();
            System.out.println("✓ Parsed @__PURE__ without space");
            assertNotNull(result);
        } catch (Exception e) {
            System.out.println("✗ Failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    public void testAtSymbolInString() {
        String code = "var e = \"react@19.2.0\";";
        Parser parser = new Parser(code);
        try {
            Object result = parser.parse();
            System.out.println("✓ Parsed @ in simple string");
            assertNotNull(result);
        } catch (Exception e) {
            System.out.println("✗ Failed: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    @Test
    public void testAtSymbolInObjectKey() {
        String code = "var e={\"react@19.2.0\":(e,t)=>{}};";
        Parser parser = new Parser(code);
        try {
            Object result = parser.parse();
            System.out.println("✓ Parsed @ in object key");
            assertNotNull(result);
        } catch (Exception e) {
            System.out.println("✗ Failed: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    @Test
    public void testRegexAfterMethodCall() {
        String code = "var x = String(e).replace(/([A-Za-z]:)?([#!\\\"$&'()*,:;<=>?@\\[\\\\\\]^`{|}])/g,\"$1\\\\$2\");";
        Parser parser = new Parser(code);
        try {
            Object result = parser.parse();
            System.out.println("✓ Parsed regex after method call with @ in character class");
            assertNotNull(result);
        } catch (Exception e) {
            System.out.println("✗ Failed: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    @Test
    public void testMinimalRegexAt() throws IOException {
        String code = Files.readString(Paths.get("/tmp/test-regex-at.js"));
        System.out.println("Code to parse: " + code);
        Parser parser = new Parser(code);
        try {
            Object result = parser.parse();
            System.out.println("✓ Parsed minimal regex with @");
            assertNotNull(result);
        } catch (Exception e) {
            System.out.println("✗ Failed: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    @Test
    public void testRegexWithBackslashLookbehind() throws IOException {
        String code = Files.readString(Paths.get("/tmp/test-regex-backslash.js"));
        System.out.println("Code to parse: " + code);
        Parser parser = new Parser(code);
        try {
            Object result = parser.parse();
            System.out.println("✓ Parsed regex with backslash lookbehind");
            assertNotNull(result);
        } catch (Exception e) {
            System.out.println("✗ Failed: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    // Removed testNextJsChunk - file was truncated/incomplete JavaScript that Acorn also rejects

    @Test
    public void testActualFailingFile() throws IOException {
        String path = "../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/next-server/server.runtime.prod.js";
        if (!Files.exists(Paths.get(path))) {
            System.out.println("⚠ File not found, skipping test");
            return;
        }

        String code = Files.readString(Paths.get(path));
        System.out.println("File size: " + code.length() + " characters");

        Parser parser = new Parser(code);
        try {
            Object result = parser.parse();
            System.out.println("✓ Parsed server.runtime.prod.js");
            assertNotNull(result);
        } catch (Exception e) {
            System.out.println("✗ Failed at: " + e.getMessage());

            // Try to extract context around the error
            String msg = e.getMessage();
            if (msg != null && msg.contains("position=")) {
                try {
                    int pos = Integer.parseInt(msg.substring(msg.indexOf("position=") + 9).split("[,\\]]")[0]);
                    int start = Math.max(0, pos - 50);
                    int end = Math.min(code.length(), pos + 50);
                    System.out.println("Context: ..." + code.substring(start, end) + "...");
                } catch (Exception ex) {
                    // ignore
                }
            }
            throw e;
        }
    }
}
