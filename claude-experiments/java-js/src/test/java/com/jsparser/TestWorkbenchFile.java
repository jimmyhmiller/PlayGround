package com.jsparser;

import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Path;

public class TestWorkbenchFile {

    @Test
    public void testWorkbenchDesktopMain() throws Exception {
        String filepath = "/Users/jimmyhmiller/Documents/Code/PlayGround/beagle-vs-code-eval/.vscode-test/vscode-darwin-arm64-1.96.2/Visual Studio Code.app/Contents/Resources/app/out/vs/workbench/workbench.desktop.main.js";
        System.out.println("Testing: " + filepath);

        String source = Files.readString(Path.of(filepath));
        System.out.println("File size: " + source.length() + " bytes");

        try {
            Lexer lexer = new Lexer(source);
            System.out.println("Lexer created, starting tokenization...");
            var tokens = lexer.tokenize();
            System.out.println("SUCCESS: Tokenized " + tokens.size() + " tokens");
        } catch (Exception e) {
            System.out.println("FAILURE: " + e.getMessage());
            e.printStackTrace();

            // Try to get more context about the error
            if (e.getMessage().contains("position=")) {
                String msg = e.getMessage();
                int posIdx = msg.indexOf("position=");
                if (posIdx >= 0) {
                    String posStr = msg.substring(posIdx + 9).split(",")[0];
                    int pos = Integer.parseInt(posStr);
                    System.out.println("\nContext around error position " + pos + ":");
                    int start = Math.max(0, pos - 100);
                    int end = Math.min(source.length(), pos + 100);
                    System.out.println(source.substring(start, end));
                }
            }

            throw e;
        }
    }
}
