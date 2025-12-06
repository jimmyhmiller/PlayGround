package com.jsparser;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

public class TestHangingFile2 {

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void testVSCodeWorkbench() throws Exception {
        Path file = Paths.get("../../beagle-vs-code-eval/.vscode-test/vscode-darwin-arm64-1.96.2/Visual Studio Code.app/Contents/Resources/app/out/vs/workbench/workbench.desktop.main.js");

        if (!Files.exists(file)) {
            System.out.println("File not found, skipping test");
            return;
        }

        String source = Files.readString(file);
        System.out.println("File size: " + source.length() + " chars");
        System.out.println("Starting parse...");

        long start = System.currentTimeMillis();
        Parser.parse(source, false);
        long end = System.currentTimeMillis();

        System.out.println("✓ Parsed in " + (end - start) + "ms");
    }
}
