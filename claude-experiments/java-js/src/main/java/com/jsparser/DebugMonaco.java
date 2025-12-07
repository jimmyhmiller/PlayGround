package com.jsparser;

import com.jsparser.ast.*;
import com.fasterxml.jackson.databind.*;
import java.nio.file.*;

public class DebugMonaco {
    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        
        // Read cache
        String cacheContent = Files.readString(Paths.get("test-oracles/adhoc-cache/..__ai-dashboard2_node_modules_monaco-editor_min_vs_editor.api-CalNCsUg.js.json"));
        JsonNode cacheRoot = mapper.readTree(cacheContent);
        JsonNode expectedAst = cacheRoot.has("ast") ? cacheRoot.get("ast") : cacheRoot;
        
        // Read source
        String source = Files.readString(Paths.get("../ai-dashboard2/node_modules/monaco-editor/min/vs/editor.api-CalNCsUg.js"));
        
        // Parse with Java
        Program actual = Parser.parse(source, true);
        String actualJson = mapper.writeValueAsString(actual);
        JsonNode actualAst = mapper.readTree(actualJson);
        
        // Find first difference - check body items
        JsonNode expectedBody = expectedAst.get("body");
        JsonNode actualBody = actualAst.get("body");
        
        System.out.println("Expected body size: " + expectedBody.size());
        System.out.println("Actual body size: " + actualBody.size());
        
        for (int i = 0; i < Math.min(expectedBody.size(), actualBody.size()); i++) {
            String expectedItem = mapper.writeValueAsString(expectedBody.get(i));
            String actualItem = mapper.writeValueAsString(actualBody.get(i));
            if (!expectedItem.equals(actualItem)) {
                System.out.println("\nFirst difference at body[" + i + "]:");
                System.out.println("Expected type: " + expectedBody.get(i).get("type"));
                System.out.println("Actual type: " + actualBody.get(i).get("type"));
                
                // Find character difference
                int minLen = Math.min(expectedItem.length(), actualItem.length());
                int diffPos = 0;
                for (int j = 0; j < minLen; j++) {
                    if (expectedItem.charAt(j) != actualItem.charAt(j)) {
                        diffPos = j;
                        break;
                    }
                }
                
                // Print around difference
                int start = Math.max(0, diffPos - 100);
                int end = Math.min(minLen, diffPos + 200);
                System.out.println("\nDiff position: " + diffPos);
                System.out.println("Expected around diff: " + expectedItem.substring(start, end));
                System.out.println("Actual around diff:   " + actualItem.substring(start, end));
                break;
            }
        }
    }
}
