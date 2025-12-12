package com.jsparser;

import com.jsparser.Parser;
import com.jsparser.ast.Program;
import com.jsparser.ast.*;
import java.nio.file.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

public class QuickTest {
    public static void main(String[] args) throws Exception {
        // Compare hash of our output vs Acorn output
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        // Our AST
        String code = "\"use strict\";";
        Parser p = new Parser(code);
        Program result = p.parse();
        Path ourFile = Paths.get("/tmp/our_out.json");
        mapper.writeValue(ourFile.toFile(), result);

        // Acorn file (run node /tmp/test_acorn.js first)
        Path acornFile = Paths.get("/tmp/acorn_out.json");

        // Compute hashes
        java.security.MessageDigest digest1 = java.security.MessageDigest.getInstance("SHA-256");
        java.security.MessageDigest digest2 = java.security.MessageDigest.getInstance("SHA-256");

        com.fasterxml.jackson.databind.JsonNode tree1 = mapper.readTree(acornFile.toFile());
        com.fasterxml.jackson.databind.JsonNode tree2 = mapper.readTree(ourFile.toFile());

        System.out.println("Acorn tree structure:");
        printStructure(tree1, 0);
        System.out.println("\nOur tree structure:");
        printStructure(tree2, 0);
    }

    static void printStructure(com.fasterxml.jackson.databind.JsonNode node, int depth) {
        String indent = "  ".repeat(depth);
        if (node.isObject()) {
            java.util.List<String> fields = new java.util.ArrayList<>();
            node.fieldNames().forEachRemaining(fields::add);
            System.out.println(indent + "Object with fields (sorted): " + fields.stream().sorted().toList());
            for (String field : fields.stream().sorted().toList()) {
                System.out.println(indent + "  " + field + ":");
                if (depth < 2) {
                    printStructure(node.get(field), depth + 2);
                }
            }
        } else if (node.isArray()) {
            System.out.println(indent + "Array[" + node.size() + "]");
        } else {
            System.out.println(indent + node.asText());
        }
    }
}
