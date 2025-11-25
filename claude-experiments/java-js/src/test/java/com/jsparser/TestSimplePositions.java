package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.*;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class TestSimplePositions {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    public void testSimplePositions() throws Exception {
        String source = "var x = function() {\n  return \"hello\";\n};";

        System.out.println("=== Source ===");
        System.out.println(source);
        System.out.println();

        // Parse with Java
        Program ast = Parser.parse(source, false);
        VariableDeclaration varDecl = (VariableDeclaration) ast.body().get(0);
        VariableDeclarator declarator = varDecl.declarations().get(0);
        FunctionExpression func = (FunctionExpression) declarator.init();
        BlockStatement block = func.body();
        ReturnStatement returnStmt = (ReturnStatement) block.body().get(0);
        Literal literal = (Literal) returnStmt.argument();

        System.out.println("=== Java Parser ===");
        System.out.println("Function start: " + func.start() + ", end: " + func.end());
        System.out.println("Block start: " + block.start() + ", end: " + block.end());
        System.out.println("Return statement start: " + returnStmt.start() + ", end: " + returnStmt.end());
        System.out.println("Literal start: " + literal.start() + ", end: " + literal.end());
        System.out.println("Literal loc: " + literal.loc());

        // Parse with Acorn
        String acornJson = parseWithAcorn(source);
        if (acornJson != null) {
            @SuppressWarnings("unchecked")
            Map<String, Object> acornAst = mapper.readValue(acornJson, Map.class);
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> body = (List<Map<String, Object>>) acornAst.get("body");
            Map<String, Object> varDeclMap = body.get(0);
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> declarations = (List<Map<String, Object>>) varDeclMap.get("declarations");
            Map<String, Object> declaratorMap = declarations.get(0);
            Map<String, Object> funcMap = (Map<String, Object>) declaratorMap.get("init");
            Map<String, Object> blockMap = (Map<String, Object>) funcMap.get("body");
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> blockBody = (List<Map<String, Object>>) blockMap.get("body");
            Map<String, Object> returnMap = blockBody.get(0);
            Map<String, Object> literalMap = (Map<String, Object>) returnMap.get("argument");

            System.out.println("\n=== Acorn ===");
            System.out.println("Function start: " + funcMap.get("start") + ", end: " + funcMap.get("end"));
            System.out.println("Block start: " + blockMap.get("start") + ", end: " + blockMap.get("end"));
            System.out.println("Return statement start: " + returnMap.get("start") + ", end: " + returnMap.get("end"));
            System.out.println("Literal start: " + literalMap.get("start") + ", end: " + literalMap.get("end"));
            System.out.println("Literal loc: " + literalMap.get("loc"));
        }
    }

    private String parseWithAcorn(String source) throws Exception {
        Path tempSource = Files.createTempFile("test-source-", ".js");
        Path tempFile = Files.createTempFile("acorn-ast-", ".json");
        Files.writeString(tempSource, source);

        try {
            String[] cmd = {
                    "node",
                    "-e",
                    "const acorn = require('acorn'); " +
                            "const fs = require('fs'); " +
                            "const source = fs.readFileSync(process.argv[1], 'utf-8'); " +
                            "const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: 'script'}); " +
                            "fs.writeFileSync(process.argv[2], JSON.stringify(ast, (k,v) => typeof v === 'bigint' ? null : v));",
                    tempSource.toAbsolutePath().toString(),
                    tempFile.toAbsolutePath().toString()
            };

            ProcessBuilder pb = new ProcessBuilder(cmd);
            Process process = pb.start();

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                return null;
            }

            return Files.readString(tempFile);
        } finally {
            Files.deleteIfExists(tempFile);
            Files.deleteIfExists(tempSource);
        }
    }
}
