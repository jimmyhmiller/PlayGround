package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.*;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DebugReactPosition {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    public void debugFirstMismatch() throws Exception {
        String filePath = "../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/react-experimental/cjs/react.development.js";
        String source = Files.readString(Paths.get(filePath));

        // The first mismatch is at:
        // .body[1].expression.right.callee.body.body[2].body.body[2].expression.right.loc.end.column: 66 != 65
        // Let's navigate to that node

        Program ast = Parser.parse(source, false);

        // body[1]
        ExpressionStatement stmt1 = (ExpressionStatement) ast.body().get(1);
        // .expression.right.callee.body.body[2].body.body[2].expression.right
        AssignmentExpression assign = (AssignmentExpression) stmt1.expression();
        CallExpression call = (CallExpression) assign.right();
        FunctionExpression func = (FunctionExpression) call.callee();
        BlockStatement block = func.body();

        // body[2]
        BlockStatement innerBlock2 = (BlockStatement) block.body().get(2);
        // body[2]
        ExpressionStatement exprStmt = (ExpressionStatement) innerBlock2.body().get(2);
        AssignmentExpression innerAssign = (AssignmentExpression) exprStmt.expression();

        Expression right = innerAssign.right();

        System.out.println("Expression type: " + right.getClass().getSimpleName());
        System.out.println("Start: " + right.start());
        System.out.println("End: " + right.end());
        System.out.println("Loc: " + right.loc());
        System.out.println("Loc end column: " + right.loc().end().column());

        // Let's also print the source text at this position
        int start = right.start();
        int end = right.end();
        String text = source.substring(start, Math.min(end, start + 200));
        System.out.println("\nSource text:");
        System.out.println(text);

        // Check what Acorn says
        String acornJson = parseWithAcorn(source, filePath);
        if (acornJson != null) {
            Object acornObj = mapper.readValue(acornJson, Object.class);
            // Navigate to the same node in Acorn's AST
            java.util.Map<?,?> acornAst = (java.util.Map<?,?>) acornObj;
            java.util.List<?> body = (java.util.List<?>) acornAst.get("body");
            java.util.Map<?,?> stmt = (java.util.Map<?,?>) body.get(1);
            java.util.Map<?,?> expr = (java.util.Map<?,?>) stmt.get("expression");
            java.util.Map<?,?> rightExpr = (java.util.Map<?,?>) expr.get("right");
            java.util.Map<?,?> callee = (java.util.Map<?,?>) rightExpr.get("callee");
            java.util.Map<?,?> funcBody = (java.util.Map<?,?>) callee.get("body");
            java.util.List<?> funcBodyBody = (java.util.List<?>) funcBody.get("body");
            java.util.Map<?,?> block2 = (java.util.Map<?,?>) funcBodyBody.get(2);
            java.util.List<?> block2Body = (java.util.List<?>) block2.get("body");
            java.util.Map<?,?> exprStmt2 = (java.util.Map<?,?>) block2Body.get(2);
            java.util.Map<?,?> exprStmtExpr = (java.util.Map<?,?>) exprStmt2.get("expression");
            java.util.Map<?,?> rightNode = (java.util.Map<?,?>) exprStmtExpr.get("right");
            java.util.Map<?,?> loc = (java.util.Map<?,?>) rightNode.get("loc");
            java.util.Map<?,?> locEnd = (java.util.Map<?,?>) loc.get("end");

            System.out.println("\nAcorn says:");
            System.out.println("Start: " + rightNode.get("start"));
            System.out.println("End: " + rightNode.get("end"));
            System.out.println("Loc end column: " + locEnd.get("column"));
        }
    }

    private String parseWithAcorn(String source, String filePath) throws Exception {
        Path tempFile = Files.createTempFile("acorn-ast-", ".json");
        try {
            String[] cmd = {
                    "node",
                    "-e",
                    "const acorn = require('acorn'); " +
                            "const fs = require('fs'); " +
                            "const source = fs.readFileSync(process.argv[1], 'utf-8'); " +
                            "try { const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: 'script'}); " +
                            "fs.writeFileSync(process.argv[2], JSON.stringify(ast, (k,v) => typeof v === 'bigint' ? null : v)); " +
                            "process.exit(0); } catch(e) { console.error(e.message); process.exit(1); }",
                    filePath,
                    tempFile.toAbsolutePath().toString()
            };

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                return null;
            }

            return Files.readString(tempFile);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }
}
