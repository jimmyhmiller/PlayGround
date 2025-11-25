package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class TestReactSpecificNode {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    @Test
    public void testSpecificNode() throws Exception {
        String filePath = "../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/react-experimental/cjs/react.development.js";
        String source = Files.readString(Paths.get(filePath));

        Program program = Parser.parse(source, false);
        String javaJson = mapper.writeValueAsString(program);

        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> javaObj = mapper.readValue(javaJson, java.util.Map.class);

        // Navigate to: .body[1].expression.right.callee.body.body[2].body.body[2].expression.right
        @SuppressWarnings("unchecked")
        java.util.List<Object> body = (java.util.List<Object>) javaObj.get("body");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> body1 = (java.util.Map<String, Object>) body.get(1);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> expr = (java.util.Map<String, Object>) body1.get("expression");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> right = (java.util.Map<String, Object>) expr.get("right");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> callee = (java.util.Map<String, Object>) right.get("callee");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> calleeBody = (java.util.Map<String, Object>) callee.get("body");
        @SuppressWarnings("unchecked")
        java.util.List<Object> calleeBodyBody = (java.util.List<Object>) calleeBody.get("body");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> stmt2 = (java.util.Map<String, Object>) calleeBodyBody.get(2);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> stmt2Body = (java.util.Map<String, Object>) stmt2.get("body");
        @SuppressWarnings("unchecked")
        java.util.List<Object> stmt2BodyBody = (java.util.List<Object>) stmt2Body.get("body");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> exprStmt = (java.util.Map<String, Object>) stmt2BodyBody.get(2);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> exprStmtExpr = (java.util.Map<String, Object>) exprStmt.get("expression");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> target = (java.util.Map<String, Object>) exprStmtExpr.get("right");

        System.out.println("Target node from Java:");
        System.out.println("  type: " + target.get("type"));
        System.out.println("  start: " + target.get("start"));
        System.out.println("  end: " + target.get("end"));

        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> loc = (java.util.Map<String, Object>) target.get("loc");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> locEnd = (java.util.Map<String, Object>) loc.get("end");
        System.out.println("  loc.end.column: " + locEnd.get("column"));

        Integer start = (Integer) target.get("start");
        Integer end = (Integer) target.get("end");
        System.out.println("  substring: " + source.substring(start, end));

        System.out.println("\nExpected from Acorn:");
        System.out.println("  end: 1732");
        System.out.println("  loc.end.column: 66");
    }
}
