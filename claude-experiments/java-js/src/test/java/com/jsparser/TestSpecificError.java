package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class TestSpecificError {

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

        // Navigate to: .body[1].expression.right.callee.body.body[8].body.body[4].consequent.cases[4].consequent[0].argument
        @SuppressWarnings("unchecked")
        java.util.List<Object> body = (java.util.List<Object>) javaObj.get("body");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> stmt = (java.util.Map<String, Object>) body.get(1);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> expr = (java.util.Map<String, Object>) stmt.get("expression");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> right = (java.util.Map<String, Object>) expr.get("right");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> callee = (java.util.Map<String, Object>) right.get("callee");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> calleeBody = (java.util.Map<String, Object>) callee.get("body");
        @SuppressWarnings("unchecked")
        java.util.List<Object> calleeBodyBody = (java.util.List<Object>) calleeBody.get("body");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> ifStmt = (java.util.Map<String, Object>) calleeBodyBody.get(8);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> ifBody = (java.util.Map<String, Object>) ifStmt.get("body");
        @SuppressWarnings("unchecked")
        java.util.List<Object> ifBodyBody = (java.util.List<Object>) ifBody.get("body");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> switchStmt = (java.util.Map<String, Object>) ifBodyBody.get(4);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> consequent = (java.util.Map<String, Object>) switchStmt.get("consequent");
        @SuppressWarnings("unchecked")
        java.util.List<Object> cases = (java.util.List<Object>) consequent.get("cases");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> case4 = (java.util.Map<String, Object>) cases.get(4);
        @SuppressWarnings("unchecked")
        java.util.List<Object> case4Consequent = (java.util.List<Object>) case4.get("consequent");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> returnStmt = (java.util.Map<String, Object>) case4Consequent.get(0);
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> argument = (java.util.Map<String, Object>) returnStmt.get("argument");

        System.out.println("Java result:");
        System.out.println("  type: " + argument.get("type"));
        System.out.println("  end: " + argument.get("end"));

        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> loc = (java.util.Map<String, Object>) argument.get("loc");
        @SuppressWarnings("unchecked")
        java.util.Map<String, Object> locEnd = (java.util.Map<String, Object>) loc.get("end");
        System.out.println("  loc.end.line: " + locEnd.get("line"));
        System.out.println("  loc.end.column: " + locEnd.get("column"));

        System.out.println("\nExpected from Acorn:");
        System.out.println("  loc.end.line: 136");
        System.out.println("  loc.end.column: 63");
        System.out.println("  end: 5037");
    }
}
