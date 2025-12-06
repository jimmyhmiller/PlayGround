package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class CheckASTStructure {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void checkOurAST() throws Exception {
        Path sourceFile = Paths.get("test-oracles/test262/test/built-ins/decodeURIComponent/S15.1.3.2_A1.1_T1.js");
        String source = Files.readString(sourceFile);

        Program program = Parser.parse(source, false);
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(program);

        System.out.println("=== Our AST (first 800 chars) ===");
        System.out.println(json.substring(0, Math.min(800, json.length())));
    }
}
