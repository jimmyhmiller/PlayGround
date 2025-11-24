package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import java.nio.file.Files;
import java.nio.file.Paths;

public class CompareHelper {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: CompareHelper <file-path> <is-module>");
            System.exit(1);
        }

        String filePath = args[0];
        boolean isModule = Boolean.parseBoolean(args[1]);

        String source = Files.readString(Paths.get(filePath));
        Program ast = Parser.parse(source, isModule);

        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(ast);
        System.out.println(json);
    }
}
