package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.nio.file.Files;
import java.nio.file.Paths;

public class CompareAST {
    public static void main(String[] args) throws Exception {
        String sourceFile = args[0];
        String outputFile = args.length > 1 ? args[1] : "/tmp/our-ast.json";
        
        System.out.println("Parsing: " + sourceFile);
        String source = Files.readString(Paths.get(sourceFile));
        System.out.println("Source size: " + source.length() + " chars");
        
        long start = System.currentTimeMillis();
        var ast = Parser.parse(source, true); // module mode
        System.out.println("Parsed in " + (System.currentTimeMillis() - start) + "ms");
        
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        
        start = System.currentTimeMillis();
        String json = mapper.writeValueAsString(ast);
        System.out.println("Serialized in " + (System.currentTimeMillis() - start) + "ms");
        System.out.println("AST JSON size: " + json.length() + " chars");
        
        Files.writeString(Paths.get(outputFile), json);
        System.out.println("Written to: " + outputFile);
    }
}
