package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.jsparser.ast.Program;

import java.nio.file.Files;
import java.nio.file.Paths;

public class ParseOne {
    public static void main(String[] args) throws Exception {
        String jsFile = args[0];
        String source = Files.readString(Paths.get(jsFile));
        
        boolean isModule = Parser.hasModuleFlag(source) || jsFile.endsWith("_FIXTURE.js");
        Program program = Parser.parse(source, isModule);
        
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        String json = mapper.writeValueAsString(program);
        System.out.println(json);
    }
}
