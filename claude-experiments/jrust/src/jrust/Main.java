package jrust;

import jrust.ast.Program;
import jrust.codegen.JvmCodegen;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: jrust <file.jrs>");
            System.exit(1);
        }

        String inputFile = args[0];
        String source = Files.readString(Path.of(inputFile));
        String outputDir = "output";

        // Lex
        Lexer lexer = new Lexer(source);
        List<Token> tokens = lexer.tokenize();

        // Parse
        Parser parser = new Parser(tokens);
        Program program = parser.parse();

        // Codegen
        JvmCodegen codegen = new JvmCodegen(program, outputDir);
        codegen.generate();

        System.out.println("Compiled to " + outputDir + "/");

        // Run - include asm.jar and build dir on classpath for JRustAsm access
        String classpath = outputDir + ":asm.jar:build";
        List<String> cmd = new java.util.ArrayList<>();
        cmd.add("java");
        cmd.add("-cp");
        cmd.add(classpath);
        cmd.add("Main");
        // Forward remaining args (after the input file)
        for (int i = 1; i < args.length; i++) {
            cmd.add(args[i]);
        }
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();
        try {
            Process proc = pb.start();
            int exitCode = proc.waitFor();
            System.exit(exitCode);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.exit(1);
        }
    }
}
