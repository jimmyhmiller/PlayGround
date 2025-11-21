package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.module.paramnames.ParameterNamesModule;
import com.jsparser.ast.Program;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Oracle parser that uses Node.js and esprima to generate reference ASTs
 */
public class OracleParser {
    private static final ObjectMapper mapper = new ObjectMapper()
            .registerModule(new ParameterNamesModule());
    private static final Path ORACLE_SCRIPT = Path.of("src/test/resources/oracle-parser.js");

    public static Program parse(String source) throws Exception {
        // Write source to temp file
        Path tempFile = Files.createTempFile("js-source", ".js");
        Files.writeString(tempFile, source);

        try {
            // Run Node.js with esprima to parse
            ProcessBuilder pb = new ProcessBuilder(
                "node",
                ORACLE_SCRIPT.toString(),
                tempFile.toString()
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream())
            );

            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new RuntimeException("Oracle parser failed: " + output);
            }

            // Parse JSON output into our AST
            return mapper.readValue(output.toString(), Program.class);
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }
}
