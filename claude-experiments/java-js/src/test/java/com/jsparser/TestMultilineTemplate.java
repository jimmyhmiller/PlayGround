package com.jsparser;

import com.jsparser.ast.*;
import org.junit.jupiter.api.Test;

public class TestMultilineTemplate {

    @Test
    public void testMultilineTemplateColumns() throws Exception {
        String source = "const checksum = \"abc\";\n" +
                       "const content = \"xyz\";\n" +
                       "return `\\\n" +
                       "${content}\n" +
                       "module.exports.__checksum = ${JSON.stringify(checksum)}\n" +
                       "`;";

        System.out.println("=== Source ===");
        System.out.println(source);
        System.out.println();

        Program ast = Parser.parse(source, false);

        // Navigate to the template literal's second expression
        ReturnStatement returnStmt = (ReturnStatement) ast.body().get(2);
        TemplateLiteral template = (TemplateLiteral) returnStmt.argument();
        Expression expr1 = template.expressions().get(1); // Second expression (JSON.stringify)

        System.out.println("Second expression (JSON.stringify): " + expr1);
        System.out.println("Location: " + expr1.loc());
        System.out.println("Start column: " + expr1.loc().start().column());
    }
}
