package com.jsparser;

import com.jsparser.ast.*;
import org.junit.jupiter.api.Test;

public class TestTemplateExpressionColumn {

    @Test
    public void testTemplateExpressionColumnAfterBackslash() throws Exception {
        String source = "const x = 1;\nconst y = `\\\n${x}`;";

        System.out.println("=== Source ===");
        for (int i = 0; i < source.length(); i++) {
            char c = source.charAt(i);
            if (c == '\n') {
                System.out.println("\\n");
            } else {
                System.out.print(c);
            }
        }
        System.out.println("\n");

        Program ast = Parser.parse(source, false);

        // Navigate to the template literal expression
        VariableDeclaration varDecl = (VariableDeclaration) ast.body().get(1);
        VariableDeclarator declarator = varDecl.declarations().get(0);
        TemplateLiteral template = (TemplateLiteral) declarator.init();
        Expression expr = template.expressions().get(0);

        System.out.println("Expression: " + expr);
        System.out.println("Expression loc: " + expr.loc());
        System.out.println("Expression loc.start: " + expr.loc().start());
        System.out.println("Expression loc.start.column: " + expr.loc().start().column());
    }
}
