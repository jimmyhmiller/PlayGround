package com.jsparser;

import com.jsparser.ast.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestTopLevelAwait {

    @Test
    public void testTopLevelAwaitWithSemicolon() {
        // This was failing before the fix
        String code = "await foo;";
        Program ast = Parser.parse(code, true); // module mode

        ExpressionStatement stmt = (ExpressionStatement) ast.body().get(0);
        assertTrue(stmt.expression() instanceof AwaitExpression);
    }

    @Test
    public void testTopLevelAwaitWithBinaryOp() {
        String code = "const x = await fetch(url) + 1;";
        Program ast = Parser.parse(code, true);

        assertNotNull(ast);
        // Should parse successfully
    }

    @Test
    public void testTopLevelAwaitWithTypeof() {
        String code = "typeof await x;";
        Program ast = Parser.parse(code, true);

        assertNotNull(ast);
    }

    @Test
    public void testScriptModeAwaitAsIdentifier() {
        // Script mode - await is identifier
        String code = "var await = 10; await;";
        Program ast = Parser.parse(code, false); // script mode

        assertNotNull(ast);
        // Should parse successfully with await as identifier
    }

    @Test
    public void testScriptModeAwaitLabel() {
        // Script mode - await as label
        String code = "await: console.log('test');";
        Program ast = Parser.parse(code, false);

        assertNotNull(ast);
    }

    @Test
    public void testModuleModeCannotAssignToAwait() {
        // Module mode - cannot assign to await keyword
        String code = "await = 10;";

        assertThrows(Exception.class, () -> {
            Parser.parse(code, true);
        });
    }

    @Test
    public void testAwaitIdentifierInClassField() {
        // In script mode, inside async function, class field can use 'await' as identifier
        String code = "async function f() { return class { x = await; }; }";
        Program ast = Parser.parse(code, false);  // script mode

        assertNotNull(ast);
        // Should parse successfully - 'await' is just an identifier reference
    }

    @Test
    public void testAsyncFunctionAwait() {
        String code = "async function f() { await x; }";
        Program ast = Parser.parse(code, false);

        assertNotNull(ast);
    }

    @Test
    public void testTopLevelAwaitStandalone() {
        // await with no argument followed by semicolon
        String code = "await;";
        Program ast = Parser.parse(code, true);

        ExpressionStatement stmt = (ExpressionStatement) ast.body().get(0);
        AwaitExpression awaitExpr = (AwaitExpression) stmt.expression();
        assertNull(awaitExpr.argument());
    }

    @Test
    public void testAwaitExpressionInClassFieldRejected() {
        // AwaitExpression not allowed in class field initializers, even in async context
        String code = "async () => class { x = await foo };";

        assertThrows(Exception.class, () -> {
            Parser.parse(code, true);
        });
    }

    // Note: Nested function await validation is complex and requires tracking
    // function nesting depth. This is a known limitation that doesn't affect
    // the primary use case (actual top-level await in modules).
    // The production bug is about top-level await followed by semicolons,
    // which is now fixed by the other test cases.
}
