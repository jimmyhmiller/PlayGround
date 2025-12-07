package com.jsparser;

import com.jsparser.ast.Expression;
import com.jsparser.ast.Statement;
import java.util.function.Function;

/**
 * Metadata for JavaScript operators used in table-driven precedence parsing.
 * Contains precedence level, associativity, and expression type information.
 */
public class OperatorInfo {

    /**
     * Functional interface for statement parsers.
     * Used in table-driven statement dispatch.
     */
    @FunctionalInterface
    public interface StatementParser {
        /**
         * Parse a statement starting at current token position.
         * @return The parsed statement, or null if this parser doesn't handle the current context
         */
        Statement parse();
    }

    /**
     * Functional interface for prefix expression parsers (NUD in Pratt parsing).
     * Handles tokens that start an expression (literals, identifiers, unary operators, etc.)
     */
    @FunctionalInterface
    public interface PrefixHandler {
        /**
         * Parse a prefix expression starting with the given token.
         * @param parser The parser instance
         * @param token The token that triggered this handler (already consumed)
         * @return The parsed expression
         */
        Expression parse(Parser parser, Token token);
    }

    /**
     * Functional interface for infix/postfix expression parsers (LED in Pratt parsing).
     * Handles tokens that appear after an expression (binary operators, member access, etc.)
     */
    @FunctionalInterface
    public interface InfixHandler {
        /**
         * Parse an infix expression with the given left operand.
         * @param parser The parser instance
         * @param left The left-hand side expression (already parsed)
         * @param token The operator token (already consumed)
         * @return The combined expression
         */
        Expression parse(Parser parser, Expression left, Token token);
    }

    /**
     * Binding power for Pratt parsing.
     * Left binding power determines when this operator takes precedence.
     * Right binding power determines the minimum precedence for the right operand.
     */
    public record BindingPower(int left, int right) {
        /**
         * Creates left-associative binding power.
         * Right BP is higher, so same-precedence operators on the right won't bind.
         * Example: a + b + c = (a + b) + c
         */
        public static BindingPower left(int bp) {
            return new BindingPower(bp, bp + 1);
        }

        /**
         * Creates right-associative binding power.
         * Right BP equals left, so same-precedence operators on the right will bind.
         * Example: a = b = c = a = (b = c)
         */
        public static BindingPower right(int bp) {
            return new BindingPower(bp, bp);
        }

        /**
         * Creates postfix binding power (no right operand).
         * Used for postfix operators like x++, member access, function calls.
         */
        public static BindingPower postfix(int bp) {
            return new BindingPower(bp, 0);
        }
    }

    /**
     * Combines an infix handler with its binding power.
     * Used in the INFIX_HANDLERS table for Pratt parsing.
     */
    public record InfixOp(BindingPower bp, InfixHandler handler) {
        public int lbp() { return bp.left(); }
        public int rbp() { return bp.right(); }

        public static InfixOp left(int bp, InfixHandler handler) {
            return new InfixOp(BindingPower.left(bp), handler);
        }

        public static InfixOp right(int bp, InfixHandler handler) {
            return new InfixOp(BindingPower.right(bp), handler);
        }

        public static InfixOp postfix(int bp, InfixHandler handler) {
            return new InfixOp(BindingPower.postfix(bp), handler);
        }
    }
    /**
     * Operator associativity - determines parsing direction.
     * LEFT: a + b + c = (a + b) + c
     * RIGHT: a = b = c = a = (b = c)
     */
    public enum Associativity {
        LEFT,
        RIGHT
    }

    /**
     * Expression type - determines which AST node to create.
     */
    public enum ExpressionType {
        BINARY,           // BinaryExpression (arithmetic, bitwise, comparison)
        LOGICAL,          // LogicalExpression (||, &&, ??)
        ASSIGNMENT,       // AssignmentExpression (=, +=, etc.)
        CONDITIONAL,      // ConditionalExpression (? :)
        UPDATE            // UpdateExpression (++, --)
    }

    private final int precedence;
    private final Associativity associativity;
    private final ExpressionType expressionType;

    /**
     * Creates operator metadata.
     *
     * @param precedence Higher values = tighter binding (e.g., * has higher precedence than +)
     * @param associativity LEFT or RIGHT
     * @param expressionType Type of AST node to create
     */
    public OperatorInfo(int precedence, Associativity associativity, ExpressionType expressionType) {
        this.precedence = precedence;
        this.associativity = associativity;
        this.expressionType = expressionType;
    }

    public int precedence() {
        return precedence;
    }

    public Associativity associativity() {
        return associativity;
    }

    public ExpressionType expressionType() {
        return expressionType;
    }

    // Helper methods for common checks

    public boolean isLeftAssociative() {
        return associativity == Associativity.LEFT;
    }

    public boolean isRightAssociative() {
        return associativity == Associativity.RIGHT;
    }

    public boolean isBinaryExpression() {
        return expressionType == ExpressionType.BINARY;
    }

    public boolean isLogicalExpression() {
        return expressionType == ExpressionType.LOGICAL;
    }

    public boolean isAssignmentExpression() {
        return expressionType == ExpressionType.ASSIGNMENT;
    }
}
