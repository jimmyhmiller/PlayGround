package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

/**
 * Base interface for all ESTree AST nodes
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
    @JsonSubTypes.Type(value = Program.class, name = "Program"),
    @JsonSubTypes.Type(value = Identifier.class, name = "Identifier"),
    @JsonSubTypes.Type(value = Literal.class, name = "Literal"),
    @JsonSubTypes.Type(value = ExpressionStatement.class, name = "ExpressionStatement"),
    @JsonSubTypes.Type(value = BinaryExpression.class, name = "BinaryExpression"),
    @JsonSubTypes.Type(value = AssignmentExpression.class, name = "AssignmentExpression"),
    @JsonSubTypes.Type(value = MemberExpression.class, name = "MemberExpression"),
    @JsonSubTypes.Type(value = CallExpression.class, name = "CallExpression"),
    @JsonSubTypes.Type(value = ArrayExpression.class, name = "ArrayExpression"),
    @JsonSubTypes.Type(value = ObjectExpression.class, name = "ObjectExpression"),
    @JsonSubTypes.Type(value = NewExpression.class, name = "NewExpression"),
    @JsonSubTypes.Type(value = VariableDeclaration.class, name = "VariableDeclaration"),
    @JsonSubTypes.Type(value = BlockStatement.class, name = "BlockStatement"),
    @JsonSubTypes.Type(value = ReturnStatement.class, name = "ReturnStatement"),
    @JsonSubTypes.Type(value = UnaryExpression.class, name = "UnaryExpression"),
    @JsonSubTypes.Type(value = LogicalExpression.class, name = "LogicalExpression"),
    @JsonSubTypes.Type(value = UpdateExpression.class, name = "UpdateExpression"),
    @JsonSubTypes.Type(value = ConditionalExpression.class, name = "ConditionalExpression"),
    @JsonSubTypes.Type(value = IfStatement.class, name = "IfStatement"),
    @JsonSubTypes.Type(value = WhileStatement.class, name = "WhileStatement"),
    @JsonSubTypes.Type(value = DoWhileStatement.class, name = "DoWhileStatement"),
    @JsonSubTypes.Type(value = ForStatement.class, name = "ForStatement"),
    @JsonSubTypes.Type(value = ForInStatement.class, name = "ForInStatement"),
    @JsonSubTypes.Type(value = ForOfStatement.class, name = "ForOfStatement"),
    @JsonSubTypes.Type(value = BreakStatement.class, name = "BreakStatement"),
    @JsonSubTypes.Type(value = ContinueStatement.class, name = "ContinueStatement"),
    @JsonSubTypes.Type(value = FunctionDeclaration.class, name = "FunctionDeclaration"),
    @JsonSubTypes.Type(value = FunctionExpression.class, name = "FunctionExpression"),
    @JsonSubTypes.Type(value = ArrowFunctionExpression.class, name = "ArrowFunctionExpression"),
    @JsonSubTypes.Type(value = TemplateLiteral.class, name = "TemplateLiteral"),
    @JsonSubTypes.Type(value = TemplateElement.class, name = "TemplateElement"),
    @JsonSubTypes.Type(value = ThisExpression.class, name = "ThisExpression"),
    @JsonSubTypes.Type(value = ObjectPattern.class, name = "ObjectPattern"),
    @JsonSubTypes.Type(value = ArrayPattern.class, name = "ArrayPattern"),
    @JsonSubTypes.Type(value = ClassBody.class, name = "ClassBody"),
    @JsonSubTypes.Type(value = MethodDefinition.class, name = "MethodDefinition"),
    @JsonSubTypes.Type(value = PropertyDefinition.class, name = "PropertyDefinition"),
    @JsonSubTypes.Type(value = ClassDeclaration.class, name = "ClassDeclaration"),
    @JsonSubTypes.Type(value = ClassExpression.class, name = "ClassExpression"),
    @JsonSubTypes.Type(value = Super.class, name = "Super"),
    @JsonSubTypes.Type(value = MetaProperty.class, name = "MetaProperty"),
    @JsonSubTypes.Type(value = ImportDeclaration.class, name = "ImportDeclaration"),
    @JsonSubTypes.Type(value = ImportSpecifier.class, name = "ImportSpecifier"),
    @JsonSubTypes.Type(value = ImportDefaultSpecifier.class, name = "ImportDefaultSpecifier"),
    @JsonSubTypes.Type(value = ImportNamespaceSpecifier.class, name = "ImportNamespaceSpecifier"),
    @JsonSubTypes.Type(value = ExportNamedDeclaration.class, name = "ExportNamedDeclaration"),
    @JsonSubTypes.Type(value = ExportDefaultDeclaration.class, name = "ExportDefaultDeclaration"),
    @JsonSubTypes.Type(value = ExportAllDeclaration.class, name = "ExportAllDeclaration"),
    @JsonSubTypes.Type(value = ExportSpecifier.class, name = "ExportSpecifier"),
    @JsonSubTypes.Type(value = ImportExpression.class, name = "ImportExpression"),
    @JsonSubTypes.Type(value = ThrowStatement.class, name = "ThrowStatement"),
    @JsonSubTypes.Type(value = TryStatement.class, name = "TryStatement"),
    @JsonSubTypes.Type(value = CatchClause.class, name = "CatchClause"),
    @JsonSubTypes.Type(value = WithStatement.class, name = "WithStatement"),
    @JsonSubTypes.Type(value = DebuggerStatement.class, name = "DebuggerStatement"),
    @JsonSubTypes.Type(value = EmptyStatement.class, name = "EmptyStatement"),
    @JsonSubTypes.Type(value = LabeledStatement.class, name = "LabeledStatement"),
    @JsonSubTypes.Type(value = SwitchStatement.class, name = "SwitchStatement"),
    @JsonSubTypes.Type(value = SwitchCase.class, name = "SwitchCase"),
    @JsonSubTypes.Type(value = Property.class, name = "Property"),
    @JsonSubTypes.Type(value = SpreadElement.class, name = "SpreadElement"),
    @JsonSubTypes.Type(value = SequenceExpression.class, name = "SequenceExpression"),
})
public sealed interface Node permits
    Program,
    Statement,
    Expression,
    Pattern,
    TemplateElement,
    Property,
    ClassBody,
    MethodDefinition,
    PropertyDefinition,
    StaticBlock,
    ImportAttribute,
    ImportSpecifier,
    ImportDefaultSpecifier,
    ImportNamespaceSpecifier,
    ExportSpecifier,
    CatchClause,
    SwitchCase {

    String type();
    int start();
    int end();
    SourceLocation loc();
}
