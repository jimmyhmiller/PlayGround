import * as ts from 'typescript';

type PrimitiveType = { kind: 'primitive'; name: 'number' | 'string' | 'boolean' | 'void' };
type Type = PrimitiveType;

const NUMBER_TYPE: Type = { kind: 'primitive', name: 'number' };
const STRING_TYPE: Type = { kind: 'primitive', name: 'string' };
const BOOLEAN_TYPE: Type = { kind: 'primitive', name: 'boolean' };

const typesEqual = (a: Type, b: Type): boolean => {
  if (a.kind !== b.kind) return false;
  if (a.kind === 'primitive' && b.kind === 'primitive') return a.name === b.name;
  return false;
};


function infer(node: ts.Node): Type {
  switch (node.kind) {
    case ts.SyntaxKind.NumericLiteral: 
      return NUMBER_TYPE;
    case ts.SyntaxKind.StringLiteral: 
      return STRING_TYPE;
    case ts.SyntaxKind.TrueKeyword:
    case ts.SyntaxKind.FalseKeyword: 
      return BOOLEAN_TYPE;
    default:
      throw new Error(`Unsupported node kind: ${ts.SyntaxKind[node.kind]}`);
  }
}

function check(node: ts.Node, expectedType: Type): void {
  const inferredType = infer(node);
  if (!typesEqual(inferredType, expectedType)) {
    throw new Error(`Expected ${expectedType}, got ${inferredType}`);
  }
};


// Simple program checker - handles exactly one statement
function checkSimpleProgram(sourceFile: ts.SourceFile, expectedType: Type): void {
  // Only handle programs with exactly one statement
  if (sourceFile.statements.length !== 1) {
    throw new Error(`Expected exactly 1 statement, got ${sourceFile.statements.length}`);
  }
  
  const statement = sourceFile.statements[0];
  
  // Only handle expression statements for now
  if (!ts.isExpressionStatement(statement)) {
    throw new Error(`Only expression statements supported, got ${ts.SyntaxKind[statement.kind]}`);
  }
  
  // Use bidirectional checking: check the expression against expected type
  check(statement.expression, expectedType);
}

function parseAndCheckSimpleProgram(code: string, type: Type) {
  const sourceFile = ts.createSourceFile('temp.ts', code, ts.ScriptTarget.Latest, true);
  checkSimpleProgram(sourceFile, type);
}

function main() {
  parseAndCheckSimpleProgram("2", NUMBER_TYPE);
  parseAndCheckSimpleProgram("true", BOOLEAN_TYPE);
  parseAndCheckSimpleProgram("'hello'", STRING_TYPE);
}
main()