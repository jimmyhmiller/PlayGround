import * as parser from "@babel/parser";
import traverse, { NodePath } from "@babel/traverse";
import generate from "@babel/generator";
import * as t from "@babel/types";
import * as path from "path";

interface ImportBinding {
  source: string;
  type: "named" | "default" | "namespace";
  imported?: string;
}

// Check if an import is external (bare module specifier, not relative)
function isExternalImport(source: string): boolean {
  return !source.startsWith(".") && !source.startsWith("/");
}

// Check if an expression is a require() call and return the source if external
function getExternalRequireSource(node: t.Expression | null | undefined): string | null {
  if (!node) return null;

  // Handle: require('module')
  if (
    t.isCallExpression(node) &&
    t.isIdentifier(node.callee) &&
    node.callee.name === "require" &&
    node.arguments.length === 1 &&
    t.isStringLiteral(node.arguments[0])
  ) {
    const source = node.arguments[0].value;
    return isExternalImport(source) ? source : null;
  }

  return null;
}

interface LocalBinding {
  type: "function" | "variable" | "class";
  preserve: boolean;
}

interface TransformOptions {
  filename: string;
  sourceRoot?: string;
}

// Check if a node is a call to our API function (once or defonce)
function isApiCall(node: t.Node, fnName: string): node is t.CallExpression {
  return (
    t.isCallExpression(node) &&
    t.isIdentifier(node.callee) &&
    node.callee.name === fnName
  );
}

// Unwrap defonce(value) -> value
function unwrapDefonce(node: t.Expression | null | undefined): t.Expression {
  if (node && isApiCall(node, "defonce") && node.arguments.length > 0) {
    return node.arguments[0] as t.Expression;
  }
  return node || t.identifier("undefined");
}

// Check if initializer is wrapped in defonce()
function isDefonce(node: t.Expression | null | undefined): boolean {
  return !!node && isApiCall(node, "defonce");
}

export function transform(code: string, options: TransformOptions): string {
  const { filename, sourceRoot = process.cwd() } = options;
  const moduleId = resolveModuleId(filename, sourceRoot);

  const ast = parser.parse(code, {
    sourceType: "module",
    plugins: ["jsx", "typescript"],
    attachComment: true,
  });

  // Track imports
  const imports = new Map<string, ImportBinding>();

  // Track API imports to skip them
  const apiImports = new Set<string>();

  // Track external import identifiers to skip in second pass
  const externalImportIds = new Set<string>();

  // Track top-level local bindings
  const locals = new Map<string, LocalBinding>();

  // Track once() counter
  let onceCounter = 0;

  // First pass: collect imports and top-level declarations
  traverse(ast, {
    ImportDeclaration(nodePath: NodePath<t.ImportDeclaration>) {
      const sourceValue = nodePath.node.source.value;

      // Check if this is importing from our API
      const isApiImport = sourceValue === "hot-reload" ||
                          sourceValue === "hot-reload/api" ||
                          sourceValue.endsWith("/api");

      // Check if this is an external import (node_modules, built-ins like 'electron')
      const isExternal = isExternalImport(sourceValue);

      const source = resolveImportSource(sourceValue, filename, sourceRoot);

      // Only track local imports for hot-reload transformation
      // External imports are left as regular require() calls
      if (!isExternal && !isApiImport) {
        for (const specifier of nodePath.node.specifiers) {
          if (t.isImportSpecifier(specifier)) {
            const imported = t.isIdentifier(specifier.imported)
              ? specifier.imported.name
              : specifier.imported.value;

            imports.set(specifier.local.name, {
              source,
              type: "named",
              imported,
            });
          } else if (t.isImportDefaultSpecifier(specifier)) {
            imports.set(specifier.local.name, {
              source,
              type: "default",
            });
          } else if (t.isImportNamespaceSpecifier(specifier)) {
            imports.set(specifier.local.name, {
              source,
              type: "namespace",
            });
          }
        }
      }

      // Track API imports for removal
      if (isApiImport) {
        for (const specifier of nodePath.node.specifiers) {
          if (t.isImportSpecifier(specifier)) {
            const imported = t.isIdentifier(specifier.imported)
              ? specifier.imported.name
              : specifier.imported.value;
            if (imported === "once" || imported === "defonce") {
              apiImports.add(specifier.local.name);
            }
          }
        }
      }

      // Transform imports based on type
      if (isApiImport) {
        // Remove API imports entirely
        nodePath.remove();
      } else if (isExternal) {
        // Convert external imports to regular require() - leave them alone
        // Transform: import { x } from 'pkg' -> const { x } = require('pkg')
        // Transform: import x from 'pkg' -> const x = require('pkg').default || require('pkg')
        // Transform: import * as x from 'pkg' -> const x = require('pkg')
        const specifiers = nodePath.node.specifiers;

        // Track all identifiers from external imports so we don't transform them
        for (const spec of specifiers) {
          externalImportIds.add(spec.local.name);
        }

        if (specifiers.length === 0) {
          // Side-effect import: import 'pkg'
          nodePath.replaceWith(
            t.expressionStatement(
              t.callExpression(t.identifier("require"), [t.stringLiteral(sourceValue)])
            )
          );
        } else {
          const requireCall = t.callExpression(t.identifier("require"), [t.stringLiteral(sourceValue)]);
          const declarations: t.VariableDeclarator[] = [];

          // Group specifiers by type
          const namedSpecifiers: t.ImportSpecifier[] = [];
          let defaultSpecifier: t.ImportDefaultSpecifier | null = null;
          let namespaceSpecifier: t.ImportNamespaceSpecifier | null = null;

          for (const spec of specifiers) {
            if (t.isImportSpecifier(spec)) {
              namedSpecifiers.push(spec);
            } else if (t.isImportDefaultSpecifier(spec)) {
              defaultSpecifier = spec;
            } else if (t.isImportNamespaceSpecifier(spec)) {
              namespaceSpecifier = spec;
            }
          }

          // Handle namespace import: import * as x from 'pkg'
          if (namespaceSpecifier) {
            declarations.push(
              t.variableDeclarator(namespaceSpecifier.local, requireCall)
            );
          }

          // Handle default import: import x from 'pkg'
          if (defaultSpecifier) {
            // Use: const x = require('pkg').default ?? require('pkg')
            // to handle both ESM default exports and CJS modules
            const req = namespaceSpecifier
              ? t.identifier(namespaceSpecifier.local.name)
              : requireCall;
            declarations.push(
              t.variableDeclarator(
                defaultSpecifier.local,
                t.logicalExpression(
                  "??",
                  t.memberExpression(
                    namespaceSpecifier ? t.identifier(namespaceSpecifier.local.name) : t.callExpression(t.identifier("require"), [t.stringLiteral(sourceValue)]),
                    t.identifier("default")
                  ),
                  namespaceSpecifier ? t.identifier(namespaceSpecifier.local.name) : t.callExpression(t.identifier("require"), [t.stringLiteral(sourceValue)])
                )
              )
            );
          }

          // Handle named imports: import { x, y } from 'pkg'
          if (namedSpecifiers.length > 0) {
            const properties: t.ObjectProperty[] = namedSpecifiers.map(spec => {
              const imported = t.isIdentifier(spec.imported)
                ? spec.imported
                : t.identifier(spec.imported.value);
              return t.objectProperty(
                imported,
                spec.local,
                false,
                imported.name === spec.local.name
              );
            });

            const req = namespaceSpecifier
              ? t.identifier(namespaceSpecifier.local.name)
              : requireCall;
            declarations.push(
              t.variableDeclarator(t.objectPattern(properties), req)
            );
          }

          nodePath.replaceWith(
            t.variableDeclaration("const", declarations)
          );
        }
      } else if (nodePath.node.specifiers.length === 0) {
        // Side-effect import for local module
        nodePath.replaceWith(
          t.expressionStatement(
            t.callExpression(
              t.memberExpression(t.identifier("__hot"), t.identifier("require")),
              [t.stringLiteral(source)]
            )
          )
        );
      } else {
        // Local import - remove (references will be replaced with __hot.get())
        nodePath.remove();
      }
    },

    FunctionDeclaration(nodePath: NodePath<t.FunctionDeclaration>) {
      if (nodePath.parent === ast.program && nodePath.node.id) {
        locals.set(nodePath.node.id.name, { type: "function", preserve: false });
      }
    },

    VariableDeclaration(nodePath: NodePath<t.VariableDeclaration>) {
      if (nodePath.parent === ast.program) {
        for (const declarator of nodePath.node.declarations) {
          // Check for CommonJS require of external module: const x = require('module')
          // Also handles destructuring: const { a, b } = require('module')
          const externalSource = getExternalRequireSource(declarator.init);
          if (externalSource) {
            // Track all identifiers from this declaration as external
            if (t.isIdentifier(declarator.id)) {
              externalImportIds.add(declarator.id.name);
            } else if (t.isObjectPattern(declarator.id)) {
              for (const prop of declarator.id.properties) {
                if (t.isObjectProperty(prop) && t.isIdentifier(prop.value)) {
                  externalImportIds.add(prop.value.name);
                } else if (t.isRestElement(prop) && t.isIdentifier(prop.argument)) {
                  externalImportIds.add(prop.argument.name);
                }
              }
            }
            continue;
          }

          if (t.isIdentifier(declarator.id)) {
            const name = declarator.id.name;

            // Skip external import identifiers (they're regular require() now)
            if (externalImportIds.has(name)) continue;

            // Check for defonce() call or let keyword
            const hasDefonce = isDefonce(declarator.init);
            const shouldPreserve = hasDefonce || nodePath.node.kind === "let";

            locals.set(name, {
              type: "variable",
              preserve: shouldPreserve,
            });
          }
        }
      }
    },

    ClassDeclaration(nodePath: NodePath<t.ClassDeclaration>) {
      if (nodePath.parent === ast.program && nodePath.node.id) {
        locals.set(nodePath.node.id.name, { type: "class", preserve: false });
      }
    },
  });

  // Second pass: transform declarations and replace references
  traverse(ast, {
    Identifier(nodePath: NodePath<t.Identifier>) {
      const name = nodePath.node.name;

      // Skip API function references (once, defonce)
      if (apiImports.has(name)) return;

      // Skip external import identifiers (they're regular require() now)
      if (externalImportIds.has(name)) return;

      const importBinding = imports.get(name);
      const localBinding = locals.get(name);

      if (!importBinding && !localBinding) return;

      const parent = nodePath.parent;

      // For locals, also transform assignment targets (e.g., mainWindow = x -> __m.mainWindow = x)
      // Check this BEFORE isReferencedIdentifier() to avoid TypeScript narrowing issues
      if (localBinding && t.isAssignmentExpression(parent) && parent.left === nodePath.node) {
        // Transform assignment target
        nodePath.replaceWith(
          t.memberExpression(t.identifier("__m"), t.identifier(name))
        );
        return;
      }

      // For imports, only transform referenced identifiers (can't assign to imports)
      if (!nodePath.isReferencedIdentifier()) return;
      if (
        t.isMemberExpression(parent) &&
        parent.property === nodePath.node &&
        !parent.computed
      ) {
        return;
      }

      if (
        (t.isFunctionDeclaration(parent) || t.isClassDeclaration(parent)) &&
        (parent as any).id === nodePath.node
      ) {
        return;
      }

      if (t.isVariableDeclarator(parent) && parent.id === nodePath.node) {
        return;
      }

      let replacement: t.Expression;

      if (importBinding) {
        if (importBinding.type === "namespace") {
          replacement = t.callExpression(
            t.memberExpression(t.identifier("__hot"), t.identifier("get")),
            [t.stringLiteral(importBinding.source)]
          );
        } else if (importBinding.type === "default") {
          replacement = t.memberExpression(
            t.callExpression(
              t.memberExpression(t.identifier("__hot"), t.identifier("get")),
              [t.stringLiteral(importBinding.source)]
            ),
            t.identifier("default")
          );
        } else {
          replacement = t.memberExpression(
            t.callExpression(
              t.memberExpression(t.identifier("__hot"), t.identifier("get")),
              [t.stringLiteral(importBinding.source)]
            ),
            t.identifier(importBinding.imported!)
          );
        }
      } else {
        replacement = t.memberExpression(
          t.identifier("__m"),
          t.identifier(name)
        );
      }

      nodePath.replaceWith(replacement);
    },

    FunctionDeclaration(nodePath: NodePath<t.FunctionDeclaration>) {
      if (nodePath.parent !== ast.program || !nodePath.node.id) return;

      const name = nodePath.node.id.name;
      nodePath.replaceWith(
        t.expressionStatement(
          t.callExpression(
            t.memberExpression(t.identifier("__hot"), t.identifier("defn")),
            [
              t.identifier("__m"),
              t.stringLiteral(name),
              t.functionExpression(
                nodePath.node.id,
                nodePath.node.params,
                nodePath.node.body,
                nodePath.node.generator,
                nodePath.node.async
              )
            ]
          )
        )
      );
    },

    VariableDeclaration(nodePath: NodePath<t.VariableDeclaration>) {
      if (nodePath.parent !== ast.program) return;

      // Check if this is an external import declaration - skip it entirely
      const hasExternalImport = nodePath.node.declarations.some(
        d => t.isIdentifier(d.id) && externalImportIds.has(d.id.name)
      );
      if (hasExternalImport) return;

      const statements: t.Statement[] = [];

      for (const declarator of nodePath.node.declarations) {
        if (t.isIdentifier(declarator.id)) {
          const name = declarator.id.name;
          const binding = locals.get(name);
          const shouldPreserve = binding?.preserve ?? false;

          // Unwrap defonce() if present
          const init = unwrapDefonce(declarator.init);

          if (shouldPreserve) {
            statements.push(
              t.expressionStatement(
                t.assignmentExpression(
                  "=",
                  t.memberExpression(t.identifier("__m"), t.identifier(name)),
                  t.logicalExpression(
                    "??",
                    t.memberExpression(t.identifier("__m"), t.identifier(name)),
                    init
                  )
                )
              )
            );
          } else {
            statements.push(
              t.expressionStatement(
                t.assignmentExpression(
                  "=",
                  t.memberExpression(t.identifier("__m"), t.identifier(name)),
                  init
                )
              )
            );
          }
        }
      }

      if (statements.length > 0) {
        nodePath.replaceWithMultiple(statements);
      }
    },

    ClassDeclaration(nodePath: NodePath<t.ClassDeclaration>) {
      if (nodePath.parent !== ast.program || !nodePath.node.id) return;

      const name = nodePath.node.id.name;
      nodePath.replaceWith(
        t.expressionStatement(
          t.assignmentExpression(
            "=",
            t.memberExpression(t.identifier("__m"), t.identifier(name)),
            t.classExpression(
              nodePath.node.id,
              nodePath.node.superClass,
              nodePath.node.body,
              nodePath.node.decorators
            )
          )
        )
      );
    },

    ExportNamedDeclaration(nodePath: NodePath<t.ExportNamedDeclaration>) {
      const { declaration, specifiers } = nodePath.node;

      if (declaration) {
        if (t.isFunctionDeclaration(declaration) && declaration.id) {
          const name = declaration.id.name;
          locals.set(name, { type: "function", preserve: false });
          nodePath.replaceWith(
            t.expressionStatement(
              t.callExpression(
                t.memberExpression(t.identifier("__hot"), t.identifier("defn")),
                [
                  t.identifier("__m"),
                  t.stringLiteral(name),
                  t.functionExpression(
                    declaration.id,
                    declaration.params,
                    declaration.body,
                    declaration.generator,
                    declaration.async
                  )
                ]
              )
            )
          );
        } else if (t.isVariableDeclaration(declaration)) {
          const statements: t.Statement[] = [];
          for (const declarator of declaration.declarations) {
            if (t.isIdentifier(declarator.id)) {
              const name = declarator.id.name;
              const hasDefonce = isDefonce(declarator.init);
              locals.set(name, { type: "variable", preserve: hasDefonce });

              const init = unwrapDefonce(declarator.init);

              if (hasDefonce) {
                statements.push(
                  t.expressionStatement(
                    t.assignmentExpression(
                      "=",
                      t.memberExpression(t.identifier("__m"), t.identifier(name)),
                      t.logicalExpression(
                        "??",
                        t.memberExpression(t.identifier("__m"), t.identifier(name)),
                        init
                      )
                    )
                  )
                );
              } else {
                statements.push(
                  t.expressionStatement(
                    t.assignmentExpression(
                      "=",
                      t.memberExpression(t.identifier("__m"), t.identifier(name)),
                      init
                    )
                  )
                );
              }
            }
          }
          nodePath.replaceWithMultiple(statements);
        } else if (t.isClassDeclaration(declaration) && declaration.id) {
          const name = declaration.id.name;
          locals.set(name, { type: "class", preserve: false });
          nodePath.replaceWith(
            t.expressionStatement(
              t.assignmentExpression(
                "=",
                t.memberExpression(t.identifier("__m"), t.identifier(name)),
                t.classExpression(
                  declaration.id,
                  declaration.superClass,
                  declaration.body,
                  declaration.decorators
                )
              )
            )
          );
        }
      } else if (specifiers.length > 0) {
        const statements: t.Statement[] = [];
        for (const specifier of specifiers) {
          if (t.isExportSpecifier(specifier)) {
            const localName = specifier.local.name;
            const exportedName = t.isIdentifier(specifier.exported)
              ? specifier.exported.name
              : specifier.exported.value;
            statements.push(
              t.expressionStatement(
                t.assignmentExpression(
                  "=",
                  t.memberExpression(t.identifier("__m"), t.identifier(exportedName)),
                  t.memberExpression(t.identifier("__m"), t.identifier(localName))
                )
              )
            );
          }
        }
        nodePath.replaceWithMultiple(statements);
      }
    },

    ExportDefaultDeclaration(nodePath: NodePath<t.ExportDefaultDeclaration>) {
      const { declaration } = nodePath.node;

      if (t.isFunctionDeclaration(declaration)) {
        if (declaration.id) {
          locals.set(declaration.id.name, { type: "function", preserve: false });
        }
        // Use __hot.defn for function declarations
        nodePath.replaceWith(
          t.expressionStatement(
            t.callExpression(
              t.memberExpression(t.identifier("__hot"), t.identifier("defn")),
              [
                t.identifier("__m"),
                t.stringLiteral("default"),
                t.functionExpression(
                  declaration.id,
                  declaration.params,
                  declaration.body,
                  declaration.generator,
                  declaration.async
                )
              ]
            )
          )
        );
      } else if (t.isClassDeclaration(declaration)) {
        if (declaration.id) {
          locals.set(declaration.id.name, { type: "class", preserve: false });
        }
        nodePath.replaceWith(
          t.expressionStatement(
            t.assignmentExpression(
              "=",
              t.memberExpression(t.identifier("__m"), t.identifier("default")),
              t.classExpression(
                declaration.id,
                declaration.superClass,
                declaration.body,
                declaration.decorators
              )
            )
          )
        );
      } else {
        nodePath.replaceWith(
          t.expressionStatement(
            t.assignmentExpression(
              "=",
              t.memberExpression(t.identifier("__m"), t.identifier("default")),
              declaration as t.Expression
            )
          )
        );
      }
    },
  });

  // Third pass: handle once() calls in expression statements
  const newBody: t.Statement[] = [];
  for (const stmt of ast.program.body) {
    // Check for once() call: once(expr) as expression statement
    if (
      t.isExpressionStatement(stmt) &&
      isApiCall(stmt.expression, "once") &&
      stmt.expression.arguments.length > 0
    ) {
      const innerExpr = stmt.expression.arguments[0] as t.Expression;
      const guardName = `__once_${onceCounter++}`;

      newBody.push(
        t.ifStatement(
          t.unaryExpression(
            "!",
            t.memberExpression(t.identifier("__m"), t.identifier(guardName))
          ),
          t.blockStatement([
            t.expressionStatement(
              t.assignmentExpression(
                "=",
                t.memberExpression(t.identifier("__m"), t.identifier(guardName)),
                t.booleanLiteral(true)
              )
            ),
            t.expressionStatement(innerExpr),
          ])
        )
      );
    } else {
      newBody.push(stmt);
    }
  }
  ast.program.body = newBody;

  // Add module preamble
  ast.program.body.unshift(
    t.variableDeclaration("const", [
      t.variableDeclarator(
        t.identifier("__m"),
        t.callExpression(
          t.memberExpression(t.identifier("__hot"), t.identifier("module")),
          [t.stringLiteral(moduleId)]
        )
      ),
    ])
  );

  const output = generate(ast, {
    retainLines: false,
    compact: false,
    comments: true,
  });

  return output.code;
}

/**
 * Transform a single expression for REPL-style evaluation.
 * Unlike full module transform, this handles individual expressions
 * like function definitions, variable declarations, or arbitrary expressions.
 */
export function transformExpression(
  expr: string,
  moduleId: string
): { code: string; type: "declaration" | "expression" } {
  // Try to parse as a statement first
  let ast: ReturnType<typeof parser.parse>;
  let isExpression = false;

  try {
    ast = parser.parse(expr, {
      sourceType: "module",
      plugins: ["jsx", "typescript"],
      allowReturnOutsideFunction: true,
    });
  } catch (e) {
    // If parsing fails, try wrapping as expression statement
    try {
      ast = parser.parse(`(${expr})`, {
        sourceType: "module",
        plugins: ["jsx", "typescript"],
      });
      isExpression = true;
    } catch (e2) {
      throw new Error(`Failed to parse expression: ${(e as Error).message}`);
    }
  }

  if (ast.program.body.length === 0) {
    return { code: "", type: "expression" };
  }

  const stmt = ast.program.body[0];
  let resultCode: string;
  let resultType: "declaration" | "expression" = "expression";

  // Handle function declaration: function foo() { ... }
  if (t.isFunctionDeclaration(stmt) && stmt.id) {
    const name = stmt.id.name;
    const funcExpr = t.functionExpression(
      stmt.id,
      stmt.params,
      stmt.body,
      stmt.generator,
      stmt.async
    );
    const assignment = t.assignmentExpression(
      "=",
      t.memberExpression(t.identifier("__m"), t.identifier(name)),
      funcExpr
    );
    resultCode = generate(assignment).code;
    resultType = "declaration";
  }
  // Handle class declaration: class Foo { ... }
  else if (t.isClassDeclaration(stmt) && stmt.id) {
    const name = stmt.id.name;
    const classExpr = t.classExpression(
      stmt.id,
      stmt.superClass,
      stmt.body,
      stmt.decorators
    );
    const assignment = t.assignmentExpression(
      "=",
      t.memberExpression(t.identifier("__m"), t.identifier(name)),
      classExpr
    );
    resultCode = generate(assignment).code;
    resultType = "declaration";
  }
  // Handle variable declaration: const x = ..., let y = ..., var z = ...
  else if (t.isVariableDeclaration(stmt)) {
    const assignments: string[] = [];
    for (const declarator of stmt.declarations) {
      if (t.isIdentifier(declarator.id) && declarator.init) {
        const name = declarator.id.name;
        // Use ??= for let/var (preserve), = for const (replace)
        const operator = stmt.kind === "const" ? "=" : "??=";
        if (operator === "??=") {
          const logicalAssign = t.assignmentExpression(
            "=",
            t.memberExpression(t.identifier("__m"), t.identifier(name)),
            t.logicalExpression(
              "??",
              t.memberExpression(t.identifier("__m"), t.identifier(name)),
              declarator.init
            )
          );
          assignments.push(generate(logicalAssign).code);
        } else {
          const assignment = t.assignmentExpression(
            "=",
            t.memberExpression(t.identifier("__m"), t.identifier(name)),
            declarator.init
          );
          assignments.push(generate(assignment).code);
        }
      }
    }
    resultCode = assignments.join(";\n");
    resultType = "declaration";
  }
  // Handle export function: export function foo() { ... }
  else if (t.isExportNamedDeclaration(stmt) && stmt.declaration) {
    if (t.isFunctionDeclaration(stmt.declaration) && stmt.declaration.id) {
      const name = stmt.declaration.id.name;
      const funcExpr = t.functionExpression(
        stmt.declaration.id,
        stmt.declaration.params,
        stmt.declaration.body,
        stmt.declaration.generator,
        stmt.declaration.async
      );
      const assignment = t.assignmentExpression(
        "=",
        t.memberExpression(t.identifier("__m"), t.identifier(name)),
        funcExpr
      );
      resultCode = generate(assignment).code;
      resultType = "declaration";
    } else if (t.isClassDeclaration(stmt.declaration) && stmt.declaration.id) {
      const name = stmt.declaration.id.name;
      const classExpr = t.classExpression(
        stmt.declaration.id,
        stmt.declaration.superClass,
        stmt.declaration.body,
        stmt.declaration.decorators
      );
      const assignment = t.assignmentExpression(
        "=",
        t.memberExpression(t.identifier("__m"), t.identifier(name)),
        classExpr
      );
      resultCode = generate(assignment).code;
      resultType = "declaration";
    } else if (t.isVariableDeclaration(stmt.declaration)) {
      const assignments: string[] = [];
      for (const declarator of stmt.declaration.declarations) {
        if (t.isIdentifier(declarator.id) && declarator.init) {
          const name = declarator.id.name;
          const assignment = t.assignmentExpression(
            "=",
            t.memberExpression(t.identifier("__m"), t.identifier(name)),
            declarator.init
          );
          assignments.push(generate(assignment).code);
        }
      }
      resultCode = assignments.join(";\n");
      resultType = "declaration";
    } else {
      resultCode = expr;
    }
  }
  // Handle export default: export default ...
  else if (t.isExportDefaultDeclaration(stmt)) {
    let value: t.Expression;
    if (t.isFunctionDeclaration(stmt.declaration)) {
      value = t.functionExpression(
        stmt.declaration.id,
        stmt.declaration.params,
        stmt.declaration.body,
        stmt.declaration.generator,
        stmt.declaration.async
      );
    } else if (t.isClassDeclaration(stmt.declaration)) {
      value = t.classExpression(
        stmt.declaration.id,
        stmt.declaration.superClass,
        stmt.declaration.body,
        stmt.declaration.decorators
      );
    } else {
      value = stmt.declaration as t.Expression;
    }
    const assignment = t.assignmentExpression(
      "=",
      t.memberExpression(t.identifier("__m"), t.identifier("default")),
      value
    );
    resultCode = generate(assignment).code;
    resultType = "declaration";
  }
  // Handle expression statement (function call, assignment, etc.)
  else if (t.isExpressionStatement(stmt)) {
    // For arbitrary expressions, just return them - they'll be evaluated
    // but also provide access to module bindings via __m
    resultCode = generate(stmt.expression).code;
    resultType = "expression";
  }
  // Anything else, just return as-is
  else {
    resultCode = expr;
    resultType = "expression";
  }

  return { code: resultCode, type: resultType };
}

function resolveModuleId(filename: string, sourceRoot: string): string {
  const absolute = path.isAbsolute(filename)
    ? filename
    : path.resolve(sourceRoot, filename);
  return path.relative(sourceRoot, absolute);
}

function resolveImportSource(
  source: string,
  fromFile: string,
  sourceRoot: string
): string {
  if (!source.startsWith(".") && !source.startsWith("/")) {
    return source;
  }

  const fromDir = path.dirname(fromFile);
  const absolute = path.resolve(fromDir, source);
  return path.relative(sourceRoot, absolute);
}
