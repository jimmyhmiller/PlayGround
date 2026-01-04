import * as parser from "@babel/parser";
import traverse, { NodePath } from "@babel/traverse";
import generate from "@babel/generator";
import * as t from "@babel/types";
import * as path from "path";
import { transformFromAstSync } from "@babel/core";

interface ImportBinding {
  source: string;
  type: "named" | "default" | "namespace";
  imported?: string;
}

// Check if an import is external (bare module specifier, not relative)
function isExternalImport(source: string): boolean {
  return !source.startsWith(".") && !source.startsWith("/");
}

// Check if an identifier is in a TypeScript type position (type annotation, type argument, etc.)
function isInTypePosition(nodePath: NodePath<t.Identifier>): boolean {
  let current: NodePath | null = nodePath;
  while (current) {
    const node = current.node;
    // Check if we're inside any TypeScript type node
    if (
      t.isTSTypeAnnotation(node) ||
      t.isTSTypeReference(node) ||
      t.isTSTypeParameterDeclaration(node) ||
      t.isTSTypeParameterInstantiation(node) ||
      t.isTSTypeLiteral(node) ||
      t.isTSFunctionType(node) ||
      t.isTSArrayType(node) ||
      t.isTSUnionType(node) ||
      t.isTSIntersectionType(node) ||
      t.isTSTypeQuery(node) ||
      t.isTSExpressionWithTypeArguments(node) ||
      t.isTSInterfaceDeclaration(node) ||
      t.isTSTypeAliasDeclaration(node)
    ) {
      return true;
    }
    current = current.parentPath;
  }
  return false;
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
  esm?: boolean; // Output ESM syntax for external imports (default: false = CJS)
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

// Unwrap lambda expressions for once(): () => expr becomes expr
// For block bodies, we call the function immediately: (() => { ... })()
function unwrapOnceArg(node: t.Expression): t.Expression {
  // Arrow function with expression body: () => expr
  if (t.isArrowFunctionExpression(node) && t.isExpression(node.body)) {
    return node.body;
  }
  // Arrow function with block body: () => { ... } -> call it
  if (t.isArrowFunctionExpression(node) && t.isBlockStatement(node.body)) {
    return t.callExpression(node, []);
  }
  // Regular function expression: function() { ... } -> call it
  if (t.isFunctionExpression(node)) {
    return t.callExpression(node, []);
  }
  return node;
}

export function transform(code: string, options: TransformOptions): string {
  const { filename, sourceRoot = process.cwd(), esm = false } = options;
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

  // Track ESM exports (only used in ESM mode)
  const esmExports: Array<{ name: string; type: 'function' | 'value' | 'default-function' | 'default-value' }> = [];

  // Track if we have a TSExportAssignment (export = x) for CJS module.exports
  let hasTSExportAssignment = false;

  // First pass: collect imports and top-level declarations
  traverse(ast, {
    ImportDeclaration(nodePath: NodePath<t.ImportDeclaration>) {
      const sourceValue = nodePath.node.source.value;

      // Skip type-only imports entirely - they don't exist at runtime
      if (nodePath.node.importKind === 'type') {
        nodePath.remove();
        return;
      }

      // Check if this is importing from our API
      const isApiImport = sourceValue === "hot-reload" ||
                          sourceValue === "hot-reload/api" ||
                          sourceValue.endsWith("/api");

      // Check if this is an external import (node_modules, built-ins like 'electron')
      const isExternal = isExternalImport(sourceValue);

      const source = resolveImportSource(sourceValue, filename, sourceRoot);

      // Only track local imports for hot-reload transformation (CJS mode only)
      // External imports are left as regular require() calls
      // In ESM mode, local imports stay as ESM imports (no tracking needed)
      if (!isExternal && !isApiImport && !esm) {
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
        // Track all identifiers from external imports so we don't transform them
        const specifiers = nodePath.node.specifiers;
        for (const spec of specifiers) {
          externalImportIds.add(spec.local.name);
        }

        // In ESM mode, keep external imports as-is (ESM syntax)
        if (esm) {
          // Don't transform - leave as ESM import
          return;
        }

        // In CJS mode, convert external imports to require() calls
        // Transform: import { x } from 'pkg' -> const { x } = require('pkg')
        // Transform: import x from 'pkg' -> const x = require('pkg').default || require('pkg')
        // Transform: import * as x from 'pkg' -> const x = require('pkg')

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
      } else if (esm) {
        // In ESM mode, keep local imports as-is
        // The imported module will export wrappers that delegate to __hot
        // Don't track these in 'imports' so references won't be transformed
        return;
      } else {
        // CJS mode: local import - replace with __hot.require() to load the module
        // References will be replaced with __hot.get() calls
        nodePath.replaceWith(
          t.expressionStatement(
            t.callExpression(
              t.memberExpression(t.identifier("__hot"), t.identifier("require")),
              [t.stringLiteral(source)]
            )
          )
        );
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

    // Handle TypeScript's `import x = require('...')` syntax
    TSImportEqualsDeclaration(nodePath: NodePath<t.TSImportEqualsDeclaration>) {
      const { id, moduleReference } = nodePath.node;

      // Only handle `import x = require('...')` style (external module reference)
      if (t.isTSExternalModuleReference(moduleReference)) {
        const source = moduleReference.expression;
        if (!t.isStringLiteral(source)) return;

        const sourceValue = source.value;
        const isExternal = isExternalImport(sourceValue);

        // Track so we don't transform references
        externalImportIds.add(id.name);

        if (esm && !isExternal) {
          // ESM mode + local import: convert to ESM default import
          nodePath.replaceWith(
            t.importDeclaration(
              [t.importDefaultSpecifier(id)],
              source
            )
          );
        } else {
          // CJS mode OR external import: use require()
          nodePath.replaceWith(
            t.variableDeclaration("const", [
              t.variableDeclarator(
                id,
                t.callExpression(t.identifier("require"), [source])
              )
            ])
          );
        }
      }
    },
  });

  // Second pass: transform declarations and replace references
  traverse(ast, {
    Identifier(nodePath: NodePath<t.Identifier>) {
      const name = nodePath.node.name;

      // Skip identifiers in type positions (type annotations, type references, etc.)
      if (isInTypePosition(nodePath)) return;

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

      // Handle type-only exports - remove them entirely
      if (nodePath.node.exportKind === 'type') {
        nodePath.remove();
        return;
      }

      // Handle empty exports (export {};) - remove them
      if (!declaration && specifiers.length === 0) {
        nodePath.remove();
        return;
      }

      if (declaration) {
        // Handle type-only declarations (export interface, export type)
        if (t.isTSInterfaceDeclaration(declaration) || t.isTSTypeAliasDeclaration(declaration)) {
          nodePath.remove();
          return;
        }

        if (t.isFunctionDeclaration(declaration) && declaration.id) {
          const name = declaration.id.name;
          locals.set(name, { type: "function", preserve: false });
          if (esm) esmExports.push({ name, type: 'function' });
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
              if (esm) esmExports.push({ name, type: 'value' });

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
          if (esm) esmExports.push({ name, type: 'value' });
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
            if (esm) esmExports.push({ name: exportedName, type: 'value' });
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
        if (esm) esmExports.push({ name: 'default', type: 'default-function' });
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
        if (esm) esmExports.push({ name: 'default', type: 'default-value' });
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
        if (esm) esmExports.push({ name: 'default', type: 'default-value' });
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

    // Handle TypeScript's `export = expression` syntax
    // This is CJS-style export that needs to become `export default` in ESM
    TSExportAssignment(nodePath: NodePath<t.TSExportAssignment>) {
      const expression = nodePath.node.expression;
      hasTSExportAssignment = true;
      if (esm) esmExports.push({ name: 'default', type: 'default-value' });
      nodePath.replaceWith(
        t.expressionStatement(
          t.assignmentExpression(
            "=",
            t.memberExpression(t.identifier("__m"), t.identifier("default")),
            expression
          )
        )
      );
    },

    // Handle `export * from './module'` re-exports
    ExportAllDeclaration(nodePath: NodePath<t.ExportAllDeclaration>) {
      const sourceValue = nodePath.node.source.value;

      // Skip type-only exports
      if (nodePath.node.exportKind === 'type') {
        nodePath.remove();
        return;
      }

      const isExternal = isExternalImport(sourceValue);
      const source = resolveImportSource(sourceValue, filename, sourceRoot);

      if (esm) {
        // In ESM mode, keep export * as-is for external modules
        if (isExternal) {
          return;
        }
        // For local modules in ESM mode, we need to re-export through __hot
        // This is tricky - for now, keep the export * and let it work normally
        // The imported module will have its own __hot wrappers
        return;
      }

      // CJS mode: Transform to load module and copy all exports to __m
      // export * from './foo' becomes:
      //   __hot.require('src/main/pipeline/foo');
      //   Object.assign(__m, __hot.get('src/main/pipeline/foo'));
      const statements: t.Statement[] = [];

      if (!isExternal) {
        // Local module: use __hot.require and __hot.get
        statements.push(
          t.expressionStatement(
            t.callExpression(
              t.memberExpression(t.identifier("__hot"), t.identifier("require")),
              [t.stringLiteral(source)]
            )
          )
        );
        statements.push(
          t.expressionStatement(
            t.callExpression(
              t.memberExpression(t.identifier("Object"), t.identifier("assign")),
              [
                t.identifier("__m"),
                t.callExpression(
                  t.memberExpression(t.identifier("__hot"), t.identifier("get")),
                  [t.stringLiteral(source)]
                )
              ]
            )
          )
        );
      } else {
        // External module: use regular require
        statements.push(
          t.expressionStatement(
            t.callExpression(
              t.memberExpression(t.identifier("Object"), t.identifier("assign")),
              [
                t.identifier("__m"),
                t.callExpression(t.identifier("require"), [t.stringLiteral(sourceValue)])
              ]
            )
          )
        );
      }

      nodePath.replaceWithMultiple(statements);
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
      const innerExpr = unwrapOnceArg(stmt.expression.arguments[0] as t.Expression);
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

  // In ESM mode, add require function for CJS compatibility
  // This allows `require()` calls to work in ESM context
  if (esm) {
    ast.program.body.unshift(
      // const require = __createRequire(import.meta.url);
      t.variableDeclaration("const", [
        t.variableDeclarator(
          t.identifier("require"),
          t.callExpression(
            t.identifier("__createRequire"),
            [t.memberExpression(
              t.metaProperty(t.identifier("import"), t.identifier("meta")),
              t.identifier("url")
            )]
          )
        ),
      ])
    );
    ast.program.body.unshift(
      // import { createRequire as __createRequire } from 'node:module';
      t.importDeclaration(
        [t.importSpecifier(t.identifier("__createRequire"), t.identifier("createRequire"))],
        t.stringLiteral("node:module")
      )
    );
  }

  // In ESM mode, add export statements at the end
  if (esm && esmExports.length > 0) {
    for (const exp of esmExports) {
      if (exp.type === 'function') {
        // For functions, export a wrapper that delegates to __m (for hot reload)
        // export function name(...args) { return __m.name.apply(this, args); }
        ast.program.body.push(
          t.exportNamedDeclaration(
            t.functionDeclaration(
              t.identifier(exp.name),
              [t.restElement(t.identifier('__args'))],
              t.blockStatement([
                t.returnStatement(
                  t.callExpression(
                    t.memberExpression(
                      t.memberExpression(t.identifier('__m'), t.identifier(exp.name)),
                      t.identifier('apply')
                    ),
                    [t.thisExpression(), t.identifier('__args')]
                  )
                )
              ])
            )
          )
        );
      } else if (exp.type === 'value') {
        // For values, export a getter via Object.defineProperty pattern
        // We use a variable declaration that reads from __m
        // export const name = __m.name;  (won't be live, but values rarely change)
        ast.program.body.push(
          t.exportNamedDeclaration(
            t.variableDeclaration('const', [
              t.variableDeclarator(
                t.identifier(exp.name),
                t.memberExpression(t.identifier('__m'), t.identifier(exp.name))
              )
            ])
          )
        );
      } else if (exp.type === 'default-function') {
        // export default function(...args) { return __m.default.apply(this, args); }
        ast.program.body.push(
          t.exportDefaultDeclaration(
            t.functionDeclaration(
              null,
              [t.restElement(t.identifier('__args'))],
              t.blockStatement([
                t.returnStatement(
                  t.callExpression(
                    t.memberExpression(
                      t.memberExpression(t.identifier('__m'), t.identifier('default')),
                      t.identifier('apply')
                    ),
                    [t.thisExpression(), t.identifier('__args')]
                  )
                )
              ])
            )
          )
        );
      } else if (exp.type === 'default-value') {
        // export default __m.default;
        ast.program.body.push(
          t.exportDefaultDeclaration(
            t.memberExpression(t.identifier('__m'), t.identifier('default'))
          )
        );
      }
    }
  }

  // In CJS mode, add module.exports at the end
  if (!esm) {
    if (hasTSExportAssignment) {
      // export = expression -> module.exports = __m.default
      ast.program.body.push(
        t.expressionStatement(
          t.assignmentExpression(
            "=",
            t.memberExpression(t.identifier("module"), t.identifier("exports")),
            t.memberExpression(t.identifier("__m"), t.identifier("default"))
          )
        )
      );
    } else {
      // Regular exports -> module.exports = __m (so all exports are accessible)
      ast.program.body.push(
        t.expressionStatement(
          t.assignmentExpression(
            "=",
            t.memberExpression(t.identifier("module"), t.identifier("exports")),
            t.identifier("__m")
          )
        )
      );
    }
  }

  // Strip TypeScript types using Babel transform
  // Use require.resolve to get absolute path - ensures plugin is found from hot-reload's node_modules
  const tsPluginPath = require.resolve('@babel/plugin-transform-typescript');
  const result = transformFromAstSync(ast, code, {
    plugins: [
      [tsPluginPath, { isTSX: true }]
    ],
    filename: filename,
    sourceType: 'module',
    retainLines: false,
    compact: false,
    comments: true,
  });

  if (!result || !result.code) {
    throw new Error('Failed to transform TypeScript');
  }

  return result.code;
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

// Strip known extensions from module IDs for consistent matching
function stripExtension(filePath: string): string {
  return filePath.replace(/\.(ts|tsx|js|jsx)$/, '');
}

function resolveModuleId(filename: string, sourceRoot: string): string {
  const absolute = path.isAbsolute(filename)
    ? filename
    : path.resolve(sourceRoot, filename);
  const relative = path.relative(sourceRoot, absolute);
  return stripExtension(relative);
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
  const relative = path.relative(sourceRoot, absolute);
  return stripExtension(relative);
}
