/**
 * Production Transform - Strip once/defonce calls
 *
 * This transform removes hot-reload API imports and replaces
 * once(expr) and defonce(value) with just their inner values.
 *
 * Use this for production builds where hot-reload is not needed.
 */

import * as parser from "@babel/parser";
import traverse, { NodePath } from "@babel/traverse";
import generate from "@babel/generator";
import * as t from "@babel/types";

export interface StripOptions {
  /** Parser plugins to use (default: jsx, typescript) */
  plugins?: parser.ParserPlugin[];
}

/**
 * Strip once() and defonce() calls from code.
 *
 * - Removes imports from 'hot-reload/api' or 'hot-reload'
 * - Replaces defonce(value) with value
 * - Replaces once(expr) with expr
 *
 * @example
 * // Input:
 * import { once, defonce } from 'hot-reload/api';
 * const cache = defonce(new Map());
 * once(console.log('init'));
 *
 * // Output:
 * const cache = new Map();
 * console.log('init');
 */
export function strip(code: string, options: StripOptions = {}): string {
  const { plugins = ["jsx", "typescript"] } = options;

  const ast = parser.parse(code, {
    sourceType: "module",
    plugins,
    attachComment: true,
  });

  // Track which local names map to once/defonce
  const apiBindings = new Map<string, "once" | "defonce">();

  // First pass: find and remove API imports, track local bindings
  traverse(ast, {
    ImportDeclaration(nodePath: NodePath<t.ImportDeclaration>) {
      const sourceValue = nodePath.node.source.value;

      // Check if this is importing from our API
      const isApiImport =
        sourceValue === "hot-reload" ||
        sourceValue === "hot-reload/api" ||
        sourceValue.endsWith("/api");

      if (!isApiImport) return;

      const remainingSpecifiers: t.ImportSpecifier[] = [];

      for (const specifier of nodePath.node.specifiers) {
        if (t.isImportSpecifier(specifier)) {
          const imported = t.isIdentifier(specifier.imported)
            ? specifier.imported.name
            : specifier.imported.value;

          if (imported === "once" || imported === "defonce") {
            // Track this binding for later replacement
            apiBindings.set(specifier.local.name, imported);
          } else {
            // Keep non-API imports
            remainingSpecifiers.push(specifier);
          }
        }
      }

      if (remainingSpecifiers.length === 0) {
        // Remove the entire import
        nodePath.remove();
      } else {
        // Keep only non-API specifiers
        nodePath.node.specifiers = remainingSpecifiers;
      }
    },
  });

  // Second pass: replace once()/defonce() calls with their arguments
  traverse(ast, {
    CallExpression(nodePath: NodePath<t.CallExpression>) {
      if (!t.isIdentifier(nodePath.node.callee)) return;

      const calleeName = nodePath.node.callee.name;
      const apiType = apiBindings.get(calleeName);

      if (!apiType) return;
      if (nodePath.node.arguments.length === 0) return;

      const arg = nodePath.node.arguments[0];

      if (t.isExpression(arg) || t.isSpreadElement(arg)) {
        // Replace the call with just the argument
        if (t.isSpreadElement(arg)) {
          // Can't replace spread element directly, wrap in array
          nodePath.replaceWith(t.arrayExpression([arg]));
        } else {
          nodePath.replaceWith(arg);
        }
      }
    },
  });

  const output = generate(ast, {
    retainLines: true,
    compact: false,
    comments: true,
  });

  return output.code;
}

/**
 * Babel plugin for stripping once/defonce calls.
 *
 * @example
 * // babel.config.js
 * module.exports = {
 *   plugins: ['hot-reload/strip']
 * };
 *
 * // Or with options:
 * module.exports = {
 *   plugins: [
 *     ['hot-reload/strip', { sourcePatterns: ['hot-reload/api', 'my-hot-reload'] }]
 *   ]
 * };
 */
export interface StripPluginOptions {
  /** Additional import sources to treat as API imports (default: hot-reload, hot-reload/api) */
  sourcePatterns?: string[];
}

export function stripPlugin(): BabelPluginObj {
  return {
    name: "hot-reload-strip",
    visitor: {
      Program: {
        enter(programPath: NodePath<t.Program>, state: BabelPluginState) {
          // Create fresh bindings map for each file
          state.set("apiBindings", new Map<string, "once" | "defonce">());

          // Get custom source patterns from options
          const opts = (state.opts || {}) as StripPluginOptions;
          const extraPatterns = opts.sourcePatterns || [];
          state.set("sourcePatterns", extraPatterns);
        },
      },

      ImportDeclaration(nodePath: NodePath<t.ImportDeclaration>, state: BabelPluginState) {
        const sourceValue = nodePath.node.source.value;
        const extraPatterns = state.get("sourcePatterns") as string[];

        const isApiImport =
          sourceValue === "hot-reload" ||
          sourceValue === "hot-reload/api" ||
          sourceValue.endsWith("/api") ||
          extraPatterns.some((p) => sourceValue === p || sourceValue.endsWith(p));

        if (!isApiImport) return;

        const apiBindings = state.get("apiBindings") as Map<string, "once" | "defonce">;
        const remainingSpecifiers: t.ImportSpecifier[] = [];

        for (const specifier of nodePath.node.specifiers) {
          if (t.isImportSpecifier(specifier)) {
            const imported = t.isIdentifier(specifier.imported)
              ? specifier.imported.name
              : specifier.imported.value;

            if (imported === "once" || imported === "defonce") {
              apiBindings.set(specifier.local.name, imported);
            } else {
              remainingSpecifiers.push(specifier);
            }
          }
        }

        if (remainingSpecifiers.length === 0) {
          nodePath.remove();
        } else {
          nodePath.node.specifiers = remainingSpecifiers;
        }
      },

      CallExpression(nodePath: NodePath<t.CallExpression>, state: BabelPluginState) {
        if (!t.isIdentifier(nodePath.node.callee)) return;

        const apiBindings = state.get("apiBindings") as Map<string, "once" | "defonce">;
        const calleeName = nodePath.node.callee.name;
        const apiType = apiBindings.get(calleeName);

        if (!apiType) return;
        if (nodePath.node.arguments.length === 0) return;

        const arg = nodePath.node.arguments[0];

        if (t.isExpression(arg)) {
          nodePath.replaceWith(arg);
        }
      },
    },
  };
}

// Default export for Babel plugin resolution
export default stripPlugin;

// Types for Babel plugin
interface BabelPluginState {
  opts?: StripPluginOptions;
  get<T>(key: string): T;
  set<T>(key: string, value: T): void;
}

interface BabelPluginObj {
  name: string;
  visitor: {
    Program?: {
      enter(path: NodePath<t.Program>, state: BabelPluginState): void;
    };
    ImportDeclaration?(path: NodePath<t.ImportDeclaration>, state: BabelPluginState): void;
    CallExpression?(path: NodePath<t.CallExpression>, state: BabelPluginState): void;
  };
}
