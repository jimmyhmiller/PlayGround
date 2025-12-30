import WebSocket from "ws";

interface EvalResult {
  success: boolean;
  value?: unknown;
  error?: string;
  type: "declaration" | "expression";
}

interface HotRuntime {
  modules: Map<string, Record<string, unknown>>;
  storage: Map<string, Record<string, unknown>>;  // backing storage for functions
  loaders: Map<string, () => void>;
  esmCache: Map<string, unknown>;  // Cache for pre-loaded ESM modules
  ws: WebSocket | null;
  sourceRoot: string;

  module(id: string): Record<string, unknown>;
  get(id: string): Record<string, unknown>;
  require(id: string): void;
  requireExternal(specifier: string): unknown;  // For external (node_modules) packages
  registerEsm(specifier: string, module: unknown): void;  // Pre-register ESM modules
  register(id: string, loader: () => void): void;
  defn(module: Record<string, unknown>, name: string, fn: Function): void;
  setSourceRoot(root: string): void;
  connect(port: number): void;
  reload(id: string, code: string): void;
  evalExpr(moduleId: string, code: string, type: "declaration" | "expression", requestId?: string): EvalResult;
}

export function createRuntime(): HotRuntime {
  const runtime: HotRuntime = {
    modules: new Map(),
    storage: new Map(),
    loaders: new Map(),
    esmCache: new Map(),
    ws: null,
    sourceRoot: process.cwd(),

    module(id: string): Record<string, unknown> {
      if (!this.modules.has(id)) {
        const mod: Record<string, unknown> = { __id__: id };
        this.modules.set(id, mod);
      }
      return this.modules.get(id)!;
    },

    get(id: string): Record<string, unknown> {
      const mod = this.modules.get(id);
      if (!mod) {
        // Might be a node_modules import that wasn't loaded yet
        // Try native require
        const path = require('path');
        const fs = require('fs');
        let native: unknown;
        let found = false;

        // Check if this is a local module ID (relative to sourceRoot)
        if (!id.startsWith('/') && !id.startsWith('.') && (id.includes('/') || id.endsWith('.ts') || id.endsWith('.js'))) {
          const basePath = path.join(this.sourceRoot, id);
          const extensions = ['', '.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.js'];
          for (const ext of extensions) {
            const fullPath = basePath + ext;
            if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
              native = require(fullPath);
              found = true;
              break;
            }
          }
        }

        if (!found) {
          try {
            native = require(id);
            found = true;
          } catch (e) {
            throw new Error(
              `[hot] Module "${id}" not found. Make sure it's loaded before accessing it.`
            );
          }
        }

        // Wrap native module to look like our module format
        const wrapper: Record<string, unknown> = { default: native };
        // Also expose all named exports
        if (native && typeof native === "object") {
          Object.assign(wrapper, native);
        }
        this.modules.set(id, wrapper);
        return wrapper;
      }
      return mod;
    },

    require(id: string): void {
      // For side-effect imports, ensure the module is loaded
      const loader = this.loaders.get(id);
      if (loader) {
        loader();
      } else {
        // Try to require the module
        // If it looks like a local module ID (not starting with / or a bare specifier), resolve it
        const path = require('path');
        const fs = require('fs');

        // Check if this is a local module ID (relative to sourceRoot)
        // Local IDs look like "src/main/events/EventStore" (no leading . or /)
        if (!id.startsWith('/') && !id.startsWith('.') && (id.includes('/') || id.endsWith('.ts') || id.endsWith('.js'))) {
          // Try with common extensions
          const basePath = path.join(this.sourceRoot, id);
          const extensions = ['', '.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.js'];
          for (const ext of extensions) {
            const fullPath = basePath + ext;
            try {
              if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
                require(fullPath);
                return;
              }
            } catch (e) {
              // File found but failed to load - this is a real error, rethrow
              throw e;
            }
          }
        }

        // Fall back to direct require
        try {
          require(id);
        } catch (e) {
          console.warn(`[hot] Could not require "${id}" for side effects`);
        }
      }
    },

    requireExternal(specifier: string): unknown {
      // Check if we have a pre-loaded ESM module
      if (this.esmCache.has(specifier)) {
        return this.esmCache.get(specifier);
      }

      try {
        return require(specifier);
      } catch (e) {
        const err = e as NodeJS.ErrnoException;
        if (err.code === 'ERR_REQUIRE_ESM') {
          throw new Error(
            `[hot-reload] Cannot require ES Module '${specifier}'. ` +
            `This package only supports ESM and cannot be loaded via require().\n\n` +
            `Solutions:\n` +
            `  1. Pre-load the ESM module in your bootstrap file:\n` +
            `     const esm = await import('${specifier}');\n` +
            `     __hot.registerEsm('${specifier}', esm);\n\n` +
            `  2. If using Node.js 22+, try running with:\n` +
            `     node --experimental-require-module ...\n\n` +
            `  3. Use a bundler (esbuild, webpack) to convert ESM dependencies to CJS`
          );
        }
        throw e;
      }
    },

    registerEsm(specifier: string, module: unknown): void {
      this.esmCache.set(specifier, module);
    },

    register(id: string, loader: () => void): void {
      this.loaders.set(id, loader);
    },

    defn(module: Record<string, unknown>, name: string, fn: Function): void {
      // Get or create backing storage for this module
      const moduleId = (module as any).__id__ as string;
      if (!this.storage.has(moduleId)) {
        this.storage.set(moduleId, {});
      }
      const store = this.storage.get(moduleId)!;

      // Store the actual function
      store[name] = fn;

      // Define getter if not already defined
      if (!Object.getOwnPropertyDescriptor(module, name)?.get) {
        Object.defineProperty(module, name, {
          get() {
            // Return wrapper that looks up fresh on each INVOCATION (not when getter is called)
            return (...args: unknown[]) => (store[name] as Function)(...args);
          },
          set(value) {
            store[name] = value;
          },
          configurable: true,
          enumerable: true
        });
      }
      // If getter exists, storage is already updated above
    },

    setSourceRoot(root: string): void {
      this.sourceRoot = root;
    },

    connect(port: number): void {
      const url = `ws://127.0.0.1:${port}`;
      console.log(`[hot] Connecting to ${url}`);

      try {
        this.ws = new WebSocket(url);
        console.log(`[hot] WebSocket created, readyState: ${this.ws.readyState}`);
      } catch (e) {
        console.error(`[hot] Failed to create WebSocket:`, e);
        return;
      }

      this.ws.on("open", () => {
        console.log("[hot] Connected to dev server");
      });

      this.ws.on("message", (data: WebSocket.Data) => {
        try {
          const message = JSON.parse(data.toString());
          if (message.type === "reload") {
            console.log(`[hot] Reloading ${message.id}`);
            this.reload(message.id, message.code);
          } else if (message.type === "eval") {
            // Expression-level evaluation from editor
            const { moduleId, code, exprType, requestId } = message;
            const result = this.evalExpr(moduleId, code, exprType || "expression", requestId);

            // Send result back to server (which forwards to editor)
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
              this.ws.send(JSON.stringify({
                type: "eval-result",
                requestId,
                moduleId,
                success: result.success,
                value: result.value,
                error: result.error,
                exprType: result.type,
              }));
            }
          }
        } catch (e) {
          console.error("[hot] Failed to parse message:", e);
        }
      });

      this.ws.on("close", () => {
        console.log("[hot] Disconnected from dev server");
        // Attempt to reconnect after a delay
        setTimeout(() => this.connect(port), 2000);
      });

      this.ws.on("error", (err: Error) => {
        console.error("[hot] WebSocket error:", (err as NodeJS.ErrnoException).code, err.message);
      });
    },

    reload(id: string, code: string): void {
      try {
        // The code already references __hot global, so just eval it
        // It will update the module's properties in place
        // Pass require, __dirname, __filename, module, exports so the code has access to Node globals
        const path = require('path');
        const modulePath = path.resolve(this.sourceRoot, id);
        const moduleDir = path.dirname(modulePath);
        // Create a fake module object for CommonJS compatibility
        const fakeModule = { exports: {} };
        const fn = new Function("__hot", "require", "__dirname", "__filename", "module", "exports", code);
        fn(this, require, moduleDir, modulePath, fakeModule, fakeModule.exports);
        console.log(`[hot] Successfully reloaded ${id}`);
      } catch (e) {
        console.error(`[hot] Failed to reload ${id}:`, e);
      }
    },

    evalExpr(moduleId: string, code: string, type: "declaration" | "expression", requestId?: string): EvalResult {
      try {
        // Get or create the module's namespace
        const __m = this.module(moduleId);

        // Build a combined object of ALL module exports
        // This makes `greet("test")` work even if greet is from another module
        const allBindings: Record<string, unknown> = {};
        for (const [modId, mod] of this.modules) {
          for (const [key, value] of Object.entries(mod)) {
            if (key !== "default" && !allBindings.hasOwnProperty(key)) {
              allBindings[key] = value;
            }
          }
        }
        // Current module's bindings take precedence
        Object.assign(allBindings, __m);

        const bindingNames = Object.keys(allBindings);
        const destructure = bindingNames.length > 0
          ? `const { ${bindingNames.join(", ")} } = __allBindings;\n`
          : "";

        // Create a function that has access to __m and __hot
        // For expressions, we want to return the value
        // For declarations, we just execute
        let evalCode: string;
        if (type === "expression") {
          evalCode = `${destructure}return (${code})`;
        } else {
          evalCode = `${destructure}${code}`;
        }

        const fn = new Function("__m", "__hot", "__allBindings", evalCode);
        const result = fn(__m, this, allBindings);

        console.log(`[hot] Eval in ${moduleId}: ${code.slice(0, 50)}${code.length > 50 ? "..." : ""}`);

        // Format result for display
        let displayValue: unknown;
        if (type === "expression") {
          displayValue = result;
        } else {
          // For declarations, show what was defined
          displayValue = `Defined in ${moduleId}`;
        }

        return {
          success: true,
          value: displayValue,
          type,
        };
      } catch (e) {
        const error = e as Error;
        console.error(`[hot] Eval error in ${moduleId}:`, error.message);
        return {
          success: false,
          error: error.message,
          type,
        };
      }
    },
  };

  return runtime;
}

// Default export for easy setup
export const __hot = createRuntime();
