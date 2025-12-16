import WebSocket from "ws";

interface EvalResult {
  success: boolean;
  value?: unknown;
  error?: string;
  type: "declaration" | "expression";
}

interface HotRuntime {
  modules: Map<string, Record<string, unknown>>;
  loaders: Map<string, () => void>;
  ws: WebSocket | null;

  module(id: string): Record<string, unknown>;
  get(id: string): Record<string, unknown>;
  require(id: string): void;
  register(id: string, loader: () => void): void;
  connect(port: number): void;
  reload(id: string, code: string): void;
  evalExpr(moduleId: string, code: string, type: "declaration" | "expression", requestId?: string): EvalResult;
}

export function createRuntime(): HotRuntime {
  const runtime: HotRuntime = {
    modules: new Map(),
    loaders: new Map(),
    ws: null,

    module(id: string): Record<string, unknown> {
      if (!this.modules.has(id)) {
        this.modules.set(id, {});
      }
      return this.modules.get(id)!;
    },

    get(id: string): Record<string, unknown> {
      const mod = this.modules.get(id);
      if (!mod) {
        // Might be a node_modules import that wasn't loaded yet
        // Try native require
        try {
          const native = require(id);
          // Wrap native module to look like our module format
          const wrapper: Record<string, unknown> = { default: native };
          // Also expose all named exports
          if (native && typeof native === "object") {
            Object.assign(wrapper, native);
          }
          this.modules.set(id, wrapper);
          return wrapper;
        } catch (e) {
          throw new Error(
            `[hot] Module "${id}" not found. Make sure it's loaded before accessing it.`
          );
        }
      }
      return mod;
    },

    require(id: string): void {
      // For side-effect imports, ensure the module is loaded
      const loader = this.loaders.get(id);
      if (loader) {
        loader();
      } else {
        // Try native require for node_modules
        try {
          require(id);
        } catch (e) {
          console.warn(`[hot] Could not require "${id}" for side effects`);
        }
      }
    },

    register(id: string, loader: () => void): void {
      this.loaders.set(id, loader);
    },

    connect(port: number): void {
      const url = `ws://localhost:${port}`;
      console.log(`[hot] Connecting to ${url}`);

      this.ws = new WebSocket(url);

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
        // Suppress ECONNREFUSED errors during reconnection
        if ((err as NodeJS.ErrnoException).code !== "ECONNREFUSED") {
          console.error("[hot] WebSocket error:", err.message);
        }
      });
    },

    reload(id: string, code: string): void {
      try {
        // The code already references __hot global, so just eval it
        // It will update the module's properties in place
        const fn = new Function("__hot", code);
        fn(this);
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
        console.log(`[hot] Available bindings for eval:`, bindingNames);
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

        const fn = new Function("__m", "__hot", evalCode);
        const result = fn(__m, this);

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
