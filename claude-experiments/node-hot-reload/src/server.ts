import * as fs from "fs";
import * as path from "path";
import { WebSocketServer, WebSocket } from "ws";
import chokidar from "chokidar";
import { transform, transformExpression } from "./transform";

interface ServerOptions {
  sourceDir: string;
  port: number;
}

export function startServer(options: ServerOptions) {
  const { sourceDir, port } = options;
  const absoluteSourceDir = path.resolve(sourceDir);

  // Track connected clients - we distinguish editors from runtimes
  // Editors send eval requests, runtimes execute them
  interface Client {
    ws: WebSocket;
    type: "runtime" | "editor";
  }
  const clients = new Map<WebSocket, Client>();

  // Track pending eval requests to route results back
  const pendingEvals = new Map<string, WebSocket>();

  // Create WebSocket server on localhost only
  const wss = new WebSocketServer({ host: "127.0.0.1", port });

  wss.on("listening", () => {
    console.log("[server] WebSocket server ready");
  });

  wss.on("error", (err) => {
    console.error("[server] WebSocket server error:", err);
  });

  wss.on("connection", (ws: WebSocket) => {
    console.log("[server] Client connected");
    // Default to runtime, editors will identify themselves
    clients.set(ws, { ws, type: "runtime" });

    ws.on("message", (data) => {
      try {
        const message = JSON.parse(data.toString());

        if (message.type === "identify") {
          // Editor identifies itself
          const client = clients.get(ws);
          if (client) {
            client.type = message.clientType || "editor";
            console.log(`[server] Client identified as ${client.type}`);

            // Send server info back to editor
            if (client.type === "editor") {
              ws.send(JSON.stringify({
                type: "server-info",
                sourceRoot: absoluteSourceDir,
                port,
              }));
            }
          }
        } else if (message.type === "eval-request") {
          // Editor sends expression to evaluate
          const { moduleId, expr, requestId } = message;
          console.log(`[server] Eval request for ${moduleId}: ${expr.slice(0, 50)}...`);

          try {
            // Transform the expression
            const { code, type: exprType } = transformExpression(expr, moduleId);

            // Track this request for routing the result back
            if (requestId) {
              pendingEvals.set(requestId, ws);
            }

            // Forward to all runtime clients
            const evalMessage = JSON.stringify({
              type: "eval",
              moduleId,
              code,
              exprType,
              requestId,
            });

            for (const [clientWs, client] of clients) {
              if (client.type === "runtime" && clientWs.readyState === WebSocket.OPEN) {
                clientWs.send(evalMessage);
              }
            }
          } catch (e) {
            // Transform error - send back to editor
            const error = e as Error;
            ws.send(JSON.stringify({
              type: "eval-result",
              requestId,
              moduleId,
              success: false,
              error: `Transform error: ${error.message}`,
            }));
          }
        } else if (message.type === "eval-result") {
          // Runtime sends result back - route to requesting editor
          const { requestId } = message;
          const editorWs = pendingEvals.get(requestId);
          if (editorWs && editorWs.readyState === WebSocket.OPEN) {
            editorWs.send(JSON.stringify(message));
            pendingEvals.delete(requestId);
          } else {
            // Broadcast to all editors
            for (const [clientWs, client] of clients) {
              if (client.type === "editor" && clientWs.readyState === WebSocket.OPEN) {
                clientWs.send(JSON.stringify(message));
              }
            }
          }
        }
      } catch (e) {
        // Not JSON or parse error - ignore
      }
    });

    ws.on("close", () => {
      console.log("[server] Client disconnected");
      clients.delete(ws);
    });
  });

  console.log(`[server] WebSocket server listening on port ${port}`);

  // Watch for file changes (.js, .ts, .jsx, .tsx)
  // Only watch main process code, not renderer (Vite handles that)
  const watcher = chokidar.watch([
    `${absoluteSourceDir}/**/*.js`,
    `${absoluteSourceDir}/**/*.ts`,
    `${absoluteSourceDir}/**/*.jsx`,
    `${absoluteSourceDir}/**/*.tsx`,
  ], {
    persistent: true,
    ignoreInitial: true,
    ignored: [
      '**/node_modules/**',
      '**/dist/**',
      '**/*.min.js',
      '**/*.d.ts',
      // Ignore frontend/renderer directories - Vite handles those
      '**/renderer/**',
      '**/frontend/**',
      '**/client/**',
      '**/web/**',
    ],
  });

  watcher.on("change", (filePath: string) => {
    console.log(`[server] File changed: ${filePath}`);

    try {
      const code = fs.readFileSync(filePath, "utf-8");
      const transformed = transform(code, {
        filename: filePath,
        sourceRoot: absoluteSourceDir,
      });

      const moduleId = path.relative(absoluteSourceDir, filePath);

      const message = JSON.stringify({
        type: "reload",
        id: moduleId,
        code: transformed,
      });

      // Broadcast to all runtime clients
      let sentCount = 0;
      for (const [clientWs, client] of clients) {
        if (client.type === "runtime" && clientWs.readyState === WebSocket.OPEN) {
          clientWs.send(message);
          sentCount++;
        }
      }

      console.log(`[server] Sent reload for ${moduleId} to ${sentCount} runtime(s)`);
    } catch (e) {
      console.error(`[server] Failed to transform ${filePath}:`, e);
    }
  });

  watcher.on("add", (filePath: string) => {
    console.log(`[server] New file: ${filePath}`);
  });

  console.log(`[server] Watching ${absoluteSourceDir} for changes`);

  return {
    close() {
      wss.close();
      watcher.close();
    },
  };
}
