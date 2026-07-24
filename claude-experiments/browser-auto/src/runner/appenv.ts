import { spawn, type ChildProcess } from "node:child_process";
import { createWriteStream, mkdirSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer } from "node:net";
import { join } from "node:path";

/**
 * Managed app instances for parallel workers. Each worker gets its own app
 * process on its own port — world isolation is structural, not cooperative.
 * `{port}` and `{index}` substitute into the command, env values, baseUrl and
 * the world's http endpoint; PORT/BAT_PORT/BAT_WORKER are set in the env.
 * All app output is retained under .bat/app-logs/ (evidence, always).
 */

export interface AppSpec {
  command: string;
  /** path polled until the app responds (default "/") */
  readyUrl?: string;
  env?: Record<string, string>;
  startupTimeoutMs?: number;
}

export interface AppEnvironment {
  index: number;
  port: number;
  baseUrl: string;
  logPath: string;
  stop(): Promise<void>;
}

export function substitute(s: string, vars: { port: number; index: number }): string {
  return s.replace(/\{port\}/g, String(vars.port)).replace(/\{index\}/g, String(vars.index));
}

export async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.listen(0, "127.0.0.1", () => {
      const address = srv.address();
      if (address === null || typeof address === "string") return reject(new Error("no port"));
      const port = address.port;
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}

export async function launchApp(
  spec: AppSpec,
  index: number,
  baseUrlTemplate: string,
  root: string,
): Promise<AppEnvironment> {
  const port = await freePort();
  const vars = { port, index };
  const baseUrl = substitute(baseUrlTemplate, vars);

  const logDir = join(root, ".bat", "app-logs");
  mkdirSync(logDir, { recursive: true });
  const logPath = join(logDir, `worker-${index}.log`);
  const log = createWriteStream(logPath, { flags: "a" });
  log.write(`\n===== ${new Date().toISOString()} worker ${index} port ${port}: ${spec.command} =====\n`);

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    PORT: String(port),
    BAT_PORT: String(port),
    BAT_WORKER: String(index),
  };
  for (const [k, v] of Object.entries(spec.env ?? {})) env[k] = substitute(v, vars);

  const child: ChildProcess = spawn("sh", ["-c", substitute(spec.command, vars)], {
    cwd: root,
    env,
    stdio: ["ignore", "pipe", "pipe"],
    detached: false,
  });
  child.stdout?.pipe(log, { end: false });
  child.stderr?.pipe(log, { end: false });

  const readyPath = spec.readyUrl ?? "/";
  const deadline = Date.now() + (spec.startupTimeoutMs ?? 30000);
  let exited = false;
  child.on("exit", () => {
    exited = true;
  });

  while (Date.now() < deadline) {
    if (exited) break;
    try {
      await fetch(new URL(readyPath, baseUrl).href, { signal: AbortSignal.timeout(2000) });
      return {
        index,
        port,
        baseUrl,
        logPath,
        stop: async () => {
          if (child.exitCode === null) {
            child.kill("SIGTERM");
            await new Promise<void>((resolve) => {
              const t = setTimeout(() => {
                child.kill("SIGKILL");
                resolve();
              }, 3000);
              child.once("exit", () => {
                clearTimeout(t);
                resolve();
              });
            });
          }
          log.end();
        },
      };
    } catch {
      await new Promise((r) => setTimeout(r, 300));
    }
  }

  child.kill("SIGKILL");
  const tail = (await readFile(logPath, "utf8").catch(() => "")).split("\n").slice(-20).join("\n");
  throw new Error(
    `worker ${index}: app did not become ready at ${baseUrl}${readyPath} within ${spec.startupTimeoutMs ?? 30000}ms` +
      (exited ? " (the process exited)" : "") +
      `\napp log tail (${logPath}):\n${tail}`,
  );
}
