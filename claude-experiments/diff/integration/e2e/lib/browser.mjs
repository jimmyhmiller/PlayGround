// A driver around the `agent-browser` CLI. One isolated session per side
// (reference / diffpack) so the two builds never share cookies, storage,
// console buffers, or refs.
//
// Every invocation is asynchronous and independently killable. `spawnSync`'s
// own `timeout` is not sufficient here: agent-browser's node wrapper forks a
// native binary that inherits stdout, so killing the wrapper leaves the pipe
// open and the synchronous call blocks forever. A stuck page must cost one
// command's timeout, never the whole run.
import { spawn, spawnSync } from "node:child_process";

const DEFAULT_TIMEOUT_MS = 40_000;

const runAgentBrowser = (args, { input, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) =>
  new Promise((resolve) => {
    const child = spawn("agent-browser", args, {
      stdio: ["pipe", "pipe", "pipe"],
      detached: true, // its own process group, so the whole tree can be killed
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };
    const killTree = (signal) => {
      try {
        process.kill(-child.pid, signal);
      } catch {
        try {
          child.kill(signal);
        } catch {}
      }
    };
    const timer = setTimeout(() => {
      killTree("SIGKILL");
      finish({ status: null, stdout, stderr, timedOut: true });
    }, timeoutMs);

    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", (error) => finish({ status: null, stdout, stderr: `${stderr}${error}`, timedOut: false }));
    child.on("close", (status) => finish({ status, stdout, stderr, timedOut: false }));

    if (input !== undefined) {
      child.stdin.end(input);
    } else {
      child.stdin.end();
    }
  });

export class Browser {
  constructor(session, { initScript } = {}) {
    this.session = session;
    this.initScript = initScript;
    this.timeouts = 0;
  }

  async #run(args, opts) {
    const result = await runAgentBrowser(["--session", this.session, ...args], opts);
    if (result.timedOut) this.timeouts++;
    return result;
  }

  async #json(args, opts) {
    const r = await this.#run([...args, "--json"], opts);
    const line = r.stdout.split("\n").find((l) => l.trim().startsWith("{"));
    if (!line) return { ok: false, raw: r.stdout + r.stderr, data: null, timedOut: r.timedOut };
    try {
      const parsed = JSON.parse(line);
      return { ok: parsed.success !== false, data: parsed.data, error: parsed.error, raw: line };
    } catch (error) {
      return { ok: false, raw: r.stdout + r.stderr, data: null, error: String(error) };
    }
  }

  async open(url) {
    // `--init-script` is a global flag: it must precede the subcommand.
    const args = this.initScript ? ["--init-script", this.initScript, "open", url] : ["open", url];
    return this.#run(args);
  }

  async reload() {
    return this.#run(["reload"]);
  }

  /** Evaluate JS in the page; the source's completion value must be JSON-serializable. */
  async eval(source, { timeoutMs } = {}) {
    const r = await this.#run(["eval", "--stdin"], { input: source, timeoutMs });
    const text = r.stdout.trim();
    if (!text) return { ok: false, value: null, raw: r.stderr || r.stdout, timedOut: r.timedOut };
    // `eval` prints the JSON encoding of the completion value.
    try {
      const decoded = JSON.parse(text);
      return { ok: true, value: typeof decoded === "string" ? JSON.parse(decoded) : decoded, raw: text };
    } catch {
      return { ok: false, value: null, raw: text + r.stderr, timedOut: r.timedOut };
    }
  }

  async consoleMessages({ clear = false } = {}) {
    const r = await this.#json(["console", ...(clear ? ["--clear"] : [])]);
    return r.data?.messages ?? [];
  }

  async pageErrors({ clear = false } = {}) {
    const r = await this.#json(["errors", ...(clear ? ["--clear"] : [])]);
    return r.data?.errors ?? [];
  }

  async networkRequests({ clear = false } = {}) {
    const r = await this.#json(["network", "requests", ...(clear ? ["--clear"] : [])]);
    return (r.data?.requests ?? []).map((req) => ({
      url: req.url,
      method: req.method,
      status: req.status ?? req.response?.status ?? null,
      resourceType: req.resourceType ?? req.type ?? null,
      failed: Boolean(req.failure ?? req.failed),
    }));
  }

  async observations() {
    return {
      console: await this.consoleMessages(),
      pageErrors: await this.pageErrors(),
      network: await this.networkRequests(),
    };
  }

  async clearObservations() {
    await this.consoleMessages({ clear: true });
    await this.pageErrors({ clear: true });
    await this.networkRequests({ clear: true });
  }

  async screenshot(absolutePath) {
    return this.#run(["screenshot", absolutePath], { timeoutMs: 20_000 });
  }

  async close() {
    return this.#run(["close"], { timeoutMs: 15_000 });
  }
}

export const closeAll = () => spawnSync("agent-browser", ["close", "--all"], { encoding: "utf8", timeout: 20_000 });
