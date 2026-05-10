// Allow running inside a Claude Code session
delete (process.env as any).CLAUDECODE;

import * as net from "net";
import * as readline from "readline";
import * as child_process from "child_process";
import * as path from "path";
import * as fs from "fs";
import { query, tool, createSdkMcpServer } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

const LOG_FILE = path.join(
  path.dirname(new URL(import.meta.url).pathname),
  "beagle_agent.log"
);

function log(level: "INFO" | "WARN" | "ERROR", msg: string, data?: unknown) {
  const ts = new Date().toISOString();
  let line = `[${ts}] ${level}: ${msg}`;
  if (data !== undefined) {
    try {
      line += ` ${JSON.stringify(data, null, 2)}`;
    } catch {
      line += ` [unstringifiable data]`;
    }
  }
  line += "\n";
  try {
    fs.appendFileSync(LOG_FILE, line);
  } catch {
    // If we can't write to log file, at least stderr it
    process.stderr.write(`[log-write-failed] ${line}`);
  }
}

// Truncate log on startup so it doesn't grow unbounded
try { fs.writeFileSync(LOG_FILE, `--- Beagle Agent started at ${new Date().toISOString()} ---\n`); } catch {}

// Catch unhandled errors globally
process.on("uncaughtException", (err) => {
  log("ERROR", "Uncaught exception", { message: err.message, stack: err.stack });
  console.error(`\n[FATAL] Uncaught exception: ${err.message}`);
  console.error(`See ${LOG_FILE} for details.`);
  process.exit(1);
});

process.on("unhandledRejection", (reason) => {
  const msg = reason instanceof Error ? reason.message : String(reason);
  const stack = reason instanceof Error ? reason.stack : undefined;
  log("ERROR", "Unhandled rejection", { message: msg, stack });
  console.error(`\n[FATAL] Unhandled rejection: ${msg}`);
  console.error(`See ${LOG_FILE} for details.`);
  process.exit(1);
});

// ---------------------------------------------------------------------------
// Beagle REPL server management
// ---------------------------------------------------------------------------

const REPL_HOST = "127.0.0.1";
const REPL_PORT = 7888;
const DEFAULT_REPL_SERVER = path.join(
  process.env.HOME ?? "",
  "Documents/Code/beagle/resources/examples/repl_server.bg"
);

let beagProcess: child_process.ChildProcess | undefined;
let serverOutput: string[] = [];

interface ProcessCrashInfo {
  exitCode: number | null;
  signal: string | null;
  stdout: string;
  stderr: string;
  startedAt: Date;
  exitedAt: Date;
  bgFile: string;
}

let lastCrash: ProcessCrashInfo | undefined;

function killBeagleServer() {
  disconnectRepl();
  if (beagProcess && !beagProcess.killed) {
    log("INFO", "Killing beagle server", { pid: beagProcess.pid });
    beagProcess.kill();
    beagProcess = undefined;
    introspectReady = false;
    serverOutput = [];
  }
}

function startBeagleServer(bgFile: string, extraArgs: string[] = []): child_process.ChildProcess {
  // Kill any existing server first
  killBeagleServer();
  serverOutput = [];

  const startedAt = new Date();
  const stdoutChunks: string[] = [];
  const stderrChunks: string[] = [];

  const proc = child_process.spawn("beag", ["run", ...extraArgs, bgFile], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  proc.stdout?.on("data", (chunk: Buffer) => {
    const text = chunk.toString();
    stdoutChunks.push(text);
    serverOutput.push(text);
  });
  proc.stderr?.on("data", (chunk: Buffer) => {
    const text = chunk.toString();
    stderrChunks.push(text);
    serverOutput.push(`[stderr] ${text}`);
  });
  proc.on("exit", (code, signal) => {
    log("WARN", "Beagle process exited", {
      bgFile,
      exitCode: code,
      signal,
      stderr: stderrChunks.join(""),
      stdout: stdoutChunks.join(""),
    });
    if (beagProcess === proc) {
      lastCrash = {
        exitCode: code,
        signal: signal,
        stdout: stdoutChunks.join(""),
        stderr: stderrChunks.join(""),
        startedAt,
        exitedAt: new Date(),
        bgFile,
      };
      beagProcess = undefined;
      introspectReady = false;
    }
  });
  return proc;
}

// Reset introspect session state (declared here, used later)
let introspectReady = false;

function waitForPort(host: string, port: number, timeoutMs = 15_000): Promise<void> {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    function tryConnect() {
      if (Date.now() - start > timeoutMs) {
        reject(new Error(`Timed out waiting for ${host}:${port}`));
        return;
      }
      const sock = new net.Socket();
      sock.once("connect", () => {
        sock.destroy();
        resolve();
      });
      sock.once("error", () => {
        sock.destroy();
        setTimeout(tryConnect, 200);
      });
      sock.connect(port, host);
    }
    tryConnect();
  });
}

// ---------------------------------------------------------------------------
// Beagle REPL persistent connection
// ---------------------------------------------------------------------------

let reqCounter = 0;

interface ReplResponse {
  value?: string;
  out?: string;
  err?: string;
  ex?: string;
  status?: string[];
  id?: string;
  "suspend-depth"?: number;
  [key: string]: unknown;
}

interface PendingEval {
  messages: ReplResponse[];
  resolve: (value: string) => void;
  timer: ReturnType<typeof setTimeout>;
}

let replSocket: net.Socket | undefined;
let replConnected = false;
let replBuffer = "";
const pendingEvals = new Map<string, PendingEval>();

function disconnectRepl() {
  if (replSocket) {
    replSocket.destroy();
    replSocket = undefined;
    replConnected = false;
    replBuffer = "";
    // Reject all pending evals
    for (const [id, pending] of pendingEvals) {
      clearTimeout(pending.timer);
      pending.resolve(JSON.stringify({ error: "Connection closed" }));
    }
    pendingEvals.clear();
  }
}

function connectRepl(): Promise<void> {
  if (replConnected && replSocket) return Promise.resolve();

  return new Promise((resolve, reject) => {
    disconnectRepl();
    const sock = new net.Socket();
    replSocket = sock;
    replBuffer = "";

    sock.connect(REPL_PORT, REPL_HOST, () => {
      replConnected = true;
      resolve();
    });

    sock.on("data", (chunk) => {
      replBuffer += chunk.toString();
      const lines = replBuffer.split("\n");
      replBuffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const parsed: ReplResponse = JSON.parse(line);
          handleReplMessage(parsed);
        } catch {
          // skip unparseable lines
        }
      }
    });

    sock.on("end", () => {
      replConnected = false;
      replSocket = undefined;
      for (const [id, pending] of pendingEvals) {
        clearTimeout(pending.timer);
        pending.resolve(formatReplResponse(pending.messages));
      }
      pendingEvals.clear();
    });

    sock.on("error", (err) => {
      replConnected = false;
      replSocket = undefined;
      for (const [id, pending] of pendingEvals) {
        clearTimeout(pending.timer);
        pending.resolve(JSON.stringify({ error: err.message }));
      }
      pendingEvals.clear();
      reject(err);
    });
  });
}

function handleReplMessage(msg: ReplResponse) {
  const id = msg.id as string | undefined;
  if (!id) return;
  const pending = pendingEvals.get(id);
  if (!pending) return;

  pending.messages.push(msg);
  if (msg.status?.includes("done") || msg.status?.includes("error")) {
    clearTimeout(pending.timer);
    pendingEvals.delete(id);
    pending.resolve(formatReplResponse(pending.messages));
  }
}

async function replRequest(op: string, extra: Record<string, string> = {}): Promise<string> {
  if (!replConnected) {
    try {
      await connectRepl();
    } catch (err: any) {
      let msg = `Connection failed: ${err.message}`;
      if (lastCrash) {
        msg += `\n\n${formatCrashInfo(lastCrash)}`;
      }
      return JSON.stringify({ error: msg });
    }
  }

  const id = String(++reqCounter);
  const msg = JSON.stringify({ op, id, ...extra }) + "\n";

  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      pendingEvals.delete(id);
      resolve(JSON.stringify({ error: "Timed out after 30s" }));
    }, 30_000);

    pendingEvals.set(id, { messages: [], resolve, timer });
    replSocket!.write(msg);
  });
}

function formatReplResponse(messages: ReplResponse[]): string {
  // Combine the nREPL-style streaming messages into a single readable result
  const parts: string[] = [];
  let value: string | undefined;
  let resumable = false;
  let suspendDepth: number | undefined;

  for (const msg of messages) {
    if (msg.out) parts.push(msg.out);
    if (msg.err) parts.push(`[stderr] ${msg.err}`);
    if (msg.ex) parts.push(`[error] ${msg.ex}`);
    if (msg.value !== undefined) value = msg.value;
    if (msg.status?.includes("resumable")) resumable = true;
    if (msg["suspend-depth"] !== undefined) suspendDepth = msg["suspend-depth"];
    // Main-thread status fields
    if ((msg as any)["main-thread"] === "suspended") {
      parts.push(`[main-thread SUSPENDED] ${(msg as any).error ?? "unknown error"}`);
      parts.push(`\nUse beagle_eval to fix the broken function, then beagle_main_resume to continue.`);
    } else if ((msg as any)["main-thread"] === "running") {
      parts.push(`[main-thread running]`);
    }
  }

  const output = parts.join("");
  let result: string;
  if (output && value !== undefined) {
    result = `${output}\n=> ${value}`;
  } else if (value !== undefined) {
    result = `=> ${value}`;
  } else if (output) {
    result = output;
  } else {
    // Fallback: return the raw messages as JSON
    result = JSON.stringify(messages, null, 2);
  }

  if (resumable) {
    result += `\n\n⚠️ RESUMABLE EXCEPTION (suspend-depth: ${suspendDepth ?? "unknown"})`;
    result += `\nThe evaluation is suspended — the program is waiting for you to provide a value.`;
    result += `\nUse beagle_resume to supply a replacement value, or beagle_abort to abandon.`;
  }

  return result;
}

// ---------------------------------------------------------------------------
// Session → namespace tracking
// ---------------------------------------------------------------------------
// Remembers the last-seen namespace per session so beagle_persist can infer
// the target namespace when the caller omits it.

const sessionNamespaces = new Map<string, string>();

function extractNamespaceFromText(text: string): string | undefined {
  const match = text.match(/^\s*namespace\s+(\S+)/m);
  return match ? match[1] : undefined;
}

function trackSessionNamespace(session: string, ns: string | undefined) {
  if (ns) sessionNamespaces.set(session, ns);
}

// ---------------------------------------------------------------------------
// Beagle string literal escape (for embedding arbitrary text in a Beagle call)
// ---------------------------------------------------------------------------

function beagleStringEscape(s: string): string {
  return s
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t");
}

function isErrorResponse(result: string): boolean {
  return result.includes("[error]") || result.includes("[stderr]");
}

function extractErrorText(result: string): string | null {
  const errMatch = result.match(/\[error\]\s*(.+)/);
  if (errMatch) return errMatch[1].trim();
  const stderrMatch = result.match(/\[stderr\]\s*(.+)/);
  if (stderrMatch) return stderrMatch[1].trim();
  return null;
}

// ---------------------------------------------------------------------------
// Eval history log (append-only JSONL)
// ---------------------------------------------------------------------------

const EVAL_HISTORY_FILE = path.join(
  path.dirname(new URL(import.meta.url).pathname),
  "eval_history.jsonl"
);

interface PersistedEntry { name: string; action: string; }
interface HistoryRecord {
  ts: string;
  session: string;
  tool: "beagle_eval" | "beagle_persist";
  code: string;
  namespace?: string;
  result: string;
  persisted: PersistedEntry[];
  error: string | null;
}

function logEvalHistory(rec: HistoryRecord) {
  try {
    fs.appendFileSync(EVAL_HISTORY_FILE, JSON.stringify(rec) + "\n");
  } catch (e: any) {
    log("WARN", "Failed to append eval history", { message: e.message });
  }
}

// Lax parser for the `[{:name "…" :action "…"} …]` return shape from
// reflect/persist — used only for the history log, so we don't need to
// understand all of Beagle's printer, just grab name/action pairs.
function parsePersistedList(result: string): PersistedEntry[] {
  const entries: PersistedEntry[] = [];
  const nameFirst = /:name\s+"([^"]*)"[^{}]*:action\s+"([^"]*)"/g;
  const actionFirst = /:action\s+"([^"]*)"[^{}]*:name\s+"([^"]*)"/g;
  let m: RegExpExecArray | null;
  while ((m = nameFirst.exec(result)) !== null) {
    entries.push({ name: m[1], action: m[2] });
  }
  while ((m = actionFirst.exec(result)) !== null) {
    // Avoid duplicating entries captured by the name-first pass
    if (!entries.some((e) => e.name === m![2] && e.action === m![1])) {
      entries.push({ name: m[2], action: m[1] });
    }
  }
  return entries;
}

// ---------------------------------------------------------------------------
// Crash info formatting
// ---------------------------------------------------------------------------

function formatCrashInfo(crash: ProcessCrashInfo): string {
  const duration = crash.exitedAt.getTime() - crash.startedAt.getTime();
  const parts = [
    `Process exited`,
    `  File: ${crash.bgFile}`,
    `  Exit code: ${crash.exitCode ?? "unknown"}`,
    `  Signal: ${crash.signal ?? "none"}`,
    `  Started: ${crash.startedAt.toISOString()}`,
    `  Exited: ${crash.exitedAt.toISOString()}`,
    `  Duration: ${duration}ms`,
  ];
  if (crash.stdout.trim()) {
    parts.push(`\n--- stdout ---\n${crash.stdout.trimEnd()}`);
  }
  if (crash.stderr.trim()) {
    parts.push(`\n--- stderr ---\n${crash.stderr.trimEnd()}`);
  }
  return parts.join("\n");
}

// ---------------------------------------------------------------------------
// MCP Tools
// ---------------------------------------------------------------------------

// Actively probe the REPL by trying to connect (if needed) and round-trip a
// describe op. Returns "connected" if the round-trip succeeds, otherwise an
// error string explaining why. This is what beagle_status uses, because the
// cached `replConnected` flag lags reality: after beagle_run, the initial
// connect can fail silently (server still booting, early connection kicked),
// leaving replConnected=false even though the next eval would reconnect fine.
// Returns null on success, or a string describing why the probe failed.
async function probeReplConnection(): Promise<string | null> {
  if (!replConnected) {
    try {
      await connectRepl();
    } catch (err: any) {
      return `connect failed: ${err.message ?? err}`;
    }
  }

  const id = String(++reqCounter);
  const msg = JSON.stringify({ op: "describe", id }) + "\n";

  return new Promise<string | null>((resolve) => {
    const timer = setTimeout(() => {
      pendingEvals.delete(id);
      resolve("describe round-trip timed out after 2s");
    }, 2_000);

    pendingEvals.set(id, {
      messages: [],
      resolve: () => {
        clearTimeout(timer);
        resolve(null);
      },
      timer,
    });
    try {
      replSocket!.write(msg);
    } catch (err: any) {
      clearTimeout(timer);
      pendingEvals.delete(id);
      resolve(`write failed: ${err.message ?? err}`);
    }
  });
}

const beagleStatus = tool(
  "beagle_status",
  "Check the status of the Beagle process. Shows whether it's running and whether the REPL " +
  "is actually reachable (probed by round-tripping a describe op, not just a cached flag). " +
  "If the process crashed, shows the exit code, signal, stdout, stderr, and timing.",
  {},
  async () => {
    const parts: string[] = [];
    if (beagProcess && !beagProcess.killed) {
      parts.push(`Process is running (PID: ${beagProcess.pid})`);
      const probeError = await probeReplConnection();
      if (probeError === null) {
        parts.push("REPL connected: true (describe round-trip ok)");
      } else {
        parts.push(`REPL connected: false (${probeError})`);
      }
    } else {
      parts.push("Process is not running.");
    }
    if (lastCrash) {
      parts.push("");
      parts.push("Last exit:");
      parts.push(formatCrashInfo(lastCrash));
    }
    return { content: [{ type: "text" as const, text: parts.join("\n") }] };
  }
);

const beagleEval = tool(
  "beagle_eval",
  "Evaluate Beagle code in a REPL session. Use this for expressions, probing state, " +
  "and throwaway experiments — NOT for definitions you want to keep. " +
  "For persisting fn/struct/enum definitions to disk, use beagle_persist. " +
  "beagle_eval does no file I/O.",
  { code: z.string().describe("Beagle code to evaluate"), session: z.string().optional().describe("Session name (default: agent)") },
  async (args) => {
    const session = args.session ?? "agent";
    // Remember any namespace directive for later beagle_persist calls.
    trackSessionNamespace(session, extractNamespaceFromText(args.code));
    const result = await replRequest("eval", { code: args.code, session });
    // After every eval, check main-thread health. The game loop may have called
    // a freshly-redefined function that throws, suspending the main thread —
    // the agent has no other signal that this happened.
    let text = result;
    try {
      const mainStatus = await replRequest("main-status", { session });
      if (mainStatus.includes("SUSPENDED")) {
        text += `\n\n${mainStatus}`;
      }
    } catch {
      // Best-effort; don't fail the eval if the status check fails.
    }
    logEvalHistory({
      ts: new Date().toISOString(),
      session,
      tool: "beagle_eval",
      code: args.code,
      result: text,
      persisted: [],
      error: isErrorResponse(text) ? extractErrorText(text) : null,
    });
    return { content: [{ type: "text" as const, text }] };
  }
);

const beagleDescribe = tool(
  "beagle_describe",
  "Get information about the REPL server's capabilities and supported operations.",
  {},
  async () => {
    const result = await replRequest("describe");
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleSessions = tool(
  "beagle_sessions",
  "List all active REPL sessions.",
  {},
  async () => {
    const result = await replRequest("ls-sessions");
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleInterrupt = tool(
  "beagle_interrupt",
  "Interrupt a long-running evaluation in a session.",
  { session: z.string().describe("Session to interrupt") },
  async (args) => {
    const result = await replRequest("interrupt", { session: args.session });
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleResume = tool(
  "beagle_resume",
  "Resume a suspended resumable exception by providing a value. When an eval hits a resumable " +
  "exception, the program suspends and waits for you to supply a replacement value. " +
  "The code you provide is evaluated, and the result becomes the return value of the " +
  "original throw expression — execution then continues from that point. " +
  "While suspended, you can also use beagle_eval in the same session to inspect state " +
  "before deciding what value to resume with.",
  {
    code: z.string().describe("Beagle expression to evaluate — its result becomes the return value of the throw site"),
    session: z.string().optional().describe("Session name (default: agent)"),
  },
  async (args) => {
    const session = args.session ?? "agent";
    const result = await replRequest("resume", { code: args.code, session });
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleAbort = tool(
  "beagle_abort",
  "Abort a suspended resumable exception, abandoning the suspended evaluation. " +
  "Use this when you don't want to resume — the exception is discarded and the " +
  "suspended evaluation terminates.",
  {
    session: z.string().optional().describe("Session name (default: agent)"),
  },
  async (args) => {
    const session = args.session ?? "agent";
    const result = await replRequest("abort", { session });
    return { content: [{ type: "text" as const, text: result }] };
  }
);

// ---------------------------------------------------------------------------
// Main-thread crash recovery tools
// ---------------------------------------------------------------------------

const beagleMainStatus = tool(
  "beagle_main_status",
  "Check whether the main thread (game loop / GUI) is running or suspended due to a crash. " +
  "If suspended, shows the error. Use beagle_main_resume or beagle_main_abort to recover.",
  {},
  async () => {
    const result = await replRequest("main-status", {});
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleMainResume = tool(
  "beagle_main_resume",
  "Resume the main thread after it crashed. The main thread (game loop) is suspended waiting " +
  "for you to fix the problem. First use beagle_eval to redefine the broken function, then " +
  "call this to resume execution. Optionally provide code to evaluate — its result becomes " +
  "the return value at the crash site.",
  {
    code: z.string().optional().describe("Optional Beagle expression — result becomes the return value at the crash site"),
  },
  async (args) => {
    const extra: Record<string, string> = {};
    if (args.code) extra.code = args.code;
    const result = await replRequest("main-resume", extra);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleMainAbort = tool(
  "beagle_main_abort",
  "Abort the suspended main thread. The main function will return and the program will " +
  "likely exit. Use this when you can't fix the crash and want to restart with beagle_run.",
  {},
  async () => {
    const result = await replRequest("main-abort", {});
    return { content: [{ type: "text" as const, text: result }] };
  }
);

// ---------------------------------------------------------------------------
// Program lifecycle tool
// ---------------------------------------------------------------------------

const beagleRun = tool(
  "beagle_run",
  "Start (or restart) a Beagle program with an embedded REPL server. " +
  "If a .bg file is given, it runs that file's main() with a REPL server embedded inside it, " +
  "so you can eval code in the context of the running program. " +
  "If no file is given, starts the default standalone REPL server. " +
  "If a server is already running, it is killed and replaced.",
  { file: z.string().optional().describe("Path to a .bg file to run (default: built-in REPL server)") },
  async (args) => {
    // Disconnect persistent REPL connection before restarting
    disconnectRepl();

    if (args.file) {
      // Run-with-REPL pattern: generate a wrapper that embeds the REPL server
      // inside the user's running program
      const sourceDir = path.dirname(path.resolve(args.file));
      const fileContent = fs.readFileSync(args.file, "utf-8");
      const nsMatch = fileContent.match(/^\s*namespace\s+(\S+)/m);
      const targetNs = nsMatch ? nsMatch[1] : undefined;

      if (!targetNs) {
        return { content: [{ type: "text" as const, text: `Error: Could not find a namespace declaration in ${args.file}. The file must have a \`namespace <name>\` at the top.` }] };
      }

      // Seed the default session's namespace so beagle_persist can infer
      // the target namespace without an explicit argument on first use.
      if (targetNs) sessionNamespaces.set("agent", targetNs);

      // Generate wrapper that imports repl-main and the target namespace
      const wrapperCode = `namespace __repl_runner

use beagle.repl-main as repl-main
use ${targetNs} as target

fn main() {
    eval("namespace ${targetNs}")
    repl-main/run-with-repl("${REPL_HOST}", ${REPL_PORT}, fn() {
        target/main()
    })
}
`;
      const wrapperPath = "/tmp/__beagle_repl_runner.bg";
      fs.writeFileSync(wrapperPath, wrapperCode);

      beagProcess = startBeagleServer(wrapperPath, ["-I", sourceDir]);
    } else {
      // Default: start the standalone REPL server
      beagProcess = startBeagleServer(DEFAULT_REPL_SERVER);
    }

    try {
      await waitForPort(REPL_HOST, REPL_PORT);
    } catch {
      const output = serverOutput.join("");
      const crashInfo = lastCrash ? `\n\n${formatCrashInfo(lastCrash)}` : "";
      killBeagleServer();
      return { content: [{ type: "text" as const, text: `Failed to start server.\n\nServer output:\n${output}${crashInfo}` }] };
    }

    // Connect the persistent socket
    try {
      await connectRepl();
    } catch {
      // Non-fatal: will auto-connect on first eval
    }

    const output = serverOutput.join("");
    const target = args.file ?? DEFAULT_REPL_SERVER;
    return { content: [{ type: "text" as const, text: `Server started: ${target}\n\nStartup output:\n${output}` }] };
  }
);

const beagleLoad = tool(
  "beagle_load",
  "Load a .bg file into the running REPL by reading it from disk and evaluating its contents. " +
  "This is how you bring user code into the live program. The file's definitions " +
  "(functions, structs, enums, etc.) become available in the REPL immediately.",
  {
    file: z.string().describe("Path to a .bg file to load"),
    session: z.string().optional().describe("Session name (default: agent)"),
  },
  async (args) => {
    let code: string;
    try {
      code = fs.readFileSync(args.file, "utf-8");
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Error reading file: ${e.message}` }] };
    }
    const session = args.session ?? "agent";
    trackSessionNamespace(session, extractNamespaceFromText(code));
    const result = await replRequest("eval", { code, session });
    return { content: [{ type: "text" as const, text: `Loaded ${args.file}\n\n${result}` }] };
  }
);

// ---------------------------------------------------------------------------
// Introspection tools — structured wrappers around beagle.reflect
// ---------------------------------------------------------------------------

const INTROSPECT_SESSION = "introspect";

async function ensureIntrospectSession() {
  if (!introspectReady) {
    await replRequest("eval", { code: "use beagle.reflect as reflect", session: INTROSPECT_SESSION });
    introspectReady = true;
  }
}

async function introspectEval(code: string): Promise<string> {
  await ensureIntrospectSession();
  return replRequest("eval", { code, session: INTROSPECT_SESSION });
}

const beagleListNamespaces = tool(
  "beagle_list_namespaces",
  "List all namespaces loaded in the running Beagle program.",
  {},
  async () => {
    const result = await introspectEval("sort(reflect/all-namespaces())");
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleNamespaceInfo = tool(
  "beagle_namespace_info",
  "Get detailed info about a namespace: its functions (with args and docs), structs, and enums. " +
  "Use this to understand what's available before writing code.",
  { namespace: z.string().describe("Namespace name, e.g. \"beagle.core\"") },
  async (args) => {
    // Get members list, then for each member get doc/args info
    const code = `
let ns = "${args.namespace.replace(/"/g, '\\"')}"
let members = sort(reflect/namespace-members(ns))
let info = reflect/namespace-info(ns)
let result = "Namespace: " + ns + "\\n"
let fns = get(info, :functions)
if fns != null {
  result = result + "\\nFunctions (" + to_string(length(fns)) + "):\\n"
  for f in fns {
    let name = get(f, :name)
    let args_list = get(f, :args)
    let doc = get(f, :doc)
    let variadic = get(f, :variadic)
    let sig = "  " + name + "("
    if args_list != null {
      sig = sig + join(args_list, ", ")
    }
    if variadic == true {
      sig = sig + "..."
    }
    sig = sig + ")"
    if doc != null {
      sig = sig + " — " + doc
    }
    result = result + sig + "\\n"
  }
}
let structs = get(info, :structs)
if structs != null {
  result = result + "\\nStructs (" + to_string(length(structs)) + "):\\n"
  for s in structs {
    let name = get(s, :name)
    let fields = get(s, :fields)
    result = result + "  " + name
    if fields != null {
      result = result + " { " + join(fields, ", ") + " }"
    }
    result = result + "\\n"
  }
}
let enums = get(info, :enums)
if enums != null {
  result = result + "\\nEnums (" + to_string(length(enums)) + "):\\n"
  for e in enums {
    let name = get(e, :name)
    let variants = get(e, :variants)
    result = result + "  " + name
    if variants != null {
      result = result + " { " + join(variants, " | ") + " }"
    }
    result = result + "\\n"
  }
}
result
`;
    const result = await introspectEval(code);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleSearch = tool(
  "beagle_search",
  "Search for functions by name or docstring substring. Returns matching fully-qualified function names.",
  { query: z.string().describe("Search term to match against function names and docstrings") },
  async (args) => {
    const code = `sort(reflect/apropos("${args.query.replace(/"/g, '\\"')}"))`;
    const result = await introspectEval(code);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleDoc = tool(
  "beagle_doc",
  "Get the documentation, arguments, and type info for a specific function or value. " +
  "Pass the fully-qualified name or a direct reference.",
  { name: z.string().describe("Fully-qualified function name, e.g. \"beagle.core/map\" or just \"map\" if in scope") },
  async (args) => {
    const code = `
let val = ${args.name}
let descriptor = reflect/type-of(val)
let info = reflect/info(descriptor)
let result = "Name: " + "${args.name.replace(/"/g, '\\"')}" + "\\n"
result = result + "Kind: " + to_string(get(info, :kind)) + "\\n"
let doc = reflect/doc(descriptor)
if doc != null {
  result = result + "Doc: " + doc + "\\n"
}
let args_list = reflect/args(descriptor)
if args_list != null {
  result = result + "Args: (" + join(args_list, ", ")
  if reflect/variadic?(descriptor) {
    result = result + "..."
  }
  result = result + ")\\n"
}
let fields = reflect/fields(descriptor)
if fields != null {
  result = result + "Fields: " + to_string(fields) + "\\n"
}
let variants = reflect/variants(descriptor)
if variants != null {
  result = result + "Variants: " + to_string(variants) + "\\n"
}
result
`;
    const result = await introspectEval(code);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleSource = tool(
  "beagle_source",
  "Get the source text of a function, struct, or enum. Pass a bare name (resolved via the " +
  "introspect session's imports — you may need to use `use foo.bar as bar` first in that " +
  "session) or a fully-qualified reference. Returns the exact source as stored by the " +
  "compiler, including any leading `///` doc comments. Returns null for REPL/eval defs, " +
  "builtins, and FFI. This is how you READ a definition before editing it — no file I/O needed.",
  { name: z.string().describe("Reference to a function, struct, or enum") },
  async (args) => {
    const code = `reflect/source(${args.name})`;
    const result = await introspectEval(code);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleNamespaceSource = tool(
  "beagle_namespace_source",
  "Get the concatenated source of every definition in a namespace (structs, then enums, " +
  "then functions). Definitions without stored source (builtins, FFI, anonymous closures) are " +
  "skipped. Use this when you want to understand or refactor a whole file.",
  { namespace: z.string().describe("Namespace name, e.g. 'my.module'") },
  async (args) => {
    const code = `reflect/namespace-source("${beagleStringEscape(args.namespace)}")`;
    const result = await introspectEval(code);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beagleLocation = tool(
  "beagle_location",
  "Get the source location of a function, struct, or enum as {:file, :byte-start, :byte-end, " +
  ":line-start, :line-end}, or null for REPL/builtin/FFI definitions. Useful for inspection; " +
  "you do NOT need to call this to decide update-vs-append — beagle_persist handles that " +
  "dispatch internally.",
  { name: z.string().describe("Reference to a function, struct, or enum") },
  async (args) => {
    const code = `reflect/location(${args.name})`;
    const result = await introspectEval(code);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

const beaglePersist = tool(
  "beagle_persist",
  "Persist one or more top-level definitions (fn, struct, enum) to disk AND into the running " +
  "program. For each def: if it already exists with an on-disk location, the file is updated " +
  "in place with a drift check; if not, the def is appended to the namespace's file. The text " +
  "is compiled with file context so source locations are tracked for later edits.\n\n" +
  "This is the canonical path for definitions you want to keep. For expressions or throwaway " +
  "experiments use beagle_eval instead. Calling beagle_eval and then beagle_persist with the " +
  "same text compiles the def twice — skip straight to beagle_persist.\n\n" +
  "Namespace: if omitted, inferred from a leading `namespace X` in text, or from the session's " +
  "last-seen namespace (set by beagle_run, beagle_load, or a prior call that had a namespace " +
  "directive). If no namespace can be inferred, the call errors and you should retry with an " +
  "explicit namespace.\n\n" +
  "Drift errors: if the file changed since the def was loaded, persist refuses and surfaces the " +
  "message. DO NOT retry blindly — re-fetch the fresh source with beagle_source (or " +
  "beagle_namespace_source), because whatever else changed may affect your edit.\n\n" +
  "Returns a list of {:name, :action} records, one per persisted def, where :action is " +
  "\"updated\" or \"appended\".",
  {
    text: z.string().describe("One or more top-level definitions (fn/struct/enum)"),
    namespace: z.string().optional().describe("Target namespace (e.g. 'my.module'). Inferred if omitted."),
    session: z.string().optional().describe("Session name (default: agent)"),
  },
  async (args) => {
    const session = args.session ?? "agent";
    const ns =
      args.namespace ??
      extractNamespaceFromText(args.text) ??
      sessionNamespaces.get(session);

    if (!ns) {
      const msg =
        "Error: could not infer target namespace. Pass `namespace` explicitly, or call " +
        "beagle_load first, or include a `namespace X` directive in `text`.";
      logEvalHistory({
        ts: new Date().toISOString(),
        session,
        tool: "beagle_persist",
        code: args.text,
        result: msg,
        persisted: [],
        error: msg,
      });
      return { content: [{ type: "text" as const, text: msg }] };
    }
    trackSessionNamespace(session, ns);

    const code = `reflect/persist("${beagleStringEscape(ns)}", "${beagleStringEscape(args.text)}")`;
    const result = await introspectEval(code);

    const persisted = parsePersistedList(result);
    const err = isErrorResponse(result) ? extractErrorText(result) : null;

    logEvalHistory({
      ts: new Date().toISOString(),
      session,
      tool: "beagle_persist",
      code: args.text,
      namespace: ns,
      result,
      persisted,
      error: err,
    });

    return { content: [{ type: "text" as const, text: result }] };
  }
);

const server = createSdkMcpServer({
  name: "beagle-repl",
  tools: [
    beagleRun, beagleLoad, beagleEval, beagleDescribe, beagleSessions, beagleInterrupt,
    beagleResume, beagleAbort, beagleStatus,
    beagleMainStatus, beagleMainResume, beagleMainAbort,
    beagleListNamespaces, beagleNamespaceInfo, beagleSearch, beagleDoc,
    beagleSource, beagleNamespaceSource, beagleLocation, beaglePersist,
  ],
});

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `\
You are a Beagle live coding agent. You interact with a running Beagle program \
through a REPL socket connection. You have NO file system tools — no Read, Glob, \
Grep, Write, Edit. Everything — reading source, writing source, exploring the \
running program, testing — goes through the MCP tools below. Source reading happens \
via beagle_source / beagle_namespace_source, source writing via beagle_persist.

## Edit loop (the thing you do most)

1. \`beagle_source(foo)\` → returns the current source of \`foo\` from the running program.
2. Compose the new version in your reply.
3. \`beagle_persist(namespace, new_text)\` → compiles it, installs it in the running \
   program, and writes it to disk in one step. For updates, Beagle drift-checks against \
   its stored copy of the file; for new defs, Beagle appends to the namespace's file.

For expressions, probes, and throwaway experiments use **beagle_eval**. beagle_eval \
does NOT touch disk. If you're going to keep a def, skip straight to beagle_persist — \
going eval → persist with the same text compiles it twice.

## Your tools

### Lifecycle
- **beagle_run**: Start (or restart) a Beagle program with an embedded REPL server. \
  If given a .bg file, runs that file's main() with the REPL server embedded inside it. \
  If no file is given, starts a standalone REPL server. Call this before anything else.
- **beagle_load**: Load a .bg file into the running REPL by reading it from disk and \
  evaluating its contents. Use when the user points you at a file.

### Source (reading / writing)
- **beagle_source(name)**: Source text of a fn/struct/enum from the running program, \
  exactly as compiled (including \`///\` doc comments). Returns null for REPL/eval \
  defs, builtins, and FFI. This replaces "read the file".
- **beagle_namespace_source(ns)**: Concatenated source for every def in a namespace — \
  structs, enums, functions. Use for understanding or refactoring a whole file.
- **beagle_location(name)**: File + byte/line range for a def, or null. For inspection; \
  you do NOT need it to choose between update and append — persist handles that.
- **beagle_persist(text, namespace?)**: The ONLY way to write source. Accepts one or \
  more top-level fn/struct/enum defs. Per def: updates in place (with drift check) if \
  it exists, appends to the namespace's file otherwise. Recompiles with file context. \
  Namespace is inferred from a leading \`namespace X\` in text, or the session's \
  last-seen namespace, if you don't pass it.

### Eval / control flow
- **beagle_eval(code)**: Evaluate an expression in a session. Use for probes and \
  experiments — NOT for definitions you want to keep.
- **beagle_interrupt**: Stop a long-running evaluation.
- **beagle_resume / beagle_abort**: Resume or abandon a suspended resumable exception.
- **beagle_status**: Is the Beagle process running? If it crashed, exit code + stdout/stderr.
- **beagle_main_status / beagle_main_resume / beagle_main_abort**: Inspect and recover \
  from main-thread (game loop / GUI) crashes.
- **beagle_describe**: What ops the REPL server supports.
- **beagle_sessions**: List active REPL sessions.

### Introspection
- **beagle_list_namespaces**: All loaded namespaces.
- **beagle_namespace_info(ns)**: Functions (signatures + docs), structs (fields), enums \
  (variants) in a namespace.
- **beagle_search(query)**: Apropos — match function name/docstring substring.
- **beagle_doc(name)**: Args, doc, kind, fields/variants for one value.

## How you work

1. **Explore first**: beagle_list_namespaces + beagle_namespace_info to orient. \
   beagle_search / beagle_doc for specifics.
2. **Read before you write**: beagle_source is always cheaper than guessing. Do it \
   before proposing an edit to an existing definition.
3. **Develop incrementally**: beagle_persist one def at a time, test with beagle_eval \
   between changes.
4. **Sessions**: "agent" is the default. "scratch" for throwaway experiments. Use \
   different sessions if you want to isolate state.
5. **Think in terms of a live program**: every persist changes both disk and memory \
   atomically. You're editing a running system, not a dead tree of files.

## Drift errors

If beagle_persist returns something like \
\`reflect/write-source: file contents have changed since this definition was loaded\`, \
the file was modified outside the runtime (another editor, git checkout, etc.) since \
Beagle loaded that def. Do NOT retry with the same text. Instead:

1. Call **beagle_source** (or **beagle_namespace_source**) to get the current state.
2. Re-derive your edit against the fresh source — whatever else changed might affect it.
3. Call beagle_persist again with the updated text.

Tell the user what drifted if it looks surprising.

## Resumable exceptions

Beagle supports **resumable exceptions** — when code throws an exception inside a \
resumable try/catch, the program suspends instead of unwinding. The REPL holds the \
continuation alive, letting you inspect state, fix the problem, and resume execution \
with a replacement value.

### How it works

1. An eval hits a throw inside a resumable handler.
2. The REPL response comes back with status \`["error", "resumable"]\` and a \`suspend-depth\`.
3. The program is **paused at the throw site**, waiting for your input.
4. While suspended, you can use **beagle_eval** in the same session to inspect \
   variables, check state, or evaluate fix code.
5. Use **beagle_resume** with a Beagle expression — it evaluates your code and the \
   result becomes the return value of the original throw, so execution continues normally.
6. Or use **beagle_abort** to abandon the suspended evaluation entirely.

### Example workflow

\`\`\`
// Eval some code that throws:
beagle_eval("let x = compute_something(bad_input)")
// → [error] "Invalid input: ..."
//   ⚠️ RESUMABLE EXCEPTION (suspend-depth: 1)

// Inspect state while suspended:
beagle_eval("bad_input")
// → => {:data nil, :format "csv"}

// Resume with a corrected value:
beagle_resume("compute_fallback(default_input)")
// → => <result of the rest of the computation>
\`\`\`

### Key points

- Resumable exceptions **nest** — you can have multiple levels of suspension. \
  Resume/abort always operates on the innermost (most recent) one.
- The \`suspend-depth\` tells you how many levels deep you are.
- While suspended, the eval thread is blocked — you can still eval new code in the \
  same session (the REPL handles nested evals), but the original computation is paused.
- This is powerful for interactive debugging: when something goes wrong, you can \
  inspect the exact state at the failure point and provide a fix value without \
  restarting the computation.

## Main-thread crash recovery

When running a GUI program (e.g. raylib game) with beagle_run, the main thread runs \
the game loop while the REPL server runs on a background thread. If you redefine a \
function with a bug and the game loop calls it, the **main thread crashes** — but \
instead of killing the process, it suspends and waits for REPL recovery.

### Workflow

1. You redefine a function via beagle_persist — it has a bug.
2. The game loop calls the broken function and throws an error.
3. The main thread suspends. Use **beagle_main_status** to see the error.
4. Fix the function with **beagle_persist** (same def, corrected body).
5. Use **beagle_main_resume** to continue the game loop from where it crashed.
6. Or use **beagle_main_abort** if you want to give up and restart with beagle_run.

**Important**: The REPL server stays alive during a main-thread crash. You can still \
eval code, persist fixes, and inspect state. The main thread is just paused waiting \
for your signal to continue.

## Beagle language basics

Beagle is a functional language with:
- \`fn name(args) { body }\` for function definitions
- \`let x = expr\` for bindings
- Pattern matching with \`match\`
- Namespaces (\`namespace foo\`)
- \`use module.name as alias\` for imports
- Protocols (similar to typeclasses/interfaces)
- Structs: \`struct Point { x, y }\`
- Enums: \`enum Color { Red, Green, Blue }\`
- Keywords: \`:name\`, \`:kind\` (like Clojure keywords)
- Persistent data structures: vectors \`[1, 2, 3]\`, maps \`{:a 1, :b 2}\`

When you persist a definition, it updates in the running program AND on disk atomically \
— no restart needed, no separate file-write step. beagle_eval, by contrast, updates the \
running program only and does not touch disk — use it for experiments.

## Reflection API (beagle.reflect)

For ad-hoc introspection inside beagle_eval, you can use the reflect API directly. \
Import with: \`use beagle.reflect as reflect\`

### Type inspection
- \`reflect/type-of(value)\` — get a type descriptor for any value
- \`reflect/kind(value)\` — returns \`:struct\`, \`:enum\`, \`:function\`, or \`:primitive\`
- \`reflect/name(descriptor)\` — get the type name as a string
- \`reflect/info(descriptor)\` — get all metadata as a map

### Structure inspection
- \`reflect/fields(struct_or_descriptor)\` — field names as vector of strings
- \`reflect/variants(enum_or_descriptor)\` — variant names as vector of strings
- \`reflect/args(fn_or_descriptor)\` — argument names as vector of strings
- \`reflect/doc(descriptor)\` — docstring or null
- \`reflect/variadic?(descriptor)\` — true if function takes variable args

### Type predicates
- \`reflect/struct?(value)\`, \`reflect/enum?(value)\`, \`reflect/function?(value)\`, \`reflect/primitive?(value)\`

### Example usage in eval
\`\`\`
use beagle.reflect as reflect
reflect/name(reflect/type-of(42))          // => "Int"
reflect/fields(Point)                       // => ["x", "y"]
reflect/args(map)                           // => ["f", "coll"]
reflect/doc(reflect/type-of(println))       // => "Print a value..."
\`\`\`
`;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  // Clean up beag process on exit
  process.on("exit", killBeagleServer);
  process.on("SIGINT", () => { killBeagleServer(); process.exit(0); });
  process.on("SIGTERM", () => { killBeagleServer(); process.exit(0); });

  console.log("Beagle Live Code Agent");
  console.log("======================");
  console.log(`Logging to: ${LOG_FILE}`);
  console.log("Ask me to run a .bg file or start the default REPL server.\n");

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  // Make stdin emit 'keypress' events so we can listen for Escape during a
  // running query without disturbing rl.question (which only consumes whole lines).
  readline.emitKeypressEvents(process.stdin);

  const prompt = (): Promise<string> =>
    new Promise((resolve) => rl.question("> ", resolve));

  let sessionId: string | undefined;

  while (true) {
    let userInput: string;
    try {
      userInput = await prompt();
    } catch {
      console.log("\nBye!");
      break;
    }

    if (!userInput.trim()) continue;

    // First turn: full options. Subsequent turns: resume the session.
    // MCP servers and permissions must be passed on every turn (not persisted by session)
    const baseOptions = {
      mcpServers: { "beagle-repl": server },
      permissionMode: "bypassPermissions" as const,
      allowDangerouslySkipPermissions: true,
    };

    const options = sessionId
      ? { ...baseOptions, resume: sessionId }
      : {
          ...baseOptions,
          systemPrompt: SYSTEM_PROMPT,
          allowedTools: [
            "mcp__beagle-repl__beagle_run",
            "mcp__beagle-repl__beagle_load",
            "mcp__beagle-repl__beagle_eval",
            "mcp__beagle-repl__beagle_describe",
            "mcp__beagle-repl__beagle_sessions",
            "mcp__beagle-repl__beagle_interrupt",
            "mcp__beagle-repl__beagle_resume",
            "mcp__beagle-repl__beagle_abort",
            "mcp__beagle-repl__beagle_status",
            "mcp__beagle-repl__beagle_main_status",
            "mcp__beagle-repl__beagle_main_resume",
            "mcp__beagle-repl__beagle_main_abort",
            "mcp__beagle-repl__beagle_list_namespaces",
            "mcp__beagle-repl__beagle_namespace_info",
            "mcp__beagle-repl__beagle_search",
            "mcp__beagle-repl__beagle_doc",
            "mcp__beagle-repl__beagle_source",
            "mcp__beagle-repl__beagle_namespace_source",
            "mcp__beagle-repl__beagle_location",
            "mcp__beagle-repl__beagle_persist",
          ],
          model: "claude-sonnet-4-6",
        };

    log("INFO", "Sending query to Claude", { promptLength: userInput.length, hasSession: !!sessionId });

    // Per-query AbortController so Escape can cancel mid-flight.
    const abortController = new AbortController();
    let aborted = false;

    // Switch stdin to raw mode while the query runs so we can catch Escape
    // (and Ctrl+C) as individual keypresses. We restore the original mode in
    // the `finally` block so the next rl.question() works normally.
    const wasRaw = (process.stdin as any).isRaw === true;
    if (process.stdin.isTTY) process.stdin.setRawMode(true);

    const onKeypress = (_str: string, key: { name?: string; ctrl?: boolean; sequence?: string } | undefined) => {
      if (!key) return;
      if (key.name === "escape") {
        if (!aborted) {
          aborted = true;
          console.log("\n\x1b[33m[escape — aborting query…]\x1b[0m");
          log("INFO", "User aborted query via escape");
          abortController.abort();
        }
      } else if (key.ctrl && key.name === "c") {
        // Preserve normal Ctrl+C behavior (the global SIGINT handler kills the
        // beag child and exits). Without this, raw mode swallows Ctrl+C.
        process.kill(process.pid, "SIGINT");
      }
    };
    process.stdin.on("keypress", onKeypress);

    try {
      for await (const message of query({ prompt: userInput, options: { ...options, abortController } })) {
        log("INFO", "Received message", { type: message.type, subtype: (message as any).subtype });

        if (message.type === "assistant") {
          // Full assistant message — print text blocks, show tool use calls
          const msg = message as any;
          for (const block of msg.message?.content ?? []) {
            if (block.type === "text" && block.text) {
              console.log(block.text);
            } else if (block.type === "tool_use") {
              const input = block.input as Record<string, unknown>;
              const summary = input.code
                ? `${block.name}(${String(input.code).slice(0, 80)}${String(input.code).length > 80 ? "..." : ""})`
                : `${block.name}(${JSON.stringify(input).slice(0, 80)})`;
              console.log(`\x1b[2m⚙ ${summary}\x1b[0m`);
              log("INFO", "Tool call", { tool: block.name, input });
            }
          }
        } else if (message.type === "result") {
          const result = (message as any).result;
          // Don't print result text — it duplicates the last assistant message's text block
          log("INFO", "Query completed with result", { resultLength: result?.length });
          // The SDK's async iterator sometimes doesn't close after the terminal
          // `result` message, leaving us hung waiting for a next message that
          // never arrives. `result` is documented as the end-of-query signal,
          // so break explicitly. The for-await semantics call iterator.return()
          // for us, which cleans up the underlying stream.
          break;
        } else if (
          message.type === "system" &&
          (message as any).subtype === "init" &&
          !sessionId
        ) {
          sessionId = (message as any).session_id;
          log("INFO", "Session initialized", { sessionId });
        }
      }
      log("INFO", "Query stream ended normally");
    } catch (err: any) {
      if (aborted) {
        // The SDK throws when its subprocess is killed by abort — that's
        // expected, not a real failure. Just log and move on.
        log("INFO", "Query aborted by user", { message: err.message });
      } else {
        log("ERROR", "Query failed", { message: err.message, stack: err.stack, name: err.name });
        console.error(`\n[ERROR] Agent query failed: ${err.message}`);
        console.error(`See ${LOG_FILE} for details.`);
        // Don't exit — let the user try again
      }
    } finally {
      process.stdin.off("keypress", onKeypress);
      if (process.stdin.isTTY) process.stdin.setRawMode(wasRaw);
    }
  }

  rl.close();
}

main();
