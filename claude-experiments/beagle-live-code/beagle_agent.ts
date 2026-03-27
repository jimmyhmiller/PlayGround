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
    log("WARN", "Beagle process exited", { bgFile, exitCode: code, signal });
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

  for (const msg of messages) {
    if (msg.out) parts.push(msg.out);
    if (msg.err) parts.push(`[stderr] ${msg.err}`);
    if (msg.ex) parts.push(`[error] ${msg.ex}`);
    if (msg.value !== undefined) value = msg.value;
  }

  const output = parts.join("");
  if (output && value !== undefined) {
    return `${output}\n=> ${value}`;
  } else if (value !== undefined) {
    return `=> ${value}`;
  } else if (output) {
    return output;
  }
  // Fallback: return the raw messages as JSON
  return JSON.stringify(messages, null, 2);
}

// ---------------------------------------------------------------------------
// Source file write-back
// ---------------------------------------------------------------------------

// Maps namespace name → absolute file path
const namespaceFiles = new Map<string, string>();
// Maps session name → current namespace
const sessionNamespaces = new Map<string, string>();

function trackFileNamespace(filePath: string, content?: string) {
  const text = content ?? fs.readFileSync(filePath, "utf-8");
  const match = text.match(/^\s*namespace\s+(\S+)/m);
  if (match) {
    namespaceFiles.set(match[1], path.resolve(filePath));
  }
}

function trackSessionNamespace(session: string, code: string) {
  // Check if the eval code sets a namespace
  const match = code.match(/^\s*namespace\s+(\S+)/m);
  if (match) {
    sessionNamespaces.set(session, match[1]);
  }
}

interface Definition {
  kind: string;   // "fn", "struct", "enum", "let"
  name: string;
  text: string;   // Full definition text
}

/** Extract top-level definitions from Beagle code */
function extractDefinitions(code: string): Definition[] {
  const defs: Definition[] = [];
  const lines = code.split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const match = line.match(/^(fn|struct|enum|let)\s+(\w+)/);
    if (!match) {
      i++;
      continue;
    }

    const kind = match[1];
    const name = match[2];
    const startLine = i;

    // Find the end of this definition by brace counting
    let depth = 0;
    let foundOpen = false;
    let endLine = i;

    for (let j = i; j < lines.length; j++) {
      for (const ch of lines[j]) {
        if (ch === "{") { depth++; foundOpen = true; }
        else if (ch === "}") { depth--; }
      }
      endLine = j;
      if (foundOpen && depth <= 0) break;
      // For simple `let x = value` with no braces, end at this line
      if (!foundOpen && j === i && kind === "let") break;
    }

    const text = lines.slice(startLine, endLine + 1).join("\n");
    defs.push({ kind, name, text });
    i = endLine + 1;
  }

  return defs;
}

/** Find the line range of an existing definition in a file.
 *  Returns [startLine, endLine] (0-indexed) or null if not found. */
function findDefinitionRange(fileLines: string[], kind: string, name: string): [number, number] | null {
  for (let i = 0; i < fileLines.length; i++) {
    const re = new RegExp(`^${kind}\\s+${name}\\b`);
    if (!re.test(fileLines[i])) continue;

    // Found the start — now find the end via brace counting
    let depth = 0;
    let foundOpen = false;
    let endLine = i;

    for (let j = i; j < fileLines.length; j++) {
      for (const ch of fileLines[j]) {
        if (ch === "{") { depth++; foundOpen = true; }
        else if (ch === "}") { depth--; }
      }
      endLine = j;
      if (foundOpen && depth <= 0) break;
      if (!foundOpen && j === i && kind === "let") break;
    }

    return [i, endLine];
  }
  return null;
}

/** After a successful eval, persist any definitions back to the source file */
function persistDefinitions(code: string, session: string) {
  const ns = sessionNamespaces.get(session);
  if (!ns) return;
  const filePath = namespaceFiles.get(ns);
  if (!filePath) return;

  const defs = extractDefinitions(code);
  if (defs.length === 0) return;

  let fileContent: string;
  try {
    fileContent = fs.readFileSync(filePath, "utf-8");
  } catch {
    return;
  }

  let fileLines = fileContent.split("\n");

  for (const def of defs) {
    const range = findDefinitionRange(fileLines, def.kind, def.name);
    const defLines = def.text.split("\n");

    if (range) {
      // Replace existing definition
      fileLines.splice(range[0], range[1] - range[0] + 1, ...defLines);
    } else {
      // Append new definition at end of file
      // Add a blank line separator if file doesn't end with one
      if (fileLines.length > 0 && fileLines[fileLines.length - 1].trim() !== "") {
        fileLines.push("");
      }
      fileLines.push(...defLines);
    }
  }

  fs.writeFileSync(filePath, fileLines.join("\n"));
}

/** Check if the REPL response indicates an error */
function isErrorResponse(result: string): boolean {
  return result.includes("[error]") || result.includes("[stderr]");
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

const beagleStatus = tool(
  "beagle_status",
  "Check the status of the Beagle process. Shows whether it's running, and if it crashed, " +
  "shows the exit code, signal, stdout, stderr, and timing information.",
  {},
  async () => {
    const parts: string[] = [];
    if (beagProcess && !beagProcess.killed) {
      parts.push(`Process is running (PID: ${beagProcess.pid})`);
      parts.push(`REPL connected: ${replConnected}`);
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
  "Evaluate Beagle code in a REPL session. Use this for everything: defining functions, " +
  "testing expressions, introspecting namespaces, running reflection queries. " +
  "The code is evaluated in the live running Beagle program. " +
  "If you define a function, it updates in the running program AND the source file automatically.",
  { code: z.string().describe("Beagle code to evaluate"), session: z.string().optional().describe("Session name (default: agent)") },
  async (args) => {
    const session = args.session ?? "agent";
    trackSessionNamespace(session, args.code);
    const result = await replRequest("eval", { code: args.code, session });
    // Auto-persist definitions back to source file on success
    if (!isErrorResponse(result)) {
      persistDefinitions(args.code, session);
    }
    return { content: [{ type: "text" as const, text: result }] };
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

      // Track namespace → file for write-back
      trackFileNamespace(args.file, fileContent);

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
    // Track namespace → file mapping for write-back
    trackFileNamespace(args.file, code);
    const session = args.session ?? "agent";
    trackSessionNamespace(session, code);
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

const server = createSdkMcpServer({
  name: "beagle-repl",
  tools: [
    beagleRun, beagleLoad, beagleEval, beagleDescribe, beagleSessions, beagleInterrupt,
    beagleStatus, beagleListNamespaces, beagleNamespaceInfo, beagleSearch, beagleDoc,
  ],
});

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `\
You are a Beagle live coding agent. You interact with a running Beagle program \
exclusively through a REPL socket connection. You do NOT have access to the file \
system — no reading files, no writing files, no editing files. Everything you do \
goes through the REPL.

## Your tools

### Lifecycle
- **beagle_run**: Start (or restart) a Beagle program with an embedded REPL server. \
  If given a .bg file, runs that file's main() with the REPL server embedded inside it — \
  so you can eval code in the context of the running program. If no file is given, \
  starts a standalone REPL server. **You must call this before using any other tool.**
- **beagle_load**: Load a .bg file into the running REPL. Reads the file from disk \
  and evaluates its entire contents in the REPL session. All definitions from the \
  file become live immediately. Use this when the user asks you to load additional files.

### Core
- **beagle_eval**: Evaluate Beagle code in a session. This is your primary tool. \
  Use it to define functions, test expressions, and run code. \
  The code is evaluated in the live running Beagle program.
- **beagle_interrupt**: Stop a long-running evaluation.
- **beagle_status**: Check if the Beagle process is running. If it crashed, shows \
  the exit code, signal, stdout, stderr, and timing. Use this to diagnose crashes.
- **beagle_describe**: Discover what operations the REPL server supports.
- **beagle_sessions**: List active REPL sessions.

### Introspection
- **beagle_list_namespaces**: List all loaded namespaces. Use this first to orient yourself.
- **beagle_namespace_info**: Get detailed info about a namespace — all its functions \
  (with signatures and docs), structs (with fields), and enums (with variants). \
  This is the best way to understand what's available.
- **beagle_search**: Search for functions by name or docstring substring (like apropos). \
  Returns fully-qualified names.
- **beagle_doc**: Get full documentation for a specific function or value — its args, \
  doc string, type kind, fields/variants.

## How you work

1. **Explore first**: When asked to work on something, use beagle_list_namespaces \
   and beagle_namespace_info to understand the running program. Use beagle_search \
   to find relevant functions. Use beagle_doc to read documentation.
2. **Develop incrementally**: Define or redefine functions one at a time, testing \
   each with beagle_eval before moving on.
3. **Use scratch sessions**: Use session "scratch" for throwaway experiments. \
   Use session "dev" for definitions that should persist.
4. **Think in terms of a live program**: You're not editing dead text in files. \
   You're modifying a running system. Every eval changes the live program.

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

When you define or redefine a function via eval, it updates in the running program \
immediately — no restart needed. The source file on disk is also updated automatically \
— you do not need to write changes to files yourself.

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
            "mcp__beagle-repl__beagle_status",
            "mcp__beagle-repl__beagle_list_namespaces",
            "mcp__beagle-repl__beagle_namespace_info",
            "mcp__beagle-repl__beagle_search",
            "mcp__beagle-repl__beagle_doc",
          ],
          model: "claude-sonnet-4-6",
        };

    log("INFO", "Sending query to Claude", { promptLength: userInput.length, hasSession: !!sessionId });

    try {
      for await (const message of query({ prompt: userInput, options })) {
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
          if (result) console.log(result);
          log("INFO", "Query completed with result", { resultLength: result?.length });
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
      log("ERROR", "Query failed", { message: err.message, stack: err.stack, name: err.name });
      console.error(`\n[ERROR] Agent query failed: ${err.message}`);
      console.error(`See ${LOG_FILE} for details.`);
      // Don't exit — let the user try again
    }
  }

  rl.close();
}

main();
