import * as readline from "readline";
import * as child_process from "child_process";
import * as path from "path";
import * as fs from "fs";
import OpenAI from "openai";
import { z } from "zod";
import {
  REPL_HOST, REPL_PORT,
  configureRepl, connectRepl, disconnectRepl, replSend, replRequest,
  formatReplResponse, isErrorResponse, extractErrorText, waitForPort, probeReplConnection,
  extractNamespaceFromText,
  reflectAllNamespaces, reflectNamespaceInfo, reflectSource, reflectNamespaceSource,
  reflectLocation, reflectApropos, reflectInfo, reflectPersist,
  formatNamespaceInfo,
} from "./beagle_repl_core";

// ---------------------------------------------------------------------------
// DeepSeek (OpenAI-compatible) configuration
// ---------------------------------------------------------------------------

const DEEPSEEK_BASE_URL = process.env.DEEPSEEK_BASE_URL ?? "https://api.deepseek.com";
const MODEL_ALIASES: Record<string, string> = {
  pro: "deepseek-v4-pro",
  flash: "deepseek-v4-flash",
};

function resolveModel(name: string): string {
  return MODEL_ALIASES[name.toLowerCase()] ?? name;
}

// ---------------------------------------------------------------------------
// Tool definition helper (replaces the Claude Agent SDK's `tool`)
// ---------------------------------------------------------------------------

type ToolResult = { content: { type: "text"; text: string }[] };

interface AgentTool {
  name: string;
  description: string;
  schema: Record<string, unknown>;
  handler: (args: any) => Promise<ToolResult>;
}

function tool<S extends z.ZodRawShape>(
  name: string,
  description: string,
  shape: S,
  handler: (args: z.infer<z.ZodObject<S>>) => Promise<ToolResult>
): AgentTool {
  const schema = z.toJSONSchema(z.object(shape)) as Record<string, unknown>;
  delete schema.$schema;
  return { name, description, schema, handler };
}

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
    }
  });
  return proc;
}

// ---------------------------------------------------------------------------
// Session → namespace tracking
// ---------------------------------------------------------------------------
// Remembers the last-seen namespace per session so beagle_persist can infer
// the target namespace when the caller omits it.

const sessionNamespaces = new Map<string, string>();

function trackSessionNamespace(session: string, ns: string | undefined) {
  if (ns) sessionNamespaces.set(session, ns);
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

// Tailor the shared REPL core to the agent: keep the default agent-tool wording
// for suspension hints, and append this server's last crash info to any
// connection-failed message (the core has no notion of our beag child).
configureRepl({
  hints: {
    mainSuspended:
      `\n\nThe main thread (game loop / GUI) threw and is PAUSED at the throw site — the process is alive.` +
      `\nThis is recoverable. Before doing anything, understand the cause: read the error's kind, message,` +
      ` and location above. A "TypeError ... mix integers and floats" whose real cause is a null operand` +
      ` usually means you read a field that does not exist on a value that is still live in the program` +
      ` (a STALE struct instance — see below).` +
      `\n\nThe live data, not your code, is often what's wrong. Changing a struct's shape does NOT migrate` +
      ` instances already alive in the running program: their new fields read as null (or their declared` +
      ` default). So a freshly-constructed value will look fine while the one inside the live world crashes.` +
      `\n\nTo recover, pick the right tool for the cause:` +
      `\n  • Fix the DATA: if state lives in an atom (e.g. a world atom), swap! a migrated value into it,` +
      ` then beagle_main_resume.` +
      `\n  • Resume PAST one bad call: beagle_main_resume("<expr>") evaluates <expr> and uses it as the` +
      ` return value of the throwing expression, so the paused frame continues. Supply the value the failed` +
      ` op should have produced (e.g. a float). Note this only steps past THIS one throw; if the underlying` +
      ` data is still wrong the next frame re-throws — fix the data instead.` +
      `\n  • If the FUNCTION is genuinely buggy: beagle_persist the corrected fn, then beagle_main_resume.` +
      ` But remember the paused frame runs the OLD code — your new fn only takes effect on the NEXT call,` +
      ` so for an in-frame crash you usually also need to resume with a value or fix the data.` +
      `\n  • Give up on this frame: beagle_main_abort.` +
      `\n\nFIRST inspect the ACTUAL live operands (e.g. eval the real world.player.stamina), never a value you` +
      ` just constructed. If you can't see the cause, say so and ask — do not thrash by editing functions blindly.`,
    resumable:
      `\nThe evaluation is suspended at the throw site — the program waits for a value.` +
      `\nbeagle_resume("<expr>") evaluates <expr> and makes it the return value of the throwing expression,` +
      ` so execution continues past the failure. Supply what the failed op should have produced.` +
      `\nWhile suspended you can beagle_eval to inspect the exact state at the failure point first.` +
      `\nbeagle_abort abandons the suspended evaluation.`,
  },
  onConnectError: () => (lastCrash ? formatCrashInfo(lastCrash) : undefined),
});

// ---------------------------------------------------------------------------
// MCP Tools
// ---------------------------------------------------------------------------

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
// Introspection tools — thin wrappers around the reflect* ops in
// beagle_repl_core (which own the beagle.reflect code strings).
// ---------------------------------------------------------------------------

// Wrap a reflect* result (or a caught error) as a tool response.
function introspectResult(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

const beagleListNamespaces = tool(
  "beagle_list_namespaces",
  "List all namespaces loaded in the running Beagle program.",
  {},
  async () => {
    try {
      const namespaces = await reflectAllNamespaces();
      namespaces.sort();
      return introspectResult(namespaces.join("\n"));
    } catch (err: any) {
      return introspectResult(`Error listing namespaces: ${err.message}`);
    }
  }
);

const beagleNamespaceInfo = tool(
  "beagle_namespace_info",
  "Get detailed info about a namespace: its functions (with args and docs), structs, and enums. " +
  "Use this to understand what's available before writing code.",
  { namespace: z.string().describe("Namespace name, e.g. \"beagle.core\"") },
  async (args) => {
    try {
      const info = await reflectNamespaceInfo(args.namespace);
      return introspectResult(formatNamespaceInfo(args.namespace, info));
    } catch (err: any) {
      return introspectResult(`Error inspecting namespace "${args.namespace}": ${err.message}`);
    }
  }
);

const beagleSearch = tool(
  "beagle_search",
  "Search for functions by name or docstring substring. Returns matching fully-qualified function names.",
  { query: z.string().describe("Search term to match against function names and docstrings") },
  async (args) => {
    try {
      const matches = await reflectApropos(args.query);
      matches.sort();
      return introspectResult(
        matches.length > 0 ? matches.join("\n") : `No matches for "${args.query}"`,
      );
    } catch (err: any) {
      return introspectResult(`Error searching for "${args.query}": ${err.message}`);
    }
  }
);

const beagleDoc = tool(
  "beagle_doc",
  "Get the documentation, arguments, and type info for a specific function or value. " +
  "Pass the fully-qualified name or a direct reference.",
  { name: z.string().describe("Fully-qualified function name, e.g. \"beagle.core/map\" or just \"map\" if in scope") },
  async (args) => {
    try {
      // `name` is a Beagle reference (resolved in the introspect session), not
      // a string literal — reflect/info takes the value directly.
      const info = await reflectInfo(args.name);
      const lines: string[] = [`Name: ${info.name ?? args.name}`];
      if (info.kind) lines.push(`Kind: ${info.kind}`);
      if (info.doc) lines.push(`Doc: ${info.doc}`);
      if (info.args) {
        lines.push(`Args: (${info.args.join(", ")}${info["variadic?"] ? "..." : ""})`);
      }
      if (info.fields) lines.push(`Fields: ${info.fields.join(", ")}`);
      if (info.variants) lines.push(`Variants: ${info.variants.join(" | ")}`);
      return introspectResult(lines.join("\n"));
    } catch (err: any) {
      return introspectResult(`Error getting doc for "${args.name}": ${err.message}`);
    }
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
    try {
      // Pass the name as a STRING literal: reflect/source resolves it via
      // resolve_by_name (which handles "ns/name" forms), so it works without
      // the target's namespace being imported as an alias in the introspect
      // session — more robust than evaluating it as a bare reference.
      const source = await reflectSource(args.name);
      return introspectResult(
        source ?? `No stored source for "${args.name}" (REPL/eval def, builtin, or FFI).`,
      );
    } catch (err: any) {
      return introspectResult(`Error reading source for "${args.name}": ${err.message}`);
    }
  }
);

const beagleNamespaceSource = tool(
  "beagle_namespace_source",
  "Get the concatenated source of every definition in a namespace (structs, then enums, " +
  "then functions). Definitions without stored source (builtins, FFI, anonymous closures) are " +
  "skipped. Use this when you want to understand or refactor a whole file.",
  { namespace: z.string().describe("Namespace name, e.g. 'my.module'") },
  async (args) => {
    try {
      const source = await reflectNamespaceSource(args.namespace);
      return introspectResult(
        source ?? `No stored source for namespace "${args.namespace}".`,
      );
    } catch (err: any) {
      return introspectResult(`Error reading source for namespace "${args.namespace}": ${err.message}`);
    }
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
    try {
      // String literal → resolved via resolve_by_name (handles "ns/name"),
      // so no alias for the target namespace is required. See beagle_source.
      const loc = await reflectLocation(args.name);
      return introspectResult(
        loc ? JSON.stringify(loc, null, 2) : `No location for "${args.name}" (REPL/builtin/FFI).`,
      );
    } catch (err: any) {
      return introspectResult(`Error getting location for "${args.name}": ${err.message}`);
    }
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

    const result = await reflectPersist(ns, args.text);

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

// ---------------------------------------------------------------------------
// Directory listing tool
// ---------------------------------------------------------------------------

const listDirectory = tool(
  "list_directory",
  "List the contents of a directory on disk. Directories are shown with a trailing '/'. " +
  "Use this to find .bg files to pass to beagle_run or beagle_load. " +
  "This is the only file system access you have — there is no file-read tool; " +
  "source is read through beagle_source / beagle_namespace_source.",
  {
    path: z.string().optional().describe("Directory to list (absolute or relative; default: current working directory)"),
  },
  async (args) => {
    const dir = path.resolve(args.path ?? process.cwd());
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch (e: any) {
      return { content: [{ type: "text" as const, text: `Error listing ${dir}: ${e.message}` }] };
    }
    const lines = entries
      .sort((a, b) => {
        // Directories first, then alphabetical
        if (a.isDirectory() !== b.isDirectory()) return a.isDirectory() ? -1 : 1;
        return a.name.localeCompare(b.name);
      })
      .map((e) => (e.isDirectory() ? `${e.name}/` : e.name));
    const text = `${dir} (${lines.length} entries)\n${lines.join("\n")}`;
    return { content: [{ type: "text" as const, text }] };
  }
);

// ---------------------------------------------------------------------------
// Tool registry
// ---------------------------------------------------------------------------

const ALL_TOOLS: AgentTool[] = [
  beagleRun, beagleLoad, beagleEval, beagleDescribe, beagleSessions, beagleInterrupt,
  beagleResume, beagleAbort, beagleStatus,
  beagleMainStatus, beagleMainResume, beagleMainAbort,
  beagleListNamespaces, beagleNamespaceInfo, beagleSearch, beagleDoc,
  beagleSource, beagleNamespaceSource, beagleLocation, beaglePersist,
  listDirectory,
];

const TOOLS_BY_NAME = new Map(ALL_TOOLS.map((t) => [t.name, t]));

const OPENAI_TOOL_SCHEMAS: OpenAI.Chat.Completions.ChatCompletionTool[] = ALL_TOOLS.map((t) => ({
  type: "function" as const,
  function: {
    name: t.name,
    description: t.description,
    parameters: t.schema,
  },
}));

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `\
You are a Beagle live coding agent. You interact with a running Beagle program \
through a REPL socket connection. You have NO file editing tools — no Read, Glob, \
Grep, Write, Edit. Everything — reading source, writing source, exploring the \
running program, testing — goes through the tools below. Source reading happens \
via beagle_source / beagle_namespace_source, source writing via beagle_persist. \
The only file system tool you have is list_directory, for locating .bg files.

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

### File system
- **list_directory(path?)**: List a directory's contents (directories shown with a \
  trailing '/'). Use it to locate .bg files for beagle_run / beagle_load. There is \
  no file-read tool — read source via beagle_source / beagle_namespace_source.

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

## Update vs. append

beagle_persist reports \`:action "updated"\` when the def already existed on disk (it \
was spliced in place) and \`:action "appended"\` when it was brand-new to the file (added \
at the end). **"appended" is NORMAL and not an error** — it just means that name had no \
prior on-disk definition. A def you created earlier with **beagle_eval** lives only in \
memory and has no disk origin, so its first beagle_persist will "append", not "update". \
If you're unsure whether a def is on disk, beagle_location returns null for in-memory / \
builtin / FFI defs.

## Drift errors

If beagle_persist fails with \
\`reflect/persist: file ... has changed since <name> was loaded (re-load and retry)\`, the \
file on disk no longer matches what the runtime recorded when it loaded that def. The \
runtime tracks each def's exact byte range; a drift means the bytes there changed out \
from under it. Common causes:

- **You edited the source file directly** (e.g. via beagle.fs/blocking-write-file or \
  blocking-append-file). DON'T do that for source you also persist — write source ONLY \
  through beagle_persist so disk and the runtime's byte tracking stay in sync.
- Another editor or a git checkout changed the file.

To recover: do NOT retry the same text blindly.

1. Call **beagle_source** / **beagle_namespace_source** to see the current state.
2. Re-derive your edit against the fresh source — whatever else changed may affect it.
3. beagle_persist again with the updated text.

Note: **beagle_load re-evaluates a file's text but does NOT refresh the runtime's on-disk \
byte tracking** (it has no file context), so it will not clear a drift. A fresh \
**beagle_run** reloads the file with file context and re-syncs tracking — use it if drift \
persists after you've reconciled the source. Tell the user what drifted if it's surprising.

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
- **resume supplies a VALUE, not a fix.** \`beagle_resume("<expr>")\` evaluates \
  \`<expr>\` and substitutes its result *as the return value of the throwing \
  expression*, then continues. So you supply what the failed operation should have \
  produced (e.g. a float for a failed multiply). It does NOT re-run the expression \
  with corrected code.

## The live-coding mental model (read before recovering from ANY crash)

You are editing a **running program that holds live data**. The single most common \
mistake is treating a crash as "a function is broken" when really **the live data is \
the wrong shape**. Internalize this:

- **Changing a struct's fields does NOT migrate instances already alive in the \
  program.** If you add a field to \`Player\`, every \`Player\` already sitting in the \
  running world keeps its old shape. Reading the new field off a stale instance yields \
  **null** (or the field's declared default if it has one) — it does not error at the \
  read.
- That null then blows up *somewhere else*, often as a misleading \
  \`TypeError: ... mix integers and floats\` (a null operand in arithmetic), pointing \
  far from the real cause. **A null operand in arithmetic almost always means a \
  stale-instance field read.**
- A value you construct fresh in beagle_eval will have the new shape and look fine. \
  **That proves nothing.** Always inspect the *actual live value* that crashed (e.g. \
  \`world.player.stamina\`), never a stand-in you just built.

## Main-thread crash recovery

When running a GUI program (e.g. raylib game) with beagle_run, the main thread runs \
the game loop while the REPL server runs on a background thread. If a function the \
loop calls throws, the **main thread suspends at the throw site** instead of killing \
the process, and waits for REPL recovery.

### Workflow — diagnose the CAUSE, then pick the matching fix

1. **Read the error** (beagle_main_status): kind, message, location. Form a hypothesis \
   about the *cause* before touching anything.
2. **Inspect the real live operands** with beagle_eval — the actual values that \
   crashed, not freshly-built ones. If a field reads null on a live instance, this is a \
   **stale-data** problem, not a code problem.
3. **Fix the right thing:**
   - **Stale data with state in an atom** (e.g. a world atom): \`swap!\` a migrated \
     value into the atom, then **beagle_main_resume**. This is the clean fix for a \
     struct-shape change.
   - **Step past one bad call:** **beagle_main_resume("<value>")** supplies the result \
     the failed op should have produced and continues the paused frame. Good for a \
     one-off; but if the underlying data is still wrong the next frame re-throws — \
     prefer fixing the data.
   - **Genuinely buggy function:** beagle_persist the corrected def, then \
     beagle_main_resume. Caveat: **the paused frame is still running the OLD code** — \
     your new def only applies to the *next* call. For an in-frame crash you typically \
     also resume-with-a-value (or fix the data) so the current frame can complete.
   - **Give up on this frame:** **beagle_main_abort** (then beagle_run to restart).
4. **If you can't see the cause, STOP and say so.** Do not edit functions blindly \
   hoping one sticks — that thrashing is the exact failure mode to avoid. Report what \
   you observe and ask.

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
// DeepSeek agent loop
// ---------------------------------------------------------------------------

type ChatMessage = OpenAI.Chat.Completions.ChatCompletionMessageParam;

function pickStartupModel(): string {
  const argv = process.argv.slice(2);
  const idx = argv.indexOf("--model");
  if (idx !== -1 && argv[idx + 1]) return resolveModel(argv[idx + 1]);
  if (process.env.DEEPSEEK_MODEL) return resolveModel(process.env.DEEPSEEK_MODEL);
  return MODEL_ALIASES.pro;
}

async function executeToolCall(name: string, rawArgs: string): Promise<string> {
  const toolDef = TOOLS_BY_NAME.get(name);
  if (!toolDef) return `Error: unknown tool "${name}"`;

  let args: Record<string, unknown> = {};
  if (rawArgs && rawArgs.trim()) {
    try {
      args = JSON.parse(rawArgs);
    } catch (e: any) {
      return `Error: tool arguments were not valid JSON: ${e.message}`;
    }
  }

  const summary = args.code
    ? `${name}(${String(args.code).slice(0, 80)}${String(args.code).length > 80 ? "..." : ""})`
    : `${name}(${JSON.stringify(args).slice(0, 80)})`;
  console.log(`\x1b[2m⚙ ${summary}\x1b[0m`);
  log("INFO", "Tool call", { tool: name, input: args });

  try {
    const result = await toolDef.handler(args);
    return result.content.map((c) => c.text).join("\n");
  } catch (e: any) {
    log("ERROR", "Tool handler threw", { tool: name, message: e.message, stack: e.stack });
    return `Error: tool ${name} failed: ${e.message}`;
  }
}

// Run one user turn: call the model, execute tool calls, repeat until the model
// answers with plain text (or the user aborts via Escape).
async function runAgentTurn(
  client: OpenAI,
  model: string,
  messages: ChatMessage[],
  signal: AbortSignal,
  isAborted: () => boolean
): Promise<void> {
  const MAX_TOOL_ROUNDS = 100;

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const response = await client.chat.completions.create(
      { model, messages, tools: OPENAI_TOOL_SCHEMAS },
      { signal }
    );

    const msg = response.choices[0]?.message;
    if (!msg) throw new Error("DeepSeek returned no choices");
    log("INFO", "Received completion", {
      finishReason: response.choices[0]?.finish_reason,
      toolCalls: msg.tool_calls?.length ?? 0,
      usage: response.usage,
    });

    // Re-append a sanitized copy of the assistant message. Never echo
    // reasoning_content (or other extra fields) back to the API.
    const toolCalls = (msg.tool_calls ?? []).filter((tc: any) => tc.type === "function");
    messages.push({
      role: "assistant",
      content: msg.content ?? "",
      ...(toolCalls.length ? { tool_calls: toolCalls as any } : {}),
    });

    if (msg.content) console.log(msg.content);
    if (!toolCalls.length) return;

    for (const tc of toolCalls as any[]) {
      // Every tool_call needs a matching tool message or the next API call 400s,
      // so on abort we answer the remaining calls with a placeholder.
      const text = isAborted()
        ? "[tool call skipped — query aborted by user]"
        : await executeToolCall(tc.function.name, tc.function.arguments);
      messages.push({ role: "tool", tool_call_id: tc.id, content: text });
    }
    if (isAborted()) return;
  }

  console.log(`\x1b[33m[stopped after ${MAX_TOOL_ROUNDS} tool rounds — ask me to continue]\x1b[0m`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  // Clean up beag process on exit
  process.on("exit", killBeagleServer);
  process.on("SIGINT", () => { killBeagleServer(); process.exit(0); });
  process.on("SIGTERM", () => { killBeagleServer(); process.exit(0); });

  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    console.error("Error: DEEPSEEK_API_KEY is not set.");
    console.error("Export your DeepSeek API key and re-run: export DEEPSEEK_API_KEY=sk-...");
    process.exit(1);
  }

  const client = new OpenAI({ baseURL: DEEPSEEK_BASE_URL, apiKey });
  let model = pickStartupModel();

  console.log("Beagle Live Code Agent (DeepSeek)");
  console.log("=================================");
  console.log(`Model: ${model} — switch with /model pro|flash`);
  console.log(`Logging to: ${LOG_FILE}`);
  console.log("Ask me to run a .bg file or start the default REPL server.\n");

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  // Make stdin emit 'keypress' events so we can listen for Escape during a
  // running query without disturbing rl.question (which only consumes whole lines).
  readline.emitKeypressEvents(process.stdin);

  const prompt = (): Promise<string> =>
    new Promise((resolve) => rl.question("> ", resolve));

  // The full conversation, persisted across turns (replaces SDK session resume).
  const messages: ChatMessage[] = [{ role: "system", content: SYSTEM_PROMPT }];

  while (true) {
    let userInput: string;
    try {
      userInput = await prompt();
    } catch {
      console.log("\nBye!");
      break;
    }

    if (!userInput.trim()) continue;

    // /model — show or switch the active model without losing conversation state.
    const modelCmd = userInput.trim().match(/^\/model(?:\s+(\S+))?$/);
    if (modelCmd) {
      if (modelCmd[1]) {
        model = resolveModel(modelCmd[1]);
        console.log(`Model set to ${model}`);
      } else {
        console.log(`Current model: ${model} (use /model pro|flash|<full-model-name>)`);
      }
      continue;
    }

    messages.push({ role: "user", content: userInput });
    log("INFO", "Sending query to DeepSeek", { promptLength: userInput.length, model });

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
      await runAgentTurn(client, model, messages, abortController.signal, () => aborted);
      log("INFO", "Turn completed");
    } catch (err: any) {
      if (aborted || err instanceof OpenAI.APIUserAbortError) {
        // Aborting the in-flight request throws — expected, not a real failure.
        // If the request died before an assistant message landed, the message
        // list is still valid (it just ends with user/tool messages).
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
