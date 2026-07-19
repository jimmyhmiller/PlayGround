#!/usr/bin/env -S npx tsx
// ---------------------------------------------------------------------------
// beagle — a self-contained CLI for driving a live Beagle REPL server.
//
// This is the *toolkit* form of the Beagle live-code agent: it shares the
// nREPL transport and beagle.reflect introspection with the DeepSeek agent via
// beagle_repl_core.ts, but exposes them as plain subcommands so ANY agent (or
// human) with shell access can drive a running Beagle program.
//
// Lifecycle differs from the agent: the agent owns its server for the life of
// its process and kills it on exit. The CLI is one process per invocation, so
// it auto-starts the server DETACHED (surviving across invocations) and finds
// an already-running server by probing the port. State (pid/file) lives in a
// small JSON file under /tmp so `down`/`status` can find the daemon later.
//
//   beagle up [file]          start/ensure the server (embed REPL in file's main)
//   beagle down               stop the detached server
//   beagle status             process + REPL reachability
//   beagle eval <code|->      evaluate (auto-starts default server if none)
//   beagle load <file>        read a .bg file and eval its contents
//   beagle namespaces         list loaded namespaces
//   beagle ns-info <ns>       functions/structs/enums of a namespace
//   beagle source <name>      stored source of a def
//   beagle ns-source <ns>     concatenated source of a whole namespace
//   beagle location <name>    {file,byte/line start/end} of a def
//   beagle search <query>     apropos over names + docstrings
//   beagle doc <name>         doc/args/kind of a def
//   beagle persist <text|->   write def(s) to disk AND the running program
//   beagle describe | sessions | interrupt <s>
//   beagle resume <code> | abort        resumable-exception recovery
//   beagle main-status | main-resume [code] | main-abort
//   beagle ls [dir]           list a directory (find .bg files)
//
// Global flags: --session <name> (default "agent"), --json (raw output),
//               --file <path> (for `up`).
// ---------------------------------------------------------------------------

import * as child_process from "child_process";
import * as path from "path";
import * as fs from "fs";
import {
  REPL_HOST, REPL_PORT,
  configureRepl, connectRepl, disconnectRepl, replSend, replRequest,
  formatReplResponse, isErrorResponse, waitForPort, isPortOpen,
  introspectEval,
  reflectAllNamespaces, reflectNamespaceInfo, reflectSource, reflectNamespaceSource,
  reflectLocation, reflectApropos, reflectInfo, reflectPersist,
  formatNamespaceInfo, extractNamespaceFromText,
  type ReflectInfo,
} from "./beagle_repl_core";

const DEFAULT_REPL_SERVER = path.join(
  process.env.HOME ?? "",
  "Documents/Code/beagle/resources/examples/repl_server.bg",
);
const STATE_FILE = "/tmp/beagle-repl-cli.json";
const SERVER_LOG = "/tmp/beagle-repl-cli.log";
const WRAPPER_PATH = "/tmp/__beagle_repl_runner.bg";

// The CLI's user-facing hints name `beagle <cmd>`, not the agent's MCP tools.
configureRepl({
  hints: {
    mainSuspended: `\nUse \`beagle eval\` to fix the broken function, then \`beagle main-resume\` to continue.`,
    resumable: `\nUse \`beagle resume <code>\` to supply a replacement value, or \`beagle abort\`.`,
  },
});

// ---------------------------------------------------------------------------
// Detached server lifecycle (CLI-specific — the daemon must outlive this process)
// ---------------------------------------------------------------------------

interface ServerState { pid: number; bgFile: string; startedAt: string; }

function readState(): ServerState | undefined {
  try {
    return JSON.parse(fs.readFileSync(STATE_FILE, "utf-8")) as ServerState;
  } catch {
    return undefined;
  }
}

function writeState(state: ServerState) {
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function clearState() {
  try { fs.unlinkSync(STATE_FILE); } catch { /* already gone */ }
}

function pidAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch { return false; }
}

// Build the run-with-REPL wrapper that embeds the REPL server inside a user
// program's main(), so evals run in that program's context.
function buildWrapper(bgFile: string): { wrapperPath: string; sourceDir: string } {
  const sourceDir = path.dirname(path.resolve(bgFile));
  const fileContent = fs.readFileSync(bgFile, "utf-8");
  const targetNs = extractNamespaceFromText(fileContent);
  if (!targetNs) {
    throw new Error(`No \`namespace <name>\` declaration found in ${bgFile}.`);
  }
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
  fs.writeFileSync(WRAPPER_PATH, wrapperCode);
  return { wrapperPath: WRAPPER_PATH, sourceDir };
}

// Spawn beag detached so it outlives this CLI process. Returns the pid.
function spawnDetached(args: string[], bgFile: string): number {
  const logFd = fs.openSync(SERVER_LOG, "a");
  const proc = child_process.spawn("beag", args, {
    detached: true,
    stdio: ["ignore", logFd, logFd],
  });
  proc.unref();
  fs.closeSync(logFd);
  writeState({ pid: proc.pid!, bgFile, startedAt: new Date().toISOString() });
  return proc.pid!;
}

// Ensure SOME server is reachable. If the port is already open we connect to
// it (started by `up`, the agent, or manually). Otherwise start the default
// standalone server detached — UNLESS we were tracking a specific program
// (a game/app started via `up`) that has since died. In that case, silently
// swapping in a bare REPL server would mask the crash and leave you talking to
// the wrong process (where your namespaces/state don't exist), so we surface it.
async function ensureServer(): Promise<void> {
  if (await isPortOpen(REPL_HOST, REPL_PORT)) {
    await connectRepl();
    return;
  }
  const state = readState();
  if (state && state.bgFile && state.bgFile !== DEFAULT_REPL_SERVER) {
    const alive = pidAlive(state.pid);
    const out = readLog();
    // Leave the tracked state in place: every subsequent command keeps
    // surfacing the dead program until you explicitly `up` (restart) or
    // `down` (clear, then a bare server auto-starts).
    throw new Error(
      `The program started with \`up\` — ${state.bgFile} (pid ${state.pid}) — is ` +
        (alive
          ? `running but not listening on ${REPL_HOST}:${REPL_PORT}.`
          : `no longer running; it exited or crashed.`) +
        `\nRefusing to silently start a bare REPL server in its place (that would ` +
        `mask the crash and put you in a process where your code/state don't exist).` +
        `\n\n--- last server log ---\n${out || "(empty)"}` +
        `\n\nRestart it:  beagle up ${state.bgFile}` +
        `\nOr start a bare REPL:  beagle down && beagle eval '1'`,
    );
  }
  await startServer(undefined);
}

// (Re)start a server. file=undefined → default standalone server.
async function startServer(file: string | undefined): Promise<string> {
  if (await isPortOpen(REPL_HOST, REPL_PORT)) {
    await stopServer();
  }
  let args: string[];
  let target: string;
  if (file) {
    const { wrapperPath, sourceDir } = buildWrapper(file);
    args = ["run", "-I", sourceDir, wrapperPath];
    target = file;
  } else {
    args = ["run", DEFAULT_REPL_SERVER];
    target = DEFAULT_REPL_SERVER;
  }
  fs.writeFileSync(SERVER_LOG, ""); // truncate so status shows this run's output
  const pid = spawnDetached(args, target);
  try {
    await waitForPort(REPL_HOST, REPL_PORT);
  } catch {
    const out = readLog();
    await stopServer();
    throw new Error(`Failed to start server (pid ${pid}).\n\n--- server log ---\n${out}`);
  }
  await connectRepl();
  return target;
}

async function stopServer(): Promise<boolean> {
  const state = readState();
  let killed = false;
  disconnectRepl();
  if (state && pidAlive(state.pid)) {
    // detached → process is a group leader; kill the whole group.
    try { process.kill(-state.pid, "SIGTERM"); killed = true; }
    catch { try { process.kill(state.pid, "SIGTERM"); killed = true; } catch { /* gone */ } }
  }
  clearState();
  return killed;
}

function readLog(): string {
  try { return fs.readFileSync(SERVER_LOG, "utf-8").trimEnd(); } catch { return "(no log)"; }
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

interface ParsedArgs { positionals: string[]; flags: Record<string, string | boolean>; }

function parseArgs(argv: string[]): ParsedArgs {
  const positionals: string[] = [];
  const flags: Record<string, string | boolean> = {};
  const booleanFlags = new Set(["json"]);
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      if (eq >= 0) {
        flags[a.slice(2, eq)] = a.slice(eq + 1);
      } else {
        const key = a.slice(2);
        if (booleanFlags.has(key)) {
          flags[key] = true;
        } else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
          flags[key] = argv[++i];
        } else {
          flags[key] = true;
        }
      }
    } else {
      positionals.push(a);
    }
  }
  return { positionals, flags };
}

// Read a positional that may be "-" meaning "read from stdin".
function readArgOrStdin(value: string | undefined): string {
  if (value === undefined) throw new UsageError("missing argument");
  if (value === "-") return fs.readFileSync(0, "utf-8");
  return value;
}

class UsageError extends Error {}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

let JSON_OUT = false;

function emit(text: string, jsonValue?: unknown) {
  if (JSON_OUT && jsonValue !== undefined) {
    process.stdout.write(JSON.stringify(jsonValue, null, 2) + "\n");
  } else {
    process.stdout.write(text + "\n");
  }
}

function fail(text: string): never {
  process.stderr.write(text + "\n");
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

const USAGE = `beagle — drive a live Beagle REPL

Lifecycle:
  up [file] [--file f]   start/ensure server (embeds REPL in file's main if given)
  down                   stop the detached server
  restart [file]         stop + start
  status                 process + REPL reachability

Code:
  eval <code|->          evaluate Beagle code (auto-starts default server)
  load <file>            read a .bg file and eval its contents
  persist <text|->       write def(s) to disk AND the running program

Introspection:
  namespaces             list loaded namespaces
  ns-info <ns>           functions/structs/enums of a namespace
  source <name>          stored source of a def
  ns-source <ns>         concatenated source of a whole namespace
  location <name>        {file,byte/line start/end} of a def
  search <query>         apropos over names + docstrings
  doc <name>             doc/args/kind of a def

REPL control:
  describe | sessions | interrupt <session>
  resume <code> | abort               resumable-exception recovery
  main-status | main-resume [code] | main-abort
  ls [dir]               list a directory (find .bg files)

Global flags: --session <name> (default "agent"), --json, --file <path>`;

async function dispatch(cmd: string, p: string[], flags: Record<string, string | boolean>): Promise<void> {
  const session = (flags.session as string) ?? "agent";

  switch (cmd) {
    case "up": {
      const file = (flags.file as string) ?? p[0];
      const target = await startServer(file);
      emit(`Server started: ${target}\n\n${readLog()}`, { started: target, log: readLog() });
      return;
    }
    case "down": {
      const killed = await stopServer();
      emit(killed ? "Server stopped." : "No running server found.", { stopped: killed });
      return;
    }
    case "restart": {
      const file = (flags.file as string) ?? p[0];
      await stopServer();
      const target = await startServer(file);
      emit(`Server restarted: ${target}`, { restarted: target });
      return;
    }
    case "status": {
      const state = readState();
      const portOpen = await isPortOpen(REPL_HOST, REPL_PORT);
      const lines: string[] = [];
      if (state && pidAlive(state.pid)) {
        lines.push(`Process running (pid ${state.pid}, file ${state.bgFile}, since ${state.startedAt})`);
      } else if (portOpen) {
        lines.push("REPL port is open, but not started by this CLI (agent or manual server).");
      } else {
        lines.push("No server running.");
      }
      if (portOpen) {
        await ensureServer();
        const probe = await replRequest("describe");
        lines.push(`REPL reachable: ${!isErrorResponse(probe)}`);
      } else {
        lines.push("REPL reachable: false");
      }
      emit(lines.join("\n"), { state, portOpen });
      return;
    }
    case "eval": {
      await ensureServer();
      const code = readArgOrStdin(p[0]);
      const messages = await replSend("eval", { code, session });
      emit(formatReplResponse(messages), messages);
      return;
    }
    case "load": {
      await ensureServer();
      const file = p[0];
      if (!file) throw new UsageError("load needs a file path");
      const code = fs.readFileSync(file, "utf-8");
      const messages = await replSend("eval", { code, session });
      emit(`Loaded ${file}\n\n${formatReplResponse(messages)}`, messages);
      return;
    }
    case "persist": {
      await ensureServer();
      const text = readArgOrStdin(p[0]);
      const nsArg = (flags.namespace as string) ?? p[1];
      const ns = nsArg ?? extractNamespaceFromText(text);
      if (!ns) throw new UsageError("persist needs a target namespace (--namespace) or a `namespace X` directive in text");
      const result = await reflectPersist(ns, text);
      emit(result, { namespace: ns, result });
      return;
    }
    case "namespaces": {
      await ensureServer();
      const namespaces = (await reflectAllNamespaces()).sort();
      emit(namespaces.join("\n"), namespaces);
      return;
    }
    case "ns-info": {
      await ensureServer();
      const ns = p[0];
      if (!ns) throw new UsageError("ns-info needs a namespace");
      const info = await reflectNamespaceInfo(ns);
      emit(formatNamespaceInfo(ns, info), info);
      return;
    }
    case "source": {
      await ensureServer();
      const name = p[0];
      if (!name) throw new UsageError("source needs a name");
      const src = await reflectSource(name);
      emit(src ?? `No stored source for "${name}" (REPL/eval def, builtin, or FFI).`, { source: src });
      return;
    }
    case "ns-source": {
      await ensureServer();
      const ns = p[0];
      if (!ns) throw new UsageError("ns-source needs a namespace");
      const src = await reflectNamespaceSource(ns);
      emit(src ?? `No stored source for namespace "${ns}".`, { source: src });
      return;
    }
    case "location": {
      await ensureServer();
      const name = p[0];
      if (!name) throw new UsageError("location needs a name");
      const loc = await reflectLocation(name);
      emit(loc ? JSON.stringify(loc, null, 2) : `No location for "${name}" (REPL/builtin/FFI).`, loc);
      return;
    }
    case "search": {
      await ensureServer();
      const query = p[0];
      if (!query) throw new UsageError("search needs a query");
      const matches = (await reflectApropos(query)).sort();
      emit(matches.length ? matches.join("\n") : `No matches for "${query}"`, matches);
      return;
    }
    case "doc": {
      await ensureServer();
      const name = p[0];
      if (!name) throw new UsageError("doc needs a name");
      const info: ReflectInfo = await reflectInfo(name);
      const lines: string[] = [`Name: ${info.name ?? name}`];
      if (info.kind) lines.push(`Kind: ${info.kind}`);
      if (info.doc) lines.push(`Doc: ${info.doc}`);
      if (info.args) lines.push(`Args: (${info.args.join(", ")}${info["variadic?"] ? "..." : ""})`);
      if (info.fields) lines.push(`Fields: ${info.fields.join(", ")}`);
      if (info.variants) lines.push(`Variants: ${info.variants.join(" | ")}`);
      emit(lines.join("\n"), info);
      return;
    }
    case "describe": {
      await ensureServer();
      const messages = await replSend("describe");
      emit(formatReplResponse(messages), messages);
      return;
    }
    case "sessions": {
      await ensureServer();
      const messages = await replSend("ls-sessions");
      emit(formatReplResponse(messages), messages);
      return;
    }
    case "interrupt": {
      await ensureServer();
      if (!p[0]) throw new UsageError("interrupt needs a session");
      emit(await replRequest("interrupt", { session: p[0] }));
      return;
    }
    case "resume": {
      await ensureServer();
      const code = readArgOrStdin(p[0]);
      emit(await replRequest("resume", { code, session }));
      return;
    }
    case "abort": {
      await ensureServer();
      emit(await replRequest("abort", { session }));
      return;
    }
    case "main-status": {
      await ensureServer();
      emit(await replRequest("main-status", {}));
      return;
    }
    case "main-resume": {
      await ensureServer();
      const extra: Record<string, string> = {};
      if (p[0]) extra.code = p[0];
      emit(await replRequest("main-resume", extra));
      return;
    }
    case "main-abort": {
      await ensureServer();
      emit(await replRequest("main-abort", {}));
      return;
    }
    case "ls": {
      const dir = path.resolve(p[0] ?? process.cwd());
      const entries = fs.readdirSync(dir, { withFileTypes: true })
        .sort((a, b) => (a.isDirectory() !== b.isDirectory() ? (a.isDirectory() ? -1 : 1) : a.name.localeCompare(b.name)))
        .map((e) => (e.isDirectory() ? `${e.name}/` : e.name));
      emit(`${dir} (${entries.length} entries)\n${entries.join("\n")}`, entries);
      return;
    }
    default:
      throw new UsageError(`unknown command: ${cmd}`);
  }
}

async function main() {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv[0] === "--help" || argv[0] === "-h" || argv[0] === "help") {
    process.stdout.write(USAGE + "\n");
    process.exit(0);
  }
  const cmd = argv[0];
  const { positionals, flags } = parseArgs(argv.slice(1));
  JSON_OUT = flags.json === true;
  try {
    await dispatch(cmd, positionals, flags);
    disconnectRepl();
    process.exit(0);
  } catch (err: any) {
    disconnectRepl();
    if (err instanceof UsageError) {
      fail(`Error: ${err.message}\n\n${USAGE}`);
    }
    fail(`Error: ${err.message ?? err}`);
  }
}

main();
