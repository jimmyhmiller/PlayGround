// ---------------------------------------------------------------------------
// beagle_repl_core — the shared Beagle live-REPL toolkit core.
//
// One source of truth for the nREPL transport, the connection/format plumbing,
// and the `beagle.reflect` introspection strings. Both the DeepSeek agent
// (beagle_agent.ts) and the CLI (beagle_repl_cli.ts) import this so the
// protocol and reflection logic can't drift between them.
//
// What lives HERE: transport + format + reflection (process-global singleton —
// each importer is its own process, so one socket per process is correct).
// What stays in the CALLERS: server lifecycle (the agent owns an attached
// child; the CLI spawns a detached daemon), logging, history, tool/command
// surfaces, and the agent loop.
//
// Per-interface wording (the hint lines that name `beagle_eval` vs `beagle
// eval`, etc.) and the connect-failure annotation are injected via
// configureRepl() so neither caller's user-facing output changes.
// ---------------------------------------------------------------------------

import * as net from "net";

export const REPL_HOST = "127.0.0.1";
export const REPL_PORT = 7888;

// ---------------------------------------------------------------------------
// Configuration hooks (let callers tailor output without forking the core)
// ---------------------------------------------------------------------------

export interface ReplFormatHints {
  // Guidance appended after "[main-thread SUSPENDED] <error>".
  mainSuspended: string;
  // Guidance appended after the "RESUMABLE EXCEPTION (suspend-depth: N)" line.
  resumable: string;
}

// Defaults use the agent's tool-name wording so an un-configured importer
// behaves exactly like the original agent.
const DEFAULT_HINTS: ReplFormatHints = {
  mainSuspended:
    `\nUse beagle_eval to fix the broken function, then beagle_main_resume to continue.`,
  resumable:
    `\nThe evaluation is suspended — the program is waiting for you to provide a value.` +
    `\nUse beagle_resume to supply a replacement value, or beagle_abort to abandon.`,
};

let currentHints: ReplFormatHints = DEFAULT_HINTS;
// Returns extra text to append to a "Connection failed" message (e.g. crash
// info), or undefined. The agent uses this to surface its server's exit info.
let onConnectError: (() => string | undefined) | undefined;

export interface ReplConfig {
  hints?: ReplFormatHints;
  onConnectError?: () => string | undefined;
}

export function configureRepl(cfg: ReplConfig): void {
  if (cfg.hints) currentHints = cfg.hints;
  if (cfg.onConnectError) onConnectError = cfg.onConnectError;
}

// ---------------------------------------------------------------------------
// nREPL-style TCP transport
// ---------------------------------------------------------------------------

export interface ReplResponse {
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
  resolve: (messages: ReplResponse[]) => void;
  timer: ReturnType<typeof setTimeout>;
}

let replSocket: net.Socket | undefined;
let replConnected = false;
let replBuffer = "";
let reqCounter = 0;
const pendingEvals = new Map<string, PendingEval>();

export function isReplConnected(): boolean {
  return replConnected;
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
    pending.resolve(pending.messages);
  }
}

export function disconnectRepl(): void {
  if (replSocket) {
    replSocket.destroy();
    replSocket = undefined;
    replConnected = false;
    replBuffer = "";
    for (const [, pending] of pendingEvals) {
      clearTimeout(pending.timer);
      pending.resolve([...pending.messages, { err: "Connection closed", status: ["error"] }]);
    }
    pendingEvals.clear();
  }
}

export function connectRepl(): Promise<void> {
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
          handleReplMessage(JSON.parse(line) as ReplResponse);
        } catch {
          // skip unparseable lines
        }
      }
    });
    sock.on("end", () => {
      replConnected = false;
      replSocket = undefined;
      for (const [, pending] of pendingEvals) {
        clearTimeout(pending.timer);
        pending.resolve(pending.messages);
      }
      pendingEvals.clear();
    });
    sock.on("error", (err) => {
      replConnected = false;
      replSocket = undefined;
      for (const [, pending] of pendingEvals) {
        clearTimeout(pending.timer);
        pending.resolve([...pending.messages, { err: err.message, status: ["error"] }]);
      }
      pendingEvals.clear();
      reject(err);
    });
  });
}

// Low-level request: resolves with the raw nREPL message stream.
export async function replSend(op: string, extra: Record<string, string> = {}): Promise<ReplResponse[]> {
  if (!replConnected) {
    try {
      await connectRepl();
    } catch (err: any) {
      let msg = `Connection failed: ${err.message ?? err}`;
      const extraText = onConnectError?.();
      if (extraText) msg += `\n\n${extraText}`;
      return [{ err: msg, status: ["error"] }];
    }
  }
  const id = String(++reqCounter);
  const msg = JSON.stringify({ op, id, ...extra }) + "\n";
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      const pending = pendingEvals.get(id);
      pendingEvals.delete(id);
      resolve([...(pending?.messages ?? []), { err: "Timed out after 30s", status: ["error"] }]);
    }, 30_000);
    pendingEvals.set(id, { messages: [], resolve, timer });
    replSocket!.write(msg);
  });
}

export async function replRequest(op: string, extra: Record<string, string> = {}): Promise<string> {
  return formatReplResponse(await replSend(op, extra));
}

export function formatReplResponse(messages: ReplResponse[]): string {
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
    if ((msg as any)["main-thread"] === "suspended") {
      parts.push(`[main-thread SUSPENDED] ${(msg as any).error ?? "unknown error"}`);
      parts.push(currentHints.mainSuspended);
    } else if ((msg as any)["main-thread"] === "running") {
      parts.push(`[main-thread running]`);
    }
  }
  const output = parts.join("");
  let result: string;
  if (output && value !== undefined) result = `${output}\n=> ${value}`;
  else if (value !== undefined) result = `=> ${value}`;
  else if (output) result = output;
  else result = JSON.stringify(messages, null, 2);
  if (resumable) {
    result += `\n\n⚠️ RESUMABLE EXCEPTION (suspend-depth: ${suspendDepth ?? "unknown"})`;
    result += currentHints.resumable;
  }
  return result;
}

// Actively probe the REPL: connect if needed, then round-trip a `describe`.
// Returns null on success, or a string describing why the probe failed. Used
// for status checks where the cached `replConnected` flag can lag reality.
export async function probeReplConnection(): Promise<string | null> {
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
      resolve: () => { clearTimeout(timer); resolve(null); },
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

// ---------------------------------------------------------------------------
// Connectivity helpers
// ---------------------------------------------------------------------------

export function waitForPort(host: string, port: number, timeoutMs = 15_000): Promise<void> {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    function tryConnect() {
      if (Date.now() - start > timeoutMs) {
        reject(new Error(`Timed out waiting for ${host}:${port}`));
        return;
      }
      const sock = new net.Socket();
      sock.once("connect", () => { sock.destroy(); resolve(); });
      sock.once("error", () => { sock.destroy(); setTimeout(tryConnect, 200); });
      sock.connect(port, host);
    }
    tryConnect();
  });
}

export function isPortOpen(host: string, port: number, timeoutMs = 500): Promise<boolean> {
  return new Promise((resolve) => {
    const sock = new net.Socket();
    const done = (open: boolean) => { sock.destroy(); resolve(open); };
    sock.setTimeout(timeoutMs);
    sock.once("connect", () => done(true));
    sock.once("timeout", () => done(false));
    sock.once("error", () => done(false));
    sock.connect(port, host);
  });
}

// ---------------------------------------------------------------------------
// Small text helpers
// ---------------------------------------------------------------------------

export function beagleStringEscape(s: string): string {
  return s
    .replace(/\\/g, "\\\\")
    // Neutralize `${...}` so embedded Beagle source (e.g. passed to
    // reflect/persist or eval) is treated as literal text, not interpolated
    // in the REPL's own scope. Beagle parses `\$` back to a literal `$`.
    // Must run after the backslash-doubling above so the inserted `\` survives.
    .replace(/\$/g, () => "\\$")
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t");
}

export function isErrorResponse(result: string): boolean {
  return result.includes("[error]") || result.includes("[stderr]");
}

export function extractErrorText(result: string): string | null {
  const errMatch = result.match(/\[error\]\s*(.+)/);
  if (errMatch) return errMatch[1].trim();
  const stderrMatch = result.match(/\[stderr\]\s*(.+)/);
  if (stderrMatch) return stderrMatch[1].trim();
  return null;
}

export function extractNamespaceFromText(text: string): string | undefined {
  const match = text.match(/^\s*namespace\s+(\S+)/m);
  return match ? match[1] : undefined;
}

// ---------------------------------------------------------------------------
// Introspection via beagle.reflect
// ---------------------------------------------------------------------------
//
// Every reflection eval runs in a dedicated session with the reflect alias
// (re)established inline. REPL sessions share ONE global current-namespace, so
// any `namespace X` eval elsewhere can otherwise strand the alias and make the
// next reflect call fail with "Namespace alias not found: reflect". Compiling
// the import together with its use keeps the alias in scope.

export const INTROSPECT_SESSION = "introspect";
const REFLECT_PRELUDE = "use beagle.reflect as reflect\n";

export class IntrospectError extends Error {}

// A failed reflection eval throws, which the session turns into a *resumable
// suspension*. Left alone, each retry stacks another and suspend-depth climbs
// without bound. Abort so the session returns clean — reflection failure is
// "fix and retry", never "resume".
async function abortIfSuspended(messages: ReplResponse[]): Promise<void> {
  if (messages.some((m) => m.status?.includes("resumable"))) {
    await replSend("abort", { session: INTROSPECT_SESSION });
  }
}

export async function introspectSend(code: string): Promise<ReplResponse[]> {
  const messages = await replSend("eval", {
    code: REFLECT_PRELUDE + code,
    session: INTROSPECT_SESSION,
  });
  await abortIfSuspended(messages);
  return messages;
}

export async function introspectEval(code: string): Promise<string> {
  return formatReplResponse(await introspectSend(code));
}

// Evaluate `code` wrapped in json-encode and return the parsed result.
export async function introspectJson<T = unknown>(code: string): Promise<T> {
  const messages = await introspectSend(`json-encode(${code})`);
  const exMsg = messages.find((m) => m.ex !== undefined)?.ex;
  if (exMsg !== undefined) throw new IntrospectError(String(exMsg));
  const errMsg = messages.find((m) => m.err !== undefined)?.err;
  if (errMsg !== undefined) throw new IntrospectError(String(errMsg));
  const valueMsg = [...messages].reverse().find((m) => m.value !== undefined);
  if (valueMsg?.value === undefined) {
    throw new IntrospectError("Reflection eval returned no value");
  }
  return JSON.parse(valueMsg.value as string) as T;
}

// Shapes returned by beagle.reflect, JSON-encoded. The variadic flag key is
// `variadic?` (with the question mark) — the runtime interns it that way.
export interface ReflectFn { name: string; doc?: string | null; args?: string[] | null; "variadic?"?: boolean }
export interface ReflectStruct { name: string; doc?: string | null; fields?: string[] | null }
export interface ReflectEnum { name: string; doc?: string | null; variants?: string[] | null }
export interface ReflectNsInfo {
  name?: string;
  functions?: ReflectFn[] | null;
  structs?: ReflectStruct[] | null;
  enums?: ReflectEnum[] | null;
}
export interface ReflectInfo {
  name?: string;
  kind?: string;
  doc?: string | null;
  args?: string[] | null;
  "variadic?"?: boolean;
  fields?: string[] | null;
  variants?: string[] | null;
}

export function formatNamespaceInfo(ns: string, info: ReflectNsInfo): string {
  const lines: string[] = [`Namespace: ${ns}`];
  const fns = info.functions ?? [];
  if (fns.length > 0) {
    lines.push(`\nFunctions (${fns.length}):`);
    for (const f of [...fns].sort((a, b) => a.name.localeCompare(b.name))) {
      const params = (f.args ?? []).join(", ");
      const variadic = f["variadic?"] ? "..." : "";
      let sig = `  ${f.name}(${params}${variadic})`;
      if (f.doc) sig += ` — ${f.doc}`;
      lines.push(sig);
    }
  }
  const structs = info.structs ?? [];
  if (structs.length > 0) {
    lines.push(`\nStructs (${structs.length}):`);
    for (const s of [...structs].sort((a, b) => a.name.localeCompare(b.name))) {
      const fields = s.fields && s.fields.length > 0 ? ` { ${s.fields.join(", ")} }` : "";
      lines.push(`  ${s.name}${fields}`);
    }
  }
  const enums = info.enums ?? [];
  if (enums.length > 0) {
    lines.push(`\nEnums (${enums.length}):`);
    for (const e of [...enums].sort((a, b) => a.name.localeCompare(b.name))) {
      const variants = e.variants && e.variants.length > 0 ? ` { ${e.variants.join(" | ")} }` : "";
      lines.push(`  ${e.name}${variants}`);
    }
  }
  if (lines.length === 1) lines.push("(no functions, structs, or enums)");
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// High-level reflect operations — the single home for the reflect code strings.
// Callers (agent tools, CLI commands) compose these instead of hand-writing
// `reflect/...` evals, so the Beagle-side API surface lives in exactly one place.
// ---------------------------------------------------------------------------

export async function reflectAllNamespaces(): Promise<string[]> {
  return introspectJson<string[]>("reflect/all-namespaces()");
}

export async function reflectNamespaceInfo(ns: string): Promise<ReflectNsInfo> {
  return introspectJson<ReflectNsInfo>(`reflect/namespace-info("${beagleStringEscape(ns)}")`);
}

// `name` resolved via resolve_by_name (handles "ns/name" forms), so the
// target namespace need not be aliased in the introspect session.
export async function reflectSource(name: string): Promise<string | null> {
  return introspectJson<string | null>(`reflect/source("${beagleStringEscape(name)}")`);
}

export async function reflectNamespaceSource(ns: string): Promise<string | null> {
  return introspectJson<string | null>(`reflect/namespace-source("${beagleStringEscape(ns)}")`);
}

export async function reflectLocation(name: string): Promise<Record<string, unknown> | null> {
  return introspectJson<Record<string, unknown> | null>(`reflect/location("${beagleStringEscape(name)}")`);
}

export async function reflectApropos(query: string): Promise<string[]> {
  return introspectJson<string[]>(`reflect/apropos("${beagleStringEscape(query)}")`);
}

// `nameRef` is a Beagle *reference* (resolved in the introspect session), not a
// string literal — reflect/info takes the value directly. Alias its namespace
// first for "ns/name" forms.
export async function reflectInfo(nameRef: string): Promise<ReflectInfo> {
  return introspectJson<ReflectInfo>(`reflect/info(${nameRef})`);
}

// Persist def(s) to disk AND the running program; returns the formatted result
// (a `[{:name, :action} ...]` list, or an error). Callers do their own parsing
// of the action list if they need it.
export async function reflectPersist(ns: string, text: string): Promise<string> {
  return introspectEval(`reflect/persist("${beagleStringEscape(ns)}", "${beagleStringEscape(text)}")`);
}
