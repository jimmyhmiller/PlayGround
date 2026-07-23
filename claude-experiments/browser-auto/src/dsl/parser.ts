import {
  ALL_KINDS,
  type Action,
  type Effect,
  type Flow,
  type Given,
  type Step,
  type Target,
  type TargetKind,
  requiresEffects,
} from "./ir.js";

export class FlowParseError extends Error {
  constructor(public file: string, public problems: string[]) {
    super(
      problems.length === 1
        ? `${file}: ${problems[0]}`
        : `${file}: ${problems.length} problems:\n  - ${problems.join("\n  - ")}`,
    );
    this.name = "FlowParseError";
  }
}

// ---------------------------------------------------------------------------
// Tokenizer (per line)

type Token =
  | { t: "word"; v: string }
  | { t: "string"; v: string }
  | { t: "number"; v: number }
  | { t: "path"; v: string }
  | { t: "var"; v: string }
  | { t: "eq" };

function tokenize(line: string, lineNo: number, problems: string[]): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  const n = line.length;
  while (i < n) {
    const c = line[i]!;
    if (c === " " || c === "\t") { i++; continue; }
    if (c === "#") break; // trailing comment
    if (c === '"') {
      let v = "";
      i++;
      let closed = false;
      while (i < n) {
        const ch = line[i]!;
        if (ch === "\\" && i + 1 < n) {
          const next = line[i + 1]!;
          v += next === "n" ? "\n" : next;
          i += 2;
        } else if (ch === '"') { i++; closed = true; break; }
        else { v += ch; i++; }
      }
      if (!closed) problems.push(`line ${lineNo}: unterminated string`);
      tokens.push({ t: "string", v });
      continue;
    }
    if (c === "=") { tokens.push({ t: "eq" }); i++; continue; }
    if (c === "$") {
      const m = /^\$([A-Za-z_][\w-]*)/.exec(line.slice(i));
      if (!m) { problems.push(`line ${lineNo}: '$' must be followed by a variable name`); i++; continue; }
      tokens.push({ t: "var", v: m[1]! });
      i += m[0].length;
      continue;
    }
    if (c === "/" || (c === "*" && line.slice(i).startsWith("**"))) {
      const m = /^[^\s#]+/.exec(line.slice(i))!;
      tokens.push({ t: "path", v: m[0] });
      i += m[0].length;
      continue;
    }
    if (/[0-9]/.test(c) || (c === "-" && /[0-9]/.test(line[i + 1] ?? ""))) {
      const iso = /^\d{4}-\d{2}-\d{2}T[\d:.]+(Z|[+-]\d{2}:?\d{2})?/.exec(line.slice(i));
      if (iso) {
        tokens.push({ t: "word", v: iso[0] });
        i += iso[0].length;
        continue;
      }
      const m = /^-?\d+(\.\d+)?/.exec(line.slice(i))!;
      tokens.push({ t: "number", v: Number(m[0]) });
      i += m[0].length;
      continue;
    }
    const m = /^[^\s#"=]+/.exec(line.slice(i));
    if (m) {
      tokens.push({ t: "word", v: m[0] });
      i += m[0].length;
    } else {
      problems.push(`line ${lineNo}: unexpected character '${c}'`);
      i++;
    }
  }
  return tokens;
}

// ---------------------------------------------------------------------------
// Parser

class LineParser {
  pos = 0;
  constructor(
    public tokens: Token[],
    public lineNo: number,
    public problems: string[],
  ) {}

  peek(): Token | undefined { return this.tokens[this.pos]; }
  next(): Token | undefined { return this.tokens[this.pos++]; }
  atEnd(): boolean { return this.pos >= this.tokens.length; }

  fail(msg: string): null {
    this.problems.push(`line ${this.lineNo}: ${msg}`);
    return null;
  }

  expectWord(...options: string[]): string | null {
    const t = this.next();
    if (t?.t === "word" && (options.length === 0 || options.includes(t.v))) return t.v;
    return this.fail(`expected ${options.length ? options.map((o) => `'${o}'`).join(" or ") : "a word"}, got ${show(t)}`);
  }

  expectString(what: string): string | null {
    const t = this.next();
    if (t?.t === "string") return t.v;
    if (t?.t === "var") return `$${t.v}`;
    return this.fail(`expected ${what} (a quoted string), got ${show(t)}`);
  }

  expectPath(): string | null {
    const t = this.next();
    if (t?.t === "path") return t.v;
    if (t?.t === "string") return t.v;
    return this.fail(`expected a path starting with '/', got ${show(t)}`);
  }

  expectEnd(context: string): boolean {
    if (this.atEnd()) return true;
    this.fail(`unexpected trailing input after ${context}: ${show(this.peek())}`);
    return false;
  }

  /** <kind> ["name"] ( (in|of) <kind> ["name"] )*  — scoping is right-to-left */
  parseTarget(): Target | null {
    const first = this.parseOneTarget();
    if (!first) return null;
    let head = first;
    let tail = first;
    while (!this.atEnd() && this.peek()!.t === "word" && ["in", "of"].includes((this.peek() as { v: string }).v)) {
      this.next();
      const outer = this.parseOneTarget();
      if (!outer) return null;
      tail.within = outer;
      tail = outer;
    }
    return head;
  }

  private parseOneTarget(): Target | null {
    const t = this.next();
    if (t?.t !== "word") return this.fail(`expected a target kind (e.g. button, link, testid), got ${show(t)}`);
    if (!ALL_KINDS.includes(t.v)) {
      return this.fail(`unknown target kind "${t.v}" — known kinds: ${ALL_KINDS.join(", ")}`);
    }
    const kind = t.v as TargetKind;
    const nxt = this.peek();
    if (nxt?.t === "string" || nxt?.t === "var") {
      const name = this.next()!;
      return { kind, name: name.t === "var" ? `$${name.v}` : (name as { v: string }).v };
    }
    if (kind === "testid" || kind === "text" || kind === "field" || kind === "placeholder") {
      return this.fail(`target kind "${kind}" requires a quoted name`);
    }
    return { kind };
  }
}

function show(t: Token | undefined): string {
  if (!t) return "end of line";
  switch (t.t) {
    case "word": return `'${t.v}'`;
    case "string": return `"${t.v}"`;
    case "number": return String(t.v);
    case "path": return t.v;
    case "var": return `$${t.v}`;
    case "eq": return "'='";
  }
}

const ACTION_WORDS = ["go", "click", "dblclick", "fill", "select", "check", "uncheck", "press", "hover", "upload"] as const;

function parseGiven(p: LineParser): Given | null {
  const what = p.next();
  if (what?.t !== "word") return p.fail(`expected a given kind after 'given', got ${show(what)}`);
  switch (what.v) {
    case "seed": {
      const name = p.expectString("a seed name");
      if (name === null) return null;
      return p.expectEnd("given seed") ? { type: "seed", name } : null;
    }
    case "patch": {
      const entityTok = p.next();
      if (entityTok?.t !== "word") return p.fail(`expected an entity type after 'given patch', got ${show(entityTok)}`);
      const key = p.expectString("a fact key");
      if (key === null) return null;
      const fieldTok = p.next();
      if (fieldTok?.t !== "word") return p.fail(`expected a field name, got ${show(fieldTok)}`);
      const valueTok = p.next();
      if (!valueTok) return p.fail("expected a value for the patch");
      let value: unknown;
      if (valueTok.t === "number") value = valueTok.v;
      else if (valueTok.t === "string") value = valueTok.v;
      else if (valueTok.t === "word") {
        if (valueTok.v === "true") value = true;
        else if (valueTok.v === "false") value = false;
        else if (valueTok.v === "null") value = null;
        else return p.fail(`patch value must be a number, quoted string, true, false, or null — got '${valueTok.v}'`);
      } else return p.fail(`patch value must be a number, quoted string, true, false, or null — got ${show(valueTok)}`);
      return p.expectEnd("given patch")
        ? { type: "patch", entity: entityTok.v, key, field: fieldTok.v, value }
        : null;
    }
    case "user": {
      const key = p.expectString("a user fact key");
      if (key === null) return null;
      if (p.expectWord("signed-in") === null) return null;
      return p.expectEnd("given user") ? { type: "user", key } : null;
    }
    case "clock": {
      const t = p.next();
      const iso = t?.t === "word" || t?.t === "path" || t?.t === "string" ? String((t as { v: string }).v) : null;
      if (!iso) return p.fail(`expected an ISO-8601 instant after 'given clock', got ${show(t)}`);
      if (Number.isNaN(Date.parse(iso))) return p.fail(`'${iso}' is not a valid ISO-8601 instant (e.g. 2026-07-22T10:00:00Z)`);
      return p.expectEnd("given clock") ? { type: "clock", iso } : null;
    }
    case "stub": {
      const method = p.expectWord();
      if (method === null) return null;
      const path = p.expectPath();
      if (path === null) return null;
      const statusTok = p.next();
      if (statusTok?.t !== "number") return p.fail(`expected a status code, got ${show(statusTok)}`);
      let body: unknown;
      if (!p.atEnd()) {
        if (p.expectWord("json") === null) return null;
        const rest = p.tokens.slice(p.pos);
        const raw = rest.map((t) => (t.t === "string" ? JSON.stringify((t as { v: string }).v) : String((t as { v: unknown }).v))).join(" ");
        try {
          body = JSON.parse(raw);
        } catch {
          return p.fail(`stub json body is not valid JSON: ${raw}`);
        }
        p.pos = p.tokens.length;
      }
      const given: Given = { type: "stub", method: method.toUpperCase(), pathPattern: path, status: statusTok.v };
      if (body !== undefined) given.body = body;
      return given;
    }
    default:
      return p.fail(`unknown given '${what.v}' — known: seed, patch, user, clock, stub`);
  }
}

function parseAction(p: LineParser, word: string): Action | null {
  switch (word) {
    case "go": {
      const path = p.expectPath();
      if (path === null) return null;
      return p.expectEnd("go") ? { type: "go", path } : null;
    }
    case "click": case "dblclick": case "hover": {
      const target = p.parseTarget();
      if (!target) return null;
      return p.expectEnd(word) ? { type: word, target } : null;
    }
    case "fill": case "select": {
      // value is the LAST string token; target is everything before it
      const last = p.tokens[p.tokens.length - 1];
      if (last?.t !== "string" && last?.t !== "var") {
        return p.fail(`${word} needs a quoted value at the end: ${word} <target> "value"`);
      }
      const valueTok = p.tokens.pop()!;
      const value = valueTok.t === "var" ? `$${(valueTok as { v: string }).v}` : (valueTok as { v: string }).v;
      const target = p.parseTarget();
      if (!target) return null;
      return p.expectEnd(word) ? { type: word, target, value } : null;
    }
    case "check": case "uncheck": {
      const target = p.parseTarget();
      if (!target) return null;
      return p.expectEnd(word) ? { type: word, target } : null;
    }
    case "press": {
      const key = p.expectString("a key (e.g. \"Enter\")");
      if (key === null) return null;
      if (p.atEnd()) return { type: "press", key };
      if (p.expectWord("in") === null) return null;
      const target = p.parseTarget();
      if (!target) return null;
      return p.expectEnd("press") ? { type: "press", key, target } : null;
    }
    case "upload": {
      const last = p.tokens[p.tokens.length - 1];
      if (last?.t !== "string") return p.fail(`upload needs a quoted file path at the end`);
      const file = (p.tokens.pop() as { v: string }).v;
      const target = p.parseTarget();
      if (!target) return null;
      return p.expectEnd("upload") ? { type: "upload", target, file } : null;
    }
    default:
      return p.fail(`unknown action '${word}' — known: ${ACTION_WORDS.join(", ")}`);
  }
}

function parseEffect(p: LineParser): Effect | null {
  const first = p.peek();
  if (first?.t === "word" && first.v === "no") {
    p.next();
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd("expect no") ? { type: "absent", target } : null;
  }
  if (first?.t === "word" && (first.v === "appear" || first.v === "gone")) {
    p.next();
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd(`expect ${first.v}`) ? { type: first.v, target } : null;
  }
  if (first?.t === "word" && (first.v === "text" || first.v === "exact")) {
    let exact = false;
    if (first.v === "exact") {
      p.next();
      if (p.expectWord("text") === null) return null;
      exact = true;
    } else {
      p.next();
    }
    const value = p.expectString("the expected text");
    if (value === null) return null;
    if (p.atEnd()) return { type: "text", value, exact };
    if (p.expectWord("in") === null) return null;
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd("expect text") ? { type: "text", target, value, exact } : null;
  }
  if (first?.t === "word" && first.v === "value") {
    p.next();
    const value = p.expectString("the expected value");
    if (value === null) return null;
    if (p.expectWord("in") === null) return null;
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd("expect value") ? { type: "value", target, value } : null;
  }
  if (first?.t === "word" && first.v === "count") {
    p.next();
    const kindTok = p.next();
    if (kindTok?.t !== "word" || !ALL_KINDS.includes(kindTok.v)) {
      return p.fail(`expected a target kind after 'count', got ${show(kindTok)}`);
    }
    let name: string | undefined;
    if (p.peek()?.t === "string") name = (p.next() as { v: string }).v;
    const nTok = p.next();
    if (nTok?.t !== "number") return p.fail(`expected a number after 'count ${kindTok.v}', got ${show(nTok)}`);
    let within: Target | undefined;
    if (!p.atEnd()) {
      if (p.expectWord("in") === null) return null;
      within = p.parseTarget() ?? undefined;
      if (!within) return null;
      if (!p.expectEnd("expect count")) return null;
    }
    const eff: Effect = { type: "count", kind: kindTok.v as TargetKind, n: nTok.v };
    if (name !== undefined) eff.name = name;
    if (within !== undefined) eff.within = within;
    return eff;
  }
  if (first?.t === "word" && first.v === "url") {
    p.next();
    const path = p.expectPath();
    if (path === null) return null;
    return p.expectEnd("expect url") ? { type: "url", path } : null;
  }
  if (first?.t === "word" && first.v === "request") {
    p.next();
    const method = p.expectWord();
    if (method === null) return null;
    const path = p.expectPath();
    if (path === null) return null;
    let status: "ok" | number = "ok";
    if (!p.atEnd()) {
      const t = p.next()!;
      if (t.t === "number") status = t.v;
      else if (t.t === "word" && t.v === "ok") status = "ok";
      else return p.fail(`expected 'ok' or a status code, got ${show(t)}`);
    }
    return p.expectEnd("expect request")
      ? { type: "request", method: method.toUpperCase(), pathPattern: path, status }
      : null;
  }
  // plain visibility: expect <target>
  const target = p.parseTarget();
  if (!target) return null;
  return p.expectEnd("expect") ? { type: "visible", target } : null;
}

function parseLet(p: LineParser): Effect | null {
  const nameTok = p.next();
  if (nameTok?.t !== "word") return p.fail(`expected a variable name after 'let', got ${show(nameTok)}`);
  const eq = p.next();
  if (eq?.t !== "eq") return p.fail(`expected '=' after 'let ${nameTok.v}'`);
  if (p.expectWord("text") === null) return null;
  if (p.expectWord("in") === null) return null;
  const from = p.parseTarget();
  if (!from) return null;
  return p.expectEnd("let") ? { type: "let", name: nameTok.v, from } : null;
}

// ---------------------------------------------------------------------------
// Static checks over the parsed flow

function varsIn(s: string): string[] {
  return [...s.matchAll(/\$([A-Za-z_][\w-]*)/g)].map((m) => m[1]!);
}

function collectStringFields(obj: unknown, out: string[]): void {
  if (typeof obj === "string") { out.push(obj); return; }
  if (Array.isArray(obj)) { obj.forEach((v) => collectStringFields(v, out)); return; }
  if (typeof obj === "object" && obj !== null) {
    for (const v of Object.values(obj)) collectStringFields(v, out);
  }
}

function staticChecks(flow: Flow, problems: string[]): void {
  const defined = new Set<string>();
  for (const step of flow.steps) {
    if (requiresEffects(step.action) && step.effects.length === 0) {
      problems.push(
        `line ${step.line}: '${step.source.trim()}' declares no effects — every action must expect at least one ` +
          `observable effect (an unobserved action is where races hide)`,
      );
    }
    const used: string[] = [];
    collectStringFields(step.action, used);
    for (const e of step.effects) {
      if (e.type === "let") continue;
      collectStringFields(e, used);
    }
    for (const s of used) {
      for (const v of varsIn(s)) {
        if (!defined.has(v)) {
          problems.push(`line ${step.line}: $${v} is used before any 'let ${v} = ...' defines it`);
        }
      }
    }
    for (const e of step.effects) {
      if (e.type === "let") {
        if (defined.has(e.name)) problems.push(`line ${step.line}: $${e.name} is defined twice`);
        defined.add(e.name);
      }
    }
  }
}

// ---------------------------------------------------------------------------

export function parseFlow(source: string, file: string): Flow {
  const problems: string[] = [];
  const lines = source.split(/\r?\n/);
  let name: string | null = null;
  const givens: Given[] = [];
  const steps: Step[] = [];
  let sawStep = false;

  for (let idx = 0; idx < lines.length; idx++) {
    const raw = lines[idx]!;
    const lineNo = idx + 1;
    const trimmed = raw.trim();
    if (trimmed === "" || trimmed.startsWith("#")) continue;
    const indented = /^[ \t]/.test(raw);
    const tokens = tokenize(trimmed, lineNo, problems);
    if (tokens.length === 0) continue;
    const p = new LineParser(tokens, lineNo, problems);
    const head = tokens[0]!;

    if (indented) {
      if (steps.length === 0) {
        problems.push(`line ${lineNo}: indented line has no action above it`);
        continue;
      }
      if (head.t === "word" && head.v === "expect") {
        p.next();
        const eff = parseEffect(p);
        if (eff) steps[steps.length - 1]!.effects.push(eff);
      } else if (head.t === "word" && head.v === "let") {
        p.next();
        const eff = parseLet(p);
        if (eff) steps[steps.length - 1]!.effects.push(eff);
      } else {
        problems.push(`line ${lineNo}: indented lines must start with 'expect' or 'let', got ${show(head)}`);
      }
      continue;
    }

    if (head.t === "word" && head.v === "flow") {
      p.next();
      const n = p.expectString("the flow name");
      if (n !== null && p.expectEnd("flow")) {
        if (name !== null) problems.push(`line ${lineNo}: duplicate 'flow' declaration`);
        name = n;
      }
      continue;
    }
    if (head.t === "word" && head.v === "given") {
      if (sawStep) problems.push(`line ${lineNo}: 'given' must come before the first action — the world is fixed per flow`);
      p.next();
      const g = parseGiven(p);
      if (g) givens.push(g);
      continue;
    }
    if (head.t === "word" && head.v === "allow") {
      p.next();
      if (p.expectWord("console-errors") !== null && p.expectEnd("allow")) {
        givens.push({ type: "allow", what: "console-errors" });
      }
      continue;
    }
    if (head.t === "word" && (ACTION_WORDS as readonly string[]).includes(head.v)) {
      p.next();
      const action = parseAction(p, head.v);
      sawStep = true;
      if (action) steps.push({ action, effects: [], line: lineNo, source: trimmed });
      continue;
    }
    problems.push(
      `line ${lineNo}: expected 'flow', 'given', an action (${ACTION_WORDS.join(", ")}), or an indented 'expect'/'let' — got ${show(head)}`,
    );
  }

  if (name === null) problems.push(`missing 'flow "<name>"' declaration`);
  if (steps.length === 0) problems.push("flow has no steps");

  const flow: Flow = { name: name ?? "(unnamed)", file, givens, steps };
  staticChecks(flow, problems);
  if (problems.length > 0) throw new FlowParseError(file, problems);
  return flow;
}
