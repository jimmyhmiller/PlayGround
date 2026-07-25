import {
  ALL_KINDS,
  type Action,
  type Effect,
  type Flow,
  type Given,
  type Step,
  type Target,
  type TargetKind,
  formatTarget,
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
          // translate the known escapes; PRESERVE unknown ones (e.g. \d, \w)
          // so regex patterns in `matches "…"` survive intact
          if (next === "n") v += "\n";
          else if (next === "t") v += "\t";
          else if (next === '"' || next === "\\") v += next;
          else v += "\\" + next;
          i += 2;
        } else if (ch === '"') { i++; closed = true; break; }
        else { v += ch; i++; }
      }
      if (!closed) problems.push(`line ${lineNo}: unterminated string`);
      tokens.push({ t: "string", v });
      continue;
    }
    if (c === ">" || c === "<") {
      const two = line.slice(i, i + 2);
      const op = two === ">=" || two === "<=" ? two : c;
      tokens.push({ t: "word", v: op });
      i += op.length;
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
    if (t?.t === "var") return `$${t.v}`; // a captured value used as a path
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
    if (first.kind === "frame") {
      return this.fail(`"frame" is a scope, not a clickable target — write '<target> in frame "..."'`);
    }
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
    if (t?.t === "var") return { kind: "ref", name: t.v }; // a `for each` loop variable used as a scope
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
    if (kind === "testid" || kind === "text" || kind === "field" || kind === "placeholder" || kind === "frame") {
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

const ACTION_WORDS = ["go", "click", "dblclick", "fill", "select", "check", "uncheck", "press", "hover", "upload", "drag", "switch", "close"] as const;

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
    case "drag": {
      const target = p.parseTarget();
      if (!target) return null;
      if (p.expectWord("to") === null) return null;
      const to = p.parseTarget();
      if (!to) return null;
      return p.expectEnd("drag") ? { type: "drag", target, to } : null;
    }
    case "switch": {
      if (p.expectWord("tab") === null) return null;
      const path = p.expectPath();
      if (path === null) return null;
      return p.expectEnd("switch tab") ? { type: "switchTab", path } : null;
    }
    case "close": {
      if (p.expectWord("tab") === null) return null;
      return p.expectEnd("close tab") ? { type: "closeTab" } : null;
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
  if (
    first?.t === "word" &&
    (first.v === "appear" || first.v === "gone" || first.v === "checked" || first.v === "unchecked" || first.v === "enabled" || first.v === "disabled")
  ) {
    p.next();
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd(`expect ${first.v}`) ? { type: first.v, target } : null;
  }
  if (first?.t === "word" && first.v === "selected") {
    p.next();
    const value = p.expectString("the selected option's label");
    if (value === null) return null;
    if (p.expectWord("in") === null) return null;
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd("expect selected") ? { type: "selected", target, value } : null;
  }
  // text / exact text / matches text / title / exact title / matches title
  if (
    first?.t === "word" &&
    (first.v === "text" || first.v === "title" || ((first.v === "exact" || first.v === "matches") && p.tokens[p.pos + 1]?.t === "word" && ["text", "title"].includes((p.tokens[p.pos + 1] as { v: string }).v)))
  ) {
    let mode: "contains" | "exact" | "matches" = "contains";
    if (first.v === "exact" || first.v === "matches") {
      mode = first.v;
      p.next();
    }
    const kw = p.next(); // text | title
    const value = p.expectString(`the expected ${(kw as { v: string }).v}`);
    if (value === null) return null;
    if ((kw as { v: string }).v === "title") {
      return p.expectEnd("expect title") ? { type: "title", value, mode } : null;
    }
    if (p.atEnd()) return { type: "text", value, mode };
    if (p.expectWord("in") === null) return null;
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd("expect text") ? { type: "text", target, value, mode } : null;
  }
  if (first?.t === "word" && first.v === "value") {
    p.next();
    let mode: "contains" | "exact" | "matches" = "exact";
    if (p.peek()?.t === "word" && (p.peek() as { v: string }).v === "matches") { p.next(); mode = "matches"; }
    const value = p.expectString("the expected value");
    if (value === null) return null;
    if (p.expectWord("in") === null) return null;
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd("expect value") ? { type: "value", target, value, mode } : null;
  }
  if (first?.t === "word" && first.v === "attribute") {
    p.next();
    const attr = p.expectString("the attribute name (e.g. \"href\")");
    if (attr === null) return null;
    let mode: "contains" | "exact" | "matches" = "exact";
    if (p.peek()?.t === "word" && (p.peek() as { v: string }).v === "matches") { p.next(); mode = "matches"; }
    const value = p.expectString("the expected attribute value");
    if (value === null) return null;
    if (p.expectWord("of") === null) return null;
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd("expect attribute") ? { type: "attribute", attr, target, value, mode } : null;
  }
  if (first?.t === "word" && first.v === "count") {
    p.next();
    const kindTok = p.next();
    if (kindTok?.t !== "word" || !ALL_KINDS.includes(kindTok.v)) {
      return p.fail(`expected a target kind after 'count', got ${show(kindTok)}`);
    }
    let name: string | undefined;
    if (p.peek()?.t === "string") name = (p.next() as { v: string }).v;
    // optional comparison operator before the number: >= <= > < (default =)
    let op: "=" | ">=" | "<=" | ">" | "<" = "=";
    const opTok = p.peek();
    if (opTok?.t === "word" && [">=", "<=", ">", "<", "="].includes(opTok.v)) {
      op = opTok.v as typeof op;
      p.next();
    }
    const nTok = p.next();
    if (nTok?.t !== "number") return p.fail(`expected a number${op === "=" ? " or comparison (>=, <=)" : ""} after 'count ${kindTok.v}', got ${show(nTok)}`);
    let within: Target | undefined;
    if (!p.atEnd()) {
      if (p.expectWord("in") === null) return null;
      within = p.parseTarget() ?? undefined;
      if (!within) return null;
      if (!p.expectEnd("expect count")) return null;
    }
    const eff: Effect = { type: "count", kind: kindTok.v as TargetKind, n: nTok.v, op };
    if (name !== undefined) eff.name = name;
    if (within !== undefined) eff.within = within;
    return eff;
  }
  // `tab` is both an ARIA role and the new-tab effect. Disambiguate by the
  // argument: a PATH (/…) means the new-tab effect; a quoted name falls
  // through to the role-`tab` visibility target.
  if (first?.t === "word" && first.v === "tab" && p.tokens[p.pos + 1]?.t === "path") {
    p.next();
    const path = p.expectPath();
    if (path === null) return null;
    return p.expectEnd("expect tab") ? { type: "tab", path } : null;
  }
  // `dialog` is both an ARIA role and the native-dialog effect. Disambiguate by
  // the accept/dismiss keyword: present → dialog effect; absent → role target.
  if (
    first?.t === "word" &&
    first.v === "dialog" &&
    p.tokens[p.pos + 1]?.t === "string" &&
    p.tokens[p.pos + 2]?.t === "word" &&
    ["accept", "dismiss"].includes((p.tokens[p.pos + 2] as { v: string }).v)
  ) {
    p.next();
    const message = p.expectString("the dialog message substring");
    if (message === null) return null;
    const response = p.expectWord("accept", "dismiss");
    if (response === null) return null;
    let text: string | undefined;
    if (!p.atEnd() && p.peek()?.t === "string") {
      text = (p.next() as { v: string }).v;
    }
    if (!p.expectEnd("expect dialog")) return null;
    const eff: Effect = { type: "dialog", message, response: response as "accept" | "dismiss" };
    if (text !== undefined) (eff as { text?: string }).text = text;
    return eff;
  }
  if (first?.t === "word" && first.v === "download") {
    p.next();
    const name = p.expectString("the filename substring");
    if (name === null) return null;
    return p.expectEnd("expect download") ? { type: "download", name } : null;
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
    let bodyContains: string | undefined;
    if (!p.atEnd() && !(p.peek()?.t === "word" && (p.peek() as { v: string }).v === "containing")) {
      const t = p.next()!;
      if (t.t === "number") status = t.v;
      else if (t.t === "word" && t.v === "ok") status = "ok";
      else return p.fail(`expected 'ok', a status code, or 'containing', got ${show(t)}`);
    }
    if (p.peek()?.t === "word" && (p.peek() as { v: string }).v === "containing") {
      p.next();
      const b = p.expectString("a request-body substring (e.g. a GraphQL operation name)");
      if (b === null) return null;
      bodyContains = b;
    }
    if (!p.expectEnd("expect request")) return null;
    const eff: Effect = { type: "request", method: method.toUpperCase(), pathPattern: path, status };
    if (bodyContains !== undefined) (eff as { bodyContains?: string }).bodyContains = bodyContains;
    return eff;
  }
  if (first?.t === "word" && first.v === "ws") {
    p.next();
    const dirWord = p.expectWord("sent", "received");
    if (dirWord === null) return null;
    const text = p.expectString("the frame substring");
    if (text === null) return null;
    let pathPattern: string | undefined;
    if (!p.atEnd()) {
      if (p.expectWord("on") === null) return null;
      const path = p.expectPath();
      if (path === null) return null;
      pathPattern = path;
    }
    if (!p.expectEnd("expect ws")) return null;
    const eff: Effect = { type: "ws", dir: dirWord as "sent" | "received", text };
    if (pathPattern !== undefined) (eff as { pathPattern?: string }).pathPattern = pathPattern;
    return eff;
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
  const kind = p.expectWord("text", "value", "attribute", "count", "query");
  if (kind === null) return null;

  if (kind === "text" || kind === "value") {
    if (p.expectWord("in") === null) return null;
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd("let") ? { type: "let", name: nameTok.v, from: { kind, target } } : null;
  }
  if (kind === "attribute") {
    const attr = p.expectString("the attribute name");
    if (attr === null) return null;
    if (p.expectWord("of") === null) return null;
    const target = p.parseTarget();
    if (!target) return null;
    return p.expectEnd("let") ? { type: "let", name: nameTok.v, from: { kind, attr, target } } : null;
  }
  if (kind === "query") {
    const param = p.expectString("the query parameter name");
    if (param === null) return null;
    return p.expectEnd("let") ? { type: "let", name: nameTok.v, from: { kind, param } } : null;
  }
  // count
  const kindTok = p.next();
  if (kindTok?.t !== "word" || !ALL_KINDS.includes(kindTok.v)) {
    return p.fail(`expected a target kind after 'count', got ${show(kindTok)}`);
  }
  let name: string | undefined;
  if (p.peek()?.t === "string") name = (p.next() as { v: string }).v;
  let within: Target | undefined;
  if (!p.atEnd()) {
    if (p.expectWord("in") === null) return null;
    within = p.parseTarget() ?? undefined;
    if (!within) return null;
  }
  const from: import("./ir.js").CaptureSource = { kind: "count", countKind: kindTok.v as TargetKind };
  if (name !== undefined) from.name = name;
  if (within !== undefined) from.within = within;
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

/** collect every ref-target var name reachable in an object (loop scopes) */
function collectRefVars(obj: unknown, out: string[]): void {
  if (Array.isArray(obj)) { obj.forEach((v) => collectRefVars(v, out)); return; }
  if (obj && typeof obj === "object") {
    const t = obj as { kind?: unknown; name?: unknown };
    if (t.kind === "ref" && typeof t.name === "string") out.push(t.name);
    for (const v of Object.values(obj)) collectRefVars(v, out);
  }
}

function checkSteps(steps: Step[], defined: Set<string>, loopScopes: Set<string>, problems: string[]): void {
  for (const step of steps) {
    if (requiresEffects(step.action) && step.effects.length === 0) {
      problems.push(
        `line ${step.line}: '${step.source.trim()}' declares no effects — every action must expect at least one ` +
          `observable effect (an unobserved action is where races hide)`,
      );
    }
    // string $var interpolation (from `let`) + ref-target ($scope) uses.
    // For a forEach, check only its COLLECTION here — the body is a separate
    // scope, checked recursively below.
    const used: string[] = [];
    const refVars: string[] = [];
    if (step.action.type === "forEach") {
      collectStringFields(step.action.collection, used);
      collectRefVars(step.action.collection, refVars);
    } else {
      collectStringFields(step.action, used);
      collectRefVars(step.action, refVars);
      for (const e of step.effects) {
        if (e.type !== "let") { collectStringFields(e, used); collectRefVars(e, refVars); }
      }
    }
    for (const s of used) {
      for (const v of varsIn(s)) {
        if (!defined.has(v)) problems.push(`line ${step.line}: $${v} is used before any 'let ${v} = ...' defines it`);
      }
    }
    // ref-target ($var used as a scope) must be an in-scope `for each` variable
    for (const v of refVars) {
      if (!loopScopes.has(v)) {
        problems.push(`line ${step.line}: $${v} is used as a scope but is not a 'for each' loop variable in scope`);
      }
    }
    if (step.action.type === "forEach") {
      const inner = new Set(loopScopes);
      inner.add(step.action.loopVar);
      checkSteps(step.action.body, new Set(defined), inner, problems);
    }
    for (const e of step.effects) {
      if (e.type === "let") defined.add(e.name); // redefinition allowed (loop bodies re-capture)
    }
  }
}

function staticChecks(flow: Flow, problems: string[]): void {
  checkSteps(flow.steps, new Set<string>(), new Set<string>(), problems);
}

// ---------------------------------------------------------------------------
// Loop expansion — a `for` over a LITERAL table is unrolled at PARSE time into
// flat steps. The DSL stays data-not-code: the step count is known before the
// browser launches, every step is still a serializable indexed record, and
// atomic single-step replay is unaffected. This is macro expansion, never
// runtime control flow — there is deliberately no way to loop over dynamic
// (runtime-sized) data, because that would break replay-by-index.
//
//   for $cat $all
//     "Electronics" "All Electronics"
//     "Clothing"    "All Clothing"
//   do
//     click link "$cat"
//       expect text "$all"

interface SrcLine {
  raw: string;
  /** original source line number (for errors / replay attribution) */
  lineNo: number;
  /** iteration note, when this line came from a loop body */
  note?: string;
}

function indentOf(line: string): number {
  const m = /^[ \t]*/.exec(line)!;
  return m[0].replace(/\t/g, "  ").length;
}

function quotedValues(line: string): string[] {
  return [...line.matchAll(/"((?:[^"\\]|\\.)*)"/g)].map((m) => m[1]!.replace(/\\"/g, '"'));
}

function substituteVar(text: string, name: string, value: string): string {
  return text.replace(new RegExp("\\$" + name + "(?![\\w-])", "g"), value);
}

/** Gather the block of lines strictly deeper than `base`, starting at index p
 * (blank/comment lines inside are kept). Returns the lines and the next index. */
function gatherDeeper(lines: SrcLine[], p: number, base: number): { block: SrcLine[]; next: number } {
  const block: SrcLine[] = [];
  let j = p;
  while (j < lines.length) {
    const t = lines[j]!.raw.trim();
    if (t === "" || t.startsWith("#")) { block.push(lines[j]!); j++; continue; }
    if (indentOf(lines[j]!.raw) > base) { block.push(lines[j]!); j++; } else break;
  }
  return { block, next: j };
}

/** Parse the header of a runtime loop: `for each <target> as $var`. */
function parseForEachHeader(tokens: Token[], lineNo: number, problems: string[]): { collection: Target; loopVar: string } | null {
  const p = new LineParser(tokens.slice(2), lineNo, problems); // drop `for` `each`
  const collection = p.parseTarget();
  if (!collection) return null;
  if (p.expectWord("as") === null) return null;
  const v = p.next();
  if (v?.t !== "var") return p.fail(`'for each … as' expects a $variable, got ${show(v)}`), null;
  if (!p.expectEnd("for each")) return null;
  return { collection, loopVar: v.v };
}

/** The recursive step builder. Handles action+effect steps, the literal-table
 * `for` (unrolled here), and the runtime `for each` (a nested forEach step). */
function buildSteps(lines: SrcLine[], problems: string[]): Step[] {
  const meaningful = lines.filter((l) => l.raw.trim() !== "" && !l.raw.trim().startsWith("#"));
  if (meaningful.length === 0) return [];
  const base = Math.min(...meaningful.map((l) => indentOf(l.raw)));
  const steps: Step[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i]!;
    const t = line.raw.trim();
    if (t === "" || t.startsWith("#")) { i++; continue; }
    if (indentOf(line.raw) > base) {
      problems.push(`line ${line.lineNo}: indented line has no action above it`);
      i++;
      continue;
    }
    const tokens = tokenize(t, line.lineNo, problems);
    const head = tokens[0];
    if (!head) { i++; continue; }

    // ---- disallow top-level keywords inside a step region
    if (head.t === "word" && (head.v === "given" || head.v === "flow" || head.v === "allow")) {
      problems.push(
        head.v === "given"
          ? `line ${line.lineNo}: 'given' must come before the first action — the world is fixed per flow`
          : `line ${line.lineNo}: '${head.v}' must come before the first action`,
      );
      i++;
      continue;
    }

    // ---- for each  (runtime loop)  and  for … (literal table, unrolled)
    if (head.t === "word" && head.v === "for") {
      const isEach = tokens[1]?.t === "word" && tokens[1].v === "each";
      if (isEach) {
        const parsed = parseForEachHeader(tokens, line.lineNo, problems);
        const { block, next } = gatherDeeper(lines, i + 1, base);
        i = next;
        const body = buildSteps(block, problems);
        if (parsed) {
          if (body.length === 0) problems.push(`line ${line.lineNo}: 'for each' body is empty`);
          const src = `for each ${formatTarget(parsed.collection)} as $${parsed.loopVar}`;
          const step: Step = { action: { type: "forEach", collection: parsed.collection, loopVar: parsed.loopVar, body }, effects: [], line: line.lineNo, source: src };
          if (line.note !== undefined) step.iteration = line.note;
          steps.push(step);
        }
        continue;
      }
      // literal table: for $a $b [in] / rows / do / body
      const forLine = line.lineNo;
      const varTokens = t.replace(/^for\s+/, "").trim().split(/\s+/);
      if (varTokens[varTokens.length - 1] === "in") varTokens.pop();
      const varNames: string[] = [];
      let headerOk = true;
      for (const vt of varTokens) {
        const vm = /^\$([A-Za-z_][\w-]*)$/.exec(vt);
        if (!vm) { problems.push(`line ${forLine}: 'for' expects loop variables like '$name', got '${vt}'`); headerOk = false; }
        else varNames.push(vm[1]!);
      }
      // rows: deeper lines up to a base-indent `do`
      const rows: Array<{ values: string[]; lineNo: number }> = [];
      let j = i + 1;
      while (j < lines.length) {
        const lj = lines[j]!; const tj = lj.raw.trim();
        if (tj === "" || tj.startsWith("#")) { j++; continue; }
        if (indentOf(lj.raw) <= base) break; // reached `do` (or a sibling)
        const values = quotedValues(lj.raw);
        if (values.length !== varNames.length) {
          problems.push(`line ${lj.lineNo}: this 'for' row has ${values.length} value(s) but the loop declares ${varNames.length} variable(s) (${varNames.map((v) => "$" + v).join(", ")})`);
        } else rows.push({ values, lineNo: lj.lineNo });
        j++;
      }
      if (lines[j]?.raw.trim() !== "do") {
        problems.push(`line ${forLine}: 'for' loop needs a 'do' line before its body`);
        i = j;
        continue;
      }
      j++; // consume `do`
      const { block, next } = gatherDeeper(lines, j, base);
      i = next;
      if (!headerOk) continue;
      if (rows.length === 0) { problems.push(`line ${forLine}: 'for' loop has no data rows`); continue; }
      const bodyMeaning = block.filter((l) => l.raw.trim() !== "" && !l.raw.trim().startsWith("#"));
      if (bodyMeaning.length === 0) { problems.push(`line ${forLine}: 'for' loop 'do' body is empty`); continue; }
      const bodyBase = Math.min(...bodyMeaning.map((l) => indentOf(l.raw)));
      rows.forEach((row, iterIdx) => {
        const label = `iteration ${iterIdx + 1}/${rows.length}: ${varNames.map((v, k) => `$${v}="${row.values[k]}"`).join(", ")}`;
        const subbed: SrcLine[] = block.map((bl) => {
          let text = bl.raw.slice(indentOf(bl.raw) >= bodyBase ? bodyBase : 0);
          varNames.forEach((v, k) => { text = substituteVar(text, v, row.values[k]!); });
          return { raw: text, lineNo: bl.lineNo, note: label };
        });
        for (const s of buildSteps(subbed, problems)) {
          steps.push(s.iteration !== undefined ? s : { ...s, iteration: label });
        }
      });
      continue;
    }

    // ---- ordinary action + its indented effects
    if (head.t === "word" && (ACTION_WORDS as readonly string[]).includes(head.v)) {
      const { block, next } = gatherDeeper(lines, i + 1, base);
      i = next;
      const p = new LineParser(tokens, line.lineNo, problems);
      p.next();
      const action = parseAction(p, head.v);
      if (!action) continue;
      const step: Step = { action, effects: [], line: line.lineNo, source: t };
      if (line.note !== undefined) step.iteration = line.note;
      for (const bl of block) {
        const bt = bl.raw.trim();
        if (bt === "" || bt.startsWith("#")) continue;
        const btoks = tokenize(bt, bl.lineNo, problems);
        const bhead = btoks[0];
        const bp = new LineParser(btoks, bl.lineNo, problems);
        if (bhead?.t === "word" && bhead.v === "expect") { bp.next(); const e = parseEffect(bp); if (e) step.effects.push(e); }
        else if (bhead?.t === "word" && bhead.v === "let") { bp.next(); const e = parseLet(bp); if (e) step.effects.push(e); }
        else problems.push(`line ${bl.lineNo}: indented lines must start with 'expect' or 'let', got ${show(bhead)}`);
      }
      steps.push(step);
      continue;
    }

    problems.push(`line ${line.lineNo}: expected an action (${ACTION_WORDS.join(", ")}, for), got ${show(head)}`);
    i++;
  }
  return steps;
}

// ---------------------------------------------------------------------------

export function parseFlow(source: string, file: string): Flow {
  const problems: string[] = [];
  const rawLines: SrcLine[] = source.split(/\r?\n/).map((raw, i) => ({ raw, lineNo: i + 1 }));
  let name: string | null = null;
  const givens: Given[] = [];

  // leading region: flow / given / allow at column 0, before the first step
  let idx = 0;
  for (; idx < rawLines.length; idx++) {
    const { raw, lineNo } = rawLines[idx]!;
    const t = raw.trim();
    if (t === "" || t.startsWith("#")) continue;
    if (indentOf(raw) !== 0) break;
    const tokens = tokenize(t, lineNo, problems);
    const head = tokens[0];
    if (head?.t === "word" && head.v === "flow") {
      const p = new LineParser(tokens, lineNo, problems); p.next();
      const n = p.expectString("the flow name");
      if (n !== null && p.expectEnd("flow")) { if (name !== null) problems.push(`line ${lineNo}: duplicate 'flow' declaration`); name = n; }
      continue;
    }
    if (head?.t === "word" && head.v === "given") {
      const p = new LineParser(tokens, lineNo, problems); p.next();
      const g = parseGiven(p); if (g) givens.push(g);
      continue;
    }
    if (head?.t === "word" && head.v === "allow") {
      const p = new LineParser(tokens, lineNo, problems); p.next();
      const what = p.expectWord("console-errors", "dialogs");
      if (what !== null && p.expectEnd("allow")) givens.push({ type: "allow", what: what as "console-errors" | "dialogs" });
      continue;
    }
    break; // first step-region line
  }

  const steps = buildSteps(rawLines.slice(idx), problems);

  if (name === null) problems.push(`missing 'flow "<name>"' declaration`);
  if (steps.length === 0) problems.push("flow has no steps");

  const flow: Flow = { name: name ?? "(unnamed)", file, givens, steps };
  staticChecks(flow, problems);
  if (problems.length > 0) throw new FlowParseError(file, problems);
  return flow;
}
