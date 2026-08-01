#!/usr/bin/env -S node --import tsx

/*
 * llparse -> Coil backend for llhttp.
 *
 * This file intentionally loads llparse from the pinned llhttp checkout instead
 * of declaring it as a dependency here.  That guarantees that generation uses
 * the exact llparse version recorded by llhttp's package-lock.json.  The graph
 * frontend is shared with the C backend; only lowering and source emission are
 * Coil-specific.
 */

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { writeFileSync } from 'node:fs';

const EXPECTED_LLHTTP_VERSION = '9.4.3';

class Wrap<T> {
  public constructor(public readonly ref: T) {}
}

/* These classes are deliberately distinct. llparse-frontend uses instanceof
 * checks to identify resumable nodes after it has translated the source graph.
 */
class Consume<T> extends Wrap<T> {}
class Empty<T> extends Wrap<T> {}
class ErrorNode<T> extends Wrap<T> {}
class Invoke<T> extends Wrap<T> {}
class Pause<T> extends ErrorNode<T> {}
class Sequence<T> extends Wrap<T> {}
class Single<T> extends Wrap<T> {}
class SpanEnd<T> extends Wrap<T> {}
class SpanStart<T> extends Wrap<T> {}
class TableLookup<T> extends Wrap<T> {}

class And<T> extends Wrap<T> {}
class External<T> extends Wrap<T> {}
class IsEqual<T> extends Wrap<T> {}
class Load<T> extends Wrap<T> {}
class MatchCode<T> extends Wrap<T> {}
class MulAdd<T> extends Wrap<T> {}
class Or<T> extends Wrap<T> {}
class SpanCode<T> extends Wrap<T> {}
class Store<T> extends Wrap<T> {}
class Test<T> extends Wrap<T> {}
class Update<T> extends Wrap<T> {}
class ValueCode<T> extends Wrap<T> {}

class ID<T> extends Wrap<T> {}
class ToLower<T> extends Wrap<T> {}
class ToLowerUnsafe<T> extends Wrap<T> {}

const implementation = {
  code: {
    And,
    External,
    IsEqual,
    Load,
    Match: MatchCode,
    MulAdd,
    Or,
    Span: SpanCode,
    Store,
    Test,
    Update,
    Value: ValueCode,
  },
  node: {
    Consume,
    Empty,
    Error: ErrorNode,
    Invoke,
    Pause,
    Sequence,
    Single,
    SpanEnd,
    SpanStart,
    TableLookup,
  },
  transform: { ID, ToLower, ToLowerUnsafe },
};

interface Args {
  upstream: string;
  output?: string;
  inspect: boolean;
}

function parseArgs(argv: readonly string[]): Args {
  let upstream = '';
  let output: string | undefined;
  let inspect = false;
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--upstream') {
      upstream = argv[++i] ?? '';
    } else if (arg === '--output') {
      output = argv[++i];
    } else if (arg === '--inspect') {
      inspect = true;
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  if (upstream === '') {
    throw new Error('usage: generate-coil.ts --upstream PATH [--output FILE] [--inspect]');
  }
  return { upstream: resolve(upstream), output, inspect };
}

function nodeKind(wrap: object): string {
  return unwrap(wrap).constructor.name;
}

function unwrap(wrap: any): any {
  return typeof wrap.get === 'function' ? wrap.get('coil') : wrap;
}

function allNodes(root: any): any[] {
  const out: any[] = [];
  const seen = new Set<any>();
  const visit = (wrap: any): void => {
    if (seen.has(wrap)) return;
    seen.add(wrap);
    out.push(wrap);
    const ref = unwrap(wrap).ref;
    const edges: any[] = [];
    if (ref.otherwise !== undefined) edges.push(ref.otherwise);
    if (ref.edge !== undefined) edges.push(ref.edge);
    if (Array.isArray(ref.edges)) edges.push(...ref.edges);
    for (const edge of edges) visit(edge.node);
  };
  visit(root);
  return out;
}

function coilString(data: Buffer | string): string {
  const bytes = typeof data === 'string' ? Buffer.from(data) : data;
  let out = '"';
  for (const byte of bytes) {
    if (byte === 0x22) out += '\\"';
    else if (byte === 0x5c) out += '\\\\';
    else if (byte === 0x0a) out += '\\n';
    else if (byte === 0x0d) out += '\\r';
    else if (byte === 0x09) out += '\\t';
    else if (byte >= 0x20 && byte <= 0x7e) out += String.fromCharCode(byte);
    else out += `\\x${byte.toString(16).padStart(2, '0')}`;
  }
  return out + '"';
}

class CoilEmitter {
  private readonly ids = new Map<any, number>();
  private readonly codeIds = new Map<string, number>();
  private readonly codeItems: any[] = [];
  private readonly externalIds = new Map<string, number>();

  public constructor(private readonly info: any,
                      private readonly nodes: any[]) {
    nodes.forEach((node, index) => this.ids.set(node, index));
  }

  public emit(): string {
    const lines: string[] = [
      '; Generated from llhttp 9.4.3 by scripts/llhttp/generate-coil.ts.',
      '; DO NOT EDIT. Regenerate from the pinned upstream graph.',
      '(module coil.llhttp.generated)',
      '(import "llhttp_types.coil" :as ll)',
      '(import "primitive.coil" :as primitive)',
      '(import "slice.coil" :as slice)',
      '',
      '(export internal-init internal-execute)',
      '',
      `(const ROOT_STATE ${this.id(this.info.root)})`,
      '',
      '(defn internal-init [(s (ptr ll/Parser))] (-> i64)',
      '  (ll/parser-zero! s)',
      '  (primitive/store! (primitive/field s current) ROOT_STATE)',
      '  0)',
      '',
      '(defn step [(s (ptr ll/Parser))',
      '            (p (mut (ptr u8)))',
      '            (endp (ptr u8))',
      '            (matched (mut i64))',
      '            (state i64)] (-> i64)',
      '  (case state',
    ];
    this.nodes.forEach((node, index) => {
      lines.push(`    ${index} ${this.emitNode(node)}`);
    });
    lines.push('    (do (ll/fail! s 1 "Invalid generated parser state" (primitive/load p) -1) -2)))');
    lines.push('');
    lines.push(...this.emitCodeDispatch());
    lines.push('');
    lines.push(...this.emitExecute());
    return lines.join('\n') + '\n';
  }

  private id(node: any): number {
    const id = this.ids.get(node);
    assert.notEqual(id, undefined, 'edge points outside enumerated graph');
    return id!;
  }

  private ref(node: any): any {
    return unwrap(node).ref;
  }

  private codeId(code: any): number {
    const item = unwrap(code);
    const ref = item.ref ?? item;
    const key = `${ref.signature}:${ref.cacheKey}`;
    let id = this.codeIds.get(key);
    if (id === undefined) {
      id = this.codeIds.size;
      this.codeIds.set(key, id);
      this.codeItems.push(item);
    }
    return id;
  }

  private transition(edge: any): string {
    const effects: string[] = [];
    if (!edge.noAdvance) {
      effects.push('(primitive/store! p (primitive/index (primitive/load p) 1))');
    }
    if (edge.value !== undefined) {
      effects.push(`(primitive/store! matched ${edge.value})`);
    }
    effects.push(String(this.id(edge.node)));
    return effects.length === 1 ? effects[0]! : `(do ${effects.join(' ')})`;
  }

  private atEnd(body: string): string {
    return `(if (= (primitive/load p) endp) -1 ${body})`;
  }

  private transform(ref: any, value: string): string {
    const name = unwrap(ref.transform).ref.name;
    if (name === 'id') return value;
    if (name === 'to_lower_unsafe') return `(primitive/ior ${value} 32)`;
    assert.equal(name, 'to_lower');
    return `(ll/ascii-lower ${value})`;
  }

  private emitNode(node: any): string {
    const impl = unwrap(node);
    const ref = impl.ref;
    const kind = impl.constructor.name;
    if (kind === 'Empty') {
      const body = this.transition(ref.otherwise);
      return ref.otherwise.noAdvance ? body : this.atEnd(body);
    }
    if (kind === 'ErrorNode') {
      return `(do (ll/fail! s ${ref.code} ${coilString(ref.reason)} (primitive/load p) -1) -2)`;
    }
    if (kind === 'Pause') {
      return `(do (ll/fail! s ${ref.code} ${coilString(ref.reason)} (primitive/load p) ${this.id(ref.otherwise.node)}) -2)`;
    }
    if (kind === 'Consume') {
      const field = this.info.properties.findIndex((p: any) => p.name === ref.field);
      assert(field >= 0);
      return `(let [avail (primitive/isub (primitive/cast i64 endp) (primitive/cast i64 (primitive/load p))) need (ll/field-load s ${field})] ` +
        `(if (primitive/icmp-ge avail need) ` +
        `(do (primitive/store! p (primitive/index (primitive/load p) need)) (ll/field-store! s ${field} 0) ${this.transition(ref.otherwise)}) ` +
        `(do (ll/field-store! s ${field} (primitive/isub need avail)) (primitive/store! p endp) -1)))`;
    }
    if (kind === 'Invoke') {
      const code = this.codeId(ref.code);
      const cases = ref.edges.map((edge: any) =>
        `${edge.code} ${this.transition({ node: edge.node, noAdvance: true })}`).join(' ');
      return `(let [result (invoke-code s ${code} (primitive/load p) endp (primitive/load matched))] ` +
        `(case result ${cases} ${this.transition(ref.otherwise)}))`;
    }
    if (kind === 'Single' || kind === 'TableLookup') {
      const current = this.transform(ref,
        '(primitive/cast i64 (primitive/load (primitive/load p)))');
      const entries: string[] = [];
      for (const edge of ref.edges) {
        const keys: number[] = kind === 'Single' ? [edge.key] : edge.keys;
        for (const key of keys) entries.push(`${key} ${this.transition(edge)}`);
      }
      return this.atEnd(`(let [current ${current}] (case current ${entries.join(' ')} ${this.transition(ref.otherwise)}))`);
    }
    if (kind === 'Sequence') {
      const bytes: number[] = [...ref.select];
      assert(bytes.length > 0, 'llparse Sequence must not be empty');
      const expected = `(case index ${bytes.map((byte, index) => `${index} ${byte}`).join(' ')} -1)`;
      const current = this.transform(ref,
        '(primitive/cast i64 (primitive/load (primitive/load p)))');
      const complete = this.transition({
        node: ref.edge.node,
        noAdvance: false,
        value: ref.edge.value,
      });
      return `(let [(mut status) 0] ` +
        `(loop (if (= (primitive/load p) endp) (break) ` +
        `(let [current ${current} index (primitive/load (primitive/field s index))] ` +
        `(if (primitive/icmp-eq current ${expected}) ` +
        `(if (primitive/icmp-eq (primitive/iadd index 1) ${bytes.length}) ` +
        `(do (primitive/store! (primitive/field s index) 0) (primitive/store! status 1) (break)) ` +
        `(do (primitive/store! (primitive/field s index) (primitive/iadd index 1)) ` +
        `(primitive/store! p (primitive/index (primitive/load p) 1)))) ` +
        `(do (primitive/store! (primitive/field s index) 0) (primitive/store! status 2) (break)))))) ` +
        `(case (primitive/load status) 1 ${complete} 2 ${this.transition(ref.otherwise)} -1))`;
    }
    if (kind === 'SpanStart') {
      const callback = this.codeId(ref.callback);
      return this.atEnd(`(do (primitive/store! (primitive/field s span-pos) (primitive/load p)) ` +
        `(primitive/store! (primitive/field s span-callback) ${callback}) ${this.transition(ref.otherwise)})`);
    }
    if (kind === 'SpanEnd') {
      const callback = this.codeId(ref.callback);
      const resumePos = ref.otherwise.noAdvance ? '(primitive/load p)' : '(primitive/index (primitive/load p) 1)';
      return `(let [start (primitive/load (primitive/field s span-pos))] ` +
        `(primitive/store! (primitive/field s span-pos) (primitive/cast (ptr u8) 0)) ` +
        `(let [err (invoke-span s ${callback} start (primitive/load p))] ` +
        `(if (primitive/icmp-ne err 0) ` +
        `(do (ll/fail! s err "Span callback error" ${resumePos} ${this.id(ref.otherwise.node)}) -2) ` +
        `${this.transition(ref.otherwise)})))`;
    }
    throw new Error(`unsupported node kind: ${kind}`);
  }

  private fieldId(name: string): number {
    const id = this.info.properties.findIndex((property: any) => property.name === name);
    assert(id >= 0, `unknown parser field: ${name}`);
    return id;
  }

  private externalId(name: string): number {
    let id = this.externalIds.get(name);
    if (id === undefined) {
      id = this.externalIds.size;
      this.externalIds.set(name, id);
    }
    return id;
  }

  private emitCodeDispatch(): string[] {
    const matchCases: string[] = [];
    const spanCases: string[] = [];
    for (let id = 0; id < this.codeItems.length; id++) {
      const item = this.codeItems[id]!;
      const ref = item.ref ?? item;
      const kind = item.constructor.name;
      let expression: string;
      if (kind === 'MatchCode' || kind === 'ValueCode') {
        expression = `(ll/invoke-external s ${this.externalId(ref.name)} p endp matched)`;
      } else if (kind === 'SpanCode') {
        spanCases.push(`    ${id} (ll/invoke-external-span s ${this.externalId(ref.name)} start endp)`);
        continue;
      } else {
        const field = this.fieldId(ref.field);
        if (kind === 'And') expression = `(do (ll/field-and! s ${field} ${ref.value}) 0)`;
        else if (kind === 'Or') expression = `(do (ll/field-or! s ${field} ${ref.value}) 0)`;
        else if (kind === 'Update') expression = `(do (ll/field-store! s ${field} ${ref.value}) 0)`;
        else if (kind === 'Store') expression = `(do (ll/field-store! s ${field} matched) 0)`;
        else if (kind === 'Load') expression = `(ll/field-load s ${field})`;
        else if (kind === 'IsEqual') expression = `(if (primitive/icmp-eq (ll/field-load s ${field}) ${ref.value}) 1 0)`;
        else if (kind === 'Test') expression = `(if (primitive/icmp-eq (primitive/iand (ll/field-load s ${field}) ${ref.value}) ${ref.value}) 1 0)`;
        else if (kind === 'MulAdd') {
          const max = ref.options.max === undefined ? -1 : ref.options.max;
          expression = `(ll/field-mul-add! s ${field} ${ref.options.base} ${max} ${ref.options.signed ? 'true' : 'false'} matched)`;
        } else throw new Error(`unsupported code kind: ${kind}`);
      }
      matchCases.push(`    ${id} ${expression}`);
    }
    const externalManifest = [...this.externalIds.entries()]
      .sort((a, b) => a[1] - b[1])
      .map(([name, id]) => `; external ${id}: ${name}`);
    return [
      ...externalManifest,
      '(defn invoke-code [(s (ptr ll/Parser)) (code i64) (p (ptr u8)) (endp (ptr u8)) (matched i64)] (-> i64)',
      '  (case code',
      ...matchCases,
      '    (do (ll/fail! s 1 "Invalid generated code id" p -1) 1)))',
      '',
      '(defn invoke-span [(s (ptr ll/Parser)) (code i64) (start (ptr u8)) (endp (ptr u8))] (-> i64)',
      '  (case code',
      ...spanCases,
      '    (do (ll/fail! s 1 "Invalid generated span id" start -1) 1)))',
    ];
  }

  private emitExecute(): string[] {
    return [
      '(defn internal-execute [(s (ptr ll/Parser)) (data (ptr u8)) (len i64)] (-> i64)',
      '  (if (primitive/icmp-ne (primitive/load (primitive/field s error)) 0)',
      '      (primitive/load (primitive/field s error))',
      '      (let [endp (primitive/index data len) (mut p) data (mut matched) 0',
      '            (mut state) (primitive/load (primitive/field s current))]',
      '        (when (primitive/icmp-ne (primitive/cast i64 (primitive/load (primitive/field s span-pos))) 0)',
      '          (primitive/store! (primitive/field s span-pos) data))',
      '        (loop',
      '          (let [next (step s (mut p) endp (mut matched) (primitive/load state))]',
      '            (cond',
      '              (primitive/icmp-ge next 0) (primitive/store! state next)',
      '              (primitive/icmp-eq next -1)',
      '                (do',
      '                  (primitive/store! (primitive/field s current) (primitive/load state))',
      '                  (if (primitive/icmp-eq (primitive/cast i64 (primitive/load (primitive/field s span-pos))) 0)',
      '                      0',
      '                      (let [err (invoke-span s',
      '                                   (primitive/load (primitive/field s span-callback))',
      '                                   (primitive/load (primitive/field s span-pos)) endp)]',
      '                        (if (primitive/icmp-eq err 0)',
      '                            0',
      '                            (ll/fail! s err "Span callback error" endp (primitive/load state)))))',
      '                  (break (primitive/load (primitive/field s error))))',
      '              :else (break (primitive/load (primitive/field s error)))))))))',
    ];
  }
}

async function buildGraph(args: Args): Promise<any> {
  const requireUpstream = createRequire(resolve(args.upstream, 'package.json'));
  const pkg = requireUpstream('./package.json');
  assert.equal(pkg.version, EXPECTED_LLHTTP_VERSION,
    `expected llhttp ${EXPECTED_LLHTTP_VERSION}, got ${pkg.version}`);

  const { LLParse } = requireUpstream('llparse');
  const frontend = requireUpstream('llparse-frontend');
  /* `npm ci` runs llhttp's prepare script, so use its compiled CommonJS here.
   * Besides being faster, this keeps this backend independent of whichever
   * TypeScript loader happened to launch it. */
  const llhttpModule = requireUpstream('./lib/llhttp/http.js');

  const llparse = new LLParse('llhttp__internal');
  const sourceRoot = new llhttpModule.HTTP(llparse).build().entry;
  const container = new frontend.Container();
  container.add('coil', implementation);
  const compiler = new frontend.Frontend('llhttp__internal', container.build());
  return compiler.compile(sourceRoot, (llparse as any).properties);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv);
  const info = await buildGraph(args);
  const nodes = allNodes(info.root);
  const counts = new Map<string, number>();
  const code = new Map<string, any>();
  const transforms = new Map<string, any>();
  const addCode = (value: any): void => {
    const item = unwrap(value);
    const ref = item.ref ?? item;
    code.set(ref.cacheKey, item);
  };
  for (const node of nodes) {
    const kind = nodeKind(node);
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
    const ref = unwrap(node).ref;
    if (ref.code !== undefined) {
      addCode(ref.code);
    }
    if (ref.callback !== undefined) {
      addCode(ref.callback);
    }
    if (ref.transform !== undefined) {
      const item = unwrap(ref.transform);
      transforms.set(item.ref.name, item);
    }
  }
  for (const span of info.spans) {
    for (const callback of span.callbacks) {
      addCode(callback);
    }
  }
  const codeCounts = new Map<string, number>();
  const externalCode: string[] = [];
  for (const item of code.values()) {
    const kind = item.constructor.name;
    const ref = item.ref ?? item;
    codeCounts.set(kind, (codeCounts.get(kind) ?? 0) + 1);
    if (kind === 'MatchCode' || kind === 'SpanCode' || kind === 'ValueCode') {
      externalCode.push(`${ref.signature}:${ref.name}`);
    }
  }

  if (args.inspect || args.output === undefined) {
    process.stdout.write(JSON.stringify({
      prefix: info.prefix,
      properties: info.properties,
      spans: info.spans.length,
      resumptionTargets: info.resumptionTargets.size,
      nodes: nodes.length,
      nodeKinds: Object.fromEntries([...counts].sort()),
      codeKinds: Object.fromEntries([...codeCounts].sort()),
      externalCode: externalCode.sort(),
      transforms: [...transforms.keys()].sort(),
    }, null, 2) + '\n');
  }

  if (args.output !== undefined) {
    const generated = new CoilEmitter(info, nodes).emit();
    writeFileSync(resolve(args.output), generated);
  }
}

await main();
