// engine.mjs — JS port of livetype_poc.py, to verify before embedding in the artifact.
// The whole language + type-condition system, then a scripted driver mirroring the POC.

// ---------- AST ----------
const TInt = { k: 'Int' };
const TRec = (name) => ({ k: 'Rec', name });
const tyEq = (a, b) => a.k === 'Int' && b.k === 'Int' ? true : (a.k === 'Rec' && b.k === 'Rec' && a.name === b.name);
const tyStr = (t) => t.k === 'Int' ? 'Int' : t.name;

// ---------- lexer ----------
const PUNCT = new Set("{}(),:=+-*.");
function lex(s) {
  let i = 0; const toks = [];
  while (i < s.length) {
    const c = s[i];
    if (/\s/.test(c)) { i++; continue; }
    if (c === '-' && s[i + 1] === '>') { toks.push(['op', '->']); i += 2; continue; }
    if (PUNCT.has(c)) { toks.push(['punct', c]); i++; continue; }
    if (/[0-9]/.test(c)) { let j = i; while (j < s.length && /[0-9]/.test(s[j])) j++; toks.push(['int', parseInt(s.slice(i, j))]); i = j; continue; }
    if (/[A-Za-z_]/.test(c)) { let j = i; while (j < s.length && /[A-Za-z0-9_]/.test(s[j])) j++; toks.push(['name', s.slice(i, j)]); i = j; continue; }
    throw new SyntaxError(`bad char ${JSON.stringify(c)}`);
  }
  toks.push(['eof', null]);
  return toks;
}

// ---------- parser ----------
class Parser {
  constructor(toks) { this.t = toks; this.i = 0; }
  peek() { return this.t[this.i]; }
  next() { return this.t[this.i++]; }
  eat(kind, val) { const [k, v] = this.next(); if (k !== kind || (val !== undefined && v !== val)) throw new SyntaxError(`expected ${kind} ${val ?? ''}, got ${k} ${v}`); return v; }
  is(kind, val) { const [k, v] = this.peek(); return k === kind && (val === undefined || v === val); }

  parseType() { const n = this.eat('name'); return n === 'Int' ? TInt : TRec(n); }

  parseExpr() {
    let e = this.parseMul();
    while (this.is('punct', '+') || this.is('punct', '-')) { const op = this.next()[1]; e = { k: 'bin', op, l: e, r: this.parseMul() }; }
    return e;
  }
  parseMul() {
    let e = this.parseTerm();
    while (this.is('punct', '*')) { this.next(); e = { k: 'bin', op: '*', l: e, r: this.parseTerm() }; }
    return e;
  }
  parseTerm() {
    let e = this.parseAtom();
    while (this.is('punct', '.')) { this.next(); const f = this.eat('name'); e = { k: 'field', obj: e, fname: f }; }
    return e;
  }
  parseAtom() {
    const [k, v] = this.peek();
    if (k === 'int') { this.next(); return { k: 'lit', val: v }; }
    if (k === 'punct' && v === '(') { this.next(); const e = this.parseExpr(); this.eat('punct', ')'); return e; }
    if (k === 'name') {
      this.next();
      if (this.is('punct', '{')) {
        this.eat('punct', '{'); const flds = [];
        if (!this.is('punct', '}')) while (true) {
          const fn = this.eat('name'); this.eat('punct', ':'); flds.push([fn, this.parseExpr()]);
          if (this.is('punct', ',')) { this.next(); continue; } break;
        }
        this.eat('punct', '}'); return { k: 'cons', tname: v, fields: flds };
      }
      if (this.is('punct', '(')) {
        this.eat('punct', '('); const args = [];
        if (!this.is('punct', ')')) while (true) { args.push(this.parseExpr()); if (this.is('punct', ',')) { this.next(); continue; } break; }
        this.eat('punct', ')'); return { k: 'call', fn: v, args };
      }
      return { k: 'var', name: v };
    }
    throw new SyntaxError(`unexpected ${k} ${v}`);
  }

  parseProgram() {
    const decls = [];
    while (!this.is('eof')) {
      if (this.is('name', 'type')) {
        this.next(); const name = this.eat('name'); this.eat('punct', '=');
        this.eat('punct', '{'); const layout = [];
        if (!this.is('punct', '}')) while (true) {
          const fn = this.eat('name'); this.eat('punct', ':'); const ty = this.parseType();
          let dflt = null; if (this.is('punct', '=')) { this.next(); dflt = this.parseExpr(); }
          layout.push({ name: fn, ty, default: dflt });
          if (this.is('punct', ',')) { this.next(); continue; } break;
        }
        this.eat('punct', '}');
        decls.push({ kind: 'type', name, layout });
      } else if (this.is('name', 'fn')) {
        this.next(); const name = this.eat('name'); this.eat('punct', '(');
        const params = [];
        if (!this.is('punct', ')')) while (true) {
          const pn = this.eat('name'); this.eat('punct', ':'); const pt = this.parseType(); params.push([pn, pt]);
          if (this.is('punct', ',')) { this.next(); continue; } break;
        }
        this.eat('punct', ')'); this.eat('op', '->'); const ret = this.parseType(); this.eat('punct', '=');
        const body = this.parseExpr();
        decls.push({ kind: 'fn', name, params, ret, body });
      } else throw new SyntaxError(`expected 'type' or 'fn', got ${JSON.stringify(this.peek())}`);
    }
    return decls;
  }
}
const parseExpr = (src) => new Parser(lex(src)).parseExpr();
const parseProgram = (src) => new Parser(lex(src)).parseProgram();

// ---------- pretty-printer (AST -> source, for repair boxes) ----------
function unparse(e, prec = 0) {
  switch (e.k) {
    case 'lit': return String(e.val);
    case 'var': return e.name;
    case 'field': return `${unparse(e.obj, 3)}.${e.fname}`;
    case 'call': return `${e.fn}(${e.args.map(a => unparse(a, 0)).join(', ')})`;
    case 'cons': return `${e.tname}{ ${e.fields.map(([f, x]) => `${f}: ${unparse(x, 0)}`).join(', ')} }`;
    case 'bin': {
      const p = e.op === '*' ? 2 : 1;
      const s = `${unparse(e.l, p)} ${e.op} ${unparse(e.r, p + 1)}`;
      return p < prec ? `(${s})` : s;
    }
  }
  return '?';
}
const unparseFn = (fn) => `fn ${fn.name}(${fn.params.map(([n, t]) => `${n}: ${tyStr(t)}`).join(', ')}) -> ${tyStr(fn.ret)} = ${unparse(fn.body)}`;

// ---------- program store ----------
function newProgram() { return { typeVersions: new Map(), fns: new Map(), migrations: new Map(), migMeta: new Map() }; }
const curVer = (p, t) => p.typeVersions.get(t).length;
const layoutAt = (p, t, v) => p.typeVersions.get(t)[v - 1];
const curLayout = (p, t) => layoutAt(p, t, curVer(p, t));
const fieldOf = (p, t, v, f) => layoutAt(p, t, v).find(fs => fs.name === f) || null;

// ---------- type checker ----------
class TypeErr extends Error {}
function check(p, e, env) {
  switch (e.k) {
    case 'lit': return TInt;
    case 'var': { if (!(e.name in env)) throw new TypeErr(`unbound variable '${e.name}'`); return env[e.name]; }
    case 'bin': {
      const lt = check(p, e.l, env), rt = check(p, e.r, env);
      if (!(lt.k === 'Int' && rt.k === 'Int')) throw new TypeErr(`'${e.op}' needs Int, Int but got ${tyStr(lt)}, ${tyStr(rt)}`);
      return TInt;
    }
    case 'field': {
      const ot = check(p, e.obj, env);
      if (ot.k !== 'Rec') throw new TypeErr(`field '.${e.fname}' on non-record ${tyStr(ot)}`);
      const fs = fieldOf(p, ot.name, curVer(p, ot.name), e.fname);
      if (!fs) throw new TypeErr(`type '${ot.name}' has no field '${e.fname}'`);
      return fs.ty;
    }
    case 'call': {
      if (!p.fns.has(e.fn)) throw new TypeErr(`unknown function '${e.fn}'`);
      const fn = p.fns.get(e.fn);
      if (e.args.length !== fn.params.length) throw new TypeErr(`'${e.fn}' expects ${fn.params.length} args, got ${e.args.length}`);
      for (let i = 0; i < e.args.length; i++) {
        const at = check(p, e.args[i], env), [pn, pt] = fn.params[i];
        if (!tyEq(at, pt)) throw new TypeErr(`'${e.fn}' arg '${pn}' expects ${tyStr(pt)}, got ${tyStr(at)}`);
      }
      return fn.ret;
    }
    case 'cons': {
      if (!p.typeVersions.has(e.tname)) throw new TypeErr(`unknown type '${e.tname}'`);
      const layout = curLayout(p, e.tname); const given = Object.fromEntries(e.fields);
      for (const fs of layout) {
        if (fs.name in given) { const at = check(p, given[fs.name], env); if (!tyEq(at, fs.ty)) throw new TypeErr(`field '${fs.name}' of ${e.tname} expects ${tyStr(fs.ty)}, got ${tyStr(at)}`); }
        else if (fs.default === null) throw new TypeErr(`missing field '${fs.name}' in ${e.tname}{...}`);
      }
      for (const gn of Object.keys(given)) if (!layout.some(fs => fs.name === gn)) throw new TypeErr(`${e.tname} has no field '${gn}'`);
      return TRec(e.tname);
    }
  }
  throw new TypeErr(`cannot type ${e.k}`);
}
function checkFn(p, fn) {
  const env = {}; for (const [pn, pt] of fn.params) env[pn] = pt;
  try { const bt = check(p, fn.body, env); if (!tyEq(bt, fn.ret)) return `body has type ${tyStr(bt)}, declared return ${tyStr(fn.ret)}`; return null; }
  catch (ex) { if (ex instanceof TypeErr) return ex.message; throw ex; }
}
function recheckAll(p) {
  const newly = [];
  for (const fn of p.fns.values()) { const err = checkFn(p, fn); const was = fn.broken; fn.broken = err; if (err && !was) newly.push([fn.name, err]); }
  return newly;
}

// ---------- runtime ----------
let _idc = 0;
function mkRecord(tname, version, fields) { return { _rec: true, _id: ++_idc, tname, version, fields }; }
const isRec = (v) => v && v._rec;
function show(v) { if (isRec(v)) return `${v.tname}@v${v.version}{${Object.entries(v.fields).map(([k, x]) => `${k}=${show(x)}`).join(', ')}}`; return String(v); }

class Pause extends Error { constructor(kind, key, message) { super(message); this.kind = kind; this.key = key; this.msg = message; } }

function migrateToCurrent(p, r, log) {
  const cur = curVer(p, r.tname);
  while (r.version < cur) {
    const step = p.migrations.get(`${r.tname}:${r.version}`);
    if (!step) throw new Pause('migration', [r.tname, r.version],
      `value ${r.tname}#${r._id} is at v${r.version} but no transformer v${r.version}->${r.version + 1} exists (change was not auto-derivable)`);
    const nf = step(r); const from = r.version;
    r.fields = nf; r.version = from + 1;            // become:-style in-place migration
    if (log) log({ t: 'migrate', text: `migrated ${r.tname}#${r._id} v${from}->v${from + 1}` });
  }
  return r;
}
function evl(p, e, env, mig = true, log = null) {
  switch (e.k) {
    case 'lit': return e.val;
    case 'var': return env[e.name];
    case 'bin': { const a = evl(p, e.l, env, mig, log), b = evl(p, e.r, env, mig, log); return e.op === '+' ? a + b : e.op === '-' ? a - b : a * b; }
    case 'field': { const o = evl(p, e.obj, env, mig, log); if (mig && isRec(o)) migrateToCurrent(p, o, log); return o.fields[e.fname]; }
    case 'call': {
      const fn = p.fns.get(e.fn);
      if (fn.broken !== null) throw new Pause('function', e.fn, `function '${e.fn}' is inconsistent: ${fn.broken}`);
      const args = e.args.map(a => evl(p, a, env, mig, log));
      const ce = {}; fn.params.forEach(([pn], i) => ce[pn] = args[i]);
      return evl(p, fn.body, ce, mig, log);
    }
    case 'cons': {
      const layout = curLayout(p, e.tname); const given = Object.fromEntries(e.fields); const fields = {};
      for (const fs of layout) fields[fs.name] = (fs.name in given) ? evl(p, given[fs.name], env, mig, log) : evl(p, fs.default, {}, mig, log);
      return mkRecord(e.tname, curVer(p, e.tname), fields);
    }
  }
  throw new Error(`cannot eval ${e.k}`);
}
function valueOk(p, v, ty) {
  if (ty.k === 'Int') return typeof v === 'number';
  if (ty.k === 'Rec') {
    if (!(isRec(v) && v.tname === ty.name)) return false;
    const layout = layoutAt(p, v.tname, v.version);
    const keys = new Set(Object.keys(v.fields));
    if (keys.size !== layout.length || !layout.every(fs => keys.has(fs.name))) return false;
    return layout.every(fs => valueOk(p, v.fields[fs.name], fs.ty));
  }
  return false;
}

// ---------- updates / reconcile ----------
const layoutSig = (layout) => layout.map(fs => `${fs.name}:${tyStr(fs.ty)}${fs.default ? '=' + JSON.stringify(fs.default) : ''}`).join(',');
function tryAutoMigration(p, tname, fromV) {
  const oldL = layoutAt(p, tname, fromV); const oldM = Object.fromEntries(oldL.map(fs => [fs.name, fs]));
  const newL = layoutAt(p, tname, fromV + 1); const plan = [];
  for (const fs of newL) {
    if (fs.name in oldM && tyEq(oldM[fs.name].ty, fs.ty)) plan.push([fs.name, 'copy']);
    else if (fs.default !== null) plan.push([fs.name, ['default', fs.default]]);
    else return null;
  }
  return (oldRec) => { const f = {}; for (const [fname, how] of plan) f[fname] = how === 'copy' ? oldRec.fields[fname] : evl(p, how[1], {}, false); return f; };
}
function compileFn(d) { return { name: d.name, params: d.params, ret: d.ret, body: d.body, broken: null }; }

// load a fresh program image (reset)
function loadFresh(image) {
  const p = newProgram();
  for (const d of image) if (d.kind === 'type') p.typeVersions.set(d.name, [d.layout]);
  for (const d of image) if (d.kind === 'fn') p.fns.set(d.name, compileFn(d));
  recheckAll(p);
  return p;
}
// hot-update the live program from a new image; keep running state
function hotUpdate(p, image, log) {
  for (const d of image) if (d.kind === 'type') {
    if (!p.typeVersions.has(d.name)) { p.typeVersions.set(d.name, [d.layout]); log && log({ t: 'update', text: `new type ${d.name}` }); }
    else if (layoutSig(curLayout(p, d.name)) !== layoutSig(d.layout)) {
      p.typeVersions.get(d.name).push(d.layout);
      const fromV = curVer(p, d.name) - 1;
      const m = tryAutoMigration(p, d.name, fromV);
      if (m) { p.migrations.set(`${d.name}:${fromV}`, m); p.migMeta.set(`${d.name}:${fromV}`, 'auto'); log && log({ t: 'update', text: `type ${d.name} -> v${fromV + 1} (migration v${fromV}->v${fromV + 1} AUTO-DERIVED)` }); }
      else { p.migMeta.set(`${d.name}:${fromV}`, 'manual-pending'); log && log({ t: 'update', text: `type ${d.name} -> v${fromV + 1} (migration v${fromV}->v${fromV + 1} needs MANUAL transformer)` }); }
    }
  }
  for (const d of image) if (d.kind === 'fn') p.fns.set(d.name, compileFn(d));
  const newly = recheckAll(p);
  if (newly.length && log) for (const [nm, err] of newly) log({ t: 'broken', text: `${nm}: ${err}` });
  return newly;
}

// one transactional tick
function stepTick(p, state, stateType, log) {
  try {
    const ns = evl(p, { k: 'call', fn: 'tick', args: [{ k: 'var', name: '__s__' }] }, { __s__: state }, true, log);
    if (!valueOk(p, ns, TRec(stateType))) throw new Error(`INVARIANT VIOLATED: ${show(ns)}`);
    return { status: 'ok', state: ns };
  } catch (e) { if (e instanceof Pause) return { status: 'paused', cond: e }; throw e; }
}
function repairFunction(p, src) { const d = parseProgram(src)[0]; const fn = compileFn(d); const err = checkFn(p, fn); if (err) return err; fn.broken = null; p.fns.set(fn.name, fn); return null; }
function repairMigration(p, tname, fromV, exprSrc) { const expr = parseExpr(exprSrc); p.migrations.set(`${tname}:${fromV}`, (oldRec) => evl(p, expr, { old: oldRec }, false).fields); p.migMeta.set(`${tname}:${fromV}`, 'supplied'); }

// ============ scripted driver mirroring the Python POC ============
function main() {
  const boot = `
type Account = { id: Int, balance: Int }
fn init() -> Account = Account{ id: 1, balance: 100 }
fn charge(a: Account, amt: Int) -> Account = Account{ id: a.id, balance: a.balance - amt }
fn tick(s: Account) -> Account = charge(s, 5)
`;
  const p = loadFresh(parseProgram(boot));
  // unparse round-trip check: unparse -> reparse -> unparse must be stable
  for (const fn of p.fns.values()) {
    const s1 = unparseFn(fn), s2 = unparseFn(compileFn(parseProgram(s1)[0]));
    if (s1 !== s2) throw new Error(`unparse not stable:\n  ${s1}\n  ${s2}`);
  }
  console.log('unparse round-trip: OK');
  const log = (ev) => console.log(`   · ${ev.t}: ${ev.text}`);
  let state = evl(p, { k: 'call', fn: 'init', args: [] }, {});   // seed from init()
  console.log('BOOT:', show(state));

  const updates = {
    2: `type Account = { id: Int, balance: Int, fee: Int = 0 }
fn charge(a: Account, amt: Int) -> Account = Account{ id: a.id, balance: a.balance - amt }
fn tick(s: Account) -> Account = charge(s, 5)`,
    3: `type Money = { cents: Int }
type Account = { id: Int, balance: Money, fee: Int = 0 }
fn charge(a: Account, amt: Int) -> Account = Account{ id: a.id, balance: a.balance - amt }
fn tick(s: Account) -> Account = charge(s, 5)`,
  };
  const fnFix = 'fn charge(a: Account, amt: Int) -> Account = Account{ id: a.id, balance: Money{ cents: a.balance.cents - amt * 100 } }';
  const migFix = { key: 'Account:2', src: 'Account{ id: old.id, balance: Money{ cents: old.balance * 100 }, fee: old.fee }' };

  for (let tick = 0; tick < 6; tick++) {
    if (updates[tick]) { console.log(`\n== HOT UPDATE before tick ${tick} ==`); hotUpdate(p, parseProgram(updates[tick]), log); }
    let guard = 0;
    while (true) {
      if (guard++ > 8) throw new Error('too many pauses');
      const r = stepTick(p, state, 'Account', log);
      if (r.status === 'ok') { state = r.state; console.log(`tick ${tick}: ${show(state)}  [${valueOk(p, state, TRec('Account')) ? '✓' : '✗'}]`); break; }
      console.log(`  PAUSE (${r.cond.kind}): ${r.cond.msg}`);
      console.log(`  QUARANTINE: committed state still ${show(state)}`);
      if (r.cond.kind === 'function') { const e = repairFunction(p, fnFix); if (e) throw new Error('bad fix ' + e); console.log(`  repaired charge; resume`); }
      else { repairMigration(p, 'Account', 2, migFix.src); console.log(`  supplied Account v2->v3 transformer; resume`); }
    }
  }
  console.log('\nDONE (engine verified)');
}
main();

// ===== verify the NEW helpers the HTML lab adds, driving the real preset sources =====
function stateTypeName(p){ const t=p.fns.get('tick'); return (t&&t.ret.k==='Rec')?t.ret.name:'Account'; }
function migrationTemplate(p,tname,fromV){
  const oldNames=new Set(layoutAt(p,tname,fromV).map(f=>f.name));
  const newL=layoutAt(p,tname,fromV+1);
  return `${tname}{ ${newL.map(fs=>`${fs.name}: ${oldNames.has(fs.name)?'old.'+fs.name:(fs.ty.k==='Int'?'0':tyStr(fs.ty)+'{ }')}`).join(', ')} }`;
}
function regenSource(p){const lines=[];
  for(const [name,vers] of p.typeVersions){const L=vers[vers.length-1]; lines.push(`type ${name} = { ${L.map(fs=>`${fs.name}: ${tyStr(fs.ty)}${fs.default?` = ${unparse(fs.default)}`:''}`).join(', ')} }`);}
  lines.push(''); for(const fn of p.fns.values()) lines.push(unparseFn(fn)); return lines.join('\n');}

const SRC={
v1:`type Account = { id: Int, balance: Int }\nfn init() -> Account = Account{ id: 1, balance: 100 }\nfn charge(a: Account, amt: Int) -> Account = Account{ id: a.id, balance: a.balance - amt }\nfn tick(s: Account) -> Account = charge(s, 5)`,
v2:`type Account = { id: Int, balance: Int, fee: Int = 0 }\nfn init() -> Account = Account{ id: 1, balance: 100 }\nfn charge(a: Account, amt: Int) -> Account = Account{ id: a.id, balance: a.balance - amt }\nfn tick(s: Account) -> Account = charge(s, 5)`,
v3:`type Money = { cents: Int }\ntype Account = { id: Int, balance: Money, fee: Int = 0 }\nfn init() -> Account = Account{ id: 1, balance: Money{ cents: 10000 } }\nfn charge(a: Account, amt: Int) -> Account = Account{ id: a.id, balance: a.balance - amt }\nfn tick(s: Account) -> Account = charge(s, 5)`,
};
function main2(){
  console.log('\n===== LAB FLOW (preset sources + UI helpers) =====');
  const p=loadFresh(parseProgram(SRC.v1));
  let state=evl(p,{k:'call',fn:'init',args:[]},{}); let tick=0; const st=()=>stateTypeName(p);
  const drive=()=>{ let g=0; while(true){ if(g++>8) throw 'loop';
    const r=stepTick(p,state,st(),null);
    if(r.status==='ok'){ state=r.state; console.log(`tick ${tick}: ${show(state)} [${valueOk(p,state,TRec(st()))?'OK':'BAD'}]`); tick++; return; }
    console.log(`  PAUSE ${r.cond.kind}: ${r.cond.msg}`);
    if(r.cond.kind==='function'){ const fix='fn charge(a: Account, amt: Int) -> Account = Account{ id: a.id, balance: Money{ cents: a.balance.cents - amt * 100 } }';
      console.log(`  prefill was: ${unparseFn(p.fns.get(r.cond.key))}`);
      const e=repairFunction(p,fix); if(e) throw e; console.log('  fixed charge'); }
    else{ const [tn,fv]=r.cond.key; console.log(`  prefill was: ${migrationTemplate(p,tn,fv)}`);
      repairMigration(p,tn,fv,'Account{ id: old.id, balance: Money{ cents: old.balance * 100 }, fee: old.fee }'); console.log('  supplied transformer'); } } };
  drive(); drive();                                  // ticks 0,1 (v1)
  hotUpdate(p,parseProgram(SRC.v2),null); console.log('[hot-update v2]'); drive();  // auto-migrate
  hotUpdate(p,parseProgram(SRC.v3),null); console.log('[hot-update v3]'); drive();  // 2 pauses then commit
  drive();
  console.log('regen editor after repairs:\n'+regenSource(p).split('\n').map(l=>'   '+l).join('\n'));
}
main2();
