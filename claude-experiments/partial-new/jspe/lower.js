// lower.js — source -> { code, entries, nslots, initState, leaders, loopHeads, loopModified }.
// Supports MULTIPLE functions + direct calls. Each function lowers into the shared `code`
// array at entries[fid]; calls push a frame at the callee's entry. Subset: arithmetic,
// comparison, var/assign, if/while, arrays (incl. index assign), named function calls.
const { parseProgram } = require("./parse.js");
const { AB } = require("./contracts.js");

// Interprocedural fixpoint: which parameter positions does each function MUTATE
// (an array param written via `p[i]=...`, directly or by passing p to a function
// that mutates that position). Lets a loop know a callee mutates the array arg.
function walk(node, fn) {
  if (!Array.isArray(node)) return;
  fn(node);
  for (const c of node) if (Array.isArray(c)) walk(c, fn);
}
function mutatedParams(funcs) {
  const params = {}; funcs.forEach((f) => (params[f[1]] = f[2]));
  const mut = {}; funcs.forEach((f) => (mut[f[1]] = new Set()));
  let changed = true;
  while (changed) {
    changed = false;
    for (const f of funcs) {
      const ps = f[2], before = mut[f[1]].size;
      walk(f[3], (node) => {
        if (node[0] === "assign" && node[1][0] === "idx" && node[1][1][0] === "var") {
          const pi = ps.indexOf(node[1][1][1]); if (pi >= 0) mut[f[1]].add(pi);
        }
        if (node[0] === "call" && node[1][0] === "var" && mut[node[1][1]]) {
          for (const j of mut[node[1][1]]) { const a = node[2][j]; if (a && a[0] === "var") { const pi = ps.indexOf(a[1]); if (pi >= 0) mut[f[1]].add(pi); } }
        }
      });
      if (mut[f[1]].size !== before) changed = true;
    }
  }
  return mut;
}

function slot(ctx, name) { if (!(name in ctx.slotOf)) ctx.slotOf[name] = ctx.nslots++; return ctx.slotOf[name]; }
// record a var name as a mutated-ref in all enclosing loops (resolved to slots later)
function markMut(ctx, name) { for (const L of ctx.loopStack) L.mutNames.add(name); }
function emitStore(ctx, sl) { ctx.code.push({ tag: "Store", slot: sl }); for (const L of ctx.loopStack) L.modified.add(sl); }

const LOWER_EXPR = {};
LOWER_EXPR.lit = (n, ctx) => ctx.code.push(typeof n[1] === "number" ? { tag: "PushNum", n: n[1] } : (typeof n[1] === "boolean" ? { tag: "PushBool", b: n[1] } : { tag: "PushStr", s: n[1] }));
LOWER_EXPR.var = (n, ctx) => ctx.code.push({ tag: "Load", slot: slot(ctx, n[1]) });
LOWER_EXPR.bin = (n, ctx) => { lowerExpr(n[2], ctx); lowerExpr(n[3], ctx); ctx.code.push({ tag: "Bin", op: n[1] }); };
LOWER_EXPR.cond = (n, ctx) => { lowerExpr(n[1], ctx); const jf = ctx.code.length; ctx.code.push({ tag: "JmpIfFalsy", target: -1 }); lowerExpr(n[2], ctx); const jmp = ctx.code.length; ctx.code.push({ tag: "Jmp", target: -1 }); ctx.code[jf].target = ctx.code.length; ctx.leaders.add(ctx.code.length); lowerExpr(n[3], ctx); ctx.code[jmp].target = ctx.code.length; ctx.leaders.add(ctx.code.length); };
LOWER_EXPR.arr = (n, ctx) => { for (const el of n[1]) lowerExpr(el, ctx); ctx.code.push({ tag: "NewArray", n: n[1].length }); };
LOWER_EXPR.idx = (n, ctx) => { lowerExpr(n[1], ctx); lowerExpr(n[2], ctx); ctx.code.push({ tag: "GetIndex" }); };
LOWER_EXPR.dot = (n, ctx) => { lowerExpr(n[1], ctx); ctx.code.push({ tag: "GetProp", k: n[2] }); };
LOWER_EXPR.call = (n, ctx) => {
  const callee = n[1];
  if (callee[0] !== "var") throw new Error("only named calls supported");
  const fid = ctx.nameToFid[callee[1]];
  if (fid === undefined) throw new Error("unknown function " + callee[1]);
  for (const a of n[2]) lowerExpr(a, ctx);
  const mp = ctx.mutated[callee[1]] || new Set();
  for (const j of mp) if (n[2][j] && n[2][j][0] === "var") markMut(ctx, n[2][j][1]);
  ctx.code.push({ tag: "Call", fid, nargs: n[2].length });
};
LOWER_EXPR.assign = (n, ctx) => {
  const lhs = n[1];
  if (lhs[0] === "var") { lowerExpr(n[2], ctx); emitStore(ctx, slot(ctx, lhs[1])); }
  else if (lhs[0] === "idx") {
    if (lhs[1][0] === "var") markMut(ctx, lhs[1][1]);
    lowerExpr(lhs[1], ctx); lowerExpr(lhs[2], ctx); lowerExpr(n[2], ctx); ctx.code.push({ tag: "SetIndexOp" });
  }
  else throw new Error("unsupported assign target " + lhs[0]);
};

const LOWER_STMT = {};
LOWER_STMT.return = (n, ctx) => { lowerExpr(n[1], ctx); ctx.code.push({ tag: "Ret" }); };
LOWER_STMT.var = (n, ctx) => { lowerExpr(n[2], ctx); emitStore(ctx, slot(ctx, n[1])); };
LOWER_STMT.expr = (n, ctx) => { lowerExpr(n[1], ctx); if (n[1][0] !== "assign") ctx.code.push({ tag: "Pop" }); };
LOWER_STMT.block = (n, ctx) => { for (const s of n[1]) lowerStmt(s, ctx); };
LOWER_STMT.if = (n, ctx) => {
  lowerExpr(n[1], ctx);
  const jf = ctx.code.length; ctx.code.push({ tag: "JmpIfFalsy", target: -1 });
  lowerStmt(n[2], ctx);
  if (n[3]) {
    const jmp = ctx.code.length; ctx.code.push({ tag: "Jmp", target: -1 });
    ctx.code[jf].target = ctx.code.length; ctx.leaders.add(ctx.code.length);
    lowerStmt(n[3], ctx);
    ctx.code[jmp].target = ctx.code.length; ctx.leaders.add(ctx.code.length);
  } else { ctx.code[jf].target = ctx.code.length; ctx.leaders.add(ctx.code.length); }
};
LOWER_STMT.while = (n, ctx) => {
  const head = ctx.code.length; ctx.leaders.add(head);
  ctx.loopStack.push({ head, modified: new Set(), mutNames: new Set() });
  lowerExpr(n[1], ctx);
  const jf = ctx.code.length; ctx.code.push({ tag: "JmpIfFalsy", target: -1 });
  lowerStmt(n[2], ctx);
  ctx.code.push({ tag: "Jmp", target: head });
  const exit = ctx.code.length; ctx.code[jf].target = exit; ctx.leaders.add(exit);
  const L = ctx.loopStack.pop();
  ctx.loopHeads.add(head); ctx.loopModified[head] = [...L.modified];
  ctx.loopMutRefs[head] = [...L.mutNames].map((nm) => ctx.slotOf[nm]).filter((x) => x !== undefined);
};

function lowerExpr(n, ctx) { const f = LOWER_EXPR[n[0]]; if (!f) throw new Error("no LOWER_EXPR." + n[0]); return f(n, ctx); }
function lowerStmt(n, ctx) { const f = LOWER_STMT[n[0]]; if (!f) throw new Error("no LOWER_STMT." + n[0]); return f(n, ctx); }

function lower(source) {
  const funcs = parseProgram(source);
  const nameToFid = {}; funcs.forEach((f, i) => (nameToFid[f[1]] = i));
  const mutated = mutatedParams(funcs);
  const code = [], entries = [], nslots = [];
  const leaders = new Set(), loopHeads = new Set(), loopModified = {}, loopMutRefs = {};
  for (let fid = 0; fid < funcs.length; fid++) {
    const f = funcs[fid];
    entries[fid] = code.length;
    const ctx = { code, slotOf: {}, nslots: 0, leaders, loopHeads, loopModified, loopMutRefs, loopStack: [], nameToFid, mutated };
    f[2].forEach((p) => slot(ctx, p));
    for (const stmt of f[3]) lowerStmt(stmt, ctx);
    code.push({ tag: "PushUndef" }, { tag: "Ret" }); // implicit return undefined
    nslots[fid] = ctx.nslots;
  }
  const mainFid = nameToFid["main"] !== undefined ? nameToFid["main"] : 0;
  const nparams = funcs[mainFid][2].length;
  const locals = [];
  for (let i = 0; i < nslots[mainFid]; i++) locals.push(i < nparams ? AB.Dyn({ tag: "Var", id: i }) : AB.Undef());
  const initState = { frames: [{ pc: entries[mainFid], func: mainFid, locals, ostack: [] }], heap: new Map(), nextAddr: 0, pendingJoins: [], handlers: [] };
  return { funcs, code, entries, nslots, initState, leaders, loopHeads, loopModified, loopMutRefs, paramName: "v0" };
}
module.exports = { LOWER_EXPR, LOWER_STMT, lowerExpr, lowerStmt, lower };
