// ir.js — residual IR emitter. REFERENCE MODULE showing the dispatch-table pattern.
// Each EMIT.* / EMITOP.* entry is ONE agent task. The dispatchers (emit/emitOp/
// emitProgram) are FROZEN — agents do not touch them.
const { UNIMPLEMENTED } = require("./contracts.js");

// ---- EMIT table: RExpr -> JS source string. `rec` is the recursive emitter. ----
const EMIT = {};
// [task ir.emit.num]    rustRef: js.rs RExpr::Num codegen
EMIT.Num = (e, rec) => "" + e.n;
// [task ir.emit.str]    string literal, JSON-quoted
EMIT.Str = (e, rec) => JSON.stringify(e.s);
// [task ir.emit.bool]
EMIT.Bool = (e, rec) => e.b ? 'true' : 'false';
// [task ir.emit.undef]
EMIT.Undef = (e, rec) => 'undefined';
// [task ir.emit.null]
EMIT.Null = (e, rec) => 'null';
// [task ir.emit.var]    e.id:int -> a variable name (use varName(e.id))
EMIT.Var = (e, rec) => varName(e.id);
// [task ir.emit.bin]    "(" + rec(e.a) + " " + e.op + " " + rec(e.b) + ")"
EMIT.Bin = (e, rec) => "(" + rec(e.a) + " " + e.op + " " + rec(e.b) + ")";
// [task ir.emit.unary]  prefix ops; "typeof"/"void" need a space
EMIT.Unary = (e, rec) => {
  const op = e.op;
  const a = rec(e.a);
  if (op === 'typeof' || op === 'void') {
    return '(' + op + ' ' + a + ')';
  }
  return '(' + op + a + ')';
};
// [task ir.emit.index]  rec(e.a) + "[" + rec(e.i) + "]"
EMIT.Index = (e, rec) => rec(e.a) + "[" + rec(e.i) + "]";
// [task ir.emit.get]    rec(e.a) + "." + e.k   (assume k is a valid identifier)
EMIT.Get = (e, rec) => rec(e.a) + "." + e.k;
// [task ir.emit.call]   rec(e.f) + "(" + e.args.map(rec).join(", ") + ")"
EMIT.Call = (e, rec) => rec(e.f) + "(" + e.args.map(rec).join(", ") + ")";
// [task ir.emit.new]    "new " + rec(e.f) + "(" + args + ")"
EMIT.New = (e, rec) => "new " + rec(e.f) + "(" + e.args.map(rec).join(", ") + ")";
// [task ir.emit.opaque] e.op + "(" + args + ")"  (op is a verbatim string)
EMIT.Opaque = (e, rec) => e.op + "(" + e.args.map(rec).join(", ") + ")";
// [task ir.emit.cond]   "(" + rec(e.c) + " ? " + rec(e.t) + " : " + rec(e.e) + ")"
EMIT.Arr = (e, rec) => "[" + e.els.map(rec).join(", ") + "]";
EMIT.Cond = (e, rec) => "(" + rec(e.c) + " ? " + rec(e.t) + " : " + rec(e.e) + ")";

// ---- EMITOP table: Op -> JS statement string ----
const EMITOP = {};
// [task ir.op.eval]     "let " + varName(o.dst) + " = " + emit(o.expr) + ";"
EMITOP.Eval = (o, emit) => "let " + varName(o.dst) + " = " + emit(o.expr) + ";";
// [task ir.op.store]    lvalue(o.name) + " = " + emit(o.expr) + ";"
EMITOP.Store = (o, emit) => lvalue(o.name) + " = " + emit(o.expr) + ";";
// [task ir.op.setindex] emit(o.a) + "[" + emit(o.i) + "] = " + emit(o.v) + ";"
EMITOP.SetIndex = (o, emit) => emit(o.a) + "[" + emit(o.i) + "] = " + emit(o.v) + ";";
// [task ir.op.setprop]
EMITOP.SetProp = (o, emit) => emit(o.a) + "." + o.k + " = " + emit(o.v) + ";";
// [task ir.op.pusharr]  emit(o.a) + ".push(" + emit(o.v) + ");"
EMITOP.PushArr = (o, emit) => emit(o.a) + ".push(" + emit(o.v) + ");";
// [task ir.op.exprstmt] emit(o.expr) + ";"
EMITOP.ExprStmt = (o, emit) => emit(o.expr) + ";";

// ---- helpers (frozen) ----
// [task ir.varname] map a Var id to a stable JS identifier (mirror js.rs naming bands)
function varName(id) { return "v" + id; }  // reference (real bands come from task ir.varname)
function lvalue(name) { return typeof name === "number" ? varName(name) : name; }

// ---- dispatchers (FROZEN — agents do not edit) ----
function emit(e) { const f = EMIT[e.tag]; if (!f) throw new Error("no EMIT." + e.tag); return f(e, emit); }
function emitOp(o) { const f = EMITOP[o.tag]; if (!f) throw new Error("no EMITOP." + o.tag); return f(o, emit); }
// [task ir.program] assemble blocks into a `switch(__pc)` dispatch loop (mirror js.rs codegen)
function emitProgram(program, paramName) {
  const { blocks, entry } = program;
  let code = `function main(${paramName}) {\n`;
  code += `  let __pc = ${entry};\n`;
  code += `  for (;;) {\n`;
  code += `    switch (__pc) {\n`;
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    code += `      case ${i}: {\n`;
    for (const op of block.ops) {
      code += `        ${emitOp(op)}\n`;
    }
    const term = block.term;
    if (term.tag === 'Halt') {
      code += `        return ${term.ret ? emit(term.ret) : ''};\n`;
    } else if (term.tag === 'Br') {
      code += `        __pc = ${term.b};\n`;
      code += `        break;\n`;
    } else if (term.tag === 'CondBr') {
      const condExpr = emit(term.cond);
      code += `        if (${condExpr}) {\n`;
      code += `          __pc = ${term.t};\n`;
      code += `        } else {\n`;
      code += `          __pc = ${term.f};\n`;
      code += `        }\n`;
      code += `        break;\n`;
    } else if (term.tag === 'Unset') {
      // do nothing
    }
    code += `      }\n`;
  }
  code += `    }\n`;
  code += `  }\n`;
  code += `}\n`;
  return code;
}

module.exports = { EMIT, EMITOP, emit, emitOp, emitProgram, varName, lvalue };
