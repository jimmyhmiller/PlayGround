// FROZEN CONTRACTS — data shapes shared by every module. Agents NEVER edit this.
// Mirrors the core of src/js.rs (skipping deob-only/builtin variants for now).

// ---------- Residual IR (the OUTPUT side) ----------
// RExpr: a residual expression. `tag` selects the shape.
//   {tag:"Num", n}            {tag:"Str", s}           {tag:"Bool", b}
//   {tag:"Undef"}             {tag:"Null"}             {tag:"Var", id}        // id:int
//   {tag:"Bin", op, a, b}     // op: "+","-","*","/","%","===","!==","<","<=",">",">=","&&","||","&","|","^","<<",">>"
//   {tag:"Unary", op, a}      // op: "!","-","~","typeof","void"
//   {tag:"Index", a, i}       // a[i]
//   {tag:"Get", a, k}         // a.k     (k:string)
//   {tag:"Call", f, args}     {tag:"New", f, args}
//   {tag:"Opaque", op, args}  // an unmodeled op rendered verbatim
//   {tag:"Cond", c, t, e}     // c ? t : e
const RE = {
  Num: (n) => ({ tag: "Num", n }), Str: (s) => ({ tag: "Str", s }), Bool: (b) => ({ tag: "Bool", b }),
  Undef: () => ({ tag: "Undef" }), Null: () => ({ tag: "Null" }), Var: (id) => ({ tag: "Var", id }),
  Bin: (op, a, b) => ({ tag: "Bin", op, a, b }), Unary: (op, a) => ({ tag: "Unary", op, a }),
  Index: (a, i) => ({ tag: "Index", a, i }), Get: (a, k) => ({ tag: "Get", a, k }),
  Call: (f, args) => ({ tag: "Call", f, args }), New: (f, args) => ({ tag: "New", f, args }),
  Opaque: (op, args) => ({ tag: "Opaque", op, args }), Cond: (c, t, e) => ({ tag: "Cond", c, t, e }),
  Arr: (els) => ({ tag: "Arr", els }),
};

// Op: a residual straight-line statement.
//   {tag:"Eval", dst, expr}     // let v<dst> = expr;
//   {tag:"Store", name, expr}   // <name> = expr;   (name: a residual lvalue string or Var id)
//   {tag:"SetIndex", a, i, v}   // a[i] = v;
//   {tag:"SetProp", a, k, v}    // a.k = v;
//   {tag:"PushArr", a, v}       // a.push(v);
//   {tag:"ExprStmt", expr}      // expr;            (effect only)
const OP = {
  Eval: (dst, expr) => ({ tag: "Eval", dst, expr }), Store: (name, expr) => ({ tag: "Store", name, expr }),
  SetIndex: (a, i, v) => ({ tag: "SetIndex", a, i, v }), SetProp: (a, k, v) => ({ tag: "SetProp", a, k, v }),
  PushArr: (a, v) => ({ tag: "PushArr", a, v }), ExprStmt: (expr) => ({ tag: "ExprStmt", expr }),
};

// Cond (branch condition) + Terminator + Block + Program
//   Cond: {tag:"Falsy", expr}              // branch on !expr
//   Terminator: {tag:"Halt"} | {tag:"Br", b} | {tag:"CondBr", cond, t, f} | {tag:"Unset"}
//   Block: {ops:Op[], term:Terminator}     Program: {blocks:Block[], entry:int}

// ---------- Abstract values (the STATE side) ----------
// Abs: a partially-static value.
//   {tag:"Num", n} {tag:"Str", s} {tag:"Bool", b} {tag:"Undef"} {tag:"Null"}
//   {tag:"Ref", addr}      // pointer into State.heap
//   {tag:"Dyn", expr}      // residual: a runtime value described by an RExpr
const AB = {
  Num: (n) => ({ tag: "Num", n }), Str: (s) => ({ tag: "Str", s }), Bool: (b) => ({ tag: "Bool", b }),
  Undef: () => ({ tag: "Undef" }), Null: () => ({ tag: "Null" }),
  Ref: (addr) => ({ tag: "Ref", addr }), Dyn: (expr) => ({ tag: "Dyn", expr }),
};

// HeapObj: {tag:"Object", fields:[[k,Abs]...]} | {tag:"Array", elems:Abs[]}
//          | {tag:"Closure", fid, captured:Abs[]}
// State:   {frames:Frame[], heap:Map<int,HeapObj>, nextAddr:int, pendingJoins:[], handlers:[]}
// Frame:   {pc:int, func:int, locals:Abs[], ostack:Abs[]}

// ---------- Object-language bytecode (the INPUT side) — contract between lower & step ----------
// Instr tags (mirrors src/js.rs Instr): PushNum n | PushStr s | PushBool b | PushUndef | PushNull
//   | Load slot | Store slot | Bin op | Unary op | GetIndex | SetIndexOp | GetProp k | SetPropOp k
//   | NewArray n | NewObject keys | MakeClosure(fid,ncap) | Call nargs | NewOp nargs | Ret
//   | Jmp t | JmpIfFalsy t | Dup | Pop
// (lower.js produces these; step/* consumes them. Keep tags identical to the Rust enum.)

// ---------- Step result (what a step handler returns) ----------
// Step: {tag:"Continue"} | {tag:"Halt"} | {tag:"Jump", state} | {tag:"Branch", cond, t, f}

function UNIMPLEMENTED(name) { throw new Error("UNIMPLEMENTED: " + name); }

module.exports = { RE, OP, AB, UNIMPLEMENTED };
