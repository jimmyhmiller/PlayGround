// state.js — abstract values, heap, gc, canonical key. Each function below is ONE
// task (see tasks.json: state.*). Ports the State/Abs/heap parts of src/js.rs.
// HeapObj: {tag:"Object",fields:[[k,Abs]...]} | {tag:"Array",elems:[Abs]} | {tag:"Closure",fid,captured:[Abs]}
// State:   {frames:[{pc,func,locals:[Abs],ostack:[Abs]}], heap:Map<int,HeapObj>, nextAddr:int, pendingJoins:[], handlers:[]}
const { UNIMPLEMENTED } = require("./contracts.js");

// [task state.alloc]
function alloc(state, heapObj) {
  const addr = state.nextAddr;
  state.heap.set(addr, heapObj);
  state.nextAddr++;
  return addr;
}
// [task state.reachable]
function reachable(heap, addr, outSet) {
  if (outSet.has(addr)) return;
  outSet.add(addr);
  const obj = heap.get(addr);
  if (!obj) return;
  if (obj.tag === 'Object') {
    for (const [k, v] of obj.fields) {
      if (v.tag === 'Ref') reachable(heap, v.addr, outSet);
    }
  } else if (obj.tag === 'Array') {
    for (const v of obj.elems) {
      if (v.tag === 'Ref') reachable(heap, v.addr, outSet);
    }
  } else if (obj.tag === 'Closure') {
    for (const v of obj.captured) {
      if (v.tag === 'Ref') reachable(heap, v.addr, outSet);
    }
  }
}
// [task state.gc]
function gc(state) {
  const reachableSet = new Set();
  for (const frame of state.frames) {
    for (const abs of frame.locals) {
      if (abs.tag === 'Ref') reachable(state.heap, abs.addr, reachableSet);
    }
    for (const abs of frame.ostack) {
      if (abs.tag === 'Ref') reachable(state.heap, abs.addr, reachableSet);
    }
  }
  for (const addr of state.heap.keys()) {
    if (!reachableSet.has(addr)) {
      state.heap.delete(addr);
    }
  }
}
// [task state.weight]
function weight(state) {
  let total = 0;
  for (const frame of state.frames) {
    for (const abs of frame.locals) total += absWeight(abs);
    for (const abs of frame.ostack) total += absWeight(abs);
  }
  for (const [addr, obj] of state.heap) {
    if (obj.tag === 'Object') {
      for (const [k, v] of obj.fields) total += absWeight(v);
    } else if (obj.tag === 'Array') {
      for (const v of obj.elems) total += absWeight(v);
    } else if (obj.tag === 'Closure') {
      for (const v of obj.captured) total += absWeight(v);
    }
  }
  return total;
}

function absWeight(abs) {
  if (abs.tag === 'Dyn') return exprWeight(abs.expr);
  return 1;
}

function exprWeight(e) {
  let w = 1;
  if (e.tag === 'Bin') w += exprWeight(e.a) + exprWeight(e.b);
  else if (e.tag === 'Unary') w += exprWeight(e.a);
  else if (e.tag === 'Index') w += exprWeight(e.a) + exprWeight(e.i);
  else if (e.tag === 'Get') w += exprWeight(e.a);
  else if (e.tag === 'Call' || e.tag === 'New') {
    w += exprWeight(e.f);
    for (const arg of e.args) w += exprWeight(arg);
  } else if (e.tag === 'Cond') w += exprWeight(e.c) + exprWeight(e.t) + exprWeight(e.e);
  else if (e.tag === 'Opaque') {
    for (const arg of e.args) w += exprWeight(arg);
  }
  return w;
}
// [task state.canonicalize]
function canonicalize(state) {
  const outSet = new Set();
  for (const frame of state.frames) {
    for (const abs of frame.locals) {
      if (abs.tag === 'Ref') reachable(state.heap, abs.addr, outSet);
    }
    for (const abs of frame.ostack) {
      if (abs.tag === 'Ref') reachable(state.heap, abs.addr, outSet);
    }
  }
  const addrMap = new Map();
  let newAddr = 0;
  const newHeap = new Map();
  for (const addr of outSet) {
    addrMap.set(addr, newAddr);
    newAddr++;
  }
  for (const addr of outSet) {
    const obj = state.heap.get(addr);
    if (!obj) continue;
    const newObj = JSON.parse(JSON.stringify(obj));
    if (newObj.tag === 'Object') {
      for (const field of newObj.fields) {
        const v = field[1];
        if (v.tag === 'Ref') field[1] = { tag: 'Ref', addr: addrMap.get(v.addr) };
      }
    } else if (newObj.tag === 'Array') {
      for (let i = 0; i < newObj.elems.length; i++) {
        const v = newObj.elems[i];
        if (v.tag === 'Ref') newObj.elems[i] = { tag: 'Ref', addr: addrMap.get(v.addr) };
      }
    } else if (newObj.tag === 'Closure') {
      for (let i = 0; i < newObj.captured.length; i++) {
        const v = newObj.captured[i];
        if (v.tag === 'Ref') newObj.captured[i] = { tag: 'Ref', addr: addrMap.get(v.addr) };
      }
    }
    newHeap.set(addrMap.get(addr), newObj);
  }
  const newFrames = state.frames.map(f => ({
    pc: f.pc,
    func: f.func,
    locals: f.locals.map(a => a.tag === 'Ref' ? { tag: 'Ref', addr: addrMap.get(a.addr) } : a),
    ostack: f.ostack.map(a => a.tag === 'Ref' ? { tag: 'Ref', addr: addrMap.get(a.addr) } : a)
  }));
  return {
    frames: newFrames,
    heap: newHeap,
    nextAddr: newAddr,
    pendingJoins: state.pendingJoins.map(x => x.slice ? x.slice() : x),
    handlers: state.handlers.slice()
  };
}
// [task state.keyOf]   (used by engine.createOrGet/resolveJump as client.key)
// Allocation-light: renumbers only NON-FROZEN reachable addresses (DFS from frame roots)
// and serializes only their objects. FROZEN addresses (immutable static heap, e.g. an
// interpreter being specialized) are keyed by stable address 'F<addr>' and never walked —
// their content is constant across all states, so it can't distinguish them. This keeps
// the key proportional to the MUTABLE working heap, not the (possibly huge) static input.
const EMPTY_SET = new Set();
function keyOf(state) {
  const frozen = state.frozen || EMPTY_SET;
  const addrMap = new Map();
  let next = 0;
  const assign = (addr) => {
    if (frozen.has(addr) || addrMap.has(addr)) return;
    addrMap.set(addr, next++);
    const obj = state.heap.get(addr);
    if (!obj) return;
    if (obj.tag === 'Object') { for (const [k, v] of obj.fields) if (v.tag === 'Ref') assign(v.addr); }
    else if (obj.tag === 'Array') { for (const v of obj.elems) if (v.tag === 'Ref') assign(v.addr); }
    else if (obj.tag === 'Closure') { for (const v of obj.captured) if (v.tag === 'Ref') assign(v.addr); }
  };
  for (const f of state.frames) {
    for (const a of f.locals) if (a.tag === 'Ref') assign(a.addr);
    for (const a of f.ostack) if (a.tag === 'Ref') assign(a.addr);
  }
  const refKey = (a) => {
    if (a.tag !== 'Ref') return absKey(a);
    if (frozen.has(a.addr)) return 'F' + a.addr;
    return 'R' + addrMap.get(a.addr);
  };
  const objKey = (obj) => {
    if (obj.tag === 'Object') return 'O{' + obj.fields.map(([k, v]) => JSON.stringify(k) + ':' + refKey(v)).join(',') + '}';
    if (obj.tag === 'Array') return 'A[' + obj.elems.map(refKey).join(',') + ']';
    if (obj.tag === 'Closure') return 'C(' + obj.fid + ',' + obj.captured.map(refKey).join(',') + ')';
    throw new Error('unknown heap obj tag');
  };
  const parts = [];
  for (const f of state.frames) {
    parts.push('pc:' + f.pc + ',func:' + f.func);
    parts.push('L:' + f.locals.map(refKey).join('|'));
    parts.push('S:' + f.ostack.map(refKey).join('|'));
  }
  parts.push('H:');
  const inv = new Array(next);
  for (const [orig, canon] of addrMap) inv[canon] = orig;
  for (let c = 0; c < next; c++) parts.push(c + ':' + objKey(state.heap.get(inv[c])));
  parts.push('PJ:' + JSON.stringify(state.pendingJoins));
  parts.push('HD:' + JSON.stringify(state.handlers));
  return parts.join(';');
}
function absKey(a) {
  if (a.tag === 'Num') return 'Num(' + a.n + ')';
  if (a.tag === 'Str') return 'Str(' + JSON.stringify(a.s) + ')';
  if (a.tag === 'Bool') return 'Bool(' + a.b + ')';
  if (a.tag === 'Undef') return 'Undef';
  if (a.tag === 'Null') return 'Null';
  if (a.tag === 'Ref') return 'Ref(' + a.addr + ')';
  if (a.tag === 'Dyn') return 'Dyn(' + exprKey(a.expr) + ')';
  throw new Error('unknown abs tag');
}
function exprKey(e) {
  let s = e.tag;
  if (e.tag === 'Num') s += '(' + e.n + ')';
  else if (e.tag === 'Str') s += '(' + JSON.stringify(e.s) + ')';
  else if (e.tag === 'Bool') s += '(' + e.b + ')';
  else if (e.tag === 'Undef') s += '()';
  else if (e.tag === 'Null') s += '()';
  else if (e.tag === 'Var') s += '(' + e.id + ')';
  else if (e.tag === 'Bin') s += '(' + e.op + ',' + exprKey(e.a) + ',' + exprKey(e.b) + ')';
  else if (e.tag === 'Unary') s += '(' + e.op + ',' + exprKey(e.a) + ')';
  else if (e.tag === 'Index') s += '(' + exprKey(e.a) + ',' + exprKey(e.i) + ')';
  else if (e.tag === 'Get') s += '(' + exprKey(e.a) + ',' + JSON.stringify(e.k) + ')';
  else if (e.tag === 'Call') s += '(' + exprKey(e.f) + ',' + e.args.map(exprKey).join(',') + ')';
  else if (e.tag === 'New') s += '(' + exprKey(e.f) + ',' + e.args.map(exprKey).join(',') + ')';
  else if (e.tag === 'Cond') s += '(' + exprKey(e.c) + ',' + exprKey(e.t) + ',' + exprKey(e.e) + ')';
  else if (e.tag === 'Opaque') s += '(' + e.op + ',' + e.args.map(exprKey).join(',') + ')';
  else throw new Error('unknown expr tag');
  return s;
}
function heapObjKey(obj) {
  if (obj.tag === 'Object') {
    const fields = obj.fields.map(([k, v]) => JSON.stringify(k) + ':' + absKey(v)).join(',');
    return 'Object{' + fields + '}';
  } else if (obj.tag === 'Array') {
    return 'Array[' + obj.elems.map(absKey).join(',') + ']';
  } else if (obj.tag === 'Closure') {
    return 'Closure(' + obj.fid + ',' + obj.captured.map(absKey).join(',') + ')';
  }
  throw new Error('unknown heap obj tag');
}
// [task state.absToRExpr]
function absToRExpr(abs) {
  const { RE } = require('./contracts.js');
  if (abs.tag === 'Num') return RE.Num(abs.n);
  if (abs.tag === 'Str') return RE.Str(abs.s);
  if (abs.tag === 'Bool') return RE.Bool(abs.b);
  if (abs.tag === 'Undef') return RE.Undef();
  if (abs.tag === 'Null') return RE.Null();
  if (abs.tag === 'Dyn') return abs.expr;
  if (abs.tag === 'Ref') throw new Error('cannot convert Ref to RExpr');
  throw new Error('unknown Abs tag: ' + abs.tag);
}
// [task state.absTruthy]   returns true | false | null(decide-at-runtime)
function absTruthy(abs) {
  if (abs.tag === 'Num') return abs.n !== 0;
  if (abs.tag === 'Str') return abs.s !== '';
  if (abs.tag === 'Bool') return abs.b;
  if (abs.tag === 'Undef' || abs.tag === 'Null') return false;
  if (abs.tag === 'Ref') return true;
  if (abs.tag === 'Dyn') return null;
  throw new Error('unknown abs tag: ' + abs.tag);
}

// (frozen) structural deep clone of a State — shared by step/whistle.
function cloneState(s) {
  return {
    frames: s.frames.map((f) => ({ pc: f.pc, func: f.func, locals: f.locals.slice(), ostack: f.ostack.slice() })),
    heap: new Map(s.heap),
    nextAddr: s.nextAddr,
    pendingJoins: s.pendingJoins.map((x) => x.slice ? x.slice() : x),
    handlers: s.handlers.slice(),
    frozen: s.frozen, // shared by reference: immutable static heap addresses (set once at init)
  };
}

module.exports = { alloc, reachable, gc, weight, canonicalize, keyOf, absToRExpr, absTruthy, cloneState };
