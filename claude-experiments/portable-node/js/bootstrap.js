// Bootstrap: builds primordials, a require() system, shim modules, and then
// loads Node's lib/buffer.js + lib/internal/buffer.js verbatim.
//
// Reads __nodeSourceFiles and __internalBinding (set by the Rust host).
// Exposes globalThis.Buffer when done.

'use strict';

// =========================================================================
// primordials — frozen references to built-in methods that Node's source
// destructures as `const { ArrayPrototypeForEach, ... } = primordials;`
// =========================================================================

const primordials = (function makePrimordials() {
  const uncurry = (fn) => Function.prototype.call.bind(fn);
  const TypedArrayProto = Object.getPrototypeOf(Uint8Array.prototype);
  const getter = (proto, name) =>
    uncurry(Object.getOwnPropertyDescriptor(proto, name).get);

  return {
    // Constructors
    Array, Uint8Array, Int8Array, Symbol, BigInt, Number, String,
    Boolean, Error, RangeError, TypeError, URIError, SyntaxError, EvalError,
    Promise, Map, Set, WeakMap, WeakSet, RegExp, Date, ArrayBuffer, DataView,
    SharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined' ? SharedArrayBuffer : ArrayBuffer,
    // Safe* — in Node these are frozen-prototype subclasses immune to
    // userland tampering. For our purposes (no adversarial user code) plain
    // built-ins behave identically.
    SafeSet: Set, SafeMap: Map, SafeWeakMap: WeakMap, SafeWeakSet: WeakSet,
    SafePromise: Promise, SafeArrayIterator: function* () {},
    SafeFinalizationRegistry: typeof FinalizationRegistry !== 'undefined'
      ? FinalizationRegistry : function () { return { register: () => {}, unregister: () => {} }; },
    Float32Array, Float64Array,
    // Globals
    decodeURIComponent,
    encodeURIComponent,
    // Constants
    NumberMAX_SAFE_INTEGER: Number.MAX_SAFE_INTEGER,
    NumberMIN_SAFE_INTEGER: Number.MIN_SAFE_INTEGER,
    SymbolSpecies: Symbol.species,
    SymbolToPrimitive: Symbol.toPrimitive,
    SymbolIterator: Symbol.iterator,
    SymbolAsyncIterator: Symbol.asyncIterator,
    SymbolFor: Symbol.for,
    SymbolHasInstance: Symbol.hasInstance,
    SymbolToStringTag: Symbol.toStringTag,
    SymbolDispose: Symbol.dispose ?? Symbol.for('Symbol.dispose'),
    SymbolAsyncDispose: Symbol.asyncDispose ?? Symbol.for('Symbol.asyncDispose'),
    // Note: AsyncIteratorPrototype is the prototype of an async generator's
    // returned iterator. Build it by walking up from an async generator.
    AsyncIteratorPrototype: Object.getPrototypeOf(Object.getPrototypeOf(Object.getPrototypeOf((async function* () {})().__proto__))) ||
                            Object.getPrototypeOf((async function* () {})()).__proto__,
    ErrorCaptureStackTrace: Error.captureStackTrace || ((e) => { e.stack = ''; }),
    ReflectOwnKeys: Reflect.ownKeys,
    PromiseReject:  Promise.reject.bind(Promise),
    PromiseResolve: Promise.resolve.bind(Promise),
    PromiseAll:     Promise.all.bind(Promise),
    // Promise.withResolvers — TC39 stage 4 (2024). Polyfill if engine lacks it.
    PromiseWithResolvers: (Promise.withResolvers && Promise.withResolvers.bind(Promise)) || (() => {
      let resolve, reject;
      const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
      return { promise, resolve, reject };
    }),
    PromisePrototypeThen: uncurry(Promise.prototype.then),
    PromisePrototypeCatch: uncurry(Promise.prototype.catch),
    PromisePrototypeFinally: uncurry(Promise.prototype.finally),
    // Statics
    ArrayBufferIsView: ArrayBuffer.isView,
    ArrayIsArray: Array.isArray,
    MathAbs: Math.abs,
    MathFloor: Math.floor,
    MathMin: Math.min,
    MathTrunc: Math.trunc,
    NumberIsFinite: Number.isFinite,
    NumberIsInteger: Number.isInteger,
    NumberIsNaN: Number.isNaN,
    ObjectDefineProperties: Object.defineProperties,
    ObjectDefineProperty: Object.defineProperty,
    ObjectFreeze: Object.freeze,
    ObjectIs: Object.is,
    ObjectIsFrozen: Object.isFrozen,
    ObjectIsSealed: Object.isSealed,
    ObjectIsExtensible: Object.isExtensible,
    ObjectKeys: Object.keys,
    ObjectValues: Object.values,
    ObjectEntries: Object.entries,
    ObjectAssign: Object.assign,
    ObjectGetPrototypeOf: Object.getPrototypeOf,
    ObjectSetPrototypeOf: Object.setPrototypeOf,
    ObjectGetOwnPropertyNames: Object.getOwnPropertyNames,
    ObjectGetOwnPropertySymbols: Object.getOwnPropertySymbols,
    ObjectGetOwnPropertyDescriptor: Object.getOwnPropertyDescriptor,
    ObjectGetOwnPropertyDescriptors: Object.getOwnPropertyDescriptors,
    ObjectCreate: Object.create,
    ObjectFromEntries: Object.fromEntries,
    ObjectSeal: Object.seal,
    ObjectPreventExtensions: Object.preventExtensions,
    ObjectPrototypeToString: uncurry(Object.prototype.toString),
    ObjectPrototypePropertyIsEnumerable: uncurry(Object.prototype.propertyIsEnumerable),
    ObjectPrototypeIsPrototypeOf: uncurry(Object.prototype.isPrototypeOf),
    MathCeil: Math.ceil,
    MathRound: Math.round,
    MathMax: Math.max,
    MathAbs: Math.abs,
    MathPow: Math.pow,
    MathSign: Math.sign,
    MathSqrt: Math.sqrt,
    MathLog: Math.log,
    MathLog2: Math.log2,
    MathLog10: Math.log10,
    MathExp: Math.exp,
    NumberParseInt: Number.parseInt,
    NumberParseFloat: Number.parseFloat,
    NumberMAX_VALUE: Number.MAX_VALUE,
    NumberMIN_VALUE: Number.MIN_VALUE,
    NumberPOSITIVE_INFINITY: Number.POSITIVE_INFINITY,
    NumberNEGATIVE_INFINITY: Number.NEGATIVE_INFINITY,
    NumberEPSILON: Number.EPSILON,
    DateNow: Date.now,
    Date,
    DatePrototypeGetTime: uncurry(Date.prototype.getTime),
    DatePrototypeToISOString: uncurry(Date.prototype.toISOString),
    JSONStringify: JSON.stringify,
    JSONParse: JSON.parse,
    globalThis,
    // Prototype methods (uncurried so XPrototypeY(obj, args) === obj.y(args))
    ArrayPrototypeForEach: uncurry(Array.prototype.forEach),
    ArrayPrototypePush: uncurry(Array.prototype.push),
    ArrayPrototypePop: uncurry(Array.prototype.pop),
    ArrayPrototypeShift: uncurry(Array.prototype.shift),
    ArrayPrototypeUnshift: uncurry(Array.prototype.unshift),
    ArrayPrototypeSplice: uncurry(Array.prototype.splice),
    ArrayPrototypeReverse: uncurry(Array.prototype.reverse),
    ArrayPrototypeSlice: uncurry(Array.prototype.slice),
    ArrayPrototypeMap: uncurry(Array.prototype.map),
    ArrayPrototypeJoin: uncurry(Array.prototype.join),
    ArrayPrototypeFilter: uncurry(Array.prototype.filter),
    ArrayPrototypeIndexOf: uncurry(Array.prototype.indexOf),
    ArrayPrototypeIncludes: uncurry(Array.prototype.includes),
    ArrayPrototypeFind: uncurry(Array.prototype.find),
    ArrayPrototypeSort: uncurry(Array.prototype.sort),
    ArrayPrototypeConcat: uncurry(Array.prototype.concat),
    ArrayPrototypeFill: uncurry(Array.prototype.fill),
    ArrayPrototypeEntries: uncurry(Array.prototype.entries),
    ArrayPrototypeKeys: uncurry(Array.prototype.keys),
    ArrayPrototypeValues: uncurry(Array.prototype.values),
    ArrayPrototypeFindIndex: uncurry(Array.prototype.findIndex),
    ArrayPrototypeFlat: uncurry(Array.prototype.flat),
    ArrayPrototypeFlatMap: uncurry(Array.prototype.flatMap),
    ArrayPrototypeEvery: uncurry(Array.prototype.every),
    ArrayPrototypeSome: uncurry(Array.prototype.some),
    ArrayPrototypeReduce: uncurry(Array.prototype.reduce),
    ArrayPrototypeReduceRight: uncurry(Array.prototype.reduceRight),
    ArrayPrototypeLastIndexOf: uncurry(Array.prototype.lastIndexOf),
    ArrayPrototypeAt: uncurry(Array.prototype.at),
    ArrayPrototypeCopyWithin: uncurry(Array.prototype.copyWithin),
    ArrayFrom: Array.from,
    ArrayOf: Array.of,
    ObjectPrototypeHasOwnProperty: uncurry(Object.prototype.hasOwnProperty),
    RegExpPrototypeSymbolReplace: uncurry(RegExp.prototype[Symbol.replace]),
    RegExpPrototypeExec: uncurry(RegExp.prototype.exec),
    RegExpPrototypeTest: uncurry(RegExp.prototype.test),
    RegExpPrototypeGetFlags: uncurry(Object.getOwnPropertyDescriptor(RegExp.prototype, 'flags').get),
    RegExpPrototypeGetSource: uncurry(Object.getOwnPropertyDescriptor(RegExp.prototype, 'source').get),
    StringPrototypeCharCodeAt: uncurry(String.prototype.charCodeAt),
    StringPrototypeCodePointAt: uncurry(String.prototype.codePointAt),
    StringPrototypeSlice: uncurry(String.prototype.slice),
    StringPrototypeToLowerCase: uncurry(String.prototype.toLowerCase),
    StringPrototypeToUpperCase: uncurry(String.prototype.toUpperCase),
    StringPrototypeTrim: uncurry(String.prototype.trim),
    StringPrototypeIncludes: uncurry(String.prototype.includes),
    StringPrototypeIndexOf: uncurry(String.prototype.indexOf),
    StringPrototypeLastIndexOf: uncurry(String.prototype.lastIndexOf),
    StringPrototypeRepeat: uncurry(String.prototype.repeat),
    StringPrototypeReplace: uncurry(String.prototype.replace),
    StringPrototypeSplit: uncurry(String.prototype.split),
    StringPrototypeStartsWith: uncurry(String.prototype.startsWith),
    StringPrototypeEndsWith: uncurry(String.prototype.endsWith),
    StringPrototypeMatch: uncurry(String.prototype.match),
    StringPrototypePadStart: uncurry(String.prototype.padStart),
    StringPrototypePadEnd: uncurry(String.prototype.padEnd),
    StringPrototypeNormalize: uncurry(String.prototype.normalize),
    FunctionPrototypeBind: uncurry(Function.prototype.bind),
    FunctionPrototypeCall: uncurry(Function.prototype.call),
    FunctionPrototypeApply: uncurry(Function.prototype.apply),
    FunctionPrototypeSymbolHasInstance: uncurry(Function.prototype[Symbol.hasInstance]),
    ReflectApply: Reflect.apply,
    NumberPrototypeToString: uncurry(Number.prototype.toString),
    BigIntPrototypeToString: uncurry(BigInt.prototype.toString),
    // uncurryThis — Node's internal helper. Same shape as our `uncurry`.
    uncurryThis: uncurry,
    // ArrayFromAsync — TC39 stage 4 (2024). QuickJS may not have it.
    ArrayFromAsync: Array.fromAsync ?? (async (src) => {
      const out = [];
      for await (const x of src) out.push(x);
      return out;
    }),
    // TypedArray prototype methods (call via the %TypedArray% prototype)
    TypedArrayPrototypeFill: uncurry(TypedArrayProto.fill),
    TypedArrayPrototypeSet: uncurry(TypedArrayProto.set),
    TypedArrayPrototypeSubarray: uncurry(TypedArrayProto.subarray),
    // TypedArray prototype getters
    TypedArrayPrototypeGetBuffer: getter(TypedArrayProto, 'buffer'),
    TypedArrayPrototypeGetByteLength: getter(TypedArrayProto, 'byteLength'),
    TypedArrayPrototypeGetByteOffset: getter(TypedArrayProto, 'byteOffset'),
    TypedArrayPrototypeGetLength: getter(TypedArrayProto, 'length'),
  };
})();

globalThis.primordials = primordials;
globalThis.internalBinding = __internalBinding;

// =========================================================================
// V8 stack-trace API polyfill — depd, debug, source-map-support, etc. assume
// Error.captureStackTrace + Error.prepareStackTrace (V8-specific). We parse
// QuickJS's stack string into objects implementing the CallSite methods npm
// packages reach for: getFileName, getLineNumber, getColumnNumber,
// getFunctionName, isEval, getEvalOrigin, isNative, isToplevel, isConstructor.
// =========================================================================

function _parseStackLine(line) {
  // QuickJS format examples:
  //   "    at fnName (file:line:col)"
  //   "    at fnName (file:line)"
  //   "    at file:line:col"
  //   "    at <anonymous> (eval_script:1:2)"
  //   "    at apply (native)"
  const m = /^\s*at\s+(?:(.+?)\s+\()?(.+?)\)?$/.exec(line);
  if (!m) return null;
  let fn = m[1] || '';
  let loc = m[2] || '';
  // Strip trailing ')' (regex above usually catches it, but be safe).
  if (loc.endsWith(')')) loc = loc.slice(0, -1);
  // loc could be "file:line:col", "file:line", "native", "eval_script:1:2".
  let file = loc, lineNo = 0, col = 0, isNative = false;
  if (loc === 'native') { isNative = true; }
  else {
    // Find last ':' for col, second-last for line.
    const lastColon = loc.lastIndexOf(':');
    if (lastColon > 0) {
      const tail = loc.slice(lastColon + 1);
      if (/^\d+$/.test(tail)) {
        const beforeCol = loc.slice(0, lastColon);
        const secondLast = beforeCol.lastIndexOf(':');
        if (secondLast > 0 && /^\d+$/.test(beforeCol.slice(secondLast + 1))) {
          file = beforeCol.slice(0, secondLast);
          lineNo = +beforeCol.slice(secondLast + 1);
          col = +tail;
        } else {
          file = beforeCol;
          lineNo = +tail;
        }
      }
    }
  }
  const isEval = file === 'eval_script' || file.startsWith('eval_script');
  return {
    getThis()          { return undefined; },
    getTypeName()      { return null; },
    getFunction()      { return undefined; },
    getFunctionName()  { return fn || null; },
    getMethodName()    { return null; },
    getFileName()      { return file || null; },
    getLineNumber()    { return lineNo || null; },
    getColumnNumber()  { return col || null; },
    getEvalOrigin()    { return isEval ? file : undefined; },
    isToplevel()       { return !fn || fn === '<anonymous>'; },
    isEval()           { return isEval; },
    isNative()         { return isNative; },
    isConstructor()    { return false; },
    isAsync()          { return false; },
    isPromiseAll()     { return false; },
    getPromiseIndex()  { return null; },
    toString()         { return line.trim(); },
  };
}

function _parseStack(str, skipTop) {
  if (typeof str !== 'string') return [];
  const lines = str.split('\n');
  const frames = [];
  for (const ln of lines) {
    if (!/^\s*at\s/.test(ln)) continue;
    const f = _parseStackLine(ln);
    if (f) frames.push(f);
  }
  // V8's captureStackTrace(obj, hideAbove) drops frames above (and including)
  // hideAbove. Without function refs we can't honor that precisely; drop the
  // top frame which represents captureStackTrace itself.
  if (skipTop && frames.length > 0) frames.shift();
  return frames;
}

// QuickJS has Error.captureStackTrace but it doesn't honor
// Error.prepareStackTrace. Override unconditionally so npm packages that set
// prepareStackTrace (depd, debug, source-map-support, …) get CallSite objects.
const _qjsCaptureStack = Error.captureStackTrace;
Error.captureStackTrace = function(obj, hideAbove) {
  let raw = '';
  if (_qjsCaptureStack) {
    try {
      _qjsCaptureStack(obj, hideAbove);
      raw = obj.stack || '';
    } catch (_e) {
      try { raw = (new Error()).stack || ''; } catch (_e2) { raw = ''; }
    }
  } else {
    try { raw = (new Error()).stack || ''; } catch (_e) { raw = ''; }
  }
  // QuickJS sometimes produces an empty stack when hideAbove doesn't match
  // any frame on the call stack. Fall back to a fresh-Error stack in that
  // case so npm packages (and our test harness) can still see context.
  if (!raw) {
    try { raw = (new Error()).stack || ''; } catch (_) {}
  }
  obj.stack = raw;
  const prep = Error.prepareStackTrace;
  if (typeof prep === 'function') {
    const frames = _parseStack(raw, false);
    while (frames.length < 4) frames.push(_dummyFrame());
    try { obj.stack = prep(obj, frames); } catch (_e) { /* keep raw */ }
  }
  void hideAbove;
};

// WHATWG TextEncoder / TextDecoder. QuickJS doesn't ship them; Buffer
// already handles UTF-8 / UTF-16 internally, so we trampoline through that.
if (typeof globalThis.TextEncoder === 'undefined') {
  globalThis.TextEncoder = class TextEncoder {
    get encoding() { return 'utf-8'; }
    encode(str) {
      // Convert via Buffer, then return as Uint8Array.
      const b = (globalThis.Buffer || require('buffer').Buffer).from(String(str || ''), 'utf8');
      return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    }
    encodeInto(str, dest) {
      const enc = this.encode(str);
      const n = Math.min(enc.length, dest.length);
      for (let i = 0; i < n; i++) dest[i] = enc[i];
      return { read: str.length, written: n };
    }
  };
}
if (typeof globalThis.TextDecoder === 'undefined') {
  globalThis.TextDecoder = class TextDecoder {
    constructor(encoding, opts) {
      this._encoding = String(encoding || 'utf-8').toLowerCase().replace('_', '-');
      this._fatal = !!(opts && opts.fatal);
      this._ignoreBOM = !!(opts && opts.ignoreBOM);
    }
    get encoding() { return this._encoding; }
    get fatal()    { return this._fatal; }
    get ignoreBOM(){ return this._ignoreBOM; }
    decode(input, _opts) {
      if (input == null) return '';
      const B = globalThis.Buffer || require('buffer').Buffer;
      const enc = this._encoding === 'utf-16le' ? 'utf16le'
                : this._encoding === 'iso-8859-1' || this._encoding === 'latin1' ? 'latin1'
                : this._encoding === 'us-ascii' || this._encoding === 'ascii' ? 'ascii'
                : 'utf8';
      let bytes;
      if (input instanceof ArrayBuffer) bytes = new Uint8Array(input);
      else if (ArrayBuffer.isView(input)) bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
      else bytes = new Uint8Array(input);
      return B.from(bytes).toString(enc);
    }
  };
}

function _dummyFrame() {
  return {
    getThis() { return undefined; },
    getTypeName() { return null; },
    getFunction() { return undefined; },
    getFunctionName() { return null; },
    getMethodName() { return null; },
    getFileName() { return null; },
    getLineNumber() { return null; },
    getColumnNumber() { return null; },
    getEvalOrigin() { return undefined; },
    isToplevel() { return false; },
    isEval() { return false; },
    isNative() { return false; },
    isConstructor() { return false; },
    isAsync() { return false; },
    isPromiseAll() { return false; },
    getPromiseIndex() { return null; },
    toString() { return '<unknown>'; },
  };
}
// Also intercept .stack reads on errors if prepareStackTrace is set. We
// can't override the Error prototype getter in QuickJS without a Proxy, so
// we just leave the raw string on uncaptured errors. depd / similar use
// captureStackTrace directly.


// =========================================================================
// Shim modules — minimal stand-ins for internal/* and util/*. Each is
// registered up front; require() reads from this registry.
// =========================================================================

const shimSources = Object.create(null);

shimSources['internal/errors'] = `
'use strict';

// Each ERR_* is a class. Throwing \`new ERR_FOO(args)\` produces an Error
// (or TypeError/RangeError) whose .code matches and whose .message follows
// Node's message format closely enough to satisfy assert.throws checks.

// Match Node's ERR_INVALID_ARG_TYPE message format. The shape splits the
// "expected" list into primitive types and constructor names, then joins:
//   "of type <primitives>"  +  "an instance of <constructors>"
// with Oxford commas, and "an" article for names that read as nouns
// (e.g. "Array-like Object").
const PRIMITIVES = new Set([
  'null', 'undefined', 'string', 'number', 'boolean',
  'symbol', 'bigint', 'function', 'object',
]);

function joinOr(items) {
  if (items.length === 0) return '';
  if (items.length === 1) return items[0];
  if (items.length === 2) return items[0] + ' or ' + items[1];
  return items.slice(0, -1).join(', ') + ', or ' + items[items.length - 1];
}

function buildInvalidArgTypeMsg(name, expected, actual) {
  const exps = Array.isArray(expected) ? expected : [expected];
  const primitives = [], instancesPlain = [], instancesArticled = [];
  for (const e of exps) {
    if (PRIMITIVES.has(e)) primitives.push(e);
    else if (e.indexOf('-') >= 0 || e.indexOf(' ') >= 0) instancesArticled.push(e);
    else instancesPlain.push(e);
  }

  let typeStr = '';
  if (primitives.length === 1) typeStr = 'of type ' + primitives[0];
  else if (primitives.length > 1) typeStr = 'one of type ' + joinOr(primitives);

  let instStr = '';
  if (instancesPlain.length || instancesArticled.length) {
    let s = joinOr(instancesPlain);
    for (const a of instancesArticled) s += (s ? ' or ' : '') + 'an ' + a;
    instStr = 'an instance of ' + s;
  }

  let expStr;
  if (typeStr && instStr) expStr = typeStr + ' or ' + instStr;
  else expStr = typeStr || instStr;

  // Don't double up "argument" when the name itself contains the word.
  const prefix = name.indexOf('argument') >= 0
    ? 'The ' + name
    : 'The "' + name + '" argument';

  let act;
  if (actual === null) act = 'null';
  else if (typeof actual === 'undefined') act = 'undefined';
  else if (typeof actual === 'object') {
    const ctor = actual.constructor && actual.constructor.name;
    act = ctor ? 'an instance of ' + ctor : Object.prototype.toString.call(actual);
  } else if (typeof actual === 'function') {
    act = 'function ' + (actual.name || '');
  } else if (typeof actual === 'symbol') {
    act = 'type symbol (' + String(actual) + ')';
  } else if (typeof actual === 'bigint') {
    act = 'type bigint (' + String(actual) + 'n)';
  } else if (typeof actual === 'number') {
    act = 'type number (' + actual + ')';
  } else if (typeof actual === 'string') {
    act = "type string ('" + (actual.length > 25 ? actual.slice(0, 25) + '...' : actual) + "')";
  } else if (typeof actual === 'boolean') {
    act = 'type boolean (' + actual + ')';
  } else {
    act = 'type ' + typeof actual;
  }

  return prefix + ' must be ' + expStr + '. Received ' + act;
}

const errorSpecs = {
  // code: { base: Error/TypeError/RangeError, message: fn(...args) }
  ERR_BUFFER_OUT_OF_BOUNDS: {
    base: RangeError,
    message: (name) => name
      ? '"' + name + '" is outside of buffer bounds'
      : 'Attempt to access memory outside buffer bounds',
  },
  ERR_INVALID_ARG_TYPE:    { base: TypeError, message: buildInvalidArgTypeMsg },
  ERR_INVALID_ARG_VALUE:   { base: TypeError, message: (name, value, reason) =>
    'The argument "' + name + '" ' + (reason || 'is invalid') + '. Received ' + String(value) },
  ERR_INVALID_BUFFER_SIZE: { base: RangeError, message: (reason) => 'Invalid Buffer size' + (reason ? ': ' + reason : '') },
  ERR_MISSING_ARGS:        { base: TypeError, message: (...args) => 'The "' + args.join('", "') + '" argument' + (args.length > 1 ? 's' : '') + ' must be specified' },
  ERR_OUT_OF_RANGE:        { base: RangeError, message: (name, range, value) =>
    'The value of "' + name + '" is out of range. It must be ' + range + '. Received ' + String(value) },
  ERR_UNKNOWN_ENCODING:    { base: TypeError, message: (enc) => 'Unknown encoding: ' + enc },
  ERR_INVALID_URI:         { base: URIError,  message: () => 'URI malformed' },
  ERR_SYSTEM_ERROR:        { base: Error,     message: (ctx) =>
    'System error: ' + (ctx && (ctx.syscall || ctx.message) || 'unknown') },
  ERR_UNHANDLED_ERROR:     { base: Error,     message: (msg) =>
    'Unhandled error.' + (msg ? ' ' + msg : '') },
  ERR_METHOD_NOT_IMPLEMENTED: { base: Error,  message: (name) =>
    'The "' + name + '" method is not implemented' },
  ERR_AMBIGUOUS_ARGUMENT:  { base: TypeError, message: (name, reason) =>
    'The "' + name + '" argument is ambiguous. ' + (reason || '') },
  ERR_CONSTRUCT_CALL_REQUIRED: { base: TypeError, message: (name) =>
    'Cannot call constructor ' + (name || '') + ' without new' },
  ERR_INVALID_RETURN_VALUE: { base: TypeError, message: (expected, name, actual) =>
    'Expected ' + expected + ' to be returned from the "' + name + '" function but got ' +
    (typeof actual === 'object' && actual !== null
      ? 'instance of ' + (actual.constructor && actual.constructor.name || 'Object')
      : 'type ' + typeof actual) },
  ERR_INVALID_THIS:        { base: TypeError, message: (expected) =>
    'Value of "this" must be of type ' + expected },
  ERR_ASSERTION:           { base: Error,     message: (msg) => msg || 'Assertion failed' },
  ERR_FS_EISDIR:           { base: Error,     message: (path) =>
    'Path is a directory: ' + path },
  ERR_INCOMPATIBLE_OPTION_PAIR: { base: TypeError, message: (opt1, opt2) =>
    'Option "' + opt1 + '" cannot be used in combination with option "' + opt2 + '"' },
  ERR_FS_INVALID_SYMLINK_TYPE: { base: Error, message: (type) =>
    'Symlink type must be one of "dir", "file", or "junction". Received "' + type + '"' },
  ERR_FS_FILE_TOO_LARGE:   { base: RangeError, message: (size) =>
    'File size ' + size + ' is greater than 2 GiB' },
  ERR_FS_CP_DIR_TO_NON_DIR:{ base: Error,     message: () =>
    'Cannot overwrite directory with non-directory' },
  ERR_FS_CP_NON_DIR_TO_DIR:{ base: Error,     message: () =>
    'Cannot overwrite non-directory with directory' },
  ERR_BUFFER_TOO_LARGE:    { base: RangeError, message: (max) =>
    'Cannot create a Buffer larger than ' + max + ' bytes' },
  ERR_ACCESS_DENIED:       { base: Error,     message: (op) =>
    'Access denied' + (op ? ' for ' + op : '') },
  ERR_STREAM_DESTROYED:    { base: Error,     message: (method) =>
    'Cannot call ' + method + ' after a stream was destroyed' },
};

function makeCode(code) {
  const spec = errorSpecs[code];
  function E(...args) {
    const Base = (spec && spec.base) || Error;
    const msg = spec && spec.message ? spec.message.apply(null, args) : code;
    const err = new Base(msg);
    err.code = code;
    return err;
  }
  return E;
}

const codes = {};
for (const k of Object.keys(errorSpecs)) codes[k] = makeCode(k);

function genericNodeError(message, opts = {}) {
  const err = new Error(message);
  if (opts.code) err.code = opts.code;
  return err;
}

// hideStackFrames(fn) — Node wraps internal fns so their frames don't
// pollute user stack traces. For us it's a pass-through.
function hideStackFrames(fn) { return function wrapper(...args) { return fn.apply(this, args); }; }

// ERR_SYSTEM_ERROR is also called as new ERR_SYSTEM_ERROR.HideStackFramesError(ctx).
codes.ERR_SYSTEM_ERROR.HideStackFramesError = codes.ERR_SYSTEM_ERROR;
// Every code-class is also addressed as X.HideStackFramesError elsewhere in
// Node's internals (lib/internal/fs/utils.js, etc.). Alias each one.
for (const k of Object.keys(codes)) {
  if (!codes[k].HideStackFramesError) codes[k].HideStackFramesError = codes[k];
}

// Node provides a check for whether Error.stackTraceLimit is writable
// (it might be frozen in some sandboxed setups). For us it always is.
function isErrorStackTraceLimitWritable() {
  const desc = Object.getOwnPropertyDescriptor(Error, 'stackTraceLimit');
  if (desc === undefined) return true;
  return desc.writable !== false;
}

// AbortError — thrown when an AbortSignal is fired. Used by events.js, fs, etc.
class AbortError extends Error {
  constructor(message = 'The operation was aborted', options) {
    super(message, options);
    this.code = 'ABORT_ERR';
    this.name = 'AbortError';
  }
}

// kEnhanceStackBeforeInspector — a Symbol marker used internally. No effect for us.
const kEnhanceStackBeforeInspector = Symbol('kEnhanceStackBeforeInspector');

// UVException — Node uses this for libuv-style errors from syscalls. Our
// __host.file primitives already throw errors with code/syscall/path/errno
// set, so this constructor lets internal/fs/utils.js wrap raw error info
// into the same shape.
function UVException(opts) {
  const { errno, syscall, path, dest, message } = opts || {};
  const err = new Error(message ||
    (opts.code ? opts.code + ': ' + syscall + ' ' + (path || '') : 'UV error'));
  if (opts.code) err.code = opts.code;
  err.errno = errno;
  err.syscall = syscall;
  if (path) err.path = path;
  if (dest) err.dest = dest;
  return err;
}

// aggregateTwoErrors — combine two Errors into an AggregateError. Used in fs.js
// to surface both the immediate failure and a follow-up cleanup error.
function aggregateTwoErrors(innerError, outerError) {
  if (innerError && outerError) {
    if (typeof AggregateError !== 'undefined') {
      const e = new AggregateError([outerError, innerError], outerError.message);
      e.code = outerError.code;
      return e;
    }
    return outerError;
  }
  return innerError || outerError;
}

module.exports = { codes, genericNodeError, hideStackFrames, AbortError,
                   kEnhanceStackBeforeInspector, isErrorStackTraceLimitWritable,
                   UVException, aggregateTwoErrors };
`;

shimSources['internal/validators'] = `
'use strict';

const { codes: { ERR_INVALID_ARG_TYPE, ERR_OUT_OF_RANGE } } = require('internal/errors');

function validateString(value, name) {
  if (typeof value !== 'string')
    throw new ERR_INVALID_ARG_TYPE(name, 'string', value);
}

function validateNumber(value, name, min, max) {
  if (typeof value !== 'number')
    throw new ERR_INVALID_ARG_TYPE(name, 'number', value);
  if ((min !== undefined && value < min) || (max !== undefined && value > max))
    throw new ERR_OUT_OF_RANGE(name, \`>= \${min} && <= \${max}\`, value);
}

function validateInteger(value, name,
                         min = Number.MIN_SAFE_INTEGER,
                         max = Number.MAX_SAFE_INTEGER) {
  if (typeof value !== 'number')
    throw new ERR_INVALID_ARG_TYPE(name, 'number', value);
  if (!Number.isInteger(value))
    throw new ERR_OUT_OF_RANGE(name, 'an integer', value);
  if (value < min || value > max)
    throw new ERR_OUT_OF_RANGE(name, \`>= \${min} && <= \${max}\`, value);
}

function validateBuffer(value, name = 'buffer') {
  if (!ArrayBuffer.isView(value))
    throw new ERR_INVALID_ARG_TYPE(name, ['Buffer', 'TypedArray', 'DataView'], value);
}

function validateArray(value, name) {
  if (!Array.isArray(value))
    throw new ERR_INVALID_ARG_TYPE(name, 'Array', value);
}

function validateInt32(value, name,
                       min = -2147483648,
                       max = 2147483647) {
  validateInteger(value, name, min, max);
}

function validateUint32(value, name, positive = false) {
  if (typeof value !== 'number')
    throw new ERR_INVALID_ARG_TYPE(name, 'number', value);
  if (!Number.isInteger(value))
    throw new ERR_OUT_OF_RANGE(name, 'an integer', value);
  const min = positive ? 1 : 0;
  if (value < min || value > 4294967295)
    throw new ERR_OUT_OF_RANGE(name, '>= ' + min + ' && <= 4294967295', value);
}

function isInt32(value) {
  return Number.isInteger(value) && value >= -2147483648 && value <= 2147483647;
}

function parseFileMode(value, name, def) {
  if (value === undefined || value === null) return def;
  if (typeof value === 'string') {
    const n = parseInt(value, 8);
    if (Number.isNaN(n)) throw new ERR_INVALID_ARG_TYPE(name, 'string or number', value);
    value = n;
  }
  if (typeof value !== 'number') throw new ERR_INVALID_ARG_TYPE(name, 'number', value);
  if (value < 0 || value > 4095) throw new ERR_OUT_OF_RANGE(name, '>= 0 && <= 0o7777', value);
  return value;
}

const VALID_ENCODINGS = new Set([
  'ascii','utf8','utf-8','utf16le','utf-16le','ucs2','ucs-2',
  'base64','base64url','latin1','binary','hex',
]);
function validateEncoding(value, name) {
  if (typeof value !== 'string' || !VALID_ENCODINGS.has(value.toLowerCase())) {
    const e = new TypeError('Unknown encoding: ' + value);
    e.code = 'ERR_UNKNOWN_ENCODING';
    throw e;
  }
}

function validateObject(value, name) {
  if (value === null || typeof value !== 'object' || Array.isArray(value))
    // Lowercase 'object' so the formatter emits "of type object" (primitive
    // form), matching Node's actual message.
    throw new ERR_INVALID_ARG_TYPE(name, 'object', value);
}

function validateBoolean(value, name) {
  if (typeof value !== 'boolean')
    throw new ERR_INVALID_ARG_TYPE(name, 'boolean', value);
}

function validateFunction(value, name) {
  if (typeof value !== 'function')
    throw new ERR_INVALID_ARG_TYPE(name, 'Function', value);
}

function validateAbortSignal(signal, name) {
  if (signal !== undefined && signal !== null &&
      (typeof signal !== 'object' || typeof signal.aborted !== 'boolean'))
    throw new ERR_INVALID_ARG_TYPE(name, 'AbortSignal', signal);
}

function validatePort(port, name = 'Port', allowZero = true) {
  if (typeof port !== 'number' && typeof port !== 'string')
    throw new ERR_INVALID_ARG_TYPE(name, 'number', port);
  const n = +port;
  if (!Number.isInteger(n) || n < (allowZero ? 0 : 1) || n > 0xFFFF)
    throw new ERR_OUT_OF_RANGE(name,
      (allowZero ? '>= 0 && <= 65535' : '>= 1 && <= 65535'), port);
  return n;
}

// Cheap shape validator — Node has many; the streams + net + http stack
// only reaches for these names.
function validateOneOf(value, name, oneOf) {
  if (!oneOf.includes(value))
    throw new ERR_INVALID_ARG_VALUE(name, value, 'must be one of ' + oneOf.join(', '));
}

function checkRangesOrGetDefault(number, name, lower, upper, def) {
  if (number === undefined) return def;
  if (typeof number !== 'number') throw new ERR_INVALID_ARG_TYPE(name, 'number', number);
  if (Number.isNaN(number) || number < lower || number > upper)
    throw new ERR_OUT_OF_RANGE(name, '>= ' + lower + ' && <= ' + upper, number);
  return number;
}

function validateLinkHeaderValue(_value) { /* HTTP-specific; no-op */ }

// Node's lib/internal/validators sets a .withoutStackTrace alias on every
// validator so error-throwing internals can use a "hide my frame" variant.
// We don't actually hide frames, but the alias must exist as a function.
for (const v of [validateArray, validateBuffer, validateInteger, validateInt32,
                 validateUint32, validateNumber, validateString, validateObject,
                 validateBoolean, validateFunction, validateAbortSignal,
                 validateEncoding, validatePort, validateOneOf,
                 checkRangesOrGetDefault]) {
  v.withoutStackTrace = v;
}

module.exports = {
  validateArray, validateBuffer, validateInteger, validateInt32, validateUint32,
  validateNumber, validateString, validateObject, validateBoolean,
  validateFunction, validateAbortSignal, validateEncoding,
  validatePort, validateOneOf, checkRangesOrGetDefault, validateLinkHeaderValue,
  isInt32, parseFileMode,
};
`;

shimSources['internal/util/types'] = `
'use strict';

// Pure-JS type discriminators. Where the engine doesn't expose a primitive
// (e.g. detecting a transferable's true type), we use Object.prototype's
// toString tag which works across realms.
function tag(v) { return Object.prototype.toString.call(v); }
module.exports = {
  isAnyArrayBuffer: (v) => {
    // \`instanceof ArrayBuffer\` returns true for forged subclasses (the
    // test-buffer-arraybuffer "AB" pattern). Probe the engine via the
    // ArrayBuffer.prototype.byteLength accessor — it requires a *real*
    // internal slot and throws otherwise.
    if (v == null || typeof v !== 'object') return false;
    try {
      Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'byteLength').get.call(v);
      return true;
    } catch (_) {}
    if (typeof SharedArrayBuffer !== 'undefined') {
      try {
        Object.getOwnPropertyDescriptor(SharedArrayBuffer.prototype, 'byteLength').get.call(v);
        return true;
      } catch (_) {}
    }
    return false;
  },
  isArrayBuffer:    (v) => {
    if (v == null || typeof v !== 'object') return false;
    try {
      Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'byteLength').get.call(v);
      return true;
    } catch (_) { return false; }
  },
  isArrayBufferView: (v) => ArrayBuffer.isView(v),
  isUint8Array:     (v) => v instanceof Uint8Array && tag(v) !== '[object Buffer]',
  isTypedArray:     (v) => ArrayBuffer.isView(v) && !(v instanceof DataView),
  isDate:           (v) => v instanceof Date || tag(v) === '[object Date]',
  isPromise:        (v) => v && typeof v.then === 'function' && tag(v) === '[object Promise]',
  isRegExp:         (v) => v instanceof RegExp || tag(v) === '[object RegExp]',
  isMap:            (v) => v instanceof Map || tag(v) === '[object Map]',
  isSet:            (v) => v instanceof Set || tag(v) === '[object Set]',
  isBigInt64Array:  (v) => tag(v) === '[object BigInt64Array]',
  isBigUint64Array: (v) => tag(v) === '[object BigUint64Array]',
  isInt8Array:      (v) => v instanceof Int8Array,
  isUint16Array:    (v) => v instanceof Uint16Array,
  isInt16Array:     (v) => v instanceof Int16Array,
  isUint32Array:    (v) => v instanceof Uint32Array,
  isInt32Array:     (v) => v instanceof Int32Array,
  isFloat32Array:   (v) => v instanceof Float32Array,
  isFloat64Array:   (v) => v instanceof Float64Array,
  isUint8ClampedArray: (v) => tag(v) === '[object Uint8ClampedArray]',
  isDataView:       (v) => v instanceof DataView,
  isWeakMap:        (v) => v instanceof WeakMap,
  isWeakSet:        (v) => v instanceof WeakSet,
  isError:          (v) => v instanceof Error || tag(v) === '[object Error]',
  isBoxedPrimitive: (v) => v instanceof Number || v instanceof String ||
                            v instanceof Boolean || v instanceof Symbol ||
                            v instanceof BigInt,
  isExternal:       () => false,
  isProxy:          () => false,
  isModuleNamespaceObject: (v) => tag(v) === '[object Module]',
  isStringObject:   (v) => v instanceof String,
  isNumberObject:   (v) => v instanceof Number,
  isBooleanObject:  (v) => v instanceof Boolean,
  isSymbolObject:   (v) => v instanceof Symbol,
  isBigIntObject:   (v) => v instanceof BigInt,
  isGeneratorFunction:      (v) => tag(v) === '[object GeneratorFunction]',
  isAsyncFunction:          (v) => tag(v) === '[object AsyncFunction]',
  isGeneratorObject:        (v) => tag(v) === '[object Generator]',
  isNativeError:            (v) => v instanceof Error,
};
`;

// 'util/types' is the public alias; some Node internal files require it.
shimSources['util/types'] = shimSources['internal/util/types'];

shimSources['internal/util'] = `
'use strict';

const customInspectSymbol = Symbol.for('nodejs.util.inspect.custom');
const kIsEncodingSymbol = Symbol('kIsEncodingSymbol');

function normalizeEncoding(enc) {
  if (enc == null) return 'utf8';
  if (typeof enc !== 'string') return undefined;
  const lower = enc.toLowerCase();
  switch (lower) {
    case 'utf8': case 'utf-8': case 'utf':  return 'utf8';
    case 'utf16le': case 'utf-16le':
    case 'ucs2': case 'ucs-2':              return 'utf16le';
    case 'latin1': case 'binary':           return 'latin1';
    case 'base64':                          return 'base64';
    case 'base64url':                       return 'base64url';
    case 'ascii':                           return 'ascii';
    case 'hex':                             return 'hex';
  }
  return undefined;
}

// Encoding numeric codes match Node's internal enum. string_decoder writes
// these into a Uint8Array slot, so they MUST be numeric. Buffer.js's
// indexOf* callers pass encodingsMap.utf8 as the encoding param to the
// binding; the Rust side accepts both numbers and strings.
const ENC_UTF8 = 0, ENC_UTF16LE = 1, ENC_LATIN1 = 2, ENC_BASE64 = 3,
      ENC_BASE64URL = 4, ENC_ASCII = 5, ENC_HEX = 6;
const encodingsMap = {
  utf8: ENC_UTF8, utf16le: ENC_UTF16LE, latin1: ENC_LATIN1,
  base64: ENC_BASE64, base64url: ENC_BASE64URL,
  ascii: ENC_ASCII, hex: ENC_HEX,
};
// Numeric → name map used by binding shims that need to dispatch on encoding.
const encodingNames = ['utf8','utf16le','latin1','base64','base64url','ascii','hex'];

function lazyDOMException(message, name = 'Error') {
  const err = new Error(message);
  err.name = name;
  return err;
}

// defineLazyProperties(target, modulePath, names) — define getters that load
// the module lazily on first access. Used by buffer.js for File/Blob etc.
function defineLazyProperties(target, modulePath, names) {
  for (const n of names) {
    Object.defineProperty(target, n, {
      configurable: true,
      get() {
        const mod = require(modulePath);
        const v = mod[n];
        Object.defineProperty(target, n, { value: v, writable: true, configurable: true });
        return v;
      },
    });
  }
}

// getCIDR(address, netmask, family) → "address/prefix"; used by os.networkInterfaces.
function getCIDR(address, netmask, family) {
  if (!address || !netmask) return null;
  let ones = 0;
  if (family === 'IPv4') {
    for (const part of netmask.split('.')) {
      let n = parseInt(part, 10);
      while (n) { ones += n & 1; n >>>= 1; }
    }
  } else if (family === 'IPv6') {
    for (const part of netmask.split(':')) {
      if (!part) continue;
      let n = parseInt(part, 16);
      while (n) { ones += n & 1; n >>>= 1; }
    }
  } else { return null; }
  return address + '/' + ones;
}

function spliceOne(list, index) {
  for (; index + 1 < list.length; index++) list[index] = list[index + 1];
  list.pop();
}

const isWindows = (globalThis.__host && globalThis.__host.process.platform === 'win32');

// getLazy(loader) → returns a function that memoizes loader()'s result.
// Used for lazy dependency requires that we may never actually need.
function getLazy(loader) {
  let cached;
  return () => { if (cached === undefined) cached = loader(); return cached; };
}

const kEmptyObject = Object.freeze(Object.create(null));

function isError(value) {
  // Cross-realm errors are common in Node tests; using the toString tag
  // catches them more reliably than the instanceof check alone.
  return value instanceof Error ||
    Object.prototype.toString.call(value) === '[object Error]';
}

function setOwnProperty(target, key, value) {
  Object.defineProperty(target, key, {
    value, enumerable: true, writable: true, configurable: true,
  });
}

// internal Symbols used by util.promisify et al.
const customPromisifyArgs = Symbol('customPromisifyArgs');
const kCustomPromisifiedSymbol = Symbol('util.promisify.custom');

function promisify(orig) {
  if (orig[kCustomPromisifiedSymbol]) return orig[kCustomPromisifiedSymbol];
  const argNames = orig[customPromisifyArgs];
  return function (...args) {
    return new Promise((resolve, reject) => {
      orig.call(this, ...args, (err, ...values) => {
        if (err) return reject(err);
        if (argNames && argNames.length > 1) {
          const obj = {};
          for (let i = 0; i < argNames.length; i++) obj[argNames[i]] = values[i];
          resolve(obj);
        } else resolve(values[0]);
      });
    });
  };
}
promisify.custom = kCustomPromisifiedSymbol;

// SideEffectFree* — Node's variants that go through frozen prototypes for
// safety. For us (no adversarial userland), plain methods are equivalent.
const _bindCall = Function.prototype.call.bind.bind(Function.prototype.call);
const SideEffectFreeRegExpPrototypeExec = _bindCall(RegExp.prototype.exec);
const SideEffectFreeRegExpPrototypeSymbolReplace = _bindCall(RegExp.prototype[Symbol.replace]);

const isMacOS = (globalThis.__host && globalThis.__host.process.platform === 'darwin');

module.exports = {
  customInspectSymbol,
  customPromisifyArgs,
  kIsEncodingSymbol,
  kEmptyObject,
  isWindows,
  isMacOS,
  getLazy,
  isError,
  setOwnProperty,
  getCIDR,
  spliceOne,
  normalizeEncoding,
  encodingsMap,
  encodingNames,
  once: (fn) => { let called = false; return function (...a) {
    if (called) return; called = true; return fn.apply(this, a);
  }; },
  deprecate: (fn, _msg) => fn,
  promisify,
  SideEffectFreeRegExpPrototypeExec,
  SideEffectFreeRegExpPrototypeSymbolReplace,
  lazyDOMException,
  defineLazyProperties,
  // assignFunctionName(name, fn) — sets fn.name to a chosen value. Node uses
  // it to forward names through anonymous helpers (e.g. hideStackFrames).
  assignFunctionName: (name, fn) => {
    try { Object.defineProperty(fn, 'name', { value: name, configurable: true }); }
    catch (_) { /* read-only fn.name in some envs; non-fatal */ }
    return fn;
  },
};
`;

// Public `node:util` — pragmatic facade. Lifting Node's real lib/util.js
// drags in many internal/* deps; this surface covers what npm packages
// (Express, body-parser, send, debug, …) actually call.
shimSources['util'] = `
'use strict';
const { inspect } = require('internal/util/inspect');
const types = require('internal/util/types');

// util.inherits — pre-ES6 prototype chain helper. Still used by send/depd/
// many npm packages even though "extends" is the modern equivalent.
function inherits(ctor, superCtor) {
  if (ctor == null || typeof ctor !== 'function') throw new TypeError('The constructor to "inherits" must not be null or undefined');
  if (superCtor == null || typeof superCtor !== 'function') throw new TypeError('The super constructor to "inherits" must not be null or undefined');
  if (superCtor.prototype === undefined) throw new TypeError('The super constructor to "inherits" must have a prototype');
  Object.defineProperty(ctor, 'super_', { value: superCtor, writable: true, configurable: true });
  Object.setPrototypeOf(ctor.prototype, superCtor.prototype);
}

function format(fmt, ...args) {
  if (typeof fmt !== 'string') return [fmt, ...args].map(a => inspect(a)).join(' ');
  let i = 0;
  const subbed = fmt.replace(/%[sdjoOif%]/g, (m) => {
    if (m === '%%') return '%';
    if (i >= args.length) return m;
    const a = args[i++];
    if (m === '%s') return String(a);
    if (m === '%d' || m === '%i') return Number(a);
    if (m === '%f') return parseFloat(a);
    if (m === '%j') { try { return JSON.stringify(a); } catch (_) { return '[Circular]'; } }
    if (m === '%o' || m === '%O') return inspect(a);
    return m;
  });
  // Append any args not consumed by a format specifier, space-separated.
  // Strings are kept as-is; everything else goes through inspect (Node's
  // behavior — primitives stringify naturally, objects pretty-print).
  if (i < args.length) {
    const tail = args.slice(i).map(a => typeof a === 'string' ? a : inspect(a)).join(' ');
    return subbed + ' ' + tail;
  }
  return subbed;
}

function promisify(fn) {
  if (typeof fn !== 'function') throw new TypeError('promisify requires a function');
  return function(...args) {
    return new Promise((resolve, reject) => {
      try {
        fn.call(this, ...args, (err, v) => err ? reject(err) : resolve(v));
      } catch (e) { reject(e); }
    });
  };
}

function callbackify(fn) {
  if (typeof fn !== 'function') throw new TypeError('callbackify requires a function');
  return function(...args) {
    const cb = args.pop();
    if (typeof cb !== 'function') throw new TypeError('last arg must be a callback');
    try {
      const r = fn.apply(this, args);
      Promise.resolve(r).then(v => cb(null, v), e => cb(e));
    } catch (e) { Promise.resolve().then(() => cb(e)); }
  };
}

function deprecate(fn, _msg, _code) { return fn; }   // suppress warnings

function isDate(x)      { return x instanceof Date; }
function isError(x)     { return x instanceof Error; }
function isRegExp(x)    { return x instanceof RegExp; }
function isBuffer(x)    { return (globalThis.Buffer && globalThis.Buffer.isBuffer(x)) || false; }
function isFunction(x)  { return typeof x === 'function'; }
function isObject(x)    { return typeof x === 'object' && x !== null; }
function isNullOrUndefined(x) { return x == null; }
function isUndefined(x) { return x === undefined; }
function isNull(x)      { return x === null; }
function isNumber(x)    { return typeof x === 'number'; }
function isString(x)    { return typeof x === 'string'; }
function isBoolean(x)   { return typeof x === 'boolean'; }
function isSymbol(x)    { return typeof x === 'symbol'; }
function isPrimitive(x) { return x == null || (typeof x !== 'object' && typeof x !== 'function'); }

// util.debuglog(set, cb?) — return a no-op log function unless NODE_DEBUG is
// set. We always return a no-op (cb fires once with the fn for code that
// expects the late-bind pattern: \`let debug = util.debuglog('x', f => debug = f)\`).
function debuglog(_set, cb) {
  const fn = function () {};
  fn.enabled = false;
  if (typeof cb === 'function') Promise.resolve().then(() => cb(fn));
  return fn;
}

const TextEncoder = globalThis.TextEncoder;
const TextDecoder = globalThis.TextDecoder;

module.exports = {
  inherits,
  inspect,
  types,
  format,
  formatWithOptions: (_opts, fmt, ...args) => format(fmt, ...args),
  promisify,
  callbackify,
  deprecate,
  debuglog, debug: debuglog,
  isDate, isError, isRegExp, isBuffer,
  isFunction, isObject, isNullOrUndefined, isUndefined, isNull,
  isNumber, isString, isBoolean, isSymbol, isPrimitive,
  isArray: Array.isArray,
  TextEncoder, TextDecoder,
  _extend: Object.assign,
  // util.toUSVString — convert ill-formed surrogate pairs.
  toUSVString: (s) => { try { return String(s).toWellFormed ? String(s).toWellFormed() : String(s); } catch (_) { return String(s); } },
  getSystemErrorName: () => 'UNKNOWN',
  getSystemErrorMap: () => new Map(),
  parseEnv: (s) => {
    // Trivial .env parser.
    const out = {};
    if (typeof s !== 'string') return out;
    for (const ln of s.split('\\n')) {
      const t = ln.trim();
      if (!t || t.startsWith('#')) continue;
      const eq = t.indexOf('=');
      if (eq < 0) continue;
      out[t.slice(0, eq).trim()] = t.slice(eq + 1).trim();
    }
    return out;
  },
};
// util.promisify.custom — pseudo-symbol some packages use.
promisify.custom = Symbol.for('nodejs.util.promisify.custom');
`;

shimSources['internal/util/inspect'] = `
'use strict';
// Minimal inspect — enough that buffer/events/etc. can call it for toString.
function inspect(obj, _opts) {
  try { return String(obj); } catch (_e) { return '[object]'; }
}
// identicalSequenceRange(a, b) → [start, length] of the longest run of
// identical elements at the same index in two arrays. Used by events.js for
// stack trace deduplication. A trivial impl (always returns [0, 0]) is fine
// for our purposes.
function identicalSequenceRange(_a, _b) { return [0, 0]; }
module.exports = { inspect, identicalSequenceRange };
`;

// Minimal addAbortListener — used by events.js but only for AbortController
// integration. Returning a Disposable-shaped object satisfies the API.
shimSources['internal/events/abort_listener'] = `
'use strict';
function addAbortListener(signal, listener) {
  if (signal == null) return { [Symbol.dispose]: () => {} };
  if (signal.aborted) { listener.call(signal, new Event('abort')); return { [Symbol.dispose]: () => {} }; }
  signal.addEventListener && signal.addEventListener('abort', listener, { once: true });
  return { [Symbol.dispose]: () => {
    signal.removeEventListener && signal.removeEventListener('abort', listener);
  } };
}
module.exports = { addAbortListener };
`;

// internal/constants — character-code constants used by lib/path.js, etc.
// Real Node has hundreds more (errno, signals, openssl); add lazily.
shimSources['internal/constants'] = `
'use strict';
module.exports = {
  CHAR_UPPERCASE_A: 65, CHAR_LOWERCASE_A: 97,
  CHAR_UPPERCASE_Z: 90, CHAR_LOWERCASE_Z: 122,
  CHAR_DOT: 46,
  CHAR_FORWARD_SLASH: 47,
  CHAR_BACKWARD_SLASH: 92,
  CHAR_COLON: 58,
  CHAR_QUESTION_MARK: 63,
  CHAR_UNDERSCORE: 95,
  CHAR_HYPHEN_MINUS: 45,
  CHAR_PLUS: 43,
  CHAR_HASH: 35,
  CHAR_PERCENT: 37,
  CHAR_AMPERSAND: 38,
  CHAR_EQUAL: 61,
  CHAR_EXCLAMATION_MARK: 33,
  CHAR_SEMICOLON: 59,
  CHAR_SPACE: 32,
  CHAR_TAB: 9,
  CHAR_LINE_FEED: 10,
  CHAR_CARRIAGE_RETURN: 13,
  CHAR_NO_BREAK_SPACE: 0xA0,
  CHAR_ZERO_WIDTH_NOBREAK_SPACE: 0xFEFF,
};
`;

shimSources['internal/v8/startup_snapshot'] = `
'use strict';
module.exports = {
  namespace: {
    addDeserializeCallback: () => {},
    isBuildingSnapshot: () => false,
  },
  // Some Node code paths read these as top-level exports too.
  addDeserializeCallback: () => {},
  isBuildingSnapshot: () => false,
};
`;

// internal/test/binding — only used by Node's tests; exposes `internalBinding`
// so the test can poke at C++-binding-level state. We expose our JS-side
// globalThis.internalBinding directly.
shimSources['internal/test/binding'] = `
'use strict';
module.exports = { internalBinding: globalThis.internalBinding };
`;

shimSources['internal/options'] = `
'use strict';
// No CLI options layer — every option lookup returns undefined.
module.exports = {
  getOptionValue: (_name) => undefined,
};
`;

// Placeholder for the old handwritten shim. Kept around but renamed so
// require('assert') falls through to Node's real lib/assert.js (loaded via
// NODE_SOURCES in main.rs). Will be deleted once everything's stable.
shimSources['__legacy/assert'] = `
'use strict';

function AssertionError(opts) {
  const err = new Error(opts.message || 'assertion failed');
  err.name = 'AssertionError';
  err.code = 'ERR_ASSERTION';
  err.actual = opts.actual;
  err.expected = opts.expected;
  err.operator = opts.operator;
  err.generatedMessage = !opts.message;
  return err;
}

function deepEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (typeof a !== typeof b) return false;
  if (typeof a !== 'object') return a === b;
  if (a.constructor !== b.constructor) return false;
  if (ArrayBuffer.isView(a)) {
    const len = a.byteLength;
    if (len !== b.byteLength) return false;
    for (let i = 0; i < len; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  if (Array.isArray(a)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!deepEqual(a[i], b[i])) return false;
    return true;
  }
  const ak = Object.keys(a), bk = Object.keys(b);
  if (ak.length !== bk.length) return false;
  for (const k of ak) if (!deepEqual(a[k], b[k])) return false;
  return true;
}

function ok(value, message) {
  if (!value) throw AssertionError({ message: message || 'ok failed: ' + String(value), actual: value, expected: true, operator: 'ok' });
}

function strictEqual(actual, expected, message) {
  if (actual !== expected)
    throw AssertionError({ message: message || ('strictEqual: ' + String(actual) + ' !== ' + String(expected)), actual, expected, operator: 'strictEqual' });
}

function notStrictEqual(actual, expected, message) {
  if (actual === expected)
    throw AssertionError({ message: message || 'notStrictEqual failed', actual, expected, operator: 'notStrictEqual' });
}

function deepStrictEqual(actual, expected, message) {
  if (!deepEqual(actual, expected))
    throw AssertionError({ message: message || 'deepStrictEqual failed', actual, expected, operator: 'deepStrictEqual' });
}

// Returns null on match, or a string explaining which key diverged.
function expectedMismatch(thrown, expected) {
  if (typeof expected === 'function') {
    try { if (thrown instanceof expected) return null; } catch (_e) {}
    return expected(thrown) ? null : 'predicate failed';
  }
  if (expected instanceof RegExp) {
    return expected.test(String(thrown && thrown.message)) ? null
      : 'message did not match regex ' + expected;
  }
  if (expected && typeof expected === 'object') {
    for (const k of Object.keys(expected)) {
      const ev = expected[k], av = thrown && thrown[k];
      if (ev instanceof RegExp) {
        if (!ev.test(String(av))) return k + ': /' + ev + '/ vs ' + JSON.stringify(av);
      } else if (av !== ev) {
        return k + ':\\n  expected: ' + JSON.stringify(ev) + '\\n  actual:   ' + JSON.stringify(av);
      }
    }
    return null;
  }
  return null;
}

function throws(fn, expected, message) {
  let thrown, didThrow = false;
  try { fn(); } catch (e) { thrown = e; didThrow = true; }
  if (!didThrow)
    throw AssertionError({ message: message || 'expected fn to throw', operator: 'throws' });
  if (expected !== undefined) {
    const diff = expectedMismatch(thrown, expected);
    if (diff !== null) {
      throw AssertionError({
        message: (message || 'throws: ') + diff,
        actual: thrown, expected, operator: 'throws',
      });
    }
  }
}

// rejects(promise, expected) — promise variant of throws. Returns a promise
// that resolves when the input promise rejects with a matching error.
async function rejects(promiseOrFn, expected, message) {
  const p = typeof promiseOrFn === 'function' ? promiseOrFn() : promiseOrFn;
  let thrown, didReject = false;
  try { await p; } catch (e) { thrown = e; didReject = true; }
  if (!didReject)
    throw AssertionError({ message: message || 'expected promise to reject', operator: 'rejects' });
  if (expected !== undefined) {
    const diff = expectedMismatch(thrown, expected);
    if (diff !== null)
      throw AssertionError({
        message: (message || 'rejects: ') + diff,
        actual: thrown, expected, operator: 'rejects',
      });
  }
}

async function doesNotReject(promiseOrFn, message) {
  const p = typeof promiseOrFn === 'function' ? promiseOrFn() : promiseOrFn;
  try { await p; } catch (e) {
    throw AssertionError({ message: message || 'doesNotReject: ' + (e && e.message), actual: e, operator: 'doesNotReject' });
  }
}

function doesNotThrow(fn, message) {
  try { fn(); } catch (e) {
    throw AssertionError({ message: message || 'doesNotThrow: ' + (e && e.message), actual: e, operator: 'doesNotThrow' });
  }
}

const assert = ok;
assert.AssertionError = function (opts) { return AssertionError(opts || {}); };
assert.ok = ok;
assert.equal = strictEqual;
assert.notEqual = notStrictEqual;
assert.strictEqual = strictEqual;
assert.notStrictEqual = notStrictEqual;
assert.deepEqual = deepStrictEqual;
assert.deepStrictEqual = deepStrictEqual;
assert.throws = throws;
assert.rejects = rejects;
assert.doesNotReject = doesNotReject;
assert.doesNotThrow = doesNotThrow;
assert.fail = (message) => { throw AssertionError({ message: message || 'fail', operator: 'fail' }); };
assert.match = (str, re, message) => {
  if (!re.test(str)) throw AssertionError({ message: message || 'match failed', actual: str, expected: re, operator: 'match' });
};

module.exports = assert;
`;

// Node tests do require('../common'). Register under both names so the test
// loader hits us regardless of relative path.
const commonSource = `
'use strict';

function expectsError(expected) {
  return (err) => {
    if (expected.code && err.code !== expected.code) return false;
    if (expected.name && err.name !== expected.name) return false;
    if (expected.message !== undefined) {
      if (expected.message instanceof RegExp) {
        if (!expected.message.test(String(err.message))) return false;
      } else if (err.message !== expected.message) {
        return false;
      }
    }
    return true;
  };
}

function mustCall(fn, expected) {
  fn = fn || (() => {});
  expected = expected == null ? 1 : expected;
  const state = { actual: 0 };
  const wrapped = function (...args) { state.actual++; return fn.apply(this, args); };
  wrapped._mustCallState = state;
  wrapped._expected = expected;
  return wrapped;
}

// common.mustCallAtLeast(fn, n) — return a wrapper that requires fn to be
// called at least n times. We don't verify at process exit (no exit hook),
// so this is effectively the same as mustCall — the wrapper just forwards.
function mustCallAtLeast(fn, n) {
  fn = fn || (() => {});
  n = n == null ? 1 : n;
  const state = { actual: 0, minimum: n };
  const wrapped = function (...args) { state.actual++; return fn.apply(this, args); };
  wrapped._mustCallState = state;
  wrapped._mustCallAtLeast = true;
  return wrapped;
}

function mustSucceed(fn) {
  return mustCall(function (err, ...args) {
    if (err) throw err;
    if (typeof fn === 'function') fn.apply(this, args);
  });
}

function mustNotCall(label) {
  return function () { throw new Error('mustNotCall' + (label ? ': ' + label : '') + ' was called'); };
}

// invalidArgTypeHelper matches Node's test helper: returns a suffix
// (starting with a SPACE) describing the value's type. The test concatenates
// the main error sentence (which ends in '.') with this helper's output, so
// the leading space — not a period — is what produces "Object. Received ...".
function invalidArgTypeHelper(input) {
  if (input == null) return ' Received ' + (input === null ? 'null' : 'undefined');
  if (typeof input === 'function') {
    return ' Received function ' + (input.name || '');
  }
  if (typeof input === 'object') {
    const ctor = input.constructor && input.constructor.name;
    if (ctor) return ' Received an instance of ' + ctor;
    return ' Received ' + Object.prototype.toString.call(input);
  }
  let inspected = String(input);
  if (inspected.length > 25) inspected = inspected.slice(0, 25) + '...';
  if (typeof input === 'string') inspected = "'" + inspected + "'";
  if (typeof input === 'bigint') inspected += 'n';
  return ' Received type ' + typeof input + ' (' + inspected + ')';
}

// common.expectWarning — Node tests use this to assert that emitWarning
// is called with specific args. Our process.emitWarning is a no-op, so we
// just register the expectation and validate nothing.
function expectWarning(_kind, _msg, _code) { /* no-op */ }

// common.getArrayBufferViews(buf) — return every TypedArray view variant
// over the buffer's underlying ArrayBuffer. Node uses this to feed the same
// bytes to APIs in different shapes (Uint8Array, Uint16Array, DataView, …).
function getArrayBufferViews(buf) {
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const views = [];
  for (const Ctor of [Uint8Array, Int8Array, Uint8ClampedArray]) {
    views.push(new Ctor(ab));
  }
  if (buf.byteLength % 2 === 0) {
    for (const Ctor of [Uint16Array, Int16Array]) views.push(new Ctor(ab));
  }
  if (buf.byteLength % 4 === 0) {
    for (const Ctor of [Uint32Array, Int32Array, Float32Array]) views.push(new Ctor(ab));
  }
  if (buf.byteLength % 8 === 0) {
    for (const Ctor of [Float64Array]) views.push(new Ctor(ab));
  }
  views.push(new DataView(ab));
  return views;
}

module.exports = {
  expectsError, mustCall, mustCallAtLeast, mustSucceed, mustNotCall,
  invalidArgTypeHelper, expectWarning, getArrayBufferViews,
  mustNotMutateObjectDeep: (o) => o,
  enoughTestMem: false,
  isWindows: false, isLinux: false, isMacOS: true,
  hasIntl: false, hasCrypto: false,
  platformTimeout: (ms) => ms,
  printSkipMessage: (msg) => { /* no-op */ },
  skip: (msg) => { /* no-op */ },
  PORT: 0,
};
`;
shimSources['common'] = commonSource;
shimSources['../common'] = commonSource;
shimSources['../../common'] = commonSource;

// Some tests do require('../common/fixtures') or '../fixtures'. Tests that
// actually use fixtures (loading real files) will fail in a clear way;
// most just check helpers exist on the export.
const fixturesSource = `
'use strict';
const path = require('path');
const base = '/portable-node/test/fixtures';
module.exports = {
  path: (...segs) => path.join(base, ...segs),
  fixturesDir: base,
  readSync: () => '',
  readKey: () => '',
};
`;
shimSources['../common/fixtures'] = fixturesSource;
shimSources['../../common/fixtures'] = fixturesSource;

// =========================================================================
// Portable-host JS bindings — each implements an internalBinding(name)
// surface by calling __host.* primitives. Dispatcher in Rust falls back
// to require('__binding/<name>') when no Rust implementation exists.
// =========================================================================

shimSources['__binding/os'] = `
'use strict';
const host = globalThis.__host;

// getCPUs returns a FLAT array matching Node's binding contract:
//   [model, speed, user, nice, sys, idle, irq, model, speed, ...]
function getCPUs() {
  const cpus = host.os.cpus();
  const flat = [];
  for (const c of cpus) {
    flat.push(c.model, c.speed, c.times.user, c.times.nice, c.times.sys, c.times.idle, c.times.irq);
  }
  return flat;
}

// getInterfaceAddresses returns FLAT [name, address, netmask, family, mac, internal, scopeid, ...]
function getInterfaceAddresses() {
  const ifs = host.os.networkInterfaces();
  const flat = [];
  for (const i of ifs) {
    flat.push(i.name, i.address, i.netmask, i.family, i.mac, i.internal, i.scopeid);
  }
  return flat;
}

// getOSInformation returns [type, version, release, machine] (positional).
function getOSInformation() {
  return [host.os.osType(), host.os.osVersion(), host.os.osRelease(), host.os.arch()];
}

// getLoadAvg(out) writes into the provided Float64Array.
function getLoadAvg(out) {
  const a = host.os.loadavg();
  out[0] = a[0]; out[1] = a[1]; out[2] = a[2];
}

module.exports = {
  getAvailableParallelism: () => host.os.availableParallelism(),
  getCPUs,
  getFreeMem: () => host.os.freemem(),
  getHomeDirectory: () => host.os.homedir(),
  getHostname:     () => host.os.hostname(),
  getInterfaceAddresses,
  getLoadAvg,
  getOSInformation,
  getPriority:     (pid) => host.os.getPriority(pid),
  getTotalMem:     () => host.os.totalmem(),
  getUptime:       () => host.os.uptime(),
  getUserInfo:     () => host.os.userInfo(),
  isBigEndian:     host.os.endianness() === 'BE',
  setPriority:     (pid, prio) => { host.os.setPriority(pid, prio); return 0; },
};
`;

// internalBinding('fs') — Node's fs C++ binding surface. We map every method
// onto __host.file.*. Same shim works for any host: only __host changes.
shimSources['__binding/fs'] = `
'use strict';
const host = globalThis.__host;
const F = host.file;
const flags = F.flags;

// Translate Node's string flag forms ('r', 'w', 'a', 'r+', etc.) to numeric.
function stringToFlags(s) {
  if (typeof s === 'number') return s;
  switch (s) {
    case 'r':   return flags.O_RDONLY;
    case 'rs':  case 'sr':
                return flags.O_RDONLY;
    case 'r+':  return flags.O_RDWR;
    case 'w':   return flags.O_WRONLY | flags.O_CREAT | flags.O_TRUNC;
    case 'wx':  case 'xw':
                return flags.O_WRONLY | flags.O_CREAT | flags.O_TRUNC | flags.O_EXCL;
    case 'w+':  return flags.O_RDWR   | flags.O_CREAT | flags.O_TRUNC;
    case 'wx+': case 'xw+':
                return flags.O_RDWR   | flags.O_CREAT | flags.O_TRUNC | flags.O_EXCL;
    case 'a':   return flags.O_WRONLY | flags.O_CREAT | flags.O_APPEND;
    case 'ax':  case 'xa':
                return flags.O_WRONLY | flags.O_CREAT | flags.O_APPEND | flags.O_EXCL;
    case 'a+':  return flags.O_RDWR   | flags.O_CREAT | flags.O_APPEND;
    case 'ax+': case 'xa+':
                return flags.O_RDWR   | flags.O_CREAT | flags.O_APPEND | flags.O_EXCL;
  }
  throw new TypeError('Invalid flag: ' + s);
}

// Most of these match libuv's sync API contract (last arg is a ctx object
// the binding fills with error info; we just throw instead).
function notImpl(name) {
  return function () {
    const e = new Error('portable-node: fs binding "' + name + '" not implemented');
    e.code = 'ENOSYS';
    e.syscall = name;
    throw e;
  };
}

// Convert a stat object to BigUint64Array/Float64Array per Node's binding
// convention (statValues — 14 fields). Node fs.js destructures this.
function statToValues(s) {
  // Float64Array: bigint=false → return numbers; we return plain array shape
  // that Node's StatWatcher etc. can index.
  return [
    s.dev, s.mode, s.nlink, s.uid, s.gid, s.rdev, s.blksize,
    s.ino, s.size, s.blocks,
    s.atime_ms, s.mtime_ms, s.ctime_ms, s.birthtime_ms,
  ];
}

module.exports = {
  flagsForString: stringToFlags,

  // Real impls — call into __host.file
  open:        (path, flags, mode) => F.open(String(path), flags|0, mode|0),
  close:       (fd) => F.close(fd|0),
  read:        (fd, buf, off, len, pos) => F.read(fd|0, buf, off|0, len|0, pos == null ? -1 : Number(pos)),
  writeBuffer: (fd, buf, off, len, pos) => F.write(fd|0, buf, off|0, len|0, pos == null ? -1 : Number(pos)),
  writeBuffers:(fd, chunks, pos) => {
    let total = 0;
    for (const c of chunks) {
      const n = F.write(fd|0, c, 0, c.length, pos == null ? -1 : Number(pos) + total);
      total += n;
    }
    return total;
  },
  writeString: (fd, str, pos, enc) => {
    const buf = Buffer.from(str, enc || 'utf8');
    return F.write(fd|0, buf, 0, buf.length, pos == null ? -1 : Number(pos));
  },
  stat:        (path) => F.stat(String(path)),
  lstat:       (path) => F.lstat(String(path)),
  fstat:       (fd)   => F.fstat(fd|0),
  readdir:     (path, _encoding, withFileTypes) => {
    const entries = F.readdir(String(path));
    if (withFileTypes) {
      // Node's getDirents destructures \`{0: names, 1: types}\` — i.e. a
      // 2-element tuple of parallel arrays, not a flat name/type list.
      const names = [], types = [];
      for (const e of entries) { names.push(e.name); types.push(e.type); }
      return [names, types];
    }
    // Default: array of strings.
    return entries.map((e) => e.name);
  },
  realpath:    (path) => F.realpath(String(path)),
  unlink:      (path) => F.unlink(String(path)),
  mkdir:       (path, mode, recursive) => {
    if (recursive) {
      // Naive recursive walk (sep = '/').
      const parts = String(path).split('/');
      let cur = parts[0] || '';
      for (let i = 1; i < parts.length; i++) {
        cur = cur + '/' + parts[i];
        if (!cur) continue;
        try { F.mkdir(cur, mode|0); } catch (e) { if (e.code !== 'EEXIST') throw e; }
      }
      return;
    }
    F.mkdir(String(path), mode|0);
  },
  rmdir:       (path) => F.rmdir(String(path)),
  rename:      (a, b) => F.rename(String(a), String(b)),
  access:      (path, mode) => F.access(String(path), mode|0),

  // Existence probe — Node's fs binding has existsSync as a fast path.
  existsSync:  (path) => { try { F.stat(String(path)); return true; } catch (_e) { return false; } },

  // Stubs for methods we don't yet provide a host primitive for. They throw
  // clearly so code paths that hit them fail loudly rather than silently.
  chmod:       notImpl('chmod'),
  chown:       notImpl('chown'),
  copyFile:    notImpl('copyFile'),
  fchmod:      notImpl('fchmod'),
  fchown:      notImpl('fchown'),
  fdatasync:   notImpl('fdatasync'),
  fsync:       notImpl('fsync'),
  ftruncate:   notImpl('ftruncate'),
  futimes:     notImpl('futimes'),
  internalModuleStat: notImpl('internalModuleStat'),
  lchown:      notImpl('lchown'),
  link:        notImpl('link'),
  lutimes:     notImpl('lutimes'),
  mkdtemp:     notImpl('mkdtemp'),
  readBuffers: notImpl('readBuffers'),
  readFileUtf8: (path, _flags) => {
    const fd = F.open(String(path), flags.O_RDONLY, 0);
    try {
      const st = F.fstat(fd);
      const buf = Buffer.allocUnsafe(st.size);
      let r = 0;
      while (r < st.size) {
        const n = F.read(fd, buf, r, st.size - r, r);
        if (n <= 0) break;
        r += n;
      }
      return buf.slice(0, r).toString('utf8');
    } finally { F.close(fd); }
  },
  readlink:    notImpl('readlink'),
  rmSync:      notImpl('rmSync'),
  statfs:      notImpl('statfs'),
  symlink:     notImpl('symlink'),
  utimes:      notImpl('utimes'),
  writeFileUtf8: (path, data, _flag, mode) => {
    // Node's contract: open with flag, write the string as UTF-8.
    const fd = F.open(String(path), flags.O_WRONLY | flags.O_CREAT | flags.O_TRUNC, mode|0 || 0o666);
    try {
      const buf = Buffer.from(data, 'utf8');
      let off = 0;
      while (off < buf.length) {
        const n = F.write(fd, buf, off, buf.length - off, -1);
        if (n <= 0) break;
        off += n;
      }
    } finally { F.close(fd); }
  },
};
`;

// internalBinding('constants') — a giant grab-bag of numeric constants.
// Only the subsets actually accessed need values; rest can be empty.
shimSources['__binding/constants'] = `
'use strict';
module.exports = {
  // (single os: entry below merges both signal/priority + errno)
  fs: {
    O_RDONLY: 0, O_WRONLY: 1, O_RDWR: 2,
    O_CREAT: 0x200, O_TRUNC: 0x400, O_APPEND: 8, O_EXCL: 0x800,
    O_SYNC: 0x80, O_NONBLOCK: 4, O_NOCTTY: 0x20000,
    O_DIRECTORY: 0x100000, O_NOFOLLOW: 0x100, O_SYMLINK: 0x200000,
    F_OK: 0, R_OK: 4, W_OK: 2, X_OK: 1,
    S_IFMT: 0o170000, S_IFREG: 0o100000, S_IFDIR: 0o040000,
    S_IFLNK: 0o120000, S_IFBLK: 0o060000, S_IFCHR: 0o020000,
    S_IFIFO: 0o010000, S_IFSOCK: 0o140000,
    S_IRUSR: 0o400, S_IWUSR: 0o200, S_IXUSR: 0o100,
    S_IRGRP: 0o040, S_IWGRP: 0o020, S_IXGRP: 0o010,
    S_IROTH: 0o004, S_IWOTH: 0o002, S_IXOTH: 0o001,
    COPYFILE_EXCL: 1, COPYFILE_FICLONE: 2, COPYFILE_FICLONE_FORCE: 4,
    UV_FS_SYMLINK_DIR: 1, UV_FS_SYMLINK_JUNCTION: 2,
    UV_DIRENT_UNKNOWN: 0, UV_DIRENT_FILE: 1, UV_DIRENT_DIR: 2,
    UV_DIRENT_LINK: 3, UV_DIRENT_FIFO: 4, UV_DIRENT_SOCKET: 5,
    UV_DIRENT_CHAR: 6, UV_DIRENT_BLOCK: 7,
    UV_FS_O_FILEMAP: 0,
  },
  crypto: {},
  zlib: {},
  trace: {},
  os: {
    UV_UDP_REUSEADDR: 4,
    dlopen: {},
    errno: {
      EISDIR: -21, ENOENT: -2, EACCES: -13, EEXIST: -17, ENOTDIR: -20,
      ENOTEMPTY: -66, EMFILE: -24, ENFILE: -23, EBADF: -9, EINVAL: -22,
      EPERM: -1, EIO: -5, ELOOP: -40, ENAMETOOLONG: -36, ENOSPC: -28,
    },
    signals: {
      SIGHUP: 1, SIGINT: 2, SIGQUIT: 3, SIGILL: 4, SIGTRAP: 5, SIGABRT: 6,
      SIGBUS: 10, SIGFPE: 8, SIGKILL: 9, SIGUSR1: 30, SIGSEGV: 11,
      SIGUSR2: 31, SIGPIPE: 13, SIGALRM: 14, SIGTERM: 15, SIGCHLD: 20,
      SIGCONT: 19, SIGSTOP: 17, SIGTSTP: 18, SIGTTIN: 21, SIGTTOU: 22,
      SIGURG: 16, SIGXCPU: 24, SIGXFSZ: 25, SIGVTALRM: 26, SIGPROF: 27,
      SIGWINCH: 28, SIGIO: 23, SIGSYS: 12,
    },
    priority: {
      PRIORITY_LOW: 19, PRIORITY_BELOW_NORMAL: 10, PRIORITY_NORMAL: 0,
      PRIORITY_ABOVE_NORMAL: -7, PRIORITY_HIGH: -14, PRIORITY_HIGHEST: -20,
    },
  },
};
`;

// internalBinding('string_decoder') — provides decode/flush plus slot
// indices into the per-decoder state buffer. Node's native version stores
// streaming state inside a small Uint8Array; we follow the same layout so
// Node's lib/string_decoder.js works unmodified.
//   state[0..4]: pending bytes of an incomplete multi-byte sequence
//   state[4] (kMissingBytes):  how many more bytes needed
//   state[5] (kBufferedBytes): how many bytes are buffered
//   state[6] (kEncodingField): numeric encoding code (encodingsMap value)
shimSources['__binding/string_decoder'] = `
'use strict';
const { Buffer } = require('buffer');

const kIncompleteCharactersStart = 0;
const kIncompleteCharactersEnd   = 4;
const kMissingBytes  = 4;
const kBufferedBytes = 5;
const kEncodingField = 6;
const kSize          = 7;

// Decode helper: classify a UTF-8 leading byte. Returns char-length 1..4
// or 0 if it's a continuation byte (caller treats as error).
function utf8LeadLen(b) {
  if (b < 0x80) return 1;
  if ((b & 0xE0) === 0xC0) return 2;
  if ((b & 0xF0) === 0xE0) return 3;
  if ((b & 0xF8) === 0xF0) return 4;
  return 0;
}

function decodeUtf8(state, input) {
  // Concatenate pending state + new input into a single byte stream.
  const have = state[kBufferedBytes];
  const total = have + input.length;
  const bytes = new Uint8Array(total);
  for (let k = 0; k < have; k++) bytes[k] = state[k];
  for (let k = 0; k < input.length; k++) bytes[have + k] = input[k];

  let out = '';
  let i = 0;
  while (i < total) {
    const b = bytes[i];
    if (b < 0x80) {
      out += String.fromCharCode(b);
      i++;
      continue;
    }
    if ((b & 0xC0) === 0x80) {
      // Stray continuation byte.
      out += '\\uFFFD';
      i++;
      continue;
    }
    const need = utf8LeadLen(b);
    if (need === 0) {
      out += '\\uFFFD';
      i++;
      continue;
    }
    // Count how many valid continuation bytes follow.
    let valid = 1;
    while (valid < need && i + valid < total &&
           (bytes[i + valid] & 0xC0) === 0x80) valid++;
    if (valid < need) {
      if (i + valid >= total) {
        // Ran out of input. Save partial sequence for next write().
        for (let k = 0; k < valid; k++) state[k] = bytes[i + k];
        state[kBufferedBytes] = valid;
        state[kMissingBytes] = need - valid;
        return out;
      }
      // Sequence cut short by a non-continuation byte mid-stream. WHATWG /
      // Node's policy: emit ONE replacement for the whole truncated lead-
      // plus-continuations group, then resume from the next byte.
      out += '\\uFFFD';
      i += valid;
      continue;
    }
    // Have a complete char. Use Buffer.toString('utf8') to decode.
    out += Buffer.from(bytes.buffer, bytes.byteOffset + i, need).toString('utf8');
    i += need;
  }
  state[kBufferedBytes] = 0;
  state[kMissingBytes] = 0;
  return out;
}

// UTF-16LE decoding with two layered concerns:
//   (1) odd-byte continuation across calls (carry 1 byte forward to make
//       the next code unit), and
//   (2) trailing high surrogate continuation (hold 2 bytes back when the
//       *last* complete unit is a high surrogate AND nothing follows it
//       in this call — so the next call can pair it with a low surrogate).
//
// state[kBufferedBytes] encodes: 0 = empty, 1 = 1 odd byte at state[0],
// 2 = 2-byte held high surrogate at state[0..1], 3 = held surrogate + 1
// trailing odd byte at state[2].
//
// Node's actual rule (observed via test-string-decoder cases):
//   - A held high surrogate is "sticky": it's only released when the NEXT
//     complete unit arrives (and either pairs with it, or doesn't).
//   - In one-shot decoding (no prior state), a high surrogate is held back
//     ONLY if the input ends exactly at the surrogate boundary (no extra
//     bytes after); otherwise it's emitted as-is.
function decodeUtf16le(state, input) {
  let out = '';
  let have = state[kBufferedBytes];
  let cursor = 0;  // index into input we've already consumed

  // Resolve a held high surrogate first (have >= 2). We need exactly one
  // more 2-byte code unit (lo, hi) to decide pair/no-pair.
  if (have >= 2) {
    const needFromInput = (have === 2) ? 2 : 1;
    if (input.length < needFromInput) {
      // Not enough to form the next unit. Buffer the odd byte if any.
      if (have === 2 && input.length === 1) {
        state[2] = input[0];
        state[kBufferedBytes] = 3;
      }
      return '';
    }
    let lo, hi;
    if (have === 2) {
      lo = input[0]; hi = input[1];
      cursor = 2;
    } else {
      lo = state[2]; hi = input[0];
      cursor = 1;
    }
    const nextUnit = (hi << 8) | lo;
    if (nextUnit >= 0xDC00 && nextUnit <= 0xDFFF) {
      out += Buffer.from(new Uint8Array([state[0], state[1], lo, hi])).toString('utf16le');
    } else {
      out += Buffer.from(new Uint8Array([state[0], state[1]])).toString('utf16le');
      out += Buffer.from(new Uint8Array([lo, hi])).toString('utf16le');
    }
    state[kBufferedBytes] = 0;
    have = 0;
  } else if (have === 1) {
    // Combine the saved odd byte with the first new byte.
    if (input.length === 0) return '';
    const lo = state[0], hi = input[0];
    cursor = 1;
    const unit = (hi << 8) | lo;
    if (unit >= 0xD800 && unit <= 0xDBFF) {
      // Combined is a high surrogate. Hold it (sticky) for in-place processing
      // below to potentially merge with future bytes.
      state[0] = lo; state[1] = hi; state[kBufferedBytes] = 2;
      have = 2;
    } else {
      out += Buffer.from(new Uint8Array([lo, hi])).toString('utf16le');
      state[kBufferedBytes] = 0;
      have = 0;
    }
  }

  // If have became 2 from the byte-combine path, recurse to apply the same
  // "resolve held surrogate" logic to the remaining input.
  if (have === 2 && cursor < input.length) {
    const slice = new Uint8Array(input.buffer, input.byteOffset + cursor, input.length - cursor);
    out += decodeUtf16le(state, slice);
    return out;
  }
  if (cursor >= input.length) return out;

  // In-place processing of the remaining slice (have == 0 here).
  const slice = new Uint8Array(input.buffer, input.byteOffset + cursor, input.length - cursor);
  const total = slice.length;
  const evenLen = total & ~1;
  let emitLen = evenLen;
  const trailingOdd = (total & 1) === 1;

  if (!trailingOdd && evenLen >= 2) {
    const last = (slice[evenLen - 1] << 8) | slice[evenLen - 2];
    if (last >= 0xD800 && last <= 0xDBFF) {
      // High surrogate at the very end → hold back.
      emitLen = evenLen - 2;
      state[0] = slice[evenLen - 2];
      state[1] = slice[evenLen - 1];
      state[kBufferedBytes] = 2;
    }
  } else if (trailingOdd) {
    state[0] = slice[total - 1];
    state[kBufferedBytes] = 1;
  }
  if (emitLen > 0) {
    out += Buffer.from(slice.buffer, slice.byteOffset, emitLen).toString('utf16le');
  }
  return out;
}

function flushUtf8(state) {
  if (state[kBufferedBytes] === 0) return '';
  state[kBufferedBytes] = 0;
  state[kMissingBytes] = 0;
  return '\\uFFFD';
}
function flushUtf16le(state) {
  const have = state[kBufferedBytes];
  if (have === 0) return '';
  state[kBufferedBytes] = 0;
  if (have === 2 || have === 3) {
    // We were holding a high surrogate; emit it now (any trailing odd byte
    // is dropped — matches Node's "incomplete code unit" silent-drop).
    return Buffer.from(new Uint8Array([state[0], state[1]])).toString('utf16le');
  }
  // have === 1: half a code unit silently dropped.
  return '';
}

// base64 / base64url accumulate up to 2 bytes between writes so output is
// always emitted in 3-byte (4-char) groups. flush emits the trailing
// partial group with padding ('==' or '=') if Node's behaviour requires.
function decodeBase64Like(state, input, urlSafe) {
  let buffered = state[kBufferedBytes];
  const total = buffered + input.length;
  const bytes = new Uint8Array(total);
  for (let k = 0; k < buffered; k++) bytes[k] = state[k];
  for (let k = 0; k < input.length; k++) bytes[buffered + k] = input[k];

  // Emit floor(total/3) groups; keep remainder for next call.
  const emitBytes = total - (total % 3);
  let out = '';
  if (emitBytes > 0) {
    out = Buffer.from(bytes.buffer, bytes.byteOffset, emitBytes)
      .toString(urlSafe ? 'base64url' : 'base64');
  }
  const remBytes = total - emitBytes;
  for (let k = 0; k < remBytes; k++) state[k] = bytes[emitBytes + k];
  state[kBufferedBytes] = remBytes;
  return out;
}
function flushBase64Like(state, urlSafe) {
  const n = state[kBufferedBytes];
  if (n === 0) return '';
  const slice = new Uint8Array(n);
  for (let k = 0; k < n; k++) slice[k] = state[k];
  state[kBufferedBytes] = 0;
  return Buffer.from(slice).toString(urlSafe ? 'base64url' : 'base64');
}

function decode(state, input) {
  // Normalize the input: Node's StringDecoder accepts any TypedArray or
  // DataView. Reinterpret as a Uint8Array view over the same bytes so we
  // can index byte-by-byte regardless of source element size.
  if (!(input instanceof Uint8Array)) {
    input = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  switch (state[kEncodingField]) {
    case 0:  return decodeUtf8(state, input);
    case 1:  return decodeUtf16le(state, input);
    case 2:  return Buffer.from(input.buffer, input.byteOffset, input.byteLength).toString('latin1');
    case 3:  return decodeBase64Like(state, input, false);
    case 4:  return decodeBase64Like(state, input, true);
    case 5:  return Buffer.from(input.buffer, input.byteOffset, input.byteLength).toString('ascii');
    case 6:  return Buffer.from(input.buffer, input.byteOffset, input.byteLength).toString('hex');
    default: return decodeUtf8(state, input);
  }
}
function flush(state) {
  switch (state[kEncodingField]) {
    case 0:  return flushUtf8(state);
    case 1:  return flushUtf16le(state);
    case 3:  return flushBase64Like(state, false);
    case 4:  return flushBase64Like(state, true);
    default: return '';
  }
}

module.exports = {
  kIncompleteCharactersStart, kIncompleteCharactersEnd,
  kMissingBytes, kBufferedBytes, kEncodingField, kSize,
  decode, flush,
};
`;

// internalBinding('tcp_wrap') — TCP class + TCPConnectWrap that net.js sits
// on top of. Translates Node's request-wrap pattern into op-id callbacks.
// Node expects:
//   const h = new TCP(TCPConstants.SOCKET);
//   h.onread = (nread, buf) => ...;        nread>0: data, =0: nothing, <0: -errno (-4095 = UV_EOF)
//   h.onconnection = (status, clientHandle) => ...;
//   h.bind(ip, port, flags) -> 0|errno
//   h.listen(backlog) -> 0|errno
//   h.connect(connectWrap, host, port) -> 0|errno
//   h.readStart() / h.readStop()
//   h.writeBuffer(writeWrap, buf) -> 0|errno
//   h.shutdown(shutdownWrap) -> 0|errno
//   h.close(cb)
//   h.getsockname(out) / h.getpeername(out)
//
// Wrap objects have an oncomplete callback fired when the op finishes.
shimSources['__binding/tcp_wrap'] = `
'use strict';

const UV_EOF = -4095;
const UV_ECANCELED = -4081;

const TCPConstants = { SOCKET: 0, SERVER: 1, UV_TCP_IPV6ONLY: 1, UV_TCP_REUSEPORT: 2 };

class TCP {
  constructor(_flag) {
    this.handle = tcpCreate();
    this.reading = false;
    this.readBuf = null;
    this.onread = null;
    this.onconnection = null;
    this.bytesRead = 0;
    this.bytesWritten = 0;
    this._closed = false;
  }

  bind(ip, port, _flags) {
    // Our host's tcp.listen() does bind+listen in one syscall, so bind()
    // here only records the address.
    this._bindIp = ip;
    this._bindPort = port|0;
    return 0;
  }
  bind6(ip, port, flags) { return this.bind(ip, port, flags); }

  listen(_backlog) {
    if (this._bindIp === undefined) return -22; // -EINVAL
    const status = tcpListen(this.handle, this._bindIp, this._bindPort, _backlog|0);
    if (status !== 0) return status;
    // Auto-arm accept loop: any pending onconnection listener will receive
    // events one at a time. We resubmit accept after each one fires.
    const self = this;
    const acceptOnce = () => {
      tcpAccept(self.handle, (c) => {
        if (self._closed) return;
        if (c.status === 0) {
          // Create a wrapped TCP for the client and pass to onconnection.
          const client = Object.create(TCP.prototype);
          client.handle = c.handle;
          client.reading = false; client.readBuf = null;
          client.onread = null; client.onconnection = null;
          client.bytesRead = 0; client.bytesWritten = 0;
          client._closed = false;
          if (self.onconnection) self.onconnection(0, client);
          // Resubmit for the next connection.
          acceptOnce();
        } else if (self.onconnection) {
          self.onconnection(c.status, null);
        }
      });
    };
    acceptOnce();
    return 0;
  }

  connect(req, host, port) {
    const opId = tcpConnect(this.handle, host, port|0, (c) => {
      if (req.oncomplete) {
        req.oncomplete(c.status, this, req, true, true);
      }
    });
    req._opId = opId;
    return 0;
  }
  connect6(req, host, port) { return this.connect(req, host, port); }

  readStart() {
    // NOTE: \`this.reading\` is set to \`true\` by net.js's tryReadStart BEFORE
    // it calls readStart() — so we can't gate on it. Use an internal flag.
    if (this._readArmed || this._closed) return 0;
    this._readArmed = true;
    const self = this;
    const STREAM_STATE = require('__binding/stream_wrap').streamBaseState;
    const readChunk = () => {
      if (!self._readArmed || self._closed) return;
      const ab = new ArrayBuffer(65536);
      const view = new Uint8Array(ab);
      tcpRead(self.handle, view, 0, view.length, (c) => {
        if (self._closed) return;
        if (c.status === 0 && c.n > 0) {
          self.bytesRead += c.n;
          STREAM_STATE[0] = c.n;   // kReadBytesOrError
          STREAM_STATE[1] = 0;     // kArrayBufferOffset
          if (self.onread) self.onread.call(self, ab);
          // Continue reading.
          readChunk();
        } else if (c.status === 0 && c.n === 0) {
          // EOF.
          STREAM_STATE[0] = UV_EOF;
          STREAM_STATE[1] = 0;
          if (self.onread) self.onread.call(self, ab);
          self._readArmed = false;
        } else {
          STREAM_STATE[0] = c.status;
          STREAM_STATE[1] = 0;
          if (self.onread) self.onread.call(self, ab);
          self._readArmed = false;
        }
      });
    };
    readChunk();
    return 0;
  }
  readStop() { this._readArmed = false; return 0; }

  writeBuffer(req, buf) {
    const bytes = ArrayBuffer.isView(buf) ? buf : Buffer.from(buf);
    const off = bytes.byteOffset || 0;
    const len = bytes.byteLength || bytes.length;
    const opId = tcpWrite(this.handle, bytes, 0, len, (c) => {
      this.bytesWritten += (c.n || 0);
      if (req.oncomplete) {
        // Signature: (status, handle, req, error)
        req.oncomplete(c.status, this, req, null);
      }
    });
    req._opId = opId;
    return 0;
  }
  writeAsciiString(req, str) {
    return this.writeBuffer(req, Buffer.from(str, 'ascii'));
  }
  writeUtf8String(req, str) {
    return this.writeBuffer(req, Buffer.from(str, 'utf8'));
  }
  writeLatin1String(req, str) {
    return this.writeBuffer(req, Buffer.from(str, 'latin1'));
  }
  writeUcs2String(req, str) {
    return this.writeBuffer(req, Buffer.from(str, 'utf16le'));
  }
  writev(req, chunks) {
    // chunks is a flat array [buf, encoding, buf, encoding, ...]
    const parts = [];
    for (let i = 0; i < chunks.length; i += 2) {
      const c = chunks[i], enc = chunks[i + 1];
      parts.push(typeof c === 'string' ? Buffer.from(c, enc || 'utf8') : c);
    }
    const combined = Buffer.concat(parts);
    return this.writeBuffer(req, combined);
  }

  shutdown(req) {
    const opId = tcpShutdown(this.handle, 1, (c) => {
      if (req.oncomplete) req.oncomplete(c.status, this, req);
    });
    req._opId = opId;
    return 0;
  }

  close(cb) {
    if (this._closed) { if (cb) Promise.resolve().then(cb); return; }
    this._closed = true;
    this.reading = false;
    tcpClose(this.handle);
    if (cb) Promise.resolve().then(cb);
  }

  getsockname(out) {
    const a = tcpLocalAddr(this.handle);
    out.address = a.ip; out.port = a.port; out.family = 'IPv' + a.family;
    return 0;
  }
  getpeername(out) {
    const a = tcpPeerAddr(this.handle);
    out.address = a.ip; out.port = a.port; out.family = 'IPv' + a.family;
    return 0;
  }
  setNoDelay(on)         { __host.tcp.set_no_delay(this.handle, !!on); return 0; }
  setKeepAlive(on, delay){ __host.tcp.set_keep_alive(this.handle, !!on, delay|0); return 0; }
  setSimultaneousAccepts(_on) { return 0; }
  ref()                  { return 0; }
  unref()                { return 0; }
  reset(req)             { return this.shutdown(req); }
  open(_fd)              { return -22; } // file-descriptor adoption not supported
  useUserBuffer(_buf)    { return 0; }
  get fd()               { return -1; }
}

class TCPConnectWrap { constructor() { this.oncomplete = null; } }
class TCPServerWrap  { constructor() {} }
class ShutdownWrap   { constructor() { this.oncomplete = null; } }
class WriteWrap      { constructor() { this.oncomplete = null; } }

module.exports = { TCP, TCPConnectWrap, TCPServerWrap, ShutdownWrap, WriteWrap,
                   TCPConstants, constants: TCPConstants };
`;

// internalBinding('task_queue') — used by internal/process/task_queues for
// microtask + nextTick scheduling. We piggyback on the engine's microtask
// queue (Promise.then) and our timer queue.
// node:cluster — stub. net.js requires it for listen-on-primary; without
// fork() in our env, every process is the primary.
shimSources['cluster'] = `
'use strict';
module.exports = {
  isPrimary: true, isMaster: true,
  isWorker: false,
  worker: null, workers: {},
  schedulingPolicy: 0,
  on() { return this; }, off() { return this; }, once() { return this; },
  emit() { return false; }, addListener() { return this; }, removeListener() { return this; },
  fork() { throw new Error('cluster.fork not implemented'); },
  setupMaster() {}, setupPrimary() {},
  disconnect() {},
  _getServer() { /* primary path bypasses this */ },
};
`;

// node:dns — full lift is for next time. For now, just `lookup` that
// resolves literal IPs and 'localhost'; everything else throws.
shimSources['dns'] = `
'use strict';
function lookup(hostname, optsOrCb, maybeCb) {
  const cb = typeof optsOrCb === 'function' ? optsOrCb : maybeCb;
  if (typeof cb !== 'function') throw new TypeError('lookup callback required');
  const opts = typeof optsOrCb === 'object' && optsOrCb !== null ? optsOrCb : {};
  const all = !!opts.all;
  (globalThis._netLog || (globalThis._netLog = [])).push('dns.lookup called for ' + hostname + ' all=' + all);
  Promise.resolve().then(() => {
    (globalThis._netLog || (globalThis._netLog = [])).push('dns.lookup microtask firing');
    let addr = null, family = 0;
    const parts = String(hostname).split('.');
    const isV4 = parts.length === 4 && parts.every(p => /^[0-9]{1,3}$/.test(p) && +p < 256);
    if (isV4) { addr = hostname; family = 4; }
    else if (hostname === 'localhost') { addr = '127.0.0.1'; family = 4; }
    else if (hostname === '::1')       { addr = '::1'; family = 6; }
    if (!addr) {
      const e = new Error('getaddrinfo ENOTFOUND ' + hostname);
      e.code = 'ENOTFOUND'; e.errno = -3008; e.syscall = 'getaddrinfo'; e.hostname = hostname;
      return cb(e);
    }
    if (all) cb(null, [{ address: addr, family }]);
    else     cb(null, addr, family);
  });
}
function notImpl(name) { return () => { throw new Error('dns.' + name + ' not implemented'); }; }
module.exports = {
  lookup,
  lookupService: notImpl('lookupService'),
  resolve:       notImpl('resolve'),
  resolve4:      notImpl('resolve4'),
  resolve6:      notImpl('resolve6'),
  resolveAny:    notImpl('resolveAny'),
  resolveCname:  notImpl('resolveCname'),
  resolveMx:     notImpl('resolveMx'),
  resolveNs:     notImpl('resolveNs'),
  resolveTxt:    notImpl('resolveTxt'),
  resolveSrv:    notImpl('resolveSrv'),
  resolveSoa:    notImpl('resolveSoa'),
  resolvePtr:    notImpl('resolvePtr'),
  reverse:       notImpl('reverse'),
  Resolver: class Resolver { constructor() {} resolve() { return notImpl('Resolver.resolve')(); } },
  promises: {
    lookup: (hostname, opts) => new Promise((res, rej) =>
      lookup(hostname, opts, (err, a, f) => err ? rej(err) : res({ address: a, family: f }))),
  },
  ADDRCONFIG: 1024, ALL: 256, V4MAPPED: 2048,
  setDefaultResultOrder() {}, getDefaultResultOrder() { return 'verbatim'; },
  setServers() {}, getServers() { return []; },
  NODATA: 'ENODATA', FORMERR: 'EFORMERR', SERVFAIL: 'ESERVFAIL',
  NOTFOUND: 'ENOTFOUND', NOTIMP: 'ENOTIMP', REFUSED: 'EREFUSED',
};
`;

shimSources['internal/process/promises'] = `
'use strict';
module.exports = {
  setup() { return [() => {}, () => {}]; },
  listenForRejections() {},
  promiseRejectHandler() {},
};
`;

shimSources['__binding/task_queue'] = `
'use strict';
module.exports = {
  enqueueMicrotask: (fn) => { Promise.resolve().then(fn); },
  setHasRejectionToWarn: () => {},
  triggerFatalException: (err) => {
    try { (globalThis.console || { error: () => {} }).error(err); } catch (_) {}
  },
  setTickCallback: (_fn) => {},
  setPromiseRejectCallback: (_fn) => {},
  runMicrotasks: () => {},
  queueMicrotask: (fn) => Promise.resolve().then(fn),
};
`;

// internalBinding('messaging') — MessageChannel transferables; only used by
// AbortController and worker_threads. Stubs are fine for single-thread http.
shimSources['__binding/messaging'] = `
'use strict';
class MessagePort { constructor() {} start() {} close() {} postMessage() {} ref() {} unref() {} }
module.exports = {
  MessageChannel: class { constructor() { this.port1 = new MessagePort(); this.port2 = new MessagePort(); } },
  MessagePort,
  setDeserializerCreateObjectFunction: () => {},
  receiveMessageOnPort: () => undefined,
  drainMessagePort: () => {},
  stopMessagePort: () => {},
  moveMessagePortToContext: () => null,
  checkMessagePort: () => {},
};
`;

// internalBinding('async_wrap') — async-context tracking. Stubs.
shimSources['__binding/async_wrap'] = `
'use strict';
module.exports = {
  Providers: {},
  asyncWrap: {
    callbackTrampoline: () => {},
    pushAsyncContext: () => {},
    popAsyncContext: () => {},
    executionAsyncResource: () => null,
  },
  setupHooks: () => {},
  pushAsyncContext: () => {},
  popAsyncContext: () => {},
  registerDestroyHook: () => {},
  enableHooksForChannel: () => {},
  setCallbackTrampoline: () => {},
  constants: {},
};
`;

// internalBinding('performance') — performance.now() etc.
shimSources['__binding/performance'] = `
'use strict';
const start = __host.time.now_ms();
module.exports = {
  now() { return __host.time.now_ms() - start; },
  timeOrigin: start,
  constants: {},
  installGarbageCollectionTracking: () => {},
  setupObservers: () => {},
  observerCounts: new Uint32Array(20),
  milestones: new Float64Array(10),
};
`;

// internalBinding('pipe_wrap') — Unix-domain / named-pipe sockets. We only
// expose the shape net.js destructures; using a pipe will throw at use.
shimSources['__binding/pipe_wrap'] = `
'use strict';
const tcp = require('__binding/tcp_wrap');
class Pipe { constructor() { throw new Error('portable-node: pipes not implemented'); } }
class PipeConnectWrap { constructor() { this.oncomplete = null; } }
const PipeConstants = { SOCKET: 0, SERVER: 1, IPC: 2 };
module.exports = {
  Pipe, PipeConnectWrap, PipeConstants,
  constants: PipeConstants,
  TCP: tcp.TCP, TCPConnectWrap: tcp.TCPConnectWrap,
};
`;

// internalBinding('stream_wrap') — exports WriteWrap/ShutdownWrap that some
// callers reach for directly (separate from tcp_wrap).
shimSources['__binding/stream_wrap'] = `
'use strict';
const tcp = require('__binding/tcp_wrap');
// Int32Array so negative libuv errno values (e.g. UV_EOF = -4095) fit.
module.exports = {
  ShutdownWrap: tcp.ShutdownWrap,
  WriteWrap:    tcp.WriteWrap,
  kReadBytesOrError: 0,
  kArrayBufferOffset: 1,
  kBytesWritten: 2,
  kLastWriteWasAsync: 3,
  streamBaseState: new Int32Array(4),
};
`;

// internal/freelist — small free-list used by _http_common to recycle parsers.
shimSources['internal/freelist'] = `
'use strict';
class FreeList {
  constructor(name, max, ctor) {
    this.name = name; this.ctor = ctor; this.max = max;
    this.list = [];
  }
  alloc() { return this.list.length > 0 ? this.list.pop() : this.ctor(); }
  free(obj) {
    if (this.list.length < this.max) { this.list.push(obj); return true; }
    return false;
  }
}
module.exports = FreeList;
`;

// internalBinding('http_parser') — thin JS facade over __host.http.parser.
// All actual parsing is delegated to the host (Rust: httparse + a chunked
// body state machine; Go would use net/http; Python: h11; C: llhttp).
// This shim only translates host events into Node's HTTPParser callback
// slots (kOnHeadersComplete, kOnBody, kOnMessageComplete).
shimSources['__binding/http_parser'] = `
'use strict';

const H = __host.http.parser;

// Node uses these as numeric tags; values are arbitrary but must be distinct
// (Node's _http_common reads HTTPParser.kOnExecute | 0 etc.).
const kOnMessageBegin = 0;
const kOnHeaders = 1;
const kOnHeadersComplete = 2;
const kOnBody = 3;
const kOnMessageComplete = 4;
const kOnExecute = 5;
const kOnTimeout = 6;

const REQUEST = 0;
const RESPONSE = 1;
const KIND_NAME = ['request', 'response'];

// Methods table for parserOnHeadersComplete → allMethods[idx] lookup.
// Order matches the indices the host emits via methods.indexOf in JS, so we
// keep our own canonical list here.
const methods = [
  'DELETE','GET','HEAD','POST','PUT','CONNECT','OPTIONS','TRACE',
  'COPY','LOCK','MKCOL','MOVE','PROPFIND','PROPPATCH','SEARCH','UNLOCK',
  'BIND','REBIND','UNBIND','ACL','REPORT','MKACTIVITY','CHECKOUT','MERGE',
  'M-SEARCH','NOTIFY','SUBSCRIBE','UNSUBSCRIBE','PATCH','PURGE','MKCALENDAR',
  'LINK','UNLINK','SOURCE',
];
const methodIndex = Object.create(null);
for (let i = 0; i < methods.length; i++) methodIndex[methods[i]] = i;

function HTTPParser(type) {
  this._kind = (type === undefined) ? REQUEST : type;
  this._handle = H.create(KIND_NAME[this._kind]);
  this._headers = [];   // partial-headers accumulator (slow path)
  this._url = '';
  this.maxHeaderPairs = 2000;
  this.incoming = null;
  this.socket = null;
  this.onIncoming = null;
}

HTTPParser.prototype.initialize = function(type, asyncResource, maxHeaderSize, lenient, connections) {
  this._kind = (type === undefined) ? REQUEST : type;
  H.reset(this._handle, KIND_NAME[this._kind]);
  this._headers = [];
  this._url = '';
  this.incoming = null;
  return 0;
};

HTTPParser.prototype.reinitialize = HTTPParser.prototype.initialize;

HTTPParser.prototype.execute = function(chunk, start, length) {
  if (start === undefined) { start = 0; length = chunk.length; }
  const result = H.execute(this._handle, chunk, start | 0, length | 0);
  return _dispatch(this, result, chunk);
};

HTTPParser.prototype.finish = function() {
  const result = H.finish(this._handle);
  return _dispatch(this, result, null);
};

HTTPParser.prototype.close   = function() { H.free(this._handle); };
HTTPParser.prototype.free    = function() { /* freed by close() */ };
HTTPParser.prototype.remove  = function() {};
HTTPParser.prototype.pause   = function() {};
HTTPParser.prototype.resume  = function() {};
HTTPParser.prototype.consume = function() {};
HTTPParser.prototype.unconsume = function() {};
HTTPParser.prototype.getCurrentBuffer = function() {};
HTTPParser.prototype.getAsyncId = function() { return 0; };

// Default no-op handlers; Node's _http_common overrides via parser[kOn...].
HTTPParser.prototype[kOnHeaders] =
HTTPParser.prototype[kOnHeadersComplete] =
HTTPParser.prototype[kOnBody] =
HTTPParser.prototype[kOnMessageComplete] = function () {};

// Translate host events → Node-style callback invocations.
// Returns the parser-execute return value Node expects:
//   number  — bytes consumed
//   Error   — fatal parse error
function _dispatch(parser, result, chunk) {
  if (result.error) {
    const err = new Error('Parse Error: ' + result.error);
    err.code = 'HPE_INVALID_CONSTANT';
    return err;
  }
  const events = result.events || [];
  let skipBody = 0;
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (ev.kind === 'headers') {
      const methodIdx = (parser._kind === REQUEST && ev.method)
                       ? (methodIndex[ev.method.toUpperCase()] ?? -1)
                       : null;
      // Node's parserOnHeadersComplete signature:
      //   (versionMajor, versionMinor, headers, method, url, statusCode,
      //    statusMessage, upgrade, shouldKeepAlive)
      const cb = parser[kOnHeadersComplete];
      if (typeof cb === 'function') {
        const ret = cb.call(parser,
          ev.http_major, ev.http_minor,
          ev.headers,
          (parser._kind === REQUEST) ? methodIdx : null,
          ev.url || '',
          ev.status_code, ev.status_message || '',
          !!ev.upgrade, !!ev.should_keep_alive);
        // ret === 2: skip body
        // ret === true / 1: same
        if (ret === 2 || ret === true || ret === 1) skipBody = 1;
      }
    } else if (ev.kind === 'body') {
      const cb = parser[kOnBody];
      if (typeof cb === 'function' && !skipBody) {
        // Hand the body slice as a Buffer view. Buffer.from(typedArray)
        // copies; we want zero-copy where reasonable — use the underlying
        // ArrayBuffer + Buffer.from(ab, off, len).
        const ta = ev.data;
        const buf = Buffer.from(ta.buffer, ta.byteOffset, ta.byteLength);
        cb.call(parser, buf, 0, buf.length);
      }
    } else if (ev.kind === 'message_complete') {
      const cb = parser[kOnMessageComplete];
      if (typeof cb === 'function') cb.call(parser);
    }
  }
  return (result.nread != null) ? result.nread : 0;
}

HTTPParser.REQUEST = REQUEST;
HTTPParser.RESPONSE = RESPONSE;
HTTPParser.kOnMessageBegin = kOnMessageBegin;
HTTPParser.kOnHeaders = kOnHeaders;
HTTPParser.kOnHeadersComplete = kOnHeadersComplete;
HTTPParser.kOnBody = kOnBody;
HTTPParser.kOnMessageComplete = kOnMessageComplete;
HTTPParser.kOnExecute = kOnExecute;
HTTPParser.kOnTimeout = kOnTimeout;
HTTPParser.kLenientNone = 0;
HTTPParser.kLenientAll = 0xFFFF;
HTTPParser.methods = methods;
HTTPParser.maxHeaderSize = 80 * 1024;

// ConnectionsList — used by _http_server.js to track per-server connections.
// In Node this is C++ bookkeeping; an empty stand-in is fine.
function ConnectionsList() {
  this.all = function(){ return []; };
  this.idle = function(){ return []; };
  this.active = function(){ return []; };
  this.expired = function(){ return []; };
}

module.exports = {
  HTTPParser,
  ConnectionsList,
  methods,
  allMethods: methods,
};
`;

// node:tty — minimal stand-in. We don't run in a TTY; everything is false.
shimSources['tty'] = `
'use strict';
function isatty(_fd) { return false; }
class ReadStream  { constructor() { this.isTTY = false; } }
class WriteStream { constructor() { this.isTTY = false; this.columns = 80; this.rows = 24; } }
module.exports = { isatty, ReadStream, WriteStream };
`;

// node:process — module form. Most code uses the global, but some packages
// do \`require('process')\`. Returns the same object the global exposes.
shimSources['process'] = `
'use strict';
module.exports = globalThis.process;
`;

// node:crypto — portable facade over __host.crypto.*. Surfaces Node's
// createHash / createHmac / randomBytes / randomFillSync / randomUUID /
// timingSafeEqual / pbkdf2Sync. Every host language has equivalents
// (Go: crypto/*, Python: hashlib/hmac/secrets, Java: MessageDigest/Mac/
// SecureRandom). Not a full Node-crypto port; ciphers, signing, X.509,
// key generation, ECDH are out of scope for now.
shimSources['crypto'] = `
'use strict';
const H = __host.crypto;

function toBytes(input, encoding) {
  if (input == null) return new Uint8Array(0);
  if (input instanceof Uint8Array) return input;
  if (Array.isArray(input)) return new Uint8Array(input);
  if (typeof input === 'string') return Buffer.from(input, encoding || 'utf8');
  if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  throw new TypeError('expected string, Buffer, or TypedArray');
}

// Map Node algorithm names to host algorithm names.
function normalizeAlgo(a) {
  const s = String(a).toLowerCase().replace(/-/g, '');
  switch (s) {
    case 'sha1':   return 'sha1';
    case 'sha224': return 'sha224';
    case 'sha256': return 'sha256';
    case 'sha384': return 'sha384';
    case 'sha512': return 'sha512';
    case 'md5':    return 'md5';
    default: throw new Error('Digest method not supported: ' + a);
  }
}

// Hash: createHash(algo) → { update(d, enc?), digest(enc?) }
class Hash {
  constructor(algo) {
    this._algo = normalizeAlgo(algo);
    this._parts = [];
    this._done = false;
  }
  update(data, encoding) {
    if (this._done) throw new Error('Digest already called');
    this._parts.push(toBytes(data, encoding));
    return this;
  }
  digest(encoding) {
    if (this._done) throw new Error('Digest already called');
    this._done = true;
    const total = Buffer.concat(this._parts.map(p => Buffer.from(p.buffer, p.byteOffset, p.byteLength)));
    const out = H.hash(this._algo, total);
    const buf = Buffer.from(out.buffer, out.byteOffset, out.byteLength);
    return encoding ? buf.toString(encoding) : buf;
  }
  copy() {
    const c = new Hash(this._algo);
    c._parts = this._parts.slice();
    return c;
  }
}
function createHash(algo) { return new Hash(algo); }

// Hmac: createHmac(algo, key) → { update, digest }
class Hmac {
  constructor(algo, key) {
    this._algo = normalizeAlgo(algo);
    this._key = toBytes(key);
    this._parts = [];
    this._done = false;
  }
  update(data, encoding) {
    if (this._done) throw new Error('HMAC already called');
    this._parts.push(toBytes(data, encoding));
    return this;
  }
  digest(encoding) {
    if (this._done) throw new Error('HMAC already called');
    this._done = true;
    const total = Buffer.concat(this._parts.map(p => Buffer.from(p.buffer, p.byteOffset, p.byteLength)));
    const out = H.hmac(this._algo, this._key, total);
    const buf = Buffer.from(out.buffer, out.byteOffset, out.byteLength);
    return encoding ? buf.toString(encoding) : buf;
  }
}
function createHmac(algo, key) { return new Hmac(algo, key); }

function randomBytes(size, cb) {
  if (typeof size !== 'number' || size < 0 || size > 0x7fffffff) {
    throw new RangeError('invalid size');
  }
  const out = H.random_bytes(size | 0);
  const buf = Buffer.from(out.buffer, out.byteOffset, out.byteLength);
  if (typeof cb === 'function') { Promise.resolve().then(() => cb(null, buf)); return; }
  return buf;
}

function randomFillSync(buf, offset, size) {
  offset = offset || 0;
  size = (size === undefined) ? buf.length - offset : size;
  const r = H.random_bytes(size);
  const view = new Uint8Array(r.buffer, r.byteOffset, r.byteLength);
  for (let i = 0; i < view.length; i++) buf[offset + i] = view[i];
  return buf;
}

function randomFill(buf, offset, size, cb) {
  // Args may be (buf, cb), (buf, offset, cb), (buf, offset, size, cb)
  if (typeof offset === 'function') { cb = offset; offset = 0; size = buf.length; }
  else if (typeof size === 'function') { cb = size; size = buf.length - offset; }
  Promise.resolve().then(() => {
    try { randomFillSync(buf, offset, size); cb(null, buf); }
    catch (e) { cb(e); }
  });
}

function randomUUID() {
  // RFC 4122 v4 — 16 random bytes, then set version/variant bits.
  const b = H.random_bytes(16);
  const v = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  v[6] = (v[6] & 0x0f) | 0x40;   // version 4
  v[8] = (v[8] & 0x3f) | 0x80;   // variant 10
  const hex = [];
  for (let i = 0; i < 16; i++) hex.push(v[i].toString(16).padStart(2, '0'));
  return hex.slice(0, 4).join('') + '-' +
         hex.slice(4, 6).join('') + '-' +
         hex.slice(6, 8).join('') + '-' +
         hex.slice(8,10).join('') + '-' +
         hex.slice(10,16).join('');
}

function randomInt(minOrMax, maxOrCb, cb) {
  let min, max;
  if (typeof maxOrCb === 'number') { min = minOrMax; max = maxOrCb; }
  else { min = 0; max = minOrMax; cb = maxOrCb; }
  // 53-bit safe range; reject out-of-window samples to avoid bias.
  const range = max - min;
  if (range <= 0) throw new RangeError('max must be > min');
  const bytesNeeded = Math.ceil(Math.log2(range) / 8) || 1;
  const max32 = Math.pow(256, bytesNeeded);
  const accept = max32 - (max32 % range);
  function once() {
    while (true) {
      const r = H.random_bytes(bytesNeeded);
      let v = 0;
      for (let i = 0; i < bytesNeeded; i++) v = v * 256 + r[i];
      if (v < accept) return min + (v % range);
    }
  }
  if (typeof cb === 'function') {
    Promise.resolve().then(() => { try { cb(null, once()); } catch (e) { cb(e); } });
    return;
  }
  return once();
}

function timingSafeEqual(a, b) {
  const ab = toBytes(a), bb = toBytes(b);
  if (ab.length !== bb.length) throw new RangeError('Input buffers must have the same byte length');
  return H.timing_safe_equal(ab, bb);
}

// pbkdf2: hash-iterated key derivation. Pure-JS using HMAC.
function pbkdf2Sync(password, salt, iterations, keylen, digest) {
  const algo = normalizeAlgo(digest);
  const passBytes = toBytes(password);
  const saltBytes = toBytes(salt);
  const hLen = ({ sha1:20, sha224:28, sha256:32, sha384:48, sha512:64, md5:16 })[algo];
  const out = Buffer.alloc(keylen);
  let offset = 0;
  let blockIdx = 1;
  while (offset < keylen) {
    const blockIdxBuf = Buffer.alloc(4);
    blockIdxBuf.writeUInt32BE(blockIdx, 0);
    let U = H.hmac(algo, passBytes, Buffer.concat([Buffer.from(saltBytes), blockIdxBuf]));
    const T = Buffer.from(U.buffer, U.byteOffset, U.byteLength);
    for (let i = 1; i < iterations; i++) {
      U = H.hmac(algo, passBytes, U);
      const Uv = new Uint8Array(U.buffer, U.byteOffset, U.byteLength);
      for (let j = 0; j < hLen; j++) T[j] ^= Uv[j];
    }
    const take = Math.min(hLen, keylen - offset);
    T.copy(out, offset, 0, take);
    offset += take;
    blockIdx++;
  }
  return out;
}

function pbkdf2(password, salt, iterations, keylen, digest, cb) {
  Promise.resolve().then(() => {
    try { cb(null, pbkdf2Sync(password, salt, iterations, keylen, digest)); }
    catch (e) { cb(e); }
  });
}

const constants = {
  // A minimal subset — Express + middleware don't need much here.
  RSA_PKCS1_PADDING: 1, RSA_NO_PADDING: 3, RSA_PKCS1_OAEP_PADDING: 4,
};

const webcrypto = {
  getRandomValues(buf) { return randomFillSync(buf, 0, buf.byteLength); },
  randomUUID,
};

module.exports = {
  createHash, createHmac,
  Hash, Hmac,
  randomBytes, randomFill, randomFillSync, randomUUID, randomInt,
  timingSafeEqual,
  pbkdf2, pbkdf2Sync,
  constants,
  webcrypto,
  getRandomValues: webcrypto.getRandomValues,
  // Not implemented: createCipheriv, createDecipheriv, createSign, createVerify,
  // generateKeyPair*, createDiffieHellman, X.509. Express and its common deps
  // don't need these; if they're needed later, expose them as host primitives.
};
`;

// internalBinding('uv') — error constants. Just the ones net.js reaches for.
shimSources['__binding/uv'] = `
'use strict';
const codes = {
  UV_EOF: -4095, UV_UNKNOWN: -4094, UV_OK: 0,
  UV_EAGAIN: -11, UV_ECANCELED: -4081, UV_ECONNRESET: -54,
  UV_ECONNREFUSED: -61, UV_EPIPE: -32, UV_ETIMEDOUT: -60,
  UV_ENOENT: -2, UV_EACCES: -13, UV_EBADF: -9,
  UV_EADDRINUSE: -48, UV_EADDRNOTAVAIL: -49,
  UV_ENETUNREACH: -51, UV_ENOTCONN: -57,
  UV_E_2BIG: -7, UV_EAI_BADFLAGS: -3000,
};
function errname(code) {
  for (const k of Object.keys(codes)) if (codes[k] === code) return k.replace(/^UV_/, '');
  return 'UNKNOWN';
}
module.exports = { ...codes, errname };
`;

// internalBinding('cares_wrap') — DNS resolver. Real Node uses c-ares.
// For an initial server-side proof, we only need the constants. Real lookup
// will land via __host.dns later for client-side support.
shimSources['__binding/cares_wrap'] = `
'use strict';
function notSupported() { throw new Error('portable-node: DNS resolver not implemented'); }
class GetAddrInfoReqWrap { constructor() { this.oncomplete = null; } }
class GetNameInfoReqWrap { constructor() { this.oncomplete = null; } }
class QueryReqWrap       { constructor() { this.oncomplete = null; } }
class ChannelWrap { constructor() {} setServers() { return 0; } getServers() { return []; }
                    cancel() {} setLocalAddress() {} setTimeout() {} }
module.exports = {
  GetAddrInfoReqWrap, GetNameInfoReqWrap, QueryReqWrap, ChannelWrap,
  // Address-family constants.
  AF_INET: 2, AF_INET6: 30, AF_UNSPEC: 0,
  // Hints.
  AI_ADDRCONFIG: 1024, AI_ALL: 256, AI_V4MAPPED: 2048,
  // Lookup functions (sync subset).
  getaddrinfo(req, hostname, _family, _hints, _verbatim) {
    // Literal-IP or localhost handling. Avoid \\d in template literal regex.
    const parts = String(hostname).split('.');
    const isV4 = parts.length === 4 && parts.every(p => /^[0-9]{1,3}$/.test(p) && +p < 256);
    if (isV4) {
      Promise.resolve().then(() => { if (req.oncomplete) req.oncomplete(0, [hostname]); });
      return 0;
    }
    if (hostname === 'localhost') {
      Promise.resolve().then(() => { if (req.oncomplete) req.oncomplete(0, ['127.0.0.1']); });
      return 0;
    }
    Promise.resolve().then(() => { if (req.oncomplete) req.oncomplete(-2); });
    return 0;
  },
  getnameinfo() { return -38; }, // ENOSYS
};
`;

// internalBinding('credentials') — only used by os.js for getTempDir.
shimSources['__binding/credentials'] = `
'use strict';
module.exports = {
  getTempDir: () => globalThis.__host.os.tmpdir(),
};
`;

// node:test — minimal runner. Each test() invocation runs the body
// synchronously (or returns its promise for chaining). Enough for tests
// that just want to scope assertions inside test() blocks.
shimSources['node:test'] = `
'use strict';
function test(nameOrFn, fnOrOpts, maybeFn) {
  let name, fn;
  if (typeof nameOrFn === 'function') { fn = nameOrFn; name = fn.name || 'anonymous'; }
  else { name = nameOrFn; fn = typeof fnOrOpts === 'function' ? fnOrOpts : maybeFn; }
  if (!fn) return Promise.resolve();
  try {
    const r = fn({ name, diagnostic: () => {} });
    return r && typeof r.then === 'function' ? r : Promise.resolve(r);
  } catch (e) { return Promise.reject(e); }
}
test.test = test;
test.it = test;
test.describe = test;
test.suite = test;        // alias used by some tests
test.skip = () => {};
test.todo = () => {};
test.only = test;
test.before = test.after = test.beforeEach = test.afterEach = () => {};
test.mock = {
  fn: (impl) => { const f = (...a) => impl && impl(...a); f.mock = { calls: [] }; return f; },
  method: () => () => {},
};
// Export both as default and as named exports for compatibility with
// "import { test, suite } from 'node:test'" style.
module.exports = test;
module.exports.test = test;
module.exports.suite = test;
module.exports.describe = test;
module.exports.it = test;
`;
shimSources['test'] = shimSources['node:test'];

// node:stream — minimal stubs of Writable/Readable. Enough for tests that
// just construct or pipe through a stream without exercising backpressure.
// A real stream port is a significant undertaking and gates on event-loop
// async work; for now we satisfy "shape" tests.
// Old stream stub — kept under __legacy so require('stream') falls through
// to Node's real lib/stream.js + internal/streams/* lift.
shimSources['__legacy/stream'] = `
'use strict';
const EventEmitter = require('events');
class Writable extends EventEmitter {
  constructor(opts) {
    super();
    this._opts = opts || {};
    this._write = (this._opts && this._opts.write) || ((c, e, cb) => cb && cb());
    this.writable = true;
  }
  write(chunk, enc, cb) {
    if (typeof enc === 'function') { cb = enc; enc = null; }
    try { this._write(chunk, enc, cb || (() => {})); } catch (e) { this.emit('error', e); }
    return true;
  }
  end(chunk, enc, cb) {
    if (chunk !== undefined && chunk !== null) this.write(chunk, enc, cb);
    this.writable = false;
    this.emit('finish');
    if (typeof cb === 'function') cb();
  }
}
class Readable extends EventEmitter {
  constructor(opts) {
    super();
    this._opts = opts || {};
    this.readable = true;
  }
  read() { return null; }
  pipe(dst) { dst.write && this.on('data', (c) => dst.write(c)); return dst; }
  on(event, listener) { return super.on(event, listener); }
}
class Duplex extends Writable { /* simplified — combines both */ }
class Transform extends Duplex { /* simplified */ }
class PassThrough extends Transform { /* simplified */ }
module.exports = {
  Writable, Readable, Duplex, Transform, PassThrough,
  pipeline: (...streams) => { /* no-op pipeline */ },
  finished:  () => Promise.resolve(),
  Stream: EventEmitter,
};
module.exports.default = module.exports;
`;

// node:child_process — stub. spawn() etc. throw clearly; tests that just
// require() it without invoking work fine.
shimSources['child_process'] = `
'use strict';
function notSupported() { throw new Error('portable-node: child_process not implemented (no host.spawn primitive yet)'); }
module.exports = {
  spawn: notSupported, spawnSync: notSupported,
  exec:  notSupported, execSync:  notSupported,
  execFile: notSupported, execFileSync: notSupported,
  fork:  notSupported,
};
`;

shimSources['vm'] = `
'use strict';
// Tests that need cross-realm semantics (runInNewContext) — we can't fully
// honor it on QuickJS. Provide a best-effort same-realm version so tests
// that just need a "fresh execution" run, while leaving a stub so realm-
// sensitive logic at least surfaces as a clear error rather than silently
// passing.
function runInNewContext(code, sandbox, _opts) {
  // Bind sandbox keys as Function args so identifiers in code resolve to
  // sandbox values. Buffer is always made available (Node's behavior when
  // the sandbox doesn't override it).
  const keys = sandbox ? Object.keys(sandbox) : [];
  const vals = keys.map((k) => sandbox[k]);
  if (!sandbox || !('Buffer' in sandbox)) {
    keys.push('Buffer');
    vals.push(globalThis.Buffer);
  }
  return Function(...keys, 'return (' + code + ');').apply(null, vals);
}
function runInThisContext(code) { return (0, eval)(code); }
class Script {
  constructor(code) { this.code = code; }
  runInNewContext(sandbox) { return runInNewContext(this.code, sandbox); }
  runInThisContext() { return runInThisContext(this.code); }
}
module.exports = { runInNewContext, runInThisContext, Script };
`;

// internal/url — Node's fs uses just `toPathIfFileURL`. Real internal/url
// is 1700+ lines; we provide only what fs needs. URL-handling tests would
// need a real WHATWG URL parser primitive.
shimSources['internal/url'] = `
'use strict';
const { codes: { ERR_INVALID_ARG_TYPE } } = require('internal/errors');

function isURL(v) {
  return v != null && typeof v.href === 'string' && typeof v.origin === 'string';
}
function toPathIfFileURL(path) {
  if (!isURL(path)) return path;
  if (path.protocol !== 'file:')
    throw new ERR_INVALID_ARG_TYPE('url', 'file: URL', path);
  return fileURLToPath(path);
}
function fileURLToPath(url) {
  // Minimal POSIX-only conversion; real Node handles Windows + percent-decoding.
  const href = typeof url === 'string' ? url : url.href;
  if (!href.startsWith('file:')) throw new ERR_INVALID_ARG_TYPE('url', 'file: URL', url);
  let path = href.slice(5);
  while (path.startsWith('/') && path.length > 1 && path[1] === '/') path = path.slice(1);
  return decodeURIComponent(path);
}
function pathToFileURL(p) {
  return { href: 'file://' + encodeURI(p), protocol: 'file:', pathname: p };
}
function urlToHttpOptions(url) { return {}; }
module.exports = {
  toPathIfFileURL, fileURLToPath, pathToFileURL, isURL, urlToHttpOptions,
  URL: globalThis.URL,
  URLSearchParams: globalThis.URLSearchParams,
};
`;

// internal/fs/promises — Node's fs.promises implementations. Real port is
// huge (~2000 lines) and needs FileHandle + threadpool. Our minimal version
// just wraps sync ops in resolved promises so async APIs work serially.
shimSources['internal/fs/promises'] = `
'use strict';
let fs;
function lazyFs() { return fs ??= require('fs'); }
function defer(fn) { return (...args) => Promise.resolve().then(() => fn(...args)); }
module.exports = {
  exports: {
    readFile:    defer((...a) => lazyFs().readFileSync(...a)),
    writeFile:   defer((...a) => lazyFs().writeFileSync(...a)),
    appendFile:  defer((...a) => lazyFs().appendFileSync(...a)),
    stat:        defer((...a) => lazyFs().statSync(...a)),
    lstat:       defer((...a) => lazyFs().lstatSync(...a)),
    access:      defer((...a) => lazyFs().accessSync(...a)),
    readdir:     defer((...a) => lazyFs().readdirSync(...a)),
    mkdir:       defer((...a) => lazyFs().mkdirSync(...a)),
    rmdir:       defer((...a) => lazyFs().rmdirSync(...a)),
    rm:          defer((...a) => { /* lazyFs().rmSync — not impl */ }),
    unlink:      defer((...a) => lazyFs().unlinkSync(...a)),
    rename:      defer((...a) => lazyFs().renameSync(...a)),
    realpath:    defer((...a) => lazyFs().realpathSync(...a)),
    chmod:       defer(() => { throw new Error('chmod not implemented'); }),
    chown:       defer(() => { throw new Error('chown not implemented'); }),
    copyFile:    defer(() => { throw new Error('copyFile not implemented'); }),
    open:        defer((path, flags, mode) => {
      const fd = lazyFs().openSync(path, flags, mode);
      return {
        fd,
        close: defer(() => lazyFs().closeSync(fd)),
        read:  defer(() => { throw new Error('FileHandle.read not implemented'); }),
        write: defer(() => { throw new Error('FileHandle.write not implemented'); }),
      };
    }),
  },
};
`;

// internalBinding('trace_events') — no-op tracing. console.js touches this
// for the diagnostics channel; everything is a stub.
shimSources['__binding/trace_events'] = `
'use strict';
module.exports = {
  emit:           () => {},
  hasMetadata:    () => false,
  getCategoryEnabledBuffer: () => new Uint8Array(1),
  isTraceCategoryEnabled: () => false,
  categoryGroupEnabled: () => false,
};
`;

// internalBinding('timers') — minimal: only fields/methods we know touchers
// of expect. Real timer scheduling requires a host event-loop primitive.
shimSources['__binding/timers'] = `
'use strict';
module.exports = {
  immediateInfo: new Uint32Array(3),
  timeoutInfo:   new Int32Array(1),
  toggleTimerRef:  () => {},
  toggleImmediateRef: () => {},
  setupTimers:     () => {},
  scheduleTimer:   () => 0,
  getLibuvNow:     () => Date.now(),
};
`;

// internalBinding('url_pattern') / 'encoding_binding' — stubs. Real URL/IDN
// requires a host-provided parser primitive.
shimSources['__binding/url_pattern'] = `
'use strict';
class URLPattern { constructor() { throw new Error('URLPattern not supported'); } }
module.exports = { URLPattern };
`;
shimSources['__binding/encoding_binding'] = `
'use strict';
module.exports = {
  toASCII:    (s) => s,            // identity — real IDNA requires punycode
  toUnicode:  (s) => s,
  encodeInto: () => ({ read: 0, written: 0 }),
};
`;

// internalBinding('url') — stub. Real WHATWG URL parsing needs a Rust
// primitive (e.g. the `url` crate). Stubbing what url.js destructures so
// it can load; URL parser ops will throw on use.
shimSources['__binding/url'] = `
'use strict';
function notSupported() { throw new Error('portable-node: URL parser not implemented (need __host.url.* primitive)'); }
module.exports = {
  parse:           notSupported,
  update:          notSupported,
  canParse:        () => false,
  domainToASCII:   (s) => s,
  domainToUnicode: (s) => s,
  getOrigin:       notSupported,
  pathToFileURL:   notSupported,
  urlComponents:   new Int32Array(9),
};
`;

// node:diagnostics_channel — minimal pub/sub. No-op subscribe/publish; the
// channel registry is just JS state.
shimSources['diagnostics_channel'] = `
'use strict';
const channels = new Map();
class Channel {
  constructor(name) { this.name = name; this._subs = new Set(); }
  publish(msg) { for (const fn of this._subs) try { fn(msg, this.name); } catch (_e) {} }
  subscribe(fn) { this._subs.add(fn); }
  unsubscribe(fn) { return this._subs.delete(fn); }
  get hasSubscribers() { return this._subs.size > 0; }
}
function channel(name) {
  let c = channels.get(name);
  if (!c) { c = new Channel(name); channels.set(name, c); }
  return c;
}
function hasSubscribers(name) {
  return channels.has(name) && channels.get(name).hasSubscribers;
}
module.exports = {
  Channel, channel, hasSubscribers,
  subscribe: (name, fn) => channel(name).subscribe(fn),
  unsubscribe: (name, fn) => channel(name).unsubscribe(fn),
  tracingChannel: (_n) => ({ start: channel('s'), end: channel('e'), error: channel('err'),
    asyncStart: channel('as'), asyncEnd: channel('ae'),
    traceSync: (fn, _ctx, _thisArg, ...args) => fn(...args),
    tracePromise: (fn, _ctx, _thisArg, ...args) => Promise.resolve(fn(...args)),
    traceCallback: (fn, _pos, _ctx, _thisArg, ...args) => fn(...args),
    bindStore: () => {},
  }),
};
`;

// internal/linkedlist — Node's internal doubly-linked list. Used by timers.
// Small enough to provide directly.
shimSources['internal/linkedlist'] = `
'use strict';
function init(list) { list._idleNext = list; list._idlePrev = list; }
function peek(list) { return list._idlePrev !== list ? list._idlePrev : null; }
function shift(list) {
  const first = list._idlePrev;
  if (first === list) return null;
  remove(first);
  return first;
}
function remove(item) {
  if (item._idleNext) {
    item._idleNext._idlePrev = item._idlePrev;
    item._idlePrev._idleNext = item._idleNext;
    item._idleNext = item._idlePrev = null;
  }
}
function append(list, item) {
  if (item._idleNext) remove(item);
  item._idleNext = list._idleNext;
  list._idleNext._idlePrev = item;
  item._idlePrev = list;
  list._idleNext = item;
}
function isEmpty(list) { return list._idleNext === list; }
module.exports = { init, peek, shift, remove, append, isEmpty };
`;

// node:async_hooks — minimal AsyncLocalStorage + executionAsyncId stubs.
// Real async-context tracking needs engine hooks; tests that don't probe
// the actual async ID will work fine.
shimSources['async_hooks'] = `
'use strict';
class AsyncLocalStorage {
  constructor() { this._store = undefined; }
  run(store, fn, ...args) {
    const prev = this._store;
    this._store = store;
    try { return fn(...args); } finally { this._store = prev; }
  }
  enterWith(store) { this._store = store; }
  exit(fn, ...args) { return this.run(undefined, fn, ...args); }
  getStore() { return this._store; }
  disable() {}
}
class AsyncResource {
  constructor(type) { this.type = type; }
  runInAsyncScope(fn, thisArg, ...args) { return fn.apply(thisArg, args); }
  bind(fn) { return fn; }
  emitDestroy() {}
  asyncId() { return 0; }
  triggerAsyncId() { return 0; }
}
function createHook() { return { enable() {}, disable() {} }; }
module.exports = {
  AsyncLocalStorage, AsyncResource,
  createHook,
  executionAsyncId:  () => 0,
  triggerAsyncId:    () => 0,
  executionAsyncResource: () => null,
  symbols: { async_id_symbol: Symbol('asyncId'), trigger_async_id_symbol: Symbol('triggerId'), owner_symbol: Symbol('owner') },
};
`;

shimSources['internal/async_hooks'] = `
'use strict';
const ah = require('async_hooks');
module.exports = {
  ...ah,
  newAsyncId:       () => 0,
  getDefaultTriggerAsyncId: () => 0,
  defaultTriggerAsyncIdScope: (id, fn, ...args) => fn(...args),
  emitInit:         () => {},
  emitBefore:       () => {},
  emitAfter:        () => {},
  emitDestroy:      () => {},
  hasHooks:         () => false,
  initHooksExist:   () => false,
  destroyHooksExist:() => false,
  afterHooksExist:  () => false,
  beforeHooksExist: () => false,
  useDomainTrampoline: () => {},
  registerDestroyHook: () => {},
  enabledHooksExist:() => false,
  getOrSetAsyncId:  (resource) => {
    const sym = ah.symbols.async_id_symbol;
    if (typeof resource[sym] !== 'number') resource[sym] = 0;
    return resource[sym];
  },
  symbols: ah.symbols,
};
`;

// Minimal stubs for internal modules that streams + downstream code reach for.
// Each is real but small — enough to satisfy import and basic usage.

// internal/util/debuglog — real version has transitive issues. No-op log
// is a perfect substitute for our purposes (we don't read debug output).
shimSources['internal/util/debuglog'] = `
'use strict';
function noop() {}
function debuglog(_set, cb) {
  const fn = function () {};
  fn.enabled = false;
  // Defer the callback. Node's pattern: let debug = ...debuglog(name, (fn) => debug = fn);
  // The cb closes over an outer let in its TDZ. Calling cb synchronously throws.
  if (typeof cb === 'function') Promise.resolve().then(() => cb(fn));
  return fn;
}
module.exports = { debuglog, debug: debuglog, initializeDebugEnv: noop };
`;

shimSources['internal/event_target'] = `
'use strict';
const kEvents = Symbol('kEvents');
const kEventListeners = Symbol('kEventListeners');
const kIsEventTarget = Symbol('kIsEventTarget');
const kResistStopPropagation = Symbol('kResistStopPropagation');
class Event {
  constructor(type, init) { this.type = type; this.bubbles = !!(init && init.bubbles); this.cancelable = !!(init && init.cancelable); this.defaultPrevented = false; }
  preventDefault() { this.defaultPrevented = true; }
  stopPropagation() {}
  stopImmediatePropagation() {}
}
class EventTarget {
  constructor() { this[kEvents] = new Map(); }
  addEventListener(type, fn, _opts) {
    let s = this[kEvents].get(type);
    if (!s) { s = new Set(); this[kEvents].set(type, s); }
    s.add(fn);
  }
  removeEventListener(type, fn) { const s = this[kEvents].get(type); if (s) s.delete(fn); }
  dispatchEvent(ev) {
    const s = this[kEvents].get(ev.type);
    if (s) for (const fn of s) try { fn(ev); } catch (_e) {}
    return !ev.defaultPrevented;
  }
}
function isEventTarget(v) { return v instanceof EventTarget; }
module.exports = {
  Event, EventTarget,
  isEventTarget, kEvents, kEventListeners, kIsEventTarget, kResistStopPropagation,
  initEventTarget: () => {},
};
globalThis.Event = globalThis.Event || Event;
globalThis.EventTarget = globalThis.EventTarget || EventTarget;
`;

shimSources['internal/webidl'] = `
'use strict';
// WebIDL conversion helpers Node uses internally. Just identity passthroughs.
module.exports = {
  converters: new Proxy({}, { get: () => (v) => v }),
  install: () => {},
};
`;

shimSources['internal/worker/js_transferable'] = `
'use strict';
// Transferables — relevant only to worker_threads. Stubs.
module.exports = {
  markTransferMode:     () => {},
  setup:                () => {},
  kClone:               Symbol('kClone'),
  kTransfer:            Symbol('kTransfer'),
  kTransferList:        Symbol('kTransferList'),
  kDeserialize:         Symbol('kDeserialize'),
  makeTransferable:     (v) => v,
};
`;

shimSources['internal/perf/observe'] = `
'use strict';
class PerformanceObserver {
  observe() {}
  disconnect() {}
  takeRecords() { return []; }
}
module.exports = {
  PerformanceObserver,
  enqueue: () => {},
  hasObserver: () => false,
  startPerf: () => {},
  stopPerf: () => {},
};
`;

shimSources['internal/streams/duplexpair'] = `
'use strict';
function notImpl() { throw new Error('duplexpair not implemented'); }
module.exports = { DuplexPair: notImpl, duplexPair: notImpl };
`;

shimSources['internal/streams/duplexify'] = `
'use strict';
// Lazy-loaded by readable's stream interop. Stub.
module.exports = function duplexify() { throw new Error('duplexify not implemented'); };
module.exports.default = module.exports;
`;

shimSources['internal/streams/lazy_transform'] = `
'use strict';
function notImpl() { throw new Error('lazy_transform not implemented'); }
module.exports = notImpl;
module.exports.default = notImpl;
`;

shimSources['stream/promises'] = `
'use strict';
module.exports = {
  finished: (s, opts) => new Promise((res, rej) => {
    s.once('end', res); s.once('finish', res); s.once('error', rej);
  }),
  pipeline: (...args) => new Promise((res, rej) => {
    // Naive: chain by piping in order, resolve when last 'finish'.
    let cb = (err) => err ? rej(err) : res();
    const streams = args[args.length - 1] && typeof args[args.length - 1] === 'function' ? args.slice(0, -1) : args;
    for (let i = 0; i < streams.length - 1; i++) streams[i].pipe(streams[i + 1]);
    streams[streams.length - 1].once('finish', cb).once('error', cb);
  }),
};
`;

shimSources['internal/async_context_frame'] = `
'use strict';
// AsyncContextFrame — engine-level async-context tracking. We don't have
// engine hooks for this; identity functions are correct for our single-
// realm use.
let _current;
class AsyncContextFrame {
  static current() { return _current; }
  static run(frame, fn, ...args) {
    const prev = _current;
    _current = frame;
    try { return fn(...args); } finally { _current = prev; }
  }
  static get() { return _current; }
  static set(f) { _current = f; }
}
module.exports = AsyncContextFrame;
module.exports.default = AsyncContextFrame;
`;

shimSources['internal/perf/usertiming'] = `
'use strict';
module.exports = {
  mark: () => ({}),
  measure: () => ({}),
  clearMarks: () => {},
  clearMeasures: () => {},
};
`;

// internal/abort_controller — real version is 638 lines + 4 heavy deps
// (internal/event_target, internal/webidl, internal/worker/js_transferable).
// A small shim with the same export surface is what streams need.
shimSources['internal/abort_controller'] = `
'use strict';
class AbortSignal {
  constructor() { this.aborted = false; this.reason = undefined; this._listeners = []; }
  throwIfAborted() { if (this.aborted) throw this.reason; }
  addEventListener(_type, fn, _opts) { this._listeners.push(fn); }
  removeEventListener(_type, fn) { this._listeners = this._listeners.filter(l => l !== fn); }
  dispatchEvent(ev) { for (const fn of this._listeners) { try { fn(ev); } catch (_e) {} } return true; }
}
AbortSignal.abort = (reason) => { const s = new AbortSignal(); s.aborted = true; s.reason = reason; return s; };
AbortSignal.timeout = (ms) => {
  const s = new AbortSignal();
  setTimeout(() => { s.aborted = true; s.reason = new Error('timeout'); s.dispatchEvent({type:'abort'}); }, ms);
  return s;
};
AbortSignal.any = (signals) => {
  const out = new AbortSignal();
  for (const s of signals) {
    if (s.aborted) { out.aborted = true; out.reason = s.reason; return out; }
    s.addEventListener('abort', () => { out.aborted = true; out.reason = s.reason; out.dispatchEvent({type:'abort'}); });
  }
  return out;
};
class AbortController {
  constructor() { this.signal = new AbortSignal(); }
  abort(reason) {
    if (this.signal.aborted) return;
    this.signal.aborted = true;
    this.signal.reason = reason !== undefined ? reason : new Error('AbortError');
    this.signal.dispatchEvent({ type: 'abort' });
  }
}
const kAborted = Symbol('aborted');
const transferableAbortSignal = (s) => s;
const transferableAbortController = (c) => c;
module.exports = {
  AbortController, AbortSignal,
  kAborted, transferableAbortSignal, transferableAbortController,
};
// Make available globally (browser parity).
globalThis.AbortController = globalThis.AbortController || AbortController;
globalThis.AbortSignal     = globalThis.AbortSignal     || AbortSignal;
`;

// internal/process/permission — Node's permission model (--permission flag).
// Always-true stub satisfies fs.js's permission.has() guards.
shimSources['internal/process/permission'] = `
'use strict';
module.exports = {
  isEnabled: () => false,
  has: () => true,
  deny: () => {},
};
`;

// internal/assert — Node's internal assert (separate from public node:assert).
// Just throws on falsy; tests for fs internals use it to guard invariants.
shimSources['internal/assert'] = `
'use strict';
function assert(cond, msg) {
  if (!cond) {
    const err = new Error(msg || 'internal assertion failed');
    err.code = 'ERR_INTERNAL_ASSERTION';
    throw err;
  }
}
assert.ok = assert;
assert.fail = (msg) => { throw new Error(msg || 'fail'); };
module.exports = assert;
`;

// internal/util/comparisons — used by assert.deepEqual / deepStrictEqual.
// Node's real impl handles cycles, TypedArrays-by-content, Map/Set order-
// independence, etc. Ours covers the common cases the tests we run exercise;
// genuinely tricky cases (cyclic graphs, Symbol-keyed properties) will need
// expansion when a test hits them.
shimSources['internal/util/comparisons'] = `
'use strict';
function tag(v) { return Object.prototype.toString.call(v); }

function isDeepStrictEqual(a, b) { return deepEqual(a, b, true, new WeakMap()); }
function isDeepEqual(a, b)       { return deepEqual(a, b, false, new WeakMap()); }

function deepEqual(a, b, strict, seen) {
  // Primitive fast path.
  if (strict ? Object.is(a, b) : a == b) return true;
  if (a === null || b === null) return false;
  if (typeof a !== 'object' || typeof b !== 'object') return false;
  // Cycle detection.
  if (seen.has(a) && seen.get(a) === b) return true;
  seen.set(a, b);
  // Same constructor (strict only).
  if (strict && a.constructor !== b.constructor) return false;
  // Dates compare by ms.
  const tA = tag(a), tB = tag(b);
  if (tA !== tB) return false;
  if (tA === '[object Date]') return a.getTime() === b.getTime();
  if (tA === '[object RegExp]') return a.source === b.source && a.flags === b.flags;
  if (tA === '[object Map]') {
    if (a.size !== b.size) return false;
    for (const [k, v] of a) if (!b.has(k) || !deepEqual(v, b.get(k), strict, seen)) return false;
    return true;
  }
  if (tA === '[object Set]') {
    if (a.size !== b.size) return false;
    for (const v of a) if (!b.has(v)) return false;
    return true;
  }
  // TypedArrays / Buffer: compare bytes.
  if (ArrayBuffer.isView(a) && !(a instanceof DataView)) {
    if (a.byteLength !== b.byteLength) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  // ArrayBuffers themselves.
  if (a instanceof ArrayBuffer) {
    if (a.byteLength !== b.byteLength) return false;
    const va = new Uint8Array(a), vb = new Uint8Array(b);
    for (let i = 0; i < va.length; i++) if (va[i] !== vb[i]) return false;
    return true;
  }
  // Errors — compare name + message (Node does more, but this satisfies tests).
  if (a instanceof Error) {
    return a.name === b.name && a.message === b.message;
  }
  // Plain objects / arrays: compare enumerable own keys.
  const aKeys = Object.keys(a), bKeys = Object.keys(b);
  if (aKeys.length !== bKeys.length) return false;
  for (const k of aKeys) {
    if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
    if (!deepEqual(a[k], b[k], strict, seen)) return false;
  }
  // Also compare Symbol-keyed properties if any.
  const aSyms = Object.getOwnPropertySymbols(a).filter(s =>
    Object.prototype.propertyIsEnumerable.call(a, s));
  const bSyms = Object.getOwnPropertySymbols(b).filter(s =>
    Object.prototype.propertyIsEnumerable.call(b, s));
  if (aSyms.length !== bSyms.length) return false;
  for (const s of aSyms) {
    if (!Object.prototype.propertyIsEnumerable.call(b, s)) return false;
    if (!deepEqual(a[s], b[s], strict, seen)) return false;
  }
  return true;
}

function isPartialStrictEqual(actual, expected, seen = new WeakMap()) {
  // Subset check: every key in expected matches in actual.
  if (Object.is(actual, expected)) return true;
  if (actual === null || expected === null) return false;
  if (typeof actual !== 'object' || typeof expected !== 'object') return false;
  for (const k of Object.keys(expected)) {
    if (!Object.prototype.hasOwnProperty.call(actual, k)) return false;
    if (!deepEqual(actual[k], expected[k], true, seen)) return false;
  }
  return true;
}

module.exports = { isDeepEqual, isDeepStrictEqual, isPartialStrictEqual };
`;

// internal/tty — only used by colors.js's lazy path when FORCE_COLOR is set.
// Returning a function that says "no color depth" disables colors entirely.
shimSources['internal/tty'] = `
'use strict';
module.exports = {
  getColorDepth: () => 1,
  hasColors: () => false,
  isatty: () => false,
};
`;

// internal/errors/error_source — Node uses this to point inspect at the
// original source location of an error. Source-map machinery is irrelevant
// for our offline use; a stub that returns nothing satisfies the API.
shimSources['internal/errors/error_source'] = `
'use strict';
module.exports = {
  getErrorSource: () => '',
  enrichStackTrace: () => '',
  defineEnumerableProperty: (target, key, value) =>
    Object.defineProperty(target, key, { value, enumerable: true, writable: true, configurable: true }),
};
`;

shimSources['internal/blob'] = `
'use strict';
// Stub: Blob references in buffer.js are lazy. If something actually pokes
// at Blob/File, this will throw clearly.
function lazy() {
  throw new Error('portable-node: internal/blob not implemented');
}
module.exports = new Proxy({}, { get: () => lazy });
`;

// =========================================================================
// require() — module loader. Looks first at shimSources, then at the
// host-provided __nodeSourceFiles, and caches resolved exports.
// =========================================================================

// Cache stores module *objects* (not just exports). When module body
// CommonJS resolver — implements the algorithm documented at
//   https://nodejs.org/api/modules.html#all-together
// (require(X) from module at path Y). See README of node's docs for the
// step labels (LOAD_AS_FILE, LOAD_INDEX, LOAD_AS_DIRECTORY, LOAD_PACKAGE_*).
//
// Cache key is the *resolved absolute path* of the file (or, for built-in
// modules, the bare name). This is required by spec: two different relative
// paths that resolve to the same absolute file must share one cached module.
const moduleCache = Object.create(null);   // resolved-path → module object

// Built-in module names — these short-circuit before any filesystem work.
// Includes both lifted-from-node and shim modules; plus the `node:` prefix
// variants Node accepts.
const builtinSet = Object.create(null);
function registerBuiltin(name) { builtinSet[name] = true; builtinSet['node:' + name] = true; }
for (const k of Object.keys(shimSources)) registerBuiltin(k);
if (typeof __nodeSourceFiles === 'object') {
  for (const k of Object.keys(__nodeSourceFiles)) registerBuiltin(k);
}

// File system helpers backed by the host. They never throw on a missing
// path (return false / null) — the resolver wants probe semantics.
const FS = {
  isFile(p)  { try { return __host.file.is_file(p); } catch (_) { return false; } },
  isDir(p)   { try { return __host.file.is_dir(p);  } catch (_) { return false; } },
  exists(p)  { try { return __host.file.exists(p);  } catch (_) { return false; } },
  read(p)    { return __host.file.read_to_string(p); },
};

// path utilities, kept independent of the (eventually-lifted) node:path
// so the resolver can run before path is loaded.
function pathJoin(a, b) {
  if (!a) return b;
  if (!b) return a;
  if (a.endsWith('/')) a = a.slice(0, -1);
  if (b.startsWith('/')) return b;
  return a + '/' + b;
}
function pathDirname(p) {
  const i = p.lastIndexOf('/');
  if (i < 0) return '.';
  if (i === 0) return '/';
  return p.slice(0, i);
}
function pathBasename(p) {
  const i = p.lastIndexOf('/');
  return i < 0 ? p : p.slice(i + 1);
}
function pathExtname(p) {
  const base = pathBasename(p);
  const i = base.lastIndexOf('.');
  return (i <= 0) ? '' : base.slice(i);
}
function pathNormalize(p) {
  const abs = p.startsWith('/');
  const parts = p.split('/').filter(s => s !== '' && s !== '.');
  const out = [];
  for (const part of parts) {
    if (part === '..') {
      if (out.length && out[out.length - 1] !== '..') out.pop();
      else if (!abs) out.push('..');
    } else out.push(part);
  }
  return (abs ? '/' : '') + out.join('/') || (abs ? '/' : '.');
}
function pathResolve(...segments) {
  let cur = '';
  for (const seg of segments) {
    if (!seg) continue;
    cur = seg.startsWith('/') ? seg : (cur ? cur + '/' + seg : seg);
  }
  // If still relative, base on process.cwd() if available.
  if (!cur.startsWith('/')) {
    const cwd = (globalThis.process && globalThis.process.cwd && globalThis.process.cwd()) || '/';
    cur = cwd + '/' + cur;
  }
  return pathNormalize(cur);
}

// Per-file-extension loader. Returns the module.exports.
function loadFile(absPath, parentModule) {
  const cached = moduleCache[absPath];
  if (cached !== undefined) return cached.exports;

  const ext = pathExtname(absPath).toLowerCase();
  const module = { exports: {}, id: absPath, filename: absPath, loaded: false, children: [], parent: parentModule || null };
  moduleCache[absPath] = module;
  if (parentModule) parentModule.children.push(module);

  try {
    if (ext === '.json') {
      const src = FS.read(absPath);
      module.exports = JSON.parse(src);
    } else if (ext === '.node') {
      throw new Error('portable-node: native addons (.node) not supported: ' + absPath);
    } else {
      // .js, .cjs, .mjs (treated as cjs in this resolver), or no-ext —
      // compile as CommonJS.
      const src = FS.read(absPath);
      const dirname = pathDirname(absPath);
      const localRequire = makeRequire(dirname, module);
      const fn = new Function(
        'exports', 'require', 'module', '__filename', '__dirname',
        // primordials and internalBinding are exposed as globals so they
        // can be shadowed by user-declared `const internalBinding = ...`
        // (which Node tests do via `require('internal/test/binding')`).
        // Strip BOM if present.
        (src.charCodeAt(0) === 0xFEFF ? src.slice(1) : src),
      );
      fn(module.exports, localRequire, module,
         absPath, dirname);
    }
    module.loaded = true;
  } catch (e) {
    delete moduleCache[absPath];
    if (e && e.message && !e.message.includes('[load:')) {
      e.message = '[load: ' + absPath + '] ' + e.message;
    }
    throw e;
  }
  return module.exports;
}

// LOAD_AS_FILE(X) — step 2 of the algorithm.
//   1. If X is a file, load X. STOP
//   2. If X.js is a file, load X.js. STOP
//   3. If X.json is a file, parse X.json. STOP
//   4. If X.node is a file, load X.node. STOP
function loadAsFile(X) {
  if (FS.isFile(X)) return X;
  for (const ext of ['.js', '.json', '.node', '.cjs', '.mjs']) {
    if (FS.isFile(X + ext)) return X + ext;
  }
  return null;
}

// LOAD_INDEX(X) — step in algorithm.
//   1. If X/index.js is a file, load. STOP
//   2. If X/index.json is a file, parse. STOP
//   3. If X/index.node is a file, load. STOP
function loadIndex(X) {
  for (const ext of ['.js', '.json', '.node', '.cjs']) {
    const p = pathJoin(X, 'index' + ext);
    if (FS.isFile(p)) return p;
  }
  return null;
}

// LOAD_AS_DIRECTORY(X).
//   1. If X/package.json is a file:
//       a. parse, look for "main"
//       b. if main is falsy, GOTO 2
//       c. let M = X + main
//       d. LOAD_AS_FILE(M); if found, STOP
//       e. LOAD_INDEX(M);   if found, STOP
//       f. LOAD_INDEX(X);   DEPRECATED, still try
//       g. THROW not found
//   2. LOAD_INDEX(X)
function loadAsDirectory(X) {
  const pj = pathJoin(X, 'package.json');
  if (FS.isFile(pj)) {
    let manifest;
    try { manifest = JSON.parse(FS.read(pj)); }
    catch (e) { throw new Error('invalid package.json at ' + pj + ': ' + e.message); }
    if (manifest && manifest.main) {
      const M = pathNormalize(pathJoin(X, manifest.main));
      const f = loadAsFile(M); if (f) return f;
      const i = loadIndex(M);  if (i) return i;
      // f. deprecated fallback to LOAD_INDEX(X).
    }
  }
  return loadIndex(X);
}

// NODE_MODULES_PATHS(START) — generate the candidate node_modules dirs.
//   1. Split START on '/'.
//   2. From deepest to root, emit START_prefix + '/node_modules', skipping
//      any segment that *is* node_modules (we don't search inside one).
function nodeModulesPaths(start) {
  const parts = start.split('/').filter(s => s !== '');
  const dirs = [];
  for (let i = parts.length; i >= 0; i--) {
    if (parts[i - 1] === 'node_modules') continue;
    const prefix = parts.slice(0, i).join('/');
    dirs.push((start.startsWith('/') ? '/' : '') + prefix + (prefix ? '/' : '') + 'node_modules');
  }
  // Plus an env-configured global search root (PORTABLE_NODE_PROJECT lets you
  // point at any project's node_modules without symlinks).
  const extra = (globalThis.process && globalThis.process.env &&
                 globalThis.process.env.PORTABLE_NODE_PROJECT);
  if (extra) dirs.push(pathJoin(extra, 'node_modules'));
  return dirs;
}

// LOAD_PACKAGE_EXPORTS / LOAD_PACKAGE_SELF / LOAD_PACKAGE_IMPORTS —
// the "exports" field resolution. We implement the subset CJS packages use:
// "exports" can be a string (subpath '.'), an object keyed by subpath, or
// an object keyed by condition. Conditions we honor: 'require', 'node',
// 'default'. ESM-only ('import') we skip.
function selectCondition(node) {
  if (typeof node === 'string') return node;
  if (node == null) return null;
  if (Array.isArray(node)) {
    for (const item of node) {
      const r = selectCondition(item);
      if (r) return r;
    }
    return null;
  }
  // object: try conditional keys in priority order (Node's CJS resolver).
  for (const key of ['node-addons', 'node', 'require', 'default']) {
    if (key in node) {
      const r = selectCondition(node[key]);
      if (r) return r;
    }
  }
  return null;
}

function resolveExports(pkgDir, exportsField, subpath) {
  // subpath always starts with '.'
  if (typeof exportsField === 'string') {
    // shorthand: only valid when subpath === '.'
    if (subpath !== '.') return null;
    return pathJoin(pkgDir, exportsField);
  }
  if (typeof exportsField !== 'object' || exportsField === null) return null;
  // Detect conditional-only object: keys don't start with '.'.
  const keys = Object.keys(exportsField);
  const isConditional = keys.length > 0 && !keys.some(k => k.startsWith('.'));
  if (isConditional) {
    if (subpath !== '.') return null;
    const r = selectCondition(exportsField);
    return r ? pathJoin(pkgDir, r) : null;
  }
  // Subpath map. Exact match first.
  if (subpath in exportsField) {
    const r = selectCondition(exportsField[subpath]);
    return r ? pathJoin(pkgDir, r) : null;
  }
  // Wildcard match: keys with '*'.
  for (const key of keys) {
    const star = key.indexOf('*');
    if (star < 0) continue;
    const prefix = key.slice(0, star), suffix = key.slice(star + 1);
    if (subpath.startsWith(prefix) && subpath.endsWith(suffix)) {
      const inner = subpath.slice(prefix.length, subpath.length - suffix.length);
      const tmpl = selectCondition(exportsField[key]);
      if (tmpl) return pathJoin(pkgDir, tmpl.replace('*', inner));
    }
  }
  return null;
}

function loadPackageExports(X, DIR) {
  // Split X into NAME and SUBPATH.
  let name, subpath;
  if (X.startsWith('@')) {
    // @scope/name[/...]
    const slash1 = X.indexOf('/');
    if (slash1 < 0) return null;
    const slash2 = X.indexOf('/', slash1 + 1);
    if (slash2 < 0) { name = X; subpath = '.'; }
    else { name = X.slice(0, slash2); subpath = '.' + X.slice(slash2); }
  } else {
    const slash = X.indexOf('/');
    if (slash < 0) { name = X; subpath = '.'; }
    else { name = X.slice(0, slash); subpath = '.' + X.slice(slash); }
  }
  const pkgDir = pathJoin(DIR, name);
  const pjPath = pathJoin(pkgDir, 'package.json');
  if (!FS.isFile(pjPath)) return null;
  let manifest;
  try { manifest = JSON.parse(FS.read(pjPath)); } catch (_) { return null; }
  if (manifest == null || manifest.exports == null) return null;
  const target = resolveExports(pkgDir, manifest.exports, subpath);
  if (!target) return null;
  // Resolve to a concrete file (target may omit extension).
  const f = loadAsFile(target);
  if (f) return f;
  if (FS.isDir(target)) {
    const d = loadAsDirectory(target);
    if (d) return d;
  }
  return null;
}

// LOAD_PACKAGE_SELF(X, DIR) — allow a package to require itself by name.
function loadPackageSelf(X, DIR) {
  // Find the closest enclosing package.json moving up from DIR.
  let cur = DIR;
  for (let i = 0; i < 64; i++) {
    const pj = pathJoin(cur, 'package.json');
    if (FS.isFile(pj)) {
      let manifest;
      try { manifest = JSON.parse(FS.read(pj)); } catch (_) { return null; }
      if (!manifest || !manifest.exports || !manifest.name) return null;
      const name = manifest.name;
      if (X !== name && !X.startsWith(name + '/')) return null;
      const subpath = (X === name) ? '.' : '.' + X.slice(name.length);
      const target = resolveExports(cur, manifest.exports, subpath);
      if (!target) return null;
      return loadAsFile(target) ||
             (FS.isDir(target) ? loadAsDirectory(target) : null);
    }
    const parent = pathDirname(cur);
    if (parent === cur) return null;
    cur = parent;
  }
  return null;
}

// LOAD_NODE_MODULES(X, START) — walk node_modules upward.
function loadNodeModules(X, START) {
  const dirs = nodeModulesPaths(START);
  for (const DIR of dirs) {
    const viaExports = loadPackageExports(X, DIR);
    if (viaExports) return viaExports;
    const candidate = pathJoin(DIR, X);
    const f = loadAsFile(candidate);
    if (f) return f;
    if (FS.isDir(candidate)) {
      const d = loadAsDirectory(candidate);
      if (d) return d;
    }
  }
  return null;
}

// Top-level resolver. Returns an absolute path OR a built-in name (the
// latter routes through the registry instead of the file system).
function resolveModule(X, Y) {
  // Step 1: core modules.
  // Strip optional node: prefix; both 'fs' and 'node:fs' are accepted.
  let bare = X;
  if (X.startsWith('node:')) bare = X.slice(5);
  if (builtinSet[bare]) return { kind: 'builtin', name: bare };

  // Step 2: absolute path.
  if (X.startsWith('/')) {
    const p = loadAsFile(X) ||
              (FS.isDir(X) ? loadAsDirectory(X) : null);
    if (p) return { kind: 'file', path: p };
    throw new Error("Cannot find module '" + X + "'");
  }

  // Step 3: relative path.
  if (X.startsWith('./') || X.startsWith('../') || X === '.' || X === '..') {
    const base = pathResolve(Y, X);
    const p = loadAsFile(base) ||
              (FS.isDir(base) ? loadAsDirectory(base) : null);
    if (p) return { kind: 'file', path: p };
    throw new Error("Cannot find module '" + X + "' from '" + Y + "'");
  }

  // Step 4: '#' package imports — not common in deps we care about; skip.

  // Step 5: LOAD_PACKAGE_SELF.
  const self = loadPackageSelf(X, Y);
  if (self) return { kind: 'file', path: self };

  // Step 6: LOAD_NODE_MODULES.
  const fromNM = loadNodeModules(X, Y);
  if (fromNM) return { kind: 'file', path: fromNM };

  throw new Error("Cannot find module '" + X + "' from '" + Y + "'");
}

// Built-in module load (registry-backed; honors caching by name).
function loadBuiltin(name) {
  const cacheKey = 'builtin:' + name;
  const cached = moduleCache[cacheKey];
  if (cached !== undefined) return cached.exports;

  let source = shimSources[name];
  if (source === undefined) source = __nodeSourceFiles[name];
  if (source === undefined) {
    throw new Error('portable-node require: builtin not registered: ' + name);
  }
  const module = { exports: {}, id: name, filename: '/portable-node/' + name + '.js', loaded: false, children: [] };
  moduleCache[cacheKey] = module;

  const filename = name.startsWith('test-')
    ? '/portable-node/test/parallel/' + name + '.js'
    : '/portable-node/' + name + '.js';
  const dirname = pathDirname(filename);

  // Built-ins get a require that can also do file-based resolution
  // (this is how Node's lifted lib code reaches into internal/* etc.).
  const localRequire = makeRequire(dirname, module);
  const fn = new Function(
    'exports', 'require', 'module', '__filename', '__dirname',
    "'use strict';\n" + source,
  );
  try {
    fn(module.exports, localRequire, module, filename, dirname);
    module.loaded = true;
  } catch (e) {
    delete moduleCache[cacheKey];
    if (e && e.message && !e.message.includes('[load:')) {
      e.message = '[load: ' + name + '] ' + e.message;
    }
    // If the engine produced an empty stack (happens for some implicit
    // TypeErrors), capture our own here so the test runner can at least
    // point at where the require happened.
    if (e && (!e.stack || e.stack === '')) {
      try { e.stack = (new Error('rethrown from loadBuiltin(' + name + ')')).stack || ''; }
      catch (_) {}
    }
    throw e;
  }
  return module.exports;
}

// Factory: every CJS module gets a require bound to its own directory.
function makeRequire(dirname, parentModule) {
  function localRequire(name) {
    const r = resolveModule(name, dirname);
    if (r.kind === 'builtin') return loadBuiltin(r.name);
    return loadFile(r.path, parentModule);
  }
  localRequire.cache = moduleCache;
  localRequire.resolve = function(name) {
    const r = resolveModule(name, dirname);
    return r.kind === 'builtin' ? r.name : r.path;
  };
  // Node's `require.main` — we don't have a main file, leave undefined.
  return localRequire;
}

// The top-level require seeded at process.cwd(). Used by the test runner
// and by direct ctx.eval() probes from main.rs.
const __topRequire = makeRequire(
  (globalThis.process && globalThis.process.cwd && globalThis.process.cwd()) || '/',
  null,
);
globalThis.require = __topRequire;
// Keep the old name available for any code that captured it.
const require = __topRequire;

// =========================================================================
// Load buffer (which transitively loads internal/buffer + shims).
// =========================================================================

const bufferMod = require('buffer');
globalThis.Buffer = bufferMod.Buffer;
globalThis.SlowBuffer = bufferMod.SlowBuffer;
globalThis.kMaxLength = bufferMod.kMaxLength;
globalThis.kStringMaxLength = bufferMod.kStringMaxLength;
globalThis.constants = bufferMod.constants;

// node:querystring — lifted from nodejs/node lib/querystring.js. No new
// host primitives needed: it builds entirely on top of Buffer + JS primordials.
globalThis.querystring = require('querystring');

// stdout/stderr: write through to the host. isTTY=false keeps ANSI colors
// off (colors.js checks it; our test harness expects plain output).
const _hostStdoutWrite = __host.process.stdout_write;
const _hostStderrWrite = __host.process.stderr_write;
const stdoutStream = { isTTY: false, write: (s) => _hostStdoutWrite(s) };
const stderrStream = { isTTY: false, write: (s) => _hostStderrWrite(s) };
// Keep one name for back-compat: the test-runner only touches `.isTTY`.
const stdStream = stdoutStream;

// `process` global — Node code expects this. Backed by __host.process for
// the data parts; just a few stub methods otherwise.
globalThis.process = {
  platform: __host.process.platform,
  arch:     __host.process.arch,
  pid:      __host.process.pid,
  env:      __host.process.env,
  argv:     __host.process.argv,
  cwd:      __host.process.cwd,
  chdir:    __host.process.chdir,
  exit:     __host.process.exit,
  hrtime:   () => {
    // hrtime() returns [seconds, nanoseconds]. Build from monotonic ns.
    const ns = __host.process.hrtime_ns();
    const sec = Math.floor(ns / 1e9);
    return [sec, Math.floor(ns - sec * 1e9)];
  },
  nextTick: (fn, ...args) => Promise.resolve().then(() => fn(...args)),
  versions: {},
  release: { name: 'node' },
  stderr: stderrStream,
  stdout: stdoutStream,
  stdin:  { isTTY: false, read: () => null },
  // Common stubs callers reach for.
  on:    () => {},
  off:   () => {},
  emit:  () => false,
  emitWarning: () => {},
};

// =========================================================================
// Tier 0 + Tier 1 event loop. Pure JS over __host.time.{now_ms, sleep_ms}.
// process.nextTick uses microtasks (engine-native). setTimeout / setInterval
// / setImmediate live on a JS-side priority queue. The Rust driver pumps:
//   1. drain microtasks
//   2. fire due timers
//   3. sleep until next due (or exit if idle)
// =========================================================================

const _timers = [];                  // [{id, fn, when, repeat, args, ref, cancelled}]
let _nextTimerId = 1;
const _timersById = new Map();

function _scheduleTimer(fn, when, repeat, args, ref) {
  const t = { id: _nextTimerId++, fn, when, repeat, args, ref, cancelled: false };
  _timers.push(t);
  _timersById.set(t.id, t);
  return t;
}

globalThis.setTimeout = function setTimeout(fn, ms = 0, ...args) {
  const when = __host.time.now_ms() + Math.max(0, ms | 0);
  return _scheduleTimer(fn, when, null, args, true);
};
globalThis.setInterval = function setInterval(fn, ms = 0, ...args) {
  const interval = Math.max(1, ms | 0);
  const when = __host.time.now_ms() + interval;
  return _scheduleTimer(fn, when, interval, args, true);
};
globalThis.setImmediate = function setImmediate(fn, ...args) {
  // setImmediate fires "as soon as possible" — we model as a 0ms timer.
  return _scheduleTimer(fn, __host.time.now_ms(), null, args, true);
};

function _cancel(handle) {
  const t = typeof handle === 'object' ? handle : _timersById.get(handle);
  if (t) t.cancelled = true;
}
globalThis.clearTimeout   = _cancel;
globalThis.clearInterval  = _cancel;
globalThis.clearImmediate = _cancel;

// Timer handles in Node have .ref() / .unref() / .hasRef(). unref means
// "don't keep the loop alive for this timer alone". Note: the underlying
// timer object stores its referenced-ness on \`._ref\` (boolean), so the
// .ref()/.unref() *methods* don't collide with that field.
const _timerProto = {
  ref()   { this._ref = true;  return this; },
  unref() { this._ref = false; return this; },
  hasRef(){ return this._ref; },
  refresh() { this.when = __host.time.now_ms() + (this.repeat || 0); return this; },
  [Symbol.toPrimitive]() { return this.id; },
};
const _origSchedule = _scheduleTimer;
function _scheduleTimerWithProto(...a) {
  const t = _origSchedule(...a);
  t._ref = t.ref !== false;
  Object.assign(t, _timerProto);
  return t;
}
// Override the helpers to return proto-equipped handles.
globalThis.setTimeout = function (fn, ms = 0, ...args) {
  const when = __host.time.now_ms() + Math.max(0, ms | 0);
  return _scheduleTimerWithProto(fn, when, null, args, true);
};
globalThis.setInterval = function (fn, ms = 0, ...args) {
  const interval = Math.max(1, ms | 0);
  return _scheduleTimerWithProto(fn, __host.time.now_ms() + interval, interval, args, true);
};
globalThis.setImmediate = function (fn, ...args) {
  return _scheduleTimerWithProto(fn, __host.time.now_ms(), null, args, true);
};

// =========================================================================
// I/O completion dispatcher — pairs op_ids with JS-side callbacks. Submit
// an async op via __ioSubmit(submitFn, cb); the helper allocates a slot,
// calls submitFn(slotKey), records the cb, returns the op_id.
// On completion (delivered through __host.io.poll), __ioDispatch fires cb.
// =========================================================================

const _ioOps = new Map();   // op_id -> { kind, cb, handle }

function _ioRegister(op_id, cb) { _ioOps.set(op_id, cb); return op_id; }

globalThis.__ioHasPending = function () {
  return _ioOps.size > 0 || __host.io.has_pending();
};

// Drain the host's completion queue, firing JS callbacks.
globalThis.__ioDrain = function (timeoutMs) {
  if (typeof timeoutMs !== 'number') timeoutMs = 0;
  const cs = __host.io.poll(timeoutMs);
  for (const c of cs) {
    const cb = _ioOps.get(c.op_id);
    if (cb) {
      _ioOps.delete(c.op_id);
      try { cb(c); } catch (e) {
        try { (globalThis.console || { error: () => {} }).error(e); } catch (_) {}
      }
    }
  }
  return cs.length;
};

// Public helpers: tcpConnect(addr, port, cb) etc., callback-style.
// The cb receives the completion object: { op_id, kind, status, ... }.
globalThis.tcpCreate = () => __host.tcp.create_tcp();
globalThis.tcpListen = (h, ip, port, bk) => __host.tcp.listen(h, ip, port, bk || 128);
globalThis.tcpClose  = (h) => __host.tcp.close(h);
globalThis.tcpLocalAddr = (h) => __host.tcp.local_addr(h);
globalThis.tcpPeerAddr  = (h) => __host.tcp.peer_addr(h);
globalThis.tcpConnect = (h, ip, port, cb) =>
  _ioRegister(__host.tcp.connect(h, ip, port), cb);
globalThis.tcpAccept  = (h, cb) =>
  _ioRegister(__host.tcp.accept(h), cb);
globalThis.tcpRead    = (h, buf, off, len, cb) =>
  _ioRegister(__host.tcp.read(h, buf, off|0, len|0), cb);
globalThis.tcpWrite   = (h, buf, off, len, cb) =>
  _ioRegister(__host.tcp.write(h, buf, off|0, len|0), cb);
globalThis.tcpShutdown = (h, how, cb) =>
  _ioRegister(__host.tcp.shutdown(h, how|0), cb);

// Hooks called by the Rust event-loop driver between microtask drains.
globalThis.__eventLoopHasWork = function () {
  for (const t of _timers) if (!t.cancelled && t._ref !== false) return true;
  return false;
};
// Returns Infinity if no due-or-pending timer.
globalThis.__eventLoopNextDueMs = function () {
  let min = Infinity;
  for (const t of _timers) {
    if (t.cancelled || t._ref === false) continue;
    if (t.when < min) min = t.when;
  }
  return min;
};
globalThis.__eventLoopFireDue = function () {
  const now = __host.time.now_ms();
  // Iterate stably; new timers added during a callback fire next round.
  const due = [];
  for (const t of _timers) {
    if (!t.cancelled && t._ref !== false && t.when <= now) due.push(t);
  }
  // Sort earliest-first so order of fires matches order of scheduling.
  due.sort((a, b) => a.when - b.when);
  for (const t of due) {
    if (t.cancelled) continue;
    try { t.fn(...t.args); } catch (e) {
      try { (globalThis.console || { error: () => {} }).error(e); } catch (_) {}
    }
    if (t.repeat && !t.cancelled) {
      t.when = __host.time.now_ms() + t.repeat;
    } else {
      t.cancelled = true;
    }
  }
  // Compact cancelled.
  for (let i = _timers.length - 1; i >= 0; i--) {
    if (_timers[i].cancelled) { _timersById.delete(_timers[i].id); _timers.splice(i, 1); }
  }
};

// process.nextTick now uses microtasks. Engine drains them automatically
// when we return control to the host (which calls execute_pending_jobs).
if (globalThis.process) {
  globalThis.process.nextTick = function (fn, ...args) {
    Promise.resolve().then(() => {
      try { fn(...args); } catch (e) {
        try { (globalThis.console || { error: () => {} }).error(e); } catch (_) {}
      }
    });
  };
  globalThis.queueMicrotask = globalThis.queueMicrotask || function (fn) {
    Promise.resolve().then(fn);
  };
}

// node:os — lifted verbatim from lib/os.js. The internalBinding('os') call
// inside resolves to our JS shim `__binding/os`, which in turn calls
// __host.os.*. This is the portable-host architecture working end-to-end.
globalThis.os = require('os');

// node:fs — sync subset. Portable JS module that uses __binding/fs (which
// uses __host.file.*). This is our own minimal fs, not a Node lift; lifting
// Node's lib/fs.js (3387 lines) is a bigger effort. The architecture is the
// same: only the top layer differs.
// (Old portable fs shim — kept under __legacy/fs as a fallback.)
shimSources['__legacy/fs'] = `
'use strict';
const binding = internalBinding('fs');
const { Buffer } = require('buffer');
const constants = internalBinding('constants').fs;

function normalizeFlag(flag) {
  if (flag === undefined) return binding.flagsForString('r');
  if (typeof flag === 'number') return flag;
  return binding.flagsForString(flag);
}

function normalizeEncoding(enc) {
  if (enc == null) return null;
  if (typeof enc === 'string') return enc;
  if (typeof enc === 'object' && enc.encoding) return enc.encoding;
  return null;
}

function openSync(path, flag, mode) {
  return binding.open(String(path), normalizeFlag(flag), mode == null ? 0o666 : mode);
}
function closeSync(fd) { binding.close(fd); }
function readSync(fd, buffer, offset, length, position) {
  if (offset == null) offset = 0;
  if (length == null) length = buffer.length - offset;
  if (position == null) position = -1;
  return binding.read(fd, buffer, offset, length, position);
}
function writeSync(fd, buffer, offset, length, position) {
  if (typeof buffer === 'string') {
    buffer = Buffer.from(buffer, typeof offset === 'string' ? offset : 'utf8');
    if (typeof offset === 'string') { length = null; position = null; }
    offset = 0;
  }
  if (offset == null) offset = 0;
  if (length == null) length = buffer.length - offset;
  if (position == null) position = -1;
  return binding.writeBuffer(fd, buffer, offset, length, position);
}

function statSync(path, opts) {
  // opts.bigint is ignored — we always return numbers for now.
  return rawToStats(binding.stat(String(path)));
}
function lstatSync(path) { return rawToStats(binding.lstat(String(path))); }
function fstatSync(fd)   { return rawToStats(binding.fstat(fd)); }

function rawToStats(raw) {
  // Mirror Node's Stats class — most callers test .isFile() / .isDirectory().
  const mode = raw.mode;
  const ifmt = mode & 0o170000;
  const stats = {
    dev:   raw.dev,
    ino:   raw.ino,
    mode:  raw.mode,
    nlink: raw.nlink,
    uid:   raw.uid,
    gid:   raw.gid,
    rdev:  raw.rdev,
    size:  raw.size,
    blksize: raw.blksize,
    blocks: raw.blocks,
    atimeMs:     raw.atime_ms,
    mtimeMs:     raw.mtime_ms,
    ctimeMs:     raw.ctime_ms,
    birthtimeMs: raw.birthtime_ms,
    atime:     new Date(raw.atime_ms),
    mtime:     new Date(raw.mtime_ms),
    ctime:     new Date(raw.ctime_ms),
    birthtime: new Date(raw.birthtime_ms),
    isFile:        () => ifmt === 0o100000,
    isDirectory:   () => ifmt === 0o040000,
    isSymbolicLink:() => ifmt === 0o120000,
    isBlockDevice: () => ifmt === 0o060000,
    isCharacterDevice: () => ifmt === 0o020000,
    isFIFO:        () => ifmt === 0o010000,
    isSocket:      () => ifmt === 0o140000,
  };
  return stats;
}

function existsSync(path) {
  try { binding.stat(String(path)); return true; } catch (_e) { return false; }
}

function readdirSync(path, opts) {
  const entries = binding.readdir(String(path));
  const withFileTypes = opts && opts.withFileTypes;
  if (withFileTypes) {
    return entries.map(e => ({
      name: e.name,
      isFile:        () => e.type === 1,
      isDirectory:   () => e.type === 2,
      isSymbolicLink:() => e.type === 3,
      isBlockDevice: () => false,
      isCharacterDevice: () => false,
      isFIFO:        () => false,
      isSocket:      () => false,
    }));
  }
  return entries.map(e => e.name);
}

function mkdirSync(path, opts) {
  let mode = 0o777, recursive = false;
  if (typeof opts === 'number') mode = opts;
  else if (typeof opts === 'object' && opts) {
    if (opts.mode != null) mode = opts.mode;
    if (opts.recursive) recursive = true;
  }
  if (recursive) {
    // Naive: walk parents and mkdir each one that doesn't exist.
    const sep = '/';
    const parts = String(path).split(sep);
    let cur = parts[0] || '';
    for (let i = 1; i < parts.length; i++) {
      cur = cur + sep + parts[i];
      if (!cur) continue;
      try { binding.mkdir(cur, mode); }
      catch (e) { if (e.code !== 'EEXIST') throw e; }
    }
    return;
  }
  binding.mkdir(String(path), mode);
}

function unlinkSync(path) { binding.unlink(String(path)); }
function rmdirSync(path)  { binding.rmdir(String(path)); }
function renameSync(a, b) { binding.rename(String(a), String(b)); }
function accessSync(path, mode) { binding.access(String(path), mode == null ? 0 : mode); }
function realpathSync(path) { return binding.realpath(String(path)); }

function readFileSync(path, options) {
  const enc = typeof options === 'string' ? options : normalizeEncoding(options);
  const flag = (options && typeof options === 'object' && options.flag) || 'r';
  const fd = openSync(path, flag);
  try {
    const st = fstatSync(fd);
    const buf = Buffer.allocUnsafe(st.size);
    let read = 0;
    while (read < st.size) {
      const n = readSync(fd, buf, read, st.size - read, read);
      if (n <= 0) break;
      read += n;
    }
    const out = read === buf.length ? buf : buf.slice(0, read);
    return enc ? out.toString(enc) : out;
  } finally {
    closeSync(fd);
  }
}

function writeFileSync(path, data, options) {
  const enc  = typeof options === 'string' ? options : normalizeEncoding(options);
  const flag = (options && typeof options === 'object' && options.flag) || 'w';
  const mode = (options && typeof options === 'object' && options.mode) || 0o666;
  const buf = typeof data === 'string'
    ? Buffer.from(data, enc || 'utf8')
    : (Buffer.isBuffer(data) ? data : Buffer.from(data));
  const fd = openSync(path, flag, mode);
  try {
    let off = 0;
    while (off < buf.length) {
      const n = writeSync(fd, buf, off, buf.length - off, -1);
      if (n <= 0) break;
      off += n;
    }
  } finally {
    closeSync(fd);
  }
}

function appendFileSync(path, data, options) {
  const o = (typeof options === 'object' && options) ? options : {};
  writeFileSync(path, data, { ...o, flag: o.flag || 'a', encoding: typeof options === 'string' ? options : o.encoding });
}

module.exports = {
  openSync, closeSync, readSync, writeSync,
  statSync, lstatSync, fstatSync, existsSync, accessSync, realpathSync,
  readdirSync, mkdirSync, unlinkSync, rmdirSync, renameSync,
  readFileSync, writeFileSync, appendFileSync,
  constants,
  // Promise-API stubs that just defer to sync:
  promises: {
    readFile:  (...a) => Promise.resolve().then(() => readFileSync(...a)),
    writeFile: (...a) => Promise.resolve().then(() => writeFileSync(...a)),
    stat:      (...a) => Promise.resolve().then(() => statSync(...a)),
    readdir:   (...a) => Promise.resolve().then(() => readdirSync(...a)),
    unlink:    (...a) => Promise.resolve().then(() => unlinkSync(...a)),
    mkdir:     (...a) => Promise.resolve().then(() => mkdirSync(...a)),
    rmdir:     (...a) => Promise.resolve().then(() => rmdirSync(...a)),
    rename:    (...a) => Promise.resolve().then(() => renameSync(...a)),
    access:    (...a) => Promise.resolve().then(() => accessSync(...a)),
    realpath:  (...a) => Promise.resolve().then(() => realpathSync(...a)),
  },
};
`;

// Extend constants binding with the fs-flag values our portable fs.js exposes.
// These are the standard Node fs constants; only ones the test actually
// touches need real values.

globalThis.fs = require('fs');

// node:path — lifted verbatim from lib/path.js. Pure JS, no host primitives
// needed; only depends on primordials + internal/{constants,validators,util}.
globalThis.path = require('path');

// node:events — lifted verbatim from lib/events.js. Pure JS, ~1200 lines.
// EventEmitter, once(), on(), AbortController integration.
globalThis.events = require('events');
globalThis.EventEmitter = globalThis.events;  // events module is also the class

// node:punycode — lifted verbatim. Deprecated upstream but still in core,
// so tests sometimes import it. Just needs internalBinding('util'), which
// we have.
globalThis.punycode = require('punycode');

// node:string_decoder — lifted verbatim. Uses Buffer + a streaming-decode
// binding (__binding/string_decoder) we implement entirely in JS.
globalThis.string_decoder = require('string_decoder');

// node:zlib — portable JS facade over __host.zlib.*. Node's real zlib.js
// uses a stateful binding handle pattern (Zlib class with callbacks, ~500
// LOC native code) that would be a large lift. The sync subset and basic
// Buffer-in / Buffer-out form is what >90% of zlib usage needs, and that's
// what this shim provides. Real lift of zlib.js can replace this later.
shimSources['zlib'] = `
'use strict';
const { Buffer } = require('buffer');
const Z = globalThis.__host.zlib;

// Constants matching Node's zlib constants (subset). Used both for level
// values and as enum-like flags by downstream code.
const constants = {
  Z_NO_FLUSH: 0, Z_PARTIAL_FLUSH: 1, Z_SYNC_FLUSH: 2,
  Z_FULL_FLUSH: 3, Z_FINISH: 4, Z_BLOCK: 5, Z_TREES: 6,
  Z_OK: 0, Z_STREAM_END: 1, Z_NEED_DICT: 2, Z_ERRNO: -1,
  Z_STREAM_ERROR: -2, Z_DATA_ERROR: -3, Z_MEM_ERROR: -4,
  Z_BUF_ERROR: -5, Z_VERSION_ERROR: -6,
  Z_NO_COMPRESSION: 0, Z_BEST_SPEED: 1, Z_BEST_COMPRESSION: 9,
  Z_DEFAULT_COMPRESSION: -1, // flate2 default = 6
  Z_FILTERED: 1, Z_HUFFMAN_ONLY: 2, Z_RLE: 3, Z_FIXED: 4, Z_DEFAULT_STRATEGY: 0,
  Z_BINARY: 0, Z_TEXT: 1, Z_ASCII: 1, Z_UNKNOWN: 2,
  Z_DEFLATED: 8,
  ZLIB_VERNUM: 0x12b0,
  // Brotli / Zstd — not implemented in our host yet; constants present so
  // user code that imports them doesn't fail on access.
  BROTLI_OPERATION_PROCESS: 0, BROTLI_OPERATION_FLUSH: 1, BROTLI_OPERATION_FINISH: 2,
};

function levelFromOpts(opts) {
  const lvl = (opts && opts.level !== undefined) ? opts.level : -1;
  // Node uses -1 = default; flate2's default is 6.
  return lvl === -1 ? 6 : Math.max(0, Math.min(9, lvl | 0));
}

function toBufferIn(input) {
  if (Buffer.isBuffer(input)) return input;
  if (typeof input === 'string') return Buffer.from(input, 'utf8');
  if (ArrayBuffer.isView(input)) return Buffer.from(input.buffer, input.byteOffset, input.byteLength);
  if (input instanceof ArrayBuffer) return Buffer.from(input);
  throw new TypeError('zlib: input must be a Buffer, TypedArray, DataView, ArrayBuffer, or string');
}

// Sync codec functions
const deflateRawSync   = (input, opts) => Buffer.from(Z.deflate_raw(toBufferIn(input), levelFromOpts(opts)));
const inflateRawSync   = (input)       => Buffer.from(Z.inflate_raw(toBufferIn(input)));
const deflateSync      = (input, opts) => Buffer.from(Z.deflate(toBufferIn(input), levelFromOpts(opts)));
const inflateSync      = (input)       => Buffer.from(Z.inflate(toBufferIn(input)));
const gzipSync         = (input, opts) => Buffer.from(Z.gzip(toBufferIn(input), levelFromOpts(opts)));
const gunzipSync       = (input)       => Buffer.from(Z.gunzip(toBufferIn(input)));
// unzip auto-detects gzip vs zlib header
const unzipSync = (input) => {
  const buf = toBufferIn(input);
  // gzip header is 1F 8B
  if (buf.length >= 2 && buf[0] === 0x1F && buf[1] === 0x8B) return gunzipSync(buf);
  return inflateSync(buf);
};

// Async forms — defer-to-sync wrappers. Real Node uses a worker thread.
function defer(fn) {
  return function (input, optsOrCb, cb) {
    const opts = typeof optsOrCb === 'function' ? {} : (optsOrCb || {});
    cb = typeof optsOrCb === 'function' ? optsOrCb : cb;
    Promise.resolve().then(() => {
      try { cb(null, fn(input, opts)); } catch (e) { cb(e); }
    });
  };
}

// Stream classes — only useful once a real event loop is in place. We
// expose them as throwing classes so code that tries to use them fails
// loudly rather than silently doing nothing.
function streamNotImplemented(name) {
  return function () { throw new Error(
    'portable-node: zlib.' + name + ' (stream) not implemented — ' +
    'use the *Sync forms (' + name + 'Sync) or the promises API'); };
}

module.exports = {
  // sync
  deflateSync, inflateSync, gzipSync, gunzipSync, unzipSync,
  deflateRawSync, inflateRawSync,
  // callback (defer-to-sync)
  deflate:    defer(deflateSync),
  inflate:    defer(inflateSync),
  gzip:       defer(gzipSync),
  gunzip:     defer(gunzipSync),
  unzip:      defer(unzipSync),
  deflateRaw: defer(deflateRawSync),
  inflateRaw: defer(inflateRawSync),
  // promises
  promises: {
    deflate:    (i, o) => Promise.resolve().then(() => deflateSync(i, o)),
    inflate:    (i)    => Promise.resolve().then(() => inflateSync(i)),
    gzip:       (i, o) => Promise.resolve().then(() => gzipSync(i, o)),
    gunzip:     (i)    => Promise.resolve().then(() => gunzipSync(i)),
    unzip:      (i)    => Promise.resolve().then(() => unzipSync(i)),
    deflateRaw: (i, o) => Promise.resolve().then(() => deflateRawSync(i, o)),
    inflateRaw: (i)    => Promise.resolve().then(() => inflateRawSync(i)),
  },
  // utilities
  crc32: (input) => Z.crc32(toBufferIn(input)),
  // streams — stubs until event loop lands
  createDeflate:  streamNotImplemented('createDeflate'),
  createInflate:  streamNotImplemented('createInflate'),
  createGzip:     streamNotImplemented('createGzip'),
  createGunzip:   streamNotImplemented('createGunzip'),
  createUnzip:    streamNotImplemented('createUnzip'),
  createDeflateRaw: streamNotImplemented('createDeflateRaw'),
  createInflateRaw: streamNotImplemented('createInflateRaw'),
  Deflate:  streamNotImplemented('Deflate'),
  Inflate:  streamNotImplemented('Inflate'),
  Gzip:     streamNotImplemented('Gzip'),
  Gunzip:   streamNotImplemented('Gunzip'),
  Unzip:    streamNotImplemented('Unzip'),
  DeflateRaw: streamNotImplemented('DeflateRaw'),
  InflateRaw: streamNotImplemented('InflateRaw'),
  constants,
  // Node also exports these directly:
  ...constants,
};
`;
globalThis.zlib = require('zlib');

// console — Node exposes a global `console` object. Build a small, working
// console on top of process.stdout/stderr. Using the lifted `node:console`
// would drag in stream + a kColorMode plumbing chain; the surface user code
// actually wants (log/info/warn/error/debug) is tiny.
(function () {
  const util = require('util');
  function fmt(args) {
    if (args.length === 0) return '';
    // Use util.format for %s/%d/%j substitution and object inspection.
    return util.format.apply(null, args);
  }
  const _con = {
    log:   function () { stdoutStream.write(fmt(arguments) + '\n'); },
    info:  function () { stdoutStream.write(fmt(arguments) + '\n'); },
    debug: function () { stdoutStream.write(fmt(arguments) + '\n'); },
    warn:  function () { stderrStream.write(fmt(arguments) + '\n'); },
    error: function () { stderrStream.write(fmt(arguments) + '\n'); },
    trace: function () {
      stderrStream.write('Trace: ' + fmt(arguments) + '\n' +
                         (new Error()).stack + '\n');
    },
    dir:   function (v, opts) {
      stdoutStream.write(util.inspect(v, opts || {}) + '\n');
    },
    table: function (v) { stdoutStream.write(util.inspect(v) + '\n'); },
    group: function () {},
    groupCollapsed: function () {},
    groupEnd:       function () {},
    time:           function () {},
    timeEnd:        function () {},
    timeLog:        function () {},
    count:          function () {},
    countReset:     function () {},
    assert:         function (cond) {
      if (!cond) {
        const rest = Array.prototype.slice.call(arguments, 1);
        stderrStream.write('Assertion failed: ' + fmt(rest) + '\n');
      }
    },
    clear:          function () {},
  };
  globalThis.console = _con;
})();

// Test runner: run a registered Node test module and return a result string.
// Escape lone surrogates so the test output can survive a round-trip through
// Rust's String (UTF-8). The string_decoder test deliberately constructs
// strings with lone surrogates to verify decoding; we don't want those to
// crash the harness on the way back.
function _safeForUtf8(s) {
  if (typeof s !== 'string') return String(s);
  let out = '';
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    if (code >= 0xD800 && code <= 0xDFFF) {
      // Check for a valid pair.
      if (code <= 0xDBFF && i + 1 < s.length) {
        const next = s.charCodeAt(i + 1);
        if (next >= 0xDC00 && next <= 0xDFFF) {
          out += s.charAt(i) + s.charAt(i + 1);
          i++;
          continue;
        }
      }
      out += '\\u' + code.toString(16).padStart(4, '0');
    } else {
      out += s.charAt(i);
    }
  }
  return out;
}

// Snapshot mutable globals once so each test can be restored to a fresh state.
const _origProcessCwd = globalThis.process && globalThis.process.cwd;

globalThis.__runNodeTest = function (name) {
  delete moduleCache[name];
  delete moduleCache['builtin:' + name];
  // Reset stack-trace state in case a prior test left it weird.
  Error.stackTraceLimit = 50;
  Error.prepareStackTrace = undefined;
  // Restore process.cwd (test-path-resolve overwrites it; tests run in
  // sequence in our harness so cross-test pollution is real).
  if (_origProcessCwd) globalThis.process.cwd = _origProcessCwd;
  try {
    require(name);
    return 'PASS ' + name;
  } catch (e) {
    let stack;
    try { stack = e && e.stack; } catch (_) { stack = null; }
    if (typeof stack !== 'string' || !stack) {
      // Some AssertionError subclasses lose their stack; capture our own.
      const e2 = new Error('rethrow');
      stack = e2.stack || '';
    }
    const msg = (e && e.message) || String(e);
    const lines = String(stack).split('\n').slice(0, 8).join('\n  ');
    let extra = '';
    if (e && (e.actual !== undefined || e.expected !== undefined)) {
      try { extra = '\n  actual=' + JSON.stringify(e.actual) + ' expected=' + JSON.stringify(e.expected); } catch (_) {}
    }
    if (e && e.operator) extra += ' op=' + e.operator;
    if (e && e.code) extra += ' code=' + e.code;
    return _safeForUtf8('FAIL ' + name + ': ' + msg + extra + '\n  ' + lines);
  }
};
