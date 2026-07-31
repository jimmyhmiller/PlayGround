// `module.exports` is a PRIMITIVE, so the interop wrapper cannot be keyed by the
// exports object — a WeakMap takes no primitive key. It still has to be one
// namespace per module: Node builds a CommonJS module's namespace once.
module.exports = 42;
