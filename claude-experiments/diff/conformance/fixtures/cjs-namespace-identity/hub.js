// Re-exporting a CommonJS module as a namespace: the namespace object a
// consumer reads must be ONE object, not a fresh interop wrapper per read.
export * as legacy from "./legacy.cjs";
