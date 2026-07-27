// `export * as ns from "cjs"` compiles to a GETTER, so every read of `hub.legacy`
// re-enters the interop. Re-entering must return the same object.
export * as legacy from "./num.cjs";
