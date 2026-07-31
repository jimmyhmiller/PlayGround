// Node, rolldown and esbuild all refuse this: a named import that the required
// CommonJS module does not provide is a hard error, never `undefined`.
import { missingName } from "./marked.cjs";
console.log("reached:" + missingName);
