import legacy from "./legacy.cjs";
export const state = "esm-ready";
export function readThroughCommonJs() {
  return legacy.read();
}
