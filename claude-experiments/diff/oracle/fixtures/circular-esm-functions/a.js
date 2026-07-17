import { fromB } from "./b.js";

export function fromA() {
  return `a:${fromB()}`;
}
