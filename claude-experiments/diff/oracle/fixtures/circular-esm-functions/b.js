import { fromA } from "./a.js";

export function fromB() {
  return fromA.name === "fromA" ? "b" : "broken";
}
