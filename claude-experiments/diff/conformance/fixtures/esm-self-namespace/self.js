import * as self from "./self.js";
export const value = 5;
export function report() {
  console.log("self:" + self.value + ":" + ("report" in self));
}
