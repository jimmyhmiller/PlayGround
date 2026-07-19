import * as values from "./values.js";

console.log(`keys:${Object.keys(values).sort().join(",")}`);
console.log(`values:${values.alpha}:${values.beta}`);
console.log(
  `meta:${values[Symbol.toStringTag]}:${Object.getOwnPropertyDescriptor(values, "alpha").configurable}`,
);
