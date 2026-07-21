import * as ns from "./vals.js";
console.log("frozen:" + Object.isFrozen(ns));
console.log("tag:" + ns[Symbol.toStringTag]);
try {
  ns.a = 99;
  console.log("write:allowed:" + ns.a);
} catch (error) {
  console.log("write:" + error.constructor.name);
}
try {
  ns.newProp = 1;
  console.log("add:allowed");
} catch (error) {
  console.log("add:" + error.constructor.name);
}
