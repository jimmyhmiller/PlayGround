import * as ns from "./lib.cjs";
console.log("keys:" + Object.keys(ns).sort().join(","));
console.log("default-one:" + ns.default.one);
console.log("named-one:" + ns.one);
