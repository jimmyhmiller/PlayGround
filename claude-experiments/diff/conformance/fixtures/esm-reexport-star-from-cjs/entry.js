import * as ns from "./hub.js";
console.log("keys:" + Object.keys(ns).join(","));
console.log(ns.foo + ":" + ns.bar);
console.log("has-default:" + ("default" in ns));
