import * as ns from "./hub.js";
console.log("keys:" + Object.keys(ns).join(","));
console.log("has-dupe:" + ("dupe" in ns));
console.log(ns.uniqueA + ":" + ns.uniqueB);
