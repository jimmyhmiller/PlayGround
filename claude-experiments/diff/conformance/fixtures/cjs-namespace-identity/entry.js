import * as hub from "./hub.js";
console.log("stable:" + (hub.legacy === hub.legacy));
console.log("value:" + hub.legacy.value);
console.log("default-is-exports:" + (hub.legacy.default.value === "legacy-value"));
