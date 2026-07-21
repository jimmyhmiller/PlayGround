import * as staticNs from "./shared.js";
const dynamicNs = await import("./shared.js");
console.log("identical:" + (staticNs === dynamicNs));
console.log("id:" + staticNs.id);
