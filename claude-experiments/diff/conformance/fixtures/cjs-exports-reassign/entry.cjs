const m = require("./mod.cjs");
console.log("keys:" + Object.keys(m).sort().join(","));
console.log("kept:" + m.kept + ":" + ("dropped" in m));
