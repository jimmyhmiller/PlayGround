console.log("entry:start");
const m = await import("./tla-mod.js");
console.log("entry:got:" + m.v);
