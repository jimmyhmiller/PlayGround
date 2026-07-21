const a = require("./shared.cjs");
const b = require("./shared.cjs");
a.n = 5;
console.log("same:" + (a === b));
console.log("n:" + b.n);
