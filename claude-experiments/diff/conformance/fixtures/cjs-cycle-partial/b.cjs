const a = require("./a.cjs");
console.log("b:sees-before:" + a.before);
console.log("b:sees-after:" + a.after);
exports.value = "b-value";
