exports.before = "a-before";
const b = require("./b.cjs");
exports.after = "a-after";
exports.reportB = () => b.value;
