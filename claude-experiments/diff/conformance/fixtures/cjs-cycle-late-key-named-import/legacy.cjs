// This module's exports are only HALF assigned when `esm.js` links against it:
// `require("./esm.js")` runs the ES module (and everything it imports, including
// this file) in the middle of this file's evaluation.
exports.early = "early";
require("./esm.js");
exports.late = "late";
