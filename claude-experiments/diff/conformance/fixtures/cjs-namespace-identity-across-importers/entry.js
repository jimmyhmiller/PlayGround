// Namespace identity is per MODULE: two importers of one CommonJS module see the
// same namespace object, and two modules that happen to export the same primitive
// do not.
import * as a from "./a.js";
import * as b from "./b.js";
import * as c from "./c.js";
console.log("same-module:" + (a.legacy === b.legacy));
console.log("different-module:" + (a.legacy === c.legacy));
