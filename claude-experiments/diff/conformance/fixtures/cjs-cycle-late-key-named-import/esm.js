// `late` does not exist on `module.exports` yet when this module links. Node
// still ACCEPTS the import — `cjs-module-lexer` found the text `exports.late =`
// in the source, so the name is in the namespace — and gives it the value it had
// when the namespace was materialized, i.e. `undefined`.
import { early, late } from "./legacy.cjs";
export function report() { return "early:" + early + " late:" + late; }
