import d, { named } from "./marked.cjs";
console.log("type:" + typeof d);
console.log("default-prop:" + (d && d.default));
console.log("named:" + named);
