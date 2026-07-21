import { count, increment } from "./counter.js";
console.log("before:" + count);
increment();
increment();
console.log("after:" + count);
