import { count, increment } from "./counter.js";

console.log(`count:${count}`);
increment();
console.log(`count:${count}:${JSON.stringify({ count })}`);
