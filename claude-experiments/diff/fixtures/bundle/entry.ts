import { add } from "./math.js";
import message from "./message.js";

console.log(`${message}: ${add(20, 22)}`);
import("./lazy.js").then(({ lazy }) => console.log(lazy));
