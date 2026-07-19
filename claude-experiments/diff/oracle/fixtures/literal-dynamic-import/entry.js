console.log("before");
import("./lazy.js").then(({ value }) => console.log(`lazy:${value}`));
