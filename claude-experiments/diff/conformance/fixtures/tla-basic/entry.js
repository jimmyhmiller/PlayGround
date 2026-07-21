console.log("tla:before");
const v = await Promise.resolve("tla-value");
console.log("tla:after:" + v);
