console.log("before");
import("./mod.js").then((m) => {
  console.log("loaded:" + m.value);
});
