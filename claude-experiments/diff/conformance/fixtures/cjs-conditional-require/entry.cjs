function load(flag) {
  return flag ? require("./used.cjs") : require("./other.cjs");
}
console.log("start");
const m = load(true);
console.log("value:" + m.value);
