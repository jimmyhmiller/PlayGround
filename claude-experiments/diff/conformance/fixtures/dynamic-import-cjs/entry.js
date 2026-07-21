import("./legacy.cjs").then((ns) => {
  console.log("default:" + ns.default.tag);
  console.log("has-tag:" + ("tag" in ns));
});
