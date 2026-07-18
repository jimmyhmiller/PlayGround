Promise.all([import("./alpha.js"), import("./beta.js")]).then(([alpha, beta]) => {
  console.log(alpha.value, beta.value);
});
