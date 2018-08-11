describe("sort", function () {
  jsc.property("idempotent", "array nat", function (arr) {
    return _.isEqual(sort(sort(arr)), sort(arr));
  });
});