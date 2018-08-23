var jsc = require("jsverify");
var assert = require('assert');
const { reverse } = require('../index.js')

describe("sort", function () {
  jsc.property("idempotent", "array nat", function (arr) {
    console.log(arr)
    return arr.length < 4
  });
});