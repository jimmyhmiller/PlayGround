module.exports = {
  read() {
    const esm = require("./esm.js");
    return esm.state;
  },
};
