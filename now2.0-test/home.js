const { send, json } = require('micro');
const { decorate } = require("./util.js")

module.exports = (req, res) => {
  send(res, 200, decorate({thing: `Hi from Now Lambda`}))
};

// routing isn't working