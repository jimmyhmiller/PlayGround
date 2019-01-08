const request = require('request');
const read = require('node-readability');


module.exports = (req, res) => {
  read("https://jimmyhmiller.github.io/incommunicability/",
    function (error, article, body) {
      res.end(error.message);
    })
};