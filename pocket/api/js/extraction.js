const mercury = require('@postlight/mercury-parser');
const querystring = require("querystring");
const url = require('url');
const { send } = require("micro");

module.exports = async (req, res) => {
  const { url: urlParam } = querystring.parse(url.parse(req.url).query);
  const result = await mercury.parse(urlParam);
  send(res, 200, result);
}

