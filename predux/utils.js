const querystring = require('querystring');
const url = require("url");

const last = (coll) => coll[coll.length-1]

const params = (req) => console.log(last(req.url.split("/"))) || ({
  id: last(req.url.split("/")),
  ...querystring.parse(url.parse(req.url).query)
}) 

module.exports = {
  params
}