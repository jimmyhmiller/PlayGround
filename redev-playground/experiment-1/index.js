const http = require("http")
const micro = require("micro")
const util = require("util");
const { send, json } = require('micro')


// Need a runtime equivilent to on build
// Need predicate based ways of transforming
// Should make api endpoints work next
// Need a way of referring to other urls for functions
// Need a way to add packages
// Need a way to execute things that aren't js (Bootstrap that?)


const resources = {
  "/": {
    headers: { "Content-Type": "text/html" },
    body: "<h1>Hello World!</h1>",
  },
};


const addResource = ({ url, payload }) => {
  if (payload.build) {
    const result = eval(`
      module = {};
      ${payload.build}
      module
    `);
    const f = result.exports;
    payload = f(payload);
  }
  resources[url] = payload;
}

const handler = async (req, res) => {

  if (req.method === "POST") {
    const body = await json(req);
    addResource({url: req.url, payload: body})
    send(res, 201, resources);
    return;
  }

  const { headers, body } = resources[req.url] || {};
  if (!body) {
    send(res, 404, {status: "Not Found"})
    return;
  }

  for ([key, value] of Object.entries(headers || {})) {
    res.setHeader(key, value);
  }
  return body;
}

module.exports = handler;
