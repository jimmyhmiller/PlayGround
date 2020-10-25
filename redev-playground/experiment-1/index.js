const http = require("http")
const micro = require("micro")
const util = require("util");
const { send, json } = require('micro')


// Need predicate based ways of transforming
// Need a way of referring to other urls for functions
// Need a way to add packages
// Need a way to execute things that aren't js (Bootstrap that?)


const resources = {
  "/": {
    headers: { "Content-Type": "text/html" },
    body: "<h1>Hello World!</h1>",
  },
  "/_meta": {
    "execute": "module.exports = async (req, res) => resources"
  }
};


const addResource = ({ url, payload }) => {
  if (payload.onBuild) {
    const result = eval(`
      module = {};
      ${payload.onBuild}
      module
    `);
    const f = result.exports;
    payload = f(payload);
  }
  resources[url] = payload;
}

const executeResource = async (req, res, resource) => {
  const handler = eval(`
      module = {};
      ${resource.execute}
      module
    `)
  const result = await handler.exports(req, res, resource);
  return result;
}

const handler = async (req, res) => {

  if (req.method === "POST" && req.url === "/routes") {
    const body = await json(req);
    addResource({url: body.url, payload: body})
    send(res, 201, resources);
    return;
  }
  if (!resources.hasOwnProperty(req.url)) {
    send(res, 404, {status: "Not Found"})
    return;
  }
  const resource = resources[req.url] || {};
  const { headers, body, execute } = resource;


  for ([key, value] of Object.entries(headers || {})) {
    res.setHeader(key, value);
  }

  if (execute) {
    const result = await executeResource(req, res, resource);
    return result;
  }
  return body;
}

module.exports = handler;
