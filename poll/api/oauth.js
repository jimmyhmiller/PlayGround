require('dotenv').config();
const request = require('request');
const url = require('url');
const querystring = require('querystring');
const { send } = require('micro');


const rootUrl = "https://slack.com/api/oauth.access"

const clientInfo = {
  client_id: process.env.CLIENT_ID,
  client_secret: process.env.CLIENT_SECRET
}

// Copied from slack tutorial, needs clean up
module.exports = (req, res) => {
  const params = querystring.parse(url.parse(req.url).query);
  const code = params.code;
  const requestParams = {
    ...clientInfo,
    code,
  }
  const options = {
    uri: `${rootUrl}?${querystring.stringify(requestParams)}`,
    method: 'GET'
  }

  request(options, (error, response, body) => {
    const json = JSON.parse(body)
    console.log(json)

    if (json.ok) {
      send(res, 200, "Success!");
    } else {
      send(res, 400,"Error encountered: \n" + JSON.stringify(json),)
    }
  })
}