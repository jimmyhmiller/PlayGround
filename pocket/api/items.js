const axios = require("axios");
const { send } = require("micro");
const url = require("url");
const querystring = require("querystring");
const cookie = require('cookie');
require("dotenv").config();

const itemsUrl = "https://getpocket.com/v3/get"
const consumer_key = process.env.POCKET_CONSUMER_KEY;

const getItems = async ({ accessToken: access_token, count=100, offset=0 }) => {
  const response = await axios.post(itemsUrl, {
    consumer_key,
    access_token,
    count,
    offset,
    detailType: "complete",
    sort: "oldest",
  })
  return response.data;
}

const getAccessToken = (req) =>
  (cookie.parse(req.headers.cookie || '').access_token) ||
  req.headers.authorization

module.exports = async (req, res) => {
  const accessToken = getAccessToken(req);
  if (!accessToken) {
    send(res, 401, "Unauthorized")
  }

  const { count, offset } = querystring.parse(url.parse(req.url).query);

  const response = await getItems({ accessToken, count, offset })
  send(res, 200, response)
};