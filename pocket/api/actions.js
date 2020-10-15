const axios = require("axios");
const { send } = require("micro");
const url = require("url");
const cookie = require('cookie');
const querystring = require("querystring");
require("dotenv").config();


const sendUrl = "https://getpocket.com/v3/send"
const consumer_key = process.env.POCKET_CONSUMER_KEY;

const getAccessToken = (req) =>
  (cookie.parse(req.headers.cookie || '').access_token) ||
  req.headers.authorization

const sendAction = async ({ accessToken: access_token, action, item_id }) => {
  const response = await axios.post(sendUrl, {
    consumer_key,
    access_token,
    actions: [{ action, item_id }],
  })
  return response.data;
}

module.exports = async (req, res) => {
  const accessToken = getAccessToken(req);
  if (!accessToken) {
    send(res, 401, "Unauthorized")
  }

  const { action, item_id } = querystring.parse(url.parse(req.url).query);

  const response = await sendAction({ accessToken, action, item_id })
  send(res, 200, response)
};