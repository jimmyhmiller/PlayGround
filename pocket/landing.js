const axios = require("axios");
const { send } = require("micro");
const querystring = require("querystring");
const url = require('url');
require("dotenv").config();

const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.FAUNA_SECRET });

const consumer_key = process.env.POCKET_CONSUMER_KEY;
const authorizeUrl = "https://getpocket.com/v3/oauth/authorize";

const itemsUrl = "https://getpocket.com/v3/get"

const getRequestToken = async userId => {
  const user = await client.query(
    q.Get(q.Ref(q.Class("pocket-users"), userId))
  );
  await client.query(q.Delete(q.Ref(q.Class("pocket-users"), userId)))
  return user.data.requestToken;
};

const getAccessTokenData = async code => {
  const response = await axios({
    url: authorizeUrl,
    method: "post",
    data: {
      code,
      consumer_key,
    },
    headers: {
      "content-type": "application/json; charset=UTF-8",
      "x-accept": "application/json"
    }
  })
  return response.data
};

const getItems = async (access_token) => {
  const response = await axios.post(itemsUrl, {
    consumer_key,
    access_token,
    count: 100,
    detailType: "complete"
  })
  return response.data;
}

module.exports = async (req, res) => {
  const { userId } = querystring.parse(url.parse(req.url).query);
  const requestToken = await getRequestToken(userId);
  const { access_token } = await getAccessTokenData(requestToken);
  const items = await getItems(access_token);
  send(res, 200, items)
};