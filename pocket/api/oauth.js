const axios = require("axios");
const { send } = require("micro");
const querystring = require("querystring");
require("dotenv").config();

const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.FAUNA_SECRET });

const consumer_key = process.env.POCKET_CONSUMER_KEY;
const redirect_url = "https://pocket-app.now.sh/api/landing";

const requestUrl = "https://getpocket.com/v3/oauth/request";
const authorizeUrl = "https://getpocket.com/auth/authorize";

const createUser = async () => {
  const user = await client.query(q.Create(q.Class("pocket-users")));
  return user.ref.id;
};

const updateUser = async (userId, requestToken) =>
  client.query(
    q.Update(q.Ref(q.Class("pocket-users"), userId), {
      data: {
        requestToken
      }
    })
  );

const getRequestToken = async redirect_uri => {
  const response = await axios({
    url: requestUrl,
    method: "POST",
    data: {
      consumer_key,
      redirect_uri
    },
    headers: {
      "Content-Type": "application/json; charset=UTF-8",
      "X-Accept": "application/json"
    }
  });

  return response.data.code;
};

module.exports = async (req, res) => {
  try {
    const userId = await createUser();
    const userRedirectUrl = `${redirect_url}?userId=${userId}`;

    const request_token = await getRequestToken(userRedirectUrl);

    await updateUser(userId, request_token);

    const redirectInfo = {
      request_token,
      redirect_uri: userRedirectUrl
    };

    res.setHeader(
      "Location",
      `${authorizeUrl}?${querystring.stringify(redirectInfo)}`
    );
    send(res, 302);
  } catch (e) {
    send(res, 500, {
      data: e.response.data,
      status: e.response.status,
      headers: e.response.headers
    });
  }
};
