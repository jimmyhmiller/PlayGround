const { send, json } = require('micro');
const { objectExists, createClient } = require('../util.js');

const client = createClient();

const userExists = async (req, res) => {
  try {
    const { user } = req.params;
    const exists = await objectExists(`user/${user}`);
    send(res, 200, exists)
  } catch (e) {
    send(res, 404)
  }
}

module.exports = userExists;