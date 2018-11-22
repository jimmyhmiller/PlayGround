const { send, json } = require('micro');
const { putObject, createClient } = require('../util.js');

const client = createClient();

const createUser = async (req, res) => {
  try {
    const body = await json(req)
    const { user } = req.params;
    await putObject(`user/${user}`, body);

    send(res, 201)
  } catch (e) {
    send(res, 404)
  }
}

module.exports = createUser;

