const { send, json } = require('micro');
const { putObject, createClient } = require('../util.js');

const client = createClient();

const postStatus = async (req, res) => {
  try {
    const body = await json(req)
    const { team, user, date } = req.params;
    await putObject(`standup/${team}/${date}/${user}.json`, body)
    send(res, 201)
  } catch (e) {
    console.log(e)
    send(res, 500, e)
  }
}

module.exports = postStatus;