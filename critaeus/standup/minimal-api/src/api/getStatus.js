const { send, json } = require('micro');
const { getObject, handleKeyNotFound, createClient } = require('../util.js');

const client = createClient();

const getDatesStatus = async (req, res) => {
  try {
    const { team, user, date } = req.params;
    const data = await getObject(`standup/${team}/${date}/${user}.json`)
    send(res, 200, data)
  } catch (e) {
    handleKeyNotFound(e, send.bind(null, res))
  }
}
