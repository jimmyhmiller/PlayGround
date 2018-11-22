const { send, json } = require('micro');
const { listDirectory, createClient } = require('../util.js');

const client = createClient();

const viewTeamStatus = async (req, res) => {
  try {
    const { team, date } = req.params;
    const statues = await listDirectory(`standup/${team}/${date}/`);
    send(res, 200, statues)
  } catch (e) {
    console.log(e)
    send(res, 404)
  }
}

module.exports = viewTeamStatus;