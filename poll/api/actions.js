const { send, json, text } = require('micro');
const parseUrlEncode = require('urlencoded-body-parser');
const { setIn } = require('zaphod/compat');


// Slack is stupid and sends form encoded json
const parseBody = async (req) => {
  const body = await parseUrlEncode(req);
  return JSON.parse(body.payload);
}

const parseField = (value) => {
  const [_, prefix, __, vote] = value.match(/^(.*?) ?(`([0-9]+)`)?$/)
  return {
    prefix,
    vote: (vote && parseInt(vote)) || 0
  }
}

const incrementVote = (index, original_message) => {
  const { prefix, vote } = parseField(original_message.attachments[0].fields[index].value)
  return setIn(original_message, ["attachments", 0, "fields", index, "value"], `${prefix} \`${vote + 1}\``)
}

const getActionIndex = (body) => {
  return parseInt(body.actions[0].value)
}

const buildMessage = (body) => {
  return {
    ...incrementVote(getActionIndex(body), body.original_message),
    response_type: "in_channel",
    replace_original: true
  }
}

module.exports = async (req, res) => {
  try {
    const body = await parseBody(req);
    send(res, 200, buildMessage(body));
  } catch (e) {
    send(res, 200, {
      text: `Failed to get body ${e.message}`,
      response_type: "ephemeral",
      replace_original: false
    })
  }
};

// module.exports = {
//   parseField,
//   incrementVote,
//   buildNewMessage: buildMessage
// }



