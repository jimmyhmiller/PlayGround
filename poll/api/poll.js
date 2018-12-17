const { send } = require('micro');
const parseUrlEncode = require('urlencoded-body-parser');

const cleanString = (str) =>
  str.replace(/(\u201C|\u201D)/g, '')
     .replace(/"/g, '')

const parseBody = async (req) => {
  const body = await parseUrlEncode(req);
  return cleanString(body.text).split(",")
}

const buildActions = (answers) => {
  return answers.map((answer, i) => ({
    text: `${answer}`,
    type: "button",
    value: `${i}`,
    name: `${i}`,
  }))
}

const buildFields = (answers) => {
  return answers.map((answer, i) => ({
    value: `• ${answer}`,
    short: false,
  }))
}

const buildMessage = ({ question, answers }) => {
  return {
    response_type: "in_channel",
    replace_original: "false",
    "attachments": [{
      "pretext": "This survey is anonymous",
        "title": question,
      "mrkdwn_in": ["fields"],
      "fields": buildFields(answers),
      "fallback": "Your interface does not support interactive messages.",
      "callback_id": "123",
      actions: buildActions(answers)
    }]
  }
}


module.exports = async (req, res) => {
  try {
    const [question, ...answers] = await parseBody(req);
    send(res, 200, buildMessage({
      question,
      answers,
    }));
  } catch (e) {
    send(res, 200, {
      text: `Failed to get body ${e.message}`,
      response_type: "ephemeral",
      replace_original: false
    })
  }
};


// module.exports = {
//   buildMessage
// }




