const { createTransactor } = require("../transactions");
const { json, send } = require("micro");
const { params } = require("../utils")
const parseUrlEncode = require('urlencoded-body-parser');

const multi = require("multiple-methods");

const reducer = multi((_, action) => action.type);

reducer.method("RELEASE", (state, { resource, }) => {
  delete state[resource];
  return state;
});

reducer.method("CLAIM", (state, { resource, user_name }) => {
  if (!state[resource]) {
    return {
      ...state,
      [resource]: `${resource} was claimed by @${user_name} at ${new Date().toISOString()}`
    }
  }
});

reducer.defaultMethod(() => {})

const transactor = createTransactor(reducer);


const parseMessage = (text="") => {
  const result = text.split(" ")

  if (result[0] === "release") {
    return {
      type: "RELEASE",
      resource: result[1]
    }
  }
  else {
    return {
      type: "CLAIM",
      resource: result[0]
    }
  }
}


const handler = async (req, res) => {
  const { text, channel_id, user_name } = await parseUrlEncode(req);
  const action = parseMessage(text);
   
  await transactor.transact(channel_id, {
    ...action,
    user_name,
  });

  const state = await transactor.getState(channel_id);

  if (action.type === "RELEASE") {
    send(res, 200, {text: `${action.resource} has been released.`})
  } else {
    console.log("here")
    send(res, 200, {text: state[action.resource]})
  }
}


module.exports = handler
