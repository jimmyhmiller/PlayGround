const { buildMessage } = require('./poll');
const { parseField, incrementVote, buildNewMessage } = require('./actions');


console.log = (...args) => {
  console.dir(args, {depth: null})
}


const message = buildMessage({ 
  question: "What are we going to eat?",
  answers: ["Chipotle", "Garden Table", "Something Else"]
})


const actionMessage = {
  original_message: message,
  "actions": [
    {
      "name": "1",
      "type": "button",
      "value": "1"
    }
  ]
}

console.log(message)
console.log(buildNewMessage(actionMessage))