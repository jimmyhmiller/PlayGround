#!/usr/bin/env node

const arg = require('arg');
const multi = require('multiple-methods');
const { magenta, green, blue, bold } = require('chalk');
const { table, getBorderCharacters } = require('table');
const inquirer = require('inquirer')


// s3-files
// /<team>/<date>/statuses


// GET  <team>/standup/summary
// { "summary": [{"user": "Jimmy", "status": "Got stuck on tech debt."}]}
// GET  <team>/standup/details
// { "details": [{"user": "Jimmy", 
//                 "status": "Got stuck on tech debt. Other things."}]}
// GET  <team>/standup/history
// { "history":
//   [{"date": "12/31/2017".
//     "statuses": [{"user": "Jimmy", 
//                   "status": "Got stuck on tech debt. Other things."}]}]}
// POST <team>/<name>
// POST <team>/standup
// {"status": "Here's my status"}
// PUT  <team>/join
// {"user": "Jimmy"}


var tab = require('tabtab')({
  name: 'standup',
  cache: false
});


tab.on('standup', function(data, done) {
  done(null, ['create:Create a new team',
              'configure:Configure standup',
              'summary:See a summary of your team\'s statuses',
              'details:See the details of your team\'s statuses',
              'history:View past standups',
              'join:Join a team',
              'status:Leave a status for your team',
              'feedback:Provide feedback on how to improve standup',
              'tutorial:Learn how to use standup']);
});

const printTable = (data) => {
  console.log(table(data, {
    border: getBorderCharacters(`void`),
    columnDefault: {
      paddingLeft: 0,
      paddingRight: 4
    },
    drawHorizontalLine: () => false
  }));
}

function getQuestion() {
  return bold('> ' + this.opt.message) + ' '
}

inquirer.prompt.prompts.input.prototype.getQuestion = getQuestion
inquirer.prompt.prompts.list.prototype.getQuestion = getQuestion

const args = arg({});

const command = multi(args => args._[0])

const team = "DevTeam6"

command.method('join', ({_: [_, team]}) => {
  console.log("")
  console.log(`+ Joined team ${team}`)
})

command.method('status', ({_: [_, message]}) => {
  console.log("")
  console.log(green(`+ Status Recorded for ${team}`))
})

command.method('list', ({_: [_, __]}) => {
  console.log("")
  console.log(`${blue.underline(team)}`)
  console.log("")
  printTable([[blue("Jimmy"), "Got stuck on some technical debt."],
              [blue("Poindexter"), "Wrote a new javascript framework"],
              [blue("Russell"), "Stacked Pickle"]])
})

command.defaultMethod(({_: args}) => {
  if (args.length > 1) {
    return;
  }
})

tab.start();

command(args)