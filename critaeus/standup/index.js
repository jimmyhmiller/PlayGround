#!/usr/bin/env node

const arg = require('arg');
const multi = require('multiple-methods');
const { magenta, green, blue, bold } = require('chalk');
const { table, getBorderCharacters } = require('table');
const inquirer = require('inquirer')

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
  console.log("")
  console.log(`Try adding a status`)
  console.log()
  console.log(`  \$ ${green('standup')} status "Got stuck on some technical debt."`)
})

command.method('status', ({_: [_, message]}) => {
  console.log("")
  console.log(green(`+ Status Recorded for ${team}`))
  console.log("")
  console.log(`To see your team's statuses run:`)
  console.log()
  console.log(`  \$ ${green('standup')} list`)
})

command.method('list', ({_: [_, __]}) => {
  console.log("")
  console.log(`${blue.underline(team)}`)
  console.log("")
  printTable([[blue("Jimmy"), "Got stuck on some technical debt."],
              [blue("Poindexter"), "Wrote a new javascript framework"],
              [blue("Russell"), "Stacked Pickle"]])
})

command.defaultMethod(() => {
  console.log("")
  console.log("To setup standup please add your name and team below")
  console.log("")
  inquirer.prompt([{name:"name", type:"input", "message": "name:"},
                   {name:"team", type:"input", "message": "team:"}])
    .then(({ team }) => {
      console.log("")
      console.log(`+ You've joined team ${team}`)
      console.log("")
      console.log(`Try adding a status`)
      console.log()
      console.log(`  \$ ${green('standup')} status "Got stuck on some technical debt."`)
    })
})


command(args)