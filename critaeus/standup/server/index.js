var express = require('express');
var app = express();
app.use(express.json())



// s3-files
// /<team>/<date>/statuses


// GET  teams/<team>/standup/summary
// { "summary": [{"user": "Jimmy", "status": "Got stuck on tech debt."}]}
// GET  teams/<team>/standup/details
// { "details": [{"user": "Jimmy", 
//                 "status": "Got stuck on tech debt. Other things."}]}
// GET  <team>/standup/history
// { "history":
//   [{"date": "2018-06-17",
//     "statuses": [{"user": "Jimmy", 
//                   "status": "Got stuck on tech debt. Other things."}]}]}
// POST teams/<team>/standup
// {"status": "Here's my status"}
// PUT  teams/<team>/join
// {"user": "Jimmy"}


const getDate = () => {
    const dateTime = new Date().toISOString();
    return dateTime.substring(0, dateTime.indexOf("T"))
}

const state = {
    teams: {}
}

const today = 

app.get('/teams/', ({params : { team }}, res) => {
    res.send(state.teams)
})

app.post('/teams/:team', ({params : { team }}, res) => {
    state.teams[team] = {standup: {}}
    res.send({team: `${team}`})
})

app.post('/teams/:team/standup', ({params : { team }, body: { status, name}}, res) => {
    const date = getDate()
    state.teams[team].standup[date] = state.teams[team].standup[date] || []
    state.teams[team].standup[date].push({status, name});
    res.send(state.teams[team].standup[date])
})

app.listen(3342, () => 
    console.log('Example app listening on port 3342!'))