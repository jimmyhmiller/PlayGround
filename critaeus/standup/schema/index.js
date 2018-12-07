require('dotenv').config()
const faunadb = require('faunadb');
const q = faunadb.query;


// This is getting really clunky. Should I just try a clojurescript repl?

const log = (...x) => console.log(x);
const client = new faunadb.Client({ secret: process.env.SECRET })

const queryLog = (query) => () => {
  console.log(query);
  return client.query(query)
    .then(log)
    .catch(log)
}

const runQuery = (query) => queryLog(query)();

const schema = [
  q.CreateDatabase({ name: "standup"}),
  q.CreateClass({ name: "users" }),
  q.CreateClass({ name: "teams" }),
  q.CreateClass({ name: "statuses" }),
]

const jimmy = q.Ref(q.Class("users"), "217993811723289100");
// runQuery(q.Create(q.Class("users"), {data: {name: "Jimmy"}}))
// runQuery(q.Update(
//     q.Ref(q.Class("users"), "217993811723289100"), 
//     { credentials: { password: "password" } }))


runQuery(q.Login(jimmy, { password: "password" }))



// q.Ref(q.Class("teams"), "217993851172815371")
// runQuery(q.Create(q.Class("teams"), {data: {name: "Awesome"}}))

// runQuery(q.Create(q.Class("teams"), {data: {name: "Awesome"}}))


const runQueries = (queries) => {
  queries
    .reduce((p, query) => {
      return p.then(queryLog(query))
    }, Promise.resolve(null))
    .then(log)
    .catch(log)
}

