require('now-env')
const multi = require("multiple-methods")
const produce = require("immer").produce
const util = require('util')

const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.FAUNA_SECRET });

const log = console.log;
console.log = x => log(util.inspect(x, false, null, true))

const translate = multi(x => x.op)
translate.method("replace", ({ op, path, value }) => {
  return x => {
    if (path && path.length) { 
      return q.Update(x, {
          data: {
            [path]: value
          }
        })
    } else {
      return q.Replace(x, {data: value})
    }
  }
})

const transform = (state, draftfn) => {
  return new Promise(resolve => {
    produce(state, draftfn, (patches) => console.log(patches) || resolve(patches.map(translate)))
  })
}

const transact = (reducer, initialState) => async (id) => {
  const state = await client.query(
    q.Do(
      q.If(q.Not(q.Exists(q.Match(q.Index("state-by-id"), id))),
        q.Create(q.Class("state"), {data: {id, ...initialState}}),
        1),
    q.Get(q.Match(q.Index("state-by-id"), id))))
  console.log(state)
  const queries = await transform(state.data, reducer)
  await client.query(q.Do(queries.map(x => x(state.ref))))
}


// I've got something seriously powerful here. A persistent reducer.
// Got to think about the other parts of redux to consider if they make sense here.
// Immer is a good stepping stone, but it doesn't record things the way I need for 
// consistency.

// Could I actually  use timestamp for consistency?
// If so, I don't have to care how your code is arranged.

// Whatever I do, asserting is also a good idea.
const run = async () => {

  try {
    await transact(x => ({ ...x, count: x.count + 1 }), {count: 0})("122")
  } catch (e) {
    console.error(e)
  }
  
}
run()






module.exports = (req, res) => {
  res.end(`Hello from Node.js on Now 2.0!`);
};