require('now-env')
const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.FAUNA_SECRET });

const createTransactor = (reducer, initialState) => async (id, action) => {
  while (true) {
    const state = await client.query(
      q.Do(
        q.If(q.Not(q.Exists(q.Match(q.Index("state-by-id"), id))),
          q.Create(q.Class("state"), {data: {id, ...await reducer(initialState, {})}}),
          1),
      q.Get(q.Match(q.Index("state-by-id"), id))))
    const newState = await reducer(state.data, action)
    try {
      const result = await client.query(
        q.Let({e: q.Get(state.ref)},
              q.If(q.Equals(q.Select("ts", q.Var("e")), state.ts),
                q.Update(state.ref, {data: newState}),
                q.Abort("Timestamp changed")))
      )
      return result.data
    } catch (e) {
      if (e.requestResult.responseContent.errors[0].description === "Timestamp changed") {
        const result = await createTransactor(reducer, initialState)(id, action)
        return result
      }
    } 
  }
}


module.exports = {
  createTransactor
}