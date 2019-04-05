const { createTransactor } = require("./transactions");
const { json, send } = require("micro");
const { params } = require("./utils")

const multi = require("multiple-methods");

const reducer = multi((_, action) => action.type);

reducer.method("INCREMENT", (state, _) => ({
  count: state.count + 1
}));

reducer.method("DECREMENT", (state, _) => ({
  count: state.count - 1
}));

reducer.method("SET", (state, { count }) => ({
  count: count
}));

reducer.method("GET", (state, _) => state);

reducer.defaultMethod(state => state || { count: 0 })



const transactor = createTransactor(reducer);

const handler = multi(req => req.method)

handler.method('GET', async (req, res) => {
  const { id } = params(req)
  const result = await transactor.getState(id)
  send(res, result.error ? 404 : 200, result)
})

handler.method('POST', async (req, res) => {
  const { id, action } = await json(req);
  const result = await transactor.transact(id, action);
  send(res, 200, result);
})

module.exports = handler
