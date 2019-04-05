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

module.exports = async (req, res) => {
  if (req.method === 'GET') {
    const { id } = params(req)
    console.log(id)
    send(res, 200, await transactor.getState(id))
    return;
  }

  const { id, action } = await json(req);
  const result = await transactor.transact(id, action);

  send(res, 200, result);
};
