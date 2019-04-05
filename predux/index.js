const { createTransactor } = require("./transactions");
const { json, send } = require("micro");
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


const transact = createTransactor(reducer);

module.exports = async (req, res) => {
  const { id, action } = await json(req);
  const result = await transact(id, action);

  send(res, 200, result);
};
