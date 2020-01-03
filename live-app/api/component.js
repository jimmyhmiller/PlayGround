const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });

module.exports = async (req, res) => {
  const { name } = req.query;
  const component = await client.query(
    q.Select("data", q.Get(q.Match("component_by_name", name)))
  );
  res.json(component);
};