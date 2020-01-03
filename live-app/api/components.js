const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });

module.exports = async (req, res) => {
  const components = await client.query(
    q.Map(
      q.Paginate(q.Match(q.Index("all_component"))),
      q.Lambda("x", q.Select("data", q.Get(q.Var("x"))))
    )
  );
  res.send(components.data);
};