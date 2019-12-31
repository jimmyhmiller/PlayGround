const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });

module.exports = async (req, res) => {
  const routes = await client.query(
    q.Map(
      q.Paginate(q.Match(q.Index("all_endpoint"))),
      q.Lambda("x", q.Select("data", q.Get(q.Var("x"))))
    )
  );
  res.send(routes.data);
};