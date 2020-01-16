const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });

const createCollection = async ({ name }) => {
  await client.query(
    q.CreateCollection({ name, history: 0 }),
  )

  await client.query(
    q.CreateIndex({name: `all_${name}`, source: q.Collection(name)})
  )

  await client.query(
    q.CreateIndex({name: `${name}_by_id`, source: q.Collection(name), terms: [{ field: ['data', 'id'] }]})
  )

  return { success: true}
}

const listCollections = () => {
  return client.query(
    q.Map(
      q.Select("data", q.Paginate(q.Collections())),
      q.Lambda("x", {name: q.Select("name", q.Get(q.Var("x")))})
    )
  );
};

module.exports = async (req, res) => {

  if (req.method === "GET") {
    res.send({collections: await listCollections()})
  }
  else if (req.method === "POST") {
    res.send(await createCollection(req.body))
  }

};