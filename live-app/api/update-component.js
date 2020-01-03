const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });

const upsert = ({ name, code }) => {
  return client.query(
    q.Let(
      { name: q.Match("component_by_name", name) },
      q.If(
        q.Exists(q.Var("name")),
        q.Update(q.Select("ref", q.Get(q.Var("name"))), { data: { code } }),
        q.Create(q.Collection("component"), {
          data: {
            name,
            code
          }
        })
      )
    )
  );
};

module.exports = async (req, res) => {
  if (req.method === "POST" || req.method === "PUT") {
    const { name, code } = req.body;
    if (!name || !code) {
      console.log(req.body);
      res.status(400);
      res.send("Missing one of name or body")
      return;
    }
    const response = await upsert({ name, code})
    res.status(200);
    res.send({success: true});
  } else {
    console.log(req.method);
    res.status(405);
    res.send({message: "method not allowed"});
  }
 
};