const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });

const upsert = ({ route, code }) => {
  return client.query(
    q.Let(
      { route: q.Match("endpoint_by_route", route) },
      q.If(
        q.Exists(q.Var("route")),
        q.Update(q.Select("ref", q.Get(q.Var("route"))), { data: { code } }),
        q.Create(q.Collection("endpoint"), {
          data: {
            route,
            code
          }
        })
      )
    )
  );
};

module.exports = async (req, res) => {
  if (req.method === "POST" || req.method === "PUT") {
    const { route, code } = req.body;
    if (!route || !code) {
      console.log(req.body);
      res.status(400);
      res.send("Missing one of route or body")
      return;
    }
    const response = await upsert({ route, code})
    res.status(200);
    res.send({success: true});
  } else {
    console.log(req.method);
    res.status(405);
    res.send({message: "method not allowed"});
  }
 
};