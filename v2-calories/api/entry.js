const faunadb = require("faunadb");
const q = faunadb.query;

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });



const db = {
  getAll: (collection) => {
    return client.query(
      q.Select(
        "data",
        q.Map(
          q.Paginate(q.Match(q.Index(`all_${collection}`))),
          q.Lambda("x", q.Select("data", q.Get(q.Var("x"))))
        )
      )
    )
  },
  create: (collection, entity) => {
    return client.query(
      q.Select("data",
        q.Create(q.Collection(collection), {
          data: entity
        })
      )
    )
  },

  delete: (collection, id) => {
    client.query(
      q.Delete(q.Select("ref", q.Get(q.Match(`${collection}_by_id`, id))))
    );
  },
}

const getTotal = () => {
  return client.query(
    q.Sum(
      q.Select(
        "data",
        q.Map(
          q.Paginate(q.Match("all_entry")),
          q.Lambda("x", q.Select(["data", "calories"], q.Get(q.Var("x"))))
        )
      )
    )
  );
};

const bmr = 2260;

const getPounds = () => {
  return client.query(
    q.Let({total: q.Sum(
                q.Select("data", q.Map(
                  q.Paginate(q.Match("all_entry")),
                  q.Lambda("x", q.Select(["data", "calories"], q.Get(q.Var("x"))))))),
        days: q.Count(q.Distinct(q.Select("data", q.Map(
                  q.Paginate(q.Match("all_entry")),
                  q.Lambda("x", q.Select(["data", "date"], q.Get(q.Var("x"))))))))},
   q.Divide(q.ToDouble(q.Subtract(q.Multiply(q.Var("days"), 2260), q.Var("total"))), 3500.0))
  )
}



module.exports = async (req, res) => {
  if (req.method === "GET" && req.query.total) {
    res.json({total: await getTotal()});
  }
  if (req.method === "GET" && req.query.pounds) {
    res.json({total: await getPounds()});
  }
  else if (req.method === "GET") {
    res.send(await db.getAll("entry"));
  }
  else if (req.method === "POST" || req.method === "PUT") {
    res.send(await db.create("entry", req.body));
  }
  else if (req.method === "DELETE") {
    res.send(await db.delete("entry", req.query.id))
  }
};
