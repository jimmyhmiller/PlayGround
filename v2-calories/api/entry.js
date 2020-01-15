const faunadb = require("faunadb");
const q = faunadb.query;
const { startOfToday } = require("date-fns"); 
const {format: formatDate, utcToZonedTime} = require("date-fns-tz");

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


// Maybe I want to combine these queries? Turn them into functions?
// Not really sure yet.
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
   q.Divide(q.ToDouble(q.Subtract(q.Multiply(q.Var("days"), bmr), q.Var("total"))), 3500.0))
  )
}

const timeZone = 'America/Chicago'
const today = () => formatDate(utcToZonedTime(startOfToday(), timeZone), "yyyy-MM-dd", { timeZone });

const getRemainingToday = () => {
  return client.query(
    // I wish they did the map and filter argument order correctly
    q.Let({total: q.Sum(
                      q.Select("data", q.Map(
                        q.Filter(
                          q.Paginate(q.Match("all_entry")),
                          q.Lambda("x", q.Equals(today(), q.Select(["data", "date"], q.Get(q.Var("x")))))),
                        q.Lambda("x", q.Select(["data", "calories"], q.Get(q.Var("x")))))),
                      )},
   // 500 = 1lb per week
   q.Subtract(bmr, 500, q.Var("total")))
  )
}





module.exports = async (req, res) => {
  if (req.method === "GET" && req.query.total) {
    res.json({total: await getTotal()});
  }
  else if (req.method === "GET" && req.query.pounds) {
    res.json({total: await getPounds()});
  }
  else if (req.method === "GET" && req.query.today) {
    res.json({remaining: await getRemainingToday()});
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
