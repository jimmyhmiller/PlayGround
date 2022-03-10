const faunadb = require("faunadb");
const q = faunadb.query;
const {format: formatDate, utcToZonedTime} = require("date-fns-tz");

const client = new faunadb.Client({ secret: process.env.DATABASE_SECRET });



const db = {
  getAll: (collection) => {
    return client.query(
      q.Select(
        "data",
        q.Map(
          q.Paginate(q.Match(q.Index(`all_${collection}`)), {size: 10000}),
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

}


const getTotal = (entries) => {
  return q.Sum(
      q.Select(
        "data",
        q.Map(
          entries,
          q.Lambda("x", q.Select(["data", "calories"], q.Get(q.Var("x"))))
        )
      )
    )
};

const numberOfDays = (entries) => {
  return q.Count(
    q.Distinct(q.Select("data", 
      q.Map(
        entries,
        q.Lambda("x", q.Select(["data", "date"], q.Get(q.Var("x"))))))));
}

const bmr = 2471;
// I got this wrong before, adjusting now.
const bmrDiff = bmr - 2260
// 500 = 1lb per week;
const goal = 500;

const getPounds = (total, numberOfDays) => {
  return q.Divide(q.ToDouble(q.Subtract(q.Multiply(numberOfDays, bmr), total)), 3500.0)

}


const extraCalories = (total, numberOfDays) => {
  return q.Subtract(q.Multiply(numberOfDays, bmr - goal), total)
}


const timeZone = 'America/New_York'
const today = () => formatDate(utcToZonedTime(new Date(), timeZone), "yyyy-MM-dd", { timeZone });


// There has to be a better way. I should probably make an index by day?
const getRemainingToday = (entries) => {
  return q.Let({total: q.Sum(
                        q.Select("data", q.Map(
                          q.Filter(
                            entries,
                            q.Lambda("x", q.Equals(today(), q.Select(["data", "date"], q.Get(q.Var("x")))))),
                          q.Lambda("x", q.Select(["data", "calories"], q.Get(q.Var("x")))))),
                      )},
   q.Subtract(bmr, goal, q.Var("total")))
}


const summary = () => {
  return client.query(
    q.Let({
      entries: q.Paginate(q.Match("all_entry"), {size: 10000}),
      total: getTotal(q.Var("entries")),
      days: numberOfDays(q.Var("entries")),
      remaining: getRemainingToday(q.Var("entries")),
      extra: extraCalories(q.Var("total"), q.Var("days")),
      pounds: getPounds(q.Var("total"), q.Var("days")),
      two: q.Subtract(q.Var("extra"), q.Multiply(q.Var("days"), 500)),
    },
      q.ToObject([
        ["remaining", q.Var("remaining")], 
        ["extraOnePound", q.Var("extra")],
        ["extraTwoPounds", q.Var("two")],
        ["extraTwoOriginal", q.Subtract(q.Var("two"), q.Multiply(bmrDiff, q.Var("days")))],
        ["days", q.Var("days")],
        ["weeks", q.Divide(q.Var("days"), 7)],
        ["projectedLoss", q.Multiply(q.Divide(q.Var("days"), 7), 2)],
        ["total", q.Var("total")],
        ["pounds", q.Var("pounds")],
        ["daily", bmr - goal],
      ]))
  )
}




module.exports = async (req, res) => {
  if (req.method === "GET" && req.query.summary) {
    res.json({summary: await summary()});
  }
  else if (req.method === "GET") {
    res.send(await db.getAll("entry"));
  }
  else if (req.method === "POST" || req.method === "PUT") {
    res.send(await db.create("entry", req.body));
  }
  // Need a way to delete. Which means I need ids? Or I could just use refs.
};
