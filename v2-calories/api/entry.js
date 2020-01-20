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


const getTotal = () => {
  return q.Sum(
      q.Select(
        "data",
        q.Map(
          q.Paginate(q.Match("all_entry"), {size: 10000}),
          q.Lambda("x", q.Select(["data", "calories"], q.Get(q.Var("x"))))
        )
      )
    )
};

const numberOfDays = () => {
  return q.Count(
    q.Distinct(q.Select("data", 
      q.Map(
        q.Paginate(q.Match("all_entry"), {size: 10000}),
        q.Lambda("x", q.Select(["data", "date"], q.Get(q.Var("x"))))))));
}

const bmr = 2260;
// 500 = 1lb per week;
const goal = 500;

const getPounds = () => {
  return q.Let({
        total: getTotal(),
        days: numberOfDays(),
      },
    q.Divide(q.ToDouble(q.Subtract(q.Multiply(q.Var("days"), bmr), q.Var("total"))), 3500.0))

}


const extraCalories = () => {
  return q.Let({
        total: getTotal(),
         days: numberOfDays(),
      },
   q.Subtract(q.Multiply(q.Var("days"), bmr-goal), q.Var("total")))
}


const timeZone = 'America/New_York'
const today = () => formatDate(utcToZonedTime(new Date(), timeZone), "yyyy-MM-dd", { timeZone });


const getRemainingToday = () => {
  return q.Let({total: q.Sum(
                        q.Select("data", q.Map(
                          q.Filter(
                            q.Paginate(q.Match("all_entry"), {size: 10000}),
                            q.Lambda("x", q.Equals(today(), q.Select(["data", "date"], q.Get(q.Var("x")))))),
                          q.Lambda("x", q.Select(["data", "calories"], q.Get(q.Var("x")))))),
                      )},
   q.Subtract(bmr, goal, q.Var("total")))
}


const summary = () => {
  return client.query(
    q.Let({
      days: numberOfDays(),
      remaining: getRemainingToday(),
      extra: extraCalories(),
      total: getTotal(),
      pounds: getPounds(),
    },
      q.ToObject([
        ["remaining", q.Var("remaining")], 
        ["extraOnePound", q.Var("extra")],
        ["extraTwoPounds", q.Subtract(q.Var("extra"), q.Multiply(q.Var("days"), 500))],
        ["total", q.Var("total")],
        ["pounds", q.Var("pounds")],
        ["daily", bmr-goal],
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
